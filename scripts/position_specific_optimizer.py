#!/usr/bin/env python3
"""
⚠️  DEPRECATED: Use scripts/train_model.py instead.

This script is deprecated in favor of the unified training CLI:
    python scripts/train_model.py position --end-gw 12
    python scripts/train_model.py full-pipeline --end-gw 12

---

Position-Specific Pipeline Optimizer for FPL Expected Points Prediction

Comprehensive optimization of position-specific ML models with:
- Multiple regressors: LightGBM, XGBoost, Random Forest, Gradient Boosting
- Extensive hyperparameter optimization per position
- Position-specific feature engineering
- Fair comparison with unified baseline

Key insight: Different positions have different scoring patterns:
- GKP: Saves (1pt/3), clean sheets (4pt), penalty saves (5pt), goals conceded (-1pt/2)
- DEF: Clean sheets (4pt), goals (6pt), goals conceded (-1pt/2)
- MID: Goals (5pt), assists (3pt), clean sheets (1pt)
- FWD: Goals (4pt), assists (3pt), no clean sheet points

Usage:
    # Optimize all positions with extensive search
    python scripts/position_specific_optimizer.py optimize-all --end-gw 10 --n-trials 50

    # Optimize single position
    python scripts/position_specific_optimizer.py optimize --position GKP --end-gw 10 --n-trials 50

    # Evaluate best position models vs unified baseline
    python scripts/position_specific_optimizer.py evaluate --end-gw 12 --holdout-gws 2 \\
        --unified-model-path models/custom/lightgbm_gw1-10_*_pipeline.joblib
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy.stats import loguniform, randint, spearmanr, ttest_rel, uniform, wilcoxon
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from ml_training_utils import (  # noqa: E402
    TemporalCVSplitter,
    create_temporal_cv_splits,
    engineer_features,
    load_training_data,
)
from fpl_team_picker.domain.ml import FeatureSelector  # noqa: E402

app = typer.Typer(
    help="Position-specific model optimization for FPL xP prediction",
    add_completion=False,
)


# =============================================================================
# POSITION-SPECIFIC CONFIGURATION
# =============================================================================

# Features to EXCLUDE for each position (reduce noise from irrelevant features)
POSITION_FEATURE_EXCLUSIONS = {
    "GKP": [
        # GKP don't score/assist often - remove attacking features
        "rolling_5gw_goals",
        "rolling_5gw_assists",
        "rolling_5gw_xg",
        "rolling_5gw_xa",
        "rolling_5gw_xgi",
        "goals_per_90",
        "assists_per_90",
        "xg_per_90",
        "xa_per_90",
        "rolling_5gw_threat",
        "rolling_5gw_creativity",
        "cumulative_goals",
        "cumulative_assists",
        "cumulative_xg",
        "cumulative_xa",
    ],
    "DEF": [
        # DEF don't make saves
        "rolling_5gw_saves",
    ],
    "MID": [
        # MID don't make saves
        "rolling_5gw_saves",
    ],
    "FWD": [
        # FWD: no saves, no clean sheet points
        "rolling_5gw_saves",
        "rolling_5gw_clean_sheets",
        "clean_sheet_rate",
        "rolling_5gw_goals_conceded",
        "rolling_5gw_xgc",
        "cumulative_clean_sheets",
        "team_rolling_5gw_clean_sheets",
    ],
}

# Position-specific feature engineering additions
# These are interaction features that capture position-specific scoring patterns
POSITION_FEATURE_ADDITIONS = {
    "GKP": [
        # GKP scoring is dominated by saves and clean sheets
        (
            "saves_x_opp_xg",
            lambda df: df.get("rolling_5gw_saves", 0)
            * df.get("opponent_rolling_5gw_xg", 1),
        ),
        (
            "clean_sheet_potential",
            lambda df: df.get("team_rolling_5gw_clean_sheets", 0)
            / (df.get("games_played", 1).clip(lower=1)),
        ),
    ],
    "DEF": [
        # DEF scoring: clean sheets + occasional goals (set pieces)
        (
            "cs_x_fixture",
            lambda df: df.get("rolling_5gw_clean_sheets", 0)
            * df.get("fixture_difficulty", 1),
        ),
        (
            "set_piece_potential",
            lambda df: (df.get("is_corner_taker", 0) + df.get("is_fk_taker", 0))
            * df.get("rolling_5gw_bps", 0),
        ),
    ],
    "MID": [
        # MID scoring: goals and assists primarily
        (
            "goal_involvement",
            lambda df: df.get("rolling_5gw_goals", 0)
            + df.get("rolling_5gw_assists", 0),
        ),
        (
            "chance_creation",
            lambda df: df.get("rolling_5gw_creativity", 0)
            * df.get("team_rolling_5gw_xg", 1),
        ),
    ],
    "FWD": [
        # FWD scoring: goals are everything
        (
            "xg_efficiency",
            lambda df: df.get("rolling_5gw_goals", 0)
            / df.get("rolling_5gw_xg", 0.1).clip(lower=0.1),
        ),
        (
            "penalty_boost",
            lambda df: df.get("is_primary_penalty_taker", 0)
            * df.get("rolling_5gw_threat", 0),
        ),
    ],
}


def get_regressor_and_param_grid(
    regressor_name: str, position: str, random_seed: int
) -> tuple:
    """
    Get regressor instance and hyperparameter search space.

    Adjusts parameter ranges based on position sample size:
    - GKP/FWD (~400-600 samples): More regularization to prevent overfitting
    - DEF/MID (~1200-1700 samples): Standard parameter ranges
    """
    small_sample = position in ["GKP", "FWD"]

    if regressor_name == "lightgbm":
        try:
            import lightgbm as lgb

            regressor = lgb.LGBMRegressor(
                random_state=random_seed,
                n_jobs=1,
                verbose=-1,
            )

            if small_sample:
                # More regularization for small datasets
                param_dist = {
                    "regressor__n_estimators": randint(100, 500),
                    "regressor__max_depth": randint(3, 7),  # Shallower
                    "regressor__learning_rate": loguniform(0.03, 0.2),
                    "regressor__num_leaves": randint(15, 50),  # Fewer leaves
                    "regressor__subsample": uniform(0.6, 0.3),
                    "regressor__colsample_bytree": uniform(0.5, 0.4),
                    "regressor__min_child_samples": randint(
                        15, 40
                    ),  # More samples per leaf
                    "regressor__reg_alpha": uniform(0.1, 1.0),  # More L1
                    "regressor__reg_lambda": uniform(0.5, 2.0),  # More L2
                }
            else:
                # Standard ranges for larger datasets
                param_dist = {
                    "regressor__n_estimators": randint(200, 1000),
                    "regressor__max_depth": randint(3, 10),
                    "regressor__learning_rate": loguniform(0.03, 0.3),
                    "regressor__num_leaves": randint(20, 80),
                    "regressor__subsample": uniform(0.6, 0.4),
                    "regressor__colsample_bytree": uniform(0.6, 0.4),
                    "regressor__min_child_samples": randint(10, 40),
                    "regressor__reg_alpha": uniform(0, 1.0),
                    "regressor__reg_lambda": uniform(0.0, 2.0),
                }
        except ImportError:
            raise ImportError("LightGBM not installed. Run: uv add lightgbm")

    elif regressor_name == "xgboost":
        try:
            import xgboost as xgb

            regressor = xgb.XGBRegressor(
                random_state=random_seed,
                n_jobs=1,
                tree_method="hist",
            )

            if small_sample:
                param_dist = {
                    "regressor__n_estimators": randint(100, 500),
                    "regressor__max_depth": randint(3, 6),
                    "regressor__learning_rate": loguniform(0.03, 0.2),
                    "regressor__subsample": uniform(0.6, 0.3),
                    "regressor__colsample_bytree": uniform(0.5, 0.4),
                    "regressor__min_child_weight": randint(3, 15),
                    "regressor__gamma": uniform(0.1, 0.5),
                    "regressor__reg_alpha": uniform(0.1, 1.0),
                    "regressor__reg_lambda": uniform(0.5, 2.0),
                }
            else:
                param_dist = {
                    "regressor__n_estimators": randint(200, 1000),
                    "regressor__max_depth": randint(3, 10),
                    "regressor__learning_rate": loguniform(0.03, 0.3),
                    "regressor__subsample": uniform(0.6, 0.4),
                    "regressor__colsample_bytree": uniform(0.6, 0.4),
                    "regressor__min_child_weight": randint(1, 10),
                    "regressor__gamma": uniform(0, 0.5),
                    "regressor__reg_alpha": uniform(0, 1.0),
                    "regressor__reg_lambda": uniform(0.0, 2.0),
                }
        except ImportError:
            raise ImportError("XGBoost not installed. Run: uv add xgboost")

    elif regressor_name == "random-forest":
        regressor = RandomForestRegressor(random_state=random_seed, n_jobs=1)

        if small_sample:
            param_dist = {
                "regressor__n_estimators": randint(100, 300),
                "regressor__max_depth": randint(5, 15),  # Shallower
                "regressor__min_samples_split": randint(5, 20),
                "regressor__min_samples_leaf": randint(5, 20),  # More regularization
                "regressor__max_features": uniform(0.3, 0.4),
            }
        else:
            param_dist = {
                "regressor__n_estimators": randint(100, 500),
                "regressor__max_depth": randint(5, 30),
                "regressor__min_samples_split": randint(2, 20),
                "regressor__min_samples_leaf": randint(1, 10),
                "regressor__max_features": uniform(0.3, 0.7),
            }

    elif regressor_name == "gradient-boost":
        regressor = GradientBoostingRegressor(random_state=random_seed)

        if small_sample:
            param_dist = {
                "regressor__n_estimators": randint(100, 400),
                "regressor__max_depth": randint(3, 6),
                "regressor__learning_rate": loguniform(0.03, 0.15),
                "regressor__subsample": uniform(0.6, 0.3),
                "regressor__min_samples_split": randint(5, 20),
                "regressor__min_samples_leaf": randint(5, 15),
            }
        else:
            param_dist = {
                "regressor__n_estimators": randint(200, 800),
                "regressor__max_depth": randint(3, 8),
                "regressor__learning_rate": loguniform(0.03, 0.2),
                "regressor__subsample": uniform(0.6, 0.4),
                "regressor__min_samples_split": randint(2, 20),
                "regressor__min_samples_leaf": randint(1, 10),
            }

    else:
        raise ValueError(f"Unknown regressor: {regressor_name}")

    return regressor, param_dist


def add_position_features(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Add position-specific engineered features."""
    df = df.copy()

    additions = POSITION_FEATURE_ADDITIONS.get(position, [])
    for feature_name, feature_fn in additions:
        try:
            df[feature_name] = feature_fn(df)
            # Handle infinities and NaNs
            df[feature_name] = df[feature_name].replace([np.inf, -np.inf], 0).fillna(0)
        except Exception as e:
            logger.warning(f"Failed to add feature {feature_name}: {e}")
            df[feature_name] = 0

    return df


def filter_features_for_position(feature_names: list, position: str) -> list:
    """Remove position-irrelevant features."""
    exclusions = POSITION_FEATURE_EXCLUSIONS.get(position, [])
    return [f for f in feature_names if f not in exclusions]


# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================


def optimize_position_model(
    position: str,
    features_df: pd.DataFrame,
    target: np.ndarray,
    cv_data: pd.DataFrame,
    regressor_name: str = "lightgbm",
    n_trials: int = 50,
    random_seed: int = 42,
) -> tuple[Pipeline, dict]:
    """
    Optimize a position-specific model with extensive hyperparameter search.

    Args:
        position: Position to train for (GKP, DEF, MID, FWD)
        features_df: Feature DataFrame with 'position' column
        target: Target values (total_points)
        cv_data: CV data with gameweek, player_id, position
        regressor_name: Regressor to use
        n_trials: Number of hyperparameter trials
        random_seed: Random seed

    Returns:
        Tuple of (trained_pipeline, metadata_dict)
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"🎯 Optimizing {position} model with {regressor_name}")
    logger.info(f"{'=' * 60}")

    # Filter to position
    pos_mask = cv_data["position"] == position
    pos_features = features_df[pos_mask].copy()
    pos_target = target[pos_mask]
    pos_cv_data = cv_data[pos_mask].copy()

    logger.info(f"📊 {position} data: {len(pos_features):,} samples")
    logger.info(
        f"   Target stats: mean={pos_target.mean():.2f}, std={pos_target.std():.2f}"
    )

    # Add position-specific engineered features
    pos_features = add_position_features(pos_features, position)

    # Get feature names and filter for position
    metadata_cols = ["gameweek", "position", "player_id"]
    all_feature_names = [c for c in pos_features.columns if c not in metadata_cols]
    pos_feature_names = filter_features_for_position(all_feature_names, position)

    # Position-specific features are already in pos_features from add_position_features()
    # Just ensure they're in the feature list (they should be, since they're in all_feature_names)
    position_additions = [
        name for name, _ in POSITION_FEATURE_ADDITIONS.get(position, [])
    ]
    for feat in position_additions:
        if feat not in pos_feature_names:
            pos_feature_names.append(feat)

    logger.info(
        f"   Features: {len(pos_feature_names)} (base: {len(all_feature_names)}, "
        f"excluded: {len(all_feature_names) - len(pos_feature_names) + len(position_additions)}, "
        f"added: {len(position_additions)})"
    )

    # Get regressor and param grid
    regressor, param_dist = get_regressor_and_param_grid(
        regressor_name, position, random_seed
    )

    # Create temporal CV splits
    cv_gws = sorted(pos_cv_data[pos_cv_data["gameweek"] >= 6]["gameweek"].unique())
    if len(cv_gws) < 3:
        raise ValueError(f"Need at least 3 gameweeks for CV, got {len(cv_gws)}")

    # Adjust CV folds based on sample size
    small_sample = position in ["GKP", "FWD"]
    n_cv_folds = min(len(cv_gws) - 1, 3 if small_sample else 5)
    logger.info(f"   CV folds: {n_cv_folds}")

    cv_splits, _ = create_temporal_cv_splits(pos_cv_data, n_cv_folds)
    cv_splitter = TemporalCVSplitter(cv_splits)

    # Create pipeline
    pipeline = Pipeline(
        [
            ("feature_selector", FeatureSelector(feature_names=pos_feature_names)),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )

    # Hyperparameter search
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_trials,
        cv=cv_splitter,
        scoring="neg_mean_absolute_error",
        random_state=random_seed,
        n_jobs=-1,
        verbose=1,
    )

    logger.info(f"\n🔍 Running {n_trials} hyperparameter trials...")
    # all_feature_names already includes position-specific features from add_position_features()
    search.fit(pos_features[all_feature_names], pos_target)

    logger.info(f"\n✅ Best {position} model ({regressor_name}):")
    logger.info(f"   CV MAE: {-search.best_score_:.3f}")
    logger.info(f"   Best params: {search.best_params_}")

    # Metadata
    metadata = {
        "position": position,
        "regressor": regressor_name,
        "n_samples": len(pos_features),
        "n_features": len(pos_feature_names),
        "position_specific_features": position_additions,
        "excluded_features": list(set(all_feature_names) - set(pos_feature_names)),
        "cv_mae": -search.best_score_,
        "best_params": {
            k: v if not isinstance(v, np.generic) else v.item()
            for k, v in search.best_params_.items()
        },
        "n_cv_folds": n_cv_folds,
        "n_trials": n_trials,
        "trained_at": datetime.now().isoformat(),
    }

    return search.best_estimator_, metadata


def optimize_all_positions(
    end_gw: int,
    regressors: list[str] = ["lightgbm", "xgboost", "random-forest", "gradient-boost"],
    n_trials: int = 50,
    output_dir: str = "models/position_specific",
    random_seed: int = 42,
) -> dict:
    """
    Optimize models for all positions, trying multiple regressors.

    For each position, tries all regressors and keeps the best one.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"\n📥 Loading training data (GW1-{end_gw})...")
    data = load_training_data(start_gw=1, end_gw=end_gw)
    (
        historical_df,
        fixtures_df,
        teams_df,
        ownership_trends_df,
        value_analysis_df,
        fixture_difficulty_df,
        betting_features_df,
        raw_players_df,
        derived_player_metrics_df,
        player_availability_snapshot_df,
        derived_team_form_df,
        players_enhanced_df,
    ) = data

    # Engineer features
    logger.info("\n🔧 Engineering features...")
    features_df, target, feature_names = engineer_features(
        historical_df,
        fixtures_df,
        teams_df,
        ownership_trends_df,
        value_analysis_df,
        fixture_difficulty_df,
        betting_features_df,
        raw_players_df,
        derived_player_metrics_df,
        player_availability_snapshot_df,
        derived_team_form_df,
        players_enhanced_df,
    )

    # Create cv_data
    cv_data = features_df[["gameweek", "position", "player_id"]].copy()

    # Results storage
    results = {}
    best_models = {}

    for position in ["GKP", "DEF", "MID", "FWD"]:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# OPTIMIZING {position}")
        logger.info(f"{'#' * 60}")

        position_results = {}
        best_mae = float("inf")
        best_regressor = None
        best_model = None
        best_metadata = None

        for regressor_name in regressors:
            try:
                model, metadata = optimize_position_model(
                    position=position,
                    features_df=features_df.copy(),
                    target=target,
                    cv_data=cv_data,
                    regressor_name=regressor_name,
                    n_trials=n_trials,
                    random_seed=random_seed,
                )

                position_results[regressor_name] = {
                    "cv_mae": metadata["cv_mae"],
                    "best_params": metadata["best_params"],
                }

                if metadata["cv_mae"] < best_mae:
                    best_mae = metadata["cv_mae"]
                    best_regressor = regressor_name
                    best_model = model
                    best_metadata = metadata

            except Exception as e:
                logger.error(f"❌ {regressor_name} failed for {position}: {e}")
                position_results[regressor_name] = {"error": str(e)}

        if best_model is not None:
            # Save best model for this position
            model_path = output_path / f"{position.lower()}_gw1-{end_gw}_best.joblib"
            joblib.dump(best_model, model_path)
            logger.info(
                f"\n💾 Saved best {position} model ({best_regressor}) to {model_path}"
            )

            # Save metadata
            meta_path = (
                output_path / f"{position.lower()}_gw1-{end_gw}_best_metadata.json"
            )
            with open(meta_path, "w") as f:
                json.dump(best_metadata, f, indent=2, default=str)

            best_models[position] = {
                "model_path": str(model_path),
                "regressor": best_regressor,
                "cv_mae": best_mae,
            }

        results[position] = {
            "all_results": position_results,
            "best_regressor": best_regressor,
            "best_cv_mae": best_mae,
        }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 OPTIMIZATION SUMMARY")
    logger.info("=" * 60)
    for pos, result in results.items():
        if result["best_regressor"]:
            logger.info(
                f"{pos}: Best={result['best_regressor']}, CV MAE={result['best_cv_mae']:.3f}"
            )
            for reg, reg_result in result["all_results"].items():
                if "cv_mae" in reg_result:
                    marker = "✓" if reg == result["best_regressor"] else " "
                    logger.info(f"   {marker} {reg}: {reg_result['cv_mae']:.3f}")
        else:
            logger.error(f"{pos}: ALL REGRESSORS FAILED")

    # Save summary
    summary_path = output_path / f"optimization_summary_gw1-{end_gw}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n📄 Summary saved to {summary_path}")

    return results


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================


def evaluate_position_vs_unified(
    position_models: dict,
    unified_model,
    test_features: pd.DataFrame,
    test_target: np.ndarray,
    test_cv_data: pd.DataFrame,
) -> dict:
    """Compare position-specific ensemble vs unified model."""
    results = {"overall": {}, "per_position": {}}

    # Get feature columns (exclude metadata)
    metadata_cols = ["gameweek", "position", "player_id"]
    feature_cols = [c for c in test_features.columns if c not in metadata_cols]

    # Unified model predictions
    unified_preds = unified_model.predict(test_features[feature_cols])

    # Position-specific predictions (combine by position)
    combined_preds = np.zeros(len(test_features))
    for position in ["GKP", "DEF", "MID", "FWD"]:
        mask = test_cv_data["position"] == position
        if mask.sum() > 0 and position in position_models:
            # Add position-specific features for this position
            pos_features = add_position_features(
                test_features.loc[mask].copy(), position
            )

            # Get all feature columns (pass all to model, FeatureSelector will filter)
            all_feature_cols = [
                c for c in pos_features.columns if c not in metadata_cols
            ]

            # Use the full pipeline - FeatureSelector will extract the right features
            model = position_models[position]
            combined_preds[mask] = model.predict(pos_features[all_feature_cols])

    y_true = test_target

    # Overall metrics
    results["overall"]["unified"] = {
        "mae": float(mean_absolute_error(y_true, unified_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, unified_preds))),
        "spearman": float(spearmanr(y_true, unified_preds)[0]),
    }

    results["overall"]["position_specific"] = {
        "mae": float(mean_absolute_error(y_true, combined_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, combined_preds))),
        "spearman": float(spearmanr(y_true, combined_preds)[0]),
    }

    # Per-position comparison
    for position in ["GKP", "DEF", "MID", "FWD"]:
        mask = test_cv_data["position"] == position
        if mask.sum() > 0:
            y_pos = y_true[mask]
            results["per_position"][position] = {
                "n_samples": int(mask.sum()),
                "unified_mae": float(mean_absolute_error(y_pos, unified_preds[mask])),
                "specific_mae": float(mean_absolute_error(y_pos, combined_preds[mask])),
            }
            results["per_position"][position]["improvement"] = (
                results["per_position"][position]["unified_mae"]
                - results["per_position"][position]["specific_mae"]
            )
            results["per_position"][position]["improvement_pct"] = (
                results["per_position"][position]["improvement"]
                / results["per_position"][position]["unified_mae"]
                * 100
            )

    # Statistical significance
    unified_errors = np.abs(y_true - unified_preds)
    specific_errors = np.abs(y_true - combined_preds)

    t_stat, t_pvalue = ttest_rel(unified_errors, specific_errors)
    w_stat, w_pvalue = wilcoxon(unified_errors, specific_errors)

    diff = unified_errors - specific_errors
    cohens_d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0

    results["significance"] = {
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "t_significant": bool(t_pvalue < 0.05),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pvalue),
        "wilcoxon_significant": bool(w_pvalue < 0.05),
        "cohens_d": cohens_d,
        "mean_improvement": float(diff.mean()),
    }

    return results


# =============================================================================
# CLI COMMANDS
# =============================================================================


@app.command()
def optimize(
    position: str = typer.Option(..., help="Position to optimize (GKP, DEF, MID, FWD)"),
    end_gw: int = typer.Option(10, help="End gameweek for training data"),
    regressor: str = typer.Option("lightgbm", help="Regressor to use"),
    n_trials: int = typer.Option(50, help="Number of hyperparameter trials"),
    output_dir: str = typer.Option("models/position_specific", help="Output directory"),
    random_seed: int = typer.Option(42, help="Random seed"),
):
    """Optimize a single position with specified regressor."""
    position = position.upper()
    if position not in ["GKP", "DEF", "MID", "FWD"]:
        raise typer.BadParameter(f"Invalid position: {position}")

    # Load and engineer features
    data = load_training_data(start_gw=1, end_gw=end_gw)
    features_df, target, _ = engineer_features(*data)
    cv_data = features_df[["gameweek", "position", "player_id"]].copy()

    # Optimize
    model, metadata = optimize_position_model(
        position=position,
        features_df=features_df,
        target=target,
        cv_data=cv_data,
        regressor_name=regressor,
        n_trials=n_trials,
        random_seed=random_seed,
    )

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / f"{position.lower()}_gw1-{end_gw}_{regressor}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"💾 Saved to {model_path}")

    meta_path = (
        output_path / f"{position.lower()}_gw1-{end_gw}_{regressor}_metadata.json"
    )
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


@app.command("optimize-all")
def optimize_all(
    end_gw: int = typer.Option(10, help="End gameweek for training data"),
    regressors: str = typer.Option(
        "lightgbm,xgboost,random-forest,gradient-boost",
        help="Comma-separated list of regressors to try",
    ),
    n_trials: int = typer.Option(
        50, help="Number of hyperparameter trials per regressor"
    ),
    output_dir: str = typer.Option("models/position_specific", help="Output directory"),
    random_seed: int = typer.Option(42, help="Random seed"),
):
    """Optimize models for all positions, trying multiple regressors."""
    regressor_list = [r.strip() for r in regressors.split(",")]

    results = optimize_all_positions(
        end_gw=end_gw,
        regressors=regressor_list,
        n_trials=n_trials,
        output_dir=output_dir,
        random_seed=random_seed,
    )

    return results


@app.command()
def evaluate(
    end_gw: int = typer.Option(12, help="End gameweek"),
    holdout_gws: int = typer.Option(
        2, help="Number of gameweeks to hold out for testing"
    ),
    position_model_dir: str = typer.Option(
        "models/position_specific", help="Position models directory"
    ),
    unified_model_path: str = typer.Option(
        ...,
        help="Path to unified model trained on GW1-(end_gw - holdout_gws) for fair comparison",
    ),
    hybrid_model_path: Optional[str] = typer.Option(
        None, help="Optional: Path to hybrid model for three-way comparison"
    ),
):
    """Evaluate position-specific models vs unified model (optionally vs hybrid)."""
    train_end_gw = end_gw - holdout_gws

    if hybrid_model_path:
        logger.info(
            f"📊 Evaluating unified vs position-specific vs hybrid (GW{train_end_gw + 1}-{end_gw} holdout)"
        )
    else:
        logger.info(
            f"📊 Evaluating position-specific vs unified (GW{train_end_gw + 1}-{end_gw} holdout)"
        )

    # Load position models (best models)
    model_dir = Path(position_model_dir)
    position_models = {}
    for position in ["GKP", "DEF", "MID", "FWD"]:
        model_path = model_dir / f"{position.lower()}_gw1-{train_end_gw}_best.joblib"
        if model_path.exists():
            position_models[position] = joblib.load(model_path)
            logger.info(f"✅ Loaded {position} model from {model_path}")
        else:
            # Try older naming convention
            old_path = model_dir / f"{position.lower()}_gw1-{train_end_gw}.joblib"
            if old_path.exists():
                position_models[position] = joblib.load(old_path)
                logger.info(f"✅ Loaded {position} model from {old_path}")
            else:
                logger.warning(f"⚠️ {position} model not found")

    if not position_models:
        logger.error("❌ No position models found. Run optimize-all first.")
        raise typer.Exit(1)

    # Load unified model
    if Path(unified_model_path).exists():
        unified_model = joblib.load(unified_model_path)
        logger.info(f"✅ Loaded unified model from {unified_model_path}")
    else:
        logger.error(f"❌ Unified model not found at {unified_model_path}")
        raise typer.Exit(1)

    # Load hybrid model if provided
    hybrid_model = None
    if hybrid_model_path:
        if Path(hybrid_model_path).exists():
            hybrid_model = joblib.load(hybrid_model_path)
            logger.info(f"✅ Loaded hybrid model from {hybrid_model_path}")
        else:
            logger.warning(
                f"⚠️ Hybrid model not found at {hybrid_model_path}, skipping hybrid comparison"
            )

    # Load test data
    logger.info(f"\n📥 Loading test data (GW{train_end_gw + 1}-{end_gw})...")
    data = load_training_data(start_gw=1, end_gw=end_gw)
    features_df, target, _ = engineer_features(*data)
    cv_data = features_df[["gameweek", "position", "player_id"]].copy()

    # Split into train/test
    test_gws = list(range(train_end_gw + 1, end_gw + 1))
    test_mask = cv_data["gameweek"].isin(test_gws)

    test_features = features_df[test_mask].reset_index(drop=True)
    test_target = target[test_mask]
    test_cv_data = cv_data[test_mask].reset_index(drop=True)

    logger.info(f"   Test samples: {len(test_features):,}")

    # Evaluate
    results = evaluate_position_vs_unified(
        position_models=position_models,
        unified_model=unified_model,
        test_features=test_features,
        test_target=test_target,
        test_cv_data=test_cv_data,
    )

    # Evaluate hybrid model if provided
    if hybrid_model is not None:
        logger.info("\n🔬 Evaluating hybrid model...")

        # Get hybrid predictions
        hybrid_preds = hybrid_model.predict(test_features)
        hybrid_mae = mean_absolute_error(test_target, hybrid_preds)

        results["hybrid"] = {
            "mae": float(hybrid_mae),
            "rmse": float(np.sqrt(mean_squared_error(test_target, hybrid_preds))),
        }

        # Per-position hybrid metrics
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            pos_mask = test_cv_data["position"] == pos
            if pos_mask.any():
                pos_mae = mean_absolute_error(
                    test_target[pos_mask], hybrid_preds[pos_mask]
                )
                results["per_position"][pos]["hybrid_mae"] = float(pos_mae)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("📊 EVALUATION RESULTS")
    logger.info("=" * 60)

    logger.info("\n🎯 Overall Metrics:")
    logger.info(f"   Unified MAE:          {results['overall']['unified']['mae']:.3f}")
    logger.info(
        f"   Position-Specific MAE: {results['overall']['position_specific']['mae']:.3f}"
    )

    improvement = (
        results["overall"]["unified"]["mae"]
        - results["overall"]["position_specific"]["mae"]
    )
    logger.info(
        f"   Improvement:          {improvement:+.3f} ({improvement / results['overall']['unified']['mae'] * 100:+.1f}%)"
    )

    if "hybrid" in results:
        logger.info(f"   Hybrid MAE:           {results['hybrid']['mae']:.3f}")
        hybrid_improvement = (
            results["overall"]["unified"]["mae"] - results["hybrid"]["mae"]
        )
        logger.info(
            f"   Hybrid Improvement:   {hybrid_improvement:+.3f} ({hybrid_improvement / results['overall']['unified']['mae'] * 100:+.1f}%)"
        )

    logger.info("\n📊 Per-Position:")
    for pos, metrics in results["per_position"].items():
        line = f"   {pos}: Unified={metrics['unified_mae']:.3f}, Specific={metrics['specific_mae']:.3f}"
        if "hybrid_mae" in metrics:
            line += f", Hybrid={metrics['hybrid_mae']:.3f}"
        line += (
            f", Δ={metrics['improvement']:+.3f} ({metrics['improvement_pct']:+.1f}%)"
        )
        logger.info(line)

    logger.info("\n📈 Statistical Significance (Unified vs Position-Specific):")
    sig = results["significance"]
    logger.info(
        f"   t-test p-value:     {sig['t_pvalue']:.4f} {'✅' if sig['t_significant'] else '❌'}"
    )
    logger.info(
        f"   Wilcoxon p-value:   {sig['wilcoxon_pvalue']:.4f} {'✅' if sig['wilcoxon_significant'] else '❌'}"
    )
    logger.info(f"   Cohen's d:          {sig['cohens_d']:.3f}")

    # Save results
    results_path = model_dir / f"evaluation_gw{train_end_gw + 1}-{end_gw}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n📄 Results saved to {results_path}")


@app.command("build-hybrid")
def build_hybrid(
    unified_model_path: str = typer.Option(
        ...,
        help="Path to unified model (e.g., models/custom/lightgbm_gw1-10_*_pipeline.joblib)",
    ),
    position_model_dir: str = typer.Option(
        "models/position_specific", help="Directory containing position-specific models"
    ),
    end_gw: int = typer.Option(
        10, help="End gameweek used for training (to find correct model files)"
    ),
    use_specific_for: str = typer.Option(
        "GKP,FWD", help="Comma-separated positions to use position-specific models for"
    ),
    output: str = typer.Option(
        "models/hybrid/hybrid.joblib", help="Output path for hybrid model"
    ),
):
    """
    Build a hybrid model combining position-specific and unified models.

    Based on experiment results:
    - GKP, FWD: Position-specific models perform better (+6.5%, +9.0%)
    - DEF, MID: Unified model benefits from more training data

    The hybrid model routes predictions based on position, using the best
    approach for each position.
    """
    from fpl_team_picker.domain.ml import HybridPositionModel

    logger.info("=" * 60)
    logger.info("🔧 Building Hybrid Position Model")
    logger.info("=" * 60)

    # Parse positions to use specific models for
    specific_positions = [p.strip().upper() for p in use_specific_for.split(",")]
    logger.info("\n📋 Configuration:")
    logger.info(f"   Position-specific for: {specific_positions}")
    logger.info(
        f"   Unified for: {[p for p in ['GKP', 'DEF', 'MID', 'FWD'] if p not in specific_positions]}"
    )

    # Load unified model
    unified_path = Path(unified_model_path)
    if not unified_path.exists():
        # Try glob pattern
        import glob

        matches = glob.glob(str(unified_path))
        if matches:
            unified_path = Path(matches[0])
        else:
            logger.error(f"❌ Unified model not found: {unified_model_path}")
            raise typer.Exit(1)

    logger.info(f"\n📥 Loading unified model from {unified_path.name}")
    unified_model = joblib.load(unified_path)

    # Load position-specific models (only for positions we need)
    model_dir = Path(position_model_dir)
    position_models = {}

    for position in specific_positions:
        # Try best model first
        model_path = model_dir / f"{position.lower()}_gw1-{end_gw}_best.joblib"
        if not model_path.exists():
            # Try older naming convention
            model_path = model_dir / f"{position.lower()}_gw1-{end_gw}.joblib"

        if model_path.exists():
            position_models[position] = joblib.load(model_path)
            logger.info(f"   ✅ Loaded {position} model: {model_path.name}")
        else:
            logger.error(f"   ❌ {position} model not found at {model_path}")
            raise typer.Exit(1)

    # Create hybrid model
    logger.info("\n🔨 Creating HybridPositionModel...")
    hybrid_model = HybridPositionModel(
        unified_model=unified_model,
        position_models=position_models,
        use_specific_for=specific_positions,
    )

    logger.info(f"   {hybrid_model}")

    # Save hybrid model
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hybrid_model, output_path)
    logger.info(f"\n💾 Saved hybrid model to {output_path}")

    # Save metadata
    metadata = {
        "unified_model": str(unified_path),
        "position_models": {
            pos: str(model_dir / f"{pos.lower()}_gw1-{end_gw}_best.joblib")
            for pos in specific_positions
        },
        "use_specific_for": specific_positions,
        "unified_positions": [
            p for p in ["GKP", "DEF", "MID", "FWD"] if p not in specific_positions
        ],
        "created_at": datetime.now().isoformat(),
        "end_gw": end_gw,
    }

    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"   Metadata saved to {metadata_path.name}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Hybrid model built successfully!")
    logger.info("=" * 60)
    logger.info("\nTo use in production, update settings.py:")
    logger.info(f'   ml_model_path = "{output_path}"')

    return hybrid_model


@app.command("full-pipeline")
def full_pipeline(
    end_gw: int = typer.Option(
        ..., help="Final gameweek (e.g., 12 = use all data up to GW12)"
    ),
    holdout_gws: int = typer.Option(
        2, help="Gameweeks to hold out for evaluation phase"
    ),
    regressors: str = typer.Option(
        "lightgbm,xgboost,random-forest,gradient-boost",
        help="Comma-separated regressors to try for position-specific models",
    ),
    unified_regressors: str = typer.Option(
        "lightgbm,xgboost,random-forest,gradient-boost",
        help="Comma-separated regressors to try for unified model",
    ),
    n_trials: int = typer.Option(
        50, help="Hyperparameter trials per regressor per position"
    ),
    scorer: str = typer.Option(
        "neg_mean_absolute_error",
        help="Scorer for hyperparameter optimization (neg_mean_absolute_error, spearman, etc.)",
    ),
    improvement_threshold: float = typer.Option(
        0.02,
        help="Minimum improvement (as fraction) to use position-specific model (default: 2%)",
    ),
    unified_model_path: Optional[str] = typer.Option(
        None, help="Path to existing unified model (skips unified optimization)"
    ),
    output_dir: str = typer.Option("models", help="Base output directory"),
    random_seed: int = typer.Option(42, help="Random seed"),
):
    """
    Full pipeline: Evaluate → Determine best config → Retrain on all data.

    This implements the proper ML workflow:
    1. EVALUATION PHASE: Train on GW1-(end-holdout), evaluate on holdout GWs
       - Tries multiple regressors for BOTH unified and position-specific models
       - Picks the best regressor for unified model
       - Determines which positions benefit from position-specific models
    2. PRODUCTION PHASE: Retrain everything on GW1-end with best config

    Example:
        # With 12 GWs of data, holdout 2 for evaluation:
        python scripts/position_specific_optimizer.py full-pipeline --end-gw 12 --holdout-gws 2

        # With custom scorer:
        python scripts/position_specific_optimizer.py full-pipeline --end-gw 12 --scorer spearman

        # Try only LightGBM and XGBoost for unified:
        python scripts/position_specific_optimizer.py full-pipeline --end-gw 12 --unified-regressors "lightgbm,xgboost"
    """
    import glob

    train_end_gw = end_gw - holdout_gws
    regressor_list = [r.strip() for r in regressors.split(",")]
    unified_regressor_list = [r.strip() for r in unified_regressors.split(",")]

    logger.info("=" * 70)
    logger.info("🚀 FULL HYBRID PIPELINE")
    logger.info("=" * 70)
    logger.info("\n📋 Configuration:")
    logger.info(f"   End gameweek: {end_gw}")
    logger.info(f"   Holdout gameweeks: {holdout_gws} (GW{train_end_gw + 1}-{end_gw})")
    logger.info(f"   Evaluation training: GW1-{train_end_gw}")
    logger.info(f"   Production training: GW1-{end_gw}")
    logger.info(f"   Position-specific regressors: {regressor_list}")
    logger.info(f"   Unified regressors: {unified_regressor_list}")
    logger.info(f"   Trials per regressor: {n_trials}")
    logger.info(f"   Scorer: {scorer}")
    logger.info(f"   Improvement threshold: {improvement_threshold * 100:.1f}%")

    # =========================================================================
    # PHASE 1: EVALUATION
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("📊 PHASE 1: EVALUATION (finding best configuration)")
    logger.info("=" * 70)

    # 1a. Train and evaluate unified models with different regressors
    best_unified_regressor = None
    best_unified_mae = float("inf")
    eval_unified_path = unified_model_path
    unified_results = {}

    if eval_unified_path is None:
        logger.info(f"\n🔧 Step 1a: Training unified models on GW1-{train_end_gw}...")
        logger.info(f"   Testing regressors: {unified_regressor_list}")

        from custom_pipeline_optimizer import run_train_mode  # noqa: E402

        # Load test data once for evaluation
        logger.info("\n   Loading evaluation data...")
        data = load_training_data(start_gw=1, end_gw=end_gw)
        features_df, target, _ = engineer_features(*data)
        cv_data = features_df[["gameweek", "position", "player_id"]].copy()

        test_gws = list(range(train_end_gw + 1, end_gw + 1))
        test_mask = cv_data["gameweek"].isin(test_gws)

        test_features = features_df[test_mask].reset_index(drop=True)
        test_target = target[test_mask]

        for reg_name in unified_regressor_list:
            logger.info(f"\n   {'─' * 50}")
            logger.info(f"   🎯 Training unified model with {reg_name}...")

            # Check for existing best params for this regressor
            params_pattern = (
                Path(output_dir)
                / "custom"
                / f"best_params_{reg_name}_gw1-{train_end_gw}_*.json"
            )
            params_files = sorted(glob.glob(str(params_pattern)))
            best_params_file = params_files[-1] if params_files else None

            if best_params_file:
                logger.info(
                    f"      Using existing params: {Path(best_params_file).name}"
                )

            try:
                run_train_mode(
                    start_gw=1,
                    end_gw=train_end_gw,
                    regressor=reg_name,
                    feature_selection="none",
                    keep_penalty_features=False,
                    preprocessing="standard",
                    n_trials=n_trials,
                    cv_folds=None,
                    scorer=scorer,
                    output_dir=f"{output_dir}/custom",
                    use_best_params_from=best_params_file,
                    random_seed=random_seed,
                    n_jobs=-1,
                    verbose=0,  # Reduce noise
                )

                # Find and evaluate the trained model
                model_pattern = (
                    Path(output_dir)
                    / "custom"
                    / f"{reg_name}_gw1-{train_end_gw}_*_pipeline.joblib"
                )
                model_files = sorted(glob.glob(str(model_pattern)))

                if model_files:
                    model_path = model_files[-1]
                    model = joblib.load(model_path)

                    # Get features the model expects
                    feature_selector = model.named_steps.get("feature_selector")
                    if feature_selector:
                        model_features = feature_selector.feature_names
                        preds = model.predict(test_features[model_features])
                    else:
                        preds = model.predict(test_features)

                    mae = mean_absolute_error(test_target, preds)
                    unified_results[reg_name] = {"mae": mae, "path": model_path}

                    logger.info(f"      ✅ {reg_name}: MAE = {mae:.4f}")

                    if mae < best_unified_mae:
                        best_unified_mae = mae
                        best_unified_regressor = reg_name
                        eval_unified_path = model_path
                else:
                    logger.warning(f"      ⚠️ Could not find trained {reg_name} model")

            except Exception as e:
                logger.warning(f"      ❌ Failed to train {reg_name}: {e}")

        if best_unified_regressor:
            logger.info(f"\n   {'═' * 50}")
            logger.info(
                f"   🏆 Best unified regressor: {best_unified_regressor} (MAE: {best_unified_mae:.4f})"
            )
            logger.info(f"   {'═' * 50}")
        else:
            logger.error("   ❌ No unified models trained successfully")
            raise typer.Exit(1)
    else:
        logger.info(f"\n🔧 Step 1a: Using provided unified model: {eval_unified_path}")
        # Try to infer regressor from filename
        path_str = str(eval_unified_path).lower()
        for reg in ["lightgbm", "xgboost", "random-forest", "gradient-boost"]:
            if reg in path_str:
                best_unified_regressor = reg
                break
        if not best_unified_regressor:
            best_unified_regressor = "unknown"

    # 1b. Optimize position-specific models on evaluation data
    logger.info(
        f"\n🔧 Step 1b: Optimizing position-specific models on GW1-{train_end_gw}..."
    )

    _eval_results = optimize_all_positions(
        end_gw=train_end_gw,
        regressors=regressor_list,
        n_trials=n_trials,
        output_dir=f"{output_dir}/position_specific",
        random_seed=random_seed,
    )

    # 1c. Evaluate to find best configuration
    logger.info(f"\n🔧 Step 1c: Evaluating on holdout GW{train_end_gw + 1}-{end_gw}...")

    # Load models and evaluate
    model_dir = Path(output_dir) / "position_specific"
    position_models = {}
    for position in ["GKP", "DEF", "MID", "FWD"]:
        model_path = model_dir / f"{position.lower()}_gw1-{train_end_gw}_best.joblib"
        if model_path.exists():
            position_models[position] = joblib.load(model_path)

    unified_model = joblib.load(eval_unified_path)

    # Load test data
    data = load_training_data(start_gw=1, end_gw=end_gw)
    features_df, target, _ = engineer_features(*data)
    cv_data = features_df[["gameweek", "position", "player_id"]].copy()

    test_gws = list(range(train_end_gw + 1, end_gw + 1))
    test_mask = cv_data["gameweek"].isin(test_gws)

    test_features = features_df[test_mask].reset_index(drop=True)
    test_target = target[test_mask]
    test_cv_data = cv_data[test_mask].reset_index(drop=True)

    results = evaluate_position_vs_unified(
        position_models=position_models,
        unified_model=unified_model,
        test_features=test_features,
        test_target=test_target,
        test_cv_data=test_cv_data,
    )

    # Determine which positions benefit from position-specific models
    positions_to_use_specific = []
    logger.info("\n📊 Evaluation Results:")
    logger.info(
        f"   {'Position':<8} {'Unified':<10} {'Specific':<10} {'Improve':<10} {'Use Specific?'}"
    )
    logger.info("   " + "-" * 50)

    for pos, metrics in results["per_position"].items():
        unified_mae = metrics["unified_mae"]
        specific_mae = metrics["specific_mae"]
        _improvement = metrics["improvement"]
        improvement_pct = metrics["improvement_pct"] / 100  # Convert to fraction

        use_specific = improvement_pct >= improvement_threshold
        if use_specific:
            positions_to_use_specific.append(pos)

        marker = "✅ YES" if use_specific else "❌ NO"
        logger.info(
            f"   {pos:<8} {unified_mae:<10.3f} {specific_mae:<10.3f} {improvement_pct:>+8.1%}   {marker}"
        )

    logger.info(
        f"\n   Positions to use specific models: {positions_to_use_specific or 'None (use unified for all)'}"
    )

    # =========================================================================
    # PHASE 2: PRODUCTION TRAINING
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("🏭 PHASE 2: PRODUCTION TRAINING (retrain on all data)")
    logger.info("=" * 70)

    # 2a. Retrain unified model on ALL data (GW1-end_gw) using best regressor
    logger.info(
        f"\n🔧 Step 2a: Retraining unified model on GW1-{end_gw} with {best_unified_regressor}..."
    )

    from custom_pipeline_optimizer import run_train_mode  # noqa: E402

    # Use best params from evaluation phase for the winning regressor
    eval_params_pattern = (
        Path(output_dir)
        / "custom"
        / f"best_params_{best_unified_regressor}_gw1-{train_end_gw}_*.json"
    )
    eval_params_files = sorted(glob.glob(str(eval_params_pattern)))
    eval_params_file = eval_params_files[-1] if eval_params_files else None

    if eval_params_file:
        logger.info(f"   Using params from evaluation: {Path(eval_params_file).name}")

    try:
        run_train_mode(
            start_gw=1,
            end_gw=end_gw,
            regressor=best_unified_regressor,
            feature_selection="none",
            keep_penalty_features=False,
            preprocessing="standard",
            n_trials=n_trials,
            cv_folds=None,
            scorer=scorer,
            output_dir=f"{output_dir}/custom",
            use_best_params_from=eval_params_file,
            random_seed=random_seed,
            n_jobs=-1,
            verbose=1,
        )
    except Exception as e:
        logger.error(f"Failed to retrain unified model: {e}")
        raise typer.Exit(1)

    # Find production unified model
    prod_unified_pattern = (
        Path(output_dir)
        / "custom"
        / f"{best_unified_regressor}_gw1-{end_gw}_*_pipeline.joblib"
    )
    prod_unified_files = sorted(glob.glob(str(prod_unified_pattern)))
    prod_unified_path = prod_unified_files[-1] if prod_unified_files else None

    if not prod_unified_path:
        logger.error("Failed to find production unified model")
        raise typer.Exit(1)

    logger.info(f"   ✅ Production unified model: {Path(prod_unified_path).name}")

    # 2b. Retrain position-specific models on ALL data (only for positions that benefit)
    if positions_to_use_specific:
        logger.info(
            f"\n🔧 Step 2b: Retraining position-specific models on GW1-{end_gw}..."
        )
        logger.info(f"   Positions: {positions_to_use_specific}")

        optimize_all_positions(
            end_gw=end_gw,
            regressors=regressor_list,
            n_trials=n_trials,
            output_dir=f"{output_dir}/position_specific",
            random_seed=random_seed,
        )
    else:
        logger.info(
            "\n🔧 Step 2b: Skipping position-specific retraining (none beat threshold)"
        )

    # 2c. Build final hybrid model
    logger.info("\n🔧 Step 2c: Building final hybrid model...")

    if positions_to_use_specific:
        from fpl_team_picker.domain.ml import HybridPositionModel

        # Load production models
        prod_unified = joblib.load(prod_unified_path)
        prod_position_models = {}

        for pos in positions_to_use_specific:
            pos_model_path = (
                Path(output_dir)
                / "position_specific"
                / f"{pos.lower()}_gw1-{end_gw}_best.joblib"
            )
            if pos_model_path.exists():
                prod_position_models[pos] = joblib.load(pos_model_path)
                logger.info(f"   ✅ Loaded {pos} model: {pos_model_path.name}")

        # Create hybrid
        hybrid = HybridPositionModel(
            unified_model=prod_unified,
            position_models=prod_position_models,
            use_specific_for=positions_to_use_specific,
        )

        # Save hybrid
        hybrid_dir = Path(output_dir) / "hybrid"
        hybrid_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hybrid_path = hybrid_dir / f"hybrid_gw1-{end_gw}_{timestamp}.joblib"
        joblib.dump(hybrid, hybrid_path)

        # Save metadata
        metadata = {
            "unified_model": str(prod_unified_path),
            "unified_regressor": best_unified_regressor,
            "unified_evaluation_results": unified_results,
            "position_models": {
                pos: str(
                    Path(output_dir)
                    / "position_specific"
                    / f"{pos.lower()}_gw1-{end_gw}_best.joblib"
                )
                for pos in positions_to_use_specific
            },
            "use_specific_for": positions_to_use_specific,
            "unified_positions": [
                p
                for p in ["GKP", "DEF", "MID", "FWD"]
                if p not in positions_to_use_specific
            ],
            "evaluation_results": results,
            "improvement_threshold": improvement_threshold,
            "scorer": scorer,
            "created_at": datetime.now().isoformat(),
            "end_gw": end_gw,
            "holdout_gws": holdout_gws,
        }

        metadata_path = hybrid_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        final_model_path = hybrid_path
        final_model_type = "Hybrid"
    else:
        # No positions benefit, just use unified
        final_model_path = prod_unified_path
        final_model_type = "Unified"

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("✅ FULL PIPELINE COMPLETE!")
    logger.info("=" * 70)

    logger.info("\n📊 Configuration Determined:")
    logger.info(f"   Best unified regressor: {best_unified_regressor}")
    if unified_results:
        logger.info("   Unified regressor comparison:")
        for reg, res in sorted(unified_results.items(), key=lambda x: x[1]["mae"]):
            marker = "👑" if reg == best_unified_regressor else "  "
            logger.info(f"     {marker} {reg}: MAE = {res['mae']:.4f}")
    if positions_to_use_specific:
        logger.info(f"   Position-specific for: {positions_to_use_specific}")
        logger.info(
            f"   Unified for: {[p for p in ['GKP', 'DEF', 'MID', 'FWD'] if p not in positions_to_use_specific]}"
        )
    else:
        logger.info("   Using unified model for all positions")

    logger.info("\n💾 Final Model:")
    logger.info(f"   Type: {final_model_type}")
    logger.info(f"   Path: {final_model_path}")

    logger.info("\n📝 To deploy, update settings.py:")
    logger.info(f'   ml_model_path = "{final_model_path}"')

    return {
        "final_model_path": str(final_model_path),
        "final_model_type": final_model_type,
        "positions_specific": positions_to_use_specific,
        "evaluation_results": results,
    }


@app.command()
def compare(
    end_gw: int = typer.Option(10, help="End gameweek"),
    position_model_dir: str = typer.Option(
        "models/position_specific", help="Position models directory"
    ),
):
    """Quick comparison of position model results."""
    model_dir = Path(position_model_dir)

    # Look for optimization summary
    summary_path = model_dir / f"optimization_summary_gw1-{end_gw}.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

        logger.info("\n📊 Position Model Optimization Results")
        logger.info("=" * 60)

        for position, result in summary.items():
            logger.info(f"\n{position}:")
            if result.get("best_regressor"):
                logger.info(
                    f"   Best: {result['best_regressor']} (CV MAE: {result['best_cv_mae']:.3f})"
                )
                for reg, reg_result in result.get("all_results", {}).items():
                    if "cv_mae" in reg_result:
                        marker = "✓" if reg == result["best_regressor"] else " "
                        logger.info(f"   {marker} {reg}: {reg_result['cv_mae']:.3f}")
    else:
        logger.warning(f"No optimization summary found at {summary_path}")
        logger.info("Run `optimize-all` first to generate results.")


if __name__ == "__main__":
    app()
