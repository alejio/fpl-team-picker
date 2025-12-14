#!/usr/bin/env python3
"""
⚠️  DEPRECATED: Use scripts/train_model.py instead.

This script is deprecated in favor of the unified training CLI:
    python scripts/train_model.py unified --end-gw 12 --regressor lightgbm
    python scripts/train_model.py full-pipeline --end-gw 12

---

Custom Pipeline Optimizer for FPL Expected Points Prediction

Fully configurable ML pipeline optimizer with:
- Manual control over feature selection with smart RFE
- Choice of regressors (XGBoost, LightGBM, RandomForest, etc.)
- Hyperparameter optimization via RandomizedSearchCV
- Temporal CV evaluation with FPL-specific metrics
- Dual modes: evaluate (fast testing on holdout) or train (full training for deployment)
- Improved hyperparameter ranges to prevent underfitting

Key Features:
- Smart RFE that preserves important penalty/set-piece features
- Keeps all 117 features or uses intelligent feature selection

Hyperparameter Improvements (2025-11-22):
- XGBoost/LightGBM: Increased n_estimators to 200-1500 (was 100-500)
- Gradient Boosting: learning_rate minimum raised from 0.01 to 0.03
- Regularization: reg_lambda minimum lowered from 0.5 to 0.0
- Prevents underfitting from learning_rate=0.01 + n_estimators=180 combination
- Uses log-uniform distribution for learning_rate (samples more in 0.03-0.1 range)

Learning Rate × N_Estimators Tradeoff:
- Low LR (0.03-0.05) needs many trees (800-1500) for full capacity
- Medium LR (0.05-0.15) needs moderate trees (300-800)
- High LR (0.15-0.3) needs fewer trees (200-400)
- Effective capacity ≈ learning_rate × n_estimators × tree_depth

Usage:
    # Evaluate mode: Test configuration on holdout set (GW9-10) before full training
    python scripts/custom_pipeline_optimizer.py evaluate --end-gw 10 --holdout-gws 2 \\
        --regressor random-forest --feature-selection rfe-smart --keep-penalty-features

    # Train mode: Full training on all data for deployment
    python scripts/custom_pipeline_optimizer.py train --end-gw 10 \\
        --regressor random-forest --feature-selection rfe-smart --keep-penalty-features --n-trials 50

    # Quick examples
    python scripts/custom_pipeline_optimizer.py evaluate --regressor xgboost
    python scripts/custom_pipeline_optimizer.py train --regressor lightgbm --feature-selection correlation
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import reusable training utilities
sys.path.insert(0, str(Path(__file__).parent))
from ml_training_utils import (  # noqa: E402
    TemporalCVSplitter,
    create_fpl_comprehensive_scorer_sklearn,
    create_fpl_position_aware_scorer_sklearn,
    create_fpl_starting_xi_scorer_sklearn,
    create_temporal_cv_splits,
    engineer_features,
    evaluate_fpl_comprehensive,
    fpl_captain_scorer_sklearn,
    fpl_hauler_capture_scorer_sklearn,
    fpl_hauler_ceiling_scorer_sklearn,
    fpl_topk_scorer_sklearn,
    fpl_weighted_huber_scorer_sklearn,
    load_training_data,
    spearman_correlation_scorer,
)

# Import custom transformers from domain layer
from fpl_team_picker.domain.ml import FeatureSelector  # noqa: E402

# Create Typer app
app = typer.Typer(
    help="Custom pipeline optimization for FPL xP prediction",
    add_completion=False,
)


# ==============================================================================
# COMMON PARAMETERS
# ==============================================================================


# Store common options in a dataclass-like dict builder
def build_config(
    start_gw: int,
    end_gw: int,
    regressor: str,
    feature_selection: str,
    keep_penalty_features: bool,
    preprocessing: str,
    n_trials: int,
    cv_folds: int | None,
    scorer: str,
    output_dir: str,
    random_seed: int,
    n_jobs: int,
    verbose: int,
) -> dict:
    """Build configuration dictionary from parameters."""
    return {
        "start_gw": start_gw,
        "end_gw": end_gw,
        "regressor": regressor,
        "feature_selection": feature_selection,
        "keep_penalty_features": keep_penalty_features,
        "preprocessing": preprocessing,
        "n_trials": n_trials,
        "cv_folds": cv_folds,
        "scorer": scorer,
        "output_dir": output_dir,
        "random_seed": random_seed,
        "n_jobs": n_jobs,
        "verbose": verbose,
    }


def get_regressor_and_param_grid(regressor_name: str, random_seed: int) -> tuple:
    """
    Get regressor instance and hyperparameter search space.

    Args:
        regressor_name: Name of regressor
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (regressor, param_distributions)
    """
    if regressor_name == "xgboost":
        try:
            import xgboost as xgb
            from scipy.stats import loguniform

            # XGBoost with improved defaults to prevent underfitting
            regressor = xgb.XGBRegressor(
                random_state=random_seed,
                n_jobs=1,
                tree_method="hist",
                # Better defaults (will be overridden by search)
                n_estimators=500,  # Higher default
                learning_rate=0.05,  # Safer default than 0.1
                max_depth=6,
                min_child_weight=3,
            )
            # Improved parameter distributions to prevent underfitting
            # Key insight: learning_rate × n_estimators controls model capacity
            # - Low LR (0.03-0.05) needs many trees (800-1500)
            # - Medium LR (0.05-0.15) needs moderate trees (300-800)
            # - High LR (0.15-0.3) needs fewer trees (200-400)
            param_dist = {
                # Increased range: 200-1500 (was 100-500)
                # This prevents underfitting when learning_rate is low
                "regressor__n_estimators": randint(200, 1500),
                # Tree structure: keep existing range
                "regressor__max_depth": randint(3, 10),
                # Learning rate: raised minimum from 0.01 to 0.03
                # 0.01 with low n_estimators causes severe underfitting
                # Using log-uniform to sample more values in 0.03-0.1 range
                "regressor__learning_rate": loguniform(0.03, 0.3),
                # Sampling parameters: keep existing
                "regressor__subsample": uniform(0.6, 0.4),  # 0.6-1.0
                "regressor__colsample_bytree": uniform(0.6, 0.4),  # 0.6-1.0
                # Regularization: keep existing
                "regressor__min_child_weight": randint(1, 10),
                "regressor__gamma": uniform(0, 0.5),
                # L1 regularization: keep existing (0-1.0)
                "regressor__reg_alpha": uniform(0, 1.0),
                # L2 regularization: lower minimum from 0.5 to 0.0
                # High L2 + low LR = extreme underfitting
                "regressor__reg_lambda": uniform(0.0, 2.0),
            }
        except ImportError:
            raise ImportError("XGBoost not installed. Run: uv add xgboost")

    elif regressor_name == "lightgbm":
        try:
            import lightgbm as lgb
            from scipy.stats import loguniform

            # LightGBM with improved defaults to prevent underfitting
            regressor = lgb.LGBMRegressor(
                random_state=random_seed,
                n_jobs=1,
                verbose=-1,
                # Better defaults (will be overridden by search)
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
            )
            # Improved parameter distributions (same reasoning as XGBoost)
            param_dist = {
                # Increased range: 200-1500 (was 100-500)
                "regressor__n_estimators": randint(200, 1500),
                # Tree structure: keep existing range
                "regressor__max_depth": randint(3, 10),
                # Learning rate: raised minimum from 0.01 to 0.03, using log-uniform
                "regressor__learning_rate": loguniform(0.03, 0.3),
                # LightGBM-specific: num_leaves
                "regressor__num_leaves": randint(20, 100),
                # Sampling parameters: keep existing
                "regressor__subsample": uniform(0.6, 0.4),  # 0.6-1.0
                "regressor__colsample_bytree": uniform(0.6, 0.4),  # 0.6-1.0
                "regressor__min_child_samples": randint(10, 50),
                # L1 regularization: keep existing (0-1.0)
                "regressor__reg_alpha": uniform(0, 1.0),
                # L2 regularization: lower minimum from 0.5 to 0.0
                "regressor__reg_lambda": uniform(0.0, 2.0),
            }
        except ImportError:
            raise ImportError("LightGBM not installed. Run: uv add lightgbm")

    elif regressor_name == "random-forest":
        regressor = RandomForestRegressor(random_state=random_seed, n_jobs=1)
        param_dist = {
            "regressor__n_estimators": randint(100, 500),
            "regressor__max_depth": randint(5, 30),
            "regressor__min_samples_split": randint(2, 20),
            "regressor__min_samples_leaf": randint(1, 10),
            "regressor__max_features": uniform(0.3, 0.7),
        }

    elif regressor_name == "gradient-boost":
        from scipy.stats import loguniform

        # Sklearn GradientBoosting with improved defaults
        regressor = GradientBoostingRegressor(
            random_state=random_seed,
            # Better defaults
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
        )
        # Improved parameter distributions
        param_dist = {
            # Increased range: 200-1000 (was 100-400)
            # Sklearn GradientBoosting is slower, so cap at 1000
            "regressor__n_estimators": randint(200, 1000),
            # Tree structure: keep existing range
            "regressor__max_depth": randint(3, 8),
            # Learning rate: raised minimum from 0.01 to 0.03, using log-uniform
            "regressor__learning_rate": loguniform(0.03, 0.2),
            # Sampling: keep existing
            "regressor__subsample": uniform(0.6, 0.4),  # 0.6-1.0
            "regressor__min_samples_split": randint(2, 20),
            "regressor__min_samples_leaf": randint(1, 10),
        }

    elif regressor_name == "adaboost":
        regressor = AdaBoostRegressor(random_state=random_seed)
        param_dist = {
            "regressor__n_estimators": randint(50, 300),
            "regressor__learning_rate": uniform(0.01, 1.0),
            "regressor__loss": ["linear", "square", "exponential"],
        }

    elif regressor_name == "ridge":
        regressor = Ridge(random_state=random_seed)
        param_dist = {
            "regressor__alpha": uniform(0.001, 10.0),
            "regressor__solver": ["auto", "svd", "cholesky", "lsqr"],
        }

    elif regressor_name == "lasso":
        regressor = Lasso(random_state=random_seed, max_iter=10000)
        param_dist = {
            "regressor__alpha": uniform(0.001, 5.0),
            "regressor__selection": ["cyclic", "random"],
        }

    elif regressor_name == "elasticnet":
        regressor = ElasticNet(random_state=random_seed, max_iter=10000)
        param_dist = {
            "regressor__alpha": uniform(0.001, 5.0),
            "regressor__l1_ratio": uniform(0.0, 1.0),
            "regressor__selection": ["cyclic", "random"],
        }

    else:
        raise ValueError(f"Unknown regressor: {regressor_name}")

    return regressor, param_dist


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    strategy: str,
    keep_penalty_features: bool,
    verbose: bool = True,
) -> list[str]:
    """
    Select features based on strategy.

    Args:
        X: Feature matrix
        y: Target variable
        feature_names: List of feature names
        strategy: Feature selection strategy
        keep_penalty_features: Force keep penalty features
        verbose: Print progress

    Returns:
        List of selected feature names
    """
    if strategy == "none":
        if verbose:
            logger.info(f"   ℹ️  Using all {len(feature_names)} features (no selection)")
        return feature_names

    penalty_features = [
        "is_primary_penalty_taker",
        "is_penalty_taker",
        "is_corner_taker",
        "is_fk_taker",
    ]

    if strategy == "correlation":
        # Remove highly correlated features (keep one from each pair)
        if verbose:
            logger.info("   🔍 Removing highly correlated features (|r| > 0.95)...")

        corr_matrix = X[feature_names].corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = set()
        for column in upper_tri.columns:
            if column in to_drop:
                continue
            high_corr = upper_tri[column][upper_tri[column] > 0.95].index.tolist()
            # Keep penalty features if requested
            if keep_penalty_features and column in penalty_features:
                to_drop.update([c for c in high_corr if c not in penalty_features])
            else:
                to_drop.update(high_corr)

        selected = [f for f in feature_names if f not in to_drop]

        if verbose:
            logger.info(
                f"   ✅ Removed {len(to_drop)} correlated features, kept {len(selected)}"
            )

    elif strategy == "permutation":
        # Use permutation importance from previous analysis
        # Keep top 60 features + penalty features if requested
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance

        if verbose:
            logger.info("   🔍 Computing permutation importance...")

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X[feature_names], y)

        perm_imp = permutation_importance(
            rf, X[feature_names], y, n_repeats=5, random_state=42, n_jobs=-1
        )

        # Get feature importance scores
        feature_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": perm_imp.importances_mean,
            }
        ).sort_values("importance", ascending=False)

        # Keep top 60 features
        top_features = feature_importance.head(60)["feature"].tolist()

        # Add penalty features if requested and not already included
        if keep_penalty_features:
            for pf in penalty_features:
                if pf not in top_features and pf in feature_names:
                    top_features.append(pf)

        selected = top_features

        if verbose:
            logger.info(
                f"   ✅ Selected top {len(selected)} features by permutation importance"
            )

    elif strategy == "rfe-smart":
        # RFE but force keep penalty features
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.feature_selection import RFE

        if verbose:
            logger.info("   🔍 Running smart RFE (keeps penalty features)...")

        # Use ExtraTreesRegressor for feature ranking
        estimator = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        # If keeping penalty features, temporarily remove them, run RFE, then add back
        if keep_penalty_features:
            non_penalty_features = [
                f for f in feature_names if f not in penalty_features
            ]

            # Run RFE on non-penalty features
            selector = RFE(
                estimator, n_features_to_select=56, step=0.1
            )  # 56 + 4 penalty = 60
            selector.fit(X[non_penalty_features], y)

            selected_non_penalty = [
                f for f, keep in zip(non_penalty_features, selector.support_) if keep
            ]
            selected = selected_non_penalty + penalty_features

        else:
            selector = RFE(estimator, n_features_to_select=60, step=0.1)
            selector.fit(X[feature_names], y)
            selected = [f for f, keep in zip(feature_names, selector.support_) if keep]

        if verbose:
            logger.info(f"   ✅ RFE selected {len(selected)} features")
            if keep_penalty_features:
                logger.info(
                    f"      (including {len(penalty_features)} penalty features)"
                )

    else:
        selected = feature_names

    # Ensure penalty features are included if requested
    if keep_penalty_features:
        for pf in penalty_features:
            if pf not in selected and pf in feature_names:
                selected.append(pf)
                if verbose:
                    logger.info(f"   ➕ Force-added penalty feature: {pf}")

    return selected


def get_feature_groups() -> dict[str, list[str]]:
    """
    Define feature groups for grouped preprocessing.

    Categorizes all 156 FPL features into appropriate preprocessing groups based on their
    distribution characteristics and semantic meaning.

    Returns:
        Dict mapping group name to list of feature names
    """
    return {
        # Count features (Poisson-like, many zeros) - Use Robust Scaler
        "count_features": [
            "games_played",
            "cumulative_goals",
            "cumulative_assists",
            "cumulative_clean_sheets",
            "cumulative_yellow_cards",
            "cumulative_red_cards",
            "rolling_5gw_goals",
            "rolling_5gw_assists",
            "rolling_5gw_clean_sheets",
            "rolling_5gw_saves",
            "rolling_5gw_goals_conceded",
            "net_transfers_gw",
            "team_rolling_5gw_goals_scored",
            "team_rolling_5gw_goals_conceded",
            "team_rolling_5gw_clean_sheets",
            "team_cumulative_goals_scored",
            "team_cumulative_goals_conceded",
            "team_cumulative_clean_sheets",
            "opponent_rolling_5gw_goals_conceded",
            "opponent_rolling_5gw_clean_sheets",
        ],
        # Binary features (0/1) - No scaling needed
        "binary_features": [
            "is_home",
            "is_primary_penalty_taker",
            "is_penalty_taker",
            "is_corner_taker",
            "is_fk_taker",
        ],
        # Percentage features (0-100 range) - Use MinMaxScaler
        "percentage_features": [
            "selected_by_percent",
            "ownership_velocity",
            "minutes_played_rate",
            "clean_sheet_rate",
            "clean_sheet_probability_enhanced",
        ],
        # Probability features (0-1 range) - Use MinMaxScaler
        "probability_features": [
            "team_win_probability",
            "opponent_win_probability",
            "draw_probability",
            "implied_clean_sheet_probability",
            "market_consensus_strength",
            "favorite_status",
        ],
        # Continuous/rate features - Use StandardScaler
        "continuous_features": [
            # Per-90 rates
            "goals_per_90",
            "assists_per_90",
            "points_per_90",
            "xg_per_90",
            "xa_per_90",
            "bps_per_90",
            "rolling_5gw_goals_per_90",
            "rolling_5gw_assists_per_90",
            "rolling_5gw_points_per_90",
            # xG/xA features
            "cumulative_xg",
            "cumulative_xa",
            "rolling_5gw_xg",
            "rolling_5gw_xa",
            "rolling_5gw_xgi",
            "rolling_5gw_xgc",
            "team_rolling_5gw_xg",
            "team_rolling_5gw_xgc",
            "team_rolling_5gw_xg_diff",
            "opponent_rolling_5gw_xgc",
            # Points/bonus/BPS
            "cumulative_points",
            "cumulative_bonus",
            "cumulative_bps",
            "rolling_5gw_points",
            "rolling_5gw_bps",
            "rolling_5gw_bonus",
            "team_rolling_5gw_points",
            "team_season_points",
            "team_rolling_5gw_goal_diff",
            # ICT index
            "rolling_5gw_ict_index",
            "rolling_5gw_influence",
            "rolling_5gw_creativity",
            "rolling_5gw_threat",
            # Volatility/consistency
            "rolling_5gw_points_std",
            "rolling_5gw_minutes_std",
            "form_trend",
            # Fixture strength
            "opponent_strength",
            "fixture_difficulty",
            "congestion_difficulty",
            "form_adjusted_difficulty",
            # Betting odds continuous
            "implied_total_goals",
            "team_expected_goals",
            "asian_handicap_line",
            "handicap_team_odds",
            "expected_goal_difference",
            "over_under_signal",
            "odds_movement_team",
            "odds_movement_magnitude",
            # Value features
            "points_per_pound",
            "value_vs_position",
            "predicted_price_change_1gw",
            "price_volatility",
            "price_risk",
            "bandwagon_score",
            "avg_net_transfers_5gw",
            # Minutes
            "cumulative_minutes",
            "rolling_5gw_minutes",
        ],
        # Encoded/ordinal features - Use StandardScaler
        "encoded_features": [
            "position_encoded",
            "team_encoded",
            "ownership_tier_encoded",
            "transfer_momentum_encoded",
            "price_band",
            "referee_encoded",
        ],
        # Price features - Use StandardScaler
        "price_features": [
            "price",
        ],
    }


def create_preprocessor(strategy: str, feature_names: list[str]):
    """
    Create a preprocessor based on the specified strategy.

    Args:
        strategy: Preprocessing strategy ("standard", "grouped", or "robust")
        feature_names: List of all available feature names (after feature selection)

    Returns:
        Preprocessor (either StandardScaler, RobustScaler, or ColumnTransformer)
    """
    if strategy == "standard":
        # Original behavior: StandardScaler for all features
        return StandardScaler()

    elif strategy == "robust":
        # RobustScaler for all features (good for linear models with outliers)
        return RobustScaler()

    elif strategy == "grouped":
        # Feature-type specific preprocessing
        feature_groups = get_feature_groups()

        # Find which features from each group are present (after feature selection)
        transformers = []

        # Count features -> RobustScaler (handles outliers from hauls/blanks)
        count_feats = [
            f for f in feature_groups["count_features"] if f in feature_names
        ]
        if count_feats:
            transformers.append(("count_scaler", RobustScaler(), count_feats))

        # Binary features -> passthrough (already 0/1)
        binary_feats = [
            f for f in feature_groups["binary_features"] if f in feature_names
        ]
        if binary_feats:
            transformers.append(("binary_passthrough", "passthrough", binary_feats))

        # Percentage features -> MinMaxScaler (already bounded 0-100)
        pct_feats = [
            f for f in feature_groups["percentage_features"] if f in feature_names
        ]
        if pct_feats:
            transformers.append(("percentage_minmax", MinMaxScaler(), pct_feats))

        # Probability features -> MinMaxScaler (already bounded 0-1)
        prob_feats = [
            f for f in feature_groups["probability_features"] if f in feature_names
        ]
        if prob_feats:
            transformers.append(("probability_minmax", MinMaxScaler(), prob_feats))

        # Continuous features -> StandardScaler (normal-ish distributions)
        cont_feats = [
            f for f in feature_groups["continuous_features"] if f in feature_names
        ]
        if cont_feats:
            transformers.append(("continuous_standard", StandardScaler(), cont_feats))

        # Encoded features -> StandardScaler
        encoded_feats = [
            f for f in feature_groups["encoded_features"] if f in feature_names
        ]
        if encoded_feats:
            transformers.append(("encoded_standard", StandardScaler(), encoded_feats))

        # Price features -> StandardScaler
        price_feats = [
            f for f in feature_groups["price_features"] if f in feature_names
        ]
        if price_feats:
            transformers.append(("price_standard", StandardScaler(), price_feats))

        # Create ColumnTransformer
        return ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",  # Pass through any features not in groups
            verbose_feature_names_out=False,  # Keep original feature names
        )

    else:
        raise ValueError(f"Unknown preprocessing strategy: {strategy}")


def optimize_pipeline(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: list,
    feature_names: list[str],
    config: dict,
    cv_data: pd.DataFrame | None = None,
    best_params: dict | None = None,
) -> tuple[Pipeline, dict]:
    """
    Optimize hyperparameters using RandomizedSearchCV or use provided best params.

    Args:
        X: Feature matrix (DataFrame with feature columns)
        y: Target variable
        cv_splits: CV splits
        feature_names: Selected feature names
        config: Configuration dictionary
        cv_data: Optional full cv_data DataFrame with position column (required for position-aware scorers)
        best_params: Optional dict of best hyperparameters to use directly (skips search)

    Returns:
        Tuple of (best_pipeline, search_results)
    """
    logger.info(f"\n🤖 Building {config['regressor']} pipeline...")
    logger.info(f"   Preprocessing strategy: {config['preprocessing']}")

    # Get regressor and param grid
    regressor, param_dist = get_regressor_and_param_grid(
        config["regressor"], config["random_seed"]
    )

    # Create preprocessor based on strategy
    preprocessor = create_preprocessor(config["preprocessing"], feature_names)

    # Build pipeline: FeatureSelector → Preprocessor → Regressor
    # This makes the pipeline self-contained - it knows which features it needs
    pipeline = Pipeline(
        [
            ("feature_selector", FeatureSelector(feature_names)),
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    # Select scorer
    # Handle position-aware scorers specially (they need cv_data with position column)
    if config["scorer"] in [
        "fpl_position_aware",
        "fpl_starting_xi",
        "fpl_comprehensive",
    ]:
        if cv_data is None:
            raise ValueError(
                f"cv_data required for position-aware scorer '{config['scorer']}'. "
                "This should be passed from the calling function."
            )
        if "position" not in cv_data.columns:
            raise ValueError(
                f"cv_data must contain 'position' column for position-aware scorer '{config['scorer']}'"
            )
        # Create the scorer using factory functions from ml_training_utils
        if config["scorer"] == "fpl_position_aware":
            scorer = create_fpl_position_aware_scorer_sklearn(cv_data)
        elif config["scorer"] == "fpl_starting_xi":
            scorer = create_fpl_starting_xi_scorer_sklearn(cv_data, "4-4-2")
        elif config["scorer"] == "fpl_comprehensive":
            scorer = create_fpl_comprehensive_scorer_sklearn(cv_data, "4-4-2")
    else:
        # Standard scorers
        scorer_map = {
            "spearman": spearman_correlation_scorer,
            "fpl_weighted_huber": fpl_weighted_huber_scorer_sklearn,
            "fpl_top_k_ranking": fpl_topk_scorer_sklearn,
            "fpl_captain_pick": fpl_captain_scorer_sklearn,
            "fpl_hauler_capture": fpl_hauler_capture_scorer_sklearn,
            "fpl_hauler_ceiling": fpl_hauler_ceiling_scorer_sklearn,  # Hauler-first with variance preservation
        }
        scorer = scorer_map.get(config["scorer"], config["scorer"])

    # Create CV splitter
    cv_splitter = TemporalCVSplitter(cv_splits)

    # If best_params provided, use them directly (skip search)
    if best_params is not None:
        logger.info("   Using provided best hyperparameters (skipping search)")
        logger.info(f"      Best params: {best_params}")
        logger.info(f"      Features: {len(feature_names)}")

        # Set hyperparameters directly on pipeline
        pipeline.set_params(**best_params)

        # Fit pipeline on all data
        logger.info("\n🔧 Training pipeline with best hyperparameters...")
        pipeline.fit(X, y)

        # Evaluate on CV for consistency (even though we're not searching)
        logger.info("\n📊 Evaluating on CV data...")
        from sklearn.model_selection import cross_val_score

        cv_scores = cross_val_score(
            pipeline, X, y, cv=cv_splitter, scoring=scorer, n_jobs=config["n_jobs"]
        )
        cv_score = cv_scores.mean()

        logger.info(f"   CV score: {cv_score:.4f}")

        return pipeline, {
            "best_score": cv_score,
            "best_params": best_params,
            "cv_results": None,  # No search results when using fixed params
        }

    # Otherwise, run hyperparameter search
    logger.info("   Hyperparameter search:")
    logger.info(f"      Trials: {config['n_trials']}")
    logger.info(f"      CV folds: {len(cv_splits)}")
    logger.info(f"      Scorer: {config['scorer']}")
    logger.info(f"      Features: {len(feature_names)}")

    # Run randomized search
    logger.info("\n🔍 Running hyperparameter optimization...")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=config["n_trials"],
        cv=cv_splitter,
        scoring=scorer,
        n_jobs=config["n_jobs"],
        random_state=config["random_seed"],
        verbose=config["verbose"],
        return_train_score=True,
    )

    # Pass ALL features - the FeatureSelector inside the pipeline will select the ones it needs
    search.fit(X, y)

    print("\n✅ Hyperparameter optimization complete!")
    print(f"   Best CV score: {search.best_score_:.4f}")
    print(f"   Best params: {search.best_params_}")

    return search.best_estimator_, {
        "best_score": search.best_score_,
        "best_params": search.best_params_,
        "cv_results": search.cv_results_,
    }


@app.command("evaluate")
def run_evaluate_mode(
    holdout_gws: int = typer.Option(
        1,
        "--holdout-gws",
        help="Gameweeks to hold out for evaluation (default: 1)",
    ),
    start_gw: int = typer.Option(1, "--start-gw", help="Training start gameweek"),
    end_gw: int = typer.Option(9, "--end-gw", help="Training end gameweek"),
    regressor: Literal[
        "xgboost",
        "lightgbm",
        "random-forest",
        "gradient-boost",
        "adaboost",
        "ridge",
        "lasso",
        "elasticnet",
    ] = typer.Option("xgboost", "--regressor", help="Regressor to use"),
    feature_selection: Literal["none", "correlation", "permutation", "rfe-smart"] = (
        typer.Option(
            "none",
            "--feature-selection",
            help="Feature selection strategy (default: none = keep all 156 features)",
        )
    ),
    keep_penalty_features: bool = typer.Option(
        False,
        "--keep-penalty-features",
        help="Force keep penalty/set-piece features (is_primary_penalty_taker, etc.)",
    ),
    preprocessing: Literal["standard", "grouped", "robust"] = typer.Option(
        "standard",
        "--preprocessing",
        help=(
            "Preprocessing strategy: 'standard' (StandardScaler for all features), "
            "'grouped' (feature-type specific scalers), 'robust' (RobustScaler for all features)"
        ),
    ),
    n_trials: int = typer.Option(
        20,
        "--n-trials",
        help="Number of hyperparameter trials (reduced in evaluate mode)",
    ),
    cv_folds: int | None = typer.Option(
        None, "--cv-folds", help="Number of CV folds (default: None = all available)"
    ),
    scorer: Literal[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "spearman",
        "fpl_weighted_huber",
        "fpl_top_k_ranking",
        "fpl_captain_pick",
        "fpl_hauler_capture",
        "fpl_hauler_ceiling",
        "fpl_position_aware",
        "fpl_starting_xi",
        "fpl_comprehensive",
    ] = typer.Option(
        "fpl_weighted_huber",
        "--scorer",
        help="Scoring metric. FPL scorers optimize for strategic objectives. Position-aware scorers require position data.",
    ),
    output_dir: str = typer.Option(
        "models/custom", "--output-dir", help="Output directory"
    ),
    random_seed: int = typer.Option(42, "--random-seed", help="Random seed"),
    n_jobs: int = typer.Option(-1, "--n-jobs", help="Parallel jobs (-1 = all CPUs)"),
    verbose: int = typer.Option(2, "--verbose", help="Verbosity level (0, 1, or 2)"),
):
    """
    Evaluate mode: Test configuration on holdout set before full training.

    Workflow:
    1. Split data into training (GW1 to end-holdout) and holdout (last N GWs)
    2. Create temporal CV on training data only
    3. Train with reduced hyperparameter search
    4. Evaluate on CV folds
    5. Evaluate on holdout set
    6. Report comprehensive metrics
    """
    # Build configuration dictionary
    config = build_config(
        start_gw,
        end_gw,
        regressor,
        feature_selection,
        keep_penalty_features,
        preprocessing,
        n_trials,
        cv_folds,
        scorer,
        output_dir,
        random_seed,
        n_jobs,
        verbose,
    )
    config["holdout_gws"] = holdout_gws

    # Calculate split point
    train_end_gw = config["end_gw"] - holdout_gws
    holdout_start_gw = train_end_gw + 1

    # Validate holdout configuration
    total_gws = config["end_gw"] - config["start_gw"] + 1
    if holdout_gws >= total_gws:
        typer.echo(
            f"Error: Holdout gameweeks ({holdout_gws}) must be less than total gameweeks ({total_gws}). "
            f"With --end-gw {config['end_gw']} and --start-gw {config['start_gw']}, "
            f"you can hold out at most {total_gws - 6} gameweeks (need at least 6 GWs for training).",
            err=True,
        )
        raise typer.Exit(code=1)

    if train_end_gw < config["start_gw"]:
        typer.echo(
            f"Error: Invalid configuration. "
            f"Training end GW ({train_end_gw}) cannot be before start GW ({config['start_gw']}). "
            f"Reduce --holdout-gws (currently {holdout_gws}) or increase --end-gw (currently {config['end_gw']}).",
            err=True,
        )
        raise typer.Exit(code=1)

    if train_end_gw < config["start_gw"] + 5:
        available_training_gws = train_end_gw - config["start_gw"] + 1
        typer.echo(
            f"Error: Not enough training data after holdout. "
            f"Training GWs: {config['start_gw']}-{train_end_gw} ({available_training_gws} gameweeks). "
            f"Need at least 6 gameweeks for temporal CV. "
            f"Reduce --holdout-gws (currently {holdout_gws}) or increase --end-gw (currently {config['end_gw']}).",
            err=True,
        )
        raise typer.Exit(code=1)

    print("=" * 80)
    print("🔬 EVALUATE MODE: Testing Configuration on Holdout Set")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"   Regressor: {config['regressor']}")
    print(f"   Feature selection: {config['feature_selection']}")
    print(f"   Preprocessing: {config['preprocessing']}")
    print(f"   Keep penalty features: {config['keep_penalty_features']}")
    print(
        f"   Training GWs: {config['start_gw']}-{train_end_gw} ({train_end_gw - config['start_gw'] + 1} weeks)"
    )

    if holdout_gws == 1:
        logger.info(f"   Holdout GW: {holdout_start_gw} (single gameweek)")
    else:
        logger.info(
            f"   Holdout GWs: {holdout_start_gw}-{config['end_gw']} ({holdout_gws} weeks)"
        )

    logger.info(
        f"   Hyperparameter trials: {config['n_trials'] // 2} (reduced for speed)"
    )
    logger.info(f"   Scorer: {config['scorer']}")

    # 1. Load training data (excluding holdout)
    logger.info("\n📦 Loading training data...")
    train_data = load_training_data(config["start_gw"], train_end_gw, verbose=True)
    (
        train_historical_df,
        train_fixtures_df,
        train_teams_df,
        train_ownership_df,
        train_value_df,
        train_fixture_diff_df,
        train_betting_df,
        train_raw_players_df,
        train_derived_player_metrics_df,
        train_player_availability_snapshot_df,
        train_derived_team_form_df,
        train_players_enhanced_df,
        train_fixture_runs_df,
    ) = train_data

    # 2. Engineer features for training data
    logger.info("\n🔧 Engineering features for training data...")
    train_features_df, train_target, all_feature_names = engineer_features(
        train_historical_df,
        train_fixtures_df,
        train_teams_df,
        train_ownership_df,
        train_value_df,
        train_fixture_diff_df,
        train_betting_df,
        train_raw_players_df,
        train_derived_player_metrics_df,
        train_player_availability_snapshot_df,
        train_derived_team_form_df,
        train_players_enhanced_df,
        train_fixture_runs_df,
        verbose=True,
    )

    # 3. Create CV splits on training data only
    cv_splits, cv_data = create_temporal_cv_splits(
        train_features_df, max_folds=config["cv_folds"], verbose=True
    )

    # 4. Feature selection on training data
    X_train = cv_data[all_feature_names].copy()
    y_train = train_target[cv_data["_original_index"].values]

    logger.info("\n🔧 Feature Selection...")
    selected_features = select_features(
        X_train,
        y_train,
        all_feature_names,
        config["feature_selection"],
        config["keep_penalty_features"],
        verbose=True,
    )

    # 5. Optimize pipeline with reduced trials
    eval_config = config.copy()
    eval_config["n_trials"] = max(
        1, config["n_trials"] // 2
    )  # Reduce trials for faster evaluation (min 1)

    best_pipeline, search_results = optimize_pipeline(
        X_train, y_train, cv_splits, selected_features, eval_config, cv_data=cv_data
    )

    # 6. Evaluate on CV data
    logger.info("\n📊 Evaluation on Training Data (CV)...")
    y_train_pred = best_pipeline.predict(X_train)

    train_metrics = evaluate_fpl_comprehensive(
        y_true=y_train,
        y_pred=y_train_pred,
        cv_data=cv_data,
        verbose=True,
    )

    # 7. Load and evaluate on holdout set
    logger.info("\n" + "=" * 80)
    logger.info("📊 Evaluation on Holdout Set")
    logger.info("=" * 80)

    # Load full data including holdout
    holdout_data = load_training_data(
        config["start_gw"], config["end_gw"], verbose=False
    )
    (
        holdout_historical_df,
        holdout_fixtures_df,
        holdout_teams_df,
        holdout_ownership_df,
        holdout_value_df,
        holdout_fixture_diff_df,
        holdout_betting_df,
        holdout_raw_players_df,
        holdout_derived_player_metrics_df,
        holdout_player_availability_snapshot_df,
        holdout_derived_team_form_df,
        holdout_players_enhanced_df,
        holdout_fixture_runs_df,
    ) = holdout_data

    # Engineer features for full dataset
    holdout_features_df, holdout_target, _ = engineer_features(
        holdout_historical_df,
        holdout_fixtures_df,
        holdout_teams_df,
        holdout_ownership_df,
        holdout_value_df,
        holdout_fixture_diff_df,
        holdout_betting_df,
        holdout_raw_players_df,
        holdout_derived_player_metrics_df,
        holdout_player_availability_snapshot_df,
        holdout_derived_team_form_df,
        holdout_players_enhanced_df,
        holdout_fixture_runs_df,
        verbose=False,
    )

    # Filter to holdout gameweeks only
    holdout_mask = holdout_features_df["gameweek"].isin(
        range(holdout_start_gw, config["end_gw"] + 1)
    )
    holdout_features_filtered = holdout_features_df[holdout_mask].copy()

    if len(holdout_features_filtered) == 0:
        logger.warning("   ⚠️  No holdout data found")
        holdout_metrics = None
    else:
        # Add metadata columns for evaluation
        holdout_features_filtered["_original_index"] = holdout_features_filtered.index
        holdout_features_filtered["_fold"] = -1  # Mark as holdout

        X_holdout = holdout_features_filtered[all_feature_names]
        y_holdout = holdout_target[holdout_features_filtered.index]

        # Predict on holdout
        y_holdout_pred = best_pipeline.predict(X_holdout)

        # Evaluate
        holdout_metrics = evaluate_fpl_comprehensive(
            y_true=y_holdout,
            y_pred=y_holdout_pred,
            cv_data=holdout_features_filtered,
            verbose=True,
        )

    # 8. Summary comparison
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<25} {'Training (CV)':<20} {'Holdout':<20}")
    print("-" * 80)

    metrics_to_compare = ["mae", "rmse", "spearman_correlation"]
    for metric in metrics_to_compare:
        train_val = train_metrics.get(metric, 0)
        holdout_val = holdout_metrics.get(metric, 0) if holdout_metrics else 0

        if metric == "spearman_correlation":
            print(f"{metric:<25} {train_val:>19.3f} {holdout_val:>19.3f}")
        else:
            print(f"{metric:<25} {train_val:>19.3f} {holdout_val:>19.3f}")

    # 9. Save best hyperparameters for use in train mode
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params_filename = f"best_params_{config['regressor']}_gw{config['start_gw']}-{train_end_gw}_{timestamp}.json"
    params_path = output_dir / params_filename

    # Save best hyperparameters and config for easy reuse
    best_params_data = {
        "best_params": search_results["best_params"],
        "best_score": search_results["best_score"],
        "config": {
            "regressor": config["regressor"],
            "feature_selection": config["feature_selection"],
            "preprocessing": config["preprocessing"],
            "keep_penalty_features": config["keep_penalty_features"],
            "scorer": config["scorer"],
            "selected_features_count": len(selected_features),
            "train_end_gw": train_end_gw,
            "holdout_gws": holdout_gws,
        },
        "selected_features": selected_features,
    }

    with open(params_path, "w") as f:
        json.dump(best_params_data, f, indent=2)

    logger.info(
        "\n✅ Evaluation complete! If results look good, run 'train' command to deploy."
    )
    logger.info(f"   Command: train --end-gw {config['end_gw']} [same other args]")
    logger.info(f"\n💾 Best hyperparameters saved to: {params_path.name}")
    logger.info("   To reuse these params in train mode, use:")
    logger.info(f"   train --use-best-params-from {params_path.name} [other args]")

    return best_pipeline, selected_features, train_metrics, holdout_metrics


@app.command("train")
def run_train_mode(
    start_gw: int = typer.Option(1, "--start-gw", help="Training start gameweek"),
    end_gw: int = typer.Option(9, "--end-gw", help="Training end gameweek"),
    regressor: Literal[
        "xgboost",
        "lightgbm",
        "random-forest",
        "gradient-boost",
        "adaboost",
        "ridge",
        "lasso",
        "elasticnet",
    ] = typer.Option(
        "xgboost",
        "--regressor",
        help="Regressor to use (ignored if --use-best-params-from is provided, uses saved config)",
    ),
    feature_selection: Literal["none", "correlation", "permutation", "rfe-smart"] = (
        typer.Option(
            "none",
            "--feature-selection",
            help="Feature selection strategy (ignored if --use-best-params-from is provided, uses saved config). Default: none = keep all 156 features",
        )
    ),
    keep_penalty_features: bool = typer.Option(
        False,
        "--keep-penalty-features",
        help="Force keep penalty/set-piece features (ignored if --use-best-params-from is provided, uses saved config)",
    ),
    preprocessing: Literal["standard", "grouped", "robust"] = typer.Option(
        "standard",
        "--preprocessing",
        help=(
            "Preprocessing strategy (ignored if --use-best-params-from is provided, uses saved config): "
            "'standard' (StandardScaler for all features), "
            "'grouped' (feature-type specific scalers), 'robust' (RobustScaler for all features)"
        ),
    ),
    n_trials: int = typer.Option(
        20,
        "--n-trials",
        help="Number of hyperparameter trials (ignored if --use-best-params-from is provided)",
    ),
    cv_folds: int | None = typer.Option(
        None,
        "--cv-folds",
        help="Number of CV folds for hyperparameter search (default: None = all available, ignored if --use-best-params-from is provided)",
    ),
    scorer: Literal[
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "spearman",
        "fpl_weighted_huber",
        "fpl_top_k_ranking",
        "fpl_captain_pick",
        "fpl_hauler_capture",
        "fpl_hauler_ceiling",
        "fpl_position_aware",
        "fpl_starting_xi",
        "fpl_comprehensive",
    ] = typer.Option(
        "fpl_weighted_huber",
        "--scorer",
        help="Scoring metric (ignored if --use-best-params-from is provided, uses saved config). FPL scorers optimize for strategic objectives.",
    ),
    output_dir: str = typer.Option(
        "models/custom", "--output-dir", help="Output directory"
    ),
    use_best_params_from: str | None = typer.Option(
        None,
        "--use-best-params-from",
        help="Path to JSON file with best hyperparameters from evaluate mode (skips search)",
    ),
    random_seed: int = typer.Option(42, "--random-seed", help="Random seed"),
    n_jobs: int = typer.Option(-1, "--n-jobs", help="Parallel jobs (-1 = all CPUs)"),
    verbose: int = typer.Option(2, "--verbose", help="Verbosity level (0, 1, or 2)"),
):
    """
    Training mode: Train on all data and save model for deployment.

    If --use-best-params-from is provided, loads saved configuration (regressor,
    feature_selection, preprocessing, scorer, hyperparameters) from the JSON file.
    Command-line arguments can override specific values if needed.
    """
    # Load saved config from JSON if provided (before building config)
    saved_config_data = None
    if use_best_params_from:
        params_path = Path(use_best_params_from)
        if not params_path.is_absolute():
            # If relative, check in output_dir first, then current directory
            temp_output_dir = output_dir if output_dir else "models/custom"
            params_path = Path(temp_output_dir) / params_path
            if not params_path.exists():
                params_path = Path(use_best_params_from)

        if not params_path.exists():
            typer.echo(
                f"Error: Best params file not found: {params_path}",
                err=True,
            )
            raise typer.Exit(code=1)

        with open(params_path) as f:
            saved_config_data = json.load(f)

        saved_config = saved_config_data.get("config", {})

        # Use saved config values for model configuration (user can still override gameweeks)
        if saved_config.get("regressor"):
            regressor = saved_config["regressor"]
        if saved_config.get("feature_selection"):
            feature_selection = saved_config["feature_selection"]
        if saved_config.get("preprocessing"):
            preprocessing = saved_config["preprocessing"]
        if saved_config.get("scorer"):
            scorer = saved_config["scorer"]
        if saved_config.get("keep_penalty_features") is not None:
            keep_penalty_features = saved_config["keep_penalty_features"]

    # Build configuration dictionary
    config = build_config(
        start_gw,
        end_gw,
        regressor,
        feature_selection,
        keep_penalty_features,
        preprocessing,
        n_trials,
        cv_folds,
        scorer,
        output_dir,
        random_seed,
        n_jobs,
        verbose,
    )

    logger.info("=" * 80)
    logger.info("🎯 TRAIN MODE: Full Training for Deployment")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")

    # Load best params if provided
    best_params = None
    selected_features_from_json = None

    if use_best_params_from and saved_config_data:
        saved_config = saved_config_data.get("config", {})
        best_params = saved_config_data["best_params"]
        selected_features_from_json = saved_config_data.get("selected_features")

        logger.info(f"   📁 Loaded from: {Path(use_best_params_from).name}")
        logger.info(f"   Regressor: {config['regressor']} (from saved config)")
        logger.info(
            f"   Feature selection: {config['feature_selection']} (from saved config)"
        )
        logger.info(f"   Preprocessing: {config['preprocessing']} (from saved config)")
        logger.info(f"   Scorer: {config['scorer']} (from saved config)")
        logger.info(
            f"   Keep penalty features: {config['keep_penalty_features']} (from saved config)"
        )
        logger.info(f"   Gameweeks: GW{config['start_gw']}-{config['end_gw']}")
        logger.info("   Using saved hyperparameters (skipping search)")
        logger.info(
            f"   Best CV score from evaluation: {saved_config_data.get('best_score', 'N/A'):.4f}"
        )
    else:
        logger.info(f"   Regressor: {config['regressor']}")
        logger.info(f"   Feature selection: {config['feature_selection']}")
        logger.info(f"   Preprocessing: {config['preprocessing']}")
        logger.info(f"   Keep penalty features: {config['keep_penalty_features']}")
        logger.info(f"   Gameweeks: GW{config['start_gw']}-{config['end_gw']}")
        logger.info(f"   Hyperparameter trials: {config['n_trials']}")
        logger.info(f"   Scorer: {config['scorer']}")

    # 1. Load data (reusable utility)
    data = load_training_data(config["start_gw"], config["end_gw"], verbose=True)
    (
        historical_df,
        fixtures_df,
        teams_df,
        ownership_df,
        value_df,
        fixture_diff_df,
        betting_df,
        raw_players_df,
        derived_player_metrics_df,
        player_availability_snapshot_df,
        derived_team_form_df,
        players_enhanced_df,
        fixture_runs_df,
    ) = data

    # 2. Engineer features (reusable utility)
    features_df, target, all_feature_names = engineer_features(
        historical_df,
        fixtures_df,
        teams_df,
        ownership_df,
        value_df,
        fixture_diff_df,
        betting_df,
        raw_players_df,
        derived_player_metrics_df,
        player_availability_snapshot_df,
        derived_team_form_df,
        players_enhanced_df,
        fixture_runs_df,
        verbose=True,
    )

    # 3. Prepare training data (use ALL data, no CV splits)
    X = features_df[all_feature_names].copy()
    y = target

    # 4. Feature selection
    logger.info("\n🔧 Feature Selection...")

    # Use saved selected_features from JSON if available (to ensure exact same features)
    if selected_features_from_json is not None:
        logger.info(
            f"   Using saved feature selection from JSON ({len(selected_features_from_json)} features)"
        )
        # Verify all saved features exist
        missing_features = set(selected_features_from_json) - set(all_feature_names)
        if missing_features:
            typer.echo(
                f"Warning: {len(missing_features)} saved features not found in current data. "
                f"Will re-run feature selection.",
                err=True,
            )
            selected_features = select_features(
                X,
                y,
                all_feature_names,
                config["feature_selection"],
                config["keep_penalty_features"],
                verbose=True,
            )
        else:
            selected_features = selected_features_from_json
    else:
        selected_features = select_features(
            X,
            y,
            all_feature_names,
            config["feature_selection"],
            config["keep_penalty_features"],
            verbose=True,
        )

    # 5. Build and train pipeline
    X_final = features_df[all_feature_names].copy()
    y_final = target

    if best_params is None:
        # Need to run hyperparameter search (uses CV internally)
        logger.info("\n🔍 Running hyperparameter optimization...")
        cv_splits_for_search, cv_data_for_search = create_temporal_cv_splits(
            features_df, max_folds=config["cv_folds"], verbose=True
        )
        X_search = cv_data_for_search[all_feature_names].copy()
        y_search = target[cv_data_for_search["_original_index"].values]

        best_pipeline, search_results = optimize_pipeline(
            X_search,
            y_search,
            cv_splits_for_search,
            selected_features,
            config,
            cv_data=cv_data_for_search,
            best_params=None,
        )

        # Now train final pipeline on ALL data with best params
        logger.info("\n🎯 Training final pipeline on 100% of available data...")
        best_pipeline.set_params(**search_results["best_params"])
        best_pipeline.fit(X_final, y_final)
    else:
        # Use provided best params, skip search entirely
        logger.info(
            "\n🎯 Training pipeline on 100% of available data with provided hyperparameters..."
        )
        # Build pipeline directly
        regressor, _ = get_regressor_and_param_grid(
            config["regressor"], config["random_seed"]
        )
        preprocessor = create_preprocessor(config["preprocessing"], selected_features)

        best_pipeline = Pipeline(
            [
                ("feature_selector", FeatureSelector(selected_features)),
                ("preprocessor", preprocessor),
                ("regressor", regressor),
            ]
        )

        # Set and fit with best params
        best_pipeline.set_params(**best_params)
        best_pipeline.fit(X_final, y_final)

        search_results = {
            "best_params": best_params,
            "best_score": None,
            "cv_results": None,
        }

    # Get predictions on all training data
    y_pred = best_pipeline.predict(X_final)

    # 8. Diagnostics (instead of evaluation)
    print("\n" + "=" * 80)
    print("📊 MODEL DIAGNOSTICS")
    print("=" * 80)

    diagnostics = {}

    # Overall predicted xP distribution
    print("\n📈 Predicted xP Distribution:")
    print(f"   Min:    {y_pred.min():.2f}")
    print(f"   Max:    {y_pred.max():.2f}")
    print(f"   Mean:   {y_pred.mean():.2f}")
    print(f"   Median: {np.median(y_pred):.2f}")
    print(f"   Std:    {y_pred.std():.2f}")

    diagnostics["pred_xp"] = {
        "min": float(y_pred.min()),
        "max": float(y_pred.max()),
        "mean": float(y_pred.mean()),
        "median": float(np.median(y_pred)),
        "std": float(y_pred.std()),
    }

    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\n📊 Predicted xP Percentiles:")
    for p in percentiles:
        val = np.percentile(y_pred, p)
        print(f"   {p:2d}th: {val:.2f}")
        diagnostics["pred_xp"][f"p{p}"] = float(val)

    # High-scoring predictions
    high_scores = {
        ">=10": (y_pred >= 10).sum(),
        ">=12": (y_pred >= 12).sum(),
        ">=15": (y_pred >= 15).sum(),
        ">=20": (y_pred >= 20).sum(),
    }
    print("\n🎯 High-Scoring Predictions:")
    for threshold, count in high_scores.items():
        pct = 100 * count / len(y_pred)
        print(f"   {threshold:>4s} xP: {count:>5d} players ({pct:>5.1f}%)")
        diagnostics[f"high_score_{threshold}"] = int(count)

    # By position (if available)
    if "position" in features_df.columns:
        print("\n📊 Predicted xP by Position:")
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            pos_mask = features_df["position"] == pos
            if pos_mask.sum() > 0:
                pos_pred = y_pred[pos_mask]
                print(
                    f"   {pos}: Mean {pos_pred.mean():.2f} | "
                    f"Max {pos_pred.max():.2f} | "
                    f"Count {pos_mask.sum()}"
                )
                diagnostics[f"{pos}_pred_xp"] = {
                    "mean": float(pos_pred.mean()),
                    "max": float(pos_pred.max()),
                    "count": int(pos_mask.sum()),
                }

    # By gameweek
    if "gameweek" in features_df.columns:
        print("\n📅 Predicted xP by Gameweek:")
        gw_stats = []
        for gw in sorted(features_df["gameweek"].unique()):
            gw_mask = features_df["gameweek"] == gw
            gw_pred = y_pred[gw_mask]
            gw_stats.append(
                {
                    "gameweek": int(gw),
                    "mean": float(gw_pred.mean()),
                    "max": float(gw_pred.max()),
                    "count": int(gw_mask.sum()),
                }
            )
            print(
                f"   GW{gw:2d}: Mean {gw_pred.mean():.2f} | "
                f"Max {gw_pred.max():.2f} | "
                f"Players {gw_mask.sum()}"
            )
        diagnostics["by_gameweek"] = gw_stats

    # Correlation with actual points (for reference, not evaluation)
    if len(y) == len(y_pred):
        corr = np.corrcoef(y, y_pred)[0, 1]
        print(f"\n📈 Correlation with actual points: {corr:.3f}")
        diagnostics["correlation_actual"] = float(corr)

    # 9. Save pipeline
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = (
        f"{config['regressor']}_gw{config['start_gw']}-{config['end_gw']}_{timestamp}"
    )

    # Save pipeline only (for inference/deployment)
    pipeline_path = output_dir / f"{model_name}_pipeline.joblib"
    joblib.dump(best_pipeline, pipeline_path)

    # Save full metadata (for analysis/debugging)
    metadata_path = output_dir / f"{model_name}.joblib"
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "feature_names": selected_features,
            "diagnostics": diagnostics,
            "search_results": search_results,
            "config": config,
            "training_samples": len(y_final),
        },
        metadata_path,
    )

    logger.info("\n💾 Model saved:")
    logger.info(f"   Pipeline (for deployment): {pipeline_path.name}")
    logger.info(f"   Metadata (for analysis): {metadata_path.name}")
    logger.info(f"   Features: {len(selected_features)}/156")
    logger.info(f"   Training samples: {len(y_final):,}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ PIPELINE TRAINING COMPLETE!")
    logger.info("=" * 80)

    return best_pipeline, selected_features, diagnostics


def main():
    """Main entry point - Typer CLI."""
    app()


if __name__ == "__main__":
    main()
