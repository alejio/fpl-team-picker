#!/usr/bin/env python3
"""
Custom Pipeline Optimizer for FPL Expected Points Prediction

Fully configurable alternative to TPOT with:
- Manual control over feature selection (addresses RFE dropping penalty features)
- Choice of regressors (XGBoost, LightGBM, RandomForest, etc.)
- Hyperparameter optimization via RandomizedSearchCV
- Identical evaluation to TPOT (temporal CV, same metrics)

Key Difference from TPOT:
- TPOT uses RFE which dropped penalty features (perm importance rank 4-8, MDI rank 92-96)
- This pipeline KEEPS all 99 features or uses smarter feature selection

Usage:
    python scripts/custom_pipeline_optimizer.py --regressor xgboost --n-trials 20
    python scripts/custom_pipeline_optimizer.py --regressor lightgbm --feature-selection correlation
    python scripts/custom_pipeline_optimizer.py --regressor random-forest --keep-penalty-features
"""

import argparse
import sys
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import reusable training utilities
sys.path.insert(0, str(Path(__file__).parent))
from ml_training_utils import (  # noqa: E402
    load_training_data,
    engineer_features,
    create_temporal_cv_splits,
    evaluate_fpl_comprehensive,
    TemporalCVSplitter,
    spearman_correlation_scorer,
    fpl_weighted_huber_scorer_sklearn,
    fpl_topk_scorer_sklearn,
    fpl_captain_scorer_sklearn,
)

# Import custom transformers from domain layer
from fpl_team_picker.domain.ml import FeatureSelector  # noqa: E402


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Custom pipeline optimization for FPL xP prediction"
    )
    parser.add_argument(
        "--start-gw", type=int, default=1, help="Training start gameweek (default: 1)"
    )
    parser.add_argument(
        "--end-gw", type=int, default=9, help="Training end gameweek (default: 9)"
    )
    parser.add_argument(
        "--regressor",
        type=str,
        default="xgboost",
        choices=[
            "xgboost",
            "lightgbm",
            "random-forest",
            "gradient-boost",
            "adaboost",
            "ridge",
            "lasso",
            "elasticnet",
        ],
        help="Regressor to use (default: xgboost)",
    )
    parser.add_argument(
        "--feature-selection",
        type=str,
        default="none",
        choices=["none", "correlation", "permutation", "rfe-smart"],
        help="Feature selection strategy (default: none = keep all 99 features)",
    )
    parser.add_argument(
        "--keep-penalty-features",
        action="store_true",
        help="Force keep penalty/set-piece features (is_primary_penalty_taker, etc.)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of hyperparameter trials (default: 20)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Number of CV folds (default: None = all available)",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default="fpl_weighted_huber",
        choices=[
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            "spearman",
            "fpl_weighted_huber",
            "fpl_top_k_ranking",
            "fpl_captain_pick",
        ],
        help="Scoring metric (default: fpl_weighted_huber). FPL scorers optimize for strategic objectives.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/custom",
        help="Output directory (default: models/custom)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Parallel jobs (default: -1 = all CPUs)"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Verbosity (default: 2)",
    )

    return parser.parse_args()


def get_regressor_and_param_grid(regressor_name: str, random_seed: int) -> Tuple:
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

            regressor = xgb.XGBRegressor(
                random_state=random_seed, n_jobs=1, tree_method="hist"
            )
            param_dist = {
                "regressor__n_estimators": randint(100, 500),
                "regressor__max_depth": randint(3, 10),
                "regressor__learning_rate": uniform(0.01, 0.3),
                "regressor__subsample": uniform(0.6, 0.4),
                "regressor__colsample_bytree": uniform(0.6, 0.4),
                "regressor__min_child_weight": randint(1, 10),
                "regressor__gamma": uniform(0, 0.5),
                "regressor__reg_alpha": uniform(0, 1.0),
                "regressor__reg_lambda": uniform(0.5, 2.0),
            }
        except ImportError:
            raise ImportError("XGBoost not installed. Run: uv add xgboost")

    elif regressor_name == "lightgbm":
        try:
            import lightgbm as lgb

            regressor = lgb.LGBMRegressor(
                random_state=random_seed, n_jobs=1, verbose=-1
            )
            param_dist = {
                "regressor__n_estimators": randint(100, 500),
                "regressor__max_depth": randint(3, 10),
                "regressor__learning_rate": uniform(0.01, 0.3),
                "regressor__num_leaves": randint(20, 100),
                "regressor__subsample": uniform(0.6, 0.4),
                "regressor__colsample_bytree": uniform(0.6, 0.4),
                "regressor__min_child_samples": randint(10, 50),
                "regressor__reg_alpha": uniform(0, 1.0),
                "regressor__reg_lambda": uniform(0.5, 2.0),
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
        regressor = GradientBoostingRegressor(random_state=random_seed)
        param_dist = {
            "regressor__n_estimators": randint(100, 400),
            "regressor__max_depth": randint(3, 8),
            "regressor__learning_rate": uniform(0.01, 0.2),
            "regressor__subsample": uniform(0.6, 0.4),
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
    feature_names: List[str],
    strategy: str,
    keep_penalty_features: bool,
    verbose: bool = True,
) -> List[str]:
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
            print(f"   ℹ️  Using all {len(feature_names)} features (no selection)")
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
            print("   🔍 Removing highly correlated features (|r| > 0.95)...")

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
            print(
                f"   ✅ Removed {len(to_drop)} correlated features, kept {len(selected)}"
            )

    elif strategy == "permutation":
        # Use permutation importance from previous analysis
        # Keep top 60 features + penalty features if requested
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance

        if verbose:
            print("   🔍 Computing permutation importance...")

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
            print(
                f"   ✅ Selected top {len(selected)} features by permutation importance"
            )

    elif strategy == "rfe-smart":
        # RFE but force keep penalty features
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import ExtraTreesRegressor

        if verbose:
            print("   🔍 Running smart RFE (keeps penalty features)...")

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
            print(f"   ✅ RFE selected {len(selected)} features")
            if keep_penalty_features:
                print(f"      (including {len(penalty_features)} penalty features)")

    else:
        selected = feature_names

    # Ensure penalty features are included if requested
    if keep_penalty_features:
        for pf in penalty_features:
            if pf not in selected and pf in feature_names:
                selected.append(pf)
                if verbose:
                    print(f"   ➕ Force-added penalty feature: {pf}")

    return selected


def optimize_pipeline(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: List,
    feature_names: List[str],
    args: argparse.Namespace,
) -> Tuple[Pipeline, Dict]:
    """
    Optimize hyperparameters using RandomizedSearchCV.

    Args:
        X: Feature matrix
        y: Target variable
        cv_splits: CV splits
        feature_names: Selected feature names
        args: Command line arguments

    Returns:
        Tuple of (best_pipeline, search_results)
    """
    print(f"\n🤖 Building {args.regressor} pipeline...")

    # Get regressor and param grid
    regressor, param_dist = get_regressor_and_param_grid(
        args.regressor, args.random_seed
    )

    # Build pipeline: FeatureSelector → Scaler → Regressor
    # This makes the pipeline self-contained - it knows which features it needs
    pipeline = Pipeline(
        [
            ("feature_selector", FeatureSelector(feature_names)),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )

    # Select scorer
    scorer_map = {
        "spearman": spearman_correlation_scorer,
        "fpl_weighted_huber": fpl_weighted_huber_scorer_sklearn,
        "fpl_top_k_ranking": fpl_topk_scorer_sklearn,
        "fpl_captain_pick": fpl_captain_scorer_sklearn,
    }
    scorer = scorer_map.get(args.scorer, args.scorer)

    # Create CV splitter
    cv_splitter = TemporalCVSplitter(cv_splits)

    print("   Hyperparameter search:")
    print(f"      Trials: {args.n_trials}")
    print(f"      CV folds: {len(cv_splits)}")
    print(f"      Scorer: {args.scorer}")
    print(f"      Features: {len(feature_names)}")

    # Run randomized search
    print("\n🔍 Running hyperparameter optimization...")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=args.n_trials,
        cv=cv_splitter,
        scoring=scorer,
        n_jobs=args.n_jobs,
        random_state=args.random_seed,
        verbose=args.verbose,
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


def main():
    """Main execution."""
    args = parse_args()

    print("=" * 80)
    print("🎯 CUSTOM PIPELINE OPTIMIZER FOR FPL XP PREDICTION")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"   Regressor: {args.regressor}")
    print(f"   Feature selection: {args.feature_selection}")
    print(f"   Keep penalty features: {args.keep_penalty_features}")
    print(f"   Gameweeks: GW{args.start_gw}-{args.end_gw}")
    print(f"   Hyperparameter trials: {args.n_trials}")
    print(f"   Scorer: {args.scorer}")

    # 1. Load data (reusable utility)
    data = load_training_data(args.start_gw, args.end_gw, verbose=True)
    (
        historical_df,
        fixtures_df,
        teams_df,
        ownership_df,
        value_df,
        fixture_diff_df,
        betting_df,
        raw_players_df,
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
        verbose=True,
    )

    # 3. Create CV splits (reusable utility)
    cv_splits, cv_data = create_temporal_cv_splits(
        features_df, max_folds=args.cv_folds, verbose=True
    )

    # Prepare training data
    X = cv_data[all_feature_names].copy()
    # Use original index to get correct target values
    y = target[cv_data["_original_index"].values]

    # 4. Feature selection
    print("\n🔧 Feature Selection...")
    selected_features = select_features(
        X,
        y,
        all_feature_names,
        args.feature_selection,
        args.keep_penalty_features,
        verbose=True,
    )

    # 5. Optimize pipeline
    best_pipeline, search_results = optimize_pipeline(
        X, y, cv_splits, selected_features, args
    )

    # 6. Evaluate on full CV data
    print("\n📊 Final Evaluation on CV data...")
    # Pass ALL features - the FeatureSelector inside the pipeline will select the ones it needs
    y_pred = best_pipeline.predict(X)

    # Use comprehensive FPL evaluation
    metrics = evaluate_fpl_comprehensive(
        y_true=y,
        y_pred=y_pred,
        cv_data=cv_data,
        verbose=True,
    )

    # 7. Save pipeline
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.regressor}_gw{args.start_gw}-{args.end_gw}_{timestamp}"

    # Save pipeline only (for inference/deployment)
    pipeline_path = output_dir / f"{model_name}_pipeline.joblib"
    joblib.dump(best_pipeline, pipeline_path)

    # Save full metadata (for analysis/debugging)
    metadata_path = output_dir / f"{model_name}.joblib"
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "feature_names": selected_features,
            "metrics": metrics,
            "search_results": search_results,
            "config": vars(args),
        },
        metadata_path,
    )

    print("\n💾 Model saved:")
    print(f"   Pipeline (for deployment): {pipeline_path.name}")
    print(f"   Metadata (for analysis): {metadata_path.name}")
    print(f"   Features: {len(selected_features)}/99")
    print(f"   MAE: {metrics['mae']:.3f}")
    print(f"   RMSE: {metrics['rmse']:.3f}")
    print(f"   Spearman: {metrics['spearman_correlation']:.3f}")

    print("\n" + "=" * 80)
    print("✅ PIPELINE OPTIMIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
