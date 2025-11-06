"""
Factory for creating scikit-learn pipelines for FPL expected points prediction.

This module provides factory functions to build complete ML pipelines that can be:
- Trained on historical data
- Saved to disk (joblib)
- Loaded for production predictions
- Integrated with gameweek_manager.py
"""

# TODO: many of these features could be created upstream
# for analytics purposes

import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from .ml_feature_engineering import FPLFeatureEngineer


def create_fpl_pipeline(
    model_type: str = "rf",
    fixtures_df: Optional[pd.DataFrame] = None,
    teams_df: Optional[pd.DataFrame] = None,
    team_strength: Optional[Dict[str, float]] = None,
    **model_kwargs,
) -> Pipeline:
    """
    Create a complete FPL prediction pipeline.

    Args:
        model_type: 'catboost' (CatBoost - best), 'lgbm', 'rf', 'gb', or 'ridge'
        fixtures_df: Fixture data for fixture-specific features
        teams_df: Teams data for opponent context
        team_strength: Team strength ratings dict
        **model_kwargs: Model hyperparameters (iterations, depth, learning_rate, etc.)

    Returns:
        sklearn Pipeline with [feature_engineer -> scaler -> model]
    """
    # Create feature engineering transformer
    feature_engineer = FPLFeatureEngineer(
        fixtures_df=fixtures_df,
        teams_df=teams_df,
        team_strength=team_strength,
    )

    # Select model
    if model_type == "catboost":
        # CatBoost with optimal hyperparameters (24.2% better MAE vs Ridge, ~1.06 MAE)
        # See: experiments/hyperparameter_tuning_valid_folds.py
        # Optimal config: iterations=200, depth=4, lr=0.1, l2=5, bagging_temp=1.0, random_strength=0
        model = CatBoostRegressor(
            iterations=model_kwargs.get("iterations", 200),
            depth=model_kwargs.get("depth", 4),
            learning_rate=model_kwargs.get("learning_rate", 0.1),
            l2_leaf_reg=model_kwargs.get("l2_leaf_reg", 5),
            bagging_temperature=model_kwargs.get("bagging_temperature", 1.0),
            random_strength=model_kwargs.get("random_strength", 0),
            loss_function=model_kwargs.get(
                "loss_function", "RMSE"
            ),  # RMSE works better than MAE
            random_seed=model_kwargs.get("random_seed", 42),
            verbose=0,  # Suppress training output
            thread_count=-1,  # Use all CPU cores
        )
        steps = [
            ("feature_engineer", feature_engineer),
            ("model", model),
        ]
    elif model_type == "lgbm":
        # LightGBM with optimal hyperparameters (4.3% better MAE vs Ridge)
        # See: experiments/hyperparameter_tuning_valid_folds.py
        model = LGBMRegressor(
            n_estimators=model_kwargs.get("n_estimators", 50),
            max_depth=model_kwargs.get("max_depth", 5),
            learning_rate=model_kwargs.get("learning_rate", 0.1),
            min_child_samples=model_kwargs.get("min_child_samples", 20),
            subsample=model_kwargs.get("subsample", 0.9),
            colsample_bytree=model_kwargs.get("colsample_bytree", 0.9),
            random_state=model_kwargs.get("random_state", 42),
            n_jobs=-1,
            verbosity=-1,  # Suppress warnings
        )
        steps = [
            ("feature_engineer", feature_engineer),
            ("model", model),
        ]
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=model_kwargs.get("n_estimators", 100),
            max_depth=model_kwargs.get("max_depth", 10),
            min_samples_split=model_kwargs.get("min_samples_split", 10),
            random_state=model_kwargs.get("random_state", 42),
            n_jobs=-1,
        )
        steps = [
            ("feature_engineer", feature_engineer),
            ("model", model),
        ]
    elif model_type == "gb":
        model = GradientBoostingRegressor(
            n_estimators=model_kwargs.get("n_estimators", 100),
            max_depth=model_kwargs.get("max_depth", 5),
            learning_rate=model_kwargs.get("learning_rate", 0.1),
            random_state=model_kwargs.get("random_state", 42),
        )
        steps = [
            ("feature_engineer", feature_engineer),
            ("model", model),
        ]
    elif model_type == "ridge":
        model = Ridge(
            alpha=model_kwargs.get("alpha", 1.0),
            random_state=model_kwargs.get("random_state", 42),
        )
        # Ridge needs scaling
        steps = [
            ("feature_engineer", feature_engineer),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Choose 'catboost', 'lgbm', 'rf', 'gb', or 'ridge'"
        )

    pipeline = Pipeline(steps)

    return pipeline


def save_pipeline(
    pipeline: Pipeline,
    path: Path,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save a trained pipeline to disk with metadata.

    Args:
        pipeline: Trained sklearn Pipeline
        path: Path to save the pipeline (e.g., 'models/fpl_xp_rf.joblib')
        metadata: Optional dict with training info (date, MAE, features, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save pipeline
    joblib.dump(pipeline, path)

    # Save metadata alongside if provided
    if metadata:
        metadata_path = path.with_suffix(".metadata.json")
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    logger.info(f"✅ Pipeline saved to {path}")
    if metadata:
        logger.info(f"✅ Metadata saved to {metadata_path}")


def load_pipeline(path: Path) -> Tuple[Pipeline, Optional[Dict]]:
    """
    Load a trained pipeline from disk with metadata.

    Args:
        path: Path to saved pipeline

    Returns:
        Tuple of (pipeline, metadata dict or None)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Pipeline not found at {path}")

    # Load pipeline
    pipeline = joblib.load(path)

    # Load metadata if exists
    metadata_path = path.with_suffix(".metadata.json")
    metadata = None
    if metadata_path.exists():
        import json

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return pipeline, metadata


def get_team_strength_ratings(
    target_gameweek: int = 1,
    teams_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Get dynamic team strength ratings using TeamAnalyticsService.

    DEPRECATED: This wrapper is kept for backward compatibility with existing code.
    New code should use TeamAnalyticsService directly.

    Args:
        target_gameweek: Target gameweek for strength calculation (default: 1 = early season baseline)
        teams_df: Teams data (if None, will load from FPLDataClient)

    Returns:
        Dict mapping team name to strength rating [0.7, 1.3]
    """
    from .team_analytics_service import TeamAnalyticsService

    # Load teams data if not provided
    if teams_df is None or teams_df.empty:
        from client import FPLDataClient

        client = FPLDataClient()
        teams_df = client.get_current_teams()

    # Use TeamAnalyticsService for dynamic calculation
    service = TeamAnalyticsService(debug=False)
    return service.get_team_strength(
        target_gameweek=target_gameweek,
        teams_data=teams_df,
        current_season_data=None,
    )


# Example usage functions for notebook/gameweek_manager integration


def train_and_save_model(
    historical_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    model_type: str = "rf",
    save_path: str = "models/fpl_xp_rf.joblib",
    **model_kwargs,
) -> Tuple[Pipeline, Dict]:
    """
    Train a model on historical data and save it.

    Args:
        historical_df: Historical player performance data
        fixtures_df: Fixture data
        teams_df: Teams data
        model_type: Model type ('rf', 'gb', 'ridge')
        save_path: Where to save the trained model
        **model_kwargs: Model hyperparameters

    Returns:
        Tuple of (trained_pipeline, training_metrics)
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    # Calculate per-gameweek team strength (no data leakage)
    # For GW N, uses team strength calculated from GW 1 to N-1
    from fpl_team_picker.domain.services.ml_feature_engineering import (
        calculate_per_gameweek_team_strength,
    )

    start_gw = 6  # First trainable gameweek (needs GW1-5 for rolling features)
    end_gw = historical_df["gameweek"].max()
    team_strength = calculate_per_gameweek_team_strength(
        start_gw=start_gw,
        end_gw=end_gw,
        teams_df=teams_df,
    )

    # Create pipeline
    pipeline = create_fpl_pipeline(
        model_type=model_type,
        fixtures_df=fixtures_df,
        teams_df=teams_df,
        team_strength=team_strength,
        **model_kwargs,
    )

    # Prepare data (GW6+ for complete rolling features)
    train_data = (
        historical_df[historical_df["gameweek"] >= 6].copy().reset_index(drop=True)
    )

    # Temporal walk-forward cross-validation (matches hyperparameter tuning)
    # Only use folds where test GW has ≥5 GW history (test GW ≥ 6)
    gws = sorted(train_data["gameweek"].unique())
    temporal_splits = []

    for i in range(len(gws) - 1):
        train_gws = gws[: i + 1]
        test_gw = gws[i + 1]

        # Only include folds where test GW ≥ 6 (has proper 5GW history)
        if test_gw >= 6:
            train_mask = train_data["gameweek"].isin(train_gws)
            test_mask = train_data["gameweek"] == test_gw

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            temporal_splits.append((train_idx, test_idx))

    # Cross-validation with temporal splits
    mae_scores = []
    rmse_scores = []
    r2_scores = []

    for train_idx, test_idx in temporal_splits:
        X_train = train_data.iloc[train_idx]
        X_test = train_data.iloc[test_idx]
        y_train = train_data.iloc[train_idx]["total_points"]
        y_test = train_data.iloc[test_idx]["total_points"]

        # Clone and fit pipeline
        from sklearn.base import clone

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)

        # Predict and score
        y_pred = fold_pipeline.predict(X_test)
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_scores.append(r2_score(y_test, y_pred))

    mae_scores = np.array(mae_scores)
    rmse_scores = np.array(rmse_scores)
    r2_scores = np.array(r2_scores)

    # Train on full data
    pipeline.fit(train_data, train_data["total_points"])

    # Prepare metadata
    import datetime

    metadata = {
        "model_type": model_type,
        "training_date": datetime.datetime.now().isoformat(),
        "cv_strategy": "temporal_walkforward",
        "cv_mae_mean": float(mae_scores.mean()),
        "cv_mae_std": float(mae_scores.std()),
        "cv_rmse_mean": float(rmse_scores.mean()),
        "cv_rmse_std": float(rmse_scores.std()),
        "cv_r2_mean": float(r2_scores.mean()),
        "cv_r2_std": float(r2_scores.std()),
        "n_folds": len(temporal_splits),
        "n_features": 65,  # Updated to match FPLFeatureEngineer output
        "training_samples": len(train_data),
        "training_players": int(train_data["player_id"].nunique()),
        "gameweek_range": f"GW{train_data['gameweek'].min()}-{train_data['gameweek'].max()}",
        **model_kwargs,
    }

    # Save pipeline
    save_pipeline(pipeline, Path(save_path), metadata)

    logger.info("\n✅ Model training complete!")
    logger.info(f"   MAE: {metadata['cv_mae_mean']:.3f} ± {metadata['cv_mae_std']:.3f}")
    logger.info(
        f"   RMSE: {metadata['cv_rmse_mean']:.3f} ± {metadata['cv_rmse_std']:.3f}"
    )
    logger.info(f"   R²: {metadata['cv_r2_mean']:.3f} ± {metadata['cv_r2_std']:.3f}")

    return pipeline, metadata


def predict_gameweek(
    pipeline_path: str,
    current_players_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Use a trained pipeline to predict xP for current gameweek.

    This is the function gameweek_manager.py will call.

    Args:
        pipeline_path: Path to saved pipeline
        current_players_df: Current player data (will be used for feature engineering)
        fixtures_df: Upcoming fixtures
        teams_df: Teams data

    Returns:
        DataFrame with player_id and predicted_xp columns
    """
    # Load pipeline
    pipeline, metadata = load_pipeline(Path(pipeline_path))

    if metadata:
        logger.info(f"📊 Using {metadata.get('model_type', 'Unknown')} model")
        logger.info(f"   Trained on {metadata.get('training_date', 'Unknown')}")
        logger.info(f"   CV MAE: {metadata.get('cv_mae_mean', 0):.3f}")

    # Make predictions
    predictions = pipeline.predict(current_players_df)

    # Return as DataFrame
    result_df = pd.DataFrame(
        {
            "player_id": current_players_df["player_id"],
            "predicted_xp": predictions,
        }
    )

    return result_df
