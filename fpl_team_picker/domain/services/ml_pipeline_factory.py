"""
Factory for creating scikit-learn pipelines for FPL expected points prediction.

This module provides factory functions to build complete ML pipelines that can be:
- Trained on historical data
- Saved to disk (joblib)
- Loaded for production predictions
- Integrated with gameweek_manager.py
"""

import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

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
        model_type: 'lgbm' (LightGBM - recommended), 'rf', 'gb', or 'ridge'
        fixtures_df: Fixture data for fixture-specific features
        teams_df: Teams data for opponent context
        team_strength: Team strength ratings dict
        **model_kwargs: Model hyperparameters (n_estimators, max_depth, etc.)

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
    if model_type == "lgbm":
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
            f"Unknown model_type: {model_type}. Choose 'lgbm', 'rf', 'gb', or 'ridge'"
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

    print(f"✅ Pipeline saved to {path}")
    if metadata:
        print(f"✅ Metadata saved to {metadata_path}")


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


def get_team_strength_ratings() -> Dict[str, float]:
    """
    Get team strength ratings based on 2023-24 final table positions.

    Returns:
        Dict mapping team name to strength rating [0.65, 1.30]
    """
    team_positions = {
        "Manchester City": 1,
        "Arsenal": 2,
        "Liverpool": 3,
        "Aston Villa": 4,
        "Tottenham": 5,
        "Chelsea": 6,
        "Newcastle": 7,
        "Manchester Utd": 8,
        "West Ham": 9,
        "Crystal Palace": 10,
        "Brighton": 11,
        "Bournemouth": 12,
        "Fulham": 13,
        "Wolves": 14,
        "Everton": 15,
        "Brentford": 16,
        "Nottingham Forest": 17,
        "Luton": 18,
        "Burnley": 19,
        "Sheffield Utd": 20,
        # Promoted teams
        "Ipswich": 21,
        "Leicester": 21,
        "Southampton": 21,
        # Aliases
        "Man City": 1,
        "Man Utd": 8,
        "Nott'm Forest": 17,
        "Spurs": 5,
    }

    strength_ratings = {}
    for team, position in team_positions.items():
        if position <= 20:
            strength = 1.3 - (position - 1) * (1.3 - 0.7) / 19
        else:
            strength = 0.65
        strength_ratings[team] = round(strength, 3)

    return strength_ratings


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
    from sklearn.model_selection import GroupKFold, cross_val_score
    import numpy as np

    # Get team strength ratings
    team_strength = get_team_strength_ratings()

    # Create pipeline
    pipeline = create_fpl_pipeline(
        model_type=model_type,
        fixtures_df=fixtures_df,
        teams_df=teams_df,
        team_strength=team_strength,
        **model_kwargs,
    )

    # Prepare data (GW6+ for complete rolling features)
    train_data = historical_df[historical_df["gameweek"] >= 6].copy()

    X = train_data
    y = train_data["total_points"]
    groups = train_data["player_id"].values

    # Cross-validation
    gkf = GroupKFold(n_splits=5)
    mae_scores = -cross_val_score(
        pipeline,
        X,
        y,
        groups=groups,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    rmse_scores = np.sqrt(
        -cross_val_score(
            pipeline,
            X,
            y,
            groups=groups,
            cv=gkf,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
    )
    r2_scores = cross_val_score(
        pipeline, X, y, groups=groups, cv=gkf, scoring="r2", n_jobs=-1
    )

    # Train on full data
    pipeline.fit(X, y)

    # Prepare metadata
    import datetime

    metadata = {
        "model_type": model_type,
        "training_date": datetime.datetime.now().isoformat(),
        "cv_mae_mean": float(mae_scores.mean()),
        "cv_mae_std": float(mae_scores.std()),
        "cv_rmse_mean": float(rmse_scores.mean()),
        "cv_rmse_std": float(rmse_scores.std()),
        "cv_r2_mean": float(r2_scores.mean()),
        "cv_r2_std": float(r2_scores.std()),
        "n_folds": 5,
        "n_features": 63,
        "training_samples": len(train_data),
        "training_players": int(train_data["player_id"].nunique()),
        "gameweek_range": f"GW{train_data['gameweek'].min()}-{train_data['gameweek'].max()}",
        **model_kwargs,
    }

    # Save pipeline
    save_pipeline(pipeline, Path(save_path), metadata)

    print("\n✅ Model training complete!")
    print(f"   MAE: {metadata['cv_mae_mean']:.3f} ± {metadata['cv_mae_std']:.3f}")
    print(f"   RMSE: {metadata['cv_rmse_mean']:.3f} ± {metadata['cv_rmse_std']:.3f}")
    print(f"   R²: {metadata['cv_r2_mean']:.3f} ± {metadata['cv_r2_std']:.3f}")

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
        print(f"📊 Using {metadata.get('model_type', 'Unknown')} model")
        print(f"   Trained on {metadata.get('training_date', 'Unknown')}")
        print(f"   CV MAE: {metadata.get('cv_mae_mean', 0):.3f}")

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
