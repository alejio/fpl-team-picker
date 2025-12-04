"""
Configuration models for ML training.

Uses Pydantic for validation and type safety.
"""

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, computed_field


class TrainingConfig(BaseModel):
    """Configuration for training a single ML model."""

    # Data range
    start_gw: int = Field(default=1, ge=1, description="Starting gameweek")
    end_gw: int = Field(default=12, ge=1, description="Ending gameweek (inclusive)")

    # Model configuration
    regressor: Literal[
        "lightgbm",
        "xgboost",
        "random-forest",
        "gradient-boost",
        "adaboost",
        "ridge",
        "lasso",
        "elasticnet",
    ] = Field(default="lightgbm", description="Regressor algorithm to use")

    # Feature engineering
    feature_selection: Literal["none", "correlation", "permutation", "rfe-smart"] = (
        Field(default="none", description="Feature selection strategy")
    )
    keep_penalty_features: bool = Field(
        default=False, description="Force keep penalty/set-piece features"
    )
    preprocessing: Literal["standard", "grouped", "robust"] = Field(
        default="standard", description="Preprocessing strategy for feature scaling"
    )

    # Hyperparameter optimization
    n_trials: int = Field(
        default=50, ge=1, description="Number of hyperparameter trials"
    )
    cv_folds: Optional[int] = Field(
        default=None, ge=2, description="CV folds (None = all available)"
    )
    scorer: str = Field(default="neg_mean_absolute_error", description="Scoring metric")

    # Output
    output_dir: Path = Field(
        default=Path("models/custom"), description="Output directory"
    )

    # Reproducibility
    random_seed: int = Field(default=42, description="Random seed")
    n_jobs: int = Field(default=-1, description="Parallel jobs (-1 = all CPUs)")
    verbose: int = Field(default=1, ge=0, le=2, description="Verbosity level")

    model_config = {"arbitrary_types_allowed": True}


class PositionTrainingConfig(TrainingConfig):
    """Configuration for training position-specific models."""

    # Positions to train
    positions: List[str] = Field(
        default=["GKP", "DEF", "MID", "FWD"],
        description="Positions to train models for",
    )

    # Regressors to try per position (will pick best)
    regressors: List[str] = Field(
        default=["lightgbm", "xgboost", "random-forest", "gradient-boost"],
        description="Regressors to try per position",
    )

    # Override output directory
    output_dir: Path = Field(
        default=Path("models/position_specific"),
        description="Output directory for position models",
    )


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    # Holdout setup
    holdout_gws: int = Field(default=2, ge=1, description="Gameweeks to hold out")

    # Metrics to compute
    compute_per_position: bool = Field(
        default=True, description="Compute per-position metrics"
    )
    compute_per_gameweek: bool = Field(
        default=True, description="Compute per-gameweek metrics"
    )

    # Captain accuracy settings
    captain_top_k: int = Field(default=3, ge=1, description="Top-K captain candidates")

    # Squad selection overlap
    squad_size: int = Field(
        default=15, ge=11, le=15, description="Squad size for overlap"
    )


class HybridPipelineConfig(BaseModel):
    """
    Configuration for the full hybrid pipeline.

    Orchestrates:
    1. Evaluation phase: Train on subset, evaluate on holdout
    2. Production phase: Retrain on all data with best config
    """

    # Data range
    end_gw: int = Field(default=12, ge=6, description="Final gameweek")
    holdout_gws: int = Field(default=2, ge=1, description="Gameweeks to hold out")

    # Regressors to try
    unified_regressors: List[str] = Field(
        default=["lightgbm", "xgboost", "random-forest", "gradient-boost"],
        description="Regressors to try for unified model",
    )
    position_regressors: List[str] = Field(
        default=["lightgbm", "xgboost", "random-forest", "gradient-boost"],
        description="Regressors to try for position-specific models",
    )

    # Optimization
    n_trials: int = Field(
        default=50, ge=1, description="Hyperparameter trials per regressor"
    )
    scorer: str = Field(default="neg_mean_absolute_error", description="Scoring metric")

    # Hybrid selection
    improvement_threshold: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Minimum improvement (fraction) to use position-specific model",
    )

    # Output
    output_dir: Path = Field(
        default=Path("models"), description="Base output directory"
    )

    # Reproducibility
    random_seed: int = Field(default=42, description="Random seed")

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @property
    def train_end_gw(self) -> int:
        """Gameweek to end evaluation training (before holdout)."""
        return self.end_gw - self.holdout_gws

    @computed_field
    @property
    def holdout_gw_range(self) -> List[int]:
        """List of holdout gameweeks."""
        return list(range(self.train_end_gw + 1, self.end_gw + 1))
