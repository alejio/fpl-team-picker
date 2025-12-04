"""
Hybrid Position Model for FPL Expected Points Prediction.

Routes predictions to position-specific or unified models based on
empirical performance analysis:
- GKP, FWD: Position-specific models (significant gains: +6.5%, +9.0%)
- DEF, MID: Unified model (benefits from larger training data)

This class implements the sklearn estimator interface (predict method)
so it can be used as a drop-in replacement for any sklearn Pipeline.
"""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


# Position-specific feature engineering additions
# These are interaction features that capture position-specific scoring patterns
# Must match those used during training in position_specific_optimizer.py
POSITION_FEATURE_ADDITIONS: dict[str, list[tuple[str, Callable]]] = {
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


def add_position_features(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """
    Add position-specific engineered features to a DataFrame.

    Args:
        df: DataFrame with base features
        position: Position to add features for ("GKP", "DEF", "MID", "FWD")

    Returns:
        DataFrame with added position-specific features
    """
    df = df.copy()

    additions = POSITION_FEATURE_ADDITIONS.get(position, [])
    for feature_name, feature_func in additions:
        if feature_name not in df.columns:
            try:
                df[feature_name] = feature_func(df)
            except Exception:
                df[feature_name] = 0.0

    return df


class HybridPositionModel:
    """
    Hybrid model that routes predictions based on player position.

    Uses position-specific models for positions where they outperform
    the unified model, and falls back to unified model for others.

    Implements sklearn's predict interface for compatibility with
    MLExpectedPointsService.

    Attributes:
        unified_model: Pipeline for positions not in use_specific_for
        position_models: Dict mapping position to Pipeline
        use_specific_for: List of positions to use specific models for

    Example:
        >>> hybrid = HybridPositionModel(
        ...     unified_model=unified_pipeline,
        ...     position_models={"GKP": gkp_pipeline, "FWD": fwd_pipeline},
        ...     use_specific_for=["GKP", "FWD"]
        ... )
        >>> predictions = hybrid.predict(features_df)
    """

    def __init__(
        self,
        unified_model: Pipeline,
        position_models: dict[str, Pipeline],
        use_specific_for: list[str] | None = None,
    ):
        """
        Initialize hybrid model.

        Args:
            unified_model: Pre-trained unified pipeline (for DEF, MID by default)
            position_models: Dict of {position: Pipeline} for position-specific models
            use_specific_for: Positions to use specific models for.
                             Default: ["GKP", "FWD"] based on experiment results.
        """
        self.unified_model = unified_model
        self.position_models = position_models
        self.use_specific_for = use_specific_for or ["GKP", "FWD"]

        # Validate that we have models for all specified positions
        for pos in self.use_specific_for:
            if pos not in self.position_models:
                raise ValueError(
                    f"Position '{pos}' in use_specific_for but no model provided. "
                    f"Available: {list(self.position_models.keys())}"
                )

        # Store metadata for introspection
        self._metadata = {
            "use_specific_for": self.use_specific_for,
            "unified_positions": [
                p
                for p in ["GKP", "DEF", "MID", "FWD"]
                if p not in self.use_specific_for
            ],
            "position_model_types": {
                pos: type(model.named_steps.get("regressor", model)).__name__
                for pos, model in self.position_models.items()
            },
            "unified_model_type": type(
                self.unified_model.named_steps.get("regressor", self.unified_model)
            ).__name__,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions by routing to appropriate model based on position.

        Automatically adds position-specific features if the model requires them
        and they are not present in the input data.

        Args:
            X: Feature DataFrame with 'position' column

        Returns:
            Array of predictions aligned with input order
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"HybridPositionModel requires DataFrame input with 'position' column, "
                f"got {type(X).__name__}"
            )

        if "position" not in X.columns:
            raise ValueError(
                "Input DataFrame must contain 'position' column for routing. "
                "Ensure the position column is preserved through feature engineering."
            )

        # Initialize predictions array
        predictions = np.zeros(len(X))

        # Track which rows have been predicted
        predicted_mask = np.zeros(len(X), dtype=bool)

        for position in ["GKP", "DEF", "MID", "FWD"]:
            # Find rows for this position
            pos_mask = X["position"] == position

            if not pos_mask.any():
                continue

            # Select appropriate model
            if position in self.use_specific_for:
                model = self.position_models[position]
            else:
                model = self.unified_model

            # Get the features this model expects
            feature_selector = model.named_steps.get("feature_selector")
            if feature_selector is not None:
                feature_names = feature_selector.feature_names
            else:
                # Fallback: try to get from model
                feature_names = list(X.columns)

            # Extract features for this position's rows
            pos_data = X.loc[pos_mask].copy()

            # Add position-specific features if needed
            if position in self.use_specific_for:
                pos_data = add_position_features(pos_data, position)

            # Check if we have all required features
            missing_features = set(feature_names) - set(pos_data.columns)
            if missing_features:
                raise ValueError(
                    f"Missing features for {position} model: {sorted(missing_features)[:10]}..."
                    if len(missing_features) > 10
                    else f"Missing features for {position} model: {sorted(missing_features)}"
                )

            # Extract only the features the model needs (in correct order)
            pos_data_features = pos_data[feature_names]

            # Generate predictions
            pos_predictions = model.predict(pos_data_features)

            # Store in result array (maintaining original order)
            predictions[pos_mask] = pos_predictions
            predicted_mask[pos_mask] = True

        # Validate all rows were predicted
        if not predicted_mask.all():
            unpredicted_positions = X.loc[~predicted_mask, "position"].unique()
            raise ValueError(
                f"Some rows were not predicted. Unknown positions: {unpredicted_positions}"
            )

        return predictions

    def get_model_for_position(self, position: str) -> Pipeline:
        """
        Get the model used for a specific position.

        Args:
            position: One of "GKP", "DEF", "MID", "FWD"

        Returns:
            Pipeline used for that position
        """
        if position in self.use_specific_for:
            return self.position_models[position]
        return self.unified_model

    @property
    def metadata(self) -> dict[str, Any]:
        """Return model metadata for introspection."""
        return self._metadata.copy()

    def __repr__(self) -> str:
        specific = ", ".join(self.use_specific_for)
        unified = ", ".join(self._metadata["unified_positions"])
        return f"HybridPositionModel(specific=[{specific}], unified=[{unified}])"

    @classmethod
    def from_paths(
        cls,
        unified_model_path: str | Path,
        position_model_paths: dict[str, str | Path],
        use_specific_for: list[str] | None = None,
    ) -> "HybridPositionModel":
        """
        Load hybrid model from saved pipeline paths.

        Args:
            unified_model_path: Path to unified model .joblib
            position_model_paths: Dict of {position: path} for position models
            use_specific_for: Positions to use specific models for

        Returns:
            HybridPositionModel instance

        Example:
            >>> hybrid = HybridPositionModel.from_paths(
            ...     unified_model_path="models/custom/lightgbm_pipeline.joblib",
            ...     position_model_paths={
            ...         "GKP": "models/position_specific/gkp_best.joblib",
            ...         "FWD": "models/position_specific/fwd_best.joblib",
            ...     },
            ...     use_specific_for=["GKP", "FWD"]
            ... )
        """
        import joblib

        unified_model = joblib.load(unified_model_path)

        position_models = {}
        for pos, path in position_model_paths.items():
            position_models[pos] = joblib.load(path)

        return cls(
            unified_model=unified_model,
            position_models=position_models,
            use_specific_for=use_specific_for,
        )
