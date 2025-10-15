"""
FPL Machine Learning Expected Points (ML-XP) Service

Advanced ML-based XP predictions using sklearn pipelines with comprehensive features.
Designed to replace or complement the rule-based XP model with improved accuracy.

Key Features:
- Scikit-learn pipelines for production-ready models
- Rich 63-feature set: 5GW rolling windows, team context, fixture difficulty
- Position-specific models (optional) for GKP/DEF/MID/FWD
- Leak-free temporal features (all features use past data only)
- Save/load capability for pre-trained models
- Drop-in replacement for old MLExpectedPointsService (same interface)
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Optional
from pathlib import Path
import logging

from sklearn.pipeline import Pipeline

from .ml_pipeline_factory import (
    create_fpl_pipeline,
    save_pipeline,
    load_pipeline,
    get_team_strength_ratings,
)

warnings.filterwarnings("ignore")

# Set up logging
logger = logging.getLogger(__name__)


class MLExpectedPointsService:
    """
    Machine Learning Expected Points Service using Sklearn Pipelines

    Uses comprehensive 63-feature set with:
    - 5GW rolling form windows (cumulative & per-90 stats)
    - Team context features (rolling team performance)
    - Fixture-specific features (opponent strength, home/away, opponent defense)
    - Leak-free temporal features (all use past data only)

    Maintains same public interface as old service for drop-in replacement in gameweek_manager.py
    """

    def __init__(
        self,
        model_type: str = "rf",
        model_path: Optional[str] = None,
        min_training_gameweeks: int = 6,
        ensemble_rule_weight: float = 0.0,
        debug: bool = False,
    ):
        """
        Initialize ML XP Service

        Args:
            model_type: Model type ('rf' for RandomForest, 'gb' for GradientBoosting, 'ridge')
            model_path: Path to pre-trained model (if None, trains on-the-fly each time)
            min_training_gameweeks: Minimum gameweeks needed for training (5+ for rolling 5GW features)
            ensemble_rule_weight: Weight for rule-based predictions in ensemble (0=pure ML, 1=pure rule-based)
            debug: Enable debug logging
        """
        self.model_type = model_type
        self.model_path = model_path
        self.min_training_gameweeks = max(
            min_training_gameweeks, 5
        )  # Minimum 5 GW for rolling features
        self.ensemble_rule_weight = ensemble_rule_weight
        self.debug = debug

        # Pipeline components
        self.pipeline: Optional[Pipeline] = None
        self.pipeline_metadata: Optional[Dict] = None

        # Load pre-trained model if path provided
        if model_path and Path(model_path).exists():
            self._load_pretrained_model()

        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug(
                f"Initialized ML XP Service: model_type={model_type}, ensemble_weight={ensemble_rule_weight}"
            )

    def _load_pretrained_model(self):
        """Load a pre-trained pipeline from disk."""
        try:
            self.pipeline, self.pipeline_metadata = load_pipeline(Path(self.model_path))
            if self.debug:
                logger.debug(f"✅ Loaded pre-trained model from {self.model_path}")
                if self.pipeline_metadata:
                    mae = self.pipeline_metadata.get("cv_mae_mean", "N/A")
                    logger.debug(f"   Model CV MAE: {mae}")
        except Exception as e:
            logger.warning(
                f"Failed to load pre-trained model: {e}. Will train on-the-fly."
            )
            self.pipeline = None
            self.pipeline_metadata = None

    def _train_pipeline(
        self,
        historical_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        teams_df: pd.DataFrame,
    ) -> Pipeline:
        """
        Train a new pipeline on historical data.

        Args:
            historical_df: Historical player performance data (with gameweek column)
            fixtures_df: Fixture data [event, home_team_id, away_team_id]
            teams_df: Teams data [team_id, name]

        Returns:
            Trained sklearn Pipeline
        """
        if self.debug:
            logger.debug(f"Training new {self.model_type} pipeline...")

        # Get team strength ratings
        team_strength = get_team_strength_ratings()

        # Create pipeline
        pipeline = create_fpl_pipeline(
            model_type=self.model_type,
            fixtures_df=fixtures_df,
            teams_df=teams_df,
            team_strength=team_strength,
            random_state=42,
        )

        # CRITICAL FIX: Transform ALL data first, THEN filter to GW6+ for training
        # The issue: Rolling 5GW features for GW6 need data from GW1-5
        # If we filter to GW6+ before transform, those earlier gameweeks aren't available!
        #
        # Strategy (matching ml_xp_notebook.py exactly):
        #   1. Transform ALL historical data (GW1-7) to calculate complete rolling features
        #   2. Filter transformed features to GW6+ for training (complete 5GW windows)
        #   3. Train model on those filtered features

        # Step 1: Transform ALL data to get complete rolling features
        feature_engineer = pipeline.named_steps["feature_engineer"]
        all_features = feature_engineer.fit_transform(
            historical_df, historical_df["total_points"]
        )

        # Add back gameweek and total_points for filtering
        # CRITICAL: historical_df is sorted by [player_id, gameweek] in feature_engineer.transform()
        historical_df_sorted = historical_df.sort_values(
            ["player_id", "gameweek"]
        ).reset_index(drop=True)
        all_features_with_meta = all_features.copy()
        all_features_with_meta["gameweek"] = historical_df_sorted["gameweek"].values
        all_features_with_meta["total_points"] = historical_df_sorted[
            "total_points"
        ].values

        # Step 2: Filter to GW6+ (now features are complete because they were calculated from all data)
        train_features = all_features_with_meta[
            all_features_with_meta["gameweek"] >= 6
        ].copy()

        if self.debug:
            print(
                f"🔧 ML Training: {len(historical_df)} records → {len(train_features)} after GW6+ filter"
            )
            print(f"   Training GWs: {sorted(train_features['gameweek'].unique())}")
            print(
                f"   Target stats: mean={train_features['total_points'].mean():.2f}, max={train_features['total_points'].max()}"
            )

        if len(train_features) < 100:
            raise ValueError(
                f"Insufficient training data: {len(train_features)} samples after GW6+ filter"
            )

        # Step 3: Train only the model (feature engineer already fitted)
        X_train = train_features[feature_engineer.feature_names_]
        y_train = train_features["total_points"]

        # Train just the model step (feature_engineer already fitted above)
        model = pipeline.named_steps["model"]
        model.fit(X_train, y_train)

        if self.debug:
            logger.debug(f"✅ Model trained on {len(train_features)} GW6+ samples")

        return pipeline

    def calculate_expected_points(
        self,
        players_data: pd.DataFrame,
        teams_data: pd.DataFrame,
        xg_rates_data: pd.DataFrame,
        fixtures_data: pd.DataFrame,
        target_gameweek: int,
        live_data: pd.DataFrame,
        gameweeks_ahead: int = 1,
        rule_based_model=None,
    ) -> pd.DataFrame:
        """
        Main interface method compatible with existing XP calculations.

        This is the method called by gameweek_manager.py and expected_points_service.py

        Args:
            players_data: Current player data (will be enriched with features)
            teams_data: Team data
            xg_rates_data: xG rates data (not used in new approach)
            fixtures_data: Fixtures data
            target_gameweek: Target gameweek for prediction
            live_data: Historical live gameweek data for training
            gameweeks_ahead: Number of gameweeks ahead (1 or 5) - not yet implemented
            rule_based_model: Optional rule-based model for ensemble predictions

        Returns:
            DataFrame with ML-based expected points in 'xP' column
        """
        try:
            # Validate we have enough historical data
            if live_data.empty:
                raise ValueError("No historical live_data provided for training")

            historical_gws = sorted(
                live_data["event"].unique()
                if "event" in live_data.columns
                else live_data["gameweek"].unique()
            )

            # Strategy: Need at least 5 completed gameweeks for rolling 5GW features
            # Example: GW1-5 data predicts GW6, GW2-6 data predicts GW7, etc.
            min_required = 5
            if len(historical_gws) < min_required:
                raise ValueError(
                    f"Need at least {min_required} historical gameweeks for rolling 5GW features. "
                    f"Found {len(historical_gws)} gameweeks: {historical_gws}. "
                    f"Strategy: Use GW1-5 to predict GW6, GW2-6 to predict GW7, etc."
                )

            # Standardize column names
            if "event" in live_data.columns and "gameweek" not in live_data.columns:
                live_data = live_data.rename(columns={"event": "gameweek"})

            # Train or use pre-trained pipeline
            if self.pipeline is None:
                if self.debug:
                    logger.debug("No pre-trained model, training on-the-fly...")

                self.pipeline = self._train_pipeline(
                    historical_df=live_data,
                    fixtures_df=fixtures_data,
                    teams_df=teams_data,
                )

            # Prepare current gameweek data for prediction
            # Need to add gameweek column and ensure all required columns exist
            current_data = players_data.copy()
            current_data["gameweek"] = target_gameweek

            # CRITICAL: Ensure current_data has same columns as historical data
            # Add missing performance columns with 0 (they haven't happened yet for future GW)
            required_cols = [
                "total_points",
                "minutes",
                "goals_scored",
                "assists",
                "clean_sheets",
                "goals_conceded",
                "yellow_cards",
                "red_cards",
                "saves",
                "bonus",
                "bps",
                "influence",
                "creativity",
                "threat",
                "ict_index",
                "expected_goals",
                "expected_assists",
                "expected_goal_involvements",
                "expected_goals_conceded",
                "value",
            ]
            for col in required_cols:
                if col not in current_data.columns:
                    if col == "value" and "price" in current_data.columns:
                        current_data["value"] = (
                            current_data["price"] * 10
                        )  # Convert price back to value
                    else:
                        current_data[col] = 0  # Future performance = 0 (unknown)

            # CRITICAL: The feature engineer needs ALL historical data to calculate rolling features!
            # Append current data to historical - pass ALL data through pipeline
            prediction_data_all = pd.concat(
                [live_data, current_data], ignore_index=True
            )

            # CRITICAL FIX: Sort by player_id and gameweek BEFORE prediction
            # The feature engineer sorts internally, so we must maintain this order
            # to correctly align predictions with player_ids
            prediction_data_all = prediction_data_all.sort_values(
                ["player_id", "gameweek"]
            ).reset_index(drop=True)

            # Make predictions for ALL gameweeks (pipeline calculates rolling features correctly)
            all_predictions = self.pipeline.predict(prediction_data_all)

            # Extract ONLY predictions for target gameweek (now properly aligned)
            target_mask = prediction_data_all["gameweek"] == target_gameweek
            predictions = all_predictions[target_mask]
            predictions = np.clip(predictions, 0, 20)  # Reasonable bounds

            # Get player_ids in the same order as predictions (after sorting)
            target_player_ids = prediction_data_all.loc[target_mask, "player_id"].values

            # Create result DataFrame and align predictions with original players_data order
            result = players_data.copy()

            # Map predictions to correct player_ids
            prediction_map = dict(zip(target_player_ids, predictions))
            result["ml_xP"] = result["player_id"].map(prediction_map).fillna(0)
            result["xP"] = result["ml_xP"]

            # Calculate expected_minutes using rule-based logic
            from .expected_points_service import ExpectedPointsService

            temp_xp_service = ExpectedPointsService()
            result = temp_xp_service._calculate_expected_minutes(result)

            # If rule-based model provided, create weighted ensemble
            if rule_based_model is not None and self.ensemble_rule_weight > 0:
                try:
                    rule_predictions = rule_based_model.calculate_expected_points(
                        players_data=players_data,
                        teams_data=teams_data,
                        xg_rates_data=xg_rates_data,
                        fixtures_data=fixtures_data,
                        target_gameweek=target_gameweek,
                        live_data=live_data,
                        gameweeks_ahead=gameweeks_ahead,
                    )

                    # Create weighted ensemble
                    ml_weight = 1 - self.ensemble_rule_weight
                    result["xP"] = (
                        ml_weight * result["ml_xP"]
                        + self.ensemble_rule_weight * rule_predictions["xP"]
                    )

                    if self.debug:
                        logger.debug(
                            f"Created ensemble: ML={ml_weight:.2f}, Rule={self.ensemble_rule_weight:.2f}"
                        )

                except Exception as e:
                    logger.warning(f"Rule-based ensemble failed, using pure ML: {e}")

            if self.debug:
                logger.debug(f"Generated ML predictions for {len(result)} players")
                logger.debug(
                    f"xP range: {result['xP'].min():.2f} - {result['xP'].max():.2f}"
                )

            return result

        except Exception as e:
            logger.error(f"ML XP calculation failed: {str(e)}")
            raise

    def save_models(self, filepath: str):
        """
        Save trained pipeline to file (maintains old interface)

        Args:
            filepath: Path to save the model
        """
        if self.pipeline is None:
            raise ValueError("No pipeline to save. Train or load a model first.")

        save_pipeline(self.pipeline, Path(filepath), self.pipeline_metadata)

        if self.debug:
            logger.debug(f"Saved pipeline to {filepath}")

    def load_models(self, filepath: str):
        """
        Load trained pipeline from file (maintains old interface)

        Args:
            filepath: Path to saved model
        """
        self.model_path = filepath
        self._load_pretrained_model()

        if self.debug:
            logger.debug(f"Loaded pipeline from {filepath}")

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Returns:
            DataFrame with features and importance scores (for tree-based models)
        """
        if self.pipeline is None:
            raise ValueError(
                "Model not trained. Call calculate_expected_points() first."
            )

        # Extract model from pipeline
        model = self.pipeline.named_steps.get("model")
        if model is None:
            raise ValueError("No model found in pipeline")

        # Get feature names from feature engineer
        feature_engineer = self.pipeline.named_steps.get("feature_engineer")
        if feature_engineer is None:
            raise ValueError("No feature engineer found in pipeline")

        feature_names = feature_engineer.get_feature_names_out()

        # Get importance scores (for tree-based models)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)
        elif hasattr(model, "coef_"):
            # For Ridge regression
            importances = np.abs(model.coef_)
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)
        else:
            raise ValueError("Model does not support feature importance extraction")

        return importance_df
