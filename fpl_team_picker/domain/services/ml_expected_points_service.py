"""
FPL Machine Learning Expected Points (ML-XP) Service

Advanced ML-based XP predictions using pre-trained sklearn pipelines.
REQUIRES pre-trained model artifacts (no on-the-fly training).

Key Features:
- Pre-trained sklearn pipelines for production deployment
- Rich 80-feature set: 5GW rolling windows, team context, ownership trends, value analysis
- Position-specific models (optional) for GKP/DEF/MID/FWD
- Leak-free temporal features (all features use past data only)
- Loads .joblib model artifacts trained via TPOT or ml_xp_experiment.py
- Drop-in replacement for old MLExpectedPointsService (same interface)

Train models using:
  - TPOT: scripts/tpot_pipeline_optimizer.py --start-gw 1 --end-gw 8
  - Manual: ml_xp_experiment.py interface
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Optional
from pathlib import Path
from loguru import logger

from sklearn.pipeline import Pipeline
from .ml_pipeline_factory import (
    create_fpl_pipeline,
    save_pipeline,
    load_pipeline,
)
from fpl_team_picker.domain.ml import HybridPositionModel


warnings.filterwarnings("ignore")

# Module-specific logger to avoid global configuration pollution
# This logger is bound to this module and doesn't affect other modules
_service_logger = logger.bind(service="ml_expected_points")


class MLExpectedPointsService:
    """
    Machine Learning Expected Points Service using Pre-trained Sklearn Pipelines

    REQUIRES pre-trained model artifact (no on-the-fly training).
    Train models using:
      - TPOT: scripts/tpot_pipeline_optimizer.py
      - Or: ml_xp_experiment.py

    Uses comprehensive 84-feature set with:
    - Base features (65): 5GW rolling form windows, team context, fixtures, price bands
    - Enhanced features (15): Ownership trends, value analysis, fixture difficulty
    - Set-piece & penalty features (4): penalty/corners/freekicks taker flags
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
            model_type: Model type ('rf' for RandomForest, 'gb' for GradientBoosting, 'ridge') - DEPRECATED, not used
            model_path: Path to pre-trained model artifact (REQUIRED - no on-the-fly training)
            min_training_gameweeks: DEPRECATED - not used (models are pre-trained)
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

        self._log_debug(
            f"Initialized ML XP Service: model_type={model_type}, ensemble_weight={ensemble_rule_weight}"
        )

    def _log_debug(self, message: str):
        """
        Log debug message only if debug mode is enabled.
        Uses module-specific logger to avoid global configuration pollution.
        """
        if self.debug:
            _service_logger.debug(message)

    def _load_pretrained_model(self):
        """
        Load a pre-trained pipeline from disk.

        For TPOT models: Wraps the bare sklearn pipeline with FPLFeatureEngineer
        since TPOT was trained on already-engineered features.

        Note: The wrapper pipeline is NOT fitted yet - it will be fitted on
        historical data during the first call to calculate_expected_points().
        """
        try:
            loaded_pipeline, self.pipeline_metadata = load_pipeline(
                Path(self.model_path)
            )

            # Check if this is a HybridPositionModel (handles its own feature routing)
            is_hybrid_model = isinstance(loaded_pipeline, HybridPositionModel)

            # Check if pipeline already has feature_engineer step
            has_feature_engineer = (
                hasattr(loaded_pipeline, "named_steps")
                and "feature_engineer" in loaded_pipeline.named_steps
            )

            if is_hybrid_model:
                # HybridPositionModel handles its own routing and feature selection
                # Do NOT wrap with feature engineer as it needs 'position' column
                self._log_debug(
                    "🔀 HybridPositionModel detected - using direct prediction (no feature wrapper)"
                )
                self.pipeline = loaded_pipeline
                self.needs_feature_wrapper = False
                self.is_hybrid_model = True
            elif not has_feature_engineer:
                # TPOT model or bare sklearn pipeline - needs feature engineering wrapper
                self._log_debug(
                    "⚙️  Bare sklearn pipeline detected (likely TPOT) - will wrap with FPLFeatureEngineer"
                )

                # Store the loaded model - we'll create the wrapper in calculate_expected_points
                # after we have historical data to fit the feature_engineer
                self.pipeline = loaded_pipeline
                self.needs_feature_wrapper = (
                    True  # Flag to create wrapper during prediction
                )
                self.is_hybrid_model = False
            else:
                # Standard pipeline already includes feature_engineer
                self.pipeline = loaded_pipeline
                self.needs_feature_wrapper = False
                self.is_hybrid_model = False

            self._log_debug(f"✅ Loaded pre-trained model from {self.model_path}")
            if self.debug and self.pipeline_metadata:
                mae = self.pipeline_metadata.get("cv_mae_mean", "N/A")
                self._log_debug(f"   Model CV MAE: {mae}")
        except Exception as e:
            raise ValueError(
                f"Failed to load pre-trained model from {self.model_path}: {e}\n"
                "MLExpectedPointsService requires a valid pre-trained model artifact.\n"
                "Train a model using:\n"
                "  - TPOT: uv run python scripts/tpot_pipeline_optimizer.py --start-gw 1 --end-gw 8\n"
                "  - Or: Use ml_xp_experiment.py to train and export a model"
            )

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
        self._log_debug(f"Training new {self.model_type} pipeline...")

        # Validate position column exists (should be enriched upstream by DataOrchestrationService)
        if "position" not in historical_df.columns:
            raise ValueError(
                "Historical data missing 'position' column. "
                "This should be enriched upstream by DataOrchestrationService. "
                "Ensure live_data includes position information before calling ML service."
            )

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

        self._log_debug(
            f"🔧 ML Training: {len(historical_df)} records → {len(train_features)} after GW6+ filter"
        )
        self._log_debug(
            f"   Training GWs: {sorted(train_features['gameweek'].unique())}"
        )
        self._log_debug(
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

        self._log_debug(f"✅ Model trained on {len(train_features)} GW6+ samples")

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
        ownership_trends_df: Optional[pd.DataFrame] = None,
        value_analysis_df: Optional[pd.DataFrame] = None,
        fixture_difficulty_df: Optional[pd.DataFrame] = None,
        raw_players_df: Optional[pd.DataFrame] = None,
        betting_features_df: Optional[pd.DataFrame] = None,
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
            ownership_trends_df: Enhanced ownership trends data (Issue #37)
            value_analysis_df: Enhanced value analysis data (Issue #37)
            fixture_difficulty_df: Enhanced fixture difficulty data (Issue #37)
            raw_players_df: Raw FPL players bootstrap data with penalty/set-piece order
                (ONLY used for inference, NOT training - avoids data leakage)
            betting_features_df: Betting odds features data (Issue #38)

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

            # Require pre-trained pipeline - no on-the-fly training
            if self.pipeline is None:
                raise ValueError(
                    "No pre-trained model loaded. MLExpectedPointsService requires a pre-trained model artifact.\n"
                    "Train a model using:\n"
                    "  - TPOT: uv run python scripts/tpot_pipeline_optimizer.py --start-gw 1 --end-gw 8\n"
                    "  - Or: Use ml_xp_experiment.py to train and export a model\n"
                    "Then set config.xp_model.ml_model_path to the trained model file."
                )
            elif getattr(self, "needs_feature_wrapper", False):
                # TPOT model needs feature engineering wrapper
                # Create wrapper: [FPLFeatureEngineer -> TPOT Model] and fit on historical data
                from sklearn.pipeline import Pipeline as SklearnPipeline
                from .ml_feature_engineering import FPLFeatureEngineer
                from .ml_pipeline_factory import get_team_strength_ratings

                self._log_debug(
                    "🔧 Creating feature engineering wrapper for TPOT model..."
                )

                # Use target_gameweek for dynamic team strength (important for fixture difficulty)
                team_strength = get_team_strength_ratings(
                    target_gameweek=target_gameweek,
                    teams_df=teams_data,
                )

                # Validate enhanced data sources for 80-feature model
                if ownership_trends_df is None or ownership_trends_df.empty:
                    raise ValueError(
                        "Enhanced ownership trends data required for 80-feature TPOT model. "
                        "DataOrchestrationService should provide this in gameweek_data."
                    )
                if value_analysis_df is None or value_analysis_df.empty:
                    raise ValueError(
                        "Enhanced value analysis data required for 80-feature TPOT model. "
                        "DataOrchestrationService should provide this in gameweek_data."
                    )
                if fixture_difficulty_df is None or fixture_difficulty_df.empty:
                    raise ValueError(
                        "Enhanced fixture difficulty data required for 80-feature TPOT model. "
                        "DataOrchestrationService should provide this in gameweek_data."
                    )

                # IMPORTANT: raw_players_df is ONLY used for inference (not training)
                # Training data uses default 0 values for penalty features (no leakage)
                # Live predictions use current penalty taker status for inference-time boost
                feature_engineer = FPLFeatureEngineer(
                    fixtures_df=fixtures_data,
                    teams_df=teams_data,
                    team_strength=team_strength,
                    ownership_trends_df=ownership_trends_df,
                    value_analysis_df=value_analysis_df,
                    fixture_difficulty_df=fixture_difficulty_df,
                    raw_players_df=raw_players_df,  # Penalty/set-piece taker features (inference only)
                    betting_features_df=betting_features_df,  # Betting odds features (Issue #38)
                )

                # Create wrapper pipeline
                tpot_model = self.pipeline  # Save reference to loaded TPOT model
                # Wrapper pipeline: feature engineering + trained model
                wrapper_pipeline = SklearnPipeline(
                    [("feature_engineer", feature_engineer), ("model", tpot_model)]
                )

                # Fit ONLY the feature_engineer step (TPOT model is already fitted)
                # We fit on historical data so rolling features are calculated correctly
                self._log_debug(
                    f"   Fitting feature_engineer on {len(live_data)} historical samples..."
                )

                # Fit only the feature engineer on historical data
                feature_engineer.fit(live_data, live_data["total_points"])

                # Replace pipeline with fitted wrapper
                self.pipeline = wrapper_pipeline
                self.needs_feature_wrapper = False  # Don't wrap again

                self._log_debug("   ✅ Feature engineering wrapper created and fitted")
            elif getattr(self, "is_hybrid_model", False):
                # HybridPositionModel needs feature engineering BUT must preserve 'position' column
                # Create a feature engineer for the hybrid model
                from .ml_feature_engineering import FPLFeatureEngineer
                from .ml_pipeline_factory import get_team_strength_ratings

                self._log_debug(
                    "🔀 Setting up feature engineering for HybridPositionModel..."
                )

                team_strength = get_team_strength_ratings(
                    target_gameweek=target_gameweek,
                    teams_df=teams_data,
                )

                # Create and store the feature engineer for hybrid model
                self._hybrid_feature_engineer = FPLFeatureEngineer(
                    fixtures_df=fixtures_data,
                    teams_df=teams_data,
                    team_strength=team_strength,
                    ownership_trends_df=ownership_trends_df,
                    value_analysis_df=value_analysis_df,
                    fixture_difficulty_df=fixture_difficulty_df,
                    raw_players_df=raw_players_df,
                    betting_features_df=betting_features_df,
                )

                # Fit on historical data
                self._hybrid_feature_engineer.fit(live_data, live_data["total_points"])
                self._log_debug("   ✅ HybridPositionModel feature engineer fitted")
            else:
                # Standard pipeline with feature_engineer already included
                # Update context if needed
                if (
                    hasattr(self.pipeline, "named_steps")
                    and "feature_engineer" in self.pipeline.named_steps
                ):
                    feature_engineer = self.pipeline.named_steps["feature_engineer"]

                    # Update context (fixtures, teams, team_strength) with current gameweek
                    from .ml_pipeline_factory import get_team_strength_ratings

                    # Use target_gameweek for dynamic team strength (important for fixture difficulty)
                    team_strength = get_team_strength_ratings(
                        target_gameweek=target_gameweek,
                        teams_df=teams_data,
                    )

                    feature_engineer.fixtures_df = (
                        fixtures_data if not fixtures_data.empty else None
                    )
                    feature_engineer.teams_df = (
                        teams_data if not teams_data.empty else None
                    )
                    feature_engineer.team_strength = (
                        team_strength if team_strength else None
                    )

                    self._log_debug(
                        "✅ Updated FPLFeatureEngineer context (fixtures, teams, team_strength)"
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
            # Filter out target gameweek from live_data to avoid duplicates (same as rule-based service)
            # live_data might include actual target_gameweek performance if available
            gw_col = "gameweek" if "gameweek" in live_data.columns else "event"
            historical_only = live_data[live_data[gw_col] < target_gameweek].copy()

            # Append current data to historical - pass ALL data through pipeline
            prediction_data_all = pd.concat(
                [historical_only, current_data], ignore_index=True
            )

            # CRITICAL FIX: Sort by player_id and gameweek BEFORE prediction
            # The feature engineer sorts internally, so we must maintain this order
            # to correctly align predictions with player_ids
            prediction_data_all = prediction_data_all.sort_values(
                ["player_id", "gameweek"]
            ).reset_index(drop=True)

            # Make predictions for ALL gameweeks (pipeline calculates rolling features correctly)
            if getattr(self, "is_hybrid_model", False):
                # HybridPositionModel: apply feature engineering while preserving 'position'
                # Save position column before transformation
                position_col = prediction_data_all["position"].copy()

                # Transform with feature engineer
                engineered_features = self._hybrid_feature_engineer.transform(
                    prediction_data_all
                )

                # Add position column back (required for HybridPositionModel routing)
                engineered_features["position"] = position_col.values

                # Now predict with HybridPositionModel
                all_predictions = self.pipeline.predict(engineered_features)
            else:
                all_predictions = self.pipeline.predict(prediction_data_all)

            # Extract uncertainty estimates for target gameweek
            target_mask = prediction_data_all["gameweek"] == target_gameweek

            # Extract uncertainty (will use RF tree variance if available)
            all_uncertainty = self._extract_uncertainty(prediction_data_all)
            uncertainty = all_uncertainty[target_mask]

            # Extract ONLY predictions for target gameweek (now properly aligned)
            predictions = all_predictions[target_mask]
            predictions = np.clip(predictions, 0, 20)  # Reasonable bounds

            # Get player_ids in the same order as predictions (after sorting)
            target_player_ids = prediction_data_all.loc[target_mask, "player_id"].values

            # Create result DataFrame and align predictions with original players_data order
            result = players_data.copy()

            # Map predictions and uncertainty to correct player_ids
            prediction_map = dict(zip(target_player_ids, predictions))
            uncertainty_map = dict(zip(target_player_ids, uncertainty))
            result["ml_xP"] = result["player_id"].map(prediction_map)
            result["xP_uncertainty"] = result["player_id"].map(uncertainty_map)

            # Validate that ALL players have ML predictions (fail fast, no silent fallbacks)
            missing_predictions = result["ml_xP"].isna()
            if missing_predictions.any():
                missing_players = result.loc[
                    missing_predictions, ["player_id", "web_name"]
                ]
                _service_logger.error(
                    f"ML prediction failed for {missing_predictions.sum()} players: "
                    f"{missing_players['web_name'].tolist()}"
                )
                _service_logger.error(
                    "This indicates upstream data quality issues (missing features, "
                    "insufficient historical data, or feature engineering failures)"
                )
                raise ValueError(
                    f"Unable to generate ML predictions for {missing_predictions.sum()} players. "
                    f"Missing: {missing_players['web_name'].tolist()}"
                )

            # Validate uncertainty extraction succeeded
            missing_uncertainty = result["xP_uncertainty"].isna()
            if missing_uncertainty.any():
                missing_players = result.loc[
                    missing_uncertainty, ["player_id", "web_name"]
                ]
                _service_logger.error(
                    f"Uncertainty extraction failed for {missing_uncertainty.sum()} players: "
                    f"{missing_players['web_name'].tolist()}"
                )
                raise ValueError(
                    f"Unable to extract uncertainty for {missing_uncertainty.sum()} players. "
                    f"Missing: {missing_players['web_name'].tolist()}"
                )

            result["xP"] = result["ml_xP"]

            # Extract fixture difficulty from fixture_difficulty_df and merge into result
            if fixture_difficulty_df is None or fixture_difficulty_df.empty:
                raise ValueError(
                    "fixture_difficulty_df is required but was not provided or is empty. "
                    "DataOrchestrationService should provide this in gameweek_data."
                )

            # Filter fixture_difficulty_df for target gameweek
            target_fixture_difficulty = fixture_difficulty_df[
                fixture_difficulty_df["gameweek"] == target_gameweek
            ].copy()

            if target_fixture_difficulty.empty:
                raise ValueError(
                    f"No fixture difficulty data found for gameweek {target_gameweek}. "
                    f"fixture_difficulty_df must contain data for the target gameweek."
                )

            if "overall_difficulty" not in target_fixture_difficulty.columns:
                raise ValueError(
                    f"fixture_difficulty_df missing required 'overall_difficulty' column. "
                    f"Available columns: {list(target_fixture_difficulty.columns)}"
                )

            # Recalculate fixture_difficulty using the same formula as training
            # This ensures consistency with the 0-2 scale used in feature engineering
            # (overall_difficulty from get_derived_fixture_difficulty uses a different 3-5 scale)
            from fpl_team_picker.domain.services.team_analytics_service import (
                TeamAnalyticsService,
            )
            from .ml_pipeline_factory import get_team_strength_ratings

            # Get team strength (same as used in feature engineering)
            analytics_service = TeamAnalyticsService(debug=False)
            historical_gws = sorted(
                live_data["event"].unique()
                if "event" in live_data.columns
                else live_data["gameweek"].unique()
            )
            if historical_gws:
                current_season_data = analytics_service.load_historical_gameweek_data(
                    start_gw=min(historical_gws), end_gw=max(historical_gws)
                )
                team_strength = analytics_service.get_team_strength(
                    target_gameweek=target_gameweek,
                    teams_data=teams_data,
                    current_season_data=current_season_data,
                )
            else:
                # Fallback: use static team strength
                team_strength = get_team_strength_ratings(
                    target_gameweek=target_gameweek,
                    teams_df=teams_data,
                )

            # Get fixtures for target gameweek
            # Handle both "gameweek" and "event" column names
            gw_col = "gameweek" if "gameweek" in fixtures_data.columns else "event"
            target_fixtures = (
                fixtures_data[fixtures_data[gw_col] == target_gameweek].copy()
                if gw_col in fixtures_data.columns
                else pd.DataFrame()
            )

            # Create team name mapping
            team_name_map = dict(zip(teams_data["team_id"], teams_data["name"]))

            # Calculate fixture_difficulty using same formula as training
            # (2.0 - opponent_strength) * 1.1 for home, 2.0 - opponent_strength for away
            fixture_difficulty_map = {}
            for _, fix in target_fixtures.iterrows():
                # Handle different column name variations
                home_id = fix.get("team_h") or fix.get("home_team_id")
                away_id = fix.get("team_a") or fix.get("away_team_id")

                if pd.isna(home_id) or pd.isna(away_id):
                    continue

                home_name = team_name_map.get(int(home_id))
                away_name = team_name_map.get(int(away_id))

                if home_name and away_name:
                    home_opponent_strength = team_strength.get(away_name, 1.0)
                    away_opponent_strength = team_strength.get(home_name, 1.0)

                    # Use relative scaling for fixture difficulty (same as ml_feature_engineering)
                    # This ensures full 0-2 range utilization and better differentiation
                    min_strength = min(team_strength.values())
                    max_strength = max(team_strength.values())
                    strength_range = max_strength - min_strength
                    if strength_range < 0.01:
                        strength_range = 0.22  # Fallback to typical range

                    # Normalize opponent strength to 0-1, then map to 0-2 range (inverted)
                    home_normalized = (
                        home_opponent_strength - min_strength
                    ) / strength_range
                    away_normalized = (
                        away_opponent_strength - min_strength
                    ) / strength_range

                    home_base = 2.0 * (
                        1.0 - home_normalized
                    )  # Weak opponent = high value
                    away_base = 2.0 * (1.0 - away_normalized)

                    # Apply home advantage
                    home_advantage = 1.1
                    home_difficulty = home_base * home_advantage
                    away_difficulty = away_base

                    fixture_difficulty_map[int(home_id)] = home_difficulty
                    fixture_difficulty_map[int(away_id)] = away_difficulty

            # Map fixture_difficulty to result using team column
            result["fixture_difficulty"] = result["team"].map(fixture_difficulty_map)

            # Fill missing values with neutral 1.0 (shouldn't happen, but safety)
            result["fixture_difficulty"] = result["fixture_difficulty"].fillna(1.0)

            # Validate that all players have fixture difficulty data
            missing_difficulty = result["fixture_difficulty"].isna()
            if missing_difficulty.any():
                missing_players = result.loc[
                    missing_difficulty, ["player_id", "web_name", "team"]
                ]
                raise ValueError(
                    f"Missing fixture difficulty data for {missing_difficulty.sum()} players. "
                    f"Missing players: {missing_players['web_name'].tolist()}. "
                    f"This indicates a data quality issue - could not calculate fixture_difficulty."
                )

            # Calculate fixture_outlook from fixture_difficulty
            # NOTE: Higher fixture_difficulty = easier fixture (inverse of opponent strength)
            # Use configurable thresholds that match the actual distribution with base_multiplier
            from fpl_team_picker.config import config

            easy_threshold = config.fixture_difficulty.easy_fixture_threshold
            average_min = config.fixture_difficulty.average_fixture_min

            def get_fixture_outlook(diff):
                if pd.isna(diff):
                    raise ValueError(
                        "fixture_difficulty is NaN - this should not happen after validation"
                    )
                elif diff >= easy_threshold:
                    return "🟢 Easy"  # High value = easy fixture
                elif diff <= average_min:
                    return "🔴 Hard"  # Low value = hard fixture
                else:
                    return "🟡 Average"

            result["fixture_outlook"] = result["fixture_difficulty"].apply(
                get_fixture_outlook
            )

            # Calculate expected_minutes using rule-based logic
            from .expected_points_service import ExpectedPointsService

            temp_xp_service = ExpectedPointsService()
            result = temp_xp_service._calculate_expected_minutes(
                result, live_data, target_gameweek
            )

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

                    # Uncertainty remains from ML model (rule-based has no uncertainty)
                    # Could theoretically combine uncertainties, but that's complex
                    # For now, keep ML uncertainty scaled by its weight
                    result["xP_uncertainty"] = ml_weight * result["xP_uncertainty"]

                    self._log_debug(
                        f"Created ensemble: ML={ml_weight:.2f}, Rule={self.ensemble_rule_weight:.2f}"
                    )

                except Exception as e:
                    _service_logger.warning(
                        f"Rule-based ensemble failed, using pure ML: {e}"
                    )

            self._log_debug(f"Generated ML predictions for {len(result)} players")
            self._log_debug(
                f"xP range: {result['xP'].min():.2f} - {result['xP'].max():.2f}"
            )

            return result

        except Exception as e:
            _service_logger.error(f"ML XP calculation failed: {str(e)}")
            raise

    def calculate_5gw_expected_points(
        self,
        players_data: pd.DataFrame,
        teams_data: pd.DataFrame,
        xg_rates_data: pd.DataFrame,
        fixtures_data: pd.DataFrame,
        target_gameweek: int,
        live_data: pd.DataFrame,
        rule_based_model=None,
        ownership_trends_df: Optional[pd.DataFrame] = None,
        value_analysis_df: Optional[pd.DataFrame] = None,
        fixture_difficulty_df: Optional[pd.DataFrame] = None,
        raw_players_df: Optional[pd.DataFrame] = None,
        betting_features_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate 5-gameweek expected points using cascading predictions.

        This method predicts each of the next 5 gameweeks sequentially, using
        predictions from earlier gameweeks to inform rolling features for later ones.
        This ensures rolling features (e.g., rolling_5gw_points) remain valid.

        Strategy:
        1. Predict GW+1 using actual historical data
        2. Predict GW+2 using historical data + predicted GW+1 points
        3. Predict GW+3 using historical data + predicted GW+1-2 points
        4. Continue for GW+4 and GW+5
        5. Sum all 5 predictions to get total 5gw xP

        Args:
            players_data: Current player data
            teams_data: Team data
            xg_rates_data: xG rates data
            fixtures_data: Fixtures data
            target_gameweek: Starting gameweek (will predict GW+1 through GW+5)
            live_data: Historical live gameweek data (actual results up to target_gameweek)
            rule_based_model: Optional rule-based model for ensemble
            ownership_trends_df: Ownership trends data
            value_analysis_df: Value analysis data
            fixture_difficulty_df: Fixture difficulty data
            raw_players_df: Raw players data
            betting_features_df: Betting odds features data

        Returns:
            DataFrame with:
            - ml_xP: 5-gameweek total expected points
            - xP_uncertainty: Combined uncertainty (sqrt of sum of variances)
            - xP: Same as ml_xP (for compatibility)
            - xP_gw1, xP_gw2, ..., xP_gw5: Per-gameweek predictions (for analysis)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not initialized. Train or load a model first.")

        self._log_debug(
            f"🔄 Calculating 5GW ML xP using cascading predictions starting from GW{target_gameweek}"
        )

        # Start with actual historical data
        historical_data = live_data.copy()
        if historical_data.empty:
            raise ValueError(
                "No historical live_data provided for cascading predictions"
            )

        # Store predictions for each gameweek
        per_gw_predictions = []
        per_gw_uncertainties = []
        per_gw_results = []

        # Predict each of the next 5 gameweeks sequentially
        for gw_offset in range(1, 6):
            current_gw = target_gameweek + gw_offset

            self._log_debug(
                f"  📊 Predicting GW{current_gw} (using data up to GW{current_gw - 1})"
            )

            # Predict this gameweek using current historical data
            result = self.calculate_expected_points(
                players_data=players_data,
                teams_data=teams_data,
                xg_rates_data=xg_rates_data,
                fixtures_data=fixtures_data,
                target_gameweek=current_gw,
                live_data=historical_data,
                gameweeks_ahead=1,
                rule_based_model=rule_based_model,
                ownership_trends_df=ownership_trends_df,
                value_analysis_df=value_analysis_df,
                fixture_difficulty_df=fixture_difficulty_df,
                raw_players_df=raw_players_df,
                betting_features_df=betting_features_df,
            )

            # Extract predictions and uncertainties
            per_gw_predictions.append(result["ml_xP"].values)
            per_gw_uncertainties.append(result["xP_uncertainty"].values)
            per_gw_results.append(result)

            # Create synthetic gameweek data from predictions for next iteration
            # This allows rolling features to use predicted points for future gameweeks
            synthetic_gw_data = self._create_synthetic_gameweek_data(
                players_data=players_data,
                predictions=result,
                gameweek=current_gw,
            )

            # Add synthetic data to historical data for next iteration
            historical_data = pd.concat(
                [historical_data, synthetic_gw_data], ignore_index=True
            )

        # Combine results: sum predictions, combine uncertainties
        result_5gw = players_data.copy()

        # Sum all 5 gameweek predictions
        xp_5gw = np.sum(per_gw_predictions, axis=0)

        # Combine uncertainties: sqrt(sum of variances) for independent predictions
        # This assumes predictions are independent (reasonable for different gameweeks)
        uncertainty_5gw = np.sqrt(np.sum(np.square(per_gw_uncertainties), axis=0))

        result_5gw["ml_xP"] = xp_5gw
        result_5gw["xP"] = xp_5gw
        result_5gw["xP_5gw"] = xp_5gw  # Alias for compatibility
        result_5gw["xP_uncertainty"] = uncertainty_5gw

        # Add per-gameweek predictions for analysis/debugging
        for i, gw_result in enumerate(per_gw_results):
            gw_num = target_gameweek + i + 1
            result_5gw[f"xP_gw{gw_num}"] = gw_result["ml_xP"].values
            result_5gw[f"uncertainty_gw{gw_num}"] = gw_result["xP_uncertainty"].values

        self._log_debug(
            f"✅ 5GW predictions complete: Total xP range {xp_5gw.min():.2f} - {xp_5gw.max():.2f}"
        )

        return result_5gw

    def calculate_3gw_expected_points(
        self,
        players_data: pd.DataFrame,
        teams_data: pd.DataFrame,
        xg_rates_data: pd.DataFrame,
        fixtures_data: pd.DataFrame,
        target_gameweek: int,
        live_data: pd.DataFrame,
        rule_based_model=None,
        ownership_trends_df: Optional[pd.DataFrame] = None,
        value_analysis_df: Optional[pd.DataFrame] = None,
        fixture_difficulty_df: Optional[pd.DataFrame] = None,
        raw_players_df: Optional[pd.DataFrame] = None,
        betting_features_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate 3-gameweek expected points using cascading predictions.

        This method predicts each of the next 3 gameweeks sequentially, using
        predictions from earlier gameweeks to inform rolling features for later ones.
        This ensures rolling features (e.g., rolling_5gw_points) remain valid.

        Strategy:
        1. Predict GW+1 using actual historical data
        2. Predict GW+2 using historical data + predicted GW+1 points
        3. Predict GW+3 using historical data + predicted GW+1-2 points
        4. Sum all 3 predictions to get total 3gw xP

        Args:
            players_data: Current player data
            teams_data: Team data
            xg_rates_data: xG rates data
            fixtures_data: Fixtures data
            target_gameweek: Starting gameweek (will predict GW+1 through GW+3)
            live_data: Historical live gameweek data (actual results up to target_gameweek)
            rule_based_model: Optional rule-based model for ensemble
            ownership_trends_df: Ownership trends data
            value_analysis_df: Value analysis data
            fixture_difficulty_df: Fixture difficulty data
            raw_players_df: Raw players data
            betting_features_df: Betting odds features data

        Returns:
            DataFrame with:
            - ml_xP: 3-gameweek total expected points
            - xP_uncertainty: Combined uncertainty (sqrt of sum of variances)
            - xP: Same as ml_xP (for compatibility)
            - xP_gw1, xP_gw2, xP_gw3: Per-gameweek predictions (for analysis)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not initialized. Train or load a model first.")

        self._log_debug(
            f"🔄 Calculating 3GW ML xP using cascading predictions starting from GW{target_gameweek}"
        )

        # Start with actual historical data
        historical_data = live_data.copy()
        if historical_data.empty:
            raise ValueError(
                "No historical live_data provided for cascading predictions"
            )

        # Store predictions for each gameweek
        per_gw_predictions = []
        per_gw_uncertainties = []
        per_gw_results = []

        # Predict each of the next 3 gameweeks sequentially
        for gw_offset in range(1, 4):
            current_gw = target_gameweek + gw_offset

            self._log_debug(
                f"  📊 Predicting GW{current_gw} (using data up to GW{current_gw - 1})"
            )

            # Predict this gameweek using current historical data
            result = self.calculate_expected_points(
                players_data=players_data,
                teams_data=teams_data,
                xg_rates_data=xg_rates_data,
                fixtures_data=fixtures_data,
                target_gameweek=current_gw,
                live_data=historical_data,
                gameweeks_ahead=1,
                rule_based_model=rule_based_model,
                ownership_trends_df=ownership_trends_df,
                value_analysis_df=value_analysis_df,
                fixture_difficulty_df=fixture_difficulty_df,
                raw_players_df=raw_players_df,
                betting_features_df=betting_features_df,
            )

            # Extract predictions and uncertainties
            per_gw_predictions.append(result["ml_xP"].values)
            per_gw_uncertainties.append(result["xP_uncertainty"].values)
            per_gw_results.append(result)

            # Create synthetic gameweek data from predictions for next iteration
            # This allows rolling features to use predicted points for future gameweeks
            synthetic_gw_data = self._create_synthetic_gameweek_data(
                players_data=players_data,
                predictions=result,
                gameweek=current_gw,
            )

            # Add synthetic data to historical data for next iteration
            historical_data = pd.concat(
                [historical_data, synthetic_gw_data], ignore_index=True
            )

        # Combine results: sum predictions, combine uncertainties
        result_3gw = players_data.copy()

        # Sum all 3 gameweek predictions
        xp_3gw = np.sum(per_gw_predictions, axis=0)

        # Combine uncertainties: sqrt(sum of variances) for independent predictions
        # This assumes predictions are independent (reasonable for different gameweeks)
        uncertainty_3gw = np.sqrt(np.sum(np.square(per_gw_uncertainties), axis=0))

        result_3gw["ml_xP"] = xp_3gw
        result_3gw["xP"] = xp_3gw
        result_3gw["xP_3gw"] = xp_3gw  # Alias for compatibility
        result_3gw["xP_uncertainty"] = uncertainty_3gw

        # Add per-gameweek predictions for analysis/debugging
        for i, gw_result in enumerate(per_gw_results):
            gw_num = target_gameweek + i + 1
            result_3gw[f"xP_gw{gw_num}"] = gw_result["ml_xP"].values
            result_3gw[f"uncertainty_gw{gw_num}"] = gw_result["xP_uncertainty"].values

        self._log_debug(
            f"✅ 3GW predictions complete: Total xP range {xp_3gw.min():.2f} - {xp_3gw.max():.2f}"
        )

        return result_3gw

    def _create_synthetic_gameweek_data(
        self,
        players_data: pd.DataFrame,
        predictions: pd.DataFrame,
        gameweek: int,
    ) -> pd.DataFrame:
        """
        Create synthetic gameweek data from ML predictions for cascading predictions.

        This generates a DataFrame that mimics actual gameweek performance data,
        using predicted points to populate performance columns. This allows
        rolling features in subsequent predictions to use predicted values.

        Args:
            players_data: Base player data with player_id, position, team, etc.
            predictions: DataFrame with ml_xP predictions
            gameweek: Gameweek number for synthetic data

        Returns:
            DataFrame with synthetic gameweek performance data
        """
        # Start with base player data
        synthetic = players_data.copy()

        # Set gameweek
        synthetic["gameweek"] = gameweek

        # Map predictions to players
        if "player_id" in predictions.columns and "ml_xP" in predictions.columns:
            pred_map = dict(zip(predictions["player_id"], predictions["ml_xP"]))
            synthetic["total_points"] = synthetic["player_id"].map(pred_map).fillna(0)
        else:
            synthetic["total_points"] = 0

        # Estimate other performance metrics from predicted points
        # These are rough heuristics - the exact values don't matter much
        # since we're mainly using total_points for rolling features
        synthetic["minutes"] = np.where(
            synthetic["total_points"] > 0, 90, 0
        )  # Assume 90 mins if points > 0

        # Estimate goals/assists from points (rough heuristic)
        # Points = 2 (appearance) + goals*points_per_goal + assists*3 + ...
        # For simplicity, assume points come from goals/assists
        points_from_goals_assists = synthetic["total_points"] - 2  # Subtract appearance
        points_from_goals_assists = np.maximum(0, points_from_goals_assists)

        # Rough estimate: assume goals worth 4-6 points on average
        synthetic["goals_scored"] = (points_from_goals_assists / 5).clip(0, 5)
        synthetic["assists"] = (points_from_goals_assists / 3).clip(0, 5)

        # Set other required columns to reasonable defaults
        synthetic["clean_sheets"] = 0
        synthetic["goals_conceded"] = 0
        synthetic["yellow_cards"] = 0
        synthetic["red_cards"] = 0
        synthetic["saves"] = 0
        synthetic["bonus"] = 0
        synthetic["bps"] = synthetic["total_points"] * 10  # Rough estimate
        synthetic["influence"] = synthetic["total_points"] * 10
        synthetic["creativity"] = synthetic["total_points"] * 5
        synthetic["threat"] = synthetic["total_points"] * 5
        synthetic["ict_index"] = (
            synthetic["influence"] + synthetic["creativity"] + synthetic["threat"]
        ) / 3
        synthetic["expected_goals"] = synthetic["goals_scored"] * 0.8  # Rough estimate
        synthetic["expected_assists"] = synthetic["assists"] * 0.8
        synthetic["expected_goal_involvements"] = (
            synthetic["expected_goals"] + synthetic["expected_assists"]
        )
        synthetic["expected_goals_conceded"] = 0

        # Ensure value column exists
        if "value" not in synthetic.columns and "price" in synthetic.columns:
            synthetic["value"] = synthetic["price"] * 10

        return synthetic

    def save_models(self, filepath: str):
        """
        Save trained pipeline to file (maintains old interface)

        Args:
            filepath: Path to save the model
        """
        if self.pipeline is None:
            raise ValueError("No pipeline to save. Train or load a model first.")

        save_pipeline(self.pipeline, Path(filepath), self.pipeline_metadata)

        self._log_debug(f"Saved pipeline to {filepath}")

    def load_models(self, filepath: str):
        """
        Load trained pipeline from file (maintains old interface)

        Args:
            filepath: Path to saved model
        """
        self.model_path = filepath
        self._load_pretrained_model()

        self._log_debug(f"Loaded pipeline from {filepath}")

    def _extract_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """
        Extract prediction uncertainty from ensemble models (Random Forest, XGBoost, LightGBM, GradientBoosting, AdaBoost).

        For Random Forest: Uses tree-level predictions to calculate standard deviation.
        For XGBoost: Uses booster tree predictions to calculate standard deviation.
        For LightGBM: Uses booster tree predictions to calculate standard deviation.
        For GradientBoosting: Uses staged predictions to calculate standard deviation.
        For AdaBoost: Uses estimator predictions to calculate standard deviation.
        For other models: Returns zeros (no uncertainty available).

        Args:
            X: Feature matrix for prediction

        Returns:
            Array of standard deviations (one per sample)
        """
        if self.pipeline is None:
            return np.zeros(len(X))

        # HybridPositionModel doesn't support uncertainty extraction (yet)
        # Return zeros for now - could aggregate from underlying models in future
        if getattr(self, "is_hybrid_model", False):
            return np.zeros(len(X))

        # Get the underlying model
        if not hasattr(self.pipeline, "named_steps"):
            return np.zeros(len(X))

        model = self.pipeline.named_steps.get("model")
        if model is None:
            return np.zeros(len(X))

        # Import ensemble model types
        from sklearn.ensemble import (
            RandomForestRegressor,
            GradientBoostingRegressor,
            AdaBoostRegressor,
        )

        try:
            from xgboost import XGBRegressor
        except ImportError:
            XGBRegressor = None

        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            LGBMRegressor = None

        # Handle nested pipelines (e.g., RFE -> RandomForest/XGBoost)
        actual_model = model
        if hasattr(model, "named_steps"):
            # It's a pipeline, extract the final estimator
            if hasattr(model, "steps") and len(model.steps) > 0:
                actual_model = model.steps[-1][1]
        elif hasattr(model, "estimator_"):
            # It's a meta-estimator (e.g., RFE)
            actual_model = model.estimator_

        # Extract tree predictions for Random Forest
        if isinstance(actual_model, RandomForestRegressor):
            try:
                # Transform features through the pipeline (excluding the model step)
                feature_engineer = self.pipeline.named_steps.get("feature_engineer")
                if feature_engineer is not None:
                    X_transformed = feature_engineer.transform(X)
                else:
                    X_transformed = X

                # Handle additional pipeline steps before the model
                # CRITICAL FIX: Check if model is a Pipeline first (handles nested pipelines like [RFE -> RandomForest])
                # When RFE is nested in a Pipeline, we must iterate through Pipeline steps rather than
                # calling transform directly on RFE, which would expect input in the original feature space.
                if hasattr(model, "named_steps") or (
                    hasattr(model, "steps") and hasattr(model, "fit")
                ):
                    # Model is a pipeline, transform through all steps except final
                    # This correctly handles nested pipelines like Pipeline([RFE, RandomForest])
                    # Each step in the pipeline expects input from the previous step, not the original feature space
                    steps = (
                        model.steps
                        if hasattr(model, "steps")
                        else list(model.named_steps.items())
                    )
                    for step_name, step_transformer in steps[:-1]:
                        try:
                            X_transformed = step_transformer.transform(X_transformed)
                        except ValueError as e:
                            # Catch dimension mismatch errors and provide helpful context
                            raise ValueError(
                                f"Dimension mismatch when transforming through step '{step_name}' "
                                f"({type(step_transformer).__name__}). "
                                f"Expected features from previous step, got {X_transformed.shape[1]} features. "
                                f"This may indicate a pipeline structure mismatch. Error: {e}"
                            ) from e
                elif hasattr(model, "estimator_"):
                    # RFE or similar meta-estimator (standalone, not in a Pipeline)
                    # RFE.transform expects input in the feature space it was fitted on
                    # Since X_transformed is already from feature_engineer (which outputs the features
                    # that the full pipeline was trained on), this should work correctly
                    if hasattr(model, "transform"):
                        try:
                            X_transformed = model.transform(X_transformed)
                        except ValueError as e:
                            # Catch dimension mismatch errors for standalone RFE
                            raise ValueError(
                                f"Dimension mismatch when transforming through {type(model).__name__}. "
                                f"Expected features matching fitted feature space, got {X_transformed.shape[1]} features. "
                                f"Error: {e}"
                            ) from e
                    else:
                        # Fallback: if no transform method, skip transformation
                        # This shouldn't happen for RFE, but handle it gracefully
                        _service_logger.warning(
                            f"Meta-estimator {type(model).__name__} has estimator_ but no transform method. "
                            "Skipping transformation step."
                        )

                # Get predictions from each tree
                tree_predictions = np.array(
                    [tree.predict(X_transformed) for tree in actual_model.estimators_]
                )

                # Calculate standard deviation across trees
                uncertainty = np.std(tree_predictions, axis=0)

                self._log_debug(
                    f"✅ Extracted uncertainty from {len(actual_model.estimators_)} RF trees. "
                    f"Mean uncertainty: {uncertainty.mean():.3f}, Range: {uncertainty.min():.3f}-{uncertainty.max():.3f}"
                )

                return uncertainty

            except Exception as e:
                _service_logger.error(
                    f"Failed to extract Random Forest uncertainty. "
                    f"Model type: {type(actual_model).__name__}, "
                    f"Has estimators_: {hasattr(actual_model, 'estimators_')}, "
                    f"Error: {e}"
                )
                raise ValueError(
                    f"Uncertainty extraction failed for Random Forest model. "
                    f"This should not happen with properly trained RF models. "
                    f"Check model training and pipeline structure. Error: {e}"
                ) from e

        # Extract tree predictions for XGBoost
        if XGBRegressor and isinstance(actual_model, XGBRegressor):
            try:
                # Transform features through the pipeline (excluding the model step)
                feature_engineer = self.pipeline.named_steps.get("feature_engineer")
                if feature_engineer is not None:
                    X_transformed = feature_engineer.transform(X)
                else:
                    X_transformed = X

                # Handle additional pipeline steps before the model
                if hasattr(model, "named_steps") or (
                    hasattr(model, "steps") and hasattr(model, "fit")
                ):
                    # Model is a pipeline, transform through all steps except final
                    steps = (
                        model.steps
                        if hasattr(model, "steps")
                        else list(model.named_steps.items())
                    )
                    for step_name, step_transformer in steps[:-1]:
                        try:
                            X_transformed = step_transformer.transform(X_transformed)
                        except ValueError as e:
                            raise ValueError(
                                f"Dimension mismatch when transforming through step '{step_name}' "
                                f"({type(step_transformer).__name__}). "
                                f"Expected features from previous step, got {X_transformed.shape[1]} features. "
                                f"This may indicate a pipeline structure mismatch. Error: {e}"
                            ) from e
                elif hasattr(model, "estimator_"):
                    # RFE or similar meta-estimator (standalone, not in a Pipeline)
                    if hasattr(model, "transform"):
                        try:
                            X_transformed = model.transform(X_transformed)
                        except ValueError as e:
                            raise ValueError(
                                f"Dimension mismatch when transforming through {type(model).__name__}. "
                                f"Expected features matching fitted feature space, got {X_transformed.shape[1]} features. "
                                f"Error: {e}"
                            ) from e
                    else:
                        _service_logger.warning(
                            f"Meta-estimator {type(model).__name__} has estimator_ but no transform method. "
                            "Skipping transformation step."
                        )

                # Convert to DMatrix for XGBoost (more efficient)
                import xgboost as xgb

                # Ensure X_transformed is a DataFrame (XGBoost works with both)
                if not isinstance(X_transformed, pd.DataFrame):
                    X_transformed = pd.DataFrame(X_transformed)

                dmatrix = xgb.DMatrix(X_transformed)

                # Get the booster object
                booster = actual_model.get_booster()

                # Get number of trees
                n_trees = len(booster.get_dump())

                if n_trees == 0:
                    _service_logger.warning(
                        "XGBoost model has no trees - returning zero uncertainty"
                    )
                    return np.zeros(len(X))

                # Get predictions from individual trees by computing incremental contributions
                # For XGBoost, we'll use cumulative predictions at each iteration to infer
                # individual tree contributions, then calculate variance

                # Get cumulative predictions at each iteration
                cumulative_preds = []
                for i in range(n_trees):
                    pred = booster.predict(dmatrix, iteration_range=(0, i + 1))
                    cumulative_preds.append(pred)

                # Convert to array: shape (n_iterations, n_samples)
                cumulative_preds = np.array(cumulative_preds)

                # Calculate individual tree contributions (differences between iterations)
                # First tree's contribution is just its prediction
                # Subsequent trees add incremental predictions
                tree_contributions = np.zeros_like(cumulative_preds)
                tree_contributions[0] = cumulative_preds[0]
                for i in range(1, n_trees):
                    tree_contributions[i] = (
                        cumulative_preds[i] - cumulative_preds[i - 1]
                    )

                # Transpose to shape (n_samples, n_trees)
                tree_contributions = tree_contributions.T

                # Calculate standard deviation across tree contributions
                # This captures disagreement between trees, similar to Random Forest
                uncertainty = np.std(tree_contributions, axis=1)

                # XGBoost trees are typically weaker learners (lower learning rate)
                # Scale uncertainty to account for this (empirical scaling factor)
                # Typical XGBoost learning rate is 0.1-0.3, so individual tree variance is smaller
                # We scale by sqrt(n_trees) to get comparable uncertainty to RF
                learning_rate = getattr(actual_model, "learning_rate", 0.3)
                if learning_rate > 0:
                    uncertainty = uncertainty / learning_rate

                self._log_debug(
                    f"✅ Extracted uncertainty from {n_trees} XGBoost trees ({len(uncertainty)} samples). "
                    f"Mean uncertainty: {uncertainty.mean():.3f}, Range: {uncertainty.min():.3f}-{uncertainty.max():.3f}, "
                    f"Learning rate: {learning_rate}"
                )

                return uncertainty

            except Exception as e:
                _service_logger.error(
                    f"Failed to extract XGBoost uncertainty. "
                    f"Model type: {type(actual_model).__name__}, "
                    f"Has get_booster: {hasattr(actual_model, 'get_booster')}, "
                    f"Error: {e}"
                )
                raise ValueError(
                    f"Uncertainty extraction failed for XGBoost model. "
                    f"This should not happen with properly trained XGBoost models. "
                    f"Check model training and pipeline structure. Error: {e}"
                ) from e

        # Extract tree predictions for LightGBM
        if LGBMRegressor and isinstance(actual_model, LGBMRegressor):
            try:
                # Transform features through the pipeline (excluding the model step)
                feature_engineer = self.pipeline.named_steps.get("feature_engineer")
                if feature_engineer is not None:
                    X_transformed = feature_engineer.transform(X)
                else:
                    X_transformed = X

                # Handle additional pipeline steps before the model
                if hasattr(model, "named_steps") or (
                    hasattr(model, "steps") and hasattr(model, "fit")
                ):
                    # Model is a pipeline, transform through all steps except final
                    steps = (
                        model.steps
                        if hasattr(model, "steps")
                        else list(model.named_steps.items())
                    )
                    for step_name, step_transformer in steps[:-1]:
                        try:
                            X_transformed = step_transformer.transform(X_transformed)
                        except ValueError as e:
                            raise ValueError(
                                f"Dimension mismatch when transforming through step '{step_name}' "
                                f"({type(step_transformer).__name__}). "
                                f"Expected features from previous step, got {X_transformed.shape[1]} features. "
                                f"This may indicate a pipeline structure mismatch. Error: {e}"
                            ) from e
                elif hasattr(model, "estimator_"):
                    # RFE or similar meta-estimator (standalone, not in a Pipeline)
                    if hasattr(model, "transform"):
                        try:
                            X_transformed = model.transform(X_transformed)
                        except ValueError as e:
                            raise ValueError(
                                f"Dimension mismatch when transforming through {type(model).__name__}. "
                                f"Expected features matching fitted feature space, got {X_transformed.shape[1]} features. "
                                f"Error: {e}"
                            ) from e
                    else:
                        _service_logger.warning(
                            f"Meta-estimator {type(model).__name__} has estimator_ but no transform method. "
                            "Skipping transformation step."
                        )

                # Ensure X_transformed is a numpy array or DataFrame
                if not isinstance(X_transformed, (pd.DataFrame, np.ndarray)):
                    X_transformed = pd.DataFrame(X_transformed)

                # Get the booster object
                booster = actual_model.booster_

                # Get number of trees
                n_trees = booster.num_trees()

                if n_trees == 0:
                    _service_logger.warning(
                        "LightGBM model has no trees - returning zero uncertainty"
                    )
                    return np.zeros(len(X))

                # Get predictions from individual trees
                # LightGBM supports prediction with different iteration ranges
                tree_predictions = []
                for i in range(n_trees):
                    pred = actual_model.predict(
                        X_transformed, start_iteration=i, num_iteration=1
                    )
                    tree_predictions.append(pred)

                # Convert to array: shape (n_trees, n_samples)
                tree_predictions = np.array(tree_predictions)

                # Calculate standard deviation across trees
                uncertainty = np.std(tree_predictions, axis=0)

                # LightGBM trees are typically weaker learners (lower learning rate)
                # Scale uncertainty to account for this
                learning_rate = getattr(actual_model, "learning_rate", 0.1)
                if learning_rate > 0:
                    uncertainty = uncertainty / learning_rate

                self._log_debug(
                    f"✅ Extracted uncertainty from {n_trees} LightGBM trees ({len(uncertainty)} samples). "
                    f"Mean uncertainty: {uncertainty.mean():.3f}, Range: {uncertainty.min():.3f}-{uncertainty.max():.3f}, "
                    f"Learning rate: {learning_rate}"
                )

                return uncertainty

            except Exception as e:
                _service_logger.error(
                    f"Failed to extract LightGBM uncertainty. "
                    f"Model type: {type(actual_model).__name__}, "
                    f"Has booster_: {hasattr(actual_model, 'booster_')}, "
                    f"Error: {e}"
                )
                raise ValueError(
                    f"Uncertainty extraction failed for LightGBM model. "
                    f"This should not happen with properly trained LightGBM models. "
                    f"Check model training and pipeline structure. Error: {e}"
                ) from e

        # Extract tree predictions for GradientBoosting
        if isinstance(actual_model, GradientBoostingRegressor):
            try:
                # Transform features through the pipeline (excluding the model step)
                feature_engineer = self.pipeline.named_steps.get("feature_engineer")
                if feature_engineer is not None:
                    X_transformed = feature_engineer.transform(X)
                else:
                    X_transformed = X

                # Handle additional pipeline steps before the model
                if hasattr(model, "named_steps") or (
                    hasattr(model, "steps") and hasattr(model, "fit")
                ):
                    # Model is a pipeline, transform through all steps except final
                    steps = (
                        model.steps
                        if hasattr(model, "steps")
                        else list(model.named_steps.items())
                    )
                    for step_name, step_transformer in steps[:-1]:
                        try:
                            X_transformed = step_transformer.transform(X_transformed)
                        except ValueError as e:
                            raise ValueError(
                                f"Dimension mismatch when transforming through step '{step_name}' "
                                f"({type(step_transformer).__name__}). "
                                f"Expected features from previous step, got {X_transformed.shape[1]} features. "
                                f"This may indicate a pipeline structure mismatch. Error: {e}"
                            ) from e
                elif hasattr(model, "estimator_"):
                    # RFE or similar meta-estimator (standalone, not in a Pipeline)
                    if hasattr(model, "transform"):
                        try:
                            X_transformed = model.transform(X_transformed)
                        except ValueError as e:
                            raise ValueError(
                                f"Dimension mismatch when transforming through {type(model).__name__}. "
                                f"Expected features matching fitted feature space, got {X_transformed.shape[1]} features. "
                                f"Error: {e}"
                            ) from e
                    else:
                        _service_logger.warning(
                            f"Meta-estimator {type(model).__name__} has estimator_ but no transform method. "
                            "Skipping transformation step."
                        )

                # Get number of estimators
                n_estimators = len(actual_model.estimators_)

                if n_estimators == 0:
                    _service_logger.warning(
                        "GradientBoosting model has no estimators - returning zero uncertainty"
                    )
                    return np.zeros(len(X))

                # Use staged_predict to get cumulative predictions at each stage
                # This gives us predictions after adding each tree
                staged_preds = list(actual_model.staged_predict(X_transformed))

                # Convert to array: shape (n_stages, n_samples)
                staged_preds = np.array(staged_preds)

                # Calculate individual tree contributions (differences between stages)
                tree_contributions = np.zeros_like(staged_preds)
                tree_contributions[0] = staged_preds[0]
                for i in range(1, len(staged_preds)):
                    tree_contributions[i] = staged_preds[i] - staged_preds[i - 1]

                # Transpose to shape (n_samples, n_stages)
                tree_contributions = tree_contributions.T

                # Calculate standard deviation across tree contributions
                uncertainty = np.std(tree_contributions, axis=1)

                # GradientBoosting uses learning rate, scale accordingly
                learning_rate = getattr(actual_model, "learning_rate", 0.1)
                if learning_rate > 0:
                    uncertainty = uncertainty / learning_rate

                self._log_debug(
                    f"✅ Extracted uncertainty from {n_estimators} GradientBoosting trees ({len(uncertainty)} samples). "
                    f"Mean uncertainty: {uncertainty.mean():.3f}, Range: {uncertainty.min():.3f}-{uncertainty.max():.3f}, "
                    f"Learning rate: {learning_rate}"
                )

                return uncertainty

            except Exception as e:
                _service_logger.error(
                    f"Failed to extract GradientBoosting uncertainty. "
                    f"Model type: {type(actual_model).__name__}, "
                    f"Has estimators_: {hasattr(actual_model, 'estimators_')}, "
                    f"Error: {e}"
                )
                raise ValueError(
                    f"Uncertainty extraction failed for GradientBoosting model. "
                    f"This should not happen with properly trained GradientBoosting models. "
                    f"Check model training and pipeline structure. Error: {e}"
                ) from e

        # Extract predictions for AdaBoost
        if isinstance(actual_model, AdaBoostRegressor):
            try:
                # Transform features through the pipeline (excluding the model step)
                feature_engineer = self.pipeline.named_steps.get("feature_engineer")
                if feature_engineer is not None:
                    X_transformed = feature_engineer.transform(X)
                else:
                    X_transformed = X

                # Handle additional pipeline steps before the model
                if hasattr(model, "named_steps") or (
                    hasattr(model, "steps") and hasattr(model, "fit")
                ):
                    # Model is a pipeline, transform through all steps except final
                    steps = (
                        model.steps
                        if hasattr(model, "steps")
                        else list(model.named_steps.items())
                    )
                    for step_name, step_transformer in steps[:-1]:
                        try:
                            X_transformed = step_transformer.transform(X_transformed)
                        except ValueError as e:
                            raise ValueError(
                                f"Dimension mismatch when transforming through step '{step_name}' "
                                f"({type(step_transformer).__name__}). "
                                f"Expected features from previous step, got {X_transformed.shape[1]} features. "
                                f"This may indicate a pipeline structure mismatch. Error: {e}"
                            ) from e
                elif hasattr(model, "estimator_"):
                    # RFE or similar meta-estimator (standalone, not in a Pipeline)
                    if hasattr(model, "transform"):
                        try:
                            X_transformed = model.transform(X_transformed)
                        except ValueError as e:
                            raise ValueError(
                                f"Dimension mismatch when transforming through {type(model).__name__}. "
                                f"Expected features matching fitted feature space, got {X_transformed.shape[1]} features. "
                                f"Error: {e}"
                            ) from e
                    else:
                        _service_logger.warning(
                            f"Meta-estimator {type(model).__name__} has estimator_ but no transform method. "
                            "Skipping transformation step."
                        )

                # Get number of estimators
                n_estimators = len(actual_model.estimators_)

                if n_estimators == 0:
                    _service_logger.warning(
                        "AdaBoost model has no estimators - returning zero uncertainty"
                    )
                    return np.zeros(len(X))

                # Use staged_predict to get cumulative predictions at each stage
                staged_preds = list(actual_model.staged_predict(X_transformed))

                # Convert to array: shape (n_stages, n_samples)
                staged_preds = np.array(staged_preds)

                # Calculate individual estimator contributions (differences between stages)
                estimator_contributions = np.zeros_like(staged_preds)
                estimator_contributions[0] = staged_preds[0]
                for i in range(1, len(staged_preds)):
                    estimator_contributions[i] = staged_preds[i] - staged_preds[i - 1]

                # Transpose to shape (n_samples, n_stages)
                estimator_contributions = estimator_contributions.T

                # Calculate standard deviation across estimator contributions
                uncertainty = np.std(estimator_contributions, axis=1)

                # AdaBoost implicitly scales contributions, no additional scaling needed
                # (learning rate is applied through estimator weights)

                self._log_debug(
                    f"✅ Extracted uncertainty from {n_estimators} AdaBoost estimators ({len(uncertainty)} samples). "
                    f"Mean uncertainty: {uncertainty.mean():.3f}, Range: {uncertainty.min():.3f}-{uncertainty.max():.3f}"
                )

                return uncertainty

            except Exception as e:
                _service_logger.error(
                    f"Failed to extract AdaBoost uncertainty. "
                    f"Model type: {type(actual_model).__name__}, "
                    f"Has estimators_: {hasattr(actual_model, 'estimators_')}, "
                    f"Error: {e}"
                )
                raise ValueError(
                    f"Uncertainty extraction failed for AdaBoost model. "
                    f"This should not happen with properly trained AdaBoost models. "
                    f"Check model training and pipeline structure. Error: {e}"
                ) from e

        # No uncertainty available for non-ensemble models (Ridge, Lasso, ElasticNet)
        self._log_debug(
            f"Model type {type(actual_model).__name__} does not support uncertainty extraction - returning zeros"
        )
        return np.zeros(len(X))

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

        # HybridPositionModel doesn't support simple feature importance extraction
        # (would need to aggregate from multiple models)
        if getattr(self, "is_hybrid_model", False):
            raise ValueError(
                "Feature importance not supported for HybridPositionModel. "
                "Use get_model_for_position() to inspect individual models."
            )

        if not hasattr(self.pipeline, "named_steps"):
            raise ValueError("Pipeline does not have named_steps")

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
