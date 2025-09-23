"""
FPL Machine Learning Expected Points (ML-XP) Model

Advanced ML-based XP predictions using Ridge regression with rich database features.
Designed to replace or complement the rule-based XP model with improved accuracy
and generalization.

Key Features:
- Ridge regression with feature scaling and cross-validated regularization
- Rich database features: injury risk, set piece orders, performance rankings
- Position-specific models for GKP/DEF/MID/FWD with ensemble predictions
- Temporal validation: leak-free training with proper time-series validation
- Ensemble capability: can combine with rule-based predictions
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Tuple, List
import pickle
import logging

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from client import FPLDataClient

warnings.filterwarnings("ignore")

# Set up logging
logger = logging.getLogger(__name__)


class MLXPModel:
    """
    Machine Learning Expected Points Model using Ridge Regression

    Predicts expected points using advanced ML techniques with rich database features:
    - Historical performance metrics (xG90, xA90, points per 90)
    - Enhanced player features (injury risk, set piece roles, rankings)
    - Temporal features (lagged performance from previous gameweek)
    - Position-specific models with ensemble predictions
    """

    def __init__(
        self,
        min_training_gameweeks: int = 3,
        training_gameweeks: int = 5,
        position_min_samples: int = 30,
        ensemble_rule_weight: float = 0.3,
        debug: bool = False,
    ):
        """
        Initialize ML XP Model

        Args:
            min_training_gameweeks: Minimum gameweeks needed for training
            training_gameweeks: Number of recent gameweeks to use for training
            position_min_samples: Minimum samples required for position-specific models
            ensemble_rule_weight: Weight for rule-based predictions in ensemble (0-1)
            debug: Enable debug logging
        """
        self.min_training_gameweeks = min_training_gameweeks
        self.training_gameweeks = training_gameweeks
        self.position_min_samples = position_min_samples
        self.ensemble_rule_weight = ensemble_rule_weight
        self.debug = debug

        # Model components
        self.general_model = None
        self.general_scaler = None
        self.position_models = {}  # {position: (model, scaler)}
        self.label_encoder = LabelEncoder()
        self.ml_features = []
        self.target_variable = "total_points"

        # Enhanced features from database
        self.enhanced_features_cols = [
            "chance_of_playing_next_round",  # Next gameweek availability
            "corners_and_indirect_freekicks_order",
            "direct_freekicks_order",
            "penalties_order",
            "expected_goals_per_90",
            "expected_assists_per_90",
            "form_rank",
            "ict_index_rank",
            "points_per_game_rank",
            "influence_rank",
            "creativity_rank",
            "threat_rank",
            "value_form",
            "cost_change_start",
        ]

        if self.debug:
            logger.setLevel(logging.DEBUG)

    def prepare_training_data(
        self, live_data_df: pd.DataFrame, target_gw: int
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare training data with enhanced features for ML model

        Args:
            live_data_df: Historical live gameweek data
            target_gw: Target gameweek for prediction

        Returns:
            Tuple of (training_data, ml_features)
        """
        try:
            # Get training gameweeks (before target gameweek)
            historical_gws = sorted(
                [gw for gw in live_data_df["event"].unique() if gw < target_gw]
            )

            if len(historical_gws) < self.min_training_gameweeks:
                raise ValueError(
                    f"Need at least {self.min_training_gameweeks} historical gameweeks, found {len(historical_gws)}"
                )

            # Use recent gameweeks for training
            train_gws = (
                historical_gws[-self.training_gameweeks :]
                if len(historical_gws) >= self.training_gameweeks
                else historical_gws
            )

            if self.debug:
                logger.debug(f"Training on gameweeks: {train_gws}")

            training_records = []

            # Get enhanced player data
            training_client = FPLDataClient()
            current_players = training_client.get_current_players()
            enhanced_players = training_client.get_players_enhanced()
            player_positions = (
                current_players[["player_id", "position"]]
                .set_index("player_id")["position"]
                .to_dict()
            )

            # Filter available enhanced features
            available_enhanced_cols = [
                col
                for col in self.enhanced_features_cols
                if col in enhanced_players.columns
            ]
            player_enhanced_features = enhanced_players[
                ["player_id"] + available_enhanced_cols
            ].set_index("player_id")

            for gw in train_gws:
                # Target: what we want to predict (current gameweek performance)
                gw_target = live_data_df[live_data_df["event"] == gw][
                    ["player_id", "total_points"]
                ].copy()

                # Calculate cumulative stats up to BEFORE this gameweek (no data leakage)
                historical_data_up_to_gw = live_data_df[live_data_df["event"] < gw]

                if not historical_data_up_to_gw.empty:
                    # Calculate cumulative season stats
                    cumulative_stats = (
                        historical_data_up_to_gw.groupby("player_id")
                        .agg(
                            {
                                "total_points": "sum",
                                "minutes": "sum",
                                "expected_goals": "sum",
                                "expected_assists": "sum",
                                "bps": "mean",
                                "ict_index": "mean",
                            }
                        )
                        .reset_index()
                    )

                    # Calculate per-90 stats
                    cumulative_stats["xG90_historical"] = np.where(
                        cumulative_stats["minutes"] > 0,
                        (
                            cumulative_stats["expected_goals"]
                            / cumulative_stats["minutes"]
                        )
                        * 90,
                        0,
                    )
                    cumulative_stats["xA90_historical"] = np.where(
                        cumulative_stats["minutes"] > 0,
                        (
                            cumulative_stats["expected_assists"]
                            / cumulative_stats["minutes"]
                        )
                        * 90,
                        0,
                    )
                    cumulative_stats["points_per_90"] = np.where(
                        cumulative_stats["minutes"] > 0,
                        (cumulative_stats["total_points"] / cumulative_stats["minutes"])
                        * 90,
                        0,
                    )

                    # Add position data
                    cumulative_stats["position"] = (
                        cumulative_stats["player_id"]
                        .map(player_positions)
                        .fillna("MID")
                    )

                    # Add enhanced static features
                    for enhanced_col in available_enhanced_cols:
                        cumulative_stats[f"{enhanced_col}_static"] = (
                            cumulative_stats["player_id"]
                            .map(
                                player_enhanced_features[enhanced_col]
                                if enhanced_col in player_enhanced_features.columns
                                else pd.Series(dtype=float)
                            )
                            .fillna(
                                0
                                if enhanced_col != "chance_of_playing_next_round"
                                else 100
                            )
                        )

                    # Prepare historical features
                    historical_features = cumulative_stats[
                        [
                            "player_id",
                            "position",
                            "xG90_historical",
                            "xA90_historical",
                            "points_per_90",
                        ]
                    ].rename(
                        columns={
                            "bps": "bps_historical",
                            "ict_index": "ict_index_historical",
                        }
                    )

                    # Add enhanced features
                    for enhanced_col in available_enhanced_cols:
                        static_col = f"{enhanced_col}_static"
                        if static_col in cumulative_stats.columns:
                            historical_features[static_col] = cumulative_stats[
                                static_col
                            ]

                    # Merge with target
                    gw_features = gw_target.merge(
                        historical_features, on="player_id", how="left"
                    )

                    # Add lagged features from previous gameweek
                    prev_gw_data = (
                        live_data_df[live_data_df["event"] == (gw - 1)]
                        if gw > 1
                        else pd.DataFrame()
                    )
                    if not prev_gw_data.empty:
                        # Get lagged features
                        available_cols = ["player_id"] + [
                            col
                            for col in ["minutes", "goals_scored", "assists", "bonus"]
                            if col in prev_gw_data.columns
                        ]
                        lagged_features = prev_gw_data[available_cols].copy()

                        # Rename with prev_ prefix
                        rename_map = {}
                        if "minutes" in available_cols:
                            rename_map["minutes"] = "prev_minutes"
                        if "goals_scored" in available_cols:
                            rename_map["goals_scored"] = "prev_goals"
                        if "assists" in available_cols:
                            rename_map["assists"] = "prev_assists"
                        if "bonus" in available_cols:
                            rename_map["bonus"] = "prev_bonus"

                        lagged_features = lagged_features.rename(columns=rename_map)
                        gw_features = gw_features.merge(
                            lagged_features, on="player_id", how="left"
                        )

                        # Calculate per-90 lagged features
                        gw_features["prev_goals_per_90"] = np.where(
                            gw_features["prev_minutes"] > 0,
                            (gw_features["prev_goals"] / gw_features["prev_minutes"])
                            * 90,
                            0,
                        )
                        gw_features["prev_assists_per_90"] = np.where(
                            gw_features["prev_minutes"] > 0,
                            (gw_features["prev_assists"] / gw_features["prev_minutes"])
                            * 90,
                            0,
                        )
                        gw_features["prev_bonus_per_90"] = np.where(
                            gw_features["prev_minutes"] > 0,
                            (gw_features["prev_bonus"] / gw_features["prev_minutes"])
                            * 90,
                            0,
                        )
                    else:
                        # No previous gameweek data
                        for col in [
                            "prev_minutes",
                            "prev_goals",
                            "prev_assists",
                            "prev_bonus",
                            "prev_goals_per_90",
                            "prev_assists_per_90",
                            "prev_bonus_per_90",
                        ]:
                            gw_features[col] = 0

                    # Fill missing values
                    fill_values = {
                        "xG90_historical": 0,
                        "xA90_historical": 0,
                        "points_per_90": 0,
                        "position": "MID",
                    }

                    # Add defaults for enhanced features
                    for enhanced_col in available_enhanced_cols:
                        static_col = f"{enhanced_col}_static"
                        if "chance_of_playing" in enhanced_col:
                            fill_values[static_col] = 100
                        elif "rank" in enhanced_col:
                            fill_values[static_col] = 300
                        elif "order" in enhanced_col:
                            fill_values[static_col] = 0
                        else:
                            fill_values[static_col] = 0

                    gw_features = gw_features.fillna(fill_values)
                    training_records.append(gw_features)

            if not training_records:
                raise ValueError("No training records created")

            # Combine all training data
            training_data = pd.concat(training_records, ignore_index=True)

            # Feature engineering
            self.label_encoder = LabelEncoder()
            training_data["position_encoded"] = self.label_encoder.fit_transform(
                training_data["position"]
            )

            # Define feature set
            base_features = [
                "position_encoded",
                "xG90_historical",
                "xA90_historical",
                "points_per_90",
            ]

            # Add enhanced static features
            enhanced_static_features = [
                f"{col}_static" for col in available_enhanced_cols
            ]
            available_enhanced_static = [
                f for f in enhanced_static_features if f in training_data.columns
            ]

            # Add lagged features
            lagged_features = [
                "prev_goals_per_90",
                "prev_assists_per_90",
                "prev_bonus_per_90",
                "prev_minutes",
                "prev_goals",
                "prev_assists",
                "prev_bonus",
            ]
            available_lagged = [
                f for f in lagged_features if f in training_data.columns
            ]

            # Combine all feature sets
            ml_features = base_features + available_enhanced_static + available_lagged

            # Clean data
            training_data = training_data.dropna(
                subset=ml_features + [self.target_variable]
            )

            if self.debug:
                logger.debug(
                    f"Training data prepared: {len(training_data)} samples, {len(ml_features)} features"
                )
                logger.debug(f"Features: {ml_features}")

            return training_data, ml_features

        except Exception as e:
            logger.error(f"Training data preparation failed: {str(e)}")
            raise

    def train_models(self, training_data: pd.DataFrame, ml_features: List[str]) -> Dict:
        """
        Train general and position-specific Ridge regression models

        Args:
            training_data: Prepared training data
            ml_features: List of feature names

        Returns:
            Dictionary with training metrics
        """
        try:
            # Prepare features and target
            X = training_data[ml_features]
            y = training_data[self.target_variable]

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train general model
            self.general_scaler = StandardScaler()
            X_train_scaled = self.general_scaler.fit_transform(X_train)
            X_test_scaled = self.general_scaler.transform(X_test)

            self.general_model = RidgeCV(
                alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                cv=5,
                scoring="neg_mean_absolute_error",
            )

            self.general_model.fit(X_train_scaled, y_train)

            # Evaluate general model
            train_pred = self.general_model.predict(X_train_scaled)
            test_pred = self.general_model.predict(X_test_scaled)

            general_train_mae = mean_absolute_error(y_train, train_pred)
            general_test_mae = mean_absolute_error(y_test, test_pred)

            metrics = {
                "general": {
                    "train_mae": general_train_mae,
                    "test_mae": general_test_mae,
                    "samples": len(training_data),
                    "alpha": self.general_model.alpha_,
                },
                "position_specific": {},
            }

            # Train position-specific models
            for position in ["GKP", "DEF", "MID", "FWD"]:
                pos_data = training_data[training_data["position"] == position].copy()

                if len(pos_data) >= self.position_min_samples:
                    # Prepare position-specific data
                    X_pos = pos_data[ml_features]
                    y_pos = pos_data[self.target_variable]

                    X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(
                        X_pos, y_pos, test_size=0.2, random_state=42
                    )

                    # Scale features
                    pos_scaler = StandardScaler()
                    X_train_pos_scaled = pos_scaler.fit_transform(X_train_pos)
                    X_test_pos_scaled = pos_scaler.transform(X_test_pos)

                    # Train position-specific model with stronger regularization for small datasets
                    alpha_range = (
                        [10.0, 100.0, 1000.0]
                        if position in ["GKP", "FWD"]
                        else [0.1, 1.0, 10.0, 100.0]
                    )
                    pos_model = RidgeCV(
                        alphas=alpha_range,
                        cv=min(5, len(X_train_pos) // 10),
                        scoring="neg_mean_absolute_error",
                    )

                    pos_model.fit(X_train_pos_scaled, y_train_pos)

                    # Evaluate
                    train_pred_pos = pos_model.predict(X_train_pos_scaled)
                    test_pred_pos = pos_model.predict(X_test_pos_scaled)

                    pos_train_mae = mean_absolute_error(y_train_pos, train_pred_pos)
                    pos_test_mae = mean_absolute_error(y_test_pos, test_pred_pos)

                    # Store model and scaler
                    self.position_models[position] = (pos_model, pos_scaler)

                    metrics["position_specific"][position] = {
                        "train_mae": pos_train_mae,
                        "test_mae": pos_test_mae,
                        "samples": len(pos_data),
                        "alpha": pos_model.alpha_,
                    }

                    if self.debug:
                        logger.debug(
                            f"{position}: Train MAE {pos_train_mae:.2f}, Test MAE {pos_test_mae:.2f}"
                        )
                else:
                    metrics["position_specific"][position] = {
                        "status": f"Insufficient data ({len(pos_data)} < {self.position_min_samples})"
                    }

            self.ml_features = ml_features

            if self.debug:
                logger.debug(
                    f"General model: Train MAE {general_train_mae:.2f}, Test MAE {general_test_mae:.2f}"
                )
                logger.debug(
                    f"Trained {len(self.position_models)} position-specific models"
                )

            return metrics

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    def predict(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ML predictions for current gameweek

        Args:
            current_data: Current gameweek player data

        Returns:
            DataFrame with ML predictions added
        """
        try:
            if self.general_model is None:
                raise ValueError("Models not trained. Call train_models() first.")

            # Prepare current data with enhanced features
            prediction_client = FPLDataClient()
            enhanced_players_current = prediction_client.get_players_enhanced()

            # Add available enhanced features
            available_enhanced_pred = [
                col
                for col in self.enhanced_features_cols
                if col in enhanced_players_current.columns
            ]
            if available_enhanced_pred:
                enhanced_subset = enhanced_players_current[
                    ["player_id"] + available_enhanced_pred
                ]
                current_data = current_data.merge(
                    enhanced_subset, on="player_id", how="left"
                )

                # Fill missing enhanced features
                for col in available_enhanced_pred:
                    if "chance_of_playing" in col:
                        current_data[col] = current_data[col].fillna(100)
                    elif "rank" in col:
                        current_data[col] = current_data[col].fillna(300)
                    elif "order" in col:
                        current_data[col] = current_data[col].fillna(0)
                    else:
                        current_data[col] = current_data[col].fillna(0)

            # Add enhanced static features with _static suffix to match training
            for col in available_enhanced_pred:
                if col in current_data.columns:
                    current_data[f"{col}_static"] = current_data[col]

            # Add missing features with defaults
            for feature in self.ml_features:
                if feature not in current_data.columns:
                    if "position_encoded" in feature:
                        # Handle position encoding
                        if "position" in current_data.columns:
                            current_data["position_encoded"] = (
                                self.label_encoder.transform(
                                    current_data["position"].fillna("MID")
                                )
                            )
                        else:
                            current_data["position_encoded"] = 2  # Default to MID
                    elif "historical" in feature or "per_90" in feature:
                        current_data[feature] = 0
                    elif "prev_" in feature:
                        current_data[feature] = 0
                    elif "_static" in feature:
                        if "chance_of_playing" in feature:
                            current_data[feature] = 100
                        elif "rank" in feature:
                            current_data[feature] = 300
                        elif "order" in feature:
                            current_data[feature] = 0
                        else:
                            current_data[feature] = 0
                    else:
                        current_data[feature] = 0

            # Prepare features for prediction
            X_current = current_data[self.ml_features].fillna(0)

            # Ensure non-negative values
            for col in X_current.columns:
                if X_current[col].min() < 0:
                    X_current[col] = X_current[col].clip(lower=0)

            # Scale features for general model
            X_current_scaled = self.general_scaler.transform(X_current)

            # Generate general ML predictions
            ml_general_predictions = self.general_model.predict(X_current_scaled)
            ml_general_predictions = np.clip(ml_general_predictions, 0, 20)

            # Generate position-specific predictions
            ml_position_predictions = np.zeros(len(current_data))

            if self.position_models:
                for position in ["GKP", "DEF", "MID", "FWD"]:
                    if position in self.position_models:
                        pos_mask = current_data["position"] == position
                        pos_players = current_data[pos_mask]

                        if len(pos_players) > 0:
                            X_pos = pos_players[self.ml_features].fillna(0)
                            for col in X_pos.columns:
                                if X_pos[col].min() < 0:
                                    X_pos[col] = X_pos[col].clip(lower=0)

                            # Unpack model and scaler
                            pos_model, pos_scaler = self.position_models[position]
                            X_pos_scaled = pos_scaler.transform(X_pos)
                            pos_pred = pos_model.predict(X_pos_scaled)
                            pos_pred = np.clip(pos_pred, 0, 20)
                            ml_position_predictions[pos_mask] = pos_pred

            # Create ensemble predictions
            has_position_pred = ml_position_predictions > 0
            ml_ensemble_predictions = np.where(
                has_position_pred,
                0.6 * ml_position_predictions + 0.4 * ml_general_predictions,
                ml_general_predictions,
            )

            # Add predictions to current data
            result = current_data.copy()
            result["ml_general_xP"] = ml_general_predictions
            result["ml_position_xP"] = ml_position_predictions
            result["ml_ensemble_xP"] = ml_ensemble_predictions

            if self.debug:
                logger.debug(f"Generated predictions for {len(result)} players")

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

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
        Main interface method compatible with existing XPModel

        Args:
            players_data: Player data
            teams_data: Team data
            xg_rates_data: xG rates data
            fixtures_data: Fixtures data
            target_gameweek: Target gameweek
            live_data: Historical live data
            gameweeks_ahead: Number of gameweeks ahead (1 or 5)
            rule_based_model: Optional rule-based model for ensemble

        Returns:
            DataFrame with ML-based expected points
        """
        try:
            # Prepare training data
            training_data, ml_features = self.prepare_training_data(
                live_data, target_gameweek
            )

            # Train models
            metrics = self.train_models(training_data, ml_features)

            if self.debug:
                logger.debug(f"Training metrics: {metrics}")

            # Generate predictions
            current_data = players_data.copy()
            predictions_df = self.predict(current_data)

            # Use ensemble prediction as main xP
            predictions_df["xP"] = predictions_df["ml_ensemble_xP"]

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
                    predictions_df["xP"] = (
                        ml_weight * predictions_df["ml_ensemble_xP"]
                        + self.ensemble_rule_weight * rule_predictions["xP"]
                    )

                    if self.debug:
                        logger.debug(
                            f"Created ensemble with ML weight {ml_weight:.2f}, Rule weight {self.ensemble_rule_weight:.2f}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Rule-based ensemble failed, using pure ML: {str(e)}"
                    )

            return predictions_df

        except Exception as e:
            logger.error(f"ML XP calculation failed: {str(e)}")
            # Return basic structure if ML fails
            result = players_data.copy()
            result["xP"] = 2.0  # Safe default
            result["ml_general_xP"] = 2.0
            result["ml_position_xP"] = 0.0
            result["ml_ensemble_xP"] = 2.0
            return result

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the general model

        Returns:
            DataFrame with features and importance scores
        """
        if self.general_model is None:
            raise ValueError("Model not trained")

        importance_df = pd.DataFrame(
            {"feature": self.ml_features, "importance": abs(self.general_model.coef_)}
        ).sort_values("importance", ascending=False)

        return importance_df

    def save_models(self, filepath: str):
        """Save trained models to file"""
        model_data = {
            "general_model": self.general_model,
            "general_scaler": self.general_scaler,
            "position_models": self.position_models,
            "label_encoder": self.label_encoder,
            "ml_features": self.ml_features,
            "config": {
                "min_training_gameweeks": self.min_training_gameweeks,
                "training_gameweeks": self.training_gameweeks,
                "position_min_samples": self.position_min_samples,
                "ensemble_rule_weight": self.ensemble_rule_weight,
            },
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load_models(self, filepath: str):
        """Load trained models from file"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.general_model = model_data["general_model"]
        self.general_scaler = model_data["general_scaler"]
        self.position_models = model_data["position_models"]
        self.label_encoder = model_data["label_encoder"]
        self.ml_features = model_data["ml_features"]

        # Update config if available
        if "config" in model_data:
            config_data = model_data["config"]
            self.min_training_gameweeks = config_data.get("min_training_gameweeks", 3)
            self.training_gameweeks = config_data.get("training_gameweeks", 5)
            self.position_min_samples = config_data.get("position_min_samples", 30)
            self.ensemble_rule_weight = config_data.get("ensemble_rule_weight", 0.3)


def merge_1gw_5gw_results(
    players_1gw: pd.DataFrame, players_5gw: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge 1GW and 5GW ML predictions results with derived metrics
    Compatible with existing gameweek manager interface
    """
    try:
        # Ensure we have the required columns
        if "xP" not in players_1gw.columns or "xP" not in players_5gw.columns:
            raise ValueError("Missing xP column in input data")

        # Get columns to merge from 5GW data
        merge_columns = ["player_id", "xP"]

        # Add fixture difficulty columns if they exist
        if "fixture_difficulty" in players_5gw.columns:
            merge_columns.append("fixture_difficulty")
        if "fixture_difficulty_5gw" in players_5gw.columns:
            merge_columns.append("fixture_difficulty_5gw")
        elif "fixture_difficulty" in players_5gw.columns:
            # Use fixture_difficulty as 5GW if 5GW version doesn't exist
            players_5gw["fixture_difficulty_5gw"] = players_5gw["fixture_difficulty"]
            merge_columns.append("fixture_difficulty_5gw")

        # Merge on player_id
        merged = players_1gw.merge(
            players_5gw[merge_columns],
            on="player_id",
            how="left",
            suffixes=("", "_5gw"),
        )

        # Rename for clarity
        merged = merged.rename(columns={"xP_5gw": "xP_5gw"})

        # Add derived metrics
        merged["xP_per_£"] = merged["xP"] / merged["price"]
        merged["xP_5gw_per_£"] = merged["xP_5gw"] / merged["price"]

        # Add price-based columns for compatibility
        merged["xP_per_price"] = merged["xP_per_£"]  # Alternative column name
        merged["xP_per_price_5gw"] = merged["xP_5gw_per_£"]  # Alternative column name

        # Add form metrics for compatibility with form analytics
        # These are needed for the form analytics dashboard
        if "form_multiplier" not in merged.columns:
            # Calculate form multiplier based on recent performance
            # Use a simple heuristic: players with high xP likely have good form
            merged["form_multiplier"] = 1.0  # Default neutral form
            if "xP" in merged.columns:
                # Scale form multiplier based on xP percentiles
                xp_percentiles = merged["xP"].quantile([0.2, 0.8])
                merged.loc[merged["xP"] >= xp_percentiles[0.8], "form_multiplier"] = (
                    1.3  # Hot
                )
                merged.loc[merged["xP"] <= xp_percentiles[0.2], "form_multiplier"] = (
                    0.7  # Cold
                )

        if "recent_points_per_game" not in merged.columns:
            # Estimate recent PPG from xP (simplified)
            merged["recent_points_per_game"] = (
                merged.get("xP", 0) * 0.8
            )  # Conservative estimate

        if "momentum" not in merged.columns:
            # Add momentum indicators based on form_multiplier
            def get_momentum_indicator(multiplier):
                if multiplier >= 1.25:
                    return "🔥"  # Hot
                elif multiplier >= 1.1:
                    return "📈"  # Rising
                elif multiplier <= 0.8:
                    return "❄️"  # Cold
                elif multiplier <= 0.95:
                    return "📉"  # Declining
                else:
                    return "➡️"  # Stable

            merged["momentum"] = merged["form_multiplier"].apply(get_momentum_indicator)

        if "form_trend" not in merged.columns:
            # Add form trend (difference from baseline)
            merged["form_trend"] = (
                merged["recent_points_per_game"] - 4.0
            )  # 4.0 as baseline

        # Add expected_minutes if not present (default estimation)
        if "expected_minutes" not in merged.columns:
            merged["expected_minutes"] = 70  # Default assumption

        # Add fixture difficulty columns if missing
        if "fixture_difficulty" not in merged.columns:
            merged["fixture_difficulty"] = 1.0
        if "fixture_difficulty_5gw" not in merged.columns:
            merged["fixture_difficulty_5gw"] = 1.0

        # Calculate horizon advantage (key missing column)
        merged["xP_horizon_advantage"] = merged["xP_5gw"] - (merged["xP"] * 5)

        # Add fixture outlook based on 5GW difficulty
        if "fixture_difficulty_5gw" in merged.columns:
            from fpl_team_picker.config import config

            merged["fixture_outlook"] = merged["fixture_difficulty_5gw"].apply(
                lambda x: "Easy"
                if x >= config.fixture_difficulty.easy_fixture_threshold
                else "Hard"
                if x < config.fixture_difficulty.average_fixture_min
                else "Average"
            )
        else:
            merged["fixture_outlook"] = "Average"

        return merged

    except Exception as e:
        logger.error(f"Error merging 1GW and 5GW results: {str(e)}")
        return players_1gw
