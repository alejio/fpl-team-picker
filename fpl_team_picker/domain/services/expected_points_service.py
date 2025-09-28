"""Expected points calculation service."""

from typing import Dict, Any, Optional
import pandas as pd


class ExpectedPointsService:
    """Service for calculating expected points using various models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the service with configuration.

        Args:
            config: Configuration dictionary for XP models
        """
        self.config = config or {}

    def calculate_expected_points(
        self,
        gameweek_data: Dict[str, Any],
        use_ml_model: bool = False,
        gameweeks_ahead: int = 1,
    ) -> pd.DataFrame:
        """Calculate expected points for players.

        Args:
            gameweek_data: Complete gameweek data from DataOrchestrationService - guaranteed clean
            use_ml_model: Whether to use ML model or rule-based model
            gameweeks_ahead: Number of gameweeks to project (1 or 5)

        Returns:
            DataFrame with expected points - guaranteed valid
        """
        # Extract required data - trust it's clean
        players = gameweek_data["players"]
        teams = gameweek_data["teams"]
        fixtures = gameweek_data["fixtures"]
        xg_rates = gameweek_data["xg_rates"]
        target_gameweek = gameweek_data["target_gameweek"]
        live_data_historical = gameweek_data.get("live_data_historical")

        # Load configuration
        from fpl_team_picker.config import config as xp_config

        if use_ml_model:
            return self._calculate_ml_expected_points(
                players,
                teams,
                xg_rates,
                fixtures,
                target_gameweek,
                live_data_historical,
                gameweeks_ahead,
                xp_config,
            )
        else:
            return self._calculate_rule_based_expected_points(
                players,
                teams,
                xg_rates,
                fixtures,
                target_gameweek,
                live_data_historical,
                gameweeks_ahead,
                xp_config,
            )

    def calculate_combined_results(
        self, gameweek_data: Dict[str, Any], use_ml_model: bool = False
    ) -> pd.DataFrame:
        """Calculate both 1GW and 5GW expected points and merge results.

        Args:
            gameweek_data: Complete gameweek data from DataOrchestrationService - guaranteed clean
            use_ml_model: Whether to use ML model or rule-based model

        Returns:
            Merged DataFrame with 1GW and 5GW predictions - guaranteed valid
        """
        # Calculate 1GW predictions
        xp_1gw_result = self.calculate_expected_points(
            gameweek_data, use_ml_model, gameweeks_ahead=1
        )

        # Calculate 5GW predictions
        xp_5gw_result = self.calculate_expected_points(
            gameweek_data, use_ml_model, gameweeks_ahead=5
        )

        # Merge results with correct model type
        return self._merge_1gw_5gw_results(xp_1gw_result, xp_5gw_result, use_ml_model)

    def _calculate_ml_expected_points(
        self,
        players: pd.DataFrame,
        teams: pd.DataFrame,
        xg_rates: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
        live_data_historical: Optional[pd.DataFrame],
        gameweeks_ahead: int,
        xp_config: Any,
    ) -> pd.DataFrame:
        """Calculate expected points using ML model."""
        try:
            from fpl_team_picker.core.ml_xp_model import MLXPModel
            from fpl_team_picker.core.xp_model import XPModel

            # Create ML model
            ml_xp_model = MLXPModel(
                min_training_gameweeks=xp_config.xp_model.ml_min_training_gameweeks,
                training_gameweeks=xp_config.xp_model.ml_training_gameweeks,
                position_min_samples=xp_config.xp_model.ml_position_min_samples,
                ensemble_rule_weight=xp_config.xp_model.ml_ensemble_rule_weight,
                debug=xp_config.xp_model.debug,
            )

            # Create rule-based model for ensemble
            rule_xp_model = XPModel(
                form_weight=xp_config.xp_model.form_weight,
                form_window=xp_config.xp_model.form_window,
                debug=xp_config.xp_model.debug,
            )

            # Calculate expected points
            return ml_xp_model.calculate_expected_points(
                players_data=players,
                teams_data=teams,
                xg_rates_data=xg_rates,
                fixtures_data=fixtures,
                target_gameweek=target_gameweek,
                live_data=live_data_historical,
                gameweeks_ahead=gameweeks_ahead,
                rule_based_model=rule_xp_model,
            )

        except ImportError:
            # Fallback to rule-based model if ML not available
            return self._calculate_rule_based_expected_points(
                players,
                teams,
                xg_rates,
                fixtures,
                target_gameweek,
                live_data_historical,
                gameweeks_ahead,
                xp_config,
            )
        except ValueError as e:
            if "Need at least" in str(e):
                # Fallback to rule-based model
                return self._calculate_rule_based_expected_points(
                    players,
                    teams,
                    xg_rates,
                    fixtures,
                    target_gameweek,
                    live_data_historical,
                    gameweeks_ahead,
                    xp_config,
                )
            else:
                raise

    def _calculate_rule_based_expected_points(
        self,
        players: pd.DataFrame,
        teams: pd.DataFrame,
        xg_rates: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
        live_data_historical: Optional[pd.DataFrame],
        gameweeks_ahead: int,
        xp_config: Any,
    ) -> pd.DataFrame:
        """Calculate expected points using rule-based model."""
        from fpl_team_picker.core.xp_model import XPModel

        xp_model = XPModel(
            form_weight=xp_config.xp_model.form_weight,
            form_window=xp_config.xp_model.form_window,
            debug=xp_config.xp_model.debug,
        )

        return xp_model.calculate_expected_points(
            players_data=players,
            teams_data=teams,
            xg_rates_data=xg_rates,
            fixtures_data=fixtures,
            target_gameweek=target_gameweek,
            live_data=live_data_historical,
            gameweeks_ahead=gameweeks_ahead,
        )

    def _merge_1gw_5gw_results(
        self,
        players_1gw: pd.DataFrame,
        players_5gw: pd.DataFrame,
        use_ml_model: bool = False,
    ) -> pd.DataFrame:
        """Merge 1GW and 5GW results with derived metrics."""
        if use_ml_model:
            # Use ML model merge function which includes form columns
            from fpl_team_picker.core.ml_xp_model import merge_1gw_5gw_results
        else:
            # Use rule-based merge function
            from fpl_team_picker.core.xp_model import merge_1gw_5gw_results

        return merge_1gw_5gw_results(players_1gw, players_5gw)

    def get_model_info(self, use_ml_model: bool) -> Dict[str, str]:
        """Get display information about the model being used.

        Args:
            use_ml_model: Whether ML model is being used

        Returns:
            Dictionary with model information for display
        """
        from fpl_team_picker.config import config as xp_config

        if use_ml_model:
            ensemble_info = ""
            if xp_config.xp_model.ml_ensemble_rule_weight > 0:
                ensemble_info = " (with rule-based ensemble)"
            return {"type": "ML", "description": f"ML{ensemble_info}"}
        else:
            return {"type": "Rule-Based", "description": "Rule-Based"}
