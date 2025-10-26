"""Expected points calculation service."""

from typing import Dict, Any, Optional
import pandas as pd
import warnings
import logging

warnings.filterwarnings("ignore")

# Set up logging
logger = logging.getLogger(__name__)


class ExpectedPointsService:
    """Service for calculating expected points using various models.

    Incorporates rule-based and ML-based XP calculation engines with:
    - Form-weighted predictions with live data integration
    - Statistical xG/xA estimation for missing data
    - Dynamic team strength ratings
    - Enhanced minutes prediction model
    - Multi-gameweek horizon capability
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the service with configuration.

        Args:
            config: Configuration dictionary for XP models
        """
        self.config = config or {}

        # Load default configuration from config module
        from fpl_team_picker.config import config as xp_config

        # XP Model parameters (for rule-based model)
        # Override with custom config if provided (for algorithm experiments)
        self.form_weight = self.config.get(
            "form_weight", xp_config.xp_model.form_weight
        )
        self.form_window = self.config.get(
            "form_window", xp_config.xp_model.form_window
        )
        self.debug = self.config.get("debug", xp_config.xp_model.debug)

        # Model component caches
        self._team_strength_cache = {}
        self._player_form_cache = {}

        if self.debug:
            print(
                f"🧠 ExpectedPointsService initialized - (Rule-based Form weight: {self.form_weight}, Window: {self.form_window} GWs"
            )

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
        from fpl_team_picker.domain.services.ml_expected_points_service import (
            MLExpectedPointsService,
        )

        # Create ML model
        ml_xp_model = MLExpectedPointsService(
            min_training_gameweeks=xp_config.xp_model.ml_min_training_gameweeks,
            training_gameweeks=xp_config.xp_model.ml_training_gameweeks,
            position_min_samples=xp_config.xp_model.ml_position_min_samples,
            ensemble_rule_weight=xp_config.xp_model.ml_ensemble_rule_weight,
            debug=xp_config.xp_model.debug,
        )

        # Create a simple wrapper that provides XPModel-compatible interface
        class RuleBasedWrapper:
            """Temporary wrapper to provide XPModel-compatible interface."""

            def __init__(self, service_instance):
                self.service = service_instance

            def calculate_expected_points(
                self,
                players_data,
                teams_data,
                xg_rates_data,
                fixtures_data,
                target_gameweek,
                live_data,
                gameweeks_ahead=1,
            ):
                gameweek_data = {
                    "players": players_data,
                    "teams": teams_data,
                    "fixtures": fixtures_data,
                    "xg_rates": xg_rates_data,
                    "target_gameweek": target_gameweek,
                    "live_data_historical": live_data,
                }
                return self.service.calculate_expected_points(
                    gameweek_data,
                    use_ml_model=False,
                    gameweeks_ahead=gameweeks_ahead,
                )

        rule_based_model_wrapper = RuleBasedWrapper(self)

        # Calculate expected points with ML model using rule-based as ensemble
        return ml_xp_model.calculate_expected_points(
            players_data=players,
            teams_data=teams,
            xg_rates_data=xg_rates,
            fixtures_data=fixtures,
            target_gameweek=target_gameweek,
            live_data=live_data_historical,
            gameweeks_ahead=gameweeks_ahead,
            rule_based_model=rule_based_model_wrapper,
        )

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
        """Calculate expected points using rule-based model (internal implementation)."""
        if self.debug:
            print(
                f"🔮 Calculating XP for GW{target_gameweek} (+{gameweeks_ahead - 1} ahead)"
            )

        # Prepare dataset
        players_xp = players.copy()

        # Merge team information
        players_xp = self._merge_team_data(players_xp, teams)

        # Calculate team strength ratings
        team_strength = self._get_dynamic_team_strength(teams, players, target_gameweek)

        # Merge xG/xA rates with statistical estimation for missing data
        players_xp = self._merge_xg_xa_rates(players_xp, xg_rates, team_strength)

        # Calculate form-weighted performance if live data available
        if live_data_historical is not None and not live_data_historical.empty:
            players_xp = self._apply_form_weighting(
                players_xp, live_data_historical, target_gameweek
            )

        # Calculate expected minutes
        players_xp = self._calculate_expected_minutes(players_xp)

        # Calculate fixture difficulty for target gameweek(s)
        fixture_difficulty = self._calculate_fixture_difficulty(
            fixtures, teams, team_strength, target_gameweek, gameweeks_ahead
        )
        players_xp = self._apply_fixture_difficulty(players_xp, fixture_difficulty)

        # Calculate XP components
        players_xp = self._calculate_xp_components(players_xp, gameweeks_ahead)

        # Add efficiency metrics
        players_xp["xP_per_price"] = players_xp["xP"] / players_xp.get(
            "price", players_xp.get("price_gbp", 1)
        )

        if self.debug:
            print(f"✅ XP calculated for {len(players_xp)} players")
            print(
                f"📊 Average XP: {players_xp['xP'].mean():.2f}, Top XP: {players_xp['xP'].max():.2f}"
            )

        return players_xp

    def _merge_1gw_5gw_results(
        self,
        players_1gw: pd.DataFrame,
        players_5gw: pd.DataFrame,
        use_ml_model: bool = False,
    ) -> pd.DataFrame:
        """Merge 1GW and 5GW results with derived metrics (internal implementation).

        Args:
            players_1gw: DataFrame with 1-gameweek XP calculations
            players_5gw: DataFrame with 5-gameweek XP calculations
            use_ml_model: Whether ML model was used (affects merge logic)

        Returns:
            DataFrame with merged results and derived metrics
        """
        if "player_id" in players_1gw.columns and "player_id" in players_5gw.columns:
            merge_cols = ["player_id"] + [
                col
                for col in [
                    "xP",
                    "xP_per_price",
                    "fixture_difficulty",
                    "expected_minutes",
                ]
                if col in players_5gw.columns
            ]

            # Create 5GW suffixed columns
            suffix_data = players_5gw[merge_cols].copy()
            for col in merge_cols:
                if col != "player_id":
                    suffix_data[f"{col}_5gw"] = suffix_data[col]
                    suffix_data = suffix_data.drop(col, axis=1)

            players_merged = players_1gw.merge(suffix_data, on="player_id", how="left")
        else:
            # Fallback: estimate 5GW from 1GW
            players_merged = players_1gw.copy()
            players_merged["xP_5gw"] = players_merged["xP"] * 4.0
            players_merged["xP_per_price_5gw"] = (
                players_merged.get("xP_per_price", players_merged["xP"]) * 4.0
            )
            players_merged["fixture_difficulty_5gw"] = players_merged.get(
                "fixture_difficulty", 1.0
            )

        # Ensure numeric dtypes for key columns (can become object during merge)
        numeric_cols = [
            "xP",
            "xP_5gw",
            "price",
            "price_gbp",
            "xP_per_price",
            "xP_per_price_5gw",
            "fixture_difficulty",
            "fixture_difficulty_5gw",
            "expected_minutes",
            "expected_minutes_5gw",
        ]
        for col in numeric_cols:
            if col in players_merged.columns:
                players_merged[col] = pd.to_numeric(
                    players_merged[col], errors="coerce"
                )

        # Ensure xP_per_price exists (needed by form analytics and UI)
        if "xP_per_price" not in players_merged.columns:
            price_col = "price" if "price" in players_merged.columns else "price_gbp"
            players_merged["xP_per_price"] = players_merged["xP"] / players_merged[
                price_col
            ].replace(0, 1)

        # Add derived metrics
        players_merged["xP_horizon_advantage"] = players_merged["xP_5gw"] - (
            players_merged["xP"] * 5
        )

        # Add fixture outlook if fixture_difficulty_5gw column exists
        from fpl_team_picker.config import config

        if "fixture_difficulty_5gw" in players_merged.columns:
            players_merged["fixture_outlook"] = players_merged[
                "fixture_difficulty_5gw"
            ].apply(
                lambda x: "🟢 Easy"
                if x >= config.fixture_difficulty.easy_fixture_threshold
                else "🟡 Average"
                if x >= config.fixture_difficulty.average_fixture_min
                else "🔴 Hard"
            )
        else:
            players_merged["fixture_outlook"] = "🟡 Average"

        # Add form analytics columns (needed for form dashboard)
        if "form_multiplier" not in players_merged.columns:
            # Calculate form multiplier based on recent performance
            # Use a simple heuristic: players with high xP likely have good form
            players_merged["form_multiplier"] = 1.0  # Default neutral form
            if "xP" in players_merged.columns:
                # Scale form multiplier based on xP percentiles
                xp_percentiles = players_merged["xP"].quantile([0.2, 0.8])
                players_merged.loc[
                    players_merged["xP"] >= xp_percentiles[0.8], "form_multiplier"
                ] = 1.3  # Hot
                players_merged.loc[
                    players_merged["xP"] <= xp_percentiles[0.2], "form_multiplier"
                ] = 0.7  # Cold

        if "recent_points_per_game" not in players_merged.columns:
            # Estimate recent PPG from xP (simplified)
            players_merged["recent_points_per_game"] = (
                players_merged.get("xP", 0) * 0.8
            )  # Conservative estimate

        if "momentum" not in players_merged.columns:
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

            players_merged["momentum"] = players_merged["form_multiplier"].apply(
                get_momentum_indicator
            )

        if "form_trend" not in players_merged.columns:
            # Add form trend (difference from baseline)
            players_merged["form_trend"] = (
                players_merged["recent_points_per_game"] - 4.0
            )  # 4.0 as baseline

        return players_merged

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

    # ========== Private XP Calculation Methods (migrated from core/xp_model.py) ==========

    def _merge_team_data(
        self, players_df: pd.DataFrame, teams_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge team information with players.

        Trusts DataOrchestrationService to provide standardized data with:
        - teams_df has 'team_id' and 'name' columns
        - players_df has 'team' column
        """
        merged = players_df.merge(
            teams_df[["team_id", "name"]],
            left_on="team",
            right_on="team_id",
            how="left",
        )

        # Clean up duplicate team_id column from merge
        if "team_id" in merged.columns:
            merged = merged.drop("team_id", axis=1)

        return merged

    def _get_dynamic_team_strength(
        self, teams_df: pd.DataFrame, players_df: pd.DataFrame, target_gameweek: int
    ) -> Dict[str, float]:
        """Calculate dynamic team strength ratings using TeamAnalyticsService."""
        try:
            from fpl_team_picker.domain.services.team_analytics_service import (
                TeamAnalyticsService,
            )

            # Initialize team analytics service
            analytics_service = TeamAnalyticsService(debug=self.debug)

            # Load current season data for dynamic calculation
            current_season_data = analytics_service.load_historical_gameweek_data(
                start_gw=1, end_gw=target_gameweek - 1
            )

            # Get dynamic team strength ratings
            strength_ratings = analytics_service.get_team_strength(
                target_gameweek=target_gameweek,
                teams_data=teams_df,
                current_season_data=current_season_data,
            )

            if self.debug:
                if target_gameweek >= 8:
                    print(
                        f"🔥 Using current season team strength (GW{target_gameweek})"
                    )
                else:
                    print(f"📊 Using weighted team strength (GW{target_gameweek})")

                strongest = max(strength_ratings.items(), key=lambda x: x[1])
                weakest = min(strength_ratings.items(), key=lambda x: x[1])
                print(f"   Strongest: {strongest[0]} ({strongest[1]})")
                print(f"   Weakest: {weakest[0]} ({weakest[1]})")

            self._team_strength_cache = strength_ratings
            return strength_ratings

        except Exception as e:
            if self.debug:
                print(f"⚠️ Dynamic team strength failed: {e}, using static fallback")

            # Fallback to static ratings
            return self._get_static_team_strength_fallback()

    def _get_static_team_strength_fallback(self) -> Dict[str, float]:
        """Fallback to static previous season table ratings if dynamic calculation fails"""
        team_positions = {
            "Liverpool": 1,
            "Arsenal": 2,
            "Manchester City": 3,
            "Chelsea": 4,
            "Newcastle United": 5,
            "Aston Villa": 6,
            "Nottingham Forest": 7,
            "Brighton & Hove Albion": 8,
            "Bournemouth": 9,
            "Brentford": 10,
            "Fulham": 11,
            "Crystal Palace": 12,
            "Everton": 13,
            "West Ham United": 14,
            "Manchester United": 15,
            "Wolverhampton Wanderers": 16,
            "Tottenham Hotspur": 17,
            "Leicester City": 18,
            "Ipswich Town": 19,
            "Southampton": 20,
            # Handle team name variations
            "Man City": 3,
            "Man Utd": 15,
            "Newcastle": 5,
            "Brighton": 8,
            "West Ham": 14,
            "Nott'm Forest": 7,
            "Spurs": 17,
            "Leicester": 18,
            "Ipswich": 19,
            "Wolves": 16,
        }

        strength_ratings = {}
        for team, position in team_positions.items():
            if position <= 20:
                strength = 1.3 - (position - 1) * (1.3 - 0.7) / 19
            else:
                strength = 0.7 - (position - 20) * 0.05
            strength_ratings[team] = round(strength, 3)

        return strength_ratings

    def _merge_xg_xa_rates(
        self,
        players_df: pd.DataFrame,
        xg_rates_df: pd.DataFrame,
        team_strength: Dict[str, float],
    ) -> pd.DataFrame:
        """Merge xG/xA rates with statistical estimation for missing data.

        Trusts DataOrchestrationService to provide standardized data with:
        - Both DataFrames have 'player_id' column
        - xg_rates_df has 'xG90' and 'xA90' columns (may be empty)
        """
        # Merge with provided xG rates
        players_with_xg = players_df.merge(
            xg_rates_df[["player_id", "xG90", "xA90"]],
            on="player_id",
            how="left",
            suffixes=("", "_xg"),
        )

        # Identify players with missing xG/xA data
        missing_xg = players_with_xg["xG90"].isna().sum()
        total_players = len(players_with_xg)

        if missing_xg > 0 and self.debug:
            print(
                f"⚙️ Missing xG/xA for {missing_xg}/{total_players} players - estimating statistically"
            )

        # Estimate missing xG/xA rates using team strength and position
        players_with_xg = self._estimate_missing_xg_xa_rates(
            players_with_xg, team_strength
        )

        return players_with_xg

    def _estimate_missing_xg_xa_rates(
        self, players_df: pd.DataFrame, team_strength: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Statistically estimate missing xG/xA rates based on:
        - Team strength (stronger teams → higher xG production)
        - Player position (FWD > MID > DEF/GK)
        - Historical median rates by position
        """
        # Position-based median xG90/xA90 from historical data
        position_medians = {
            "FWD": {"xG90": 0.45, "xA90": 0.15},
            "MID": {"xG90": 0.20, "xA90": 0.15},
            "DEF": {"xG90": 0.05, "xA90": 0.08},
            "GKP": {"xG90": 0.00, "xA90": 0.00},
        }

        # Map team strength to each player
        players_df["team_strength"] = players_df["name"].map(team_strength).fillna(1.0)

        # Estimate missing xG90
        missing_xg_mask = players_df["xG90"].isna()
        if missing_xg_mask.sum() > 0:
            players_df.loc[missing_xg_mask, "xG90"] = players_df.loc[
                missing_xg_mask
            ].apply(
                lambda row: position_medians.get(row["position"], {"xG90": 0.10})[
                    "xG90"
                ]
                * row["team_strength"],
                axis=1,
            )

        # Estimate missing xA90
        missing_xa_mask = players_df["xA90"].isna()
        if missing_xa_mask.sum() > 0:
            players_df.loc[missing_xa_mask, "xA90"] = players_df.loc[
                missing_xa_mask
            ].apply(
                lambda row: position_medians.get(row["position"], {"xA90": 0.05})[
                    "xA90"
                ]
                * row["team_strength"],
                axis=1,
            )

        return players_df

    def _apply_form_weighting(
        self,
        players_df: pd.DataFrame,
        live_data: pd.DataFrame,
        target_gameweek: int,
    ) -> pd.DataFrame:
        """Apply form-weighted adjustments to player performance metrics"""
        # Calculate recent form scores
        form_data = self._calculate_recent_form(live_data, target_gameweek)

        if form_data.empty:
            if self.debug:
                print(
                    "⚠️ No form data available - skipping form weighting (early season)"
                )
            return players_df

        # Merge form data
        players_with_form = players_df.merge(
            form_data[["player_id", "form_multiplier"]],
            on="player_id",
            how="left",
        )

        # Fill missing form values (new players) with neutral 1.0
        players_with_form["form_multiplier"] = players_with_form[
            "form_multiplier"
        ].fillna(1.0)

        # Apply form weighting to xG/xA rates
        players_with_form["xG90"] = (
            players_with_form["xG90"] * players_with_form["form_multiplier"]
        )
        players_with_form["xA90"] = (
            players_with_form["xA90"] * players_with_form["form_multiplier"]
        )

        if self.debug:
            improved = (players_with_form["form_multiplier"] > 1.05).sum()
            declined = (players_with_form["form_multiplier"] < 0.95).sum()
            print(
                f"📈 Form adjustments: {improved} players boosted, {declined} players reduced"
            )

        return players_with_form

    def _calculate_recent_form(
        self, live_data: pd.DataFrame, target_gameweek: int
    ) -> pd.DataFrame:
        """Calculate recent form scores based on last N gameweeks of performance.

        Trusts DataOrchestrationService to provide standardized data with:
        - live_data has 'gameweek', 'total_points', 'minutes', 'player_id' columns
        """
        # Filter to recent gameweeks only
        form_window_start = max(1, target_gameweek - self.form_window)
        recent_data = live_data[
            (live_data["gameweek"] >= form_window_start)
            & (live_data["gameweek"] < target_gameweek)
        ].copy()

        if recent_data.empty:
            return pd.DataFrame()

        # Calculate actual points per 90 minutes
        recent_data["pts_per_90"] = (
            recent_data["total_points"] / recent_data["minutes"].replace(0, 1) * 90
        )

        # Aggregate form metrics per player
        form_data = (
            recent_data.groupby("player_id")
            .agg(
                {
                    "pts_per_90": "mean",
                    "minutes": "sum",
                    "gameweek": "count",  # Number of gameweeks played
                }
            )
            .reset_index()
        )

        form_data = form_data.rename(
            columns={"gameweek": "gws_played", "pts_per_90": "recent_pts_per_90"}
        )

        # Calculate form multiplier (1.0 = average, >1.0 = hot form, <1.0 = cold form)
        # Use weighted approach: form_weight parameter controls influence
        global_avg_pts_per_90 = recent_data["pts_per_90"].mean()

        form_data["form_multiplier"] = (
            1.0
            + self.form_weight
            * (form_data["recent_pts_per_90"] - global_avg_pts_per_90)
            / global_avg_pts_per_90
        )

        # Clamp form multipliers to reasonable range [0.5, 1.5]
        form_data["form_multiplier"] = form_data["form_multiplier"].clip(0.5, 1.5)

        # Reduce confidence for players with limited recent minutes
        min_minutes_threshold = 180  # 2 full games worth
        form_data.loc[
            form_data["minutes"] < min_minutes_threshold, "form_multiplier"
        ] = 1.0

        return form_data

    def _calculate_expected_minutes(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced minutes prediction model"""
        from fpl_team_picker.config import config

        # Use recent minutes trend if available (last 3 gameweeks weighted)
        if "minutes_3gw_avg" in players_df.columns:
            players_df["expected_minutes"] = (
                players_df["minutes_3gw_avg"]
                .fillna(0)
                .clip(
                    config.minutes_prediction.min_predicted_minutes,
                    config.minutes_prediction.max_predicted_minutes,
                )
            )
        else:
            # Fallback to position-based estimates
            position_minutes = {"GKP": 90, "DEF": 80, "MID": 75, "FWD": 70}
            players_df["expected_minutes"] = players_df["position"].map(
                position_minutes
            )

        # Reduce for high-risk players (injury concerns, suspensions, rotations)
        if "chance_of_playing_next_round" in players_df.columns:
            injury_risk = players_df["chance_of_playing_next_round"].fillna(100) / 100.0
            players_df["expected_minutes"] = (
                players_df["expected_minutes"] * injury_risk
            )

        return players_df

    def _calculate_fixture_difficulty(
        self,
        fixtures_df: pd.DataFrame,
        teams_df: pd.DataFrame,
        team_strength: Dict[str, float],
        target_gameweek: int,
        gameweeks_ahead: int,
    ) -> pd.DataFrame:
        """Calculate fixture difficulty scores for each team.

        Trusts DataOrchestrationService to provide standardized data with:
        - fixtures_df has 'gameweek', 'team_h', 'team_a' columns
        - teams_df has 'team_id' and 'name' columns
        """

        # Filter fixtures for target gameweek range
        target_fixtures = fixtures_df[
            (fixtures_df["gameweek"] >= target_gameweek)
            & (fixtures_df["gameweek"] < target_gameweek + gameweeks_ahead)
        ].copy()

        if target_fixtures.empty:
            # Return default neutral difficulty for all teams
            return pd.DataFrame(
                {
                    "team_id": teams_df["team_id"],
                    "fixture_difficulty": 1.0,
                    "num_fixtures": 0,
                }
            )

        # Calculate difficulty for each fixture
        fixture_difficulties = []

        for _, fix in target_fixtures.iterrows():
            home_team_id = fix["team_h"]
            away_team_id = fix["team_a"]

            # Get team names from team IDs
            home_team_name = teams_df[teams_df["team_id"] == home_team_id][
                "name"
            ].values
            away_team_name = teams_df[teams_df["team_id"] == away_team_id][
                "name"
            ].values

            if len(home_team_name) == 0 or len(away_team_name) == 0:
                continue

            home_team_name = home_team_name[0]
            away_team_name = away_team_name[0]

            # Get opponent strength
            home_opponent_strength = team_strength.get(away_team_name, 1.0)
            away_opponent_strength = team_strength.get(home_team_name, 1.0)

            # Fixture difficulty: higher value = easier fixture (inverse of opponent strength)
            # Home teams get slight advantage boost (10%)
            home_advantage_multiplier = 1.1
            home_difficulty = (2.0 - home_opponent_strength) * home_advantage_multiplier
            away_difficulty = 2.0 - away_opponent_strength

            fixture_difficulties.append(
                {
                    "team_id": home_team_id,
                    "fixture_difficulty": home_difficulty,
                    "gameweek": fix["gameweek"],
                }
            )

            fixture_difficulties.append(
                {
                    "team_id": away_team_id,
                    "fixture_difficulty": away_difficulty,
                    "gameweek": fix["gameweek"],
                }
            )

        if not fixture_difficulties:
            return pd.DataFrame(
                {
                    "team_id": teams_df["team_id"],
                    "fixture_difficulty": 1.0,
                    "num_fixtures": 0,
                }
            )

        difficulty_df = pd.DataFrame(fixture_difficulties)

        # Aggregate difficulty across multiple gameweeks
        aggregated = (
            difficulty_df.groupby("team_id")
            .agg({"fixture_difficulty": "mean", "gameweek": "count"})
            .reset_index()
        )

        aggregated = aggregated.rename(columns={"gameweek": "num_fixtures"})

        return aggregated

    def _apply_fixture_difficulty(
        self, players_df: pd.DataFrame, fixture_difficulty: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply fixture difficulty multipliers to player XP.

        Trusts DataOrchestrationService to provide standardized data with:
        - players_df has 'team' column
        - fixture_difficulty has 'team_id' and 'fixture_difficulty' columns
        """
        # Merge fixture difficulty
        players_with_fixtures = players_df.merge(
            fixture_difficulty[["team_id", "fixture_difficulty"]],
            left_on="team",
            right_on="team_id",
            how="left",
        )

        # Fill missing fixture difficulty with neutral 1.0
        players_with_fixtures["fixture_difficulty"] = players_with_fixtures[
            "fixture_difficulty"
        ].fillna(1.0)

        # Clean up duplicate team_id column from merge
        if "team_id" in players_with_fixtures.columns:
            players_with_fixtures = players_with_fixtures.drop("team_id", axis=1)

        return players_with_fixtures

    def _calculate_xp_components(
        self, players_df: pd.DataFrame, gameweeks_ahead: int
    ) -> pd.DataFrame:
        """Calculate expected points from all FPL scoring components.

        Uses standard FPL points system (hardcoded, not configurable):
        - Appearance: 1 pt (<60 min), 2 pts (60+ min)
        - Goals: 6 (GKP/DEF), 5 (MID), 4 (FWD)
        - Assists: 3 pts
        - Clean sheets: 4 (GKP/DEF), 1 (MID)
        - Cards: -1 (yellow), -3 (red)
        """
        # Expected minutes per gameweek
        minutes_per_gw = players_df["expected_minutes"]

        # Scale xG/xA rates to expected minutes
        xG_per_gw = players_df["xG90"] * (minutes_per_gw / 90.0)
        xA_per_gw = players_df["xA90"] * (minutes_per_gw / 90.0)

        # Apply fixture difficulty multiplier
        fixture_multiplier = players_df["fixture_difficulty"]
        xG_per_gw = xG_per_gw * fixture_multiplier
        xA_per_gw = xA_per_gw * fixture_multiplier

        # === Component-based XP calculation ===

        # 1. Appearance points (FPL: 60+ minutes = 2 pts, under 60 = 1 pt)
        prob_60_min = (minutes_per_gw >= 60).astype(float)
        xP_appearance = (
            prob_60_min * 2.0
            + (1 - prob_60_min) * (minutes_per_gw > 0).astype(float) * 1.0
        )

        # 2. Goal points (FPL: GKP/DEF=6, MID=5, FWD=4)
        position_goal_pts = {
            "GKP": 6,
            "DEF": 6,
            "MID": 5,
            "FWD": 4,
        }
        goal_points = players_df["position"].map(position_goal_pts)
        xP_goals = xG_per_gw * goal_points

        # 3. Assist points (FPL: 3 pts for all positions)
        xP_assists = xA_per_gw * 3

        # 4. Clean sheet probability (defenders/keepers only)
        # Estimate based on team defensive strength
        team_defensive_strength = players_df.get("team_strength", 1.0)
        base_cs_prob = 0.30  # Average clean sheet probability
        cs_prob = (base_cs_prob * team_defensive_strength).clip(0.1, 0.6)

        xP_clean_sheet = 0.0
        if "position" in players_df.columns:
            # FPL: GKP/DEF = 4 pts, MID = 1 pt (only for 60+ mins)
            xP_clean_sheet = (
                (
                    (players_df["position"] == "GKP").astype(float)
                    * cs_prob
                    * 4
                    * (minutes_per_gw >= 60).astype(float)
                )
                + (
                    (players_df["position"] == "DEF").astype(float)
                    * cs_prob
                    * 4
                    * (minutes_per_gw >= 60).astype(float)
                )
                + (
                    (players_df["position"] == "MID").astype(float)
                    * cs_prob
                    * 1
                    * (minutes_per_gw >= 60).astype(float)
                )
            )

        # 5. Bonus points (heuristic: top performers get bonus)
        # Estimate 0.5-1.5 bonus per game for high xG+xA players
        xP_bonus = ((xG_per_gw + xA_per_gw) * 0.3).clip(0, 2)

        # 6. Expected yellow/red cards (FPL: yellow=-1, red=-3)
        # Heuristic: ~0.1 yellow cards per game, 0.01 red cards per game
        xP_cards = -0.1 * 1 - 0.01 * 3

        # === Total XP for 1 gameweek ===
        players_df["xP_1gw"] = (
            xP_appearance + xP_goals + xP_assists + xP_clean_sheet + xP_bonus + xP_cards
        )

        # === Scale to multiple gameweeks ===
        players_df["xP"] = players_df["xP_1gw"] * gameweeks_ahead

        return players_df

    def enrich_players_with_season_stats(
        self, players_with_xp: pd.DataFrame
    ) -> pd.DataFrame:
        """Enrich expected points data with additional season statistics using repository pattern.

        Args:
            players_with_xp: DataFrame with XP calculations

        Returns:
            DataFrame with additional season stats merged in

        Raises:
            ValueError: If enrichment fails
        """
        try:
            # Use repository pattern for data access
            from fpl_team_picker.adapters.database_repositories import (
                DatabasePlayerRepository,
            )

            player_repo = DatabasePlayerRepository()
            enriched_result = player_repo.get_enriched_players_dataframe()

            if enriched_result.is_failure:
                print(
                    f"⚠️ Could not load enriched data: {enriched_result.error.message}"
                )
                print("⚠️ Falling back to basic player data")
                return players_with_xp

            enriched_players = enriched_result.value
            print(
                f"✅ Loaded enriched data for {len(enriched_players)} players with {len(enriched_players.columns)} additional attributes"
            )

            # Merge with existing XP data, handling overlapping columns
            enriched_xp = players_with_xp.merge(
                enriched_players,
                on="player_id",
                how="left",
                suffixes=(
                    "",
                    "_season",
                ),  # Keep original names, add _season to enriched data when conflict
            )

            # Validate merge was successful
            if len(enriched_xp) != len(players_with_xp):
                raise ValueError(
                    "Player enrichment resulted in unexpected row count change"
                )

            # Fill missing values with 0 for numeric columns (except news)
            for col in enriched_players.columns:
                if col != "player_id" and col in enriched_xp.columns:
                    if col == "news":
                        enriched_xp[col] = enriched_xp[col].fillna("")
                    else:
                        enriched_xp[col] = enriched_xp[col].fillna(0)

            print(
                f"✅ Successfully enriched {len(enriched_xp)} players with additional season statistics"
            )
            return enriched_xp

        except Exception as e:
            # If enrichment fails, return original data with warning
            print(f"⚠️ Warning: Could not enrich player data - {str(e)}")
            return players_with_xp
