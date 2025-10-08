"""Team analytics service for team strength calculations.

Provides dynamic team strength ratings that evolve throughout the season.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional

from fpl_team_picker.config import config

warnings.filterwarnings("ignore")


class TeamAnalyticsService:
    """Service for team analytics including dynamic strength calculations.

    Calculates team strength ratings that evolve throughout the season:
    - GW1-7: Weighted combination of previous season baseline + current season
    - GW8+: Pure current season performance

    This provides stable early season ratings while becoming fully responsive
    to current season performance once sufficient data is available.
    """

    # Previous season (2024-25) Premier League final standings - used as baseline
    HISTORICAL_POSITIONS = {
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
        # Handle common team name variations
        "Newcastle": 5,
        "Brighton": 8,
        "West Ham": 14,
        "Man Utd": 15,
        "Manchester Utd": 15,
        "Wolves": 16,
        "Spurs": 17,
        "Tottenham": 17,
        "Leicester": 18,
        "Ipswich": 19,
        "Man City": 3,
        "Nott'm Forest": 7,
    }

    def __init__(self, debug: bool = None):
        """Initialize team analytics service.

        Args:
            debug: Enable debug logging (defaults to config)
        """
        self.debug = debug if debug is not None else config.team_strength.debug
        self._historical_cache = {}
        self._current_season_cache = {}

        # Transition parameters from config
        self.HISTORICAL_TRANSITION_GW = config.team_strength.historical_transition_gw
        self.ROLLING_WINDOW_SIZE = config.team_strength.rolling_window_size

        if self.debug:
            print(
                f"🏟️ TeamAnalyticsService initialized - Transition at GW{self.HISTORICAL_TRANSITION_GW}"
            )

    def get_team_strength(
        self,
        target_gameweek: int,
        teams_data: pd.DataFrame,
        current_season_data: Optional[List[pd.DataFrame]] = None,
    ) -> Dict[str, float]:
        """Get team strength ratings for target gameweek.

        Args:
            target_gameweek: Target gameweek for strength calculation
            teams_data: Team reference data for name mapping
            current_season_data: List of gameweek DataFrames for current season

        Returns:
            Dict mapping team name to strength rating [0.7, 1.3]
        """
        if target_gameweek >= self.HISTORICAL_TRANSITION_GW:
            return self._get_current_season_strength(
                target_gameweek, teams_data, current_season_data
            )
        else:
            return self._get_weighted_strength(
                target_gameweek, teams_data, current_season_data
            )

    def load_historical_gameweek_data(
        self, start_gw: int = 1, end_gw: Optional[int] = None
    ) -> List[pd.DataFrame]:
        """Load current season gameweek data for team strength calculation.

        Args:
            start_gw: Starting gameweek to load
            end_gw: Ending gameweek to load (None = latest available)

        Returns:
            List of gameweek DataFrames
        """
        from client import FPLDataClient

        client = FPLDataClient()

        gameweek_data = []
        max_gw = end_gw if end_gw else 38

        for gw in range(start_gw, max_gw + 1):
            try:
                gw_data = client.get_gameweek_live_data(gw)
                if not gw_data.empty:
                    gameweek_data.append(gw_data)
            except Exception:
                # No data available for this gameweek
                continue

        return gameweek_data

    # ========== Private Internal Methods ==========

    def _get_current_season_strength(
        self,
        target_gameweek: int,
        teams_data: pd.DataFrame,
        current_season_data: Optional[List[pd.DataFrame]],
    ) -> Dict[str, float]:
        """Calculate team strength using current season multi-factor approach (GW8+)."""
        if self.debug:
            print(
                f"🔄 Calculating current season multi-factor strength for GW{target_gameweek}"
            )

        try:
            strength_ratings = self._calculate_multi_factor_strength(
                teams_data, target_gameweek=target_gameweek
            )

            if self.debug:
                print(
                    f"✅ Current season multi-factor strength calculated for GW{target_gameweek}"
                )

            return strength_ratings

        except Exception as e:
            if self.debug:
                print(
                    f"⚠️ Error in current season calculation, falling back to historical: {e}"
                )
            return self._get_historical_baseline(teams_data)

    def _get_weighted_strength(
        self,
        target_gameweek: int,
        teams_data: pd.DataFrame,
        current_season_data: Optional[List[pd.DataFrame]],
    ) -> Dict[str, float]:
        """Calculate early season team strength using multi-factor approach."""
        if self.debug:
            print(
                f"🔄 Calculating early season multi-factor strength for GW{target_gameweek}"
            )

        try:
            strength_ratings = self._calculate_multi_factor_strength(
                teams_data, target_gameweek=target_gameweek
            )

            if self.debug:
                print(
                    f"✅ Early season multi-factor strength calculated for GW{target_gameweek}"
                )

            return strength_ratings

        except Exception as e:
            if self.debug:
                print(
                    f"⚠️ Error in early season calculation, falling back to historical: {e}"
                )
            return self._get_historical_baseline(teams_data)

    def _get_historical_baseline(self, teams_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate team strength using multi-factor approach."""
        if "historical_baseline" in self._historical_cache:
            return self._historical_cache["historical_baseline"]

        try:
            strength_ratings = self._calculate_multi_factor_strength(
                teams_data, target_gameweek=1
            )

            self._historical_cache["historical_baseline"] = strength_ratings

            if self.debug:
                print(
                    f"✅ Multi-factor team strength calculated for {len(strength_ratings)} teams"
                )
                strongest = max(strength_ratings.items(), key=lambda x: x[1])
                weakest = min(strength_ratings.items(), key=lambda x: x[1])
                print(f"🏆 Strongest: {strongest[0]} ({strongest[1]})")
                print(f"📉 Weakest: {weakest[0]} ({weakest[1]})")

            return strength_ratings

        except Exception as e:
            if self.debug:
                print(f"⚠️ Error in multi-factor calculation: {e}")

            return self._get_static_fallback_ratings()

    def _calculate_multi_factor_strength(
        self, teams_data: pd.DataFrame, target_gameweek: Optional[int] = None
    ) -> Dict[str, float]:
        """Calculate team strength using comprehensive multi-factor approach.

        Factors:
        1) League Position (25% weight) - Current/historical league standing
        2) Player Quality (35% weight) - Squad value and total points
        3) Reputation (20% weight) - Historical success over 3-5 seasons
        4) Recent Form (20% weight) - Last 6 gameweeks performance
        """
        if self.debug:
            print(
                f"🔍 Calculating multi-factor team strength for GW{target_gameweek or 'current'}"
            )

        try:
            from client import FPLDataClient

            client = FPLDataClient()

            players = client.get_current_players()

            if target_gameweek is None:
                target_gameweek = 1

            strength_components = {}
            all_teams = self._get_all_team_names(teams_data, players)

            for team_name in all_teams:
                strength_components[team_name] = {
                    "position": 1.0,
                    "quality": 1.0,
                    "reputation": 1.0,
                    "form": 1.0,
                }

            # Calculate each factor
            position_strengths = self._calculate_position_strength(
                target_gameweek, all_teams
            )
            quality_strengths = self._calculate_player_quality_strength(
                players, all_teams
            )
            reputation_strengths = self._calculate_reputation_strength(all_teams)
            form_strengths = self._calculate_recent_form_strength(
                target_gameweek, all_teams
            )

            # Get seasonal weights based on gameweek
            weights = self._get_seasonal_weights(target_gameweek)

            # Combine factors with seasonal weighting
            final_strengths = {}
            for team_name in all_teams:
                combined_strength = (
                    position_strengths.get(team_name, 1.0) * weights["position"]
                    + quality_strengths.get(team_name, 1.0) * weights["quality"]
                    + reputation_strengths.get(team_name, 1.0) * weights["reputation"]
                    + form_strengths.get(team_name, 1.0) * weights["form"]
                )

                final_strengths[team_name] = round(
                    np.clip(combined_strength, 0.7, 1.3), 3
                )

            if self.debug:
                print(f"✅ Multi-factor calculation complete with weights: {weights}")
                sample_team = next(iter(final_strengths.keys()))
                print(
                    f"📊 Example ({sample_team}): pos={position_strengths.get(sample_team, 1.0):.3f}, "
                    f"quality={quality_strengths.get(sample_team, 1.0):.3f}, "
                    f"rep={reputation_strengths.get(sample_team, 1.0):.3f}, "
                    f"form={form_strengths.get(sample_team, 1.0):.3f} → {final_strengths[sample_team]:.3f}"
                )

            return final_strengths

        except Exception as e:
            if self.debug:
                print(f"⚠️ Error in multi-factor calculation: {e}")
            return self._get_static_fallback_ratings()

    def _get_all_team_names(
        self, teams_data: pd.DataFrame, players: pd.DataFrame
    ) -> List[str]:
        """Get comprehensive list of all team names from available data sources."""
        team_names = set()

        if "name" in teams_data.columns:
            team_names.update(teams_data["name"].dropna().tolist())

        if "team_name" in players.columns:
            team_names.update(players["team_name"].dropna().unique().tolist())

        if not team_names:
            team_names = {
                "Arsenal",
                "Aston Villa",
                "Bournemouth",
                "Brentford",
                "Brighton",
                "Burnley",
                "Chelsea",
                "Crystal Palace",
                "Everton",
                "Fulham",
                "Liverpool",
                "Leeds",
                "Leicester",
                "Manchester City",
                "Manchester United",
                "Newcastle",
                "Nottingham Forest",
                "Southampton",
                "Tottenham",
                "West Ham",
                "Wolves",
            }

        return sorted(list(team_names))

    def _calculate_position_strength(
        self, target_gameweek: int, all_teams: List[str]
    ) -> Dict[str, float]:
        """Factor 1: League Position (25% weight)."""
        try:
            position_strengths = {}
            for team_name in all_teams:
                position = self.HISTORICAL_POSITIONS.get(team_name, 15)
                strength = 1.3 - (position - 1) * (1.3 - 0.7) / 19
                position_strengths[team_name] = round(np.clip(strength, 0.7, 1.3), 3)

            if self.debug:
                print(
                    f"📊 Position strengths calculated for {len(position_strengths)} teams"
                )

            return position_strengths

        except Exception as e:
            if self.debug:
                print(f"⚠️ Error calculating position strength: {e}")
            return {team: 1.0 for team in all_teams}

    def _calculate_player_quality_strength(
        self, players: pd.DataFrame, all_teams: List[str]
    ) -> Dict[str, float]:
        """Factor 2: Player Quality (35% weight) - Squad value and total points."""
        try:
            quality_strengths = {}
            team_metrics = {}

            for team_name in all_teams:
                if "team_name" in players.columns:
                    team_players = players[players["team_name"] == team_name]
                else:
                    team_players = pd.DataFrame()

                if len(team_players) > 0:
                    squad_value = team_players["now_cost"].sum() / 10.0
                    squad_points = team_players["total_points"].sum()

                    team_metrics[team_name] = {
                        "value": squad_value,
                        "points": squad_points,
                    }
                else:
                    team_metrics[team_name] = {"value": 50.0, "points": 100}

            if team_metrics:
                league_avg_value = np.mean([m["value"] for m in team_metrics.values()])
                league_avg_points = np.mean(
                    [m["points"] for m in team_metrics.values()]
                )
            else:
                league_avg_value, league_avg_points = 50.0, 100

            for team_name in all_teams:
                metrics = team_metrics[team_name]

                value_ratio = (
                    metrics["value"] / league_avg_value if league_avg_value > 0 else 1.0
                )
                points_ratio = (
                    metrics["points"] / league_avg_points
                    if league_avg_points > 0
                    else 1.0
                )

                combined_quality = (value_ratio + points_ratio) / 2

                strength = 0.7 + (combined_quality - 0.5) * (1.3 - 0.7) / (1.5 - 0.5)
                quality_strengths[team_name] = round(np.clip(strength, 0.7, 1.3), 3)

            if self.debug:
                print(
                    f"💎 Player quality strengths calculated for {len(quality_strengths)} teams"
                )
                print(
                    f"💰 League avg value: £{league_avg_value:.1f}m, avg points: {league_avg_points:.0f}"
                )

            return quality_strengths

        except Exception as e:
            if self.debug:
                print(f"⚠️ Error calculating player quality: {e}")
            return {team: 1.0 for team in all_teams}

    def _calculate_reputation_strength(self, all_teams: List[str]) -> Dict[str, float]:
        """Factor 3: Reputation (20% weight) - Historical success over 3-5 seasons."""
        reputation_ratings = {
            # Big 6
            "Liverpool": 1.3,
            "Manchester City": 1.25,
            "Arsenal": 1.25,
            "Chelsea": 1.2,
            "Manchester United": 1.15,
            "Tottenham": 1.15,
            # Strong teams
            "Newcastle": 1.1,
            "Aston Villa": 1.05,
            # Solid mid-table
            "Brighton": 1.0,
            "West Ham": 1.0,
            "Crystal Palace": 0.95,
            "Fulham": 0.95,
            "Brentford": 0.9,
            "Bournemouth": 0.9,
            # Lower mid-table
            "Everton": 0.85,
            "Wolves": 0.85,
            "Nottingham Forest": 0.8,
            # Historically weaker/promoted teams
            "Leicester": 0.85,
            "Southampton": 0.8,
            "Leeds": 0.8,
            "Burnley": 0.75,
            "Sheffield United": 0.7,
            "Luton": 0.7,
            "Ipswich": 0.7,
        }

        reputation_strengths = {}
        for team_name in all_teams:
            lookup_name = team_name
            if team_name == "Man City":
                lookup_name = "Manchester City"
            elif team_name == "Man Utd":
                lookup_name = "Manchester United"
            elif team_name == "Nott'm Forest":
                lookup_name = "Nottingham Forest"
            elif team_name == "Spurs":
                lookup_name = "Tottenham"

            reputation_strengths[team_name] = reputation_ratings.get(lookup_name, 0.9)

        if self.debug:
            print(
                f"🏆 Reputation strengths assigned for {len(reputation_strengths)} teams"
            )

        return reputation_strengths

    def _calculate_recent_form_strength(
        self, target_gameweek: int, all_teams: List[str]
    ) -> Dict[str, float]:
        """Factor 4: Recent Form (20% weight) - Last 6 gameweeks performance."""
        try:
            from client import FPLDataClient

            client = FPLDataClient()

            form_strengths = {}

            if target_gameweek <= 6:
                if self.debug:
                    print(
                        f"⚡ Using neutral form for early season (GW{target_gameweek})"
                    )
                return {team: 1.0 for team in all_teams}

            recent_gws = []
            for gw in range(max(1, target_gameweek - 6), target_gameweek):
                try:
                    gw_data = client.get_gameweek_live_data(gw)
                    if not gw_data.empty:
                        recent_gws.append(gw_data)
                except Exception:
                    continue

            if not recent_gws:
                if self.debug:
                    print("⚡ No recent form data available, using neutral form")
                return {team: 1.0 for team in all_teams}

            for team_name in all_teams:
                form_strengths[team_name] = 1.0

            if self.debug:
                print(
                    f"⚡ Recent form calculated for {len(form_strengths)} teams over {len(recent_gws)} gameweeks"
                )

            return form_strengths

        except Exception as e:
            if self.debug:
                print(f"⚠️ Error calculating recent form: {e}")
            return {team: 1.0 for team in all_teams}

    def _get_seasonal_weights(self, target_gameweek: int) -> Dict[str, float]:
        """Get seasonal weighting factors based on gameweek."""
        if target_gameweek <= 7:
            # Early season: Higher weight on reputation + historical position
            return {
                "position": 0.20,
                "quality": 0.35,
                "reputation": 0.30,
                "form": 0.15,
            }
        elif target_gameweek <= 25:
            # Mid season: Balanced weights
            return {"position": 0.25, "quality": 0.35, "reputation": 0.20, "form": 0.20}
        else:
            # Late season: Higher weight on current position + recent form
            return {
                "position": 0.30,
                "quality": 0.30,
                "reputation": 0.15,
                "form": 0.25,
            }

    def _get_static_fallback_ratings(self) -> Dict[str, float]:
        """Fallback to static previous season final table ratings if dynamic calculation fails."""
        strength_ratings = {}
        for team, position in self.HISTORICAL_POSITIONS.items():
            if position <= 20:
                strength = 1.3 - (position - 1) * (1.3 - 0.7) / 19
            else:
                strength = 0.7 - (position - 20) * 0.05
            strength_ratings[team] = round(strength, 3)

        if self.debug:
            print("⚠️ Using static fallback ratings")

        return strength_ratings
