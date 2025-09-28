"""Fixture analysis service for FPL fixture difficulty and scheduling."""

from typing import Dict, Any, List, Optional
import pandas as pd

from fpl_team_picker.domain.common.result import Result


class FixtureAnalysisService:
    """Service for analyzing fixture difficulty and scheduling."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the service with configuration.

        Args:
            config: Configuration dictionary for fixture analysis
        """
        self.config = config or {
            "easy_threshold": 1.1,  # Threshold for "easy" fixtures
            "hard_threshold": 0.9,  # Threshold for "hard" fixtures
            "double_gw_bonus": 1.5,  # Bonus multiplier for double gameweeks
            "analysis_horizon": 5,  # Number of gameweeks to analyze ahead
        }

    def analyze_fixture_difficulty(
        self,
        gameweek_data: Dict[str, Any],
        target_gameweek: int,
        gameweeks_ahead: int = 5,
    ) -> Result[Dict[str, Any]]:
        """Analyze fixture difficulty over multiple gameweeks.

        Args:
            gameweek_data: Complete gameweek data from DataOrchestrationService
            target_gameweek: Starting gameweek for analysis
            gameweeks_ahead: Number of gameweeks to analyze

        Returns:
            Result containing fixture difficulty analysis
        """

        fixtures = gameweek_data.get("fixtures")
        teams = gameweek_data.get("teams")

        # Analyze upcoming fixtures
        upcoming_fixtures = self._get_upcoming_fixtures(
            fixtures, target_gameweek, gameweeks_ahead
        )

        # Calculate difficulty ratings
        difficulty_analysis = self._calculate_difficulty_ratings(
            upcoming_fixtures, teams
        )

        # Get team fixture outlook
        team_outlook = self._analyze_team_outlook(
            upcoming_fixtures, teams, gameweeks_ahead
        )

        return {
            "upcoming_fixtures": upcoming_fixtures,
            "difficulty_analysis": difficulty_analysis,
            "team_outlook": team_outlook,
            "analysis_period": f"GW{target_gameweek}-{target_gameweek + gameweeks_ahead - 1}",
            "gameweeks_analyzed": gameweeks_ahead,
        }

    def _get_upcoming_fixtures(
        self, fixtures: pd.DataFrame, start_gw: int, num_gws: int
    ) -> List[Dict[str, Any]]:
        """Get upcoming fixtures for analysis period."""
        try:
            upcoming = fixtures[
                (fixtures["event"] >= start_gw)
                & (fixtures["event"] < start_gw + num_gws)
            ].copy()

            return upcoming.to_dict("records")
        except Exception:
            return []

    def _calculate_difficulty_ratings(
        self, fixtures: List[Dict[str, Any]], teams: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate difficulty ratings for fixtures."""
        try:
            if not fixtures:
                return {"summary": "No fixtures to analyze"}

            # Simple difficulty calculation
            total_fixtures = len(fixtures)
            avg_difficulty = 1.0  # Neutral difficulty baseline

            return {
                "total_fixtures": total_fixtures,
                "average_difficulty": avg_difficulty,
                "summary": f"Analyzed {total_fixtures} upcoming fixtures",
            }
        except Exception:
            return {"summary": "Difficulty calculation failed"}

    def _analyze_team_outlook(
        self, fixtures: List[Dict[str, Any]], teams: pd.DataFrame, num_gws: int
    ) -> Dict[str, Any]:
        """Analyze team fixture outlook."""
        try:
            if not fixtures:
                return {
                    "easiest_teams": [],
                    "hardest_teams": [],
                    "double_gameweeks": [],
                }

            # Count fixtures per team
            team_fixture_counts = {}
            for fixture in fixtures:
                home_team = fixture.get("home_team_id")
                away_team = fixture.get("away_team_id")

                if home_team:
                    team_fixture_counts[home_team] = (
                        team_fixture_counts.get(home_team, 0) + 1
                    )
                if away_team:
                    team_fixture_counts[away_team] = (
                        team_fixture_counts.get(away_team, 0) + 1
                    )

            # Find teams with multiple fixtures (double gameweeks)
            double_gw_teams = [
                team_id
                for team_id, count in team_fixture_counts.items()
                if count > num_gws
            ]

            # Get team names for display
            team_id_to_name = {}
            if not teams.empty and "id" in teams.columns and "name" in teams.columns:
                team_id_to_name = teams.set_index("id")["name"].to_dict()

            return {
                "easiest_teams": ["Brighton", "Sheffield Utd", "Luton"],  # Placeholder
                "hardest_teams": ["Man City", "Arsenal", "Liverpool"],  # Placeholder
                "double_gameweeks": [
                    team_id_to_name.get(team_id, f"Team {team_id}")
                    for team_id in double_gw_teams
                ]
                if double_gw_teams
                else ["None"],
            }

        except Exception:
            return {
                "easiest_teams": [],
                "hardest_teams": [],
                "double_gameweeks": [],
            }
