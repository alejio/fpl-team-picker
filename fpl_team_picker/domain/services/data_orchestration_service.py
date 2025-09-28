"""Data orchestration service for gameweek management."""

from typing import Dict, List, Any
import pandas as pd

from fpl_team_picker.domain.repositories.player_repository import PlayerRepository
from fpl_team_picker.domain.repositories.team_repository import TeamRepository
from fpl_team_picker.domain.repositories.fixture_repository import FixtureRepository


class DataOrchestrationService:
    """Service for orchestrating data loading and validation for gameweek management."""

    def __init__(
        self,
        player_repository: PlayerRepository = None,
        team_repository: TeamRepository = None,
        fixture_repository: FixtureRepository = None,
    ):
        self.player_repository = player_repository
        self.team_repository = team_repository
        self.fixture_repository = fixture_repository

    def load_gameweek_data(
        self, target_gameweek: int, form_window: int = 5
    ) -> Dict[str, Any]:
        """Load all required data for gameweek analysis - FAIL FAST if data is bad.

        Args:
            target_gameweek: The gameweek to analyze
            form_window: Number of recent gameweeks for form analysis

        Returns:
            Clean data dictionary - guaranteed to be valid

        Raises:
            ValueError: If data doesn't meet contracts
        """
        # Use existing data loading - let it fail fast
        from fpl_team_picker.core.data_loader import (
            fetch_fpl_data,
            get_current_gameweek_info,
            fetch_manager_team,
            process_current_squad,
        )

        # Get current gameweek info - fail fast if not available
        gw_info = get_current_gameweek_info()
        if not gw_info:
            raise ValueError(
                "Could not determine current gameweek - check FPL API connection"
            )

        # Fetch comprehensive FPL data - let core loader handle failures
        fpl_data = fetch_fpl_data(
            target_gameweek=target_gameweek, form_window=form_window
        )
        (
            players,
            teams,
            xg_rates,
            fixtures,
            actual_target_gameweek,
            live_data_historical,
        ) = fpl_data

        # Validate core data contracts
        if len(players) < 100:
            raise ValueError(
                f"Invalid player data: expected >100 players, got {len(players)}"
            )

        if len(teams) != 20:
            raise ValueError(f"Invalid team data: expected 20 teams, got {len(teams)}")

        # Get manager team data - fail if we can't get current squad
        manager_team = None
        current_squad = None
        if gw_info.get("current_gameweek", 0) > 1:
            manager_team = fetch_manager_team(
                previous_gameweek=gw_info["current_gameweek"] - 1
            )
            if manager_team is not None:
                current_squad = process_current_squad(manager_team, players, teams)
                if len(current_squad) != 15:
                    raise ValueError(
                        f"Invalid squad: expected 15 players, got {len(current_squad)}"
                    )

        return {
            "players": players,
            "teams": teams,
            "fixtures": fixtures,
            "xg_rates": xg_rates,
            "live_data_historical": live_data_historical,
            "gameweek_info": gw_info,
            "manager_team": manager_team,
            "current_squad": current_squad,
            "target_gameweek": target_gameweek,
            "form_window": form_window,
        }

    def get_current_gameweek_info(self) -> Dict[str, Any]:
        """Get current gameweek information.

        Returns:
            Current gameweek info - guaranteed clean
        """
        from fpl_team_picker.core.data_loader import get_current_gameweek_info

        gw_info = get_current_gameweek_info()
        if not gw_info:
            raise ValueError(
                "Could not determine current gameweek - check FPL API connection"
            )

        return gw_info

    def validate_gameweek_data(self, data: Dict[str, Any]) -> bool:
        """Validate that the loaded gameweek data is complete and valid.

        Args:
            data: The gameweek data dictionary - guaranteed clean

        Returns:
            True if validation passes
        """
        required_keys = [
            "players",
            "teams",
            "fixtures",
            "xg_rates",
            "gameweek_info",
            "target_gameweek",
        ]
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            raise ValueError(f"Missing required data keys: {missing_keys}")

        # Validate data completeness
        players = data["players"]
        teams = data["teams"]
        fixtures = data["fixtures"]

        if players.empty:
            raise ValueError("Players data is empty")

        if teams.empty:
            raise ValueError("Teams data is empty")

        if fixtures.empty:
            raise ValueError("Fixtures data is empty")

        # Validate target gameweek
        target_gw = data["target_gameweek"]
        if not isinstance(target_gw, int) or target_gw < 1 or target_gw > 38:
            raise ValueError(f"Invalid target gameweek: {target_gw}")

        return True

    def _convert_players_to_dataframe(self, players: List[Any]) -> pd.DataFrame:
        """Convert player domain models to DataFrame for compatibility."""
        data = []
        for player in players:
            data.append(
                {
                    "player_id": player.player_id,
                    "web_name": player.web_name,
                    "first_name": player.first_name,
                    "last_name": player.last_name,
                    "team_id": player.team_id,
                    "position": player.position.value
                    if hasattr(player.position, "value")
                    else player.position,
                    "price": player.price,
                    "selected_by_percent": player.selected_by_percent,
                    "availability_status": player.availability_status.value
                    if hasattr(player.availability_status, "value")
                    else (
                        player.availability_status
                        if player.availability_status
                        else "a"
                    ),
                    "as_of_utc": player.as_of_utc,
                }
            )
        return pd.DataFrame(data)

    def _convert_teams_to_dataframe(self, teams: List[Any]) -> pd.DataFrame:
        """Convert team domain models to DataFrame for compatibility."""
        data = []
        for team in teams:
            data.append(
                {
                    "team_id": team.team_id,
                    "name": team.name,
                    "short_name": team.short_name,
                    "strength_overall_home": team.strength_overall_home,
                    "strength_overall_away": team.strength_overall_away,
                    "strength_attack_home": team.strength_attack_home,
                    "strength_attack_away": team.strength_attack_away,
                    "strength_defence_home": team.strength_defence_home,
                    "strength_defence_away": team.strength_defence_away,
                }
            )
        return pd.DataFrame(data)

    def _convert_fixtures_to_dataframe(self, fixtures: List[Any]) -> pd.DataFrame:
        """Convert fixture domain models to DataFrame for compatibility."""
        data = []
        for fixture in fixtures:
            data.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "event": fixture.event,
                    "home_team_id": fixture.home_team_id,
                    "away_team_id": fixture.away_team_id,
                    "kickoff_time": fixture.kickoff_time,
                    "difficulty_home": fixture.difficulty_home,
                    "difficulty_away": fixture.difficulty_away,
                    "finished": fixture.finished,
                }
            )
        return pd.DataFrame(data)
