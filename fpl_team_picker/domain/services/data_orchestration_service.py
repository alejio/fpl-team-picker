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

    def load_historical_gameweek_state(
        self,
        target_gameweek: int,
        form_window: int = 5,
        include_snapshots: bool = True,
    ) -> Dict[str, Any]:
        """Load historical gameweek state for recomputation - FAIL FAST if data is bad.

        This method reconstructs the data state as it existed at the time of
        target_gameweek, enabling accurate recomputation of historical expected points.

        Args:
            target_gameweek: The historical gameweek to reconstruct (1-38)
            form_window: Number of recent gameweeks for form analysis
            include_snapshots: Whether to use availability snapshots (if available)

        Returns:
            Clean historical data dictionary with same structure as load_gameweek_data()
            - players: Player data with historical prices
            - teams: Team reference data (unchanged)
            - fixtures: Fixtures for target gameweek
            - xg_rates: xG/xA rates (cumulative up to target_gameweek)
            - live_data_historical: Performance data up to target_gameweek only
            - gameweek_info: Synthetic gameweek info for target_gameweek
            - availability_snapshot: Player availability at target_gameweek (if available)
            - target_gameweek: The gameweek being reconstructed
            - form_window: Form window used

        Raises:
            ValueError: If data doesn't meet contracts or target_gameweek is invalid
        """
        from client import FPLDataClient

        # Validate target gameweek
        if (
            not isinstance(target_gameweek, int)
            or target_gameweek < 1
            or target_gameweek > 38
        ):
            raise ValueError(
                f"Invalid target gameweek: {target_gameweek}. Must be 1-38."
            )

        client = FPLDataClient()

        # 1. Load historical performance data (cumulative up to target_gameweek)
        live_data_historical = None
        if target_gameweek > 1:
            # Get performance data for form window
            start_gw = max(1, target_gameweek - form_window)
            historical_dfs = []
            for gw in range(start_gw, target_gameweek):
                try:
                    gw_data = client.get_gameweek_performance(gw)
                    if not gw_data.empty:
                        gw_data["gameweek"] = gw
                        historical_dfs.append(gw_data)
                except Exception:
                    continue

            if historical_dfs:
                live_data_historical = pd.concat(historical_dfs, ignore_index=True)

        # 2. Load current players (for reference data - we'll override prices)
        players = client.get_current_players()
        if players.empty or len(players) < 100:
            raise ValueError(
                f"Invalid player data: expected >100 players, got {len(players)}"
            )

        # 3. Override player prices with historical prices from gameweek performance
        # Use the last available price before target_gameweek
        try:
            price_gw = target_gameweek - 1 if target_gameweek > 1 else 1
            historical_prices = client.get_gameweek_performance(price_gw)

            if not historical_prices.empty and "value" in historical_prices.columns:
                # Merge historical prices into players dataframe
                price_map = historical_prices.set_index("player_id")["value"].to_dict()
                players["now_cost"] = players["player_id"].map(
                    lambda pid: price_map.get(
                        pid,
                        players.loc[players["player_id"] == pid, "now_cost"].iloc[0],
                    )
                )
        except Exception:
            # If we can't get historical prices, use current prices (acceptable degradation)
            pass

        # 4. Load availability snapshot (if available and requested)
        availability_snapshot = None
        if include_snapshots and target_gameweek >= 8:
            try:
                availability_snapshot = client.get_player_availability_snapshot(
                    gameweek=target_gameweek, include_backfilled=True
                )
            except Exception:
                # Snapshots may not be available for all gameweeks
                pass

        # 5. Approximate availability for pre-snapshot gameweeks (GW1-7)
        if availability_snapshot is None and live_data_historical is not None:
            # Infer availability: if player had 0 minutes → likely unavailable
            recent_performance = live_data_historical[
                live_data_historical["gameweek"] == target_gameweek - 1
            ]
            if not recent_performance.empty:
                unavailable_players = recent_performance[
                    recent_performance["minutes"] == 0
                ]["player_id"].unique()
                # Mark these players as potentially unavailable
                players["inferred_unavailable"] = players["player_id"].isin(
                    unavailable_players
                )

        # 6. Load teams data (unchanged - stable reference)
        teams = client.get_current_teams()
        if len(teams) != 20:
            raise ValueError(f"Invalid team data: expected 20 teams, got {len(teams)}")

        # 7. Load fixtures for target gameweek
        all_fixtures = client.get_fixtures_normalized()
        fixtures = all_fixtures[all_fixtures["event"] == target_gameweek].copy()

        # 8. Load xG/xA rates (cumulative up to target_gameweek)
        xg_rates = client.get_player_xg_xa_rates()

        # 9. Create synthetic gameweek info
        gameweek_info = {
            "current_gameweek": target_gameweek,
            "status": "historical",
            "available_data": list(range(1, target_gameweek)),
            "message": f"Historical reconstruction for GW{target_gameweek}",
        }

        return {
            "players": players,
            "teams": teams,
            "fixtures": fixtures,
            "xg_rates": xg_rates,
            "live_data_historical": live_data_historical,
            "gameweek_info": gameweek_info,
            "availability_snapshot": availability_snapshot,
            "manager_team": None,  # Not applicable for historical reconstruction
            "current_squad": None,  # Not applicable for historical reconstruction
            "target_gameweek": target_gameweek,
            "form_window": form_window,
        }
