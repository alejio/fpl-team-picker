"""Data orchestration service for gameweek management."""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from datetime import datetime

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
        # Get current gameweek info - fail fast if not available
        gw_info = self._get_current_gameweek_info_internal()
        if not gw_info:
            raise ValueError(
                "Could not determine current gameweek - check FPL API connection"
            )

        # Fetch comprehensive FPL data - let internal method handle failures
        fpl_data = self._fetch_fpl_data_internal(
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
            manager_team = self._fetch_manager_team_internal(
                previous_gameweek=gw_info["current_gameweek"] - 1
            )
            if manager_team is not None:
                current_squad = self._process_current_squad_internal(
                    manager_team, players, teams
                )
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
        gw_info = self._get_current_gameweek_info_internal()
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
        price_gw = target_gameweek - 1 if target_gameweek > 1 else 1
        historical_prices = client.get_gameweek_performance(price_gw)

        if historical_prices.empty:
            raise ValueError(
                f"No historical price data available for GW{price_gw}. "
                f"Cannot recompute GW{target_gameweek} without accurate historical prices. "
                f"Fix data quality in dataset-builder."
            )

        if "value" not in historical_prices.columns:
            raise ValueError(
                f"Historical price data for GW{price_gw} missing 'value' column. "
                f"Data contract violation - fix dataset-builder schema."
            )

        # Merge historical prices into players dataframe
        price_map = historical_prices.set_index("player_id")["value"].to_dict()

        # Validate all players have historical prices
        missing_prices = set(players["player_id"]) - set(price_map.keys())
        if missing_prices:
            raise ValueError(
                f"Historical prices missing for {len(missing_prices)} players in GW{price_gw}. "
                f"Data quality issue - fix dataset-builder. Missing player IDs: {list(missing_prices)[:10]}"
            )

        players["now_cost"] = players["player_id"].map(price_map)

        # 4. Load availability snapshot (required for accurate recomputation)
        availability_snapshot = None
        if include_snapshots:
            availability_snapshot = client.get_player_availability_snapshot(
                gameweek=target_gameweek, include_backfilled=True
            )

            if availability_snapshot.empty:
                raise ValueError(
                    f"No availability snapshot data for GW{target_gameweek}. "
                    f"Cannot recompute without player availability/injury status. "
                    f"Run snapshot capture in dataset-builder first: "
                    f"'uv run main.py snapshot --gameweek {target_gameweek}'"
                )

            # Validate snapshot has required fields
            required_snapshot_fields = ["player_id", "status", "gameweek"]
            missing_fields = [
                f
                for f in required_snapshot_fields
                if f not in availability_snapshot.columns
            ]
            if missing_fields:
                raise ValueError(
                    f"Availability snapshot for GW{target_gameweek} missing required fields: {missing_fields}. "
                    f"Data contract violation - fix dataset-builder snapshot schema."
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

    # ========== Private Internal Methods (Migrated from core/data_loader.py) ==========

    def _get_current_gameweek_info_internal(self) -> Dict:
        """
        Automatically detect current gameweek based on fixture dates and data availability.

        Returns:
            Dictionary containing:
            - current_gameweek: The gameweek we should be preparing for
            - status: 'upcoming', 'in_progress', 'completed', or 'unknown'
            - available_data: List of gameweeks with live data available
            - message: Human-readable status message
        """
        from client import FPLDataClient

        client = FPLDataClient()

        try:
            # Get fixture data to determine current gameweek
            fixtures = client.get_fixtures_normalized()
            if fixtures.empty:
                return {
                    "current_gameweek": 1,
                    "status": "unknown",
                    "available_data": [],
                    "message": "⚠️ No fixture data available",
                }

            # Convert kickoff times to dates
            fixtures["kickoff_datetime"] = pd.to_datetime(fixtures["kickoff_utc"])
            fixtures["kickoff_date"] = fixtures["kickoff_datetime"].dt.date
            today = datetime.now().date()

            # Find the current/next gameweek based on fixture dates
            upcoming_fixtures = fixtures[fixtures["kickoff_date"] >= today].sort_values(
                "kickoff_datetime"
            )

            if not upcoming_fixtures.empty:
                target_gameweek = int(upcoming_fixtures.iloc[0]["event"])
            else:
                # All fixtures are in the past - get the latest gameweek
                past_fixtures = fixtures[fixtures["kickoff_date"] < today].sort_values(
                    "kickoff_datetime", ascending=False
                )
                if not past_fixtures.empty:
                    target_gameweek = int(past_fixtures.iloc[0]["event"]) + 1
                else:
                    target_gameweek = 1

            # Check what gameweek data is available
            available_gws = []
            for gw in range(
                1, min(target_gameweek + 2, 39)
            ):  # Check up to 2 gameweeks ahead
                try:
                    # Try both live data and performance data methods
                    gw_data = client.get_gameweek_live_data(gw)
                    if gw_data.empty:
                        gw_data = client.get_gameweek_performance(gw)

                    if not gw_data.empty:
                        available_gws.append(gw)
                except Exception:
                    continue

            # Determine status based on fixture timing and data availability
            current_gw_fixtures = fixtures[fixtures["event"] == target_gameweek]
            if not current_gw_fixtures.empty:
                earliest_kickoff = current_gw_fixtures["kickoff_datetime"].min()
                latest_kickoff = current_gw_fixtures["kickoff_datetime"].max()

                now = datetime.now()

                # Check if we're in the middle of the gameweek
                if earliest_kickoff <= now <= latest_kickoff + pd.Timedelta(hours=2):
                    status = "in_progress"
                    message = f"🕐 GW{target_gameweek} is currently in progress"
                elif now > latest_kickoff + pd.Timedelta(hours=2):
                    # Gameweek completed, data should be available
                    if target_gameweek in available_gws:
                        status = "completed"
                        message = f"✅ GW{target_gameweek} completed, data available"
                        # Don't increment - target_gameweek already points to next gameweek from upcoming fixtures
                    else:
                        status = "completed"
                        message = (
                            f"⏳ GW{target_gameweek} completed, waiting for data update"
                        )
                else:
                    # Gameweek is upcoming
                    if target_gameweek == 1:
                        status = "upcoming"
                        message = f"🚀 Preparing for GW{target_gameweek} (Season start)"
                    elif (target_gameweek - 1) in available_gws:
                        status = "upcoming"
                        message = f"📋 Preparing for GW{target_gameweek}"
                    else:
                        status = "upcoming"
                        message = f"⚠️ Preparing for GW{target_gameweek} - previous gameweek data not yet available"
            else:
                status = "unknown"
                message = f"❓ Unable to determine status for GW{target_gameweek}"

            return {
                "current_gameweek": target_gameweek,
                "status": status,
                "available_data": available_gws,
                "message": message,
            }

        except Exception as e:
            return {
                "current_gameweek": 1,
                "status": "unknown",
                "available_data": [],
                "message": f"❌ Error detecting gameweek: {str(e)[:50]}",
            }

    def _fetch_fpl_data_internal(
        self, target_gameweek: int, form_window: int = 5
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, pd.DataFrame
    ]:
        """
        Fetch FPL data from database for specified gameweek with historical form data

        Args:
            target_gameweek: The gameweek to optimize for
            form_window: Number of previous gameweeks to include for form analysis

        Returns:
            Tuple of (players, teams, xg_rates, fixtures, target_gameweek, live_data_historical)
        """
        from client import FPLDataClient

        # Initialize robust client
        client = FPLDataClient()

        print(f"🔄 Loading FPL data from database for gameweek {target_gameweek}...")

        # Get base data from database using enhanced client
        players_base = client.get_current_players()
        teams_df = client.get_current_teams()

        # Get historical live data for form calculation using enhanced methods
        historical_data = []

        for gw in range(max(1, target_gameweek - form_window), target_gameweek):
            try:
                # Try gameweek-specific performance data first (more detailed)
                gw_data = client.get_gameweek_performance(gw)
                if not gw_data.empty:
                    # Standardize column names for compatibility
                    if "gameweek" in gw_data.columns and "event" not in gw_data.columns:
                        gw_data["event"] = gw_data["gameweek"]
                    elif "event" not in gw_data.columns:
                        gw_data["event"] = gw

                    historical_data.append(gw_data)
                    print(
                        f"✅ Loaded detailed form data for GW{gw} ({len(gw_data)} players)"
                    )
                else:
                    # Fallback to legacy live data
                    gw_data = client.get_gameweek_live_data(gw)
                    if not gw_data.empty:
                        # Ensure event column exists
                        if "event" not in gw_data.columns:
                            gw_data["event"] = gw
                        historical_data.append(gw_data)
                        print(f"✅ Loaded legacy form data for GW{gw}")
            except Exception as e:
                print(f"⚠️  No form data for GW{gw}: {str(e)[:50]}")
                continue

        # Combine historical data for form analysis
        if historical_data:
            live_data_historical = pd.concat(historical_data, ignore_index=True)
            print(f"📊 Combined form data from {len(historical_data)} gameweeks")
        else:
            live_data_historical = pd.DataFrame()
            print("⚠️  No historical form data available")

        # Try to get current gameweek live data with enhanced methods
        try:
            # Try detailed gameweek performance first
            current_live_data = client.get_gameweek_performance(target_gameweek)
            if not current_live_data.empty:
                print(
                    f"✅ Found detailed live data for GW{target_gameweek} ({len(current_live_data)} players)"
                )
                # Standardize column names for compatibility
                if (
                    "gameweek" in current_live_data.columns
                    and "event" not in current_live_data.columns
                ):
                    current_live_data["event"] = current_live_data["gameweek"]
                elif "event" not in current_live_data.columns:
                    current_live_data["event"] = target_gameweek
            else:
                # Fallback to legacy live data
                current_live_data = client.get_gameweek_live_data(target_gameweek)
                if not current_live_data.empty:
                    print(f"✅ Found legacy live data for GW{target_gameweek}")
                    # Ensure event column exists
                    if "event" not in current_live_data.columns:
                        current_live_data["event"] = target_gameweek
        except Exception as e:
            print(f"⚠️  No current live data for GW{target_gameweek}: {str(e)[:50]}")
            current_live_data = pd.DataFrame()

        # Combine historical and current live data
        all_live_data = []
        if not live_data_historical.empty:
            all_live_data.append(live_data_historical)
        if not current_live_data.empty:
            all_live_data.append(current_live_data)

        if all_live_data:
            live_data_combined = pd.concat(all_live_data, ignore_index=True)
            print(
                f"📊 Combined live data: {len(live_data_combined)} records across {len(live_data_combined['event'].unique()) if 'event' in live_data_combined.columns else 0} gameweeks"
            )

            # Convert string numeric columns to proper numeric types with validation
            numeric_columns = [
                "expected_goals",
                "expected_assists",
                "ict_index",
                "influence",
                "creativity",
                "threat",
            ]
            for col in numeric_columns:
                if col in live_data_combined.columns:
                    # Check for invalid values before conversion
                    invalid_mask = (
                        pd.to_numeric(live_data_combined[col], errors="coerce").isna()
                        & live_data_combined[col].notna()
                    )
                    if invalid_mask.any():
                        invalid_values = live_data_combined.loc[
                            invalid_mask, col
                        ].unique()
                        print(
                            f"⚠️  Warning: Invalid numeric values in {col}: {invalid_values}"
                        )
                        # Fill invalid values with 0.0 instead of NaN for better data quality
                        live_data_combined[col] = pd.to_numeric(
                            live_data_combined[col], errors="coerce"
                        ).fillna(0.0)
                    else:
                        live_data_combined[col] = pd.to_numeric(live_data_combined[col])

        else:
            live_data_combined = live_data_historical  # Fallback to historical only

        # Merge players with current gameweek stats if available
        if not current_live_data.empty:
            # Standardize column names for compatibility
            if (
                "gameweek" in current_live_data.columns
                and "event" not in current_live_data.columns
            ):
                current_live_data["event"] = current_live_data["gameweek"]

            players = players_base.merge(current_live_data, on="player_id", how="left")

            # Ensure event column exists
            if "event" not in players.columns:
                players["event"] = target_gameweek
        else:
            # Use baseline data without live stats
            players = players_base.copy()
            players["total_points"] = 0  # Reset for new gameweek
            players["event"] = target_gameweek

        # Get additional data for XP calculation using enhanced client
        xg_rates = client.get_player_xg_xa_rates()
        fixtures = client.get_fixtures_normalized()

        # Standardize column names for compatibility
        rename_dict = {
            "price_gbp": "price",
            "selected_by_percentage": "selected_by_percent",
            "availability_status": "status",
        }

        # Handle team column naming - keep original name to avoid conflicts
        if "team_id" in players.columns and "team" not in players.columns:
            rename_dict["team_id"] = "team"

        players = players.rename(columns=rename_dict)

        # Add calculated fields with robust handling
        # Ensure event column exists and has valid values
        if "event" not in players.columns:
            players["event"] = target_gameweek

        players["points_per_game"] = players["total_points"].fillna(0) / players[
            "event"
        ].fillna(target_gameweek).replace(0, 1)
        players["form"] = players["total_points"].fillna(0).astype(float)

        teams = teams_df.rename(columns={"team_id": "id"})

        print(f"✅ Loaded {len(players)} players, {len(teams)} teams from database")
        print(f"📅 Target GW: {target_gameweek}")

        return players, teams, xg_rates, fixtures, target_gameweek, live_data_combined

    def _fetch_manager_team_internal(self, previous_gameweek: int) -> Optional[Dict]:
        """
        Fetch manager's team from previous gameweek

        Args:
            previous_gameweek: The gameweek to fetch team data from

        Returns:
            Dictionary with team info including picks, bank balance, etc., or None if failed
        """
        print(f"🔄 Fetching your team from GW{previous_gameweek}...")

        try:
            from client import FPLDataClient

            client = FPLDataClient()

            # Get manager data
            manager_data = client.get_my_manager_data()
            if manager_data.empty:
                print("❌ No manager data found in database")
                return None

            # Get picks from previous gameweek with enhanced methods
            try:
                # Try to get historical picks for specific gameweek
                historical_picks = client.get_my_picks_history(
                    start_gw=previous_gameweek, end_gw=previous_gameweek
                )
                if not historical_picks.empty:
                    current_picks = historical_picks
                    print(f"✅ Found historical picks for GW{previous_gameweek}")
                else:
                    # Fallback to current picks
                    current_picks = client.get_my_current_picks()
                    if not current_picks.empty:
                        print("⚠️ Using current picks as fallback")
                    else:
                        print("❌ No picks found in database")
                        return None
            except Exception as e:
                print(f"⚠️ Error getting picks: {str(e)[:50]}")
                # Last resort fallback to current picks
                current_picks = client.get_my_current_picks()
                if current_picks.empty:
                    print("❌ No current picks found in database")
                    return None

            # Convert picks to the expected format
            picks = []
            for _, pick in current_picks.iterrows():
                picks.append(
                    {
                        "element": pick["player_id"],
                        "is_captain": pick.get("is_captain", False),
                        "is_vice_captain": pick.get("is_vice_captain", False),
                        "multiplier": pick.get("multiplier", 1),
                    }
                )

            # Get manager info from first row
            manager_info = manager_data.iloc[0]

            # Get chip usage data
            chips_available, chips_used = self._get_chip_status_internal(
                client, previous_gameweek
            )

            team_info = {
                "manager_id": manager_info.get("manager_id", 0),
                "entry_name": manager_info.get("entry_name", "My Team"),
                "total_points": manager_info.get("summary_overall_points", 0),
                "bank": manager_info.get("bank", 0)
                / 10.0,  # Convert from 0.1M units to millions
                "team_value": manager_info.get("team_value", 1000)
                / 10.0,  # Convert from 0.1M units to millions
                "picks": picks,
                "free_transfers": 1,  # Default to 1 free transfer
                "chips_available": chips_available,
                "chips_used": chips_used,
            }

            print(
                f"✅ Loaded team from GW{previous_gameweek}: {team_info['entry_name']}"
            )
            return team_info

        except Exception as e:
            print(f"❌ Error fetching team from database: {e}")
            print("💡 Make sure manager data is available in the database")
            return None

    def _get_chip_status_internal(
        self, client, current_gameweek: int
    ) -> Tuple[List[str], List[str]]:
        """
        Get available and used chips based on current gameweek and season progress

        Args:
            client: FPL data client
            current_gameweek: Current gameweek number

        Returns:
            Tuple of (chips_available, chips_used)
        """
        # Standard chips available at season start
        all_chips = ["wildcard", "bench_boost", "triple_captain", "free_hit"]

        # In practice, you'd query the database for actual chip usage
        # For now, implement simple heuristics based on gameweek
        chips_used = []

        # Simplified chip availability logic
        # In a real implementation, this would query manager's chip usage history
        if current_gameweek <= 19:
            # First half of season - all chips potentially available
            chips_available = all_chips.copy()
            # Wildcard gets a second one after GW19
        else:
            # Second half - would need to check if first wildcard was used
            chips_available = ["wildcard", "bench_boost", "triple_captain", "free_hit"]

        try:
            # Attempt to get actual chip usage from database
            # This would need to be implemented in the FPL data client
            # For now, return defaults
            pass
        except Exception:
            # Fallback to defaults
            pass

        return chips_available, chips_used

    def _process_current_squad_internal(
        self, team_data: Dict, players: pd.DataFrame, teams: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process current squad data by merging team picks with player details

        Args:
            team_data: Team data dictionary from fetch_manager_team
            players: Players DataFrame
            teams: Teams DataFrame

        Returns:
            Processed current squad DataFrame with captain info and team names
        """
        if not team_data:
            return pd.DataFrame()

        # Get player details and merge with current data
        player_ids = [pick["element"] for pick in team_data["picks"]]
        current_squad = players[players["player_id"].isin(player_ids)].copy()

        # Ensure standardized team column naming
        if "team_id" not in current_squad.columns:
            # Look for alternative team column names and standardize
            team_col_mapping = {
                "team": "team_id",
                "team_id_x": "team_id",
                "team_id_y": "team_id",
            }

            found_team_col = None
            for alt_col, standard_col in team_col_mapping.items():
                if alt_col in current_squad.columns:
                    current_squad = current_squad.rename(
                        columns={alt_col: standard_col}
                    )
                    found_team_col = alt_col
                    print(f"📋 Standardized team column: {alt_col} -> {standard_col}")
                    break

            if not found_team_col:
                raise ValueError(
                    f"No valid team column found in current squad data. "
                    f"Available columns: {list(current_squad.columns)}. "
                    f"Expected one of: {list(team_col_mapping.keys())}"
                )

        # Merge with team data using standardized column name
        if "team_id" in current_squad.columns:
            current_squad = current_squad.merge(
                teams[["id", "name"]], left_on="team_id", right_on="id", how="left"
            )
            # Fill any missing team names (defensive programming)
            if "name" in current_squad.columns:
                current_squad["name"] = current_squad["name"].fillna("Unknown Team")

        # Add captain info
        captain_id = next(
            (pick["element"] for pick in team_data["picks"] if pick["is_captain"]), None
        )
        vice_captain_id = next(
            (pick["element"] for pick in team_data["picks"] if pick["is_vice_captain"]),
            None,
        )

        current_squad["role"] = current_squad["player_id"].apply(
            lambda x: "(C)"
            if x == captain_id
            else "(VC)"
            if x == vice_captain_id
            else ""
        )

        return current_squad
