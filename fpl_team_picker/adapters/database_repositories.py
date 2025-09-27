"""Database implementations of repository interfaces."""

import pandas as pd
from datetime import datetime
from typing import List, Optional

from client import FPLDataClient
from pydantic import ValidationError

from ..domain.common.result import DomainError, Result
from ..domain.models.fixture import FixtureDomain
from ..domain.models.player import (
    AvailabilityStatus,
    LiveDataDomain,
    PlayerDomain,
    Position,
)
from ..domain.models.team import TeamDomain
from ..domain.repositories.fixture_repository import FixtureRepository
from ..domain.repositories.player_repository import PlayerRepository
from ..domain.repositories.team_repository import TeamRepository


class DatabasePlayerRepository(PlayerRepository):
    """Database implementation of PlayerRepository using FPLDataClient."""

    def __init__(self, client: Optional[FPLDataClient] = None):
        self.client = client or FPLDataClient()

    def get_current_players(self) -> Result[List[PlayerDomain]]:
        """Get all current players with strict validation."""
        try:
            players_df = self.client.get_current_players()
            if players_df.empty:
                return Result.failure(
                    DomainError.data_not_found("No player data available")
                )

            players = []
            validation_errors = []

            for _, row in players_df.iterrows():
                try:
                    # Map column names from database to domain model
                    player_data = {
                        "player_id": int(row["player_id"]),
                        "web_name": str(row["web_name"]),
                        "first_name": str(row.get("first", "")).strip() or None,
                        "last_name": str(row.get("second", "")).strip() or None,
                        "team_id": int(row["team_id"]),
                        "position": Position(row["position"]),
                        "price": float(row["price_gbp"]),
                        "selected_by_percent": float(row["selected_by_percentage"]),
                        "availability_status": AvailabilityStatus(
                            row.get("availability_status", "a")
                        ),
                        "as_of_utc": pd.to_datetime(row["as_of_utc"]).to_pydatetime(),
                        "mapped_player_id": int(row["mapped_player_id"])
                        if pd.notna(row.get("mapped_player_id"))
                        else None,
                    }

                    player = PlayerDomain(**player_data)
                    players.append(player)

                except (ValueError, ValidationError) as e:
                    validation_errors.append(
                        {"player_id": row.get("player_id", "unknown"), "error": str(e)}
                    )

            if validation_errors:
                return Result.failure(
                    DomainError.validation_error(
                        f"Failed to validate {len(validation_errors)} players",
                        details={"validation_errors": validation_errors},
                    )
                )

            return Result.success(players)

        except Exception as e:
            return Result.failure(
                DomainError.external_api_error(f"Failed to fetch player data: {str(e)}")
            )

    def get_player_by_id(self, player_id: int) -> Result[Optional[PlayerDomain]]:
        """Get a specific player by ID."""
        result = self.get_current_players()
        if result.is_failure:
            return result

        player = next((p for p in result.value if p.player_id == player_id), None)
        return Result.success(player)

    def get_players_by_team(self, team_id: int) -> Result[List[PlayerDomain]]:
        """Get all players for a specific team."""
        result = self.get_current_players()
        if result.is_failure:
            return result

        team_players = [p for p in result.value if p.team_id == team_id]
        return Result.success(team_players)

    def get_players_by_position(self, position: str) -> Result[List[PlayerDomain]]:
        """Get all players for a specific position."""
        try:
            position_enum = Position(position)
        except ValueError:
            return Result.failure(
                DomainError.validation_error(
                    f"Invalid position: {position}. Must be one of: {[p.value for p in Position]}"
                )
            )

        result = self.get_current_players()
        if result.is_failure:
            return result

        position_players = [p for p in result.value if p.position == position_enum]
        return Result.success(position_players)

    def get_live_data_for_gameweek(self, gameweek: int) -> Result[List[LiveDataDomain]]:
        """Get live performance data for all players in a specific gameweek."""
        if not (1 <= gameweek <= 38):
            return Result.failure(
                DomainError.validation_error(
                    f"Invalid gameweek: {gameweek}. Must be between 1 and 38."
                )
            )

        try:
            live_df = self.client.get_gameweek_live_data(gameweek)
            if live_df.empty:
                return Result.failure(
                    DomainError.data_not_found(
                        f"No live data available for gameweek {gameweek}"
                    )
                )

            live_data = []
            validation_errors = []

            for _, row in live_df.iterrows():
                try:
                    # Strict validation - no coercion
                    live_data_dict = {
                        "player_id": int(row["player_id"]),
                        "gameweek": gameweek,
                        "minutes": int(row.get("minutes", 0)),
                        "total_points": int(row.get("total_points", 0)),
                        "goals_scored": int(row.get("goals_scored", 0)),
                        "assists": int(row.get("assists", 0)),
                        "clean_sheets": int(row.get("clean_sheets", 0)),
                        "goals_conceded": int(row.get("goals_conceded", 0)),
                        "yellow_cards": int(row.get("yellow_cards", 0)),
                        "red_cards": int(row.get("red_cards", 0)),
                        "saves": int(row.get("saves", 0)),
                        "bonus": int(row.get("bonus", 0)),
                        "bps": int(row.get("bps", 0)),
                        "influence": float(row.get("influence", 0.0)),
                        "creativity": float(row.get("creativity", 0.0)),
                        "threat": float(row.get("threat", 0.0)),
                        "ict_index": float(row.get("ict_index", 0.0)),
                        "expected_goals": float(row["expected_goals"])
                        if pd.notna(row.get("expected_goals"))
                        else None,
                        "expected_assists": float(row["expected_assists"])
                        if pd.notna(row.get("expected_assists"))
                        else None,
                        "expected_goal_involvements": float(
                            row["expected_goal_involvements"]
                        )
                        if pd.notna(row.get("expected_goal_involvements"))
                        else None,
                        "expected_goals_conceded": float(row["expected_goals_conceded"])
                        if pd.notna(row.get("expected_goals_conceded"))
                        else None,
                        "value": float(row.get("value", 4.0)),
                        "was_home": bool(row.get("was_home", False)),
                        "opponent_team": int(row.get("opponent_team", 1)),
                    }

                    live_data_obj = LiveDataDomain(**live_data_dict)
                    live_data.append(live_data_obj)

                except (ValueError, ValidationError) as e:
                    validation_errors.append(
                        {"player_id": row.get("player_id", "unknown"), "error": str(e)}
                    )

            if validation_errors:
                return Result.failure(
                    DomainError.validation_error(
                        f"Failed to validate {len(validation_errors)} live data records",
                        details={"validation_errors": validation_errors},
                    )
                )

            return Result.success(live_data)

        except Exception as e:
            return Result.failure(
                DomainError.external_api_error(
                    f"Failed to fetch live data for gameweek {gameweek}: {str(e)}"
                )
            )

    def get_player_live_data(
        self, player_id: int, gameweek: int
    ) -> Result[Optional[LiveDataDomain]]:
        """Get live data for a specific player in a specific gameweek."""
        result = self.get_live_data_for_gameweek(gameweek)
        if result.is_failure:
            return result

        player_data = next(
            (ld for ld in result.value if ld.player_id == player_id), None
        )
        return Result.success(player_data)


class DatabaseTeamRepository(TeamRepository):
    """Database implementation of TeamRepository using FPLDataClient."""

    def __init__(self, client: Optional[FPLDataClient] = None):
        self.client = client or FPLDataClient()

    def get_all_teams(self) -> Result[List[TeamDomain]]:
        """Get all Premier League teams with strict validation."""
        try:
            teams_df = self.client.get_current_teams()
            if teams_df.empty:
                return Result.failure(
                    DomainError.data_not_found("No team data available")
                )

            teams = []
            validation_errors = []

            for _, row in teams_df.iterrows():
                try:
                    team_data = {
                        "team_id": int(row["team_id"]),
                        "name": str(row["name"]),
                        "short_name": str(row["short_name"]),
                        "as_of_utc": pd.to_datetime(row["as_of_utc"]).to_pydatetime(),
                    }

                    team = TeamDomain(**team_data)
                    teams.append(team)

                except (ValueError, ValidationError) as e:
                    validation_errors.append(
                        {"team_id": row.get("team_id", "unknown"), "error": str(e)}
                    )

            if validation_errors:
                return Result.failure(
                    DomainError.validation_error(
                        f"Failed to validate {len(validation_errors)} teams",
                        details={"validation_errors": validation_errors},
                    )
                )

            return Result.success(teams)

        except Exception as e:
            return Result.failure(
                DomainError.external_api_error(f"Failed to fetch team data: {str(e)}")
            )

    def get_team_by_id(self, team_id: int) -> Result[Optional[TeamDomain]]:
        """Get a specific team by ID."""
        result = self.get_all_teams()
        if result.is_failure:
            return result

        team = next((t for t in result.value if t.team_id == team_id), None)
        return Result.success(team)

    def get_team_by_name(self, name: str) -> Result[Optional[TeamDomain]]:
        """Get a team by name (full or short name)."""
        result = self.get_all_teams()
        if result.is_failure:
            return result

        name_lower = name.lower()
        team = next(
            (
                t
                for t in result.value
                if t.name.lower() == name_lower or t.short_name.lower() == name_lower
            ),
            None,
        )
        return Result.success(team)


class DatabaseFixtureRepository(FixtureRepository):
    """Database implementation of FixtureRepository using FPLDataClient."""

    def __init__(self, client: Optional[FPLDataClient] = None):
        self.client = client or FPLDataClient()

    def get_all_fixtures(self) -> Result[List[FixtureDomain]]:
        """Get all fixtures with strict validation."""
        try:
            fixtures_df = self.client.get_fixtures_normalized()
            if fixtures_df.empty:
                return Result.failure(
                    DomainError.data_not_found("No fixture data available")
                )

            fixtures = []
            validation_errors = []

            for _, row in fixtures_df.iterrows():
                try:
                    fixture_data = {
                        "fixture_id": int(row["fixture_id"]),
                        "event": int(row["event"]),
                        "kickoff_utc": pd.to_datetime(
                            row["kickoff_utc"]
                        ).to_pydatetime(),
                        "home_team_id": int(row["home_team_id"]),
                        "away_team_id": int(row["away_team_id"]),
                        "as_of_utc": pd.to_datetime(row["as_of_utc"]).to_pydatetime(),
                    }

                    fixture = FixtureDomain(**fixture_data)
                    fixtures.append(fixture)

                except (ValueError, ValidationError) as e:
                    validation_errors.append(
                        {
                            "fixture_id": row.get("fixture_id", "unknown"),
                            "error": str(e),
                        }
                    )

            if validation_errors:
                return Result.failure(
                    DomainError.validation_error(
                        f"Failed to validate {len(validation_errors)} fixtures",
                        details={"validation_errors": validation_errors},
                    )
                )

            return Result.success(fixtures)

        except Exception as e:
            return Result.failure(
                DomainError.external_api_error(
                    f"Failed to fetch fixture data: {str(e)}"
                )
            )

    def get_fixtures_for_gameweek(self, gameweek: int) -> Result[List[FixtureDomain]]:
        """Get all fixtures for a specific gameweek."""
        if not (1 <= gameweek <= 38):
            return Result.failure(
                DomainError.validation_error(
                    f"Invalid gameweek: {gameweek}. Must be between 1 and 38."
                )
            )

        result = self.get_all_fixtures()
        if result.is_failure:
            return result

        gameweek_fixtures = [f for f in result.value if f.event == gameweek]
        return Result.success(gameweek_fixtures)

    def get_fixtures_for_team(self, team_id: int) -> Result[List[FixtureDomain]]:
        """Get all fixtures for a specific team."""
        if not (1 <= team_id <= 20):
            return Result.failure(
                DomainError.validation_error(
                    f"Invalid team_id: {team_id}. Must be between 1 and 20."
                )
            )

        result = self.get_all_fixtures()
        if result.is_failure:
            return result

        team_fixtures = [
            f
            for f in result.value
            if f.home_team_id == team_id or f.away_team_id == team_id
        ]
        return Result.success(team_fixtures)

    def get_upcoming_fixtures(
        self, team_id: Optional[int] = None, limit: int = 5
    ) -> Result[List[FixtureDomain]]:
        """Get upcoming fixtures, optionally filtered by team."""
        result = self.get_all_fixtures()
        if result.is_failure:
            return result

        now = datetime.utcnow()
        upcoming = [f for f in result.value if f.kickoff_utc > now]

        if team_id is not None:
            if not (1 <= team_id <= 20):
                return Result.failure(
                    DomainError.validation_error(
                        f"Invalid team_id: {team_id}. Must be between 1 and 20."
                    )
                )
            upcoming = [
                f
                for f in upcoming
                if f.home_team_id == team_id or f.away_team_id == team_id
            ]

        # Sort by kickoff time and limit
        upcoming.sort(key=lambda f: f.kickoff_utc)
        return Result.success(upcoming[:limit])
