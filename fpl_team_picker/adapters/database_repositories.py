"""Database repository implementations."""

import sys
import os
from typing import List, Optional
import pandas as pd

from fpl_team_picker.domain.repositories.player_repository import PlayerRepository
from fpl_team_picker.domain.models.player import (
    PlayerDomain,
    LiveDataDomain,
    EnrichedPlayerDomain,
)
from fpl_team_picker.domain.common.result import Result, DomainError, ErrorType


class DatabasePlayerRepository(PlayerRepository):
    """Database implementation of PlayerRepository using fpl-dataset-builder."""

    def __init__(self):
        """Initialize repository with database access."""
        # Add dataset builder to path
        dataset_builder_path = os.path.join(
            os.path.dirname(__file__), "../../fpl-dataset-builder"
        )
        if dataset_builder_path not in sys.path:
            sys.path.append(dataset_builder_path)

    def get_current_players(self) -> Result[List[PlayerDomain]]:
        """Get all current players for the season."""
        try:
            from client import FPLDataClient

            client = FPLDataClient()
            players_df = client.get_current_players()

            if players_df.empty:
                return Result.failure(
                    DomainError.data_not_found("No player data available")
                )

            # Convert DataFrame to domain models (simplified for now)
            players = []
            for _, row in players_df.iterrows():
                # Basic conversion - would be more comprehensive in practice
                player = PlayerDomain(
                    player_id=row.get("player_id", 0),
                    web_name=row.get("web_name", ""),
                    position=row.get("position", ""),
                    team_id=row.get("team_id", 0),
                    price=row.get("price_gbp", 0.0),
                )
                players.append(player)

            return Result.success(players)

        except Exception as e:
            return Result.failure(
                DomainError(
                    error_type=ErrorType.DATA_ACCESS_ERROR,
                    message=f"Failed to load players: {str(e)}",
                )
            )

    def get_enriched_players_dataframe(self) -> Result[pd.DataFrame]:
        """Get all current players with enriched season statistics.

        Returns pandas DataFrame for compatibility with existing code.
        Eventually this would return List[EnrichedPlayerDomain].
        """
        try:
            # Import database operations
            from db.operations import DatabaseOperations

            db_ops = DatabaseOperations()

            # Get raw player data with all attributes
            raw_players = db_ops.get_raw_players_bootstrap()
            raw_positions = db_ops.get_raw_element_types()

            if raw_players.empty:
                return Result.failure(
                    DomainError.data_not_found("No player data available")
                )

            if raw_positions.empty:
                return Result.failure(
                    DomainError.data_not_found("No position data available")
                )

            # Create position mapping
            position_mapping = {}
            if not raw_positions.empty:
                position_mapping = dict(
                    zip(
                        raw_positions["position_id"],
                        raw_positions["singular_name_short"],
                        strict=False,
                    )
                )

            # Create enriched player DataFrame
            enriched_players = pd.DataFrame(
                {
                    "player_id": raw_players["player_id"],
                    "web_name": raw_players["web_name"],
                    "first": raw_players["first_name"],
                    "second": raw_players["second_name"],
                    "team_id": raw_players["team_id"],
                    "position": raw_players["position_id"].map(position_mapping),
                    "price": raw_players["now_cost"] / 10.0,
                    "selected_by_percent": pd.to_numeric(
                        raw_players["selected_by_percent"], errors="coerce"
                    ),
                    "status": raw_players["status"],
                    # Season performance stats
                    "total_points_season": pd.to_numeric(
                        raw_players["total_points"], errors="coerce"
                    ).fillna(0),
                    "form_season": pd.to_numeric(
                        raw_players["form"], errors="coerce"
                    ).fillna(0),
                    "points_per_game_season": pd.to_numeric(
                        raw_players["points_per_game"], errors="coerce"
                    ).fillna(0),
                    "minutes": pd.to_numeric(
                        raw_players["minutes"], errors="coerce"
                    ).fillna(0),
                    "starts": pd.to_numeric(
                        raw_players["starts"], errors="coerce"
                    ).fillna(0),
                    # Match stats
                    "goals_scored": pd.to_numeric(
                        raw_players["goals_scored"], errors="coerce"
                    ).fillna(0),
                    "assists": pd.to_numeric(
                        raw_players["assists"], errors="coerce"
                    ).fillna(0),
                    "clean_sheets": pd.to_numeric(
                        raw_players["clean_sheets"], errors="coerce"
                    ).fillna(0),
                    "goals_conceded": pd.to_numeric(
                        raw_players["goals_conceded"], errors="coerce"
                    ).fillna(0),
                    "yellow_cards": pd.to_numeric(
                        raw_players["yellow_cards"], errors="coerce"
                    ).fillna(0),
                    "red_cards": pd.to_numeric(
                        raw_players["red_cards"], errors="coerce"
                    ).fillna(0),
                    "saves": pd.to_numeric(
                        raw_players["saves"], errors="coerce"
                    ).fillna(0),
                    # Bonus and BPS
                    "bonus": pd.to_numeric(
                        raw_players["bonus"], errors="coerce"
                    ).fillna(0),
                    "bps": pd.to_numeric(raw_players["bps"], errors="coerce").fillna(0),
                    # ICT Index components
                    "influence": pd.to_numeric(
                        raw_players["influence"], errors="coerce"
                    ).fillna(0),
                    "creativity": pd.to_numeric(
                        raw_players["creativity"], errors="coerce"
                    ).fillna(0),
                    "threat": pd.to_numeric(
                        raw_players["threat"], errors="coerce"
                    ).fillna(0),
                    "ict_index": pd.to_numeric(
                        raw_players["ict_index"], errors="coerce"
                    ).fillna(0),
                    # Expected stats
                    "expected_goals": pd.to_numeric(
                        raw_players["expected_goals"], errors="coerce"
                    ).fillna(0),
                    "expected_assists": pd.to_numeric(
                        raw_players["expected_assists"], errors="coerce"
                    ).fillna(0),
                    "expected_goals_per_90": pd.to_numeric(
                        raw_players["expected_goals_per_90"], errors="coerce"
                    ).fillna(0),
                    "expected_assists_per_90": pd.to_numeric(
                        raw_players["expected_assists_per_90"], errors="coerce"
                    ).fillna(0),
                    # Market data
                    "value_form": pd.to_numeric(
                        raw_players["value_form"], errors="coerce"
                    ).fillna(0),
                    "value_season": pd.to_numeric(
                        raw_players["value_season"], errors="coerce"
                    ).fillna(0),
                    "transfers_in": pd.to_numeric(
                        raw_players["transfers_in"], errors="coerce"
                    ).fillna(0),
                    "transfers_out": pd.to_numeric(
                        raw_players["transfers_out"], errors="coerce"
                    ).fillna(0),
                    "transfers_in_event": pd.to_numeric(
                        raw_players["transfers_in_event"], errors="coerce"
                    ).fillna(0),
                    "transfers_out_event": pd.to_numeric(
                        raw_players["transfers_out_event"], errors="coerce"
                    ).fillna(0),
                    # Availability
                    "chance_of_playing_this_round": pd.to_numeric(
                        raw_players["chance_of_playing_this_round"], errors="coerce"
                    ),
                    "chance_of_playing_next_round": pd.to_numeric(
                        raw_players["chance_of_playing_next_round"], errors="coerce"
                    ),
                    # Set pieces and news
                    "penalties_order": pd.to_numeric(
                        raw_players["penalties_order"], errors="coerce"
                    ),
                    "corners_and_indirect_freekicks_order": pd.to_numeric(
                        raw_players["corners_and_indirect_freekicks_order"],
                        errors="coerce",
                    ),
                    "news": raw_players["news"].fillna(""),
                    # Metadata
                    "as_of_utc": raw_players["as_of_utc"],
                }
            )

            # Data validation
            if len(enriched_players) < 100:
                return Result.failure(
                    DomainError.validation_error(
                        f"Invalid player data: expected >100 players, got {len(enriched_players)}"
                    )
                )

            if enriched_players["position"].isna().sum() > 0:
                return Result.failure(
                    DomainError.validation_error(
                        "Position mapping failed - some players have no position"
                    )
                )

            return Result.success(enriched_players)

        except ImportError as e:
            return Result.failure(
                DomainError(
                    error_type=ErrorType.CONFIGURATION_ERROR,
                    message=f"Could not import database operations: {str(e)}",
                )
            )
        except Exception as e:
            return Result.failure(
                DomainError(
                    error_type=ErrorType.DATA_ACCESS_ERROR,
                    message=f"Failed to load enriched players: {str(e)}",
                )
            )

    def get_enriched_players(self) -> Result[List[EnrichedPlayerDomain]]:
        """Get all current players with enriched season statistics as Pydantic domain objects.

        This method provides the future-facing implementation that returns validated
        Pydantic domain objects with comprehensive type safety and business logic.
        """
        try:
            # Get the DataFrame first (reuse existing logic)
            df_result = self.get_enriched_players_dataframe()
            if df_result.is_failure:
                return Result.failure(df_result.error)

            enriched_df = df_result.value

            # Convert DataFrame rows to validated Pydantic domain objects
            enriched_players = []
            validation_errors = []

            for _, row in enriched_df.iterrows():
                try:
                    # Create Pydantic domain object with full validation
                    player = EnrichedPlayerDomain(
                        # Core PlayerDomain fields
                        player_id=int(row["player_id"]),
                        web_name=str(row["web_name"]),
                        first_name=row.get("first"),
                        last_name=row.get("second"),
                        team_id=int(row["team_id"]),
                        position=str(row["position"]),
                        price=float(row["price"]),
                        selected_by_percent=float(row["selected_by_percent"]),
                        availability_status=str(row.get("status", "a")),
                        as_of_utc=row["as_of_utc"],
                        # EnrichedPlayerDomain specific fields
                        total_points_season=int(row["total_points_season"]),
                        form_season=float(row["form_season"]),
                        points_per_game_season=float(row["points_per_game_season"]),
                        minutes=int(row["minutes"]),
                        starts=int(row["starts"]),
                        goals_scored=int(row["goals_scored"]),
                        assists=int(row["assists"]),
                        clean_sheets=int(row["clean_sheets"]),
                        goals_conceded=int(row["goals_conceded"]),
                        yellow_cards=int(row["yellow_cards"]),
                        red_cards=int(row["red_cards"]),
                        saves=int(row["saves"]),
                        bonus=int(row["bonus"]),
                        bps=int(row["bps"]),
                        influence=float(row["influence"]),
                        creativity=float(row["creativity"]),
                        threat=float(row["threat"]),
                        ict_index=float(row["ict_index"]),
                        expected_goals=float(row["expected_goals"]),
                        expected_assists=float(row["expected_assists"]),
                        expected_goals_per_90=float(row["expected_goals_per_90"]),
                        expected_assists_per_90=float(row["expected_assists_per_90"]),
                        value_form=float(row["value_form"]),
                        value_season=float(row["value_season"]),
                        transfers_in=int(row["transfers_in"]),
                        transfers_out=int(row["transfers_out"]),
                        transfers_in_event=int(row["transfers_in_event"]),
                        transfers_out_event=int(row["transfers_out_event"]),
                        chance_of_playing_this_round=row.get(
                            "chance_of_playing_this_round"
                        ),
                        chance_of_playing_next_round=row.get(
                            "chance_of_playing_next_round"
                        ),
                        penalties_order=row.get("penalties_order"),
                        corners_and_indirect_freekicks_order=row.get(
                            "corners_and_indirect_freekicks_order"
                        ),
                        news=str(row.get("news", "")),
                    )
                    enriched_players.append(player)

                except Exception as e:
                    validation_errors.append(
                        f"Player {row.get('player_id', 'unknown')}: {str(e)}"
                    )

            # Check if we have too many validation errors
            if (
                validation_errors and len(validation_errors) > len(enriched_df) * 0.1
            ):  # >10% failure rate
                return Result.failure(
                    DomainError.validation_error(
                        f"Too many validation errors ({len(validation_errors)}/{len(enriched_df)}). "
                        f"Sample errors: {validation_errors[:3]}"
                    )
                )
            elif validation_errors:
                # Log warnings but continue if <10% failure rate
                print(
                    f"⚠️ {len(validation_errors)} players failed validation: {validation_errors[:3]}"
                )

            print(
                f"✅ Successfully created {len(enriched_players)} validated EnrichedPlayerDomain objects"
            )
            return Result.success(enriched_players)

        except Exception as e:
            return Result.failure(
                DomainError(
                    error_type=ErrorType.DATA_ACCESS_ERROR,
                    message=f"Failed to create enriched player domain objects: {str(e)}",
                )
            )

    def get_player_by_id(self, player_id: int) -> Result[Optional[PlayerDomain]]:
        """Get a specific player by ID."""
        # Implementation would filter by player_id
        return Result.failure(
            DomainError(
                error_type=ErrorType.SYSTEM_ERROR, message="Not implemented yet"
            )
        )

    def get_players_by_team(self, team_id: int) -> Result[List[PlayerDomain]]:
        """Get all players for a specific team."""
        # Implementation would filter by team_id
        return Result.failure(
            DomainError(
                error_type=ErrorType.SYSTEM_ERROR, message="Not implemented yet"
            )
        )

    def get_players_by_position(self, position: str) -> Result[List[PlayerDomain]]:
        """Get all players for a specific position."""
        # Implementation would filter by position
        return Result.failure(
            DomainError(
                error_type=ErrorType.SYSTEM_ERROR, message="Not implemented yet"
            )
        )

    def get_live_data_for_gameweek(self, gameweek: int) -> Result[List[LiveDataDomain]]:
        """Get live performance data for all players in a specific gameweek."""
        # Implementation would get live data
        return Result.failure(
            DomainError(
                error_type=ErrorType.SYSTEM_ERROR, message="Not implemented yet"
            )
        )

    def get_player_live_data(
        self, player_id: int, gameweek: int
    ) -> Result[Optional[LiveDataDomain]]:
        """Get live data for a specific player in a specific gameweek."""
        # Implementation would get specific player live data
        return Result.failure(
            DomainError(
                error_type=ErrorType.SYSTEM_ERROR, message="Not implemented yet"
            )
        )
