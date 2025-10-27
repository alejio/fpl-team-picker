"""Database repository implementations."""

import sys
import os
from typing import List, Optional, Dict, Union
from datetime import datetime
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from fpl_team_picker.domain.repositories.player_repository import PlayerRepository
from fpl_team_picker.domain.models.player import (
    PlayerDomain,
    LiveDataDomain,
    EnrichedPlayerDomain,
    Position,
    AvailabilityStatus,
)
from fpl_team_picker.domain.common.result import Result, DomainError, ErrorType


class RawPlayerData(BaseModel):
    """Pydantic model for validating raw player data from FPL API/database."""

    player_id: int = Field(..., gt=0, description="Player ID must be positive")
    web_name: str = Field(..., min_length=1, description="Web name cannot be empty")
    position: str = Field(..., description="Position code")
    team_id: int = Field(..., ge=1, le=20, description="Team ID must be 1-20")
    price_gbp: Optional[float] = Field(None, ge=0.0, description="Price in GBP")
    now_cost: Optional[int] = Field(
        None, ge=39, le=150, description="Price in 0.1m units"
    )
    selected_by_percent: Optional[str] = Field(
        None, description="Selection percentage as string"
    )
    status: Optional[str] = Field(None, description="Player status code")
    first_name: Optional[str] = Field(None, description="First name")
    second_name: Optional[str] = Field(None, description="Last name")
    as_of_utc: Optional[datetime] = Field(None, description="Data timestamp")

    @field_validator("price_gbp")
    @classmethod
    def validate_price_gbp(cls, v: Optional[float]) -> Optional[float]:
        """Validate GBP price if provided."""
        if v is not None and (v < 3.9 or v > 15.0):
            raise ValueError(f"Price must be between £3.9m and £15.0m, got {v}")
        return v

    @field_validator("selected_by_percent")
    @classmethod
    def validate_selection_percentage(cls, v: Optional[str]) -> Optional[str]:
        """Validate selection percentage string format."""
        if v is not None:
            try:
                percentage = float(v)
                if not (0.0 <= percentage <= 100.0):
                    raise ValueError(
                        f"Selection percentage must be 0-100, got {percentage}"
                    )
            except ValueError as e:
                if "Selection percentage must be" in str(e):
                    raise
                raise ValueError(f"Invalid selection percentage format: {v}")
        return v


class RawEnrichedPlayerData(BaseModel):
    """Pydantic model for validating comprehensive raw player data."""

    # Core fields
    player_id: int = Field(..., gt=0)
    web_name: str = Field(..., min_length=1)
    first_name: Optional[str] = Field(None)
    second_name: Optional[str] = Field(None)
    team_id: int = Field(..., ge=1, le=20)
    position_id: int = Field(..., ge=1, le=4, description="Position ID 1-4")
    now_cost: int = Field(..., ge=38, le=150, description="Price in 0.1m units")
    selected_by_percent: str = Field(..., description="Selection percentage")
    status: str = Field(default="a", description="Availability status")
    as_of_utc: datetime = Field(..., description="Data timestamp")

    # Season performance (allow both strings and numbers for flexible DB sources)
    total_points: Optional[Union[str, int, float]] = Field(None)
    form: Optional[Union[str, float]] = Field(None)
    points_per_game: Optional[Union[str, float]] = Field(None)
    minutes: Optional[Union[str, int]] = Field(None)
    starts: Optional[Union[str, int]] = Field(None)

    # Match statistics
    goals_scored: Optional[Union[str, int]] = Field(None)
    assists: Optional[Union[str, int]] = Field(None)
    clean_sheets: Optional[Union[str, int]] = Field(None)
    goals_conceded: Optional[Union[str, int]] = Field(None)
    yellow_cards: Optional[Union[str, int]] = Field(None)
    red_cards: Optional[Union[str, int]] = Field(None)
    saves: Optional[Union[str, int]] = Field(None)

    # Bonus and ICT
    bonus: Optional[Union[str, int]] = Field(None)
    bps: Optional[Union[str, int]] = Field(None)
    influence: Optional[Union[str, float]] = Field(None)
    creativity: Optional[Union[str, float]] = Field(None)
    threat: Optional[Union[str, float]] = Field(None)
    ict_index: Optional[Union[str, float]] = Field(None)

    # Expected stats
    expected_goals: Optional[Union[str, float]] = Field(None)
    expected_assists: Optional[Union[str, float]] = Field(None)
    expected_goals_per_90: Optional[Union[str, float]] = Field(None)
    expected_assists_per_90: Optional[Union[str, float]] = Field(None)

    # Market data
    value_form: Optional[Union[str, float]] = Field(None)
    value_season: Optional[Union[str, float]] = Field(None)
    transfers_in: Optional[Union[str, int]] = Field(None)
    transfers_out: Optional[Union[str, int]] = Field(None)
    transfers_in_event: Optional[Union[str, int]] = Field(None)
    transfers_out_event: Optional[Union[str, int]] = Field(None)

    # Availability
    chance_of_playing_this_round: Optional[Union[str, float]] = Field(None)
    chance_of_playing_next_round: Optional[Union[str, float]] = Field(None)

    # Set pieces
    penalties_order: Optional[Union[str, int]] = Field(None)
    corners_and_indirect_freekicks_order: Optional[Union[str, int]] = Field(None)

    # News
    news: Optional[str] = Field(None)

    class Config:
        """Allow extra fields for forward compatibility."""

        extra = "allow"

    def to_numeric(self, field_name: str, default: float = 0.0) -> float:
        """Safely convert Union[str, int, float] field to numeric with validation."""
        value = getattr(self, field_name)
        if value is None or value == "":
            return default
        # Handle pandas NaN values
        if isinstance(value, float) and pd.isna(value):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric value for {field_name}: {value}")

    def to_int(self, field_name: str, default: int = 0) -> int:
        """Safely convert Union[str, int, float] field to integer with validation."""
        numeric_value = self.to_numeric(field_name, default)
        return int(numeric_value)


class DatabaseConfiguration(BaseModel):
    """Pydantic model for database operation configuration."""

    min_expected_players: int = Field(
        default=100, ge=50, description="Minimum expected players"
    )
    max_validation_error_rate: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Max error rate (0-50%)"
    )
    require_position_mapping: bool = Field(
        default=True, description="Require valid position mapping"
    )
    require_team_mapping: bool = Field(
        default=True, description="Require valid team mapping"
    )

    class Config:
        """Configuration for database settings."""

        extra = "forbid"


class DatabasePlayerRepository(PlayerRepository):
    """Database implementation of PlayerRepository using fpl-dataset-builder."""

    def __init__(self, config: Optional[DatabaseConfiguration] = None):
        """Initialize repository with validated configuration."""
        # Validate configuration with Pydantic
        self.config = config or DatabaseConfiguration()

        # Add dataset builder to path
        dataset_builder_path = os.path.join(
            os.path.dirname(__file__), "../../fpl-dataset-builder"
        )
        if dataset_builder_path not in sys.path:
            sys.path.append(dataset_builder_path)

    @staticmethod
    def _safe_optional_int(value) -> Optional[int]:
        """Convert value to int or None, handling pandas nan."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return int(value)

    @staticmethod
    def _safe_optional_float(value) -> Optional[float]:
        """Convert value to float or None, handling pandas nan."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)

    def get_current_players(self) -> Result[List[PlayerDomain]]:
        """Get all current players with comprehensive Pydantic validation."""
        try:
            from client import FPLDataClient

            client = FPLDataClient()
            players_df = client.get_current_players()

            if players_df.empty:
                return Result.failure(
                    DomainError.data_not_found("No player data available")
                )

            # Validate minimum data quality expectations
            if len(players_df) < self.config.min_expected_players:
                return Result.failure(
                    DomainError.validation_error(
                        f"Insufficient player data: expected >={self.config.min_expected_players}, got {len(players_df)}"
                    )
                )

            # Convert DataFrame to validated domain models
            players = []
            validation_errors = []

            for _, row in players_df.iterrows():
                try:
                    # First validate raw data with Pydantic
                    raw_data = RawPlayerData(
                        player_id=row.get("player_id"),
                        web_name=row.get("web_name"),
                        position=row.get("position"),
                        team_id=row.get("team_id"),
                        price_gbp=row.get("price_gbp"),
                        now_cost=row.get("now_cost"),
                        selected_by_percent=row.get("selected_by_percent"),
                        status=row.get("status"),
                        first_name=row.get("first_name"),
                        second_name=row.get("second_name"),
                        as_of_utc=row.get("as_of_utc", datetime.now()),
                    )

                    # Convert to domain model with proper defaults and validation
                    price = raw_data.price_gbp
                    if price is None and raw_data.now_cost is not None:
                        price = raw_data.now_cost / 10.0  # Convert from 0.1m units

                    # Map position string to enum
                    position = self._map_position_to_enum(raw_data.position)

                    # Parse selection percentage
                    selection_pct = 0.0
                    if raw_data.selected_by_percent:
                        selection_pct = float(raw_data.selected_by_percent)

                    # Map status to availability enum
                    availability = self._map_status_to_availability(raw_data.status)

                    # Create validated domain object
                    player = PlayerDomain(
                        player_id=raw_data.player_id,
                        web_name=raw_data.web_name,
                        first_name=raw_data.first_name,
                        last_name=raw_data.second_name,
                        team_id=raw_data.team_id,
                        position=position,
                        price=price or 4.0,  # Default to minimum FPL price
                        selected_by_percent=selection_pct,
                        availability_status=availability,
                        as_of_utc=raw_data.as_of_utc or datetime.now(),
                    )
                    players.append(player)

                except ValidationError as e:
                    validation_errors.append(
                        f"Player {row.get('player_id', 'unknown')}: {str(e)}"
                    )
                except Exception as e:
                    validation_errors.append(
                        f"Player {row.get('player_id', 'unknown')}: Unexpected error - {str(e)}"
                    )

            # Check validation error rate
            error_rate = (
                len(validation_errors) / len(players_df)
                if players_df is not None and len(players_df) > 0
                else 0
            )
            if error_rate > self.config.max_validation_error_rate:
                return Result.failure(
                    DomainError.validation_error(
                        f"Too many validation errors: {len(validation_errors)}/{len(players_df)} "
                        f"({error_rate:.1%} > {self.config.max_validation_error_rate:.1%}). "
                        f"Sample errors: {validation_errors[:3]}"
                    )
                )

            if validation_errors:
                print(
                    f"⚠️ {len(validation_errors)} players failed validation (within tolerance): {validation_errors[:3]}"
                )

            print(f"✅ Successfully validated {len(players)} PlayerDomain objects")
            return Result.success(players)

        except ImportError as e:
            return Result.failure(
                DomainError(
                    error_type=ErrorType.CONFIGURATION_ERROR,
                    message=f"Could not import FPL data client: {str(e)}",
                )
            )
        except Exception as e:
            return Result.failure(
                DomainError(
                    error_type=ErrorType.DATA_ACCESS_ERROR,
                    message=f"Failed to load players: {str(e)}",
                )
            )

    def _map_position_to_enum(self, position_str: Optional[str]) -> Position:
        """Map position string to Position enum with validation."""
        if not position_str:
            raise ValueError("Position cannot be empty")

        position_mapping = {
            "GKP": Position.GKP,
            "DEF": Position.DEF,
            "MID": Position.MID,
            "FWD": Position.FWD,
            "Goalkeeper": Position.GKP,
            "Defender": Position.DEF,
            "Midfielder": Position.MID,
            "Forward": Position.FWD,
        }

        if position_str not in position_mapping:
            raise ValueError(f"Unknown position: {position_str}")

        return position_mapping[position_str]

    def _map_status_to_availability(
        self, status_str: Optional[str]
    ) -> AvailabilityStatus:
        """Map status string to AvailabilityStatus enum."""
        if not status_str:
            return AvailabilityStatus.AVAILABLE

        status_mapping = {
            "a": AvailabilityStatus.AVAILABLE,
            "d": AvailabilityStatus.DOUBTFUL,
            "i": AvailabilityStatus.INJURED,
            "s": AvailabilityStatus.SUSPENDED,
            "u": AvailabilityStatus.UNAVAILABLE,
            "n": AvailabilityStatus.UNKNOWN,
        }

        return status_mapping.get(status_str, AvailabilityStatus.UNKNOWN)

    def _create_validated_position_mapping(
        self, raw_positions: pd.DataFrame
    ) -> Dict[int, str]:
        """Create validated position mapping with Pydantic-style error checking."""
        if raw_positions.empty:
            if self.config.require_position_mapping:
                raise ValueError(
                    "Position mapping required but no position data available"
                )
            return {}

        # Validate position data structure
        required_columns = ["position_id", "singular_name_short"]
        missing_columns = [
            col for col in required_columns if col not in raw_positions.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Position data missing required columns: {missing_columns}"
            )

        # Create mapping with validation
        position_mapping = {}
        expected_positions = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}

        for _, row in raw_positions.iterrows():
            position_id = row.get("position_id")
            position_name = row.get("singular_name_short")

            # Validate position ID
            if (
                not isinstance(position_id, (int, float))
                or position_id < 1
                or position_id > 4
            ):
                raise ValueError(f"Invalid position_id: {position_id}")

            # Validate position name
            if not position_name or position_name not in expected_positions:
                raise ValueError(f"Invalid position name: {position_name}")

            # Validate consistency
            if expected_positions[position_name] != int(position_id):
                raise ValueError(
                    f"Position mapping inconsistency: {position_name} should have ID {expected_positions[position_name]}, got {position_id}"
                )

            position_mapping[int(position_id)] = position_name

        # Validate completeness
        if self.config.require_position_mapping and len(position_mapping) != 4:
            missing_positions = set(range(1, 5)) - set(position_mapping.keys())
            raise ValueError(
                f"Incomplete position mapping: missing position IDs {missing_positions}"
            )

        print(f"✅ Created validated position mapping: {position_mapping}")
        return position_mapping

    def get_enriched_players_dataframe(self) -> Result[pd.DataFrame]:
        """Get enriched players with comprehensive Pydantic validation at every step.

        This method demonstrates Pydantic's role in ensuring data integrity
        throughout the complex data transformation pipeline.
        """
        try:
            # Import database operations
            from db.operations import DatabaseOperations

            db_ops = DatabaseOperations()

            # Get raw data with validation expectations
            raw_players = db_ops.get_raw_players_bootstrap()
            raw_positions = db_ops.get_raw_element_types()

            # Validate data availability
            if raw_players.empty:
                return Result.failure(
                    DomainError.data_not_found("No player data available")
                )

            if raw_positions.empty and self.config.require_position_mapping:
                return Result.failure(
                    DomainError.data_not_found("No position data available")
                )

            # Validate minimum data expectations
            if len(raw_players) < self.config.min_expected_players:
                return Result.failure(
                    DomainError.validation_error(
                        f"Insufficient enriched player data: expected >={self.config.min_expected_players}, got {len(raw_players)}"
                    )
                )

            # Create validated position mapping with Pydantic validation
            position_mapping = self._create_validated_position_mapping(raw_positions)

            # Validate raw player data with Pydantic models
            validated_players = []
            validation_errors = []

            for _, row in raw_players.iterrows():
                try:
                    # Use Pydantic to validate comprehensive raw data
                    raw_enriched = RawEnrichedPlayerData(**row.to_dict())
                    validated_players.append(raw_enriched)
                except ValidationError as e:
                    validation_errors.append(
                        f"Player {row.get('player_id', 'unknown')}: {str(e)}"
                    )
                except Exception as e:
                    validation_errors.append(
                        f"Player {row.get('player_id', 'unknown')}: Unexpected error - {str(e)}"
                    )

            # Check validation error rate using configured threshold
            error_rate = len(validation_errors) / len(raw_players)
            if error_rate > self.config.max_validation_error_rate:
                return Result.failure(
                    DomainError.validation_error(
                        f"Too many validation errors: {len(validation_errors)}/{len(raw_players)} "
                        f"({error_rate:.1%} > {self.config.max_validation_error_rate:.1%}). "
                        f"Sample errors: {validation_errors[:3]}"
                    )
                )

            if validation_errors:
                print(
                    f"⚠️ {len(validation_errors)} players failed validation (within tolerance): {validation_errors[:3]}"
                )

            print(
                f"✅ Pydantic validated {len(validated_players)} enriched player records"
            )

            # Transform validated Pydantic models to DataFrame using type-safe conversion
            df_rows = []
            for player in validated_players:
                try:
                    # Map position ID to position name using validated mapping
                    position_name = position_mapping.get(player.position_id, "Unknown")
                    if (
                        position_name == "Unknown"
                        and self.config.require_position_mapping
                    ):
                        raise ValueError(
                            f"No position mapping for position_id {player.position_id}"
                        )

                    # Use Pydantic model methods for safe numeric conversion
                    row_data = {
                        "player_id": player.player_id,
                        "web_name": player.web_name,
                        "first": player.first_name,
                        "second": player.second_name,
                        "team_id": player.team_id,
                        "position": position_name,
                        "price": player.now_cost / 10.0,  # Convert to millions
                        "selected_by_percent": player.to_numeric("selected_by_percent"),
                        "status": player.status,
                        # Season performance stats using validated conversion
                        "total_points_season": player.to_int("total_points"),
                        "form_season": player.to_numeric("form"),
                        "points_per_game_season": player.to_numeric("points_per_game"),
                        "minutes": player.to_int("minutes"),
                        "starts": player.to_int("starts"),
                        # Match stats
                        "goals_scored": player.to_int("goals_scored"),
                        "assists": player.to_int("assists"),
                        "clean_sheets": player.to_int("clean_sheets"),
                        "goals_conceded": player.to_int("goals_conceded"),
                        "yellow_cards": player.to_int("yellow_cards"),
                        "red_cards": player.to_int("red_cards"),
                        "saves": player.to_int("saves"),
                        # Bonus and BPS
                        "bonus": player.to_int("bonus"),
                        "bps": player.to_int("bps"),
                        # ICT Index components
                        "influence": player.to_numeric("influence"),
                        "creativity": player.to_numeric("creativity"),
                        "threat": player.to_numeric("threat"),
                        "ict_index": player.to_numeric("ict_index"),
                        # Expected stats
                        "expected_goals": player.to_numeric("expected_goals"),
                        "expected_assists": player.to_numeric("expected_assists"),
                        "expected_goals_per_90": player.to_numeric(
                            "expected_goals_per_90"
                        ),
                        "expected_assists_per_90": player.to_numeric(
                            "expected_assists_per_90"
                        ),
                        # Market data
                        "value_form": player.to_numeric("value_form"),
                        "value_season": player.to_numeric("value_season"),
                        "transfers_in": player.to_int("transfers_in"),
                        "transfers_out": player.to_int("transfers_out"),
                        "transfers_in_event": player.to_int("transfers_in_event"),
                        "transfers_out_event": player.to_int("transfers_out_event"),
                        # Availability (nullable fields)
                        "chance_of_playing_this_round": player.to_numeric(
                            "chance_of_playing_this_round"
                        )
                        if player.chance_of_playing_this_round
                        else None,
                        "chance_of_playing_next_round": player.to_numeric(
                            "chance_of_playing_next_round"
                        )
                        if player.chance_of_playing_next_round
                        else None,
                        # Set pieces (nullable fields)
                        "penalties_order": player.to_int("penalties_order")
                        if player.penalties_order
                        else None,
                        "corners_and_indirect_freekicks_order": player.to_int(
                            "corners_and_indirect_freekicks_order"
                        )
                        if player.corners_and_indirect_freekicks_order
                        else None,
                        "news": player.news or "",
                        # Metadata
                        "as_of_utc": player.as_of_utc,
                    }
                    df_rows.append(row_data)

                except Exception as e:
                    validation_errors.append(
                        f"Player {player.player_id}: DataFrame conversion error - {str(e)}"
                    )

            # Create DataFrame from validated data
            if not df_rows:
                return Result.failure(
                    DomainError.validation_error(
                        "No valid player data after Pydantic validation"
                    )
                )

            enriched_players = pd.DataFrame(df_rows)

            # Final validation using configuration
            if len(enriched_players) < self.config.min_expected_players:
                return Result.failure(
                    DomainError.validation_error(
                        f"Invalid player data: expected >={self.config.min_expected_players} players, got {len(enriched_players)}"
                    )
                )

            if (
                self.config.require_position_mapping
                and enriched_players["position"].isna().sum() > 0
            ):
                return Result.failure(
                    DomainError.validation_error(
                        "Position mapping failed - some players have no position"
                    )
                )

            print(
                f"✅ Successfully created DataFrame with {len(enriched_players)} validated enriched players"
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

        This method merges data from:
        - get_enriched_players_dataframe() (47 enhanced fields)
        - get_derived_player_metrics() (29 derived analytics fields)

        Returns fully enriched EnrichedPlayerDomain objects with 70+ validated attributes.
        """
        try:
            from client import FPLDataClient

            # Get the enriched DataFrame first (reuse existing logic)
            df_result = self.get_enriched_players_dataframe()
            if df_result.is_failure:
                return Result.failure(df_result.error)

            enriched_df = df_result.value

            # Get derived metrics from dataset-builder
            client = FPLDataClient()
            derived_metrics = client.get_derived_player_metrics()

            if derived_metrics.empty:
                return Result.failure(
                    DomainError.data_not_found("No derived player metrics available")
                )

            # Merge enriched data with derived metrics on player_id
            merged_df = enriched_df.merge(
                derived_metrics, on="player_id", how="left", suffixes=("", "_derived")
            )

            # Convert DataFrame rows to validated Pydantic domain objects
            enriched_players = []
            validation_errors = []

            for _, row in merged_df.iterrows():
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
                        # EnrichedPlayerDomain enhanced fields
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
                        chance_of_playing_this_round=self._safe_optional_float(
                            row.get("chance_of_playing_this_round")
                        ),
                        chance_of_playing_next_round=self._safe_optional_float(
                            row.get("chance_of_playing_next_round")
                        ),
                        penalties_order=self._safe_optional_int(
                            row.get("penalties_order")
                        ),
                        corners_and_indirect_freekicks_order=self._safe_optional_int(
                            row.get("corners_and_indirect_freekicks_order")
                        ),
                        news=str(row.get("news", "")),
                        # Derived analytics metrics (29 fields from get_derived_player_metrics)
                        points_per_million=float(row.get("points_per_million", 0.0)),
                        form_per_million=float(row.get("form_per_million", 0.0)),
                        value_score=float(row.get("value_score", 0.0)),
                        value_confidence=float(row.get("value_confidence", 0.0)),
                        form_trend=str(row.get("form_trend", "stable")),
                        form_momentum=float(row.get("form_momentum", 0.0)),
                        recent_form_5gw=float(row.get("recent_form_5gw", 0.0)),
                        season_consistency=float(row.get("season_consistency", 0.0)),
                        expected_points_per_game=float(
                            row.get("expected_points_per_game", 0.0)
                        ),
                        points_above_expected=float(
                            row.get("points_above_expected", 0.0)
                        ),
                        overperformance_risk=float(
                            row.get("overperformance_risk", 0.0)
                        ),
                        ownership_trend=str(row.get("ownership_trend", "stable")),
                        transfer_momentum=float(row.get("transfer_momentum", 0.0)),
                        ownership_risk=float(row.get("ownership_risk", 0.0)),
                        set_piece_priority=float(row.get("set_piece_priority", 0.0)),
                        penalty_taker=bool(row.get("penalty_taker", False)),
                        corner_taker=bool(row.get("corner_taker", False)),
                        freekick_taker=bool(row.get("freekick_taker", False)),
                        injury_risk=float(row.get("injury_risk", 0.0)),
                        rotation_risk=float(row.get("rotation_risk", 0.0)),
                        data_quality_score=float(row.get("data_quality_score", 1.0)),
                    )
                    enriched_players.append(player)

                except Exception as e:
                    validation_errors.append(
                        f"Player {row.get('player_id', 'unknown')}: {str(e)}"
                    )

            # Check if we have too many validation errors
            if (
                validation_errors and len(validation_errors) > len(merged_df) * 0.1
            ):  # >10% failure rate
                return Result.failure(
                    DomainError.validation_error(
                        f"Too many validation errors ({len(validation_errors)}/{len(merged_df)}). "
                        f"Sample errors: {validation_errors[:3]}"
                    )
                )
            elif validation_errors:
                # Log warnings but continue if <10% failure rate
                print(
                    f"⚠️ {len(validation_errors)} players failed validation: {validation_errors[:3]}"
                )

            print(
                f"✅ Successfully created {len(enriched_players)} validated EnrichedPlayerDomain objects with full derived metrics"
            )
            return Result.success(enriched_players)

        except ImportError as e:
            return Result.failure(
                DomainError(
                    error_type=ErrorType.CONFIGURATION_ERROR,
                    message=f"Could not import FPL data client: {str(e)}",
                )
            )
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
