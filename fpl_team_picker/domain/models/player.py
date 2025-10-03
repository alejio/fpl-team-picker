"""Player domain model with strict FPL validation."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Position(str, Enum):
    """FPL player positions."""

    GKP = "GKP"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"


class AvailabilityStatus(str, Enum):
    """Player availability status."""

    AVAILABLE = "a"
    DOUBTFUL = "d"
    INJURED = "i"
    SUSPENDED = "s"
    UNAVAILABLE = "u"
    UNKNOWN = "n"  # Sometimes appears in FPL data


class PlayerDomain(BaseModel):
    """
    Domain model for FPL players with strict validation.

    Enforces FPL rules and data quality at the domain level.
    """

    player_id: int = Field(..., gt=0, description="Unique FPL player ID")
    web_name: str = Field(..., min_length=1, max_length=50, description="Display name")
    first_name: Optional[str] = Field(None, max_length=50, description="First name")
    last_name: Optional[str] = Field(None, max_length=50, description="Last name")
    team_id: int = Field(
        ..., ge=1, le=20, description="Team ID (1-20 for Premier League)"
    )
    position: Position = Field(..., description="Player position")
    price: float = Field(..., ge=3.9, le=15.0, description="Price in millions")
    selected_by_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Selection percentage"
    )
    availability_status: AvailabilityStatus = Field(
        default=AvailabilityStatus.AVAILABLE, description="Availability status"
    )
    as_of_utc: datetime = Field(..., description="Data timestamp")
    mapped_player_id: Optional[int] = Field(
        None, description="Mapped player ID for historical data"
    )

    @field_validator("price")
    @classmethod
    def validate_price_increment(cls, v: float) -> float:
        """FPL prices must be in 0.1m increments."""
        if round(v * 10) != v * 10:
            raise ValueError("Price must be in 0.1m increments")
        return v

    @field_validator("web_name", "first_name", "last_name")
    @classmethod
    def validate_name_fields(cls, v: Optional[str]) -> Optional[str]:
        """Ensure name fields are properly trimmed and non-empty."""
        if v is None:
            return None
        trimmed = v.strip()
        if not trimmed:
            return None
        return trimmed

    @property
    def full_name(self) -> str:
        """Generate full name from first and last name, fallback to web_name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}".strip()
        return self.web_name

    @property
    def is_available(self) -> bool:
        """Check if player is available for selection."""
        return self.availability_status == AvailabilityStatus.AVAILABLE

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}
        use_enum_values = True


class LiveDataDomain(BaseModel):
    """Live performance data for a player in a specific gameweek."""

    player_id: int = Field(..., gt=0)
    gameweek: int = Field(..., ge=1, le=38)
    minutes: int = Field(..., ge=0, le=120)
    total_points: int = Field(...)
    goals_scored: int = Field(..., ge=0)
    assists: int = Field(..., ge=0)
    clean_sheets: int = Field(..., ge=0, le=1)
    goals_conceded: int = Field(..., ge=0)
    yellow_cards: int = Field(..., ge=0)
    red_cards: int = Field(..., ge=0, le=1)
    saves: int = Field(..., ge=0)
    bonus: int = Field(..., ge=0, le=3)
    bps: int = Field(..., ge=0)
    influence: float = Field(..., ge=0.0)
    creativity: float = Field(..., ge=0.0)
    threat: float = Field(..., ge=0.0)
    ict_index: float = Field(..., ge=0.0)
    expected_goals: Optional[float] = Field(None, ge=0.0)
    expected_assists: Optional[float] = Field(None, ge=0.0)
    expected_goal_involvements: Optional[float] = Field(None, ge=0.0)
    expected_goals_conceded: Optional[float] = Field(None, ge=0.0)
    value: float = Field(
        ..., ge=3.9, le=15.0, description="Player price at time of gameweek"
    )
    was_home: bool = Field(...)
    opponent_team: int = Field(..., ge=1, le=20)

    @field_validator("value")
    @classmethod
    def validate_value_increment(cls, v: float) -> float:
        """FPL prices must be in 0.1m increments."""
        if round(v * 10) != v * 10:
            raise ValueError("Value must be in 0.1m increments")
        return v


class EnrichedPlayerDomain(PlayerDomain):
    """
    Enhanced player domain model with complete season statistics.

    Extends base PlayerDomain with all available FPL statistics
    for comprehensive analysis and decision making.
    """

    # Season Performance Stats (beyond basic PlayerDomain)
    total_points_season: int = Field(ge=0, description="Total FPL points this season")
    form_season: float = Field(ge=0.0, description="Recent form score")
    points_per_game_season: float = Field(ge=0.0, description="Average points per game")
    minutes: int = Field(ge=0, description="Total minutes played")
    starts: int = Field(ge=0, description="Number of starts")

    # Match Statistics
    goals_scored: int = Field(ge=0, description="Goals scored this season")
    assists: int = Field(ge=0, description="Assists this season")
    clean_sheets: int = Field(ge=0, description="Clean sheets (for DEF/GKP)")
    goals_conceded: int = Field(ge=0, description="Goals conceded (for DEF/GKP)")
    yellow_cards: int = Field(ge=0, description="Yellow cards received")
    red_cards: int = Field(ge=0, description="Red cards received")
    saves: int = Field(ge=0, description="Saves (for GKP)")

    # Bonus Points System
    bonus: int = Field(ge=0, description="Bonus points earned")
    bps: int = Field(ge=0, description="Bonus points system score")

    # ICT Index Components (0-200 scale typically)
    influence: float = Field(ge=0.0, description="Influence score")
    creativity: float = Field(ge=0.0, description="Creativity score")
    threat: float = Field(ge=0.0, description="Threat score")
    ict_index: float = Field(ge=0.0, description="Combined ICT index")

    # Expected Statistics
    expected_goals: float = Field(ge=0.0, description="Expected goals (xG)")
    expected_assists: float = Field(ge=0.0, description="Expected assists (xA)")
    expected_goals_per_90: float = Field(
        ge=0.0, description="Expected goals per 90 minutes"
    )
    expected_assists_per_90: float = Field(
        ge=0.0, description="Expected assists per 90 minutes"
    )

    # Market Data
    value_form: float = Field(ge=0.0, description="Value based on recent form")
    value_season: float = Field(ge=0.0, description="Value based on season performance")
    transfers_in: int = Field(ge=0, description="Total transfers in")
    transfers_out: int = Field(ge=0, description="Total transfers out")
    transfers_in_event: int = Field(ge=0, description="Transfers in this gameweek")
    transfers_out_event: int = Field(ge=0, description="Transfers out this gameweek")

    # Availability Information
    chance_of_playing_this_round: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Chance of playing this round (%)"
    )
    chance_of_playing_next_round: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Chance of playing next round (%)"
    )

    # Set Piece Responsibilities
    penalties_order: Optional[int] = Field(
        None, ge=1, le=5, description="Penalty taking order (1=first choice)"
    )
    corners_and_indirect_freekicks_order: Optional[int] = Field(
        None, ge=1, le=5, description="Corner/free kick taking order"
    )

    # News and Updates
    news: str = Field(default="", description="Latest injury/availability news")

    @field_validator("total_points_season", "goals_scored", "assists")
    @classmethod
    def validate_non_negative_stats(cls, v: int) -> int:
        """Ensure key stats are non-negative."""
        if v < 0:
            raise ValueError(f"Stat cannot be negative: {v}")
        return v

    @field_validator("chance_of_playing_this_round", "chance_of_playing_next_round")
    @classmethod
    def validate_percentage_or_none(cls, v: Optional[float]) -> Optional[float]:
        """Validate percentage values or None."""
        if v is not None and not (0.0 <= v <= 100.0):
            raise ValueError(f"Percentage must be between 0-100 or None: {v}")
        return v

    @property
    def goals_per_90(self) -> float:
        """Calculate goals per 90 minutes."""
        if self.minutes == 0:
            return 0.0
        return (self.goals_scored * 90) / self.minutes

    @property
    def assists_per_90(self) -> float:
        """Calculate assists per 90 minutes."""
        if self.minutes == 0:
            return 0.0
        return (self.assists * 90) / self.minutes

    @property
    def is_penalty_taker(self) -> bool:
        """Check if player is likely penalty taker."""
        return self.penalties_order is not None and self.penalties_order <= 2

    @property
    def is_set_piece_taker(self) -> bool:
        """Check if player takes set pieces."""
        return (self.penalties_order is not None and self.penalties_order <= 3) or (
            self.corners_and_indirect_freekicks_order is not None
            and self.corners_and_indirect_freekicks_order <= 3
        )

    class Config:
        """Pydantic configuration for enriched player data."""

        str_strip_whitespace = True
        validate_assignment = True
        extra = "forbid"  # Prevent extra fields for data integrity
        json_encoders = {datetime: lambda v: v.isoformat()}
        use_enum_values = True
