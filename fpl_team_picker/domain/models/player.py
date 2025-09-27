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
