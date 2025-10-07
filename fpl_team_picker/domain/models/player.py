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
    form_season: float = Field(description="Recent form score (can be negative)")
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
    bps: int = Field(description="Bonus points system score (can be negative)")

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
    value_form: float = Field(
        description="Value based on recent form (can be negative)"
    )
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
        None, ge=1, description="Penalty taking order (1=first choice)"
    )
    corners_and_indirect_freekicks_order: Optional[int] = Field(
        None, ge=1, description="Corner/free kick taking order"
    )

    # News and Updates
    news: str = Field(default="", description="Latest injury/availability news")

    # Derived Analytics Metrics (from dataset-builder get_derived_player_metrics)
    points_per_million: float = Field(ge=0.0, description="Total points per £1m price")
    form_per_million: float = Field(ge=0.0, description="Form score per £1m price")
    value_score: float = Field(
        ge=0.0, le=100.0, description="Overall value rating (0-100)"
    )
    value_confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in value score (0-1)"
    )
    form_trend: str = Field(description="Form trend: improving/stable/declining")
    form_momentum: float = Field(
        ge=-1.0, le=1.0, description="Form momentum indicator (-1 to +1)"
    )
    recent_form_5gw: float = Field(ge=0.0, description="Average form over last 5 GWs")
    season_consistency: float = Field(
        ge=0.0, le=1.0, description="Performance consistency score (0-1)"
    )
    expected_points_per_game: float = Field(
        ge=0.0, description="Expected points per game based on xG/xA"
    )
    points_above_expected: float = Field(
        description="Actual points minus expected points (overperformance)"
    )
    overperformance_risk: float = Field(
        ge=0.0, le=1.0, description="Risk of regression to mean (0-1)"
    )
    ownership_trend: str = Field(description="Ownership trend: rising/stable/falling")
    transfer_momentum: float = Field(
        description="Net transfer velocity (transfers_in - transfers_out)"
    )
    ownership_risk: float = Field(
        ge=0.0, le=1.0, description="Risk from ownership dynamics (0-1)"
    )
    set_piece_priority: float = Field(
        ge=0.0, description="Overall set piece priority score"
    )
    penalty_taker: bool = Field(description="First choice penalty taker flag")
    corner_taker: bool = Field(description="Primary corner taker flag")
    freekick_taker: bool = Field(description="Primary free kick taker flag")
    injury_risk: float = Field(
        ge=0.0, le=1.0, description="Injury probability based on history (0-1)"
    )
    rotation_risk: float = Field(
        ge=0.0, le=1.0, description="Risk of rotation/benching (0-1)"
    )
    data_quality_score: float = Field(
        ge=0.0, le=1.0, description="Quality of underlying data for metrics (0-1)"
    )

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

    @property
    def is_high_value(self) -> bool:
        """Check if player is high value (score >= 80)."""
        return self.value_score >= 80.0

    @property
    def has_injury_concern(self) -> bool:
        """Check if player has injury concern (risk > 50%)."""
        return self.injury_risk > 0.5

    @property
    def has_rotation_concern(self) -> bool:
        """Check if player has rotation concern (risk > 50%)."""
        return self.rotation_risk > 0.5

    @property
    def is_form_improving(self) -> bool:
        """Check if player's form is improving."""
        return self.form_trend.lower() == "improving"

    @property
    def is_form_declining(self) -> bool:
        """Check if player's form is declining."""
        return self.form_trend.lower() == "declining"

    @property
    def is_ownership_rising(self) -> bool:
        """Check if player's ownership is rising."""
        return self.ownership_trend.lower() == "rising"

    @property
    def has_overperformance_risk(self) -> bool:
        """Check if player has high overperformance risk (>= 0.7)."""
        return self.overperformance_risk >= 0.7

    @property
    def is_reliable_data(self) -> bool:
        """Check if player has reliable data quality (>= 0.7)."""
        return self.data_quality_score >= 0.7

    # Position helpers
    @property
    def is_goalkeeper(self) -> bool:
        """Is a goalkeeper."""
        return self.position == Position.GKP

    @property
    def is_defender(self) -> bool:
        """Is a defender."""
        return self.position == Position.DEF

    @property
    def is_midfielder(self) -> bool:
        """Is a midfielder."""
        return self.position == Position.MID

    @property
    def is_forward(self) -> bool:
        """Is a forward."""
        return self.position == Position.FWD

    # Price brackets
    @property
    def is_budget_enabler(self) -> bool:
        """Cheap player (<= £5.0m) enabling premium purchases."""
        return self.price <= 5.0

    @property
    def is_premium(self) -> bool:
        """Premium player (>= £10.0m)."""
        return self.price >= 10.0

    @property
    def is_mid_price(self) -> bool:
        """Mid-price player (£6.5m - £9.5m)."""
        return 6.5 <= self.price <= 9.5

    # Ownership indicators
    @property
    def is_differential(self) -> bool:
        """Low ownership (< 5%) potential differential pick."""
        return self.selected_by_percent < 5.0

    @property
    def is_template(self) -> bool:
        """High ownership (> 40%) template player."""
        return self.selected_by_percent > 40.0

    # Performance per 90
    @property
    def goal_involvement_per_90(self) -> float:
        """Goals + Assists per 90 minutes."""
        if self.minutes == 0:
            return 0.0
        return ((self.goals_scored + self.assists) * 90) / self.minutes

    @property
    def expected_involvement_per_90(self) -> float:
        """xG + xA per 90 (expected goal involvements)."""
        return self.expected_goals_per_90 + self.expected_assists_per_90

    @property
    def bonus_per_90(self) -> float:
        """Bonus points per 90 minutes."""
        if self.minutes == 0:
            return 0.0
        return (self.bonus * 90) / self.minutes

    # Performance analysis
    @property
    def is_outperforming_xg(self) -> bool:
        """Scoring significantly more than expected (overperforming)."""
        return self.points_above_expected > 5.0

    @property
    def is_underperforming_xg(self) -> bool:
        """Scoring significantly less than expected (underperforming)."""
        return self.points_above_expected < -5.0

    # Risk aggregations
    @property
    def has_any_concern(self) -> bool:
        """Any risk flag raised (injury, rotation, overperformance)."""
        return (
            self.has_injury_concern
            or self.has_rotation_concern
            or self.has_overperformance_risk
        )

    @property
    def is_safe_pick(self) -> bool:
        """Low risk profile (no concerns, good data quality, available)."""
        return not self.has_any_concern and self.is_reliable_data and self.is_available

    # Trend indicators
    @property
    def is_rising_star(self) -> bool:
        """Improving form + rising ownership."""
        return self.is_form_improving and self.is_ownership_rising

    @property
    def is_falling_knife(self) -> bool:
        """Declining form + falling ownership - avoid!"""
        return self.is_form_declining and self.ownership_trend.lower() == "falling"

    @property
    def is_under_the_radar(self) -> bool:
        """Good value + low ownership + improving form."""
        return self.is_high_value and self.is_differential and self.is_form_improving

    @property
    def is_attacking_defender(self) -> bool:
        """Defender with attacking returns (goals + assists >= 3)."""
        return self.position == Position.DEF and (self.goals_scored + self.assists) >= 3

    class Config:
        """Pydantic configuration for enriched player data."""

        str_strip_whitespace = True
        validate_assignment = True
        extra = "forbid"  # Prevent extra fields for data integrity
        json_encoders = {datetime: lambda v: v.isoformat()}
        use_enum_values = True
