"""Transfer plan domain models for multi-gameweek planning."""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class StrategyMode(str, Enum):
    """Multi-GW planning strategy modes."""

    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    DGW_STACKER = "dgw_stacker"


class Transfer(BaseModel):
    """Single transfer between two players."""

    model_config = ConfigDict(use_enum_values=True)

    player_out_id: int = Field(..., gt=0, description="Player ID being transferred out")
    player_out_name: str = Field(..., min_length=1, description="Player name out")
    player_in_id: int = Field(..., gt=0, description="Player ID being transferred in")
    player_in_name: str = Field(..., min_length=1, description="Player name in")
    cost: float = Field(
        ..., description="Transfer cost (price diff, negative if downgrade)"
    )

    def __str__(self) -> str:
        """Human-readable transfer representation."""
        return f"{self.player_out_name} → {self.player_in_name} (£{self.cost:+.1f}m)"


class WeeklyTransferPlan(BaseModel):
    """Transfer plan for a single gameweek."""

    model_config = ConfigDict(use_enum_values=True)

    gameweek: int = Field(..., ge=1, le=38, description="Gameweek number")
    transfers: List[Transfer] = Field(
        default_factory=list, description="Transfers for this week (empty = hold FT)"
    )
    expected_xp: float = Field(..., ge=0, description="Expected points with transfers")
    baseline_xp: float = Field(
        ..., ge=0, description="Expected points without transfers"
    )
    xp_gain: float = Field(..., description="xP improvement from transfers")
    hit_cost: int = Field(
        ..., description="Points deduction from hits (0, -4, -8, etc)"
    )
    net_gain: float = Field(..., description="xP gain minus hit cost")
    reasoning: str = Field(..., min_length=1, description="Agent's reasoning")
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Agent's confidence level"
    )

    @property
    def num_transfers(self) -> int:
        """Number of transfers in this week."""
        return len(self.transfers)

    @property
    def is_hold(self) -> bool:
        """Whether this week is holding free transfers."""
        return len(self.transfers) == 0


class MultiGWPlan(BaseModel):
    """Complete multi-gameweek transfer plan from agent."""

    model_config = ConfigDict(use_enum_values=True)

    # Plan metadata
    start_gameweek: int = Field(..., ge=1, le=38, description="First gameweek")
    end_gameweek: int = Field(..., ge=1, le=38, description="Last gameweek")
    strategy_mode: StrategyMode = Field(..., description="Strategy used")

    # Weekly breakdown
    weekly_plans: List[WeeklyTransferPlan] = Field(
        ..., min_length=1, description="Week-by-week transfer plans"
    )

    # Summary metrics
    total_xp_gain: float = Field(..., description="Total xP improvement over horizon")
    total_hit_cost: int = Field(..., description="Total points from hits")
    net_roi: float = Field(..., description="Net points (xP gain - hits)")

    # Agent reasoning
    opportunities_identified: List[str] = Field(
        default_factory=list,
        description="Key opportunities found (DGW, fixtures, etc)",
    )
    constraints_considered: List[str] = Field(
        default_factory=list, description="Constraints applied (budget, FT, etc)"
    )
    trade_offs: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Options considered with pros/cons",
    )
    final_reasoning: str = Field(
        ..., min_length=1, description="Overall strategic reasoning"
    )

    # Risk analysis
    best_case_roi: float = Field(..., description="Optimistic scenario (xP + 2σ)")
    worst_case_roi: float = Field(..., description="Pessimistic scenario (xP - 2σ)")
    template_comparison: Dict[str, Any] = Field(
        default_factory=dict, description="Comparison with template picks"
    )

    # Optional chip recommendations (Phase 4)
    chip_timing: Optional[Dict[str, Any]] = Field(
        None, description="Chip recommendations if consider_chips=True"
    )

    @property
    def planning_horizon(self) -> int:
        """Number of gameweeks in planning horizon."""
        return self.end_gameweek - self.start_gameweek + 1

    @property
    def total_transfers(self) -> int:
        """Total number of transfers across all weeks."""
        return sum(week.num_transfers for week in self.weekly_plans)


class AgentState(BaseModel):
    """Stateful context maintained during agent execution (for internal use)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core planning parameters
    current_gameweek: int = Field(..., ge=1, le=38)
    planning_horizon: int = Field(..., ge=1, le=10)
    current_squad: List[int] = Field(..., min_length=15, max_length=15)
    budget: float = Field(..., ge=0.0, le=100.0)
    free_transfers: int = Field(..., ge=0, le=15)

    # Cached intermediate results (populated by tools)
    xp_predictions: Optional[Any] = Field(
        None, description="Cached xP predictions DataFrame"
    )
    dgw_bgw_info: Optional[Dict[str, Any]] = Field(
        None, description="Cached DGW/BGW detection results"
    )
    fixture_swings: Optional[List[Dict[str, Any]]] = Field(
        None, description="Cached fixture difficulty changes"
    )

    # User constraints
    must_include_ids: List[int] = Field(
        default_factory=list, description="Players that must stay"
    )
    must_exclude_ids: List[int] = Field(
        default_factory=list, description="Players to avoid"
    )
    hit_roi_threshold: float = Field(
        default=5.0, ge=3.0, le=10.0, description="Min xP gain for -4 hit"
    )
