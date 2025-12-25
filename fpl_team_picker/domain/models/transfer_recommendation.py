"""Transfer recommendation domain models for single-gameweek planning with multi-GW context."""

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

from .transfer_plan import Transfer  # Reuse existing Transfer model


class TransferScenario(BaseModel):
    """Single transfer scenario with multi-GW analysis."""

    model_config = ConfigDict(use_enum_values=True)

    # Identity
    scenario_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier (e.g., 'option_1', 'option_2')",
    )
    transfers: List[Transfer] = Field(
        default_factory=list,
        description="Transfers for this scenario (empty = hold)",
    )

    # Single-GW metrics (immediate impact)
    xp_gw1: float = Field(..., ge=0, description="Expected points for GW N")
    xp_gain_gw1: float = Field(..., description="xP improvement vs baseline (GW N)")
    hit_cost: int = Field(0, description="Points deduction from hits (0, -4, -8)")
    net_gain_gw1: float = Field(..., description="xP gain minus hit cost (GW N)")

    # Multi-GW context (3 GW horizon)
    xp_3gw: float = Field(..., ge=0, description="Total xP over next 3 gameweeks")
    xp_gain_3gw: float = Field(..., description="3GW xP improvement vs baseline")
    net_roi_3gw: float = Field(..., description="3GW net gain (xP gain - hits)")

    # Strategic reasoning
    reasoning: str = Field(
        ..., min_length=1, description="Why this scenario is recommended"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence level"
    )

    # Context flags (make agent reasoning explicit)
    leverages_dgw: bool = Field(False, description="Targets DGW opportunity")
    leverages_fixture_swing: bool = Field(
        False, description="Exploits fixture difficulty change"
    )
    prepares_for_chip: bool = Field(False, description="Positions for upcoming chip")

    # SA validation
    sa_validated: bool = Field(
        False, description="Whether SA optimizer validated this scenario"
    )
    sa_deviation: float | None = Field(
        None, description="xP difference vs SA solution (+/- xP)"
    )

    @property
    def num_transfers(self) -> int:
        """Number of transfers in this scenario."""
        return len(self.transfers)

    @property
    def is_hold(self) -> bool:
        """Whether this scenario holds free transfers."""
        return len(self.transfers) == 0


class HoldOption(BaseModel):
    """Baseline no-transfer scenario."""

    model_config = ConfigDict(use_enum_values=True)

    xp_gw1: float = Field(..., ge=0, description="Expected points if holding FT")
    xp_3gw: float = Field(..., ge=0, description="Total xP over 3 GW if holding")
    free_transfers_next_week: int = Field(
        ..., ge=1, le=2, description="FTs banked for next week (1 or 2)"
    )
    reasoning: str = Field(
        ..., min_length=1, description="Strategic value of banking FT"
    )


class SingleGWRecommendation(BaseModel):
    """Complete single-GW recommendation with ranked options."""

    model_config = ConfigDict(use_enum_values=True)

    # Metadata
    target_gameweek: int = Field(..., ge=1, le=38, description="Target gameweek")
    current_free_transfers: int = Field(..., ge=0, le=15, description="Available FTs")
    budget_available: float = Field(..., ge=0, description="Budget in the bank (£m)")

    # Hold option (baseline)
    hold_option: HoldOption = Field(..., description="Baseline no-transfer scenario")

    # Recommended scenarios (ranked by net_roi_3gw)
    recommended_scenarios: List[TransferScenario] = Field(
        ..., min_length=3, max_length=5, description="Top 3-5 transfer scenarios"
    )

    # Multi-GW context
    context_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategic context (DGWs, fixture swings, chip timing)",
    )

    # SA benchmarking
    sa_benchmark: Dict[str, Any] = Field(
        default_factory=dict,
        description="SA optimizer results for comparison",
    )

    # Final recommendation
    top_recommendation_id: str = Field(
        ..., description="ID of top-ranked scenario (or 'hold')"
    )
    final_reasoning: str = Field(
        ..., min_length=1, description="Overall strategic recommendation"
    )

    @property
    def best_scenario(self) -> TransferScenario | HoldOption:
        """Get the top-ranked scenario."""
        if self.top_recommendation_id == "hold":
            return self.hold_option
        for scenario in self.recommended_scenarios:
            if scenario.scenario_id == self.top_recommendation_id:
                return scenario
        # Fallback to first scenario
        return self.recommended_scenarios[0]

    @property
    def total_scenarios(self) -> int:
        """Total number of scenarios including hold."""
        return len(self.recommended_scenarios) + 1
