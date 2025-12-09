"""
Global Configuration System for FPL Team Picker

Centralized configuration management for all hardcoded values across modules.
Provides type-safe configuration with validation and environment variable support.
"""

import os
from typing import Dict, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator


class XPModelConfig(BaseModel):
    """Expected Points Model Configuration"""

    # Model selection
    use_ml_model: bool = Field(
        default=True, description="Use ML model instead of rule-based model"
    )
    ml_model_path: str = Field(
        default="models/hybrid/hybrid_gw1-15_20251209_195913.joblib",
        description="Trained using scripts/train_model.py full-pipeline --end-gw 14 --holdout-gws 2 --scorer fpl_hauler_ceiling --n-trials 75",
    )
    ml_ensemble_rule_weight: float = Field(
        default=0.0,
        description="Weight for rule-based in ML ensemble (0=pure ML, 1=pure rule-based). Default 0.0 for pure ML predictions.",
        ge=0.0,
        le=1.0,
    )

    # Form weighting
    form_weight: float = Field(
        default=0.7, description="70% recent form, 30% season baseline", ge=0.0, le=1.0
    )
    form_window: int = Field(
        default=5, description="5 gameweek form window", ge=1, le=10
    )
    debug: bool = Field(default=True, description="Enable debug logging")

    # ML Model Configuration
    ml_min_training_gameweeks: int = Field(
        default=3, description="Minimum gameweeks for ML training", ge=2, le=10
    )
    ml_training_gameweeks: int = Field(
        default=5, description="Number of recent gameweeks for ML training", ge=3, le=15
    )
    ml_position_min_samples: int = Field(
        default=30,
        description="Minimum samples for position-specific models",
        ge=10,
        le=100,
    )


class TeamStrengthConfig(BaseModel):
    """Dynamic Team Strength Configuration"""

    # Transition gameweek
    historical_transition_gw: int = Field(
        default=8, description="GW8+ = current season only", ge=1, le=38
    )
    rolling_window_size: int = Field(
        default=6, description="Games for current season rolling average", ge=1, le=15
    )
    debug: bool = Field(default=False, description="Enable debug logging")

    # Strength scaling
    max_strength: float = Field(
        default=1.3, description="Maximum team strength", ge=1.0, le=2.0
    )
    min_strength: float = Field(
        default=0.7, description="Minimum team strength", ge=0.1, le=1.0
    )
    fallback_strength: float = Field(
        default=1.0, description="Fallback strength for missing data", ge=0.5, le=1.5
    )

    # Team name mappings for lookups
    team_name_mappings: Dict[str, str] = Field(
        default_factory=lambda: {
            "Man City": "Manchester City",
            "Man Utd": "Manchester United",
            "Nott'm Forest": "Nottingham Forest",
            "Spurs": "Tottenham",
        },
        description="Team name mappings for data consistency",
    )

    @field_validator("max_strength", "min_strength")
    @classmethod
    def validate_strength_range(cls, v, info):
        if (
            info.field_name == "max_strength"
            and hasattr(info, "data")
            and "min_strength" in info.data
        ):
            if v <= info.data["min_strength"]:
                raise ValueError("max_strength must be greater than min_strength")
        return v


class MinutesModelConfig(BaseModel):
    """Enhanced Minutes Prediction Configuration"""

    # EWMA (Exponential Weighted Moving Average) parameters
    use_ewma: bool = Field(
        default=True, description="Use EWMA for expected minutes calculation"
    )
    ewma_span: int = Field(
        default=5, description="EWMA span (lookback window in games)", ge=2, le=10
    )
    ewma_min_games: int = Field(
        default=1,
        description="Minimum games required to use EWMA (otherwise fallback to position default)",
        ge=1,
        le=5,
    )

    # Selected By Percentage (SBP) thresholds
    sbp_very_high_threshold: float = Field(
        default=40.0, description=">= 40% ownership", ge=0.0, le=100.0
    )
    sbp_high_threshold: float = Field(
        default=20.0, description=">= 20% ownership", ge=0.0, le=100.0
    )
    sbp_medium_threshold: float = Field(
        default=10.0, description=">= 10% ownership", ge=0.0, le=100.0
    )
    sbp_low_threshold: float = Field(
        default=5.0, description=">= 5% ownership", ge=0.0, le=100.0
    )
    sbp_very_low_threshold: float = Field(
        default=2.0, description=">= 2% ownership", ge=0.0, le=100.0
    )
    sbp_minimal_threshold: float = Field(
        default=0.5, description=">= 0.5% ownership", ge=0.0, le=100.0
    )

    # Price-based durability (GKP vs outfield)
    gkp_avg_minutes: float = Field(
        default=90.0, description="GKP average minutes", ge=0.0, le=90.0
    )
    gkp_full_game_prob: float = Field(
        default=0.95, description="GKP full game probability", ge=0.0, le=1.0
    )
    gkp_sub_prob: float = Field(
        default=0.05, description="GKP substitute probability", ge=0.0, le=1.0
    )

    # Premium outfield (>= £7.0m)
    premium_price_threshold: float = Field(
        default=7.0, description="Premium player price threshold", ge=4.0, le=15.0
    )
    premium_avg_minutes: float = Field(
        default=85.0, description="Premium average minutes", ge=0.0, le=90.0
    )
    premium_full_game_prob: float = Field(
        default=0.80, description="Premium full game probability", ge=0.0, le=1.0
    )
    premium_sub_prob: float = Field(
        default=0.25, description="Premium substitute probability", ge=0.0, le=1.0
    )

    # Mid-tier outfield (>= £5.0m)
    mid_tier_price_threshold: float = Field(
        default=5.0, description="Mid-tier price threshold", ge=4.0, le=10.0
    )
    mid_tier_avg_minutes: float = Field(
        default=75.0, description="Mid-tier average minutes", ge=0.0, le=90.0
    )
    mid_tier_full_game_prob: float = Field(
        default=0.70, description="Mid-tier full game probability", ge=0.0, le=1.0
    )
    mid_tier_sub_prob: float = Field(
        default=0.30, description="Mid-tier substitute probability", ge=0.0, le=1.0
    )

    # Budget outfield (< £5.0m)
    budget_avg_minutes: float = Field(
        default=70.0, description="Budget average minutes", ge=0.0, le=90.0
    )
    budget_full_game_prob: float = Field(
        default=0.60, description="Budget full game probability", ge=0.0, le=1.0
    )
    budget_sub_prob: float = Field(
        default=0.35, description="Budget substitute probability", ge=0.0, le=1.0
    )

    # Injury/availability adjustments
    injury_full_game_multiplier: float = Field(
        default=0.7, description="Injury full game multiplier", ge=0.0, le=1.0
    )
    injury_avg_minutes_multiplier: float = Field(
        default=0.8, description="Injury average minutes multiplier", ge=0.0, le=1.0
    )


class StatisticalEstimationConfig(BaseModel):
    """Statistical xG/xA Estimation Configuration"""

    # xG90 estimation multipliers by position and price
    premium_price_threshold: float = Field(
        default=8.0, description="Premium player price threshold", ge=4.0, le=15.0
    )
    mid_tier_price_threshold: float = Field(
        default=6.0, description="Mid-tier price threshold", ge=4.0, le=10.0
    )
    budget_price_threshold: float = Field(
        default=4.5, description="Budget price threshold", ge=4.0, le=8.0
    )
    min_price_threshold: float = Field(
        default=4.0, description="Minimum price threshold", ge=4.0, le=6.0
    )

    # xG90 price multipliers (goals)
    xg_premium_base_multiplier: float = Field(
        default=1.4, description="xG premium base multiplier", ge=0.5, le=3.0
    )
    xg_premium_scale_factor: float = Field(
        default=0.15, description="xG premium scale factor", ge=0.0, le=1.0
    )
    xg_mid_tier_base_multiplier: float = Field(
        default=1.1, description="xG mid-tier base multiplier", ge=0.5, le=3.0
    )
    xg_mid_tier_scale_factor: float = Field(
        default=0.15, description="xG mid-tier scale factor", ge=0.0, le=1.0
    )
    xg_budget_base_multiplier: float = Field(
        default=0.8, description="xG budget base multiplier", ge=0.1, le=2.0
    )
    xg_budget_scale_factor: float = Field(
        default=0.2, description="xG budget scale factor", ge=0.0, le=1.0
    )
    xg_min_base_multiplier: float = Field(
        default=0.5, description="xG minimum base multiplier", ge=0.1, le=2.0
    )
    xg_min_scale_factor: float = Field(
        default=0.6, description="xG minimum scale factor", ge=0.0, le=2.0
    )

    # xA90 price multipliers (assists)
    xa_premium_price_threshold: float = Field(
        default=7.5, description="xA premium price threshold", ge=4.0, le=15.0
    )
    xa_mid_tier_price_threshold: float = Field(
        default=5.5, description="xA mid-tier price threshold", ge=4.0, le=10.0
    )
    xa_premium_base_multiplier: float = Field(
        default=1.5, description="xA premium base multiplier", ge=0.5, le=3.0
    )
    xa_premium_scale_factor: float = Field(
        default=0.2, description="xA premium scale factor", ge=0.0, le=1.0
    )
    xa_mid_tier_base_multiplier: float = Field(
        default=1.0, description="xA mid-tier base multiplier", ge=0.5, le=3.0
    )
    xa_mid_tier_scale_factor: float = Field(
        default=0.25, description="xA mid-tier scale factor", ge=0.0, le=1.0
    )
    xa_budget_base_multiplier: float = Field(
        default=0.6, description="xA budget base multiplier", ge=0.1, le=2.0
    )
    xa_budget_scale_factor: float = Field(
        default=0.25, description="xA budget scale factor", ge=0.0, le=1.0
    )

    # Team strength multipliers
    xg_team_strength_factor: float = Field(
        default=0.8, description="Goals influence by team strength", ge=0.0, le=2.0
    )
    xa_team_strength_factor: float = Field(
        default=1.0, description="Assists influence by team strength", ge=0.0, le=2.0
    )

    # SBP (popularity) influence
    xg_sbp_influence: float = Field(
        default=0.3, description="SBP influence on xG", ge=0.0, le=1.0
    )
    xa_sbp_influence: float = Field(
        default=0.25, description="SBP influence on xA", ge=0.0, le=1.0
    )

    # ICT index factors
    threat_factor_influence: float = Field(
        default=0.4, description="Threat factor influence on xG", ge=0.0, le=2.0
    )
    ict_factor_influence: float = Field(
        default=0.2, description="ICT factor general influence", ge=0.0, le=2.0
    )
    creativity_factor_influence: float = Field(
        default=0.5, description="Creativity factor influence on xA", ge=0.0, le=2.0
    )
    influence_factor_influence: float = Field(
        default=0.3, description="Influence factor for assists", ge=0.0, le=2.0
    )


class FixtureDifficultyConfig(BaseModel):
    """Fixture Difficulty Analysis Configuration"""

    # Difficulty multiplier calculation
    base_difficulty_multiplier: float = Field(
        default=2.2,
        description="Base calculation: 2.2 - opponent_strength (increased from 2.0 for better differentiation)",
        ge=1.0,
        le=5.0,
    )

    # Difficulty classification thresholds (calibrated for relative scaling: 0.0-2.2 range)
    # Based on 25th/75th percentiles of GW11 distribution with relative scaling
    easy_fixture_threshold: float = Field(
        default=1.538,
        description=">= 1.538 = Easy (🟢) - top 25% of fixtures",
        ge=1.0,
        le=2.2,
    )
    average_fixture_min: float = Field(
        default=0.791,
        description="0.791-1.538 = Average (🟡), <= 0.791 = Hard (🔴)",
        ge=0.0,
        le=1.5,
    )
    # Relative scaling produces 0.0-2.2 range, so thresholds adjusted accordingly

    # Temporal weighting for multi-gameweek
    temporal_weights: List[float] = Field(
        default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.6],
        description="GW1=1.0, GW2=0.9, GW3=0.8, GW4=0.7, GW5=0.6",
    )

    @field_validator("temporal_weights")
    @classmethod
    def validate_temporal_weights(cls, v):
        if not v or len(v) < 1:
            raise ValueError("temporal_weights must have at least one value")
        if any(w < 0 or w > 1 for w in v):
            raise ValueError("temporal_weights values must be between 0 and 1")
        return v


class XPCalibrationConfig(BaseModel):
    """Simple Probabilistic XP Calibration Configuration"""

    enabled: bool = Field(
        default=True,
        description="Enable probabilistic calibration of ML predictions",
    )

    # Simplified categorization (2-tier fixture only)
    premium_price_threshold: float = Field(
        default=8.0,
        description="Price threshold for premium tier (>= this price = premium)",
        ge=4.0,
        le=15.0,
    )
    mid_price_threshold: float = Field(
        default=6.0,
        description="Price threshold for mid tier (>= this price = mid, < premium)",
        ge=4.0,
        le=10.0,
    )
    easy_fixture_threshold: float = Field(
        default=1.538,
        description="Fixture difficulty threshold for easy fixtures (>= this = easy)",
        ge=1.0,
        le=2.2,
    )

    # Distribution file
    distributions_path: str = Field(
        default="data/calibration/distributions.json",
        description="Path to fitted distributions JSON file (relative to project root)",
    )

    # Regularization (for distribution fitting)
    regularization_lambda: float = Field(
        default=0.3,
        description="L2 regularization strength (higher = more shrinkage toward prior)",
        ge=0.0,
        le=1.0,
    )

    # Minimum sample size
    minimum_sample_size: int = Field(
        default=30,
        description="Minimum samples required to use fitted distribution (else use prior)",
        ge=10,
        le=100,
    )

    # Weight for additive fixture effect correction
    # This controls how much of the historical (tier × fixture) effect to apply
    # The calibration adds: empirical_blend_weight * (distribution_mean - tier_baseline)
    # where fixture_effect captures easy vs hard fixture difference
    empirical_blend_weight: float = Field(
        default=0.5,
        description=(
            "Weight for fixture effect correction (0=pure ML, 1=full historical effect). "
            "This adds an adjustment to ML predictions based on historical tier×fixture data. "
            "0.5 = apply 50% of the historical fixture effect difference. "
            "Higher values increase the easy/hard fixture differential for premium players."
        ),
        ge=0.0,
        le=1.0,
    )

    # Risk adjustment multiplier (how many std deviations for conservative/risk-taking)
    risk_adjustment_multiplier: float = Field(
        default=0.67,
        description=(
            "Standard deviation multiplier for risk profiles. "
            "0.67 = ~25th/75th percentile, 1.0 = ~16th/84th percentile. "
            "Higher values create larger differences between risk profiles."
        ),
        ge=0.0,
        le=2.0,
    )

    # Debug logging
    debug: bool = Field(
        default=False,
        description="Enable debug logging for calibration service",
    )


class OptimizationConfig(BaseModel):
    """Transfer Optimization Configuration"""

    # Transfer costs and penalties
    transfer_cost: float = Field(
        default=4.0,
        description="Points penalty per transfer beyond free transfers",
        ge=0.0,
        le=10.0,
    )

    # Optimization horizon
    optimization_horizon: str = Field(
        default="5gw",
        description="Optimization horizon: '1gw' for current gameweek only, '3gw' for 3-gameweek planning, '5gw' for 5-gameweek strategic planning",
    )

    @field_validator("optimization_horizon")
    @classmethod
    def validate_optimization_horizon(cls, v):
        if v not in ["1gw", "3gw", "5gw"]:
            raise ValueError(
                "optimization_horizon must be either '1gw', '3gw', or '5gw'"
            )
        return v

    # Transfer optimization method (removed - always uses simulated_annealing)
    # This field is kept for backwards compatibility but is no longer used
    transfer_optimization_method: str = Field(
        default="simulated_annealing",
        description="Transfer optimization method (deprecated - always uses simulated_annealing)",
    )

    # Simulated Annealing parameters
    sa_iterations: int = Field(
        default=2000,
        description="Number of SA iterations per restart for transfer optimization",
        ge=100,
        le=10000,
    )
    sa_restarts: int = Field(
        default=3,
        description="Number of SA restarts to run (best result is kept). More restarts = more reliable but slower.",
        ge=1,
        le=10,
    )
    sa_max_transfers_per_iteration: int = Field(
        default=3,
        description="Maximum number of players to swap in a single SA iteration",
        ge=1,
        le=5,
    )
    sa_random_seed: Optional[int] = Field(
        default=None,
        description="Random seed for SA reproducibility (None = different results each run). Set to integer for deterministic results.",
    )
    sa_deterministic_mode: bool = Field(
        default=False,
        description="Deterministic mode: prefer top candidates over random sampling when xP difference is large (>0.5). More stable but less exploration.",
    )
    sa_consensus_runs: int = Field(
        default=5,
        description="Number of full optimization runs for consensus mode. Runs multiple times and recommends most frequent optimal transfer. Higher = more reliable but slower.",
        ge=1,
        le=20,
    )
    sa_use_consensus_mode: bool = Field(
        default=True,
        description="Consensus mode: run optimization multiple times and aggregate results to find truly optimal solution. Recommended for finding best transfer.",
    )
    sa_exhaustive_search_max_transfers: int = Field(
        default=2,
        description="Use exhaustive search (guaranteed optimal) for 0-N transfers. Set to 0 to disable, 2 means exhaustive for 0-2 transfers. Higher = slower but guaranteed optimal.",
        ge=0,
        le=3,
    )
    sa_wildcard_iterations: int = Field(
        default=5000,
        description="Number of SA iterations for wildcard optimization (15 transfers). Higher than normal since search space is much larger.",
        ge=1000,
        le=20000,
    )
    sa_wildcard_restarts: int = Field(
        default=5,
        description="Number of SA restarts for wildcard optimization. More restarts = more reliable but slower.",
        ge=1,
        le=10,
    )
    sa_wildcard_use_consensus: bool = Field(
        default=True,
        description="Use consensus mode for wildcard: run multiple full optimizations and find consensus best squad. Recommended for finding truly optimal wildcard squad.",
    )

    # Free Hit parameters
    free_hit_budget: float = Field(
        default=100.0,
        description="Budget for Free Hit chip in millions. FPL rules give exactly £100m regardless of team value.",
        ge=100.0,
        le=100.0,
    )
    sa_free_hit_iterations: int = Field(
        default=5000,
        description="Number of SA iterations for Free Hit optimization (15 transfers). Same as wildcard since search space is identical.",
        ge=1000,
        le=20000,
    )
    sa_free_hit_restarts: int = Field(
        default=5,
        description="Number of SA restarts for Free Hit optimization.",
        ge=1,
        le=10,
    )
    sa_free_hit_use_consensus: bool = Field(
        default=True,
        description="Use consensus mode for Free Hit optimization.",
    )

    # Scenario analysis (greedy method)
    max_transfers: int = Field(
        default=3, description="Maximum transfers to analyze (0-3)", ge=0, le=5
    )
    scenario_display_limit: int = Field(
        default=7, description="Number of scenarios to display in table", ge=1, le=20
    )

    # Premium acquisition planning
    top_premium_targets: int = Field(
        default=3, description="Number of premium targets to analyze", ge=1, le=10
    )

    # Budget thresholds
    premium_player_threshold: float = Field(
        default=8.0, description="£8m+ considered premium", ge=5.0, le=15.0
    )
    budget_friendly_threshold: float = Field(
        default=7.5, description="<= £7.5m considered budget-friendly", ge=4.0, le=12.0
    )

    # Hauler-first strategy: ceiling bonus configuration
    # Based on top 1% manager analysis: they capture 3.0 haulers/GW vs 2.4 for average
    ceiling_bonus_enabled: bool = Field(
        default=True,
        description="Add ceiling bonus to optimization objective, favoring players with haul potential",
    )
    ceiling_bonus_factor: float = Field(
        default=0.15,
        description="Bonus factor per point of ceiling above threshold (ceiling = xP + 1.645*uncertainty). "
        "Default 0.15 means a player with 20 ceiling gets +1.5 bonus vs 10 ceiling player.",
        ge=0.0,
        le=0.5,
    )
    ceiling_bonus_threshold: float = Field(
        default=10.0,
        description="Ceiling threshold above which bonus applies. Players with ceiling > this get bonus.",
        ge=5.0,
        le=15.0,
    )


class VisualizationConfig(BaseModel):
    """Visualization and Display Configuration"""

    # Team strength display
    strength_display_rows: int = Field(
        default=20, description="Team strength display rows", ge=5, le=50
    )

    # Player trends
    max_gameweeks_to_load: int = Field(
        default=38, description="Maximum gameweeks in season", ge=1, le=50
    )
    max_consecutive_failures: int = Field(
        default=2, description="API failure tolerance", ge=1, le=10
    )

    # Table pagination
    default_page_size: int = Field(
        default=15, description="Default table page size", ge=5, le=100
    )
    squad_page_size: int = Field(
        default=15, description="Squad table page size", ge=5, le=50
    )
    scenario_page_size: int = Field(
        default=7, description="Scenario table page size", ge=3, le=20
    )
    large_table_page_size: int = Field(
        default=20, description="Large table page size", ge=10, le=100
    )

    # Chart styling
    chart_width: int = Field(
        default=800, description="Chart width in pixels", ge=400, le=2000
    )
    chart_height: int = Field(
        default=400, description="Chart height in pixels", ge=200, le=1000
    )


class CaptainSelectionConfig(BaseModel):
    """Intelligent Captain Selection Configuration"""

    # Strategy mode
    strategy_mode: str = Field(
        default="auto",
        description="Captain strategy mode: 'auto' (situation-aware), 'template_lock' (always highest owned), "
        "'protect_rank' (minimize downside), 'balanced' (xP-weighted), 'chase_rank' (differential focus), "
        "'maximum_upside' (pure ceiling-seeking)",
    )

    @field_validator("strategy_mode")
    @classmethod
    def validate_strategy_mode(cls, v):
        valid_modes = [
            "auto",
            "template_lock",
            "protect_rank",
            "balanced",
            "chase_rank",
            "maximum_upside",
        ]
        if v not in valid_modes:
            raise ValueError(f"strategy_mode must be one of {valid_modes}")
        return v

    # Rank thresholds for auto-detection
    elite_rank_threshold: int = Field(
        default=100_000,
        description="Rank threshold for 'protect_rank' recommendation",
        ge=1,
    )
    comfortable_rank_threshold: int = Field(
        default=500_000,
        description="Rank threshold for 'balanced' recommendation",
        ge=1,
    )
    chasing_rank_threshold: int = Field(
        default=2_000_000,
        description="Rank threshold for 'chase_rank' recommendation",
        ge=1,
    )

    # Template ownership thresholds
    template_ownership_threshold: float = Field(
        default=50.0,
        description="Ownership % above which a player is considered 'template'",
        ge=20.0,
        le=80.0,
    )
    high_ownership_threshold: float = Field(
        default=30.0,
        description="Ownership % for 'high ownership' classification",
        ge=10.0,
        le=60.0,
    )

    # Haul probability thresholds (from xP uncertainty)
    blank_threshold: int = Field(
        default=3,
        description="Points at or below this are considered a 'blank'",
        ge=0,
        le=5,
    )
    return_threshold: int = Field(
        default=8,
        description="Points at or below this are considered a 'return' (above blank)",
        ge=4,
        le=12,
    )
    # Above return_threshold is a 'haul'

    # Rank impact estimation
    show_rank_impact: bool = Field(
        default=True,
        description="Show expected rank impact for each captain choice",
    )
    show_haul_probabilities: bool = Field(
        default=True,
        description="Show blank/return/haul probability matrix",
    )

    # Risk adjustment
    differential_bonus_factor: float = Field(
        default=1.5,
        description="Multiplier for differential captain xP gain in chase_rank mode",
        ge=1.0,
        le=3.0,
    )
    template_safety_factor: float = Field(
        default=1.3,
        description="Multiplier for template captain in protect_rank mode",
        ge=1.0,
        le=2.0,
    )


class ChipCalendarConfig(BaseModel):
    """Chip Calendar and Deadline Configuration for 2025-26 Season"""

    # Chip set deadlines
    first_set_deadline_gw: int = Field(
        default=19,
        description="First set of chips must be used by this gameweek (GW19 = 30 Dec 2025)",
        ge=1,
        le=38,
    )
    second_set_deadline_gw: int = Field(
        default=38,
        description="Second set of chips deadline (end of season)",
        ge=20,
        le=38,
    )

    # Warning thresholds
    expiry_warning_threshold: int = Field(
        default=3,
        description="Warn when this many GWs or fewer remain to use chips",
        ge=1,
        le=10,
    )

    # Chip scoring weights for optimal gameweek calculation
    dgw_bonus_multiplier: float = Field(
        default=1.5,
        description="Bonus multiplier for DGW when scoring Free Hit/Triple Captain",
        ge=1.0,
        le=3.0,
    )
    fixture_difficulty_weight: float = Field(
        default=0.3,
        description="Weight of fixture difficulty in chip scoring",
        ge=0.0,
        le=1.0,
    )
    chip_lookahead_horizon: int = Field(
        default=5,
        description="Number of gameweeks to scan ahead for optimal chip usage",
        ge=1,
        le=10,
    )


class FPLConfig(BaseModel):
    """Master FPL Configuration Container"""

    xp_model: XPModelConfig = Field(
        default_factory=XPModelConfig, description="Expected Points Model Configuration"
    )
    team_strength: TeamStrengthConfig = Field(
        default_factory=TeamStrengthConfig, description="Team Strength Configuration"
    )
    minutes_model: MinutesModelConfig = Field(
        default_factory=MinutesModelConfig,
        description="Minutes Prediction Configuration",
    )
    statistical_estimation: StatisticalEstimationConfig = Field(
        default_factory=StatisticalEstimationConfig,
        description="Statistical Estimation Configuration",
    )
    fixture_difficulty: FixtureDifficultyConfig = Field(
        default_factory=FixtureDifficultyConfig,
        description="Fixture Difficulty Configuration",
    )
    xp_calibration: XPCalibrationConfig = Field(
        default_factory=XPCalibrationConfig,
        description="Probabilistic XP Calibration Configuration",
    )
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig, description="Optimization Configuration"
    )
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig, description="Visualization Configuration"
    )
    chip_calendar: ChipCalendarConfig = Field(
        default_factory=ChipCalendarConfig, description="Chip Calendar Configuration"
    )
    captain_selection: CaptainSelectionConfig = Field(
        default_factory=CaptainSelectionConfig,
        description="Intelligent Captain Selection Configuration",
    )

    @model_validator(mode="after")
    def validate_config_consistency(self):
        """Validate cross-field consistency"""
        # Ensure team strength range is consistent
        if self.team_strength.max_strength <= self.team_strength.min_strength:
            raise ValueError(
                "team_strength.max_strength must be greater than min_strength"
            )

        # Ensure fixture difficulty thresholds are ordered correctly
        if (
            self.fixture_difficulty.easy_fixture_threshold
            <= self.fixture_difficulty.average_fixture_min
        ):
            raise ValueError(
                "fixture_difficulty.easy_fixture_threshold must be greater than average_fixture_min"
            )

        return self


def load_config(
    config_path: Optional[Path] = None, config_data: Optional[Dict] = None
) -> FPLConfig:
    """
    Load configuration with environment variable overrides and optional config file

    Args:
        config_path: Optional path to JSON/YAML configuration file
        config_data: Optional dictionary of configuration data

    Environment variables can override any config value using the pattern:
    FPL_{SECTION}_{FIELD} = value

    Example: FPL_XP_MODEL_FORM_WEIGHT=0.8
    """
    # Start with default configuration
    config_dict = {}

    # Load from file if provided
    if config_path and config_path.exists():
        import json

        try:
            with open(config_path, "r") as f:
                if config_path.suffix.lower() == ".json":
                    config_dict = json.load(f)
                # Could add YAML support here if needed
        except Exception as e:
            print(f"⚠️ Warning: Failed to load config file {config_path}: {e}")

    # Override with provided config data
    if config_data:
        config_dict.update(config_data)

    # Environment variable overrides
    env_overrides = {}
    for env_var, value in os.environ.items():
        if env_var.startswith("FPL_"):
            # Parse FPL_SECTION_FIELD pattern
            parts = env_var.split("_")[1:]  # Remove FPL_ prefix
            if len(parts) >= 2:
                section = parts[0].lower()
                field = "_".join(parts[1:]).lower()

                if section not in env_overrides:
                    env_overrides[section] = {}

                # Try to convert value to appropriate type
                try:
                    # Try boolean
                    if value.lower() in ("true", "false"):
                        env_overrides[section][field] = value.lower() == "true"
                    # Try integer
                    elif value.isdigit() or (
                        value.startswith("-") and value[1:].isdigit()
                    ):
                        env_overrides[section][field] = int(value)
                    # Try float
                    elif "." in value:
                        env_overrides[section][field] = float(value)
                    # Keep as string
                    else:
                        env_overrides[section][field] = value
                except ValueError:
                    env_overrides[section][field] = value

    # Merge environment overrides into config_dict
    for section, fields in env_overrides.items():
        if section not in config_dict:
            config_dict[section] = {}
        config_dict[section].update(fields)

    # Create and validate the configuration
    try:
        config = FPLConfig(**config_dict)
        return config
    except Exception as e:
        print(f"⚠️ Warning: Configuration validation failed: {e}")
        print("Using default configuration...")
        return FPLConfig()


# Global configuration instance
config = load_config()
