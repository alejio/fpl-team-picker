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
    ml_ensemble_rule_weight: float = Field(
        default=0.22,
        description="Weight for rule-based in ML ensemble (0-1) - optimized based on 0.78 MAE performance",
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

    # UNUSED PARAMETERS - DEPRECATED 2025-01-21
    # These were designed for form analytics features but are not currently used
    # in any interfaces. Form analytics are handled directly in charts.py
    # Keep commented for potential future use.

    # # Form classification thresholds
    # hot_threshold: float = Field(
    #     default=6.0, description="Points per gameweek for 'hot' players", ge=0.0
    # )
    # rising_threshold: float = Field(
    #     default=5.0, description="Points per gameweek for 'rising' players", ge=0.0
    # )
    # stable_threshold: float = Field(
    #     default=3.0, description="Points per gameweek for 'stable' players", ge=0.0
    # )
    # declining_threshold: float = Field(
    #     default=1.0, description="Points per gameweek for 'declining' players", ge=0.0
    # )

    # # Form multiplier thresholds
    # hot_multiplier_threshold: float = Field(
    #     default=1.4, description=">= 1.4 = 🔥 Hot", ge=1.0
    # )
    # rising_multiplier_threshold: float = Field(
    #     default=1.15, description=">= 1.15 = 📈 Rising", ge=1.0
    # )
    # stable_multiplier_threshold: float = Field(
    #     default=0.85, description=">= 0.85 = ➡️ Stable", ge=0.0, le=1.0
    # )
    # declining_multiplier_threshold: float = Field(
    #     default=0.7, description=">= 0.7 = 📉 Declining (else ❄️ Cold)", ge=0.0, le=1.0
    # )


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

    # Difficulty classification thresholds (adjusted for better elite team recognition)
    easy_fixture_threshold: float = Field(
        default=1.20, description="> 1.20 = Easy (🟢)", ge=1.0, le=2.0
    )
    average_fixture_min: float = Field(
        default=0.90, description="0.90-1.20 = Average (🟡)", ge=0.5, le=1.0
    )
    # < 0.90 = Hard (🔴) - now properly captures elite teams like Liverpool

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
        description="Optimization horizon: '1gw' for current gameweek only, '5gw' for 5-gameweek strategic planning",
    )

    @field_validator("optimization_horizon")
    @classmethod
    def validate_optimization_horizon(cls, v):
        if v not in ["1gw", "5gw"]:
            raise ValueError("optimization_horizon must be either '1gw' or '5gw'")
        return v

    # Scenario analysis
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
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig, description="Optimization Configuration"
    )
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig, description="Visualization Configuration"
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
