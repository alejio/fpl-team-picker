"""
FPL Team Picker Configuration Module

Provides centralized configuration management for the entire application.
Import the global config instance to access all configuration values.

Usage:
    from fpl_team_picker.config import config
    
    # Access XP model configuration
    form_weight = config.xp_model.form_weight
    
    # Access team strength configuration  
    transition_gw = config.team_strength.historical_transition_gw
    
    # Access optimization configuration
    transfer_cost = config.optimization.transfer_cost
"""

from .settings import (
    FPLConfig,
    XPModelConfig,
    TeamStrengthConfig,
    MinutesModelConfig,
    StatisticalEstimationConfig,
    FixtureDifficultyConfig,
    OptimizationConfig,
    VisualizationConfig,
    DataLoadingConfig,
    config,
    load_config
)

__all__ = [
    'FPLConfig',
    'XPModelConfig', 
    'TeamStrengthConfig',
    'MinutesModelConfig',
    'StatisticalEstimationConfig',
    'FixtureDifficultyConfig',
    'OptimizationConfig',
    'VisualizationConfig',
    'DataLoadingConfig',
    'config',
    'load_config'
]