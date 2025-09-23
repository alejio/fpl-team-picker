# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Fantasy Premier League (FPL) analysis suite that provides both season-start team building and weekly gameweek management. The project implements a comprehensive mathematical framework combining Poisson distributions, team ratings, player performance metrics, fixture analysis, and live data integration for optimal FPL decision making.

The project is structured as a Python package with modular architecture and provides two main interfaces via Marimo notebooks for interactive data analysis.

## Project Structure

```
fpl-team-picker/
├── fpl_team_picker/           # Main Python package
│   ├── config/               # Configuration system
│   │   ├── settings.py       # Global configuration with Pydantic models
│   │   └── utils.py          # Configuration utilities
│   ├── core/                 # Core business logic
│   │   ├── data_loader.py    # Data loading and preprocessing
│   │   ├── team_strength.py  # Dynamic team strength calculations
│   │   ├── xp_model.py       # Expected points calculation engine
│   │   └── chip_assessment.py # Chip timing analysis and recommendations
│   ├── optimization/         # Transfer and team optimization
│   │   └── optimizer.py      # Smart transfer optimization logic
│   ├── visualization/        # Charts and visual components
│   │   └── charts.py         # Plotly-based interactive visualizations
│   ├── interfaces/           # User interfaces (Marimo notebooks)
│   │   ├── season_planner.py # Season-start team building interface
│   │   ├── gameweek_manager.py # Weekly gameweek management interface
│   │   └── ml_xp_experiment.py # ML expected points model development and validation
│   └── utils/                # Utility functions
│       └── helpers.py        # Common helper functions
├── pyproject.toml            # Project configuration and dependencies
├── config_example.json       # Example configuration file
└── fpl_rules.md             # FPL rules reference
```

## Data Loading

This project uses the fpl-dataset-builder database client library for all data operations.

### Usage
```python
from client import FPLDataClient

client = FPLDataClient()
players = client.get_current_players()
teams = client.get_current_teams()
fixtures = client.get_fixtures_normalized()
xg_rates = client.get_player_xg_xa_rates()
live_data = client.get_gameweek_live_data(gw)
```

### Available Data Functions
- `get_current_players()` - Current season player data (prices, positions, teams)
- `get_current_teams()` - Team reference data for ID lookups
- `get_fixtures_normalized()` - Normalized fixture data with team IDs and kickoff times
- `get_player_xg_xa_rates()` - Expected goals (xG90) and expected assists (xA90) rates per 90 minutes
- `get_gameweek_live_data(gw)` - Real-time player performance data for active/completed gameweeks
- `get_gameweek_performance(gw)` - Detailed performance data for specific gameweek
- `get_player_deltas_current()` - Week-over-week performance and market movement tracking

### Installation
The fpl-dataset-builder is configured as a local dependency:

```bash
uv sync
```

**Configuration in pyproject.toml:**
```toml
dependencies = [
    "fpl-dataset-builder",
    # ... other dependencies
]

[tool.uv.sources]
fpl-dataset-builder = { path = "../fpl-dataset-builder", editable = true }
```

## Core Architecture

### 1. Configuration System (`fpl_team_picker/config/`)

**Centralized configuration management with Pydantic validation:**
- **Type-safe configuration** with field validation and environment variable support
- **Modular config sections**: XP Model, Team Strength, Minutes Model, Statistical Estimation, etc.
- **Environment variable overrides** using pattern `FPL_{SECTION}_{FIELD}`
- **JSON configuration file support** with graceful fallbacks

Key configuration sections:
- `XPModelConfig` - Form weighting, thresholds, debug settings
- `TeamStrengthConfig` - Dynamic strength calculation parameters
- `MinutesModelConfig` - Enhanced minutes prediction parameters
- `StatisticalEstimationConfig` - xG/xA estimation parameters
- `OptimizationConfig` - Transfer optimization settings
- `VisualizationConfig` - Chart and display settings
- `ChipAssessmentConfig` - Chip recommendation thresholds and parameters

### 2. Core Business Logic (`fpl_team_picker/core/`)

#### Data Loader (`data_loader.py`)
**Centralized data orchestration and preprocessing:**
- `fetch_fpl_data(target_gameweek, form_window)` - Main data loading with historical form
- `fetch_manager_team(previous_gameweek)` - Manager team data retrieval
- `process_current_squad(team_data, players, teams)` - Squad processing with captain info
- `load_gameweek_datasets(target_gameweek)` - Comprehensive gameweek data orchestration

#### Expected Points Engine (`xp_model.py`)
**Dedicated XP calculation engine with form weighting:**
```python
from fpl_team_picker.core.xp_model import XPModel

xp_model = XPModel(form_weight=0.7, form_window=5)
xp_results = xp_model.calculate_expected_points(players, fixtures, ...)
```

Key features:
- **Form-weighted predictions** - 70% recent form + 30% season baseline
- **Statistical xG/xA estimation** - Multi-factor estimation for missing data
- **Enhanced minutes prediction** - SBP + availability + form-based modeling
- **Multi-gameweek capability** - 5-gameweek temporal weighting

#### Dynamic Team Strength (`team_strength.py`)
**Evolving team strength ratings throughout the season:**
- **Previous season transition logic** - Weighted previous season (2024-25) baseline → current season (2025-26) focus
- **GW8+ pure current season** - Responsive to current form once sufficient data
- **Rolling window analysis** - 6-gameweek performance windows
- **Venue-specific adjustments** - Home/away advantage incorporation

#### Chip Assessment Engine (`chip_assessment.py`)
**Heuristic-based analysis for optimal chip usage timing:**
- **Wildcard Assessment** - Transfer opportunity analysis and fixture run quality evaluation
- **Free Hit Analysis** - Double gameweek detection and temporary squad improvement potential
- **Bench Boost Evaluation** - Bench strength calculation and rotation risk assessment
- **Triple Captain Identification** - Premium captain candidate analysis and fixture quality review
- **Traffic Light Recommendations** - 🟢 RECOMMENDED, 🟡 CONSIDER, 🔴 HOLD status system

### 3. Optimization Engine (`fpl_team_picker/optimization/`)

#### Transfer Optimizer (`optimizer.py`)
**Smart transfer optimization and budget analysis:**
- `optimize_team_with_transfers()` - Smart 0-3 transfer optimization with scenario analysis
- `calculate_total_budget_pool()` - Advanced budget analysis including sellable player values
- `premium_acquisition_planner()` - Multi-transfer scenarios for expensive targets
- `get_best_starting_11()` - Formation-flexible lineup selection
- `select_captain()` - Risk-adjusted captaincy recommendations

### 4. Visualization Suite (`fpl_team_picker/visualization/`)

#### Interactive Charts (`charts.py`)
**Plotly-based interactive visualizations:**
- `create_xp_results_display()` - Expected points results with filtering
- `create_form_analytics_display()` - Hot/cold player analysis
- `create_player_trends_visualization()` - Historical performance trends
- `create_fixture_difficulty_visualization()` - Fixture difficulty heatmaps

### 5. User Interfaces (`fpl_team_picker/interfaces/`)

#### Season Planner (`season_planner.py`)
**Marimo notebook for season-start team building:**
- Multi-gameweek xP calculations (GW1-5 weighted horizon)
- Simulated Annealing team optimization with constraint satisfaction
- Formation-flexible starting 11 selection (8 valid formations)
- Transfer risk analysis and squad stability metrics

#### Gameweek Manager (`gameweek_manager.py`)
**Marimo notebook for weekly gameweek management:**
- Form-weighted predictions with live data integration
- Dynamic team strength analysis
- Transfer optimization with 0-3 transfer scenarios
- Captain selection tools with risk assessment
- Player performance trends and form analytics
- Chip assessment with smart timing recommendations for Wildcard, Free Hit, Bench Boost, and Triple Captain

#### ML Expected Points Experiment (`ml_xp_experiment.py`)
**Marimo notebook for ML model development and validation:**
- XGBoost-based expected points prediction with leak-free temporal validation
- Enhanced feature engineering with historical per-90 metrics
- Position-specific models (GKP, DEF, MID, FWD) with ensemble predictions
- Form-weighted training data using historical gameweek performance
- Model comparison and performance validation against actual FPL results

## Command Line Interface

The project provides CLI entry points for both interfaces:

```bash
# Season-start team building
fpl-season-planner

# Weekly gameweek management
fpl-gameweek-manager

# ML model development
marimo run fpl_team_picker/interfaces/ml_xp_experiment.py

# Or run directly with Marimo
marimo run fpl_team_picker/interfaces/season_planner.py
marimo run fpl_team_picker/interfaces/gameweek_manager.py
marimo run fpl_team_picker/interfaces/ml_xp_experiment.py
```

## ML Model Development Data Sources

### Historical Gameweek Performance Data
The fpl-dataset-builder project provides comprehensive historical data for ML model development via the `FPLDataClient`:

**Available Historical Data:**
- `get_gameweek_performance(gw)` - All players' performance for specific gameweek
- `get_player_gameweek_history(player_id, start_gw, end_gw)` - Historical performance for individual players
- `get_my_picks_history(start_gw, end_gw)` - Historical team selections across gameweeks

**Core Performance Fields (Leak-Free for ML Training):**
- **Basic Stats**: `total_points`, `minutes`, `goals_scored`, `assists`, `clean_sheets`, `goals_conceded`
- **Advanced Metrics**: `expected_goals`, `expected_assists`, `ict_index`, `bps`, `influence`, `creativity`, `threat`
- **Context**: `team_id`, `opponent_team`, `was_home`, `value` (player price at time of gameweek)

**ML Feature Engineering (Temporal Validation):**
The ML experiment interface implements proper temporal validation to prevent data leakage:
- Historical per-90 metrics calculated from cumulative data up to each training gameweek
- Position-specific modeling with separate XGBoost models for GKP/DEF/MID/FWD
- Ensemble predictions combining general + position-specific model outputs
- Leak-free features: `xG90_historical`, `xA90_historical`, `points_per_90`, `bps_historical`, `ict_index_historical`

**Usage Example:**
```python
from client import FPLDataClient

client = FPLDataClient()

# Get training data for multiple gameweeks with proper temporal validation
for gw in range(2, 6):  # GW2-5 for training
    gw_data = client.get_gameweek_performance(gw)
    # Calculate cumulative historical stats up to previous gameweek only
    # Use for training to predict current gameweek performance
```

## Expected Points Models

### Multi-Gameweek Model (Season Planning)
**Fast implementation for season-start team building:**

1. **League Baselines** - μ_home = 1.43, μ_away = 1.15 (Premier League averages)
2. **Team Strength Ratings** - Dynamic calculations with historical baselines
3. **Enhanced Minutes Model** - SBP + availability + position-specific durability
4. **Multi-Gameweek xP** - 5-week horizon with temporal weighting (1.0, 0.9, 0.8, 0.7, 0.6)

### Single Gameweek Model (Weekly Management)
**Form-weighted predictions with live data integration:**

1. **Live Data Integration** - Real-time performance and market intelligence
2. **Form-Weighted Calculations** - 70% recent form + 30% season average
3. **Dynamic Performance Adjustments** - Form multipliers and momentum tracking

### ML-Based Expected Points Model (Experimental)
**XGBoost machine learning approach with leak-free temporal validation:**

1. **Enhanced Feature Engineering** - Historical per-90 metrics with cumulative calculations
2. **Position-Specific Models** - Separate XGBoost models for GKP, DEF, MID, FWD positions
3. **Ensemble Predictions** - Combines general + position-specific model outputs
4. **Temporal Validation** - Proper train/test splits preventing data leakage
5. **Hyperparameter Optimization** - Tuned XGBoost parameters for FPL prediction accuracy

**Key Features:**
- Uses only historical data available at prediction time (no future information)
- Incorporates gameweek-by-gameweek performance history from fpl-dataset-builder
- Validates against actual FPL results with performance metrics
- Experimental interface for model development and accuracy testing

## Development Commands

**Primary Interfaces:**
```bash
# Install dependencies
uv sync

# Run season planner
fpl-season-planner
# or: marimo run fpl_team_picker/interfaces/season_planner.py

# Run gameweek manager
fpl-gameweek-manager
# or: marimo run fpl_team_picker/interfaces/gameweek_manager.py

# Development mode
marimo edit fpl_team_picker/interfaces/season_planner.py
marimo edit fpl_team_picker/interfaces/gameweek_manager.py
marimo edit fpl_team_picker/interfaces/ml_xp_experiment.py
```

**Code Quality:**
```bash
# Lint code
ruff check fpl_team_picker/

# Format code
ruff format fpl_team_picker/

# Find unused code
vulture fpl_team_picker/
```

## Technical Specifications

**Dependencies:**
- Python 3.13+
- marimo>=0.14.16 (interactive notebook interface)
- pandas>=2.3.1 (data manipulation)
- numpy>=2.3.2 (numerical computations)
- plotly>=6.3.0 (interactive visualizations)
- pydantic (configuration validation)
- requests (HTTP client)
- xgboost (machine learning for expected points models)
- scikit-learn (ML utilities and preprocessing)
- fpl-dataset-builder (local dependency for historical data)

**Key Features:**
- **Modular architecture** - Separated concerns with clear interfaces
- **Type-safe configuration** - Pydantic models with validation
- **Interactive notebooks** - Marimo-based user interfaces
- **CLI integration** - Command-line entry points
- **Comprehensive testing** - Form analytics and model validation
- **Live data integration** - Real-time FPL API data
- **Advanced visualizations** - Interactive Plotly charts

## FPL Rules Reference

All implementations must comply with official FPL rules documented in `fpl_rules.md`. Key constraints include:
- Squad composition: 2 GKP, 5 DEF, 5 MID, 3 FWD
- £100m budget limit
- Max 3 players per real team
- Valid formations for starting 11

## Configuration

The project uses a comprehensive configuration system. Create `config.json` to override defaults:

```json
{
  "xp_model": {
    "form_weight": 0.7,
    "form_window": 5
  },
  "team_strength": {
    "historical_transition_gw": 8
  },
  "optimization": {
    "transfer_cost": 4.0,
    "max_transfers": 3
  }
}
```

Environment variables can override any setting:
```bash
export FPL_XP_MODEL_FORM_WEIGHT=0.8
export FPL_TEAM_STRENGTH_HISTORICAL_TRANSITION_GW=10
```

## Performance Benchmark

**2024-25 FPL Winner**: 2,810 points (74 points per gameweek average)
- Target: 5-week xP projections should align with ~370 points (5 × 74) for competitive squads
- Validation: Premium players should project 8-12 xP per gameweek, budget options 4-7 xP

## Development Workflow

1. **Season Start**: Use season planner for initial 15-player squad building
2. **Weekly Planning**: Use gameweek manager for lineup and transfer decisions, including chip assessment analysis
3. **Post-Gameweek**: Analyze results and validate model predictions
4. **ML Model Development**: Use ML experiment interface for expected points model improvement
   - Train models on historical gameweek performance data
   - Validate with proper temporal splits to prevent data leakage
   - Compare ML predictions against actual FPL results
   - Iterate on feature engineering and hyperparameter optimization
5. **Continuous Improvement**: Update configuration and model parameters

The modular architecture supports rapid iteration and testing of different strategies while maintaining code quality and type safety. The ML experiment interface provides a dedicated environment for developing and validating machine learning approaches to expected points prediction using comprehensive historical data from the fpl-dataset-builder.
