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
│   ├── domain/               # Clean architecture domain layer
│   │   ├── services/         # Domain services (business logic)
│   │   │   ├── data_orchestration_service.py    # Data loading & validation
│   │   │   ├── expected_points_service.py       # XP calculation engine
│   │   │   ├── transfer_optimization_service.py # Transfer optimization
│   │   │   ├── chip_assessment_service.py       # Chip timing analysis
│   │   │   ├── performance_analytics_service.py # Form & performance analysis
│   │   │   ├── fixture_analysis_service.py      # Fixture difficulty analysis
│   │   │   ├── squad_management_service.py      # Squad analysis & management
│   │   │   └── visualization_service.py         # Chart generation
│   │   ├── models/           # Domain models
│   │   │   ├── player.py     # Player domain models
│   │   │   ├── team.py       # Team domain models
│   │   │   └── fixture.py    # Fixture domain models
│   │   ├── repositories/     # Data access contracts
│   │   │   ├── player_repository.py   # Player data access interface
│   │   │   ├── team_repository.py     # Team data access interface
│   │   │   └── fixture_repository.py  # Fixture data access interface
│   │   └── common/           # Shared domain utilities
│   │       └── result.py     # Result types for error handling
│   ├── adapters/             # Infrastructure implementations
│   │   └── database_repositories.py   # Database repository implementations
│   ├── core/                 # Legacy core business logic (gradually migrating to domain/)
│   │   ├── data_loader.py    # Data loading and preprocessing
│   │   ├── team_strength.py  # Dynamic team strength calculations
│   │   ├── xp_model.py       # Expected points calculation engine
│   │   └── chip_assessment.py # Chip timing analysis and recommendations
│   ├── optimization/         # Transfer and team optimization
│   │   └── optimizer.py      # Smart transfer optimization logic
│   ├── visualization/        # Charts and visual components
│   │   └── charts.py         # Plotly-based interactive visualizations
│   ├── interfaces/           # User interfaces (Frontend adapters)
│   │   ├── season_planner.py # Season-start team building interface (2257 lines)
│   │   ├── gameweek_manager.py # Weekly gameweek management interface (969 lines - clean)
│   │   └── ml_xp_experiment.py # ML expected points model development (2947 lines)
│   └── utils/                # Utility functions
│       └── helpers.py        # Common helper functions
├── tests/                    # Comprehensive testing suite
│   ├── domain/               # Domain layer tests
│   │   ├── models/           # Domain model tests
│   │   └── services/         # Domain service tests (integration & unit)
│   ├── adapters/             # Infrastructure tests
│   └── interfaces/           # Interface compatibility tests
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

The project follows **Clean Architecture principles** with clear separation between domain logic, infrastructure, and presentation layers. This design enables frontend flexibility and comprehensive testing while maintaining code quality.

### 1. Domain Layer (`fpl_team_picker/domain/`) - **Frontend-Agnostic Business Logic**

The domain layer contains all business logic and is completely independent of any framework or UI technology.

#### Domain Services - **Pure Business Logic**

**Data Orchestration Service** (`data_orchestration_service.py`)
```python
from fpl_team_picker.domain.services import DataOrchestrationService

# Boundary validation - fail fast with clean data guarantee
orchestration_service = DataOrchestrationService()
gameweek_data = orchestration_service.load_gameweek_data(target_gameweek=10, form_window=5)
# Returns: Guaranteed valid data dictionary - all downstream operations are deterministic
```

**Expected Points Service** (`expected_points_service.py`)
```python
from fpl_team_picker.domain.services import ExpectedPointsService

# Pure XP calculation - works with any frontend
xp_service = ExpectedPointsService()
players_with_xp = xp_service.calculate_combined_results(
    gameweek_data, use_ml_model=False
)
# Returns: 1GW + 5GW expected points with model metadata
```

**Transfer Optimization Service** (`transfer_optimization_service.py`)
```python
from fpl_team_picker.domain.services import TransferOptimizationService

# Smart transfer scenarios - 0-3 transfers with constraint support
transfer_service = TransferOptimizationService()
optimization_results = transfer_service.optimize_transfers(
    players_with_xp=players_with_xp,
    current_squad=current_squad,
    team_data=team_data,
    must_include_ids={player_id_1, player_id_2},
    must_exclude_ids={player_id_3}
)
# Returns: Multiple transfer scenarios with net XP analysis
```

**Chip Assessment Service** (`chip_assessment_service.py`)
```python
from fpl_team_picker.domain.services import ChipAssessmentService

# Traffic light chip recommendations
chip_service = ChipAssessmentService()
chip_recommendations = chip_service.assess_all_chips(
    current_squad=squad_with_xp,
    all_players=players_with_xp,
    fixtures=fixtures,
    available_chips=["wildcard", "bench_boost"]
)
# Returns: 🟢 RECOMMENDED, 🟡 CONSIDER, 🔴 HOLD status for each chip
```

**Performance Analytics Service** (`performance_analytics_service.py`)
```python
from fpl_team_picker.domain.services import PerformanceAnalyticsService

# Form analysis and hot/cold player identification
analytics_service = PerformanceAnalyticsService()
form_analysis = analytics_service.analyze_player_form(
    players_with_xp, form_window=5
)
# Returns: Form trends, momentum indicators, hot/cold classifications
```

**Fixture Analysis** - *Handled by visualization layer*
```python
# Fixture difficulty analysis is now handled directly in charts.py
from fpl_team_picker.visualization.charts import create_fixture_difficulty_visualization

fixture_analysis = create_fixture_difficulty_visualization(
    target_gameweek=10, gameweeks_ahead=5, mo
)
# Returns: Interactive Marimo fixture difficulty display
```

**Squad Management Service** (`squad_management_service.py`)
```python
from fpl_team_picker.domain.services import SquadManagementService

# Squad analysis and lineup optimization
squad_service = SquadManagementService()
lineup_analysis = squad_service.analyze_squad_composition(
    current_squad, players_with_xp
)
# Returns: Squad balance, formation flexibility, value distribution
```

**Player Analytics Service** (`player_analytics_service.py`)
```python
from fpl_team_picker.domain.services import PlayerAnalyticsService
from fpl_team_picker.adapters.database_repositories import DatabasePlayerRepository

# Type-safe player operations using domain models
player_repo = DatabasePlayerRepository()
analytics_service = PlayerAnalyticsService(player_repo)

# Get all players with 70+ validated attributes (enriched + derived metrics)
players: List[EnrichedPlayerDomain] = analytics_service.get_all_players_enriched()

# Type-safe filtering and analysis
penalty_takers = analytics_service.get_penalty_takers()
high_value = analytics_service.get_high_value_players(min_value_score=80)
injury_risks = analytics_service.get_injury_risks(min_risk=0.5)
form_improving = analytics_service.get_form_improving_players()
top_value = analytics_service.get_top_players_by_value(limit=10)

# Returns: List[EnrichedPlayerDomain] with full type safety and computed properties
# player.is_high_value, player.has_injury_concern, player.is_penalty_taker, etc.
```

**Visualization** - *Handled by charts layer*
```python
# Chart generation is handled directly in charts.py for Marimo
from fpl_team_picker.visualization.charts import create_xp_results_display

xp_display = create_xp_results_display(players_with_xp, target_gameweek, mo)
# Returns: Ready-to-use Marimo visualization components
```

#### Domain Models (`domain/models/`)
- **PlayerDomain**: Core player entity with business rules
- **TeamDomain**: Team entity with strength calculations
- **FixtureDomain**: Fixture entity with difficulty analysis

#### Repository Contracts (`domain/repositories/`)
- **PlayerRepository**: Abstract player data access
- **TeamRepository**: Abstract team data access
- **FixtureRepository**: Abstract fixture data access

### 2. Infrastructure Layer (`fpl_team_picker/adapters/`) - **Framework Implementations**

**Database Repositories** (`database_repositories.py`)
```python
from fpl_team_picker.adapters.database_repositories import DatabasePlayerRepository

# Concrete implementations of repository contracts
player_repo = DatabasePlayerRepository()  # Uses fpl-dataset-builder
team_repo = DatabaseTeamRepository()      # FPL API integration
fixture_repo = DatabaseFixtureRepository() # Fixture data access
```

### 3. Configuration System (`fpl_team_picker/config/`)

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

### 4. Legacy Core Business Logic (`fpl_team_picker/core/`) - **Migration in Progress**

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

### 5. Presentation Layer (`fpl_team_picker/interfaces/`) - **Frontend Adapters**

The interfaces serve as **presentation adapters** that orchestrate domain services for specific frontends. They contain zero business logic and focus purely on composition and display.

#### Clean Architecture Benefits Achieved:
- **969 lines** (vs 1457 lines original) - **33% reduction** through domain service extraction
- **Zero business logic** in presentation cells
- **Frontend migration ready** - same services work for FastAPI, React, CLI
- **Comprehensive testing** - domain services tested independently of UI

#### Gameweek Manager (`gameweek_manager.py`) - **969 lines - Clean Architecture**
**Marimo presentation adapter for weekly gameweek management:**

**Service Orchestration Pattern:**
```python
# Pure composition - no business logic
@app.cell
def _(gameweek_data, mo):
    if gameweek_data:
        # Domain service usage
        from fpl_team_picker.domain.services import ExpectedPointsService
        xp_service = ExpectedPointsService()
        players_with_xp = xp_service.calculate_combined_results(gameweek_data)

        # Pure presentation
        xp_results_display = create_xp_results_display(players_with_xp, mo)
    else:
        xp_results_display = mo.md("Load gameweek data first")
    return xp_results_display,
```

**Key Features:**
- **Data Loading**: Uses `DataOrchestrationService` for boundary validation
- **Expected Points**: Orchestrates `ExpectedPointsService` (ML + rule-based models)
- **Transfer Optimization**: Uses `TransferOptimizationService` with 0-3 transfer scenarios
- **Form Analytics**: Leverages `PerformanceAnalyticsService` for hot/cold analysis
- **Chip Assessment**: Integrates `ChipAssessmentService` with traffic light recommendations
- **Captain Selection**: Risk-adjusted recommendations via optimization results
- **Fixture Analysis**: Uses `create_fixture_difficulty_visualization()` for difficulty heatmaps

#### Season Planner (`season_planner.py`) - **2257 lines**
**Marimo notebook for season-start team building:**
- Multi-gameweek xP calculations (GW1-5 weighted horizon)
- Simulated Annealing team optimization with constraint satisfaction
- Formation-flexible starting 11 selection (8 valid formations)
- Transfer risk analysis and squad stability metrics

#### ML Expected Points Experiment (`ml_xp_experiment.py`) - **2947 lines**
**Marimo notebook for ML model development and validation:**
- XGBoost-based expected points prediction with leak-free temporal validation
- Enhanced feature engineering with historical per-90 metrics
- Position-specific models (GKP, DEF, MID, FWD) with ensemble predictions
- Form-weighted training data using historical gameweek performance
- Model comparison and performance validation against actual FPL results

#### xP Accuracy Tracking (`xp_accuracy_tracking.py`) - **Experimentation Notebook**
**Marimo notebook for model accuracy analysis and algorithm optimization:**

**Purpose:** Development and validation tool for algorithm experimentation (not for end-user predictions)

**Key Features:**
- **Historical Accuracy Tracking** - Monitor MAE/RMSE/correlation trends over completed gameweeks
- **Algorithm A/B Testing** - Compare multiple algorithm versions side-by-side
- **Position-Specific Analysis** - Identify which positions need model improvements
- **Parameter Optimization** - Test form_weight (0.5, 0.7, 0.9) and form_window (3GW, 5GW, 8GW) variations

**Interactive Controls:**
- Gameweek range selection for analysis
- Algorithm version selector (current, experimental_high_form, experimental_low_form, v1.0)
- Position-specific accuracy breakdown
- Real-time algorithm comparison with winner recommendations

**Use Cases:**
1. **Model Validation** - Measure prediction accuracy against actual FPL results
2. **Algorithm Development** - Test new parameters on historical data before deployment
3. **Performance Monitoring** - Track accuracy trends to detect model degradation
4. **Evidence-Based Optimization** - Data-driven algorithm selection for gameweek manager

**Workflow:**
```bash
# Run accuracy tracking notebook
fpl-xp-accuracy

# Analyze accuracy trends → Identify improvements → Test algorithm variants →
# Validate on historical data → Select best performer → Update gameweek manager algorithm
```

## Command Line Interface

The project provides CLI entry points for all interfaces:

```bash
# Season-start team building
fpl-season-planner

# Weekly gameweek management (production)
fpl-gameweek-manager

# Model accuracy analysis & experimentation (development)
fpl-xp-accuracy

# ML model development
fpl-ml-experiment

# Or run directly with Marimo
marimo run fpl_team_picker/interfaces/season_planner.py
marimo run fpl_team_picker/interfaces/gameweek_manager.py
marimo run fpl_team_picker/interfaces/xp_accuracy_tracking.py
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

## Historical xP Recomputation Framework

**Recomputation Strategy** - Instead of storing predictions at prediction time, we recompute historical xP using arbitrary algorithms to enable retroactive testing and model improvement.

### Why Recomputation Over Storage?

**Benefits:**
- ✅ Test unlimited algorithm variations retroactively on historical data
- ✅ Not locked into predictions made with old/buggy algorithm versions
- ✅ Enable scientific hypothesis testing and A/B comparisons
- ✅ Future-proof for evaluating new algorithms against 2+ seasons of data
- ✅ Minimal storage overhead (reuse existing gameweek_performance data)

### Historical Data Requirements

**Required Data (Fail Fast if Missing):**
- ✅ **Historical prices**: `raw_player_gameweek_performance.value` field
- ✅ **Historical performance**: Goals, assists, minutes, bonus points, ICT
- ✅ **Historical fixtures**: Team matchups, home/away, difficulty
- ✅ **Historical team data**: Stable team references
- ✅ **Historical availability**: `get_player_availability_snapshot(gw)` from dataset-builder
  - Player status (available, injured, suspended, etc.)
  - Chance of playing percentages
  - Injury/suspension news

**Data Quality Philosophy:**
- ❌ **No fallbacks** - If data is missing, fail fast with actionable error message
- ✅ **Boundary validation** - Validate at data loading boundary, trust downstream
- ✅ **Upstream fixes** - Missing data indicates dataset-builder quality issue to fix

**Before Recomputing Historical Gameweeks:**
Ensure dataset-builder has captured snapshots for target gameweeks:
```bash
# Capture snapshot for specific gameweek
cd ../fpl-dataset-builder
uv run main.py snapshot --gameweek 8

# Verify snapshot exists
python -c "from client import FPLDataClient; print(FPLDataClient().get_player_availability_snapshot(8))"
```

**Error Handling:**
Recomputation will fail with clear error messages if:
- Historical prices missing for any gameweek
- Availability snapshots not captured
- Required data fields missing from dataset-builder schema
- Player data incomplete or inconsistent

Fix upstream data quality issues in dataset-builder rather than working around them.

### Core Components

#### 1. Historical Data Loader (`DataOrchestrationService`)

```python
from fpl_team_picker.domain.services import DataOrchestrationService

orchestration_service = DataOrchestrationService()

# Load gameweek state as it existed historically
historical_data = orchestration_service.load_historical_gameweek_state(
    target_gameweek=5,
    form_window=5,
    include_snapshots=True  # Use availability snapshots if available
)

# Returns: Same structure as load_gameweek_data() but with historical values
# - players: with historical prices from gameweek_performance.value
# - availability_snapshot: from dataset-builder (required - fails if missing)
# - live_data_historical: cumulative up to target_gameweek only (no future data)

# Fails fast with actionable error if:
# - Historical prices missing for target gameweek
# - Availability snapshot not captured
# - Required fields missing from data
```

#### 2. Algorithm Version Management (`PerformanceAnalyticsService`)

```python
from fpl_team_picker.domain.services.performance_analytics_service import (
    AlgorithmVersion,
    ALGORITHM_VERSIONS,
)

# Pydantic-based algorithm configuration
class AlgorithmVersion(BaseModel):
    name: str
    form_weight: float  # 0.0-1.0
    form_window: int    # 1-10 gameweeks
    use_team_strength: bool
    team_strength_params: Dict[str, Any]
    minutes_model_params: Dict[str, Any]
    statistical_estimation_params: Dict[str, Any]

# Pre-configured algorithm versions
ALGORITHM_VERSIONS = {
    "v1.0": AlgorithmVersion(form_weight=0.7, form_window=5, ...),
    "current": AlgorithmVersion(form_weight=0.7, form_window=5, ...),
    "experimental_high_form": AlgorithmVersion(form_weight=0.9, form_window=3, ...),
    "experimental_low_form": AlgorithmVersion(form_weight=0.5, form_window=8, ...),
}
```

#### 3. Historical Recomputation Engine

```python
from fpl_team_picker.domain.services import PerformanceAnalyticsService

analytics_service = PerformanceAnalyticsService()

# Recompute xP for a single historical gameweek
predictions_gw5 = analytics_service.recompute_historical_xp(
    target_gameweek=5,
    algorithm_version="current",
    include_snapshots=True
)
# Returns: DataFrame with xP, xP_5gw, algorithm_version, computed_at, target_gameweek

# Batch recompute across multiple gameweeks and algorithms
comparison_df = analytics_service.batch_recompute_season(
    start_gw=1,
    end_gw=7,
    algorithm_versions=["current", "experimental_high_form", "experimental_low_form"],
    include_snapshots=True
)
# Returns: Multi-index DataFrame (gameweek, player_id, algorithm_version) → xP
```

#### 4. Accuracy Metrics Calculation

```python
# Get actual results from dataset-builder
from client import FPLDataClient
client = FPLDataClient()
actual_results = client.get_gameweek_performance(5)

# Calculate accuracy metrics
accuracy_metrics = analytics_service.calculate_accuracy_metrics(
    predictions_df=predictions_gw5,
    actual_results_df=actual_results,
    by_position=True  # Position-specific metrics
)

# Returns:
# {
#   "overall": {"mae": 2.1, "rmse": 2.8, "correlation": 0.65, ...},
#   "by_position": {
#     "GKP": {"mae": 1.5, "rmse": 2.0, "correlation": 0.70, ...},
#     "DEF": {"mae": 2.0, "rmse": 2.5, "correlation": 0.60, ...},
#     "MID": {"mae": 2.3, "rmse": 3.0, "correlation": 0.62, ...},
#     "FWD": {"mae": 2.5, "rmse": 3.2, "correlation": 0.58, ...}
#   },
#   "player_count": 587
# }
```

#### 5. Accuracy Tracking Visualizations

**Model Accuracy Tracking** - Visualize accuracy trends across historical gameweeks:

```python
from fpl_team_picker.visualization.charts import create_model_accuracy_visualization

# Track model accuracy over last 5 completed gameweeks
accuracy_viz = create_model_accuracy_visualization(
    target_gameweek=8,  # Current gameweek
    lookback_gameweeks=5,  # Analyze GW3-7
    algorithm_versions=["current"],  # Can compare multiple algorithms
    mo_ref=mo  # Marimo reference
)
# Returns: Interactive charts with MAE trends, correlation analysis, and gameweek breakdown
```

**Position-Specific Accuracy** - Analyze accuracy by position (GKP, DEF, MID, FWD):

```python
from fpl_team_picker.visualization.charts import create_position_accuracy_visualization

# Analyze position-specific accuracy for a completed gameweek
position_accuracy = create_position_accuracy_visualization(
    target_gameweek=7,  # Must be completed
    algorithm_version="current",
    mo_ref=mo
)
# Returns: Position-wise MAE/correlation charts and breakdown tables
```

**Algorithm Comparison** - Compare multiple algorithm versions to find best performer:

```python
from fpl_team_picker.visualization.charts import create_algorithm_comparison_visualization

# Compare algorithm performance across historical gameweeks
comparison = create_algorithm_comparison_visualization(
    start_gw=3,
    end_gw=7,
    algorithm_versions=["current", "experimental_high_form", "experimental_low_form"],
    mo_ref=mo
)
# Returns: Side-by-side comparison with winner recommendations
```

**Integration with Gameweek Manager:**

The accuracy tracking visualizations can be added to the gameweek manager interface:

```python
# In gameweek_manager.py
@app.cell
def _(target_gameweek, mo):
    from fpl_team_picker.visualization.charts import create_model_accuracy_visualization

    # Show accuracy tracking for completed gameweeks
    if target_gameweek > 1:
        accuracy_display = create_model_accuracy_visualization(
            target_gameweek=target_gameweek,
            lookback_gameweeks=5,
            mo_ref=mo
        )
    else:
        accuracy_display = mo.md("*No completed gameweeks for accuracy tracking*")

    return accuracy_display,
```

### Use Cases

**1. Algorithm Optimization:**
```python
# Test 3 different form weights on GW1-7
results = analytics_service.batch_recompute_season(
    start_gw=1, end_gw=7,
    algorithm_versions=["current", "experimental_high_form", "experimental_low_form"]
)

# Compare accuracy for each algorithm version
for algo in ["current", "experimental_high_form", "experimental_low_form"]:
    algo_predictions = results.xs(algo, level="algorithm_version")
    metrics = analytics_service.calculate_accuracy_metrics(
        algo_predictions, actual_results
    )
    print(f"{algo}: MAE={metrics['overall']['mae']}")
```

**2. Transfer ROI Analysis:**
```python
# Reconstruct what xP would have predicted at GW3 (before transfer decision)
gw3_predictions = analytics_service.recompute_historical_xp(target_gameweek=3)

# Compare transferred-in vs transferred-out players
# Calculate net points over 5GW horizon
# Identify systematic transfer biases
```

**3. Model Validation:**
```python
# Validate current algorithm against last 10 gameweeks
validation_results = []
for gw in range(1, 11):
    predictions = analytics_service.recompute_historical_xp(target_gameweek=gw)
    actual = client.get_gameweek_performance(gw)
    metrics = analytics_service.calculate_accuracy_metrics(predictions, actual)
    validation_results.append(metrics)

# Analyze accuracy trends over time
```

### Testing

```bash
# Run historical recomputation and accuracy tracking tests
pytest tests/domain/services/test_performance_analytics_service.py::TestAccuracyTracking -v

# Test coverage includes:
# - Historical data loading with temporal consistency
# - Algorithm version validation
# - Single gameweek recomputation
# - Batch recomputation across gameweeks and algorithms
# - Accuracy metrics calculation (overall + position-specific)
# - Algorithm version registry validation
# - Error handling for invalid inputs and missing data

# Run all performance analytics tests
pytest tests/domain/services/test_performance_analytics_service.py -v
```

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
# Lint and fix code issues
ruff check fpl_team_picker/ --fix

# Format code
ruff format fpl_team_picker/

# Check and fix marimo notebooks
marimo check fpl_team_picker/interfaces/ --fix

# Find unused code
vulture fpl_team_picker/

# Run domain service tests
pytest tests/domain/services/ -v

# Run integration tests
pytest tests/domain/services/test_integration.py -v
```

## Technical Specifications

**Dependencies:**
- Python 3.13+
- marimo>=0.16.2 (interactive notebook interface - upgraded)
- pandas>=2.3.1 (data manipulation)
- numpy>=2.3.2 (numerical computations)
- plotly>=6.3.0 (interactive visualizations)
- pydantic>=2.11.0 (configuration validation and domain models)
- requests>=2.31.0 (HTTP client)
- xgboost>=3.0.5 (machine learning for expected points models)
- scikit-learn>=1.7.2 (ML utilities and preprocessing)
- pytest>=8.0.0 (testing framework)
- ruff>=0.12.10 (linting and formatting)
- fpl-dataset-builder (local dependency for historical data)

**Key Architectural Features:**
- **Clean Architecture** - Domain, infrastructure, and presentation layer separation
- **Frontend-Agnostic Design** - Ready for FastAPI, React, CLI, mobile frontends
- **Boundary Validation Pattern** - Fail fast at data boundaries, deterministic business logic
- **Domain Services** - Pure business logic with zero framework dependencies
- **Comprehensive Testing** - Domain service unit tests and integration tests
- **Type-Safe Configuration** - Pydantic models with validation and environment overrides
- **Repository Pattern** - Abstract data access with concrete implementations
- **Interactive Notebooks** - Marimo-based presentation adapters (33% code reduction achieved)
- **CLI Integration** - Command-line entry points for all interfaces

## Data Contract & Boundary Validation Principles

### "Fail Fast, Validate Once" Pattern

The system implements a strict **boundary validation pattern** where all data validation occurs at the boundary (DataOrchestrationService), ensuring deterministic business logic downstream.

#### Core Principle: NO FALLBACKS
**❌ Never use fallbacks or silent error handling:**
```python
# WRONG - Silent failure hides data integrity issues
team_name = team_map.get(team_id, "Unknown Team")
player_data = data.fillna("Default Value")
```

**✅ Always fail fast with actionable error messages:**
```python
# CORRECT - Explicit validation with clear error messages
if not set(team_values).issubset(available_teams):
    missing_teams = set(team_values) - available_teams
    raise ValueError(f"Data contract violation: team IDs {missing_teams} not found in teams data")
```

#### Boundary Validation Contract

**DataOrchestrationService** - THE validation boundary:
```python
def load_gameweek_data(self, target_gameweek: int) -> Dict[str, Any]:
    """FAIL FAST if data is bad - guarantee clean data downstream"""

    # Comprehensive validation here
    if len(teams) != 20:
        raise ValueError(f"Invalid team data: expected 20 teams, got {len(teams)}")

    if players.empty:
        raise ValueError("No player data available")

    # Validate team ID consistency
    player_teams = set(players["team_id"].unique())
    available_teams = set(teams["team_id"].unique())
    if not player_teams.issubset(available_teams):
        missing = player_teams - available_teams
        raise ValueError(f"Team ID mismatch: players reference teams {missing} not in teams data")

    # Return: "Clean data dictionary - guaranteed to be valid"
    return validated_data
```

**Domain Services** - Trust the data contract:
```python
def calculate_expected_points(self, gameweek_data: Dict[str, Any]) -> pd.DataFrame:
    """Args: gameweek_data - guaranteed clean data
    Returns: DataFrame with expected points - deterministic"""

    # No validation needed - trust the contract
    players = gameweek_data["players"]  # Known to be valid
    return mathematical_transformation(players)
```

#### Data Integrity Enforcement

**In optimization and display layers:**
```python
# Validate data contract before mapping operations
team_values = df["team"].unique()
available_teams = set(teams["team_id"])

if not set(team_values).issubset(available_teams):
    missing_teams = set(team_values) - available_teams
    raise ValueError(f"Data contract violation: team IDs {missing_teams} not found")

# Only proceed with mapping if validation passes
df["name"] = df["team"].map(team_map)  # Guaranteed to work
```

#### Benefits of This Approach

1. **Deterministic Behavior** - No silent failures or unpredictable fallbacks
2. **Fast Debugging** - Clear error messages pinpoint exact data issues
3. **Data Quality Assurance** - Forces upstream data problems to be fixed
4. **Maintainable Code** - Business logic can trust data contracts
5. **No Technical Debt** - Prevents accumulation of workaround code

#### Common Anti-Patterns to Avoid

```python
# ❌ ANTI-PATTERN: Silent fallbacks hide problems
df["name"] = df["team"].map(team_map).fillna("Unknown")

# ❌ ANTI-PATTERN: Generic error handling
try:
    df["name"] = df["team"].map(team_map)
except:
    df["name"] = "Error"

# ❌ ANTI-PATTERN: Default value masking
team_name = team_map.get(team_id, "N/A")
```

```python
# ✅ CORRECT PATTERN: Explicit validation and clear failures
if team_id not in team_map:
    raise ValueError(f"Team ID {team_id} not found in teams data. Available: {list(team_map.keys())}")
team_name = team_map[team_id]
```

This pattern ensures that when something fails, you get **actionable debugging information** rather than mysterious "object" values or silent data corruption.

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

## Clean Architecture Patterns & Frontend Migration

### **Boundary Validation Pattern - "Fail Fast, Validate Once"**

The system implements a sophisticated validation pattern where all data validation occurs at the boundary, ensuring deterministic business logic:

```python
# DataOrchestrationService - THE validation boundary
def load_gameweek_data(self, target_gameweek: int) -> Dict[str, Any]:
    """FAIL FAST if data is bad - guarantee clean data downstream"""

    # Comprehensive validation here
    if len(teams) != 20:
        raise ValueError(f"Invalid team data: expected 20 teams, got {len(teams)}")

    # Return: "Clean data dictionary - guaranteed to be valid"
    return validated_data

# ExpectedPointsService - Trusts the data contract
def calculate_expected_points(self, gameweek_data: Dict[str, Any]) -> pd.DataFrame:
    """Args: gameweek_data - guaranteed clean data
    Returns: DataFrame with expected points - deterministic"""

    # No validation needed - trust the contract
    players = gameweek_data["players"]  # Known to be valid
    return mathematical_transformation(players)
```

### **Frontend Migration Readiness**

The clean architecture enables seamless migration to any frontend technology:

```python
# 1. Current: Marimo Notebook
@app.cell
def _(gameweek_input, mo):
    orchestration_service = DataOrchestrationService()
    gameweek_data = orchestration_service.load_gameweek_data(gameweek_input.value)

    xp_service = ExpectedPointsService()
    players_with_xp = xp_service.calculate_combined_results(gameweek_data)

    return create_marimo_display(players_with_xp)

# 2. Future: FastAPI Backend
@app.post("/analyze-gameweek")
def analyze_gameweek(request: GameweekRequest):
    orchestration_service = DataOrchestrationService()
    gameweek_data = orchestration_service.load_gameweek_data(request.gameweek)

    xp_service = ExpectedPointsService()
    players_with_xp = xp_service.calculate_combined_results(gameweek_data)

    return {"status": "success", "data": players_with_xp.to_dict()}

# 3. Future: React Frontend (via API)
const GameweekAnalysis = ({ gameweek }) => {
    const { data: analysisData } = useQuery(
        ['gameweek-analysis', gameweek],
        () => fetch(`/api/analyze-gameweek`, {
            method: 'POST',
            body: JSON.stringify({ gameweek })
        }).then(res => res.json())
    );

    return <AnalysisDisplay data={analysisData} />;
};

# 4. Future: CLI Interface
def cli_analyze_gameweek(gameweek: int):
    orchestration_service = DataOrchestrationService()
    gameweek_data = orchestration_service.load_gameweek_data(gameweek)

    xp_service = ExpectedPointsService()
    players_with_xp = xp_service.calculate_combined_results(gameweek_data)

    print(format_cli_table(players_with_xp))
```

### **Testing Strategy**

**Domain Service Testing** - Complete isolation from UI:
```python
# tests/domain/services/test_integration.py
def test_expected_points_service_integration(data_service, xp_service):
    """Test XP service with real data - no UI dependencies"""
    gameweek_data = data_service.load_gameweek_data(target_gameweek=1)

    players_with_xp = xp_service.calculate_combined_results(gameweek_data)

    assert isinstance(players_with_xp, pd.DataFrame)
    assert players_with_xp["xP"].min() >= 0
    assert players_with_xp["xP"].max() <= 20
```

**Interface Compatibility Testing**:
```python
# tests/interfaces/test_marimo_compatibility.py
def test_gameweek_manager_loads_without_errors():
    """Ensure marimo interface works with domain services"""
    # Test marimo notebook can import and use domain services
    from fpl_team_picker.domain.services import DataOrchestrationService
    service = DataOrchestrationService()
    assert service is not None
```

## Development Workflow

### **Current Development (Clean Architecture)**
1. **Season Start**: Use season planner for initial 15-player squad building
2. **Weekly Planning**: Use clean gameweek manager (969 lines) with domain service orchestration
3. **Post-Gameweek**: Analyze results using domain services independently
4. **ML Model Development**: Use ML experiment interface for model improvement
5. **Domain Service Development**: Add/modify business logic in isolated domain services
6. **Frontend Experimentation**: Build new interfaces using existing domain services

### **Future Development (Multi-Frontend)**
1. **FastAPI Backend**: Expose domain services as REST API endpoints
2. **React Frontend**: Build modern web interface consuming REST API
3. **Mobile Development**: iOS/Android apps using same business logic via API
4. **CLI Tools**: Command-line utilities for power users and automation
5. **Third-Party Integrations**: Enable other developers to use FPL analysis services

### **Testing & Validation Workflow**
```bash
# 1. Domain service testing (business logic)
pytest tests/domain/services/ -v

# 2. Integration testing (end-to-end)
pytest tests/domain/services/test_integration.py -v

# 3. Interface compatibility testing
pytest tests/interfaces/ -v

# 4. Code quality and formatting
ruff check fpl_team_picker/ --fix
marimo check fpl_team_picker/interfaces/ --fix
```

The clean architecture transformation enables rapid iteration across multiple frontends while maintaining a single source of truth for all FPL analysis logic. Domain services can be developed, tested, and deployed independently of any specific UI technology.
- Remember to run ruff formatting and linting/checking. Create tests parsimoniously but where necessary don't omit them. Stop to make commits. Remember to update @CLAUDE.md
- Use pydantic instead of dataclass.
- Don't do fallbacks in application code when the issue lies in upstream data quality problems.
