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

**Fixture Analysis Service** (`fixture_analysis_service.py`)
```python
from fpl_team_picker.domain.services import FixtureAnalysisService

# Fixture difficulty and scheduling analysis
fixture_service = FixtureAnalysisService()
fixture_analysis = fixture_service.analyze_fixtures(
    target_gameweek=10, gameweeks_ahead=5
)
# Returns: Fixture difficulty ratings, double gameweeks, rotation risks
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

**Visualization Service** (`visualization_service.py`)
```python
from fpl_team_picker.domain.services import VisualizationService

# Chart generation for any frontend
viz_service = VisualizationService()
chart_config = viz_service.generate_xp_chart_config(players_with_xp)
# Returns: Frontend-agnostic chart configurations (Plotly, React, etc.)
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
- **Fixture Analysis**: Uses `FixtureAnalysisService` for difficulty heatmaps

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
