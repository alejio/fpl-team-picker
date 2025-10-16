# CLAUDE.md

Fantasy Premier League (FPL) analysis suite with season-start team building and weekly gameweek management. Clean architecture with domain-driven design, implementing mathematical frameworks for expected points, optimization, and fixture analysis.

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
│   │   │   ├── expected_points_service.py       # Rule-based XP calculation
│   │   │   ├── ml_expected_points_service.py    # ML-based XP calculation
│   │   │   ├── team_analytics_service.py        # Dynamic team strength
│   │   │   ├── optimization_service.py          # FPL optimization algorithms
│   │   │   ├── transfer_optimization_service.py # Transfer orchestration
│   │   │   ├── squad_management_service.py      # Squad analysis & management
│   │   │   ├── chip_assessment_service.py       # Chip timing analysis
│   │   │   ├── performance_analytics_service.py # Historical recomputation & accuracy
│   │   │   └── player_analytics_service.py      # Type-safe player operations
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
│   ├── visualization/        # Charts and visual components
│   │   └── charts.py         # Plotly-based interactive visualizations (with display helpers)
│   └── interfaces/           # User interfaces (Frontend adapters)
│       ├── season_planner.py # Season-start team building interface (2257 lines)
│       ├── gameweek_manager.py # Weekly gameweek management interface (969 lines - clean)
│       └── ml_xp_experiment.py # ML expected points model development (2947 lines)
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

Uses fpl-dataset-builder client library (`from client import FPLDataClient`).

**Key methods**: `get_current_players()`, `get_current_teams()`, `get_fixtures_normalized()`, `get_player_xg_xa_rates()`, `get_gameweek_live_data(gw)`, `get_gameweek_performance(gw)`, `get_player_deltas_current()`

Install: `uv sync` (local dependency configured in pyproject.toml)

## Core Architecture

Clean Architecture with domain/infrastructure/presentation separation. All business logic in `domain/services/`.

### Domain Services

- **DataOrchestrationService**: Boundary validation, guaranteed clean data
- **ExpectedPointsService**: Rule-based XP (1GW + 5GW projections)
- **MLExpectedPointsService**: Ridge regression ML predictions
- **TeamAnalyticsService**: Dynamic team strength ratings
- **OptimizationService**: Transfer optimization, starting XI, captain selection, squad generation
- **ChipAssessmentService**: Traffic light chip recommendations
- **PerformanceAnalyticsService**: Form analysis, historical recomputation
- **PlayerAnalyticsService**: Type-safe player operations (70+ attributes)

### Domain Models & Repositories

- **EnrichedPlayerDomain**: 70+ validated attributes (47 enhanced + 29 derived + computed properties)
- **PlayerRepository/TeamRepository/FixtureRepository**: Abstract data access
- **DatabasePlayerRepository** (adapters): Concrete implementation via fpl-dataset-builder

### Configuration (`config/`)

Pydantic models with env var overrides (`FPL_{SECTION}_{FIELD}`). Config sections: XPModel, TeamStrength, MinutesModel, StatisticalEstimation, Optimization, Visualization, ChipAssessment.

### Presentation Layer (`interfaces/`)

Marimo notebooks orchestrating domain services (zero business logic):
- **gameweek_manager.py** (969 lines): Weekly management with type-safe player operations
- **season_planner.py** (2257 lines): Initial squad building
- **ml_xp_experiment.py** (2947 lines): ML model development
- **xp_accuracy_tracking.py**: Model validation & algorithm A/B testing


## CLI Commands

`fpl-season-planner` | `fpl-gameweek-manager` | `fpl-xp-accuracy` | `fpl-ml-experiment`

Or: `marimo run fpl_team_picker/interfaces/{notebook}.py`

## MCP Server (AI Integration)

Marimo MCP (Model Context Protocol) server enables AI assistants like Claude Code to interact with notebooks programmatically.

**Quick Start**: `./start-mcp-server.sh`

**Manual Start**: `uv run marimo edit fpl_team_picker/interfaces/ml_xp_notebook.py --headless --no-token --port 2718 --mcp`

**Configuration**:
- MCP URL: `http://localhost:2718/mcp/server`
- Add to Claude Code: `claude mcp add --transport http marimo http://localhost:2718/mcp/server`
- Check status: `claude mcp list`

**Requirements**: `marimo[mcp]>=0.17.0` (already in pyproject.toml)

**Note**: MCP server only works with `marimo edit` (not `marimo run`), requires `--mcp` flag (hidden option)

## ML Development

**Historical data**: `get_gameweek_performance(gw)`, `get_player_gameweek_history()`, `get_my_picks_history()`

**Leak-free features**: Historical per-90 metrics, position-specific models (GKP/DEF/MID/FWD), temporal validation, team context features (rolling 5GW team form, cumulative team stats)

**Team features rationale**: Safe with player-based GroupKFold validation - testing "can we predict NEW players on KNOWN teams?" not future outcomes. All team features use shift(1) to exclude current gameweek.

## Expected Points Models

**Rule-Based** (`ExpectedPointsService`): Form-weighted (70/30), live data, dynamic team strength, 1GW+5GW projections. Fast, works from GW1+.

**ML-Based** (`MLExpectedPointsService`): Ridge regression, position-specific models, temporal validation, optional ensemble. Requires 3+ GWs.

## Historical xP Recomputation

Recompute historical xP with arbitrary algorithms for retroactive testing. Benefits: test algorithm variations, A/B comparisons, future-proof evaluation.

**Required data**: Historical prices, performance, fixtures, availability snapshots (`get_player_availability_snapshot(gw)`). Fail fast if missing.

**Key APIs**:
- `DataOrchestrationService.load_historical_gameweek_state()` - Load historical state
- `PerformanceAnalyticsService.recompute_historical_xp()` - Single gameweek
- `PerformanceAnalyticsService.batch_recompute_season()` - Multiple gameweeks/algorithms
- `calculate_accuracy_metrics()` - MAE/RMSE/correlation by position
- Visualization: `create_model_accuracy_visualization()`, `create_algorithm_comparison_visualization()`

**Algorithm versions**: v1.0, current, experimental_high_form, experimental_low_form (Pydantic configs)

Tests: `pytest tests/domain/services/test_performance_analytics_service.py -v`

## Development Commands

**Setup**: `uv sync`

**Code Quality**: `ruff check/format fpl_team_picker/`, `marimo check fpl_team_picker/interfaces/ --fix`, `vulture fpl_team_picker/`

**Tests**: `pytest tests/domain/services/ -v`, `pytest tests/domain/services/test_integration.py -v`

## Technical Specs

**Stack**: Python 3.13+, marimo, pandas, numpy, plotly, pydantic, xgboost, scikit-learn, pytest, ruff, fpl-dataset-builder

**Architecture**: Clean Architecture, frontend-agnostic, boundary validation, domain services, type-safe Pydantic models (70+ player attributes), repository pattern, 29/29 tests passing

## Data Contract & Boundary Validation

**"Fail Fast, Validate Once"** - All validation at DataOrchestrationService boundary, deterministic downstream.

**NO FALLBACKS**: ❌ `.get(key, "default")`, `.fillna()` → ✅ Explicit validation with clear error messages

Benefits: Deterministic behavior, fast debugging, data quality assurance, maintainable code

## FPL Rules

See `fpl_rules.md`. Key: 2 GKP, 5 DEF, 5 MID, 3 FWD; £100m budget; max 3 players/team; valid formations

## Configuration

Override via `config.json` or env vars (`FPL_{SECTION}_{FIELD}`)

## Performance Benchmark

2024-25 winner: 2,810 points (74/GW avg). Target: ~370 pts over 5GW. Premium 8-12 xP, budget 4-7 xP.

## Frontend Migration Ready

Domain services work with Marimo (current), FastAPI, React, CLI, or mobile. Zero UI dependencies in business logic. Test domain services independently.

## Domain Models (Issue #31 ✅)

**EnrichedPlayerDomain**: 70+ validated attributes (core, performance, advanced, market, set pieces, derived, computed properties)

**PlayerAnalyticsService**: Type-safe queries (`get_penalty_takers()`, `get_high_value_players()`, `get_injury_risks()`, etc.)

Full Pydantic validation, 29/29 tests passing, production-ready in gameweek_manager.py

---

## Development Guidelines

- Remember to run ruff formatting and linting/checking. Create tests parsimoniously but where necessary don't omit them. Stop to make commits. Remember to update @CLAUDE.md
- Use pydantic instead of dataclass for all domain models
- Don't do fallbacks in application code when the issue lies in upstream data quality problems
- Prefer domain models (`PlayerAnalyticsService`) for UI/display logic where type safety matters
- Use DataFrames directly for performance-critical operations (optimization, ML training) where domain model overhead provides no benefit
- **All business logic belongs in `domain/services/`** - The `core/` directory has been fully migrated and removed
- **Separate services for separate concerns** - Keep services focused (~1,000 lines each) rather than creating monolithic files
- **ML and rule-based models are separate services** - `ExpectedPointsService` (rule-based) and `MLExpectedPointsService` (ML) for maintainability
