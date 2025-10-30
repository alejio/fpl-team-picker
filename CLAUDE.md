# CLAUDE.md

Fantasy Premier League (FPL) analysis suite with season-start team building and weekly gameweek management. Clean architecture with domain-driven design, implementing mathematical frameworks for expected points, optimization, and fixture analysis.

## Architecture

**Clean Architecture**: domain/infrastructure/presentation separation. Business logic in `domain/services/`.

**Key Services**:
- **DataOrchestrationService**: Boundary validation, guaranteed clean data
- **ExpectedPointsService**: Rule-based XP (1GW + 5GW projections)
- **MLExpectedPointsService**: Pre-trained ML predictions (99 features)
- **TeamAnalyticsService**: Dynamic team strength ratings
- **OptimizationService**: Simulated annealing for transfers, XI selection, captain picks
- **ChipAssessmentService**: Traffic light chip recommendations
- **PerformanceAnalyticsService**: Historical recomputation & accuracy tracking
- **PlayerAnalyticsService**: Type-safe player operations (70+ attributes)

**Interfaces** (`interfaces/`): Marimo notebooks (gameweek_manager.py, season_planner.py, ml_xp_experiment.py)

## Data Loading

Uses `fpl-dataset-builder` client: `from client import FPLDataClient`

**Key methods**: `get_current_players()`, `get_current_teams()`, `get_fixtures_normalized()`, `get_gameweek_performance(gw)`, `get_derived_ownership_trends()`, `get_derived_value_analysis()`, `get_derived_fixture_difficulty()`, `get_derived_betting_features()`

Install: `uv sync`

## ML Feature Engineering (99 features)

**FPLFeatureEngineer** (sklearn transformer):
- **Base (65)**: Cumulative stats, rolling 5GW form, per-90 rates, team context, fixture features
- **Enhanced (15 - #37)**: Ownership trends (7), value analysis (5), enhanced fixture difficulty (3)
- **Penalty/set-piece (4)**: Primary/backup penalty takers, corner/FK takers
- **Betting odds (15 - #38)**: Implied probabilities (6), market confidence (4), Asian Handicap (3), match context (2)

**Data sources**:
- Base: `get_current_players()`, `get_gameweek_performance(gw)`, `get_fixtures_normalized()`
- Enhanced: `get_derived_ownership_trends()`, `get_derived_value_analysis()`, `get_derived_fixture_difficulty()`
- Betting: `get_derived_betting_features()`

**Leak-free**: All features use shift(1) or historical lookback. Betting odds are forward-looking (available pre-match) - no shift needed.

**TPOT**: `scripts/tpot_pipeline_optimizer.py` - Automated ML pipeline discovery with temporal CV
Quick start: `uv run python scripts/tpot_pipeline_optimizer.py --start-gw 1 --end-gw 8 --generations 10`

## Expected Points Models

**PRIMARY: ML-Based** (`MLExpectedPointsService`): Pre-trained sklearn pipelines (99 features), position-specific models, temporal validation. Requires .joblib artifact. Train using TPOT or ml_xp_experiment.py.

**Legacy: Rule-Based** (`ExpectedPointsService`): Form-weighted (70/30), live data, dynamic team strength. Used for GW1-5 (insufficient ML training data) and as ML benchmark only.

## Historical xP Recomputation

Retroactive testing with arbitrary algorithms for A/B comparisons.

**Key APIs**:
- `DataOrchestrationService.load_historical_gameweek_state()`
- `PerformanceAnalyticsService.recompute_historical_xp(gw)`
- `PerformanceAnalyticsService.batch_recompute_season()`
- Metrics: MAE/RMSE/correlation by position

## CLI Commands

`fpl-season-planner` | `fpl-gameweek-manager` | `fpl-xp-accuracy` | `fpl-ml-experiment`

Or: `marimo run fpl_team_picker/interfaces/{notebook}.py`

## Development

**Setup**: `uv sync`

**Code Quality**:
```bash
ruff check fpl_team_picker/ && ruff format fpl_team_picker/
marimo check fpl_team_picker/interfaces/ --fix
```

**Tests**: `pytest tests/domain/services/ -v`

## Configuration

Pydantic models with env var overrides (`FPL_{SECTION}_{FIELD}`). Override via `config.json`.

Config sections: XPModel, TeamStrength, MinutesModel, StatisticalEstimation, Optimization, Visualization, ChipAssessment.

## Data Contract & Boundary Validation

**"Fail Fast, Validate Once"** - All validation at DataOrchestrationService boundary.

**NO FALLBACKS**: ❌ `.get(key, "default")`, `.fillna()` → ✅ Explicit validation with clear error messages

Benefits: Deterministic behavior, fast debugging, data quality assurance.

## FPL Rules

See `fpl_rules.md`. Key: 2 GKP, 5 DEF, 5 MID, 3 FWD; £100m budget; max 3 players/team; valid formations.

## Performance Benchmark

2024-25 winner: 2,810 points (74/GW avg). Target: ~370 pts over 5GW. Premium 8-12 xP, budget 4-7 xP.

## Development Guidelines

- Run ruff formatting/linting. Create tests where necessary. Make commits. Update CLAUDE.md.
- Use pydantic (not dataclass) for domain models
- No fallbacks when issue is upstream data quality - fail fast with clear errors
- Prefer domain models for UI/display (type safety), DataFrames for performance-critical ops (optimization, ML)
- **All business logic in `domain/services/`** - No business logic in interfaces
- **Separate services for separate concerns** - Keep focused (~1,000 lines each)
- **ML and rule-based are separate services** - ExpectedPointsService vs MLExpectedPointsService

## Tech Stack

Python 3.13+, marimo, pandas, numpy, plotly, pydantic, xgboost, scikit-learn, lightgbm, tpot, pytest, ruff, fpl-dataset-builder

**Architecture**: Clean Architecture, frontend-agnostic, boundary validation, domain services, type-safe Pydantic models, repository pattern, 29/29 tests passing
