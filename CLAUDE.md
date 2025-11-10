# CLAUDE.md

Fantasy Premier League (FPL) analysis suite for the 2025-26 season with season-start team building and weekly gameweek management. 2025-2026 season is in progress. We are competing in the FPL league and are using this tool to help us win. The tool is underperforming and needs to be improved. Clean architecture with domain-driven design, implementing mathematical frameworks for expected points, optimization, and fixture analysis.

## 2025-26 Season Context

**Promoted Teams** (from Championship): Leeds, Burnley, Sunderland
**Relegated Teams** (from Premier League): Ipswich, Leicester, Southampton

**Key Context**: Promoted teams often fight harder at home, making away fixtures tougher than raw team strength suggests. Avoid assuming promoted team fixtures are "easy" - they can be competitive, especially at their home grounds.

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

## ML Pipeline Training

**Reusable Training Infrastructure** (`scripts/ml_training_utils.py`):
- `load_training_data()` - Load all 8 data sources (historical, fixtures, teams, ownership, value, fixture difficulty, betting, raw players)
- `engineer_features()` - FPLFeatureEngineer with leak-free per-GW team strength (FIXED: data alignment bug)
- `create_temporal_cv_splits()` - Walk-forward validation (GW6→7, GW6-7→8, etc.)
- `evaluate_fpl_comprehensive()` - MAE/RMSE/Spearman/top-15 overlap/captain accuracy
- Ensures "apples to apples" comparison across experiments

**Custom ML Transformers** (`fpl_team_picker/domain/ml/`):
- `FeatureSelector` - Self-contained transformer that selects features by name (makes pipelines service-agnostic)
- Enables pipelines to accept all 99 features and internally select the subset they need
- Fixes pickle issues - properly importable from domain layer

**Custom Pipeline Optimizer** (`scripts/custom_pipeline_optimizer.py`):
- **Production Model**: `models/custom/random-forest_gw1-9_20251031_140131_pipeline.joblib` ✅ IN USE
- **Performance**: RandomForest with RFE-smart (60 features + penalties): MAE=0.632, Spearman=0.818
- **64% better than TPOT** (MAE 0.632 vs 1.752, same scorer/data/CV)
- **43.6% better on GW10 test data** (MAE 1.002 vs 1.777)
- **Correctly predicted Haaland > Kudus** (GW10: prevented -24 point captain disaster)
- 8 regressors: XGBoost, LightGBM, RandomForest, GradientBoosting, AdaBoost, Ridge, Lasso, ElasticNet
- 4 feature selection strategies: none, correlation, permutation, rfe-smart
- `--keep-penalty-features` flag to force-keep critical domain features
- Self-contained pipelines: FeatureSelector → StandardScaler → Regressor
- Quick start: `uv run python scripts/custom_pipeline_optimizer.py --regressor random-forest --feature-selection rfe-smart --keep-penalty-features --scorer fpl_weighted_huber`

**TPOT (Reference)** (`scripts/tpot_pipeline_optimizer.py`):
- MAE=1.752, Spearman=0.794 (trained 8hrs with fpl_weighted_huber)
- **Not recommended for production** - inferior accuracy, predicted Kudus > Haaland (incorrect)
- Use for comparison/exploration only
- Quick start: `uv run python scripts/tpot_pipeline_optimizer.py --start-gw 1 --end-gw 9 --scorer fpl_weighted_huber --max-time-mins 480`

**GW10 Empirical Validation** (`experiments/GW10_MODEL_COMPARISON_REPORT.md`):
- Custom RF vs TPOT tested on held-out GW10 data
- Custom RF: 43.6% better MAE, correctly ranked Haaland > Kudus
- TPOT: Systematically overestimated Kudus, leading to captain failure
- Conclusion: Custom RF is production-ready, TPOT deprecated

## Expected Points Models

**PRIMARY: ML-Based** (`MLExpectedPointsService`):
- Pre-trained sklearn pipelines (99 features), position-specific models, temporal validation
- **Uncertainty Quantification**: Random Forest tree-level variance (returns `xP_uncertainty` column)
- Extracts per-player prediction uncertainty from ensemble disagreement
- Requires .joblib artifact from custom_pipeline_optimizer.py
- Config: `config.xp_model.ml_model_path = "models/custom/random-forest_gw1-9_20251031_140131_pipeline.joblib"`

**Legacy: Rule-Based** (`ExpectedPointsService`):
- Form-weighted (70/30), live data, dynamic team strength
- Used for GW1-5 (insufficient ML training data) and as ML benchmark only
- No uncertainty quantification

## Captain Selection (Uncertainty-Aware)

**OptimizationService.get_captain_recommendation()** implements:
1. **Uncertainty Penalty**: Prefers players with lower prediction variance (reliable picks)
   - Formula: `risk_adjusted_score = (xP * 2) / (1 + uncertainty_penalty)`
   - High uncertainty (>40% of xP) = reduced captain score
2. **Template Protection**: High ownership (>50%) players get 5-10% bonus
   - Protects against rank swings if template captain hauls
3. **Risk Assessment**: Includes uncertainty in risk factors display
4. **Full Transparency**: Returns uncertainty metrics in candidate analysis

**Key Feature**: Would have prevented GW10 disaster (-24 points from Kudus captain)

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

uv (don't forget to use uv), Python 3.13+, marimo, pandas, numpy, plotly, pydantic, xgboost, scikit-learn, lightgbm, tpot, pytest, ruff, fpl-dataset-builder

**Architecture**: Clean Architecture, frontend-agnostic, boundary validation, domain services, type-safe Pydantic models, repository pattern, 29/29 tests passing
