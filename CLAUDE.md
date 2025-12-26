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
- **XPCalibrationService**: Probabilistic calibration using additive fixture effect correction
- **TeamAnalyticsService**: Dynamic team strength ratings
- **OptimizationService**: SA for transfers (exploratory), XI selection, captain picks
- **ChipAssessmentService**: Traffic light chip recommendations
- **PerformanceAnalyticsService**: Historical recomputation & accuracy tracking
- **PlayerAnalyticsService**: Type-safe player operations (70+ attributes)
- **PredictionStorageService**: Save/load committed predictions for accurate performance tracking
- **AFCONExclusionService**: Tournament-based player exclusions (AFCON 2025: GW17-22)
- **TransferPlanningAgentService**: LLM-based single-GW transfer recommendations using Claude Agent SDK

**Interfaces** (`interfaces/`): Marimo notebooks (gameweek_manager.py)

## Data Loading

Uses `fpl-dataset-builder` client: `from client import FPLDataClient`

**Key methods**: `get_current_players()`, `get_current_teams()`, `get_fixtures_normalized()`, `get_gameweek_performance(gw)`, `get_derived_ownership_trends()`, `get_derived_value_analysis()`, `get_derived_fixture_difficulty()`, `get_derived_betting_features()`

Install: `uv sync`

## ML Feature Engineering (156 features)

**FPLFeatureEngineer** (sklearn transformer):
- **Base (65)**: Cumulative stats, rolling 5GW form, per-90 rates, team context, fixture features
- **Enhanced (15 - #37)**: Ownership trends (7), value analysis (5), enhanced fixture difficulty (3)
- **Penalty/set-piece (4)**: Primary/backup penalty takers, corner/FK takers
- **Betting odds (15 - #38)**: Implied probabilities (6), market confidence (4), Asian Handicap (3), match context (2)
- **Injury & rotation risk (5 - Phase 1)**: injury_risk, rotation_risk, chance_of_playing_next_round, status_encoded, overperformance_risk
- **Venue-specific team strength (6 - Phase 2)**: home_attack_strength, away_attack_strength, home_defense_strength, away_defense_strength, home_advantage, venue_consistency
- **Player rankings & context (7 - Phase 3)**: form_rank, ict_index_rank, points_per_game_rank, defensive_contribution, tackles, recoveries, form_momentum
- **Elite player × fixture interactions (4 - Phase 4)**: is_elite (price>=£10m), elite_x_fixture_difficulty, elite_x_opponent_strength, elite_x_is_away

**Data sources**:
- Base: `get_current_players()`, `get_gameweek_performance(gw)`, `get_fixtures_normalized()`
- Enhanced: `get_derived_ownership_trends()`, `get_derived_value_analysis()`, `get_derived_fixture_difficulty()`
- Betting: `get_derived_betting_features()`
- Phase 1: `get_derived_player_metrics()`, `get_player_availability_snapshot(gw)`
- Phase 2: `get_derived_team_form()`
- Phase 3: `get_players_enhanced()`

**Leak-free**: All features use shift(1) or historical lookback. Betting odds are forward-looking (available pre-match) - no shift needed.

**Imputation Strategy (Phase 4)**: Domain-aware defaults replace generic `.fillna(0)`:
- Risk/Probability: injury_risk=0.1, rotation_risk=0.2, chance_of_playing=100
- Rankings: -1 for "unranked" instead of 0
- Status: 0 (available) as default
- Position-aware: DEF/MID have higher defaults for tackles/recoveries

## ML Pipeline Training

**Unified Training CLI** (`scripts/train_model.py`):
```bash
# Train unified model (default scorer: neg_mean_absolute_error)
uv run python scripts/train_model.py unified --end-gw 14 --regressor lightgbm

# Train with hauler-focused scorer (recommended for production)
uv run python scripts/train_model.py unified --end-gw 14 --regressor random-forest --scorer fpl_hauler_capture

# Train position-specific models
uv run python scripts/train_model.py position --end-gw 14

# Full pipeline (evaluate → determine hybrid config → retrain on all data)
# PARALLELIZED: Unified models train in parallel (default: 4 workers)
uv run python scripts/train_model.py full-pipeline --end-gw 14 --holdout-gws 2

# Full pipeline with custom parallelization (e.g., 8 workers for high-CPU systems)
uv run python scripts/train_model.py full-pipeline --end-gw 14 --holdout-gws 2 --max-workers 8

# Evaluate existing model
uv run python scripts/train_model.py evaluate --model-path models/hybrid/model.joblib --end-gw 14
```

**ML Training Module** (`fpl_team_picker/domain/services/ml_training/`):
- `TrainingConfig`, `HybridPipelineConfig` - Pydantic config models with validation
- `MLTrainer` - Core training orchestrator (unified, position-specific, all positions)
- `ModelEvaluator` - Evaluation metrics, hybrid config determination
- `build_pipeline()`, `get_param_space()` - Pipeline construction utilities
- **Parallel Training**: `full-pipeline` trains unified regressors in parallel (ProcessPoolExecutor)
  - Default: 4 workers, customizable via `--max-workers`
  - Speedup: ~4x faster for Step 1a (e.g., 40min → 10min with 4 regressors)

**Hybrid Model** (`fpl_team_picker/domain/ml/`):
- `HybridPositionModel` - Routes predictions to position-specific or unified models
- `FeatureSelector` - Self-contained transformer for feature selection by name
- Based on experiment results: GKP/FWD use position-specific, DEF/MID use unified
- **Position-Specific Features**: `POSITION_FEATURE_ADDITIONS` includes BOTH current and legacy features for backward compatibility with old models

**Training Utilities** (`scripts/ml_training_utils.py`):
- `load_training_data()` - Load all 12 data sources
- `engineer_features()` - FPLFeatureEngineer with leak-free per-GW team strength
- `create_temporal_cv_splits()` - Walk-forward validation (GW6→7, etc.)
- `evaluate_fpl_comprehensive()` - MAE/RMSE/Spearman/top-15 overlap/captain accuracy

**Hyperparameter Configuration**:
- XGBoost/LightGBM: n_estimators 200-1500, learning_rate 0.03-0.3 (log-uniform)
- RandomForest/GradientBoosting: Standard sklearn param spaces
- 4 feature selection strategies: none, correlation, permutation, rfe-smart
- Self-contained pipelines: FeatureSelector → StandardScaler → Regressor

**Position-Specific Feature Exclusions** (`POSITION_FEATURE_EXCLUSIONS`):
- **Logic-based filtering**: Only excludes features that are impossible for each position
- **GKP** (13 excluded): Player goals/xG, threat, creativity, tackles/recoveries → ~109 features
- **DEF** (1 excluded): saves → ~121 features
- **MID** (1 excluded): saves → ~121 features
- **FWD** (6 excluded): saves, clean sheets (FWD don't get CS points) → ~116 features
- **Team-level features kept**: Team stats (goals, xG, clean sheets) retained for all positions
- **Applied before feature selection**: Exclusions happen first, then permutation/RFE filters further

**Training Scorers** (`scripts/ml_training_utils.py`):
- `neg_mean_absolute_error`: Default sklearn MAE (baseline)
- `fpl_hauler_capture`: **Recommended** - Hauler precision@15 + point efficiency + captain accuracy
- `fpl_hauler_ceiling`: Hauler capture + variance preservation (prevents prediction compression)
- `fpl_weighted_huber`: Huber loss with premium/budget weighting
- `fpl_topk`, `fpl_captain`: Legacy ranking-focused scorers

**Scorer-Consistent Evaluation**: When training with custom scorers, `full-pipeline` now:
- Reports both the scorer value AND MAE for transparency
- Uses the scorer (not MAE) for model comparison/selection
- Example: `✅ random-forest: fpl_hauler_ceiling=0.4015, MAE=1.08`

**Multi-Model Comparison Tool** (`scripts/compare_all_models.py`):
- **Compare all 5 uncertainty-supporting models in one command**
- Trains & evaluates RandomForest, XGBoost, LightGBM, GradientBoosting, AdaBoost
- Automatic ranking by composite score (MAE 40%, Spearman 30%, RMSE 20%, Captain Accuracy 10%)
- Parallel execution support (`--parallel` flag for simultaneous training)
- Saves comparison report to `models/comparisons/comparison_{timestamp}.json`
- Identifies best model automatically with comprehensive metrics

**Top Manager Analysis** (`scripts/fetch_top_manager_ids.py`):
- Fetches top 1% manager IDs from FPL API for strategy analysis
- Usage: `uv run python scripts/fetch_top_manager_ids.py [num_managers]`
- Saves to `experiments/top_manager_ids.json` and `experiments/top_manager_ids.txt`

## Expected Points Models

**PRIMARY: ML-Based** (`MLExpectedPointsService`):
- Pre-trained sklearn pipelines (99 features), position-specific models, temporal validation
- **Uncertainty Quantification**: Tree-level variance for ensemble models (returns `xP_uncertainty` column)
  - **Random Forest**: Standard deviation across individual tree predictions
  - **XGBoost**: Standard deviation of incremental tree contributions, scaled by learning rate
  - **LightGBM**: Standard deviation across individual tree predictions, scaled by learning rate
  - **GradientBoosting**: Standard deviation of tree contributions from staged predictions, scaled by learning rate
  - **AdaBoost**: Standard deviation of estimator contributions from staged predictions
  - **Ridge/Lasso/ElasticNet**: Returns zero uncertainty (non-ensemble models)
- Extracts per-player prediction uncertainty from ensemble disagreement
- Requires .joblib artifact from custom_pipeline_optimizer.py
- Config: `config.xp_model.ml_model_path = "models/custom/random-forest_gw1-14_20251205_205638_pipeline.joblib"`
- Current model trained with `fpl_hauler_capture` scorer on GW1-14 data

**Legacy: Rule-Based** (`ExpectedPointsService`):
- Form-weighted (70/30), live data, dynamic team strength
- Used for GW1-5 (insufficient ML training data) and as ML benchmark only
- No uncertainty quantification

**Fixture Difficulty Consistency Architecture**:
- **Single Source of Truth**: `fixture_difficulty` is computed ONCE by `FPLFeatureEngineer.transform()`


## XP Calibration (Probabilistic)

**XPCalibrationService** (`domain/services/xp_calibration_service.py`):
- **Additive fixture effect correction** for ML predictions
- Addresses ML underweighting fixture difficulty for premium players (fixture difficulty is just 1 of 150+ features)
- Uses historical (tier × fixture) distributions from `data/calibration/distributions.json`

**Approach**:
1. Map player to (tier, fixture) combination based on price and fixture difficulty
2. Get fitted distribution mean for that combination (or fallback to priors)
3. Calculate tier baseline = average of easy and hard distribution means
4. Calculate fixture effect = distribution_mean - tier_baseline
5. Apply additive correction: `calibrated_xp = ml_xp + empirical_blend_weight * fixture_effect`

**Why Additive (Not Blending)**:
- Blending toward distribution means creates weak adjustments when ML is close
- Additive correction preserves ML as baseline while adding fixture effect
- Captures full historical easy/hard differential that ML underweights

**Example** (premium player, `empirical_blend_weight=0.5`):
- tier_baseline = (premium_easy_mean + premium_hard_mean) / 2 = (4.62 + 3.37) / 2 = 4.0
- easy fixture_effect = 4.62 - 4.0 = +0.62
- hard fixture_effect = 3.37 - 4.0 = -0.63
- ML predicts 5.0 for both → after calibration:
  - easy: 5.0 + 0.5 × 0.62 = **5.31**
  - hard: 5.0 + 0.5 × (-0.63) = **4.69**
- Creates **0.62 point differential** (50% of 1.25 historical difference)

**Risk Profiles**:
- **Conservative**: Uses 25th percentile (interpolated) → lower fixture effect
- **Balanced**: Uses distribution mean (default)
- **Risk-taking**: Uses 75th percentile (interpolated) → higher fixture effect

**Configuration** (`XPCalibrationConfig`):
- `enabled`: Enable/disable calibration (default: True)
- `empirical_blend_weight`: Weight for fixture effect (default: 0.5 = 50% of historical effect)
- `risk_adjustment_multiplier`: Std multiplier for risk profiles (default: 0.67)
- `premium_price_threshold`: Price threshold for premium tier (default: 8.0)
- `mid_price_threshold`: Price threshold for mid tier (default: 6.0)
- `easy_fixture_threshold`: Fixture difficulty threshold for easy (default: 1.538)
- `minimum_sample_size`: Minimum samples required to use fitted distribution (default: 30)

**Distributions File** (`data/calibration/distributions.json`):
- Contains fitted means, stds, percentiles for 6 combinations: premium/mid/budget × easy/hard
- Generated from historical FPL data analysis
- Includes sample sizes for quality assessment

## Expected Minutes Model (EWMA)

**Implementation** (`ExpectedPointsService._calculate_expected_minutes()`):
- **EWMA (Exponential Weighted Moving Average)** with configurable span (default: 5 games)
- Uses historical minutes data from `live_data_historical`
- Automatically adapts to rotation patterns, injuries, and manager trust
- Fallback to position-based estimates (GKP:90, DEF:80, MID:75, FWD:70) for new players

**Configuration** (`MinutesModelConfig`):
- `use_ewma`: Enable/disable EWMA (default: True)
- `ewma_span`: Lookback window in games (default: 5)
- `ewma_min_games`: Minimum games required for EWMA (default: 1, otherwise fallback)

**Key Features**:
- Recency bias: Recent games weighted more heavily
- Smoothing: Reduces noise from one-off benchings
- Injury adjustment: Scales by `chance_of_playing_next_round`
- Non-negative guarantee: xP clipped to >= 0

**Example**: Player with [90, 90, 85, 90, 75] minutes → EWMA = 84.3 (weighted toward recent 75)

## Captain Selection (Intelligent, Situation-Aware)

**Two methods available:**

### Basic: `get_captain_recommendation()`
- **Ceiling-seeking**: Uses 95th percentile (xP + 1.645 × uncertainty) for high-variance explosive players
- **Consistent players**: Uses 90th percentile (xP + 1.28 × uncertainty) for players with uncertainty ≤ 1.5
- **Haul bonus**: Players with ceiling (xP + 2×uncertainty) > 12 get additional bonus
- Template protection (20-40% boost for >50% owned in good fixtures)
- Matchup quality bonus from betting odds

### Advanced: `get_intelligent_captain_recommendation()`
Situation-aware captain selection with configurable strategy modes:

**Strategy Modes** (`CaptainSelectionConfig.strategy_mode`):
- `auto`: Recommends based on rank, season phase, chip status
- `template_lock`: Always captain highest-owned (>50%) - protects rank
- `protect_rank`: Minimize downside, prefer safe picks
- `balanced`: xP-weighted with template protection (default)
- `chase_rank`: Differential focus for climbers
- `maximum_upside`: Pure ceiling-seeking

**Features:**
1. **Situation Analysis**: Rank category, season phase, momentum, chip influence
2. **Haul Probability Matrix**: Blank/Return/Haul probabilities from xP ± uncertainty
3. **Rank Impact Estimation**: Expected rank change for each captain choice
4. **Template Comparison**: Shows risk/reward vs popular pick
5. **Strategy-Adjusted Scoring**: Applies mode-specific bonuses/penalties

**Auto-Detection Logic:**
- Rank < 100k → `protect_rank` or `template_lock` (late season)
- Rank 100k-500k → `balanced`
- Rank 500k-2M → `chase_rank` (if not improving) or `balanced`
- Rank > 2M → `maximum_upside`
- Triple Captain active → `maximum_upside`
- Free Hit active → `chase_rank`

**Marimo UI**: Strategy dropdown, rank impact toggle, haul probability matrix

## Hauler-First Strategy (Top 1% Analysis)

Based on empirical analysis of top 1% FPL managers comparing to average players:

| Metric | Top 1% | Average | Difference |
|--------|--------|---------|------------|
| Captain Points | 18.7 | 15.0 | +24% |
| Haulers/GW | 3.0 | 2.4 | +25% |

**Key Insight**: Top managers don't just find haulers - they find *optimal* haulers with higher ceilings.

**Implementation** (see `experiments/HAULER_STRATEGY_IMPLEMENTATION.md`):
1. **Training**: Use `fpl_hauler_capture` or `fpl_hauler_ceiling` scorer instead of MAE
2. **Variance Preservation**: `fpl_hauler_ceiling_scorer` penalizes models that compress predictions to 5-7 xP for everyone
3. **Captain Selection**: 95th percentile for high-uncertainty players (ceiling-seeking)
4. **Optimization Bonus**: SA optimizer adds ceiling bonus for players with haul potential (ceiling > 10)

**Configuration** (`OptimizationConfig`):
- `ceiling_bonus_enabled`: Add ceiling bonus to SA objective (default: True)
- `ceiling_bonus_factor`: Bonus per point above threshold (default: 0.15)
- `ceiling_bonus_threshold`: Ceiling threshold for bonus (default: 10.0)

## Historical xP Recomputation

Retroactive testing with arbitrary algorithms for A/B comparisons.

**Key APIs**:
- `DataOrchestrationService.load_historical_gameweek_state()`
- `PerformanceAnalyticsService.recompute_historical_xp(gw)`
- `PerformanceAnalyticsService.batch_recompute_season()`
- Metrics: MAE/RMSE/correlation by position

## Prediction Storage (Performance Tracking)

**PredictionStorageService** - Save committed predictions for accurate historical performance analysis.

**Saved Data**:
- Player predictions: `ml_xp`, `calibrated_xp`, `xp_uncertainty`, fixture context
- Squad composition: Captain/vice, starting XI, bench positions
- Squad financials: Total value, in the bank, free transfers
- Model metadata: Model path, calibration config, risk profile


## CLI Commands

`fpl-season-planner` | `fpl-gameweek-manager` | `fpl-xp-accuracy` | `fpl-ml-experiment`

Or: `marimo run fpl_team_picker/interfaces/{notebook}.py`

## Development

**Setup**: `uv sync`

**Code Quality**:
```bash
uv run ruff check fpl_team_picker/ && ruff format fpl_team_picker/
uv run marimo check fpl_team_picker/interfaces/ --fix
```

**Tests**: `uv run pytest -n auto --cov=fpl_team_picker/ --cov-report=html -v`

## Transfer Optimization

**SIMULATED ANNEALING**: Probabilistic search with temperature-based acceptance
- Good for non-linear objectives and exploration
- **Ceiling bonus**: Adds bonus for high-upside players (ceiling > 10) when `ceiling_bonus_enabled=True`
- Used for weekly transfers (1-3 players), wildcards, and initial squad generation
- Typical runtime: ~10-45 seconds depending on iterations

## Single-Gameweek Transfer Recommendations (LLM Agent)

**TransferPlanningAgentService** - Strategic single-GW transfer recommendations using Claude Agent SDK with orchestrator-workers pattern.

**Architecture (from Anthropic's "Building Effective Agents")**:
- **Orchestrator-Workers Pattern**: Main LLM agent delegates to 5 specialized tools
- **Start Simple**: Core 5 tools in Phase 1, extensible for future enhancements
- **ACI Investment**: Comprehensive tool docstrings with examples, edge cases, performance hints
- **Thinking Space**: 6-step workflow in system prompt encourages planning before execution
- **SA as Validator**: Agent reasons strategically first, SA optimizer validates/benchmarks
- **Hallucination Prevention**: Explicit instructions to only use team_name from tool outputs (never training data) to prevent incorrect player-team associations

**Capabilities:**
- **Single-GW Focus**: Recommend transfers for target gameweek only (not sequential multi-week)
- **Multi-GW Context**: Looks ahead 3 gameweeks for DGWs, fixture swings, chip timing
- **Multiple Options**: Returns 3-5 ranked transfer scenarios + hold baseline
- **Strategic Modes**: balanced, conservative, aggressive
- **Hit Analysis**: Configurable ROI threshold for -4 hits (default: +5 xP over 3 GW)
- **SA Validation**: Compares agent scenarios against mathematically optimal SA solution

**5 Agent Tools:**
1. `get_multi_gw_xp_predictions`: 1/3/5 GW xP forecasts with per-GW breakdowns
2. `analyze_fixture_context`: DGW/BGW detection + fixture difficulty runs
3. `run_sa_optimizer`: SA optimizer validation/benchmarking
4. `analyze_squad_weaknesses`: Identify upgrade targets (low xP, rotation, injury risks)
5. `get_template_players`: High ownership (>30%) players for safety consideration

**Output Structure** (`SingleGWRecommendation` Pydantic model):
- **Hold Option**: Baseline no-transfer scenario with xP projections
- **Transfer Scenarios**: 3-5 ranked by 3GW net ROI, each with:
  - Transfers list (player_out → player_in)
  - Single-GW metrics (xP, gain, hit cost, net gain)
  - Multi-GW metrics (3GW xP, gain, ROI)
  - Strategic reasoning and confidence level
  - Context flags (DGW leverage, fixture swing, chip prep)
  - SA validation deviation
- **Context Analysis**: DGW opportunities, fixture swings, chip timing
- **Final Recommendation**: Top-ranked scenario ID + strategic summary

**Usage:**
```bash
export ANTHROPIC_API_KEY=your_key_here

# Default: 5 scenarios, balanced strategy, GW18
uv run python scripts/transfer_recommendation_agent.py --gameweek 18

# Conservative strategy with 3 options
uv run python scripts/transfer_recommendation_agent.py -g 18 -s conservative -n 3

# Aggressive with lower hit threshold
uv run python scripts/transfer_recommendation_agent.py -g 18 -s aggressive -r 4.0

# Use faster/cheaper Haiku model
uv run python scripts/transfer_recommendation_agent.py -g 18 -m claude-haiku-3-7
```

**Configuration** (`TransferPlanningAgentConfig`):
- `model`: Claude model (default: "claude-sonnet-4-5")
- `api_key`: Anthropic API key (env: ANTHROPIC_API_KEY or FPL_TRANSFER_PLANNING_AGENT_API_KEY)
- `default_strategy`: "balanced", "conservative", "aggressive"
- `default_hit_roi_threshold`: Minimum xP gain for -4 hit (default: 5.0)

**Example Output:**
```
📊 Transfer Recommendations for GW18
Budget: £2.3m | Free Transfers: 1

🛑 HOLD OPTION (Baseline)
  GW1: 63.2 xP | 3GW: 189.5 xP
  Bank FT → 2 FTs next week
  Reasoning: Strong fixtures, worth banking for future flexibility

⭐ OPTION 1: PREMIUM_UPGRADE
  Transfer: Watkins → Haaland (£3.1m)
  GW1: 65.8 xP (+2.6) | Net: -1.4 (with -4 hit)
  3GW: 197.2 xP (+7.7) | ROI: +3.7
  Hit: -4 | Confidence: high
  SA Deviation: +0.3 xP vs optimizer
  Reasoning: Haaland has Liverpool(H) DGW in GW20, worth taking hit for 3GW gain

📍 OPTION 2: DIFFERENTIAL_PUNT
  Transfer: Saka → Palmer (£0.5m)
  GW1: 64.1 xP (+0.9) | Net: +0.9
  3GW: 192.3 xP (+2.8) | ROI: +2.8
  Hit: 0 | Confidence: medium
  Reasoning: Palmer has easy fixture run (GW18-20), differential at 18% ownership

🏆 TOP RECOMMENDATION: PREMIUM_UPGRADE
  3GW ROI (+3.7) justifies the hit, DGW opportunity in GW20
```

**Design Principles:**
- **Agent reasons strategically**: Considers fixtures, DGWs, template safety, chip timing
- **SA validates mathematically**: Ensures agent picks are near-optimal xP-wise
- **User maintains control**: Multiple ranked options allow informed decision-making
- **Transparent reasoning**: Each scenario explains WHY it's recommended

**Status**: Phase 1 complete (core backend + CLI). Phase 2-4 pending (tests, integration, UI).

See: Draft PR #45 and plan at `/Users/alex/.claude/plans/prancy-launching-emerson.md` for implementation details.

## AFCON Player Exclusions (GW16-22, 2025-26 Season)

**AFCONExclusionService** - Automatic exclusion of players participating in Africa Cup of Nations 2025.

**Tournament Details**:
- **Dates**: 21 December 2025 - 18 January 2026
- **Player release**: 15 December 2025
- **Affected gameweeks**: GW17-22
- **Total players**: 19 Premier League players

**High-Impact Players** (>5% ownership):
- Mohamed Salah (Liverpool) - 17.1% owned
- Bryan Mbeumo (Brentford) - 26.1% owned
- Omar Marmoush (Man City) - 2.5% owned
- Amad Diallo (Man Utd) - 1.6% owned

**Usage in Marimo UI**:
- Checkbox: "Exclude AFCON Players (GW17-22: Salah, Mbeumo, etc.)"
- When enabled, automatically excludes all 19 AFCON players from optimization
- Particularly important for GW16 (5 free transfers given for AFCON planning)

**Data Source**: `data/afcon_2025_players.json` with player_id matching for exact identification

**API**:
```python
from fpl_team_picker.domain.services.afcon_exclusion_service import AFCONExclusionService

service = AFCONExclusionService()
afcon_ids = service.get_afcon_player_ids()  # All 19 players
high_impact = service.get_afcon_player_ids(impact_filter=["high"])  # Just high-impact
```

## Configuration

Pydantic models with env var overrides (`FPL_{SECTION}_{FIELD}`). Override via `config.json`.

Config sections: XPModel, XPCalibration, TeamStrength, MinutesModel, StatisticalEstimation, Optimization, Visualization, ChipAssessment.

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

uv (don't forget to use uv), Python 3.13+, marimo, pandas, numpy, plotly, pydantic, xgboost, scikit-learn, lightgbm, pytest, ruff, fpl-dataset-builder

**Architecture**: Clean Architecture, frontend-agnostic, boundary validation, domain services, type-safe Pydantic models, repository pattern, 48/48 tests passing
