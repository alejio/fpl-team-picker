# TPOT Pipeline Optimization Setup

## What Was Created

A dedicated script for automated ML pipeline optimization using TPOT (Tree-based Pipeline Optimization Tool).

### Files Added

1. **scripts/tpot_pipeline_optimizer.py** - Main optimization script (450+ lines)
2. **scripts/README.md** - Comprehensive documentation with examples
3. **TPOT_SETUP.md** - This file (setup guide)

### Dependencies Added

- `tpot>=1.1.0` - AutoML library for pipeline optimization
- `setuptools>=75.0.0` - Required by TPOT dependencies

Updated: `pyproject.toml`

## What TPOT Does

TPOT automatically discovers optimal ML pipelines by:

1. **Algorithm Selection** - Tests multiple ML algorithms (Ridge, RandomForest, GradientBoosting, XGBoost, etc.)
2. **Hyperparameter Tuning** - Optimizes parameters for each algorithm
3. **Feature Engineering** - Tests different preprocessing steps (scaling, PCA, etc.)
4. **Pipeline Assembly** - Combines components into optimal sklearn Pipeline objects

**Result**: Production-ready sklearn pipeline code you can integrate directly into your project.

## Why This Matters for FPL xP Prediction

Current approach (ml_xp_notebook.py):
- Manual algorithm testing (Ridge, LightGBM, RandomForest, GradientBoosting)
- Manual hyperparameter selection
- Time-consuming to test new algorithms

With TPOT:
- **Automated discovery** - Tests hundreds of pipeline combinations
- **Optimized hyperparameters** - Uses genetic algorithms to find best settings
- **Novel combinations** - May discover pipelines you wouldn't try manually
- **Reproducible** - Export exact pipeline for production use

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

This installs TPOT 1.1.0 and all required dependencies.

### 2. Run Quick Test (5 minutes)

```bash
uv run python scripts/tpot_pipeline_optimizer.py \
  --start-gw 1 \
  --end-gw 8 \
  --max-time-mins 5
```

**Expected output**: Best pipeline exported to `models/tpot/`

### 3. Run Production Optimization (1 hour)

```bash
uv run python scripts/tpot_pipeline_optimizer.py \
  --start-gw 1 \
  --end-gw 8 \
  --max-time-mins 60 \
  --verbose 2
```

**Expected output**: Thoroughly optimized pipeline with better performance

**Note**: TPOT 1.1.0 uses **time-based optimization** with Dask distributed computing instead of generations/population_size. The longer it runs, the more pipelines it evaluates.

## How It Works (Technical Details)

### 1. Data Loading

Uses same data pipeline as ml_xp_notebook.py:
- `FPLDataClient` for historical gameweek data (GW1-8)
- Position enrichment from current_players
- Fixtures and teams for context features

### 2. Feature Engineering

Uses **production FPLFeatureEngineer** (same as notebook):
- 63 features: price, position, games_played
- Cumulative stats: minutes, goals, assists, points, xG, xA, BPS
- Per-90 rates: goals_per_90, assists_per_90, points_per_90
- Rolling 5GW form: recent performance trends
- Team context: team strength, rolling team stats
- Fixture features: opponent strength, home/away, fixture difficulty

**Guarantee**: TPOT uses IDENTICAL features to notebook (no drift).

### 3. Temporal Cross-Validation

Walk-forward validation (same as notebook):
```
Fold 1: Train on GW6     → Test on GW7
Fold 2: Train on GW6-7   → Test on GW8
Fold 3: Train on GW6-8   → Test on GW9
...
```

**Why temporal CV?**
- Realistic: Predicts future gameweeks (not random players)
- No leakage: Training only uses past data
- Robust: Multiple test points across season

### 4. TPOT 1.1.0 Optimization

Time-based evolutionary algorithm with Dask:
1. **Initialize**: Create Dask LocalCluster for distributed optimization
2. **Random Pipelines**: Generate initial population of ML pipelines
3. **Evaluate**: Test each pipeline with temporal CV (parallel via Dask)
4. **Select**: Keep best performers
5. **Mutate**: Modify pipelines (change algorithms, hyperparameters, preprocessing)
6. **Crossover**: Combine successful pipelines
7. **Repeat**: Continue until `max_time_mins` reached

**Key Differences from Classic TPOT:**
- Uses **time budget** instead of generation count
- Leverages **Dask** for distributed parallel evaluation
- Automatically stops after time limit (no manual convergence)

**Result**: Best pipeline discovered within time budget.

### 5. Export & Evaluation

Exports:
- **Pipeline file**: Standalone sklearn Pipeline (copy-paste ready)
- **Metadata file**: CV scores, hyperparameters, configuration

Evaluation metrics:
- Overall: MAE, RMSE, R²
- Position-specific: MAE for GKP, DEF, MID, FWD
- Gameweek-specific: MAE for each GW in test set

## Example Output

```
================================================================================
TPOT Pipeline Optimizer for FPL Expected Points
================================================================================

📥 Loading historical data (GW1 to GW8)...
   ✅ GW1: 543 players
   ✅ GW2: 543 players
   ✅ GW3: 543 players
   ✅ GW4: 543 players
   ✅ GW5: 543 players
   ✅ GW6: 543 players
   ✅ GW7: 543 players
   ✅ GW8: 543 players

✅ Total records: 4,344
   Unique players: 543

🔧 Engineering features (production FPLFeatureEngineer)...
   ✅ Created 63 features
   Total samples: 4,344

📊 Creating temporal CV splits (walk-forward validation)...
   Fold 1: Train on GW6-6 (543 samples) → Test on GW7 (543 samples)
   Fold 2: Train on GW6-7 (1,086 samples) → Test on GW8 (543 samples)

✅ Created 2 temporal CV folds

📊 Final training data shape:
   X: 1,086 samples × 63 features
   y: 1,086 targets

🤖 Initializing TPOT 1.1.0 optimizer...
   Scorer: neg_mean_absolute_error
   CV folds: 2
   Max time: 60 mins
   Max eval time: 5 mins
   Random seed: 42
   Parallel jobs: -1
   Note: TPOT 1.1.0 uses time-based optimization with Dask

🔧 Starting Dask LocalCluster...
   Workers: auto
   ✅ Dask cluster ready: http://127.0.0.1:8787/status

🚀 Starting TPOT optimization...
================================================================================
[TPOT 1.1.0 runs optimization for 60 minutes, evaluating pipelines continuously]
================================================================================

✅ TPOT optimization complete!
   Duration: 60.15 minutes

🧹 Dask cluster shut down

💾 Exporting best pipeline to: models/tpot/tpot_pipeline_gw1-8_20251020_143022.py
   ✅ Pipeline exported successfully
   ✅ Joblib model saved: models/tpot/tpot_pipeline_gw1-8_20251020_143022.joblib
   ✅ Metadata saved to: models/tpot/tpot_pipeline_gw1-8_20251020_143022_metadata.txt

📊 Evaluating best pipeline...

📈 Overall Metrics:
   MAE:  0.768 points
   RMSE: 1.643 points
   R²:   0.508

📊 Position-Specific MAE:
   DEF: 0.894 points
   FWD: 0.832 points
   GKP: 0.433 points
   MID: 0.745 points

📅 Gameweek-Specific MAE:
   GW6: 0.749 points
   GW7: 0.784 points
   GW8: 0.770 points

================================================================================
✅ TPOT optimization complete!
================================================================================

📁 Exported pipeline: models/tpot/tpot_pipeline_gw1-8_20251020_143022.py

💡 Next steps:
   1. Review the exported pipeline code
   2. Test the pipeline in ml_xp_notebook.py
   3. Compare against current models (LightGBM, Ridge, etc.)
   4. If better, integrate into MLExpectedPointsService
```

## Integration Workflow

### Step 1: Run TPOT

```bash
uv run python scripts/tpot_pipeline_optimizer.py \
  --start-gw 1 \
  --end-gw 8 \
  --max-time-mins 60
```

### Step 2: Review Exported Pipeline

Check `models/tpot/tpot_pipeline_gw1-8_*.py`:

```python
# Example exported pipeline
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Average CV score on the training set was: -2.2263
exported_pipeline = make_pipeline(
    StandardScaler(),
    GradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=7,
        n_estimators=200,
        subsample=0.8
    )
)
```

### Step 3: Test in ml_xp_notebook.py

Add to model training cell:

```python
# TPOT-discovered pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

tpot_model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=7,
        n_estimators=200,
        subsample=0.8
    ))
])

# Evaluate with temporal CV
_tpot_mae_scores = -cross_val_score(
    tpot_model,
    X_cv,
    y_cv,
    scoring='neg_mean_absolute_error',
    cv=_cv_splits,  # Temporal splits
    n_jobs=-1
)

# Train final model
tpot_model.fit(X_cv, y_cv)
trained_models['tpot'] = tpot_model

cv_results['tpot'] = {
    'MAE_mean': _tpot_mae_scores.mean(),
    'MAE_std': _tpot_mae_scores.std(),
    # ... other metrics
}
```

### Step 4: Compare Results

Notebook will show:
- TPOT vs Ridge vs LightGBM vs RandomForest
- Feature importance (what drives predictions?)
- Position-specific errors (which positions need improvement?)
- Prediction bias (over/under-prediction patterns)

### Step 5: Production Integration (if TPOT wins)

If TPOT model has better MAE:

1. **Update MLExpectedPointsService**:
   ```python
   # In ml_expected_points_service.py
   self.model = Pipeline([
       ('scaler', StandardScaler()),
       ('regressor', GradientBoostingRegressor(
           learning_rate=0.05,
           max_depth=7,
           n_estimators=200,
           subsample=0.8
       ))
   ])
   ```

2. **Retrain on full dataset**:
   ```python
   self.model.fit(X_train, y_train)
   ```

3. **Test in gameweek_manager.py**:
   ```python
   xp_df = xp_service.calculate_expected_points(
       gameweek_data=gameweek_data,
       use_ml_model=True,  # Use TPOT pipeline
       gameweeks_ahead=1
   )
   ```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tpot'"

**Solution**: Run `uv sync` to install dependencies.

### Issue: "Need at least 2 gameweeks for temporal CV"

**Solution**: Increase `--end-gw` to at least 7:
```bash
uv run python scripts/tpot_pipeline_optimizer.py --end-gw 7
```

### Issue: TPOT runs forever

**Solution**: Set time limit:
```bash
uv run python scripts/tpot_pipeline_optimizer.py --max-time-mins 30
```

### Issue: Out of memory

**Solution**: Reduce CV folds or number of Dask workers:
```bash
uv run python scripts/tpot_pipeline_optimizer.py \
  --cv-folds 2 \
  --n-jobs 2
```

### Issue: TPOT finds worse pipeline than manual models

**Possible causes**:
1. Not enough optimization time (try `--max-time-mins 120` or more)
2. Too few Dask workers (try `--n-jobs -1` for all CPUs)
3. Time limit too strict (increase `--max-time-mins`)
4. Unlucky random seed (try different `--random-seed`)

**Solution**: Run longer optimization or try different random seeds.

## Best Practices

### 1. Start Small, Scale Up

```bash
# Day 1: Quick test (5 mins)
uv run python scripts/tpot_pipeline_optimizer.py \
  --max-time-mins 5

# Day 2: Medium run (30 mins)
uv run python scripts/tpot_pipeline_optimizer.py \
  --max-time-mins 30

# Day 3: Production run (2 hours)
uv run python scripts/tpot_pipeline_optimizer.py \
  --max-time-mins 120
```

### 2. Use Consistent Random Seeds

For reproducibility:
```bash
uv run python scripts/tpot_pipeline_optimizer.py \
  --random-seed 42
```

### 3. Monitor Progress

Use verbose level 2 or 3:
```bash
uv run python scripts/tpot_pipeline_optimizer.py \
  --verbose 3
```

### 4. Try Different Scoring Metrics

MAE (default, focus on absolute errors):
```bash
uv run python scripts/tpot_pipeline_optimizer.py \
  --scorer neg_mean_absolute_error
```

MSE (penalize large errors more):
```bash
uv run python scripts/tpot_pipeline_optimizer.py \
  --scorer neg_mean_squared_error
```

R² (focus on variance explained):
```bash
uv run python scripts/tpot_pipeline_optimizer.py \
  --scorer r2
```

### 5. Save Results

Keep track of experiments:
```bash
# Create experiment log
mkdir -p experiments/tpot_runs

# Run with timestamp in output
uv run python scripts/tpot_pipeline_optimizer.py \
  --max-time-mins 60 \
  --output-dir experiments/tpot_runs/run_$(date +%Y%m%d_%H%M%S)
```

## Advanced Usage

### Custom CV Folds

Limit CV folds for faster iteration:
```bash
uv run python scripts/tpot_pipeline_optimizer.py \
  --cv-folds 3  # Only use 3 folds instead of all available
```

### Parallel Processing

Control CPU usage:
```bash
# Use all CPUs (default)
uv run python scripts/tpot_pipeline_optimizer.py --n-jobs -1

# Use 4 CPUs
uv run python scripts/tpot_pipeline_optimizer.py --n-jobs 4

# Use 1 CPU (sequential)
uv run python scripts/tpot_pipeline_optimizer.py --n-jobs 1
```

## References

- [TPOT Documentation](https://epistasislab.github.io/tpot/latest/)
- [TPOT GitHub](https://github.com/EpistasisLab/tpot)
- [TPOT Paper (JMLR 2016)](http://jmlr.org/papers/v17/15-166.html)
- [Sklearn Pipeline Guide](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [AutoML Survey Paper](https://arxiv.org/abs/1908.00709)

## Next Steps

1. **Run quick test**: Verify TPOT works with your data
2. **Run production optimization**: Find best pipeline (1-2 hours)
3. **Evaluate in notebook**: Compare against existing models
4. **Integrate if better**: Update MLExpectedPointsService
5. **Iterate**: Try different configurations, scoring metrics, CV strategies

Happy optimizing! 🚀
