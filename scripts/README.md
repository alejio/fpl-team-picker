# Scripts

Standalone scripts for ML pipeline optimization and experimentation.

## TPOT Pipeline Optimizer

`tpot_pipeline_optimizer.py` - Automated ML pipeline discovery using TPOT (Tree-based Pipeline Optimization Tool).

### Features

- Uses the same temporal cross-validation (walk-forward) as `ml_xp_notebook.py`
- Leak-free feature engineering via `FPLFeatureEngineer` (80 features: 65 base + 15 enhanced)
- **Custom FPL-optimized scorer** (`fpl_weighted_huber`): Weighted Huber loss with position-aware penalties
- Automatic hyperparameter tuning and algorithm selection
- Exports production-ready sklearn pipelines (.py and .joblib)
- Enhanced evaluation metrics: position-specific MAE, ranking correlation, top-15 selection accuracy, captain accuracy, chaos detection

### Quick Start

```bash
# Install TPOT dependency
uv sync

# Run with default MAE scorer (fast test)
uv run python scripts/tpot_pipeline_optimizer.py --start-gw 1 --end-gw 9 --max-time-mins 10

# Run with custom FPL-optimized scorer (recommended)
uv run python scripts/tpot_pipeline_optimizer.py --start-gw 1 --end-gw 9 --scorer fpl_weighted_huber --max-time-mins 30

# Production run with more data (better model)
uv run python scripts/tpot_pipeline_optimizer.py --start-gw 1 --end-gw 15 --scorer fpl_weighted_huber --max-time-mins 60
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--start-gw` | 1 | Training start gameweek |
| `--end-gw` | 8 | Training end gameweek (inclusive) |
| `--max-time-mins` | 60 | Maximum runtime in minutes (TPOT 1.1.0 time-based) |
| `--max-eval-time-mins` | 5 | Max evaluation time per pipeline in minutes |
| `--cv-folds` | auto | Number of temporal CV folds (auto = all available) |
| `--scorer` | neg_mean_absolute_error | Scoring metric (MAE, MSE, R², **fpl_weighted_huber**) |
| `--output-dir` | models/tpot | Output directory for exported pipelines |
| `--random-seed` | 42 | Random seed for reproducibility |
| `--verbose` | 2 | TPOT verbosity level (0-3) |
| `--n-jobs` | -1 | Number of parallel jobs (-1 = all CPUs) |

### Custom FPL Scorer

The `fpl_weighted_huber` scorer optimizes for FPL-specific strategic objectives:

**Design Principles:**
1. **Huber loss** (δ=2.0): Balances MAE/MSE - robust to outliers, penalizes large errors
2. **Value-based weighting**: Errors on high xP predictions (8+ = captaincy candidates) penalized 2x more
3. **Asymmetric penalty**: Underestimation (missing hauls) penalized 1.3x vs overestimation
4. **Position-aware**: Supports sample weighting for position-specific models

**Why use it?**
- Standard MAE treats all errors equally (predicting budget player ±2pts = premium player ±2pts)
- FPL strategy cares more about correctly identifying captains and avoiding premium player mistakes
- Asymmetric penalty reflects FPL reality: missing a 15pt haul hurts more than overestimating a blank

**Test it:**
```bash
uv run python scripts/test_custom_scorer.py
```

### Enhanced Evaluation Metrics

Beyond standard MAE/RMSE/R², the optimizer reports:

**FPL Strategic Metrics:**
- **Spearman correlation**: Ranking accuracy (how well model orders players)
- **Top-15 selection accuracy**: Per-gameweek overlap with actual top performers (squad building)
- **Captain accuracy**: Did model identify the best player to captain?

**Chaos Detection:**
- Flags gameweeks with extreme variance, outlier hauls (20+ pts), or many high scorers
- Helps explain MAE spikes (e.g., GW9 had 23-point defender haul + rotation chaos)
- Format: `GW9: MAE 1.028 | Std 2.44 | Max 23 | 10+ pts: 14 | 0 pts: 60% ⚠️ [extreme_haul]`

### Output

TPOT exports three files to `models/tpot/`:

1. **Pipeline file** (`tpot_pipeline_gw1-9_YYYYMMDD_HHMMSS.py`) - Standalone sklearn pipeline code
2. **Model file** (`tpot_pipeline_gw1-9_YYYYMMDD_HHMMSS.joblib`) - Serialized model for immediate use
3. **Metadata file** (`tpot_pipeline_gw1-9_YYYYMMDD_HHMMSS_metadata.txt`) - Run configuration and CV scores

### Example Output

```
================================================================================
TPOT Pipeline Optimizer for FPL Expected Points
================================================================================

📥 Loading historical data (GW1 to GW8)...
   ✅ GW1: 543 players
   ✅ GW2: 543 players
   ...
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

🤖 Initializing TPOT optimizer...
   Generations: 10
   Population size: 50
   Scoring: neg_mean_absolute_error
   CV folds: 2
   Max time: No limit
   Max eval time: 5 mins
   Random seed: 42
   Parallel jobs: -1

🚀 Starting TPOT optimization...
================================================================================
Generation 1 - Current best internal CV score: -2.345
Generation 2 - Current best internal CV score: -2.289
...
================================================================================

✅ TPOT optimization complete!
   Duration: 12.34 minutes
   Best CV score: 0.8234

💾 Exporting best pipeline to: models/tpot/tpot_pipeline_gw1-8_20251020_143022.py
   ✅ Pipeline exported successfully
   ✅ Metadata saved to: models/tpot/tpot_pipeline_gw1-8_20251020_143022_metadata.txt

📊 Evaluating best pipeline...

📈 Overall Metrics:
   MAE:  2.289 points
   RMSE: 3.145 points
   R²:   0.234

📊 Position-Specific MAE:
   DEF: 2.123 points
   FWD: 2.567 points
   GKP: 1.845 points
   MID: 2.301 points
```

### Integration with ML Notebook

After TPOT finds an optimal pipeline, test it in `ml_xp_notebook.py`:

1. Copy the exported pipeline code from `models/tpot/tpot_pipeline_*.py`
2. Add it as a new model in the notebook's model training section
3. Compare against existing models (LightGBM, Ridge, etc.)
4. If better, integrate into `MLExpectedPointsService`

### Example Integration

```python
# In ml_xp_notebook.py, add to model training cell:

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# TPOT-discovered pipeline (example)
tpot_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=7,
        n_estimators=200,
        subsample=0.8
    ))
])

# Cross-validation
tpot_mae_scores = -cross_val_score(
    tpot_pipeline,
    X_cv,
    y_cv,
    scoring='neg_mean_absolute_error',
    cv=_gkf,
    groups=cv_groups,
    n_jobs=-1
)

# Train final model
tpot_pipeline.fit(X_cv, y_cv)
trained_models['tpot'] = tpot_pipeline

cv_results['tpot'] = {
    'MAE_mean': tpot_mae_scores.mean(),
    'MAE_std': tpot_mae_scores.std(),
    # ... other metrics
}
```

### Tips for Best Results

1. **Start small**: Test with `--generations 5` to verify everything works
2. **Scale up**: Use `--generations 20 --population-size 100` for production
3. **Use time limits**: `--max-time-mins 60` prevents overnight runs
4. **Compare scoring metrics**: Try different `--scoring` options (MAE vs MSE vs R²)
5. **Reproducibility**: Always use same `--random-seed` for consistent results
6. **Monitor progress**: Use `--verbosity 2` or `--verbosity 3` to watch optimization

### Troubleshooting

**Issue**: "Need at least 2 gameweeks for temporal CV"
- **Solution**: Increase `--end-gw` to at least 7 (GW6 train, GW7 test)

**Issue**: TPOT takes too long
- **Solution**: Reduce `--generations` or set `--max-time-mins`

**Issue**: Out of memory errors
- **Solution**: Reduce `--population-size` or limit `--cv-folds`

**Issue**: Pipeline overfits
- **Solution**: Increase `--cv-folds` or add more training data (higher `--end-gw`)

### References

- [TPOT Documentation](https://epistasislab.github.io/tpot/latest/)
- [TPOT Paper (JMLR 2016)](http://jmlr.org/papers/v17/15-166.html)
- [Sklearn Pipeline Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
