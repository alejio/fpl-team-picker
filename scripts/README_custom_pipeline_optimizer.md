# Custom Pipeline Optimizer

Fully configurable ML pipeline optimizer for FPL expected points prediction. Alternative to TPOT with manual control over feature selection, regressors, preprocessing, and hyperparameter optimization.

## Key Features

- **Manual feature selection** - Keeps penalty/set-piece features that TPOT's RFE drops
- **Multiple regressors** - XGBoost, LightGBM, Random Forest, Gradient Boost, AdaBoost, Ridge, Lasso, ElasticNet
- **Smart preprocessing** - Standard, grouped (feature-type specific), or robust scaling
- **Position-aware scorers** - FPL-specific metrics that respect positional constraints
- **Hyperparameter reuse** - Save best params from evaluation and reuse in training

## Quick Start

### Evaluate Mode (Test Configuration)

Test a configuration on a holdout set before full training:

```bash
python scripts/custom_pipeline_optimizer.py evaluate \
  --regressor random-forest \
  --preprocessing grouped \
  --end-gw 10 \
  --holdout-gws 1 \
  --scorer fpl_weighted_huber
```

This will:
1. Train on GW1-9, hold out GW10
2. Run hyperparameter search (with reduced trials)
3. Evaluate on CV and holdout
4. Save best hyperparameters to `best_params_*.json`

### Train Mode (Deploy Model)

Train on all data for deployment:

```bash
# Option 1: Use saved hyperparameters from evaluate
python scripts/custom_pipeline_optimizer.py train \
  --regressor random-forest \
  --preprocessing grouped \
  --end-gw 10 \
  --use-best-params-from best_params_random-forest_gw1-9_*.json

# Option 2: Run fresh hyperparameter search
python scripts/custom_pipeline_optimizer.py train \
  --regressor random-forest \
  --preprocessing grouped \
  --end-gw 10 \
  --n-trials 20
```

## Common Options

### Regressors
- `xgboost`, `lightgbm`, `random-forest`, `gradient-boost`, `adaboost`, `ridge`, `lasso`, `elasticnet`

### Feature Selection
- `none` - Use all 99 features
- `correlation` - Remove highly correlated features (|r| > 0.95)
- `permutation` - Top 60 features by permutation importance
- `rfe-smart` - RFE that keeps penalty features

### Preprocessing
- `standard` - StandardScaler for all features
- `grouped` - Feature-type specific scalers (RobustScaler for counts, MinMaxScaler for percentages, etc.)
- `robust` - RobustScaler for all features (good for linear models)

### Scorers
- `fpl_weighted_huber` - Weighted Huber loss (default)
- `fpl_top_k_ranking` - Top-K ranking accuracy
- `fpl_captain_pick` - Captain selection accuracy
- `fpl_position_aware` - Position-specific top-K overlap
- `fpl_starting_xi` - Starting XI point efficiency
- `fpl_comprehensive` - Combined position-aware + XI + captain

## Examples

```bash
# Quick test with XGBoost
python scripts/custom_pipeline_optimizer.py evaluate --regressor xgboost

# Full training with position-aware scorer
python scripts/custom_pipeline_optimizer.py train \
  --regressor lightgbm \
  --feature-selection rfe-smart \
  --keep-penalty-features \
  --preprocessing grouped \
  --scorer fpl_comprehensive \
  --end-gw 10

# Train with saved params (fastest)
python scripts/custom_pipeline_optimizer.py train \
  --regressor random-forest \
  --use-best-params-from models/custom/best_params_random-forest_gw1-9_*.json
```

## Output

- **Evaluate mode**: Prints metrics, saves `best_params_*.json`
- **Train mode**: Saves trained pipeline and metadata to `models/custom/`

## Workflow

1. **Evaluate** different configurations quickly (reduced trials)
2. Check holdout performance to avoid overfitting
3. **Train** with best params on all data for deployment
4. Use saved pipeline for predictions

## Notes

- Position-aware scorers require position data (automatically included)
- `--keep-penalty-features` ensures penalty/set-piece features aren't dropped
- `--holdout-gws` must leave at least 6 gameweeks for training
- Saved hyperparameters include config validation to prevent mismatches
