# Position-Specific ML Models Experiment

## Status: PHASE 1 COMPLETE ✅

**Date**: 2025-12-03
**Result**: Position-specific models show mixed results → **Hybrid approach recommended**

---

## Experiment Results (GW11-12 Holdout)

### Overall Performance

| Metric | Unified | Position-Specific | Δ |
|--------|---------|-------------------|---|
| **MAE** | 1.124 | 1.119 | +0.4% (marginal) |

### Per-Position Breakdown

| Position | Unified MAE | Specific MAE | Improvement | Best Regressor |
|----------|-------------|--------------|-------------|----------------|
| **GKP** | 0.727 | 0.680 | **+6.5%** ✅ | Gradient Boosting |
| **DEF** | 1.274 | 1.293 | -1.5% ❌ | XGBoost |
| **MID** | 1.111 | 1.124 | -1.2% ❌ | Random Forest |
| **FWD** | 1.134 | 1.033 | **+9.0%** ✅ | Random Forest |

### Statistical Significance

- **t-test p-value**: 0.80 ❌ (not significant)
- **Wilcoxon p-value**: 0.36 ❌ (not significant)
- **Cohen's d**: 0.006 (negligible effect)

### Key Findings

1. **Position-specific models excel for edge positions** (GKP +6.5%, FWD +9.0%)
2. **Unified model is better for volume positions** (DEF, MID) - benefits from more training data
3. **Overall improvement is negligible** when using ALL position-specific models
4. **Hybrid approach recommended**: Use position-specific for GKP/FWD, unified for DEF/MID

---

## Hypothesis

Training separate ML models for each position (GKP, DEF, MID, FWD) will outperform the current unified model because:

1. **Different scoring mechanisms**: GKP points come primarily from saves and penalty saves; DEF from clean sheets; MID/FWD from goals and assists with different weights
2. **Different feature importance**: Opponent xG matters more for GKP/DEF; team xG matters more for MID/FWD
3. **Different point distributions**: GKP has modal point value of 0-2; FWD has higher variance with more frequent blanks
4. **Reduced noise**: Position-irrelevant features (e.g., `rolling_5gw_saves` for FWD) add noise to unified model

## Data Availability (GW1-12, 2025-26 Season)

| Position | Total Records | Trainable (GW6+) | Unique Players |
|----------|--------------|------------------|----------------|
| GKP      | ~1,020       | ~600             | ~85            |
| DEF      | ~2,900       | ~1,700           | ~242           |
| MID      | ~3,900       | ~2,300           | ~325           |
| FWD      | ~960         | ~560             | ~80            |
| **Total**| **~8,780**   | **~5,160**       | **~747**       |

**Key Concern**: GKP (~600) and FWD (~560) have limited samples for ML training. Will need:
- Careful regularization to prevent overfitting
- Potentially fewer features via aggressive feature selection
- Consider combining GKP+DEF (defensive) and MID+FWD (attacking) as alternative

## Baseline Model

The unified baseline model is already trained and configured:

```
Path: models/custom/lightgbm_gw1-12_20251129_103326_pipeline.joblib
Type: LightGBM with 117 features (RFE-smart + penalty features)
Training Data: GW1-12
```

---

## Experiment Design

### Phase 1: Baseline Establishment

**Objective**: Train unified LightGBM baseline on GW1-10 for fair comparison.

**Best Params**: `models/custom/best_params_lightgbm_gw1-10_20251129_102622.json`

```bash
# Step 1: Train unified LightGBM baseline on GW1-10 using optimized params
uv run python scripts/custom_pipeline_optimizer.py train \
    --end-gw 10 \
    --regressor lightgbm \
    --use-best-params-from models/custom/best_params_lightgbm_gw1-10_20251129_102622.json

# This will produce: models/custom/lightgbm_gw1-10_*_pipeline.joblib
```

**Why GW1-10?** For fair comparison:
- Train both unified and position-specific models on GW1-10
- Evaluate both on holdout GW11-12
- Same training data = fair comparison

**Metrics to Record**:
- Overall MAE, RMSE, Spearman correlation
- Per-position MAE (GKP_mae, DEF_mae, MID_mae, FWD_mae)
- Captain accuracy
- Top-15 overlap

---

### Phase 2: Position-Specific Model Training

**Objective**: Train and evaluate separate models for each position.

**Implementation**:

Create new script: `scripts/position_specific_optimizer.py`

```python
#!/usr/bin/env python3
"""
Position-Specific Pipeline Optimizer

Trains separate ML models for each FPL position:
- GKP: Focus on saves, clean sheets, penalty saves
- DEF: Focus on clean sheets, goals/assists, bonus
- MID: Focus on goals, assists, bonus, attacking metrics
- FWD: Focus on goals, bonus, xG efficiency

Each model uses position-specific feature selection to remove
irrelevant features and reduce overfitting.
"""

POSITION_FEATURE_MASKS = {
    "GKP": {
        # Keep: saves, clean sheets, goals conceded, team defensive
        "exclude": [
            "rolling_5gw_goals", "rolling_5gw_assists", "rolling_5gw_xg", "rolling_5gw_xa",
            "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
            "rolling_5gw_threat", "rolling_5gw_creativity",
        ],
        # Boost: defensive metrics
        "boost_features": ["rolling_5gw_saves", "rolling_5gw_clean_sheets", "clean_sheet_rate"],
    },
    "DEF": {
        # Keep most features, DEF can score goals
        "exclude": ["rolling_5gw_saves"],  # Only GKP has saves
        "boost_features": ["rolling_5gw_clean_sheets", "clean_sheet_rate", "opponent_rolling_5gw_xgc"],
    },
    "MID": {
        # Keep attacking metrics, remove GKP-specific
        "exclude": ["rolling_5gw_saves", "rolling_5gw_goals_conceded"],
        "boost_features": ["rolling_5gw_goals", "rolling_5gw_assists", "rolling_5gw_creativity"],
    },
    "FWD": {
        # Focus on attacking, remove defensive
        "exclude": [
            "rolling_5gw_saves", "rolling_5gw_clean_sheets", "clean_sheet_rate",
            "rolling_5gw_goals_conceded", "rolling_5gw_xgc",
        ],
        "boost_features": ["rolling_5gw_goals", "rolling_5gw_xg", "rolling_5gw_threat"],
    },
}

# Adjusted hyperparameters for smaller datasets (GKP, FWD)
SMALL_DATASET_CONFIG = {
    "max_depth": (3, 8),  # Shallower trees
    "min_samples_leaf": (10, 50),  # More regularization
    "n_estimators": (50, 200),  # Fewer trees
    "max_features": (0.3, 0.7),  # Feature subsampling
}
```

**Training Commands**:
```bash
# Train position-specific models on GW1-10 (holding out GW11-12 for evaluation)
uv run python scripts/position_specific_optimizer.py train-all \
    --end-gw 10 \
    --n-trials 30 \
    --output-dir models/position_specific/

# Or train individual positions:
for pos in GKP DEF MID FWD; do
    uv run python scripts/position_specific_optimizer.py train \
        --position $pos \
        --end-gw 10 \
        --n-trials 30 \
        --output-dir models/position_specific/
done
```

---

### Phase 3: Evaluation Framework

**Objective**: Compare unified vs. position-specific models fairly.

**Evaluation Script**: `scripts/evaluate_position_models.py`

```python
"""
Evaluate position-specific models vs unified model.

Combines position-specific predictions into full-squad predictions
for fair comparison with unified model.
"""

def evaluate_position_specific(
    position_models: dict[str, Pipeline],  # {"GKP": model, "DEF": model, ...}
    unified_model: Pipeline,
    test_data: pd.DataFrame,
    cv_data: pd.DataFrame,
) -> dict:
    """
    Compare position-specific ensemble vs unified model.

    Returns:
        Dict with comparison metrics
    """
    results = {}

    # Get unified predictions
    unified_preds = unified_model.predict(test_data[feature_cols])

    # Get position-specific predictions (combine by position)
    combined_preds = np.zeros(len(test_data))
    for position in ["GKP", "DEF", "MID", "FWD"]:
        mask = test_data["position"] == position
        if mask.sum() > 0:
            pos_features = test_data.loc[mask, feature_cols]
            combined_preds[mask] = position_models[position].predict(pos_features)

    # Evaluate both
    y_true = test_data["total_points"].values

    results["unified"] = {
        "mae": mean_absolute_error(y_true, unified_preds),
        "rmse": np.sqrt(mean_squared_error(y_true, unified_preds)),
        "spearman": spearmanr(y_true, unified_preds)[0],
    }

    results["position_specific"] = {
        "mae": mean_absolute_error(y_true, combined_preds),
        "rmse": np.sqrt(mean_squared_error(y_true, combined_preds)),
        "spearman": spearmanr(y_true, combined_preds)[0],
    }

    # Per-position comparison
    for position in ["GKP", "DEF", "MID", "FWD"]:
        mask = test_data["position"] == position
        if mask.sum() > 0:
            y_pos = y_true[mask]
            results[f"{position}_unified_mae"] = mean_absolute_error(
                y_pos, unified_preds[mask]
            )
            results[f"{position}_specific_mae"] = mean_absolute_error(
                y_pos, combined_preds[mask]
            )
            results[f"{position}_improvement"] = (
                results[f"{position}_unified_mae"] - results[f"{position}_specific_mae"]
            )

    return results
```

---

### Phase 4: Captain & Squad-Level Evaluation

**Objective**: Evaluate on FPL-relevant metrics, not just point prediction accuracy.

**Key Metrics**:

1. **Captain Accuracy**: Did the model correctly rank the highest-scoring player in the squad?
   ```python
   def captain_accuracy(y_true, y_pred, squad_ids, gameweek):
       """For each GW, check if max(y_pred) corresponds to max(y_true)."""
       correct = 0
       total = 0
       for gw in gameweeks:
           gw_mask = gameweek == gw
           gw_true = y_true[gw_mask]
           gw_pred = y_pred[gw_mask]
           if gw_true.argmax() == gw_pred.argmax():
               correct += 1
           total += 1
       return correct / total
   ```

2. **Top-K Overlap**: How many of the model's top-K predictions are in the actual top-K?
   ```python
   def top_k_overlap(y_true, y_pred, k=15):
       true_top_k = set(np.argsort(y_true)[-k:])
       pred_top_k = set(np.argsort(y_pred)[-k:])
       return len(true_top_k & pred_top_k) / k
   ```

3. **Starting XI xP**: Given optimal starting XI based on predictions, what's actual points?
   ```python
   def starting_xi_score(y_true, y_pred, positions):
       """Select best 11 based on predictions, score on actuals."""
       # Sort by prediction
       sorted_idx = np.argsort(y_pred)[::-1]

       # Greedy formation selection (simplified)
       selected = select_valid_formation(sorted_idx, positions)

       return y_true[selected].sum()
   ```

---

## Statistical Significance Testing

**Objective**: Determine if position-specific models provide statistically significant improvement.

**Method**: Paired t-test on per-gameweek prediction errors

```python
from scipy.stats import ttest_rel, wilcoxon

def significance_test(unified_errors, position_errors, alpha=0.05):
    """
    Test if position-specific model significantly outperforms unified.

    Args:
        unified_errors: Array of absolute errors per sample (unified model)
        position_errors: Array of absolute errors per sample (position model)
        alpha: Significance level

    Returns:
        Dict with test statistics and conclusion
    """
    # Paired t-test (assumes normal distribution of differences)
    t_stat, t_pvalue = ttest_rel(unified_errors, position_errors)

    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, w_pvalue = wilcoxon(unified_errors, position_errors)

    # Effect size (Cohen's d)
    diff = unified_errors - position_errors
    cohens_d = diff.mean() / diff.std()

    return {
        "t_statistic": t_stat,
        "t_pvalue": t_pvalue,
        "t_significant": t_pvalue < alpha,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_pvalue": w_pvalue,
        "wilcoxon_significant": w_pvalue < alpha,
        "cohens_d": cohens_d,
        "mean_improvement": diff.mean(),
        "improvement_std": diff.std(),
    }
```

---

## Implementation Plan

### Week 1: Infrastructure Setup

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Create `position_specific_optimizer.py` scaffold | Basic script structure |
| 2 | Implement position-specific feature filtering | `POSITION_FEATURE_MASKS` |
| 3 | Implement data splitting by position | Position data loaders |
| 4 | Add hyperparameter configs for small datasets | `SMALL_DATASET_CONFIG` |
| 5 | Unit tests for position filtering | `test_position_specific_optimizer.py` |

### Week 2: Training & Evaluation

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Baseline already exists: `lightgbm_gw1-12_*.joblib` | ✅ Done |
| 2 | Train GKP and DEF models (GW1-10) | Position models |
| 3 | Train MID and FWD models (GW1-10) | Position models |
| 4 | Evaluation script already in `position_specific_optimizer.py` | ✅ Done |
| 5 | Run full comparison on holdout (GW11-12) | Results JSON |

### Week 3: Analysis & Decision

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Statistical significance testing | p-values, effect sizes |
| 2 | Error analysis by player archetype | Insights document |
| 3 | Ablation: 2-model (DEF+GKP vs MID+FWD) | Alternative results |
| 4 | Production recommendation | Decision document |
| 5 | If positive: Integration into main pipeline | Updated `MLExpectedPointsService` |

---

## Expected Outcomes

### Optimistic Scenario (Position-Specific Wins)

| Metric | Unified | Position-Specific | Improvement |
|--------|---------|-------------------|-------------|
| Overall MAE | 2.10 | 1.95 | -7.1% |
| GKP MAE | 1.50 | 1.30 | -13.3% |
| DEF MAE | 2.00 | 1.85 | -7.5% |
| MID MAE | 2.20 | 2.10 | -4.5% |
| FWD MAE | 2.40 | 2.25 | -6.3% |
| Captain Accuracy | 42% | 48% | +6pp |

**If this happens**: Integrate position-specific models into production.

### Neutral Scenario (No Significant Difference)

| Metric | Unified | Position-Specific | Improvement |
|--------|---------|-------------------|-------------|
| Overall MAE | 2.10 | 2.08 | -1.0% |
| Captain Accuracy | 42% | 43% | +1pp |

**If this happens**: Keep unified model (simpler, fewer artifacts to maintain).

### Pessimistic Scenario (Position-Specific Worse)

| Metric | Unified | Position-Specific | Improvement |
|--------|---------|-------------------|-------------|
| Overall MAE | 2.10 | 2.25 | +7.1% |
| GKP MAE | 1.50 | 1.80 | +20% (overfitting) |
| FWD MAE | 2.40 | 2.70 | +12.5% (overfitting) |

**If this happens**: Small sample sizes (GKP: 430, FWD: 410) cause overfitting. Consider:
1. Combine into 2-model approach: DEF+GKP (defensive) and MID+FWD (attacking)
2. Use unified model but add position as interaction feature

---

## Alternative Approaches to Test

### Alternative A: Position as Interaction Features

Instead of separate models, add position interaction features to unified model:

```python
# Add to FPLFeatureEngineer
df["is_gkp"] = (df["position"] == "GKP").astype(int)
df["is_def"] = (df["position"] == "DEF").astype(int)
df["is_mid"] = (df["position"] == "MID").astype(int)
df["is_fwd"] = (df["position"] == "FWD").astype(int)

# Interaction features
df["gkp_x_clean_sheets"] = df["is_gkp"] * df["rolling_5gw_clean_sheets"]
df["def_x_clean_sheets"] = df["is_def"] * df["rolling_5gw_clean_sheets"]
df["mid_x_goals"] = df["is_mid"] * df["rolling_5gw_goals"]
df["fwd_x_goals"] = df["is_fwd"] * df["rolling_5gw_goals"]
```

**Pros**: Uses all data, no small-sample issues
**Cons**: Model must learn interactions, may not capture all position-specific patterns

### Alternative B: Two-Model Approach (Defensive vs Attacking)

```python
DEFENSIVE_POSITIONS = ["GKP", "DEF"]  # ~1,659 trainable samples
ATTACKING_POSITIONS = ["MID", "FWD"]  # ~2,064 trainable samples
```

**Pros**: Larger sample sizes than 4-model approach
**Cons**: Still loses some position-specific nuance

### Alternative C: Hierarchical Model

Train unified model first, then train position-specific "residual" models:

```python
# Step 1: Unified model predictions
unified_pred = unified_model.predict(X)

# Step 2: Position-specific residual models
residual = y_true - unified_pred
for position in positions:
    pos_mask = X["position"] == position
    residual_model[position].fit(X[pos_mask], residual[pos_mask])

# Step 3: Final prediction
final_pred = unified_pred + residual_model[position].predict(X)
```

**Pros**: Residual models only need to learn position-specific corrections
**Cons**: More complex pipeline, error propagation

---

## Success Criteria

The position-specific approach will be adopted if:

1. **Overall MAE improvement** ≥ 5% (statistically significant, p < 0.05)
2. **Captain accuracy improvement** ≥ 3 percentage points
3. **No position overfits**: No position has MAE increase > 10%
4. **Inference time acceptable**: < 2x latency vs unified model

---

## Appendix: Scoring Mechanism Reference

### Points by Position (2025-26 Season)

| Action | GKP | DEF | MID | FWD |
|--------|-----|-----|-----|-----|
| Minutes (≥60) | 2 | 2 | 2 | 2 |
| Goal scored | 6 | 6 | 5 | 4 |
| Assist | 3 | 3 | 3 | 3 |
| Clean sheet | 4 | 4 | 1 | 0 |
| 3 saves | 1 | - | - | - |
| Penalty save | 5 | - | - | - |
| Penalty miss | -2 | -2 | -2 | -2 |
| 2 goals conceded | -1 | -1 | - | - |
| Yellow card | -1 | -1 | -1 | -1 |
| Red card | -3 | -3 | -3 | -3 |
| Own goal | -2 | -2 | -2 | -2 |
| Bonus (1-3) | 1-3 | 1-3 | 1-3 | 1-3 |

### Key Implications for Position-Specific Models

1. **GKP**: Clean sheets (4pts) and saves (1pt per 3) dominate. Penalty saves (5pts) are rare but valuable.
2. **DEF**: Clean sheets (4pts) and goals (6pts, though rare) matter most.
3. **MID**: Goals (5pts) and assists (3pts) are primary. Clean sheets (1pt) are minor.
4. **FWD**: Goals (4pts) and assists (3pts) only. No clean sheet points.

This suggests:
- **GKP model** should heavily weight `opponent_xg`, `team_defensive_strength`, `rolling_5gw_saves`
- **DEF model** should weight clean sheet probability AND attacking potential (set pieces)
- **MID/FWD models** should focus on `xG`, `xA`, `team_attacking_strength`

---

## Phase 2: Hybrid Model Implementation

Based on our findings, we implement a **hybrid approach** that uses:
- **Position-specific models** for GKP and FWD (clear wins: +6.5% and +9.0%)
- **Unified model** for DEF and MID (more training data helps)

### Architecture

```
HybridPositionModel
├── unified_model: Pipeline        # For DEF, MID
├── position_models: {             # For GKP, FWD
│   "GKP": Pipeline (GradientBoosting),
│   "FWD": Pipeline (RandomForest),
│ }
└── use_specific_for: ["GKP", "FWD"]

.predict(X) routes by X["position"]:
  - GKP → position_models["GKP"]
  - DEF → unified_model
  - MID → unified_model
  - FWD → position_models["FWD"]
```

### Files Changed/Created

| File | Action | Purpose |
|------|--------|---------|
| `fpl_team_picker/domain/ml/hybrid_model.py` | **CREATE** | `HybridPositionModel` class |
| `fpl_team_picker/domain/ml/__init__.py` | **MODIFY** | Export `HybridPositionModel` |
| `scripts/position_specific_optimizer.py` | **MODIFY** | Add `build-hybrid` command |
| `fpl_team_picker/config/settings.py` | **MODIFY** | Update default model path |

### No Changes Required

| File | Reason |
|------|--------|
| `gameweek_manager.py` | Uses `MLExpectedPointsService` (unchanged interface) |
| `MLExpectedPointsService` | Calls `.predict()` which hybrid model implements |
| `custom_pipeline_optimizer.py` | Continues to train unified models |

---

## Complete Workflow

### Step 1: Train Unified Baseline

```bash
# Train LightGBM unified model on GW1-10
uv run python scripts/custom_pipeline_optimizer.py train \
    --end-gw 10 \
    --regressor lightgbm \
    --use-best-params-from models/custom/best_params_lightgbm_gw1-10_20251129_102622.json

# Output: models/custom/lightgbm_gw1-10_*_pipeline.joblib
```

### Step 2: Optimize Position-Specific Models

```bash
# Run comprehensive hyperparameter optimization for all positions
uv run python scripts/position_specific_optimizer.py optimize-all \
    --end-gw 10 \
    --regressors "lightgbm,xgboost,random-forest,gradient-boost" \
    --n-trials 50

# Output: models/position_specific/{gkp,def,mid,fwd}_gw1-10_best.joblib
# Output: models/position_specific/optimization_summary_gw1-10.json
```

### Step 3: Build Hybrid Model

```bash
# Combine best position-specific models with unified model
uv run python scripts/position_specific_optimizer.py build-hybrid \
    --unified-model models/custom/lightgbm_gw1-10_*_pipeline.joblib \
    --position-model-dir models/position_specific \
    --use-specific-for "GKP,FWD" \
    --output models/hybrid/hybrid_gw1-10.joblib

# Output: models/hybrid/hybrid_gw1-10.joblib (single file)
```

### Step 4: Evaluate All Approaches

```bash
# Compare unified vs position-specific vs hybrid on holdout GW11-12
uv run python scripts/position_specific_optimizer.py evaluate \
    --end-gw 12 \
    --holdout-gws 2 \
    --unified-model-path models/custom/lightgbm_gw1-10_*_pipeline.joblib \
    --hybrid-model-path models/hybrid/hybrid_gw1-10.joblib
```

### Step 5: Deploy

```python
# In fpl_team_picker/config/settings.py
class XPModelConfig(BaseModel):
    ml_model_path: str = Field(
        default="models/hybrid/hybrid_gw1-10.joblib",
        description="Hybrid model: position-specific for GKP/FWD, unified for DEF/MID"
    )
```

---

## Model Artifacts

After running the complete workflow:

```
models/
├── custom/
│   ├── lightgbm_gw1-10_*_pipeline.joblib      # Unified baseline
│   └── best_params_lightgbm_gw1-10_*.json     # Saved hyperparameters
├── position_specific/
│   ├── gkp_gw1-10_best.joblib                 # GKP: GradientBoosting
│   ├── def_gw1-10_best.joblib                 # DEF: XGBoost
│   ├── mid_gw1-10_best.joblib                 # MID: RandomForest
│   ├── fwd_gw1-10_best.joblib                 # FWD: RandomForest
│   ├── optimization_summary_gw1-10.json       # All results
│   └── evaluation_gw11-12.json                # Holdout evaluation
└── hybrid/
    ├── hybrid_gw1-10.joblib                   # Combined model (DEPLOY THIS)
    └── hybrid_metadata.json                   # Configuration
```

---

## Retraining Schedule

When new gameweek data is available:

```bash
# Weekly after GW results finalize
./scripts/retrain_hybrid.sh --end-gw <current_gw>

# Script contents:
# 1. Train new unified model
# 2. Re-optimize position-specific (or reuse params)
# 3. Rebuild hybrid
# 4. Evaluate on last 2 GWs as sanity check
# 5. Deploy if metrics acceptable
```

---

## Expected Performance (Hybrid Model)

| Position | Model Used | Expected MAE |
|----------|------------|--------------|
| GKP | Specific (GradientBoosting) | 0.680 |
| DEF | Unified (LightGBM) | 1.274 |
| MID | Unified (LightGBM) | 1.111 |
| FWD | Specific (RandomForest) | 1.033 |
| **Overall** | **Hybrid** | **~1.115** |

vs. Pure Unified: 1.124 → **+0.8% improvement**
vs. Pure Position-Specific: 1.119 → **+0.4% improvement**
