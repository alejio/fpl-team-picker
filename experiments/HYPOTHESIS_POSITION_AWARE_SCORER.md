# Hypothesis: Position-Aware Scorer Training Experiment

## Research Question

**Can training with a position-aware FPL decision scorer produce better models than training with point-based loss (MAE, Huber)?**

## Background

### Current State (2025-10-31)

| Model | Scorer | MAE | Position Top-K | XI Efficiency | Captain | Comprehensive |
|-------|--------|-----|----------------|---------------|---------|---------------|
| **TPOT (8hr)** | fpl_weighted_huber | 1.752 | 0.635 | 0.938 | 1.000 | **0.827** |
| **Custom RF (2min)** | fpl_weighted_huber | **0.632** | 0.585 | 0.884 | 0.560 | 0.695 |

### Key Findings

1. **Lower MAE ≠ Better FPL Performance**
   - Custom RF: 64% better MAE but 16% worse comprehensive score
   - TPOT: Worse MAE but perfect captain accuracy and better XI selection

2. **Variance Compression Problem**
   - Custom RF: Max prediction 8.98 (compressed)
   - TPOT: Max prediction 12.61 (preserved)
   - Actual: Max points 23.0
   - **Custom RF can't predict hauls!**

3. **Position-Aware Metrics Reveal Truth**
   - Old "top-15 overall" metric: Custom RF wins (9.2 vs 8.2)
   - New "top-10 per position" metric: TPOT wins (0.635 vs 0.585)
   - Position-awareness mirrors how OptimizationService actually works

---

## Hypothesis

**Training with position-aware comprehensive scorer will produce a model that:**

1. ✅ Preserves variance (can identify hauls)
2. ✅ Optimizes for actual FPL decisions (XI selection, captain)
3. ✅ Better position-specific rankings (mirrors optimizer)
4. ❓ May have higher MAE (but that's OK - optimizer doesn't care!)

**Expected Result**: Better comprehensive FPL score than both TPOT and Custom RF

---

## Experimental Setup

### New Training Approach

**Model**: TPOT (5-minute quick run)

**Scorer**: `fpl_comprehensive_team_scorer`
```python
def fpl_comprehensive_team_scorer(y_true, y_pred, position_labels, formation="4-4-2"):
    """
    Combined FPL scorer that evaluates:
    1. Position-aware top-10 overlap (40%)
    2. Starting XI point efficiency (40%)
    3. Captain accuracy within XI (20%)

    This directly mirrors what OptimizationService optimizes.
    """
```

### Why This Should Work

1. **Variance Preservation**: Can't achieve high overlap without differentiating players
2. **Position-Aware**: Trains on same criteria optimizer uses (top-K per position)
3. **Formation-Aware**: Tests all 8 formations, mimics SA objective function
4. **Captain-Aware**: Forces model to identify extreme values (hauls)

### Training Parameters

- Data: GW1-9 (2,976 samples from GW6-9 CV set)
- Features: 99 (same as TPOT/Custom RF)
- CV: 3-fold temporal walk-forward (GW6→7, GW6-7→8, GW6-8→9)
- Time: 5 minutes (quick test)
- Population: 20
- Config: TPOT light (faster search space)

---

## Expected Outcomes

### Scenario A: Hypothesis Confirmed ✅

**Position-Aware TPOT** outperforms both baselines:

| Metric | TPOT (Huber) | Custom RF (Huber) | **New (Position)** |
|--------|--------------|-------------------|-------------------|
| MAE | 1.752 | 0.632 | ~1.2-1.5 |
| Position Top-K | 0.635 | 0.585 | **>0.65** |
| XI Efficiency | 0.938 | 0.884 | **>0.94** |
| Captain Accuracy | 1.000 | 0.560 | **>0.90** |
| **Comprehensive** | 0.827 | 0.695 | **>0.85** |

**Interpretation**:
- Training on FPL decisions beats training on point predictions
- Position-aware loss preserves variance while optimizing decisions
- MAE is irrelevant - optimizer uses relative rankings, not absolute values

---

### Scenario B: Partial Success ⚠️

**Position-Aware TPOT** improves some metrics:

| Metric | Result | Why |
|--------|--------|-----|
| Position Top-K | ✅ Better | Directly optimized |
| XI Efficiency | ✅ Better | Formation testing in scorer |
| Captain Accuracy | ❌ Similar | Hard to optimize directly (sparse signal) |
| Comprehensive | ✅ Slightly better | Net positive from top-K + XI |

**Interpretation**:
- Position-aware training helps but isn't magic
- Captain accuracy may need separate model or post-hoc calibration
- Still beats Custom RF, competitive with TPOT (Huber)

---

### Scenario C: Hypothesis Rejected ❌

**Position-Aware TPOT** underperforms:

| Metric | Result | Why |
|--------|--------|-----|
| All metrics | ❌ Worse | Scorer too complex, signal too noisy |
| Training | ⏳ Slow | Expensive scorer (O(n²) formations) |
| Convergence | ❌ Poor | Optimization landscape too rough |

**Interpretation**:
- Position-aware scorer may be too expensive for TPOT's genetic search
- Need to simplify scorer (maybe just position top-K, drop formations)
- OR use gradient-based optimization (XGBoost with custom loss)

---

## Alternative Approaches (if hypothesis rejected)

### Option 1: Simplified Position-Aware Scorer

```python
def fpl_position_ranking_loss(y_true, y_pred, position_labels):
    """
    Simpler: Just optimize position-specific Spearman correlation.

    No formation enumeration, no XI simulation.
    Much faster, but less direct FPL alignment.
    """
    scores = []
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        mask = position_labels == pos
        if mask.sum() > 2:
            spearman = spearmanr(y_true[mask], y_pred[mask])[0]
            scores.append(spearman)

    return np.mean(scores)
```

**Pros**: Fast, preserves variance, position-aware
**Cons**: Less direct FPL alignment

---

### Option 2: XGBoost with Custom Gradient

```python
# Custom loss that penalizes variance compression
def fpl_variance_preserving_loss(y_pred, y_true):
    """
    MAE + variance penalty.

    Minimize: MAE + λ * max(0, target_std - pred_std)

    Forces model to preserve variance while minimizing error.
    """
    mae = np.abs(y_true - y_pred).mean()
    std_penalty = max(0, y_true.std() - y_pred.std())

    return mae + 0.5 * std_penalty
```

**Pros**: Gradient-based (efficient), explicit variance preservation
**Cons**: Indirect FPL optimization, hyperparameter tuning needed

---

### Option 3: Ensemble Post-Calibration

```python
# Train on MAE, post-process to preserve variance
def calibrate_predictions(y_pred, y_train_actual):
    """
    Stretch predictions to match actual variance.

    y_calibrated = mean + (y_pred - mean) * (actual_std / pred_std)
    """
    pred_mean = y_pred.mean()
    pred_std = y_pred.std()
    actual_std = y_train_actual.std()

    scale_factor = actual_std / pred_std
    y_calibrated = pred_mean + (y_pred - pred_mean) * scale_factor

    return y_calibrated
```

**Pros**: Works with any model, simple post-processing
**Cons**: Doesn't fix position-specific issues, crude adjustment

---

## Success Criteria

### Minimum Viable Success

- Position Top-K: >0.62 (better than Custom RF, close to TPOT)
- Comprehensive: >0.75 (significantly better than Custom RF)
- Training Time: <10 minutes (practical for retraining)

### Strong Success

- Position Top-K: >0.65 (better than both baselines)
- XI Efficiency: >0.94 (near-optimal team selection)
- Captain Accuracy: >0.80 (consistent haul detection)
- Comprehensive: >0.85 (best overall FPL performance)

### Game-Changing Success

- Position Top-K: >0.70 (major improvement)
- Comprehensive: >0.90 (transformative FPL model)
- Generalizes to GW10+ (not just training set overfitting)

---

## Next Steps After Experiment

### If Successful ✅

1. Train full 8-hour TPOT with position-aware scorer
2. Validate on GW10+ (true out-of-sample)
3. Deploy to production (replace current ML service)
4. Update CLAUDE.md with new training methodology

### If Partially Successful ⚠️

1. Try simplified position-ranking scorer (faster)
2. Experiment with variance-preserving loss (XGBoost)
3. Consider ensemble of models (MAE + position-aware)

### If Unsuccessful ❌

1. Investigate why (slow? noisy signal? wrong metric?)
2. Fall back to TPOT (Huber) for production
3. Focus on post-hoc calibration instead
4. Consider non-ML approaches (rule-based with variance)

---

## Timeline

- **Training Start**: 2025-10-31 19:48 UTC
- **Training End**: 2025-10-31 19:53 UTC (5 min)
- **Evaluation**: 2025-10-31 19:55 UTC
- **Decision**: 2025-10-31 20:00 UTC

---

## Key Insight

**The optimal FPL scorer is NOT the one with lowest MAE.**

**The optimal FPL scorer is the one that:**
1. Preserves position-specific variance
2. Enables optimizer to identify differentials
3. Aligns with actual FPL decision criteria

**This experiment tests whether we can bake this directly into training.**

---

*Experiment in progress: 2025-10-31*
*Hypothesis: Position-aware training > point-based training*
*Status: Training (5 min)*
