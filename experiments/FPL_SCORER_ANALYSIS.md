# FPL Scorer Analysis: What Should We Actually Optimize?

## The Variance Compression Problem

**Current Situation**:
- Both TPOT and Custom RF trained with `fpl_weighted_huber` (3x underest., 5x haul focus)
- Custom RF: MAE 0.632, **max prediction 8.98**, no 10+ predictions, 68.6% in 0-2 range
- TPOT: MAE 1.752, **max prediction 12.61**, 0.2% 10+ predictions, 53.0% in 0-2 range
- Actual: **max 23 pts**, 26 point range, std dev 2.367

**Key Finding**: Despite haul-focused scorer, Custom RF compresses variance by 21% vs TPOT (range ratio 0.79x).

---

## What Are We Actually Optimizing For in FPL?

FPL is NOT about minimizing prediction error - it's about **making optimal decisions**:

### 1. **Team Selection** (GW Planning)
   - Pick top 15 players → Need accurate **relative ranking**
   - Don't care about exact points → Care about **who beats who**
   - Metric: **Top-15 overlap**, **Spearman correlation**

### 2. **Captain Selection** (2x Multiplier)
   - Pick THE top scorer → Need to identify **hauls**
   - Extreme penalty for missing differentials → Need **variance**
   - Metric: **Captain accuracy**, **Top-1 precision**

### 3. **Transfer Decisions** (Limited budget)
   - Identify value plays → Need **risk-adjusted returns**
   - Avoid traps (expensive duds) → Penalize overestimation for premiums
   - Metric: **Value picks accuracy**, **bust avoidance**

### 4. **Chip Timing** (Once per season)
   - Bench Boost: Need all 15 to score → Need **floor predictions**
   - Triple Captain: Need hauls → Need **ceiling predictions**
   - Metric: **Calibrated confidence intervals**

---

## Current Scorer Limitations

### `fpl_weighted_huber` (Used by Both Models)

```python
# Simplified logic:
# - 3x penalty for underestimation (pred < actual)
# - 5x penalty for missing hauls (actual >= 10)
# - Huber loss for robustness
```

**Problems**:
1. ❌ **Still point-based** - Optimizes prediction error, not decision quality
2. ❌ **Doesn't preserve variance** - RF averaging smooths out extremes
3. ❌ **No ranking component** - Doesn't care if Haaland beats Salah
4. ❌ **Arbitrary haul threshold** - Why 10? Why not 8 or 12?
5. ❌ **No position context** - GKP 10pts ≠ FWD 10pts

---

## Alternative Scoring Frameworks

### Option 1: **Ranking-Based Loss** (Best for Team Selection)

Focus on **pairwise comparisons** rather than absolute error:

```python
# Pairwise Ranking Loss (LambdaRank, RankNet)
# - Penalize when pred_i < pred_j but actual_i > actual_j
# - Discount lower-ranked pairs (who cares about #501 vs #502?)
# - Position-weighted (top-15 matters most)

def fpl_pairwise_ranking_loss(y_true, y_pred):
    """
    Optimizes ranking quality with top-k focus.

    For all pairs (i, j) where actual_i > actual_j:
        - Penalize if pred_i < pred_j (wrong order)
        - Weight by position (top-15 pairs weighted 10x)
        - Discount by margin (big gaps matter more)
    """
    pass
```

**Pros**:
- ✅ Directly optimizes team selection
- ✅ Preserves relative differences (variance)
- ✅ Position-aware (top-15 focus)

**Cons**:
- ⚠️ Doesn't optimize absolute values (captain needs exact haul detection)
- ⚠️ Computationally expensive (O(n²) pairs)

---

### Option 2: **Top-K Precision Loss** (Best for Captain Pick)

Focus on **correctly identifying the extreme scorers**:

```python
def fpl_top_k_precision_loss(y_true, y_pred, k=15):
    """
    Penalizes missing top-k scorers.

    1. Identify actual top-k players
    2. Identify predicted top-k players
    3. Heavy penalty for each missed top-k player
    4. Bonus for correct captain (top-1)
    """
    actual_top_k = argsort(y_true)[-k:]
    pred_top_k = argsort(y_pred)[-k:]

    # Miss penalty: exponential with rank
    miss_penalty = sum([
        (k - rank) ** 2 for rank, player in enumerate(actual_top_k)
        if player not in pred_top_k
    ])

    # Captain bonus
    captain_bonus = -100 if actual_top_k[-1] == pred_top_k[-1] else 0

    return miss_penalty + captain_bonus
```

**Pros**:
- ✅ Directly targets FPL decisions (team + captain)
- ✅ Preserves variance (need extreme values to rank high)
- ✅ Differentiates importance (missing captain >> missing #15)

**Cons**:
- ⚠️ Ignores middle/bottom players (less robust)
- ⚠️ High variance (small changes = big penalty swings)

---

### Option 3: **Calibrated Quantile Loss** (Best for Risk Management)

Predict **confidence intervals** rather than point estimates:

```python
def fpl_quantile_loss(y_true, y_pred_quantiles):
    """
    Predict P10, P50, P90 for each player.

    Loss components:
    1. Median accuracy (P50 vs actual) - base prediction
    2. Calibration (P90-P10 should contain 80% of outcomes)
    3. Ceiling identification (P90 > 10 for actual hauls)
    """
    p10, p50, p90 = y_pred_quantiles

    # Median error
    median_loss = huber_loss(y_true, p50)

    # Calibration: penalize if interval too narrow/wide
    coverage = (y_true >= p10) & (y_true <= p90)
    calibration_loss = abs(coverage.mean() - 0.80)

    # Haul ceiling: P90 must be high for actual hauls
    haul_mask = y_true >= 10
    ceiling_loss = max(0, 10 - p90[haul_mask]).mean()

    return median_loss + 5 * calibration_loss + 10 * ceiling_loss
```

**Pros**:
- ✅ Preserves variance (explicit ceiling predictions)
- ✅ Useful for all FPL decisions (team/captain/chips)
- ✅ Handles uncertainty properly

**Cons**:
- ⚠️ Requires quantile regression (more complex)
- ⚠️ 3x predictions per player (slower inference)

---

### Option 4: **FPL Expected Value Loss** (Best for Overall Utility)

Optimize **expected FPL points** from decisions:

```python
def fpl_expected_value_loss(y_true, y_pred, player_prices):
    """
    Simulates actual FPL decisions and measures points.

    1. Select top-15 by predicted xP (respecting budget/formation)
    2. Captain = argmax(pred)
    3. Calculate actual FPL points from those decisions
    4. Loss = negative actual points
    """
    # Simplified (ignoring budget/formation for now)
    top_15_pred = argsort(y_pred)[-15:]
    captain = argmax(y_pred)

    # Actual points from this selection
    team_points = y_true[top_15_pred].sum()
    captain_points = y_true[captain]  # +1x from captain
    total_points = team_points + captain_points

    return -total_points  # Minimize negative = maximize points
```

**Pros**:
- ✅ **Directly optimizes FPL score** - the ultimate metric!
- ✅ Handles budget/formation constraints
- ✅ Automatically balances team vs captain

**Cons**:
- ⚠️ Non-differentiable (need RL/evolution)
- ⚠️ Sparse signal (only 15 players contribute)
- ⚠️ Ignores bench/future transfers

---

## Variance Preservation Analysis

**Why does Custom RF compress variance despite weighted Huber?**

1. **Ensemble Averaging**: RandomForest = average of 100 trees
   - Each tree has full variance
   - Average compresses extremes (regression to mean)
   - No amount of loss weighting fixes this architectural issue

2. **Scorer Design**: `fpl_weighted_huber` penalizes underestimation but doesn't **require variance**
   - A model predicting everyone at 5pts could have low loss if most actuals are 3-7
   - Need a scorer that explicitly rewards variance (e.g., predicting 15 for someone who gets 15)

3. **Training Data**: Few extreme outcomes (0.2% hauls)
   - Model learns "safe" predictions (6-8 range)
   - Rare events underweighted even with 5x penalty

---

## Recommendations: Optimal FPL Scorer Design

### Primary Scorer: **Hybrid Ranking + Top-K Loss**

```python
def fpl_optimal_scorer(y_true, y_pred):
    """
    Combines ranking quality with top-k precision.

    Components:
    1. Spearman correlation (60%) - preserve relative order
    2. Top-15 overlap (30%) - correct team selection
    3. Captain accuracy (10%) - identify top scorer

    All position-weighted, variance-preserving.
    """

    # 1. Ranking correlation (all players)
    spearman = scipy.stats.spearmanr(y_true, y_pred)[0]

    # 2. Top-15 precision
    actual_top15 = set(argsort(y_true)[-15:])
    pred_top15 = set(argsort(y_pred)[-15:])
    top15_overlap = len(actual_top15 & pred_top15) / 15

    # 3. Captain accuracy (top-1)
    captain_correct = 1.0 if argmax(y_true) == argmax(y_pred) else 0.0

    # Weighted combination (maximize, so return negative for minimization)
    score = 0.60 * spearman + 0.30 * top15_overlap + 0.10 * captain_correct

    return -score  # Negative because sklearn minimizes
```

### Secondary Scorers for Experimentation:

1. **`fpl_variance_preserving_loss`**: Penalize if `std(y_pred) < 0.8 * std(y_true)`
2. **`fpl_differential_detector`**: Bonus for correctly predicting 15+ hauls
3. **`fpl_value_optimizer`**: Points per million accuracy for budget players

---

## Proposed Experiment

**Goal**: Find scorer that maximizes FPL decisions without variance compression.

### Test Matrix:

| Scorer | Focus | Expected Outcome |
|--------|-------|------------------|
| `mae` (baseline) | Min error | Variance compression, poor top-k |
| `fpl_weighted_huber` (current) | Haul penalty | Some compression (proven) |
| `fpl_ranking_loss` (new) | Relative order | Better ranking, preserved variance |
| `fpl_top_k_precision` (new) | Team selection | Best overlap, may sacrifice MAE |
| `fpl_expected_value` (new) | Direct FPL points | **Optimal for FPL**, complex training |

### Metrics to Track:

1. **MAE** - Statistical accuracy (reference only)
2. **Spearman** - Ranking quality (primary)
3. **Top-15 overlap** - Team selection (primary)
4. **Captain accuracy** - Differential detection (primary)
5. **Prediction std dev** - Variance preservation (critical!)
6. **P90 prediction** - Ceiling detection for hauls

---

## Hypothesis

**The optimal FPL scorer is NOT point-based but decision-based**:

- Minimize: Top-15 selection errors
- Maximize: Captain pick accuracy
- Preserve: Prediction variance (std dev within 20% of actual)
- Bonus: Correctly identify 15+ hauls (differentials)

**Expected Result**: Model with worse MAE but better Spearman/Top-15 overlap will deliver more FPL points.

---

## Next Steps

1. ✅ Implement `fpl_ranking_loss` scorer (pairwise ranking)
2. ✅ Implement `fpl_top_k_precision` scorer (team selection focus)
3. ✅ Implement `fpl_variance_penalty` (explicit variance requirement)
4. 🔄 Retrain Custom RF with each scorer
5. 🔄 Compare variance, ranking quality, and top-k overlap
6. 🔄 Test on GW10 holdout data to measure generalization

**Ultimate Test**: Deploy models in parallel on GW11-15, measure actual FPL points.

---

*Generated: 2025-10-31*
*Status: Hypothesis - needs empirical validation*
