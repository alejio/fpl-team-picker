# Hauler-First Strategy Implementation

Based on empirical analysis of top 1% managers, this document outlines changes to reflect their strategy: **Hauler identification matters more than balanced scoring**.

## ✅ Implementation Status

| Change | Status | Files Modified |
|--------|--------|----------------|
| Variance-preserving scorer | ✅ Done | `scripts/ml_training_utils.py` |
| Captain ceiling enhancement | ✅ Done | `captain_service.py` |
| Register new scorer | ✅ Done | `trainer.py`, `custom_pipeline_optimizer.py` |
| SA optimization ceiling bonus | ✅ Done | `transfer_sa.py` |
| Configuration options | ✅ Done | `settings.py` |
| LP optimization bonus | ⏳ Pending | `transfer_lp.py` |

## Key Finding

| Metric | Top 1% | Average | Impact |
|--------|--------|---------|--------|
| **Captain Points** | 18.7 | 15.0 | +24% ⭐ |
| **Haulers per GW** | 3.0 | 2.4 | +25% ⭐ |
| Squad Consistency | 0.052 | 0.078 | Better |
| Balanced Players | 6.6 | 6.0 | +10% |

**Conclusion**: Top managers don't just identify haulers - they identify the **optimal haulers** with higher ceilings.

---

## Implementation Areas

### 1. ML Training: Use Hauler-Focused Scorer (HIGH PRIORITY)

**Problem**: Current default scorer is `neg_mean_absolute_error` which penalizes all errors equally. This compresses variance and makes the model predict average returns for everyone.

**Solution**: Use `fpl_hauler_capture_scorer` as the primary training metric.

**File**: `scripts/train_model.py`

```python
# Change default scorer from neg_mean_absolute_error to fpl_hauler_capture
@app.command("unified")
def train_unified(
    ...
    scorer: str = typer.Option("fpl_hauler_capture", help="Scoring metric"),  # CHANGED
    ...
)
```

**File**: `fpl_team_picker/domain/services/ml_training/trainer.py`

Add `fpl_hauler_capture_scorer` to the available scorers.

---

### 2. ML Training: Preserve Variance (HIGH PRIORITY)

**Problem**: Models with low MAE often compress predictions to the mean (5-7 xP for everyone), missing haulers entirely.

**Solution**: Add a variance preservation penalty to the loss function.

**New scorer** in `scripts/ml_training_utils.py`:

```python
def fpl_hauler_ceiling_scorer(y_true, y_pred, sample_weight=None):
    """
    Hauler-optimized scorer that preserves prediction variance.

    Components:
    1. Hauler capture (50%): Identify the 8+ point scorers
    2. Captain accuracy (30%): Get the top scorer right
    3. Variance preservation (20%): Don't compress predictions
    """
    # Hauler capture (existing)
    hauler_score = fpl_hauler_capture_scorer(y_true, y_pred)

    # Captain accuracy (top-3 overlap)
    captain_score = fpl_captain_pick_scorer(y_true, y_pred)

    # Variance preservation
    # Penalize models that compress predictions
    actual_variance = np.var(y_true)
    pred_variance = np.var(y_pred)
    variance_ratio = pred_variance / actual_variance if actual_variance > 0 else 0
    variance_score = min(variance_ratio, 1.0)  # Cap at 1.0 (don't reward over-variance)

    # Combine with weights emphasizing hauler identification
    combined = 0.50 * hauler_score + 0.30 * captain_score + 0.20 * variance_score

    return combined
```

---

### 3. Captain Selection: Ceiling Over Safety (MEDIUM PRIORITY)

**Current state**: Already uses 90th percentile (xP + 1.28*uncertainty), which is good.

**Enhancement**: Increase the ceiling multiplier for players with high haul potential.

**File**: `fpl_team_picker/domain/services/optimization/captain_service.py`

```python
# Line ~78-84: Enhance the upside calculation
# Current:
xp_upside = xp_value + (1.28 * uncertainty)

# Enhanced: Use 95th percentile for high-ceiling players
if uncertainty > 1.5:  # High variance = haul potential
    # 95th percentile for explosive players
    xp_upside = xp_value + (1.645 * uncertainty)  # 95th percentile
else:
    # 90th percentile for consistent players
    xp_upside = xp_value + (1.28 * uncertainty)   # 90th percentile
```

**Additional**: Add a "haul probability" bonus to captain scoring:

```python
# Bonus for players with high ceiling (xP + 2*uncertainty > 12)
haul_ceiling = xp_value + 2 * uncertainty
if haul_ceiling > 12:  # Potential for big haul
    haul_bonus = (haul_ceiling - 12) * 0.1  # Up to 20% bonus for 15+ ceiling
    final_score *= (1 + haul_bonus)
```

---

### 4. Optimization Objective: Hauler Potential Bonus (MEDIUM PRIORITY)

**Problem**: LP optimization maximizes pure xP sum, treating a player with 6.0 xP the same regardless of ceiling.

**Solution**: Add an "upside bonus" to the objective function.

**File**: `fpl_team_picker/domain/services/optimization/transfer_lp.py`

```python
# Current objective (line ~200):
prob += pulp.lpSum([xp_values[i] * x[i] for i in range(n_players)])

# Enhanced objective: Add ceiling bonus
# For each player, add bonus based on their upside potential
ceiling_bonus = {}
for i, player in enumerate(players_list):
    xp = player.get(xp_column, 0)
    uncertainty = player.get("xP_uncertainty", 0)

    # Calculate ceiling (95th percentile)
    ceiling = xp + 1.645 * uncertainty

    # Bonus for high-ceiling players (ceiling > 10)
    if ceiling > 10:
        ceiling_bonus[i] = (ceiling - 10) * 0.15  # Up to 1.5 bonus for 20 ceiling
    else:
        ceiling_bonus[i] = 0

# Modified objective
prob += pulp.lpSum([
    (xp_values[i] + ceiling_bonus[i]) * x[i]
    for i in range(n_players)
])
```

---

### 5. Configuration: Hauler Strategy Settings (LOW PRIORITY)

**File**: `fpl_team_picker/config/settings.py`

Add new configuration section for hauler strategy:

```python
class HaulerStrategyConfig(BaseModel):
    """Hauler-first strategy configuration"""

    # Thresholds
    hauler_threshold: int = Field(
        default=8,
        description="Points threshold for a 'hauler' (8+ points)"
    )
    ceiling_multiplier: float = Field(
        default=1.645,  # 95th percentile
        description="Standard deviations for ceiling calculation"
    )

    # Weights for training scorer
    hauler_capture_weight: float = Field(
        default=0.50,
        description="Weight for hauler capture in training scorer"
    )
    captain_accuracy_weight: float = Field(
        default=0.30,
        description="Weight for captain accuracy in training scorer"
    )
    variance_preservation_weight: float = Field(
        default=0.20,
        description="Weight for variance preservation in training scorer"
    )

    # Optimization bonuses
    ceiling_bonus_enabled: bool = Field(
        default=True,
        description="Add ceiling bonus to optimization objective"
    )
    ceiling_bonus_factor: float = Field(
        default=0.15,
        description="Bonus factor per point above threshold (default 0.15)"
    )
```

---

## Implementation Order

### Phase 1: Quick Wins (Immediate Impact)

1. **Change default training scorer to `fpl_hauler_capture`**
   - File: `scripts/train_model.py`
   - Impact: Model will optimize for hauler identification

2. **Add `fpl_hauler_capture_scorer` to trainer**
   - File: `fpl_team_picker/domain/services/ml_training/trainer.py`
   - Enables using hauler scorer in unified CLI

### Phase 2: Model Enhancement (1-2 hours)

3. **Create `fpl_hauler_ceiling_scorer` with variance preservation**
   - File: `scripts/ml_training_utils.py`
   - Prevents variance compression

4. **Enhance captain selection ceiling calculation**
   - File: `fpl_team_picker/domain/services/optimization/captain_service.py`
   - Use 95th percentile for high-variance players

### Phase 3: Optimization Enhancement (2-3 hours)

5. **Add ceiling bonus to LP objective**
   - File: `fpl_team_picker/domain/services/optimization/transfer_lp.py`
   - Prefer players with hauler potential

6. **Add configuration options**
   - File: `fpl_team_picker/config/settings.py`
   - Make strategy configurable

---

## Validation

After implementing, validate by:

1. **Retrain model** with `fpl_hauler_capture` scorer
2. **Compare predictions**: Check variance is preserved (not compressed)
3. **Backtest captain picks**: Compare to top 1% captain success rate
4. **Track hauler capture**: Do we now capture 3.0+ haulers per GW?

Target metrics:
- Captain hauler rate: >75%
- Avg captain points: >17
- Haulers per GW: >2.8
- Prediction variance: ~50%+ of actual variance
