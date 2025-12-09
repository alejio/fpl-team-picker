# Probabilistic XP Calibration: Final Implementation Plan

## Core Goal

**Problem**: ML model underweights fixture difficulty for premium players because:
- Fixture difficulty is 1 of 150+ features → signal gets diluted
- Model trained on ALL players (including bench) → learns weak average effect
- Premium players (Haaland) vs bench players respond very differently to easy fixtures

**Solution**: Calibrate ML predictions using simple probabilistic approach that:
- Adjusts for **player quality × fixture difficulty** interaction
- Uses historical data to learn this interaction (15 gameweeks available)
- Provides risk-adjusted values (conservative, balanced, risk-taking)

**Principle**: **Simplicity over complexity** - start simple, validate, iterate.

---

## Approach: Simple Regularized Negative Binomial

### Why This Approach?

**Chosen Over**:
- ❌ Empirical CDF: Percentiles unreliable with small samples
- ❌ Zero-Inflated Models: Too complex, higher overfitting risk
- ❌ Full Bayesian (PyMC): Unnecessary complexity for calibration

**Chosen Because**:
- ✅ **Simple**: 2 parameters (mean, dispersion), regularized MLE
- ✅ **Sufficient**: Captures right-skew, overdispersion (what we need)
- ✅ **Robust**: Regularization prevents overfitting with small samples
- ✅ **Fast**: scipy.optimize, no complex libraries needed

### Model Structure

**For each (tier, fixture) combination**:
```python
points ~ NegativeBinomial(mean, dispersion)

# Regularized toward prior
mean = argmax[log_likelihood(data) - λ * (mean - prior_mean)²]
```

**Parameters**:
- `mean`: Expected points (calibrated for tier × fixture)
- `dispersion`: Overdispersion parameter (variance > mean)

**Regularization**:
- Shrink estimates toward prior (domain knowledge)
- Prevents overfitting with small samples

---

## Implementation

### Phase 1: Data Preparation & Analysis (Day 1)

**Script**: `scripts/analyze_calibration_data.py`

**Tasks**:
1. Load GW1-15 historical data
2. For each player observation:
   - Determine tier: Premium (≥£8m), Mid (£6-8m), Budget (<£6m)
   - Determine fixture: Easy (≥1.538), Hard (<1.538) - **2-tier only for simplicity**
   - Get actual points scored
3. Calculate sample sizes per combination
4. Identify combinations with <30 samples (use fallback)

**Output**: Data quality report
- Sample sizes per combination
- Recommendations (which combinations to use/merge)

**Expected Sample Sizes** (2-tier fixture):
- Premium + Easy: ~50-80 (adequate)
- Premium + Hard: ~100-150 (good)
- Mid + Easy: ~150-200 (good)
- Mid + Hard: ~200-250 (good)
- Budget + Easy: ~200-300 (good)
- Budget + Hard: ~300-400 (good)

---

### Phase 2: Distribution Fitting (Day 1-2)

**Script**: `scripts/build_calibration_distributions.py`

**Approach**: Regularized Maximum Likelihood Estimation

**For each (tier, fixture) combination**:

```python
from scipy.stats import nbinom
from scipy.optimize import minimize

def fit_regularized_negative_binomial(
    data: np.array,
    prior_mean: float,
    prior_dispersion: float = 2.0,
    lambda_reg: float = 0.3
) -> Tuple[float, float]:
    """Fit Negative Binomial with L2 regularization toward prior.

    Args:
        data: Array of actual points scored
        prior_mean: Prior mean (from domain knowledge)
        prior_dispersion: Prior dispersion parameter
        lambda_reg: Regularization strength (higher = more shrinkage)

    Returns:
        (fitted_mean, fitted_dispersion)
    """
    def negative_log_likelihood(params):
        mean, dispersion = params

        # Convert to negative binomial parameters
        # NB(r, p): mean = r*(1-p)/p, so p = r/(r+mean)
        r = dispersion
        p = r / (r + mean)

        # Log likelihood
        ll = np.sum(nbinom.logpmf(data, r, p))

        # L2 regularization (shrink toward prior)
        reg = lambda_reg * (
            (mean - prior_mean)**2 +
            (dispersion - prior_dispersion)**2
        )

        return -(ll - reg)  # Negative log likelihood + regularization

    # Optimize
    result = minimize(
        negative_log_likelihood,
        x0=[prior_mean, prior_dispersion],
        method='L-BFGS-B',
        bounds=[(0.1, 20.0), (0.1, 10.0)]  # Reasonable bounds
    )

    return result.x[0], result.x[1]  # mean, dispersion
```

**Prior Specification** (Domain Knowledge):
```python
PRIORS = {
    "tier_means": {
        "premium": 8.0,  # Premium players average ~8 points
        "mid": 6.0,      # Mid players average ~6 points
        "budget": 4.5    # Budget players average ~4.5 points
    },
    "fixture_boosts": {
        "easy": 2.0,     # Easy fixtures: +2 points
        "hard": -1.0     # Hard fixtures: -1 point
    },
    "dispersion": 2.0   # Moderate overdispersion
}

# For each combination
prior_mean = PRIORS["tier_means"][tier] + PRIORS["fixture_boosts"][fixture]
```

**Output**: `data/calibration/distributions.json`
```json
{
  "premium_easy": {
    "mean": 9.2,
    "dispersion": 2.3,
    "std": 4.1,
    "sample_size": 65,
    "percentile_25": 5.8,
    "percentile_75": 12.5
  },
  "premium_hard": {...},
  "mid_easy": {...},
  "mid_hard": {...},
  "budget_easy": {...},
  "budget_hard": {...}
}
```

**Validation**:
- Check fitted means make sense (premium_easy > premium_hard)
- Check sample sizes (warn if <30)
- Check percentiles are reasonable

---

### Phase 3: Calibration Service (Day 2-3)

**File**: `fpl_team_picker/domain/services/xp_calibration_service.py`

**Service**:
```python
class XPCalibrationService:
    """Simple probabilistic calibration of ML xP predictions.

    Uses regularized Negative Binomial distributions to adjust
    ML predictions based on player quality × fixture difficulty.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.distributions = self._load_distributions()
        self.priors = self._get_priors()

    def calibrate_predictions(
        self,
        players_df: pd.DataFrame,
        risk_profile: str = "balanced",
        fixture_difficulty_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Calibrate ML predictions using fitted distributions.

        Simple approach:
        1. Map player to (tier, fixture) combination
        2. Get fitted distribution for that combination
        3. Combine ML prediction with distribution mean (precision-weighted)
        4. Extract risk-adjusted value
        """
        result = players_df.copy()

        for idx, player in result.iterrows():
            # Get player attributes
            tier = self._get_player_tier(player["price"])
            fixture = self._get_fixture_category(
                player.get("fixture_difficulty", 1.0)
            )

            # Get ML prediction
            ml_xp = player.get("xP", 0)
            ml_std = player.get("xP_uncertainty", 1.5)

            # Get fitted distribution
            distribution = self.distributions.get(f"{tier}_{fixture}")
            if not distribution or distribution["sample_size"] < 30:
                # Fallback: use prior only
                posterior_mean = (
                    self.priors["tier_means"][tier] +
                    self.priors["fixture_boosts"][fixture]
                )
                posterior_std = 3.0  # High uncertainty
            else:
                posterior_mean = distribution["mean"]
                posterior_std = distribution["std"]

            # Precision-weighted combination (Bayesian update)
            ml_precision = 1.0 / (ml_std ** 2)
            posterior_precision = 1.0 / (posterior_std ** 2)
            total_precision = ml_precision + posterior_precision

            calibrated_mean = (
                (ml_precision / total_precision) * ml_xp +
                (posterior_precision / total_precision) * posterior_mean
            )

            calibrated_std = np.sqrt(1.0 / total_precision)

            # Risk-adjusted value
            if risk_profile == "conservative":
                # Use lower credible interval (~25th percentile)
                calibrated_xp = calibrated_mean - 0.67 * calibrated_std
            elif risk_profile == "risk-taking":
                # Use upper credible interval (~75th percentile)
                calibrated_xp = calibrated_mean + 0.67 * calibrated_std
            else:  # balanced
                calibrated_xp = calibrated_mean

            # Update player
            result.at[idx, "xP"] = calibrated_xp
            result.at[idx, "xP_uncertainty"] = calibrated_std
            result.at[idx, "xP_calibrated"] = True
            result.at[idx, "xP_raw"] = ml_xp  # Keep original for comparison

        return result
```

**Helper Methods**:
```python
def _get_player_tier(self, price: float) -> str:
    """Determine player tier from price."""
    if price >= 8.0:
        return "premium"
    elif price >= 6.0:
        return "mid"
    else:
        return "budget"

def _get_fixture_category(self, fixture_difficulty: float) -> str:
    """Determine fixture category (2-tier only for simplicity)."""
    if fixture_difficulty >= 1.538:
        return "easy"
    else:
        return "hard"
```

---

### Phase 4: Configuration (Day 3)

**File**: `fpl_team_picker/config/settings.py`

**Add Config**:
```python
class XPCalibrationConfig(BaseModel):
    """Simple Probabilistic XP Calibration Configuration"""

    enabled: bool = Field(
        default=True,
        description="Enable probabilistic calibration of ML predictions"
    )

    # Simplified categorization (2-tier fixture only)
    premium_price_threshold: float = Field(default=8.0)
    mid_price_threshold: float = Field(default=6.0)
    easy_fixture_threshold: float = Field(default=1.538)

    # Distribution file
    distributions_path: str = Field(
        default="data/calibration/distributions.json"
    )

    # Regularization (for distribution fitting)
    regularization_lambda: float = Field(
        default=0.3,
        description="L2 regularization strength (higher = more shrinkage toward prior)"
    )

    # Minimum sample size
    minimum_sample_size: int = Field(
        default=30,
        description="Minimum samples required to use fitted distribution (else use prior)"
    )
```

**Add to FPLConfig**:
```python
class FPLConfig(BaseModel):
    ...
    xp_calibration: XPCalibrationConfig = Field(
        default_factory=XPCalibrationConfig
    )
```

---

### Phase 5: Integration (Day 3-4)

**File**: `fpl_team_picker/interfaces/gameweek_manager.py`

**Add Risk Profile Selector** (in optimization configuration section):
```python
risk_profile_selector = mo.ui.radio(
    options=["conservative", "balanced", "risk-taking"],
    value="balanced",
    label="Optimization Risk Profile:"
)
```

**Apply Calibration** (after ML prediction, before optimization):
```python
# After ML prediction
players_with_xp = ml_xp_service.calculate_expected_points(...)

# Apply calibration if enabled
if config.xp_calibration.enabled:
    from fpl_team_picker.domain.services import XPCalibrationService

    calibration_service = XPCalibrationService()
    players_with_xp = calibration_service.calibrate_predictions(
        players_with_xp,
        risk_profile=risk_profile_selector.value,
        fixture_difficulty_df=gameweek_data.get("fixture_difficulty")
    )

    if config.xp_model.debug:
        logger.info(
            f"✅ Applied probabilistic calibration "
            f"(risk_profile={risk_profile_selector.value})"
        )
```

**Optimization uses calibrated xP automatically** (no changes needed).

---

### Phase 6: Validation (Day 4-5)

**Temporal Validation**:
- Train on GW1-12, test on GW13-15
- Compare calibrated vs raw ML predictions
- Measure improvement for premium players in easy fixtures

**Success Criteria**:
- ✅ Calibrated xP better predicts actuals for premium + easy
- ✅ Optimization selects more premium players in easy fixtures
- ✅ No degradation for other player types
- ✅ Calibration adjustments reasonable (not extreme)

**If Validation Fails**:
- Adjust regularization strength
- Adjust prior means
- Consider using prior only (disable calibration)

---

## File Structure

```
fpl-team-picker/
├── scripts/
│   ├── analyze_calibration_data.py          # Phase 1: Data analysis
│   └── build_calibration_distributions.py  # Phase 2: Distribution fitting
├── fpl_team_picker/
│   ├── domain/
│   │   └── services/
│   │       └── xp_calibration_service.py    # Phase 3: Calibration service
│   └── config/
│       └── settings.py                      # Phase 4: Configuration
├── data/
│   └── calibration/
│       └── distributions.json              # Fitted distributions
└── interfaces/
    └── gameweek_manager.py                  # Phase 5: Integration
```

---

## Key Design Decisions (Simplicity First)

### 1. **2-Tier Fixture Difficulty** (Not 3-Tier)
- **Why**: More samples per combination, simpler
- **Trade-off**: Less granularity, but sufficient for calibration

### 2. **Simple Negative Binomial** (Not Zero-Inflated)
- **Why**: 2 parameters vs 3, less overfitting risk
- **Trade-off**: Doesn't explicitly model "2 points" spike, but approximates it

### 3. **Regularized MLE** (Not Full Bayesian)
- **Why**: Simpler, faster, sufficient for calibration
- **Trade-off**: Less sophisticated uncertainty, but adequate

### 4. **Precision-Weighted Combination** (Not Complex Bayesian Update)
- **Why**: Simple, interpretable, works well
- **Trade-off**: Less theoretically rigorous, but pragmatic

### 5. **Conservative Defaults** (90% ML, 10% Empirical Initially)
- **Why**: Trust ML model more with small samples
- **Trade-off**: Less correction, but safer

---

## Expected Impact

**With 15 gameweeks of data**:
- **Primary Goal**: Better identification of premium players in easy fixtures
- **Expected Improvement**: +2-5 points per gameweek
- **Risk Profiles**: Provide meaningful differentiation (conservative vs risk-taking)

**As More Data Accumulates** (GW20, GW30, etc.):
- Can increase empirical weight (move toward 70/30)
- Can use 3-tier fixture difficulty
- Can add position-specific distributions
- Can consider zero-inflation if needed

---

## Testing Strategy

### Unit Tests
- Calibration service logic
- Tier/fixture categorization
- Fallback handling
- Risk profile calculations

### Integration Tests
- Full pipeline (ML → Calibration → Optimization)
- Validation on GW13-15
- Compare calibrated vs raw ML

### Validation Tests
- Distribution fitting produces reasonable values
- Calibration improves predictions for premium + easy
- No extreme values or errors

---

## Success Metrics

**Calibration is Working If**:
1. ✅ Calibrated xP better predicts actuals for premium players in easy fixtures
2. ✅ Optimization selects more premium players in easy fixtures (vs raw ML)
3. ✅ Risk profiles produce different squad selections
4. ✅ No performance degradation (calibration adds <100ms)
5. ✅ No increase in errors (calibration failures <1%)

**Calibration is NOT Working If**:
1. ❌ Calibrated predictions worse than raw ML on validation set
2. ❌ No difference in optimization results
3. ❌ Extreme values (calibrated xP <0 or >20)
4. ❌ Frequent fallbacks to defaults

---

## Implementation Timeline

### Week 1: Core Implementation
- **Day 1**: Data analysis script, distribution fitting script
- **Day 2**: Calibration service implementation
- **Day 3**: Configuration, integration into gameweek_manager
- **Day 4**: Testing, validation on GW13-15
- **Day 5**: Bug fixes, tuning, documentation

### Week 2: Validation & Iteration
- Validate on GW13-15
- Tune regularization strength
- Adjust priors if needed
- A/B test optimization results

---

## Risk Mitigation

### High Priority Risks

1. **Insufficient Data**:
   - Use 2-tier fixture (6 combinations, not 9)
   - Require ≥30 samples (not 100)
   - Aggressive fallbacks (use prior if insufficient)

2. **Overfitting**:
   - Regularization (lambda=0.3, shrink toward prior)
   - Conservative weighting (trust ML more)
   - Validate on held-out gameweeks

3. **Edge Cases**:
   - Hierarchical fallbacks (exact → tier → default)
   - Use prior if combination missing
   - Extensive logging

### Medium Priority Risks

4. **Calibration Weight Tuning**:
   - Start conservative (90/10 ML/empirical)
   - Tune based on validation results
   - Make configurable

5. **Distribution Drift**:
   - Rebuild distributions weekly as new data arrives
   - Use rolling window (last 12 gameweeks)
   - Monitor for changes

---

## Configuration Example

```python
# Default configuration (conservative)
xp_calibration: XPCalibrationConfig(
    enabled=True,
    premium_price_threshold=8.0,
    mid_price_threshold=6.0,
    easy_fixture_threshold=1.538,
    regularization_lambda=0.3,
    minimum_sample_size=30,
    distributions_path="data/calibration/distributions.json"
)
```

---

## Conclusion

**This plan prioritizes simplicity**:
- ✅ Simple Negative Binomial (2 parameters)
- ✅ Regularized MLE (not full Bayesian)
- ✅ 2-tier fixture difficulty (not 3-tier)
- ✅ Precision-weighted combination (not complex Bayesian update)
- ✅ Conservative defaults (trust ML more with small samples)

**Core Goal**: Calibrate ML predictions to better identify premium players in easy fixtures, using simple probabilistic approach that works with limited data.

**Next Steps**: Implement Phase 1 (data analysis) to validate approach with actual data.
