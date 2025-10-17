# ML Expected Points (xP) Model Improvement Plan

**Date:** 2025-10-17
**Current Status:** Temporal CV implemented, baseline established
**Current Best MAE:** Ridge 1.26 (temporal CV), RF 0.61 (player-based CV - was incorrectly measured)

---

## Executive Summary

We've discovered that the original RF 0.61 MAE was measured with **player-based CV** (tests generalization to NEW players, mixes gameweeks), not **temporal CV** (tests prediction of NEXT future gameweek). After implementing proper temporal CV, we need to re-evaluate all models and implement improvements.

**Key Finding:** Player-based CV gives overly optimistic results. Ridge goes from 0.84 MAE (player-based) → **1.26 MAE (temporal)** - a 50% degradation when properly validated.

---

## Current Data Situation

### Available Data
- **Gameweeks:** GW6-7 only (2 gameweeks)
- **Samples per GW:** ~742 players
- **Temporal CV:** Only 1 fold (train GW6 → test GW7)
- **Target:** `total_points` - per-gameweek FPL points (0-20 range, confirmed NOT cumulative)

### Data Quality Issues Identified

**Problem 1: Bench Warmers Pollute Training Data**
- ~30-40% of players have `cumulative_minutes = 0` (never played)
- `rolling_5gw_points = 0`, `rolling_5gw_minutes = 0`
- Model can't learn from players with no history
- They inflate error metrics (predicting 2 pts when they score 0 = 2 MAE)

**Problem 2: Small Sample Size**
- Only 742 samples per gameweek
- After filtering bench warmers, ~400-500 effective samples
- Makes complex models (deep trees, many features) prone to overfitting

**Problem 3: Limited Temporal Validation**
- Only 1 temporal fold (GW6→GW7) means no variance estimates
- Can't properly tune hyperparameters with temporal CV
- Will improve as more gameweeks become available

---

## Training Data Structure (Verified)

### Sample Row (Player 5, GW6):
```
player_id: 5
gameweek: 6
total_points: 2           ← TARGET (what we predict)
price: 5.6
position_encoded: 0       (GKP=0, DEF=1, MID=2, FWD=3)
rolling_5gw_points: 5.2
rolling_5gw_minutes: 90
fixture_difficulty: 0.889
cumulative_minutes: 450   (5 games × 90 mins)
cumulative_points: 26     (season total up to GW5)
... (64 features total)
```

### Feature Categories (64 total):
1. **Static (3):** price, position_encoded, games_played
2. **Cumulative Season Stats (11):** minutes, goals, assists, points, bonus, clean_sheets, xG, xA, BPS, cards
3. **Cumulative Per-90 Rates (7):** goals/90, assists/90, points/90, xG/90, xA/90, BPS/90, CS rate
4. **Rolling 5GW Form (13):** points, minutes, goals, assists, xG, xA, BPS, bonus, clean_sheets, ICT components
5. **Rolling 5GW Per-90 (3):** goals/90, assists/90, points/90
6. **Defensive Metrics (4):** goals_conceded, saves, xGI, xGC
7. **Consistency & Volatility (4):** points_std, minutes_std, minutes_played_rate, form_trend
8. **Team Context (13):** team_encoded, rolling 5GW team stats, cumulative team stats, goal differentials
9. **Fixture Features (6):** is_home, opponent_strength, fixture_difficulty, opponent defensive metrics

### Training Process:
```python
# Cross-Validation (evaluation only, doesn't affect final model)
- Temporal CV: Train on GW6 → Test on GW7 (1 fold)
- Player-Based CV: 5 folds, mixes gameweeks (overly optimistic)

# Final Model (what gets deployed)
- Trains on ALL available data (GW6 + GW7 combined)
- CV strategy only affects evaluation metrics, not the final trained model
```

---

## Current Model Performance

### Ridge Regression (Baseline)
| CV Strategy | MAE | RMSE | R² | Fold Scores |
|------------|-----|------|-----|-------------|
| **Temporal** (correct) | **1.26 ± 0.00** | 2.59 | -6.57 | [1.26] (only 1 fold) |
| Player-Based (optimistic) | 0.84 ± 0.08 | 1.69 | 0.11 | [0.83, 0.78, 0.80, 0.81, 1.00] |

**Interpretation:**
- Temporal CV is 50% harder than player-based (1.26 vs 0.84)
- R² = -6.57 means model performs MUCH worse than predicting the mean
- This is expected with only 1 temporal fold and noisy data (bench warmers)

### Random Forest
| CV Strategy | MAE | Status |
|------------|-----|--------|
| **Temporal** | **Not yet tested** | Need to re-run with fixed CV |
| Player-Based | 0.61 ± 0.06 | Was measured incorrectly (CV bug) |

**Note:** Need to re-test RF with temporal CV to get true performance.

### Gradient Boosting
| CV Strategy | MAE | Status |
|------------|-----|--------|
| **Temporal** | **Not yet tested** | Need to test |
| Player-Based | Unknown | Need to test |

---

## Improvement Plan - Priority Order

### ✅ **COMPLETED: Step 1 - Fix CV Strategy**

**What we did:**
- Implemented temporal walk-forward validation
- Added CV strategy selector (temporal vs player-based)
- Fixed bug where dropdown value wasn't being read correctly
- Fixed index alignment issue (pandas .index vs positional indices)

**Code changes:** [ml_xp_notebook.py](fpl_team_picker/interfaces/ml_xp_notebook.py:760-810)

**Result:** Temporal CV now works correctly, reveals true generalization performance

---

### ❌ **COMPLETED: Step 2 - Filter Low-Information Cases** (REJECTED)

**Hypothesis:**
- 30-40% of training data is bench warmers (0 minutes, no history)
- They add noise and inflate error metrics
- Model wastes capacity trying to predict zeros

**Experiment:**
```python
# Filter to players with meaningful playing history
cv_data = cv_data[cv_data['cumulative_minutes'] >= 45]  # At least half a game
```

**Actual Results:** ⚠️ **FILTERING DEGRADES ACCURACY**

| Metric | Unfiltered (1,484 samples) | Filtered (226 samples) | Change |
|--------|---------------------------|------------------------|---------|
| MAE | **0.966 ± 0.065** | 1.173 ± 0.183 | **-21.4% worse** |
| Samples | 1,484 | 226 (84.8% removed) | - |

**Why filtering HURT accuracy:**

1. **Sample size too small:** Removing 84.8% of data (1,484 → 226) leaves insufficient samples for generalization
2. **Selection bias:** Filtered dataset only has regular starters, missing the distribution of bench/rotation players
3. **Bench warmers are EASY to predict:** Model can learn "0 minutes = 0 points" perfectly (MAE=0 for bench)
4. **Errors come from STARTERS, not bench warmers:** The hard cases are predicting 2 vs 8 for starting players
5. **Per-90 normalization already handles minutes:** Features like `goals_per_90` already account for playing time

**Conclusion:**
- **DO NOT filter bench warmers from training data**
- Including zeros actually helps model learn the full distribution
- Focus instead on improving predictions for STARTING players (the hard cases)

**Key Learning:**
- Intuition was wrong: "noisy bench warmers" are actually the EASIEST cases
- Real challenge is predicting point variance among regular starters
- Removing easy cases makes validation metrics look worse (model only tested on hard cases)

**Files modified:**
- [test_bench_warmer_filtering.py](test_bench_warmer_filtering.py) - Standalone experiment script

---

### ✅ **COMPLETED: Step 3 - Test More Algorithms**

**Tested 6 algorithms** with both Player-Based and Temporal CV:
1. Ridge Regression (baseline)
2. ElasticNet
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. LightGBM

**Results (Temporal CV - realistic):**

| Rank | Model | Temporal MAE | vs Ridge | Optimism Gap* |
|------|-------|--------------|----------|---------------|
| 🥇 1st | **Gradient Boosting** | **0.958** | **+6.2%** | -2.9% |
| 🥈 2nd | **LightGBM** | **0.963** | **+5.7%** | -1.6% |
| 🥉 3rd | Ridge (baseline) | 1.021 | — | +5.7% |
| 4th | XGBoost | 1.137 | -11.4% | +15.8% |
| 5th | ElasticNet | 1.162 | -13.8% | +16.8% |
| 6th | Random Forest | 1.198 | -17.3% | **+32.5%** |

*Optimism Gap = Temporal MAE - Player-Based MAE (negative is good, model generalizes better than expected)

**Key Findings:**

1. **🏆 Gradient Boosting is the winner** - 6.2% better than Ridge (0.958 vs 1.021 MAE)
2. **LightGBM is a close second** - 5.7% better than Ridge (0.963 MAE)
3. **Random Forest severely overfits** - 32.5% optimism gap, worst temporal performance
4. **XGBoost disappoints** - 11.4% worse than Ridge, high optimism gap (15.8%)
5. **ElasticNet adds no value** - L1 regularization doesn't help, worse than Ridge
6. **Tree models need careful tuning** - Vanilla hyperparameters cause overfitting

**Why Gradient Boosting wins:**
- Sequential learning reduces overfitting vs RF (which trains trees independently)
- Better regularization than XGBoost with our default hyperparameters
- Handles non-linear patterns (fixture difficulty, form) better than Ridge
- Smaller optimism gap (-2.9%) means it generalizes well

**Production Recommendation:**
- **Use Gradient Boosting as primary model** (0.958 MAE)
- Consider ensemble with LightGBM for robustness (both ~0.96 MAE)
- Avoid Random Forest (overfits badly on small temporal data)

**Files:**
- [algorithm_comparison_experiment.py](experiments/algorithm_comparison_experiment.py) - Standalone experiment
- Added `lightgbm==4.6.0` to dependencies

**⚠️ CRITICAL ISSUE DISCOVERED:**
The initial experiment only loaded GW6-7 data, which meant:
- GW6 rows had NO historical features (shift(1) on first row = 0)
- GW7 rows had features from only 1 gameweek
- **Rolling 5GW windows were incomplete!**

The `FPLFeatureEngineer` uses `groupby().shift(1).cumsum()` on the input DataFrame only - it doesn't query the database for historical data. This means we MUST load ALL gameweeks (GW1-7) to get proper features.

**Action:** Re-running experiment with ALL gameweeks (see Step 3b below)

---

### ⏭️ **FUTURE: Step 4 - Add Missing Features**

**Goal:** Enrich feature set with FPL-specific data

**No-brainer FPL features to add:**

1. **Market Sentiment:**
   - `selected_by_percent` - Ownership % (proxy for player quality)
   - `transfers_in_event` - Transfers in this gameweek
   - `transfers_out_event` - Transfers out this gameweek
   - Rationale: Wisdom of crowds - high ownership = better players

2. **Bonus Predictors:**
   - `bonus_per_90` - Historical bonus points rate
   - `bps_rank_in_team` - Is this player the bonus magnet?
   - Rationale: Some players consistently get bonus (defenders who score)

3. **Recent Form (shorter window):**
   - `minutes_last_3gw` - More recent than 5GW rolling average
   - `points_last_3gw` - Captures very recent form changes
   - Rationale: Form can change quickly (injury, rotation, tactics)

4. **Position-Specific:**
   - `is_penalty_taker` - Binary flag
   - `corners_and_indirect_freekicks_order` - Set piece priority
   - `direct_freekicks_order` - Direct FK priority
   - Rationale: Set pieces are high-leverage scoring opportunities

5. **Team Fixture Context:**
   - `team_fdr_next_5` - Fixture difficulty rating
   - `team_home_away_split` - Some teams much stronger at home
   - Rationale: Some teams perform drastically different home vs away

**Implementation:**
- Add to [ml_feature_engineering.py](fpl_team_picker/domain/services/ml_feature_engineering.py:14-799)
- Ensure all features use `.shift(1)` to be leak-free
- Update feature count (64 → ~75-80 features)

**Expected Impact:**
- MAE improvement: 5-10% (e.g., 1.0 → 0.90-0.95)
- Better capture of player quality signals
- More informative for casual vs premium players

**Data availability check:**
```python
# Check if features exist in current_players
players = client.get_current_players()
print(players.columns)  # Verify column names

# Features we expect:
# - selected_by_percent ✓
# - transfers_in_event ✓
# - transfers_out_event ✓
# - bonus ✓ (already have, need per-90)
# - corners_and_indirect_freekicks_order ✓
# - direct_freekicks_order ✓
```

---

### ⏭️ **FUTURE: Step 5 - Hyperparameter Tuning**

**Goal:** Optimize algorithm hyperparameters

**Current issue:**
- RF uses defaults: `n_estimators=100, max_depth=10, min_samples_split=10`
- These were never tuned for FPL xP prediction

**Grid Search Strategy:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [5, 10, 15],
    'max_features': ['sqrt', 'log2', 0.5],
}

# Use temporal CV for tuning
grid_search = GridSearchCV(
    RandomForestRegressor(),
    param_grid,
    cv=temporal_splits,  # Custom splits from our temporal CV
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
```

**Challenge:**
- With only 1 temporal fold, GridSearchCV won't work well
- Need 3-4 gameweeks minimum for proper tuning
- Alternative: Tune on player-based CV, validate on temporal

**Expected Impact:**
- MAE improvement: 3-8% (e.g., 0.90 → 0.83-0.87)
- Better regularization (less overfitting)
- Optimal tree depth/complexity for our feature set

---

### ⏭️ **FUTURE: Step 6 - Ensemble Methods**

**Goal:** Combine multiple models for robustness

**Stacking Strategy:**
```python
from sklearn.ensemble import StackingRegressor

# Level 0: Diverse base models
estimators = [
    ('rf', RandomForestRegressor(**best_rf_params)),
    ('xgb', XGBRegressor(**best_xgb_params)),
    ('ridge', Ridge(alpha=1.0))
]

# Level 1: Meta-learner
stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=0.1),
    cv=temporal_splits
)
```

**Why this helps:**
- Different models make different errors
- RF good at interactions, Ridge good at linear trends
- Meta-learner learns optimal weighting

**Expected Impact:**
- MAE improvement: 2-5% (e.g., 0.85 → 0.81-0.83)
- More stable predictions (less variance)
- Better performance on edge cases

---

### ⏭️ **FUTURE: Step 7 - Target Engineering**

**Goal:** Transform target to improve model performance

**Options:**

**Option A: Sqrt Transform**
```python
# Train on transformed target
y_train = np.sqrt(cv_data['total_points'])
model.fit(X_train, y_train)

# Predict and inverse transform
predictions = model.predict(X_test)
final_predictions = predictions ** 2
```

**Rationale:**
- Reduces impact of outliers (hauls: 15-20 pts)
- Stabilizes variance (0-20 range → 0-4.5 range)
- May help model focus on typical performance

**Option B: Winsorization**
```python
from scipy.stats import mstats
# Cap extreme values at 95th percentile
y_train = mstats.winsorize(cv_data['total_points'], limits=[0, 0.05])
```

**Option C: Multi-Task Learning**
```python
# Predict multiple related targets
Target 1: total_points (main)
Target 2: minutes (auxiliary - easier to predict)
Target 3: goals + assists (auxiliary - interpretable)

# Share hidden layers between tasks
# Forces model to learn useful representations
```

**Expected Impact:**
- MAE improvement: 2-5% (e.g., 0.85 → 0.81-0.83)
- Better calibration (predictions match actual distribution)
- Potentially worse on hauls, better on typical scores

---

## Success Metrics

### Short-Term (Next 2-3 weeks)
- **Temporal CV MAE < 1.0** (currently 1.26 for Ridge)
- **Beat rule-based model** (need to measure baseline)
- **R² > 0** (currently -6.57, model worse than mean)

### Medium-Term (Next 1-2 months with more data)
- **Temporal CV MAE < 0.80** (competitive with player-based Ridge 0.84)
- **Multiple temporal folds** (4-5 gameweeks available)
- **Production deployment** (replace rule-based or ensemble)

### Long-Term (Full season)
- **Temporal CV MAE < 0.65** (matches current RF player-based performance)
- **Live validation** (track predictions vs actual each gameweek)
- **Transfer recommendation accuracy** (did recommended transfers improve score?)

---

## Key Learnings & Principles

### 1. **Validation Strategy Matters Critically**
- Player-based CV: Tests "can we predict new players?" (optimistic)
- Temporal CV: Tests "can we predict future?" (realistic)
- **Always use temporal CV for FPL** (real use case is predicting next GW)

### 2. **Data Quality > Model Complexity**
- Filtering bench warmers likely improves MAE more than fancy algorithms
- Clean data with simple model > dirty data with complex model

### 3. **Small Sample Size Limitations**
- 742 samples per GW is small for ML
- Can't train very deep models (overfitting)
- Feature engineering more important than algorithm choice

### 4. **Cross-Validation Is For Evaluation Only**
- CV doesn't change the final deployed model
- Final model trains on ALL data (GW6 + GW7)
- CV tells us how well it will generalize

### 5. **Temporal Validation Requires Multiple Gameweeks**
- 1 fold gives no variance estimate
- Need 4-5 GWs for reliable temporal CV
- Be patient - performance will improve with more data

---

## Questions to Investigate

### 1. **Why is R² so negative (-6.57)?**
Possible reasons:
- Model predicting ~5 pts on average, actual mean is ~3 pts
- Large errors on hauls (predicted 5, actual 15 = 10 error²)
- Only 1 fold means outliers dominate
- Bench warmers inflating error

**Test:** Filter to `cumulative_minutes > 90` and re-measure

### 2. **Should we predict points or points per 90?**
Current: Predicting raw `total_points` (0-20)
Alternative: Predict `points_per_90` then multiply by `expected_minutes`

**Pros of per-90:**
- Handles rotation better (separates ability from playing time)
- More stable target (less variance)

**Cons:**
- Two-stage prediction (more error propagation)
- Division by small minutes creates outliers

**Test:** Try both and compare MAE

### 3. **Do we need position-specific models?**
Current: Single model for all positions
Alternative: Separate models for GKP/DEF/MID/FWD

**Rationale:**
- Different scoring patterns (GKP clean sheets vs FWD goals)
- Different feature importance

**Test:** Train 4 separate models, compare aggregated MAE

### 4. **How much do team/fixture features help?**
Current: 19 features for team + fixture context
Test: Ablation study
- Remove team features → measure MAE increase
- Remove fixture features → measure MAE increase

**Expected:** Team features matter more than fixture (player quality > matchup)

---

## Code Structure & Files

### Main Files
- **[ml_xp_notebook.py](fpl_team_picker/interfaces/ml_xp_notebook.py)** - Interactive notebook for development
- **[ml_feature_engineering.py](fpl_team_picker/domain/services/ml_feature_engineering.py)** - FPLFeatureEngineer transformer
- **[ml_expected_points_service.py](fpl_team_picker/domain/services/ml_expected_points_service.py)** - Production service
- **[ml_pipeline_factory.py](fpl_team_picker/domain/services/ml_pipeline_factory.py)** - Pipeline construction

### Shared Logic
Both `gameweek_manager.py` and `ml_xp_notebook.py` use:
- Same `FPLFeatureEngineer` (line 357-361 in notebook)
- Same feature list (64 features)
- Same data loading (`client.get_gameweek_performance()`)

**This ensures notebook experiments translate directly to production!**

---

## Next Session Checklist

When you return to this work:

1. **Check data availability:**
   ```bash
   # How many gameweeks do we have now?
   cd /Users/alex/dev/FPL/fpl-dataset-builder
   uv run python -c "from client import FPLDataClient; c = FPLDataClient(); print([gw for gw in range(1, 20) if not c.get_gameweek_performance(gw).empty])"
   ```

2. **Run notebook:**
   ```bash
   cd /Users/alex/dev/FPL/fpl-team-picker
   uv run marimo run fpl_team_picker/interfaces/ml_xp_notebook.py --no-token
   ```

3. **Prioritize based on data:**
   - If still only GW6-7: Focus on **Step 2 (filtering)** and **Step 3 (algorithms)**
   - If GW6-10+ available: Can do proper temporal CV with multiple folds, enable hyperparameter tuning

4. **Remember:**
   - Temporal CV is the truth (player-based is too optimistic)
   - Filtering bench warmers is the highest-leverage improvement
   - Don't over-engineer until we have more gameweeks

---

## References

- **FPL Scoring Rules:** [fpl_rules.md](fpl_rules.md)
- **Project README:** [CLAUDE.md](CLAUDE.md)
- **ML Model Docs:** [ML_XP_MODEL.md](ML_XP_MODEL.md)

---

**Last Updated:** 2025-10-17
**Next Review:** After Step 2 (filtering) is implemented
