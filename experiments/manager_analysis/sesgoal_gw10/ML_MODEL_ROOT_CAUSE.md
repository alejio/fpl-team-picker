# ML Model Root Cause Analysis: Bad GW11 Team Selection

## The Problem

**ML model xP predictions don't properly reflect fixture difficulty, causing optimizer to select players with hard fixtures (Haaland vs Liverpool 🔴, Roefs vs Arsenal 🔴) over players with easy fixtures (João Pedro vs Wolves 🟢).**

---

## Root Cause: **ML Model Output Missing Fixture Difficulty**

### The Issue

**The ML model:**
1. ✅ **Receives** `fixture_difficulty_df` as input (line 277, 513)
2. ✅ **Uses** fixture difficulty in feature engineering (`FPLFeatureEngineer._add_fixture_difficulty_features`, line 1185-1246)
3. ✅ **Learns** from fixture difficulty features during training
4. ❌ **Does NOT return** fixture difficulty in output DataFrame
5. ❌ **Gets overridden** to 1.0 (neutral) in `gameweek_manager.py` (line 578)

### Evidence

**In `ml_expected_points_service.py` (lines 514-603):**
```python
# Create result DataFrame
result = players_data.copy()

# Map predictions and uncertainty
result["ml_xP"] = result["player_id"].map(prediction_map)
result["xP_uncertainty"] = result["player_id"].map(uncertainty_map)
result["xP"] = result["ml_xP"]

# Calculate expected_minutes using rule-based logic
result = temp_xp_service._calculate_expected_minutes(result)

# Returns result - BUT NO FIXTURE_DIFFICULTY COLUMN!
return result
```

**The ML model returns:**
- ✅ `xP` (expected points)
- ✅ `xP_uncertainty` (uncertainty)
- ✅ `expected_minutes`
- ❌ **NO `fixture_difficulty` column**

**Then in `gameweek_manager.py` (lines 578-580):**
```python
players_with_xp["fixture_difficulty"] = 1.0  # Neutral
players_with_xp["fixture_difficulty_5gw"] = 1.0  # Neutral
players_with_xp["fixture_outlook"] = "Average"  # Placeholder
```

**This hardcodes fixture difficulty to 1.0 (neutral) for ALL players!**

---

## Why This Happens

### 1. **ML Model Uses Fixture Difficulty as Feature, Not Output**

**The ML model architecture:**
- Fixture difficulty is used as an **input feature** during training and prediction
- The model learns: "Players with easy fixtures (low difficulty) score more points"
- The model's xP predictions **implicitly include** fixture difficulty
- BUT: The model doesn't **explicitly return** fixture difficulty as a separate column

**Example:**
- João Pedro vs Wolves (🟢 easy): Model predicts 6.87 xP (includes fixture advantage)
- Haaland vs Liverpool (🔴 hard): Model predicts 6.87 xP (includes fixture penalty)
- **Problem:** Both have same xP, but we can't see WHY (fixture difficulty is hidden)

### 2. **Feature Engineering Adds Fixture Difficulty, But It's Not Preserved**

**In `ml_feature_engineering.py` (lines 1185-1246):**
```python
def _add_fixture_difficulty_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Merges fixture_difficulty_df into df
    # Adds: congestion_difficulty, form_adjusted_difficulty, clean_sheet_probability_enhanced
    # BUT: These are FEATURES (inputs), not OUTPUTS
    return df
```

**The feature engineer:**
- Adds fixture difficulty features to the input data
- Model uses these features to predict xP
- BUT: These features are NOT included in the output DataFrame

### 3. **gameweek_manager.py Overrides Fixture Difficulty**

**After ML prediction (line 578):**
```python
players_with_xp["fixture_difficulty"] = 1.0  # Neutral
players_with_xp["fixture_outlook"] = "Average"  # Placeholder
```

**This is a workaround/hack:**
- ML model doesn't return fixture difficulty
- So gameweek_manager.py hardcodes it to 1.0
- This loses all fixture difficulty information
- Optimizer can't distinguish easy vs hard fixtures

---

## Impact on GW11 Team Selection

### What Should Happen

**With proper fixture difficulty:**
- João Pedro: 6.87 xP, fixture_difficulty = 0.8 (🟢 easy) → **Should be preferred**
- Haaland: 6.87 xP, fixture_difficulty = 1.2 (🔴 hard) → **Should be avoided**
- Roefs: 3.33 xP, fixture_difficulty = 1.3 (🔴 very hard) → **Should be avoided**

**Optimizer would:**
- Prefer João Pedro over Haaland (same xP, better fixture)
- Avoid Roefs (hard fixture, low xP)
- Select players with easy fixtures when xP is similar

### What Actually Happens

**With hardcoded fixture_difficulty = 1.0:**
- João Pedro: 6.87 xP, fixture_difficulty = 1.0 (neutral) → **Looks same as Haaland**
- Haaland: 6.87 xP, fixture_difficulty = 1.0 (neutral) → **Looks same as João Pedro**
- Roefs: 3.33 xP, fixture_difficulty = 1.0 (neutral) → **Looks acceptable**

**Optimizer:**
- Sees same xP for João Pedro and Haaland → **Can't distinguish**
- Sees acceptable xP for Roefs → **Selects him**
- **Makes bad decisions because fixture difficulty is hidden**

---

## The Fix

### Solution 1: Extract Fixture Difficulty from Feature Engineering (Recommended)

**In `ml_expected_points_service.py`, after prediction (around line 556):**

```python
# After ML prediction
result["xP"] = result["ml_xP"]

# EXTRACT fixture difficulty from feature engineering
# The feature_engineer has fixture_difficulty_df merged into prediction_data_all
if hasattr(self.pipeline, 'named_steps') and 'feature_engineer' in self.pipeline.named_steps:
    feature_engineer = self.pipeline.named_steps['feature_engineer']

    # Get fixture difficulty from the feature-engineered data
    target_mask = prediction_data_all["gameweek"] == target_gameweek
    target_data = prediction_data_all[target_mask].copy()

    # Extract fixture difficulty columns if they exist
    if "fixture_difficulty" in target_data.columns:
        fixture_map = dict(zip(target_data["player_id"], target_data["fixture_difficulty"]))
        result["fixture_difficulty"] = result["player_id"].map(fixture_map)
    else:
        # Fallback: calculate from fixture_difficulty_df
        if fixture_difficulty_df is not None and not fixture_difficulty_df.empty:
            # Merge fixture difficulty
            result = result.merge(
                fixture_difficulty_df[["team_id", "gameweek", "fixture_difficulty"]],
                left_on=["team", "gameweek"],
                right_on=["team_id", "gameweek"],
                how="left"
            )
            result["fixture_difficulty"] = result["fixture_difficulty"].fillna(1.0)
    else:
        result["fixture_difficulty"] = 1.0  # Fallback
else:
    result["fixture_difficulty"] = 1.0  # Fallback

# Calculate fixture_outlook from fixture_difficulty
def get_fixture_outlook(diff):
    if pd.isna(diff) or diff == 1.0:
        return "Average"
    elif diff <= 0.8:
        return "🟢 Easy"
    elif diff >= 1.2:
        return "🔴 Hard"
    else:
        return "🟡 Average"

result["fixture_outlook"] = result["fixture_difficulty"].apply(get_fixture_outlook)
```

### Solution 2: Remove Hardcoding in gameweek_manager.py

**In `gameweek_manager.py` (line 578), REMOVE:**
```python
# REMOVE THIS:
players_with_xp["fixture_difficulty"] = 1.0  # Neutral
players_with_xp["fixture_difficulty_5gw"] = 1.0  # Neutral
players_with_xp["fixture_outlook"] = "Average"  # Placeholder

# REPLACE WITH:
# Use fixture difficulty from ML model if available
if "fixture_difficulty" not in players_with_xp.columns:
    # Calculate from fixture_difficulty_df if ML model didn't provide it
    if "fixture_difficulty" in gameweek_data:
        fixture_df = gameweek_data["fixture_difficulty"]
        players_with_xp = players_with_xp.merge(
            fixture_df[["team_id", "gameweek", "fixture_difficulty"]],
            left_on=["team", "gameweek"],
            right_on=["team_id", "gameweek"],
            how="left"
        )
        players_with_xp["fixture_difficulty"] = players_with_xp["fixture_difficulty"].fillna(1.0)
    else:
        players_with_xp["fixture_difficulty"] = 1.0  # Fallback only if no data

# Calculate fixture_outlook
def get_fixture_outlook(diff):
    if pd.isna(diff) or diff == 1.0:
        return "Average"
    elif diff <= 0.8:
        return "🟢 Easy"
    elif diff >= 1.2:
        return "🔴 Hard"
    else:
        return "🟡 Average"

players_with_xp["fixture_outlook"] = players_with_xp["fixture_difficulty"].apply(get_fixture_outlook)
```

### Solution 3: Add Fixture Difficulty to ML Model Output (Best Long-term)

**Modify `ml_expected_points_service.py` to explicitly return fixture difficulty:**

1. Extract fixture difficulty from `fixture_difficulty_df` after prediction
2. Merge it into result DataFrame
3. Calculate fixture_outlook from fixture_difficulty
4. Return both in result

**This ensures fixture difficulty is always available in the output.**

---

## Why ML Model xP Values Are Similar Despite Different Fixtures

### The Model IS Learning from Fixture Difficulty

**Evidence:**
- Model receives fixture difficulty features during training
- Model learns: "Easy fixtures → more points, Hard fixtures → fewer points"
- Model's xP predictions DO account for fixture difficulty

**BUT:**
- The effect might be **too small** in the model
- Or fixture difficulty features have **low feature importance**
- Or the model is **overfitting** to other features (form, price, etc.)

### Possible Issues

1. **Feature Importance:**
   - Fixture difficulty might have low importance in the trained model
   - Other features (form, price, ownership) might dominate
   - Model might not weight fixture difficulty heavily enough

2. **Model Architecture:**
   - Tree-based models (Random Forest) might not capture fixture difficulty well
   - Linear relationships might be better for fixture difficulty
   - Ensemble might be diluting fixture difficulty signal

3. **Training Data:**
   - If training data has imbalanced fixture difficulties
   - Or fixture difficulty doesn't correlate strongly with points in training
   - Model might learn to ignore it

---

## Recommended Fix Priority

### Immediate (This Week)
1. **Remove hardcoding** in `gameweek_manager.py` (line 578)
2. **Extract fixture difficulty** from `fixture_difficulty_df` and add to output
3. **Calculate fixture_outlook** from actual fixture difficulty

### Short-term (This Month)
1. **Verify ML model** is using fixture difficulty features correctly
2. **Check feature importance** - is fixture difficulty weighted properly?
3. **Add fixture difficulty penalty** in optimizer for very hard fixtures (difficulty ≥ 1.2)

### Long-term (Next Season)
1. **Re-train ML model** with explicit fixture difficulty weighting
2. **Add fixture difficulty as explicit output** from ML model
3. **Optimizer constraint** for minimum fixture quality

---

## Conclusion

**Root Cause: ML Model Output Missing Fixture Difficulty**

**The ML model:**
- ✅ Uses fixture difficulty as input feature
- ✅ Learns from fixture difficulty
- ❌ Doesn't return fixture difficulty in output
- ❌ Gets overridden to 1.0 in gameweek_manager.py

**Impact:**
- Optimizer can't distinguish easy vs hard fixtures
- Makes bad decisions (selects Haaland vs Liverpool, Roefs vs Arsenal)
- All players show "Average" fixture outlook (hardcoded)

**Fix:**
- Extract fixture difficulty from feature engineering or fixture_difficulty_df
- Add to ML model output DataFrame
- Remove hardcoding in gameweek_manager.py
- Calculate fixture_outlook from actual difficulty

**This is a data pipeline issue, not a model training issue. The model is learning correctly, but the information is being lost in the output.**

---

**Analysis Date:** GW11
**Files Analyzed:**
- `ml_expected_points_service.py` (lines 265-603)
- `ml_feature_engineering.py` (lines 1185-1246)
- `gameweek_manager.py` (lines 578-580)
