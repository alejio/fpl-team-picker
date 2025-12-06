# Recipe: Adding New Features to FPLFeatureEngineer

This recipe documents all the places that need to be updated when adding new features to the ML feature engineering pipeline.

## Overview

When adding new features to `FPLFeatureEngineer`, you must update multiple files to keep the codebase consistent. This checklist ensures nothing is missed.

**Current Feature Count:** 156 features

---

## Step-by-Step Checklist

### 1. Feature Engineering Implementation

#### 1.1 Add Feature Creation Method
- [ ] Add new method `_add_<feature_category>_features()` in `fpl_team_picker/domain/services/ml_feature_engineering.py`
- [ ] Method should accept `df: pd.DataFrame` and return `pd.DataFrame`
- [ ] Ensure features are leak-free (only use past data)
- [ ] Handle missing data gracefully with domain-aware defaults

#### 1.2 Integrate into Transform Pipeline
- [ ] Call the new method in `transform()` method
- [ ] Place in appropriate phase section (Phase 1-6 or new phase)
- [ ] Ensure proper ordering (dependencies resolved)

#### 1.3 Add to Feature List
- [ ] Add feature names to `_get_feature_columns()` method
- [ ] Add in appropriate category section with comment
- [ ] Count features in each category to verify totals

---

### 2. Documentation Updates

#### 2.1 Class Docstring
- [ ] Update total feature count in main docstring
- [ ] Update feature breakdown summary (e.g., "64 base + 12 enhanced + ...")
- [ ] Add new feature category description with count
- [ ] List all new feature names in the category section

**File:** `fpl_team_picker/domain/services/ml_feature_engineering.py` (lines ~67-147)

#### 2.2 Method Docstring
- [ ] Update `_get_feature_columns()` docstring with new total count
- [ ] Update `transform()` method docstring with new feature count

**File:** `fpl_team_picker/domain/services/ml_feature_engineering.py`

---

### 3. Test Updates

#### 3.1 Feature Count Tests
- [ ] Update `test_total_feature_count_156()` in `test_ml_feature_engineering_phase_1_4.py`
  - Change expected count from old → new
  - Update test name if needed
  - Update class docstring
- [ ] Update `test_total_feature_count_includes_betting_odds()` in `test_ml_feature_engineering_betting_odds.py`
- [ ] Update `test_feature_engineer_produces_117_features_with_betting_odds()` in `test_ml_expected_points_service_integration.py`
  - Update test name
  - Update docstring
  - Update assertion

**Files:**
- `tests/domain/services/test_ml_feature_engineering_phase_1_4.py`
- `tests/domain/services/test_ml_feature_engineering_betting_odds.py`
- `tests/domain/services/test_ml_expected_points_service_integration.py`

#### 3.2 Feature-Specific Tests (if applicable)
- [ ] Add tests for new feature category
- [ ] Test feature creation logic
- [ ] Test default values when data missing
- [ ] Test feature merging/joining

---

### 4. Interface/UI Updates

#### 4.1 Gameweek Manager
- [ ] Update feature count in model info display
- [ ] Update feature description if needed

**File:** `fpl_team_picker/interfaces/gameweek_manager.py` (line ~666)

**Search for:** `"features": "156 features"`

---

### 5. Training Pipeline Updates

#### 5.1 Training Utils
- [ ] Update log message with new feature count

**File:** `scripts/ml_training_utils.py` (line ~342)

**Search for:** `"Engineering features (FPLFeatureEngineer with"`

#### 5.2 Custom Pipeline Optimizer
- [ ] Update feature count in docstring/comments
- [ ] Update help text for feature selection
- [ ] Update log messages

**File:** `scripts/custom_pipeline_optimizer.py`

**Search for:**
- `"122 FPL features"` or current count
- `"keep all 156 features"` or current count
- `f"Features: {len(selected_features)}/156"` or current count

#### 5.3 Feature Importance Analysis
- [ ] Update log message with new feature count

**File:** `scripts/feature_importance_analysis.py` (line ~171)

**Search for:** `"Engineering features (production FPLFeatureEngineer with"`

#### 5.4 Training Pipeline Docs
- [ ] Update `get_feature_groups()` docstring if needed

**File:** `fpl_team_picker/domain/services/ml_training/pipelines.py` (line ~217)

**Search for:** `"Categorizes all 156 FPL features"`

---

### 6. Documentation Files

#### 6.1 Main Documentation
- [ ] Update `CLAUDE.md` if it mentions feature count

**File:** `CLAUDE.md`

**Search for:** `"ML Feature Engineering (156 features)"`

#### 6.2 LaTeX Documentation (if applicable)
- [ ] Update `docs/ALGORITHM_ANALYSIS_REPORT.tex` if it mentions feature count
- [ ] Update `docs/EMBEDDING_AI_ANALYSIS.tex` if it mentions feature count

**Note:** These are historical documents - update only if actively maintained

---

### 7. Model Metadata (Historical - Optional)

#### 7.1 Saved Model JSON Files
- [ ] **Note:** Model JSON files in `models/` contain historical `selected_features_count`
- [ ] **Do NOT update** - these are historical records
- [ ] New models will automatically have correct count

---

### 8. Verification Steps

#### 8.1 Run Tests
```bash
# Run feature engineering tests
uv run pytest tests/domain/services/test_ml_feature_engineering_phase_1_4.py::TestTotalFeatureCount -xvs
uv run pytest tests/domain/services/test_ml_feature_engineering_betting_odds.py::TestBettingOddsFeatures::test_total_feature_count_includes_betting_odds -xvs
uv run pytest tests/domain/services/test_ml_expected_points_service_integration.py::TestFPLFeatureEngineer118Features::test_feature_engineer_produces_117_features_with_betting_odds -xvs
```

#### 8.2 Verify Feature Count
```python
from fpl_team_picker.domain.services.ml_feature_engineering import FPLFeatureEngineer

engineer = FPLFeatureEngineer()
features = engineer._get_feature_columns()
print(f"Total features: {len(features)}")
assert len(features) == 156  # Update to new count
```

#### 8.3 Check for Linter Errors
```bash
uv run ruff check fpl_team_picker/domain/services/ml_feature_engineering.py
```

#### 8.4 Search for Old Count
```bash
# Search for old feature count to catch any missed references
grep -r "147.*feature\|feature.*147" --include="*.py" --include="*.md" .
grep -r "\b147\b" tests/domain/services/
```

---

## Quick Reference: Files to Update

### Core Implementation
1. `fpl_team_picker/domain/services/ml_feature_engineering.py`
   - Add feature creation method
   - Add to transform pipeline
   - Add to `_get_feature_columns()`
   - Update docstrings

### Tests
2. `tests/domain/services/test_ml_feature_engineering_phase_1_4.py`
3. `tests/domain/services/test_ml_feature_engineering_betting_odds.py`
4. `tests/domain/services/test_ml_expected_points_service_integration.py`

### Interfaces
5. `fpl_team_picker/interfaces/gameweek_manager.py`

### Scripts
6. `scripts/ml_training_utils.py`
7. `scripts/custom_pipeline_optimizer.py`
8. `scripts/feature_importance_analysis.py`

### Training Pipeline
9. `fpl_team_picker/domain/services/ml_training/pipelines.py`

### Documentation
10. `CLAUDE.md`
11. `docs/ALGORITHM_ANALYSIS_REPORT.tex` (if maintained)
12. `docs/EMBEDDING_AI_ANALYSIS.tex` (if maintained)

---

## Example: Adding 3 New Features

If adding 3 new features (bringing total from 156 → 159):

1. **Implementation:**
   ```python
   def _add_my_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
       df["new_feature_1"] = ...
       df["new_feature_2"] = ...
       df["new_feature_3"] = ...
       return df
   ```

2. **Update docstring:**
   ```python
   Generates 158 features (64 base + 12 enhanced + ... + 8 position-specific + 3 new):

   New category (3) - Phase 7:
   - new_feature_1: Description
   - new_feature_2: Description
   - new_feature_3: Description
   ```

3. **Update all test assertions:**
   ```python
   assert result.shape[1] == 158, f"Expected 158 features, got {result.shape[1]}"
   ```

4. **Update all log messages:**
   ```python
   "Engineering features (FPLFeatureEngineer with 158 features)..."
   ```

5. **Search and replace:**
   ```bash
   # Find all references
   grep -r "155" --include="*.py" --include="*.md" .
   # Update systematically
   ```

---

## Tips

- **Use grep to find all references:** `grep -r "155" --include="*.py" --include="*.md" .`
- **Update tests first** - they'll catch if you miss something
- **Update docstrings early** - helps document your thinking
- **Run tests frequently** - catch issues before they compound
- **Check linter** - catches syntax errors in docstrings

---

## Common Mistakes to Avoid

1. ❌ Forgetting to update test assertions
2. ❌ Updating feature count in code but not in docstrings
3. ❌ Missing log messages in training scripts
4. ❌ Forgetting gameweek manager UI display
5. ❌ Not updating help text in CLI tools
6. ❌ Updating historical model JSON files (don't do this!)

---

## Version History

- **2025-12-06**: Created recipe after adding Phase 6 position-specific features (147 → 155)
- **2025-12-06**: Updated to 156 features after adding opponent_rolling_5gw_xg
- Current feature count: **156 features**
