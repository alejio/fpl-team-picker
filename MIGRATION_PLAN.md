# Business Logic Consolidation - Migration Plan

**Goal**: Move all business logic from `core/` into `domain/services/` to fix dependency inversion and eliminate code duplication.

**Total Effort**: 30-40 hours (6-8 hours per module)
**Start Date**: TBD
**Target Completion**: TBD

---

## Current State Analysis

### Dependency Violations Found

**Domain services importing from core/ (8 imports):**
```
chip_assessment_service.py:56    → core.chip_assessment.ChipAssessmentEngine
data_orchestration_service.py:40 → core.data_loader (fetch_fpl_data, etc.)
data_orchestration_service.py:109 → core.data_loader.get_current_gameweek_info
expected_points_service.py:106   → core.ml_xp_model.MLXPModel
expected_points_service.py:107   → core.xp_model.XPModel
expected_points_service.py:177   → core.xp_model.XPModel
expected_points_service.py:204   → core.ml_xp_model.merge_1gw_5gw_results
expected_points_service.py:207   → core.xp_model.merge_1gw_5gw_results
```

**Domain services importing from optimization/ (5 imports):**
```
squad_management_service.py:189         → optimization.optimizer
transfer_optimization_service.py:47     → optimization.optimizer.optimize_team_with_transfers
transfer_optimization_service.py:87     → optimization.optimizer.get_best_starting_11
transfer_optimization_service.py:188    → optimization.optimizer
transfer_optimization_service.py:212    → optimization.optimizer
```

**External dependencies on core/:**
- `visualization/charts.py` → `core.team_strength.DynamicTeamStrength` (2 usages)
- `core/xp_model.py` → `core.team_strength.DynamicTeamStrength` (internal)

---

## Module Analysis

### Module 1: chip_assessment.py (601 LOC) ⭐ START HERE
**Type**: Single class with helper methods
**Complexity**: Low - No external dependencies
**Current State**: Thin wrapper exists in domain
**Migration**: Straightforward copy-paste + refactor

**Structure:**
```python
class ChipRecommendation(dataclass):  # 19-26
class ChipAssessmentEngine:          # 29-601
    __init__(config)
    assess_all_chips()
    assess_wildcard()
    assess_free_hit()
    assess_bench_boost()
    assess_triple_captain()
    # + 20 private helper methods
```

**Dependencies**: None (pure business logic)
**Used by**: `chip_assessment_service.py:56`
**Estimated time**: 4-6 hours

---

### Module 2: data_loader.py (564 LOC) ⭐ SECOND
**Type**: Module with functions (no classes)
**Complexity**: Low-Medium - FPL API calls
**Current State**: Partial implementation in domain
**Migration**: Merge into existing service

**Structure:**
```python
def get_current_gameweek_info() -> Dict:           # 17-132
def fetch_fpl_data(target_gameweek, form_window):  # 135-322
def fetch_manager_team(previous_gameweek):         # 325-411
def _get_chip_status():                            # 414-451
def process_current_squad():                       # 454-521
def load_gameweek_datasets():                      # 524-564
```

**Dependencies**: None (uses fpl-dataset-builder client)
**Used by**: `data_orchestration_service.py:40, 109`
**Estimated time**: 4-6 hours

---

### Module 3: team_strength.py (636 LOC) ⚠️ COMPLEX
**Type**: Single class + helper function
**Complexity**: Medium - Used by xp_model internally
**Current State**: No domain service exists (NEW)
**Migration**: Create new `team_analytics_service.py`

**Structure:**
```python
class DynamicTeamStrength:                   # 22-606
    __init__(config, debug)
    calculate_team_strengths()               # Main calculation
    _calculate_previous_season_baseline()
    _calculate_current_season_ratings()
    _apply_venue_adjustments()
    # + 15 helper methods

def load_historical_gameweek_data():        # 607-636 (utility)
```

**Dependencies**: None
**Used by**:
- `core/xp_model.py:186` (internal)
- `visualization/charts.py` (2 usages) ← External!

**Challenge**: Breaking change for visualization layer
**Estimated time**: 6-8 hours (includes creating new service)

---

### Module 4: xp_model.py (1,293 LOC) 🔴 LARGEST
**Type**: Single class with extensive logic
**Complexity**: HIGH - Complex calculations + dependencies
**Current State**: Thin wrapper in domain
**Migration**: Major refactoring required

**Structure:**
```python
class XPModel:                                    # 25-1293
    __init__(form_weight, form_window, debug)

    # Main calculation method
    calculate_expected_points()                   # 65-146

    # Multi-gameweek support
    calculate_multi_gameweek_xp()                # 148-181

    # Core calculation engines (20+ methods)
    _calculate_single_gameweek()
    _apply_form_weighting()
    _estimate_xg_xa_statistically()
    _predict_minutes()
    _calculate_goalkeeper_points()
    _calculate_defender_points()
    _calculate_midfielder_points()
    _calculate_forward_points()
    # + 15 more helper methods
```

**Dependencies**:
- `core.team_strength.DynamicTeamStrength` (line 186) ← Must migrate Module 3 first!

**Used by**: `expected_points_service.py:107, 177, 207`
**Estimated time**: 10-12 hours (largest module)

---

### Module 5: ml_xp_model.py (940 LOC) 🔴 COMPLEX
**Type**: Single class (ML variant)
**Complexity**: HIGH - ML model + XGBoost
**Current State**: Thin wrapper in domain
**Migration**: Merge into expected_points_service.py

**Structure:**
```python
class MLXPModel:                                  # 25-940
    __init__(form_weight, form_window)

    # Main ML calculation
    calculate_expected_points()                   # ML-based

    # Feature engineering
    _prepare_features()
    _train_model()
    _predict_with_ml()
    # + helper methods

# Module-level utility
def merge_1gw_5gw_results():                     # Utility function
```

**Dependencies**: xgboost, sklearn
**Used by**: `expected_points_service.py:106, 204`
**Estimated time**: 8-10 hours

---

## Migration Order & Dependencies

```
┌─────────────────────────────────────────────────┐
│  Migration Dependency Graph                     │
└─────────────────────────────────────────────────┘

Module 1: chip_assessment.py (No dependencies)
    ↓ Can migrate independently
    ✅ Start here for quick win

Module 2: data_loader.py (No dependencies)
    ↓ Can migrate independently
    ✅ Second quick win

Module 3: team_strength.py (No dependencies)
    ↓ Must migrate BEFORE Module 4!
    ⚠️ Creates breaking change for visualization/
    ↓
Module 4: xp_model.py (Depends on Module 3)
    ↓ Largest module
    🔴 High complexity
    ↓
Module 5: ml_xp_model.py (Independent but related to Module 4)
    ↓ ML variant
    🔴 High complexity
```

**Critical Path**: Module 3 → Module 4 (team_strength must come before xp_model)

---

## Detailed Migration Strategy

### Phase 1: Quick Wins (8-12 hours)

**Module 1: chip_assessment.py → chip_assessment_service.py**
```
✅ Step 1: Read core/chip_assessment.py thoroughly
✅ Step 2: Copy ChipRecommendation dataclass to domain/services/chip_assessment_service.py
✅ Step 3: Copy ChipAssessmentEngine class and all methods
✅ Step 4: Remove `from fpl_team_picker.core.chip_assessment import ChipAssessmentEngine` (line 56)
✅ Step 5: Update ChipAssessmentService to use internal ChipAssessmentEngine
✅ Step 6: Run tests: pytest tests/domain/services/test_chip_assessment_service.py -v
✅ Step 7: Update docstrings and add type hints
✅ Step 8: Delete core/chip_assessment.py
✅ Step 9: Commit: "refactor: merge chip_assessment.py into domain service"
```

**Module 2: data_loader.py → data_orchestration_service.py**
```
✅ Step 1: Read core/data_loader.py thoroughly
✅ Step 2: Identify which functions are already implemented in DataOrchestrationService
✅ Step 3: Copy remaining functions (get_current_gameweek_info, etc.) as private methods
✅ Step 4: Remove imports from core/data_loader (lines 40, 109)
✅ Step 5: Refactor: convert module functions to class methods
✅ Step 6: Run tests: pytest tests/domain/services/test_data_orchestration_service.py -v
✅ Step 7: Delete core/data_loader.py
✅ Step 8: Commit: "refactor: merge data_loader.py into domain service"
```

---

### Phase 2: Team Strength (6-8 hours) ⚠️ BREAKING CHANGE

**Module 3: team_strength.py → NEW team_analytics_service.py**

**Challenge**: `visualization/charts.py` imports `DynamicTeamStrength` directly!

**Strategy**: Create new service + update visualization imports

```
✅ Step 1: Create domain/services/team_analytics_service.py
✅ Step 2: Copy DynamicTeamStrength class (lines 22-606)
✅ Step 3: Create TeamAnalyticsService wrapper class
✅ Step 4: Update visualization/charts.py imports:
    BEFORE: from ..core.team_strength import DynamicTeamStrength
    AFTER: from ..domain.services.team_analytics_service import TeamAnalyticsService

✅ Step 5: Update core/xp_model.py imports (prepare for Module 4 migration):
    BEFORE: from .team_strength import DynamicTeamStrength
    AFTER: from ..domain.services.team_analytics_service import TeamAnalyticsService

✅ Step 6: Create tests: tests/domain/services/test_team_analytics_service.py
✅ Step 7: Run all affected tests:
    pytest tests/domain/services/test_team_analytics_service.py -v
    pytest tests/visualization/ -v (if exists)

✅ Step 8: Delete core/team_strength.py
✅ Step 9: Commit: "refactor: create team_analytics_service and migrate team_strength.py"
```

**Migration Risk**: Visualization layer is presentation, not domain - this is acceptable coupling.

---

### Phase 3: Expected Points Model (10-12 hours) 🔴 LARGEST

**Module 4: xp_model.py → expected_points_service.py**

**Prerequisites**: Module 3 MUST be complete!

**Strategy**: Incremental refactoring with tests at each step

```
✅ Step 1: Review current expected_points_service.py structure
✅ Step 2: Copy XPModel class (1,293 LOC) into ExpectedPointsService
✅ Step 3: Refactor XPModel methods as private methods of ExpectedPointsService:
    - XPModel.calculate_expected_points() → ExpectedPointsService._calculate_xp_internal()
    - XPModel.__init__() → Merge into ExpectedPointsService.__init__()
    - All private methods remain private (_calculate_single_gameweek, etc.)

✅ Step 4: Update import statements:
    BEFORE: from fpl_team_picker.core.xp_model import XPModel
    AFTER: Use internal methods (no import needed)

✅ Step 5: Replace wrapper methods with direct calls:
    BEFORE:
        xp_model = XPModel(...)
        return xp_model.calculate_expected_points(...)
    AFTER:
        return self._calculate_xp_internal(...)

✅ Step 6: Update team strength usage:
    BEFORE: from .team_strength import DynamicTeamStrength
    AFTER: from .team_analytics_service import TeamAnalyticsService

✅ Step 7: Run tests incrementally:
    pytest tests/domain/services/test_expected_points_service.py -v -k "test_calculate"

✅ Step 8: Delete core/xp_model.py
✅ Step 9: Commit: "refactor: merge xp_model.py into expected_points_service"
```

**Testing Strategy**: Keep XPModel logic identical initially - optimize later!

---

### Phase 4: ML Model (8-10 hours) 🔴 COMPLEX

**Module 5: ml_xp_model.py → expected_points_service.py**

**Strategy**: Add as alternative calculation strategy

```
✅ Step 1: Copy MLXPModel class into ExpectedPointsService
✅ Step 2: Refactor as private methods:
    - MLXPModel.calculate_expected_points() → _calculate_ml_xp_internal()

✅ Step 3: Use strategy pattern or flag:
    def _calculate_ml_expected_points(self, ...):
        if use_ml_model:
            return self._calculate_ml_xp_internal(...)
        else:
            return self._calculate_xp_internal(...)

✅ Step 4: Copy merge_1gw_5gw_results() as private method
✅ Step 5: Update imports (remove core.ml_xp_model)
✅ Step 6: Run ML tests:
    pytest tests/domain/services/test_expected_points_service.py -v -k "ml"

✅ Step 7: Delete core/ml_xp_model.py
✅ Step 8: Commit: "refactor: merge ml_xp_model.py into expected_points_service"
```

---

### Phase 5: Cleanup (2-4 hours)

```
✅ Step 1: Verify core/ directory is empty
✅ Step 2: Delete core/ directory entirely:
    rm -rf fpl_team_picker/core/

✅ Step 3: Search for remaining imports:
    grep -r "from fpl_team_picker.core" fpl_team_picker/ tests/

✅ Step 4: Fix any remaining imports in:
    - interfaces/ (if any direct imports)
    - tests/ (if any test fixtures)

✅ Step 5: Update __init__.py files to remove core exports
✅ Step 6: Run full test suite:
    pytest tests/ -v

✅ Step 7: Update CLAUDE.md documentation
✅ Step 8: Commit: "refactor: complete core/ migration and cleanup"
```

---

## Testing Strategy

### Per-Module Testing

After each module migration:
```bash
# 1. Run specific service tests
pytest tests/domain/services/test_<service_name>.py -v

# 2. Run integration tests
pytest tests/domain/services/test_integration.py -v

# 3. Run interface compatibility tests (if exist)
pytest tests/interfaces/ -v

# 4. Quick smoke test with actual notebook
fpl-gameweek-manager  # Load and verify no errors
```

### Final Validation

After all migrations:
```bash
# 1. Full test suite
pytest tests/ -v --tb=short

# 2. Code quality
ruff check fpl_team_picker/
ruff format fpl_team_picker/

# 3. Import validation
python -c "from fpl_team_picker.domain.services import *"

# 4. Manual smoke tests
fpl-gameweek-manager    # Weekly interface
fpl-season-planner      # Season interface
```

---

## Risk Mitigation

### High-Risk Areas

1. **team_strength.py → visualization/charts.py dependency**
   - **Risk**: Breaking change for presentation layer
   - **Mitigation**: Update visualization imports simultaneously
   - **Rollback**: Keep git commit isolated, easy to revert

2. **xp_model.py complexity (1,293 LOC)**
   - **Risk**: Introducing bugs during copy-paste
   - **Mitigation**: Keep logic identical initially, refactor later
   - **Rollback**: Comprehensive tests at each step

3. **ml_xp_model.py ML dependencies**
   - **Risk**: XGBoost/sklearn version compatibility
   - **Mitigation**: Test thoroughly with actual ML predictions
   - **Rollback**: Keep ML model isolated in separate methods

### Rollback Strategy

Each migration is a separate commit:
```bash
# If Module X migration fails, rollback:
git revert <commit-hash>  # Revert specific migration
git push origin main

# Or reset to before migration:
git reset --hard <pre-migration-commit>
```

---

## Success Criteria

### Per-Module Success
- [ ] All tests passing for affected services
- [ ] No imports from core/ in domain services
- [ ] Core module deleted successfully
- [ ] Interfaces still functional (smoke test)
- [ ] Commit message clear and descriptive

### Overall Success
- [ ] All 5 modules migrated successfully
- [ ] core/ directory deleted entirely
- [ ] Zero imports from core/ anywhere in codebase
- [ ] All tests passing (pytest tests/ -v)
- [ ] All interfaces functional (manual testing)
- [ ] Documentation updated (CLAUDE.md)
- [ ] Clean Architecture Dependency Rule respected

---

## Timeline Estimate

| Phase | Modules | Estimated Time | Cumulative |
|-------|---------|----------------|------------|
| **Phase 1: Quick Wins** | 1-2 | 8-12 hours | 8-12 hours |
| **Phase 2: Team Strength** | 3 | 6-8 hours | 14-20 hours |
| **Phase 3: XP Model** | 4 | 10-12 hours | 24-32 hours |
| **Phase 4: ML Model** | 5 | 8-10 hours | 32-42 hours |
| **Phase 5: Cleanup** | - | 2-4 hours | 34-46 hours |

**Total: 34-46 hours**

**Recommended Schedule**: 2-3 weeks (2 hours/day, 5 days/week)

---

## Benefits Achieved

After completing this migration:

✅ **Dependency Inversion Fixed**: Domain layer no longer depends on core/
✅ **Code Consolidation**: All business logic in domain/services/
✅ **Zero Duplication**: Single source of truth for all calculations
✅ **Clean Architecture**: Proper layering respected
✅ **Testability**: Domain services testable in isolation
✅ **Maintainability**: Clear ownership and structure
✅ **Documentation**: Architecture aligns with documented structure

---

## Next Steps

1. **Review this plan** with team/stakeholders
2. **Create git branch**: `git checkout -b refactor/consolidate-business-logic`
3. **Start with Module 1**: chip_assessment.py (4-6 hours)
4. **Proceed incrementally**: One module at a time with tests
5. **Merge to main**: After all modules complete and tested

---

**Ready to begin?** Start with Module 1 (chip_assessment.py) for a quick win! ⭐
