# Simulated Annealing Stability Guide

## Problem
Simulated annealing uses randomness for exploration, which means you get different results each time you run optimization. This can be frustrating when you want consistent recommendations.

## Solutions Implemented

### 1. Random Seed Configuration ✅

Set a fixed random seed for reproducible results:

```python
# In config/settings.py or via environment variable
config.optimization.sa_random_seed = 42  # Any integer
```

**How it works:**
- When `sa_random_seed` is set, the algorithm uses the same random sequence each run
- Each restart uses `seed + restart_number` for diversity while maintaining reproducibility
- **Result:** Same input → Same output (deterministic)

**Usage:**
```python
from fpl_team_picker.config import config
config.optimization.sa_random_seed = 42  # Set once, get same results every time
```

### 2. Deterministic Mode ✅

Enable deterministic mode to prefer top candidates when the xP gap is large:

```python
config.optimization.sa_deterministic_mode = True
```

**How it works:**
- When top candidate is >0.5 xP better than 2nd place → always pick top
- When gap is small (<0.5 xP) → still use weighted sampling for exploration
- **Result:** More stable recommendations, less random exploration

**Trade-off:**
- ✅ More stable results
- ✅ Faster convergence
- ⚠️ Less exploration (might miss some good alternatives)

### 3. Increase Iterations/Restarts

More iterations and restarts = better convergence = more stable results:

```python
config.optimization.sa_iterations = 5000  # Default: 2000
config.optimization.sa_restarts = 5       # Default: 3
```

**Recommendation:**
- For stability: `iterations=5000`, `restarts=5`
- For speed: `iterations=2000`, `restarts=3` (default)

## Recommended Settings for Stability

### Option A: Maximum Stability (Deterministic)
```python
config.optimization.sa_random_seed = 42
config.optimization.sa_deterministic_mode = True
config.optimization.sa_iterations = 5000
config.optimization.sa_restarts = 5
```
**Result:** Same results every time, prefers best candidates

### Option B: Reproducible but Exploratory
```python
config.optimization.sa_random_seed = 42
config.optimization.sa_deterministic_mode = False
config.optimization.sa_iterations = 5000
config.optimization.sa_restarts = 5
```
**Result:** Same results every time, but still explores alternatives

### Option C: Balanced (Default + Seed)
```python
config.optimization.sa_random_seed = 42
config.optimization.sa_deterministic_mode = False
config.optimization.sa_iterations = 2000  # Default
config.optimization.sa_restarts = 3        # Default
```
**Result:** Reproducible with default exploration

## How to Configure

### Method 1: Edit `config/settings.py`
```python
class OptimizationConfig(BaseModel):
    sa_random_seed: Optional[int] = Field(
        default=42,  # Change from None to 42
        description="Random seed for SA reproducibility"
    )
    sa_deterministic_mode: bool = Field(
        default=True,  # Change from False to True
        description="Deterministic mode for stability"
    )
```

### Method 2: Environment Variable (Future)
```bash
export FPL_SA_RANDOM_SEED=42
export FPL_SA_DETERMINISTIC_MODE=true
```

### Method 3: Runtime Override
```python
from fpl_team_picker.config import config
config.optimization.sa_random_seed = 42
config.optimization.sa_deterministic_mode = True
```

## Understanding the Differences

| Setting | Stability | Exploration | Speed |
|---------|-----------|-------------|-------|
| No seed, normal mode | ❌ Low | ✅ High | ✅ Fast |
| Seed, normal mode | ✅ High | ✅ High | ✅ Fast |
| Seed, deterministic | ✅✅ Highest | ⚠️ Medium | ✅ Fast |
| Seed, deterministic, more iterations | ✅✅✅ Highest | ⚠️ Medium | ⚠️ Slower |

## Testing Stability

Run optimization twice with same settings and compare:

```python
# First run
result1 = optimization_service.optimize_transfers(...)

# Second run (should be identical if seed is set)
result2 = optimization_service.optimize_transfers(...)

assert result1 == result2  # Should pass if seed is set
```

## Troubleshooting

**Q: Still getting different results with seed set?**
- Check that `sa_random_seed` is not `None`
- Ensure you're not modifying the seed between runs
- Check that data inputs are identical (same gameweek, same squad)

**Q: Deterministic mode too conservative?**
- Set `sa_deterministic_mode = False`
- Increase `sa_restarts` to explore more alternatives
- The algorithm will still use weighted sampling for exploration

**Q: Want more exploration but still reproducible?**
- Set `sa_random_seed = 42` (for reproducibility)
- Set `sa_deterministic_mode = False` (for exploration)
- Increase `sa_restarts = 5` (more diverse restarts)

## Technical Details

### Random Seed Implementation
- Seed is set at start of `_optimize_transfers_sa()`
- Each restart uses `seed + restart_number` for diversity
- Seed affects: `random.random()`, `random.randint()`, `random.sample()`, `random.choice()`

### Deterministic Mode Logic
```python
if top_xp - second_xp > 0.5:
    # Clear winner → always pick top
    pick_top_candidate()
else:
    # Close race → use weighted sampling
    weighted_random_sample()
```

### Why Different Results Without Seed?
1. Random player selection in `_swap_transfer_players()`
2. Random acceptance of worse moves (SA exploration)
3. Random sampling from top candidates
4. Random initial state generation (for initial squad)

All of these use `random` module, which is non-deterministic without a seed.
