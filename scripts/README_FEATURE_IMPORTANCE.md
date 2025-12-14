# Feature Importance Analysis

Comprehensive feature importance analysis script for FPL ML xP prediction model.

## Overview

This script analyzes the importance of all 117 features used in the FPL ML xP prediction model using multiple complementary methods:

1. **Random Forest MDI** (Mean Decrease in Impurity): Fast, built-in feature importance
2. **Permutation Importance**: Model-agnostic, tests real predictive power by shuffling
3. **SHAP Values** (optional): Explains individual predictions using game theory
4. **Correlation Analysis**: Identifies redundant/highly correlated features

## Quick Start

```bash
# Basic usage (GW1-9, no SHAP)
python experiments/feature_importance_analysis.py

# With SHAP analysis (slower but more detailed)
python experiments/feature_importance_analysis.py --shap-analysis

# Custom gameweek range
python experiments/feature_importance_analysis.py --start-gw 1 --end-gw 9

# Show top 50 features instead of default 30
python experiments/feature_importance_analysis.py --top-n 50
```

## Full Options

```bash
python experiments/feature_importance_analysis.py \
  --start-gw 1 \              # Training start gameweek (default: 1)
  --end-gw 9 \                # Training end gameweek (default: 9)
  --n-estimators 200 \        # Number of RF trees (default: 200)
  --max-depth 12 \            # Max tree depth (default: 12)
  --random-seed 42 \          # Random seed (default: 42)
  --top-n 30 \                # Top N features to display (default: 30)
  --shap-analysis \           # Run SHAP analysis (slower)
  --output-dir experiments/feature_importance_output  # Output directory
```

## Output Files

All results are saved to `experiments/feature_importance_output/` (configurable):

### CSV Files
- `mdi_importance.csv` - MDI feature importance rankings
- `permutation_importance.csv` - Permutation importance rankings
- `shap_importance.csv` - SHAP value rankings (if `--shap-analysis` used)
- `high_correlations.csv` - Highly correlated feature pairs (|r| > 0.8)

### Visualizations
- `mdi_importance.png` - Bar chart of top features by MDI
- `permutation_importance.png` - Bar chart of top features by permutation
- `shap_summary.png` - SHAP summary plot (bee swarm, if `--shap-analysis` used)
- `shap_bar.png` - SHAP bar plot (if `--shap-analysis` used)
- `comparison.png` - Side-by-side comparison of MDI vs Permutation

## Understanding the Methods

### 1. MDI (Mean Decrease in Impurity)
- **What it measures**: How much each feature contributes to reducing variance across all trees
- **Pros**: Fast, built-in to Random Forest
- **Cons**: Biased toward high-cardinality features (many unique values)
- **Best for**: Getting a quick overview of feature importance

### 2. Permutation Importance
- **What it measures**: How much MAE increases when feature values are randomly shuffled
- **Pros**: Model-agnostic, measures actual predictive power
- **Cons**: Slower than MDI
- **Best for**: Reliable importance rankings, identifying truly predictive features

### 3. SHAP Values
- **What it measures**: Each feature's contribution to individual predictions using game theory
- **Pros**: Explains both global and local feature importance, handles feature interactions
- **Cons**: Much slower than other methods
- **Best for**: Understanding how features interact, explaining specific predictions

### 4. Correlation Analysis
- **What it measures**: Linear correlation between features (|r| > 0.8 flagged)
- **Pros**: Identifies redundant features quickly
- **Cons**: Only captures linear relationships
- **Best for**: Feature selection, removing redundancy

## Interpreting Results

### Consensus Features
Features that appear in the top N for **all methods** are the most reliable predictors:
```
✅ Consensus Top Features (appear in all 3 methods' top 20):
   • cum_goals_scored
   • rolling_5gw_total_points
   • cum_total_points
   • ...
```

### Underutilized Features
Features with high permutation importance but low MDI importance have strong predictive power but aren't used much by the trees:
```
🔍 Underutilized Features (high permutation, low MDI):
   • team_win_probability (Perm Rank: 15, MDI Rank: 67)
```
**Action**: Consider feature engineering or ensemble methods to better utilize these features.

### High Correlations
Features with correlation |r| > 0.8 are redundant:
```
⚠️  Found 3 highly correlated pairs (|r| > 0.8):
   cum_goals_scored <-> rolling_5gw_goals_scored r=0.85
```
**Action**: Consider removing one feature from each pair to reduce model complexity.

## Example Workflow

1. **Initial analysis** (quick, no SHAP):
   ```bash
   python experiments/feature_importance_analysis.py --start-gw 1 --end-gw 9
   ```

2. **Review consensus features** in terminal output and `experiments/feature_importance_output/`

3. **Deep dive with SHAP** (slower but more detailed):
   ```bash
   python experiments/feature_importance_analysis.py --start-gw 1 --end-gw 9 --shap-analysis
   ```

4. **Feature selection**: Use consensus features + underutilized features for next iteration

5. **Remove redundancy**: Drop one feature from each highly correlated pair

## Performance Notes

- **Without SHAP**: ~2-5 minutes for GW1-9 (depends on CPU cores)
- **With SHAP**: ~10-20 minutes for GW1-9 (uses 1000-sample subsample)
- **Memory**: ~2-4GB RAM for full dataset

## Troubleshooting

### "NaN values in features"
- Check data quality in `fpl-dataset-builder`
- Ensure all gameweeks have complete data
- Run: `python -c "from client import FPLDataClient; client = FPLDataClient(); print(client.get_gameweek_performance(1).isnull().sum())"`

### "Need at least 2 gameweeks for temporal CV"
- Increase `--end-gw` to at least 7 (needs GW6 train, GW7 test minimum)
- First trainable gameweek is GW6 (needs GW1-5 for rolling features)

### SHAP analysis too slow
- Skip `--shap-analysis` flag for faster results
- Reduce `--n-estimators` (e.g., `--n-estimators 100`)
- Script already uses 1000-sample subsample for SHAP

## Next Steps

After running this analysis:

1. **Feature selection**: Create a reduced feature set using consensus features
2. **Model improvement**: Address underutilized features with feature engineering
3. **Redundancy removal**: Drop highly correlated features to reduce overfitting

## References

- Random Forest: Breiman (2001) - "Random Forests"
- Permutation Importance: Breiman (2001), Fisher et al. (2019)
- SHAP: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
