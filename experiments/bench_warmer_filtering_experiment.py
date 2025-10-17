#!/usr/bin/env python3
"""
Standalone test script for bench warmer filtering experiment.
Measures MAE before/after removing bench warmers from training data.
"""

import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from client import FPLDataClient
from fpl_team_picker.domain.services.ml_feature_engineering import FPLFeatureEngineer

print("=" * 80)
print("🧪 BENCH WARMER FILTERING EXPERIMENT")
print("=" * 80)

# Initialize data client
client = FPLDataClient()

# Load data for GW6 and GW7 (hardcoded based on notebook)
print("\n📊 Loading training data...")
gw6_data = client.get_gameweek_performance(6)
gw7_data = client.get_gameweek_performance(7)

# Combine into training dataset
cv_data = pd.concat([gw6_data, gw7_data], ignore_index=True)
print(f"   Total samples: {len(cv_data):,}")

# Enrich with position data (required by FPLFeatureEngineer)
print("   Enriching with player position data...")
players_data = client.get_current_players()
cv_data = cv_data.merge(
    players_data[["player_id", "position"]],
    on="player_id",
    how="left",
)
if cv_data["position"].isna().any():
    missing_count = cv_data["position"].isna().sum()
    print(f"   ⚠️ Warning: {missing_count} records missing position data")

# Preserve target and metadata before feature engineering
target = cv_data["total_points"].copy()
player_ids = cv_data["player_id"].copy()
positions = cv_data["position"].copy()

# Feature engineering
print("\n🔧 Engineering features...")
feature_engineer = FPLFeatureEngineer()
cv_data_enriched = feature_engineer.fit_transform(cv_data)

# Add back target, position, and player_id (needed for filtering and grouping)
cv_data_enriched["total_points"] = target
cv_data_enriched["player_id"] = player_ids
cv_data_enriched["position"] = positions

# Get feature columns (all except metadata and target)
exclude_cols = {
    "player_id",
    "player_name",
    "team_id",
    "team_name",
    "position",
    "gameweek",
    "total_points",
    "minutes",
    "opponent_team_id",
}
feature_cols = [col for col in cv_data_enriched.columns if col not in exclude_cols]
print(f"   Features: {len(feature_cols)}")

# Prepare UNFILTERED data
cv_data_unfiltered = cv_data_enriched.copy()
X_cv_unfiltered = cv_data_unfiltered[feature_cols].fillna(0)
y_cv_unfiltered = cv_data_unfiltered["total_points"]
cv_groups_unfiltered = cv_data_unfiltered["player_id"].values

# Apply bench warmer filter
print("\n🔬 Applying filter (cumulative_minutes > 90)...")
print(
    f"   DEBUG: Available columns: {sorted(cv_data_enriched.columns.tolist())[:20]}..."
)  # First 20 columns
print(
    f"   DEBUG: 'cumulative_minutes' in columns: {'cumulative_minutes' in cv_data_enriched.columns}"
)

if "cumulative_minutes" in cv_data_enriched.columns:
    print("   DEBUG: cumulative_minutes stats:")
    print(f"           min={cv_data_enriched['cumulative_minutes'].min()}")
    print(f"           max={cv_data_enriched['cumulative_minutes'].max()}")
    print(f"           mean={cv_data_enriched['cumulative_minutes'].mean():.1f}")
    print(
        f"           >= 45: {(cv_data_enriched['cumulative_minutes'] >= 45).sum()} samples"
    )
    print(
        "   NOTE: This appears to be per-90 normalized, not cumulative season minutes!"
    )
    print("   Using threshold >= 45 (meaningful playing time)")

    # Use >= 45 threshold (meaningful playing time across gameweeks)
    cv_data_filtered = cv_data_enriched[cv_data_enriched["cumulative_minutes"] >= 45]
else:
    # Fallback: use cumsum_minutes or minutes_played feature
    minutes_col = None
    for col in ["cumsum_minutes", "minutes_played", "cumulative_season_minutes"]:
        if col in cv_data_enriched.columns:
            minutes_col = col
            break

    if minutes_col:
        print(f"   Using '{minutes_col}' instead of 'cumulative_minutes'")
        cv_data_filtered = cv_data_enriched[cv_data_enriched[minutes_col] > 90]
    else:
        print("   ⚠️ No minutes column found - cannot filter bench warmers!")
        cv_data_filtered = cv_data_enriched.copy()
X_cv_filtered = cv_data_filtered[feature_cols].fillna(0)
y_cv_filtered = cv_data_filtered["total_points"]
cv_groups_filtered = cv_data_filtered["player_id"].values

removed_count = len(cv_data_unfiltered) - len(cv_data_filtered)
removed_pct = (removed_count / len(cv_data_unfiltered)) * 100

print(f"   Before filtering: {len(cv_data_unfiltered):,} samples")
print(f"   After filtering:  {len(cv_data_filtered):,} samples")
print(f"   Removed:          {removed_count:,} bench warmers ({removed_pct:.1f}%)")

# Cross-validation setup
print("\n📈 Training Ridge models with 5-fold Player-Based GroupKFold CV...")
n_folds = 5
gkf = GroupKFold(n_splits=n_folds)

# Model pipeline
ridge_pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])

# Evaluate UNFILTERED model
print("\n   Testing UNFILTERED model...")
unfiltered_mae_scores = -cross_val_score(
    ridge_pipeline,
    X_cv_unfiltered,
    y_cv_unfiltered,
    groups=cv_groups_unfiltered,
    cv=gkf,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
unfiltered_mae_mean = unfiltered_mae_scores.mean()
unfiltered_mae_std = unfiltered_mae_scores.std()

print(f"   ✅ Unfiltered MAE: {unfiltered_mae_mean:.3f} ± {unfiltered_mae_std:.3f}")

# Evaluate FILTERED model
print("\n   Testing FILTERED model...")
filtered_mae_scores = -cross_val_score(
    ridge_pipeline,
    X_cv_filtered,
    y_cv_filtered,
    groups=cv_groups_filtered,
    cv=gkf,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
)
filtered_mae_mean = filtered_mae_scores.mean()
filtered_mae_std = filtered_mae_scores.std()

print(f"   ✅ Filtered MAE:   {filtered_mae_mean:.3f} ± {filtered_mae_std:.3f}")

# Calculate improvement
mae_improvement = unfiltered_mae_mean - filtered_mae_mean
mae_improvement_pct = (mae_improvement / unfiltered_mae_mean) * 100

print("\n" + "=" * 80)
print("🎯 EXPERIMENT RESULTS")
print("=" * 80)
print(
    f"BEFORE (unfiltered): MAE = {unfiltered_mae_mean:.3f} ± {unfiltered_mae_std:.3f} ({len(cv_data_unfiltered):,} samples)"
)
print(
    f"AFTER (filtered):    MAE = {filtered_mae_mean:.3f} ± {filtered_mae_std:.3f} ({len(cv_data_filtered):,} samples)"
)
print(f"\nImprovement: {mae_improvement:.3f} ({mae_improvement_pct:+.1f}%)")
print(f"Samples removed: {removed_count:,} ({removed_pct:.1f}%)")

if mae_improvement > 0:
    print("\n✅ CONCLUSION: Filtering bench warmers IMPROVES model accuracy!")
    print("   Recommendation: Keep this filter for production training.")
elif mae_improvement < -0.01:
    print("\n⚠️  CONCLUSION: Filtering bench warmers DEGRADES model accuracy.")
    print("   Recommendation: Do NOT use this filter.")
else:
    print("\n➖ CONCLUSION: Filtering has NEGLIGIBLE effect on accuracy.")
    print("   Recommendation: Optional - consider for cleaner training data.")

print("=" * 80)
