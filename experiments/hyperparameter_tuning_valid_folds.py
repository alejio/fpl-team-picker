#!/usr/bin/env python3
"""
Hyperparameter Tuning Experiment - VALID FOLDS ONLY

Only evaluates on folds where the test gameweek has a proper 5GW historical window.

Valid Folds (test GW has ≥5 GW history):
- Fold 5: Train GW1-5 → Test GW6 (GW6 has GW1-5 history)
- Fold 6: Train GW1-6 → Test GW7 (GW7 has GW2-6 history)

Invalid Folds (insufficient history):
- Fold 1: Train GW1 → Test GW2 (only 1 GW history) ❌
- Fold 2: Train GW1-2 → Test GW3 (only 2 GW history) ❌
- Fold 3: Train GW1-3 → Test GW4 (only 3 GW history) ❌
- Fold 4: Train GW1-4 → Test GW5 (only 4 GW history) ❌
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from itertools import product
import time

from client import FPLDataClient
from fpl_team_picker.domain.services.ml_feature_engineering import FPLFeatureEngineer

print("=" * 80)
print("🔬 HYPERPARAMETER TUNING - VALID FOLDS ONLY")
print("=" * 80)
print("\n⚠️  Only using folds where test GW has ≥5 gameweeks of history")
print("   This ensures proper 5GW rolling window features\n")

# Initialize data client
client = FPLDataClient()

# Load ALL available gameweeks
print("📊 Loading ALL available training data...")
available_gws = []
for gw in range(1, 15):
    try:
        data = client.get_gameweek_performance(gw)
        if not data.empty:
            available_gws.append(gw)
    except Exception:
        break

print(f"   Available gameweeks: {available_gws}")
print(f"   Total: {len(available_gws)} gameweeks")

# Load all data
all_data = []
for gw in available_gws:
    gw_data = client.get_gameweek_performance(gw)
    all_data.append(gw_data)

cv_data = pd.concat(all_data, ignore_index=True)
print(f"   Total samples: {len(cv_data):,}")

# Enrich with position data
print("   Enriching with player position data...")
players_data = client.get_current_players()
cv_data = cv_data.merge(
    players_data[["player_id", "position"]],
    on="player_id",
    how="left",
)

# Preserve target and metadata
target = cv_data["total_points"].copy()
player_ids = cv_data["player_id"].copy()
gameweeks = cv_data["gameweek"].copy()

# Feature engineering
print("\n🔧 Engineering features...")
feature_engineer = FPLFeatureEngineer()
cv_data_enriched = feature_engineer.fit_transform(cv_data)

# Add back target, gameweek, player_id
cv_data_enriched["total_points"] = target
cv_data_enriched["player_id"] = player_ids
cv_data_enriched["gameweek"] = gameweeks

# Get feature columns
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

# Prepare data
X = cv_data_enriched[feature_cols].fillna(0)
y = cv_data_enriched["total_points"]

print("\n📈 Dataset Summary:")
print(f"   Total samples: {len(X):,}")
print(f"   Unique players: {cv_data_enriched['player_id'].nunique():,}")
print(f"   Gameweeks: {sorted(cv_data_enriched['gameweek'].unique())}")
print(f"   Features: {len(feature_cols)}")

# ============================================================================
# Setup VALID Temporal Cross-Validation Folds
# ============================================================================

print("\n" + "=" * 80)
print("📊 VALID TEMPORAL CROSS-VALIDATION SETUP")
print("=" * 80)

gws = sorted(cv_data_enriched["gameweek"].unique())
all_temporal_splits = []
valid_temporal_splits = []

# Reset index for positional indexing
cv_data_reset = cv_data_enriched.reset_index(drop=True)

for i in range(len(gws) - 1):
    train_gws = gws[: i + 1]
    test_gw = gws[i + 1]

    train_mask = cv_data_reset["gameweek"].isin(train_gws)
    test_mask = cv_data_reset["gameweek"] == test_gw

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    fold_split = (train_idx, test_idx)
    all_temporal_splits.append(fold_split)

    # Only include folds where test GW has ≥5 GW history
    # Test GW N needs history from GW1 to GW(N-1), which means N-1 ≥ 5, so N ≥ 6
    is_valid = test_gw >= 6

    status = "✅ VALID" if is_valid else "❌ SKIP"
    print(
        f"Fold {i + 1}: Train GW{min(train_gws)}-{max(train_gws)} ({len(train_idx):,} samples) → Test GW{test_gw} ({len(test_idx):,} samples) {status}"
    )

    if is_valid:
        valid_temporal_splits.append(fold_split)

print(
    f"\n✅ Total valid folds: {len(valid_temporal_splits)} (out of {len(all_temporal_splits)} total)"
)
print("   Using only folds where test GW ≥ 6 (has proper 5GW history)")

# ============================================================================
# Hyperparameter Grids (Reduced for faster iteration on 2 folds)
# ============================================================================

print("\n" + "=" * 80)
print("🎛️  HYPERPARAMETER GRIDS")
print("=" * 80)

# Gradient Boosting parameter grid
gb_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "min_samples_split": [10, 20],
    "min_samples_leaf": [5, 10],
}

print("\n🌲 Gradient Boosting Grid:")
for param, values in gb_param_grid.items():
    print(f"   {param}: {values}")
gb_combinations = np.prod([len(v) for v in gb_param_grid.values()])
print(f"   Total combinations: {gb_combinations}")

# LightGBM parameter grid
lgbm_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "min_child_samples": [10, 20, 30],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
}

print("\n💡 LightGBM Grid:")
for param, values in lgbm_param_grid.items():
    print(f"   {param}: {values}")
lgbm_combinations = np.prod([len(v) for v in lgbm_param_grid.values()])
print(f"   Total combinations: {lgbm_combinations}")

# ============================================================================
# Grid Search Function
# ============================================================================


def grid_search_cv(model_class, param_grid, model_name, cv_splits, X, y):
    """Manual grid search with temporal CV"""
    print(f"\n{'=' * 80}")
    print(f"🔍 GRID SEARCH: {model_name}")
    print(f"{'=' * 80}")

    results = []
    param_combinations = list(product(*param_grid.values()))
    total = len(param_combinations)

    print(f"\nTesting {total} parameter combinations...")
    print(f"Each tested with {len(cv_splits)} VALID temporal CV folds\n")

    start_time = time.time()

    for idx, param_values in enumerate(param_combinations, 1):
        params = dict(zip(param_grid.keys(), param_values))

        # Add fixed params
        if model_name == "Gradient Boosting":
            params["random_state"] = 42
        else:  # LightGBM
            params["random_state"] = 42
            params["n_jobs"] = -1
            params["verbose"] = -1

        # Create model
        if model_name == "Gradient Boosting":
            model = GradientBoostingRegressor(**params)
        else:
            model = LGBMRegressor(**params)

        # Create pipeline
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

        # Cross-validate
        mae_scores = -cross_val_score(
            pipeline, X, y, cv=cv_splits, scoring="neg_mean_absolute_error", n_jobs=1
        )

        mae_mean = mae_scores.mean()
        mae_std = mae_scores.std()

        results.append(
            {
                **params,
                "MAE_mean": mae_mean,
                "MAE_std": mae_std,
                "MAE_scores": mae_scores,
            }
        )

        # Progress update
        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - start_time
            eta = (elapsed / idx) * (total - idx)
            print(
                f"   [{idx}/{total}] Best so far: {min(r['MAE_mean'] for r in results):.4f} | "
                f"Elapsed: {elapsed / 60:.1f}m | ETA: {eta / 60:.1f}m"
            )

    total_time = time.time() - start_time
    print(f"\n✅ Grid search completed in {total_time / 60:.1f} minutes")

    # Sort by MAE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAE_mean")

    return results_df


# ============================================================================
# Run Grid Search for Gradient Boosting
# ============================================================================

gb_results = grid_search_cv(
    GradientBoostingRegressor,
    gb_param_grid,
    "Gradient Boosting",
    valid_temporal_splits,
    X,
    y,
)

# ============================================================================
# Run Grid Search for LightGBM
# ============================================================================

lgbm_results = grid_search_cv(
    LGBMRegressor, lgbm_param_grid, "LightGBM", valid_temporal_splits, X, y
)

# ============================================================================
# Test Ridge Baseline
# ============================================================================

print("\n" + "=" * 80)
print("📊 RIDGE BASELINE")
print("=" * 80)

ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])

ridge_mae_scores = -cross_val_score(
    ridge, X, y, cv=valid_temporal_splits, scoring="neg_mean_absolute_error", n_jobs=-1
)

ridge_mae_mean = ridge_mae_scores.mean()
ridge_mae_std = ridge_mae_scores.std()

print(f"\nRidge (baseline): {ridge_mae_mean:.4f} ± {ridge_mae_std:.4f}")
print(f"Per-fold MAE: {ridge_mae_scores}")

# ============================================================================
# Compare Best Models
# ============================================================================

print("\n" + "=" * 80)
print("🏆 FINAL RESULTS (VALID FOLDS ONLY)")
print("=" * 80)

gb_best = gb_results.iloc[0]
lgbm_best = lgbm_results.iloc[0]

print("\n🌲 Best Gradient Boosting:")
print(f"   MAE: {gb_best['MAE_mean']:.4f} ± {gb_best['MAE_std']:.4f}")
print(f"   Per-fold: {gb_best['MAE_scores']}")
print("   Parameters:")
for param in gb_param_grid.keys():
    print(f"      {param}: {gb_best[param]}")

print("\n💡 Best LightGBM:")
print(f"   MAE: {lgbm_best['MAE_mean']:.4f} ± {lgbm_best['MAE_std']:.4f}")
print(f"   Per-fold: {lgbm_best['MAE_scores']}")
print("   Parameters:")
for param in lgbm_param_grid.keys():
    print(f"      {param}: {lgbm_best[param]}")

print("\n📊 Ridge (baseline):")
print(f"   MAE: {ridge_mae_mean:.4f} ± {ridge_mae_std:.4f}")
print(f"   Per-fold: {ridge_mae_scores}")

# ============================================================================
# Head-to-Head Comparison
# ============================================================================

print("\n" + "=" * 80)
print("🥊 HEAD-TO-HEAD COMPARISON")
print("=" * 80)

models = [
    ("LightGBM", lgbm_best["MAE_mean"], lgbm_best["MAE_std"]),
    ("Gradient Boosting", gb_best["MAE_mean"], gb_best["MAE_std"]),
    ("Ridge", ridge_mae_mean, ridge_mae_std),
]

models_sorted = sorted(models, key=lambda x: x[1])

print("\n🏆 RANKING:")
for rank, (name, mae, std) in enumerate(models_sorted, 1):
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")
    print(f"{medal} {name:<20} MAE: {mae:.4f} ± {std:.4f}")

winner_name, winner_mae, winner_std = models_sorted[0]
baseline_mae = ridge_mae_mean

improvement = baseline_mae - winner_mae
improvement_pct = (improvement / baseline_mae) * 100

print(f"\n✅ Best Model: {winner_name}")
print(f"   MAE: {winner_mae:.4f} ± {winner_std:.4f}")
print(f"   vs Ridge: {improvement:+.4f} ({improvement_pct:+.2f}%)")

# Save results
print("\n" + "=" * 80)
print("💾 SAVING RESULTS")
print("=" * 80)

gb_results.head(5).to_csv("experiments/results/gb_valid_folds_top5.csv", index=False)
lgbm_results.head(5).to_csv(
    "experiments/results/lgbm_valid_folds_top5.csv", index=False
)

print("\n✅ Saved top 5 configurations:")
print("   - experiments/results/gb_valid_folds_top5.csv")
print("   - experiments/results/lgbm_valid_folds_top5.csv")

print("\n" + "=" * 80)
print("🎯 PRODUCTION RECOMMENDATION")
print("=" * 80)

print(f"\nUse {winner_name} with optimized hyperparameters")
print(f"Expected MAE on future gameweeks: ~{winner_mae:.2f} points")
print(f"This is {improvement_pct:.1f}% better than Ridge baseline")

print("\n" + "=" * 80)
