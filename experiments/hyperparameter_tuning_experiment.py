#!/usr/bin/env python3
"""
Hyperparameter Tuning Experiment for Top 2 Models

Performs grid search on Gradient Boosting and LightGBM to find optimal hyperparameters.
Uses ALL available gameweeks (GW1-7) for more robust temporal cross-validation.

Temporal CV Strategy:
- Train GW1 → Test GW2 (fold 1)
- Train GW1-2 → Test GW3 (fold 2)
- Train GW1-3 → Test GW4 (fold 3)
- Train GW1-4 → Test GW5 (fold 4)
- Train GW1-5 → Test GW6 (fold 5)
- Train GW1-6 → Test GW7 (fold 6)

Total: 6 temporal folds for reliable variance estimates
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from itertools import product
import time

from client import FPLDataClient
from fpl_team_picker.domain.services.ml_feature_engineering import FPLFeatureEngineer

print("=" * 80)
print("🔬 HYPERPARAMETER TUNING EXPERIMENT")
print("=" * 80)

# Initialize data client
client = FPLDataClient()

# Load ALL available gameweeks
print("\n📊 Loading ALL available training data...")
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
print(f"   Features: {len(feature_cols)}")

# Prepare data
X = cv_data_enriched[feature_cols].fillna(0)
y = cv_data_enriched["total_points"]

print("\n📈 Dataset Summary:")
print(f"   Total samples: {len(X):,}")
print(f"   Unique players: {cv_data_enriched['player_id'].nunique():,}")
print(f"   Gameweeks: {sorted(cv_data_enriched['gameweek'].unique())}")
print(f"   Features: {len(feature_cols)}")

# ============================================================================
# Setup Temporal Cross-Validation
# ============================================================================

print("\n" + "=" * 80)
print("📊 TEMPORAL CROSS-VALIDATION SETUP")
print("=" * 80)

gws = sorted(cv_data_enriched["gameweek"].unique())
temporal_splits = []

# Reset index for positional indexing
cv_data_reset = cv_data_enriched.reset_index(drop=True)

for i in range(len(gws) - 1):
    train_gws = gws[: i + 1]
    test_gw = gws[i + 1]

    train_mask = cv_data_reset["gameweek"].isin(train_gws)
    test_mask = cv_data_reset["gameweek"] == test_gw

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    temporal_splits.append((train_idx, test_idx))
    print(
        f"Fold {i + 1}: Train GW{min(train_gws)}-{max(train_gws)} ({len(train_idx):,} samples) → Test GW{test_gw} ({len(test_idx):,} samples)"
    )

print(f"\n✅ Total temporal folds: {len(temporal_splits)}")

# ============================================================================
# Hyperparameter Grids
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
print(f"   n_estimators: {gb_param_grid['n_estimators']}")
print(f"   max_depth: {gb_param_grid['max_depth']}")
print(f"   learning_rate: {gb_param_grid['learning_rate']}")
print(f"   min_samples_split: {gb_param_grid['min_samples_split']}")
print(f"   min_samples_leaf: {gb_param_grid['min_samples_leaf']}")
gb_combinations = (
    len(gb_param_grid["n_estimators"])
    * len(gb_param_grid["max_depth"])
    * len(gb_param_grid["learning_rate"])
    * len(gb_param_grid["min_samples_split"])
    * len(gb_param_grid["min_samples_leaf"])
)
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
print(f"   n_estimators: {lgbm_param_grid['n_estimators']}")
print(f"   max_depth: {lgbm_param_grid['max_depth']}")
print(f"   learning_rate: {lgbm_param_grid['learning_rate']}")
print(f"   min_child_samples: {lgbm_param_grid['min_child_samples']}")
print(f"   subsample: {lgbm_param_grid['subsample']}")
print(f"   colsample_bytree: {lgbm_param_grid['colsample_bytree']}")
lgbm_combinations = (
    len(lgbm_param_grid["n_estimators"])
    * len(lgbm_param_grid["max_depth"])
    * len(lgbm_param_grid["learning_rate"])
    * len(lgbm_param_grid["min_child_samples"])
    * len(lgbm_param_grid["subsample"])
    * len(lgbm_param_grid["colsample_bytree"])
)
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
    print(f"Each tested with {len(cv_splits)} temporal CV folds\n")

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
            pipeline,
            X,
            y,
            cv=cv_splits,
            scoring="neg_mean_absolute_error",
            n_jobs=1,  # Already parallel in model
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

        # Progress update every 10 combinations
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

print("\n" + "=" * 80)
print("🌲 GRADIENT BOOSTING TUNING")
print("=" * 80)

gb_results = grid_search_cv(
    GradientBoostingRegressor, gb_param_grid, "Gradient Boosting", temporal_splits, X, y
)

# ============================================================================
# Run Grid Search for LightGBM
# ============================================================================

print("\n" + "=" * 80)
print("💡 LIGHTGBM TUNING")
print("=" * 80)

lgbm_results = grid_search_cv(
    LGBMRegressor, lgbm_param_grid, "LightGBM", temporal_splits, X, y
)

# ============================================================================
# Compare Best Models
# ============================================================================

print("\n" + "=" * 80)
print("🏆 FINAL RESULTS")
print("=" * 80)

gb_best = gb_results.iloc[0]
lgbm_best = lgbm_results.iloc[0]

print("\n🌲 Best Gradient Boosting:")
print(f"   MAE: {gb_best['MAE_mean']:.4f} ± {gb_best['MAE_std']:.4f}")
print("   Parameters:")
for param in gb_param_grid.keys():
    print(f"      {param}: {gb_best[param]}")

print("\n💡 Best LightGBM:")
print(f"   MAE: {lgbm_best['MAE_mean']:.4f} ± {lgbm_best['MAE_std']:.4f}")
print("   Parameters:")
for param in lgbm_param_grid.keys():
    print(f"      {param}: {lgbm_best[param]}")

print("\n" + "=" * 80)
print("📊 HEAD-TO-HEAD COMPARISON")
print("=" * 80)

if gb_best["MAE_mean"] < lgbm_best["MAE_mean"]:
    winner = "Gradient Boosting"
    winner_mae = gb_best["MAE_mean"]
    loser_mae = lgbm_best["MAE_mean"]
else:
    winner = "LightGBM"
    winner_mae = lgbm_best["MAE_mean"]
    loser_mae = gb_best["MAE_mean"]

improvement = loser_mae - winner_mae
improvement_pct = (improvement / loser_mae) * 100

print(f"\n🏆 WINNER: {winner}")
print(f"   MAE: {winner_mae:.4f}")
print(f"   Beat opponent by: {improvement:.4f} ({improvement_pct:.2f}%)")

# Save top 5 configurations for each model
print("\n" + "=" * 80)
print("💾 SAVING RESULTS")
print("=" * 80)

gb_results.head(5).to_csv("experiments/results/gb_top5_configs.csv", index=False)
lgbm_results.head(5).to_csv("experiments/results/lgbm_top5_configs.csv", index=False)

print("\n✅ Saved top 5 configurations:")
print("   - experiments/results/gb_top5_configs.csv")
print("   - experiments/results/lgbm_top5_configs.csv")

print("\n" + "=" * 80)
print("🎯 PRODUCTION RECOMMENDATION")
print("=" * 80)

print(f"\nUse {winner} with these hyperparameters:")
if winner == "Gradient Boosting":
    for param in gb_param_grid.keys():
        print(f"   {param}: {gb_best[param]}")
else:
    for param in lgbm_param_grid.keys():
        print(f"   {param}: {lgbm_best[param]}")

print(f"\nExpected temporal CV MAE: {winner_mae:.4f}")
print("\n" + "=" * 80)
