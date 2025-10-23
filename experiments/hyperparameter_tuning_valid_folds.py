#!/usr/bin/env python3
"""
Hyperparameter Tuning Experiment - CatBoost Optimization

Comprehensive grid search for CatBoost (winner from algorithm comparison).
Only evaluates on folds where the test gameweek has a proper 5GW historical window.

Valid Folds (test GW has ≥5 GW history):
- Fold 5: Train GW1-5 → Test GW6 (GW6 has GW1-5 history)
- Fold 6: Train GW1-6 → Test GW7 (GW7 has GW2-6 history)
- Fold 7: Train GW1-7 → Test GW8 (GW8 has GW3-7 history)

Focus: CatBoost hyperparameter optimization for production deployment
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from itertools import product
import time

from client import FPLDataClient
from fpl_team_picker.domain.services.ml_feature_engineering import FPLFeatureEngineer
from fpl_team_picker.domain.services.ml_pipeline_factory import (
    get_team_strength_ratings,
)

print("=" * 80)
print("🔬 CATBOOST HYPERPARAMETER TUNING - VALID FOLDS ONLY")
print("=" * 80)
print("\n⚠️  Only using folds where test GW has ≥5 gameweeks of history")
print("   This ensures proper 5GW rolling window features")
print("\n🎯 Goal: Find optimal CatBoost hyperparameters for production\n")

# Initialize data client
client = FPLDataClient()

# Load GW1-8 (matching algorithm comparison experiment)
print("📊 Loading training data (GW1-8)...")
historical_data = []
start_gw = 1
end_gw = 8

for gw in range(start_gw, end_gw + 1):
    gw_performance = client.get_gameweek_performance(gw)
    if not gw_performance.empty:
        gw_performance["gameweek"] = gw
        historical_data.append(gw_performance)
        print(f"   ✅ GW{gw}: {len(gw_performance):,} players")
    else:
        print(f"   ⚠️  GW{gw}: No data available")

if not historical_data:
    raise ValueError("No historical data loaded. Check gameweek range.")

cv_data = pd.concat(historical_data, ignore_index=True)
print(f"\n   📊 Total samples: {len(cv_data):,}")
print(f"   📅 Gameweeks: GW{start_gw}-GW{end_gw}")

# Load fixtures and teams data for feature engineering
print("\n🔧 Loading context data for feature engineering...")
fixtures_df = client.get_fixtures_normalized()
teams_df = client.get_current_teams()
players_data = client.get_current_players()

print(f"   ✅ Fixtures: {len(fixtures_df):,}")
print(f"   ✅ Teams: {len(teams_df):,}")
print(f"   ✅ Players: {len(players_data):,}")

# Enrich with position data
cv_data = cv_data.merge(
    players_data[["player_id", "position"]],
    on="player_id",
    how="left",
)

# Convert value to price for analysis
if "value" in cv_data.columns and "price" not in cv_data.columns:
    cv_data["price"] = cv_data["value"] / 10.0

# Preserve target and metadata
target = cv_data["total_points"].copy()
player_ids = cv_data["player_id"].copy()
gameweeks = cv_data["gameweek"].copy()
prices = cv_data["price"].copy()

# Feature engineering (with full context)
print("\n🔧 Engineering features...")
team_strength = get_team_strength_ratings()
feature_engineer = FPLFeatureEngineer(
    fixtures_df=fixtures_df,
    teams_df=teams_df,
    team_strength=team_strength,
)
cv_data_enriched = feature_engineer.fit_transform(cv_data, target)

# Add back target, gameweek, player_id, price
cv_data_enriched["total_points"] = target.values
cv_data_enriched["player_id"] = player_ids.values
cv_data_enriched["gameweek"] = gameweeks.values
cv_data_enriched["price"] = prices.values

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
# Hyperparameter Grids - CatBoost Comprehensive Tuning
# ============================================================================

print("\n" + "=" * 80)
print("🎛️  CATBOOST HYPERPARAMETER GRID")
print("=" * 80)

# CatBoost parameter grid (comprehensive search)
catboost_param_grid = {
    # Number of trees
    "iterations": [100, 200, 300, 500],
    # Tree depth (CatBoost is robust to deeper trees due to ordered boosting)
    "depth": [4, 6, 8, 10],
    # Learning rate
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    # L2 regularization (prevents overfitting)
    "l2_leaf_reg": [1, 3, 5, 7, 10],
    # Bagging temperature (controls randomness in Bayesian bootstrap)
    "bagging_temperature": [0, 0.5, 1.0],
    # Random strength (regularization for splits)
    "random_strength": [0, 1, 2],
}

print("\n🐱 CatBoost Grid:")
for param, values in catboost_param_grid.items():
    print(f"   {param}: {values}")
catboost_combinations = np.prod([len(v) for v in catboost_param_grid.values()])
print(f"   Total combinations: {catboost_combinations}")
print(
    f"\n⚠️  This will take significant time: ~{catboost_combinations * 3 * 2 / 60:.0f} minutes"
)
print(f"   (3 CV folds × ~2 sec per model × {catboost_combinations} combinations)")

# Also test LightGBM and XGBoost with smaller grids for comparison
lgbm_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 6, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [31, 50],
    "min_child_samples": [15, 20, 25],
    "reg_alpha": [0, 0.1, 0.3],
    "reg_lambda": [0, 0.1, 0.3],
}

print("\n💡 LightGBM Grid (for comparison):")
for param, values in lgbm_param_grid.items():
    print(f"   {param}: {values}")
lgbm_combinations = np.prod([len(v) for v in lgbm_param_grid.values()])
print(f"   Total combinations: {lgbm_combinations}")

xgb_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 6, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [2, 3, 4],
    "gamma": [0, 0.1, 0.2],
    "reg_alpha": [0, 0.1, 0.3],
    "reg_lambda": [0, 0.1, 0.3],
}

print("\n🚀 XGBoost Grid (for comparison):")
for param, values in xgb_param_grid.items():
    print(f"   {param}: {values}")
xgb_combinations = np.prod([len(v) for v in xgb_param_grid.values()])
print(f"   Total combinations: {xgb_combinations}")

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

        # Add fixed params based on model type
        if model_name == "CatBoost":
            params["random_seed"] = 42
            params["verbose"] = 0
            params["thread_count"] = -1
            params["loss_function"] = "MAE"  # Optimize MAE directly
            model = CatBoostRegressor(**params)
        elif model_name == "LightGBM":
            params["random_state"] = 42
            params["n_jobs"] = -1
            params["verbose"] = -1
            model = LGBMRegressor(**params)
        elif model_name == "XGBoost":
            params["random_state"] = 42
            params["n_jobs"] = -1
            params["verbosity"] = 0
            model = XGBRegressor(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

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

        # Progress update (more frequent for slower models)
        update_freq = 5 if model_name == "CatBoost" else 10
        if idx % update_freq == 0 or idx == total:
            elapsed = time.time() - start_time
            eta = (elapsed / idx) * (total - idx)
            print(
                f"   [{idx:4d}/{total}] Best so far: {min(r['MAE_mean'] for r in results):.4f} | "
                f"Elapsed: {elapsed / 60:.1f}m | ETA: {eta / 60:.1f}m"
            )

    total_time = time.time() - start_time
    print(f"\n✅ Grid search completed in {total_time / 60:.1f} minutes")

    # Sort by MAE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAE_mean")

    return results_df


# ============================================================================
# Run Grid Search for CatBoost (PRIMARY FOCUS)
# ============================================================================

catboost_results = grid_search_cv(
    CatBoostRegressor,
    catboost_param_grid,
    "CatBoost",
    valid_temporal_splits,
    X,
    y,
)

# ============================================================================
# Run Grid Search for LightGBM (comparison)
# ============================================================================

lgbm_results = grid_search_cv(
    LGBMRegressor, lgbm_param_grid, "LightGBM", valid_temporal_splits, X, y
)

# ============================================================================
# Run Grid Search for XGBoost (comparison)
# ============================================================================

xgb_results = grid_search_cv(
    XGBRegressor, xgb_param_grid, "XGBoost", valid_temporal_splits, X, y
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

catboost_best = catboost_results.iloc[0]
lgbm_best = lgbm_results.iloc[0]
xgb_best = xgb_results.iloc[0]

print("\n🐱 Best CatBoost (PRIMARY):")
print(f"   MAE: {catboost_best['MAE_mean']:.4f} ± {catboost_best['MAE_std']:.4f}")
print(f"   Per-fold: {catboost_best['MAE_scores']}")
print("   Parameters:")
for param in catboost_param_grid.keys():
    print(f"      {param}: {catboost_best[param]}")

print("\n💡 Best LightGBM:")
print(f"   MAE: {lgbm_best['MAE_mean']:.4f} ± {lgbm_best['MAE_std']:.4f}")
print(f"   Per-fold: {lgbm_best['MAE_scores']}")
print("   Parameters:")
for param in lgbm_param_grid.keys():
    print(f"      {param}: {lgbm_best[param]}")

print("\n🚀 Best XGBoost:")
print(f"   MAE: {xgb_best['MAE_mean']:.4f} ± {xgb_best['MAE_std']:.4f}")
print(f"   Per-fold: {xgb_best['MAE_scores']}")
print("   Parameters:")
for param in xgb_param_grid.keys():
    print(f"      {param}: {xgb_best[param]}")

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
    ("CatBoost (tuned)", catboost_best["MAE_mean"], catboost_best["MAE_std"]),
    ("LightGBM (tuned)", lgbm_best["MAE_mean"], lgbm_best["MAE_std"]),
    ("XGBoost (tuned)", xgb_best["MAE_mean"], xgb_best["MAE_std"]),
    ("Ridge (baseline)", ridge_mae_mean, ridge_mae_std),
]

models_sorted = sorted(models, key=lambda x: x[1])

print("\n🏆 RANKING:")
for rank, (name, mae, std) in enumerate(models_sorted, 1):
    medal = {1: "🥇", 2: "🥈", 3: "🥉", 4: "4️⃣"}.get(rank, f"{rank}.")
    print(f"{medal} {name:<22} MAE: {mae:.4f} ± {std:.4f}")

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

catboost_results.head(10).to_csv(
    "experiments/results/catboost_valid_folds_top10.csv", index=False
)
lgbm_results.head(10).to_csv(
    "experiments/results/lgbm_valid_folds_top10.csv", index=False
)
xgb_results.head(10).to_csv(
    "experiments/results/xgb_valid_folds_top10.csv", index=False
)

print("\n✅ Saved top 10 configurations:")
print("   - experiments/results/catboost_valid_folds_top10.csv")
print("   - experiments/results/lgbm_valid_folds_top10.csv")
print("   - experiments/results/xgb_valid_folds_top10.csv")

print("\n" + "=" * 80)
print("🎯 PRODUCTION RECOMMENDATION")
print("=" * 80)

print(f"\n✅ Best Model: {winner_name}")
print(f"   Expected MAE: ~{winner_mae:.2f} points")
print(f"   Improvement: {improvement_pct:.1f}% better than Ridge")
print("\n📋 Optimal Hyperparameters:")

if "CatBoost" in winner_name:
    for param in catboost_param_grid.keys():
        print(f"   {param}: {catboost_best[param]}")
elif "LightGBM" in winner_name:
    for param in lgbm_param_grid.keys():
        print(f"   {param}: {lgbm_best[param]}")
else:  # XGBoost
    for param in xgb_param_grid.keys():
        print(f"   {param}: {xgb_best[param]}")

print("\n💡 Next Steps:")
print("   1. Use these hyperparameters in MLExpectedPointsService")
print("   2. Validate on holdout gameweeks (GW9+)")
print("   3. A/B test against current rule-based model")

print("\n" + "=" * 80)
