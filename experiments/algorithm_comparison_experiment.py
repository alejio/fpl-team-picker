#!/usr/bin/env python3
"""
Algorithm Comparison Experiment for ML Expected Points (xP) Prediction

Tests multiple regression algorithms with proper temporal cross-validation to identify
the best performer for FPL expected points prediction.

Algorithms tested:
1. Ridge Regression (baseline)
2. ElasticNet (L1+L2 regularization)
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. LightGBM

Cross-validation strategy:
- Temporal walk-forward CV: Train on GW6 → Test on GW7 (realistic)
- Player-based GroupKFold: 5 folds (optimistic, for comparison)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from client import FPLDataClient
from fpl_team_picker.domain.services.ml_feature_engineering import FPLFeatureEngineer

print("=" * 80)
print("🤖 ALGORITHM COMPARISON EXPERIMENT")
print("=" * 80)

# Initialize data client
client = FPLDataClient()

# Load data for GW6 and GW7
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

# Preserve target and metadata before feature engineering
target = cv_data["total_points"].copy()
player_ids = cv_data["player_id"].copy()
gameweeks = cv_data["gameweek"].copy()

# Feature engineering
print("\n🔧 Engineering features...")
feature_engineer = FPLFeatureEngineer()
cv_data_enriched = feature_engineer.fit_transform(cv_data)

# Add back target, gameweek, and player_id
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
player_groups = cv_data_enriched["player_id"].values
gameweek_values = cv_data_enriched["gameweek"].values

print("\n📈 Dataset Summary:")
print(f"   Total samples: {len(X):,}")
print(f"   Unique players: {len(np.unique(player_groups)):,}")
print(f"   Gameweeks: {sorted(cv_data_enriched['gameweek'].unique())}")
print(f"   Features: {len(feature_cols)}")

# ============================================================================
# Setup Cross-Validation Strategies
# ============================================================================

print("\n" + "=" * 80)
print("📊 CROSS-VALIDATION SETUP")
print("=" * 80)

# 1. Temporal Walk-Forward CV (realistic)
gws = sorted(cv_data_enriched["gameweek"].unique())
temporal_splits = []

if len(gws) >= 2:
    # Reset index to get positional indices
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
            f"Temporal Fold {i + 1}: Train on GW{min(train_gws)}-{max(train_gws)} ({len(train_idx)} samples) → Test on GW{test_gw} ({len(test_idx)} samples)"
        )

    print(f"\n✅ Temporal CV: {len(temporal_splits)} fold(s)")
else:
    print("⚠️ Need at least 2 gameweeks for temporal CV")
    temporal_splits = None

# 2. Player-Based GroupKFold (optimistic)
n_folds = 5
player_gkf = GroupKFold(n_splits=n_folds)
print(f"✅ Player-Based CV: {n_folds} folds (GroupKFold)")

# ============================================================================
# Define Models
# ============================================================================

print("\n" + "=" * 80)
print("🤖 MODEL DEFINITIONS")
print("=" * 80)

models = {
    "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    "ElasticNet": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000)),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
    "Gradient Boosting": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                ),
            ),
        ]
    ),
    "XGBoost": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
    "LightGBM": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LGBMRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    min_child_samples=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    ),
}

for name, model in models.items():
    print(f"✅ {name}: {model['model'].__class__.__name__}")

# ============================================================================
# Run Experiments
# ============================================================================

print("\n" + "=" * 80)
print("🧪 RUNNING EXPERIMENTS")
print("=" * 80)

results = []

# Test each model with both CV strategies
for model_name, model_pipeline in models.items():
    print(f"\n{'─' * 80}")
    print(f"Testing: {model_name}")
    print(f"{'─' * 80}")

    # 1. Player-Based CV (optimistic)
    print(f"   Player-Based CV ({n_folds} folds)...", end=" ", flush=True)
    player_mae_scores = -cross_val_score(
        model_pipeline,
        X,
        y,
        groups=player_groups,
        cv=player_gkf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    player_mae_mean = player_mae_scores.mean()
    player_mae_std = player_mae_scores.std()
    print(f"MAE = {player_mae_mean:.3f} ± {player_mae_std:.3f}")

    # 2. Temporal CV (realistic)
    if temporal_splits:
        print(f"   Temporal CV ({len(temporal_splits)} fold)...", end=" ", flush=True)
        temporal_mae_scores = -cross_val_score(
            model_pipeline,
            X,
            y,
            cv=temporal_splits,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        temporal_mae_mean = temporal_mae_scores.mean()
        temporal_mae_std = temporal_mae_scores.std()
        print(f"MAE = {temporal_mae_mean:.3f} ± {temporal_mae_std:.3f}")
    else:
        temporal_mae_mean = None
        temporal_mae_std = None
        print("   Temporal CV: SKIPPED (not enough gameweeks)")

    # Store results
    results.append(
        {
            "Model": model_name,
            "Player-Based MAE": player_mae_mean,
            "Player-Based Std": player_mae_std,
            "Temporal MAE": temporal_mae_mean,
            "Temporal Std": temporal_mae_std,
        }
    )

# ============================================================================
# Results Summary
# ============================================================================

print("\n" + "=" * 80)
print("📊 RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)

# Sort by temporal MAE (realistic metric)
if temporal_splits:
    results_df = results_df.sort_values("Temporal MAE")
    print("\n🏆 RANKING (by Temporal CV - most realistic):")
    print("─" * 80)
    for idx, row in results_df.iterrows():
        print(
            f"{idx + 1}. {row['Model']:<20} Temporal MAE: {row['Temporal MAE']:.3f} ± {row['Temporal Std']:.3f}"
        )
else:
    results_df = results_df.sort_values("Player-Based MAE")
    print("\n🏆 RANKING (by Player-Based CV - only metric available):")
    print("─" * 80)
    for idx, row in results_df.iterrows():
        print(
            f"{idx + 1}. {row['Model']:<20} Player-Based MAE: {row['Player-Based MAE']:.3f} ± {row['Player-Based Std']:.3f}"
        )

print("\n" + "=" * 80)
print("📋 FULL COMPARISON TABLE")
print("=" * 80)
print(results_df.to_string(index=False))

# ============================================================================
# Key Insights
# ============================================================================

print("\n" + "=" * 80)
print("💡 KEY INSIGHTS")
print("=" * 80)

if temporal_splits:
    best_temporal = results_df.iloc[0]
    print(f"\n✅ Best Model (Temporal CV): {best_temporal['Model']}")
    print(
        f"   MAE: {best_temporal['Temporal MAE']:.3f} ± {best_temporal['Temporal Std']:.3f}"
    )

    # Compare to Ridge baseline
    ridge_row = results_df[results_df["Model"] == "Ridge"].iloc[0]
    if best_temporal["Model"] != "Ridge":
        improvement = ridge_row["Temporal MAE"] - best_temporal["Temporal MAE"]
        improvement_pct = (improvement / ridge_row["Temporal MAE"]) * 100
        print(
            f"\n   Improvement over Ridge: {improvement:.3f} ({improvement_pct:+.1f}%)"
        )

    # Check Player-Based vs Temporal gap
    print("\n📊 Optimism Gap (Player-Based vs Temporal CV):")
    for _, row in results_df.iterrows():
        if row["Temporal MAE"] is not None:
            gap = row["Temporal MAE"] - row["Player-Based MAE"]
            gap_pct = (gap / row["Player-Based MAE"]) * 100
            print(f"   {row['Model']:<20} Gap: {gap:+.3f} ({gap_pct:+.1f}%)")

print("\n" + "=" * 80)
print("🎯 RECOMMENDATIONS")
print("=" * 80)

if temporal_splits:
    print(
        f"\n1. Use {best_temporal['Model']} for production (best temporal CV performance)"
    )
    print(
        f"2. Expected MAE on next gameweek: ~{best_temporal['Temporal MAE']:.2f} points"
    )
    print(
        f"3. Player-based CV is {results_df['Temporal MAE'].mean() / results_df['Player-Based MAE'].mean() - 1:.1%} too optimistic"
    )
else:
    best_player = results_df.iloc[0]
    print(f"\n1. Best model (Player-Based CV): {best_player['Model']}")
    print(
        f"2. MAE: {best_player['Player-Based MAE']:.3f} ± {best_player['Player-Based Std']:.3f}"
    )
    print("3. ⚠️ Need more gameweeks for realistic temporal CV validation")

print("\n" + "=" * 80)
