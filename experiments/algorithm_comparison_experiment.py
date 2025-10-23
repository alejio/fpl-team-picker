#!/usr/bin/env python3
"""
Algorithm Comparison Experiment for ML Expected Points (xP) Prediction

Tests multiple regression algorithms with proper temporal cross-validation to identify
the best performer for FPL expected points prediction.

Algorithms tested (Tier 1 recommendations):
1. LightGBM ⭐⭐⭐⭐⭐ (best for categorical/ordinal features like price_band)
2. XGBoost ⭐⭐⭐⭐⭐ (proven track record for structured data)
3. CatBoost ⭐⭐⭐⭐ (best categorical handling, overfitting protection)
4. HistGradientBoosting ⭐⭐⭐⭐ (native sklearn, fast, categorical support)
5. Random Forest ⭐⭐⭐⭐ (solid baseline)
6. Gradient Boosting ⭐⭐⭐ (traditional sklearn baseline)
7. Ridge Regression ⭐⭐ (linear baseline for comparison)

Cross-validation strategy:
- Temporal walk-forward CV: Train on GW1-N → Test on GW N+1 (realistic)
- Player-based GroupKFold: 5 folds (optimistic, for comparison)

Training data: GW1-8 (matches TPOT pipeline optimizer setup)
Feature set: 65 features including new price_band ordinal feature
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from client import FPLDataClient
from fpl_team_picker.domain.services.ml_feature_engineering import FPLFeatureEngineer
from fpl_team_picker.domain.services.ml_pipeline_factory import (
    get_team_strength_ratings,
)

print("=" * 80)
print("🤖 ALGORITHM COMPARISON EXPERIMENT")
print("=" * 80)

# Initialize data client
client = FPLDataClient()

# Load data for GW1-8 (matching TPOT setup)
print("\n📊 Loading training data (GW1-8)...")
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

# Combine into training dataset
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

# Enrich with position data (required by FPLFeatureEngineer)
cv_data = cv_data.merge(
    players_data[["player_id", "position"]],
    on="player_id",
    how="left",
)

# Convert value to price (gameweek_performance has 'value', we need 'price' for analysis)
# value is in tenths (45 = £4.5M), price is in pounds
if "value" in cv_data.columns and "price" not in cv_data.columns:
    cv_data["price"] = cv_data["value"] / 10.0

# Preserve target and metadata before feature engineering
target = cv_data["total_points"].copy()
player_ids = cv_data["player_id"].copy()
gameweeks = cv_data["gameweek"].copy()
prices = cv_data["price"].copy()

# Feature engineering (with fixtures and teams context)
print("\n🔧 Engineering features...")
team_strength = get_team_strength_ratings()
feature_engineer = FPLFeatureEngineer(
    fixtures_df=fixtures_df,
    teams_df=teams_df,
    team_strength=team_strength,
)
cv_data_enriched = feature_engineer.fit_transform(cv_data, target)

# Add back target, gameweek, player_id, and price
cv_data_enriched["total_points"] = target.values
cv_data_enriched["player_id"] = player_ids.values
cv_data_enriched["gameweek"] = gameweeks.values
cv_data_enriched["price"] = prices.values

# Get feature columns (price is already in features as engineered feature)
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
print(f"   ✅ Features: {len(feature_cols)} (including new price_band feature)")

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
    # Baseline
    "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    # Tier 1 - Top Recommendations (Gradient Boosting variants)
    "LightGBM": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LGBMRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
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
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    ),
    "CatBoost": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                CatBoostRegressor(
                    iterations=200,
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=3,
                    random_seed=42,
                    verbose=0,
                    thread_count=-1,
                ),
            ),
        ]
    ),
    "HistGradientBoosting": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    max_depth=6,
                    learning_rate=0.05,
                    min_samples_leaf=20,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
        ]
    ),
    # Tier 2 - Solid alternatives
    "Random Forest": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features="sqrt",
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
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    subsample=0.8,
                    random_state=42,
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

# ============================================================================
# Elite Player Performance Analysis (addressing underestimation issue)
# ============================================================================

print("\n" + "=" * 80)
print("💎 ELITE PLAYER PERFORMANCE ANALYSIS")
print("=" * 80)
print("\nAnalyzing prediction accuracy by price band...")
print("(Testing if gradient boosting fixes elite player underestimation)\n")

# Train best models on full dataset and analyze predictions by price band
if temporal_splits:
    best_model_name = best_temporal["Model"]
    best_model = models[best_model_name]

    # Train on full dataset
    best_model.fit(X, y)
    predictions = best_model.predict(X)

    # Add predictions to enriched data
    cv_data_enriched["predicted_points"] = predictions

    # Define price bands
    cv_data_enriched["price_band_label"] = pd.cut(
        cv_data_enriched["price"],
        bins=[0, 5.0, 7.0, 9.0, float("inf")],
        labels=["Budget (<£5M)", "Mid (£5-7M)", "Premium (£7-9M)", "Elite (£9M+)"],
    )

    # Calculate MAE by price band
    print(f"🏆 Best Model: {best_model_name}")
    print("-" * 80)
    print(
        f"{'Price Band':<20} {'Count':>8} {'Avg Actual':>12} {'Avg Predicted':>15} {'MAE':>10} {'Bias':>10}"
    )
    print("-" * 80)

    for band in ["Budget (<£5M)", "Mid (£5-7M)", "Premium (£7-9M)", "Elite (£9M+)"]:
        band_data = cv_data_enriched[cv_data_enriched["price_band_label"] == band]
        if len(band_data) > 0:
            count = len(band_data)
            avg_actual = band_data["total_points"].mean()
            avg_predicted = band_data["predicted_points"].mean()
            mae = np.abs(
                band_data["total_points"] - band_data["predicted_points"]
            ).mean()
            bias = avg_predicted - avg_actual  # Negative = underestimation

            bias_indicator = "⚠️ " if bias < -0.5 else "✅ "
            print(
                f"{band:<20} {count:>8,} {avg_actual:>12.2f} {avg_predicted:>15.2f} {mae:>10.3f} {bias_indicator}{bias:>9.2f}"
            )

    print("-" * 80)
    print("\n💡 Interpretation:")
    print("   - Bias close to 0: Model is well-calibrated for that price band")
    print("   - Negative bias: Model underestimates (your original problem)")
    print("   - Positive bias: Model overestimates")
    print("\n   ✅ Goal: All bias values close to 0, especially for Elite players")

print("\n" + "=" * 80)
