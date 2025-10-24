"""
ML Architecture Analysis: First Principles Approach
====================================================

Systematic analysis to determine which ML pipeline is best suited for FPL xP prediction
based on data characteristics, not just empirical performance.

Analysis Steps:
1. Target distribution analysis (skewness, zeros, outliers)
2. Feature-target relationships (linear vs non-linear)
3. Feature interaction importance
4. Model comparison with identical CV strategy
5. Error analysis by player type
6. Computational requirements

Usage:
    python experiments/ml_architecture_analysis.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, SGDRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fpl_team_picker.domain.services.ml_feature_engineering import FPLFeatureEngineer
from client import FPLDataClient


def analyze_target_distribution(y: pd.Series) -> dict:
    """Analyze characteristics of target variable."""
    print("\n" + "=" * 80)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("=" * 80)

    stats_dict = {
        "mean": y.mean(),
        "median": y.median(),
        "std": y.std(),
        "min": y.min(),
        "max": y.max(),
        "skewness": stats.skew(y),
        "kurtosis": stats.kurtosis(y),
        "pct_zeros": (y == 0).sum() / len(y) * 100,
        "pct_high_scorers": (y >= 10).sum() / len(y) * 100,
    }

    print("\nBasic Statistics:")
    print(f"  Mean: {stats_dict['mean']:.2f}")
    print(f"  Median: {stats_dict['median']:.2f}")
    print(f"  Std: {stats_dict['std']:.2f}")
    print(f"  Min: {stats_dict['min']:.2f}")
    print(f"  Max: {stats_dict['max']:.2f}")

    print("\nDistribution Shape:")
    print(
        f"  Skewness: {stats_dict['skewness']:.2f} {'(right-skewed)' if stats_dict['skewness'] > 0.5 else '(approximately normal)' if abs(stats_dict['skewness']) < 0.5 else '(left-skewed)'}"
    )
    print(
        f"  Kurtosis: {stats_dict['kurtosis']:.2f} {'(heavy tails)' if stats_dict['kurtosis'] > 1 else '(light tails)' if stats_dict['kurtosis'] < -1 else '(normal tails)'}"
    )

    print("\nSparsity:")
    print(f"  % Zeros: {stats_dict['pct_zeros']:.1f}%")
    print(f"  % High scorers (≥10): {stats_dict['pct_high_scorers']:.1f}%")

    # Implications for model choice
    print(f"\n{'IMPLICATIONS FOR MODEL CHOICE:'}")
    if stats_dict["skewness"] > 1.0:
        print("  ⚠️  Heavy right skew → Tree models may underpredict high values")
        print(
            "     Consider: Quantile regression, asymmetric loss, or sample weighting"
        )

    if stats_dict["pct_zeros"] > 20:
        print(f"  ⚠️  {stats_dict['pct_zeros']:.0f}% zeros → High sparsity")
        print("     Consider: Zero-inflated models or two-stage prediction")

    if stats_dict["max"] / stats_dict["mean"] > 5:
        print(
            f"  ⚠️  Large outliers (max/mean = {stats_dict['max'] / stats_dict['mean']:.1f})"
        )
        print("     Consider: Robust loss functions (Huber, quantile)")

    return stats_dict


def analyze_feature_relationships(
    X: pd.DataFrame, y: pd.Series, top_n: int = 10
) -> pd.DataFrame:
    """Analyze linear and non-linear feature-target relationships."""
    print("\n" + "=" * 80)
    print("FEATURE-TARGET RELATIONSHIP ANALYSIS")
    print("=" * 80)

    # Linear correlations
    correlations = []
    for col in X.columns:
        pearson_corr = X[col].corr(y)
        spearman_corr = X[col].corr(y, method="spearman")
        correlations.append(
            {
                "feature": col,
                "pearson": pearson_corr,
                "spearman": spearman_corr,
                "nonlinearity_score": abs(spearman_corr - pearson_corr),
            }
        )

    corr_df = pd.DataFrame(correlations).sort_values(
        "pearson", key=abs, ascending=False
    )

    print(f"\nTop {top_n} Most Predictive Features (by absolute Pearson correlation):")
    print(corr_df.head(top_n).to_string(index=False))

    # Assess linearity
    avg_nonlinearity = corr_df["nonlinearity_score"].mean()
    print("\nLinearity Assessment:")
    print(f"  Average |Spearman - Pearson|: {avg_nonlinearity:.3f}")

    if avg_nonlinearity < 0.05:
        print("  ✅ Strong evidence of LINEAR relationships")
        print(
            "     Recommendation: Linear models (Ridge, SGD, ElasticNet) should work well"
        )
    elif avg_nonlinearity < 0.15:
        print("  ⚠️  Moderate non-linearity detected")
        print("     Recommendation: Consider polynomial features or tree models")
    else:
        print("  ⚠️  Strong NON-LINEAR relationships")
        print("     Recommendation: Tree models or neural networks required")

    return corr_df


def test_feature_interactions(X: pd.DataFrame, y: pd.Series) -> dict:
    """Test if feature interactions are important using model comparison."""
    print("\n" + "=" * 80)
    print("FEATURE INTERACTION IMPORTANCE")
    print("=" * 80)

    # Linear model (no interactions)
    linear_model = Ridge(alpha=1.0)
    linear_scores = cross_val_score(
        linear_model, X, y, cv=3, scoring="neg_mean_absolute_error"
    )
    linear_mae = -linear_scores.mean()

    # Tree model (automatic interactions)
    tree_model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    tree_scores = cross_val_score(
        tree_model, X, y, cv=3, scoring="neg_mean_absolute_error"
    )
    tree_mae = -tree_scores.mean()

    improvement = (linear_mae - tree_mae) / linear_mae * 100

    print("\nModel Comparison (3-fold CV):")
    print(f"  Ridge (linear, no interactions): MAE = {linear_mae:.3f}")
    print(f"  RandomForest (interactions):     MAE = {tree_mae:.3f}")
    print(f"  Improvement from interactions:   {improvement:+.1f}%")

    if improvement > 10:
        print("\n  ✅ Feature interactions are IMPORTANT (>10% improvement)")
        print("     Recommendation: Use tree models or add polynomial features")
    elif improvement > 5:
        print("\n  ⚠️  Feature interactions provide MODERATE benefit (5-10%)")
        print(
            "     Recommendation: Consider model complexity vs interpretability tradeoff"
        )
    else:
        print("\n  ✅ Feature interactions provide MINIMAL benefit (<5%)")
        print("     Recommendation: Linear models sufficient; prefer simpler models")

    return {
        "linear_mae": linear_mae,
        "tree_mae": tree_mae,
        "interaction_importance": improvement,
    }


def compare_model_architectures(
    X: pd.DataFrame, y: pd.Series, cv_folds: list
) -> pd.DataFrame:
    """Systematic comparison of different model architectures with IDENTICAL CV."""
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE COMPARISON (Temporal CV)")
    print("=" * 80)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        "SGDRegressor": SGDRegressor(
            loss="squared_epsilon_insensitive",
            alpha=0.01,
            epsilon=0.1,
            max_iter=10000,
            random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=200,
            depth=4,
            learning_rate=0.1,
            l2_leaf_reg=5,
            random_seed=42,
            verbose=0,
        ),
    }

    results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...", end=" ")

        fold_maes = []
        fold_rmses = []
        fold_r2s = []

        for train_idx, test_idx in cv_folds:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_maes.append(mean_absolute_error(y_test, y_pred))
            fold_rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            fold_r2s.append(r2_score(y_test, y_pred))

        results.append(
            {
                "Model": model_name,
                "CV_MAE": np.mean(fold_maes),
                "CV_MAE_std": np.std(fold_maes),
                "CV_RMSE": np.mean(fold_rmses),
                "CV_R2": np.mean(fold_r2s),
            }
        )

        print(f"MAE = {np.mean(fold_maes):.3f} ± {np.std(fold_maes):.3f}")

    results_df = pd.DataFrame(results).sort_values("CV_MAE")

    print(f"\n{'MODEL RANKING (by CV MAE):'}")
    print(results_df.to_string(index=False))

    # Statistical significance
    best_mae = results_df["CV_MAE"].iloc[0]
    best_model = results_df["Model"].iloc[0]
    second_best_mae = results_df["CV_MAE"].iloc[1]
    second_best_model = results_df["Model"].iloc[1]

    print(f"\n{'WINNER:'} {best_model} (MAE = {best_mae:.3f})")
    print(
        f"  vs {second_best_model}: {((second_best_mae - best_mae) / best_mae * 100):.1f}% better"
    )

    return results_df


def analyze_prediction_errors(
    X: pd.DataFrame, y: pd.Series, model, cv_folds: list, player_data: pd.DataFrame
) -> dict:
    """Analyze prediction errors by player type."""
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS BY PLAYER TYPE")
    print("=" * 80)

    all_predictions = []
    all_actuals = []
    all_player_info = []

    for train_idx, test_idx in cv_folds:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
        all_player_info.extend(player_data.iloc[test_idx].to_dict("records"))

    error_df = pd.DataFrame(
        {
            "actual": all_actuals,
            "predicted": all_predictions,
            "error": np.array(all_predictions) - np.array(all_actuals),
            "abs_error": np.abs(np.array(all_predictions) - np.array(all_actuals)),
            "position": [p["position"] for p in all_player_info],
            "now_cost": [p["now_cost"] for p in all_player_info],
        }
    )

    # By position
    print("\nMAE by Position:")
    for position in ["GKP", "DEF", "MID", "FWD"]:
        pos_mae = error_df[error_df["position"] == position]["abs_error"].mean()
        pos_bias = error_df[error_df["position"] == position]["error"].mean()
        print(
            f"  {position}: MAE = {pos_mae:.3f}, Bias = {pos_bias:+.3f} {'(underpredict)' if pos_bias < 0 else '(overpredict)'}"
        )

    # By price tier
    print("\nMAE by Price Tier:")
    error_df["price_tier"] = pd.cut(
        error_df["now_cost"],
        bins=[0, 50, 70, 90, 200],
        labels=["Budget (<5.0)", "Mid (5.0-7.0)", "Premium (7.0-9.0)", "Elite (9.0+)"],
    )

    for tier in ["Budget (<5.0)", "Mid (5.0-7.0)", "Premium (7.0-9.0)", "Elite (9.0+)"]:
        tier_data = error_df[error_df["price_tier"] == tier]
        if len(tier_data) > 0:
            tier_mae = tier_data["abs_error"].mean()
            tier_bias = tier_data["error"].mean()
            print(
                f"  {tier}: MAE = {tier_mae:.3f}, Bias = {tier_bias:+.3f} {'(underpredict)' if tier_bias < 0 else '(overpredict)'}"
            )

    # Check for systematic bias on high scorers
    high_scorers = error_df[error_df["actual"] >= 10]
    if len(high_scorers) > 0:
        high_scorer_bias = high_scorers["error"].mean()
        print("\nHigh Scorers (actual ≥10 points):")
        print(
            f"  Bias: {high_scorer_bias:+.3f} {'⚠️ UNDERPREDICTING premium performances' if high_scorer_bias < -1 else '✅ Good calibration'}"
        )

    return error_df


def main():
    """Run comprehensive ML architecture analysis."""
    print("=" * 80)
    print("ML ARCHITECTURE ANALYSIS: FIRST PRINCIPLES")
    print("=" * 80)
    print("\nLoading data...")

    # Load data
    client = FPLDataClient()
    players = client.get_current_players()
    teams = client.get_current_teams()
    fixtures = client.get_fixtures_normalized()

    # Load historical gameweek data
    all_data = []
    for gw in range(1, 9):  # GW1-8
        try:
            gw_data = client.get_gameweek_performance(gw)
            gw_data["gameweek"] = gw
            all_data.append(gw_data)
        except Exception as e:
            print(f"Warning: Could not load GW{gw}: {e}")

    historical_data = pd.concat(all_data, ignore_index=True)

    # Merge with players to get position and price
    player_info = players[
        ["player_id", "web_name", "position", "price_gbp", "team_id"]
    ].copy()

    # Convert price from £ to pence (to match FPL API format)
    player_info["now_cost"] = (player_info["price_gbp"] * 10).astype(int)

    historical_data = historical_data.merge(
        player_info[["player_id", "position", "now_cost", "team_id"]],
        on="player_id",
        how="left",
    )

    # Filter to GW6+ (when 5GW rolling windows are valid) BEFORE feature engineering
    historical_data = historical_data[historical_data["gameweek"] >= 6].copy()

    print(f"Training data: {len(historical_data)} player-gameweeks from GW6-8")

    # Engineer features
    print("Engineering features...")
    engineer = FPLFeatureEngineer(
        fixtures_df=fixtures,
        teams_df=teams,
        team_strength=None,  # Will use defaults
    )

    # Keep metadata for analysis
    metadata_cols = ["player_id", "gameweek", "position", "now_cost", "total_points"]
    metadata = historical_data[metadata_cols].copy()

    # Engineer features (returns only feature columns)
    X = engineer.fit_transform(historical_data)
    y = metadata["total_points"]

    feature_cols = X.columns.tolist()

    print(f"\nDataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Gameweeks: {sorted(metadata['gameweek'].unique())}")

    # Create temporal CV folds (same as production)
    gws = sorted(metadata["gameweek"].unique())
    temporal_folds = []

    for i in range(len(gws) - 1):
        train_gws = gws[: i + 1]
        test_gw = gws[i + 1]

        train_mask = metadata["gameweek"].isin(train_gws)
        test_mask = metadata["gameweek"] == test_gw

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        temporal_folds.append((train_idx, test_idx))

    print(f"Cross-validation: {len(temporal_folds)} temporal folds")

    # Run analyses
    target_stats = analyze_target_distribution(y)

    _ = analyze_feature_relationships(X, y, top_n=15)

    interaction_results = test_feature_interactions(X, y)

    model_comparison = compare_model_architectures(X, y, temporal_folds)

    # Detailed error analysis for best model
    best_model_name = model_comparison["Model"].iloc[0]
    print(f"\n{'=' * 80}")
    print(f"DETAILED ERROR ANALYSIS: {best_model_name}")
    print(f"{'=' * 80}")

    if best_model_name == "SGDRegressor":
        best_model = SGDRegressor(
            loss="squared_epsilon_insensitive",
            alpha=0.01,
            epsilon=0.1,
            max_iter=10000,
            random_state=42,
        )
    elif best_model_name == "Ridge":
        best_model = Ridge(alpha=1.0)
    elif best_model_name == "CatBoost":
        best_model = CatBoostRegressor(
            iterations=200,
            depth=4,
            learning_rate=0.1,
            l2_leaf_reg=5,
            random_seed=42,
            verbose=0,
        )
    else:
        print(f"Error analysis not implemented for {best_model_name}")
        return

    _ = analyze_prediction_errors(
        X, y, best_model, temporal_folds, metadata[["position", "now_cost"]]
    )

    # Final recommendation
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)

    print(f"\n✅ BEST MODEL: {best_model_name}")
    print(
        f"   CV MAE: {model_comparison[model_comparison['Model'] == best_model_name]['CV_MAE'].iloc[0]:.3f}"
    )

    print(f"\n{'EVIDENCE SUMMARY:'}")
    print(
        f"  • Target distribution: {'Skewed' if target_stats['skewness'] > 1 else 'Normal'}, "
        f"{target_stats['pct_zeros']:.0f}% zeros"
    )
    print(
        f"  • Feature relationships: {'Non-linear' if interaction_results['interaction_importance'] > 10 else 'Mostly linear'}"
    )
    print(
        f"  • Interaction importance: {interaction_results['interaction_importance']:+.1f}% improvement"
    )

    print(f"\n{'DEPLOYMENT RECOMMENDATION:'}")
    if best_model_name in ["Ridge", "SGDRegressor", "ElasticNet"]:
        print(f"  ✅ Use {best_model_name} - fast, interpretable, performs best")
        print("  • Training time: Fast (~seconds)")
        print("  • Inference time: Very fast (~milliseconds)")
        print("  • Interpretability: High (linear coefficients)")
    else:
        print(f"  ✅ Use {best_model_name} - best predictive performance")
        print("  • Training time: Moderate (~minutes)")
        print("  • Inference time: Fast (~milliseconds)")
        print("  • Interpretability: Moderate (feature importance)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
