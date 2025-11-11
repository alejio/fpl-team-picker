#!/usr/bin/env python3
"""
Feature Importance Analysis for FPL ML Expected Points (xP) Prediction

Replicates the feature engineering and training pipeline from TPOT optimizer to
identify the most important features for FPL xP prediction.

Methods used:
1. Random Forest feature importance (MDI - Mean Decrease in Impurity)
2. Permutation importance (model-agnostic, tests real predictive power)
3. SHAP values (explains individual predictions and global feature impact)
4. Correlation analysis (identifies redundant features)

Provides actionable insights for feature selection and model improvement.

Usage:
    python experiments/feature_importance_analysis.py --start-gw 1 --end-gw 9
"""

from types import SimpleNamespace
from typing import Any
import typer
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import BaseCrossValidator

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: uv pip install shap")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client import FPLDataClient  # noqa: E402
from fpl_team_picker.domain.services.ml_feature_engineering import (  # noqa: E402
    FPLFeatureEngineer,
    calculate_per_gameweek_team_strength,
)


class TemporalCVSplitter(BaseCrossValidator):
    """Custom CV splitter for temporal walk-forward validation."""

    def __init__(self, splits):
        self.splits = splits

    def split(self, X, y=None, groups=None):
        for train_idx, test_idx in self.splits:
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.splits)


app = typer.Typer(help="Feature importance analysis for FPL xP prediction")


def load_historical_data(start_gw: int, end_gw: int):
    """
    Load historical gameweek performance data and enhanced data sources.

    Replicates TPOT optimizer data loading pipeline.
    """
    client = FPLDataClient()
    historical_data = []

    print(f"\n📥 Loading historical data (GW{start_gw} to GW{end_gw})...")

    for gw in range(start_gw, end_gw + 1):
        gw_performance = client.get_gameweek_performance(gw)
        if not gw_performance.empty:
            gw_performance["gameweek"] = gw
            historical_data.append(gw_performance)
            print(f"   ✅ GW{gw}: {len(gw_performance)} players")
        else:
            print(f"   ⚠️  GW{gw}: No data available")

    if not historical_data:
        raise ValueError("No historical data loaded. Check gameweek range.")

    historical_df = pd.concat(historical_data, ignore_index=True)

    # Enrich with position data
    print("\n📊 Enriching with position data...")
    players_data = client.get_current_players()
    if (
        "position" not in players_data.columns
        or "player_id" not in players_data.columns
    ):
        raise ValueError(
            f"Position column not found in current_players data. "
            f"Available columns: {list(players_data.columns)}"
        )

    historical_df = historical_df.merge(
        players_data[["player_id", "position"]], on="player_id", how="left"
    )

    missing_count = historical_df["position"].isna().sum()
    if missing_count > 0:
        print(f"   ⚠️  Warning: {missing_count} records missing position data")

    # Load fixtures and teams
    print("\n🏟️  Loading fixtures and teams...")
    fixtures_df = client.get_fixtures_normalized()
    teams_df = client.get_current_teams()

    print(f"   ✅ Fixtures: {len(fixtures_df)} | Teams: {len(teams_df)}")

    # Load enhanced data sources (Issue #37)
    print("\n📊 Loading enhanced data sources...")
    ownership_trends_df = client.get_derived_ownership_trends()
    value_analysis_df = client.get_derived_value_analysis()
    fixture_difficulty_df = client.get_derived_fixture_difficulty()

    print(f"   ✅ Ownership trends: {len(ownership_trends_df)} records")
    print(f"   ✅ Value analysis: {len(value_analysis_df)} records")
    print(f"   ✅ Fixture difficulty: {len(fixture_difficulty_df)} records")

    # Load betting odds features (Issue #38)
    print("\n🎲 Loading betting odds features...")
    try:
        betting_features_df = client.get_derived_betting_features()
        print(f"   ✅ Betting features: {len(betting_features_df)} records")
    except (AttributeError, Exception) as e:
        print(f"   ⚠️  Betting features unavailable: {e}")
        print("   ℹ️  Continuing with neutral defaults (features will be 0/neutral)")
        betting_features_df = pd.DataFrame()

    print(f"\n✅ Total records: {len(historical_df):,}")
    print(f"   Unique players: {historical_df['player_id'].nunique():,}")

    return (
        historical_df,
        fixtures_df,
        teams_df,
        ownership_trends_df,
        value_analysis_df,
        fixture_difficulty_df,
        betting_features_df,
    )


def engineer_features(
    historical_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    ownership_trends_df: pd.DataFrame,
    value_analysis_df: pd.DataFrame,
    fixture_difficulty_df: pd.DataFrame,
    betting_features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Engineer features using production FPLFeatureEngineer.

    Replicates TPOT optimizer feature engineering pipeline.
    """
    print(
        "\n🔧 Engineering features (production FPLFeatureEngineer with 117 features)..."
    )

    # Calculate per-gameweek team strength (no data leakage)
    print("   📊 Calculating per-gameweek team strength (leak-free)...")

    start_gw = 6  # First trainable gameweek (needs GW1-5 for rolling features)
    end_gw = historical_df["gameweek"].max()
    team_strength = calculate_per_gameweek_team_strength(
        start_gw=start_gw,
        end_gw=end_gw,
        teams_df=teams_df,
    )
    print(f"   ✅ Team strength calculated for GW{start_gw}-{end_gw}")

    # Load per-gameweek set-piece/penalty taker data
    raw_players_df = None
    try:
        if hasattr(client, "get_players_set_piece_orders"):
            raw_players_df = client.get_players_set_piece_orders()
            print(f"   ✅ Loaded per-gameweek penalty data: {len(raw_players_df)} rows")
        else:
            raw_players_df = client.get_raw_players_bootstrap()
            print(f"   ✅ Loaded bootstrap penalty data: {len(raw_players_df)} rows")

        # Verify required columns exist
        required_cols = [
            "penalties_order",
            "corners_and_indirect_freekicks_order",
            "direct_freekicks_order",
        ]
        missing_cols = [
            col for col in required_cols if col not in raw_players_df.columns
        ]
        if missing_cols:
            print(f"   ⚠️  Missing penalty columns: {missing_cols}")
            raw_players_df = None
        else:
            print(f"   ✅ All penalty columns present: {required_cols}")

    except Exception as e:
        print(f"   ❌ Error loading penalty data: {e}")
        raw_players_df = None

    # Initialize production feature engineer with enhanced data sources
    feature_engineer = FPLFeatureEngineer(
        fixtures_df=fixtures_df if not fixtures_df.empty else None,
        teams_df=teams_df if not teams_df.empty else None,
        team_strength=team_strength if team_strength else None,
        ownership_trends_df=ownership_trends_df
        if not ownership_trends_df.empty
        else None,
        value_analysis_df=value_analysis_df if not value_analysis_df.empty else None,
        fixture_difficulty_df=fixture_difficulty_df
        if not fixture_difficulty_df.empty
        else None,
        raw_players_df=raw_players_df
        if raw_players_df is not None and not raw_players_df.empty
        else None,
        betting_features_df=betting_features_df
        if not betting_features_df.empty
        else None,
    )

    # Transform historical data
    features_df = feature_engineer.fit_transform(
        historical_df, historical_df["total_points"]
    )

    # Add back metadata for analysis
    historical_df_sorted = historical_df.sort_values(
        ["player_id", "gameweek"]
    ).reset_index(drop=True)
    features_df["player_id"] = historical_df_sorted["player_id"].values
    features_df["gameweek"] = historical_df_sorted["gameweek"].values
    features_df["total_points"] = historical_df_sorted["total_points"].values

    # Add position metadata
    if "position" not in historical_df_sorted.columns:
        raise ValueError(
            "Position column missing from historical data after merge. "
            "Check that get_current_players() returns position data."
        )

    if historical_df_sorted["position"].isna().any():
        missing_count = historical_df_sorted["position"].isna().sum()
        raise ValueError(
            f"Position data missing for {missing_count} records. "
            "Cannot proceed with incomplete position data. "
            "Check data quality in get_current_players()."
        )

    features_df["position"] = historical_df_sorted["position"].values

    # Get production feature names
    production_feature_cols = list(feature_engineer.get_feature_names_out())

    print(f"   ✅ Created {len(production_feature_cols)} features")
    print(f"   Total samples: {len(features_df):,}")

    return features_df, production_feature_cols


def create_temporal_cv_splits(
    features_df: pd.DataFrame, max_folds: int = None
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """
    Create temporal walk-forward cross-validation splits.

    Replicates TPOT optimizer CV strategy.
    """
    print("\n📊 Creating temporal CV splits (walk-forward validation)...")

    # Filter to GW6+ (need 5 preceding GWs for features)
    cv_data = features_df[features_df["gameweek"] >= 6].copy().reset_index(drop=True)

    cv_gws = sorted(cv_data["gameweek"].unique())

    if len(cv_gws) < 2:
        raise ValueError(
            f"Need at least 2 gameweeks for temporal CV. Found: {cv_gws}. "
            "Ensure end_gw >= 7 (GW6 train, GW7 test minimum)."
        )

    cv_splits = []

    # Limit folds if requested
    n_folds = len(cv_gws) - 1
    if max_folds is not None:
        n_folds = min(n_folds, max_folds)

    for i in range(n_folds):
        train_gws = cv_gws[: i + 1]  # GW6 to GW(6+i)
        test_gw = cv_gws[i + 1]  # GW(6+i+1)

        # Get positional indices
        train_mask = cv_data["gameweek"].isin(train_gws)
        test_mask = cv_data["gameweek"] == test_gw

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        cv_splits.append((train_idx, test_idx))
        print(
            f"   Fold {i + 1}: Train on GW{min(train_gws)}-{max(train_gws)} "
            f"({len(train_idx)} samples) → Test on GW{test_gw} ({len(test_idx)} samples)"
        )

    print(f"\n✅ Created {len(cv_splits)} temporal CV folds")

    return cv_splits, cv_data


def train_model(X: pd.DataFrame, y: pd.Series, args: Any):
    """Train Random Forest model for feature importance analysis."""
    print("\n🌲 Training Random Forest model...")
    print(f"   n_estimators: {args.n_estimators}")
    print(f"   max_depth: {args.max_depth}")
    print(f"   random_state: {args.random_seed}")

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=args.random_seed,
        n_jobs=-1,
        verbose=1,
    )

    model.fit(X, y)
    print("   ✅ Model trained")

    # Calculate performance metrics
    train_score = model.score(X, y)
    predictions = model.predict(X)
    mae = np.abs(predictions - y).mean()
    rmse = np.sqrt(((predictions - y) ** 2).mean())

    print("\n📊 Training Performance:")
    print(f"   R² Score: {train_score:.3f}")
    print(f"   MAE: {mae:.3f} points")
    print(f"   RMSE: {rmse:.3f} points")

    return model


def analyze_mdi_importance(
    model: RandomForestRegressor, feature_names: list[str], args: Any
) -> pd.DataFrame:
    """
    Analyze Mean Decrease in Impurity (MDI) feature importance.

    MDI: Measures how much each feature contributes to decreasing impurity
    (variance for regression) across all trees in the forest.
    """
    print("\n📊 Analyzing MDI Feature Importance...")

    # Get feature importances from trained model
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    # Create DataFrame
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
            "std": std,
        }
    ).sort_values("importance", ascending=False)

    # Display top features
    print(f"\n🏆 Top {args.top_n} Features (MDI):")
    print("-" * 80)
    for idx, row in importance_df.head(args.top_n).iterrows():
        print(f"{row['feature']:<50} {row['importance']:.4f} ± {row['std']:.4f}")

    return importance_df


def analyze_permutation_importance(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    args: Any,
) -> pd.DataFrame:
    """
    Analyze permutation importance.

    Permutation importance: Model-agnostic method that measures how much
    performance drops when a feature's values are randomly shuffled.
    More reliable than MDI for features with high cardinality.
    """
    print("\n🔀 Analyzing Permutation Importance...")
    print("   (This may take a few minutes...)")

    # Calculate permutation importance
    perm_importance = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=args.random_seed,
        n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )

    # Create DataFrame (convert to positive MAE)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": -perm_importance.importances_mean,  # Negative MAE -> positive
            "std": perm_importance.importances_std,
        }
    ).sort_values("importance", ascending=False)

    # Display top features
    print(f"\n🏆 Top {args.top_n} Features (Permutation):")
    print("-" * 80)
    for idx, row in importance_df.head(args.top_n).iterrows():
        print(f"{row['feature']:<50} {row['importance']:.4f} ± {row['std']:.4f}")

    return importance_df


def analyze_shap_importance(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    feature_names: list[str],
    args: Any,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Analyze SHAP (SHapley Additive exPlanations) values.

    SHAP: Explains model predictions by computing the contribution of each
    feature to the prediction. Based on game theory (Shapley values).
    """
    if not SHAP_AVAILABLE:
        print("\n⚠️  SHAP analysis skipped (shap not installed)")
        print("   Install with: uv pip install shap")
        return None, None, None

    print("\n🎯 Analyzing SHAP Values...")
    print("   (This may take several minutes...)")

    # Sample data for SHAP analysis (use 1000 samples for speed)
    sample_size = min(1000, len(X))
    X_sample = X.sample(n=sample_size, random_state=args.random_seed)

    print(f"   Using {sample_size} samples for SHAP analysis")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create DataFrame
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": mean_abs_shap,
        }
    ).sort_values("importance", ascending=False)

    # Display top features
    print(f"\n🏆 Top {args.top_n} Features (SHAP):")
    print("-" * 80)
    for idx, row in importance_df.head(args.top_n).iterrows():
        print(f"{row['feature']:<50} {row['importance']:.4f}")

    return importance_df, shap_values, X_sample


def analyze_correlations(
    X: pd.DataFrame, feature_names: list[str], top_features: list[str]
) -> pd.DataFrame:
    """
    Analyze correlations between top features to identify redundancy.
    """
    print("\n🔗 Analyzing Feature Correlations (Top Features)...")

    # Calculate correlation matrix for top features
    top_X = X[top_features]
    corr_matrix = top_X.corr().abs()

    # Find highly correlated pairs (> 0.8)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.8:
                high_corr_pairs.append(
                    {
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j],
                    }
                )

    if high_corr_pairs:
        print(f"\n⚠️  Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8):")
        print("-" * 80)
        for pair in high_corr_pairs:
            print(
                f"{pair['feature_1']:<30} <-> {pair['feature_2']:<30} r={pair['correlation']:.3f}"
            )
    else:
        print("   ✅ No highly correlated pairs found (all |r| <= 0.8)")

    return pd.DataFrame(high_corr_pairs)


def create_visualizations(
    mdi_importance: pd.DataFrame,
    perm_importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    args: Any,
    output_dir: Path,
):
    """Create visualizations for feature importance analysis."""
    print(f"\n📈 Creating visualizations (saving to {output_dir})...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 10)

    # 1. MDI Importance Bar Plot
    print("   Creating MDI importance plot...")
    fig, ax = plt.subplots(figsize=(14, 10))
    top_mdi = mdi_importance.head(args.top_n)
    ax.barh(range(len(top_mdi)), top_mdi["importance"], xerr=top_mdi["std"])
    ax.set_yticks(range(len(top_mdi)))
    ax.set_yticklabels(top_mdi["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in Impurity (MDI)")
    ax.set_title(
        f"Top {args.top_n} Features by MDI Importance", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "mdi_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Permutation Importance Bar Plot
    print("   Creating permutation importance plot...")
    fig, ax = plt.subplots(figsize=(14, 10))
    top_perm = perm_importance.head(args.top_n)
    ax.barh(range(len(top_perm)), top_perm["importance"], xerr=top_perm["std"])
    ax.set_yticks(range(len(top_perm)))
    ax.set_yticklabels(top_perm["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (MAE Increase)")
    ax.set_title(
        f"Top {args.top_n} Features by Permutation Importance",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "permutation_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. SHAP Summary Plot (if SHAP analysis was run)
    if SHAP_AVAILABLE and shap_importance is not None and shap_values is not None:
        print("   Creating SHAP summary plot...")
        fig, ax = plt.subplots(figsize=(14, 10))
        shap.summary_plot(
            shap_values,
            X_sample,
            max_display=args.top_n,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 4. SHAP Bar Plot
        print("   Creating SHAP bar plot...")
        fig, ax = plt.subplots(figsize=(14, 10))
        shap.summary_plot(
            shap_values,
            X_sample,
            plot_type="bar",
            max_display=args.top_n,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 5. Comparison Plot (MDI vs Permutation)
    print("   Creating comparison plot...")
    fig, ax = plt.subplots(figsize=(14, 10))

    # Merge top features from both methods
    comparison_df = pd.merge(
        mdi_importance.head(args.top_n)[["feature", "importance"]].rename(
            columns={"importance": "mdi"}
        ),
        perm_importance.head(args.top_n)[["feature", "importance"]].rename(
            columns={"importance": "permutation"}
        ),
        on="feature",
        how="outer",
    ).fillna(0)

    comparison_df = comparison_df.sort_values("mdi", ascending=False)

    x = np.arange(len(comparison_df))
    width = 0.35

    ax.barh(x - width / 2, comparison_df["mdi"], width, label="MDI")
    ax.barh(x + width / 2, comparison_df["permutation"], width, label="Permutation")

    ax.set_yticks(x)
    ax.set_yticklabels(comparison_df["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(
        f"Feature Importance Comparison: MDI vs Permutation (Top {args.top_n})",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   ✅ All plots saved to {output_dir}")


def save_results(
    mdi_importance: pd.DataFrame,
    perm_importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
    corr_pairs: pd.DataFrame,
    output_dir: Path,
):
    """Save results to CSV files."""
    print(f"\n💾 Saving results to CSV (saving to {output_dir})...")

    output_dir.mkdir(parents=True, exist_ok=True)

    mdi_importance.to_csv(output_dir / "mdi_importance.csv", index=False)
    print(f"   ✅ MDI importance: {output_dir / 'mdi_importance.csv'}")

    perm_importance.to_csv(output_dir / "permutation_importance.csv", index=False)
    print(f"   ✅ Permutation importance: {output_dir / 'permutation_importance.csv'}")

    if shap_importance is not None:
        shap_importance.to_csv(output_dir / "shap_importance.csv", index=False)
        print(f"   ✅ SHAP importance: {output_dir / 'shap_importance.csv'}")

    if not corr_pairs.empty:
        corr_pairs.to_csv(output_dir / "high_correlations.csv", index=False)
        print(f"   ✅ High correlations: {output_dir / 'high_correlations.csv'}")


def generate_recommendations(
    mdi_importance: pd.DataFrame,
    perm_importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
    args: Any,
):
    """Generate actionable recommendations based on feature importance analysis."""
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    # Identify consensus top features (appear in top N for all methods)
    top_n_features = 20

    mdi_top = set(mdi_importance.head(top_n_features)["feature"])
    perm_top = set(perm_importance.head(top_n_features)["feature"])

    if shap_importance is not None:
        shap_top = set(shap_importance.head(top_n_features)["feature"])
        consensus = mdi_top & perm_top & shap_top
        print(
            f"\n✅ Consensus Top Features (appear in all 3 methods' top {top_n_features}):"
        )
    else:
        consensus = mdi_top & perm_top
        print(
            f"\n✅ Consensus Top Features (appear in both methods' top {top_n_features}):"
        )

    if consensus:
        print("-" * 80)
        for feature in sorted(consensus):
            print(f"   • {feature}")
        print(f"\nTotal consensus features: {len(consensus)}")
    else:
        print("   ⚠️  No consensus features found. Methods disagree significantly.")

    # Identify features that appear high in permutation but low in MDI
    # (suggests high predictive power but not used much by trees)
    print("\n🔍 Underutilized Features (high permutation, low MDI):")
    print("-" * 80)

    # Get ranks
    mdi_importance["mdi_rank"] = range(1, len(mdi_importance) + 1)
    perm_importance["perm_rank"] = range(1, len(perm_importance) + 1)

    merged = pd.merge(
        mdi_importance[["feature", "mdi_rank"]],
        perm_importance[["feature", "perm_rank"]],
        on="feature",
    )

    # Find features with high permutation rank but low MDI rank
    merged["rank_diff"] = merged["mdi_rank"] - merged["perm_rank"]
    underutilized = merged[merged["rank_diff"] > 30].sort_values(
        "rank_diff", ascending=False
    )

    if not underutilized.empty:
        for _, row in underutilized.head(10).iterrows():
            print(
                f"   • {row['feature']:<50} "
                f"(Perm Rank: {row['perm_rank']}, MDI Rank: {row['mdi_rank']})"
            )
        print("\n   💡 Consider: These features have strong predictive power but trees")
        print("      don't use them much. Try feature engineering or ensemble methods.")
    else:
        print("   ✅ No significantly underutilized features found.")

    # Summary stats
    print("\n📊 Feature Importance Summary:")
    print("-" * 80)
    print(f"   Total features analyzed: {len(mdi_importance)}")
    print(f"   Top {args.top_n} features account for:")
    print(
        f"      • MDI: {mdi_importance.head(args.top_n)['importance'].sum():.1%} of total importance"
    )
    print(
        f"      • Permutation: {perm_importance.head(args.top_n)['importance'].sum():.3f} MAE increase"
    )

    if shap_importance is not None:
        print(
            f"      • SHAP: {shap_importance.head(args.top_n)['importance'].sum():.3f} mean |SHAP| value"
        )


def main(args: Any):
    """Main execution function."""

    print("\n" + "=" * 80)
    print("Feature Importance Analysis for FPL ML xP Prediction")
    print("=" * 80)

    try:
        # Load data (replicates TPOT optimizer)
        (
            historical_df,
            fixtures_df,
            teams_df,
            ownership_trends_df,
            value_analysis_df,
            fixture_difficulty_df,
            betting_features_df,
        ) = load_historical_data(args.start_gw, args.end_gw)

        # Engineer features (replicates TPOT optimizer)
        features_df, feature_cols = engineer_features(
            historical_df,
            fixtures_df,
            teams_df,
            ownership_trends_df,
            value_analysis_df,
            fixture_difficulty_df,
            betting_features_df,
        )

        # Create temporal CV splits (replicates TPOT optimizer)
        cv_splits, cv_data = create_temporal_cv_splits(features_df)

        # Prepare X and y
        X = cv_data[feature_cols]
        nan_counts = X.isna().sum()
        if nan_counts.any():
            nan_features = nan_counts[nan_counts > 0]
            raise ValueError(
                f"Features contain NaN values. Cannot proceed with incomplete data.\n"
                f"Features with NaN:\n{nan_features.to_string()}\n"
                f"This indicates upstream data quality issues."
            )

        y = cv_data["total_points"]

        print("\n📊 Final training data shape:")
        print(f"   X: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"   y: {len(y)} targets")

        # Train model
        model = train_model(X, y, args)

        # Analyze MDI importance
        mdi_importance = analyze_mdi_importance(model, feature_cols, args)

        # Analyze permutation importance
        perm_importance = analyze_permutation_importance(
            model, X, y, feature_cols, args
        )

        # Analyze SHAP importance (optional - slower)
        shap_importance = None
        shap_values = None
        X_sample = None
        if args.shap_analysis:
            shap_importance, shap_values, X_sample = analyze_shap_importance(
                model, X, feature_cols, args
            )

        # Analyze correlations
        top_features = mdi_importance.head(args.top_n)["feature"].tolist()
        corr_pairs = analyze_correlations(X, feature_cols, top_features)

        # Create visualizations
        output_dir = Path(args.output_dir)
        create_visualizations(
            mdi_importance,
            perm_importance,
            shap_importance,
            shap_values,
            X_sample,
            args,
            output_dir,
        )

        # Save results
        save_results(
            mdi_importance, perm_importance, shap_importance, corr_pairs, output_dir
        )

        # Generate recommendations
        generate_recommendations(mdi_importance, perm_importance, shap_importance, args)

        print("\n" + "=" * 80)
        print("✅ Feature importance analysis complete!")
        print("=" * 80)
        print(f"\n📁 Results saved to: {output_dir}")
        print("\n💡 Next steps:")
        print("   1. Review consensus top features for feature selection")
        print("   2. Examine underutilized features for potential improvement")
        print("   3. Check high correlation pairs for redundancy")
        print("   4. Use SHAP plots to understand feature interactions")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


@app.command()
def run(
    start_gw: int = typer.Option(1, help="Training start gameweek"),
    end_gw: int = typer.Option(9, help="Training end gameweek"),
    n_estimators: int = typer.Option(200, help="Number of trees in Random Forest"),
    max_depth: int = typer.Option(12, help="Max depth of Random Forest trees"),
    random_seed: int = typer.Option(42, help="Random seed"),
    top_n: int = typer.Option(30, help="Number of top features to display"),
    shap_analysis: bool = typer.Option(
        False, "--shap-analysis", help="Run SHAP analysis (slower)"
    ),
    output_dir: Path = typer.Option(
        Path("experiments/feature_importance_output"),
        help="Output directory for plots and CSVs",
    ),
):
    """Run feature importance analysis replicating the TPOT pipeline."""
    args = SimpleNamespace(
        start_gw=start_gw,
        end_gw=end_gw,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_seed=random_seed,
        top_n=top_n,
        shap_analysis=shap_analysis,
        output_dir=str(output_dir),
    )
    main(args)


if __name__ == "__main__":
    # Store client globally for use in functions
    client = FPLDataClient()
    app()
