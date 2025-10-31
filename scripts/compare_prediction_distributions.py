"""
Compare prediction distributions between TPOT and Custom RF models.

Examines variance, range, and top player predictions to understand
if models are actually differentiating between players effectively.
"""

import sys
from pathlib import Path
import joblib
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from ml_training_utils import (  # noqa: E402
    load_training_data,
    engineer_features,
    create_temporal_cv_splits,
)


def analyze_predictions(y_true, y_pred, model_name, cv_data, features_df):
    """Analyze prediction distribution and characteristics."""
    print(f"\n{'=' * 80}")
    print(f"📊 {model_name.upper()} PREDICTION ANALYSIS")
    print(f"{'=' * 80}")

    # Basic statistics
    print("\n📈 Prediction Statistics:")
    print(f"   Mean:     {np.mean(y_pred):.3f}")
    print(f"   Std Dev:  {np.std(y_pred):.3f}")
    print(f"   Min:      {np.min(y_pred):.3f}")
    print(f"   Max:      {np.max(y_pred):.3f}")
    print(f"   Range:    {np.max(y_pred) - np.min(y_pred):.3f}")
    print(f"   Median:   {np.median(y_pred):.3f}")

    print("\n📈 Actual Points Statistics:")
    print(f"   Mean:     {np.mean(y_true):.3f}")
    print(f"   Std Dev:  {np.std(y_true):.3f}")
    print(f"   Min:      {np.min(y_true):.3f}")
    print(f"   Max:      {np.max(y_true):.3f}")
    print(f"   Range:    {np.max(y_true) - np.min(y_true):.3f}")

    # Percentiles
    print("\n📊 Prediction Percentiles:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"   {p:2d}th: {np.percentile(y_pred, p):.3f}")

    # Distribution bins
    print("\n📊 Prediction Distribution:")
    bins = [0, 2, 4, 6, 8, 10, 100]
    labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10+"]
    counts = np.histogram(y_pred, bins=bins)[0]
    for label, count in zip(labels, counts):
        pct = 100 * count / len(y_pred)
        print(f"   {label:8s}: {count:4d} players ({pct:5.1f}%)")

    return {
        "mean": np.mean(y_pred),
        "std": np.std(y_pred),
        "min": np.min(y_pred),
        "max": np.max(y_pred),
        "range": np.max(y_pred) - np.min(y_pred),
        "predictions": y_pred,
    }


def compare_top_players(
    tpot_pred, custom_pred, y_true, cv_data, raw_players_df, n_players=50
):
    """Compare top N players from both models."""
    print(f"\n{'=' * 80}")
    print(f"🏆 TOP {n_players} PLAYERS COMPARISON")
    print(f"{'=' * 80}")

    # Create results DataFrame
    results_df = cv_data.copy()
    results_df["actual_points"] = y_true
    results_df["tpot_pred"] = tpot_pred
    results_df["custom_pred"] = custom_pred
    results_df["tpot_error"] = np.abs(tpot_pred - y_true)
    results_df["custom_error"] = np.abs(custom_pred - y_true)

    # Get player names from raw_players_df (only use available columns)
    available_cols = ["player_id", "web_name"]
    if "position" in raw_players_df.columns:
        available_cols.append("position")
    player_info = raw_players_df[available_cols].drop_duplicates()
    results_df = results_df.merge(
        player_info,
        left_on="player_id",
        right_on="player_id",
        how="left",
    )

    # Use position from cv_data if not in raw_players_df
    if "position" not in available_cols and "position" in cv_data.columns:
        # Position already in results_df from cv_data
        pass

    # Top players by actual points
    print("\n📊 TOP 50 BY ACTUAL POINTS:")
    top_actual = results_df.nlargest(n_players, "actual_points")

    has_position = "position" in results_df.columns
    if has_position:
        print(
            f"\n{'Rank':<6} {'Player':<25} {'Pos':<5} {'Actual':<8} {'TPOT':<8} {'Custom':<8} {'Δ TPOT':<8} {'Δ Custom':<8}"
        )
        print("-" * 90)
    else:
        print(
            f"\n{'Rank':<6} {'Player':<30} {'Actual':<8} {'TPOT':<8} {'Custom':<8} {'Δ TPOT':<8} {'Δ Custom':<8}"
        )
        print("-" * 85)

    for i, row in enumerate(top_actual.itertuples(), 1):
        player_name = getattr(row, "web_name", f"Player_{row.player_id}")[
            : 24 if has_position else 29
        ]
        if has_position:
            pos = getattr(row, "position", "?")
            print(
                f"{i:<6} {player_name:<25} {pos:<5} {row.actual_points:7.1f}  {row.tpot_pred:7.2f}  {row.custom_pred:7.2f}  {row.tpot_error:7.2f}   {row.custom_error:7.2f}"
            )
        else:
            print(
                f"{i:<6} {player_name:<30} {row.actual_points:7.1f}  {row.tpot_pred:7.2f}  {row.custom_pred:7.2f}  {row.tpot_error:7.2f}   {row.custom_error:7.2f}"
            )

    # Top players by TPOT prediction
    print("\n\n📊 TOP 50 BY TPOT PREDICTION:")
    top_tpot = results_df.nlargest(n_players, "tpot_pred")
    if has_position:
        print(
            f"\n{'Rank':<6} {'Player':<25} {'Pos':<5} {'TPOT':<8} {'Actual':<8} {'Custom':<8} {'Error':<8}"
        )
        print("-" * 85)
    else:
        print(
            f"\n{'Rank':<6} {'Player':<30} {'TPOT':<8} {'Actual':<8} {'Custom':<8} {'Error':<8}"
        )
        print("-" * 75)

    for i, row in enumerate(top_tpot.itertuples(), 1):
        player_name = getattr(row, "web_name", f"Player_{row.player_id}")[
            : 24 if has_position else 29
        ]
        error = row.actual_points - row.tpot_pred
        if has_position:
            pos = getattr(row, "position", "?")
            print(
                f"{i:<6} {player_name:<25} {pos:<5} {row.tpot_pred:7.2f}  {row.actual_points:7.1f}  {row.custom_pred:7.2f}  {error:+7.2f}"
            )
        else:
            print(
                f"{i:<6} {player_name:<30} {row.tpot_pred:7.2f}  {row.actual_points:7.1f}  {row.custom_pred:7.2f}  {error:+7.2f}"
            )

    # Top players by Custom RF prediction
    print("\n\n📊 TOP 50 BY CUSTOM RF PREDICTION:")
    top_custom = results_df.nlargest(n_players, "custom_pred")
    if has_position:
        print(
            f"\n{'Rank':<6} {'Player':<25} {'Pos':<5} {'Custom':<8} {'Actual':<8} {'TPOT':<8} {'Error':<8}"
        )
        print("-" * 85)
    else:
        print(
            f"\n{'Rank':<6} {'Player':<30} {'Custom':<8} {'Actual':<8} {'TPOT':<8} {'Error':<8}"
        )
        print("-" * 75)

    for i, row in enumerate(top_custom.itertuples(), 1):
        player_name = getattr(row, "web_name", f"Player_{row.player_id}")[
            : 24 if has_position else 29
        ]
        error = row.actual_points - row.custom_pred
        if has_position:
            pos = getattr(row, "position", "?")
            print(
                f"{i:<6} {player_name:<25} {pos:<5} {row.custom_pred:7.2f}  {row.actual_points:7.1f}  {row.tpot_pred:7.2f}  {error:+7.2f}"
            )
        else:
            print(
                f"{i:<6} {player_name:<30} {row.custom_pred:7.2f}  {row.actual_points:7.1f}  {row.tpot_pred:7.2f}  {error:+7.2f}"
            )

    # Overlap analysis
    print(f"\n\n📊 TOP {n_players} OVERLAP ANALYSIS:")
    tpot_top_ids = set(top_tpot["player_id"].values)
    custom_top_ids = set(top_custom["player_id"].values)
    actual_top_ids = set(top_actual["player_id"].values)

    tpot_actual_overlap = len(tpot_top_ids & actual_top_ids)
    custom_actual_overlap = len(custom_top_ids & actual_top_ids)
    tpot_custom_overlap = len(tpot_top_ids & custom_top_ids)

    print(
        f"   TPOT ∩ Actual:   {tpot_actual_overlap}/{n_players} ({100 * tpot_actual_overlap / n_players:.0f}%)"
    )
    print(
        f"   Custom ∩ Actual: {custom_actual_overlap}/{n_players} ({100 * custom_actual_overlap / n_players:.0f}%)"
    )
    print(
        f"   TPOT ∩ Custom:   {tpot_custom_overlap}/{n_players} ({100 * tpot_custom_overlap / n_players:.0f}%)"
    )

    # Prediction variance by position
    print("\n\n📊 PREDICTION VARIANCE BY POSITION:")
    print(f"\n{'Position':<10} {'TPOT Std':<12} {'Custom Std':<12} {'Actual Std':<12}")
    print("-" * 50)

    for pos in ["GKP", "DEF", "MID", "FWD"]:
        pos_mask = results_df["position"] == pos
        if pos_mask.sum() > 0:
            tpot_std = results_df.loc[pos_mask, "tpot_pred"].std()
            custom_std = results_df.loc[pos_mask, "custom_pred"].std()
            actual_std = results_df.loc[pos_mask, "actual_points"].std()
            print(f"{pos:<10} {tpot_std:11.3f}  {custom_std:11.3f}  {actual_std:11.3f}")


def main():
    print("=" * 80)
    print("🔬 PREDICTION DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Model paths
    tpot_path = (
        project_root / "models" / "tpot" / "tpot_pipeline_gw1-9_20251031_064636.joblib"
    )
    custom_path = (
        project_root
        / "models"
        / "custom"
        / "random-forest_gw1-9_20251031_140131_pipeline.joblib"
    )

    # Load models
    print("\n📥 Loading models...")
    tpot_model = joblib.load(tpot_path)
    custom_model = joblib.load(custom_path)
    print("   ✅ Models loaded")

    # Load and prepare data
    print("\n📊 Loading data...")
    (
        historical_df,
        fixtures_df,
        teams_df,
        ownership_trends_df,
        value_analysis_df,
        fixture_difficulty_df,
        betting_features_df,
        raw_players_df,
    ) = load_training_data(start_gw=1, end_gw=9, verbose=False)

    features_df, target, feature_names = engineer_features(
        historical_df,
        fixtures_df,
        teams_df,
        ownership_trends_df,
        value_analysis_df,
        fixture_difficulty_df,
        betting_features_df,
        raw_players_df,
        verbose=False,
    )

    cv_splits, cv_data = create_temporal_cv_splits(features_df, verbose=False)

    # Extract features
    X_cv = cv_data[feature_names]
    y_cv = target[cv_data["_original_index"].values]

    print(f"   ✅ Data loaded: {len(X_cv)} samples")

    # Generate predictions
    print("\n🔮 Generating predictions...")
    tpot_pred = tpot_model.predict(X_cv.values)
    custom_pred = custom_model.predict(X_cv)
    print("   ✅ Predictions generated")

    # Analyze distributions
    tpot_stats = analyze_predictions(y_cv, tpot_pred, "TPOT", cv_data, features_df)
    custom_stats = analyze_predictions(
        y_cv, custom_pred, "Custom RF", cv_data, features_df
    )

    # Compare distributions
    print(f"\n{'=' * 80}")
    print("📊 DISTRIBUTION COMPARISON")
    print(f"{'=' * 80}")

    print("\n📈 Variance Comparison:")
    print(f"   TPOT Std Dev:   {tpot_stats['std']:.3f}")
    print(f"   Custom Std Dev: {custom_stats['std']:.3f}")
    print(f"   Actual Std Dev: {np.std(y_cv):.3f}")
    print(
        f"   Variance Ratio (Custom/TPOT): {custom_stats['std'] / tpot_stats['std']:.2f}x"
    )

    print("\n📈 Range Comparison:")
    print(f"   TPOT Range:   {tpot_stats['range']:.3f}")
    print(f"   Custom Range: {custom_stats['range']:.3f}")
    print(f"   Actual Range: {np.max(y_cv) - np.min(y_cv):.3f}")
    print(
        f"   Range Ratio (Custom/TPOT): {custom_stats['range'] / tpot_stats['range']:.2f}x"
    )

    # Compare top players
    compare_top_players(
        tpot_pred, custom_pred, y_cv, cv_data, raw_players_df, n_players=50
    )

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
