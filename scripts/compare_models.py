"""
Side-by-side comparison of TPOT and Custom Random Forest models.

Evaluates both models using identical data, infrastructure, and metrics
to provide a direct, fair comparison.
"""

import sys
from pathlib import Path
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from ml_training_utils import (  # noqa: E402
    load_training_data,
    engineer_features,
    create_temporal_cv_splits,
    evaluate_fpl_comprehensive,
)


def evaluate_model(model_path, model_name, feature_names, X_cv, y_cv, cv_data):
    """Evaluate a single model and return metrics."""
    print(f"\n{'=' * 80}")
    print(f"📊 EVALUATING {model_name.upper()}")
    print(f"{'=' * 80}")

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None

    # Load model
    print(f"\n📥 Loading model: {model_path.name}")
    model = joblib.load(model_path)
    print(f"   ✅ Model loaded: {type(model).__name__}")

    # Inspect pipeline
    print("\n🔍 Pipeline steps:")
    if hasattr(model, "named_steps"):
        for i, (name, step) in enumerate(model.named_steps.items(), 1):
            print(f"   {i}. {name}: {type(step).__name__}")
            if hasattr(step, "n_features_to_select"):
                print(f"      → Features to select: {step.n_features_to_select}")
            if hasattr(step, "support_"):
                n_selected = step.support_.sum()
                n_total = len(step.support_)
                print(f"      → Features selected: {n_selected}/{n_total}")
    else:
        print(f"   Single estimator: {type(model).__name__}")

    # Make predictions
    print("\n🔮 Generating predictions...")
    # Check if model needs DataFrame or numpy array
    # FeatureSelector requires DataFrame, TPOT expects numpy array
    needs_dataframe = False
    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            if type(step).__name__ == "FeatureSelector":
                needs_dataframe = True
                break

    if needs_dataframe:
        print("   ℹ️  Using DataFrame input (FeatureSelector detected)...")
        y_pred = model.predict(X_cv)
    else:
        print("   ℹ️  Using numpy array input...")
        y_pred = model.predict(X_cv.values)

    # Evaluate
    metrics = evaluate_fpl_comprehensive(
        y_true=y_cv,
        y_pred=y_pred,
        cv_data=cv_data,
        verbose=True,
    )

    # Check feature selection
    feature_info = {"selected": [], "dropped": [], "penalty_status": {}}

    if hasattr(model, "named_steps"):
        # Find RFE or FeatureSelector step
        for name, step in model.named_steps.items():
            if hasattr(step, "support_"):  # RFE
                feature_info["selected"] = [
                    f for f, keep in zip(feature_names, step.support_) if keep
                ]
                feature_info["dropped"] = [
                    f for f, keep in zip(feature_names, step.support_) if not keep
                ]
                break
            elif hasattr(step, "feature_names"):  # FeatureSelector
                feature_info["selected"] = step.feature_names
                feature_info["dropped"] = [
                    f for f in feature_names if f not in step.feature_names
                ]
                break

    # Check penalty features
    penalty_features = [
        "is_primary_penalty_taker",
        "is_penalty_taker",
        "is_corner_taker",
        "is_fk_taker",
    ]

    for pf in penalty_features:
        if pf in feature_names:
            status = "✅ KEPT" if pf in feature_info["selected"] else "❌ DROPPED"
            feature_info["penalty_status"][pf] = status

    return {
        "model": model,
        "metrics": metrics,
        "feature_info": feature_info,
    }


def print_comparison_table(tpot_results, custom_results):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("📊 SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    tpot_m = tpot_results["metrics"]
    custom_m = custom_results["metrics"]

    print(
        "\n┌─────────────────────┬─────────────────┬─────────────────┬──────────────┐"
    )
    print("│ Metric              │ TPOT Model      │ Custom RF       │ Winner       │")
    print("├─────────────────────┼─────────────────┼─────────────────┼──────────────┤")

    # MAE
    mae_winner = "Custom RF" if custom_m["mae"] < tpot_m["mae"] else "TPOT"
    mae_improvement = abs(1 - custom_m["mae"] / tpot_m["mae"]) * 100
    print(
        f"│ MAE                 │ {tpot_m['mae']:15.3f} │ {custom_m['mae']:15.3f} │ {mae_winner:12s} │"
    )
    print(
        f"│                     │                 │                 │ ({mae_improvement:4.0f}% better)│"
    )

    # RMSE
    rmse_winner = "Custom RF" if custom_m["rmse"] < tpot_m["rmse"] else "TPOT"
    rmse_improvement = abs(1 - custom_m["rmse"] / tpot_m["rmse"]) * 100
    print(
        f"│ RMSE                │ {tpot_m['rmse']:15.3f} │ {custom_m['rmse']:15.3f} │ {rmse_winner:12s} │"
    )
    print(
        f"│                     │                 │                 │ ({rmse_improvement:4.0f}% better)│"
    )

    # Spearman
    spearman_winner = (
        "Custom RF"
        if custom_m["spearman_correlation"] > tpot_m["spearman_correlation"]
        else "TPOT"
    )
    spearman_improvement = (
        abs(custom_m["spearman_correlation"] - tpot_m["spearman_correlation"])
        / tpot_m["spearman_correlation"]
        * 100
    )
    print(
        f"│ Spearman            │ {tpot_m['spearman_correlation']:15.3f} │ {custom_m['spearman_correlation']:15.3f} │ {spearman_winner:12s} │"
    )
    print(
        f"│                     │                 │                 │ ({spearman_improvement:4.1f}% better) │"
    )

    # Top-15 overlap
    top15_winner = (
        "Custom RF"
        if custom_m["avg_top15_overlap"] > tpot_m["avg_top15_overlap"]
        else "TPOT"
    )
    top15_improvement = (
        abs(custom_m["avg_top15_overlap"] - tpot_m["avg_top15_overlap"])
        / tpot_m["avg_top15_overlap"]
        * 100
    )
    print(
        f"│ Avg Top-15 Overlap  │ {tpot_m['avg_top15_overlap']:11.1f}/15 │ {custom_m['avg_top15_overlap']:11.1f}/15 │ {top15_winner:12s} │"
    )
    print(
        f"│                     │ ({100 * tpot_m['avg_top15_overlap'] / 15:4.0f}%)         │ ({100 * custom_m['avg_top15_overlap'] / 15:4.0f}%)         │ ({top15_improvement:4.0f}% better)│"
    )

    # Captain accuracy
    cap_winner = (
        "Custom RF"
        if custom_m["captain_accuracy"] > tpot_m["captain_accuracy"]
        else "TPOT"
    )
    print(
        f"│ Captain Accuracy    │ {100 * tpot_m['captain_accuracy']:11.0f}%    │ {100 * custom_m['captain_accuracy']:11.0f}%    │ {cap_winner:12s} │"
    )

    print("└─────────────────────┴─────────────────┴─────────────────┴──────────────┘")

    # Feature selection comparison
    print("\n" + "=" * 80)
    print("🎯 FEATURE SELECTION COMPARISON")
    print("=" * 80)

    tpot_f = tpot_results["feature_info"]
    custom_f = custom_results["feature_info"]

    print("\n📊 Features selected:")
    print(f"   TPOT: {len(tpot_f['selected'])}/99")
    print(f"   Custom RF: {len(custom_f['selected'])}/99")

    print("\n⚽ Penalty/Set-piece Features:")
    print(f"   {'Feature':<30} {'TPOT':<15} {'Custom RF':<15}")
    print(f"   {'-' * 30} {'-' * 15} {'-' * 15}")

    for pf in [
        "is_primary_penalty_taker",
        "is_penalty_taker",
        "is_corner_taker",
        "is_fk_taker",
    ]:
        tpot_status = tpot_f["penalty_status"].get(pf, "N/A")
        custom_status = custom_f["penalty_status"].get(pf, "N/A")
        print(f"   {pf:<30} {tpot_status:<15} {custom_status:<15}")

    # Key insights
    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS")
    print("=" * 80)

    insights = []

    if custom_m["mae"] < tpot_m["mae"]:
        improvement = (1 - custom_m["mae"] / tpot_m["mae"]) * 100
        insights.append(
            f"✅ Custom RF achieves {improvement:.0f}% better MAE ({custom_m['mae']:.3f} vs {tpot_m['mae']:.3f})"
        )

    if custom_m["avg_top15_overlap"] > tpot_m["avg_top15_overlap"]:
        improvement = (
            (custom_m["avg_top15_overlap"] - tpot_m["avg_top15_overlap"])
            / tpot_m["avg_top15_overlap"]
            * 100
        )
        insights.append(
            f"✅ Custom RF has {improvement:.0f}% better top-15 overlap ({custom_m['avg_top15_overlap']:.1f} vs {tpot_m['avg_top15_overlap']:.1f})"
        )

    # Check if Custom kept penalties while TPOT dropped them
    tpot_dropped_penalties = any(
        "DROPPED" in status for status in tpot_f["penalty_status"].values()
    )
    custom_kept_penalties = all(
        "KEPT" in status for status in custom_f["penalty_status"].values()
    )

    if tpot_dropped_penalties and custom_kept_penalties:
        insights.append(
            "⚽ Custom RF preserves all penalty features, TPOT drops them (explains MAE difference)"
        )

    for insight in insights:
        print(f"\n{insight}")

    print(
        "\n⚠️  Note: Both models evaluated on training data (GW6-9). Test on GW10+ for true generalization."
    )


def main():
    print("=" * 80)
    print("🔬 MODEL COMPARISON: TPOT vs CUSTOM RANDOM FOREST")
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

    # Load data (once, used for both models)
    print("\n" + "=" * 80)
    print("📊 LOADING DATA (SHARED INFRASTRUCTURE)")
    print("=" * 80)

    (
        historical_df,
        fixtures_df,
        teams_df,
        ownership_trends_df,
        value_analysis_df,
        fixture_difficulty_df,
        betting_features_df,
        raw_players_df,
    ) = load_training_data(start_gw=1, end_gw=9, verbose=True)

    features_df, target, feature_names = engineer_features(
        historical_df,
        fixtures_df,
        teams_df,
        ownership_trends_df,
        value_analysis_df,
        fixture_difficulty_df,
        betting_features_df,
        raw_players_df,
        verbose=True,
    )

    cv_splits, cv_data = create_temporal_cv_splits(features_df, verbose=True)

    # Extract features and target for CV data
    X_cv = cv_data[feature_names]  # Keep as DataFrame for FeatureSelector
    y_cv = target[cv_data["_original_index"].values]

    print(f"\n✅ Data loaded: {len(X_cv)} samples, {len(feature_names)} features")

    # Evaluate both models
    tpot_results = evaluate_model(tpot_path, "TPOT", feature_names, X_cv, y_cv, cv_data)

    custom_results = evaluate_model(
        custom_path, "Custom Random Forest", feature_names, X_cv, y_cv, cv_data
    )

    # Print comparison
    if tpot_results and custom_results:
        print_comparison_table(tpot_results, custom_results)

    print("\n" + "=" * 80)
    print("✅ COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
