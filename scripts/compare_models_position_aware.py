"""
Compare TPOT vs Custom RF models using position-aware FPL metrics.

Evaluates models on realistic FPL criteria:
- Top-K per position (not overall top-15)
- Starting XI point efficiency across formations
- Captain selection within XI constraints
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

from position_aware_scorer import evaluate_position_aware_metrics  # noqa: E402


def evaluate_model_position_aware(model, model_name, X_cv, y_cv, position_labels):
    """Evaluate a model using position-aware FPL metrics."""
    print(f"\n{'=' * 80}")
    print(f"📊 {model_name.upper()} - POSITION-AWARE EVALUATION")
    print(f"{'=' * 80}")

    # Make predictions
    print("\n🔮 Generating predictions...")
    # Check if model needs DataFrame or numpy array
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

    # Evaluate position-aware metrics
    print("\n📈 Evaluating position-aware FPL metrics...")
    metrics = evaluate_position_aware_metrics(
        y_cv,
        y_pred,
        position_labels,
        formations=["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1"],
    )

    # Print results
    print("\n🎯 Position-Aware Top-K Overlap:")
    print(f"   Overall: {metrics['position_topk_overlap']:.3f}")
    print("   (Top-5 GKP, Top-10 DEF/MID/FWD)")

    print("\n⚽ Starting XI Efficiency (by formation):")
    for formation in ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1"]:
        efficiency = metrics[f"xi_efficiency_{formation}"]
        print(f"   {formation}: {efficiency:.3f} (% of optimal points)")

    print("\n👑 Captain Accuracy (within XI, by formation):")
    for formation in ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1"]:
        accuracy = metrics[f"captain_accuracy_{formation}"]
        print(f"   {formation}: {accuracy:.3f}")

    print("\n📊 Comprehensive FPL Score (4-4-2 formation):")
    print(f"   {metrics['comprehensive_442']:.3f}")
    print("   (40% top-K + 40% XI efficiency + 20% captain)")

    return metrics


def print_comparison_table(tpot_metrics, custom_metrics):
    """Print side-by-side comparison of position-aware metrics."""
    print("\n" + "=" * 80)
    print("📊 POSITION-AWARE COMPARISON: TPOT vs CUSTOM RF")
    print("=" * 80)

    print(
        "\n┌────────────────────────────────┬─────────────┬─────────────┬─────────────┐"
    )
    print(
        "│ Metric                         │ TPOT        │ Custom RF   │ Winner      │"
    )
    print(
        "├────────────────────────────────┼─────────────┼─────────────┼─────────────┤"
    )

    # Position-aware top-K
    tpot_topk = tpot_metrics["position_topk_overlap"]
    custom_topk = custom_metrics["position_topk_overlap"]
    topk_winner = "Custom RF" if custom_topk > tpot_topk else "TPOT"
    improvement = abs(custom_topk - tpot_topk) / tpot_topk * 100
    print(
        f"│ Position Top-K Overlap         │ {tpot_topk:11.3f} │ {custom_topk:11.3f} │ {topk_winner:11s} │"
    )
    print(
        f"│                                │             │             │ ({improvement:4.1f}% {'better' if custom_topk > tpot_topk else 'worse':>6s})│"
    )

    # XI efficiency (average across formations)
    formations = ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1"]
    tpot_xi_avg = np.mean([tpot_metrics[f"xi_efficiency_{f}"] for f in formations])
    custom_xi_avg = np.mean([custom_metrics[f"xi_efficiency_{f}"] for f in formations])
    xi_winner = "Custom RF" if custom_xi_avg > tpot_xi_avg else "TPOT"
    xi_improvement = abs(custom_xi_avg - tpot_xi_avg) / tpot_xi_avg * 100
    print(
        f"│ Avg XI Efficiency (all forms)  │ {tpot_xi_avg:11.3f} │ {custom_xi_avg:11.3f} │ {xi_winner:11s} │"
    )
    print(
        f"│                                │             │             │ ({xi_improvement:4.1f}% {'better' if custom_xi_avg > tpot_xi_avg else 'worse':>6s})│"
    )

    # Captain accuracy (average across formations)
    tpot_cap_avg = np.mean([tpot_metrics[f"captain_accuracy_{f}"] for f in formations])
    custom_cap_avg = np.mean(
        [custom_metrics[f"captain_accuracy_{f}"] for f in formations]
    )
    cap_winner = "Custom RF" if custom_cap_avg > tpot_cap_avg else "TPOT"
    cap_improvement = (
        abs(custom_cap_avg - tpot_cap_avg) / tpot_cap_avg * 100
        if tpot_cap_avg > 0
        else 0
    )
    print(
        f"│ Avg Captain Accuracy (all)     │ {tpot_cap_avg:11.3f} │ {custom_cap_avg:11.3f} │ {cap_winner:11s} │"
    )
    print(
        f"│                                │             │             │ ({cap_improvement:4.1f}% {'better' if custom_cap_avg > tpot_cap_avg else 'worse':>6s})│"
    )

    # Comprehensive score
    tpot_comp = tpot_metrics["comprehensive_442"]
    custom_comp = custom_metrics["comprehensive_442"]
    comp_winner = "Custom RF" if custom_comp > tpot_comp else "TPOT"
    comp_improvement = abs(custom_comp - tpot_comp) / tpot_comp * 100
    print(
        f"│ Comprehensive FPL Score (4-4-2)│ {tpot_comp:11.3f} │ {custom_comp:11.3f} │ {comp_winner:11s} │"
    )
    print(
        f"│                                │             │             │ ({comp_improvement:4.1f}% {'better' if custom_comp > tpot_comp else 'worse':>6s})│"
    )

    print(
        "└────────────────────────────────┴─────────────┴─────────────┴─────────────┘"
    )

    # Detailed breakdown by formation
    print("\n" + "=" * 80)
    print("⚽ FORMATION-SPECIFIC ANALYSIS")
    print("=" * 80)

    for formation in formations:
        print(f"\n{formation}:")
        tpot_xi = tpot_metrics[f"xi_efficiency_{formation}"]
        custom_xi = custom_metrics[f"xi_efficiency_{formation}"]
        tpot_cap = tpot_metrics[f"captain_accuracy_{formation}"]
        custom_cap = custom_metrics[f"captain_accuracy_{formation}"]

        print(
            f"   XI Efficiency:    TPOT {tpot_xi:.3f} | Custom {custom_xi:.3f} | {'✅ Custom' if custom_xi > tpot_xi else '✅ TPOT'}"
        )
        print(
            f"   Captain Accuracy: TPOT {tpot_cap:.3f} | Custom {custom_cap:.3f} | {'✅ Custom' if custom_cap > tpot_cap else '✅ TPOT'}"
        )

    # Key insights
    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS")
    print("=" * 80)

    insights = []

    if custom_topk > tpot_topk:
        insights.append(
            f"✅ Custom RF better at identifying top players per position ({custom_topk:.3f} vs {tpot_topk:.3f})"
        )
    else:
        insights.append(
            f"⚠️  TPOT better at identifying top players per position ({tpot_topk:.3f} vs {custom_topk:.3f})"
        )

    if custom_xi_avg > tpot_xi_avg:
        insights.append(
            f"✅ Custom RF builds more optimal XIs on average ({custom_xi_avg:.3f} vs {tpot_xi_avg:.3f})"
        )
    else:
        insights.append(
            f"⚠️  TPOT builds more optimal XIs on average ({tpot_xi_avg:.3f} vs {custom_xi_avg:.3f})"
        )

    if custom_cap_avg > tpot_cap_avg:
        insights.append(
            f"✅ Custom RF better at captain selection within XI ({custom_cap_avg:.3f} vs {tpot_cap_avg:.3f})"
        )
    else:
        insights.append(
            f"⚠️  TPOT better at captain selection within XI ({tpot_cap_avg:.3f} vs {custom_cap_avg:.3f})"
        )

    for insight in insights:
        print(f"\n{insight}")

    print(
        "\n⚠️  Remember: These are training set metrics (GW6-9). Validate on GW10+ for generalization."
    )


def main():
    print("=" * 80)
    print("🏆 POSITION-AWARE FPL MODEL COMPARISON")
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

    # Extract features and position labels
    X_cv = cv_data[feature_names]
    y_cv = target[cv_data["_original_index"].values]
    position_labels = cv_data["position"].values

    print(f"   ✅ Data loaded: {len(X_cv)} samples")
    print(
        f"   ✅ Positions: GKP={sum(position_labels == 'GKP')}, "
        f"DEF={sum(position_labels == 'DEF')}, "
        f"MID={sum(position_labels == 'MID')}, "
        f"FWD={sum(position_labels == 'FWD')}"
    )

    # Evaluate both models
    tpot_metrics = evaluate_model_position_aware(
        tpot_model, "TPOT", X_cv, y_cv, position_labels
    )

    custom_metrics = evaluate_model_position_aware(
        custom_model, "Custom RF", X_cv, y_cv, position_labels
    )

    # Compare
    print_comparison_table(tpot_metrics, custom_metrics)

    print("\n" + "=" * 80)
    print("✅ POSITION-AWARE COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
