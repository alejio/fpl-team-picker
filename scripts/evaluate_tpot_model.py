"""
Evaluate TPOT model using standardized ml_training_utils for fair comparison.

This script loads the TPOT pipeline and evaluates it using the same
data loading, feature engineering, and evaluation methodology as the custom
pipelines to ensure "apples to apples" comparison.
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


def main():
    print("=" * 80)
    print("🔍 TPOT MODEL EVALUATION (STANDARDIZED)")
    print("=" * 80)

    # Load TPOT model
    model_path = (
        project_root / "models" / "tpot" / "tpot_pipeline_gw1-9_20251031_064636.joblib"
    )
    print(f"\n📥 Loading TPOT model: {model_path.name}")

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    tpot_pipeline = joblib.load(model_path)
    print(f"   ✅ Model loaded: {type(tpot_pipeline).__name__}")

    # Inspect pipeline steps
    print("\n🔍 Pipeline steps:")
    for i, (name, step) in enumerate(tpot_pipeline.named_steps.items(), 1):
        print(f"   {i}. {name}: {type(step).__name__}")
        if hasattr(step, "n_features_to_select"):
            print(f"      → Features to select: {step.n_features_to_select}")
        if hasattr(step, "n_features_in_"):
            print(f"      → Features in: {step.n_features_in_}")
        if hasattr(step, "support_"):
            n_selected = step.support_.sum()
            n_total = len(step.support_)
            print(f"      → Features selected: {n_selected}/{n_total}")

    # Load and engineer features using standardized utils
    print("\n" + "=" * 80)
    print("📊 LOADING DATA (STANDARDIZED)")
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
    X_cv = cv_data[feature_names].values
    # Use original index to get correct target values
    y_cv = target[cv_data["_original_index"].values]

    print("\n" + "=" * 80)
    print("📊 EVALUATING TPOT MODEL")
    print("=" * 80)

    # Make predictions
    print("\n🔮 Generating predictions...")
    y_pred = tpot_pipeline.predict(X_cv)

    # Evaluate with comprehensive FPL metrics
    metrics = evaluate_fpl_comprehensive(
        y_true=y_cv,
        y_pred=y_pred,
        cv_data=cv_data,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    print("\n📊 TPOT Model Summary:")
    print(f"   MAE: {metrics['mae']:.3f}")
    print(f"   RMSE: {metrics['rmse']:.3f}")
    print(f"   Spearman: {metrics['spearman_correlation']:.3f}")
    print(f"   Avg Top-15 Overlap: {metrics['avg_top15_overlap']:.1f}/15")
    print(f"   Captain Accuracy: {100 * metrics['captain_accuracy']:.0f}%")

    # Check which features were selected by RFE
    rfe_step = None
    for name, step in tpot_pipeline.named_steps.items():
        if hasattr(step, "support_"):
            rfe_step = step
            break

    if rfe_step is not None:
        print("\n" + "=" * 80)
        print("🔍 FEATURE SELECTION ANALYSIS")
        print("=" * 80)

        selected_features = [
            f for f, keep in zip(feature_names, rfe_step.support_) if keep
        ]
        dropped_features = [
            f for f, keep in zip(feature_names, rfe_step.support_) if not keep
        ]

        print(f"\n✅ Selected features ({len(selected_features)}):")
        for f in sorted(selected_features)[:20]:  # Show first 20
            print(f"   • {f}")
        if len(selected_features) > 20:
            print(f"   ... and {len(selected_features) - 20} more")

        print(f"\n❌ Dropped features ({len(dropped_features)}):")
        for f in sorted(dropped_features)[:20]:  # Show first 20
            print(f"   • {f}")
        if len(dropped_features) > 20:
            print(f"   ... and {len(dropped_features) - 20} more")

        # Check penalty features specifically
        penalty_features = [
            "is_primary_penalty_taker",
            "is_penalty_taker",
            "is_corner_taker",
            "is_fk_taker",
        ]

        print("\n⚽ Penalty/Set-piece Features Status:")
        for pf in penalty_features:
            if pf in feature_names:
                idx = feature_names.index(pf)
                status = "✅ KEPT" if rfe_step.support_[idx] else "❌ DROPPED"
                print(f"   {pf}: {status}")


if __name__ == "__main__":
    main()
