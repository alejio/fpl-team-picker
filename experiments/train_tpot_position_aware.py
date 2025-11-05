"""
Train TPOT model with position-aware FPL scorer.

This experiment tests the hypothesis that training with position-aware
decision-based loss produces better FPL models than point-based loss (MAE, Huber).

Key Innovation: Use fpl_comprehensive_team_scorer that mirrors what
OptimizationService actually optimizes (position-aware XI selection + captain).
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import make_scorer
from tpot import TPOTRegressor
import joblib
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from ml_training_utils import (  # noqa: E402
    load_training_data,
    engineer_features,
)

from position_aware_scorer import (  # noqa: E402
    fpl_comprehensive_team_scorer,
)


def create_position_aware_scorer_for_sklearn():
    """
    Create sklearn-compatible scorer that extracts position from data.

    Challenge: sklearn scorers don't normally have access to metadata.
    Solution: Store positions globally and use CV splitter to track indices.
    """
    # Global storage for position labels and current CV indices
    position_storage = {
        "positions": None,
        "player_ids": None,
        "current_train_idx": None,
        "current_test_idx": None,
    }

    def scorer_with_positions(estimator, X, y):
        """
        Scorer function that uses stored position labels.

        Args:
            estimator: Trained model
            X: Features (numpy array - subset from CV split)
            y: Target (actual points)
        """
        # Get predictions
        y_pred = estimator.predict(X)

        # Get position labels for this fold
        if position_storage["positions"] is None:
            raise ValueError(
                "Position labels not set! Must call set_positions() before scoring"
            )

        # Determine which indices we're scoring
        # TPOT/sklearn passes subsets, so we need to use stored indices
        # The scorer is called with test set during CV, so use current_test_idx
        test_idx = position_storage["current_test_idx"]

        if test_idx is None:
            # Fallback: if indices not set, assume we're scoring the last len(X) positions
            # This is fragile but better than crashing
            if len(X) <= len(position_storage["positions"]):
                current_positions = position_storage["positions"][-len(X) :]
            else:
                # If X is larger, something is wrong - use all positions
                current_positions = position_storage["positions"]
        else:
            # Use the stored test indices to get correct positions
            if len(test_idx) == len(X):
                current_positions = position_storage["positions"][test_idx]
            else:
                # Mismatch - use fallback
                current_positions = position_storage["positions"][: len(X)]

        # Calculate position-aware score
        score = fpl_comprehensive_team_scorer(
            y, y_pred, current_positions, formation="4-4-2"
        )

        return score

    # Create sklearn scorer
    sklearn_scorer = make_scorer(scorer_with_positions, greater_is_better=True)

    # Attach storage reference so we can update it
    sklearn_scorer._position_storage = position_storage

    return sklearn_scorer


def train_tpot_position_aware(
    start_gw=1,
    end_gw=9,
    max_time_mins=5,
    population_size=20,
    generations=10,
):
    """
    Train TPOT model with position-aware comprehensive scorer.

    Args:
        start_gw: Starting gameweek for training data
        end_gw: Ending gameweek for training data
        max_time_mins: Maximum training time in minutes
        population_size: TPOT population size
        generations: TPOT generations (None = unlimited within time)
    """
    print("=" * 80)
    print("🧬 TPOT TRAINING WITH POSITION-AWARE SCORER")
    print("=" * 80)

    # Load data
    print(f"\n📊 Loading training data (GW{start_gw}-{end_gw})...")
    (
        historical_df,
        fixtures_df,
        teams_df,
        ownership_trends_df,
        value_analysis_df,
        fixture_difficulty_df,
        betting_features_df,
        raw_players_df,
    ) = load_training_data(start_gw=start_gw, end_gw=end_gw, verbose=True)

    # Engineer features
    print("\n🔧 Engineering features...")
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

    # Keep only training gameweeks (GW6+)
    train_mask = features_df["gameweek"] >= 6
    features_df_train = features_df[train_mask].copy()
    target_train = target[train_mask]

    print(f"\n✅ Training set: {len(features_df_train)} samples from GW6-{end_gw}")

    # Extract features and metadata
    X_train = features_df_train[feature_names].values
    y_train = target_train
    position_labels = features_df_train["position"].values
    player_ids = features_df_train["player_id"].values

    # Create position-aware scorer
    print("\n🎯 Creating position-aware scorer...")
    print("   Scorer components:")
    print("   - 40% position-aware top-K overlap (GKP/DEF/MID/FWD)")
    print("   - 40% starting XI efficiency (formation optimization)")
    print("   - 20% captain accuracy (within XI)")

    scorer = create_position_aware_scorer_for_sklearn()

    # Set position labels in scorer storage
    scorer._position_storage["positions"] = position_labels
    scorer._position_storage["player_ids"] = player_ids

    print("\n✅ Position distribution:")
    print(f"   GKP: {sum(position_labels == 'GKP')}")
    print(f"   DEF: {sum(position_labels == 'DEF')}")
    print(f"   MID: {sum(position_labels == 'MID')}")
    print(f"   FWD: {sum(position_labels == 'FWD')}")

    # Create temporal CV splits (walk-forward)
    print("\n📊 Creating temporal CV splits (walk-forward)...")
    gameweeks = features_df_train["gameweek"].values
    unique_gws = sorted(features_df_train["gameweek"].unique())

    cv_splits = []
    for i in range(len(unique_gws) - 1):
        train_gws = unique_gws[: i + 1]
        test_gw = unique_gws[i + 1]

        train_idx = np.where(np.isin(gameweeks, train_gws))[0]
        test_idx = np.where(gameweeks == test_gw)[0]

        cv_splits.append((train_idx, test_idx))
        print(f"   Fold {i + 1}: Train on GW{train_gws} → Test on GW{test_gw}")

    print(f"\n✅ Created {len(cv_splits)} temporal CV folds")

    # Create sklearn-compatible CV splitter that tracks indices for scorer
    from sklearn.model_selection import BaseCrossValidator

    class PreSplitCV(BaseCrossValidator):
        """Wrapper for pre-computed CV splits to work with sklearn/TPOT.

        Also tracks current test indices so position-aware scorer can use them.
        """

        def __init__(self, splits, position_storage=None):
            self.splits = splits
            self.position_storage = position_storage

        def split(self, X, y=None, groups=None):
            for train_idx, test_idx in self.splits:
                # Store current indices for scorer to use
                if self.position_storage is not None:
                    self.position_storage["current_train_idx"] = train_idx
                    self.position_storage["current_test_idx"] = test_idx
                yield train_idx, test_idx

        def get_n_splits(self, X=None, y=None, groups=None):
            return len(self.splits)

    cv_splitter = PreSplitCV(cv_splits, position_storage=scorer._position_storage)

    # Configure TPOT
    print("\n🧬 Configuring TPOT...")
    print(f"   Max time: {max_time_mins} minutes")
    print("   Max eval time: 3 minutes per pipeline")
    print("   Scorer: Position-aware comprehensive (FPL decision-based)")
    print(f"   CV: {len(cv_splits)}-fold temporal walk-forward")
    print(
        "   Note: population_size and generations not used in TPOT 1.1.0 (time-based optimization)"
    )

    # Set up Dask for TPOT 1.1.0
    from dask.distributed import Client, LocalCluster

    print("\n🔧 Setting up Dask cluster...")
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        dashboard_address=":0",
        silence_logs=True,  # Reduce noise
    )
    client = Client(cluster)
    print("   ✅ Dask cluster ready")

    try:
        tpot = TPOTRegressor(
            scorers=[scorer],  # TPOT 1.1.0 uses 'scorers' (list)
            cv=cv_splitter,  # Use CV splitter object
            max_time_mins=max_time_mins,
            max_eval_time_mins=3,  # Max 3 minutes per pipeline eval
            random_state=42,
            verbose=2,
            n_jobs=1,  # TPOT 1.1.0 uses Dask for parallelization
            client=client,  # Pass Dask client
        )

        # Train
        print("\n" + "=" * 80)
        print("🚀 STARTING TPOT TRAINING")
        print("=" * 80)

        tpot.fit(X_train, y_train)

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE")
        print("=" * 80)

    finally:
        # Clean up Dask resources
        client.close()
        cluster.close()
        print("\n🧹 Dask cluster shut down")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = project_root / "models" / "tpot"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        model_dir / f"tpot_position_aware_gw{start_gw}-{end_gw}_{timestamp}.joblib"
    )

    print(f"\n💾 Saving model to: {model_path.name}")
    joblib.dump(tpot.fitted_pipeline_, model_path)

    # Save metadata
    metadata_path = model_path.with_suffix(".txt")
    with open(metadata_path, "w") as f:
        f.write("TPOT Position-Aware Model\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training Data: GW{start_gw}-{end_gw}\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Features: {len(feature_names)}\n")
        f.write("Scorer: Position-aware comprehensive (FPL decision-based)\n")
        f.write("  - 40% position top-K overlap\n")
        f.write("  - 40% XI efficiency\n")
        f.write("  - 20% captain accuracy\n\n")
        f.write("TPOT Config:\n")
        f.write(f"  Max time: {max_time_mins} minutes\n")
        f.write("  Max eval time: 3 minutes per pipeline\n")
        f.write(f"  CV folds: {len(cv_splits)} (temporal)\n\n")
        f.write("Best Pipeline:\n")
        f.write(f"{tpot.fitted_pipeline_}\n\n")
        f.write(f"CV Score: {tpot.score(X_train, y_train):.3f}\n")
        f.write(f"Timestamp: {timestamp}\n")

    print(f"💾 Saved metadata to: {metadata_path.name}")

    # Evaluate on training set
    print("\n" + "=" * 80)
    print("📊 TRAINING SET EVALUATION")
    print("=" * 80)

    y_pred = tpot.fitted_pipeline_.predict(X_train)

    # Position-aware score
    position_score = fpl_comprehensive_team_scorer(
        y_train, y_pred, position_labels, formation="4-4-2"
    )
    print(f"\n🎯 Position-Aware Comprehensive Score: {position_score:.3f}")

    # Traditional metrics for comparison
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import spearmanr

    mae = mean_absolute_error(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    spearman = spearmanr(y_train, y_pred)[0]

    print("\n📈 Traditional Metrics (for comparison):")
    print(f"   MAE:     {mae:.3f}")
    print(f"   RMSE:    {rmse:.3f}")
    print(f"   Spearman: {spearman:.3f}")

    # Prediction distribution
    print("\n📊 Prediction Distribution:")
    print(f"   Mean: {np.mean(y_pred):.3f}")
    print(f"   Std:  {np.std(y_pred):.3f}")
    print(f"   Min:  {np.min(y_pred):.3f}")
    print(f"   Max:  {np.max(y_pred):.3f}")
    print(f"   Range: {np.max(y_pred) - np.min(y_pred):.3f}")

    print("\n📊 Actual Distribution:")
    print(f"   Mean: {np.mean(y_train):.3f}")
    print(f"   Std:  {np.std(y_train):.3f}")
    print(f"   Min:  {np.min(y_train):.3f}")
    print(f"   Max:  {np.max(y_train):.3f}")
    print(f"   Range: {np.max(y_train) - np.min(y_train):.3f}")

    print("\n" + "=" * 80)
    print("✅ TPOT POSITION-AWARE TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n📦 Model saved: {model_path}")
    print(f"📦 Metadata saved: {metadata_path}")

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "position_score": position_score,
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "pred_std": np.std(y_pred),
        "pred_range": np.max(y_pred) - np.min(y_pred),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train TPOT with position-aware scorer"
    )
    parser.add_argument("--start-gw", type=int, default=1, help="Start gameweek")
    parser.add_argument("--end-gw", type=int, default=9, help="End gameweek")
    parser.add_argument(
        "--max-time-mins", type=int, default=5, help="Max training time (minutes)"
    )
    parser.add_argument(
        "--population-size", type=int, default=20, help="TPOT population size"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="TPOT generations (None=unlimited)",
    )

    args = parser.parse_args()

    results = train_tpot_position_aware(
        start_gw=args.start_gw,
        end_gw=args.end_gw,
        max_time_mins=args.max_time_mins,
        population_size=args.population_size,
        generations=args.generations,
    )

    print(
        f"\n🎉 Training complete! Position-aware score: {results['position_score']:.3f}"
    )
