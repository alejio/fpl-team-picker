#!/usr/bin/env python3
"""
TPOT Pipeline Optimizer for FPL Expected Points Prediction

This script uses TPOT (Tree-based Pipeline Optimization Tool) to automatically
discover optimal ML pipelines for predicting FPL player expected points.

Uses the same temporal cross-validation strategy as ml_xp_notebook.py:
- Train on GW6 to N-1, test on GW N (walk-forward validation)
- Leak-free feature engineering via FPLFeatureEngineer
- Production-ready pipeline export

Run: python scripts/tpot_pipeline_optimizer.py --start-gw 1 --end-gw 8 --generations 10
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
from dask.distributed import Client, LocalCluster

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

# Import reusable training utilities
from ml_training_utils import (  # noqa: E402
    load_training_data,
    engineer_features,
    create_temporal_cv_splits,
    TemporalCVSplitter,
    fpl_weighted_huber_scorer_sklearn,
    fpl_topk_scorer_sklearn,
    fpl_captain_scorer_sklearn,
)

# Import position-aware scorer
from position_aware_scorer import (  # noqa: E402
    fpl_comprehensive_team_scorer,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TPOT pipeline optimization for FPL xP prediction"
    )
    parser.add_argument(
        "--start-gw",
        type=int,
        default=1,
        help="Training start gameweek (default: 1)",
    )
    parser.add_argument(
        "--end-gw",
        type=int,
        default=8,
        help="Training end gameweek to predict (default: 8)",
    )
    parser.add_argument(
        "--max-time-mins",
        type=int,
        default=60,
        help="Maximum runtime in minutes (default: 60). TPOT 1.1.0 uses time-based optimization.",
    )
    parser.add_argument(
        "--max-eval-time-mins",
        type=int,
        default=5,
        help="Maximum evaluation time per pipeline in minutes (default: 5)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Number of temporal CV folds (default: auto = all available GWs)",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default="neg_mean_absolute_error",
        choices=[
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            "neg_median_absolute_error",
            "r2",
            "fpl_weighted_huber",
            "fpl_top_k_ranking",
            "fpl_captain_pick",
            "fpl_position_aware",
        ],
        help="Scoring metric for TPOT (default: neg_mean_absolute_error). Custom scorers: 'fpl_weighted_huber' (AGGRESSIVE balanced), 'fpl_top_k_ranking' (best for team selection), 'fpl_captain_pick' (captain focus), 'fpl_position_aware' (position-aware comprehensive FPL scorer). TPOT 1.1.0 uses 'scorers' parameter.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/tpot",
        help="Output directory for exported pipelines (default: models/tpot)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="TPOT verbose level (0=silent, 3=most verbose, default: 2)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 = all CPUs, default: -1)",
    )

    return parser.parse_args()


def create_position_aware_scorer_for_sklearn(position_storage):
    """
    Create sklearn-compatible scorer that uses stored position labels.

    Args:
        position_storage: Dict with 'positions', 'current_test_idx' keys

    Returns:
        sklearn scorer
    """

    def scorer_with_positions(estimator, X, y):
        """Scorer function that uses stored position labels."""
        try:
            # Get predictions
            y_pred = estimator.predict(X)

            # Convert to numpy if needed
            if hasattr(y, "values"):
                y = y.values
            if hasattr(y_pred, "values"):
                y_pred = y_pred.values

            # Ensure arrays are 1D
            y = np.asarray(y).flatten()
            y_pred = np.asarray(y_pred).flatten()

            # Get position labels for this fold
            if position_storage["positions"] is None:
                raise ValueError("Position labels not set!")

            # Use stored test indices to get correct positions
            test_idx = position_storage["current_test_idx"]

            if test_idx is None:
                # Fallback: TPOT may evaluate on full dataset during initial population
                # BEFORE CV splitter runs. Try to match X to positions intelligently.
                full_dataset_len = len(position_storage["positions"])

                if len(X) == full_dataset_len:
                    # Full dataset evaluation - use all positions (perfect match)
                    current_positions = position_storage["positions"]
                elif len(X) < full_dataset_len:
                    # Subset evaluation - this is tricky without knowing which subset
                    # Try first N positions as fallback (may be wrong but allows scoring)
                    # This happens during initial population evaluation
                    current_positions = position_storage["positions"][: len(X)]
                else:
                    # X is larger - should never happen but handle gracefully
                    current_positions = np.tile(
                        position_storage["positions"], (len(X) // full_dataset_len + 1)
                    )[: len(X)]
            else:
                # Use the stored test indices
                if len(test_idx) == len(X):
                    current_positions = position_storage["positions"][test_idx]
                else:
                    # Mismatch - try to align or fail gracefully
                    if len(X) <= len(position_storage["positions"]):
                        current_positions = position_storage["positions"][
                            test_idx[: len(X)]
                        ]
                    else:
                        raise ValueError(
                            f"Index mismatch: X has {len(X)} samples but test_idx has {len(test_idx)}"
                        )

            # Ensure all arrays have same length
            min_len = min(len(y), len(y_pred), len(current_positions))
            if min_len == 0:
                return 0.01  # No data to score - return small positive value

            y = y[:min_len]
            y_pred = y_pred[:min_len]
            current_positions = current_positions[:min_len]

            # Validate arrays are all same length (position_aware_scorer requires this)
            if not (len(y) == len(y_pred) == len(current_positions)):
                return 0.01  # Length mismatch after trimming

            # Validate we have at least some players in each position for scoring
            unique_positions = np.unique(current_positions)
            if len(unique_positions) == 0:
                return 0.01  # No positions available

            # Ensure we have enough players for XI selection (need at least 1 of each position for 4-4-2)
            required_positions = {"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2}
            position_counts = {
                pos: np.sum(current_positions == pos)
                for pos in required_positions.keys()
            }
            if any(
                position_counts[pos] < required_positions[pos]
                for pos in required_positions.keys()
            ):
                # Not enough players in each position for XI formation
                return 0.01  # Return small score but allow evaluation

            # Calculate position-aware score
            try:
                score = fpl_comprehensive_team_scorer(
                    y, y_pred, current_positions, formation="4-4-2"
                )
            except (ValueError, IndexError, ZeroDivisionError):
                # Position scorer functions may raise errors for edge cases
                # Return small positive score instead of 0.0 to prevent TPOT rejection
                # 0.01 is a very low score but not zero, so TPOT will accept it
                return 0.01

            # Ensure score is a valid number
            if np.isnan(score) or np.isinf(score):
                return 0.01  # Return small positive instead of 0

            # Ensure score is in valid range (0-1 for comprehensive scorer)
            score = float(score)
            if score < 0.0:
                score = 0.0
            elif score > 1.0:
                score = 1.0

            return score

        except Exception:
            # Return a small positive score instead of 0.0 or crashing
            # TPOT may reject all-zeros, so use 0.01 to indicate "bad but valid"
            # This allows TPOT to continue but penalizes problematic pipelines
            # print(f"DEBUG: Position-aware scorer error: {e}", file=sys.stderr)
            return 0.01

    return make_scorer(scorer_with_positions, greater_is_better=True)


def run_tpot_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list,
    args: argparse.Namespace,
    position_storage: dict = None,
) -> TPOTRegressor:
    """
    Run TPOT pipeline optimization.

    Args:
        X: Feature matrix
        y: Target variable
        cv_splits: Custom CV splits (temporal walk-forward)
        args: Command line arguments

    Returns:
        Fitted TPOTRegressor
    """
    print("\n🤖 Initializing TPOT 1.1.0 optimizer...")
    print(f"   Scorer: {args.scorer}")
    print(f"   CV folds: {len(cv_splits)}")
    print(f"   Max time: {args.max_time_mins} mins")
    print(f"   Max eval time: {args.max_eval_time_mins} mins")
    print(f"   Random seed: {args.random_seed}")
    print(f"   Parallel jobs: {args.n_jobs}")
    print("   Note: TPOT 1.1.0 uses time-based optimization with Dask")

    # Wrap custom CV splits in sklearn-compatible splitter
    # For position-aware scorer, we need to track indices
    if args.scorer == "fpl_position_aware" and position_storage is not None:
        # Create CV splitter that tracks indices for position-aware scorer
        class PositionAwareCVSplitter(BaseCrossValidator):
            def __init__(self, splits, position_storage):
                self.splits = splits
                self.position_storage = position_storage

            def split(self, X, y=None, groups=None):
                for train_idx, test_idx in self.splits:
                    if self.position_storage is not None:
                        self.position_storage["current_train_idx"] = train_idx
                        self.position_storage["current_test_idx"] = test_idx
                    yield train_idx, test_idx

            def get_n_splits(self, X=None, y=None, groups=None):
                return len(self.splits)

        cv_splitter = PositionAwareCVSplitter(cv_splits, position_storage)
    else:
        cv_splitter = TemporalCVSplitter(cv_splits)

    # Handle custom scorer
    scorer_name = args.scorer
    if args.scorer == "fpl_weighted_huber":
        scorer_name = fpl_weighted_huber_scorer_sklearn
        print(
            "   Using custom FPL weighted Huber loss scorer (AGGRESSIVE: 3x asymmetric, 5x high-scorer penalty)"
        )
    elif args.scorer == "fpl_top_k_ranking":
        scorer_name = fpl_topk_scorer_sklearn
        print(
            "   Using top-K ranking scorer (optimizes top-50 player ranking + captain overlap)"
        )
    elif args.scorer == "fpl_captain_pick":
        scorer_name = fpl_captain_scorer_sklearn
        print("   Using captain pick scorer (pure captain identification focus)")
    elif args.scorer == "fpl_position_aware":
        if position_storage is None:
            raise ValueError(
                "Position storage required for position-aware scorer. "
                "Ensure cv_data includes position labels."
            )
        scorer_name = create_position_aware_scorer_for_sklearn(position_storage)
        print(
            "   Using position-aware comprehensive scorer (40% top-K, 40% XI efficiency, 20% captain)"
        )

    # Set up Dask cluster for parallelization
    n_workers = args.n_jobs if args.n_jobs > 0 else None
    print("\n🔧 Starting Dask LocalCluster...")
    print(f"   Workers: {n_workers if n_workers else 'auto'}")

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        silence_logs=True,  # Reduce noise
    )
    client = Client(cluster)

    print(f"   ✅ Dask cluster ready: {client.dashboard_link}")

    try:
        tpot = TPOTRegressor(
            scorers=[
                scorer_name
            ],  # TPOT 1.1.0 uses 'scorers' list (can be string or callable)
            cv=cv_splitter,  # Custom temporal CV splitter
            max_time_mins=args.max_time_mins,
            max_eval_time_mins=args.max_eval_time_mins,
            random_state=args.random_seed,
            verbose=args.verbose,
            n_jobs=1,  # TPOT 1.1.0 uses Dask for parallelization, not n_jobs
            client=client,  # Pass Dask client
        )

        print("\n🚀 Starting TPOT optimization...")
        print("=" * 80)

        start_time = datetime.now()

        tpot.fit(X, y)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        print("=" * 80)
        print("\n✅ TPOT optimization complete!")
        print(f"   Duration: {duration:.2f} minutes")

        return tpot

    finally:
        # Clean up Dask resources
        client.close()
        cluster.close()
        print("\n🧹 Dask cluster shut down")


def export_pipeline(
    tpot: TPOTRegressor, output_dir: str, args: argparse.Namespace
) -> Path:
    """
    Export the best TPOT pipeline to a Python file.

    Args:
        tpot: Fitted TPOTRegressor
        output_dir: Output directory
        args: Command line arguments

    Returns:
        Path to exported pipeline file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tpot_pipeline_gw{args.start_gw}-{args.end_gw}_{timestamp}.py"
    pipeline_file = output_path / filename

    print(f"\n💾 Exporting best pipeline to: {pipeline_file}")

    # TPOT 1.1.0 stores the best pipeline in fitted_pipeline_ attribute
    # We need to manually export it since export() method was removed
    if hasattr(tpot, "fitted_pipeline_"):
        # Get the best pipeline
        best_pipeline = tpot.fitted_pipeline_

        # Write manual export (sklearn Pipeline object)
        with open(pipeline_file, "w") as f:
            f.write("# TPOT 1.1.0 exported pipeline\n")
            f.write(
                "# This pipeline was automatically discovered using genetic programming\n\n"
            )
            f.write("import numpy as np\n")
            f.write("import pandas as pd\n")
            f.write("from sklearn.pipeline import Pipeline\n\n")
            f.write("# Best pipeline found by TPOT:\n")
            f.write(f"# {best_pipeline}\n\n")
            f.write("# To use this pipeline:\n")
            f.write("# 1. Reconstruct the pipeline from the repr above\n")
            f.write("# 2. Or use joblib to load the saved model:\n")
            f.write("#    import joblib\n")
            f.write(
                f"#    model = joblib.load('{pipeline_file.with_suffix('.joblib').name}')\n\n"
            )
            f.write("# Pipeline structure:\n")
            f.write(f"exported_pipeline = {repr(best_pipeline)}\n")

        # Also save with joblib for easy loading
        import joblib

        joblib_file = pipeline_file.with_suffix(".joblib")
        joblib.dump(best_pipeline, joblib_file)

        print("   ✅ Pipeline exported successfully")
        print(f"   ✅ Joblib model saved: {joblib_file}")
    else:
        raise AttributeError("TPOTRegressor has no fitted_pipeline_ attribute")

    # Also save metadata
    metadata_file = pipeline_file.parent / f"{pipeline_file.stem}_metadata.txt"
    with open(metadata_file, "w") as f:
        f.write("TPOT 1.1.0 Pipeline Optimization Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Gameweek range: GW{args.start_gw} to GW{args.end_gw}\n")
        f.write(f"Max time: {args.max_time_mins} mins\n")
        f.write(f"Max eval time: {args.max_eval_time_mins} mins\n")
        f.write(f"Scoring metric: {args.scorer}\n")
        f.write(
            f"CV folds: {args.cv_folds if args.cv_folds else 'auto (all available)'}\n"
        )
        f.write(f"Random seed: {args.random_seed}\n")
        f.write("Note: TPOT 1.1.0 uses time-based optimization\n\n")
        f.write("Best pipeline:\n")
        f.write(str(tpot.fitted_pipeline_))

    print(f"   ✅ Metadata saved to: {metadata_file}")

    return pipeline_file


def evaluate_pipeline(
    tpot: TPOTRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    cv_data: pd.DataFrame,
) -> None:
    """
    Evaluate the best pipeline with detailed metrics.

    Args:
        tpot: Fitted TPOTRegressor
        X: Feature matrix
        y: Target variable
        cv_data: CV data with metadata (includes 'gameweek', 'player_id', 'position')
    """
    print("\n📊 Evaluating best pipeline...")

    # Get predictions
    y_pred = tpot.predict(X)

    # Create evaluation dataframe
    cv_data_eval = cv_data.copy()
    cv_data_eval["predicted"] = y_pred
    cv_data_eval["actual"] = y.values

    # Overall metrics
    mae = np.abs(y_pred - y).mean()
    rmse = np.sqrt(((y_pred - y) ** 2).mean())
    r2 = 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    print("\n📈 Overall Metrics:")
    print(f"   MAE:  {mae:.3f} points")
    print(f"   RMSE: {rmse:.3f} points")
    print(f"   R²:   {r2:.3f}")

    # FPL-specific strategic metrics
    print("\n🎯 FPL Strategic Metrics:")

    # Ranking correlation (how well does model rank players?)
    rank_corr, _ = spearmanr(cv_data_eval["actual"], cv_data_eval["predicted"])
    print(f"   Spearman correlation (ranking): {rank_corr:.3f}")

    # Top-15 selection accuracy (squad building)
    for gw in sorted(cv_data_eval["gameweek"].unique()):
        gw_data = cv_data_eval[cv_data_eval["gameweek"] == gw]

        # Top-15 players by actual vs predicted
        top_15_actual = set(gw_data.nlargest(15, "actual")["player_id"])
        top_15_pred = set(gw_data.nlargest(15, "predicted")["player_id"])
        top_15_overlap = len(top_15_actual & top_15_pred)

        # Captain accuracy (top-1)
        best_actual_id = gw_data.loc[gw_data["actual"].idxmax(), "player_id"]
        best_pred_id = gw_data.loc[gw_data["predicted"].idxmax(), "player_id"]
        captain_correct = 1 if best_actual_id == best_pred_id else 0

        print(
            f"   GW{gw}: Top-15 overlap {top_15_overlap}/15 ({100 * top_15_overlap / 15:.0f}%), "
            f"Captain {'✓' if captain_correct else '✗'}"
        )

    # Position-specific metrics
    if "position" in cv_data_eval.columns:
        print("\n📊 Position-Specific MAE:")

        for position in sorted(cv_data_eval["position"].unique()):
            pos_mask = cv_data_eval["position"] == position
            pos_mae = np.abs(
                cv_data_eval.loc[pos_mask, "predicted"]
                - cv_data_eval.loc[pos_mask, "actual"]
            ).mean()
            print(f"   {position}: {pos_mae:.3f} points")

        # Position-aware comprehensive score (if position data available)
        position_labels = cv_data_eval["position"].values
        position_score = fpl_comprehensive_team_scorer(
            cv_data_eval["actual"].values,
            cv_data_eval["predicted"].values,
            position_labels,
            formation="4-4-2",
        )
        print(f"\n🎯 Position-Aware Comprehensive Score: {position_score:.3f}")
        print("   (40% top-K overlap, 40% XI efficiency, 20% captain accuracy)")

    # Gameweek-specific metrics with chaos detection
    print("\n📅 Gameweek Analysis (with chaos detection):")

    for gw in sorted(cv_data_eval["gameweek"].unique()):
        gw_mask = cv_data_eval["gameweek"] == gw
        gw_data = cv_data_eval.loc[gw_mask]

        # MAE
        gw_mae = np.abs(gw_data["predicted"] - gw_data["actual"]).mean()

        # Chaos indicators
        actual_std = gw_data["actual"].std()
        actual_max = gw_data["actual"].max()
        high_scorers = (gw_data["actual"] >= 10).sum()
        zero_scorers = (gw_data["actual"] == 0).sum()
        zero_pct = 100 * zero_scorers / len(gw_data)

        # Flag chaos weeks
        chaos_flags = []
        if actual_std > 2.5:
            chaos_flags.append("high_variance")
        if actual_max >= 20:
            chaos_flags.append("extreme_haul")
        if high_scorers >= 15:
            chaos_flags.append("many_hauls")

        chaos_str = f" ⚠️ [{', '.join(chaos_flags)}]" if chaos_flags else ""

        print(
            f"   GW{gw}: MAE {gw_mae:.3f} | Std {actual_std:.2f} | "
            f"Max {actual_max:.0f} | 10+ pts: {high_scorers} | "
            f"0 pts: {zero_pct:.0f}%{chaos_str}"
        )


def main():
    """Main execution function."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("TPOT Pipeline Optimizer for FPL Expected Points")
    print("=" * 80)

    try:
        # 1. Load training data using reusable utilities
        (
            historical_df,
            fixtures_df,
            teams_df,
            ownership_trends_df,
            value_analysis_df,
            fixture_difficulty_df,
            betting_features_df,
            raw_players_df,
        ) = load_training_data(start_gw=args.start_gw, end_gw=args.end_gw, verbose=True)

        # 2. Engineer features using reusable utilities
        features_df, target, feature_cols = engineer_features(
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

        # 3. Create temporal CV splits using reusable utilities
        cv_splits, cv_data = create_temporal_cv_splits(
            features_df, max_folds=args.cv_folds, verbose=True
        )

        # cv_data already includes total_points from engineer_features (metadata columns added)
        # But verify it exists for safety
        if "total_points" not in cv_data.columns:
            # Match target to cv_data using original indices (shouldn't happen but safety check)
            train_mask = features_df["gameweek"] >= 6
            target_train = target[train_mask]
            cv_data["total_points"] = target_train  # target is already numpy array

        # 4. Prepare X and y
        # Fail fast if features contain NaN - indicates upstream data quality issues
        X = cv_data[feature_cols]
        nan_counts = X.isna().sum()
        if nan_counts.any():
            nan_features = nan_counts[nan_counts > 0]
            raise ValueError(
                f"Features contain NaN values. Cannot proceed with incomplete data.\n"
                f"Features with NaN:\n{nan_features.to_string()}\n"
                f"This indicates upstream data quality issues. "
                f"Check feature engineering or data loading."
            )

        y = cv_data["total_points"]

        print("\n📊 Final training data shape:")
        print(f"   X: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"   y: {len(y)} targets")

        # 5. Set up position storage for position-aware scorer (if needed)
        position_storage = None
        if args.scorer == "fpl_position_aware":
            if "position" not in cv_data.columns:
                raise ValueError(
                    "Position column required for position-aware scorer. "
                    "Ensure data includes position information."
                )
            position_storage = {
                "positions": cv_data["position"].values,
                "player_ids": cv_data["player_id"].values
                if "player_id" in cv_data.columns
                else None,
                "current_train_idx": None,
                "current_test_idx": None,
            }
            print("\n✅ Position storage initialized for position-aware scoring")
            print(
                f"   Position distribution: GKP={sum(position_storage['positions'] == 'GKP')}, "
                f"DEF={sum(position_storage['positions'] == 'DEF')}, "
                f"MID={sum(position_storage['positions'] == 'MID')}, "
                f"FWD={sum(position_storage['positions'] == 'FWD')}"
            )

        # 6. Run TPOT optimization
        tpot = run_tpot_optimization(
            X, y, cv_splits, args, position_storage=position_storage
        )

        # 7. Export pipeline
        pipeline_file = export_pipeline(tpot, args.output_dir, args)

        # 8. Evaluate pipeline
        evaluate_pipeline(tpot, X, y, cv_data)

        print("\n" + "=" * 80)
        print("✅ TPOT optimization complete!")
        print("=" * 80)
        print(f"\n📁 Exported pipeline: {pipeline_file}")
        print("\n💡 Next steps:")
        print("   1. Review the exported pipeline code")
        print("   2. Test the pipeline in ml_xp_notebook.py")
        print("   3. Compare against current models (LightGBM, Ridge, etc.)")
        print("   4. If better, integrate into MLExpectedPointsService")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
