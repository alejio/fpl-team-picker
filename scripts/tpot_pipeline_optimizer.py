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
from dask.distributed import Client, LocalCluster

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client import FPLDataClient  # noqa: E402
from fpl_team_picker.domain.services.ml_feature_engineering import (  # noqa: E402
    FPLFeatureEngineer,
)
from fpl_team_picker.domain.services.ml_pipeline_factory import (  # noqa: E402
    get_team_strength_ratings,
)


class TemporalCVSplitter(BaseCrossValidator):
    """
    Custom CV splitter for temporal walk-forward validation.
    Wraps list of (train_idx, test_idx) tuples for sklearn compatibility.
    """

    def __init__(self, splits):
        """
        Args:
            splits: List of (train_idx, test_idx) tuples
        """
        self.splits = splits

    def split(self, X, y=None, groups=None):
        """Yield train/test indices."""
        for train_idx, test_idx in self.splits:
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        return len(self.splits)


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
        ],
        help="Scoring metric for TPOT (default: neg_mean_absolute_error). TPOT 1.1.0 uses 'scorers' parameter.",
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


def load_historical_data(
    start_gw: int, end_gw: int
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Load historical gameweek performance data and enhanced data sources.

    Args:
        start_gw: Starting gameweek
        end_gw: Ending gameweek (inclusive)

    Returns:
        Tuple of (historical_df, fixtures_df, teams_df, ownership_trends_df, value_analysis_df, fixture_difficulty_df)
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

    print(f"\n✅ Total records: {len(historical_df):,}")
    print(f"   Unique players: {historical_df['player_id'].nunique():,}")

    return (
        historical_df,
        fixtures_df,
        teams_df,
        ownership_trends_df,
        value_analysis_df,
        fixture_difficulty_df,
    )


def engineer_features(
    historical_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    ownership_trends_df: pd.DataFrame,
    value_analysis_df: pd.DataFrame,
    fixture_difficulty_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Engineer features using production FPLFeatureEngineer.

    Args:
        historical_df: Historical gameweek performance
        fixtures_df: Fixture data
        teams_df: Team data
        ownership_trends_df: Ownership trends data (Issue #37)
        value_analysis_df: Value analysis data (Issue #37)
        fixture_difficulty_df: Enhanced fixture difficulty data (Issue #37)

    Returns:
        Tuple of (features_df, feature_column_names)
    """
    print(
        "\n🔧 Engineering features (production FPLFeatureEngineer with 80 features)..."
    )

    # Get team strength ratings
    # TODO: Team strength hardcoded. Should be dynamically calculated each gw
    team_strength = get_team_strength_ratings()

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
    # TODO: is this fallback required?
    if "position" in historical_df_sorted.columns:
        features_df["position"] = historical_df_sorted["position"].values

    # Get production feature names
    production_feature_cols = list(feature_engineer.get_feature_names_out())

    print(f"   ✅ Created {len(production_feature_cols)} features")
    print(f"   Total samples: {len(features_df):,}")

    return features_df, production_feature_cols


def create_temporal_cv_splits(
    features_df: pd.DataFrame, max_folds: int = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create temporal walk-forward cross-validation splits.

    Train on GW6 to N-1, test on GW N (iterate forward).

    Args:
        features_df: Feature-engineered DataFrame with 'gameweek' column
        max_folds: Maximum number of folds (default: None = all available)

    Returns:
        List of (train_idx, test_idx) tuples
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

        # Get positional indices (not DataFrame .index values)
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

    # Return both splits and the filtered data
    return cv_splits, cv_data


def run_tpot_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list,
    args: argparse.Namespace,
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
    cv_splitter = TemporalCVSplitter(cv_splits)

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
            scorers=[args.scorer],  # TPOT 1.1.0 uses 'scorers' list
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
        cv_data: CV data with metadata
    """
    print("\n📊 Evaluating best pipeline...")

    # Get predictions
    y_pred = tpot.predict(X)

    # Overall metrics
    mae = np.abs(y_pred - y).mean()
    rmse = np.sqrt(((y_pred - y) ** 2).mean())
    r2 = 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    print("\n📈 Overall Metrics:")
    print(f"   MAE:  {mae:.3f} points")
    print(f"   RMSE: {rmse:.3f} points")
    print(f"   R²:   {r2:.3f}")

    # Position-specific metrics
    if "position" in cv_data.columns:
        print("\n📊 Position-Specific MAE:")
        cv_data_eval = cv_data.copy()
        cv_data_eval["predicted"] = y_pred
        cv_data_eval["actual"] = y.values

        for position in sorted(cv_data_eval["position"].unique()):
            pos_mask = cv_data_eval["position"] == position
            pos_mae = np.abs(
                cv_data_eval.loc[pos_mask, "predicted"]
                - cv_data_eval.loc[pos_mask, "actual"]
            ).mean()
            print(f"   {position}: {pos_mae:.3f} points")

    # Gameweek-specific metrics
    print("\n📅 Gameweek-Specific MAE:")
    cv_data_eval = cv_data.copy()
    cv_data_eval["predicted"] = y_pred
    cv_data_eval["actual"] = y.values

    for gw in sorted(cv_data_eval["gameweek"].unique()):
        gw_mask = cv_data_eval["gameweek"] == gw
        gw_mae = np.abs(
            cv_data_eval.loc[gw_mask, "predicted"] - cv_data_eval.loc[gw_mask, "actual"]
        ).mean()
        print(f"   GW{gw}: {gw_mae:.3f} points")


def main():
    """Main execution function."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("TPOT Pipeline Optimizer for FPL Expected Points")
    print("=" * 80)

    try:
        # 1. Load historical data (includes enhanced data sources)
        (
            historical_df,
            fixtures_df,
            teams_df,
            ownership_trends_df,
            value_analysis_df,
            fixture_difficulty_df,
        ) = load_historical_data(args.start_gw, args.end_gw)

        # 2. Engineer features (80 features: 65 base + 15 enhanced)
        features_df, feature_cols = engineer_features(
            historical_df,
            fixtures_df,
            teams_df,
            ownership_trends_df,
            value_analysis_df,
            fixture_difficulty_df,
        )

        # 3. Create temporal CV splits
        cv_splits, cv_data = create_temporal_cv_splits(
            features_df, max_folds=args.cv_folds
        )

        # 4. Prepare X and y
        # TODO: why are we doing fillna? What values are missing?
        X = cv_data[feature_cols].fillna(0)
        y = cv_data["total_points"]

        print("\n📊 Final training data shape:")
        print(f"   X: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"   y: {len(y)} targets")

        # 5. Run TPOT optimization
        tpot = run_tpot_optimization(X, y, cv_splits, args)

        # 6. Export pipeline
        pipeline_file = export_pipeline(tpot, args.output_dir, args)

        # 7. Evaluate pipeline
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
