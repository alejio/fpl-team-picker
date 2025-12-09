#!/usr/bin/env python3
"""
Phase 1: Data Preparation & Analysis for Probabilistic XP Calibration

Analyzes historical gameweek data (defaults to GW1 to latest available) to:
1. Categorize players by tier (Premium, Mid, Budget) and fixture difficulty (Easy, Hard)
2. Calculate sample sizes per combination
3. Identify combinations with insufficient data (<30 samples)
4. Generate data quality report with recommendations

Usage:
    # Auto-detect latest gameweek (default)
    python scripts/analyze_calibration_data.py

    # Custom gameweek range
    python scripts/analyze_calibration_data.py --start-gw 1 --end-gw 15

    # With custom minimum sample size
    python scripts/analyze_calibration_data.py --min-samples 30

    # Save summary to file
    python scripts/analyze_calibration_data.py --output data/calibration/summary.csv
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import typer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add fpl-dataset-builder to path for client import
dataset_builder_path = project_root.parent / "fpl-dataset-builder"
if str(dataset_builder_path) not in sys.path:
    sys.path.insert(0, str(dataset_builder_path))

from client import FPLDataClient  # noqa: E402


def get_player_tier(price: float) -> str:
    """Determine player tier from price.

    Args:
        price: Player price in millions (e.g., 8.5)

    Returns:
        "premium", "mid", or "budget"
    """
    if price >= 8.0:
        return "premium"
    elif price >= 6.0:
        return "mid"
    else:
        return "budget"


def get_fixture_category(fixture_difficulty: float, threshold: float = 1.538) -> str:
    """Determine fixture category (2-tier only for simplicity).

    Args:
        fixture_difficulty: Overall fixture difficulty rating
        threshold: Threshold for easy vs hard (default 1.538)

    Returns:
        "easy" or "hard"
    """
    if pd.isna(fixture_difficulty):
        return "unknown"
    if fixture_difficulty >= threshold:
        return "easy"
    else:
        return "hard"


def detect_latest_gameweek(client: FPLDataClient, max_check: int = 38) -> int:
    """Detect the latest gameweek with available performance data.

    Args:
        client: FPLDataClient instance
        max_check: Maximum gameweek to check (default 38)

    Returns:
        Latest gameweek number with data, or 1 if none found
    """
    print("\n🔍 Detecting latest available gameweek...")

    # Check backwards from max_check to find latest available
    for gw in range(max_check, 0, -1):
        try:
            gw_performance = client.get_gameweek_performance(gw)
            if not gw_performance.empty:
                print(f"   ✅ Latest gameweek with data: GW{gw}")
                return gw
        except Exception:
            continue

    print("   ⚠️  No gameweek data found, defaulting to GW1")
    return 1


def load_historical_data(
    start_gw: int, end_gw: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical gameweek performance and fixture difficulty data.

    Args:
        start_gw: Starting gameweek (inclusive)
        end_gw: Ending gameweek (inclusive). If None, auto-detects latest available.

    Returns:
        Tuple of (performance_df, fixture_difficulty_df)
    """
    client = FPLDataClient()

    # Auto-detect latest gameweek if not specified
    if end_gw is None:
        end_gw = detect_latest_gameweek(client)

    print(f"\n📥 Loading historical data (GW{start_gw} to GW{end_gw})...")

    # Load gameweek performance data
    performance_data = []
    for gw in range(start_gw, end_gw + 1):
        gw_performance = client.get_gameweek_performance(gw)
        if not gw_performance.empty:
            gw_performance["gameweek"] = gw
            performance_data.append(gw_performance)
            print(f"   ✅ GW{gw}: {len(gw_performance)} players")
        else:
            print(f"   ⚠️  GW{gw}: No data available")

    if not performance_data:
        raise ValueError("No historical data loaded. Check gameweek range.")

    performance_df = pd.concat(performance_data, ignore_index=True)

    # Load fixture difficulty data
    print("\n🏟️  Loading fixture difficulty data...")
    fixture_difficulty_df = client.get_derived_fixture_difficulty()
    print(f"   ✅ Fixture difficulty: {len(fixture_difficulty_df)} records")

    return performance_df, fixture_difficulty_df


def enrich_with_fixture_difficulty(
    performance_df: pd.DataFrame, fixture_difficulty_df: pd.DataFrame
) -> pd.DataFrame:
    """Join performance data with fixture difficulty.

    Args:
        performance_df: Gameweek performance data
        fixture_difficulty_df: Fixture difficulty data

    Returns:
        Enriched performance dataframe with fixture difficulty
    """
    print("\n🔗 Joining performance data with fixture difficulty...")

    # Prepare fixture difficulty for joining
    # Match on: team_id, opponent_team (from performance) = opponent_id (from fixture), gameweek, was_home = is_home
    fixture_join = fixture_difficulty_df[
        ["team_id", "opponent_id", "gameweek", "is_home", "overall_difficulty"]
    ].copy()

    # Rename for join
    fixture_join = fixture_join.rename(
        columns={"opponent_id": "opponent_team", "is_home": "was_home"}
    )

    # Join
    enriched_df = performance_df.merge(
        fixture_join,
        on=["team_id", "opponent_team", "gameweek", "was_home"],
        how="left",
    )

    missing_count = enriched_df["overall_difficulty"].isna().sum()
    total_count = len(enriched_df)
    match_rate = (1 - missing_count / total_count) * 100 if total_count > 0 else 0

    print(
        f"   ✅ Joined: {match_rate:.1f}% matched ({total_count - missing_count}/{total_count})"
    )
    if missing_count > 0:
        print(f"   ⚠️  Warning: {missing_count} records missing fixture difficulty")

    return enriched_df


def categorize_players(performance_df: pd.DataFrame) -> pd.DataFrame:
    """Categorize players by tier and fixture difficulty.

    Args:
        performance_df: Performance data with fixture difficulty

    Returns:
        Dataframe with tier and fixture_category columns
    """
    print("\n📊 Categorizing players by tier and fixture difficulty...")

    result_df = performance_df.copy()

    # Convert price from 0.1M units to millions
    result_df["price_millions"] = result_df["value"] / 10.0

    # Categorize by tier
    result_df["tier"] = result_df["price_millions"].apply(get_player_tier)

    # Categorize by fixture difficulty
    result_df["fixture_category"] = result_df["overall_difficulty"].apply(
        get_fixture_category
    )

    # Filter out records with missing data
    initial_count = len(result_df)
    result_df = result_df[
        result_df["tier"].notna()
        & result_df["fixture_category"].notna()
        & (result_df["fixture_category"] != "unknown")
        & result_df["total_points"].notna()
    ].copy()

    filtered_count = initial_count - len(result_df)
    if filtered_count > 0:
        print(f"   ⚠️  Filtered out {filtered_count} records with missing data")

    print(f"   ✅ Categorized {len(result_df)} player observations")

    return result_df


def calculate_sample_sizes(categorized_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sample sizes and statistics per combination.

    Args:
        categorized_df: Categorized performance data

    Returns:
        Dataframe with sample sizes and statistics per combination
    """
    print("\n📈 Calculating sample sizes per combination...")

    combinations = []

    for tier in ["premium", "mid", "budget"]:
        for fixture in ["easy", "hard"]:
            combo_df = categorized_df[
                (categorized_df["tier"] == tier)
                & (categorized_df["fixture_category"] == fixture)
            ]

            if len(combo_df) > 0:
                points = combo_df["total_points"].values
                combinations.append(
                    {
                        "combination": f"{tier}_{fixture}",
                        "tier": tier,
                        "fixture": fixture,
                        "sample_size": len(combo_df),
                        "mean_points": points.mean(),
                        "std_points": points.std(),
                        "min_points": points.min(),
                        "max_points": points.max(),
                        "percentile_25": pd.Series(points).quantile(0.25),
                        "percentile_50": pd.Series(points).quantile(0.50),
                        "percentile_75": pd.Series(points).quantile(0.75),
                    }
                )

    summary_df = pd.DataFrame(combinations)

    if not summary_df.empty:
        summary_df = summary_df.sort_values("combination")
        print(f"   ✅ Calculated statistics for {len(summary_df)} combinations")

    return summary_df


def generate_report(summary_df: pd.DataFrame, minimum_sample_size: int = 30) -> str:
    """Generate data quality report.

    Args:
        summary_df: Summary statistics per combination
        minimum_sample_size: Minimum samples required (default 30)

    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PROBABILISTIC XP CALIBRATION - DATA QUALITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Overall summary
    total_samples = summary_df["sample_size"].sum()
    report_lines.append(f"Total Player Observations: {total_samples:,}")
    report_lines.append(f"Combinations Analyzed: {len(summary_df)}")
    report_lines.append(f"Minimum Sample Size Required: {minimum_sample_size}")
    report_lines.append("")

    # Combination details
    report_lines.append("-" * 80)
    report_lines.append("COMBINATION DETAILS")
    report_lines.append("-" * 80)
    report_lines.append("")

    for _, row in summary_df.iterrows():
        combo = row["combination"]
        n = row["sample_size"]
        mean_pts = row["mean_points"]
        std_pts = row["std_points"]
        status = "✅" if n >= minimum_sample_size else "⚠️ "

        report_lines.append(f"{status} {combo.upper()}")
        report_lines.append(f"   Sample Size: {n}")
        report_lines.append(f"   Mean Points: {mean_pts:.2f}")
        report_lines.append(f"   Std Points: {std_pts:.2f}")
        report_lines.append(
            f"   Range: {row['min_points']:.0f} - {row['max_points']:.0f}"
        )
        report_lines.append(
            f"   Percentiles: 25th={row['percentile_25']:.1f}, 50th={row['percentile_50']:.1f}, 75th={row['percentile_75']:.1f}"
        )
        if n < minimum_sample_size:
            report_lines.append(
                f"   ⚠️  WARNING: Insufficient samples (need {minimum_sample_size}, have {n})"
            )
        report_lines.append("")

    # Recommendations
    report_lines.append("-" * 80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 80)
    report_lines.append("")

    insufficient = summary_df[summary_df["sample_size"] < minimum_sample_size]
    sufficient = summary_df[summary_df["sample_size"] >= minimum_sample_size]

    if len(insufficient) > 0:
        report_lines.append(
            f"⚠️  {len(insufficient)} combinations have insufficient data:"
        )
        for _, row in insufficient.iterrows():
            report_lines.append(
                f"   - {row['combination']}: {row['sample_size']} samples (need {minimum_sample_size})"
            )
        report_lines.append("")
        report_lines.append(
            "   RECOMMENDATION: Use fallback to prior for these combinations."
        )
        report_lines.append("")
    else:
        report_lines.append("✅ All combinations have sufficient data.")
        report_lines.append("")

    if len(sufficient) > 0:
        report_lines.append(
            f"✅ {len(sufficient)} combinations are ready for distribution fitting:"
        )
        for _, row in sufficient.iterrows():
            report_lines.append(
                f"   - {row['combination']}: {row['sample_size']} samples"
            )
        report_lines.append("")

    # Expected sample sizes (from plan)
    report_lines.append("-" * 80)
    report_lines.append("EXPECTED SAMPLE SIZES (from plan)")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append("Premium + Easy:   ~50-80  (adequate)")
    report_lines.append("Premium + Hard:   ~100-150 (good)")
    report_lines.append("Mid + Easy:       ~150-200 (good)")
    report_lines.append("Mid + Hard:       ~200-250 (good)")
    report_lines.append("Budget + Easy:     ~200-300 (good)")
    report_lines.append("Budget + Hard:     ~300-400 (good)")
    report_lines.append("")

    # Next steps
    report_lines.append("-" * 80)
    report_lines.append("NEXT STEPS")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append("1. Review sample sizes above")
    report_lines.append(
        "2. If sufficient data, proceed to Phase 2: Distribution Fitting"
    )
    report_lines.append("3. If insufficient data, adjust thresholds or use fallbacks")
    report_lines.append("")

    return "\n".join(report_lines)


app = typer.Typer(
    help="Phase 1: Data Preparation & Analysis for Probabilistic XP Calibration. "
    "Automatically detects latest available gameweek by default."
)


@app.command()
def analyze(
    start_gw: int = typer.Option(
        1, "--start-gw", help="Starting gameweek (default: 1)"
    ),
    end_gw: Optional[int] = typer.Option(
        None,
        "--end-gw",
        help="Ending gameweek (default: auto-detect latest available gameweek with data)",
    ),
    min_samples: int = typer.Option(
        30, "--min-samples", help="Minimum sample size required (default: 30)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Output file path for summary CSV (optional)"
    ),
):
    """Analyze historical data for probabilistic XP calibration.

    Categorizes players by tier and fixture difficulty, calculates sample sizes,
    and generates a data quality report with recommendations.
    """
    try:
        # Load data (auto-detect latest gameweek if not specified)
        performance_df, fixture_difficulty_df = load_historical_data(start_gw, end_gw)

        # Enrich with fixture difficulty
        enriched_df = enrich_with_fixture_difficulty(
            performance_df, fixture_difficulty_df
        )

        # Categorize players
        categorized_df = categorize_players(enriched_df)

        # Calculate sample sizes
        summary_df = calculate_sample_sizes(categorized_df)

        # Generate and print report
        report = generate_report(summary_df, min_samples)
        print("\n" + report)

        # Save summary CSV if requested
        if output:
            summary_df.to_csv(output, index=False)
            print(f"\n💾 Summary saved to: {output}")

        # Save detailed categorized data for Phase 2
        output_dir = Path(__file__).parent.parent / "data" / "calibration"
        output_dir.mkdir(parents=True, exist_ok=True)
        categorized_output = output_dir / "categorized_data.csv"
        categorized_df.to_csv(categorized_output, index=False)
        print(f"💾 Categorized data saved to: {categorized_output}")

        print("\n✅ Analysis complete!")

    except Exception as e:
        typer.echo(f"\n❌ Error: {e}", err=True)
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
