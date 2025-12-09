#!/usr/bin/env python3
"""
Phase 2: Distribution Fitting for Probabilistic XP Calibration

Fits regularized Negative Binomial distributions for each (tier, fixture) combination
using Maximum Likelihood Estimation with L2 regularization toward priors.

Usage:
    # Use default categorized data from Phase 1
    python scripts/build_calibration_distributions.py

    # Custom input file
    python scripts/build_calibration_distributions.py --input data/calibration/categorized_data.csv

    # Custom regularization strength
    python scripts/build_calibration_distributions.py --lambda-reg 0.5
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import typer
from scipy.optimize import minimize
from scipy.stats import nbinom

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Prior specification (domain knowledge)
PRIORS = {
    "tier_means": {
        "premium": 8.0,  # Premium players average ~8 points
        "mid": 6.0,  # Mid players average ~6 points
        "budget": 4.5,  # Budget players average ~4.5 points
    },
    "fixture_boosts": {
        "easy": 2.0,  # Easy fixtures: +2 points
        "hard": -1.0,  # Hard fixtures: -1 point
    },
    "dispersion": 2.0,  # Moderate overdispersion
}


def get_prior_mean(tier: str, fixture: str) -> float:
    """Calculate prior mean for a (tier, fixture) combination.

    Args:
        tier: Player tier ("premium", "mid", "budget")
        fixture: Fixture category ("easy", "hard")

    Returns:
        Prior mean points
    """
    base_mean = PRIORS["tier_means"].get(tier, 5.0)
    fixture_boost = PRIORS["fixture_boosts"].get(fixture, 0.0)
    return base_mean + fixture_boost


def fit_regularized_negative_binomial(
    data: np.ndarray,
    prior_mean: float,
    prior_dispersion: float = 2.0,
    lambda_reg: float = 0.3,
) -> Tuple[float, float, Dict]:
    """Fit Negative Binomial with L2 regularization toward prior.

    Uses scipy's nbinom parameterization: nbinom(n, p) where:
    - n = dispersion parameter (r)
    - p = probability parameter
    - mean = n * (1 - p) / p
    - variance = n * (1 - p) / p²

    Args:
        data: Array of actual points scored (non-negative integers)
        prior_mean: Prior mean (from domain knowledge)
        prior_dispersion: Prior dispersion parameter (default 2.0)
        lambda_reg: Regularization strength (higher = more shrinkage)

    Returns:
        Tuple of (fitted_mean, fitted_dispersion, fit_info)
    """
    # Ensure data is non-negative integers
    data = np.array(data, dtype=int)
    data = np.maximum(data, 0)  # Ensure non-negative

    if len(data) == 0:
        return prior_mean, prior_dispersion, {"error": "No data provided"}

    # Convert to scipy nbinom parameters
    # We'll optimize in (mean, dispersion) space and convert to (n, p)
    def negative_log_likelihood(params):
        mean, dispersion = params

        # Ensure positive values
        mean = max(0.1, mean)
        dispersion = max(0.1, dispersion)

        # Convert to scipy nbinom parameters
        # mean = n * (1 - p) / p  =>  p = n / (n + mean)
        n = dispersion
        p = n / (n + mean)

        # Ensure p is in valid range [0, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)

        try:
            # Log likelihood
            ll = np.sum(nbinom.logpmf(data, n, p))

            # L2 regularization (shrink toward prior)
            reg = lambda_reg * (
                (mean - prior_mean) ** 2 + (dispersion - prior_dispersion) ** 2
            )

            return -(ll - reg)  # Negative log likelihood + regularization
        except Exception:
            # Return large penalty if optimization fails
            return 1e10

    # Initial guess (use prior)
    x0 = [prior_mean, prior_dispersion]

    # Optimize
    try:
        result = minimize(
            negative_log_likelihood,
            x0=x0,
            method="L-BFGS-B",
            bounds=[(0.1, 20.0), (0.1, 10.0)],  # Reasonable bounds
            options={"maxiter": 1000, "ftol": 1e-6},
        )

        if result.success:
            fitted_mean = max(0.1, result.x[0])
            fitted_dispersion = max(0.1, result.x[1])

            # Calculate standard deviation from fitted parameters
            n = fitted_dispersion
            p = n / (n + fitted_mean)
            np.sqrt(n * (1 - p) / (p**2))

            fit_info = {
                "success": True,
                "n_iterations": result.nit,
                "final_loss": result.fun,
            }
        else:
            # Fallback to prior if optimization fails
            fitted_mean = prior_mean
            fitted_dispersion = prior_dispersion
            n = fitted_dispersion
            p = n / (n + fitted_mean)
            np.sqrt(n * (1 - p) / (p**2))

            fit_info = {
                "success": False,
                "error": result.message,
                "used_prior": True,
            }
    except Exception as e:
        # Fallback to prior on error
        fitted_mean = prior_mean
        fitted_dispersion = prior_dispersion
        n = fitted_dispersion
        p = n / (n + fitted_mean)
        np.sqrt(n * (1 - p) / (p**2))

        fit_info = {"success": False, "error": str(e), "used_prior": True}

    return fitted_mean, fitted_dispersion, fit_info


def fit_distributions(
    categorized_df: pd.DataFrame,
    lambda_reg: float = 0.3,
    minimum_sample_size: int = 30,
) -> Dict:
    """Fit distributions for all combinations in the data.

    Args:
        categorized_df: Categorized performance data from Phase 1
        lambda_reg: Regularization strength (default 0.3)
        minimum_sample_size: Minimum samples required (default 30)

    Returns:
        Dictionary of fitted distributions
    """
    print("\n🔧 Fitting Negative Binomial distributions...")
    print(f"   Regularization strength (λ): {lambda_reg}")
    print(f"   Minimum sample size: {minimum_sample_size}")

    distributions = {}

    # Get unique combinations
    for tier in ["premium", "mid", "budget"]:
        for fixture in ["easy", "hard"]:
            combo_name = f"{tier}_{fixture}"
            combo_df = categorized_df[
                (categorized_df["tier"] == tier)
                & (categorized_df["fixture_category"] == fixture)
            ]

            if len(combo_df) == 0:
                print(f"   ⚠️  {combo_name}: No data available (skipping)")
                continue

            points = combo_df["total_points"].values
            sample_size = len(points)

            # Get prior
            prior_mean = get_prior_mean(tier, fixture)
            prior_dispersion = PRIORS["dispersion"]

            if sample_size < minimum_sample_size:
                print(
                    f"   ⚠️  {combo_name}: {sample_size} samples (need {minimum_sample_size}, using prior)"
                )
                # Use prior only
                fitted_mean = prior_mean
                fitted_dispersion = prior_dispersion
                fit_info = {
                    "success": True,
                    "used_prior": True,
                    "reason": "insufficient_samples",
                }
            else:
                print(f"   🔄 {combo_name}: {sample_size} samples (fitting...)")
                fitted_mean, fitted_dispersion, fit_info = (
                    fit_regularized_negative_binomial(
                        points, prior_mean, prior_dispersion, lambda_reg
                    )
                )

            # Calculate statistics
            n = fitted_dispersion
            p = n / (n + fitted_mean)
            fitted_std = np.sqrt(n * (1 - p) / (p**2))

            # Calculate percentiles from fitted distribution
            percentile_25 = nbinom.ppf(0.25, n, p)
            percentile_50 = nbinom.ppf(0.50, n, p)
            percentile_75 = nbinom.ppf(0.75, n, p)

            # Store distribution
            distributions[combo_name] = {
                "mean": float(fitted_mean),
                "dispersion": float(fitted_dispersion),
                "std": float(fitted_std),
                "sample_size": int(sample_size),
                "percentile_25": float(percentile_25),
                "percentile_50": float(percentile_50),
                "percentile_75": float(percentile_75),
                "prior_mean": float(prior_mean),
                "fit_info": fit_info,
            }

            # Validation checks
            if fitted_mean < 0 or fitted_mean > 20:
                print(f"      ⚠️  WARNING: Mean out of range: {fitted_mean:.2f}")
            if fitted_std < 0 or fitted_std > 10:
                print(f"      ⚠️  WARNING: Std out of range: {fitted_std:.2f}")

            print(
                f"      ✅ Mean: {fitted_mean:.2f}, Std: {fitted_std:.2f}, "
                f"Percentiles: 25th={percentile_25:.1f}, 50th={percentile_50:.1f}, 75th={percentile_75:.1f}"
            )

    return distributions


def validate_distributions(distributions: Dict) -> Tuple[bool, list]:
    """Validate fitted distributions for reasonableness.

    Args:
        distributions: Dictionary of fitted distributions

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    is_valid = True

    # Check that premium_easy > premium_hard (if both exist)
    if "premium_easy" in distributions and "premium_hard" in distributions:
        if (
            distributions["premium_easy"]["mean"]
            <= distributions["premium_hard"]["mean"]
        ):
            warnings.append(
                "premium_easy mean should be > premium_hard mean "
                f"(got {distributions['premium_easy']['mean']:.2f} vs "
                f"{distributions['premium_hard']['mean']:.2f})"
            )

    # Check that mid_easy > mid_hard (if both exist)
    if "mid_easy" in distributions and "mid_hard" in distributions:
        if distributions["mid_easy"]["mean"] <= distributions["mid_hard"]["mean"]:
            warnings.append(
                "mid_easy mean should be > mid_hard mean "
                f"(got {distributions['mid_easy']['mean']:.2f} vs "
                f"{distributions['mid_hard']['mean']:.2f})"
            )

    # Check that budget_easy > budget_hard (if both exist)
    if "budget_easy" in distributions and "budget_hard" in distributions:
        if distributions["budget_easy"]["mean"] <= distributions["budget_hard"]["mean"]:
            warnings.append(
                "budget_easy mean should be > budget_hard mean "
                f"(got {distributions['budget_easy']['mean']:.2f} vs "
                f"{distributions['budget_hard']['mean']:.2f})"
            )

    # Check that premium > mid > budget for same fixture (if all exist)
    for fixture in ["easy", "hard"]:
        premium_key = f"premium_{fixture}"
        mid_key = f"mid_{fixture}"
        budget_key = f"budget_{fixture}"

        if all(k in distributions for k in [premium_key, mid_key, budget_key]):
            premium_mean = distributions[premium_key]["mean"]
            mid_mean = distributions[mid_key]["mean"]
            budget_mean = distributions[budget_key]["mean"]

            if not (premium_mean >= mid_mean >= budget_mean):
                warnings.append(
                    f"For {fixture} fixtures, expected premium >= mid >= budget, "
                    f"got {premium_mean:.2f} >= {mid_mean:.2f} >= {budget_mean:.2f}"
                )

    # Check for extreme values
    for combo_name, dist in distributions.items():
        if dist["mean"] < 0 or dist["mean"] > 20:
            warnings.append(
                f"{combo_name}: Mean out of reasonable range: {dist['mean']:.2f}"
            )
            is_valid = False
        if dist["std"] < 0 or dist["std"] > 10:
            warnings.append(
                f"{combo_name}: Std out of reasonable range: {dist['std']:.2f}"
            )
            is_valid = False

    return is_valid, warnings


def generate_report(distributions: Dict) -> str:
    """Generate distribution fitting report.

    Args:
        distributions: Dictionary of fitted distributions

    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PROBABILISTIC XP CALIBRATION - DISTRIBUTION FITTING REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append(f"Total Combinations Fitted: {len(distributions)}")
    report_lines.append("")

    # Distribution details
    report_lines.append("-" * 80)
    report_lines.append("FITTED DISTRIBUTIONS")
    report_lines.append("-" * 80)
    report_lines.append("")

    for combo_name in sorted(distributions.keys()):
        dist = distributions[combo_name]
        report_lines.append(f"📊 {combo_name.upper()}")
        report_lines.append(f"   Sample Size: {dist['sample_size']}")
        report_lines.append(f"   Fitted Mean: {dist['mean']:.2f}")
        report_lines.append(f"   Prior Mean: {dist['prior_mean']:.2f}")
        report_lines.append(f"   Std: {dist['std']:.2f}")
        report_lines.append(
            f"   Percentiles: 25th={dist['percentile_25']:.1f}, "
            f"50th={dist['percentile_50']:.1f}, 75th={dist['percentile_75']:.1f}"
        )

        fit_info = dist.get("fit_info", {})
        if fit_info.get("used_prior"):
            report_lines.append(
                f"   ⚠️  Used prior (reason: {fit_info.get('reason', 'unknown')})"
            )
        elif not fit_info.get("success"):
            report_lines.append(f"   ⚠️  Fit failed: {fit_info.get('error', 'unknown')}")
        else:
            report_lines.append(
                f"   ✅ Fit successful (iterations: {fit_info.get('n_iterations', 'N/A')})"
            )

        report_lines.append("")

    # Validation
    is_valid, warnings = validate_distributions(distributions)

    report_lines.append("-" * 80)
    report_lines.append("VALIDATION")
    report_lines.append("-" * 80)
    report_lines.append("")

    if is_valid and len(warnings) == 0:
        report_lines.append("✅ All distributions passed validation checks.")
    else:
        if not is_valid:
            report_lines.append("❌ Some distributions failed validation.")
        if len(warnings) > 0:
            report_lines.append(f"⚠️  {len(warnings)} warning(s):")
            for warning in warnings:
                report_lines.append(f"   - {warning}")

    report_lines.append("")

    return "\n".join(report_lines)


app = typer.Typer(
    help="Phase 2: Distribution Fitting for Probabilistic XP Calibration. "
    "Fits regularized Negative Binomial distributions for each (tier, fixture) combination."
)


@app.command()
def build(
    input_file: Path = typer.Option(
        None,
        "--input",
        help="Input categorized data CSV (default: data/calibration/categorized_data.csv)",
    ),
    output_file: Path = typer.Option(
        None,
        "--output",
        help="Output distributions JSON (default: data/calibration/distributions.json)",
    ),
    lambda_reg: float = typer.Option(
        0.3, "--lambda-reg", help="Regularization strength (default: 0.3)"
    ),
    min_samples: int = typer.Option(
        30, "--min-samples", help="Minimum sample size required (default: 30)"
    ),
):
    """Build calibration distributions from categorized data.

    Fits Negative Binomial distributions with L2 regularization toward priors
    for each (tier, fixture) combination found in the data.
    """
    try:
        # Default paths
        if input_file is None:
            input_file = project_root / "data" / "calibration" / "categorized_data.csv"

        if output_file is None:
            output_file = project_root / "data" / "calibration" / "distributions.json"

        # Check input file exists
        if not input_file.exists():
            typer.echo(f"❌ Error: Input file not found: {input_file}", err=True)
            raise typer.Exit(code=1)

        print(f"\n📥 Loading categorized data from: {input_file}")
        categorized_df = pd.read_csv(input_file)

        print(f"   ✅ Loaded {len(categorized_df)} player observations")

        # Fit distributions
        distributions = fit_distributions(
            categorized_df, lambda_reg=lambda_reg, minimum_sample_size=min_samples
        )

        if len(distributions) == 0:
            typer.echo("❌ Error: No distributions fitted. Check input data.", err=True)
            raise typer.Exit(code=1)

        # Validate
        is_valid, warnings = validate_distributions(distributions)

        # Generate report
        report = generate_report(distributions)
        print("\n" + report)

        # Save distributions
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Prepare JSON-serializable output (remove fit_info for cleaner JSON)
        output_dict = {}
        for combo_name, dist in distributions.items():
            output_dict[combo_name] = {k: v for k, v in dist.items() if k != "fit_info"}

        with open(output_file, "w") as f:
            json.dump(output_dict, f, indent=2)

        print(f"\n💾 Distributions saved to: {output_file}")

        if warnings:
            print(f"\n⚠️  {len(warnings)} validation warning(s) - review above")
        else:
            print("\n✅ Distribution fitting complete!")

    except Exception as e:
        typer.echo(f"\n❌ Error: {e}", err=True)
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
