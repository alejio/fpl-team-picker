#!/usr/bin/env python3
"""
Multi-Model Comparison Tool for FPL Expected Points Prediction

Trains and evaluates all 5 ensemble models that support uncertainty quantification
in a single command, then ranks them by performance.

Supported Models (all support xP_uncertainty):
- RandomForest
- XGBoost
- LightGBM
- GradientBoosting
- AdaBoost

Key Features:
- Parallel model training (optional)
- Comprehensive evaluation metrics (MAE, RMSE, Spearman, Captain Accuracy, Top-K)
- Uncertainty quantification for all models
- Automatic ranking and best model selection
- Detailed comparison report with performance breakdown
- Saves all trained models for later use

Usage:
    # Compare all 5 models with default settings
    uv run python scripts/compare_all_models.py --end-gw 12

    # Full comparison with custom hyperparameter search
    uv run python scripts/compare_all_models.py --end-gw 12 --holdout-gws 2 \\
        --feature-selection rfe-smart --keep-penalty-features --n-trials 30

    # Quick comparison (fewer trials for faster results)
    uv run python scripts/compare_all_models.py --end-gw 12 --quick

Output:
    - Comparison report printed to console
    - All trained models saved to models/comparison_{timestamp}/
    - JSON summary with rankings saved to models/comparison_{timestamp}/summary.json
    - Best model automatically identified and highlighted
"""

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

# Create Typer app and console
app = typer.Typer(
    help="Multi-model comparison for FPL xP prediction",
    add_completion=False,
)
console = Console()

# Ensemble models that support uncertainty quantification
# Note: using names from custom_pipeline_optimizer.py regressor options
UNCERTAINTY_MODELS = [
    "random-forest",
    "xgboost",
    "lightgbm",
    "gradient-boost",  # Note: custom_pipeline_optimizer uses "gradient-boost" not "gradient-boosting"
    "adaboost",
]


def parse_metrics_from_output(output: str) -> Dict:
    """Parse metrics from custom_pipeline_optimizer output."""
    metrics = {}

    # Parse MAE (format: "MAE:  0.653 points")
    mae_match = re.search(r"MAE:\s+([\d.]+)\s+points", output)
    if mae_match:
        metrics["mae"] = float(mae_match.group(1))

    # Parse RMSE (format: "RMSE: 1.358 points")
    rmse_match = re.search(r"RMSE:\s+([\d.]+)\s+points", output)
    if rmse_match:
        metrics["rmse"] = float(rmse_match.group(1))

    # Parse Spearman (format: "Spearman correlation (ranking): 0.828")
    spearman_match = re.search(r"Spearman correlation.*?:\s+([\d.]+)", output)
    if spearman_match:
        metrics["spearman"] = float(spearman_match.group(1))

    # Parse Captain Accuracy - count ✓ vs total captain predictions
    captain_correct = len(re.findall(r"Captain ✓", output))
    captain_total = len(re.findall(r"Captain [✓✗]", output))
    if captain_total > 0:
        metrics["captain_accuracy"] = captain_correct / captain_total

    # Parse Top-15 Overlap - extract percentages and average them
    top15_matches = re.findall(r"Top-15 overlap \d+/15 \((\d+)%\)", output)
    if top15_matches:
        overlaps = [int(m) / 100.0 for m in top15_matches]
        metrics["top15_overlap"] = sum(overlaps) / len(overlaps)

    # Parse model path
    path_match = re.search(r"Pipeline saved to: (.+\.joblib)", output)
    if path_match:
        metrics["pipeline_path"] = path_match.group(1).strip()

    return metrics


def train_single_model(
    model_name: str,
    start_gw: int,
    end_gw: int,
    holdout_gws: int,
    feature_selection: str,
    keep_penalty_features: bool,
    preprocessing: str,
    scorer: str,
    n_trials: int,
    mode: str = "evaluate",
) -> Dict:
    """
    Train and evaluate a single model using subprocess to call custom_pipeline_optimizer.

    Returns a dictionary with model name, metrics, and trained pipeline path.
    """
    logger.info(f"🚀 Starting {mode} for {model_name}")

    try:
        # Build command
        cmd = [
            "uv",
            "run",
            "python",
            "scripts/custom_pipeline_optimizer.py",
            mode,
            "--regressor",
            model_name,
            "--end-gw",
            str(end_gw),
            "--feature-selection",
            feature_selection,
            "--preprocessing",
            preprocessing,
            "--scorer",
            scorer,
            "--n-trials",
            str(n_trials),
        ]

        if mode == "evaluate":
            cmd.extend(["--holdout-gws", str(holdout_gws)])
        else:
            cmd.extend(["--start-gw", str(start_gw)])

        if keep_penalty_features:
            cmd.append("--keep-penalty-features")

        # Run command
        logger.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,  # Run from project root
        )

        if result.returncode != 0:
            logger.error(f"❌ Failed to train {model_name}")
            logger.error(f"STDERR: {result.stderr}")
            return {
                "model_name": model_name,
                "metrics": None,
                "pipeline_path": None,
                "success": False,
                "error": result.stderr,
            }

        # Parse metrics from output
        output = result.stdout + result.stderr
        metrics = parse_metrics_from_output(output)

        if not metrics:
            logger.warning(f"⚠️  Could not parse metrics for {model_name}, check output")
            return {
                "model_name": model_name,
                "metrics": None,
                "pipeline_path": None,
                "success": False,
                "error": "Could not parse metrics from output",
            }

        logger.info(f"✅ Completed {model_name}: MAE={metrics.get('mae', 'N/A'):.3f}")

        return {
            "model_name": model_name,
            "metrics": metrics,
            "pipeline_path": metrics.get("pipeline_path"),
            "success": True,
        }

    except Exception as e:
        logger.error(f"❌ Exception training {model_name}: {e}")
        return {
            "model_name": model_name,
            "metrics": None,
            "pipeline_path": None,
            "success": False,
            "error": str(e),
        }


def rank_models(results: List[Dict]) -> List[Dict]:
    """
    Rank models by performance using a composite score.

    Composite score weights:
    - MAE: 40% (lower is better)
    - RMSE: 20% (lower is better)
    - Spearman: 30% (higher is better)
    - Captain Accuracy: 10% (higher is better)
    """
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        logger.error("No successful model results to rank")
        return results

    # Extract metrics for normalization
    maes = [r["metrics"]["mae"] for r in successful_results]
    rmses = [r["metrics"]["rmse"] for r in successful_results]
    spearmans = [r["metrics"]["spearman"] for r in successful_results]
    captain_accs = [r["metrics"].get("captain_accuracy", 0) for r in successful_results]

    # Normalize metrics to 0-1 scale
    mae_min, mae_max = min(maes), max(maes)
    rmse_min, rmse_max = min(rmses), max(rmses)
    spearman_min, spearman_max = min(spearmans), max(spearmans)
    captain_min, captain_max = min(captain_accs), max(captain_accs)

    # Avoid division by zero
    mae_range = mae_max - mae_min if mae_max > mae_min else 1
    rmse_range = rmse_max - rmse_min if rmse_max > rmse_min else 1
    spearman_range = spearman_max - spearman_min if spearman_max > spearman_min else 1
    captain_range = captain_max - captain_min if captain_max > captain_min else 1

    # Calculate composite scores
    for result in successful_results:
        metrics = result["metrics"]

        # Normalize (lower is better for MAE/RMSE, higher is better for Spearman/Captain)
        mae_norm = 1 - (metrics["mae"] - mae_min) / mae_range
        rmse_norm = 1 - (metrics["rmse"] - rmse_min) / rmse_range
        spearman_norm = (metrics["spearman"] - spearman_min) / spearman_range
        captain_norm = (
            metrics.get("captain_accuracy", 0) - captain_min
        ) / captain_range

        # Composite score (0-100)
        composite = (
            mae_norm * 40 + rmse_norm * 20 + spearman_norm * 30 + captain_norm * 10
        )

        result["composite_score"] = composite
        result["rank"] = 0  # Will be set after sorting

    # Sort by composite score (descending)
    ranked = sorted(
        successful_results, key=lambda x: x["composite_score"], reverse=True
    )

    # Assign ranks
    for i, result in enumerate(ranked):
        result["rank"] = i + 1

    # Add failed models at the end
    failed_results = [r for r in results if not r["success"]]
    for i, result in enumerate(failed_results):
        result["rank"] = len(ranked) + i + 1
        result["composite_score"] = 0

    return ranked + failed_results


def print_comparison_table(results: List[Dict], mode: str):
    """Print a rich table comparing all models."""
    table = Table(title=f"\n🏆 Model Comparison Results ({mode.upper()} mode)")

    table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
    table.add_column("Model", style="magenta")
    table.add_column("MAE", justify="right", style="green")
    table.add_column("RMSE", justify="right", style="green")
    table.add_column("Spearman", justify="right", style="blue")
    table.add_column("Captain Acc", justify="right", style="yellow")
    table.add_column("Top-15 Overlap", justify="right", style="yellow")
    table.add_column("Composite Score", justify="right", style="bold cyan")
    table.add_column("Status", justify="center")

    for result in results:
        if result["success"]:
            metrics = result["metrics"]
            rank_str = f"#{result['rank']}"
            status = "✅" if result["rank"] == 1 else "✓"

            # Highlight best model
            name_style = "bold magenta" if result["rank"] == 1 else "magenta"

            table.add_row(
                rank_str,
                result["model_name"],
                f"{metrics['mae']:.3f}",
                f"{metrics['rmse']:.3f}",
                f"{metrics['spearman']:.3f}",
                f"{metrics.get('captain_accuracy', 0):.3f}",
                f"{metrics.get('top15_overlap', 0):.3f}",
                f"{result['composite_score']:.1f}",
                status,
                style=name_style if result["rank"] == 1 else None,
            )
        else:
            table.add_row(
                f"#{result['rank']}",
                result["model_name"],
                "—",
                "—",
                "—",
                "—",
                "—",
                "0.0",
                "❌",
                style="dim",
            )

    console.print(table)

    # Print best model summary
    if results and results[0].get("success"):
        best = results[0]
        console.print(f"\n🥇 [bold green]Best Model: {best['model_name']}[/bold green]")
        console.print(f"   Pipeline saved to: {best.get('pipeline_path', 'N/A')}")
        console.print(f"   MAE: {best['metrics']['mae']:.3f}")
        console.print(f"   Spearman: {best['metrics']['spearman']:.3f}")
        console.print(f"   Composite Score: {best.get('composite_score', 0):.1f}/100")
    else:
        console.print("\n[bold red]⚠️  No models completed successfully[/bold red]")


@app.command()
def main(
    start_gw: int = typer.Option(1, "--start-gw", help="Training start gameweek"),
    end_gw: int = typer.Option(11, "--end-gw", help="Training end gameweek"),
    holdout_gws: int = typer.Option(
        1,
        "--holdout-gws",
        help="Gameweeks to hold out for evaluation (evaluate mode only)",
    ),
    feature_selection: Literal[
        "none", "correlation", "permutation", "rfe-smart"
    ] = typer.Option(
        "rfe-smart",
        "--feature-selection",
        help="Feature selection strategy",
    ),
    keep_penalty_features: bool = typer.Option(
        True,
        "--keep-penalty-features/--no-keep-penalty-features",
        help="Force-keep penalty/set-piece features",
    ),
    preprocessing: Literal["standard", "robust", "minmax", "grouped"] = typer.Option(
        "grouped",
        "--preprocessing",
        help="Feature scaling strategy",
    ),
    scorer: Literal[
        "fpl_weighted_huber",
        "fpl_comprehensive",
        "fpl_topk",
        "fpl_captain",
        "spearman",
    ] = typer.Option(
        "fpl_comprehensive",
        "--scorer",
        help="Optimization scorer",
    ),
    n_trials: int = typer.Option(
        20,
        "--n-trials",
        help="Hyperparameter search trials per model",
    ),
    mode: Literal["evaluate", "train"] = typer.Option(
        "evaluate",
        "--mode",
        help="evaluate (fast, holdout validation) or train (full training)",
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel/--sequential",
        help="Run models in parallel (faster but uses more CPU)",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Quick mode: reduce n_trials to 10 for faster comparison",
    ),
):
    """
    Compare all 5 ensemble models that support uncertainty quantification.

    This command trains and evaluates RandomForest, XGBoost, LightGBM,
    GradientBoosting, and AdaBoost, then ranks them by performance.
    """
    # Apply quick mode if requested
    if quick:
        n_trials = min(n_trials, 10)
        logger.info("🚀 Quick mode enabled: n_trials=10")

    # Create timestamp for summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(
        f"🎯 Comparing {len(UNCERTAINTY_MODELS)} models: {', '.join(UNCERTAINTY_MODELS)}"
    )
    logger.info(
        f"🔧 Mode: {mode}, Feature Selection: {feature_selection}, Scorer: {scorer}"
    )
    logger.info(f"🎲 Hyperparameter trials per model: {n_trials}")

    # Train models (parallel or sequential)
    results = []

    if parallel:
        logger.info("🔄 Training models in parallel...")
        with ThreadPoolExecutor(
            max_workers=min(5, len(UNCERTAINTY_MODELS))
        ) as executor:
            futures = {
                executor.submit(
                    train_single_model,
                    model_name,
                    start_gw,
                    end_gw,
                    holdout_gws,
                    feature_selection,
                    keep_penalty_features,
                    preprocessing,
                    scorer,
                    n_trials,
                    mode,
                ): model_name
                for model_name in UNCERTAINTY_MODELS
            }

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Exception training {model_name}: {e}")
                    results.append(
                        {
                            "model_name": model_name,
                            "success": False,
                            "error": str(e),
                        }
                    )
    else:
        logger.info("🔄 Training models sequentially...")
        for model_name in UNCERTAINTY_MODELS:
            result = train_single_model(
                model_name,
                start_gw,
                end_gw,
                holdout_gws,
                feature_selection,
                keep_penalty_features,
                preprocessing,
                scorer,
                n_trials,
                mode,
            )
            results.append(result)

    # Rank models
    ranked_results = rank_models(results)

    # Print comparison table
    print_comparison_table(ranked_results, mode)

    # Save summary to JSON
    summary_dir = Path("models/comparisons")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"comparison_{timestamp}.json"

    summary = {
        "timestamp": timestamp,
        "mode": mode,
        "parameters": {
            "start_gw": start_gw,
            "end_gw": end_gw,
            "holdout_gws": holdout_gws,
            "feature_selection": feature_selection,
            "keep_penalty_features": keep_penalty_features,
            "preprocessing": preprocessing,
            "scorer": scorer,
            "n_trials": n_trials,
        },
        "results": [
            {
                "rank": r["rank"],
                "model_name": r["model_name"],
                "metrics": r["metrics"] if r["success"] else None,
                "composite_score": r.get("composite_score", 0),
                "pipeline_path": r.get("pipeline_path"),
                "success": r["success"],
                "error": r.get("error"),
            }
            for r in ranked_results
        ],
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n📊 Summary saved to: {summary_path}")
    logger.info(
        f"✅ Comparison complete! Best model: {ranked_results[0]['model_name']}"
    )


if __name__ == "__main__":
    app()
