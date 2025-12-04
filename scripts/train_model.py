#!/usr/bin/env python3
"""
Unified ML Training CLI for FPL Expected Points Prediction.

Single entry point for all ML training operations:
- train unified: Train a unified model on all positions
- train position: Train position-specific model(s)
- train full-pipeline: Full evaluation + production workflow

Usage:
    # Train unified model
    python scripts/train_model.py unified --end-gw 12 --regressor lightgbm

    # Train position-specific models
    python scripts/train_model.py position --end-gw 12 --position GKP
    python scripts/train_model.py position --end-gw 12  # All positions

    # Full pipeline (evaluate → determine config → retrain)
    python scripts/train_model.py full-pipeline --end-gw 12 --holdout-gws 2

    # Evaluate existing model
    python scripts/train_model.py evaluate --model-path models/custom/model.joblib --end-gw 12
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import typer
from loguru import logger

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fpl_team_picker.domain.services.ml_training import (  # noqa: E402
    TrainingConfig,
    PositionTrainingConfig,
    MLTrainer,
    ModelEvaluator,
)
from fpl_team_picker.domain.ml import HybridPositionModel  # noqa: E402

# Reuse data loading utilities
sys.path.insert(0, str(Path(__file__).parent))
from ml_training_utils import load_training_data, engineer_features  # noqa: E402

app = typer.Typer(
    help="Unified ML training CLI for FPL xP prediction",
    add_completion=False,
)


# =============================================================================
# UNIFIED MODEL TRAINING
# =============================================================================


@app.command("unified")
def train_unified(
    end_gw: int = typer.Option(..., help="End gameweek for training"),
    start_gw: int = typer.Option(1, help="Start gameweek"),
    regressor: str = typer.Option("lightgbm", help="Regressor to use"),
    n_trials: int = typer.Option(50, help="Hyperparameter optimization trials"),
    scorer: str = typer.Option("neg_mean_absolute_error", help="Scoring metric"),
    feature_selection: str = typer.Option("none", help="Feature selection strategy"),
    preprocessing: str = typer.Option("standard", help="Preprocessing strategy"),
    output_dir: str = typer.Option("models/custom", help="Output directory"),
    random_seed: int = typer.Option(42, help="Random seed"),
    verbose: int = typer.Option(1, help="Verbosity level"),
):
    """Train a unified model on all positions."""

    logger.info("=" * 70)
    logger.info("🎯 UNIFIED MODEL TRAINING")
    logger.info("=" * 70)

    config = TrainingConfig(
        start_gw=start_gw,
        end_gw=end_gw,
        regressor=regressor,
        n_trials=n_trials,
        scorer=scorer,
        feature_selection=feature_selection,
        preprocessing=preprocessing,
        output_dir=Path(output_dir),
        random_seed=random_seed,
        verbose=verbose,
    )

    logger.info("\n📋 Configuration:")
    logger.info(f"   Gameweeks: GW{start_gw}-{end_gw}")
    logger.info(f"   Regressor: {regressor}")
    logger.info(f"   Trials: {n_trials}")
    logger.info(f"   Scorer: {scorer}")

    trainer = MLTrainer(config)
    pipeline, metadata = trainer.train_unified()

    model_path = trainer.save_model(pipeline, metadata)

    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"   Model: {model_path}")
    logger.info(f"   CV Score: {metadata.get('best_score', 'N/A')}")


# =============================================================================
# POSITION-SPECIFIC TRAINING
# =============================================================================


@app.command("position")
def train_position(
    end_gw: int = typer.Option(..., help="End gameweek for training"),
    start_gw: int = typer.Option(1, help="Start gameweek"),
    position: Optional[str] = typer.Option(
        None, help="Position (GKP/DEF/MID/FWD) or None for all"
    ),
    regressors: str = typer.Option(
        "lightgbm,xgboost,random-forest,gradient-boost",
        help="Comma-separated regressors to try",
    ),
    n_trials: int = typer.Option(50, help="Trials per regressor"),
    scorer: str = typer.Option("neg_mean_absolute_error", help="Scoring metric"),
    output_dir: str = typer.Option("models/position_specific", help="Output directory"),
    random_seed: int = typer.Option(42, help="Random seed"),
):
    """Train position-specific model(s)."""

    logger.info("=" * 70)
    logger.info("📍 POSITION-SPECIFIC TRAINING")
    logger.info("=" * 70)

    regressor_list = [r.strip() for r in regressors.split(",")]
    positions = [position] if position else ["GKP", "DEF", "MID", "FWD"]

    config = PositionTrainingConfig(
        start_gw=start_gw,
        end_gw=end_gw,
        positions=positions,
        regressors=regressor_list,
        n_trials=n_trials,
        scorer=scorer,
        output_dir=Path(output_dir),
        random_seed=random_seed,
    )

    logger.info("\n📋 Configuration:")
    logger.info(f"   Gameweeks: GW{start_gw}-{end_gw}")
    logger.info(f"   Positions: {positions}")
    logger.info(f"   Regressors: {regressor_list}")

    trainer = MLTrainer(config)
    results = trainer.train_all_positions(regressors=regressor_list)

    # Save each model
    for pos, (pipeline, metadata) in results.items():
        trainer.save_model(pipeline, metadata, name_prefix=pos.lower())

    logger.info("\n" + "=" * 70)
    logger.info("✅ POSITION TRAINING COMPLETE")
    logger.info("=" * 70)

    for pos, (_, metadata) in results.items():
        logger.info(
            f"   {pos}: {metadata['regressor']} (score: {metadata['best_score']:.4f})"
        )


# =============================================================================
# FULL HYBRID PIPELINE
# =============================================================================


@app.command("full-pipeline")
def full_pipeline(
    end_gw: int = typer.Option(..., help="Final gameweek (all data)"),
    holdout_gws: int = typer.Option(2, help="Gameweeks to hold out for evaluation"),
    unified_regressors: str = typer.Option(
        "lightgbm,xgboost,random-forest,gradient-boost",
        help="Regressors for unified model",
    ),
    position_regressors: str = typer.Option(
        "lightgbm,xgboost,random-forest,gradient-boost",
        help="Regressors for position-specific models",
    ),
    n_trials: int = typer.Option(50, help="Trials per regressor"),
    scorer: str = typer.Option("neg_mean_absolute_error", help="Scoring metric"),
    improvement_threshold: float = typer.Option(
        0.02, help="Minimum improvement to use position-specific (default: 2%)"
    ),
    output_dir: str = typer.Option("models", help="Base output directory"),
    random_seed: int = typer.Option(42, help="Random seed"),
):
    """
    Full pipeline: Evaluate → Determine best config → Retrain on all data.

    Phase 1 (Evaluation):
    - Train unified models with each regressor on GW1-(end-holdout)
    - Train position-specific models on GW1-(end-holdout)
    - Evaluate on holdout GWs
    - Determine: best unified regressor + which positions need specific models

    Phase 2 (Production):
    - Retrain best unified on GW1-end
    - Retrain needed position-specific on GW1-end
    - Build and save hybrid model
    """

    unified_reg_list = [r.strip() for r in unified_regressors.split(",")]
    position_reg_list = [r.strip() for r in position_regressors.split(",")]

    train_end_gw = end_gw - holdout_gws

    logger.info("=" * 70)
    logger.info("🚀 FULL HYBRID PIPELINE")
    logger.info("=" * 70)
    logger.info("\n📋 Configuration:")
    logger.info(f"   End gameweek: {end_gw}")
    logger.info(f"   Holdout: GW{train_end_gw + 1}-{end_gw} ({holdout_gws} GWs)")
    logger.info(f"   Evaluation training: GW1-{train_end_gw}")
    logger.info(f"   Unified regressors: {unified_reg_list}")
    logger.info(f"   Position regressors: {position_reg_list}")
    logger.info(f"   Scorer: {scorer}")
    logger.info(f"   Improvement threshold: {improvement_threshold * 100:.1f}%")

    output_path = Path(output_dir)

    # =========================================================================
    # PHASE 1: EVALUATION
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("📊 PHASE 1: EVALUATION")
    logger.info("=" * 70)

    # Load evaluation + holdout data
    logger.info("\n📦 Loading data...")
    data = load_training_data(1, end_gw, verbose=True)
    features_df, target, feature_names = engineer_features(*data, verbose=True)

    # Split into train/holdout
    cv_data = features_df[["gameweek", "position", "player_id"]].copy()
    holdout_gws_list = list(range(train_end_gw + 1, end_gw + 1))

    _train_mask = ~cv_data["gameweek"].isin(holdout_gws_list)
    test_mask = cv_data["gameweek"].isin(holdout_gws_list)

    X_holdout = features_df[test_mask].reset_index(drop=True)
    y_holdout = target[test_mask]
    holdout_cv_data = cv_data[test_mask].reset_index(drop=True)

    # 1a. Train and evaluate unified models
    logger.info(f"\n🔧 Step 1a: Training unified models on GW1-{train_end_gw}...")

    unified_results = {}
    best_unified_regressor = None
    best_unified_mae = float("inf")
    best_unified_model = None

    for reg in unified_reg_list:
        logger.info(f"\n   Training {reg}...")

        config = TrainingConfig(
            start_gw=1,
            end_gw=train_end_gw,
            regressor=reg,
            n_trials=n_trials,
            scorer=scorer,
            output_dir=output_path / "custom",
            random_seed=random_seed,
            verbose=0,
        )

        try:
            trainer = MLTrainer(config)
            model, metadata = trainer.train_unified()

            # Evaluate on holdout
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_model(
                model, X_holdout, y_holdout, holdout_cv_data
            )

            mae = metrics["mae"]
            unified_results[reg] = {"mae": mae, "model": model, "metadata": metadata}

            logger.info(f"   ✅ {reg}: MAE = {mae:.4f}")

            if mae < best_unified_mae:
                best_unified_mae = mae
                best_unified_regressor = reg
                best_unified_model = model

        except Exception as e:
            logger.warning(f"   ❌ {reg} failed: {e}")

    if not best_unified_regressor:
        logger.error("No unified models trained successfully")
        raise typer.Exit(1)

    logger.info(
        f"\n   🏆 Best unified: {best_unified_regressor} (MAE: {best_unified_mae:.4f})"
    )

    # 1b. Train position-specific models
    logger.info(
        f"\n🔧 Step 1b: Training position-specific models on GW1-{train_end_gw}..."
    )

    pos_config = PositionTrainingConfig(
        start_gw=1,
        end_gw=train_end_gw,
        regressors=position_reg_list,
        n_trials=n_trials,
        scorer=scorer,
        output_dir=output_path / "position_specific",
        random_seed=random_seed,
        verbose=0,
    )

    pos_trainer = MLTrainer(pos_config)
    position_results = pos_trainer.train_all_positions(regressors=position_reg_list)
    position_models = {pos: model for pos, (model, _) in position_results.items()}

    # 1c. Compare and determine hybrid config
    logger.info(f"\n🔧 Step 1c: Evaluating on holdout GW{train_end_gw + 1}-{end_gw}...")

    evaluator = ModelEvaluator()
    comparison = evaluator.compare_unified_vs_position(
        best_unified_model,
        position_models,
        X_holdout,
        y_holdout,
        holdout_cv_data["position"],
    )

    positions_for_specific = evaluator.determine_hybrid_config(
        comparison,
        threshold=improvement_threshold,
    )

    # =========================================================================
    # PHASE 2: PRODUCTION TRAINING
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("🏭 PHASE 2: PRODUCTION TRAINING")
    logger.info("=" * 70)

    # 2a. Retrain best unified on all data
    logger.info(f"\n🔧 Step 2a: Retraining {best_unified_regressor} on GW1-{end_gw}...")

    prod_config = TrainingConfig(
        start_gw=1,
        end_gw=end_gw,
        regressor=best_unified_regressor,
        n_trials=n_trials,
        scorer=scorer,
        output_dir=output_path / "custom",
        random_seed=random_seed,
        verbose=1,
    )

    prod_trainer = MLTrainer(prod_config)

    # Use best params from evaluation
    eval_best_params = unified_results[best_unified_regressor]["metadata"][
        "best_params"
    ]
    prod_unified_model, prod_unified_metadata = prod_trainer.train_unified(
        best_params=eval_best_params
    )
    prod_unified_path = prod_trainer.save_model(
        prod_unified_model, prod_unified_metadata
    )

    # 2b. Retrain position-specific models if needed
    prod_position_models = {}

    if positions_for_specific:
        logger.info(
            f"\n🔧 Step 2b: Retraining position-specific for {positions_for_specific}..."
        )

        pos_prod_config = PositionTrainingConfig(
            start_gw=1,
            end_gw=end_gw,
            positions=positions_for_specific,
            regressors=position_reg_list,
            n_trials=n_trials,
            scorer=scorer,
            output_dir=output_path / "position_specific",
            random_seed=random_seed,
        )

        pos_prod_trainer = MLTrainer(pos_prod_config)

        for pos in positions_for_specific:
            # Use best params from evaluation
            eval_metadata = position_results[pos][1]
            best_reg = eval_metadata["regressor"]
            best_params = eval_metadata["best_params"]

            model, metadata = pos_prod_trainer.train_position(
                pos,
                regressor=best_reg,
                best_params=best_params,
            )
            prod_position_models[pos] = model
            pos_prod_trainer.save_model(model, metadata, name_prefix=pos.lower())
    else:
        logger.info("\n🔧 Step 2b: Skipping position-specific (none beat threshold)")

    # 2c. Build hybrid model
    logger.info("\n🔧 Step 2c: Building hybrid model...")

    if positions_for_specific:
        hybrid = HybridPositionModel(
            unified_model=prod_unified_model,
            position_models=prod_position_models,
            use_specific_for=positions_for_specific,
        )

        hybrid_dir = output_path / "hybrid"
        hybrid_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hybrid_path = hybrid_dir / f"hybrid_gw1-{end_gw}_{timestamp}.joblib"
        joblib.dump(hybrid, hybrid_path)

        # Save metadata
        metadata = {
            "unified_model": str(prod_unified_path),
            "unified_regressor": best_unified_regressor,
            "unified_results": {
                k: {"mae": v["mae"]} for k, v in unified_results.items()
            },
            "position_models": {
                pos: str(
                    output_path
                    / "position_specific"
                    / f"{pos.lower()}_gw1-{end_gw}*.joblib"
                )
                for pos in positions_for_specific
            },
            "use_specific_for": positions_for_specific,
            "unified_positions": [
                p
                for p in ["GKP", "DEF", "MID", "FWD"]
                if p not in positions_for_specific
            ],
            "comparison_results": {
                pos: {
                    k: v for k, v in metrics.items() if not isinstance(v, (list, dict))
                }
                for pos, metrics in comparison.get("per_position", {}).items()
            },
            "improvement_threshold": improvement_threshold,
            "scorer": scorer,
            "created_at": datetime.now().isoformat(),
            "end_gw": end_gw,
            "holdout_gws": holdout_gws,
        }

        with open(hybrid_path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        final_path = hybrid_path
        final_type = "Hybrid"
    else:
        final_path = prod_unified_path
        final_type = "Unified"

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("✅ FULL PIPELINE COMPLETE")
    logger.info("=" * 70)

    logger.info("\n📊 Best Configuration:")
    logger.info(f"   Unified regressor: {best_unified_regressor}")

    if unified_results:
        logger.info("   Unified comparison:")
        for reg, res in sorted(unified_results.items(), key=lambda x: x[1]["mae"]):
            marker = "👑" if reg == best_unified_regressor else "  "
            logger.info(f"     {marker} {reg}: MAE = {res['mae']:.4f}")

    if positions_for_specific:
        logger.info(f"   Position-specific for: {positions_for_specific}")
        logger.info(
            f"   Unified for: {[p for p in ['GKP', 'DEF', 'MID', 'FWD'] if p not in positions_for_specific]}"
        )
    else:
        logger.info("   Using unified for all positions")

    logger.info("\n💾 Final Model:")
    logger.info(f"   Type: {final_type}")
    logger.info(f"   Path: {final_path}")

    logger.info("\n📝 To use in production, update settings.py:")
    logger.info(f'   ml_model_path = "{final_path}"')


# =============================================================================
# MODEL EVALUATION
# =============================================================================


@app.command("evaluate")
def evaluate_model(
    model_path: str = typer.Option(..., help="Path to model (.joblib)"),
    end_gw: int = typer.Option(..., help="End gameweek for test data"),
    start_gw: int = typer.Option(1, help="Start gameweek"),
    holdout_gws: int = typer.Option(2, help="Gameweeks to use as test set"),
):
    """Evaluate an existing model on holdout data."""

    logger.info("=" * 70)
    logger.info("📊 MODEL EVALUATION")
    logger.info("=" * 70)

    # Load model
    model = joblib.load(model_path)
    logger.info(f"   Model: {model_path}")

    # Load data
    train_end_gw = end_gw - holdout_gws
    data = load_training_data(start_gw, end_gw, verbose=True)
    features_df, target, _ = engineer_features(*data, verbose=True)

    # Filter to holdout
    holdout_gws_list = list(range(train_end_gw + 1, end_gw + 1))
    cv_data = features_df[["gameweek", "position", "player_id"]].copy()
    test_mask = cv_data["gameweek"].isin(holdout_gws_list)

    X_test = features_df[test_mask].reset_index(drop=True)
    y_test = target[test_mask]
    test_cv_data = cv_data[test_mask].reset_index(drop=True)

    logger.info(f"\n   Holdout GWs: {holdout_gws_list}")
    logger.info(f"   Test samples: {len(X_test)}")

    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(model, X_test, y_test, test_cv_data)

    logger.info("\n" + evaluator.format_results(metrics))


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
