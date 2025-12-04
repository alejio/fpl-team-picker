"""
ModelEvaluator - Evaluation and comparison of ML models.

Provides comprehensive evaluation metrics for FPL ML models including:
- Standard regression metrics (MAE, RMSE, R²)
- FPL-specific metrics (captain accuracy, squad overlap)
- Position-specific analysis
- Unified vs position-specific comparison
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import EvaluationConfig


class ModelEvaluator:
    """
    Evaluate and compare ML models for FPL prediction.

    Handles holdout evaluation, model comparison, and
    determination of optimal hybrid configurations.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()

    def evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        cv_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a model on test data.

        Args:
            model: Trained sklearn pipeline
            X: Test features
            y: Test target
            cv_data: Optional DataFrame with position/gameweek for detailed metrics

        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred = model.predict(X)

        # Basic metrics
        metrics = {
            "mae": mean_absolute_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "r2": 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        }

        # Spearman correlation (ranking quality)
        corr, _ = spearmanr(y, y_pred)
        metrics["spearman"] = corr

        # FPL-specific metrics if cv_data provided
        if cv_data is not None and "gameweek" in cv_data.columns:
            eval_df = cv_data.copy()
            eval_df["predicted"] = y_pred
            eval_df["actual"] = y

            # Per-gameweek captain accuracy
            captain_hits = []
            top15_overlaps = []

            for gw in sorted(eval_df["gameweek"].unique()):
                gw_data = eval_df[eval_df["gameweek"] == gw]

                # Captain accuracy (did we identify top scorer?)
                actual_best = gw_data.loc[gw_data["actual"].idxmax(), "player_id"]
                pred_best = gw_data.loc[gw_data["predicted"].idxmax(), "player_id"]
                captain_hits.append(1 if actual_best == pred_best else 0)

                # Top-15 overlap (squad building)
                top15_actual = set(gw_data.nlargest(15, "actual")["player_id"])
                top15_pred = set(gw_data.nlargest(15, "predicted")["player_id"])
                overlap = len(top15_actual & top15_pred)
                top15_overlaps.append(overlap)

            metrics["captain_accuracy"] = np.mean(captain_hits)
            metrics["avg_top15_overlap"] = np.mean(top15_overlaps)

            # Per-position metrics
            if self.config.compute_per_position and "position" in cv_data.columns:
                for pos in ["GKP", "DEF", "MID", "FWD"]:
                    mask = eval_df["position"] == pos
                    if mask.sum() > 0:
                        pos_mae = mean_absolute_error(
                            eval_df.loc[mask, "actual"], eval_df.loc[mask, "predicted"]
                        )
                        metrics[f"{pos}_mae"] = pos_mae

        return metrics

    def evaluate_per_position(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        positions: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance per position.

        Args:
            model: Trained model
            X: Test features
            y: Test target
            positions: Position labels for each sample

        Returns:
            Dict mapping position -> metrics dict
        """
        y_pred = model.predict(X)

        results = {}
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            mask = positions == pos
            if mask.sum() == 0:
                continue

            y_pos = y[mask]
            y_pred_pos = y_pred[mask]

            results[pos] = {
                "mae": mean_absolute_error(y_pos, y_pred_pos),
                "rmse": np.sqrt(mean_squared_error(y_pos, y_pred_pos)),
                "spearman": spearmanr(y_pos, y_pred_pos)[0],
                "n_samples": int(mask.sum()),
            }

        return results

    def compare_unified_vs_position(
        self,
        unified_model: Any,
        position_models: Dict[str, Any],
        X: pd.DataFrame,
        y: np.ndarray,
        positions: pd.Series,
    ) -> Dict[str, Any]:
        """
        Compare unified model vs position-specific models.

        Args:
            unified_model: Unified model (all positions)
            position_models: Dict mapping position -> position-specific model
            X: Test features
            y: Test target
            positions: Position labels

        Returns:
            Comparison results with per-position breakdown
        """
        results = {
            "unified_overall": {},
            "position_specific_overall": {},
            "per_position": {},
        }

        # Unified predictions
        unified_preds = unified_model.predict(X)
        results["unified_overall"] = {
            "mae": mean_absolute_error(y, unified_preds),
            "rmse": np.sqrt(mean_squared_error(y, unified_preds)),
            "spearman": spearmanr(y, unified_preds)[0],
        }

        # Position-specific predictions (combined)
        position_preds = np.zeros_like(unified_preds)

        for pos in ["GKP", "DEF", "MID", "FWD"]:
            mask = positions == pos
            if not mask.any():
                continue

            X_pos = X.loc[mask].copy()
            y_pos = y[mask]

            # Get unified performance for this position
            unified_pos_preds = unified_preds[mask]
            unified_mae = mean_absolute_error(y_pos, unified_pos_preds)

            # Get position-specific performance
            if pos in position_models:
                model = position_models[pos]

                # Get features model expects
                try:
                    feature_names = model.named_steps["feature_selector"].feature_names

                    # Check for missing features and add them
                    missing = set(feature_names) - set(X_pos.columns)
                    if missing:
                        # Add position-specific features if needed
                        from .trainer import MLTrainer

                        trainer = MLTrainer.__new__(MLTrainer)
                        additions = trainer._get_position_feature_additions(pos)
                        for feat_name, feat_func in additions:
                            if feat_name in missing and feat_name not in X_pos.columns:
                                X_pos[feat_name] = feat_func(X_pos)

                    specific_preds = model.predict(X_pos[feature_names])
                except Exception as e:
                    logger.warning(f"Error predicting {pos}: {e}")
                    specific_preds = unified_pos_preds
            else:
                specific_preds = unified_pos_preds

            specific_mae = mean_absolute_error(y_pos, specific_preds)
            position_preds[mask] = specific_preds

            # Calculate improvement
            improvement = unified_mae - specific_mae
            improvement_pct = (
                (improvement / unified_mae) * 100 if unified_mae > 0 else 0
            )

            results["per_position"][pos] = {
                "unified_mae": unified_mae,
                "specific_mae": specific_mae,
                "improvement": improvement,
                "improvement_pct": improvement_pct,
                "n_samples": int(mask.sum()),
            }

        # Overall position-specific performance
        results["position_specific_overall"] = {
            "mae": mean_absolute_error(y, position_preds),
            "rmse": np.sqrt(mean_squared_error(y, position_preds)),
            "spearman": spearmanr(y, position_preds)[0],
        }

        return results

    def determine_hybrid_config(
        self,
        comparison_results: Dict[str, Any],
        threshold: float = 0.02,
    ) -> List[str]:
        """
        Determine which positions should use position-specific models.

        Args:
            comparison_results: Output from compare_unified_vs_position
            threshold: Minimum improvement (as fraction) to use specific model

        Returns:
            List of positions that should use position-specific models
        """
        positions_for_specific = []

        logger.info(f"\n📊 Hybrid Configuration (threshold: {threshold * 100:.1f}%)")
        logger.info(
            f"   {'Position':<8} {'Unified':<10} {'Specific':<10} {'Improve':<10} {'Use Specific?'}"
        )
        logger.info("   " + "-" * 50)

        for pos, metrics in comparison_results.get("per_position", {}).items():
            unified_mae = metrics["unified_mae"]
            specific_mae = metrics["specific_mae"]
            improvement_pct = metrics["improvement_pct"] / 100  # Convert to fraction

            use_specific = improvement_pct >= threshold
            if use_specific:
                positions_for_specific.append(pos)

            marker = "✅ YES" if use_specific else "❌ NO"
            logger.info(
                f"   {pos:<8} {unified_mae:<10.3f} {specific_mae:<10.3f} "
                f"{improvement_pct:>+8.1%}   {marker}"
            )

        logger.info(
            f"\n   Result: {positions_for_specific or 'None (use unified for all)'}"
        )

        return positions_for_specific

    def format_results(self, metrics: Dict[str, Any]) -> str:
        """
        Format evaluation results for display.

        Args:
            metrics: Evaluation metrics dict

        Returns:
            Formatted string
        """
        lines = ["📊 Evaluation Results:"]
        lines.append(f"   MAE:      {metrics.get('mae', 0):.3f}")
        lines.append(f"   RMSE:     {metrics.get('rmse', 0):.3f}")
        lines.append(f"   R²:       {metrics.get('r2', 0):.3f}")
        lines.append(f"   Spearman: {metrics.get('spearman', 0):.3f}")

        if "captain_accuracy" in metrics:
            lines.append(
                f"   Captain accuracy: {metrics['captain_accuracy'] * 100:.1f}%"
            )
        if "avg_top15_overlap" in metrics:
            lines.append(
                f"   Avg top-15 overlap: {metrics['avg_top15_overlap']:.1f}/15"
            )

        return "\n".join(lines)
