"""Tests for ModelEvaluator."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from fpl_team_picker.domain.services.ml_training.evaluator import ModelEvaluator
from fpl_team_picker.domain.services.ml_training.config import EvaluationConfig


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return ModelEvaluator()

    @pytest.fixture
    def mock_model(self):
        """Create mock model that returns fixed predictions."""
        model = MagicMock()
        model.predict = MagicMock(return_value=np.array([2.0, 3.0, 4.0, 5.0, 6.0]))
        return model

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        X = pd.DataFrame({
            "feat1": [1, 2, 3, 4, 5],
            "feat2": [5, 4, 3, 2, 1],
        })
        y = np.array([2.5, 3.5, 4.5, 5.5, 6.5])
        cv_data = pd.DataFrame({
            "gameweek": [10, 10, 11, 11, 11],
            "position": ["GKP", "DEF", "MID", "MID", "FWD"],
            "player_id": [1, 2, 3, 4, 5],
        })
        return X, y, cv_data

    def test_evaluate_model_basic_metrics(self, evaluator, mock_model, sample_data):
        """Test basic evaluation metrics."""
        X, y, _ = sample_data

        metrics = evaluator.evaluate_model(mock_model, X, y)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "spearman" in metrics

        # Check MAE calculation (predictions are [2,3,4,5,6], actual is [2.5,3.5,4.5,5.5,6.5])
        # All errors are 0.5, so MAE should be 0.5
        assert metrics["mae"] == pytest.approx(0.5, rel=0.01)

    def test_evaluate_model_with_cv_data(self, evaluator, mock_model, sample_data):
        """Test evaluation with CV data for FPL metrics."""
        X, y, cv_data = sample_data

        metrics = evaluator.evaluate_model(mock_model, X, y, cv_data)

        # Should have FPL-specific metrics
        assert "captain_accuracy" in metrics
        assert "avg_top15_overlap" in metrics

    def test_evaluate_model_per_position(self, evaluator, mock_model, sample_data):
        """Test per-position metrics are computed."""
        X, y, cv_data = sample_data

        # Enable per-position metrics
        evaluator.config.compute_per_position = True

        metrics = evaluator.evaluate_model(mock_model, X, y, cv_data)

        # Should have position-specific MAE
        assert "MID_mae" in metrics or "DEF_mae" in metrics

    def test_evaluate_per_position(self, evaluator):
        """Test evaluate_per_position method."""
        # Create mock model
        model = MagicMock()
        model.predict = MagicMock(return_value=np.array([1, 2, 3, 4, 5, 6, 7, 8]))

        X = pd.DataFrame({
            "feat1": range(8),
        })
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
        positions = pd.Series(["GKP", "GKP", "DEF", "DEF", "MID", "MID", "FWD", "FWD"])

        results = evaluator.evaluate_per_position(model, X, y, positions)

        assert "GKP" in results
        assert "DEF" in results
        assert "MID" in results
        assert "FWD" in results

        # Each position should have mae, rmse, spearman, n_samples
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            assert "mae" in results[pos]
            assert "rmse" in results[pos]
            assert "spearman" in results[pos]
            assert "n_samples" in results[pos]
            assert results[pos]["n_samples"] == 2

    def test_determine_hybrid_config_all_below_threshold(self, evaluator):
        """Test hybrid config when no position beats threshold."""
        comparison = {
            "per_position": {
                "GKP": {"unified_mae": 1.0, "specific_mae": 0.99, "improvement": 0.01, "improvement_pct": 1.0},
                "DEF": {"unified_mae": 1.0, "specific_mae": 0.99, "improvement": 0.01, "improvement_pct": 1.0},
                "MID": {"unified_mae": 1.0, "specific_mae": 0.99, "improvement": 0.01, "improvement_pct": 1.0},
                "FWD": {"unified_mae": 1.0, "specific_mae": 0.99, "improvement": 0.01, "improvement_pct": 1.0},
            }
        }

        # Threshold is 2%, all improvements are 1%
        positions = evaluator.determine_hybrid_config(comparison, threshold=0.02)

        assert positions == []

    def test_determine_hybrid_config_some_above_threshold(self, evaluator):
        """Test hybrid config when some positions beat threshold."""
        comparison = {
            "per_position": {
                "GKP": {"unified_mae": 1.0, "specific_mae": 0.95, "improvement": 0.05, "improvement_pct": 5.0},
                "DEF": {"unified_mae": 1.0, "specific_mae": 0.99, "improvement": 0.01, "improvement_pct": 1.0},
                "MID": {"unified_mae": 1.0, "specific_mae": 0.99, "improvement": 0.01, "improvement_pct": 1.0},
                "FWD": {"unified_mae": 1.0, "specific_mae": 0.92, "improvement": 0.08, "improvement_pct": 8.0},
            }
        }

        # Threshold is 2%, GKP (5%) and FWD (8%) should pass
        positions = evaluator.determine_hybrid_config(comparison, threshold=0.02)

        assert "GKP" in positions
        assert "FWD" in positions
        assert "DEF" not in positions
        assert "MID" not in positions

    def test_format_results(self, evaluator):
        """Test results formatting."""
        metrics = {
            "mae": 1.234,
            "rmse": 1.567,
            "r2": 0.789,
            "spearman": 0.856,
            "captain_accuracy": 0.5,
            "avg_top15_overlap": 10.5,
        }

        formatted = evaluator.format_results(metrics)

        assert "MAE" in formatted
        assert "1.234" in formatted
        assert "Captain accuracy" in formatted
        assert "50.0%" in formatted


class TestEvaluatorWithConfig:
    """Tests for ModelEvaluator with custom config."""

    def test_custom_config(self):
        """Test evaluator with custom config."""
        config = EvaluationConfig(
            holdout_gws=3,
            compute_per_position=False,
            captain_top_k=5,
        )

        evaluator = ModelEvaluator(config)

        assert evaluator.config.holdout_gws == 3
        assert evaluator.config.compute_per_position is False
        assert evaluator.config.captain_top_k == 5

    def test_default_config(self):
        """Test evaluator creates default config if none provided."""
        evaluator = ModelEvaluator()

        assert evaluator.config is not None
        assert evaluator.config.holdout_gws == 2  # Default
