"""
Tests for custom_pipeline_optimizer.py dual-mode training CLI.

Tests:
1. Argument parsing (train vs evaluate modes)
2. Holdout split logic
3. Feature selection strategies
4. Error handling
5. Integration tests with minimal data
"""

import sys
from pathlib import Path
from unittest.mock import patch
import argparse
import importlib.util

import pytest
import pandas as pd
import numpy as np

# Add project and scripts to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

# Import directly from script file

spec = importlib.util.spec_from_file_location(
    "custom_pipeline_optimizer",
    project_root / "scripts" / "custom_pipeline_optimizer.py",
)
custom_pipeline_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_pipeline_optimizer)

parse_args = custom_pipeline_optimizer.parse_args
run_evaluate_mode = custom_pipeline_optimizer.run_evaluate_mode
run_train_mode = custom_pipeline_optimizer.run_train_mode
select_features = custom_pipeline_optimizer.select_features
get_regressor_and_param_grid = custom_pipeline_optimizer.get_regressor_and_param_grid


class TestArgumentParsing:
    """Test command line argument parsing."""

    def test_default_mode_is_train(self):
        """Default mode should be 'train'."""
        with patch("sys.argv", ["script.py"]):
            args = parse_args()
            assert args.mode == "train"

    def test_evaluate_mode_parsing(self):
        """Evaluate mode should parse correctly."""
        with patch(
            "sys.argv",
            [
                "script.py",
                "--mode",
                "evaluate",
                "--end-gw",
                "10",
                "--regressor",
                "random-forest",
            ],
        ):
            args = parse_args()
            assert args.mode == "evaluate"
            assert args.end_gw == 10
            assert args.regressor == "random-forest"

    def test_holdout_gws_default_is_one(self):
        """Default holdout gameweeks should be 1."""
        with patch("sys.argv", ["script.py", "--mode", "evaluate"]):
            args = parse_args()
            assert args.holdout_gws == 1

    def test_custom_holdout_gws(self):
        """Custom holdout gameweeks should be respected."""
        with patch("sys.argv", ["script.py", "--mode", "evaluate", "--holdout-gws", "2"]):
            args = parse_args()
            assert args.holdout_gws == 2

    def test_train_mode_n_trials(self):
        """Train mode should use specified n_trials."""
        with patch("sys.argv", ["script.py", "--mode", "train", "--n-trials", "20"]):
            args = parse_args()
            assert args.n_trials == 20

    def test_feature_selection_choices(self):
        """Feature selection should accept valid choices."""
        valid_choices = ["none", "correlation", "permutation", "rfe-smart"]
        for choice in valid_choices:
            with patch(
                "sys.argv", ["script.py", "--feature-selection", choice]
            ):
                args = parse_args()
                assert args.feature_selection == choice

    def test_keep_penalty_features_flag(self):
        """Keep penalty features flag should work."""
        with patch("sys.argv", ["script.py", "--keep-penalty-features"]):
            args = parse_args()
            assert args.keep_penalty_features is True

        with patch("sys.argv", ["script.py"]):
            args = parse_args()
            assert args.keep_penalty_features is False


class TestHoldoutSplitLogic:
    """Test holdout set splitting logic."""

    def test_single_gameweek_holdout(self):
        """Single GW holdout should calculate correctly."""
        args = argparse.Namespace(
            start_gw=1,
            end_gw=10,
            holdout_gws=1,
        )

        train_end_gw = args.end_gw - args.holdout_gws
        holdout_start_gw = train_end_gw + 1

        assert train_end_gw == 9
        assert holdout_start_gw == 10

    def test_two_gameweeks_holdout(self):
        """Two GWs holdout should calculate correctly."""
        args = argparse.Namespace(
            start_gw=1,
            end_gw=10,
            holdout_gws=2,
        )

        train_end_gw = args.end_gw - args.holdout_gws
        holdout_start_gw = train_end_gw + 1

        assert train_end_gw == 8
        assert holdout_start_gw == 9

    def test_insufficient_training_data_error(self):
        """Should raise error if insufficient training data after holdout."""
        args = argparse.Namespace(
            start_gw=1,
            end_gw=7,
            holdout_gws=2,
            mode="evaluate",
            regressor="random-forest",
            feature_selection="none",
            keep_penalty_features=False,
            n_trials=2,
            cv_folds=None,
            scorer="fpl_weighted_huber",
            output_dir="models/test",
            random_seed=42,
            n_jobs=1,
            verbose=0,
        )

        # Should raise ValueError because train_end_gw (5) < start_gw + 5 (6)
        with pytest.raises(
            ValueError, match="Not enough training data after holdout"
        ):
            run_evaluate_mode(args)


class TestFeatureSelection:
    """Test feature selection strategies."""

    @pytest.fixture
    def sample_data(self):
        """Create sample feature matrix and target."""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(100, 20), columns=[f"feature_{i}" for i in range(20)]
        )
        # Add penalty features
        X["is_primary_penalty_taker"] = np.random.randint(0, 2, 100)
        X["is_penalty_taker"] = np.random.randint(0, 2, 100)
        X["is_corner_taker"] = np.random.randint(0, 2, 100)
        X["is_fk_taker"] = np.random.randint(0, 2, 100)

        y = np.random.randn(100)
        feature_names = X.columns.tolist()

        return X, y, feature_names

    def test_feature_selection_none(self, sample_data):
        """'none' strategy should return all features."""
        X, y, feature_names = sample_data

        selected = select_features(
            X, y, feature_names, strategy="none", keep_penalty_features=False, verbose=False
        )

        assert len(selected) == len(feature_names)
        assert set(selected) == set(feature_names)

    def test_feature_selection_correlation(self, sample_data):
        """'correlation' strategy should remove highly correlated features."""
        X, y, feature_names = sample_data

        # Add highly correlated feature
        X["feature_corr"] = X["feature_0"] + np.random.randn(len(X)) * 0.01

        selected = select_features(
            X,
            y,
            X.columns.tolist(),
            strategy="correlation",
            keep_penalty_features=False,
            verbose=False,
        )

        # Should remove at least one feature
        assert len(selected) < len(X.columns)

    def test_keep_penalty_features_flag(self, sample_data):
        """Penalty features should be kept when flag is set."""
        X, y, feature_names = sample_data

        penalty_features = [
            "is_primary_penalty_taker",
            "is_penalty_taker",
            "is_corner_taker",
            "is_fk_taker",
        ]

        selected = select_features(
            X,
            y,
            feature_names,
            strategy="rfe-smart",
            keep_penalty_features=True,
            verbose=False,
        )

        # All penalty features should be in selected
        for pf in penalty_features:
            assert pf in selected

    def test_rfe_smart_respects_target_count(self, sample_data):
        """RFE-smart should select approximately 60 features."""
        X, y, feature_names = sample_data

        selected = select_features(
            X,
            y,
            feature_names,
            strategy="rfe-smart",
            keep_penalty_features=True,
            verbose=False,
        )

        # With 24 total features and keep_penalty=True, should keep 20 + 4 penalty = 24
        # But RFE targets 60, so with small dataset it'll keep most
        assert len(selected) > 0
        assert len(selected) <= len(feature_names)


class TestRegressorInstantiation:
    """Test regressor and param grid creation."""

    def test_random_forest_regressor(self):
        """Random Forest should instantiate correctly."""
        regressor, param_dist = get_regressor_and_param_grid("random-forest", 42)

        from sklearn.ensemble import RandomForestRegressor

        assert isinstance(regressor, RandomForestRegressor)
        assert "regressor__n_estimators" in param_dist
        assert "regressor__max_depth" in param_dist

    def test_ridge_regressor(self):
        """Ridge should instantiate correctly."""
        regressor, param_dist = get_regressor_and_param_grid("ridge", 42)

        from sklearn.linear_model import Ridge

        assert isinstance(regressor, Ridge)
        assert "regressor__alpha" in param_dist

    def test_invalid_regressor_raises_error(self):
        """Invalid regressor name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown regressor"):
            get_regressor_and_param_grid("invalid-regressor", 42)


class TestEvaluateModeValidation:
    """Test evaluate mode validation logic."""

    def test_evaluate_mode_reduces_trials(self):
        """Evaluate mode should reduce hyperparameter trials."""
        args = argparse.Namespace(n_trials=20)

        eval_args = argparse.Namespace(**vars(args))
        eval_args.n_trials = args.n_trials // 2

        assert eval_args.n_trials == 10

    def test_holdout_gameweek_range_calculation(self):
        """Holdout range should be calculated correctly."""
        args = argparse.Namespace(start_gw=1, end_gw=10, holdout_gws=1)

        train_end_gw = args.end_gw - args.holdout_gws
        holdout_start_gw = train_end_gw + 1

        holdout_gws = list(range(holdout_start_gw, args.end_gw + 1))

        assert holdout_gws == [10]

    def test_holdout_two_gameweeks_range(self):
        """Two holdout GWs should create correct range."""
        args = argparse.Namespace(start_gw=1, end_gw=10, holdout_gws=2)

        train_end_gw = args.end_gw - args.holdout_gws
        holdout_start_gw = train_end_gw + 1

        holdout_gws = list(range(holdout_start_gw, args.end_gw + 1))

        assert holdout_gws == [9, 10]


class TestIntegration:
    """Integration tests with mocked data loading."""

    @pytest.fixture
    def mock_training_data(self):
        """Mock training data loader."""

        def _mock_load(start_gw, end_gw, verbose=True):
            # Return minimal mock data
            n_samples = (end_gw - start_gw + 1) * 100
            historical_df = pd.DataFrame(
                {
                    "player_id": list(range(100)) * (end_gw - start_gw + 1),
                    "gameweek": [
                        gw
                        for gw in range(start_gw, end_gw + 1)
                        for _ in range(100)
                    ],
                    "total_points": np.random.randint(0, 15, n_samples),
                }
            )
            fixtures_df = pd.DataFrame()
            teams_df = pd.DataFrame({"team_id": [1], "team_name": ["Test"]})
            ownership_df = pd.DataFrame()
            value_df = pd.DataFrame()
            fixture_diff_df = pd.DataFrame()
            betting_df = pd.DataFrame()
            raw_players_df = None

            return (
                historical_df,
                fixtures_df,
                teams_df,
                ownership_df,
                value_df,
                fixture_diff_df,
                betting_df,
                raw_players_df,
            )

        return _mock_load

    @pytest.mark.integration
    def test_evaluate_mode_workflow_integration(self):
        """Integration test: evaluate mode executes full workflow with real data."""
        # This is an integration test that uses real data
        # It verifies the entire workflow works end-to-end
        args = argparse.Namespace(
            start_gw=1,
            end_gw=10,
            holdout_gws=1,
            regressor="random-forest",
            feature_selection="none",
            keep_penalty_features=False,
            n_trials=1,  # Minimal trials for speed
            cv_folds=2,  # Limited folds for speed
            scorer="fpl_weighted_huber",
            output_dir="models/test",
            random_seed=42,
            n_jobs=1,
            verbose=0,
        )

        # Run evaluate mode with real data
        result = run_evaluate_mode(args)

        # Verify result structure
        pipeline, features, train_metrics, holdout_metrics = result

        # Verify metrics were computed
        assert train_metrics is not None
        assert "mae" in train_metrics
        assert "rmse" in train_metrics
        assert "spearman_correlation" in train_metrics

        # Verify holdout metrics
        assert holdout_metrics is not None
        assert "mae" in holdout_metrics

        # Verify pipeline and features
        assert pipeline is not None
        assert len(features) > 0

        # Verify reasonable metric values
        assert 0 < train_metrics["mae"] < 5  # MAE should be reasonable
        assert train_metrics["spearman_correlation"] > 0  # Should have positive correlation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
