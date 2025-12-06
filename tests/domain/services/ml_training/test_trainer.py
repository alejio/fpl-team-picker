"""Tests for MLTrainer."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from fpl_team_picker.domain.services.ml_training.trainer import MLTrainer
from fpl_team_picker.domain.services.ml_training.config import (
    TrainingConfig,
    PositionTrainingConfig,
)


class TestMLTrainerInit:
    """Tests for MLTrainer initialization."""

    def test_init_with_config(self):
        """Test trainer initializes with config."""
        config = TrainingConfig(end_gw=10, regressor="lightgbm")
        trainer = MLTrainer(config)

        assert trainer.config == config
        assert trainer._data_cache is None
        assert trainer._features_cache is None

    def test_init_with_position_config(self):
        """Test trainer works with PositionTrainingConfig."""
        config = PositionTrainingConfig(
            end_gw=10,
            positions=["GKP", "FWD"],
        )
        trainer = MLTrainer(config)

        assert trainer.config == config


class TestMLTrainerSelectFeatures:
    """Tests for feature selection methods."""

    @pytest.fixture
    def trainer(self):
        """Create trainer with default config."""
        config = TrainingConfig(feature_selection="none")
        return MLTrainer(config)

    @pytest.fixture
    def sample_features(self):
        """Create sample feature data."""
        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({
            f"feat_{i}": np.random.randn(n_samples) for i in range(10)
        })
        y = np.random.randn(n_samples)
        feature_names = list(X.columns)

        return X, y, feature_names

    def test_select_features_none(self, sample_features):
        """Test feature selection with 'none' strategy."""
        X, y, feature_names = sample_features

        config = TrainingConfig(feature_selection="none")
        trainer = MLTrainer(config)

        selected = trainer.select_features(X, y, feature_names)

        assert selected == feature_names
        assert len(selected) == 10

    def test_select_features_correlation(self, sample_features):
        """Test feature selection with correlation strategy."""
        X, y, feature_names = sample_features

        # Add a highly correlated feature
        X["feat_corr"] = X["feat_0"] * 1.001 + 0.001  # Nearly identical
        feature_names.append("feat_corr")

        config = TrainingConfig(feature_selection="correlation")
        trainer = MLTrainer(config)

        selected = trainer.select_features(X, y, feature_names)

        # Either feat_0 or feat_corr should be removed
        assert len(selected) < len(feature_names)

    def test_select_features_keeps_penalty_features(self, sample_features):
        """Test that penalty features are kept when requested."""
        X, y, feature_names = sample_features

        # Add penalty features
        X["is_primary_penalty_taker"] = np.random.randint(0, 2, len(X))
        X["is_penalty_taker"] = np.random.randint(0, 2, len(X))
        feature_names.extend(["is_primary_penalty_taker", "is_penalty_taker"])

        config = TrainingConfig(
            feature_selection="correlation",
            keep_penalty_features=True,
        )
        trainer = MLTrainer(config)

        selected = trainer.select_features(X, y, feature_names)

        assert "is_primary_penalty_taker" in selected
        assert "is_penalty_taker" in selected


class TestMLTrainerPositionFeatures:
    """Tests for position-specific feature additions."""

    @pytest.fixture
    def trainer(self):
        """Create trainer instance."""
        config = TrainingConfig()
        return MLTrainer(config)

    def test_gkp_features(self, trainer):
        """Test GKP-specific features."""
        additions = trainer._get_position_feature_additions("GKP")

        feature_names = [name for name, _ in additions]
        assert "saves_x_opp_xg" in feature_names
        assert "clean_sheet_potential" in feature_names

    def test_def_features(self, trainer):
        """Test DEF-specific features."""
        additions = trainer._get_position_feature_additions("DEF")

        feature_names = [name for name, _ in additions]
        assert "cs_x_minutes" in feature_names
        assert "goal_threat_def" in feature_names

    def test_mid_features(self, trainer):
        """Test MID-specific features."""
        additions = trainer._get_position_feature_additions("MID")

        feature_names = [name for name, _ in additions]
        assert "xgi_combined" in feature_names
        assert "creativity_x_threat" in feature_names

    def test_fwd_features(self, trainer):
        """Test FWD-specific features."""
        additions = trainer._get_position_feature_additions("FWD")

        feature_names = [name for name, _ in additions]
        assert "xg_x_minutes" in feature_names
        assert "goal_involvement" in feature_names

    def test_feature_functions_callable(self, trainer):
        """Test that feature functions are callable on DataFrame."""
        # Create sample data with required columns
        df = pd.DataFrame({
            "rolling_5gw_saves": [5, 10, 15],
            "opponent_rolling_5gw_xg": [1.5, 2.0, 1.0],
            "clean_sheet_probability_enhanced": [0.3, 0.4, 0.5],
            "rolling_5gw_minutes": [270, 360, 450],
            "rolling_5gw_xg": [0.5, 1.0, 1.5],
            "rolling_5gw_xa": [0.2, 0.3, 0.4],
            "rolling_5gw_threat": [50, 60, 70],
            "rolling_5gw_creativity": [40, 50, 60],
        })

        for position in ["GKP", "DEF", "MID", "FWD"]:
            additions = trainer._get_position_feature_additions(position)

            for feat_name, feat_func in additions:
                result = feat_func(df)
                assert len(result) == len(df)
                assert not np.isnan(result).all()


class TestMLTrainerSaveModel:
    """Tests for model saving functionality."""

    @pytest.fixture
    def trainer(self, tmp_path):
        """Create trainer with temp output dir."""
        config = TrainingConfig(
            output_dir=tmp_path / "models",
            regressor="lightgbm",
            start_gw=1,
            end_gw=10,
        )
        return MLTrainer(config)

    def test_save_model_creates_files(self, trainer, tmp_path):
        """Test that save_model creates model and metadata files."""
        # Create mock pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge()),
        ])

        # Fit on dummy data
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        pipeline.fit(X, y)

        metadata = {
            "best_params": {"regressor__alpha": 1.0},
            "best_score": -0.5,
        }

        model_path = trainer.save_model(pipeline, metadata)

        assert model_path.exists()
        assert model_path.suffix == ".joblib"
        assert "_pipeline" in model_path.name

        # Check metadata file exists (base_name.json, not base_name_pipeline.json)
        # Model is saved as {base_name}_pipeline.joblib, metadata as {base_name}.json
        json_name = model_path.name.replace("_pipeline.joblib", ".json")
        json_path = model_path.parent / json_name
        assert json_path.exists()

    def test_save_model_with_prefix(self, trainer, tmp_path):
        """Test save_model with custom name prefix."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge()),
        ])

        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        pipeline.fit(X, y)

        model_path = trainer.save_model(pipeline, {}, name_prefix="gkp")

        assert "gkp" in model_path.name


class TestMLTrainerIntegration:
    """Integration tests for MLTrainer (require data mocking)."""

    @pytest.fixture
    def mock_data(self):
        """Create mock training data."""
        np.random.seed(42)
        n_samples = 200

        # Create mock historical data
        historical_df = pd.DataFrame({
            "player_id": np.tile(np.arange(20), 10),
            "gameweek": np.repeat(np.arange(6, 16), 20),
            "position": np.tile(["GKP", "GKP"] + ["DEF"]*4 + ["MID"]*8 + ["FWD"]*6, 10),
            "total_points": np.random.randint(0, 15, n_samples),
            "minutes": np.random.randint(0, 91, n_samples),
        })

        features_df = pd.DataFrame({
            f"feat_{i}": np.random.randn(n_samples) for i in range(10)
        })
        features_df["gameweek"] = historical_df["gameweek"].values
        features_df["position"] = historical_df["position"].values
        features_df["player_id"] = historical_df["player_id"].values

        target = historical_df["total_points"].values.astype(float)
        feature_names = [f"feat_{i}" for i in range(10)]

        return features_df, target, feature_names

    @patch("fpl_team_picker.domain.services.ml_training.trainer.load_training_data")
    @patch("fpl_team_picker.domain.services.ml_training.trainer.engineer_features")
    def test_train_unified_with_mocked_data(
        self, mock_engineer, mock_load, mock_data
    ):
        """Test train_unified with mocked data loading."""
        features_df, target, feature_names = mock_data

        # Setup mocks
        mock_load.return_value = tuple([pd.DataFrame()] * 12)
        mock_engineer.return_value = (features_df, target, feature_names)

        config = TrainingConfig(
            end_gw=15,
            regressor="random-forest",  # Faster than LightGBM for tests
            n_trials=2,  # Minimal for testing
            verbose=0,
        )
        trainer = MLTrainer(config)

        # This should work without actual data
        pipeline, metadata = trainer.train_unified()

        assert pipeline is not None
        assert "selected_features" in metadata
        assert "n_samples" in metadata
