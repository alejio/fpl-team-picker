"""Tests for ML training configuration models."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from fpl_team_picker.domain.services.ml_training.config import (
    TrainingConfig,
    PositionTrainingConfig,
    EvaluationConfig,
    HybridPipelineConfig,
)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.start_gw == 1
        assert config.end_gw == 12
        assert config.regressor == "lightgbm"
        assert config.feature_selection == "none"
        assert config.keep_penalty_features is False
        assert config.preprocessing == "standard"
        assert config.n_trials == 50
        assert config.cv_folds is None
        assert config.scorer == "neg_mean_absolute_error"
        assert config.random_seed == 42
        assert config.n_jobs == -1
        assert config.verbose == 1

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = TrainingConfig(
            start_gw=3,
            end_gw=15,
            regressor="xgboost",
            n_trials=100,
            scorer="spearman",
        )

        assert config.start_gw == 3
        assert config.end_gw == 15
        assert config.regressor == "xgboost"
        assert config.n_trials == 100
        assert config.scorer == "spearman"

    def test_regressor_validation(self):
        """Test that invalid regressors are rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(regressor="invalid_regressor")

    def test_feature_selection_validation(self):
        """Test that invalid feature selection is rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(feature_selection="invalid_method")

    def test_preprocessing_validation(self):
        """Test that invalid preprocessing is rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(preprocessing="invalid_preprocessing")

    def test_n_trials_validation(self):
        """Test that n_trials must be positive."""
        with pytest.raises(ValidationError):
            TrainingConfig(n_trials=0)

        with pytest.raises(ValidationError):
            TrainingConfig(n_trials=-5)

    def test_cv_folds_validation(self):
        """Test that cv_folds must be at least 2 if provided."""
        # None is allowed
        config = TrainingConfig(cv_folds=None)
        assert config.cv_folds is None

        # 2+ is valid
        config = TrainingConfig(cv_folds=5)
        assert config.cv_folds == 5

        # Less than 2 is invalid
        with pytest.raises(ValidationError):
            TrainingConfig(cv_folds=1)

    def test_verbose_validation(self):
        """Test that verbose must be 0, 1, or 2."""
        for v in [0, 1, 2]:
            config = TrainingConfig(verbose=v)
            assert config.verbose == v

        with pytest.raises(ValidationError):
            TrainingConfig(verbose=3)

    def test_output_dir_path_conversion(self):
        """Test that output_dir is converted to Path."""
        config = TrainingConfig(output_dir=Path("custom/path"))
        assert isinstance(config.output_dir, Path)
        assert str(config.output_dir) == "custom/path"


class TestPositionTrainingConfig:
    """Tests for PositionTrainingConfig."""

    def test_default_positions(self):
        """Test default positions."""
        config = PositionTrainingConfig()
        assert config.positions == ["GKP", "DEF", "MID", "FWD"]

    def test_default_regressors(self):
        """Test default regressors list."""
        config = PositionTrainingConfig()
        assert "lightgbm" in config.regressors
        assert "xgboost" in config.regressors
        assert "random-forest" in config.regressors
        assert "gradient-boost" in config.regressors

    def test_custom_positions(self):
        """Test custom positions list."""
        config = PositionTrainingConfig(positions=["GKP", "FWD"])
        assert config.positions == ["GKP", "FWD"]

    def test_inherits_training_config(self):
        """Test that it inherits from TrainingConfig."""
        config = PositionTrainingConfig(
            end_gw=10,
            n_trials=30,
        )
        assert config.end_gw == 10
        assert config.n_trials == 30

    def test_default_output_dir(self):
        """Test default output directory."""
        config = PositionTrainingConfig()
        assert "position_specific" in str(config.output_dir)


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""

    def test_default_values(self):
        """Test default evaluation config."""
        config = EvaluationConfig()

        assert config.holdout_gws == 2
        assert config.compute_per_position is True
        assert config.compute_per_gameweek is True
        assert config.captain_top_k == 3
        assert config.squad_size == 15

    def test_holdout_validation(self):
        """Test holdout_gws validation."""
        config = EvaluationConfig(holdout_gws=1)
        assert config.holdout_gws == 1

        with pytest.raises(ValidationError):
            EvaluationConfig(holdout_gws=0)

    def test_squad_size_validation(self):
        """Test squad_size must be 11-15."""
        config = EvaluationConfig(squad_size=11)
        assert config.squad_size == 11

        with pytest.raises(ValidationError):
            EvaluationConfig(squad_size=10)

        with pytest.raises(ValidationError):
            EvaluationConfig(squad_size=16)


class TestHybridPipelineConfig:
    """Tests for HybridPipelineConfig."""

    def test_default_values(self):
        """Test default hybrid config."""
        config = HybridPipelineConfig()

        assert config.end_gw == 12
        assert config.holdout_gws == 2
        assert config.improvement_threshold == 0.02

    def test_train_end_gw_computed(self):
        """Test train_end_gw is computed correctly."""
        config = HybridPipelineConfig(end_gw=12, holdout_gws=2)
        assert config.train_end_gw == 10

        config = HybridPipelineConfig(end_gw=15, holdout_gws=3)
        assert config.train_end_gw == 12

    def test_holdout_gw_range_computed(self):
        """Test holdout_gw_range is computed correctly."""
        config = HybridPipelineConfig(end_gw=12, holdout_gws=2)
        assert config.holdout_gw_range == [11, 12]

        config = HybridPipelineConfig(end_gw=15, holdout_gws=3)
        assert config.holdout_gw_range == [13, 14, 15]

    def test_improvement_threshold_validation(self):
        """Test improvement_threshold is between 0 and 1."""
        config = HybridPipelineConfig(improvement_threshold=0.05)
        assert config.improvement_threshold == 0.05

        config = HybridPipelineConfig(improvement_threshold=0.0)
        assert config.improvement_threshold == 0.0

        with pytest.raises(ValidationError):
            HybridPipelineConfig(improvement_threshold=-0.1)

        with pytest.raises(ValidationError):
            HybridPipelineConfig(improvement_threshold=1.5)

    def test_unified_regressors_default(self):
        """Test default unified regressors."""
        config = HybridPipelineConfig()
        assert len(config.unified_regressors) == 4
        assert "lightgbm" in config.unified_regressors

    def test_position_regressors_default(self):
        """Test default position regressors."""
        config = HybridPipelineConfig()
        assert len(config.position_regressors) == 4
        assert "xgboost" in config.position_regressors
