"""Tests for ML pipeline construction."""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

from fpl_team_picker.domain.services.ml_training.pipelines import (
    get_regressor,
    get_param_space,
    create_preprocessor,
    build_pipeline,
    get_feature_groups,
    REGRESSOR_MAP,
)


class TestGetRegressor:
    """Tests for get_regressor function."""

    def test_lightgbm_regressor(self):
        """Test LightGBM regressor creation."""
        reg = get_regressor("lightgbm", random_seed=42)
        assert reg is not None
        assert hasattr(reg, "fit")
        assert hasattr(reg, "predict")

    def test_xgboost_regressor(self):
        """Test XGBoost regressor creation."""
        reg = get_regressor("xgboost", random_seed=42)
        assert reg is not None
        assert hasattr(reg, "fit")

    def test_random_forest_regressor(self):
        """Test Random Forest regressor creation."""
        reg = get_regressor("random-forest", random_seed=42)
        assert reg is not None
        assert reg.random_state == 42

    def test_gradient_boost_regressor(self):
        """Test Gradient Boosting regressor creation."""
        reg = get_regressor("gradient-boost", random_seed=42)
        assert reg is not None

    def test_ridge_regressor(self):
        """Test Ridge regressor creation."""
        reg = get_regressor("ridge", random_seed=42)
        assert reg is not None

    def test_invalid_regressor(self):
        """Test that invalid regressor raises ValueError."""
        with pytest.raises(ValueError, match="Unknown regressor"):
            get_regressor("invalid_regressor")

    def test_random_seed_propagation(self):
        """Test that random seed is propagated."""
        reg1 = get_regressor("random-forest", random_seed=42)
        reg2 = get_regressor("random-forest", random_seed=123)

        assert reg1.random_state == 42
        assert reg2.random_state == 123


class TestGetParamSpace:
    """Tests for get_param_space function."""

    def test_lightgbm_param_space(self):
        """Test LightGBM parameter space."""
        params = get_param_space("lightgbm")

        assert "regressor__n_estimators" in params
        assert "regressor__learning_rate" in params
        assert "regressor__max_depth" in params
        assert "regressor__num_leaves" in params

    def test_xgboost_param_space(self):
        """Test XGBoost parameter space."""
        params = get_param_space("xgboost")

        assert "regressor__n_estimators" in params
        assert "regressor__learning_rate" in params
        assert "regressor__reg_alpha" in params
        assert "regressor__reg_lambda" in params

    def test_random_forest_param_space(self):
        """Test Random Forest parameter space."""
        params = get_param_space("random-forest")

        assert "regressor__n_estimators" in params
        assert "regressor__max_depth" in params
        assert "regressor__min_samples_split" in params

    def test_ridge_param_space(self):
        """Test Ridge parameter space."""
        params = get_param_space("ridge")

        assert "regressor__alpha" in params
        assert "regressor__solver" in params

    def test_invalid_regressor_param_space(self):
        """Test that invalid regressor raises ValueError."""
        with pytest.raises(ValueError, match="Unknown regressor"):
            get_param_space("invalid_regressor")

    def test_all_regressors_have_param_spaces(self):
        """Test that all regressors in REGRESSOR_MAP have param spaces."""
        for regressor_name in REGRESSOR_MAP.keys():
            params = get_param_space(regressor_name)
            assert isinstance(params, dict)
            assert len(params) > 0


class TestCreatePreprocessor:
    """Tests for create_preprocessor function."""

    def test_standard_preprocessor(self):
        """Test standard preprocessing creates StandardScaler."""
        preprocessor = create_preprocessor("standard", ["feat1", "feat2"])
        assert isinstance(preprocessor, StandardScaler)

    def test_robust_preprocessor(self):
        """Test robust preprocessing creates RobustScaler."""
        preprocessor = create_preprocessor("robust", ["feat1", "feat2"])
        assert isinstance(preprocessor, RobustScaler)

    def test_grouped_preprocessor(self):
        """Test grouped preprocessing creates ColumnTransformer."""
        from sklearn.compose import ColumnTransformer

        # Use actual feature names from the groups
        feature_groups = get_feature_groups()
        feature_names = (
            feature_groups["binary_features"][:2]
            + feature_groups["continuous_features"][:2]
        )

        preprocessor = create_preprocessor("grouped", feature_names)
        assert isinstance(preprocessor, ColumnTransformer)

    def test_invalid_preprocessor(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preprocessing"):
            create_preprocessor("invalid_strategy", ["feat1"])


class TestBuildPipeline:
    """Tests for build_pipeline function."""

    def test_pipeline_structure(self):
        """Test pipeline has correct structure."""
        feature_names = ["feat1", "feat2", "feat3"]
        pipeline = build_pipeline("lightgbm", feature_names)

        assert isinstance(pipeline, Pipeline)
        assert "feature_selector" in pipeline.named_steps
        assert "preprocessor" in pipeline.named_steps
        assert "regressor" in pipeline.named_steps

    def test_feature_selector_in_pipeline(self):
        """Test that FeatureSelector is configured correctly."""
        feature_names = ["feat1", "feat2"]
        pipeline = build_pipeline("lightgbm", feature_names)

        selector = pipeline.named_steps["feature_selector"]
        assert selector.feature_names == feature_names

    def test_pipeline_with_params(self):
        """Test pipeline can be built with initial params."""
        feature_names = ["feat1", "feat2"]
        params = {"regressor__n_estimators": 100}

        pipeline = build_pipeline(
            "lightgbm",
            feature_names,
            params=params,
        )

        # Check params were set
        regressor = pipeline.named_steps["regressor"]
        assert regressor.n_estimators == 100

    def test_pipeline_with_different_preprocessing(self):
        """Test pipeline with different preprocessing strategies."""
        feature_names = ["feat1", "feat2"]

        # Standard
        pipeline = build_pipeline("lightgbm", feature_names, preprocessing="standard")
        assert isinstance(pipeline.named_steps["preprocessor"], StandardScaler)

        # Robust
        pipeline = build_pipeline("lightgbm", feature_names, preprocessing="robust")
        assert isinstance(pipeline.named_steps["preprocessor"], RobustScaler)

    def test_pipeline_fit_predict(self):
        """Test that built pipeline can fit and predict."""
        # Create simple test data
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feat1": np.random.randn(100),
                "feat2": np.random.randn(100),
            }
        )
        y = np.random.randn(100)

        pipeline = build_pipeline("random-forest", ["feat1", "feat2"])
        pipeline.fit(X, y)

        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)


class TestFeatureGroups:
    """Tests for feature groups definition."""

    def test_feature_groups_structure(self):
        """Test feature groups are properly structured."""
        groups = get_feature_groups()

        assert "binary_features" in groups
        assert "count_features" in groups
        assert "continuous_features" in groups
        assert "percentage_features" in groups
        assert "probability_features" in groups

    def test_binary_features_are_binary(self):
        """Test binary features are correct."""
        groups = get_feature_groups()
        binary = groups["binary_features"]

        assert "is_home" in binary
        assert "is_primary_penalty_taker" in binary

    def test_no_duplicate_features(self):
        """Test no feature appears in multiple groups."""
        groups = get_feature_groups()

        all_features = []
        for group_features in groups.values():
            all_features.extend(group_features)

        # Check for duplicates
        assert len(all_features) == len(set(all_features)), (
            "Duplicate features in groups"
        )
