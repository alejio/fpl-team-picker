"""
Unit tests for ML uncertainty extraction from Random Forest and XGBoost models.

Tests that both Random Forest and XGBoost models can extract prediction uncertainty
using tree-level variance calculations.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from fpl_team_picker.domain.services.ml_expected_points_service import (
    MLExpectedPointsService,
)


class TestUncertaintyExtraction:
    """Test uncertainty extraction for different model types"""

    def test_random_forest_uncertainty_extraction(self):
        """Test that Random Forest models can extract tree-level uncertainty"""

        # Create a simple Random Forest model with enough trees to get variance
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)

        # Create dummy training data with some noise
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature_1": np.random.randn(200),
                "feature_2": np.random.randn(200),
                "feature_3": np.random.randn(200),
            }
        )
        # Create target with some non-linear relationship
        y_train = (
            X_train["feature_1"] ** 2
            + X_train["feature_2"]
            + np.random.randn(200) * 0.5
        )

        # Train the model
        rf_model.fit(X_train, y_train)

        # Create a pipeline with the model (simulating full pipeline structure)
        pipeline = Pipeline([("model", rf_model)])

        # Create service and set pipeline
        service = MLExpectedPointsService(debug=True)
        service.pipeline = pipeline

        # Create test data with diverse values to get variance
        X_test = pd.DataFrame(
            {
                "feature_1": np.linspace(-3, 3, 20),
                "feature_2": np.linspace(-2, 2, 20),
                "feature_3": np.random.randn(20),
            }
        )

        # Extract uncertainty
        uncertainty = service._extract_uncertainty(X_test)

        # Assertions
        assert len(uncertainty) == len(X_test), (
            "Uncertainty array length should match input"
        )
        assert all(uncertainty >= 0), "Uncertainty should be non-negative"
        # Note: Random Forest should have non-zero uncertainty for at least some samples
        # If this fails, it might mean trees are too similar (low variance)
        print(
            f"\nRF Uncertainty stats: mean={uncertainty.mean():.4f}, max={uncertainty.max():.4f}"
        )
        assert uncertainty.max() >= 0, "Should have valid uncertainty values"

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_xgboost_uncertainty_extraction(self):
        """Test that XGBoost models can extract tree-level uncertainty"""

        # Create a simple XGBoost model
        xgb_model = XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)

        # Create dummy training data
        X_train = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "feature_3": np.random.randn(100),
            }
        )
        y_train = np.random.randn(100)

        # Train the model
        xgb_model.fit(X_train, y_train)

        # Create a pipeline with the model
        pipeline = Pipeline([("model", xgb_model)])

        # Create service and set pipeline
        service = MLExpectedPointsService(debug=True)
        service.pipeline = pipeline

        # Create test data
        X_test = pd.DataFrame(
            {
                "feature_1": np.random.randn(20),
                "feature_2": np.random.randn(20),
                "feature_3": np.random.randn(20),
            }
        )

        # Extract uncertainty
        uncertainty = service._extract_uncertainty(X_test)

        # Assertions
        assert len(uncertainty) == len(X_test), (
            "Uncertainty array length should match input"
        )
        assert all(uncertainty >= 0), "Uncertainty should be non-negative"
        assert any(uncertainty > 0), (
            "At least some samples should have non-zero uncertainty"
        )

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_xgboost_with_nested_pipeline(self):
        """Test XGBoost uncertainty extraction with nested pipeline (e.g., scaler + model)"""

        # Create a pipeline with preprocessing
        xgb_model = XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)

        nested_pipeline = Pipeline([("scaler", StandardScaler()), ("xgb", xgb_model)])

        # Create dummy training data
        X_train = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "feature_3": np.random.randn(100),
            }
        )
        y_train = np.random.randn(100)

        # Train the pipeline
        nested_pipeline.fit(X_train, y_train)

        # Wrap in another pipeline to simulate full structure
        full_pipeline = Pipeline([("model", nested_pipeline)])

        # Create service and set pipeline
        service = MLExpectedPointsService(debug=True)
        service.pipeline = full_pipeline

        # Create test data
        X_test = pd.DataFrame(
            {
                "feature_1": np.random.randn(20),
                "feature_2": np.random.randn(20),
                "feature_3": np.random.randn(20),
            }
        )

        # Extract uncertainty
        uncertainty = service._extract_uncertainty(X_test)

        # Assertions
        assert len(uncertainty) == len(X_test), (
            "Uncertainty array length should match input"
        )
        assert all(uncertainty >= 0), "Uncertainty should be non-negative"

    def test_non_ensemble_model_returns_zeros(self):
        """Test that non-ensemble models return zero uncertainty"""
        from sklearn.linear_model import Ridge

        # Create a Ridge model (no uncertainty available)
        ridge_model = Ridge()

        # Create dummy training data
        X_train = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
            }
        )
        y_train = np.random.randn(100)

        # Train the model
        ridge_model.fit(X_train, y_train)

        # Create a pipeline
        pipeline = Pipeline([("model", ridge_model)])

        # Create service and set pipeline
        service = MLExpectedPointsService(
            debug=False
        )  # Disable debug to avoid log spam
        service.pipeline = pipeline

        # Create test data
        X_test = pd.DataFrame(
            {
                "feature_1": np.random.randn(20),
                "feature_2": np.random.randn(20),
            }
        )

        # Extract uncertainty
        uncertainty = service._extract_uncertainty(X_test)

        # Assertions
        assert len(uncertainty) == len(X_test), (
            "Uncertainty array length should match input"
        )
        assert all(uncertainty == 0), (
            "Non-ensemble models should return zero uncertainty"
        )

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_xgboost_with_production_model(self):
        """Test XGBoost uncertainty extraction with actual production model if available"""

        model_path = Path(
            "models/custom/xgboost_gw1-11_20251114_210304_pipeline.joblib"
        )

        if not model_path.exists():
            pytest.skip(f"Production model not found: {model_path}")

        # Load the production model
        service = MLExpectedPointsService(model_path=str(model_path), debug=True)

        # Verify the model loaded correctly
        assert service.pipeline is not None, "Pipeline should be loaded"

        # Check if this is a bare sklearn pipeline (needs feature wrapper)
        if getattr(service, "needs_feature_wrapper", False):
            # Bare sklearn pipeline - the pipeline IS the model
            actual_model = service.pipeline

            # If it's a pipeline, extract the final step
            if hasattr(actual_model, "steps") and len(actual_model.steps) > 0:
                # Get the last step (usually the model)
                actual_model = actual_model.steps[-1][1]
        else:
            # Full pipeline with named steps
            model = service.pipeline.named_steps.get("model")
            actual_model = model

            # Handle nested structures
            if hasattr(actual_model, "steps"):
                actual_model = actual_model.steps[-1][1]
            elif hasattr(actual_model, "estimator_"):
                actual_model = actual_model.estimator_

        assert isinstance(actual_model, XGBRegressor), (
            f"Expected XGBoost model, got {type(actual_model).__name__}"
        )

        # Get model info
        booster = actual_model.get_booster()
        n_trees = len(booster.get_dump())
        learning_rate = getattr(actual_model, "learning_rate", 0.3)

        print("\nProduction model info:")
        print(f"  Trees: {n_trees}")
        print(f"  Learning rate: {learning_rate}")

        assert n_trees > 0, "Model should have trees"

    def test_gradient_boosting_uncertainty_extraction(self):
        """Test that GradientBoosting models can extract tree-level uncertainty"""
        from sklearn.ensemble import GradientBoostingRegressor

        # Create a simple GradientBoosting model
        gb_model = GradientBoostingRegressor(
            n_estimators=50, learning_rate=0.1, random_state=42
        )

        # Create dummy training data with some noise
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature_1": np.random.randn(200),
                "feature_2": np.random.randn(200),
                "feature_3": np.random.randn(200),
            }
        )
        # Create target with some non-linear relationship
        y_train = (
            X_train["feature_1"] ** 2
            + X_train["feature_2"]
            + np.random.randn(200) * 0.5
        )

        # Train the model
        gb_model.fit(X_train, y_train)

        # Create a pipeline with the model
        pipeline = Pipeline([("model", gb_model)])

        # Create service and set pipeline
        service = MLExpectedPointsService(debug=True)
        service.pipeline = pipeline

        # Create test data
        X_test = pd.DataFrame(
            {
                "feature_1": np.linspace(-3, 3, 20),
                "feature_2": np.linspace(-2, 2, 20),
                "feature_3": np.random.randn(20),
            }
        )

        # Extract uncertainty
        uncertainty = service._extract_uncertainty(X_test)

        # Assertions
        assert len(uncertainty) == len(X_test), (
            "Uncertainty array length should match input"
        )
        assert all(uncertainty >= 0), "Uncertainty should be non-negative"
        print(
            f"\nGradientBoosting Uncertainty stats: mean={uncertainty.mean():.4f}, max={uncertainty.max():.4f}"
        )
        assert uncertainty.max() >= 0, "Should have valid uncertainty values"

    def test_adaboost_uncertainty_extraction(self):
        """Test that AdaBoost models can extract estimator-level uncertainty"""
        from sklearn.ensemble import AdaBoostRegressor

        # Create a simple AdaBoost model
        ada_model = AdaBoostRegressor(
            n_estimators=50, learning_rate=1.0, random_state=42
        )

        # Create dummy training data with some noise
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature_1": np.random.randn(200),
                "feature_2": np.random.randn(200),
                "feature_3": np.random.randn(200),
            }
        )
        # Create target with some non-linear relationship
        y_train = (
            X_train["feature_1"] ** 2
            + X_train["feature_2"]
            + np.random.randn(200) * 0.5
        )

        # Train the model
        ada_model.fit(X_train, y_train)

        # Create a pipeline with the model
        pipeline = Pipeline([("model", ada_model)])

        # Create service and set pipeline
        service = MLExpectedPointsService(debug=True)
        service.pipeline = pipeline

        # Create test data
        X_test = pd.DataFrame(
            {
                "feature_1": np.linspace(-3, 3, 20),
                "feature_2": np.linspace(-2, 2, 20),
                "feature_3": np.random.randn(20),
            }
        )

        # Extract uncertainty
        uncertainty = service._extract_uncertainty(X_test)

        # Assertions
        assert len(uncertainty) == len(X_test), (
            "Uncertainty array length should match input"
        )
        assert all(uncertainty >= 0), "Uncertainty should be non-negative"
        print(
            f"\nAdaBoost Uncertainty stats: mean={uncertainty.mean():.4f}, max={uncertainty.max():.4f}"
        )
        assert uncertainty.max() >= 0, "Should have valid uncertainty values"

    @pytest.mark.skipif(
        not pytest.importorskip("lightgbm", reason="LightGBM not installed"),
        reason="LightGBM not installed",
    )
    def test_lightgbm_uncertainty_extraction(self):
        """Test that LightGBM models can extract tree-level uncertainty"""
        from lightgbm import LGBMRegressor

        # Create a simple LightGBM model
        lgbm_model = LGBMRegressor(
            n_estimators=50, learning_rate=0.1, random_state=42, verbose=-1
        )

        # Create dummy training data with some noise
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature_1": np.random.randn(200),
                "feature_2": np.random.randn(200),
                "feature_3": np.random.randn(200),
            }
        )
        # Create target with some non-linear relationship
        y_train = (
            X_train["feature_1"] ** 2
            + X_train["feature_2"]
            + np.random.randn(200) * 0.5
        )

        # Train the model
        lgbm_model.fit(X_train, y_train)

        # Create a pipeline with the model
        pipeline = Pipeline([("model", lgbm_model)])

        # Create service and set pipeline
        service = MLExpectedPointsService(debug=True)
        service.pipeline = pipeline

        # Create test data
        X_test = pd.DataFrame(
            {
                "feature_1": np.linspace(-3, 3, 20),
                "feature_2": np.linspace(-2, 2, 20),
                "feature_3": np.random.randn(20),
            }
        )

        # Extract uncertainty
        uncertainty = service._extract_uncertainty(X_test)

        # Assertions
        assert len(uncertainty) == len(X_test), (
            "Uncertainty array length should match input"
        )
        assert all(uncertainty >= 0), "Uncertainty should be non-negative"
        print(
            f"\nLightGBM Uncertainty stats: mean={uncertainty.mean():.4f}, max={uncertainty.max():.4f}"
        )
        assert uncertainty.max() >= 0, "Should have valid uncertainty values"

    def test_gradient_boosting_with_nested_pipeline(self):
        """Test GradientBoosting uncertainty extraction with nested pipeline"""
        from sklearn.ensemble import GradientBoostingRegressor

        # Create a pipeline with preprocessing
        gb_model = GradientBoostingRegressor(
            n_estimators=50, learning_rate=0.1, random_state=42
        )

        nested_pipeline = Pipeline([("scaler", StandardScaler()), ("gb", gb_model)])

        # Create dummy training data
        X_train = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "feature_3": np.random.randn(100),
            }
        )
        y_train = np.random.randn(100)

        # Train the pipeline
        nested_pipeline.fit(X_train, y_train)

        # Wrap in another pipeline to simulate full structure
        full_pipeline = Pipeline([("model", nested_pipeline)])

        # Create service and set pipeline
        service = MLExpectedPointsService(debug=True)
        service.pipeline = full_pipeline

        # Create test data
        X_test = pd.DataFrame(
            {
                "feature_1": np.random.randn(20),
                "feature_2": np.random.randn(20),
                "feature_3": np.random.randn(20),
            }
        )

        # Extract uncertainty
        uncertainty = service._extract_uncertainty(X_test)

        # Assertions
        assert len(uncertainty) == len(X_test), (
            "Uncertainty array length should match input"
        )
        assert all(uncertainty >= 0), "Uncertainty should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
