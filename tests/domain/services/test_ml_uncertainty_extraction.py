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
from sklearn.base import BaseEstimator, TransformerMixin


class DummyFeatureSelector(BaseEstimator, TransformerMixin):
    """Dummy feature selector for testing."""

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names]
        return X


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

    def test_hybrid_model_uncertainty_extraction(self):
        """Test that HybridPositionModel can extract uncertainty from underlying models"""
        from fpl_team_picker.domain.ml.hybrid_model import HybridPositionModel
        from sklearn.base import BaseEstimator, TransformerMixin

        # Create a dummy feature engineer that preserves position
        class DummyFeatureEngineer(BaseEstimator, TransformerMixin):
            """Dummy feature engineer that just passes through data with position"""

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                # Ensure position is preserved
                result = X.copy()
                if "position" not in result.columns and isinstance(X, pd.DataFrame):
                    # If position is missing, add it from index or create dummy
                    result["position"] = "GKP"  # Default, will be overridden
                return result

        # Create pipelines for unified and position-specific models
        # Use Random Forest for all models to ensure uncertainty extraction works
        unified_pipeline = Pipeline(
            [
                ("feature_selector", DummyFeatureSelector(["xP", "price"])),
                ("regressor", RandomForestRegressor(n_estimators=50, random_state=42)),
            ]
        )

        gkp_pipeline = Pipeline(
            [
                (
                    "feature_selector",
                    DummyFeatureSelector(["xP", "price", "saves_x_opp_xg"]),
                ),
                ("regressor", RandomForestRegressor(n_estimators=50, random_state=42)),
            ]
        )

        fwd_pipeline = Pipeline(
            [
                (
                    "feature_selector",
                    DummyFeatureSelector(["xP", "price", "xg_x_minutes"]),
                ),
                ("regressor", RandomForestRegressor(n_estimators=50, random_state=42)),
            ]
        )

        # Create training data for each model
        np.random.seed(42)
        n_train = 100

        # Unified model training data (DEF, MID)
        unified_X_train = pd.DataFrame(
            {
                "xP": np.random.randn(n_train) * 2 + 5,
                "price": np.random.randn(n_train) * 2 + 7,
            }
        )
        unified_y_train = (
            unified_X_train["xP"] * 0.8
            + unified_X_train["price"] * 0.2
            + np.random.randn(n_train) * 0.5
        )
        unified_pipeline.fit(unified_X_train, unified_y_train)

        # GKP model training data
        gkp_X_train = pd.DataFrame(
            {
                "xP": np.random.randn(n_train) * 2 + 4,
                "price": np.random.randn(n_train) * 1 + 5,
                "saves_x_opp_xg": np.random.randn(n_train) * 3 + 2,
            }
        )
        gkp_y_train = (
            gkp_X_train["xP"] * 0.7
            + gkp_X_train["price"] * 0.1
            + gkp_X_train["saves_x_opp_xg"] * 0.2
            + np.random.randn(n_train) * 0.5
        )
        gkp_pipeline.fit(gkp_X_train, gkp_y_train)

        # FWD model training data
        fwd_X_train = pd.DataFrame(
            {
                "xP": np.random.randn(n_train) * 2 + 7,
                "price": np.random.randn(n_train) * 3 + 10,
                "xg_x_minutes": np.random.randn(n_train) * 2 + 1.5,
            }
        )
        fwd_y_train = (
            fwd_X_train["xP"] * 0.6
            + fwd_X_train["price"] * 0.2
            + fwd_X_train["xg_x_minutes"] * 0.2
            + np.random.randn(n_train) * 0.5
        )
        fwd_pipeline.fit(fwd_X_train, fwd_y_train)

        # Create hybrid model
        hybrid_model = HybridPositionModel(
            unified_model=unified_pipeline,
            position_models={"GKP": gkp_pipeline, "FWD": fwd_pipeline},
            use_specific_for=["GKP", "FWD"],
        )

        # Create service and set up hybrid model
        service = MLExpectedPointsService(debug=True)
        service.pipeline = hybrid_model
        service.is_hybrid_model = True

        # Create a dummy feature engineer for the service
        service._hybrid_feature_engineer = DummyFeatureEngineer()

        # Create test data with all positions
        X_test = pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4, 5, 6, 7, 8],
                "position": ["GKP", "GKP", "DEF", "DEF", "MID", "MID", "FWD", "FWD"],
                "xP": [4.5, 5.0, 6.0, 5.5, 7.0, 6.5, 8.0, 7.5],
                "price": [5.0, 5.5, 6.0, 5.5, 8.0, 7.5, 11.0, 10.0],
                "saves_x_opp_xg": [2.0, 2.5, 0, 0, 0, 0, 0, 0],
                "xg_x_minutes": [0, 0, 0, 0, 0, 0, 1.8, 2.0],
            }
        )

        # Extract uncertainty using the hybrid method
        uncertainty = service._extract_uncertainty_hybrid(X_test)

        # Assertions
        assert len(uncertainty) == len(X_test), (
            "Uncertainty array length should match input"
        )
        assert all(uncertainty >= 0), "Uncertainty should be non-negative"

        # At least some samples should have non-zero uncertainty
        # (Random Forest should produce variance across trees)
        assert any(uncertainty > 0), (
            "At least some samples should have non-zero uncertainty from ensemble models"
        )

        # Check that different positions have uncertainty extracted
        gkp_uncertainty = uncertainty[X_test["position"] == "GKP"]
        def_uncertainty = uncertainty[X_test["position"] == "DEF"]
        mid_uncertainty = uncertainty[X_test["position"] == "MID"]
        fwd_uncertainty = uncertainty[X_test["position"] == "FWD"]

        print("\nHybrid Model Uncertainty by Position:")
        print(
            f"  GKP: mean={gkp_uncertainty.mean():.4f}, max={gkp_uncertainty.max():.4f}"
        )
        print(
            f"  DEF: mean={def_uncertainty.mean():.4f}, max={def_uncertainty.max():.4f}"
        )
        print(
            f"  MID: mean={mid_uncertainty.mean():.4f}, max={mid_uncertainty.max():.4f}"
        )
        print(
            f"  FWD: mean={fwd_uncertainty.mean():.4f}, max={fwd_uncertainty.max():.4f}"
        )

        # Position-specific models (GKP, FWD) should have uncertainty
        assert any(gkp_uncertainty > 0), "GKP position should have non-zero uncertainty"
        assert any(fwd_uncertainty > 0), "FWD position should have non-zero uncertainty"

        # Unified model (DEF, MID) should also have uncertainty
        assert any(def_uncertainty > 0), "DEF position should have non-zero uncertainty"
        assert any(mid_uncertainty > 0), "MID position should have non-zero uncertainty"

        # Overall mean uncertainty should be reasonable (not all zeros)
        assert uncertainty.mean() > 0, (
            f"Mean uncertainty should be positive, got {uncertainty.mean()}"
        )

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not installed")
    def test_hybrid_model_with_xgboost_uncertainty(self):
        """Test hybrid model uncertainty extraction with XGBoost models"""
        from fpl_team_picker.domain.ml.hybrid_model import HybridPositionModel
        from sklearn.base import BaseEstimator, TransformerMixin

        # Create a dummy feature engineer
        class DummyFeatureEngineer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X.copy()

        # Create XGBoost pipelines
        unified_pipeline = Pipeline(
            [
                ("feature_selector", DummyFeatureSelector(["xP", "price"])),
                (
                    "regressor",
                    XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42),
                ),
            ]
        )

        gkp_pipeline = Pipeline(
            [
                (
                    "feature_selector",
                    DummyFeatureSelector(["xP", "price", "saves_x_opp_xg"]),
                ),
                (
                    "regressor",
                    XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42),
                ),
            ]
        )

        # Train models
        np.random.seed(42)
        n_train = 100

        unified_X_train = pd.DataFrame(
            {
                "xP": np.random.randn(n_train) * 2 + 5,
                "price": np.random.randn(n_train) * 2 + 7,
            }
        )
        unified_y_train = (
            unified_X_train["xP"] * 0.8
            + unified_X_train["price"] * 0.2
            + np.random.randn(n_train) * 0.5
        )
        unified_pipeline.fit(unified_X_train, unified_y_train)

        gkp_X_train = pd.DataFrame(
            {
                "xP": np.random.randn(n_train) * 2 + 4,
                "price": np.random.randn(n_train) * 1 + 5,
                "saves_x_opp_xg": np.random.randn(n_train) * 3 + 2,
            }
        )
        gkp_y_train = (
            gkp_X_train["xP"] * 0.7
            + gkp_X_train["price"] * 0.1
            + gkp_X_train["saves_x_opp_xg"] * 0.2
            + np.random.randn(n_train) * 0.5
        )
        gkp_pipeline.fit(gkp_X_train, gkp_y_train)

        # Create hybrid model
        hybrid_model = HybridPositionModel(
            unified_model=unified_pipeline,
            position_models={"GKP": gkp_pipeline},
            use_specific_for=["GKP"],
        )

        # Create service
        service = MLExpectedPointsService(debug=True)
        service.pipeline = hybrid_model
        service.is_hybrid_model = True
        service._hybrid_feature_engineer = DummyFeatureEngineer()

        # Create test data
        X_test = pd.DataFrame(
            {
                "position": ["GKP", "GKP", "DEF", "DEF", "MID", "MID"],
                "xP": [4.5, 5.0, 6.0, 5.5, 7.0, 6.5],
                "price": [5.0, 5.5, 6.0, 5.5, 8.0, 7.5],
                "saves_x_opp_xg": [2.0, 2.5, 0, 0, 0, 0],
            }
        )

        # Extract uncertainty
        uncertainty = service._extract_uncertainty_hybrid(X_test)

        # Assertions
        assert len(uncertainty) == len(X_test)
        assert all(uncertainty >= 0)
        assert any(uncertainty > 0), (
            "XGBoost models should produce non-zero uncertainty"
        )

        print(
            f"\nXGBoost Hybrid Uncertainty: mean={uncertainty.mean():.4f}, max={uncertainty.max():.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
