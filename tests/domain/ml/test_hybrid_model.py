"""Unit tests for Hybrid Position Model.

Tests for:
- Model initialization and validation
- Position-based routing (GKP/FWD use specific, DEF/MID use unified)
- Position-specific feature addition
- Prediction with missing features
- from_paths class method
- Metadata and introspection
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin

from fpl_team_picker.domain.ml.hybrid_model import (
    HybridPositionModel,
    add_position_features,
)


class DummyFeatureSelector(BaseEstimator, TransformerMixin):
    """Dummy feature selector for testing."""

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names] if isinstance(X, pd.DataFrame) else X


@pytest.fixture
def create_pipeline():
    """Create a dummy sklearn pipeline for testing."""

    def _create(feature_names=None):
        if feature_names is None:
            feature_names = ["xP", "price", "position"]

        pipeline = Pipeline(
            [
                ("feature_selector", DummyFeatureSelector(feature_names)),
                ("scaler", StandardScaler()),
                ("regressor", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )
        return pipeline

    return _create


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "position": ["GKP", "GKP", "DEF", "DEF", "MID", "MID", "FWD", "FWD"],
            "xP": [5.0, 4.5, 6.0, 5.5, 7.0, 6.5, 8.0, 7.5],
            "price": [5.5, 5.0, 6.0, 5.5, 8.0, 7.5, 11.0, 10.0],
            "rolling_5gw_saves": [3.0, 2.5, 0, 0, 0, 0, 0, 0],
            "rolling_5gw_xg": [0, 0, 0.5, 0.4, 1.2, 1.0, 2.0, 1.8],
            "rolling_5gw_minutes": [90, 85, 90, 88, 90, 87, 90, 85],
            "clean_sheet_probability_enhanced": [0.4, 0.35, 0.5, 0.45, 0, 0, 0, 0],
        }
    )


class TestAddPositionFeatures:
    """Test add_position_features function."""

    def test_add_gkp_features(self):
        """Test adding GKP-specific features."""
        df = pd.DataFrame(
            {
                "rolling_5gw_saves": [3.0],
                "opponent_rolling_5gw_xg": [1.5],
                "clean_sheet_probability_enhanced": [0.4],
                "rolling_5gw_minutes": [90],
            }
        )

        result = add_position_features(df, "GKP")

        assert "saves_x_opp_xg" in result.columns
        assert "clean_sheet_potential" in result.columns

    def test_add_def_features(self):
        """Test adding DEF-specific features."""
        df = pd.DataFrame(
            {
                "clean_sheet_probability_enhanced": [0.5],
                "rolling_5gw_minutes": [90],
                "rolling_5gw_xg": [0.5],
                "rolling_5gw_threat": [50],
            }
        )

        result = add_position_features(df, "DEF")

        assert "cs_x_minutes" in result.columns
        assert "goal_threat_def" in result.columns

    def test_add_mid_features(self):
        """Test adding MID-specific features."""
        df = pd.DataFrame(
            {
                "rolling_5gw_xg": [1.0],
                "rolling_5gw_xa": [0.8],
                "rolling_5gw_creativity": [100],
                "rolling_5gw_threat": [80],
            }
        )

        result = add_position_features(df, "MID")

        assert "xgi_combined" in result.columns
        assert "creativity_x_threat" in result.columns

    def test_add_fwd_features(self):
        """Test adding FWD-specific features."""
        df = pd.DataFrame(
            {
                "rolling_5gw_xg": [2.0],
                "rolling_5gw_xa": [0.5],
                "rolling_5gw_minutes": [90],
            }
        )

        result = add_position_features(df, "FWD")

        assert "xg_x_minutes" in result.columns
        assert "goal_involvement" in result.columns

    def test_add_features_handles_missing_columns(self):
        """Test that missing columns are handled gracefully."""
        df = pd.DataFrame({"xP": [5.0]})

        # Should not raise error, just add 0.0 for missing features
        result = add_position_features(df, "GKP")

        assert isinstance(result, pd.DataFrame)


class TestHybridPositionModelInit:
    """Test HybridPositionModel initialization."""

    def test_init_default_use_specific(self, create_pipeline):
        """Test initialization with default use_specific_for."""
        unified = create_pipeline()
        position_models = {
            "GKP": create_pipeline(),
            "FWD": create_pipeline(),
        }

        model = HybridPositionModel(
            unified_model=unified, position_models=position_models
        )

        assert model.use_specific_for == ["GKP", "FWD"]
        assert model.unified_model == unified
        assert model.position_models == position_models

    def test_init_custom_use_specific(self, create_pipeline):
        """Test initialization with custom use_specific_for."""
        unified = create_pipeline()
        position_models = {
            "GKP": create_pipeline(),
            "DEF": create_pipeline(),
        }

        model = HybridPositionModel(
            unified_model=unified,
            position_models=position_models,
            use_specific_for=["GKP", "DEF"],
        )

        assert model.use_specific_for == ["GKP", "DEF"]

    def test_init_missing_position_model_raises_error(self, create_pipeline):
        """Test that missing position model raises ValueError."""
        unified = create_pipeline()
        position_models = {"GKP": create_pipeline()}

        with pytest.raises(ValueError, match="Position 'FWD' in use_specific_for"):
            HybridPositionModel(
                unified_model=unified,
                position_models=position_models,
                use_specific_for=["GKP", "FWD"],
            )

    def test_init_metadata(self, create_pipeline):
        """Test that metadata is stored correctly."""
        unified = create_pipeline()
        position_models = {"GKP": create_pipeline()}

        model = HybridPositionModel(
            unified_model=unified,
            position_models=position_models,
            use_specific_for=["GKP"],
        )

        assert "use_specific_for" in model.metadata
        assert "unified_positions" in model.metadata
        assert model.metadata["use_specific_for"] == ["GKP"]
        assert "DEF" in model.metadata["unified_positions"]
        assert "MID" in model.metadata["unified_positions"]
        assert "FWD" in model.metadata["unified_positions"]


class TestHybridPositionModelPredict:
    """Test HybridPositionModel.predict method."""

    def test_predict_routes_to_position_models(self, create_pipeline, sample_features):
        """Test that predictions route to correct models."""
        # Create pipelines with different feature sets
        unified = create_pipeline(["xP", "price"])
        gkp_model = create_pipeline(["xP", "price"])
        fwd_model = create_pipeline(["xP", "price"])

        # Fit models with simple data
        unified.fit(sample_features[["xP", "price"]], [1.0] * len(sample_features))
        gkp_data = sample_features[sample_features["position"] == "GKP"][
            ["xP", "price"]
        ]
        gkp_model.fit(gkp_data, [1.0] * len(gkp_data))
        fwd_data = sample_features[sample_features["position"] == "FWD"][
            ["xP", "price"]
        ]
        fwd_model.fit(fwd_data, [1.0] * len(fwd_data))

        model = HybridPositionModel(
            unified_model=unified,
            position_models={"GKP": gkp_model, "FWD": fwd_model},
            use_specific_for=["GKP", "FWD"],
        )

        # Use features that match what models expect
        predictions = model.predict(sample_features[["xP", "price", "position"]])

        assert len(predictions) == len(sample_features)
        assert isinstance(predictions, np.ndarray)

    def test_predict_requires_position_column(self, create_pipeline, sample_features):
        """Test that predict requires position column."""
        unified = create_pipeline(["xP", "price"])
        unified.fit(sample_features[["xP", "price"]], [1.0] * len(sample_features))
        # Create model with dummy position models to satisfy validation
        dummy_model = create_pipeline(["xP", "price"])
        dummy_model.fit(sample_features[["xP", "price"]], [1.0] * len(sample_features))
        model = HybridPositionModel(
            unified_model=unified,
            position_models={"GKP": dummy_model, "FWD": dummy_model},
            use_specific_for=["GKP", "FWD"],
        )

        df_without_position = pd.DataFrame({"xP": [5.0], "price": [6.0]})

        with pytest.raises(ValueError, match="must contain 'position' column"):
            model.predict(df_without_position)

    def test_predict_requires_dataframe(self, create_pipeline, sample_features):
        """Test that predict requires DataFrame input."""
        unified = create_pipeline(["xP", "price"])
        unified.fit(sample_features[["xP", "price"]], [1.0] * len(sample_features))
        dummy_model = create_pipeline(["xP", "price"])
        dummy_model.fit(sample_features[["xP", "price"]], [1.0] * len(sample_features))
        model = HybridPositionModel(
            unified_model=unified,
            position_models={"GKP": dummy_model, "FWD": dummy_model},
            use_specific_for=["GKP", "FWD"],
        )

        with pytest.raises(TypeError, match="requires DataFrame input"):
            model.predict(np.array([[1, 2, 3]]))

    def test_predict_all_positions(self, create_pipeline, sample_features):
        """Test that all positions are predicted."""
        unified = create_pipeline(["xP", "price"])
        unified.fit(sample_features[["xP", "price"]], [1.0] * len(sample_features))
        gkp_model = create_pipeline(["xP", "price"])
        fwd_model = create_pipeline(["xP", "price"])
        gkp_model.fit(
            sample_features[sample_features["position"] == "GKP"][["xP", "price"]],
            [1.0] * 2,
        )
        fwd_model.fit(
            sample_features[sample_features["position"] == "FWD"][["xP", "price"]],
            [1.0] * 2,
        )

        model = HybridPositionModel(
            unified_model=unified,
            position_models={"GKP": gkp_model, "FWD": fwd_model},
            use_specific_for=["GKP", "FWD"],
        )

        predictions = model.predict(sample_features[["xP", "price", "position"]])

        assert len(predictions) == len(sample_features)
        assert all(not np.isnan(p) for p in predictions)


class TestHybridPositionModelFromPaths:
    """Test HybridPositionModel.from_paths class method."""

    def test_from_paths_loads_models(self, create_pipeline, sample_features):
        """Test loading models from file paths."""
        with TemporaryDirectory() as tmpdir:
            unified_path = Path(tmpdir) / "unified.joblib"
            gkp_path = Path(tmpdir) / "gkp.joblib"

            unified = create_pipeline(["xP", "price"])
            gkp_model = create_pipeline(["xP", "price"])

            unified.fit(sample_features[["xP", "price"]], [1.0] * len(sample_features))
            gkp_model.fit(
                sample_features[sample_features["position"] == "GKP"][["xP", "price"]],
                [1.0] * 2,
            )

            joblib.dump(unified, unified_path)
            joblib.dump(gkp_model, gkp_path)

            model = HybridPositionModel.from_paths(
                unified_model_path=str(unified_path),
                position_model_paths={"GKP": str(gkp_path)},
                use_specific_for=["GKP"],
            )

            assert isinstance(model, HybridPositionModel)
            assert model.use_specific_for == ["GKP"]
            assert len(model.position_models) == 1


class TestHybridPositionModelMetadata:
    """Test HybridPositionModel metadata and introspection."""

    def test_get_model_for_position_specific(self, create_pipeline):
        """Test getting position-specific model."""
        unified = create_pipeline()
        gkp_model = create_pipeline()

        model = HybridPositionModel(
            unified_model=unified,
            position_models={"GKP": gkp_model},
            use_specific_for=["GKP"],
        )

        assert model.get_model_for_position("GKP") == gkp_model

    def test_get_model_for_position_unified(self, create_pipeline):
        """Test getting unified model for non-specific position."""
        unified = create_pipeline()
        gkp_model = create_pipeline()

        model = HybridPositionModel(
            unified_model=unified,
            position_models={"GKP": gkp_model},
            use_specific_for=["GKP"],
        )

        assert model.get_model_for_position("DEF") == unified
        assert model.get_model_for_position("MID") == unified
        assert model.get_model_for_position("FWD") == unified

    def test_repr(self, create_pipeline):
        """Test string representation."""
        unified = create_pipeline()
        gkp_model = create_pipeline()

        model = HybridPositionModel(
            unified_model=unified,
            position_models={"GKP": gkp_model},
            use_specific_for=["GKP"],
        )

        repr_str = repr(model)
        assert "GKP" in repr_str
        assert "DEF" in repr_str or "MID" in repr_str or "FWD" in repr_str
