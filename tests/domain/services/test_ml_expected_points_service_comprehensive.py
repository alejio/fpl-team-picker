"""Comprehensive tests for MLExpectedPointsService.

Tests for:
- Model loading (hybrid, legacy, standard pipelines)
- calculate_expected_points with different model types
- calculate_3gw_expected_points
- Uncertainty extraction for different model types
- Feature importance extraction
- Model save/load
- Error handling paths
- Ensemble predictions
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from fpl_team_picker.domain.services.ml_expected_points_service import (
    MLExpectedPointsService,
)
from fpl_team_picker.domain.ml import HybridPositionModel


@pytest.fixture
def sample_players_data():
    """Sample current player data."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "web_name": ["Player1", "Player2", "Player3"],
            "position": ["DEF", "MID", "FWD"],
            "team": ["Team1", "Team2", "Team3"],
            "team_id": [1, 2, 3],
            "price": [5.0, 8.0, 10.0],
        }
    )


@pytest.fixture
def sample_teams_data():
    """Sample teams data."""
    return pd.DataFrame(
        {
            "team_id": [1, 2, 3],
            "name": ["Team1", "Team2", "Team3"],
        }
    )


@pytest.fixture
def sample_fixtures_data():
    """Sample fixtures data."""
    return pd.DataFrame(
        {
            "event": [11],
            "home_team_id": [1],
            "away_team_id": [2],
        }
    )


@pytest.fixture
def sample_historical_data():
    """Historical live data for GW1-10."""
    data = []
    for gw in range(1, 11):
        for player_id in [1, 2, 3]:
            data.append(
                {
                    "player_id": player_id,
                    "gameweek": gw,
                    "event": gw,
                    "total_points": 5 + (player_id * gw % 3),
                    "minutes": 90,
                    "goals_scored": 0,
                    "assists": 0,
                    "clean_sheets": 1 if player_id == 1 else 0,
                    "goals_conceded": 1,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "saves": 0,
                    "bonus": 0,
                    "bps": 20,
                    "influence": 30.0,
                    "creativity": 20.0,
                    "threat": 10.0,
                    "ict_index": 6.0,
                    "expected_goals": 0.2,
                    "expected_assists": 0.1,
                    "expected_goal_involvements": 0.3,
                    "expected_goals_conceded": 1.0,
                    "value": 50 + player_id * 10,
                    "position": "DEF"
                    if player_id == 1
                    else ("MID" if player_id == 2 else "FWD"),
                    "team_id": player_id,
                }
            )
    return pd.DataFrame(data)


@pytest.fixture
def sample_enhanced_data():
    """Sample enhanced data sources."""
    ownership_trends = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "gameweek": [11, 11, 11],
            "selected_by_percent": [10.0, 15.0, 20.0],
        }
    )
    value_analysis = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "gameweek": [11, 11, 11],
            "points_per_pound": [1.0, 1.0, 1.0],
        }
    )
    fixture_difficulty = pd.DataFrame(
        {
            "team_id": [1, 2, 3],
            "gameweek": [11, 11, 11],
            "congestion_difficulty": [1.0, 1.0, 1.0],
        }
    )
    return {
        "ownership_trends_df": ownership_trends,
        "value_analysis_df": value_analysis,
        "fixture_difficulty_df": fixture_difficulty,
    }


class TestModelLoading:
    """Test model loading functionality."""

    def test_load_standard_pipeline(self, tmp_path):
        """Test loading a standard pipeline with feature_engineer."""
        # Create a pipeline with feature_engineer step
        # Use a simple transformer as feature_engineer
        from sklearn.preprocessing import FunctionTransformer

        pipeline = Pipeline(
            [
                ("feature_engineer", FunctionTransformer()),
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )

        model_path = tmp_path / "model.joblib"
        joblib.dump(pipeline, model_path)

        # Mock load_pipeline to return our pipeline
        with patch(
            "fpl_team_picker.domain.services.ml_expected_points_service.load_pipeline"
        ) as mock_load:
            mock_load.return_value = (pipeline, {})
            service = MLExpectedPointsService(model_path=str(model_path))
            assert service.pipeline is not None
            assert not getattr(service, "needs_feature_wrapper", False)

    def test_load_legacy_pipeline(self, tmp_path):
        """Test loading a legacy bare sklearn pipeline."""
        # Create a bare model (no feature_engineer)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model_path = tmp_path / "legacy_model.joblib"
        joblib.dump(model, model_path)

        service = MLExpectedPointsService(model_path=str(model_path))
        assert service.pipeline is not None
        assert getattr(service, "needs_feature_wrapper", True)

    def test_load_hybrid_model(self, tmp_path):
        """Test loading a HybridPositionModel."""
        # Create a hybrid model
        unified = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )
        gkp_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )
        hybrid = HybridPositionModel(
            unified_model=unified,
            position_models={"GKP": gkp_model},
            use_specific_for=["GKP"],
        )

        model_path = tmp_path / "hybrid_model.joblib"
        joblib.dump(hybrid, model_path)

        service = MLExpectedPointsService(model_path=str(model_path))
        assert service.pipeline is not None
        assert getattr(service, "is_hybrid_model", False)  # Will be False until loaded

    def test_load_missing_file_raises_error(self):
        """Test that loading missing file raises error."""
        # The error is raised during _load_pretrained_model which is called in __init__
        # But it might not raise if the file doesn't exist - let's check the actual behavior
        service = MLExpectedPointsService(model_path="nonexistent.joblib")
        # The service might initialize but pipeline will be None
        # Actually, looking at the code, it should raise during __init__ if file doesn't exist
        # Let's test by calling _load_pretrained_model directly
        service = MLExpectedPointsService()
        service.model_path = "nonexistent.joblib"
        with pytest.raises(ValueError, match="Failed to load pre-trained model"):
            service._load_pretrained_model()

    def test_init_without_model_path(self):
        """Test initialization without model path."""
        service = MLExpectedPointsService()
        assert service.pipeline is None


class TestCalculateExpectedPoints:
    """Test calculate_expected_points method."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing."""
        pipeline = Mock()
        pipeline.named_steps = {
            "feature_engineer": Mock(),
        }
        pipeline.predict = Mock(return_value=np.array([5.0, 6.0, 7.0]))
        return pipeline

    def test_calculate_expected_points_no_pipeline_raises_error(
        self,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_historical_data,
    ):
        """Test that missing pipeline raises error."""
        service = MLExpectedPointsService()
        service.pipeline = None

        with pytest.raises(ValueError, match="No pre-trained model loaded"):
            service.calculate_expected_points(
                players_data=sample_players_data,
                teams_data=sample_teams_data,
                xg_rates_data=pd.DataFrame(),
                fixtures_data=sample_fixtures_data,
                target_gameweek=11,
                live_data=sample_historical_data,
            )

    def test_calculate_expected_points_insufficient_historical_data(
        self,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        mock_pipeline,
    ):
        """Test that insufficient historical data raises error."""
        service = MLExpectedPointsService()
        service.pipeline = mock_pipeline

        # Only 3 gameweeks (need 5+)
        short_data = pd.DataFrame(
            [
                {"player_id": 1, "gameweek": 1, "total_points": 5, "minutes": 90},
                {"player_id": 1, "gameweek": 2, "total_points": 5, "minutes": 90},
                {"player_id": 1, "gameweek": 3, "total_points": 5, "minutes": 90},
            ]
        )

        with pytest.raises(ValueError, match="Need at least 5 historical gameweeks"):
            service.calculate_expected_points(
                players_data=sample_players_data,
                teams_data=sample_teams_data,
                xg_rates_data=pd.DataFrame(),
                fixtures_data=sample_fixtures_data,
                target_gameweek=4,
                live_data=short_data,
            )

    def test_calculate_expected_points_empty_historical_data(
        self,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        mock_pipeline,
    ):
        """Test that empty historical data raises error."""
        service = MLExpectedPointsService()
        service.pipeline = mock_pipeline

        with pytest.raises(ValueError, match="No historical live_data provided"):
            service.calculate_expected_points(
                players_data=sample_players_data,
                teams_data=sample_teams_data,
                xg_rates_data=pd.DataFrame(),
                fixtures_data=sample_fixtures_data,
                target_gameweek=11,
                live_data=pd.DataFrame(),
            )


class TestCalculate3GWExpectedPoints:
    """Test calculate_3gw_expected_points method."""

    def test_calculate_3gw_expected_points_basic(
        self,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_historical_data,
    ):
        """Test basic 3GW cascading prediction."""
        service = MLExpectedPointsService()
        service.pipeline = Mock()
        service.pipeline.named_steps = {"feature_engineer": Mock()}
        service.pipeline.predict = Mock(return_value=np.array([5.0, 6.0, 7.0]))

        # Mock calculate_expected_points
        def mock_calculate(*args, **kwargs):
            result = sample_players_data.copy()
            target_gw = kwargs.get("target_gameweek", 11)
            result["ml_xP"] = [5.0 + target_gw, 6.0 + target_gw, 7.0 + target_gw]
            result["xP"] = result["ml_xP"]
            result["xP_uncertainty"] = [1.0, 1.5, 2.0]
            return result

        service.calculate_expected_points = Mock(side_effect=mock_calculate)

        result = service.calculate_3gw_expected_points(
            players_data=sample_players_data,
            teams_data=sample_teams_data,
            xg_rates_data=pd.DataFrame(),
            fixtures_data=sample_fixtures_data,
            target_gameweek=10,
            live_data=sample_historical_data,
        )

        assert "xP_3gw" in result.columns
        assert "xP_uncertainty" in result.columns
        assert service.calculate_expected_points.call_count == 3

    def test_calculate_3gw_no_pipeline_raises_error(
        self,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_historical_data,
    ):
        """Test that missing pipeline raises error."""
        service = MLExpectedPointsService()
        service.pipeline = None

        with pytest.raises(ValueError, match="Pipeline not initialized"):
            service.calculate_3gw_expected_points(
                players_data=sample_players_data,
                teams_data=sample_teams_data,
                xg_rates_data=pd.DataFrame(),
                fixtures_data=sample_fixtures_data,
                target_gameweek=10,
                live_data=sample_historical_data,
            )


class TestUncertaintyExtraction:
    """Test uncertainty extraction for different model types."""

    def test_extract_uncertainty_random_forest(self, sample_players_data):
        """Test uncertainty extraction for Random Forest."""
        service = MLExpectedPointsService()

        # Create a Random Forest model
        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        # Fit on dummy data
        X_dummy = np.random.rand(10, 5)
        y_dummy = np.random.rand(10)
        rf_model.fit(X_dummy, y_dummy)

        # Create pipeline
        pipeline = Pipeline(
            [
                ("feature_engineer", Mock()),
                ("model", rf_model),
            ]
        )
        pipeline.named_steps["feature_engineer"].transform = Mock(
            return_value=pd.DataFrame(np.random.rand(3, 5))
        )

        service.pipeline = pipeline

        uncertainty = service._extract_uncertainty(sample_players_data)
        assert len(uncertainty) == len(sample_players_data)
        assert all(u >= 0 for u in uncertainty)

    def test_extract_uncertainty_no_pipeline(self, sample_players_data):
        """Test uncertainty extraction with no pipeline."""
        service = MLExpectedPointsService()
        service.pipeline = None

        uncertainty = service._extract_uncertainty(sample_players_data)
        assert len(uncertainty) == len(sample_players_data)
        assert all(u == 0 for u in uncertainty)

    def test_extract_uncertainty_hybrid_model(self, sample_players_data):
        """Test uncertainty extraction for hybrid model (returns zeros)."""
        service = MLExpectedPointsService()
        service.is_hybrid_model = True
        service.pipeline = Mock()

        uncertainty = service._extract_uncertainty(sample_players_data)
        assert len(uncertainty) == len(sample_players_data)
        assert all(u == 0 for u in uncertainty)

    def test_extract_uncertainty_non_ensemble_model(self, sample_players_data):
        """Test uncertainty extraction for non-ensemble model (Ridge)."""
        service = MLExpectedPointsService()

        ridge_model = Ridge()
        ridge_model.fit(np.random.rand(10, 5), np.random.rand(10))

        pipeline = Pipeline(
            [
                ("feature_engineer", Mock()),
                ("model", ridge_model),
            ]
        )
        pipeline.named_steps["feature_engineer"].transform = Mock(
            return_value=pd.DataFrame(np.random.rand(3, 5))
        )

        service.pipeline = pipeline

        uncertainty = service._extract_uncertainty(sample_players_data)
        assert len(uncertainty) == len(sample_players_data)
        assert all(u == 0 for u in uncertainty)  # Ridge has no uncertainty


class TestFeatureImportance:
    """Test feature importance extraction."""

    def test_get_feature_importance_random_forest(self):
        """Test feature importance for Random Forest."""
        service = MLExpectedPointsService()

        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        rf_model.fit(np.random.rand(10, 5), np.random.rand(10))

        feature_engineer = Mock()
        feature_engineer.get_feature_names_out = Mock(
            return_value=[f"feature_{i}" for i in range(5)]
        )

        pipeline = Pipeline(
            [
                ("feature_engineer", feature_engineer),
                ("model", rf_model),
            ]
        )

        service.pipeline = pipeline

        importance_df = service.get_feature_importance()
        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == 5

    def test_get_feature_importance_no_pipeline_raises_error(self):
        """Test that missing pipeline raises error."""
        service = MLExpectedPointsService()
        service.pipeline = None

        with pytest.raises(ValueError, match="Model not trained"):
            service.get_feature_importance()

    def test_get_feature_importance_hybrid_model_raises_error(self):
        """Test that hybrid model raises error."""
        service = MLExpectedPointsService()
        service.is_hybrid_model = True
        service.pipeline = Mock()

        with pytest.raises(ValueError, match="Feature importance not supported"):
            service.get_feature_importance()


class TestModelSaveLoad:
    """Test model save/load functionality."""

    def test_save_models(self, tmp_path):
        """Test saving models."""
        service = MLExpectedPointsService()
        service.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )
        service.pipeline_metadata = {"cv_mae_mean": 1.5}

        model_path = tmp_path / "saved_model.joblib"
        service.save_models(str(model_path))

        assert model_path.exists()

    def test_save_models_no_pipeline_raises_error(self):
        """Test that saving without pipeline raises error."""
        service = MLExpectedPointsService()
        service.pipeline = None

        with pytest.raises(ValueError, match="No pipeline to save"):
            service.save_models("dummy.joblib")

    def test_load_models(self, tmp_path):
        """Test loading models."""
        # Create and save a model
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
            ]
        )
        model_path = tmp_path / "model.joblib"
        joblib.dump(pipeline, model_path)

        service = MLExpectedPointsService()
        service.load_models(str(model_path))

        assert service.pipeline is not None
        assert service.model_path == str(model_path)


class TestSyntheticDataCreation:
    """Test synthetic gameweek data creation."""

    def test_create_synthetic_gameweek_data(self, sample_players_data):
        """Test creating synthetic gameweek data."""
        service = MLExpectedPointsService()

        predictions = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "ml_xP": [5.0, 6.0, 7.0],
            }
        )

        synthetic = service._create_synthetic_gameweek_data(
            players_data=sample_players_data,
            predictions=predictions,
            gameweek=11,
        )

        assert len(synthetic) == len(sample_players_data)
        assert (synthetic["gameweek"] == 11).all()
        assert "total_points" in synthetic.columns
        assert "minutes" in synthetic.columns
        assert synthetic["total_points"].iloc[0] == 5.0

    def test_create_synthetic_data_missing_predictions(self, sample_players_data):
        """Test synthetic data when predictions missing some players."""
        service = MLExpectedPointsService()

        # Predictions missing player 2
        predictions = pd.DataFrame(
            {
                "player_id": [1, 3],
                "ml_xP": [5.0, 7.0],
            }
        )

        synthetic = service._create_synthetic_gameweek_data(
            players_data=sample_players_data,
            predictions=predictions,
            gameweek=11,
        )

        # Player 2 should have 0 points
        player_2_points = synthetic[synthetic["player_id"] == 2]["total_points"].iloc[0]
        assert player_2_points == 0


class TestDebugLogging:
    """Test debug logging functionality."""

    def test_debug_logging_enabled(self):
        """Test that debug logging works when enabled."""
        service = MLExpectedPointsService(debug=True)
        # Should not raise error
        service._log_debug("Test message")

    def test_debug_logging_disabled(self):
        """Test that debug logging is silent when disabled."""
        service = MLExpectedPointsService(debug=False)
        # Should not raise error
        service._log_debug("Test message")
