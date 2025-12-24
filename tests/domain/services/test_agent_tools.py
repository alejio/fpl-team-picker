"""Tests for agent tools and AgentDeps."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from pydantic_ai import RunContext

from fpl_team_picker.domain.services.agent_tools import (
    AgentDeps,
    get_multi_gw_xp_predictions,
)


class TestAgentDeps:
    """Test AgentDeps dependency injection class."""

    def test_agent_deps_creation_minimal(self):
        """Test creating AgentDeps with minimal required data."""
        players = pd.DataFrame({"player_id": [1, 2], "web_name": ["A", "B"]})
        teams = pd.DataFrame({"team_id": [1, 2], "team_name": ["Team1", "Team2"]})
        fixtures = pd.DataFrame({"fixture_id": [1]})
        live_data = pd.DataFrame()

        deps = AgentDeps(
            players_data=players,
            teams_data=teams,
            fixtures_data=fixtures,
            live_data=live_data,
        )

        assert deps.players_data is players
        assert deps.teams_data is teams
        assert deps.fixtures_data is fixtures
        assert deps.live_data is live_data
        assert deps.ownership_trends is None
        assert deps.value_analysis is None

    def test_agent_deps_creation_full(self):
        """Test creating AgentDeps with all optional data."""
        players = pd.DataFrame({"player_id": [1]})
        teams = pd.DataFrame({"team_id": [1]})
        fixtures = pd.DataFrame({"fixture_id": [1]})
        live_data = pd.DataFrame()
        ownership = pd.DataFrame({"player_id": [1], "ownership": [0.5]})
        value = pd.DataFrame({"player_id": [1], "value": [1.2]})
        fixture_diff = pd.DataFrame({"team_id": [1], "difficulty": [2]})
        betting = pd.DataFrame({"player_id": [1], "odds": [2.0]})
        metrics = pd.DataFrame({"player_id": [1], "metric": [1.0]})
        availability = pd.DataFrame({"player_id": [1], "available": [True]})
        team_form = pd.DataFrame({"team_id": [1], "form": [1.5]})
        enhanced = pd.DataFrame({"player_id": [1], "enhanced": [1.0]})
        xg_rates = pd.DataFrame({"team_id": [1], "xg": [1.5]})

        deps = AgentDeps(
            players_data=players,
            teams_data=teams,
            fixtures_data=fixtures,
            live_data=live_data,
            ownership_trends=ownership,
            value_analysis=value,
            fixture_difficulty=fixture_diff,
            betting_features=betting,
            player_metrics=metrics,
            player_availability=availability,
            team_form=team_form,
            players_enhanced=enhanced,
            xg_rates=xg_rates,
        )

        assert deps.ownership_trends is ownership
        assert deps.value_analysis is value
        assert deps.fixture_difficulty is fixture_diff
        assert deps.betting_features is betting
        assert deps.player_metrics is metrics
        assert deps.player_availability is availability
        assert deps.team_form is team_form
        assert deps.players_enhanced is enhanced
        assert deps.xg_rates is xg_rates


class TestGetMultiGWXPPredictions:
    """Test get_multi_gw_xp_predictions tool function."""

    @pytest.fixture
    def sample_players_data(self):
        """Create sample players DataFrame."""
        return pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "web_name": ["Salah", "Son", "Haaland"],
                "position": ["MID", "MID", "FWD"],
                "team": [1, 2, 3],
            }
        )

    @pytest.fixture
    def sample_teams_data(self):
        """Create sample teams DataFrame."""
        return pd.DataFrame(
            {
                "team_id": [1, 2, 3],
                "team_name": ["Liverpool", "Spurs", "Man City"],
            }
        )

    @pytest.fixture
    def sample_fixtures_data(self):
        """Create sample fixtures DataFrame."""
        return pd.DataFrame(
            {
                "fixture_id": [1, 2, 3],
                "team_h": [1, 2, 3],
                "team_a": [2, 3, 1],
                "gameweek": [18, 18, 18],
            }
        )

    @pytest.fixture
    def sample_agent_deps(
        self, sample_players_data, sample_teams_data, sample_fixtures_data
    ):
        """Create sample AgentDeps."""
        return AgentDeps(
            players_data=sample_players_data,
            teams_data=sample_teams_data,
            fixtures_data=sample_fixtures_data,
            live_data=pd.DataFrame(),
        )

    @pytest.fixture
    def mock_run_context(self, sample_agent_deps):
        """Create mock RunContext."""
        ctx = Mock(spec=RunContext)
        ctx.deps = sample_agent_deps
        return ctx

    @pytest.fixture
    def mock_ml_service(self):
        """Create mock MLExpectedPointsService."""
        service = Mock()
        predictions_df = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "web_name": ["Salah", "Son", "Haaland"],
                "position": ["MID", "MID", "FWD"],
                "team_name": ["Liverpool", "Spurs", "Man City"],
                "price": [13.0, 10.0, 14.0],
                "selected_by_percent": [50.0, 30.0, 60.0],
                "ml_xP": [15.0, 12.0, 18.0],
                "xP_gw1": [5.0, 4.0, 6.0],
                "xP_gw2": [5.0, 4.0, 6.0],
                "xP_gw3": [5.0, 4.0, 6.0],
                "xP_uncertainty": [1.0, 1.2, 0.8],
            }
        )
        service.calculate_3gw_expected_points.return_value = predictions_df
        service.calculate_5gw_expected_points.return_value = predictions_df
        return service

    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    @patch("fpl_team_picker.domain.services.agent_tools.config")
    def test_get_multi_gw_xp_predictions_3gw(
        self, mock_config, mock_ml_service_class, mock_run_context, mock_ml_service
    ):
        """Test get_multi_gw_xp_predictions for 3-gameweek horizon."""
        mock_ml_service_class.return_value = mock_ml_service
        mock_config.xp_model.ml_model_path = "models/test.joblib"
        mock_config.xp_model.ml_ensemble_rule_weight = 0.5
        mock_config.xp_model.debug = False

        result = get_multi_gw_xp_predictions(
            ctx=mock_run_context, start_gameweek=18, num_gameweeks=3
        )

        # Verify ML service was called correctly
        mock_ml_service.calculate_3gw_expected_points.assert_called_once()
        call_kwargs = mock_ml_service.calculate_3gw_expected_points.call_args[1]
        assert call_kwargs["target_gameweek"] == 18

        # Verify result structure
        assert "players" in result
        assert "summary" in result
        assert "metadata" in result

        # Verify players list
        assert len(result["players"]) <= 100  # Should be limited to top 100
        assert len(result["players"]) == 3  # We have 3 players
        assert result["players"][0]["web_name"] == "Haaland"  # Sorted by xP desc

        # Verify summary statistics
        assert "total_players" in result["summary"]
        assert "avg_xp_total" in result["summary"]
        assert "max_xp_total" in result["summary"]
        assert "min_xp_total" in result["summary"]
        assert result["summary"]["total_players"] == 3
        assert result["summary"]["avg_xp_total"] == 15.0

        # Verify metadata
        assert result["metadata"]["start_gameweek"] == 18
        assert result["metadata"]["num_gameweeks"] == 3

    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    @patch("fpl_team_picker.domain.services.agent_tools.config")
    def test_get_multi_gw_xp_predictions_5gw(
        self, mock_config, mock_ml_service_class, mock_run_context, mock_ml_service
    ):
        """Test get_multi_gw_xp_predictions for 5-gameweek horizon."""
        mock_ml_service_class.return_value = mock_ml_service
        mock_config.xp_model.ml_model_path = "models/test.joblib"
        mock_config.xp_model.ml_ensemble_rule_weight = 0.5
        mock_config.xp_model.debug = False

        result = get_multi_gw_xp_predictions(
            ctx=mock_run_context, start_gameweek=20, num_gameweeks=5
        )

        # Verify ML service was called with 5GW method
        mock_ml_service.calculate_5gw_expected_points.assert_called_once()
        call_kwargs = mock_ml_service.calculate_5gw_expected_points.call_args[1]
        assert call_kwargs["target_gameweek"] == 20

        # Verify result structure
        assert "players" in result
        assert "summary" in result
        assert result["metadata"]["num_gameweeks"] == 5

    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    @patch("fpl_team_picker.domain.services.agent_tools.config")
    def test_get_multi_gw_xp_predictions_invalid_horizon(
        self, mock_config, mock_ml_service_class, mock_run_context, mock_ml_service
    ):
        """Test that invalid horizon raises ValueError."""
        mock_ml_service_class.return_value = mock_ml_service
        mock_config.xp_model.ml_model_path = "models/test.joblib"
        mock_config.xp_model.ml_ensemble_rule_weight = 0.5
        mock_config.xp_model.debug = False

        with pytest.raises(ValueError, match="Unsupported horizon"):
            get_multi_gw_xp_predictions(
                ctx=mock_run_context, start_gameweek=18, num_gameweeks=2
            )

        with pytest.raises(ValueError, match="Unsupported horizon"):
            get_multi_gw_xp_predictions(
                ctx=mock_run_context, start_gameweek=18, num_gameweeks=4
            )

    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    @patch("fpl_team_picker.domain.services.agent_tools.config")
    def test_get_multi_gw_xp_predictions_with_optional_data(
        self,
        mock_config,
        mock_ml_service_class,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        mock_ml_service,
    ):
        """Test that optional data is passed through correctly."""
        mock_ml_service_class.return_value = mock_ml_service
        mock_config.xp_model.ml_model_path = "models/test.joblib"
        mock_config.xp_model.ml_ensemble_rule_weight = 0.5
        mock_config.xp_model.debug = False

        # Create deps with optional data
        ownership = pd.DataFrame({"player_id": [1], "ownership": [0.5]})
        value = pd.DataFrame({"player_id": [1], "value": [1.2]})
        xg_rates = pd.DataFrame({"team_id": [1], "xg": [1.5]})

        deps = AgentDeps(
            players_data=sample_players_data,
            teams_data=sample_teams_data,
            fixtures_data=sample_fixtures_data,
            live_data=pd.DataFrame(),
            ownership_trends=ownership,
            value_analysis=value,
            xg_rates=xg_rates,
        )

        ctx = Mock(spec=RunContext)
        ctx.deps = deps


        # Verify optional data was passed to ML service
        call_kwargs = mock_ml_service.calculate_3gw_expected_points.call_args[1]
        assert call_kwargs["ownership_trends_df"] is ownership
        assert call_kwargs["value_analysis_df"] is value
        assert call_kwargs["xg_rates_data"] is xg_rates

    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    @patch("fpl_team_picker.domain.services.agent_tools.config")
    def test_get_multi_gw_xp_predictions_sorts_by_xp(
        self, mock_config, mock_ml_service_class, mock_run_context, mock_ml_service
    ):
        """Test that results are sorted by total xP descending."""
        # Create predictions with different xP values
        predictions_df = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "web_name": ["Low", "High", "Mid"],
                "position": ["MID", "MID", "MID"],
                "team_name": ["Team1", "Team2", "Team3"],
                "price": [10.0, 10.0, 10.0],
                "selected_by_percent": [10.0, 10.0, 10.0],
                "ml_xP": [5.0, 15.0, 10.0],  # Different xP values
                "xP_gw1": [1.67, 5.0, 3.33],
                "xP_gw2": [1.67, 5.0, 3.33],
                "xP_gw3": [1.67, 5.0, 3.33],
            }
        )
        mock_ml_service.calculate_3gw_expected_points.return_value = predictions_df
        mock_ml_service_class.return_value = mock_ml_service
        mock_config.xp_model.ml_model_path = "models/test.joblib"
        mock_config.xp_model.ml_ensemble_rule_weight = 0.5
        mock_config.xp_model.debug = False

        result = get_multi_gw_xp_predictions(
            ctx=mock_run_context, start_gameweek=18, num_gameweeks=3
        )

        # Verify sorting - highest xP first
        assert result["players"][0]["web_name"] == "High"
        assert result["players"][0]["ml_xP"] == 15.0
        assert result["players"][1]["web_name"] == "Mid"
        assert result["players"][1]["ml_xP"] == 10.0
        assert result["players"][2]["web_name"] == "Low"
        assert result["players"][2]["ml_xP"] == 5.0

    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    @patch("fpl_team_picker.domain.services.agent_tools.config")
    def test_get_multi_gw_xp_predictions_handles_missing_columns(
        self, mock_config, mock_ml_service_class, mock_run_context, mock_ml_service
    ):
        """Test that missing optional columns are handled gracefully."""
        # Create predictions without uncertainty column
        predictions_df = pd.DataFrame(
            {
                "player_id": [1, 2],
                "web_name": ["A", "B"],
                "position": ["MID", "FWD"],
                "team_name": ["Team1", "Team2"],
                "price": [10.0, 11.0],
                "selected_by_percent": [10.0, 20.0],
                "ml_xP": [10.0, 12.0],
                "xP_gw1": [3.33, 4.0],
                "xP_gw2": [3.33, 4.0],
                "xP_gw3": [3.33, 4.0],
                # No xP_uncertainty column
            }
        )
        mock_ml_service.calculate_3gw_expected_points.return_value = predictions_df
        mock_ml_service_class.return_value = mock_ml_service
        mock_config.xp_model.ml_model_path = "models/test.joblib"
        mock_config.xp_model.ml_ensemble_rule_weight = 0.5
        mock_config.xp_model.debug = False

        result = get_multi_gw_xp_predictions(
            ctx=mock_run_context, start_gameweek=18, num_gameweeks=3
        )

        # Should still work without uncertainty column
        assert len(result["players"]) == 2
        assert "xP_uncertainty" not in result["players"][0]

    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    @patch("fpl_team_picker.domain.services.agent_tools.config")
    def test_get_multi_gw_xp_predictions_error_handling(
        self, mock_config, mock_ml_service_class, mock_run_context
    ):
        """Test that errors are properly logged and re-raised."""
        mock_ml_service = Mock()
        mock_ml_service.calculate_3gw_expected_points.side_effect = Exception(
            "ML service error"
        )
        mock_ml_service_class.return_value = mock_ml_service
        mock_config.xp_model.ml_model_path = "models/test.joblib"
        mock_config.xp_model.ml_ensemble_rule_weight = 0.5
        mock_config.xp_model.debug = False

        with pytest.raises(Exception, match="ML service error"):
            get_multi_gw_xp_predictions(
                ctx=mock_run_context, start_gameweek=18, num_gameweeks=3
            )
