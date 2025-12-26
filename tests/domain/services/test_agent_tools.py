"""Tests for Agent Tools used by TransferPlanningAgentService.

Tests for the 5 agent tools:
1. get_multi_gw_xp_predictions
2. analyze_fixture_context
3. run_sa_optimizer
4. analyze_squad_weaknesses
5. get_template_players
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from pydantic_ai import RunContext

from fpl_team_picker.domain.services.agent_tools import (
    AGENT_TOOLS,
    AgentDeps,
    analyze_fixture_context,
    analyze_squad_weaknesses,
    get_multi_gw_xp_predictions,
    get_template_players,
    run_sa_optimizer,
)


class TestAgentDeps:
    """Test AgentDeps data class."""

    def test_agent_deps_creation(self):
        """Test creating AgentDeps with required data."""
        players_df = pd.DataFrame(
            {
                "player_id": [1, 2],
                "web_name": ["Haaland", "Salah"],
                "position": ["FWD", "MID"],
            }
        )
        teams_df = pd.DataFrame({"team_id": [1, 2], "name": ["Arsenal", "Liverpool"]})
        fixtures_df = pd.DataFrame(
            {"id": [1], "event": [18], "team_h": [1], "team_a": [2]}
        )
        live_df = pd.DataFrame(
            {"player_id": [1], "gameweek": [17], "total_points": [8]}
        )

        deps = AgentDeps(
            players_data=players_df,
            teams_data=teams_df,
            fixtures_data=fixtures_df,
            live_data=live_df,
        )

        assert not deps.players_data.empty
        assert not deps.teams_data.empty
        assert not deps.fixtures_data.empty
        assert not deps.live_data.empty

    def test_agent_deps_with_optional_data(self):
        """Test AgentDeps with optional enriched data."""
        players_df = pd.DataFrame({"player_id": [1]})
        teams_df = pd.DataFrame({"team_id": [1]})
        fixtures_df = pd.DataFrame({"id": [1]})
        live_df = pd.DataFrame()

        ownership_df = pd.DataFrame({"player_id": [1], "ownership_delta": [0.5]})
        fixture_difficulty_df = pd.DataFrame(
            {
                "team_id": [1],
                "opponent_id": [2],
                "gameweek": [18],
                "overall_difficulty": [2.5],
            }
        )

        deps = AgentDeps(
            players_data=players_df,
            teams_data=teams_df,
            fixtures_data=fixtures_df,
            live_data=live_df,
            ownership_trends=ownership_df,
            fixture_difficulty=fixture_difficulty_df,
        )

        assert deps.ownership_trends is not None
        assert deps.fixture_difficulty is not None
        assert not deps.ownership_trends.empty


class TestGetMultiGWXPPredictions:
    """Test get_multi_gw_xp_predictions tool."""

    @pytest.fixture
    def mock_context(self):
        """Create mock RunContext with sample data."""
        players_df = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "web_name": ["Haaland", "Salah", "Saka"],
                "position": ["FWD", "MID", "MID"],
                "price": [15.0, 13.0, 9.0],
            }
        )

        deps = AgentDeps(
            players_data=players_df,
            teams_data=pd.DataFrame({"team_id": [1]}),
            fixtures_data=pd.DataFrame({"id": [1], "event": [18]}),
            live_data=pd.DataFrame(),
            xg_rates=pd.DataFrame(),
        )

        ctx = Mock(spec=RunContext)
        ctx.deps = deps
        return ctx

    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    def test_get_multi_gw_xp_predictions_1gw(self, mock_ml_service_class, mock_context):
        """Test getting 1GW xP predictions."""
        # Mock ML service
        mock_service = Mock()
        mock_service.calculate_expected_points.return_value = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "web_name": ["Haaland", "Salah", "Saka"],
                "position": ["FWD", "MID", "MID"],
                "team_name": ["Man City", "Liverpool", "Arsenal"],
                "price": [15.0, 13.0, 9.0],
                "selected_by_percent": [65.0, 55.0, 35.0],
                "ml_xP": [8.2, 7.5, 6.1],
                "xP_gw1": [8.2, 7.5, 6.1],
                "xP_uncertainty": [1.5, 1.2, 1.0],
            }
        )
        mock_ml_service_class.return_value = mock_service

        result = get_multi_gw_xp_predictions(
            ctx=mock_context, start_gameweek=18, num_gameweeks=1
        )

        assert result["metadata"]["num_gameweeks"] == 1
        assert len(result["players"]) == 3
        assert result["players"][0]["player_id"] == 1
        assert result["players"][0]["web_name"] == "Haaland"
        assert result["players"][0]["xP_gw1"] == 8.2
        assert result["players"][0]["xP_uncertainty"] == 1.5

    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    def test_get_multi_gw_xp_predictions_3gw(self, mock_ml_service_class, mock_context):
        """Test getting 3GW xP predictions with per-GW breakdowns."""
        # Mock ML service to return 3GW data with all per-GW columns
        mock_service = Mock()
        mock_service.calculate_3gw_expected_points.return_value = pd.DataFrame(
            {
                "player_id": [1],
                "web_name": ["Haaland"],
                "position": ["FWD"],
                "team_name": ["Man City"],
                "price": [15.0],
                "selected_by_percent": [65.0],
                "ml_xP": [25.5],  # Total over 3 GW
                "xP_gw1": [8.2],
                "xP_gw2": [7.8],
                "xP_gw3": [9.5],
                "xP_uncertainty": [1.5],
            }
        )
        mock_ml_service_class.return_value = mock_service

        result = get_multi_gw_xp_predictions(
            ctx=mock_context, start_gameweek=18, num_gameweeks=3
        )

        assert result["metadata"]["num_gameweeks"] == 3
        assert len(result["players"]) == 1
        player = result["players"][0]
        assert player["xP_gw1"] == 8.2
        assert player["xP_gw2"] == 7.8
        assert player["xP_gw3"] == 9.5
        assert player["ml_xP"] == 25.5

    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    def test_get_multi_gw_xp_summary_stats(self, mock_ml_service_class, mock_context):
        """Test summary statistics calculation."""
        mock_service = Mock()
        mock_service.calculate_expected_points.return_value = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "web_name": ["Haaland", "Salah", "Saka"],
                "position": ["FWD", "MID", "MID"],
                "team_name": ["Man City", "Liverpool", "Arsenal"],
                "price": [15.0, 13.0, 9.0],
                "selected_by_percent": [65.0, 55.0, 35.0],
                "ml_xP": [8.2, 7.5, 6.1],
                "xP_gw1": [8.2, 7.5, 6.1],
                "xP_uncertainty": [1.5, 1.2, 1.0],
            }
        )
        mock_ml_service_class.return_value = mock_service

        result = get_multi_gw_xp_predictions(
            ctx=mock_context, start_gameweek=18, num_gameweeks=1
        )

        assert "summary" in result
        assert result["summary"]["total_players"] == 3
        assert "avg_xp_total" in result["summary"]
        assert result["summary"]["avg_xp_gw1"] == pytest.approx((8.2 + 7.5 + 6.1) / 3)

    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    def test_get_multi_gw_xp_metadata(self, mock_ml_service_class, mock_context):
        """Test metadata is included in response."""
        mock_service = Mock()
        mock_service.calculate_expected_points.return_value = pd.DataFrame(
            {
                "player_id": [1],
                "web_name": ["Haaland"],
                "position": ["FWD"],
                "team_name": ["Man City"],
                "price": [15.0],
                "selected_by_percent": [65.0],
                "ml_xP": [8.2],
                "xP_gw1": [8.2],
            }
        )
        mock_ml_service_class.return_value = mock_service

        result = get_multi_gw_xp_predictions(
            ctx=mock_context, start_gameweek=18, num_gameweeks=1
        )

        assert "metadata" in result
        assert result["metadata"]["start_gameweek"] == 18
        assert result["metadata"]["num_gameweeks"] == 1


class TestAnalyzeFixtureContext:
    """Test analyze_fixture_context tool."""

    @pytest.fixture
    def mock_context_with_fixtures(self):
        """Create mock context with fixture data."""
        teams_df = pd.DataFrame(
            {
                "team_id": [1, 2, 3, 4],
                "name": ["Arsenal", "Liverpool", "Chelsea", "Man City"],
            }
        )

        # Create fixtures with DGW for Arsenal (team_id=1) in GW18
        fixtures_df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "event": [18, 18, 19, 19],  # GW18 has 2 fixtures, GW19 has 2
                "team_h": [1, 1, 2, 3],  # Arsenal plays twice in GW18 (home)
                "team_a": [2, 3, 4, 1],
            }
        )

        # Fixture difficulty is team-level, not player-level
        fixture_difficulty_df = pd.DataFrame(
            {
                "team_id": [1, 2, 3],
                "opponent_id": [2, 3, 1],
                "gameweek": [18, 18, 18],
                "overall_difficulty": [2.0, 4.5, 3.0],
            }
        )

        deps = AgentDeps(
            players_data=pd.DataFrame(),
            teams_data=teams_df,
            fixtures_data=fixtures_df,
            live_data=pd.DataFrame(),
            fixture_difficulty=fixture_difficulty_df,
        )

        ctx = Mock(spec=RunContext)
        ctx.deps = deps
        return ctx

    def test_analyze_fixture_context_detects_dgw(self, mock_context_with_fixtures):
        """Test DGW detection."""
        result = analyze_fixture_context(
            ctx=mock_context_with_fixtures, start_gameweek=18, num_gameweeks=2
        )

        assert result["dgw_bgw_calendar"]["has_dgw"]
        assert 18 in result["dgw_bgw_calendar"]["dgw_teams"]
        assert "Arsenal" in result["dgw_bgw_calendar"]["dgw_teams"][18]
        assert result["dgw_bgw_calendar"]["next_dgw_gameweek"] == 18

    def test_analyze_fixture_context_fixture_runs(self, mock_context_with_fixtures):
        """Test fixture difficulty runs analysis."""
        result = analyze_fixture_context(
            ctx=mock_context_with_fixtures, start_gameweek=18, num_gameweeks=2
        )

        assert "fixture_runs" in result
        assert "easy_runs" in result["fixture_runs"]
        assert "hard_runs" in result["fixture_runs"]
        assert "fixture_swings" in result["fixture_runs"]

    def test_analyze_fixture_context_no_fixture_difficulty(self):
        """Test when fixture_difficulty data is missing."""
        teams_df = pd.DataFrame({"team_id": [1], "name": ["Arsenal"]})
        fixtures_df = pd.DataFrame(
            {"id": [1], "event": [18], "team_h": [1], "team_a": [2]}
        )

        deps = AgentDeps(
            players_data=pd.DataFrame(),
            teams_data=teams_df,
            fixtures_data=fixtures_df,
            live_data=pd.DataFrame(),
            fixture_difficulty=None,  # Missing
        )

        ctx = Mock(spec=RunContext)
        ctx.deps = deps

        result = analyze_fixture_context(ctx=ctx, start_gameweek=18, num_gameweeks=1)

        # Should return empty fixture runs but not fail
        assert result["fixture_runs"]["easy_runs"] == []
        assert result["fixture_runs"]["hard_runs"] == []


class TestRunSAOptimizer:
    """Test run_sa_optimizer tool."""

    @pytest.fixture
    def mock_context_for_sa(self):
        """Create mock context for SA optimizer."""
        players_df = pd.DataFrame(
            {
                "player_id": list(range(1, 16)),
                "web_name": [f"Player{i}" for i in range(1, 16)],
                "position": ["GKP", "GKP"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
                "price": [4.5] * 15,
            }
        )

        deps = AgentDeps(
            players_data=players_df,
            teams_data=pd.DataFrame({"team_id": [1]}),
            fixtures_data=pd.DataFrame({"id": [1], "event": [18]}),
            live_data=pd.DataFrame(),
            xg_rates=pd.DataFrame(),
        )

        ctx = Mock(spec=RunContext)
        ctx.deps = deps
        return ctx

    def test_run_sa_optimizer_input_validation(self, mock_context_for_sa):
        """Test input validation for SA optimizer."""
        # Test invalid squad size
        with pytest.raises(ValueError, match="exactly 15 players"):
            run_sa_optimizer(
                ctx=mock_context_for_sa,
                current_squad_ids=[1, 2, 3],  # Only 3 players
                num_transfers=1,
                target_gameweek=18,
            )

        # Test invalid num_transfers
        with pytest.raises(ValueError, match="num_transfers must be 1-15"):
            run_sa_optimizer(
                ctx=mock_context_for_sa,
                current_squad_ids=list(range(1, 16)),
                num_transfers=20,  # Too many
                target_gameweek=18,
            )

        # Test invalid gameweek
        with pytest.raises(ValueError, match="target_gameweek must be 1-38"):
            run_sa_optimizer(
                ctx=mock_context_for_sa,
                current_squad_ids=list(range(1, 16)),
                num_transfers=1,
                target_gameweek=50,  # Invalid
            )

        # Test invalid horizon
        with pytest.raises(ValueError, match="horizon must be 1, 3, or 5"):
            run_sa_optimizer(
                ctx=mock_context_for_sa,
                current_squad_ids=list(range(1, 16)),
                num_transfers=1,
                target_gameweek=18,
                horizon=7,  # Invalid
            )

    @patch("fpl_team_picker.domain.services.agent_tools.OptimizationService")
    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    def test_run_sa_optimizer_success(
        self, mock_ml_service_class, mock_opt_service_class, mock_context_for_sa
    ):
        """Test successful SA optimization run."""
        # Mock ML service with 3-GW predictions (default horizon)
        mock_ml_service = Mock()
        mock_ml_service.calculate_3gw_expected_points.return_value = pd.DataFrame(
            {
                "player_id": list(range(1, 16)),
                "web_name": [f"Player{i}" for i in range(1, 16)],
                "ml_xP": [5.0] * 15,
                "xP_3gw": [15.0] * 15,  # 3-GW total
            }
        )
        mock_ml_service_class.return_value = mock_ml_service

        # Mock optimization service
        mock_opt_service = Mock()
        optimal_squad_df = pd.DataFrame({"player_id": list(range(1, 16))})
        best_scenario = {
            "net_xp": 65.0,
            "xp_gain": 4.0,
            "penalty": -4,
            "formation": "4-4-2",
        }
        metadata = {
            "transfers_out": [{"player_id": 5, "web_name": "Player5", "price": 5.0}],
            "transfers_in": [
                {
                    "player_id": 42,
                    "web_name": "NewPlayer",
                    "price": 5.5,
                    "position": "MID",
                }
            ],
        }
        mock_opt_service.optimize_transfers.return_value = (
            optimal_squad_df,
            best_scenario,
            metadata,
        )
        mock_opt_service_class.return_value = mock_opt_service

        result = run_sa_optimizer(
            ctx=mock_context_for_sa,
            current_squad_ids=list(range(1, 16)),
            num_transfers=1,
            target_gameweek=18,
        )

        assert result["expected_xp"] == 65.0
        assert result["xp_gain"] == 4.0
        assert result["hit_cost"] == -4
        assert len(result["transfers"]) == 1
        assert result["transfers"][0]["in_name"] == "NewPlayer"
        assert "runtime_seconds" in result


class TestAnalyzeSquadWeaknesses:
    """Test analyze_squad_weaknesses tool."""

    @pytest.fixture
    def mock_context_with_squad(self):
        """Create mock context with squad data."""
        players_df = pd.DataFrame(
            {
                "player_id": list(range(1, 16)),
                "web_name": [f"Player{i}" for i in range(1, 16)],
                "position": ["GKP", "GKP"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
                "price": [4.5] * 15,
                "team": [1] * 15,  # All players from team 1
                "team_id": [1] * 15,  # Match real data structure
            }
        )

        player_metrics_df = pd.DataFrame(
            {
                "player_id": [3, 4, 5],  # Some DEF players
                "rotation_risk": [0.7, 0.3, 0.6],
            }
        )

        deps = AgentDeps(
            players_data=players_df,
            teams_data=pd.DataFrame({"team_id": [1], "name": ["Arsenal"]}),
            fixtures_data=pd.DataFrame({"id": [1], "event": [18]}),
            live_data=pd.DataFrame(),
            player_metrics=player_metrics_df,
            xg_rates=pd.DataFrame(),
        )

        ctx = Mock(spec=RunContext)
        ctx.deps = deps
        return ctx

    def test_analyze_squad_weaknesses_input_validation(self, mock_context_with_squad):
        """Test input validation."""
        with pytest.raises(ValueError, match="exactly 15 players"):
            analyze_squad_weaknesses(
                ctx=mock_context_with_squad,
                current_squad_ids=[1, 2, 3],  # Only 3
                target_gameweek=18,
            )

    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    def test_analyze_squad_weaknesses_identifies_low_xp(
        self, mock_ml_service_class, mock_context_with_squad
    ):
        """Test identification of low xP players."""
        mock_service = Mock()
        # ML service returns predictions with team_id (matches reality)
        mock_service.calculate_expected_points.return_value = pd.DataFrame(
            {
                "player_id": list(range(1, 16)),
                "web_name": [f"Player{i}" for i in range(1, 16)],
                "position": ["GKP", "GKP"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
                "team_id": [1] * 15,  # ML service preserves team_id
                "ml_xP": [
                    2.0,
                    2.5,
                    2.1,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                ],  # First 3 are low
            }
        )
        mock_ml_service_class.return_value = mock_service

        result = analyze_squad_weaknesses(
            ctx=mock_context_with_squad,
            current_squad_ids=list(range(1, 16)),
            target_gameweek=18,
        )

        assert len(result["low_xp_players"]) == 3
        assert all(p["xp_gw1"] < 3.0 for p in result["low_xp_players"])
        assert result["summary"]["total_weaknesses"] >= 3


class TestGetTemplatePlayers:
    """Test get_template_players tool."""

    @pytest.fixture
    def mock_context_for_templates(self):
        """Create mock context for template analysis."""
        players_df = pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4, 5],
                "web_name": ["Haaland", "Salah", "Saka", "Palmer", "Budget"],
                "position": ["FWD", "MID", "MID", "MID", "DEF"],
                "price": [15.0, 13.0, 9.0, 11.0, 4.5],
                "selected_by_percent": [65.0, 55.0, 35.0, 25.0, 5.0],
            }
        )

        deps = AgentDeps(
            players_data=players_df,
            teams_data=pd.DataFrame({"team_id": [1]}),
            fixtures_data=pd.DataFrame({"id": [1], "event": [18]}),
            live_data=pd.DataFrame(),
            xg_rates=pd.DataFrame(),
        )

        ctx = Mock(spec=RunContext)
        ctx.deps = deps
        return ctx

    def test_get_template_players_ownership_threshold_validation(
        self, mock_context_for_templates
    ):
        """Test ownership threshold validation."""
        with pytest.raises(ValueError, match="ownership_threshold must be 10.0-50.0"):
            get_template_players(
                ctx=mock_context_for_templates,
                target_gameweek=18,
                ownership_threshold=60.0,  # Too high
            )

    @patch("fpl_team_picker.domain.services.agent_tools.MLExpectedPointsService")
    def test_get_template_players_identifies_high_ownership(
        self, mock_ml_service_class, mock_context_for_templates
    ):
        """Test identification of template players."""
        mock_service = Mock()
        mock_service.calculate_expected_points.return_value = pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4, 5],
                "web_name": ["Haaland", "Salah", "Saka", "Palmer", "Budget"],
                "position": ["FWD", "MID", "MID", "MID", "DEF"],
                "price": [15.0, 13.0, 9.0, 11.0, 4.5],
                "selected_by_percent": [65.0, 55.0, 35.0, 25.0, 5.0],
                "ml_xP": [8.0, 7.5, 6.5, 6.0, 3.0],
            }
        )
        mock_ml_service_class.return_value = mock_service

        result = get_template_players(
            ctx=mock_context_for_templates,
            target_gameweek=18,
            ownership_threshold=30.0,
        )

        # Should find Haaland (65%), Salah (55%), Saka (35%)
        assert len(result["template_players"]) == 3
        assert result["template_players"][0]["web_name"] == "Haaland"
        assert result["template_players"][0]["ownership"] == 65.0


class TestAgentToolsRegistry:
    """Test AGENT_TOOLS registry."""

    def test_agent_tools_registry_has_all_tools(self):
        """Test that all 5 tools are registered."""
        assert len(AGENT_TOOLS) == 5

    def test_agent_tools_registry_contains_correct_functions(self):
        """Test that registry contains the right functions."""
        assert get_multi_gw_xp_predictions in AGENT_TOOLS
        assert analyze_fixture_context in AGENT_TOOLS
        assert run_sa_optimizer in AGENT_TOOLS
        assert analyze_squad_weaknesses in AGENT_TOOLS
        assert get_template_players in AGENT_TOOLS
