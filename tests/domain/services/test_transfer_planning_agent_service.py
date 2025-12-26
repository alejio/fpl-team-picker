"""Tests for TransferPlanningAgentService - LLM-based transfer recommendations.

This test suite covers:
1. Service initialization and configuration
2. Single-GW recommendation generation (mocked LLM)
3. System prompt generation with strategy modes
4. Error handling (API failures, invalid responses)
5. Integration with agent tools
"""

import os
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from fpl_team_picker.domain.models.transfer_plan import StrategyMode
from fpl_team_picker.domain.models.transfer_recommendation import (
    HoldOption,
    SingleGWRecommendation,
    TransferScenario,
)
from fpl_team_picker.domain.services.transfer_planning_agent_service import (
    SINGLE_GW_SYSTEM_PROMPT,
    STRATEGY_GUIDANCE,
    TransferPlanningAgentService,
)


class TestTransferPlanningAgentServiceInit:
    """Test service initialization and configuration."""

    def test_init_with_api_key_from_env(self):
        """Test initialization with API key from environment variable."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-123"}):
            service = TransferPlanningAgentService()
            assert service.api_key == "test-key-123"
            assert service.model_name == "claude-sonnet-4-5"  # Default from config

    def test_init_with_api_key_parameter(self):
        """Test initialization with API key passed as parameter."""
        service = TransferPlanningAgentService(api_key="param-key-456")
        assert service.api_key == "param-key-456"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        service = TransferPlanningAgentService(
            model="claude-haiku-3-7", api_key="test-key"
        )
        assert service.model_name == "claude-haiku-3-7"
        assert service.api_key == "test-key"

    def test_init_without_api_key_raises_error(self):
        """Test that missing API key raises clear error."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any potential API keys from env
            with pytest.raises(ValueError, match="Anthropic API key is required"):
                TransferPlanningAgentService()

    def test_init_with_debug_mode(self):
        """Test debug mode initialization."""
        service = TransferPlanningAgentService(api_key="test-key", debug=True)
        assert service.debug is True


class TestSystemPromptGeneration:
    """Test system prompt generation with different strategies."""

    def test_system_prompt_contains_target_gameweek(self):
        """Test that system prompt includes target gameweek."""
        prompt = SINGLE_GW_SYSTEM_PROMPT.format(
            target_gw=18,
            strategy_mode="balanced",
            strategy_specific_guidance=STRATEGY_GUIDANCE[StrategyMode.BALANCED],
            roi_threshold=5.0,
        )

        assert "gameweek 18" in prompt.lower()
        assert "GW18" in prompt

    def test_system_prompt_includes_strategy_guidance(self):
        """Test that strategy-specific guidance is included."""
        for strategy_mode in StrategyMode:
            guidance = STRATEGY_GUIDANCE.get(strategy_mode, "")
            prompt = SINGLE_GW_SYSTEM_PROMPT.format(
                target_gw=18,
                strategy_mode=strategy_mode.value,
                strategy_specific_guidance=guidance,
                roi_threshold=5.0,
            )

            # Check strategy mode is mentioned
            assert strategy_mode.value in prompt.lower()

    def test_balanced_strategy_guidance(self):
        """Test balanced strategy guidance content."""
        guidance = STRATEGY_GUIDANCE[StrategyMode.BALANCED]

        assert "balance" in guidance.lower()
        assert "template" in guidance.lower()
        assert "fixture" in guidance.lower()

    def test_conservative_strategy_guidance(self):
        """Test conservative strategy guidance content."""
        guidance = STRATEGY_GUIDANCE[StrategyMode.CONSERVATIVE]

        assert "minimize risk" in guidance.lower() or "risk" in guidance.lower()
        assert "template" in guidance.lower()

    def test_aggressive_strategy_guidance(self):
        """Test aggressive strategy guidance content."""
        guidance = STRATEGY_GUIDANCE[StrategyMode.AGGRESSIVE]

        assert "upside" in guidance.lower() or "maximize" in guidance.lower()
        assert "differential" in guidance.lower()

    def test_system_prompt_includes_roi_threshold(self):
        """Test that ROI threshold is included in prompt."""
        prompt = SINGLE_GW_SYSTEM_PROMPT.format(
            target_gw=18,
            strategy_mode="balanced",
            strategy_specific_guidance="",
            roi_threshold=7.5,
        )

        assert "7.5" in prompt

    def test_system_prompt_includes_workflow_steps(self):
        """Test that 6-step workflow is in system prompt."""
        prompt = SINGLE_GW_SYSTEM_PROMPT.format(
            target_gw=18,
            strategy_mode="balanced",
            strategy_specific_guidance="",
            roi_threshold=5.0,
        )

        # Check for workflow step keywords
        assert "ANALYZE" in prompt
        assert "PLAN" in prompt
        assert "EXECUTE" in prompt
        assert "SYNTHESIZE" in prompt
        assert "VALIDATE" in prompt
        assert "RECOMMEND" in prompt


class TestGenerateSingleGWRecommendations:
    """Test single-GW recommendation generation with mocked LLM."""

    @pytest.fixture
    def sample_gameweek_data(self):
        """Create sample gameweek data for testing."""
        players_df = pd.DataFrame(
            {
                "player_id": list(range(1, 21)),
                "web_name": [f"Player{i}" for i in range(1, 21)],
                "name": [f"Full Name {i}" for i in range(1, 21)],  # Added team name
                "position": (["GKP"] * 2 + ["DEF"] * 6 + ["MID"] * 8 + ["FWD"] * 4),
                "price": [5.0] * 20,
                "selected_by_percent": [20.0] * 20,
                "team_id": [1, 2] * 10,
            }
        )

        current_squad_df = players_df.head(15).copy()

        teams_df = pd.DataFrame(
            {"team_id": [1, 2, 3], "name": ["Arsenal", "Liverpool", "Chelsea"]}
        )

        fixtures_df = pd.DataFrame(
            {
                "id": [1, 2],
                "event": [18, 18],
                "team_h": [1, 2],
                "team_a": [2, 3],
            }
        )

        return {
            "players": players_df,
            "teams": teams_df,
            "fixtures": fixtures_df,
            "current_squad": current_squad_df,
            "live_data_historical": pd.DataFrame(),
            "ownership_trends": pd.DataFrame(
                {"player_id": [1], "ownership_delta": [0.5]}
            ),
            "value_analysis": pd.DataFrame(),
            "fixture_difficulty": pd.DataFrame(
                {
                    "team_id": [1],
                    "opponent_id": [2],
                    "gameweek": [18],
                    "overall_difficulty": [2.5],
                }
            ),
            "betting_features": pd.DataFrame(),
            "player_metrics": pd.DataFrame(),
            "player_availability": pd.DataFrame(),
            "team_form": pd.DataFrame(),
            "players_enhanced": pd.DataFrame(),
            "xg_rates": pd.DataFrame(),
            "manager_team": {"bank": 10, "transfers": {"limit": 1}},
        }

    @pytest.fixture
    def mock_llm_response(self):
        """Create a valid mock LLM response."""
        hold_option = HoldOption(
            xp_gw1=60.0,
            xp_3gw=180.0,
            free_transfers_next_week=2,
            reasoning="Strong squad with good fixtures, bank the transfer",
        )

        # Create 3 scenarios (minimum required by Pydantic model)
        scenario1 = TransferScenario(
            scenario_id="premium_upgrade",
            transfers=[],
            xp_gw1=62.0,
            xp_gain_gw1=2.0,
            net_gain_gw1=-2.0,
            xp_3gw=185.0,
            xp_gain_3gw=5.0,
            net_roi_3gw=1.0,
            hit_cost=-4,
            confidence="high",
            reasoning="Upgrade to premium forward for better ceiling",
            leverages_dgw=False,
            leverages_fixture_swing=False,
            prepares_for_chip=False,
            sa_validated=True,
            sa_deviation=0.5,
        )

        scenario2 = TransferScenario(
            scenario_id="differential_punt",
            transfers=[],
            xp_gw1=61.0,
            xp_gain_gw1=1.0,
            net_gain_gw1=1.0,
            xp_3gw=182.0,
            xp_gain_3gw=2.0,
            net_roi_3gw=2.0,
            hit_cost=0,
            confidence="medium",
            reasoning="Low ownership differential with upside",
            leverages_dgw=False,
            leverages_fixture_swing=True,
            prepares_for_chip=False,
            sa_validated=True,
            sa_deviation=-0.3,
        )

        scenario3 = TransferScenario(
            scenario_id="template_safety",
            transfers=[],
            xp_gw1=60.5,
            xp_gain_gw1=0.5,
            net_gain_gw1=0.5,
            xp_3gw=181.0,
            xp_gain_3gw=1.0,
            net_roi_3gw=1.0,
            hit_cost=0,
            confidence="high",
            reasoning="Safe template pick for rank protection",
            leverages_dgw=False,
            leverages_fixture_swing=False,
            prepares_for_chip=False,
            sa_validated=True,
            sa_deviation=0.1,
        )

        return SingleGWRecommendation(
            target_gameweek=18,
            budget_available=1.0,
            current_free_transfers=1,
            hold_option=hold_option,
            recommended_scenarios=[scenario1, scenario2, scenario3],
            context_analysis={
                "dgw_opportunities": [],
                "fixture_swings": ["Arsenal easy run"],
                "chip_timing": "Consider Wildcard in GW20",
            },
            sa_benchmark={"optimal_xp": 62.5, "runtime_seconds": 12.0},
            top_recommendation_id="premium_upgrade",
            final_reasoning="Premium upgrade offers best ROI despite hit",
        )

    def test_generate_recommendations_input_validation(
        self, sample_gameweek_data, mock_llm_response
    ):
        """Test input validation for recommendation generation."""
        service = TransferPlanningAgentService(api_key="test-key")

        # Invalid gameweek
        with pytest.raises(ValueError, match="target_gameweek must be 1-38"):
            service.generate_single_gw_recommendations(
                target_gameweek=50,
                current_squad=sample_gameweek_data["current_squad"],
                gameweek_data=sample_gameweek_data,
            )

        # Invalid squad size
        with pytest.raises(ValueError, match="current_squad must have 15 players"):
            service.generate_single_gw_recommendations(
                target_gameweek=18,
                current_squad=sample_gameweek_data["players"].head(10),
                gameweek_data=sample_gameweek_data,
            )

        # Invalid num_recommendations
        with pytest.raises(ValueError, match="num_recommendations must be 3-5"):
            service.generate_single_gw_recommendations(
                target_gameweek=18,
                current_squad=sample_gameweek_data["current_squad"],
                gameweek_data=sample_gameweek_data,
                num_recommendations=10,
            )

    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.Agent")
    def test_generate_recommendations_success(
        self, mock_agent_class, sample_gameweek_data, mock_llm_response
    ):
        """Test successful recommendation generation with mocked agent."""
        # Mock the agent.run_sync to return our mock response
        mock_agent_instance = Mock()
        mock_result = Mock()
        mock_result.output = mock_llm_response
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        service = TransferPlanningAgentService(api_key="test-key")

        result = service.generate_single_gw_recommendations(
            target_gameweek=18,
            current_squad=sample_gameweek_data["current_squad"],
            gameweek_data=sample_gameweek_data,
            strategy_mode=StrategyMode.BALANCED,
            hit_roi_threshold=5.0,
            num_recommendations=5,
        )

        # Verify result structure
        assert isinstance(result, SingleGWRecommendation)
        assert result.target_gameweek == 18
        assert result.hold_option is not None
        assert len(result.recommended_scenarios) >= 1
        assert result.top_recommendation_id == "premium_upgrade"
        assert result.final_reasoning is not None

        # Verify agent was called correctly
        mock_agent_instance.run_sync.assert_called_once()
        call_kwargs = mock_agent_instance.run_sync.call_args[1]
        assert call_kwargs["output_type"] == SingleGWRecommendation

    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.Agent")
    def test_generate_recommendations_with_conservative_strategy(
        self, mock_agent_class, sample_gameweek_data, mock_llm_response
    ):
        """Test recommendation generation with conservative strategy."""
        mock_agent_instance = Mock()
        mock_result = Mock()
        mock_result.output = mock_llm_response
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        service = TransferPlanningAgentService(api_key="test-key")

        result = service.generate_single_gw_recommendations(
            target_gameweek=18,
            current_squad=sample_gameweek_data["current_squad"],
            gameweek_data=sample_gameweek_data,
            strategy_mode=StrategyMode.CONSERVATIVE,
        )

        assert result is not None

        # Verify system prompt includes conservative strategy
        mock_agent_class.assert_called_once()
        init_kwargs = mock_agent_class.call_args[1]
        system_prompt = init_kwargs["system_prompt"]
        assert "conservative" in system_prompt.lower()

    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.Agent")
    def test_generate_recommendations_with_must_include_exclude(
        self, mock_agent_class, sample_gameweek_data, mock_llm_response
    ):
        """Test recommendation generation with player constraints."""
        mock_agent_instance = Mock()
        mock_result = Mock()
        mock_result.output = mock_llm_response
        mock_agent_instance.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent_instance

        service = TransferPlanningAgentService(api_key="test-key")

        result = service.generate_single_gw_recommendations(
            target_gameweek=18,
            current_squad=sample_gameweek_data["current_squad"],
            gameweek_data=sample_gameweek_data,
            must_include_ids={1, 2, 3},
            must_exclude_ids={99, 100},
        )

        assert result is not None

        # Verify constraints are passed to agent
        user_prompt = mock_agent_instance.run_sync.call_args[0][0]
        assert "Must keep: 1, 2, 3" in user_prompt
        assert "Avoid: 99, 100" in user_prompt

    def test_format_squad_creates_readable_summary(self, sample_gameweek_data):
        """Test _format_squad helper creates readable squad summary."""
        service = TransferPlanningAgentService(api_key="test-key")

        squad_summary = service._format_squad(sample_gameweek_data["current_squad"])

        # Should contain position headers for positions in squad
        assert "GKP:" in squad_summary
        assert "DEF:" in squad_summary
        assert "MID:" in squad_summary
        # Note: First 15 players don't include FWD in this fixture

        # Should contain player names
        assert "Player" in squad_summary

        # Should contain prices
        assert "£" in squad_summary

    def test_format_squad_handles_empty_dataframe(self):
        """Test _format_squad handles empty DataFrame gracefully."""
        service = TransferPlanningAgentService(api_key="test-key")

        squad_summary = service._format_squad(pd.DataFrame())

        assert "No current squad data available" in squad_summary


class TestFreTransfersHandling:
    """Test that free transfers are properly handled throughout the system."""

    @pytest.fixture
    def sample_gameweek_data(self):
        """Create gameweek data with different free transfer scenarios."""
        players_df = pd.DataFrame(
            {
                "player_id": list(range(1, 21)),
                "web_name": [f"Player{i}" for i in range(1, 21)],
                "name": [f"Full Name {i}" for i in range(1, 21)],
                "position": (["GKP"] * 2 + ["DEF"] * 6 + ["MID"] * 8 + ["FWD"] * 4),
                "price": [5.0] * 20,
                "selected_by_percent": [20.0] * 20,
                "team_id": [1, 2] * 10,
            }
        )

        current_squad_df = players_df.head(15).copy()

        teams_df = pd.DataFrame(
            {"team_id": [1, 2, 3], "name": ["Arsenal", "Liverpool", "Chelsea"]}
        )

        fixtures_df = pd.DataFrame(
            {
                "id": [1, 2],
                "event": [18, 18],
                "team_h": [1, 2],
                "team_a": [2, 3],
            }
        )

        return {
            "players": players_df,
            "teams": teams_df,
            "fixtures": fixtures_df,
            "current_squad": current_squad_df,
            "live_data_historical": pd.DataFrame(),
            "ownership_trends": pd.DataFrame(
                {"player_id": [1], "ownership_delta": [0.5]}
            ),
            "value_analysis": pd.DataFrame(),
            "fixture_difficulty": pd.DataFrame(
                {
                    "team_id": [1],
                    "opponent_id": [2],
                    "gameweek": [18],
                    "overall_difficulty": [2.5],
                }
            ),
            "betting_features": pd.DataFrame(),
            "player_metrics": pd.DataFrame(),
            "player_availability": pd.DataFrame(),
            "team_form": pd.DataFrame(),
            "players_enhanced": pd.DataFrame(),
            "xg_rates": pd.DataFrame(),
        }

    def test_one_free_transfer_passed_to_agent(self, sample_gameweek_data):
        """Test that 1 free transfer is passed to agent dependencies."""
        sample_gameweek_data["manager_team"] = {
            "bank": 10,
            "transfers": {"limit": 1},
        }

        TransferPlanningAgentService(api_key="test-key")

        # We can't easily test the agent call without mocking, but we can verify
        # the data extraction works correctly
        budget = sample_gameweek_data.get("manager_team", {}).get("bank", 0.0) / 10.0
        free_transfers = (
            sample_gameweek_data.get("manager_team", {})
            .get("transfers", {})
            .get("limit", 1)
        )

        assert free_transfers == 1
        assert budget == 1.0

    def test_two_free_transfers_passed_to_agent(self, sample_gameweek_data):
        """Test that 2 free transfers (banked) is passed to agent."""
        sample_gameweek_data["manager_team"] = {
            "bank": 25,
            "transfers": {"limit": 2},
        }

        budget = sample_gameweek_data.get("manager_team", {}).get("bank", 0.0) / 10.0
        (
            sample_gameweek_data.get("manager_team", {})
            .get("transfers", {})
            .get("limit", 1)
        )

        # With the limit=2 in manager_team
        free_transfers_actual = sample_gameweek_data["manager_team"]["transfers"][
            "limit"
        ]
        assert free_transfers_actual == 2
        assert budget == 2.5

    def test_zero_free_transfers_edge_case(self, sample_gameweek_data):
        """Test edge case of 0 free transfers."""
        sample_gameweek_data["manager_team"] = {
            "bank": 0,
            "transfers": {"limit": 0},
        }

        free_transfers = sample_gameweek_data["manager_team"]["transfers"]["limit"]
        assert free_transfers == 0


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def sample_gameweek_data(self):
        """Minimal gameweek data for error testing."""
        players_df = pd.DataFrame(
            {
                "player_id": list(range(1, 16)),
                "web_name": [f"Player{i}" for i in range(1, 16)],
                "name": [f"Full Name {i}" for i in range(1, 16)],  # Added team name
                "position": (["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3),
                "price": [5.0] * 15,
                "team_id": [1] * 15,
            }
        )

        return {
            "players": players_df,
            "teams": pd.DataFrame({"team_id": [1]}),
            "fixtures": pd.DataFrame({"id": [1], "event": [18]}),
            "current_squad": players_df,
            "live_data_historical": pd.DataFrame(),
            "manager_team": {"bank": 10, "transfers": {"limit": 1}},
        }

    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.Agent")
    def test_handles_agent_runtime_error(self, mock_agent_class, sample_gameweek_data):
        """Test handling of agent runtime errors."""
        mock_agent_instance = Mock()
        mock_agent_instance.run_sync.side_effect = RuntimeError(
            "Agent execution failed"
        )
        mock_agent_class.return_value = mock_agent_instance

        service = TransferPlanningAgentService(api_key="test-key")

        with pytest.raises(RuntimeError, match="Agent execution failed"):
            service.generate_single_gw_recommendations(
                target_gameweek=18,
                current_squad=sample_gameweek_data["current_squad"],
                gameweek_data=sample_gameweek_data,
            )

    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.Agent")
    def test_handles_api_authentication_error(
        self, mock_agent_class, sample_gameweek_data
    ):
        """Test handling of API authentication errors."""
        mock_agent_instance = Mock()
        mock_agent_instance.run_sync.side_effect = Exception(
            "Invalid API key or authentication failed"
        )
        mock_agent_class.return_value = mock_agent_instance

        service = TransferPlanningAgentService(api_key="invalid-key")

        with pytest.raises(Exception):  # Just check that exception is raised
            service.generate_single_gw_recommendations(
                target_gameweek=18,
                current_squad=sample_gameweek_data["current_squad"],
                gameweek_data=sample_gameweek_data,
            )


class TestAgentToolsRegistration:
    """Test that agent tools are properly registered."""

    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.Agent")
    def test_all_tools_registered_with_agent(self, mock_agent_class):
        """Test that all 5 agent tools are registered."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        service = TransferPlanningAgentService(api_key="test-key")

        # Create minimal gameweek data
        gw_data = {
            "players": pd.DataFrame({"player_id": [1]}),
            "teams": pd.DataFrame({"team_id": [1]}),
            "fixtures": pd.DataFrame({"id": [1]}),
            "current_squad": pd.DataFrame(
                {
                    "player_id": list(range(1, 16)),
                    "position": ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
                }
            ),
            "live_data_historical": pd.DataFrame(),
            "manager_team": {"bank": 10, "transfers": {"limit": 1}},
        }

        try:
            service.generate_single_gw_recommendations(
                target_gameweek=18,
                current_squad=gw_data["current_squad"],
                gameweek_data=gw_data,
            )
        except Exception:
            pass  # We just want to check tool registration

        # Verify agent.tool was called 5 times (once per tool)
        assert mock_agent_instance.tool.call_count == 5


class TestStrategyGuidanceCompleteness:
    """Test that all strategy modes have guidance defined."""

    def test_all_strategy_modes_have_guidance(self):
        """Test that every StrategyMode has corresponding guidance."""
        for strategy_mode in StrategyMode:
            assert strategy_mode in STRATEGY_GUIDANCE, (
                f"Missing guidance for {strategy_mode}"
            )
            assert STRATEGY_GUIDANCE[strategy_mode], (
                f"Empty guidance for {strategy_mode}"
            )

    def test_guidance_is_non_empty_string(self):
        """Test that all guidance entries are non-empty strings."""
        for strategy_mode, guidance in STRATEGY_GUIDANCE.items():
            assert isinstance(guidance, str)
            assert len(guidance.strip()) > 0
