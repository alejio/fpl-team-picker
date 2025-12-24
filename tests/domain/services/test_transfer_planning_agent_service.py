"""Tests for TransferPlanningAgentService."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
import os

from fpl_team_picker.domain.services.transfer_planning_agent_service import (
    TransferPlanningAgentService,
    SYSTEM_PROMPTS,
)
from fpl_team_picker.domain.models.transfer_plan import (
    StrategyMode,
    MultiGWPlan,
    WeeklyTransferPlan,
    Transfer,
)


class TestTransferPlanningAgentService:
    """Test TransferPlanningAgentService."""

    @pytest.fixture
    def sample_squad(self):
        """Create a sample 15-player squad."""
        return pd.DataFrame(
            {
                "player_id": list(range(1, 16)),
                "web_name": [f"Player{i}" for i in range(1, 16)],
                "name": [f"Player {i}" for i in range(1, 16)],
                "position": ["GKP", "GKP"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
                "price": [5.0] * 2 + [6.0] * 5 + [8.0] * 5 + [10.0] * 3,
            }
        )

    @pytest.fixture
    def sample_gameweek_data(self):
        """Create sample gameweek data dictionary."""
        return {
            "players": pd.DataFrame(
                {
                    "player_id": [1, 2, 3],
                    "web_name": ["Salah", "Son", "Haaland"],
                    "position": ["MID", "MID", "FWD"],
                }
            ),
            "teams": pd.DataFrame({"team_id": [1, 2, 3], "team_name": ["A", "B", "C"]}),
            "fixtures": pd.DataFrame({"fixture_id": [1, 2, 3]}),
            "manager_team": {"bank": 50.0, "transfers": {"limit": 1}},
        }

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing."""
        return "test-api-key-12345"

    def test_service_initialization_with_api_key(self, mock_api_key):
        """Test service initialization with explicit API key."""
        service = TransferPlanningAgentService(api_key=mock_api_key)
        assert service.api_key == mock_api_key
        assert service.model_name is not None

    def test_service_initialization_with_model(self, mock_api_key):
        """Test service initialization with custom model."""
        service = TransferPlanningAgentService(
            model="claude-sonnet-4-5", api_key=mock_api_key
        )
        assert service.model_name == "claude-sonnet-4-5"

    @patch.dict(os.environ, {}, clear=True)
    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.config")
    def test_service_initialization_missing_api_key(self, mock_config):
        """Test that missing API key raises ValueError."""
        mock_config.transfer_planning_agent.api_key = None
        mock_config.transfer_planning_agent.model = "claude-sonnet-4-5"

        with pytest.raises(ValueError, match="Anthropic API key is required"):
            TransferPlanningAgentService()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"})
    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.config")
    def test_service_initialization_from_env(self, mock_config):
        """Test service initialization from environment variable."""
        mock_config.transfer_planning_agent.api_key = None
        mock_config.transfer_planning_agent.model = "claude-sonnet-4-5"

        service = TransferPlanningAgentService()
        assert service.api_key == "env-key"

    def test_format_squad(self, mock_api_key, sample_squad):
        """Test _format_squad method formats squad correctly."""
        service = TransferPlanningAgentService(api_key=mock_api_key)
        formatted = service._format_squad(sample_squad)

        assert "GKP:" in formatted
        assert "DEF:" in formatted
        assert "MID:" in formatted
        assert "FWD:" in formatted
        assert "Player1" in formatted

    def test_format_squad_empty(self, mock_api_key):
        """Test _format_squad with empty squad."""
        service = TransferPlanningAgentService(api_key=mock_api_key)
        formatted = service._format_squad(pd.DataFrame())
        assert formatted == "No current squad data available"

    def test_system_prompts_exist_for_all_strategies(self):
        """Test that system prompts exist for all strategy modes."""
        for strategy in StrategyMode:
            assert strategy in SYSTEM_PROMPTS
            prompt = SYSTEM_PROMPTS[strategy]
            # BALANCED uses {horizon}, others may not
            if strategy == StrategyMode.BALANCED:
                assert "{horizon}" in prompt
            # All prompts should have {roi_threshold} (except DGW_STACKER which doesn't use it)
            if strategy != StrategyMode.DGW_STACKER:
                assert "{roi_threshold}" in prompt
            # All prompts should be non-empty
            assert len(prompt) > 0

    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.Agent")
    @patch(
        "fpl_team_picker.domain.services.transfer_planning_agent_service.AnthropicProvider"
    )
    @patch(
        "fpl_team_picker.domain.services.transfer_planning_agent_service.AnthropicModel"
    )
    def test_generate_multi_gw_plan_basic(
        self,
        mock_model_class,
        mock_provider_class,
        mock_agent_class,
        mock_api_key,
        sample_squad,
        sample_gameweek_data,
    ):
        """Test generate_multi_gw_plan with mocked agent."""
        # Create mock agent and result
        mock_agent = Mock()
        mock_result = Mock()
        mock_plan = MultiGWPlan(
            start_gameweek=18,
            end_gameweek=20,
            strategy_mode=StrategyMode.BALANCED,
            weekly_plans=[
                WeeklyTransferPlan(
                    gameweek=18,
                    transfers=[],
                    expected_xp=60.0,
                    baseline_xp=60.0,
                    xp_gain=0.0,
                    hit_cost=0,
                    net_gain=0.0,
                    reasoning="Hold FT",
                    confidence="medium",
                )
            ],
            total_xp_gain=0.0,
            total_hit_cost=0,
            net_roi=0.0,
            best_case_roi=2.0,
            worst_case_roi=-2.0,
            final_reasoning="Test plan",
        )
        mock_result.response = mock_plan
        mock_agent.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent

        service = TransferPlanningAgentService(api_key=mock_api_key)
        plan = service.generate_multi_gw_plan(
            start_gameweek=18,
            planning_horizon=3,
            current_squad=sample_squad,
            gameweek_data=sample_gameweek_data,
            strategy_mode=StrategyMode.BALANCED,
        )

        # Verify agent was initialized correctly
        mock_agent_class.assert_called_once()
        assert mock_agent.tool.called  # Tools should be registered

        # Verify agent.run_sync was called
        mock_agent.run_sync.assert_called_once()
        call_args = mock_agent.run_sync.call_args
        assert call_args[0][0]  # User prompt should be provided
        assert call_args[1]["deps"]  # Dependencies should be provided
        assert call_args[1]["output_type"] == MultiGWPlan

        # Verify returned plan
        assert plan.start_gameweek == 18
        assert plan.end_gameweek == 20
        assert plan.strategy_mode == StrategyMode.BALANCED

    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.Agent")
    @patch(
        "fpl_team_picker.domain.services.transfer_planning_agent_service.AnthropicProvider"
    )
    @patch(
        "fpl_team_picker.domain.services.transfer_planning_agent_service.AnthropicModel"
    )
    def test_generate_multi_gw_plan_with_transfers(
        self,
        mock_model_class,
        mock_provider_class,
        mock_agent_class,
        mock_api_key,
        sample_squad,
        sample_gameweek_data,
    ):
        """Test generate_multi_gw_plan with transfer recommendations."""
        mock_agent = Mock()
        mock_result = Mock()

        # Create plan with transfers
        transfer = Transfer(
            player_out_id=1,
            player_out_name="Out",
            player_in_id=2,
            player_in_name="In",
            cost=0.5,
        )
        weekly_plan = WeeklyTransferPlan(
            gameweek=18,
            transfers=[transfer],
            expected_xp=65.0,
            baseline_xp=60.0,
            xp_gain=5.0,
            hit_cost=-4,
            net_gain=1.0,
            reasoning="Good fixture swing",
            confidence="high",
        )

        mock_plan = MultiGWPlan(
            start_gameweek=18,
            end_gameweek=20,
            strategy_mode=StrategyMode.AGGRESSIVE,
            weekly_plans=[weekly_plan],
            total_xp_gain=5.0,
            total_hit_cost=-4,
            net_roi=1.0,
            best_case_roi=8.0,
            worst_case_roi=-2.0,
            final_reasoning="Aggressive transfer",
        )
        mock_result.response = mock_plan
        mock_agent.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent

        service = TransferPlanningAgentService(api_key=mock_api_key)
        plan = service.generate_multi_gw_plan(
            start_gameweek=18,
            planning_horizon=3,
            current_squad=sample_squad,
            gameweek_data=sample_gameweek_data,
            strategy_mode=StrategyMode.AGGRESSIVE,
            hit_roi_threshold=4.0,
        )

        assert plan.total_transfers == 1
        assert plan.total_hit_cost == -4
        assert plan.net_roi == 1.0
        assert plan.strategy_mode == StrategyMode.AGGRESSIVE

    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.Agent")
    @patch(
        "fpl_team_picker.domain.services.transfer_planning_agent_service.AnthropicProvider"
    )
    @patch(
        "fpl_team_picker.domain.services.transfer_planning_agent_service.AnthropicModel"
    )
    def test_generate_multi_gw_plan_all_strategies(
        self,
        mock_model_class,
        mock_provider_class,
        mock_agent_class,
        mock_api_key,
        sample_squad,
        sample_gameweek_data,
    ):
        """Test generate_multi_gw_plan with all strategy modes."""
        mock_agent = Mock()
        mock_result = Mock()

        for strategy in StrategyMode:
            mock_plan = MultiGWPlan(
                start_gameweek=18,
                end_gameweek=20,
                strategy_mode=strategy,
                weekly_plans=[
                    WeeklyTransferPlan(
                        gameweek=18,
                        transfers=[],
                        expected_xp=60.0,
                        baseline_xp=60.0,
                        xp_gain=0.0,
                        hit_cost=0,
                        net_gain=0.0,
                        reasoning="Test",
                        confidence="medium",
                    )
                ],
                total_xp_gain=0.0,
                total_hit_cost=0,
                net_roi=0.0,
                best_case_roi=2.0,
                worst_case_roi=-2.0,
                final_reasoning="Test",
            )
            mock_result.response = mock_plan
            mock_agent.run_sync.return_value = mock_result
            mock_agent_class.return_value = mock_agent

            service = TransferPlanningAgentService(api_key=mock_api_key)
            plan = service.generate_multi_gw_plan(
                start_gameweek=18,
                planning_horizon=3,
                current_squad=sample_squad,
                gameweek_data=sample_gameweek_data,
                strategy_mode=strategy,
            )

            assert plan.strategy_mode == strategy

    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.Agent")
    @patch(
        "fpl_team_picker.domain.services.transfer_planning_agent_service.AnthropicProvider"
    )
    @patch(
        "fpl_team_picker.domain.services.transfer_planning_agent_service.AnthropicModel"
    )
    def test_generate_multi_gw_plan_with_constraints(
        self,
        mock_model_class,
        mock_provider_class,
        mock_agent_class,
        mock_api_key,
        sample_squad,
        sample_gameweek_data,
    ):
        """Test generate_multi_gw_plan with must_include and must_exclude."""
        mock_agent = Mock()
        mock_result = Mock()
        mock_plan = MultiGWPlan(
            start_gameweek=18,
            end_gameweek=20,
            strategy_mode=StrategyMode.BALANCED,
            weekly_plans=[
                WeeklyTransferPlan(
                    gameweek=18,
                    transfers=[],
                    expected_xp=60.0,
                    baseline_xp=60.0,
                    xp_gain=0.0,
                    hit_cost=0,
                    net_gain=0.0,
                    reasoning="Test",
                    confidence="medium",
                )
            ],
            total_xp_gain=0.0,
            total_hit_cost=0,
            net_roi=0.0,
            best_case_roi=2.0,
            worst_case_roi=-2.0,
            final_reasoning="Test",
        )
        mock_result.response = mock_plan
        mock_agent.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent

        service = TransferPlanningAgentService(api_key=mock_api_key)
        service.generate_multi_gw_plan(
            start_gameweek=18,
            planning_horizon=3,
            current_squad=sample_squad,
            gameweek_data=sample_gameweek_data,
            strategy_mode=StrategyMode.BALANCED,
            must_include=["Salah", "Haaland"],
            must_exclude=["Injured Player"],
        )

        # Verify constraints were included in prompt
        call_args = mock_agent.run_sync.call_args
        user_prompt = call_args[0][0]
        assert "Must keep: Salah, Haaland" in user_prompt
        assert "Avoid: Injured Player" in user_prompt

    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.Agent")
    @patch(
        "fpl_team_picker.domain.services.transfer_planning_agent_service.AnthropicProvider"
    )
    @patch(
        "fpl_team_picker.domain.services.transfer_planning_agent_service.AnthropicModel"
    )
    def test_generate_multi_gw_plan_agent_deps_creation(
        self,
        mock_model_class,
        mock_provider_class,
        mock_agent_class,
        mock_api_key,
        sample_squad,
        sample_gameweek_data,
    ):
        """Test that AgentDeps are created correctly from gameweek_data."""
        mock_agent = Mock()
        mock_result = Mock()
        mock_plan = MultiGWPlan(
            start_gameweek=18,
            end_gameweek=20,
            strategy_mode=StrategyMode.BALANCED,
            weekly_plans=[
                WeeklyTransferPlan(
                    gameweek=18,
                    transfers=[],
                    expected_xp=60.0,
                    baseline_xp=60.0,
                    xp_gain=0.0,
                    hit_cost=0,
                    net_gain=0.0,
                    reasoning="Test",
                    confidence="medium",
                )
            ],
            total_xp_gain=0.0,
            total_hit_cost=0,
            net_roi=0.0,
            best_case_roi=2.0,
            worst_case_roi=-2.0,
            final_reasoning="Test",
        )
        mock_result.response = mock_plan
        mock_agent.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent

        # Add optional data
        sample_gameweek_data["ownership_trends"] = pd.DataFrame({"player_id": [1]})
        sample_gameweek_data["value_analysis"] = pd.DataFrame({"player_id": [1]})
        sample_gameweek_data["live_data_historical"] = pd.DataFrame()

        service = TransferPlanningAgentService(api_key=mock_api_key)
        service.generate_multi_gw_plan(
            start_gameweek=18,
            planning_horizon=3,
            current_squad=sample_squad,
            gameweek_data=sample_gameweek_data,
            strategy_mode=StrategyMode.BALANCED,
        )

        # Verify deps were passed to agent
        call_args = mock_agent.run_sync.call_args
        deps = call_args[1]["deps"]
        assert deps.players_data is sample_gameweek_data["players"]
        assert deps.teams_data is sample_gameweek_data["teams"]
        assert deps.fixtures_data is sample_gameweek_data["fixtures"]

    @patch("fpl_team_picker.domain.services.transfer_planning_agent_service.Agent")
    @patch(
        "fpl_team_picker.domain.services.transfer_planning_agent_service.AnthropicProvider"
    )
    @patch(
        "fpl_team_picker.domain.services.transfer_planning_agent_service.AnthropicModel"
    )
    def test_generate_multi_gw_plan_user_prompt_content(
        self,
        mock_model_class,
        mock_provider_class,
        mock_agent_class,
        mock_api_key,
        sample_squad,
        sample_gameweek_data,
    ):
        """Test that user prompt contains expected information."""
        mock_agent = Mock()
        mock_result = Mock()
        mock_plan = MultiGWPlan(
            start_gameweek=18,
            end_gameweek=20,
            strategy_mode=StrategyMode.BALANCED,
            weekly_plans=[
                WeeklyTransferPlan(
                    gameweek=18,
                    transfers=[],
                    expected_xp=60.0,
                    baseline_xp=60.0,
                    xp_gain=0.0,
                    hit_cost=0,
                    net_gain=0.0,
                    reasoning="Test",
                    confidence="medium",
                )
            ],
            total_xp_gain=0.0,
            total_hit_cost=0,
            net_roi=0.0,
            best_case_roi=2.0,
            worst_case_roi=-2.0,
            final_reasoning="Test",
        )
        mock_result.response = mock_plan
        mock_agent.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent

        service = TransferPlanningAgentService(api_key=mock_api_key)
        service.generate_multi_gw_plan(
            start_gameweek=18,
            planning_horizon=3,
            current_squad=sample_squad,
            gameweek_data=sample_gameweek_data,
            strategy_mode=StrategyMode.BALANCED,
            hit_roi_threshold=6.0,
        )

        # Verify prompt content
        call_args = mock_agent.run_sync.call_args
        user_prompt = call_args[0][0]
        assert "18-gameweek" in user_prompt or "3-gameweek" in user_prompt
        assert "GW18" in user_prompt
        assert "Budget:" in user_prompt or "£" in user_prompt
        assert "Free Transfers:" in user_prompt
        assert "6.0" in user_prompt  # ROI threshold
