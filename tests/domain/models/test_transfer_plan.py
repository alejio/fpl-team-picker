"""Tests for transfer plan domain models."""

import pytest
from pydantic import ValidationError

from fpl_team_picker.domain.models.transfer_plan import (
    StrategyMode,
    Transfer,
    WeeklyTransferPlan,
    MultiGWPlan,
    AgentState,
)


class TestStrategyMode:
    """Test StrategyMode enum."""

    def test_strategy_mode_values(self):
        """Test all strategy mode values are correct."""
        assert StrategyMode.BALANCED == "balanced"
        assert StrategyMode.CONSERVATIVE == "conservative"
        assert StrategyMode.AGGRESSIVE == "aggressive"
        assert StrategyMode.DGW_STACKER == "dgw_stacker"

    def test_strategy_mode_from_string(self):
        """Test creating StrategyMode from string."""
        assert StrategyMode("balanced") == StrategyMode.BALANCED
        assert StrategyMode("conservative") == StrategyMode.CONSERVATIVE
        assert StrategyMode("aggressive") == StrategyMode.AGGRESSIVE
        assert StrategyMode("dgw_stacker") == StrategyMode.DGW_STACKER


class TestTransfer:
    """Test Transfer model."""

    def test_transfer_creation(self):
        """Test creating a valid transfer."""
        transfer = Transfer(
            player_out_id=1,
            player_out_name="Player Out",
            player_in_id=2,
            player_in_name="Player In",
            cost=0.5,
        )
        assert transfer.player_out_id == 1
        assert transfer.player_out_name == "Player Out"
        assert transfer.player_in_id == 2
        assert transfer.player_in_name == "Player In"
        assert transfer.cost == 0.5

    def test_transfer_negative_cost(self):
        """Test transfer with negative cost (downgrade)."""
        transfer = Transfer(
            player_out_id=1,
            player_out_name="Expensive",
            player_in_id=2,
            player_in_name="Cheap",
            cost=-2.0,
        )
        assert transfer.cost == -2.0

    def test_transfer_string_representation(self):
        """Test transfer string representation."""
        transfer = Transfer(
            player_out_id=1,
            player_out_name="Salah",
            player_in_id=2,
            player_in_name="Son",
            cost=1.5,
        )
        assert "Salah" in str(transfer)
        assert "Son" in str(transfer)
        assert "+1.5" in str(transfer)

    def test_transfer_validation_player_id_positive(self):
        """Test that player IDs must be positive."""
        with pytest.raises(ValidationError):
            Transfer(
                player_out_id=0,
                player_out_name="Player Out",
                player_in_id=2,
                player_in_name="Player In",
                cost=0.5,
            )

        with pytest.raises(ValidationError):
            Transfer(
                player_out_id=1,
                player_out_name="Player Out",
                player_in_id=-1,
                player_in_name="Player In",
                cost=0.5,
            )

    def test_transfer_validation_name_not_empty(self):
        """Test that player names cannot be empty."""
        with pytest.raises(ValidationError):
            Transfer(
                player_out_id=1,
                player_out_name="",
                player_in_id=2,
                player_in_name="Player In",
                cost=0.5,
            )


class TestWeeklyTransferPlan:
    """Test WeeklyTransferPlan model."""

    @pytest.fixture
    def sample_transfer(self):
        """Create a sample transfer."""
        return Transfer(
            player_out_id=1,
            player_out_name="Out",
            player_in_id=2,
            player_in_name="In",
            cost=0.5,
        )

    def test_weekly_plan_with_transfers(self, sample_transfer):
        """Test weekly plan with transfers."""
        plan = WeeklyTransferPlan(
            gameweek=18,
            transfers=[sample_transfer],
            expected_xp=65.5,
            baseline_xp=60.0,
            xp_gain=5.5,
            hit_cost=-4,
            net_gain=1.5,
            reasoning="Good fixture swing",
            confidence="high",
        )
        assert plan.gameweek == 18
        assert len(plan.transfers) == 1
        assert plan.num_transfers == 1
        assert not plan.is_hold
        assert plan.expected_xp == 65.5
        assert plan.baseline_xp == 60.0
        assert plan.xp_gain == 5.5
        assert plan.hit_cost == -4
        assert plan.net_gain == 1.5

    def test_weekly_plan_hold_ft(self):
        """Test weekly plan holding free transfers."""
        plan = WeeklyTransferPlan(
            gameweek=18,
            transfers=[],
            expected_xp=60.0,
            baseline_xp=60.0,
            xp_gain=0.0,
            hit_cost=0,
            net_gain=0.0,
            reasoning="No clear opportunities, banking FT",
            confidence="medium",
        )
        assert plan.num_transfers == 0
        assert plan.is_hold
        assert plan.hit_cost == 0

    def test_weekly_plan_multiple_transfers(self, sample_transfer):
        """Test weekly plan with multiple transfers."""
        transfer2 = Transfer(
            player_out_id=3,
            player_out_name="Out2",
            player_in_id=4,
            player_in_name="In2",
            cost=1.0,
        )
        plan = WeeklyTransferPlan(
            gameweek=18,
            transfers=[sample_transfer, transfer2],
            expected_xp=70.0,
            baseline_xp=60.0,
            xp_gain=10.0,
            hit_cost=-8,
            net_gain=2.0,
            reasoning="Double transfer for DGW",
            confidence="high",
        )
        assert plan.num_transfers == 2
        assert plan.hit_cost == -8

    def test_weekly_plan_validation_gameweek_range(self):
        """Test gameweek must be in valid range."""
        with pytest.raises(ValidationError):
            WeeklyTransferPlan(
                gameweek=0,
                transfers=[],
                expected_xp=60.0,
                baseline_xp=60.0,
                xp_gain=0.0,
                hit_cost=0,
                net_gain=0.0,
                reasoning="Test",
                confidence="low",
            )

        with pytest.raises(ValidationError):
            WeeklyTransferPlan(
                gameweek=39,
                transfers=[],
                expected_xp=60.0,
                baseline_xp=60.0,
                xp_gain=0.0,
                hit_cost=0,
                net_gain=0.0,
                reasoning="Test",
                confidence="low",
            )

    def test_weekly_plan_validation_xp_non_negative(self):
        """Test expected and baseline xP must be non-negative."""
        with pytest.raises(ValidationError):
            WeeklyTransferPlan(
                gameweek=18,
                transfers=[],
                expected_xp=-1.0,
                baseline_xp=60.0,
                xp_gain=0.0,
                hit_cost=0,
                net_gain=0.0,
                reasoning="Test",
                confidence="low",
            )

    def test_weekly_plan_validation_reasoning_not_empty(self):
        """Test reasoning cannot be empty."""
        with pytest.raises(ValidationError):
            WeeklyTransferPlan(
                gameweek=18,
                transfers=[],
                expected_xp=60.0,
                baseline_xp=60.0,
                xp_gain=0.0,
                hit_cost=0,
                net_gain=0.0,
                reasoning="",
                confidence="low",
            )

    def test_weekly_plan_validation_confidence_levels(self):
        """Test confidence must be one of valid levels."""
        valid_levels = ["high", "medium", "low"]
        for level in valid_levels:
            plan = WeeklyTransferPlan(
                gameweek=18,
                transfers=[],
                expected_xp=60.0,
                baseline_xp=60.0,
                xp_gain=0.0,
                hit_cost=0,
                net_gain=0.0,
                reasoning="Test",
                confidence=level,
            )
            assert plan.confidence == level


class TestMultiGWPlan:
    """Test MultiGWPlan model."""

    @pytest.fixture
    def sample_weekly_plan(self):
        """Create a sample weekly plan."""
        return WeeklyTransferPlan(
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

    def test_multi_gw_plan_creation(self, sample_weekly_plan):
        """Test creating a multi-GW plan."""
        plan = MultiGWPlan(
            start_gameweek=18,
            end_gameweek=20,
            strategy_mode=StrategyMode.BALANCED,
            weekly_plans=[sample_weekly_plan],
            total_xp_gain=5.0,
            total_hit_cost=-4,
            net_roi=1.0,
            best_case_roi=8.0,
            worst_case_roi=-2.0,
            final_reasoning="Test reasoning",
        )
        assert plan.start_gameweek == 18
        assert plan.end_gameweek == 20
        assert plan.planning_horizon == 3
        assert plan.strategy_mode == StrategyMode.BALANCED
        assert len(plan.weekly_plans) == 1
        assert plan.total_xp_gain == 5.0
        assert plan.total_hit_cost == -4
        assert plan.net_roi == 1.0

    def test_multi_gw_plan_planning_horizon(self, sample_weekly_plan):
        """Test planning horizon calculation."""
        plan = MultiGWPlan(
            start_gameweek=18,
            end_gameweek=22,
            strategy_mode=StrategyMode.BALANCED,
            weekly_plans=[sample_weekly_plan] * 5,
            total_xp_gain=10.0,
            total_hit_cost=0,
            net_roi=10.0,
            best_case_roi=15.0,
            worst_case_roi=5.0,
            final_reasoning="Test",
        )
        assert plan.planning_horizon == 5

    def test_multi_gw_plan_total_transfers(self, sample_weekly_plan):
        """Test total transfers calculation."""
        transfer = Transfer(
            player_out_id=1,
            player_out_name="Out",
            player_in_id=2,
            player_in_name="In",
            cost=0.5,
        )
        plan_with_transfer = WeeklyTransferPlan(
            gameweek=19,
            transfers=[transfer],
            expected_xp=65.0,
            baseline_xp=60.0,
            xp_gain=5.0,
            hit_cost=-4,
            net_gain=1.0,
            reasoning="Transfer",
            confidence="high",
        )

        plan = MultiGWPlan(
            start_gameweek=18,
            end_gameweek=20,
            strategy_mode=StrategyMode.BALANCED,
            weekly_plans=[sample_weekly_plan, plan_with_transfer, sample_weekly_plan],
            total_xp_gain=5.0,
            total_hit_cost=-4,
            net_roi=1.0,
            best_case_roi=8.0,
            worst_case_roi=-2.0,
            final_reasoning="Test",
        )
        assert plan.total_transfers == 1

    def test_multi_gw_plan_validation_min_weekly_plans(self):
        """Test that at least one weekly plan is required."""
        with pytest.raises(ValidationError):
            MultiGWPlan(
                start_gameweek=18,
                end_gameweek=20,
                strategy_mode=StrategyMode.BALANCED,
                weekly_plans=[],
                total_xp_gain=5.0,
                total_hit_cost=-4,
                net_roi=1.0,
                best_case_roi=8.0,
                worst_case_roi=-2.0,
                final_reasoning="Test",
            )

    def test_multi_gw_plan_validation_gameweek_range(self, sample_weekly_plan):
        """Test gameweek range validation."""
        with pytest.raises(ValidationError):
            MultiGWPlan(
                start_gameweek=0,
                end_gameweek=20,
                strategy_mode=StrategyMode.BALANCED,
                weekly_plans=[sample_weekly_plan],
                total_xp_gain=5.0,
                total_hit_cost=-4,
                net_roi=1.0,
                best_case_roi=8.0,
                worst_case_roi=-2.0,
                final_reasoning="Test",
            )

    def test_multi_gw_plan_optional_fields(self, sample_weekly_plan):
        """Test optional fields can be omitted."""
        plan = MultiGWPlan(
            start_gameweek=18,
            end_gameweek=20,
            strategy_mode=StrategyMode.BALANCED,
            weekly_plans=[sample_weekly_plan],
            total_xp_gain=5.0,
            total_hit_cost=-4,
            net_roi=1.0,
            best_case_roi=8.0,
            worst_case_roi=-2.0,
            final_reasoning="Test",
        )
        # Optional fields should have defaults
        assert plan.opportunities_identified == []
        assert plan.constraints_considered == []
        assert plan.trade_offs == []
        assert plan.template_comparison == {}
        assert plan.chip_timing is None


class TestAgentState:
    """Test AgentState model."""

    def test_agent_state_creation(self):
        """Test creating agent state."""
        state = AgentState(
            current_gameweek=18,
            planning_horizon=3,
            current_squad=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            budget=2.5,
            free_transfers=1,
        )
        assert state.current_gameweek == 18
        assert state.planning_horizon == 3
        assert len(state.current_squad) == 15
        assert state.budget == 2.5
        assert state.free_transfers == 1

    def test_agent_state_validation_squad_size(self):
        """Test squad must be exactly 15 players."""
        with pytest.raises(ValidationError):
            AgentState(
                current_gameweek=18,
                planning_horizon=3,
                current_squad=[1, 2, 3],  # Too few
                budget=2.5,
                free_transfers=1,
            )

        with pytest.raises(ValidationError):
            AgentState(
                current_gameweek=18,
                planning_horizon=3,
                current_squad=list(range(1, 20)),  # Too many
                budget=2.5,
                free_transfers=1,
            )

    def test_agent_state_validation_budget_range(self):
        """Test budget must be in valid range."""
        with pytest.raises(ValidationError):
            AgentState(
                current_gameweek=18,
                planning_horizon=3,
                current_squad=list(range(1, 16)),
                budget=-1.0,  # Negative
                free_transfers=1,
            )

        with pytest.raises(ValidationError):
            AgentState(
                current_gameweek=18,
                planning_horizon=3,
                current_squad=list(range(1, 16)),
                budget=101.0,  # Too high
                free_transfers=1,
            )

    def test_agent_state_validation_free_transfers_range(self):
        """Test free transfers must be in valid range."""
        with pytest.raises(ValidationError):
            AgentState(
                current_gameweek=18,
                planning_horizon=3,
                current_squad=list(range(1, 16)),
                budget=2.5,
                free_transfers=-1,  # Negative
            )

        with pytest.raises(ValidationError):
            AgentState(
                current_gameweek=18,
                planning_horizon=3,
                current_squad=list(range(1, 16)),
                budget=2.5,
                free_transfers=16,  # Too high
            )

    def test_agent_state_optional_fields(self):
        """Test optional fields can be None."""
        state = AgentState(
            current_gameweek=18,
            planning_horizon=3,
            current_squad=list(range(1, 16)),
            budget=2.5,
            free_transfers=1,
        )
        assert state.xp_predictions is None
        assert state.dgw_bgw_info is None
        assert state.fixture_swings is None
        assert state.must_include_ids == []
        assert state.must_exclude_ids == []
        assert state.hit_roi_threshold == 5.0
