"""Tests for Transfer Recommendation models."""

import pytest

from fpl_team_picker.domain.models.transfer_plan import Transfer
from fpl_team_picker.domain.models.transfer_recommendation import (
    HoldOption,
    SingleGWRecommendation,
    TransferScenario,
)


class TestTransferScenario:
    """Test TransferScenario model validation."""

    def test_valid_transfer_scenario_creation(self):
        """Test creating a valid transfer scenario."""
        transfer = Transfer(
            player_out_id=5,
            player_in_id=42,
            player_out_name="Smith",
            player_in_name="Jones",
            position="MID",
            cost=0.5,
        )

        scenario = TransferScenario(
            scenario_id="option_1",
            transfers=[transfer],
            xp_gw1=65.8,
            xp_gain_gw1=2.6,
            hit_cost=-4,
            net_gain_gw1=-1.4,
            xp_3gw=197.2,
            xp_gain_3gw=7.7,
            net_roi_3gw=3.7,
            reasoning="Haaland has Liverpool(H) DGW in GW20",
            confidence="high",
        )

        assert scenario.scenario_id == "option_1"
        assert len(scenario.transfers) == 1
        assert scenario.xp_gw1 == 65.8
        assert scenario.hit_cost == -4
        assert scenario.net_roi_3gw == 3.7
        assert scenario.confidence == "high"
        assert scenario.num_transfers == 1
        assert not scenario.is_hold

    def test_hold_scenario_creation(self):
        """Test creating a hold scenario (no transfers)."""
        scenario = TransferScenario(
            scenario_id="hold",
            transfers=[],
            xp_gw1=63.2,
            xp_gain_gw1=0.0,
            hit_cost=0,
            net_gain_gw1=0.0,
            xp_3gw=189.5,
            xp_gain_3gw=0.0,
            net_roi_3gw=0.0,
            reasoning="Strong fixtures, worth banking FT",
            confidence="medium",
        )

        assert scenario.is_hold
        assert scenario.num_transfers == 0
        assert scenario.hit_cost == 0

    def test_scenario_with_context_flags(self):
        """Test scenario with DGW and fixture swing flags."""
        scenario = TransferScenario(
            scenario_id="dgw_option",
            transfers=[],
            xp_gw1=67.0,
            xp_gain_gw1=3.8,
            hit_cost=-4,
            net_gain_gw1=-0.2,
            xp_3gw=205.0,
            xp_gain_3gw=15.5,
            net_roi_3gw=11.5,
            reasoning="Target DGW opportunity",
            confidence="high",
            leverages_dgw=True,
            leverages_fixture_swing=True,
            prepares_for_chip=False,
        )

        assert scenario.leverages_dgw
        assert scenario.leverages_fixture_swing
        assert not scenario.prepares_for_chip

    def test_scenario_with_sa_validation(self):
        """Test scenario with SA validation data."""
        scenario = TransferScenario(
            scenario_id="validated",
            transfers=[],
            xp_gw1=66.0,
            xp_gain_gw1=2.8,
            hit_cost=0,
            net_gain_gw1=2.8,
            xp_3gw=198.0,
            xp_gain_3gw=8.5,
            net_roi_3gw=8.5,
            reasoning="SA validated pick",
            confidence="high",
            sa_validated=True,
            sa_deviation=0.3,
        )

        assert scenario.sa_validated
        assert scenario.sa_deviation == 0.3

    def test_scenario_id_validation(self):
        """Test scenario_id cannot be empty."""
        with pytest.raises(ValueError):
            TransferScenario(
                scenario_id="",  # Empty string not allowed
                transfers=[],
                xp_gw1=60.0,
                xp_gain_gw1=0.0,
                hit_cost=0,
                net_gain_gw1=0.0,
                xp_3gw=180.0,
                xp_gain_3gw=0.0,
                net_roi_3gw=0.0,
                reasoning="Test",
                confidence="medium",
            )

    def test_xp_validation(self):
        """Test xP values must be non-negative."""
        with pytest.raises(ValueError):
            TransferScenario(
                scenario_id="invalid",
                transfers=[],
                xp_gw1=-5.0,  # Negative xP not allowed
                xp_gain_gw1=0.0,
                hit_cost=0,
                net_gain_gw1=0.0,
                xp_3gw=180.0,
                xp_gain_3gw=0.0,
                net_roi_3gw=0.0,
                reasoning="Test",
                confidence="medium",
            )

    def test_confidence_validation(self):
        """Test confidence must be high/medium/low."""
        with pytest.raises(ValueError):
            TransferScenario(
                scenario_id="invalid",
                transfers=[],
                xp_gw1=60.0,
                xp_gain_gw1=0.0,
                hit_cost=0,
                net_gain_gw1=0.0,
                xp_3gw=180.0,
                xp_gain_3gw=0.0,
                net_roi_3gw=0.0,
                reasoning="Test",
                confidence="very_high",  # Invalid confidence level
            )


class TestHoldOption:
    """Test HoldOption model validation."""

    def test_valid_hold_option_creation(self):
        """Test creating a valid hold option."""
        hold = HoldOption(
            xp_gw1=63.2,
            xp_3gw=189.5,
            free_transfers_next_week=2,
            reasoning="Strong fixtures, worth banking for future flexibility",
        )

        assert hold.xp_gw1 == 63.2
        assert hold.xp_3gw == 189.5
        assert hold.free_transfers_next_week == 2
        assert "banking" in hold.reasoning.lower()

    def test_xp_validation(self):
        """Test xP values must be non-negative."""
        with pytest.raises(ValueError):
            HoldOption(
                xp_gw1=-10.0,  # Negative not allowed
                xp_3gw=180.0,
                free_transfers_next_week=1,
                reasoning="Test",
            )

    def test_free_transfers_validation(self):
        """Test free transfers must be 1 or 2."""
        with pytest.raises(ValueError):
            HoldOption(
                xp_gw1=60.0,
                xp_3gw=180.0,
                free_transfers_next_week=3,  # Max is 2
                reasoning="Test",
            )

        with pytest.raises(ValueError):
            HoldOption(
                xp_gw1=60.0,
                xp_3gw=180.0,
                free_transfers_next_week=0,  # Min is 1
                reasoning="Test",
            )

    def test_reasoning_validation(self):
        """Test reasoning cannot be empty."""
        with pytest.raises(ValueError):
            HoldOption(
                xp_gw1=60.0, xp_3gw=180.0, free_transfers_next_week=1, reasoning=""
            )


class TestSingleGWRecommendation:
    """Test SingleGWRecommendation model validation."""

    def test_valid_recommendation_creation(self):
        """Test creating a valid single-GW recommendation."""
        hold = HoldOption(
            xp_gw1=63.2,
            xp_3gw=189.5,
            free_transfers_next_week=2,
            reasoning="Bank FT",
        )

        scenario1 = TransferScenario(
            scenario_id="option_1",
            transfers=[],
            xp_gw1=65.8,
            xp_gain_gw1=2.6,
            hit_cost=-4,
            net_gain_gw1=-1.4,
            xp_3gw=197.2,
            xp_gain_3gw=7.7,
            net_roi_3gw=3.7,
            reasoning="Premium upgrade",
            confidence="high",
        )

        scenario2 = TransferScenario(
            scenario_id="option_2",
            transfers=[],
            xp_gw1=64.1,
            xp_gain_gw1=0.9,
            hit_cost=0,
            net_gain_gw1=0.9,
            xp_3gw=192.3,
            xp_gain_3gw=2.8,
            net_roi_3gw=2.8,
            reasoning="Differential punt",
            confidence="medium",
        )

        scenario3 = TransferScenario(
            scenario_id="option_3",
            transfers=[],
            xp_gw1=63.5,
            xp_gain_gw1=0.3,
            hit_cost=0,
            net_gain_gw1=0.3,
            xp_3gw=190.8,
            xp_gain_3gw=1.3,
            net_roi_3gw=1.3,
            reasoning="Safe pick",
            confidence="medium",
        )

        recommendation = SingleGWRecommendation(
            target_gameweek=18,
            current_free_transfers=1,
            budget_available=2.3,
            hold_option=hold,
            recommended_scenarios=[scenario1, scenario2, scenario3],
            context_analysis={
                "dgw_opportunities": ["Arsenal", "Chelsea"],
                "fixture_swings": ["Newcastle"],
            },
            sa_benchmark={"expected_xp": 66.0, "transfers": 1},
            top_recommendation_id="option_1",
            final_reasoning="3GW ROI justifies the hit",
        )

        assert recommendation.target_gameweek == 18
        assert recommendation.current_free_transfers == 1
        assert recommendation.budget_available == 2.3
        assert len(recommendation.recommended_scenarios) == 3
        assert recommendation.top_recommendation_id == "option_1"
        assert recommendation.total_scenarios == 4  # 3 scenarios + hold

    def test_best_scenario_property(self):
        """Test best_scenario property returns correct scenario."""
        hold = HoldOption(
            xp_gw1=63.2, xp_3gw=189.5, free_transfers_next_week=2, reasoning="Bank FT"
        )

        scenario1 = TransferScenario(
            scenario_id="option_1",
            transfers=[],
            xp_gw1=65.8,
            xp_gain_gw1=2.6,
            hit_cost=-4,
            net_gain_gw1=-1.4,
            xp_3gw=197.2,
            xp_gain_3gw=7.7,
            net_roi_3gw=3.7,
            reasoning="Best option",
            confidence="high",
        )

        scenario2 = TransferScenario(
            scenario_id="option_2",
            transfers=[],
            xp_gw1=64.0,
            xp_gain_gw1=0.8,
            hit_cost=0,
            net_gain_gw1=0.8,
            xp_3gw=192.0,
            xp_gain_3gw=2.5,
            net_roi_3gw=2.5,
            reasoning="Second option",
            confidence="medium",
        )

        scenario3 = TransferScenario(
            scenario_id="option_3",
            transfers=[],
            xp_gw1=63.0,
            xp_gain_gw1=0.0,
            hit_cost=0,
            net_gain_gw1=0.0,
            xp_3gw=190.0,
            xp_gain_3gw=0.5,
            net_roi_3gw=0.5,
            reasoning="Third option",
            confidence="low",
        )

        recommendation = SingleGWRecommendation(
            target_gameweek=18,
            current_free_transfers=1,
            budget_available=2.3,
            hold_option=hold,
            recommended_scenarios=[scenario1, scenario2, scenario3],
            top_recommendation_id="option_1",
            final_reasoning="Best ROI",
        )

        best = recommendation.best_scenario
        assert isinstance(best, TransferScenario)
        assert best.scenario_id == "option_1"

    def test_best_scenario_hold_option(self):
        """Test best_scenario returns hold_option when top_recommendation_id is 'hold'."""
        hold = HoldOption(
            xp_gw1=65.0, xp_3gw=195.0, free_transfers_next_week=2, reasoning="Bank FT"
        )

        scenario1 = TransferScenario(
            scenario_id="option_1",
            transfers=[],
            xp_gw1=64.0,
            xp_gain_gw1=-1.0,
            hit_cost=-4,
            net_gain_gw1=-5.0,
            xp_3gw=192.0,
            xp_gain_3gw=-3.0,
            net_roi_3gw=-7.0,
            reasoning="Worse than hold",
            confidence="low",
        )

        scenario2 = TransferScenario(
            scenario_id="option_2",
            transfers=[],
            xp_gw1=64.5,
            xp_gain_gw1=-0.5,
            hit_cost=0,
            net_gain_gw1=-0.5,
            xp_3gw=193.5,
            xp_gain_3gw=-1.5,
            net_roi_3gw=-1.5,
            reasoning="Also worse",
            confidence="low",
        )

        scenario3 = TransferScenario(
            scenario_id="option_3",
            transfers=[],
            xp_gw1=64.2,
            xp_gain_gw1=-0.8,
            hit_cost=0,
            net_gain_gw1=-0.8,
            xp_3gw=193.0,
            xp_gain_3gw=-2.0,
            net_roi_3gw=-2.0,
            reasoning="Still worse",
            confidence="low",
        )

        recommendation = SingleGWRecommendation(
            target_gameweek=18,
            current_free_transfers=1,
            budget_available=2.3,
            hold_option=hold,
            recommended_scenarios=[scenario1, scenario2, scenario3],
            top_recommendation_id="hold",
            final_reasoning="Hold is best option",
        )

        best = recommendation.best_scenario
        assert isinstance(best, HoldOption)
        assert best.xp_gw1 == 65.0

    def test_min_scenarios_validation(self):
        """Test at least 3 scenarios required."""
        hold = HoldOption(
            xp_gw1=63.2, xp_3gw=189.5, free_transfers_next_week=1, reasoning="Hold"
        )

        scenario1 = TransferScenario(
            scenario_id="option_1",
            transfers=[],
            xp_gw1=65.0,
            xp_gain_gw1=1.8,
            hit_cost=0,
            net_gain_gw1=1.8,
            xp_3gw=195.0,
            xp_gain_3gw=5.5,
            net_roi_3gw=5.5,
            reasoning="Only option",
            confidence="medium",
        )

        scenario2 = TransferScenario(
            scenario_id="option_2",
            transfers=[],
            xp_gw1=64.0,
            xp_gain_gw1=0.8,
            hit_cost=0,
            net_gain_gw1=0.8,
            xp_3gw=192.0,
            xp_gain_3gw=2.5,
            net_roi_3gw=2.5,
            reasoning="Second option",
            confidence="low",
        )

        # Only 2 scenarios - should fail
        with pytest.raises(ValueError):
            SingleGWRecommendation(
                target_gameweek=18,
                current_free_transfers=1,
                budget_available=2.3,
                hold_option=hold,
                recommended_scenarios=[scenario1, scenario2],  # Only 2, need 3-5
                top_recommendation_id="option_1",
                final_reasoning="Test",
            )

    def test_max_scenarios_validation(self):
        """Test maximum 5 scenarios allowed."""
        hold = HoldOption(
            xp_gw1=63.2, xp_3gw=189.5, free_transfers_next_week=1, reasoning="Hold"
        )

        scenarios = [
            TransferScenario(
                scenario_id=f"option_{i}",
                transfers=[],
                xp_gw1=65.0,
                xp_gain_gw1=1.8,
                hit_cost=0,
                net_gain_gw1=1.8,
                xp_3gw=195.0,
                xp_gain_3gw=5.5,
                net_roi_3gw=5.5,
                reasoning=f"Option {i}",
                confidence="medium",
            )
            for i in range(1, 7)  # 6 scenarios
        ]

        # 6 scenarios - should fail (max is 5)
        with pytest.raises(ValueError):
            SingleGWRecommendation(
                target_gameweek=18,
                current_free_transfers=1,
                budget_available=2.3,
                hold_option=hold,
                recommended_scenarios=scenarios,
                top_recommendation_id="option_1",
                final_reasoning="Test",
            )

    def test_gameweek_validation(self):
        """Test gameweek must be 1-38."""
        hold = HoldOption(
            xp_gw1=63.2, xp_3gw=189.5, free_transfers_next_week=1, reasoning="Hold"
        )

        scenarios = [
            TransferScenario(
                scenario_id=f"option_{i}",
                transfers=[],
                xp_gw1=65.0,
                xp_gain_gw1=1.8,
                hit_cost=0,
                net_gain_gw1=1.8,
                xp_3gw=195.0,
                xp_gain_3gw=5.5,
                net_roi_3gw=5.5,
                reasoning=f"Option {i}",
                confidence="medium",
            )
            for i in range(1, 4)
        ]

        with pytest.raises(ValueError):
            SingleGWRecommendation(
                target_gameweek=39,  # Invalid (max is 38)
                current_free_transfers=1,
                budget_available=2.3,
                hold_option=hold,
                recommended_scenarios=scenarios,
                top_recommendation_id="option_1",
                final_reasoning="Test",
            )

    def test_budget_validation(self):
        """Test budget must be non-negative."""
        hold = HoldOption(
            xp_gw1=63.2, xp_3gw=189.5, free_transfers_next_week=1, reasoning="Hold"
        )

        scenarios = [
            TransferScenario(
                scenario_id=f"option_{i}",
                transfers=[],
                xp_gw1=65.0,
                xp_gain_gw1=1.8,
                hit_cost=0,
                net_gain_gw1=1.8,
                xp_3gw=195.0,
                xp_gain_3gw=5.5,
                net_roi_3gw=5.5,
                reasoning=f"Option {i}",
                confidence="medium",
            )
            for i in range(1, 4)
        ]

        with pytest.raises(ValueError):
            SingleGWRecommendation(
                target_gameweek=18,
                current_free_transfers=1,
                budget_available=-1.0,  # Negative not allowed
                hold_option=hold,
                recommended_scenarios=scenarios,
                top_recommendation_id="option_1",
                final_reasoning="Test",
            )
