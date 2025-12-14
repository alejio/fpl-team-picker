"""Unit tests for Squad Selection functionality.

Tests for:
- find_optimal_starting_11 with various formations
- find_bench_players ordering
- calculate_budget_pool with constraints
- Edge cases (unavailable players, insufficient players)
"""

import pytest
import pandas as pd
from fpl_team_picker.domain.services.optimization_service import OptimizationService


@pytest.fixture
def sample_squad():
    """Create a sample squad of 15 players."""
    return pd.DataFrame(
        [
            # 2 GKP
            {"player_id": 1, "web_name": "GKP1", "position": "GKP", "xP": 5.0, "status": "a"},
            {"player_id": 2, "web_name": "GKP2", "position": "GKP", "xP": 4.5, "status": "a"},
            # 5 DEF
            {"player_id": 3, "web_name": "DEF1", "position": "DEF", "xP": 6.5, "status": "a"},
            {"player_id": 4, "web_name": "DEF2", "position": "DEF", "xP": 6.0, "status": "a"},
            {"player_id": 5, "web_name": "DEF3", "position": "DEF", "xP": 5.5, "status": "a"},
            {"player_id": 6, "web_name": "DEF4", "position": "DEF", "xP": 5.0, "status": "a"},
            {"player_id": 7, "web_name": "DEF5", "position": "DEF", "xP": 4.5, "status": "a"},
            # 5 MID
            {"player_id": 8, "web_name": "MID1", "position": "MID", "xP": 8.0, "status": "a"},
            {"player_id": 9, "web_name": "MID2", "position": "MID", "xP": 7.5, "status": "a"},
            {"player_id": 10, "web_name": "MID3", "position": "MID", "xP": 7.0, "status": "a"},
            {"player_id": 11, "web_name": "MID4", "position": "MID", "xP": 6.5, "status": "a"},
            {"player_id": 12, "web_name": "MID5", "position": "MID", "xP": 6.0, "status": "a"},
            # 3 FWD
            {"player_id": 13, "web_name": "FWD1", "position": "FWD", "xP": 9.0, "status": "a"},
            {"player_id": 14, "web_name": "FWD2", "position": "FWD", "xP": 8.5, "status": "a"},
            {"player_id": 15, "web_name": "FWD3", "position": "FWD", "xP": 8.0, "status": "a"},
        ]
    )


@pytest.fixture
def service():
    """Create OptimizationService instance."""
    return OptimizationService()


class TestFindOptimalStarting11:
    """Test find_optimal_starting_11 method."""

    def test_find_optimal_starting_11_basic(self, service, sample_squad):
        """Test basic starting 11 selection."""
        starting_11, formation, total_xp = service.find_optimal_starting_11(
            sample_squad
        )

        assert len(starting_11) == 11
        assert formation in ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"]
        assert total_xp > 0
        # Should have exactly 1 GKP
        gkp_count = sum(1 for p in starting_11 if p["position"] == "GKP")
        assert gkp_count == 1

    def test_find_optimal_starting_11_with_xp_column(self, service, sample_squad):
        """Test starting 11 selection with custom xP column."""
        sample_squad["xP_3gw"] = sample_squad["xP"] * 1.1

        starting_11, formation, total_xp = service.find_optimal_starting_11(
            sample_squad, xp_column="xP_3gw"
        )

        assert len(starting_11) == 11
        assert total_xp > 0

    def test_find_optimal_starting_11_excludes_unavailable(self, service, sample_squad):
        """Test that unavailable players are excluded."""
        # Mark one player as injured
        sample_squad.loc[sample_squad["player_id"] == 8, "status"] = "i"

        starting_11, formation, total_xp = service.find_optimal_starting_11(
            sample_squad
        )

        assert len(starting_11) == 11
        # Injured player should not be in starting 11
        starting_11_ids = {p["player_id"] for p in starting_11}
        assert 8 not in starting_11_ids

    def test_find_optimal_starting_11_excludes_suspended(self, service, sample_squad):
        """Test that suspended players are excluded."""
        # Mark one player as suspended
        sample_squad.loc[sample_squad["player_id"] == 13, "status"] = "s"

        starting_11, formation, total_xp = service.find_optimal_starting_11(
            sample_squad
        )

        starting_11_ids = {p["player_id"] for p in starting_11}
        assert 13 not in starting_11_ids

    def test_find_optimal_starting_11_insufficient_players(self, service):
        """Test handling when squad has fewer than 11 players."""
        small_squad = pd.DataFrame(
            [
                {"player_id": 1, "position": "GKP", "xP": 5.0, "status": "a"},
                {"player_id": 2, "position": "DEF", "xP": 6.0, "status": "a"},
            ]
        )

        starting_11, formation, total_xp = service.find_optimal_starting_11(
            small_squad
        )

        assert starting_11 == []
        assert formation == ""
        assert total_xp == 0

    def test_find_optimal_starting_11_empty_squad(self, service):
        """Test handling of empty squad."""
        empty_squad = pd.DataFrame()

        starting_11, formation, total_xp = service.find_optimal_starting_11(
            empty_squad
        )

        assert starting_11 == []
        assert formation == ""
        assert total_xp == 0

    def test_find_optimal_starting_11_fallback_xp_column(self, service, sample_squad):
        """Test fallback to xP_3gw or xP_5gw if specified column missing."""
        # Remove xP column, add xP_3gw
        sample_squad = sample_squad.drop(columns=["xP"])
        sample_squad["xP_3gw"] = [5.0] * len(sample_squad)

        starting_11, formation, total_xp = service.find_optimal_starting_11(
            sample_squad, xp_column="xP"
        )

        # Should fallback to xP_3gw
        assert len(starting_11) == 11


class TestFindBenchPlayers:
    """Test find_bench_players method."""

    def test_find_bench_players_basic(self, service, sample_squad):
        """Test basic bench player selection."""
        starting_11, _, _ = service.find_optimal_starting_11(sample_squad)
        bench = service.find_bench_players(sample_squad, starting_11)

        assert len(bench) == 4
        # Bench players should not be in starting 11
        starting_11_ids = {p["player_id"] for p in starting_11}
        bench_ids = {p["player_id"] for p in bench}
        assert bench_ids.isdisjoint(starting_11_ids)

    def test_find_bench_players_ordered_by_xp(self, service, sample_squad):
        """Test that bench players are ordered by xP (highest first)."""
        starting_11, _, _ = service.find_optimal_starting_11(sample_squad)
        bench = service.find_bench_players(sample_squad, starting_11)

        if len(bench) > 1:
            xp_values = [p.get("xP", 0) for p in bench]
            assert xp_values == sorted(xp_values, reverse=True)

    def test_find_bench_players_with_xp_column(self, service, sample_squad):
        """Test bench selection with custom xP column."""
        sample_squad["xP_3gw"] = sample_squad["xP"] * 1.1
        starting_11, _, _ = service.find_optimal_starting_11(sample_squad)
        bench = service.find_bench_players(sample_squad, starting_11, xp_column="xP_3gw")

        assert len(bench) <= 4

    def test_find_bench_players_insufficient_squad(self, service):
        """Test bench selection with squad smaller than 15."""
        small_squad = pd.DataFrame(
            [
                {"player_id": i, "position": "DEF", "xP": 5.0, "status": "a"}
                for i in range(12)
            ]
        )
        starting_11 = [{"player_id": i} for i in range(11)]
        bench = service.find_bench_players(small_squad, starting_11)

        assert len(bench) <= 1  # Only 1 player left (or 0 if squad too small)

    def test_find_bench_players_empty_squad(self, service):
        """Test bench selection with empty squad."""
        empty_squad = pd.DataFrame()
        starting_11 = []
        bench = service.find_bench_players(empty_squad, starting_11)

        assert bench == []

    def test_find_bench_players_max_4(self, service, sample_squad):
        """Test that bench is limited to 4 players."""
        # Add extra players to squad
        extra_players = pd.DataFrame(
            [
                {"player_id": 16, "position": "DEF", "xP": 3.0, "status": "a"},
                {"player_id": 17, "position": "MID", "xP": 3.5, "status": "a"},
            ]
        )
        large_squad = pd.concat([sample_squad, extra_players])

        starting_11, _, _ = service.find_optimal_starting_11(large_squad)
        bench = service.find_bench_players(large_squad, starting_11)

        assert len(bench) == 4


class TestCalculateBudgetPool:
    """Test calculate_budget_pool method."""

    def test_calculate_budget_pool_basic(self, service, sample_squad):
        """Test basic budget pool calculation."""
        sample_squad["price"] = [5.0] * len(sample_squad)

        budget_pool = service.calculate_budget_pool(
            current_squad=sample_squad, bank_balance=1.0
        )

        assert budget_pool["bank_balance"] == 1.0
        assert budget_pool["sellable_value"] > 0
        assert budget_pool["total_budget"] == budget_pool["bank_balance"] + budget_pool["sellable_value"]
        assert budget_pool["sellable_players"] == 15

    def test_calculate_budget_pool_with_players_to_keep(self, service, sample_squad):
        """Test budget pool with players that must be kept."""
        sample_squad["price"] = [5.0] * len(sample_squad)
        players_to_keep = {1, 2, 3}  # Keep first 3 players

        budget_pool = service.calculate_budget_pool(
            current_squad=sample_squad,
            bank_balance=1.0,
            players_to_keep=players_to_keep,
        )

        assert budget_pool["players_to_keep"] == 3
        assert budget_pool["sellable_players"] == 12
        # Sellable value should be less (3 players not sellable)
        assert budget_pool["sellable_value"] < 15 * 5.0

    def test_calculate_budget_pool_empty_squad(self, service):
        """Test budget pool calculation with empty squad."""
        empty_squad = pd.DataFrame()

        budget_pool = service.calculate_budget_pool(
            current_squad=empty_squad, bank_balance=5.0
        )

        assert budget_pool["bank_balance"] == 5.0
        assert budget_pool["sellable_value"] == 0.0
        assert budget_pool["total_budget"] == 5.0
        assert budget_pool["sellable_players"] == 0

    def test_calculate_budget_pool_no_players_to_keep(self, service, sample_squad):
        """Test budget pool with no keep constraints."""
        sample_squad["price"] = [5.0] * len(sample_squad)

        budget_pool = service.calculate_budget_pool(
            current_squad=sample_squad, bank_balance=1.0, players_to_keep=None
        )

        assert budget_pool["sellable_players"] == 15
        assert budget_pool["players_to_keep"] == 0
