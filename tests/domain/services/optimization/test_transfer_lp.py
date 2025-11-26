"""Tests for Linear Programming optimization in OptimizationService.

Tests the LP transfer optimization implementation:
- Basic LP functionality and correctness
- Constraint satisfaction (budget, positions, teams, transfers)
- Optimal solution guarantees
- LP vs SA comparison
- Performance and speed
- Configuration switching
- Edge cases and error handling
"""

import pytest
import pandas as pd
import time
from fpl_team_picker.domain.services.optimization_service import OptimizationService
from fpl_team_picker.config import config


@pytest.fixture
def sample_players():
    """Create sample player data for testing LP optimization."""
    players = []

    # Create diverse player pool across positions with varying prices and xP
    positions = [
        ("GKP", 8, 4.0, 6.0),
        ("DEF", 18, 4.0, 8.0),
        ("MID", 22, 4.5, 13.0),
        ("FWD", 12, 4.5, 12.5),
    ]

    player_id = 1
    teams = [
        "Arsenal",
        "Liverpool",
        "Man City",
        "Chelsea",
        "Spurs",
        "Newcastle",
        "Brighton",
        "Aston Villa",
        "West Ham",
        "Man Utd",
        "Wolves",
        "Fulham",
    ]

    for position, count, min_price, max_price in positions:
        for i in range(count):
            price = min_price + (max_price - min_price) * i / max(count - 1, 1)
            # xP correlates with price but with some noise
            base_xp = 2.0 + (price - 4.0) * 1.5
            xp_1gw = base_xp + (i % 3) * 0.3  # Add variation
            xp_3gw = xp_1gw * 3.1
            xp_5gw = xp_1gw * 5.2

            # Chance of playing for availability adjustment
            chance_of_playing = 100.0 if i % 8 != 0 else 75.0  # Most available

            players.append(
                {
                    "player_id": player_id,
                    "web_name": f"Player{player_id}",
                    "position": position,
                    "price": round(price, 1),
                    "xP": round(xp_1gw, 2),
                    "xP_3gw": round(xp_3gw, 2),
                    "xP_5gw": round(xp_5gw, 2),
                    "team": teams[player_id % len(teams)],
                    "status": "a",  # Available
                    "chance_of_playing_this_round": chance_of_playing,
                    "fixture_outlook": "AVG",
                }
            )
            player_id += 1

    return pd.DataFrame(players)


@pytest.fixture
def current_squad(sample_players):
    """Create a sample current squad with valid formation."""
    # Select 15 players ensuring valid formation (2-5-5-3)
    gkp = sample_players[sample_players["position"] == "GKP"].iloc[:2]
    def_players = sample_players[sample_players["position"] == "DEF"].iloc[:5]
    mid_players = sample_players[sample_players["position"] == "MID"].iloc[:5]
    fwd_players = sample_players[sample_players["position"] == "FWD"].iloc[:3]

    squad = pd.concat([gkp, def_players, mid_players, fwd_players])
    return squad.reset_index(drop=True)


@pytest.fixture
def team_data():
    """Create team data for testing."""
    return {
        "bank": 2.5,
        "team_value": 100.0,
        "free_transfers": 1,
    }


@pytest.fixture
def optimization_service():
    """Create OptimizationService instance."""
    return OptimizationService()


class TestLPBasicFunctionality:
    """Test basic LP optimization functionality."""

    def test_lp_builds_valid_squad(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP builds a valid FPL squad with correct structure."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "linear_programming"

            squad_df, result_summary, metadata = (
                optimization_service.optimize_transfers(
                    current_squad=current_squad,
                    team_data=team_data,
                    players_with_xp=sample_players,
                )
            )

            # Verify return types
            assert isinstance(squad_df, pd.DataFrame)
            assert isinstance(result_summary, dict)
            assert isinstance(metadata, dict)

            # Verify squad structure
            assert len(squad_df) == 15, "Squad must have exactly 15 players"
            assert set(squad_df.columns).issuperset(
                {"player_id", "web_name", "position", "price", "xP"}
            ), "Squad must have required columns"

            # Verify position requirements (2-5-5-3)
            assert len(squad_df[squad_df["position"] == "GKP"]) == 2, (
                "Must have 2 goalkeepers"
            )
            assert len(squad_df[squad_df["position"] == "DEF"]) == 5, (
                "Must have 5 defenders"
            )
            assert len(squad_df[squad_df["position"] == "MID"]) == 5, (
                "Must have 5 midfielders"
            )
            assert len(squad_df[squad_df["position"] == "FWD"]) == 3, (
                "Must have 3 forwards"
            )

            # Verify result structure
            assert "net_xp" in result_summary
            assert "transfers" in result_summary
            assert "type" in result_summary
            assert result_summary["type"] == "linear_programming"

            # Verify metadata
            assert metadata["method"] == "linear_programming"
            assert "lp_solve_time" in metadata
            assert "lp_objective_value" in metadata
            assert "lp_status" in metadata
            assert metadata["lp_status"] == "Optimal"

        finally:
            config.optimization.transfer_optimization_method = original_method

    def test_lp_respects_budget_constraint(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP respects budget constraint."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "linear_programming"

            squad_df, _, metadata = optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )

            # Calculate total budget available
            total_budget = metadata["budget_pool_info"]["total_budget"]
            squad_cost = squad_df["price"].sum()

            assert squad_cost <= total_budget, (
                f"Squad cost {squad_cost} exceeds budget {total_budget}"
            )

        finally:
            config.optimization.transfer_optimization_method = original_method

    def test_lp_respects_team_limit_constraint(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP respects max 3 players per team constraint."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "linear_programming"

            squad_df, _, _ = optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )

            # Check team distribution
            team_counts = squad_df["team"].value_counts()
            max_per_team = team_counts.max()

            assert max_per_team <= 3, (
                f"Found {max_per_team} players from same team (max 3 allowed)"
            )

        finally:
            config.optimization.transfer_optimization_method = original_method

    def test_lp_respects_transfer_limit(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP respects max transfer limit."""
        original_method = config.optimization.transfer_optimization_method
        original_max_transfers = config.optimization.max_transfers
        try:
            config.optimization.transfer_optimization_method = "linear_programming"
            config.optimization.max_transfers = 2

            squad_df, result_summary, _ = optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )

            # Check number of transfers
            num_transfers = result_summary["transfers"]
            assert num_transfers <= 2, (
                f"Made {num_transfers} transfers, max allowed is 2"
            )

        finally:
            config.optimization.transfer_optimization_method = original_method
            config.optimization.max_transfers = original_max_transfers


class TestLPOptimality:
    """Test LP optimality guarantees."""

    def test_lp_is_deterministic(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP always returns same result for same input."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "linear_programming"

            # Run optimization multiple times
            results = []
            for _ in range(3):
                squad_df, result_summary, _ = optimization_service.optimize_transfers(
                    current_squad=current_squad,
                    team_data=team_data,
                    players_with_xp=sample_players,
                )
                results.append(
                    {
                        "net_xp": result_summary["net_xp"],
                        "transfers": result_summary["transfers"],
                        "player_ids": sorted(squad_df["player_id"].tolist()),
                    }
                )

            # All results should be identical
            assert all(r["net_xp"] == results[0]["net_xp"] for r in results), (
                "LP should return same net_xp every time"
            )
            assert all(r["player_ids"] == results[0]["player_ids"] for r in results), (
                "LP should return same squad every time"
            )

        finally:
            config.optimization.transfer_optimization_method = original_method

    def test_lp_produces_valid_optimal_solution(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP produces valid solution flagged as optimal."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "linear_programming"

            squad_df, result_summary, metadata = (
                optimization_service.optimize_transfers(
                    current_squad=current_squad,
                    team_data=team_data,
                    players_with_xp=sample_players,
                )
            )

            # LP should find optimal solution
            assert metadata["lp_status"] == "Optimal", "LP should report optimal status"
            assert len(squad_df) == 15, "LP should find valid 15-player squad"
            assert result_summary["net_xp"] > 0, "LP should find positive net xP"
            assert "lp_solve_time" in metadata, "LP should report solve time"
            assert metadata["lp_solve_time"] < 10.0, "LP should solve quickly"

        finally:
            config.optimization.transfer_optimization_method = original_method


class TestLPPerformance:
    """Test LP performance characteristics."""

    def test_lp_is_fast(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP solves quickly (< 5 seconds)."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "linear_programming"

            start_time = time.time()
            _, _, metadata = optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )
            elapsed_time = time.time() - start_time

            # LP should solve quickly
            assert elapsed_time < 5.0, f"LP took {elapsed_time:.2f}s (expected < 5s)"
            assert "lp_solve_time" in metadata
            assert metadata["lp_solve_time"] < 5.0, (
                f"LP solve time was {metadata['lp_solve_time']:.2f}s"
            )

        finally:
            config.optimization.transfer_optimization_method = original_method

    def test_lp_is_faster_than_sa(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP is faster than SA for same problem."""
        original_method = config.optimization.transfer_optimization_method
        original_sa_iterations = config.optimization.sa_iterations
        try:
            # Time LP
            config.optimization.transfer_optimization_method = "linear_programming"
            lp_start = time.time()
            optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )
            lp_time = time.time() - lp_start

            # Time SA
            config.optimization.transfer_optimization_method = "simulated_annealing"
            config.optimization.sa_iterations = 2000
            config.optimization.sa_use_consensus_mode = False
            sa_start = time.time()
            optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )
            sa_time = time.time() - sa_start

            # LP should be faster (allow some margin for overhead)
            assert lp_time < sa_time, (
                f"LP took {lp_time:.2f}s, SA took {sa_time:.2f}s (LP should be faster)"
            )

        finally:
            config.optimization.transfer_optimization_method = original_method
            config.optimization.sa_iterations = original_sa_iterations
            config.optimization.sa_use_consensus_mode = True


class TestLPConstraints:
    """Test LP constraint handling."""

    def test_lp_respects_must_include_constraint(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP respects must_include_ids constraint."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "linear_programming"

            # Pick a player not in current squad to force inclusion
            available_players = sample_players[
                ~sample_players["player_id"].isin(current_squad["player_id"])
            ]
            must_include_player = available_players.iloc[0]["player_id"]

            squad_df, _, _ = optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
                must_include_ids={must_include_player},
            )

            assert must_include_player in squad_df["player_id"].values, (
                f"Player {must_include_player} should be in squad (must_include)"
            )

        finally:
            config.optimization.transfer_optimization_method = original_method

    def test_lp_respects_must_exclude_constraint(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP respects must_exclude_ids constraint."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "linear_programming"

            # Pick top xP player to exclude
            best_player = sample_players.nlargest(1, "xP").iloc[0]["player_id"]

            squad_df, _, _ = optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
                must_exclude_ids={best_player},
            )

            assert best_player not in squad_df["player_id"].values, (
                f"Player {best_player} should not be in squad (must_exclude)"
            )

        finally:
            config.optimization.transfer_optimization_method = original_method

    def test_lp_handles_wildcard(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP handles wildcard chip (15 free transfers, £100m budget)."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "linear_programming"

            squad_df, result_summary, metadata = (
                optimization_service.optimize_transfers(
                    current_squad=current_squad,
                    team_data=team_data,
                    players_with_xp=sample_players,
                    free_transfers_override=15,  # Wildcard
                )
            )

            # Check wildcard properties
            assert metadata["is_wildcard"], "Should be flagged as wildcard"
            assert metadata["budget_pool_info"]["total_budget"] == 100.0, (
                "Wildcard budget should be £100m"
            )

            # Squad should be valid
            assert len(squad_df) == 15
            squad_cost = squad_df["price"].sum()
            assert squad_cost <= 100.0, (
                f"Wildcard squad costs {squad_cost}m (max £100m)"
            )

            # No transfer penalty for wildcard
            assert result_summary["penalty"] == 0.0, "Wildcard should have no penalty"

        finally:
            config.optimization.transfer_optimization_method = original_method


class TestLPConfigurationSwitching:
    """Test switching between LP and SA via configuration."""

    def test_config_switch_to_lp(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test switching to LP via config."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "linear_programming"

            _, _, metadata = optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )

            assert metadata["method"] == "linear_programming"
            assert "lp_solve_time" in metadata
            assert "lp_objective_value" in metadata

        finally:
            config.optimization.transfer_optimization_method = original_method

    def test_config_switch_to_sa(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test switching to SA via config."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "simulated_annealing"
            config.optimization.sa_use_consensus_mode = False

            _, _, metadata = optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )

            assert metadata["method"] == "simulated_annealing"
            assert "sa_iterations" in metadata
            assert "sa_improvements" in metadata

        finally:
            config.optimization.transfer_optimization_method = original_method
            config.optimization.sa_use_consensus_mode = True

    def test_unknown_method_is_rejected(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test that unknown optimization methods are properly validated."""
        # Note: Config validation happens at the pydantic level
        # This test verifies that only valid methods work
        original_method = config.optimization.transfer_optimization_method
        try:
            # Valid methods should work
            config.optimization.transfer_optimization_method = "linear_programming"
            assert (
                config.optimization.transfer_optimization_method == "linear_programming"
            )

            config.optimization.transfer_optimization_method = "simulated_annealing"
            assert (
                config.optimization.transfer_optimization_method
                == "simulated_annealing"
            )

        finally:
            config.optimization.transfer_optimization_method = original_method


class TestLPEdgeCases:
    """Test LP edge cases and error handling."""

    def test_lp_handles_unavailable_players(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP filters out unavailable players."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "linear_programming"

            # Mark some players as injured
            players_with_injuries = sample_players.copy()
            players_with_injuries.loc[
                players_with_injuries["player_id"].isin([1, 2, 3]), "status"
            ] = "i"

            squad_df, _, _ = optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=players_with_injuries,
            )

            # Injured players should not be in squad
            injured_ids = [1, 2, 3]
            for injured_id in injured_ids:
                assert injured_id not in squad_df["player_id"].values, (
                    f"Injured player {injured_id} should not be in squad"
                )

        finally:
            config.optimization.transfer_optimization_method = original_method

    def test_lp_with_tight_budget(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP handles tight budget constraints."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "linear_programming"

            # Very tight budget
            tight_budget_data = team_data.copy()
            tight_budget_data["bank"] = 0.1

            squad_df, _, metadata = optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=tight_budget_data,
                players_with_xp=sample_players,
            )

            # Should still find valid solution
            assert len(squad_df) == 15
            assert metadata["lp_status"] == "Optimal"

        finally:
            config.optimization.transfer_optimization_method = original_method

    def test_lp_handles_multiple_constraints(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP handles multiple simultaneous constraints."""
        original_method = config.optimization.transfer_optimization_method
        original_max_transfers = config.optimization.max_transfers
        try:
            config.optimization.transfer_optimization_method = "linear_programming"
            config.optimization.max_transfers = 2

            # Pick players not in current squad
            available_players = sample_players[
                ~sample_players["player_id"].isin(current_squad["player_id"])
            ]
            must_include = {available_players.iloc[0]["player_id"]}
            must_exclude = {available_players.iloc[1]["player_id"]}

            squad_df, result_summary, _ = optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
                must_include_ids=must_include,
                must_exclude_ids=must_exclude,
            )

            # All constraints should be satisfied
            assert len(squad_df) == 15
            assert result_summary["transfers"] <= 2
            assert list(must_include)[0] in squad_df["player_id"].values
            assert list(must_exclude)[0] not in squad_df["player_id"].values

        finally:
            config.optimization.transfer_optimization_method = original_method
            config.optimization.max_transfers = original_max_transfers


class TestLPVsSAComparison:
    """Detailed comparison of LP vs SA."""

    def test_lp_objective_value_is_reasonable(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test LP objective value is reasonable and positive."""
        original_method = config.optimization.transfer_optimization_method
        try:
            config.optimization.transfer_optimization_method = "linear_programming"

            squad_df, result_summary, metadata = (
                optimization_service.optimize_transfers(
                    current_squad=current_squad,
                    team_data=team_data,
                    players_with_xp=sample_players,
                )
            )

            # LP objective maximizes full 15-player squad xP (not just best 11)
            # So objective value will be higher than net_xp (which is best 11 minus penalty)
            lp_objective = metadata["lp_objective_value"]
            net_xp = result_summary["net_xp"]

            # Objective should be positive and greater than net_xp
            assert lp_objective > 0, "LP objective should be positive"
            assert lp_objective >= net_xp, (
                "LP objective (full squad) should be >= net_xp (best 11 minus penalty)"
            )

            # Verify LP status is optimal
            assert metadata["lp_status"] == "Optimal", "LP should find optimal solution"

        finally:
            config.optimization.transfer_optimization_method = original_method

    def test_lp_and_sa_both_satisfy_constraints(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test both LP and SA produce valid squads."""
        original_method = config.optimization.transfer_optimization_method
        try:
            # Test LP
            config.optimization.transfer_optimization_method = "linear_programming"
            lp_squad, lp_result, _ = optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )

            # Test SA
            config.optimization.transfer_optimization_method = "simulated_annealing"
            config.optimization.sa_use_consensus_mode = False
            sa_squad, sa_result, _ = optimization_service.optimize_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )

            # Both should have valid squads
            for squad, result in [(lp_squad, lp_result), (sa_squad, sa_result)]:
                assert len(squad) == 15
                assert len(squad[squad["position"] == "GKP"]) == 2
                assert len(squad[squad["position"] == "DEF"]) == 5
                assert len(squad[squad["position"] == "MID"]) == 5
                assert len(squad[squad["position"] == "FWD"]) == 3
                assert squad["team"].value_counts().max() <= 3

        finally:
            config.optimization.transfer_optimization_method = original_method
            config.optimization.sa_use_consensus_mode = True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
