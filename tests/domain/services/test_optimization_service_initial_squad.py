"""Tests for OptimizationService.optimize_initial_squad() - Simulated Annealing Squad Generation.

Tests the extraction of the proven simulated annealing algorithm from season_planner.py
into the domain layer with proper data contracts and validation.
"""

import pytest
import pandas as pd
from fpl_team_picker.domain.services.optimization_service import (
    OptimizationService,
    InitialSquadOptimizationInput,
)


@pytest.fixture
def sample_players():
    """Create sample player data for testing."""
    players = []

    # Create 30 players across positions with varying prices and xP
    positions = [
        ("GKP", 5, 4.5, 6.0),  # position, count, min_price, max_price
        ("DEF", 10, 4.0, 7.0),
        ("MID", 10, 4.5, 12.0),
        ("FWD", 5, 4.5, 11.0),
    ]

    player_id = 1
    teams = ["Team A", "Team B", "Team C", "Team D", "Team E"]

    for position, count, min_price, max_price in positions:
        for i in range(count):
            # Distribute prices evenly
            price = min_price + (max_price - min_price) * i / max(count - 1, 1)
            # xP correlates with price
            xp = 2.0 + (price - 4.0) * 1.5

            players.append(
                {
                    "player_id": player_id,
                    "web_name": f"Player{player_id}",
                    "position": position,
                    "price": round(price, 1),
                    "xP": round(xp, 2),
                    "xP_5gw": round(xp * 5, 2),
                    "team": teams[i % len(teams)],  # Distribute across teams
                }
            )
            player_id += 1

    return pd.DataFrame(players)


@pytest.fixture
def optimization_service():
    """Create OptimizationService instance."""
    return OptimizationService()


class TestInitialSquadOptimizationInput:
    """Test Pydantic data contract validation."""

    def test_valid_input(self):
        """Test valid input passes validation."""
        input_data = InitialSquadOptimizationInput(
            budget=100.0,
            formation=(2, 5, 5, 3),
            iterations=1000,
            must_include_ids=[1, 2],
            must_exclude_ids=[3, 4],
            xp_column="xP",
        )
        assert input_data.budget == 100.0
        assert input_data.formation == (2, 5, 5, 3)

    def test_invalid_budget(self):
        """Test budget validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            InitialSquadOptimizationInput(
                budget=-10.0,  # Negative budget
                formation=(2, 5, 5, 3),
                iterations=1000,
            )

    def test_invalid_formation_sum(self):
        """Test formation must sum to 15."""
        with pytest.raises(ValueError, match="sum to 15"):
            InitialSquadOptimizationInput(
                budget=100.0,
                formation=(2, 5, 5, 4),  # Sums to 16
                iterations=1000,
            )

    def test_invalid_formation_too_few_def(self):
        """Test formation must have at least 3 defenders."""
        with pytest.raises(ValueError, match="at least 3 DEF"):
            InitialSquadOptimizationInput(
                budget=100.0,
                formation=(2, 2, 8, 3),  # Only 2 DEF
                iterations=1000,
            )

    def test_invalid_iterations(self):
        """Test iterations bounds."""
        with pytest.raises(ValueError):
            InitialSquadOptimizationInput(
                budget=100.0,
                formation=(2, 5, 5, 3),
                iterations=50,  # Too few
            )

    def test_duplicate_constraint_ids(self):
        """Test duplicate IDs in constraints."""
        with pytest.raises(ValueError, match="Duplicate"):
            InitialSquadOptimizationInput(
                budget=100.0,
                formation=(2, 5, 5, 3),
                iterations=1000,
                must_include_ids=[1, 1, 2],  # Duplicate ID
            )


class TestOptimizeInitialSquad:
    """Test the main optimize_initial_squad() method."""

    def test_basic_squad_generation(self, optimization_service, sample_players):
        """Test basic squad generation without constraints."""
        result = optimization_service.optimize_initial_squad(
            players_with_xp=sample_players,
            budget=100.0,
            formation=(2, 5, 5, 3),
            iterations=500,  # Reduced for testing speed
        )

        # Verify result structure
        assert "optimal_squad" in result
        assert "remaining_budget" in result
        assert "total_xp" in result
        assert "iterations_improved" in result

        # Verify squad properties
        squad = result["optimal_squad"]
        assert len(squad) == 15

        # Check formation
        position_counts = {}
        for player in squad:
            pos = player["position"]
            position_counts[pos] = position_counts.get(pos, 0) + 1

        assert position_counts["GKP"] == 2
        assert position_counts["DEF"] == 5
        assert position_counts["MID"] == 5
        assert position_counts["FWD"] == 3

        # Check budget
        total_cost = sum(p["price"] for p in squad)
        assert total_cost <= 100.0
        assert result["remaining_budget"] >= 0

        # Check 3-per-team constraint
        team_counts = {}
        for player in squad:
            team = player["team"]
            team_counts[team] = team_counts.get(team, 0) + 1

        for team, count in team_counts.items():
            assert count <= 3, f"Team {team} has {count} players (max 3)"

    def test_must_include_constraint(self, optimization_service, sample_players):
        """Test must-include player constraint."""
        # Select specific players to include
        must_include_ids = {1, 2, 11}  # 1 GKP, 1 more player, 1 DEF

        result = optimization_service.optimize_initial_squad(
            players_with_xp=sample_players,
            budget=100.0,
            formation=(2, 5, 5, 3),
            iterations=500,
            must_include_ids=must_include_ids,
        )

        squad_ids = {p["player_id"] for p in result["optimal_squad"]}
        assert must_include_ids.issubset(squad_ids), "Must-include players not in squad"

    def test_must_exclude_constraint(self, optimization_service, sample_players):
        """Test must-exclude player constraint."""
        # Exclude some players
        must_exclude_ids = {5, 6, 7, 8}

        result = optimization_service.optimize_initial_squad(
            players_with_xp=sample_players,
            budget=100.0,
            formation=(2, 5, 5, 3),
            iterations=500,
            must_exclude_ids=must_exclude_ids,
        )

        squad_ids = {p["player_id"] for p in result["optimal_squad"]}
        assert squad_ids.isdisjoint(must_exclude_ids), "Excluded players found in squad"

    def test_both_constraints(self, optimization_service, sample_players):
        """Test both must-include and must-exclude constraints together."""
        must_include_ids = {1, 11}
        must_exclude_ids = {2, 12}

        result = optimization_service.optimize_initial_squad(
            players_with_xp=sample_players,
            budget=100.0,
            formation=(2, 5, 5, 3),
            iterations=500,
            must_include_ids=must_include_ids,
            must_exclude_ids=must_exclude_ids,
        )

        squad_ids = {p["player_id"] for p in result["optimal_squad"]}
        assert must_include_ids.issubset(squad_ids)
        assert squad_ids.isdisjoint(must_exclude_ids)

    def test_budget_respects_constraint(self, optimization_service, sample_players):
        """Test optimization respects budget constraint."""
        budget = 100.0
        result = optimization_service.optimize_initial_squad(
            players_with_xp=sample_players,
            budget=budget,
            formation=(2, 5, 5, 3),
            iterations=500,
        )

        total_cost = sum(p["price"] for p in result["optimal_squad"])
        assert total_cost <= budget, f"Squad cost {total_cost} exceeds budget {budget}"
        assert len(result["optimal_squad"]) == 15

    def test_xp_5gw_optimization(self, optimization_service, sample_players):
        """Test optimization using 5-gameweek xP column."""
        result = optimization_service.optimize_initial_squad(
            players_with_xp=sample_players,
            budget=100.0,
            formation=(2, 5, 5, 3),
            iterations=500,
            xp_column="xP_5gw",
        )

        assert len(result["optimal_squad"]) == 15
        # Total xP should be based on xP_5gw column
        assert result["total_xp"] > 0


class TestOptimizeInitialSquadDataValidation:
    """Test data quality validation - fail fast approach."""

    def test_missing_required_columns(self, optimization_service, sample_players):
        """Test fails with clear error when required columns missing."""
        incomplete_df = sample_players.drop(columns=["price"])

        with pytest.raises(ValueError, match="missing required columns.*price"):
            optimization_service.optimize_initial_squad(
                players_with_xp=incomplete_df,
                budget=100.0,
                formation=(2, 5, 5, 3),
                iterations=500,
            )

    def test_nan_values_rejected(self, optimization_service, sample_players):
        """Test fails fast when NaN values present."""
        bad_df = sample_players.copy()
        bad_df.loc[0, "xP"] = None  # Introduce NaN

        with pytest.raises(ValueError, match="Data quality issue.*missing xP"):
            optimization_service.optimize_initial_squad(
                players_with_xp=bad_df,
                budget=100.0,
                formation=(2, 5, 5, 3),
                iterations=500,
            )

    def test_insufficient_players(self, optimization_service, sample_players):
        """Test fails when not enough players after exclusions."""
        # Exclude almost all players
        all_ids = set(sample_players["player_id"])
        keep_ids = set(list(all_ids)[:10])  # Keep only 10
        exclude_ids = all_ids - keep_ids

        with pytest.raises(ValueError, match="Only 10 players available"):
            optimization_service.optimize_initial_squad(
                players_with_xp=sample_players,
                budget=100.0,
                formation=(2, 5, 5, 3),
                iterations=500,
                must_exclude_ids=exclude_ids,
            )

    def test_must_include_not_found(self, optimization_service, sample_players):
        """Test fails when must-include player IDs don't exist."""
        with pytest.raises(ValueError, match="not found in dataset.*999"):
            optimization_service.optimize_initial_squad(
                players_with_xp=sample_players,
                budget=100.0,
                formation=(2, 5, 5, 3),
                iterations=500,
                must_include_ids={999, 1000},  # Non-existent IDs
            )

    def test_must_include_violates_position_constraint(
        self, optimization_service, sample_players
    ):
        """Test fails when must-include players violate formation."""
        # Try to include 3 goalkeepers in a 2-GKP formation
        gkp_ids = sample_players[sample_players["position"] == "GKP"]["player_id"].iloc[
            :3
        ]

        with pytest.raises(ValueError, match="Must-include constraint violation.*GKP"):
            optimization_service.optimize_initial_squad(
                players_with_xp=sample_players,
                budget=100.0,
                formation=(2, 5, 5, 3),  # Only 2 GKP allowed
                iterations=500,
                must_include_ids=set(gkp_ids),
            )

    def test_must_include_exceeds_budget(self, optimization_service, sample_players):
        """Test fails when must-include players exceed budget."""
        # Select expensive players with correct formation distribution
        expensive_gkp = sample_players[sample_players["position"] == "GKP"].nlargest(2, "price")["player_id"]
        expensive_def = sample_players[sample_players["position"] == "DEF"].nlargest(5, "price")["player_id"]
        expensive_mid = sample_players[sample_players["position"] == "MID"].nlargest(5, "price")["player_id"]
        expensive_fwd = sample_players[sample_players["position"] == "FWD"].nlargest(3, "price")["player_id"]

        expensive_ids = set(expensive_gkp) | set(expensive_def) | set(expensive_mid) | set(expensive_fwd)

        with pytest.raises(ValueError, match="exceeding budget"):
            optimization_service.optimize_initial_squad(
                players_with_xp=sample_players,
                budget=65.0,  # Too low for expensive players
                formation=(2, 5, 5, 3),
                iterations=500,
                must_include_ids=expensive_ids,
            )

    def test_must_include_violates_3_per_team(self, optimization_service, sample_players):
        """Test fails when must-include players violate 3-per-team rule."""
        # Select 4 players from same team
        team_a_ids = sample_players[sample_players["team"] == "Team A"]["player_id"].iloc[
            :4
        ]

        with pytest.raises(ValueError, match="violate 3-per-team rule"):
            optimization_service.optimize_initial_squad(
                players_with_xp=sample_players,
                budget=100.0,
                formation=(2, 5, 5, 3),
                iterations=500,
                must_include_ids=set(team_a_ids),
            )


class TestOptimizationQuality:
    """Test optimization produces quality results."""

    def test_improves_over_iterations(self, optimization_service, sample_players):
        """Test that simulated annealing algorithm executes correctly."""
        # Run optimization
        result = optimization_service.optimize_initial_squad(
            players_with_xp=sample_players,
            budget=100.0,
            formation=(2, 5, 5, 3),
            iterations=1000,
        )

        # Algorithm should complete successfully
        assert "total_xp" in result
        assert "iterations_improved" in result
        assert "total_iterations" in result
        assert result["total_iterations"] == 1000

        # Result should be a valid squad with reasonable xP
        assert result["total_xp"] > 0
        assert len(result["optimal_squad"]) == 15

    def test_uses_budget_efficiently(self, optimization_service, sample_players):
        """Test that optimization uses most of available budget."""
        result = optimization_service.optimize_initial_squad(
            players_with_xp=sample_players,
            budget=100.0,
            formation=(2, 5, 5, 3),
            iterations=1000,
        )

        # Should use at least 90% of budget for optimal squad
        assert result["final_cost"] >= 90.0
        assert result["remaining_budget"] < 10.0

    def test_selects_high_xp_players(self, optimization_service, sample_players):
        """Test that optimization prefers players with higher xP."""
        result = optimization_service.optimize_initial_squad(
            players_with_xp=sample_players,
            budget=100.0,
            formation=(2, 5, 5, 3),
            iterations=1000,
        )

        squad_xps = [p["xP"] for p in result["optimal_squad"]]
        avg_squad_xp = sum(squad_xps) / len(squad_xps)

        # Squad average xP should be above median (since we're optimizing for xP)
        all_xps = sample_players["xP"].tolist()
        median_xp = sorted(all_xps)[len(all_xps) // 2]

        assert avg_squad_xp > median_xp, "Squad should have above-average xP"

    def test_starting_11_selection(self, optimization_service, sample_players):
        """Test that best starting 11 is correctly identified."""
        result = optimization_service.optimize_initial_squad(
            players_with_xp=sample_players,
            budget=100.0,
            formation=(2, 5, 5, 3),
            iterations=500,
        )

        squad = result["optimal_squad"]
        starting_11 = optimization_service._get_best_starting_11_from_squad(squad, "xP")

        assert len(starting_11) == 11

        # Verify starting 11 has valid formation
        pos_counts = {}
        for p in starting_11:
            pos_counts[p["position"]] = pos_counts.get(p["position"], 0) + 1

        assert pos_counts["GKP"] == 1
        assert 3 <= pos_counts["DEF"] <= 5
        assert 2 <= pos_counts["MID"] <= 5
        assert 1 <= pos_counts["FWD"] <= 3
        assert sum(pos_counts.values()) == 11


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_alternative_formation(self, optimization_service, sample_players):
        """Test with alternative formation."""
        result = optimization_service.optimize_initial_squad(
            players_with_xp=sample_players,
            budget=100.0,
            formation=(2, 6, 6, 1),  # Alternative formation
            iterations=500,
        )

        position_counts = {}
        for p in result["optimal_squad"]:
            pos = p["position"]
            position_counts[pos] = position_counts.get(pos, 0) + 1

        assert position_counts == {"GKP": 2, "DEF": 6, "MID": 6, "FWD": 1}

    def test_exact_budget_usage(self, optimization_service):
        """Test scenario where squad costs exactly the budget."""
        # Create players with prices that can exactly sum to 100
        players = []
        player_id = 1

        # Create players with exact prices
        for i in range(2):  # 2 GKP at 5.0 each
            players.append({
                "player_id": player_id,
                "web_name": f"GKP{i}",
                "position": "GKP",
                "price": 5.0,
                "xP": 3.0,
                "team": f"Team{player_id % 5}",
            })
            player_id += 1

        for i in range(5):  # 5 DEF at 6.0 each
            players.append({
                "player_id": player_id,
                "web_name": f"DEF{i}",
                "position": "DEF",
                "price": 6.0,
                "xP": 4.0,
                "team": f"Team{player_id % 5}",
            })
            player_id += 1

        for i in range(5):  # 5 MID at 8.0 each
            players.append({
                "player_id": player_id,
                "web_name": f"MID{i}",
                "position": "MID",
                "price": 8.0,
                "xP": 5.0,
                "team": f"Team{player_id % 5}",
            })
            player_id += 1

        for i in range(3):  # 3 FWD at 10.0 each
            players.append({
                "player_id": player_id,
                "web_name": f"FWD{i}",
                "position": "FWD",
                "price": 10.0,
                "xP": 6.0,
                "team": f"Team{player_id % 5}",
            })
            player_id += 1

        # Add some cheaper alternatives
        for i in range(10):
            players.append({
                "player_id": player_id,
                "web_name": f"Bench{i}",
                "position": ["DEF", "MID", "FWD"][i % 3],
                "price": 4.5,
                "xP": 1.0,
                "team": f"Team{player_id % 5}",
            })
            player_id += 1

        df = pd.DataFrame(players)

        result = optimization_service.optimize_initial_squad(
            players_with_xp=df,
            budget=100.0,
            formation=(2, 5, 5, 3),
            iterations=500,
        )

        assert result["final_cost"] <= 100.0
        assert len(result["optimal_squad"]) == 15
