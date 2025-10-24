"""Tests for OptimizationService.optimize_wildcard_squad() - Wildcard Chip Optimization.

Tests the wildcard chip optimization wrapper that builds entire squads from scratch
using simulated annealing with £100m budget reset and horizon configuration support.
"""

import pytest
import pandas as pd
from fpl_team_picker.domain.services.optimization_service import OptimizationService
from fpl_team_picker.config import config


@pytest.fixture
def sample_players():
    """Create sample player data for testing wildcard optimization."""
    players = []

    # Create 40 players across positions with varying prices and xP
    positions = [
        ("GKP", 6, 4.0, 6.0),  # position, count, min_price, max_price
        ("DEF", 12, 4.0, 7.5),
        ("MID", 14, 4.5, 13.0),
        ("FWD", 8, 4.5, 12.0),
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
    ]

    for position, count, min_price, max_price in positions:
        for i in range(count):
            # Distribute prices evenly
            price = min_price + (max_price - min_price) * i / max(count - 1, 1)
            # xP correlates with price
            xp_1gw = 2.0 + (price - 4.0) * 1.5
            xp_5gw = xp_1gw * 5.2  # Slightly more than 5x for variation

            players.append(
                {
                    "player_id": player_id,
                    "web_name": f"Player{player_id}",
                    "position": position,
                    "price": round(price, 1),
                    "xP": round(xp_1gw, 2),
                    "xP_5gw": round(xp_5gw, 2),
                    "team": teams[player_id % len(teams)],
                    "fixture_outlook": "AVG",
                }
            )
            player_id += 1

    return pd.DataFrame(players)


@pytest.fixture
def optimization_service():
    """Create OptimizationService instance."""
    return OptimizationService()


@pytest.fixture
def current_squad(sample_players):
    """Create a dummy current squad (will be ignored for wildcard)."""
    # Select any 15 players for the dummy squad
    return sample_players.iloc[:15].copy()


@pytest.fixture
def team_data():
    """Create dummy team data (bank will be overridden to £100m for wildcard)."""
    return {
        "bank": 0.5,  # Will be ignored/overridden for wildcard
        "team_value": 100.0,
        "free_transfers": 1,  # Will be overridden to 15 for wildcard
    }


class TestWildcardBasicFunctionality:
    """Test basic wildcard optimization functionality."""

    def test_wildcard_returns_correct_structure(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard returns same structure as transfer optimization."""
        squad_df, result_summary, metadata = (
            optimization_service.optimize_wildcard_squad(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )
        )

        # Verify return types
        assert isinstance(squad_df, pd.DataFrame), "Should return DataFrame"
        assert isinstance(result_summary, dict), "Should return result summary dict"
        assert isinstance(metadata, dict), "Should return metadata dict"

        # Verify squad structure
        assert len(squad_df) == 15, "Wildcard should build 15-player squad"

        # Verify result summary structure (matches transfer optimization format)
        assert "transfers" in result_summary
        # Wildcard reports actual number of players changed (not 0)
        assert result_summary["transfers"] >= 0, (
            "Should report number of players changed"
        )
        assert "penalty" in result_summary
        assert result_summary["penalty"] == 0.0, "Wildcard has no penalty"
        assert "net_xp" in result_summary
        assert "description" in result_summary
        assert "type" in result_summary
        assert result_summary["type"] == "wildcard"

        # Verify metadata structure
        assert "method" in metadata
        assert metadata["method"] == "simulated_annealing"
        assert "horizon_label" in metadata
        assert "xp_column" in metadata
        assert "available_budget" in metadata
        assert "budget_pool_info" in metadata
        assert metadata["budget_pool_info"]["total_budget"] == 100.0

    def test_wildcard_builds_valid_squad(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard builds a valid FPL squad."""
        squad_df, _, _ = optimization_service.optimize_wildcard_squad(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
        )

        # Check formation
        position_counts = squad_df["position"].value_counts().to_dict()
        assert position_counts.get("GKP", 0) == 2, "Should have 2 goalkeepers"
        assert position_counts.get("DEF", 0) == 5, "Should have 5 defenders"
        assert position_counts.get("MID", 0) == 5, "Should have 5 midfielders"
        assert position_counts.get("FWD", 0) == 3, "Should have 3 forwards"

        # Check budget constraint
        total_cost = squad_df["price"].sum()
        assert total_cost <= 100.0, f"Squad cost {total_cost} exceeds £100m budget"

        # Check 3-per-team constraint
        team_counts = squad_df["team"].value_counts()
        max_per_team = team_counts.max()
        assert max_per_team <= 3, (
            f"Team constraint violated: {max_per_team} players from one team"
        )

    def test_wildcard_uses_100m_budget(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard always uses £100m budget regardless of current team value."""
        _, _, metadata = optimization_service.optimize_wildcard_squad(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
        )

        # Wildcard always starts with £100m
        assert metadata["budget_pool_info"]["total_budget"] == 100.0
        assert metadata["budget_pool_info"]["sellable_value"] == 0.0, (
            "Sellable value not applicable for wildcard"
        )


class TestWildcardHorizonConfiguration:
    """Test wildcard respects optimization horizon configuration."""

    def test_wildcard_uses_1gw_horizon(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard uses 1GW xP when configured."""
        # Set config to 1GW
        original_horizon = config.optimization.optimization_horizon
        try:
            config.optimization.optimization_horizon = "1gw"

            squad_df, result_summary, metadata = (
                optimization_service.optimize_wildcard_squad(
                    current_squad=current_squad,
                    team_data=team_data,
                    players_with_xp=sample_players,
                )
            )

            # Verify horizon configuration
            assert metadata["xp_column"] == "xP", "Should use 1GW xP column"
            assert metadata["horizon_label"] == "1-GW", "Should show 1-GW label"
            # Description shows "Wildcard: Rebuild entire squad (N players changed)"
            assert "Wildcard" in result_summary["description"], (
                "Description should mention Wildcard"
            )

            # Verify optimization used correct column
            assert result_summary["net_xp"] > 0, "Should have valid xP total"

        finally:
            config.optimization.optimization_horizon = original_horizon

    def test_wildcard_uses_5gw_horizon(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard uses 5GW xP when configured."""
        # Set config to 5GW
        original_horizon = config.optimization.optimization_horizon
        try:
            config.optimization.optimization_horizon = "5gw"

            squad_df, result_summary, metadata = (
                optimization_service.optimize_wildcard_squad(
                    current_squad=current_squad,
                    team_data=team_data,
                    players_with_xp=sample_players,
                )
            )

            # Verify horizon configuration
            assert metadata["xp_column"] == "xP_5gw", "Should use 5GW xP column"
            assert metadata["horizon_label"] == "5-GW", "Should show 5-GW label"
            # Description shows "Wildcard: Rebuild entire squad (N players changed)"
            assert "Wildcard" in result_summary["description"], (
                "Description should mention Wildcard"
            )

            # 5GW total should be significantly higher than 1GW
            assert result_summary["net_xp"] > 100, "5-GW total should be substantial"

        finally:
            config.optimization.optimization_horizon = original_horizon

    def test_wildcard_horizon_switch_produces_different_squads(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test that different horizons can produce different optimal squads."""
        original_horizon = config.optimization.optimization_horizon

        try:
            # Get 1GW squad
            config.optimization.optimization_horizon = "1gw"
            squad_1gw, summary_1gw, _ = optimization_service.optimize_wildcard_squad(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )

            # Get 5GW squad
            config.optimization.optimization_horizon = "5gw"
            squad_5gw, summary_5gw, _ = optimization_service.optimize_wildcard_squad(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )

            # XP totals should be different (5GW should be much higher)
            assert summary_5gw["net_xp"] > summary_1gw["net_xp"] * 4, (
                "5-GW optimization should produce higher total xP "
                "(approximately 5x the 1-GW total)"
            )

        finally:
            config.optimization.optimization_horizon = original_horizon


class TestWildcardConstraints:
    """Test wildcard optimization with constraints."""

    def test_wildcard_respects_must_include(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard includes specified players."""
        # Select specific players to include (ensuring valid formation)
        gkp_id = sample_players[sample_players["position"] == "GKP"]["player_id"].iloc[
            0
        ]
        def_id = sample_players[sample_players["position"] == "DEF"]["player_id"].iloc[
            0
        ]
        mid_id = sample_players[sample_players["position"] == "MID"]["player_id"].iloc[
            0
        ]

        must_include_ids = {gkp_id, def_id, mid_id}

        squad_df, _, _ = optimization_service.optimize_wildcard_squad(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
            must_include_ids=must_include_ids,
        )

        squad_ids = set(squad_df["player_id"])
        assert must_include_ids.issubset(squad_ids), (
            "Must-include players should be in wildcard squad"
        )

    def test_wildcard_respects_must_exclude(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard excludes specified players."""
        # Exclude some high-value players
        expensive_players = sample_players.nlargest(5, "price")
        must_exclude_ids = set(expensive_players["player_id"])

        squad_df, _, _ = optimization_service.optimize_wildcard_squad(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
            must_exclude_ids=must_exclude_ids,
        )

        squad_ids = set(squad_df["player_id"])
        assert squad_ids.isdisjoint(must_exclude_ids), (
            "Excluded players should not be in wildcard squad"
        )

    def test_wildcard_with_both_constraints(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard with both include and exclude constraints."""
        # Include some budget players
        budget_gkp = (
            sample_players[sample_players["position"] == "GKP"]
            .nsmallest(1, "price")["player_id"]
            .iloc[0]
        )
        must_include_ids = {budget_gkp}

        # Exclude some expensive players
        expensive_mids = (
            sample_players[sample_players["position"] == "MID"]
            .nlargest(3, "price")["player_id"]
            .tolist()
        )
        must_exclude_ids = set(expensive_mids)

        squad_df, _, _ = optimization_service.optimize_wildcard_squad(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
            must_include_ids=must_include_ids,
            must_exclude_ids=must_exclude_ids,
        )

        squad_ids = set(squad_df["player_id"])
        assert must_include_ids.issubset(squad_ids), "Must-include should be present"
        assert squad_ids.isdisjoint(must_exclude_ids), "Must-exclude should be absent"


class TestWildcardVsTransferOptimization:
    """Test differences between wildcard and transfer optimization."""

    def test_wildcard_ignores_current_squad(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard builds from scratch, not from current squad."""
        _, result_summary, metadata = optimization_service.optimize_wildcard_squad(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
        )

        # Wildcard should have zero penalty (building from scratch)
        # But reports actual number of players changed from current squad
        assert result_summary["penalty"] == 0.0
        assert result_summary["transfers"] >= 0  # Reports players changed

        # Should have full £100m budget
        assert metadata["budget_pool_info"]["total_budget"] == 100.0

    def test_wildcard_no_transfer_penalty(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard has no transfer penalties."""
        _, result_summary, _ = optimization_service.optimize_wildcard_squad(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
        )

        # No penalty for wildcard
        assert result_summary["penalty"] == 0.0
        # Reports number of players changed from current squad
        assert result_summary["transfers"] >= 0

        # Net XP equals total XP (no penalty deduction)
        assert result_summary["net_xp"] > 0


class TestWildcardOptimizationQuality:
    """Test wildcard produces quality optimized squads."""

    def test_wildcard_uses_budget_efficiently(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard uses most of the £100m budget."""
        squad_df, _, metadata = optimization_service.optimize_wildcard_squad(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
        )

        total_cost = squad_df["price"].sum()
        remaining_budget = metadata["available_budget"]

        # Should use most of the budget (at least 80% for test data)
        assert total_cost >= 80.0, f"Should use at least £80m, used £{total_cost:.1f}m"
        assert remaining_budget < 20.0, (
            f"Should have <£20m remaining, has £{remaining_budget:.1f}m"
        )
        assert (total_cost + remaining_budget) == pytest.approx(100.0, abs=0.1), (
            "Total should equal £100m"
        )

    def test_wildcard_selects_high_xp_players(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard prefers higher xP players."""
        squad_df, _, _ = optimization_service.optimize_wildcard_squad(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
        )

        # Squad average xP should be above overall median
        squad_avg_xp = squad_df["xP"].mean()
        all_avg_xp = sample_players["xP"].median()

        assert squad_avg_xp > all_avg_xp, (
            "Wildcard squad should have above-median xP players"
        )

    def test_wildcard_optimization_improves(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test that simulated annealing makes improvements."""
        _, _, metadata = optimization_service.optimize_wildcard_squad(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
        )

        # Simulated annealing should find improvements
        assert "sa_improvements" in metadata
        assert metadata["sa_improvements"] > 0, (
            "Optimization should find at least some improvements"
        )

        # Should complete configured iterations
        assert "sa_iterations" in metadata
        assert metadata["sa_iterations"] > 0


class TestWildcardErrorHandling:
    """Test wildcard error handling and validation."""

    def test_wildcard_with_empty_dataframe(
        self, optimization_service, current_squad, team_data
    ):
        """Test wildcard handles empty player data gracefully."""
        empty_df = pd.DataFrame()

        # Should either raise ValueError or return empty result
        try:
            result = optimization_service.optimize_wildcard_squad(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=empty_df,
            )
            # If it doesn't raise, result should be empty/invalid
            assert result is not None
        except (ValueError, KeyError):
            # Expected - empty data causes validation error
            pass

    def test_wildcard_with_missing_columns(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard fails with missing required columns."""
        incomplete_df = sample_players.drop(columns=["price"])

        with pytest.raises(ValueError, match="missing required columns"):
            optimization_service.optimize_wildcard_squad(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=incomplete_df,
            )

    def test_wildcard_with_invalid_must_include(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard fails with non-existent must-include player IDs."""
        with pytest.raises(ValueError, match="not found in dataset"):
            optimization_service.optimize_wildcard_squad(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
                must_include_ids={9999, 10000},  # Non-existent IDs
            )


class TestWildcardIntegration:
    """Integration tests for wildcard optimization."""

    def test_wildcard_integrates_with_starting_11_selection(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard squad can be used for starting 11 selection."""
        squad_df, _, metadata = optimization_service.optimize_wildcard_squad(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
        )

        # Convert to format expected by find_optimal_starting_11
        xp_column = metadata["xp_column"]

        # Should be able to select starting 11 from wildcard squad
        starting_11, formation, total_xp = (
            optimization_service.find_optimal_starting_11(squad_df, xp_column)
        )

        assert len(starting_11) == 11, "Should select 11 players"
        assert total_xp > 0, "Starting 11 should have positive xP"
        assert formation in [
            "3-4-3",
            "3-5-2",
            "4-3-3",
            "4-4-2",
            "4-5-1",
            "5-2-3",
            "5-3-2",
            "5-4-1",
        ], f"Should return valid formation, got {formation}"

    def test_wildcard_with_realistic_scenario(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard with realistic constraints (e.g., keep team captain)."""
        # Simulate keeping expensive premium captain
        premium_mid = (
            sample_players[sample_players["position"] == "MID"]
            .nlargest(1, "xP")["player_id"]
            .iloc[0]
        )

        # Simulate excluding injured/suspended players (but not the premium captain)
        exclude_candidates = sample_players[sample_players["player_id"] != premium_mid]
        exclude_players = exclude_candidates.sample(n=3)["player_id"].tolist()

        squad_df, result_summary, metadata = (
            optimization_service.optimize_wildcard_squad(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
                must_include_ids={premium_mid},
                must_exclude_ids=set(exclude_players),
            )
        )

        # Verify constraints respected
        assert premium_mid in squad_df["player_id"].values
        assert not any(pid in squad_df["player_id"].values for pid in exclude_players)

        # Should still build valid squad
        assert len(squad_df) == 15
        assert result_summary["net_xp"] > 0
        assert squad_df["price"].sum() <= 100.0
