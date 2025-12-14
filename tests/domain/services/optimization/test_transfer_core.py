"""Unit tests for Transfer Core optimization.

Tests for:
- optimize_transfers entry point with chip support
- plan_premium_acquisition scenarios
- Budget handling for wildcard/free hit
- Transfer constraints (must_include, must_exclude)
"""

import pytest
import pandas as pd
from fpl_team_picker.domain.services.optimization_service import OptimizationService


@pytest.fixture
def sample_players():
    """Create sample player data for testing."""
    players = []
    positions = [
        ("GKP", 4, 5.0, 6.0),
        ("DEF", 10, 4.5, 7.0),
        ("MID", 12, 5.0, 13.0),
        ("FWD", 8, 5.5, 12.0),
    ]

    player_id = 1
    teams = ["Arsenal", "Liverpool", "Man City", "Chelsea", "Spurs"]

    for position, count, min_price, max_price in positions:
        for i in range(count):
            price = min_price + (max_price - min_price) * i / max(count - 1, 1)
            xp = 2.0 + (price - 4.0) * 1.5

            players.append(
                {
                    "player_id": player_id,
                    "web_name": f"Player{player_id}",
                    "position": position,
                    "price": round(price, 1),
                    "xP": round(xp, 2),
                    "xP_5gw": round(xp * 5, 2),
                    "team": teams[player_id % len(teams)],
                    "status": "a",
                    "expected_minutes": 90,
                }
            )
            player_id += 1

    return pd.DataFrame(players)


@pytest.fixture
def current_squad(sample_players):
    """Create a sample current squad."""
    gkp = sample_players[sample_players["position"] == "GKP"].iloc[:2]
    def_players = sample_players[sample_players["position"] == "DEF"].iloc[:5]
    mid_players = sample_players[sample_players["position"] == "MID"].iloc[:5]
    fwd_players = sample_players[sample_players["position"] == "FWD"].iloc[:3]

    squad = pd.concat([gkp, def_players, mid_players, fwd_players])
    return squad.reset_index(drop=True)


@pytest.fixture
def team_data():
    """Create dummy team data."""
    return {
        "bank": 1.0,
        "team_value": 100.0,
        "free_transfers": 1,
    }


@pytest.fixture
def service():
    """Create OptimizationService instance."""
    return OptimizationService()


class TestOptimizeTransfers:
    """Test optimize_transfers entry point."""

    def test_optimize_transfers_normal_gameweek(
        self, service, current_squad, sample_players, team_data
    ):
        """Test normal gameweek optimization (1 free transfer)."""
        squad_df, scenario, metadata = service.optimize_transfers(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
        )

        assert isinstance(squad_df, pd.DataFrame)
        assert len(squad_df) == 15  # Full squad
        assert isinstance(scenario, dict)
        assert isinstance(metadata, dict)

    def test_optimize_transfers_with_saved_transfer(
        self, service, current_squad, sample_players, team_data
    ):
        """Test optimization with saved transfer (2 free transfers)."""
        team_data_copy = team_data.copy()
        team_data_copy["free_transfers"] = 2

        squad_df, scenario, metadata = service.optimize_transfers(
            current_squad=current_squad,
            team_data=team_data_copy,
            players_with_xp=sample_players,
        )

        assert isinstance(squad_df, pd.DataFrame)
        assert len(squad_df) == 15

    def test_optimize_transfers_wildcard(
        self, service, current_squad, sample_players, team_data
    ):
        """Test wildcard optimization (15 free transfers, budget resets)."""
        squad_df, scenario, metadata = service.optimize_transfers(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
            free_transfers_override=15,
        )

        assert isinstance(squad_df, pd.DataFrame)
        assert len(squad_df) == 15
        # Wildcard should reset budget to £100m
        assert metadata.get("budget_used", 0) <= 100.0

    def test_optimize_transfers_free_hit(
        self, service, current_squad, sample_players, team_data
    ):
        """Test Free Hit optimization (1GW only, squad reverts)."""
        squad_df, scenario, metadata = service.optimize_transfers(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
            free_transfers_override=15,
            is_free_hit=True,
        )

        assert isinstance(squad_df, pd.DataFrame)
        assert len(squad_df) == 15
        # Free Hit uses special budget (from config)
        assert isinstance(metadata, dict)

    def test_optimize_transfers_bench_boost(
        self, service, current_squad, sample_players, team_data
    ):
        """Test Bench Boost optimization (all 15 players score)."""
        squad_df, scenario, metadata = service.optimize_transfers(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
            is_bench_boost=True,
        )

        assert isinstance(squad_df, pd.DataFrame)
        assert len(squad_df) == 15

    def test_optimize_transfers_with_must_include(
        self, service, current_squad, sample_players, team_data
    ):
        """Test optimization with must_include constraint."""
        must_include_id = current_squad.iloc[0]["player_id"]

        squad_df, scenario, metadata = service.optimize_transfers(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
            must_include_ids={must_include_id},
        )

        assert isinstance(squad_df, pd.DataFrame)
        assert must_include_id in squad_df["player_id"].values

    def test_optimize_transfers_with_must_exclude(
        self, service, current_squad, sample_players, team_data
    ):
        """Test optimization with must_exclude constraint."""
        # Exclude a player that can be easily replaced (not a unique position/team)
        # Pick a MID player (more options available)
        mid_players = current_squad[current_squad["position"] == "MID"]
        if len(mid_players) > 0:
            must_exclude_id = mid_players.iloc[0]["player_id"]
        else:
            must_exclude_id = current_squad.iloc[0]["player_id"]

        squad_df, scenario, metadata = service.optimize_transfers(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=sample_players,
            must_exclude_ids={must_exclude_id},
        )

        assert isinstance(squad_df, pd.DataFrame)
        # If optimization succeeded, excluded player should not be in squad
        if len(squad_df) > 0:
            assert must_exclude_id not in squad_df["player_id"].values

    def test_optimize_transfers_empty_squad(
        self, service, sample_players, team_data
    ):
        """Test optimization with empty squad."""
        empty_squad = pd.DataFrame()

        squad_df, scenario, metadata = service.optimize_transfers(
            current_squad=empty_squad,
            team_data=team_data,
            players_with_xp=sample_players,
        )

        # Should handle gracefully
        assert isinstance(squad_df, pd.DataFrame)
        assert isinstance(metadata, dict)


class TestPlanPremiumAcquisition:
    """Test plan_premium_acquisition method."""

    def test_plan_premium_acquisition_basic(
        self, service, current_squad, sample_players
    ):
        """Test basic premium acquisition planning."""
        # Add a premium player not in squad
        premium_player = pd.DataFrame(
            [
                {
                    "player_id": 999,
                    "web_name": "Premium",
                    "position": "MID",
                    "price": 12.0,
                    "xP": 10.0,
                    "team": "Man City",
                    "status": "a",
                }
            ]
        )
        all_players = pd.concat([sample_players, premium_player])

        budget_pool = service.calculate_budget_pool(
            current_squad=current_squad, bank_balance=1.0
        )

        scenarios = service.plan_premium_acquisition(
            current_squad=current_squad,
            all_players=all_players,
            budget_pool_info=budget_pool,
            top_n=3,
        )

        assert isinstance(scenarios, list)
        # May be empty if no viable scenarios, but should not crash

    def test_plan_premium_acquisition_no_premium_targets(
        self, service, current_squad, sample_players
    ):
        """Test when no premium targets available."""
        budget_pool = service.calculate_budget_pool(
            current_squad=current_squad, bank_balance=50.0  # High bank
        )

        scenarios = service.plan_premium_acquisition(
            current_squad=current_squad,
            all_players=sample_players,
            budget_pool_info=budget_pool,
            top_n=3,
        )

        assert scenarios == []  # No premium targets if bank covers all

    def test_plan_premium_acquisition_empty_squad(
        self, service, sample_players
    ):
        """Test premium acquisition with empty squad."""
        empty_squad = pd.DataFrame()
        budget_pool = {"bank_balance": 1.0, "total_budget": 1.0}

        scenarios = service.plan_premium_acquisition(
            current_squad=empty_squad,
            all_players=sample_players,
            budget_pool_info=budget_pool,
            top_n=3,
        )

        assert scenarios == []

    def test_plan_premium_acquisition_empty_players(
        self, service, current_squad
    ):
        """Test premium acquisition with no available players."""
        empty_players = pd.DataFrame()
        budget_pool = service.calculate_budget_pool(
            current_squad=current_squad, bank_balance=1.0
        )

        scenarios = service.plan_premium_acquisition(
            current_squad=current_squad,
            all_players=empty_players,
            budget_pool_info=budget_pool,
            top_n=3,
        )

        assert scenarios == []

    def test_plan_premium_acquisition_top_n_limit(
        self, service, current_squad, sample_players
    ):
        """Test that top_n limits returned scenarios."""
        # Add multiple premium players
        premium_players = pd.DataFrame(
            [
                {
                    "player_id": 999 + i,
                    "web_name": f"Premium{i}",
                    "position": "MID",
                    "price": 12.0 + i,
                    "xP": 10.0 + i,
                    "team": "Man City",
                    "status": "a",
                }
                for i in range(10)
            ]
        )
        all_players = pd.concat([sample_players, premium_players])

        budget_pool = service.calculate_budget_pool(
            current_squad=current_squad, bank_balance=1.0
        )

        scenarios = service.plan_premium_acquisition(
            current_squad=current_squad,
            all_players=all_players,
            budget_pool_info=budget_pool,
            top_n=3,
        )

        assert len(scenarios) <= 3

    def test_plan_premium_acquisition_scenario_structure(
        self, service, current_squad, sample_players
    ):
        """Test that scenarios have expected structure."""
        premium_player = pd.DataFrame(
            [
                {
                    "player_id": 999,
                    "web_name": "Premium",
                    "position": "MID",
                    "price": 12.0,
                    "xP": 10.0,
                    "team": "Man City",
                    "status": "a",
                }
            ]
        )
        all_players = pd.concat([sample_players, premium_player])

        budget_pool = service.calculate_budget_pool(
            current_squad=current_squad, bank_balance=1.0
        )

        scenarios = service.plan_premium_acquisition(
            current_squad=current_squad,
            all_players=all_players,
            budget_pool_info=budget_pool,
            top_n=1,
        )

        if scenarios:  # If any scenarios found
            scenario = scenarios[0]
            assert "type" in scenario
            assert "target_player" in scenario
            assert "xp_gain" in scenario
            assert "transfers" in scenario
