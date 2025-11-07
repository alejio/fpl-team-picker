"""Tests for OptimizationService.optimize_wildcard_squad() - Wildcard Chip Optimization.

Tests the wildcard chip optimization wrapper that builds entire squads from scratch
using simulated annealing with £100m budget reset and horizon configuration support.
"""

import pytest
import pandas as pd
from fpl_team_picker.domain.services.optimization_service import OptimizationService


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

    def test_wildcard_builds_valid_squad(
        self, optimization_service, current_squad, team_data, sample_players
    ):
        """Test wildcard builds a valid FPL squad with correct structure and constraints."""
        squad_df, result_summary, metadata = (
            optimization_service.optimize_wildcard_squad(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=sample_players,
            )
        )

        # Verify return types and structure
        assert isinstance(squad_df, pd.DataFrame), "Should return DataFrame"
        assert isinstance(result_summary, dict), "Should return result summary dict"
        assert isinstance(metadata, dict), "Should return metadata dict"
        assert len(squad_df) == 15, "Wildcard should build 15-player squad"

        # Verify result summary structure
        assert result_summary["type"] == "wildcard"
        assert result_summary["penalty"] == 0.0, "Wildcard has no penalty"
        assert result_summary["net_xp"] > 0

        # Verify budget
        assert metadata["budget_pool_info"]["total_budget"] == 100.0

        # Check formation
        position_counts = squad_df["position"].value_counts().to_dict()
        assert position_counts.get("GKP", 0) == 2, "Should have 2 goalkeepers"
        assert position_counts.get("DEF", 0) == 5, "Should have 5 defenders"
        assert position_counts.get("MID", 0) == 5, "Should have 5 midfielders"
        assert position_counts.get("FWD", 0) == 3, "Should have 3 forwards"

        # Check budget constraint
        total_cost = squad_df["price"].sum()
        assert total_cost <= 100.0 + 1e-9, (
            f"Squad cost {total_cost} exceeds £100m budget"
        )

        # Check 3-per-team constraint
        team_counts = squad_df["team"].value_counts()
        max_per_team = team_counts.max()
        assert max_per_team <= 3, (
            f"Team constraint violated: {max_per_team} players from one team"
        )
