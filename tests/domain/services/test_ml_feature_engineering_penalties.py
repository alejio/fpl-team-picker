"""
Tests for penalty and set-piece taker features in ML feature engineering.

Tests the implementation of penalty_order, corners_order, and free_kick_order
features from raw_players_bootstrap data.

Test coverage:
1. is_primary_penalty_taker correctly identifies order=1 penalty takers
2. is_penalty_taker correctly identifies any penalty taker (order >= 1)
3. is_corner_taker correctly identifies primary corner takers
4. is_fk_taker correctly identifies primary free kick takers
5. Features default to 0/False when raw_players_df not provided
6. Merging by player_id works correctly across gameweeks
"""

import pytest
import pandas as pd

from fpl_team_picker.domain.services.ml_feature_engineering import FPLFeatureEngineer


@pytest.fixture
def mock_teams_data():
    """Mock teams DataFrame."""
    return pd.DataFrame(
        {
            "team_id": [1, 2, 3],
            "name": ["Arsenal", "Liverpool", "Man City"],
        }
    )


@pytest.fixture
def mock_fixtures_data():
    """Mock fixtures DataFrame.

    GW6: Arsenal (1) vs Man City (3), Liverpool (2) at home
    GW7: Arsenal (1) at home, Liverpool (2) vs Man City (3)

    Each team plays exactly once per gameweek to avoid duplicate merges.
    """
    return pd.DataFrame(
        {
            "event": [6, 7],
            "home_team_id": [1, 2],
            "away_team_id": [3, 3],
        }
    )


@pytest.fixture
def mock_player_data():
    """Mock player performance data for testing.

    Player 1: Salah (penalty taker, order=1)
    Player 2: Palmer (penalty + corner + FK taker, all order=1)
    Player 3: Regular player (no set-piece duties)
    Player 4: Backup penalty taker (order=2)
    """
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "gameweek": [6, 7, 6, 7, 6, 7, 6, 7],
            "team_id": [1, 1, 2, 2, 3, 3, 1, 1],
            "position": ["MID", "MID", "MID", "MID", "FWD", "FWD", "FWD", "FWD"],
            "minutes": [90, 90, 85, 90, 80, 75, 60, 55],
            "goals_scored": [1, 0, 2, 1, 1, 0, 0, 0],
            "assists": [0, 1, 1, 0, 0, 0, 0, 0],
            "total_points": [7, 5, 12, 6, 5, 2, 2, 1],
            "bonus": [0, 1, 3, 0, 0, 0, 0, 0],
            "bps": [25, 30, 55, 25, 20, 10, 8, 5],
            "clean_sheets": [0, 0, 0, 0, 0, 0, 0, 0],
            "expected_goals": [0.8, 0.3, 1.8, 0.9, 0.6, 0.2, 0.1, 0.1],
            "expected_assists": [0.2, 0.6, 0.7, 0.3, 0.1, 0.0, 0.0, 0.0],
            "ict_index": [8.5, 9.2, 14.5, 9.0, 6.5, 4.0, 3.0, 2.5],
            "influence": [40, 50, 75, 45, 30, 20, 15, 10],
            "creativity": [30, 45, 60, 35, 15, 10, 8, 5],
            "threat": [50, 35, 90, 50, 35, 15, 10, 8],
            "value": [85, 85, 105, 105, 75, 75, 60, 60],
            "yellow_cards": [0, 0, 0, 0, 1, 0, 0, 0],
            "red_cards": [0, 0, 0, 0, 0, 0, 0, 0],
            "goals_conceded": [2, 1, 1, 2, 2, 3, 2, 1],
            "saves": [0, 0, 0, 0, 0, 0, 0, 0],
            "expected_goal_involvements": [1.0, 0.9, 2.5, 1.2, 0.7, 0.2, 0.1, 0.1],
            "expected_goals_conceded": [1.5, 1.2, 1.0, 1.8, 1.5, 2.0, 1.5, 1.2],
        }
    )


@pytest.fixture
def mock_raw_players_data():
    """Mock raw_players_bootstrap data with penalty/set-piece order.

    Player 1 (Salah): Primary penalty taker only
    Player 2 (Palmer): Primary penalty, corner, and FK taker (all order=1)
    Player 3: No set-piece duties (all order=0)
    Player 4: Backup penalty taker (order=2)
    """
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "web_name": ["Salah", "Palmer", "Regular", "Backup"],
            "team_id": [1, 2, 3, 1],
            "penalties_order": [1, 1, 0, 2],
            "corners_and_indirect_freekicks_order": [0, 1, 0, 0],
            "direct_freekicks_order": [0, 1, 0, 0],
        }
    )


@pytest.fixture
def minimal_enhanced_data():
    """Minimal ownership, value, and fixture difficulty data."""
    ownership = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "gameweek": [5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7],
            "selected_by_percent": [25.0] * 3 + [15.0] * 3 + [3.0] * 3 + [1.0] * 3,
            "net_transfers_gw": [100] * 3 + [50] * 3 + [0] * 3 + [0] * 3,
            "avg_net_transfers_5gw": [80] * 3 + [40] * 3 + [0] * 3 + [0] * 3,
            "transfer_momentum": ["rising"] * 3 + ["rising"] * 3 + ["neutral"] * 6,
            "ownership_velocity": [1.0] * 3 + [0.8] * 3 + [0] * 6,
            "ownership_tier": ["premium"] * 3 + ["popular"] * 3 + ["budget"] * 6,
            "bandwagon_score": [0.8] * 3 + [0.6] * 3 + [0] * 6,
        }
    )

    value = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "gameweek": [5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7],
            "points_per_pound": [1.2] * 3 + [1.5] * 3 + [0.8] * 3 + [0.5] * 3,
            "value_vs_position": [1.1] * 3 + [1.3] * 3 + [0.9] * 3 + [0.7] * 3,
            "predicted_price_change_1gw": [0] * 12,
            "price_volatility": [0] * 12,
            "price_risk": [0] * 12,
        }
    )

    fixture_diff = pd.DataFrame(
        {
            "team_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "gameweek": [5, 6, 7, 5, 6, 7, 5, 6, 7],
            "congestion_difficulty": [1.0] * 9,
            "form_difficulty": [1.0] * 9,
            "clean_sheet_probability": [0.3] * 9,
        }
    )

    return {
        "ownership": ownership,
        "value": value,
        "fixture_difficulty": fixture_diff,
    }


class TestPenaltySetPieceFeatures:
    """Test suite for penalty and set-piece taker features."""

    def test_is_primary_penalty_taker_identifies_order_1(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_raw_players_data,
        minimal_enhanced_data,
    ):
        """Test that is_primary_penalty_taker=1 only for players with penalties_order=1."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            raw_players_df=mock_raw_players_data,
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        result = engineer.fit_transform(
            mock_player_data, mock_player_data["total_points"]
        )

        # Check feature exists
        assert "is_primary_penalty_taker" in result.columns

        # Player 1 (Salah, rows 0-1): penalties_order=1 → is_primary_penalty_taker=1
        salah_rows = result.iloc[0:2]
        assert all(salah_rows["is_primary_penalty_taker"] == 1), (
            "Salah should be primary penalty taker"
        )

        # Player 2 (Palmer, rows 2-3): penalties_order=1 → is_primary_penalty_taker=1
        palmer_rows = result.iloc[2:4]
        assert all(palmer_rows["is_primary_penalty_taker"] == 1), (
            "Palmer should be primary penalty taker"
        )

        # Player 3 (Regular, rows 4-5): penalties_order=0 → is_primary_penalty_taker=0
        regular_rows = result.iloc[4:6]
        assert all(regular_rows["is_primary_penalty_taker"] == 0), (
            "Regular player should NOT be primary penalty taker"
        )

        # Player 4 (Backup, rows 6-7): penalties_order=2 → is_primary_penalty_taker=0
        backup_rows = result.iloc[6:8]
        assert all(backup_rows["is_primary_penalty_taker"] == 0), (
            "Backup taker should NOT be primary"
        )

    def test_is_penalty_taker_identifies_any_taker(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_raw_players_data,
        minimal_enhanced_data,
    ):
        """Test that is_penalty_taker=1 for any player with penalties_order >= 1."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            raw_players_df=mock_raw_players_data,
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        result = engineer.fit_transform(
            mock_player_data, mock_player_data["total_points"]
        )

        # Check feature exists
        assert "is_penalty_taker" in result.columns

        # Player 1 (Salah, rows 0-1): penalties_order >= 1 → is_penalty_taker=1
        salah_rows = result.iloc[0:2]
        assert all(salah_rows["is_penalty_taker"] == 1)

        # Player 2 (Palmer, rows 2-3): penalties_order >= 1 → is_penalty_taker=1
        palmer_rows = result.iloc[2:4]
        assert all(palmer_rows["is_penalty_taker"] == 1)

        # Player 4 (Backup, rows 6-7, order=2): should also be is_penalty_taker=1
        backup_rows = result.iloc[6:8]
        assert all(backup_rows["is_penalty_taker"] == 1), (
            "Backup penalty taker (order=2) should be is_penalty_taker=1"
        )

        # Player 3 (Regular, rows 4-5): penalties_order=0 → is_penalty_taker=0
        regular_rows = result.iloc[4:6]
        assert all(regular_rows["is_penalty_taker"] == 0)

    def test_is_corner_taker_identifies_primary_takers(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_raw_players_data,
        minimal_enhanced_data,
    ):
        """Test that is_corner_taker=1 only for primary corner takers (order=1)."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            raw_players_df=mock_raw_players_data,
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        result = engineer.fit_transform(
            mock_player_data, mock_player_data["total_points"]
        )

        assert "is_corner_taker" in result.columns

        # Only Player 2 (Palmer, rows 2-3) has corners_order=1
        palmer_rows = result.iloc[2:4]
        assert all(palmer_rows["is_corner_taker"] == 1)

        # Others should be 0
        salah_rows = result.iloc[0:2]
        assert all(salah_rows["is_corner_taker"] == 0)

        regular_rows = result.iloc[4:6]
        assert all(regular_rows["is_corner_taker"] == 0)

    def test_is_fk_taker_identifies_primary_takers(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_raw_players_data,
        minimal_enhanced_data,
    ):
        """Test that is_fk_taker=1 only for primary FK takers (order=1)."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            raw_players_df=mock_raw_players_data,
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        result = engineer.fit_transform(
            mock_player_data, mock_player_data["total_points"]
        )

        assert "is_fk_taker" in result.columns

        # Only Player 2 (Palmer, rows 2-3) has direct_freekicks_order=1
        palmer_rows = result.iloc[2:4]
        assert all(palmer_rows["is_fk_taker"] == 1)

        # Others should be 0
        salah_rows = result.iloc[0:2]
        assert all(salah_rows["is_fk_taker"] == 0)

    def test_features_default_to_zero_without_raw_players(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        minimal_enhanced_data,
    ):
        """Test that features default to 0 when raw_players_df not provided."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            # raw_players_df NOT provided
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        result = engineer.fit_transform(
            mock_player_data, mock_player_data["total_points"]
        )

        # All features should exist and be 0
        assert "is_primary_penalty_taker" in result.columns
        assert "is_penalty_taker" in result.columns
        assert "is_corner_taker" in result.columns
        assert "is_fk_taker" in result.columns

        assert all(result["is_primary_penalty_taker"] == 0)
        assert all(result["is_penalty_taker"] == 0)
        assert all(result["is_corner_taker"] == 0)
        assert all(result["is_fk_taker"] == 0)

    def test_features_persist_across_gameweeks(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_raw_players_data,
        minimal_enhanced_data,
    ):
        """Test that penalty/set-piece features are consistent across gameweeks for same player."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            raw_players_df=mock_raw_players_data,
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        result = engineer.fit_transform(
            mock_player_data, mock_player_data["total_points"]
        )

        # Palmer (player_id=2) appears in rows 2 (GW6) and 3 (GW7)
        palmer_gw6 = result.iloc[2]
        palmer_gw7 = result.iloc[3]

        # Should have same values in both gameweeks
        assert (
            palmer_gw6["is_primary_penalty_taker"]
            == palmer_gw7["is_primary_penalty_taker"]
            == 1
        )
        assert palmer_gw6["is_penalty_taker"] == palmer_gw7["is_penalty_taker"] == 1
        assert palmer_gw6["is_corner_taker"] == palmer_gw7["is_corner_taker"] == 1
        assert palmer_gw6["is_fk_taker"] == palmer_gw7["is_fk_taker"] == 1
