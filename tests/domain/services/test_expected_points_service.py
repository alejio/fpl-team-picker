"""Tests for ExpectedPointsService - EWMA minutes calculation."""

import pandas as pd
import pytest
from fpl_team_picker.domain.services.expected_points_service import (
    ExpectedPointsService,
)


class TestEWMAMinutesCalculation:
    """Test EWMA-based expected minutes calculation."""

    @pytest.fixture
    def xp_service(self):
        """Create ExpectedPointsService instance."""
        return ExpectedPointsService(config={"debug": False})

    @pytest.fixture
    def historical_data(self):
        """Sample historical gameweek data."""
        return pd.DataFrame(
            {
                "player_id": [1, 1, 1, 1, 1, 2, 2, 2],
                "gameweek": [1, 2, 3, 4, 5, 3, 4, 5],
                "minutes": [90, 90, 85, 90, 75, 60, 0, 30],
                "total_points": [5, 8, 6, 10, 4, 3, 0, 2],
            }
        )

    @pytest.fixture
    def players_data(self):
        """Sample player data."""
        return pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "web_name": ["Salah", "Rotation Risk", "New Player"],
                "position": ["MID", "MID", "FWD"],
                "team": [1, 1, 2],
                "price_gbp": [13.0, 5.0, 7.0],
            }
        )

    def test_ewma_calculation_with_full_history(
        self, xp_service, players_data, historical_data
    ):
        """Test EWMA calculation for player with 5+ games history."""
        result = xp_service._calculate_expected_minutes(
            players_data.copy(), historical_data, target_gameweek=6
        )

        # Player 1 (Salah): [90, 90, 85, 90, 75] -> EWMA ~84
        salah_minutes = result[result["player_id"] == 1]["expected_minutes"].values[0]
        assert 80 <= salah_minutes <= 86, f"Expected ~84, got {salah_minutes}"

        # Player 2 (Rotation): [60, 0, 30] -> EWMA ~30-40
        rotation_minutes = result[result["player_id"] == 2]["expected_minutes"].values[
            0
        ]
        assert 25 <= rotation_minutes <= 45, f"Expected ~35, got {rotation_minutes}"

    def test_fallback_to_position_defaults_for_new_players(
        self, xp_service, players_data, historical_data
    ):
        """Test position-based fallback for players with no history."""
        result = xp_service._calculate_expected_minutes(
            players_data.copy(), historical_data, target_gameweek=6
        )

        # Player 3 (New Player): No history -> FWD default = 70
        new_player_minutes = result[result["player_id"] == 3][
            "expected_minutes"
        ].values[0]
        assert new_player_minutes == 70, f"Expected 70 (FWD default), got {new_player_minutes}"

        # Verify all players have expected_minutes
        assert result["expected_minutes"].notna().all()
        assert (result["expected_minutes"] >= 0).all()
        assert (result["expected_minutes"] <= 90).all()
