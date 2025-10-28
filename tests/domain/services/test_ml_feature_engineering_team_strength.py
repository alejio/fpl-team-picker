"""
Tests for per-gameweek team strength in ML feature engineering.

Tests the implementation of per-gameweek team strength to eliminate data leakage
in ML training. Ensures that GW N predictions use team strength calculated from
GW 1 to N-1 only.

Test coverage:
1. calculate_per_gameweek_team_strength() returns correct format
2. FPLFeatureEngineer correctly detects per-gameweek format
3. FPLFeatureEngineer applies per-gameweek strength correctly
4. Backward compatibility with static team strength format
5. No data leakage (GW N uses strength from GW N-1)
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from fpl_team_picker.domain.services.ml_feature_engineering import (
    FPLFeatureEngineer,
    calculate_per_gameweek_team_strength,
)


@pytest.fixture
def mock_teams_data():
    """Mock teams DataFrame."""
    return pd.DataFrame(
        {
            "team_id": [1, 2, 3, 4],
            "name": ["Arsenal", "Liverpool", "Man City", "Leicester"],
        }
    )


@pytest.fixture
def mock_fixtures_data():
    """Mock fixtures DataFrame.

    GW6: Arsenal (1) vs Man City (3), Liverpool (2) vs Leicester (4)
    GW7: Arsenal (1) vs Liverpool (2), Man City (3) vs Leicester (4)

    Each team plays exactly once per gameweek to avoid duplicate merges.
    """
    return pd.DataFrame(
        {
            "event": [6, 6, 7, 7],
            "home_team_id": [1, 2, 1, 3],
            "away_team_id": [3, 4, 2, 4],
        }
    )


@pytest.fixture
def mock_player_data():
    """Mock player performance data for testing."""
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2],
            "gameweek": [6, 7, 6, 7],
            "team_id": [1, 1, 2, 2],
            "position": ["MID", "MID", "FWD", "FWD"],
            "minutes": [90, 85, 60, 75],
            "goals_scored": [1, 0, 2, 1],
            "assists": [0, 1, 0, 0],
            "total_points": [7, 5, 10, 6],
            "bonus": [0, 1, 2, 0],
            "bps": [25, 30, 45, 20],
            "clean_sheets": [0, 0, 0, 0],
            "expected_goals": [0.8, 0.3, 1.5, 0.9],
            "expected_assists": [0.2, 0.6, 0.1, 0.1],
            "ict_index": [8.5, 9.2, 12.3, 7.1],
            "influence": [40, 50, 60, 35],
            "creativity": [30, 45, 20, 15],
            "threat": [50, 35, 80, 40],
            "value": [85, 85, 95, 95],
            "yellow_cards": [0, 0, 1, 0],
            "red_cards": [0, 0, 0, 0],
            "goals_conceded": [2, 1, 1, 2],
            "saves": [0, 0, 0, 0],
            "expected_goal_involvements": [1.0, 0.9, 1.6, 1.0],
            "expected_goals_conceded": [1.5, 1.2, 1.0, 1.8],
        }
    )


@pytest.fixture
def minimal_enhanced_data():
    """Minimal ownership, value, and fixture difficulty data for testing."""
    # Need GW5 data for shift(1) to work with GW6-7 predictions
    ownership = pd.DataFrame({
        "player_id": [1, 1, 1, 2, 2, 2],
        "gameweek": [5, 6, 7, 5, 6, 7],
        "selected_by_percent": [5.0, 5.0, 5.0, 10.0, 10.0, 10.0],
        "net_transfers_gw": [0, 0, 0, 100, 100, 100],
        "avg_net_transfers_5gw": [0, 0, 0, 50, 50, 50],
        "transfer_momentum": ["neutral"] * 6,
        "ownership_velocity": [0] * 6,
        "ownership_tier": ["budget"] * 3 + ["popular"] * 3,
        "bandwagon_score": [0] * 6,
    })

    value = pd.DataFrame({
        "player_id": [1, 1, 1, 2, 2, 2],
        "gameweek": [5, 6, 7, 5, 6, 7],
        "points_per_pound": [0.8] * 3 + [1.0] * 3,
        "value_vs_position": [1.0] * 6,
        "predicted_price_change_1gw": [0] * 6,
        "price_volatility": [0] * 6,
        "price_risk": [0] * 6,
    })

    fixture_diff = pd.DataFrame({
        "team_id": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
        "gameweek": [5, 6, 7, 5, 6, 7, 6, 7, 6, 7],
        "congestion_difficulty": [1.0] * 10,
        "form_difficulty": [1.0] * 10,
        "clean_sheet_probability": [0.3] * 10,
    })

    return {
        "ownership": ownership,
        "value": value,
        "fixture_difficulty": fixture_diff,
    }


class TestCalculatePerGameweekTeamStrength:
    """Test suite for calculate_per_gameweek_team_strength() utility function."""

    @patch(
        "fpl_team_picker.domain.services.team_analytics_service.TeamAnalyticsService"
    )
    def test_returns_correct_format(self, mock_team_analytics_class, mock_teams_data):
        """Test that function returns Dict[int, Dict[str, float]] format."""
        # Mock TeamAnalyticsService
        mock_service = Mock()
        mock_team_analytics_class.return_value = mock_service

        # Mock get_team_strength to return different values per gameweek
        def mock_get_strength(target_gameweek, teams_data, current_season_data=None):
            return {
                "Arsenal": 1.2 + (target_gameweek * 0.01),
                "Liverpool": 1.25 + (target_gameweek * 0.01),
                "Man City": 1.23 + (target_gameweek * 0.01),
                "Leicester": 1.1 + (target_gameweek * 0.01),
            }

        mock_service.get_team_strength = mock_get_strength

        # Calculate per-gameweek strength
        result = calculate_per_gameweek_team_strength(
            start_gw=6, end_gw=8, teams_df=mock_teams_data
        )

        # Verify structure: Dict[int, Dict[str, float]]
        assert isinstance(result, dict)
        assert 6 in result
        assert 7 in result
        assert 8 in result
        assert isinstance(result[6], dict)
        assert "Arsenal" in result[6]
        assert isinstance(result[6]["Arsenal"], float)

    @patch(
        "fpl_team_picker.domain.services.team_analytics_service.TeamAnalyticsService"
    )
    def test_no_data_leakage(self, mock_team_analytics_class, mock_teams_data):
        """Test that GW N uses strength from GW N-1 (no leakage)."""
        mock_service = Mock()
        mock_team_analytics_class.return_value = mock_service

        # Track which gameweek was requested
        called_gameweeks = []

        def mock_get_strength(target_gameweek, teams_data, current_season_data=None):
            called_gameweeks.append(target_gameweek)
            return {"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23, "Leicester": 1.1}

        mock_service.get_team_strength = mock_get_strength

        # Calculate strength for GW6-8
        calculate_per_gameweek_team_strength(
            start_gw=6, end_gw=8, teams_df=mock_teams_data
        )

        # Verify no leakage:
        # - GW6 prediction should use GW5 strength (called_gameweeks[0] == 5)
        # - GW7 prediction should use GW6 strength (called_gameweeks[1] == 6)
        # - GW8 prediction should use GW7 strength (called_gameweeks[2] == 7)
        assert called_gameweeks == [5, 6, 7]

    @patch(
        "fpl_team_picker.domain.services.team_analytics_service.TeamAnalyticsService"
    )
    def test_handles_gw1_edge_case(self, mock_team_analytics_class, mock_teams_data):
        """Test that GW1 doesn't request GW0 (uses GW1 instead)."""
        mock_service = Mock()
        mock_team_analytics_class.return_value = mock_service

        called_gameweeks = []

        def mock_get_strength(target_gameweek, teams_data, current_season_data=None):
            called_gameweeks.append(target_gameweek)
            return {"Arsenal": 1.2}

        mock_service.get_team_strength = mock_get_strength

        # Edge case: start from GW1
        calculate_per_gameweek_team_strength(
            start_gw=1, end_gw=1, teams_df=mock_teams_data
        )

        # Should request GW1 (not GW0)
        assert called_gameweeks == [1]


class TestFPLFeatureEngineerPerGameweekStrength:
    """Test suite for FPLFeatureEngineer with per-gameweek team strength."""

    def test_detects_per_gameweek_format(self, mock_teams_data, mock_fixtures_data):
        """Test that _detect_strength_format correctly identifies per-gameweek format."""
        per_gw_strength = {
            6: {"Arsenal": 1.2, "Liverpool": 1.25},
            7: {"Arsenal": 1.21, "Liverpool": 1.24},
        }

        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength=per_gw_strength,
        )

        assert engineer._is_per_gameweek_strength is True

    def test_detects_static_format(self, mock_teams_data, mock_fixtures_data):
        """Test that _detect_strength_format correctly identifies static format."""
        static_strength = {"Arsenal": 1.2, "Liverpool": 1.25}

        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength=static_strength,
        )

        assert engineer._is_per_gameweek_strength is False

    def test_applies_per_gameweek_strength_correctly(
        self, mock_teams_data, mock_fixtures_data, mock_player_data, minimal_enhanced_data
    ):
        """Test that per-gameweek strength is applied correctly to features."""
        # Different strength for GW6 vs GW7
        per_gw_strength = {
            6: {"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23, "Leicester": 1.1},
            7: {"Arsenal": 1.3, "Liverpool": 1.35, "Man City": 1.33, "Leicester": 1.2},
        }

        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength=per_gw_strength,
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        # Transform the data
        result = engineer.fit_transform(mock_player_data)

        # Verify opponent_strength differs between GW6 and GW7
        # (This tests that per-gameweek strength is being applied)
        assert "opponent_strength" in result.columns

        # Get opponent strengths for each gameweek
        gw6_strengths = result[result.index.isin([0, 2])]["opponent_strength"].values
        gw7_strengths = result[result.index.isin([1, 3])]["opponent_strength"].values

        # GW6 and GW7 should have different opponent strengths due to per-GW strength
        assert not np.array_equal(gw6_strengths, gw7_strengths)

    def test_backward_compatible_with_static_strength(
        self, mock_teams_data, mock_fixtures_data, mock_player_data, minimal_enhanced_data
    ):
        """Test that static format still works (backward compatibility)."""
        static_strength = {"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23, "Leicester": 1.1}

        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength=static_strength,
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        # Should not raise any errors
        result = engineer.fit_transform(mock_player_data)

        assert "opponent_strength" in result.columns
        assert len(result) == len(mock_player_data)

    def test_invalid_format_raises_error(
        self, mock_teams_data, mock_fixtures_data, mock_player_data
    ):
        """Test that invalid team_strength format raises descriptive error."""
        # Invalid: int keys but float values (not dict)
        invalid_strength = {6: 1.2, 7: 1.3}

        with pytest.raises(ValueError, match="Invalid team_strength format"):
            FPLFeatureEngineer(
                fixtures_df=mock_fixtures_data,
                teams_df=mock_teams_data,
                team_strength=invalid_strength,
            )

    def test_empty_strength_dict_handled(
        self, mock_teams_data, mock_fixtures_data, mock_player_data, minimal_enhanced_data
    ):
        """Test that empty team_strength dict is handled gracefully."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={},  # Empty dict
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        # Should not crash, should use default 1.0 strength
        result = engineer.fit_transform(mock_player_data)
        assert "opponent_strength" in result.columns


class TestIntegration:
    """Integration tests for per-gameweek team strength in full ML pipeline."""

    @patch(
        "fpl_team_picker.domain.services.team_analytics_service.TeamAnalyticsService"
    )
    def test_full_pipeline_with_per_gameweek_strength(
        self,
        mock_team_analytics_class,
        mock_teams_data,
        mock_fixtures_data,
        mock_player_data,
        minimal_enhanced_data,
    ):
        """Test full workflow: calculate per-GW strength -> feature engineering."""
        # Mock TeamAnalyticsService
        mock_service = Mock()
        mock_team_analytics_class.return_value = mock_service

        def mock_get_strength(target_gameweek, teams_data, current_season_data=None):
            # Return different values per gameweek to simulate evolution
            base = {"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23, "Leicester": 1.1}
            return {team: val + (target_gameweek * 0.01) for team, val in base.items()}

        mock_service.get_team_strength = mock_get_strength

        # Step 1: Calculate per-gameweek strength
        team_strength = calculate_per_gameweek_team_strength(
            start_gw=6, end_gw=7, teams_df=mock_teams_data
        )

        # Step 2: Use in feature engineering
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength=team_strength,
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        result = engineer.fit_transform(mock_player_data)

        # Verify features were created successfully
        assert "opponent_strength" in result.columns
        assert "fixture_difficulty" in result.columns
        assert len(result) == len(mock_player_data)

        # Verify per-gameweek strength was used (not static)
        assert engineer._is_per_gameweek_strength is True
