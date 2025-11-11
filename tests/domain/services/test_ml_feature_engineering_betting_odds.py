"""
Tests for betting odds features in ML feature engineering.

Tests the implementation of betting odds features from derived betting data.

Test coverage:
1. All 15 betting odds features are created correctly
2. Features merge correctly by player_id + gameweek
3. Features default to neutral values when betting_features_df not provided
4. Missing betting data (e.g., GW1) fills with neutral defaults
5. Betting features have correct values from source data
6. Feature count includes 15 betting odds features (99 total)
"""

import pytest
import pandas as pd
import numpy as np

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
    """Mock fixtures DataFrame."""
    return pd.DataFrame(
        {
            "event": [6, 7],
            "home_team_id": [1, 2],
            "away_team_id": [3, 3],
        }
    )


@pytest.fixture
def mock_player_data():
    """Mock player performance data for testing."""
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2, 3, 3],
            "gameweek": [6, 7, 6, 7, 6, 7],
            "team_id": [1, 1, 2, 2, 3, 3],
            "position": ["MID", "MID", "FWD", "FWD", "DEF", "DEF"],
            "minutes": [90, 90, 85, 90, 80, 75],
            "goals_scored": [1, 0, 2, 1, 0, 0],
            "assists": [0, 1, 1, 0, 0, 0],
            "total_points": [7, 5, 12, 6, 2, 1],
            "bonus": [0, 1, 3, 0, 0, 0],
            "bps": [25, 30, 55, 25, 20, 10],
            "clean_sheets": [0, 0, 0, 0, 1, 0],
            "expected_goals": [0.8, 0.3, 1.8, 0.9, 0.1, 0.1],
            "expected_assists": [0.2, 0.6, 0.7, 0.3, 0.0, 0.0],
            "ict_index": [8.5, 9.2, 14.5, 9.0, 4.0, 3.5],
            "influence": [40, 50, 75, 45, 20, 15],
            "creativity": [30, 45, 60, 35, 10, 8],
            "threat": [50, 35, 90, 50, 10, 8],
            "value": [85, 85, 105, 105, 45, 45],
            "yellow_cards": [0, 0, 0, 0, 0, 0],
            "red_cards": [0, 0, 0, 0, 0, 0],
            "goals_conceded": [2, 1, 1, 2, 1, 3],
            "saves": [0, 0, 0, 0, 0, 0],
            "expected_goal_involvements": [1.0, 0.9, 2.5, 1.2, 0.1, 0.1],
            "expected_goals_conceded": [1.5, 1.2, 1.0, 1.8, 1.0, 2.0],
        }
    )


@pytest.fixture
def mock_betting_features():
    """Mock betting odds features from get_derived_betting_features().

    Player 1 (Salah): Team favorite, high win probability
    Player 2 (Haaland): Strong favorite, very high win probability
    Player 3 (Defender): Underdog team, low win probability
    """
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2, 3, 3],
            "gameweek": [6, 7, 6, 7, 6, 7],
            # Implied probabilities (6)
            "team_win_probability": [0.55, 0.60, 0.70, 0.65, 0.25, 0.30],
            "opponent_win_probability": [0.25, 0.20, 0.15, 0.20, 0.55, 0.50],
            "draw_probability": [0.20, 0.20, 0.15, 0.15, 0.20, 0.20],
            "implied_clean_sheet_probability": [0.40, 0.45, 0.50, 0.48, 0.20, 0.25],
            "implied_total_goals": [2.8, 2.6, 3.2, 2.9, 2.4, 2.5],
            "team_expected_goals": [1.6, 1.7, 2.1, 1.9, 0.8, 1.0],
            # Market confidence (4)
            "market_consensus_strength": [0.7, 0.8, 0.9, 0.85, 0.4, 0.5],
            "odds_movement_team": [0.05, -0.10, -0.15, 0.0, 0.10, 0.05],
            "odds_movement_magnitude": [0.15, 0.20, 0.30, 0.10, 0.15, 0.12],
            "favorite_status": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            # Asian Handicap (3)
            "asian_handicap_line": [-0.5, -1.0, -1.5, -1.0, 1.0, 0.5],
            "handicap_team_odds": [1.85, 1.75, 1.70, 1.80, 2.10, 2.00],
            "expected_goal_difference": [-0.35, -0.70, -1.05, -0.70, 0.70, 0.35],
            # Match context (2)
            "over_under_signal": [0.15, 0.10, 0.35, 0.20, -0.10, 0.0],
            "referee_encoded": [5, 12, 8, 5, 10, 12],
        }
    )


@pytest.fixture
def minimal_enhanced_data():
    """Minimal ownership, value, and fixture difficulty data."""
    ownership = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "gameweek": [5, 6, 7, 5, 6, 7, 5, 6, 7],
            "selected_by_percent": [25.0] * 3 + [35.0] * 3 + [5.0] * 3,
            "net_transfers_gw": [100] * 3 + [200] * 3 + [10] * 3,
            "avg_net_transfers_5gw": [80] * 3 + [150] * 3 + [5] * 3,
            "transfer_momentum": ["rising"] * 6 + ["neutral"] * 3,
            "ownership_velocity": [1.0] * 3 + [1.5] * 3 + [0.1] * 3,
            "ownership_tier": ["premium"] * 3 + ["premium"] * 3 + ["budget"] * 3,
            "bandwagon_score": [0.8] * 3 + [0.9] * 3 + [0.1] * 3,
        }
    )

    value = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "gameweek": [5, 6, 7, 5, 6, 7, 5, 6, 7],
            "points_per_pound": [1.2] * 3 + [1.5] * 3 + [0.6] * 3,
            "value_vs_position": [1.1] * 3 + [1.3] * 3 + [0.8] * 3,
            "predicted_price_change_1gw": [0] * 9,
            "price_volatility": [0] * 9,
            "price_risk": [0] * 9,
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


class TestBettingOddsFeatures:
    """Test suite for betting odds features."""

    def test_all_15_betting_features_created(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_betting_features,
        minimal_enhanced_data,
    ):
        """Test that all 15 betting odds features are created."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            betting_features_df=mock_betting_features,
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        result = engineer.fit_transform(
            mock_player_data, mock_player_data["total_points"]
        )

        # Check all 15 betting odds features exist
        expected_features = [
            # Implied probabilities (6)
            "team_win_probability",
            "opponent_win_probability",
            "draw_probability",
            "implied_clean_sheet_probability",
            "implied_total_goals",
            "team_expected_goals",
            # Market confidence (4)
            "market_consensus_strength",
            "odds_movement_team",
            "odds_movement_magnitude",
            "favorite_status",
            # Asian Handicap (3)
            "asian_handicap_line",
            "handicap_team_odds",
            "expected_goal_difference",
            # Match context (2)
            "over_under_signal",
            "referee_encoded",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing betting feature: {feature}"

    def test_betting_features_merge_correctly(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_betting_features,
        minimal_enhanced_data,
    ):
        """Test that betting features merge correctly by player_id + gameweek."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            betting_features_df=mock_betting_features,
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        result = engineer.fit_transform(
            mock_player_data, mock_player_data["total_points"]
        )

        # Player 1, GW6 (row 0): team_win_probability should be 0.55
        assert np.isclose(result.iloc[0]["team_win_probability"], 0.55, atol=0.01), (
            "Player 1 GW6 should have team_win_prob=0.55"
        )

        # Player 2, GW7 (row 3): team_win_probability should be 0.65
        assert np.isclose(result.iloc[3]["team_win_probability"], 0.65, atol=0.01), (
            "Player 2 GW7 should have team_win_prob=0.65"
        )

        # Player 3, GW6 (row 4): underdog team, favorite_status should be 0
        assert result.iloc[4]["favorite_status"] == 0.0, (
            "Player 3 GW6 should not be favorite"
        )

    def test_fails_fast_without_betting_data(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        minimal_enhanced_data,
    ):
        """Test that FPLFeatureEngineer fails fast when betting_features_df not provided (FAIL FAST principle)."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            # betting_features_df NOT provided - should fail
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        # Should fail with clear error message
        with pytest.raises(ValueError, match="betting_features_df is empty"):
            engineer.fit_transform(mock_player_data, mock_player_data["total_points"])

    def test_asian_handicap_features_have_correct_values(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_betting_features,
        minimal_enhanced_data,
    ):
        """Test that Asian Handicap features have correct values from source."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            betting_features_df=mock_betting_features,
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        result = engineer.fit_transform(
            mock_player_data, mock_player_data["total_points"]
        )

        # Player 2 (Haaland), GW6 (row 2): Strong favorite, handicap -1.5
        assert np.isclose(result.iloc[2]["asian_handicap_line"], -1.5, atol=0.01), (
            "Haaland GW6 should have handicap=-1.5"
        )
        assert np.isclose(
            result.iloc[2]["expected_goal_difference"], -1.05, atol=0.01
        ), "Haaland GW6 expected_goal_diff should be -1.05"

        # Player 3 (Defender), GW6 (row 4): Underdog, positive handicap
        assert np.isclose(result.iloc[4]["asian_handicap_line"], 1.0, atol=0.01), (
            "Underdog GW6 should have positive handicap"
        )
        assert np.isclose(
            result.iloc[4]["expected_goal_difference"], 0.70, atol=0.01
        ), "Underdog GW6 expected_goal_diff should be 0.70"

    def test_market_confidence_features_have_correct_values(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_betting_features,
        minimal_enhanced_data,
    ):
        """Test that market confidence features reflect betting data correctly."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            betting_features_df=mock_betting_features,
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
        )

        result = engineer.fit_transform(
            mock_player_data, mock_player_data["total_points"]
        )

        # Player 2, GW6 (row 2): High market consensus (0.9), negative odds movement
        assert np.isclose(
            result.iloc[2]["market_consensus_strength"], 0.9, atol=0.01
        ), "Haaland GW6 should have high consensus"
        assert np.isclose(result.iloc[2]["odds_movement_team"], -0.15, atol=0.01), (
            "Haaland GW6 odds sharpened (-0.15)"
        )

        # Player 3, GW6 (row 4): Low consensus (0.4), positive odds movement
        assert np.isclose(
            result.iloc[4]["market_consensus_strength"], 0.4, atol=0.01
        ), "Underdog GW6 should have low consensus"
        assert np.isclose(result.iloc[4]["odds_movement_team"], 0.10, atol=0.01), (
            "Underdog GW6 odds drifted (+0.10)"
        )

    def test_total_feature_count_includes_betting_odds(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_betting_features,
        minimal_enhanced_data,
    ):
        """Test that total feature count is 118 (117 - 4 redundant + 5 data quality indicators)."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            betting_features_df=mock_betting_features,
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            # Phase 1-3 data sources not provided - will use defaults
        )

        result = engineer.fit_transform(
            mock_player_data, mock_player_data["total_points"]
        )

        feature_names = engineer.get_feature_names_out()

        # Should be 118 features total (117 - 4 redundant + 5 data quality indicators)
        assert len(feature_names) == 118, (
            f"Expected 118 features, got {len(feature_names)}"
        )
        assert result.shape[1] == 118, (
            f"Result should have 118 columns, got {result.shape[1]}"
        )
