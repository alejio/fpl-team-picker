"""
Tests for Phase 1-4 feature engineering enhancements.

Phase 1: Injury & rotation risk features (5 features)
Phase 2: Venue-specific team strength (6 features)
Phase 3: Player rankings & context (7 features)
Phase 4: Imputation strategy overhaul

Test coverage:
1. All Phase 1 features are created correctly (injury_risk, rotation_risk, etc.)
2. All Phase 2 features are created correctly (home_attack_strength, etc.)
3. All Phase 3 features are created correctly (form_rank, tackles, etc.)
4. Features merge correctly by player_id + gameweek
5. Features default to sensible values when data sources not provided
6. Imputation uses domain-aware defaults (not generic .fillna(0))
7. Feature count includes all 18 new features (117 total)
8. Ranking normalization works correctly (0-1 scale, lower rank = better)
9. Position-aware imputation for defensive features
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
def mock_derived_player_metrics():
    """Mock derived player metrics data (Phase 1)."""
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2, 3, 3],
            "gameweek": [5, 6, 5, 6, 5, 6],  # Shifted by +1 in feature engineering
            "injury_risk": [0.15, 0.20, 0.05, 0.10, 0.30, 0.25],
            "rotation_risk": [0.10, 0.15, 0.05, 0.08, 0.40, 0.35],
            "overperformance_risk": [0.25, 0.30, 0.10, 0.15, 0.50, 0.45],
            "form_momentum": [1.0, 0.5, 1.0, 0.0, -1.0, -0.5],  # +1 improving, 0 stable, -1 declining
        }
    )


@pytest.fixture
def mock_player_availability_snapshot():
    """Mock player availability snapshot data (Phase 1)."""
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2, 3, 3],
            "gameweek": [6, 7, 6, 7, 6, 7],  # No shift - snapshot is at correct gameweek
            "status": ["a", "a", "a", "d", "i", "i"],  # a=available, d=doubtful, i=injured
            "chance_of_playing_next_round": [100, 100, 100, 75, 0, 0],
        }
    )


@pytest.fixture
def mock_derived_team_form():
    """Mock derived team form data (Phase 2)."""
    return pd.DataFrame(
        {
            "team_id": [1, 1, 2, 2, 3, 3],
            "gameweek": [5, 6, 5, 6, 5, 6],  # Shifted by +1 in feature engineering
            "home_attack_strength": [1.2, 1.25, 1.3, 1.35, 1.1, 1.15],
            "away_attack_strength": [1.0, 1.05, 1.1, 1.15, 0.9, 0.95],
            "home_defense_strength": [0.9, 0.85, 0.95, 0.90, 1.0, 1.05],
            "away_defense_strength": [1.1, 1.15, 1.05, 1.10, 1.2, 1.25],
            "home_advantage": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "venue_consistency": [0.7, 0.75, 0.8, 0.85, 0.6, 0.65],
        }
    )


@pytest.fixture
def mock_players_enhanced():
    """Mock players enhanced data (Phase 3)."""
    return pd.DataFrame(
        {
            "player_id": [1, 1, 2, 2, 3, 3],
            "gameweek": [5, 6, 5, 6, 5, 6],  # Shifted by +1 in feature engineering
            "form_rank": [10, 15, 5, 8, 50, 55],  # Lower rank = better form
            "ict_index_rank": [12, 18, 3, 5, 60, 65],
            "points_per_game_rank": [8, 12, 2, 4, 45, 50],
            "tackles": [2.0, 2.5, 0.5, 0.8, 3.5, 4.0],  # DEF/MID have more
            "recoveries": [5.0, 5.5, 1.0, 1.5, 8.0, 8.5],
            "defensive_contribution": [15.0, 16.0, 3.0, 4.0, 25.0, 26.0],
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

    betting = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "gameweek": [5, 6, 7, 5, 6, 7, 5, 6, 7],
            "team_win_probability": [0.5] * 9,
            "opponent_win_probability": [0.3] * 9,
            "draw_probability": [0.2] * 9,
            "implied_clean_sheet_probability": [0.35] * 9,
            "implied_total_goals": [2.5] * 9,
            "team_expected_goals": [1.3] * 9,
            "market_consensus_strength": [0.6] * 9,
            "odds_movement_team": [0.0] * 9,
            "odds_movement_magnitude": [0.1] * 9,
            "favorite_status": [0.5] * 9,
            "asian_handicap_line": [0.0] * 9,
            "handicap_team_odds": [1.9] * 9,
            "expected_goal_difference": [0.0] * 9,
            "over_under_signal": [0.1] * 9,
            "referee_encoded": [5] * 9,
        }
    )

    return {
        "ownership": ownership,
        "value": value,
        "fixture_difficulty": fixture_diff,
        "betting": betting,
    }


class TestPhase1InjuryRotationFeatures:
    """Test suite for Phase 1: Injury & rotation risk features."""

    def test_all_phase1_features_created(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_derived_player_metrics,
        mock_player_availability_snapshot,
        minimal_enhanced_data,
    ):
        """Test that all 5 Phase 1 features are created."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            betting_features_df=minimal_enhanced_data["betting"],
            derived_player_metrics_df=mock_derived_player_metrics,
            player_availability_snapshot_df=mock_player_availability_snapshot,
        )

        result = engineer.fit_transform(mock_player_data)

        # Check all 5 Phase 1 features exist
        expected_features = [
            "injury_risk",
            "rotation_risk",
            "chance_of_playing_next_round",
            "status_encoded",
            "overperformance_risk",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing Phase 1 feature: {feature}"

    def test_injury_rotation_features_merge_correctly(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_derived_player_metrics,
        mock_player_availability_snapshot,
        minimal_enhanced_data,
    ):
        """Test that Phase 1 features merge correctly by player_id + gameweek."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            betting_features_df=minimal_enhanced_data["betting"],
            derived_player_metrics_df=mock_derived_player_metrics,
            player_availability_snapshot_df=mock_player_availability_snapshot,
        )

        result = engineer.fit_transform(mock_player_data)

        # Merge player_id back from original data to access by player
        result_with_id = result.copy()
        result_with_id["player_id"] = mock_player_data["player_id"].values

        # Player 1, GW6: injury_risk from GW5 data (shifted +1) should be 0.15
        # But we're predicting GW6, so we use GW5 metrics (shifted to GW6)
        player1_gw6 = result_with_id[result_with_id["player_id"] == 1].iloc[0]
        assert np.isclose(player1_gw6["injury_risk"], 0.15, atol=0.01), (
            "Player 1 GW6 should have injury_risk=0.15 from GW5 data"
        )

        # Player 2, GW6: status_encoded should be 0 (available)
        player2_gw6 = result_with_id[result_with_id["player_id"] == 2].iloc[0]
        assert player2_gw6["status_encoded"] == 0, (
            "Player 2 GW6 should be available (status=0)"
        )

        # Player 2, GW7: status_encoded should be 1 (doubtful)
        player2_gw7 = result_with_id[result_with_id["player_id"] == 2].iloc[1]
        assert player2_gw7["status_encoded"] == 1, (
            "Player 2 GW7 should be doubtful (status=1)"
        )

        # Player 3, GW6: status_encoded should be 2 (injured)
        player3_gw6 = result_with_id[result_with_id["player_id"] == 3].iloc[0]
        assert player3_gw6["status_encoded"] == 2, (
            "Player 3 GW6 should be injured (status=2)"
        )

    def test_phase1_features_default_when_data_missing(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        minimal_enhanced_data,
    ):
        """Test that Phase 1 features default to sensible values when data sources not provided."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            betting_features_df=minimal_enhanced_data["betting"],
            # Phase 1 data sources NOT provided
        )

        result = engineer.fit_transform(mock_player_data)

        # Should default to domain-aware values
        assert np.allclose(result["injury_risk"], 0.1, atol=0.01), (
            "injury_risk should default to 0.1"
        )
        assert np.allclose(result["rotation_risk"], 0.2, atol=0.01), (
            "rotation_risk should default to 0.2"
        )
        assert np.allclose(result["chance_of_playing_next_round"], 100.0, atol=0.01), (
            "chance_of_playing_next_round should default to 100"
        )
        assert np.allclose(result["status_encoded"], 0, atol=0.01), (
            "status_encoded should default to 0 (available)"
        )
        assert np.allclose(result["overperformance_risk"], 0.0, atol=0.01), (
            "overperformance_risk should default to 0.0"
        )


class TestPhase2VenueSpecificStrength:
    """Test suite for Phase 2: Venue-specific team strength."""

    def test_all_phase2_features_created(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_derived_team_form,
        minimal_enhanced_data,
    ):
        """Test that all 6 Phase 2 features are created."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            betting_features_df=minimal_enhanced_data["betting"],
            derived_team_form_df=mock_derived_team_form,
        )

        result = engineer.fit_transform(mock_player_data)

        # Check all 6 Phase 2 features exist
        expected_features = [
            "home_attack_strength",
            "away_attack_strength",
            "home_defense_strength",
            "away_defense_strength",
            "home_advantage",
            "venue_consistency",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing Phase 2 feature: {feature}"

    def test_venue_features_merge_correctly(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_derived_team_form,
        minimal_enhanced_data,
    ):
        """Test that Phase 2 features merge correctly by team_id + gameweek."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            betting_features_df=minimal_enhanced_data["betting"],
            derived_team_form_df=mock_derived_team_form,
        )

        result = engineer.fit_transform(mock_player_data)

        # Merge team_id back from original data to access by team
        result_with_team = result.copy()
        result_with_team["team_id"] = mock_player_data["team_id"].values

        # Team 1 (Arsenal), GW6: home_attack_strength from GW5 data (shifted +1) should be 1.2
        team1_gw6 = result_with_team[result_with_team["team_id"] == 1].iloc[0]
        assert np.isclose(team1_gw6["home_attack_strength"], 1.2, atol=0.01), (
            "Team 1 GW6 should have home_attack_strength=1.2 from GW5 data"
        )

    def test_phase2_features_default_when_data_missing(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        minimal_enhanced_data,
    ):
        """Test that Phase 2 features default to neutral values when data not provided."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            betting_features_df=minimal_enhanced_data["betting"],
            # Phase 2 data source NOT provided
        )

        result = engineer.fit_transform(mock_player_data)

        # Should default to neutral values
        assert np.allclose(result["home_attack_strength"], 1.0, atol=0.01), (
            "home_attack_strength should default to 1.0"
        )
        assert np.allclose(result["away_attack_strength"], 1.0, atol=0.01), (
            "away_attack_strength should default to 1.0"
        )
        assert np.allclose(result["home_defense_strength"], 1.0, atol=0.01), (
            "home_defense_strength should default to 1.0"
        )
        assert np.allclose(result["away_defense_strength"], 1.0, atol=0.01), (
            "away_defense_strength should default to 1.0"
        )
        assert np.allclose(result["home_advantage"], 0.0, atol=0.01), (
            "home_advantage should default to 0.0"
        )
        assert np.allclose(result["venue_consistency"], 0.5, atol=0.01), (
            "venue_consistency should default to 0.5"
        )


class TestPhase3RankingFeatures:
    """Test suite for Phase 3: Player rankings & context."""

    def test_all_phase3_features_created(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_players_enhanced,
        mock_derived_player_metrics,
        minimal_enhanced_data,
    ):
        """Test that all 7 Phase 3 features are created."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            betting_features_df=minimal_enhanced_data["betting"],
            players_enhanced_df=mock_players_enhanced,
            derived_player_metrics_df=mock_derived_player_metrics,
        )

        result = engineer.fit_transform(mock_player_data)

        # Check all 7 Phase 3 features exist
        expected_features = [
            "form_rank",
            "ict_index_rank",
            "points_per_game_rank",
            "defensive_contribution",
            "tackles",
            "recoveries",
            "form_momentum",
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing Phase 3 feature: {feature}"

    def test_ranking_features_merge_correctly(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_players_enhanced,
        mock_derived_player_metrics,
        minimal_enhanced_data,
    ):
        """Test that Phase 3 features merge correctly by player_id + gameweek."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            betting_features_df=minimal_enhanced_data["betting"],
            players_enhanced_df=mock_players_enhanced,
            derived_player_metrics_df=mock_derived_player_metrics,
        )

        result = engineer.fit_transform(mock_player_data)

        # Merge player_id back from original data to access by player
        result_with_id = result.copy()
        result_with_id["player_id"] = mock_player_data["player_id"].values

        # Player 3 (DEF), GW6: tackles should be higher (DEF/MID have more)
        player3_gw6 = result_with_id[result_with_id["player_id"] == 3].iloc[0]
        assert np.isclose(player3_gw6["tackles"], 3.5, atol=0.01), (
            "Player 3 (DEF) GW6 should have tackles=3.5"
        )

        # Player 2 (FWD), GW6: tackles should be lower
        player2_gw6 = result_with_id[result_with_id["player_id"] == 2].iloc[0]
        assert np.isclose(player2_gw6["tackles"], 0.5, atol=0.01), (
            "Player 2 (FWD) GW6 should have tackles=0.5"
        )

    def test_ranking_normalization(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_players_enhanced,
        minimal_enhanced_data,
    ):
        """Test that ranking features are normalized to 0-1 scale (lower rank = better)."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            betting_features_df=minimal_enhanced_data["betting"],
            players_enhanced_df=mock_players_enhanced,
            derived_player_metrics_df=pd.DataFrame(),  # No form_momentum
        )

        result = engineer.fit_transform(mock_player_data)

        # Merge player_id back from original data to access by player
        result_with_id = result.copy()
        result_with_id["player_id"] = mock_player_data["player_id"].values

        # Rankings should be normalized to 0-1 (lower original rank = higher normalized value)
        # Player 2 has form_rank=5 (best), Player 1 has form_rank=10, Player 3 has form_rank=50 (worst)
        # After normalization: Player 2 should have highest value, Player 3 lowest

        player2_gw6 = result_with_id[result_with_id["player_id"] == 2].iloc[0]
        player1_gw6 = result_with_id[result_with_id["player_id"] == 1].iloc[0]
        player3_gw6 = result_with_id[result_with_id["player_id"] == 3].iloc[0]

        # Player 2 (rank 5) should have higher normalized value than Player 1 (rank 10)
        assert player2_gw6["form_rank"] > player1_gw6["form_rank"], (
            "Player 2 (better rank) should have higher normalized form_rank"
        )
        # Player 1 (rank 10) should have higher normalized value than Player 3 (rank 50)
        assert player1_gw6["form_rank"] > player3_gw6["form_rank"], (
            "Player 1 (better rank) should have higher normalized form_rank than Player 3"
        )

        # All normalized ranks should be between 0 and 1
        assert 0 <= player1_gw6["form_rank"] <= 1, "form_rank should be normalized to 0-1"
        assert 0 <= player2_gw6["form_rank"] <= 1, "form_rank should be normalized to 0-1"
        assert 0 <= player3_gw6["form_rank"] <= 1, "form_rank should be normalized to 0-1"

    def test_phase3_features_default_when_data_missing(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        minimal_enhanced_data,
    ):
        """Test that Phase 3 features default to sensible values when data not provided."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            betting_features_df=minimal_enhanced_data["betting"],
            # Phase 3 data sources NOT provided
        )

        result = engineer.fit_transform(mock_player_data)

        # Rankings should default to -1 (unranked)
        assert np.allclose(result["form_rank"], -1.0, atol=0.01), (
            "form_rank should default to -1.0 (unranked)"
        )
        assert np.allclose(result["ict_index_rank"], -1.0, atol=0.01), (
            "ict_index_rank should default to -1.0 (unranked)"
        )
        assert np.allclose(result["points_per_game_rank"], -1.0, atol=0.01), (
            "points_per_game_rank should default to -1.0 (unranked)"
        )

        # Defensive features should default to 0 (position-aware only applies when data is partially available)
        # When no data source provided, all default to 0.0
        assert np.allclose(result["defensive_contribution"], 0.0, atol=0.01), (
            "defensive_contribution should default to 0.0 when no data provided"
        )
        assert np.allclose(result["tackles"], 0.0, atol=0.01), (
            "tackles should default to 0.0 when no data provided"
        )
        assert np.allclose(result["recoveries"], 0.0, atol=0.01), (
            "recoveries should default to 0.0 when no data provided"
        )
        assert np.allclose(result["form_momentum"], 0.0, atol=0.01), (
            "form_momentum should default to 0.0 (stable)"
        )


class TestPhase4ImputationStrategy:
    """Test suite for Phase 4: Imputation strategy overhaul."""

    def test_position_aware_imputation(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        minimal_enhanced_data,
    ):
        """Test that defensive features use position-aware imputation when data is partially available."""
        # Create partial Phase 3 data (only for player 1) to trigger position-aware imputation
        partial_players_enhanced = pd.DataFrame(
            {
                "player_id": [1, 1],
                "gameweek": [5, 6],  # Shifted by +1
                "form_rank": [10, 15],
                "ict_index_rank": [12, 18],
                "points_per_game_rank": [8, 12],
                "tackles": [2.0, 2.5],  # Only player 1 has data
                "recoveries": [5.0, 5.5],
                "defensive_contribution": [15.0, 16.0],
            }
        )

        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            betting_features_df=minimal_enhanced_data["betting"],
            players_enhanced_df=partial_players_enhanced,  # Partial data - triggers position-aware
        )

        result = engineer.fit_transform(mock_player_data)

        # Merge position back from original data to check position-aware imputation
        result_with_position = result.copy()
        result_with_position["position"] = mock_player_data["position"].values
        result_with_position["player_id"] = mock_player_data["player_id"].values

        # DEF/MID should have higher defaults for tackles/recoveries when data is missing
        # Player 1 (MID) has data, so uses that
        # Player 3 (DEF) has no data, so should get position-aware default (2.0)
        def_players = result_with_position[result_with_position["position"] == "DEF"]
        mid_players = result_with_position[result_with_position["position"] == "MID"]
        fwd_players = result_with_position[result_with_position["position"] == "FWD"]

        # DEF players without data should have tackles=2.0 (position-aware default)
        def_without_data = def_players[def_players["player_id"] == 3]
        assert np.allclose(def_without_data["tackles"], 2.0, atol=0.01), (
            "DEF players without data should have tackles=2.0 (position-aware default)"
        )

        # MID players with data should use their data, not default
        mid_with_data = mid_players[mid_players["player_id"] == 1]
        assert np.allclose(mid_with_data["tackles"], 2.0, atol=0.5), (
            "MID players with data should use their data"
        )

        # FWD players without data should have tackles=0.0
        fwd_without_data = fwd_players[fwd_players["player_id"] == 2]
        assert np.allclose(fwd_without_data["tackles"], 0.0, atol=0.01), (
            "FWD players without data should have tackles=0.0 default"
        )

    def test_risk_probability_imputation(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        minimal_enhanced_data,
    ):
        """Test that risk/probability fields use domain-aware defaults."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            betting_features_df=minimal_enhanced_data["betting"],
            # No Phase 1 data - should use domain-aware defaults
        )

        result = engineer.fit_transform(mock_player_data)

        # Risk fields should use neutral defaults, not 0
        assert np.allclose(result["injury_risk"], 0.1, atol=0.01), (
            "injury_risk should default to 0.1 (10% base risk), not 0"
        )
        assert np.allclose(result["rotation_risk"], 0.2, atol=0.01), (
            "rotation_risk should default to 0.2 (20% base rotation), not 0"
        )
        assert np.allclose(result["chance_of_playing_next_round"], 100.0, atol=0.01), (
            "chance_of_playing_next_round should default to 100 (healthy), not 0"
        )


class TestTotalFeatureCount:
    """Test suite for total feature count (117 features)."""

    def test_total_feature_count_117(
        self,
        mock_player_data,
        mock_teams_data,
        mock_fixtures_data,
        mock_derived_player_metrics,
        mock_player_availability_snapshot,
        mock_derived_team_form,
        mock_players_enhanced,
        minimal_enhanced_data,
    ):
        """Test that total feature count is 117 (99 + 18 new features)."""
        engineer = FPLFeatureEngineer(
            fixtures_df=mock_fixtures_data,
            teams_df=mock_teams_data,
            team_strength={"Arsenal": 1.2, "Liverpool": 1.25, "Man City": 1.23},
            ownership_trends_df=minimal_enhanced_data["ownership"],
            value_analysis_df=minimal_enhanced_data["value"],
            fixture_difficulty_df=minimal_enhanced_data["fixture_difficulty"],
            betting_features_df=minimal_enhanced_data["betting"],
            derived_player_metrics_df=mock_derived_player_metrics,
            player_availability_snapshot_df=mock_player_availability_snapshot,
            derived_team_form_df=mock_derived_team_form,
            players_enhanced_df=mock_players_enhanced,
        )

        result = engineer.fit_transform(mock_player_data)
        feature_names = engineer.get_feature_names_out()

        # Should be 117 features total (99 base + 18 new)
        assert len(feature_names) == 117, (
            f"Expected 117 features, got {len(feature_names)}"
        )
        assert result.shape[1] == 117, (
            f"Result should have 117 columns, got {result.shape[1]}"
        )

        # Verify all phase features are in the list
        phase1_features = [
            "injury_risk",
            "rotation_risk",
            "chance_of_playing_next_round",
            "status_encoded",
            "overperformance_risk",
        ]
        phase2_features = [
            "home_attack_strength",
            "away_attack_strength",
            "home_defense_strength",
            "away_defense_strength",
            "home_advantage",
            "venue_consistency",
        ]
        phase3_features = [
            "form_rank",
            "ict_index_rank",
            "points_per_game_rank",
            "defensive_contribution",
            "tackles",
            "recoveries",
            "form_momentum",
        ]

        for feature in phase1_features + phase2_features + phase3_features:
            assert feature in feature_names, f"Missing feature in feature_names: {feature}"
