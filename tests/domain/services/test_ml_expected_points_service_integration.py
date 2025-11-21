"""
Integration tests for FPLFeatureEngineer with 122 features (including Phase 1-4 enhancements + elite interactions).

Tests the feature engineering pipeline:
1. Feature engineer produces exactly 122 features when all data sources provided
2. Feature engineer fails fast without betting data (FAIL FAST principle)
3. All 15 betting odds features are present in output
4. All 18 Phase 1-3 features are present (with defaults if data not provided)
"""

import pytest
import pandas as pd


class TestFPLFeatureEngineer118Features:
    """Test FPLFeatureEngineer with 118-feature output (117 - 4 redundant + 5 data quality indicators)."""

    @pytest.fixture
    def sample_historical_data(self):
        """Historical data for GW1-8 (need 5+ GWs for rolling features)."""
        data = []
        for gw in range(1, 9):  # GW1-8
            for player_id in [1, 2, 3]:
                data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "total_points": 5 + (player_id * gw % 3),
                        "minutes": 90,
                        "goals_scored": 0,
                        "assists": 0,
                        "clean_sheets": 1 if player_id == 1 else 0,
                        "goals_conceded": 1,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "saves": 0,
                        "bonus": 0,
                        "bps": 20,
                        "influence": 30.0,
                        "creativity": 20.0,
                        "threat": 10.0,
                        "ict_index": 6.0,
                        "expected_goals": 0.2,
                        "expected_assists": 0.1,
                        "expected_goal_involvements": 0.3,
                        "expected_goals_conceded": 1.0,
                        "value": 50 + player_id * 10,
                        "position": "DEF"
                        if player_id == 1
                        else ("MID" if player_id == 2 else "FWD"),
                        "team_id": player_id,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_teams_data(self):
        """Teams data."""
        return pd.DataFrame(
            {
                "team_id": [1, 2, 3],
                "name": ["Team1", "Team2", "Team3"],
            }
        )

    @pytest.fixture
    def sample_fixtures_data(self):
        """Fixtures for GW9."""
        return pd.DataFrame(
            {
                "event": [9, 9, 9],
                "home_team_id": [1, 2, 3],
                "away_team_id": [3, 1, 2],
            }
        )

    @pytest.fixture
    def sample_ownership_trends(self):
        """Ownership trends data - need GW1+ for shift(1) to work."""
        data = []
        for gw in range(1, 10):  # GW1-9 to cover historical data
            for player_id in [1, 2, 3]:
                data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "selected_by_percent": 10.0 + player_id * 5,
                        "net_transfers_gw": 100 * player_id,
                        "avg_net_transfers_5gw": 80 * player_id,
                        "transfer_momentum": "rising",
                        "ownership_velocity": 0.5,
                        "ownership_tier": "popular",
                        "bandwagon_score": 0.5,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_value_analysis(self):
        """Value analysis data - need GW1+ for shift(1) to work."""
        data = []
        for gw in range(1, 10):  # GW1-9
            for player_id in [1, 2, 3]:
                data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "points_per_pound": 1.0,
                        "value_vs_position": 1.0,
                        "predicted_price_change_1gw": 0.0,
                        "price_volatility": 0.0,
                        "price_risk": 0.0,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_fixture_difficulty(self):
        """Enhanced fixture difficulty data - need GW1+ for shift(1) to work."""
        data = []
        for gw in range(1, 10):  # GW1-9
            for team_id in [1, 2, 3]:
                data.append(
                    {
                        "team_id": team_id,
                        "gameweek": gw,
                        "congestion_difficulty": 1.0,
                        "form_difficulty": 1.0,
                        "clean_sheet_probability": 0.3,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_betting_features(self):
        """Betting odds features data."""
        data = []
        for gw in range(1, 10):  # GW1-9
            for player_id in [1, 2, 3]:
                data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        # Implied probabilities (6)
                        "team_win_probability": 0.40 + player_id * 0.1,
                        "opponent_win_probability": 0.30,
                        "draw_probability": 0.30,
                        "implied_clean_sheet_probability": 0.35,
                        "implied_total_goals": 2.5,
                        "team_expected_goals": 1.3,
                        # Market confidence (4)
                        "market_consensus_strength": 0.6,
                        "odds_movement_team": 0.0,
                        "odds_movement_magnitude": 0.1,
                        "favorite_status": 1.0 if player_id == 3 else 0.0,
                        # Asian Handicap (3)
                        "asian_handicap_line": -0.5 if player_id == 3 else 0.5,
                        "handicap_team_odds": 1.9,
                        "expected_goal_difference": -0.35 if player_id == 3 else 0.35,
                        # Match context (2)
                        "over_under_signal": 0.1,
                        "referee_encoded": 5,
                    }
                )
        return pd.DataFrame(data)

    def test_feature_engineer_produces_117_features_with_betting_odds(
        self,
        sample_historical_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_ownership_trends,
        sample_value_analysis,
        sample_fixture_difficulty,
        sample_betting_features,
    ):
        """Test that FPLFeatureEngineer produces exactly 122 features (118 + 4 elite interactions)."""
        from fpl_team_picker.domain.services.ml_feature_engineering import (
            FPLFeatureEngineer,
        )

        engineer = FPLFeatureEngineer(
            fixtures_df=sample_fixtures_data,
            teams_df=sample_teams_data,
            team_strength={"Team1": 1.0, "Team2": 1.0, "Team3": 1.0},
            ownership_trends_df=sample_ownership_trends,
            value_analysis_df=sample_value_analysis,
            fixture_difficulty_df=sample_fixture_difficulty,
            betting_features_df=sample_betting_features,
            # Phase 1-3 data sources not provided - will use defaults
        )

        result = engineer.fit_transform(
            sample_historical_data, sample_historical_data["total_points"]
        )

        # Should have exactly 122 features (118 + 4 elite interactions)
        assert result.shape[1] == 122, f"Expected 122 features, got {result.shape[1]}"

        # Verify betting odds features are present
        betting_features = [
            "team_win_probability",
            "opponent_win_probability",
            "draw_probability",
            "implied_clean_sheet_probability",
            "implied_total_goals",
            "team_expected_goals",
            "market_consensus_strength",
            "odds_movement_team",
            "odds_movement_magnitude",
            "favorite_status",
            "asian_handicap_line",
            "handicap_team_odds",
            "expected_goal_difference",
            "over_under_signal",
            "referee_encoded",
        ]
        for feature in betting_features:
            assert feature in result.columns, f"Missing betting feature: {feature}"

    def test_feature_engineer_fails_without_betting_data(
        self,
        sample_historical_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_ownership_trends,
        sample_value_analysis,
        sample_fixture_difficulty,
    ):
        """Test that FPLFeatureEngineer fails fast without betting data (FAIL FAST principle)."""
        from fpl_team_picker.domain.services.ml_feature_engineering import (
            FPLFeatureEngineer,
        )

        engineer = FPLFeatureEngineer(
            fixtures_df=sample_fixtures_data,
            teams_df=sample_teams_data,
            team_strength={"Team1": 1.0, "Team2": 1.0, "Team3": 1.0},
            ownership_trends_df=sample_ownership_trends,
            value_analysis_df=sample_value_analysis,
            fixture_difficulty_df=sample_fixture_difficulty,
            betting_features_df=None,  # No betting data - should fail
        )

        # Should fail with clear error message
        with pytest.raises(ValueError, match="betting_features_df is empty"):
            engineer.fit_transform(
                sample_historical_data, sample_historical_data["total_points"]
            )
