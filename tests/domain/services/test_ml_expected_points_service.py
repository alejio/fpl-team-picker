"""
Tests for MLExpectedPointsService - Focus on target gameweek filtering logic.

Critical test scenarios:
1. Filtering out target gameweek from live_data prevents duplicates
2. Ownership merge works correctly after filtering
3. Predictions use correct historical context (GW1-N-1 to predict GW N)
"""

import pandas as pd
import pytest
from pathlib import Path


class TestMLExpectedPointsServiceFiltering:
    """Test that ML service correctly filters target gameweek from live_data."""

    @pytest.fixture
    def mock_model_path(self):
        """Get the most recent TPOT model for testing."""
        import glob

        # Find most recent TPOT model
        models = glob.glob("models/tpot/*.joblib")
        if not models:
            pytest.skip("No ML models available for testing")

        # Use most recent model
        model_path = Path(sorted(models)[-1])
        return model_path

    @pytest.fixture
    def sample_live_data_with_target_gw(self):
        """Create sample live data that INCLUDES target gameweek (GW9)."""
        # Simulate GW1-9 historical data for 3 players
        data = []
        for gw in range(1, 10):  # GW1-9
            for player_id in [100, 101, 102]:
                data.append({
                    "player_id": player_id,
                    "gameweek": gw,
                    "total_points": 5 if gw < 9 else 8,  # GW9 has actual points!
                    "minutes": 90,
                    "goals_scored": 0,
                    "assists": 0,
                    "clean_sheets": 1 if player_id == 100 else 0,
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
                    "value": 50,
                    "position": "DEF" if player_id == 100 else "MID",
                    "team_id": 1,
                })
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_live_data_without_target_gw(self):
        """Create sample live data that does NOT include target gameweek (GW1-8 only)."""
        # Simulate GW1-8 historical data for 3 players
        data = []
        for gw in range(1, 9):  # GW1-8
            for player_id in [100, 101, 102]:
                data.append({
                    "player_id": player_id,
                    "gameweek": gw,
                    "total_points": 5,
                    "minutes": 90,
                    "goals_scored": 0,
                    "assists": 0,
                    "clean_sheets": 1 if player_id == 100 else 0,
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
                    "value": 50,
                    "position": "DEF" if player_id == 100 else "MID",
                    "team_id": 1,
                })
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_players_data(self):
        """Create sample current players data for GW9."""
        return pd.DataFrame([
            {
                "player_id": 100,
                "web_name": "Player A",
                "position": "DEF",
                "team_id": 1,
                "price": 5.0,
                "status": "a",
            },
            {
                "player_id": 101,
                "web_name": "Player B",
                "position": "MID",
                "team_id": 1,
                "price": 6.0,
                "status": "a",
            },
            {
                "player_id": 102,
                "web_name": "Player C",
                "position": "MID",
                "team_id": 1,
                "price": 7.0,
                "status": "a",
            },
        ])

    @pytest.fixture
    def sample_teams_data(self):
        """Create sample teams data."""
        return pd.DataFrame([
            {
                "team_id": 1,
                "name": "Test Team",
                "short_name": "TST",
                "strength": 4,
                "strength_overall_home": 1200,
                "strength_overall_away": 1100,
                "strength_attack_home": 1200,
                "strength_attack_away": 1100,
                "strength_defence_home": 1200,
                "strength_defence_away": 1100,
            }
        ])

    @pytest.fixture
    def sample_fixtures_data(self):
        """Create sample fixtures for GW9."""
        return pd.DataFrame([
            {
                "event": 9,
                "home_team_id": 1,
                "away_team_id": 2,
                "team_h_difficulty": 3,
                "team_a_difficulty": 3,
                "finished": False,
            }
        ])

    @pytest.fixture
    def sample_ownership_trends(self):
        """Create sample ownership trends for GW1-9."""
        data = []
        for gw in range(1, 10):  # GW1-9
            for player_id in [100, 101, 102]:
                data.append({
                    "player_id": player_id,
                    "gameweek": gw,
                    "selected_by_percent": 10.0 + gw,  # Increasing ownership
                    "net_transfers_gw": 100,
                    "avg_net_transfers_5gw": 100.0,
                    "transfer_momentum": "neutral",
                    "ownership_tier": "popular",
                    "bandwagon_score": 0.5,
                    "ownership_velocity": 1.0,
                })
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_value_analysis(self):
        """Create sample value analysis."""
        data = []
        for gw in range(1, 10):
            for player_id in [100, 101, 102]:
                data.append({
                    "player_id": player_id,
                    "gameweek": gw,
                    "points_per_pound": 1.0,
                    "value_vs_position": 0.0,
                    "predicted_price_change": 0.0,
                    "price_volatility": 0.1,
                    "price_risk": 0.0,
                })
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_fixture_difficulty(self):
        """Create sample fixture difficulty."""
        return pd.DataFrame([
            {
                "team_id": 1,
                "gameweek": 9,
                "congestion_difficulty": 3.0,
                "form_adjusted_difficulty": 3.0,
                "clean_sheet_probability_enhanced": 0.3,
            }
        ])

    def test_filtering_removes_target_gameweek_from_live_data(
        self,
        mock_model_path,
        sample_live_data_with_target_gw,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_ownership_trends,
        sample_value_analysis,
        sample_fixture_difficulty,
    ):
        """Test that target gameweek is filtered out from live_data before concatenation."""
        from fpl_team_picker.domain.services import MLExpectedPointsService

        service = MLExpectedPointsService(model_path=str(mock_model_path))

        # live_data includes GW1-9 (target is GW9)
        result = service.calculate_expected_points(
            players_data=sample_players_data,
            teams_data=sample_teams_data,
            xg_rates_data=pd.DataFrame(),
            fixtures_data=sample_fixtures_data,
            target_gameweek=9,
            live_data=sample_live_data_with_target_gw,
            gameweeks_ahead=1,
            ownership_trends_df=sample_ownership_trends,
            value_analysis_df=sample_value_analysis,
            fixture_difficulty_df=sample_fixture_difficulty,
        )

        # Should return predictions for all 3 players
        assert len(result) == 3
        assert "xP" in result.columns
        assert all(result["xP"] >= 0)  # Predictions should be non-negative

    def test_no_duplicate_gameweek_rows(
        self,
        mock_model_path,
        sample_live_data_with_target_gw,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_ownership_trends,
        sample_value_analysis,
        sample_fixture_difficulty,
    ):
        """Test that filtering prevents duplicate GW9 rows in prediction_data_all."""
        from fpl_team_picker.domain.services import MLExpectedPointsService

        service = MLExpectedPointsService(model_path=str(mock_model_path), debug=True)

        # Monkey-patch to inspect prediction_data_all
        original_predict = service.pipeline.predict

        prediction_data_captured = None

        def capture_predict(X):
            nonlocal prediction_data_captured
            prediction_data_captured = X.copy()
            return original_predict(X)

        service.pipeline.predict = capture_predict

        # Run prediction
        service.calculate_expected_points(
            players_data=sample_players_data,
            teams_data=sample_teams_data,
            xg_rates_data=pd.DataFrame(),
            fixtures_data=sample_fixtures_data,
            target_gameweek=9,
            live_data=sample_live_data_with_target_gw,
            gameweeks_ahead=1,
            ownership_trends_df=sample_ownership_trends,
            value_analysis_df=sample_value_analysis,
            fixture_difficulty_df=sample_fixture_difficulty,
        )

        # Check that prediction_data_all has no duplicate (player_id, gameweek) pairs
        assert prediction_data_captured is not None

        # Before feature engineering, check raw input
        gw_col = "gameweek" if "gameweek" in prediction_data_captured.columns else "event"
        if gw_col in prediction_data_captured.columns and "player_id" in prediction_data_captured.columns:
            duplicates = prediction_data_captured.duplicated(subset=["player_id", gw_col], keep=False)
            duplicate_rows = prediction_data_captured[duplicates]

            assert not duplicates.any(), (
                f"Found duplicate (player_id, {gw_col}) rows:\n{duplicate_rows}"
            )

    def test_both_scenarios_produce_same_predictions(
        self,
        mock_model_path,
        sample_live_data_with_target_gw,
        sample_live_data_without_target_gw,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_ownership_trends,
        sample_value_analysis,
        sample_fixture_difficulty,
    ):
        """Test that predictions are the same whether target GW is in live_data or not."""
        from fpl_team_picker.domain.services import MLExpectedPointsService

        service = MLExpectedPointsService(model_path=str(mock_model_path))

        # Scenario 1: live_data includes GW9
        result_with_gw9 = service.calculate_expected_points(
            players_data=sample_players_data,
            teams_data=sample_teams_data,
            xg_rates_data=pd.DataFrame(),
            fixtures_data=sample_fixtures_data,
            target_gameweek=9,
            live_data=sample_live_data_with_target_gw,
            gameweeks_ahead=1,
            ownership_trends_df=sample_ownership_trends,
            value_analysis_df=sample_value_analysis,
            fixture_difficulty_df=sample_fixture_difficulty,
        )

        # Scenario 2: live_data does NOT include GW9
        result_without_gw9 = service.calculate_expected_points(
            players_data=sample_players_data,
            teams_data=sample_teams_data,
            xg_rates_data=pd.DataFrame(),
            fixtures_data=sample_fixtures_data,
            target_gameweek=9,
            live_data=sample_live_data_without_target_gw,
            gameweeks_ahead=1,
            ownership_trends_df=sample_ownership_trends,
            value_analysis_df=sample_value_analysis,
            fixture_difficulty_df=sample_fixture_difficulty,
        )

        # Predictions should be identical (filtering makes them equivalent)
        pd.testing.assert_frame_equal(
            result_with_gw9[["player_id", "xP"]].sort_values("player_id").reset_index(drop=True),
            result_without_gw9[["player_id", "xP"]].sort_values("player_id").reset_index(drop=True),
            check_dtype=False,
        )

    def test_ownership_merge_works_after_filtering(
        self,
        mock_model_path,
        sample_live_data_with_target_gw,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_ownership_trends,
        sample_value_analysis,
        sample_fixture_difficulty,
    ):
        """Test that ownership data merges correctly after target GW filtering."""
        from fpl_team_picker.domain.services import MLExpectedPointsService

        service = MLExpectedPointsService(model_path=str(mock_model_path))

        # This should not raise ValueError about missing ownership data
        result = service.calculate_expected_points(
            players_data=sample_players_data,
            teams_data=sample_teams_data,
            xg_rates_data=pd.DataFrame(),
            fixtures_data=sample_fixtures_data,
            target_gameweek=9,
            live_data=sample_live_data_with_target_gw,
            gameweeks_ahead=1,
            ownership_trends_df=sample_ownership_trends,
            value_analysis_df=sample_value_analysis,
            fixture_difficulty_df=sample_fixture_difficulty,
        )

        # Should complete successfully without ownership errors
        assert len(result) == 3
        assert "xP" in result.columns
