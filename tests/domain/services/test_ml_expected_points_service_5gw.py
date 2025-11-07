"""
Tests for 5-gameweek cascading predictions in MLExpectedPointsService.

Tests the cascading prediction approach where:
1. GW+1 is predicted using actual historical data
2. GW+2 is predicted using historical data + predicted GW+1 points
3. GW+3-5 continue the cascade
4. All 5 predictions are summed to get total 5gw xP
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from fpl_team_picker.domain.services.ml_expected_points_service import (
    MLExpectedPointsService,
)


class TestCascading5GWPredictions:
    """Test cascading 5-gameweek prediction functionality."""

    @pytest.fixture
    def sample_players_data(self):
        """Sample current player data."""
        return pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "web_name": ["Player1", "Player2", "Player3"],
                "position": ["DEF", "MID", "FWD"],
                "team": ["Team1", "Team2", "Team3"],
                "team_id": [1, 2, 3],
                "price": [5.0, 8.0, 10.0],
            }
        )

    @pytest.fixture
    def sample_teams_data(self):
        """Sample teams data."""
        return pd.DataFrame(
            {
                "team_id": [1, 2, 3],
                "name": ["Team1", "Team2", "Team3"],
            }
        )

    @pytest.fixture
    def sample_fixtures_data(self):
        """Sample fixtures data for GW11-15."""
        data = []
        for gw in range(11, 16):
            data.append(
                {
                    "gameweek": gw,
                    "event": gw,
                    "team_h": 1,
                    "team_a": 2,
                }
            )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_historical_data(self):
        """Historical live data for GW1-10."""
        data = []
        for gw in range(1, 11):
            for player_id in [1, 2, 3]:
                data.append(
                    {
                        "player_id": player_id,
                        "gameweek": gw,
                        "event": gw,
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
    def mock_ml_service(self, sample_historical_data):
        """Create a mock ML service with pipeline."""
        service = MLExpectedPointsService(debug=False)
        # Mock the pipeline
        service.pipeline = Mock()
        service.pipeline.named_steps = {"feature_engineer": Mock()}
        return service

    def test_calculate_5gw_expected_points_basic(
        self,
        mock_ml_service,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_historical_data,
    ):
        """Test basic 5gw cascading prediction functionality."""

        # Mock calculate_expected_points to return predictable results
        def mock_calculate_expected_points(*args, **kwargs):
            target_gw = kwargs.get("target_gameweek", args[4] if len(args) > 4 else 11)
            result = sample_players_data.copy()
            # Return different xP for each gameweek to verify cascade
            result["ml_xP"] = [5.0 + target_gw, 6.0 + target_gw, 7.0 + target_gw]
            result["xP"] = result["ml_xP"]
            result["xP_uncertainty"] = [1.0, 1.5, 2.0]
            return result

        mock_ml_service.calculate_expected_points = Mock(
            side_effect=mock_calculate_expected_points
        )

        # Call 5gw method
        result = mock_ml_service.calculate_5gw_expected_points(
            players_data=sample_players_data,
            teams_data=sample_teams_data,
            xg_rates_data=pd.DataFrame(),
            fixtures_data=sample_fixtures_data,
            target_gameweek=10,
            live_data=sample_historical_data,
        )

        # Verify it was called 5 times (once for each gameweek)
        assert mock_ml_service.calculate_expected_points.call_count == 5

        # Verify 5gw xP is sum of all 5 gameweeks
        # GW11: [5+11, 6+11, 7+11] = [16, 17, 18]
        # GW12: [5+12, 6+12, 7+12] = [17, 18, 19]
        # GW13: [5+13, 6+13, 7+13] = [18, 19, 20]
        # GW14: [5+14, 6+14, 7+14] = [19, 20, 21]
        # GW15: [5+15, 6+15, 7+15] = [20, 21, 22]
        # Sum: [90, 95, 100]
        expected_5gw = [90.0, 95.0, 100.0]
        np.testing.assert_array_almost_equal(
            result["xP_5gw"].values, expected_5gw, decimal=1
        )
        np.testing.assert_array_almost_equal(
            result["ml_xP"].values, expected_5gw, decimal=1
        )

        # Verify per-gameweek predictions are included
        assert "xP_gw11" in result.columns
        assert "xP_gw12" in result.columns
        assert "xP_gw13" in result.columns
        assert "xP_gw14" in result.columns
        assert "xP_gw15" in result.columns

        # Verify uncertainty columns
        assert "uncertainty_gw11" in result.columns
        assert "uncertainty_gw15" in result.columns

    def test_uncertainty_combination(
        self,
        mock_ml_service,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_historical_data,
    ):
        """Test that uncertainties are properly combined using sqrt(sum of variances)."""

        # Mock calculate_expected_points to return fixed uncertainties
        def mock_calculate_expected_points(*args, **kwargs):
            result = sample_players_data.copy()
            result["ml_xP"] = [5.0, 6.0, 7.0]
            result["xP"] = result["ml_xP"]
            result["xP_uncertainty"] = [2.0, 3.0, 4.0]  # Fixed uncertainties
            return result

        mock_ml_service.calculate_expected_points = Mock(
            side_effect=mock_calculate_expected_points
        )

        result = mock_ml_service.calculate_5gw_expected_points(
            players_data=sample_players_data,
            teams_data=sample_teams_data,
            xg_rates_data=pd.DataFrame(),
            fixtures_data=sample_fixtures_data,
            target_gameweek=10,
            live_data=sample_historical_data,
        )

        # Uncertainty should be sqrt(sum of variances)
        # For player 1: sqrt(2^2 + 2^2 + 2^2 + 2^2 + 2^2) = sqrt(20) ≈ 4.47
        # For player 2: sqrt(3^2 * 5) = sqrt(45) ≈ 6.71
        # For player 3: sqrt(4^2 * 5) = sqrt(80) ≈ 8.94
        expected_uncertainties = [
            np.sqrt(5 * 2.0**2),
            np.sqrt(5 * 3.0**2),
            np.sqrt(5 * 4.0**2),
        ]
        np.testing.assert_array_almost_equal(
            result["xP_uncertainty"].values, expected_uncertainties, decimal=2
        )

    def test_synthetic_data_creation(self, mock_ml_service, sample_players_data):
        """Test that synthetic gameweek data is created correctly from predictions."""
        # Create a mock prediction result
        predictions = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "ml_xP": [5.0, 6.0, 7.0],
                "xP_uncertainty": [1.0, 1.5, 2.0],
            }
        )

        synthetic = mock_ml_service._create_synthetic_gameweek_data(
            players_data=sample_players_data,
            predictions=predictions,
            gameweek=11,
        )

        # Verify gameweek is set
        assert (synthetic["gameweek"] == 11).all()

        # Verify total_points matches predictions
        np.testing.assert_array_almost_equal(
            synthetic["total_points"].values, [5.0, 6.0, 7.0], decimal=1
        )

        # Verify required columns exist
        required_cols = [
            "minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "bps",
            "influence",
            "creativity",
            "threat",
            "ict_index",
            "expected_goals",
            "expected_assists",
            "expected_goal_involvements",
            "expected_goals_conceded",
        ]
        for col in required_cols:
            assert col in synthetic.columns, f"Missing column: {col}"

        # Verify minutes are set (90 if points > 0)
        assert (synthetic[synthetic["total_points"] > 0]["minutes"] == 90).all()

    def test_cascading_uses_previous_predictions(
        self,
        mock_ml_service,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_historical_data,
    ):
        """Test that later predictions use synthetic data from earlier predictions."""
        call_historical_data = []

        def mock_calculate_expected_points(*args, **kwargs):
            # Capture the historical data passed to each call
            live_data = kwargs.get(
                "live_data", args[5] if len(args) > 5 else pd.DataFrame()
            )
            call_historical_data.append(len(live_data))
            kwargs.get("target_gameweek", args[4] if len(args) > 4 else 11)

            result = sample_players_data.copy()
            result["ml_xP"] = [5.0, 6.0, 7.0]
            result["xP"] = result["ml_xP"]
            result["xP_uncertainty"] = [1.0, 1.5, 2.0]
            return result

        mock_ml_service.calculate_expected_points = Mock(
            side_effect=mock_calculate_expected_points
        )

        mock_ml_service.calculate_5gw_expected_points(
            players_data=sample_players_data,
            teams_data=sample_teams_data,
            xg_rates_data=pd.DataFrame(),
            fixtures_data=sample_fixtures_data,
            target_gameweek=10,
            live_data=sample_historical_data,
        )

        # Verify that historical data grows with each call (synthetic data added)
        # Initial: 30 rows (3 players * 10 gameweeks)
        # After GW11: 33 rows (30 + 3 synthetic)
        # After GW12: 36 rows (33 + 3 synthetic)
        # etc.
        assert len(call_historical_data) == 5
        assert call_historical_data[0] == len(sample_historical_data)  # Initial size
        assert call_historical_data[1] > call_historical_data[0]  # Should grow
        assert call_historical_data[4] > call_historical_data[0]  # Should keep growing

    def test_empty_historical_data_raises_error(
        self,
        mock_ml_service,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
    ):
        """Test that empty historical data raises appropriate error."""
        with pytest.raises(ValueError, match="No historical live_data"):
            mock_ml_service.calculate_5gw_expected_points(
                players_data=sample_players_data,
                teams_data=sample_teams_data,
                xg_rates_data=pd.DataFrame(),
                fixtures_data=sample_fixtures_data,
                target_gameweek=10,
                live_data=pd.DataFrame(),  # Empty
            )

    def test_no_pipeline_raises_error(
        self,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_historical_data,
    ):
        """Test that missing pipeline raises appropriate error."""
        service = MLExpectedPointsService(debug=False)
        service.pipeline = None  # No pipeline

        with pytest.raises(ValueError, match="Pipeline not initialized"):
            service.calculate_5gw_expected_points(
                players_data=sample_players_data,
                teams_data=sample_teams_data,
                xg_rates_data=pd.DataFrame(),
                fixtures_data=sample_fixtures_data,
                target_gameweek=10,
                live_data=sample_historical_data,
            )

    def test_per_gameweek_predictions_included(
        self,
        mock_ml_service,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_historical_data,
    ):
        """Test that per-gameweek predictions are included in result."""

        def mock_calculate_expected_points(*args, **kwargs):
            target_gw = kwargs.get("target_gameweek", args[4] if len(args) > 4 else 11)
            result = sample_players_data.copy()
            result["ml_xP"] = [
                float(target_gw),
                float(target_gw + 1),
                float(target_gw + 2),
            ]
            result["xP"] = result["ml_xP"]
            result["xP_uncertainty"] = [1.0, 1.5, 2.0]
            return result

        mock_ml_service.calculate_expected_points = Mock(
            side_effect=mock_calculate_expected_points
        )

        result = mock_ml_service.calculate_5gw_expected_points(
            players_data=sample_players_data,
            teams_data=sample_teams_data,
            xg_rates_data=pd.DataFrame(),
            fixtures_data=sample_fixtures_data,
            target_gameweek=10,
            live_data=sample_historical_data,
        )

        # Verify per-gameweek columns exist
        for gw in [11, 12, 13, 14, 15]:
            assert f"xP_gw{gw}" in result.columns
            assert f"uncertainty_gw{gw}" in result.columns

        # Verify values are correct
        # GW11 should have [11, 12, 13]
        np.testing.assert_array_almost_equal(
            result["xP_gw11"].values, [11.0, 12.0, 13.0], decimal=1
        )

    def test_synthetic_data_missing_predictions(
        self, mock_ml_service, sample_players_data
    ):
        """Test synthetic data creation when predictions are missing some players."""
        # Predictions missing player 2
        predictions = pd.DataFrame(
            {
                "player_id": [1, 3],
                "ml_xP": [5.0, 7.0],
            }
        )

        synthetic = mock_ml_service._create_synthetic_gameweek_data(
            players_data=sample_players_data,
            predictions=predictions,
            gameweek=11,
        )

        # Player 2 should have 0 points (not in predictions)
        assert synthetic[synthetic["player_id"] == 2]["total_points"].iloc[0] == 0
        # Players 1 and 3 should have their predicted points
        assert synthetic[synthetic["player_id"] == 1]["total_points"].iloc[0] == 5.0
        assert synthetic[synthetic["player_id"] == 3]["total_points"].iloc[0] == 7.0

    def test_5gw_vs_1gw_multiplication_difference(
        self,
        mock_ml_service,
        sample_players_data,
        sample_teams_data,
        sample_fixtures_data,
        sample_historical_data,
    ):
        """Test that 5gw predictions differ from simple 1gw * 5 multiplication."""

        # Mock to return different xP for each gameweek
        def mock_calculate_expected_points(*args, **kwargs):
            target_gw = kwargs.get("target_gameweek", args[4] if len(args) > 4 else 11)
            result = sample_players_data.copy()
            # Vary predictions by gameweek to show cascade effect
            # GW11: [5, 6, 7], GW12: [6, 7, 8], etc.
            base_xp = [5.0, 6.0, 7.0]
            result["ml_xP"] = [b + (target_gw - 11) for b in base_xp]
            result["xP"] = result["ml_xP"]
            result["xP_uncertainty"] = [1.0, 1.5, 2.0]
            return result

        mock_ml_service.calculate_expected_points = Mock(
            side_effect=mock_calculate_expected_points
        )

        result = mock_ml_service.calculate_5gw_expected_points(
            players_data=sample_players_data,
            teams_data=sample_teams_data,
            xg_rates_data=pd.DataFrame(),
            fixtures_data=sample_fixtures_data,
            target_gameweek=10,
            live_data=sample_historical_data,
        )

        # Calculate what 1gw * 5 would be (using GW11 prediction)
        one_gw_xp = [5.0, 6.0, 7.0]  # GW11 predictions
        naive_5gw = [x * 5 for x in one_gw_xp]  # [25, 30, 35]

        # Actual 5gw should be different (sum of increasing predictions)
        # GW11: [5, 6, 7], GW12: [6, 7, 8], GW13: [7, 8, 9], GW14: [8, 9, 10], GW15: [9, 10, 11]
        # Sum: [35, 40, 45]
        actual_5gw = result["xP_5gw"].values

        # Verify they're different
        assert not np.allclose(actual_5gw, naive_5gw, rtol=0.1)

        # Verify actual is higher (since predictions increase)
        assert (actual_5gw > naive_5gw).all()
