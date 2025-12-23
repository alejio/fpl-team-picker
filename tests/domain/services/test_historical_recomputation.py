"""Integration tests for historical xP recomputation framework."""

import pytest
import pandas as pd
from fpl_team_picker.domain.services.performance_analytics_service import (
    PerformanceAnalyticsService,
    ALGORITHM_VERSIONS,
)
from fpl_team_picker.domain.services.data_orchestration_service import (
    DataOrchestrationService,
)


class TestHistoricalRecomputation:
    """Test suite for historical xP recomputation functionality."""

    def test_load_historical_gameweek_state_valid(self):
        """Test loading historical gameweek state with valid data."""
        orchestration_service = DataOrchestrationService()

        # Load historical state for GW1
        historical_data = orchestration_service.load_historical_gameweek_state(
            target_gameweek=1, form_window=5
        )

        # Validate structure
        assert "players" in historical_data
        assert "teams" in historical_data
        assert "fixtures" in historical_data
        assert "xg_rates" in historical_data
        assert "gameweek_info" in historical_data
        assert "target_gameweek" in historical_data

        # Validate data types
        assert isinstance(historical_data["players"], pd.DataFrame)
        assert isinstance(historical_data["teams"], pd.DataFrame)
        assert isinstance(historical_data["fixtures"], pd.DataFrame)

        # Validate gameweek info
        assert historical_data["target_gameweek"] == 1
        assert historical_data["gameweek_info"]["status"] == "historical"

    def test_load_historical_gameweek_state_invalid_gameweek(self):
        """Test that invalid gameweek raises ValueError."""
        orchestration_service = DataOrchestrationService()

        # Test gameweek 0 (invalid)
        with pytest.raises(ValueError, match="Invalid target gameweek"):
            orchestration_service.load_historical_gameweek_state(target_gameweek=0)

        # Test gameweek 39 (invalid)
        with pytest.raises(ValueError, match="Invalid target gameweek"):
            orchestration_service.load_historical_gameweek_state(target_gameweek=39)

    def test_recompute_historical_xp_single_gameweek(self):
        """Test recomputing xP for a single historical gameweek."""
        analytics_service = PerformanceAnalyticsService()

        # Recompute GW1 with current algorithm
        result = analytics_service.recompute_historical_xp(
            target_gameweek=1, algorithm_version="current"
        )

        # Validate result structure
        assert isinstance(result, pd.DataFrame)
        assert "player_id" in result.columns
        assert "xP" in result.columns
        assert "algorithm_version" in result.columns
        assert "computed_at" in result.columns
        assert "target_gameweek" in result.columns

        # Validate metadata
        assert (result["algorithm_version"] == "current").all()
        assert (result["target_gameweek"] == 1).all()

        # Validate xP values are reasonable
        assert result["xP"].min() >= 0
        assert result["xP"].max() <= 20  # No player should have >20 xP

    def test_recompute_historical_xp_invalid_algorithm(self):
        """Test that invalid algorithm version raises ValueError."""
        analytics_service = PerformanceAnalyticsService()

        with pytest.raises(ValueError, match="Unknown algorithm version"):
            analytics_service.recompute_historical_xp(
                target_gameweek=1, algorithm_version="nonexistent_algo"
            )

    def test_batch_recompute_season_multiple_gameweeks(self):
        """Test batch recomputation across multiple gameweeks."""
        analytics_service = PerformanceAnalyticsService()

        # Batch recompute GW1-2 with current algorithm
        result = analytics_service.batch_recompute_season(
            start_gw=1,
            end_gw=2,
            algorithm_versions=["current"],
        )

        # Validate result structure
        assert isinstance(result, pd.DataFrame)
        assert result.index.names == [
            "target_gameweek",
            "player_id",
            "algorithm_version",
        ]

        # Validate gameweeks present
        gameweeks = result.index.get_level_values("target_gameweek").unique()
        assert 1 in gameweeks
        assert 2 in gameweeks

    def test_batch_recompute_season_multiple_algorithms(self):
        """Test batch recomputation with multiple algorithm versions."""
        analytics_service = PerformanceAnalyticsService()

        # Batch recompute GW1 with multiple algorithms
        result = analytics_service.batch_recompute_season(
            start_gw=1,
            end_gw=1,
            algorithm_versions=["current", "experimental_high_form"],
        )

        # Validate algorithms present
        algorithms = result.index.get_level_values("algorithm_version").unique()
        assert "current" in algorithms
        assert "experimental_high_form" in algorithms

    def test_batch_recompute_season_invalid_range(self):
        """Test that invalid gameweek range raises ValueError."""
        analytics_service = PerformanceAnalyticsService()

        # Test start > end
        with pytest.raises(ValueError, match="Invalid gameweek range"):
            analytics_service.batch_recompute_season(start_gw=5, end_gw=1)

        # Test invalid start
        with pytest.raises(ValueError, match="Invalid gameweek range"):
            analytics_service.batch_recompute_season(start_gw=0, end_gw=5)

    def test_calculate_accuracy_metrics_basic(self):
        """Test accuracy metrics calculation with mock data."""
        analytics_service = PerformanceAnalyticsService()

        # Create mock predictions and actual results
        predictions = pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4],
                "web_name": ["Player A", "Player B", "Player C", "Player D"],
                "position": ["FWD", "MID", "DEF", "GKP"],
                "xP": [6.0, 5.0, 4.0, 3.0],
            }
        )

        actual_results = pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4],
                "total_points": [8, 4, 5, 2],
            }
        )

        # Calculate metrics
        metrics = analytics_service.calculate_accuracy_metrics(
            predictions, actual_results, by_position=True
        )

        # Validate overall metrics exist
        assert "overall" in metrics
        assert "mae" in metrics["overall"]
        assert "rmse" in metrics["overall"]
        assert "correlation" in metrics["overall"]
        assert "player_count" in metrics

        # Validate position metrics exist
        assert "by_position" in metrics
        assert len(metrics["by_position"]) > 0

    def test_calculate_accuracy_metrics_no_matching_players(self):
        """Test accuracy metrics when no players match."""
        analytics_service = PerformanceAnalyticsService()

        # Create predictions and results with no overlap
        predictions = pd.DataFrame(
            {
                "player_id": [1, 2],
                "web_name": ["Player A", "Player B"],
                "position": ["FWD", "MID"],
                "xP": [6.0, 5.0],
            }
        )

        actual_results = pd.DataFrame(
            {
                "player_id": [3, 4],
                "total_points": [8, 4],
            }
        )

        # Calculate metrics - should return error
        metrics = analytics_service.calculate_accuracy_metrics(
            predictions, actual_results
        )

        assert "error" in metrics
        assert metrics["player_count"] == 0

    def test_algorithm_versions_registry(self):
        """Test that algorithm version registry is properly configured."""
        # Validate registry exists and has expected versions
        assert "v1.0" in ALGORITHM_VERSIONS
        assert "current" in ALGORITHM_VERSIONS
        assert "experimental_high_form" in ALGORITHM_VERSIONS
        assert "experimental_low_form" in ALGORITHM_VERSIONS

        # Validate algorithm structure
        for version_name, algo_version in ALGORITHM_VERSIONS.items():
            assert algo_version.name == version_name
            assert 0.0 <= algo_version.form_weight <= 1.0
            assert 1 <= algo_version.form_window <= 10
            assert isinstance(algo_version.use_team_strength, bool)
            assert isinstance(algo_version.team_strength_params, dict)

    def test_historical_prices_loaded(self):
        """Test that historical prices are correctly loaded from gameweek performance."""
        orchestration_service = DataOrchestrationService()

        # Load historical state for GW2 (should have prices from GW1)
        historical_data = orchestration_service.load_historical_gameweek_state(
            target_gameweek=2
        )

        players = historical_data["players"]

        # Validate price column exists
        assert "now_cost" in players.columns

        # Validate prices are reasonable (between 3.8 and 20.0 typically)
        assert (
            players["now_cost"].min() >= 38
        )  # API stores as 10x (£3.8m is valid minimum)
        assert players["now_cost"].max() <= 200
