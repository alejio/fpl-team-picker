"""Integration tests for PerformanceAnalyticsService."""

import pytest
import pandas as pd

from fpl_team_picker.domain.services import (
    DataOrchestrationService,
    ExpectedPointsService,
)
from fpl_team_picker.domain.services.performance_analytics_service import (
    PerformanceAnalyticsService,
)


class TestPerformanceAnalyticsServiceIntegration:
    """Integration tests for PerformanceAnalyticsService with real data."""

    @pytest.fixture
    def analytics_service(self):
        """Create performance analytics service."""
        return PerformanceAnalyticsService()

    @pytest.fixture
    def sample_gameweek_data(self):
        """Load sample gameweek data with XP calculations."""
        # Create services
        data_service = DataOrchestrationService()
        xp_service = ExpectedPointsService()

        # Load gameweek data
        gameweek_data = data_service.load_gameweek_data(
            target_gameweek=1, form_window=3
        )

        # Calculate XP
        players_with_xp = xp_service.calculate_combined_results(
            gameweek_data, use_ml_model=False
        )
        gameweek_data["players_with_xp"] = players_with_xp

        return gameweek_data

    def test_statistical_insights(self, analytics_service, sample_gameweek_data):
        """Test statistical insights generation."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Test statistical insights (should work even without form data)
        insights = analytics_service.get_statistical_insights(players_with_xp)
        assert isinstance(insights, dict)
        assert "xp_distribution" in insights
        assert insights["xp_distribution"]["mean"] > 0
        assert insights["xp_distribution"]["max"] > insights["xp_distribution"]["min"]
        assert insights["xp_distribution"]["std"] >= 0

    def test_position_trends_analysis(self, analytics_service, sample_gameweek_data):
        """Test position trends analysis."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Test position trends for all positions
        position_data = analytics_service.analyze_position_trends(players_with_xp)
        assert isinstance(position_data, dict)
        assert "position_analysis" in position_data
        assert len(position_data["position_analysis"]) > 0

        # Check that we have data for standard positions
        positions = position_data["position_analysis"].keys()
        expected_positions = {"GKP", "DEF", "MID", "FWD"}
        assert expected_positions.intersection(positions), (
            "Should have at least some standard positions"
        )

    def test_specific_position_analysis(self, analytics_service, sample_gameweek_data):
        """Test analysis for a specific position."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Test specific position analysis
        mid_data = analytics_service.analyze_position_trends(
            players_with_xp, position="MID"
        )
        assert isinstance(mid_data, dict)
        assert "position_analysis" in mid_data
        assert "MID" in mid_data["position_analysis"]

        mid_stats = mid_data["position_analysis"]["MID"]
        assert "total_players" in mid_stats
        assert "avg_xp" in mid_stats
        assert "top_player" in mid_stats
        assert mid_stats["total_players"] > 0

    def test_breakout_player_detection(self, analytics_service, sample_gameweek_data):
        """Test breakout player detection."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Test breakout player detection
        breakout_players = analytics_service.detect_breakout_players(
            players_with_xp, price_threshold=7.0, xp_threshold=5.0
        )
        assert isinstance(breakout_players, list)

        # Check that all breakout players meet criteria
        for player in breakout_players:
            assert player["price"] <= 7.0
            assert player["xP"] >= 5.0

    def test_form_analysis_without_form_data(
        self, analytics_service, sample_gameweek_data
    ):
        """Test form analysis when form data is not available."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Remove form columns if they exist to test fallback behavior
        players_no_form = players_with_xp.drop(
            columns=["momentum", "form_multiplier"], errors="ignore"
        )

        # This should work or return empty results when no form data is available
        form_result = analytics_service.analyze_player_form(players_no_form)
        assert isinstance(form_result, dict)

    def test_error_handling(self, analytics_service):
        """Test error handling in analytics service."""
        empty_df = pd.DataFrame()

        # Test with empty data - should raise KeyError for missing columns
        with pytest.raises(KeyError):
            analytics_service.get_statistical_insights(empty_df)

        # Test with invalid position but proper column structure
        df_with_columns = pd.DataFrame(
            {"xP": [], "position": [], "price": [], "web_name": []}
        )

        # Invalid position should handle gracefully or return empty results
        result = analytics_service.analyze_position_trends(
            df_with_columns, position="INVALID"
        )
        assert isinstance(result, dict)  # Should return something, even if empty

        # Test breakout detection with empty data - should raise exception
        with pytest.raises((KeyError, ValueError)):
            analytics_service.detect_breakout_players(empty_df)

    def test_price_performance_correlation(
        self, analytics_service, sample_gameweek_data
    ):
        """Test price-performance correlation analysis."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Ensure we have price column
        if "price" not in players_with_xp.columns:
            pytest.skip("No price data available for correlation test")

        insights = analytics_service.get_statistical_insights(players_with_xp)
        if "price_performance_correlation" in insights:
            correlation = insights["price_performance_correlation"]
            # Correlation should be a valid number between -1 and 1
            assert -1 <= correlation <= 1
            assert not pd.isna(correlation)


class TestAccuracyTracking:
    """Tests for model accuracy tracking and recomputation functionality."""

    @pytest.fixture
    def analytics_service(self):
        """Create performance analytics service."""
        return PerformanceAnalyticsService()

    def test_recompute_historical_xp(self, analytics_service):
        """Test historical xP recomputation for a single gameweek."""
        # Test recomputation for GW7 (where we have complete historical data)
        # Note: Using include_snapshots=False since snapshots aren't available for past GWs
        predictions = analytics_service.recompute_historical_xp(
            target_gameweek=7, algorithm_version="current", include_snapshots=False
        )

        # Validate results structure
        assert isinstance(predictions, pd.DataFrame)
        assert not predictions.empty
        assert "player_id" in predictions.columns
        assert "xP" in predictions.columns
        assert "algorithm_version" in predictions.columns
        assert "computed_at" in predictions.columns
        assert "target_gameweek" in predictions.columns

        # Validate metadata
        assert all(predictions["algorithm_version"] == "current")
        assert all(predictions["target_gameweek"] == 7)

        # Validate xP values are reasonable
        assert predictions["xP"].min() >= 0
        assert predictions["xP"].max() <= 20  # Reasonable upper bound

    def test_recompute_invalid_algorithm(self, analytics_service):
        """Test that invalid algorithm version raises error."""
        with pytest.raises(ValueError, match="Unknown algorithm version"):
            analytics_service.recompute_historical_xp(
                target_gameweek=7,
                algorithm_version="invalid_algo",
                include_snapshots=False,
            )

    def test_batch_recompute_season(self, analytics_service):
        """Test batch recomputation across multiple gameweeks."""
        # Test batch recomputation for GW7 with current algorithm
        # Note: Using include_snapshots=False since snapshots aren't available for past GWs
        # Note: GW6 has data quality issues, so testing with GW7 only
        predictions_df = analytics_service.batch_recompute_season(
            start_gw=7,
            end_gw=7,
            algorithm_versions=["current"],
            include_snapshots=False,
        )

        # Validate structure
        assert isinstance(predictions_df, pd.DataFrame)
        assert not predictions_df.empty

        # Check multi-index structure
        assert predictions_df.index.names == [
            "target_gameweek",
            "player_id",
            "algorithm_version",
        ]

        # Check we have data for gameweek 7
        gameweeks = predictions_df.index.get_level_values("target_gameweek").unique()
        assert 7 in gameweeks

    def test_batch_recompute_multiple_algorithms(self, analytics_service):
        """Test batch recomputation with multiple algorithm versions."""
        # Note: Using include_snapshots=False since snapshots aren't available for past GWs
        predictions_df = analytics_service.batch_recompute_season(
            start_gw=7,
            end_gw=7,
            algorithm_versions=["current", "experimental_high_form"],
            include_snapshots=False,
        )

        # Validate we have results for both algorithms
        algorithms = predictions_df.index.get_level_values("algorithm_version").unique()
        assert "current" in algorithms
        assert "experimental_high_form" in algorithms

    def test_batch_recompute_invalid_gameweek_range(self, analytics_service):
        """Test that invalid gameweek range raises error."""
        with pytest.raises(ValueError, match="Invalid gameweek range"):
            analytics_service.batch_recompute_season(
                start_gw=10, end_gw=5, algorithm_versions=["current"]
            )

        with pytest.raises(ValueError, match="Invalid gameweek range"):
            analytics_service.batch_recompute_season(
                start_gw=0, end_gw=5, algorithm_versions=["current"]
            )

        with pytest.raises(ValueError, match="Invalid gameweek range"):
            analytics_service.batch_recompute_season(
                start_gw=1, end_gw=39, algorithm_versions=["current"]
            )

    def test_calculate_accuracy_metrics(self, analytics_service):
        """Test accuracy metrics calculation."""
        from client import FPLDataClient

        client = FPLDataClient()

        # Recompute predictions for GW7
        # Note: Using include_snapshots=False since snapshots aren't available for past GWs
        predictions = analytics_service.recompute_historical_xp(
            target_gameweek=7, algorithm_version="current", include_snapshots=False
        )

        # Get actual results
        actual_results = client.get_gameweek_performance(7)

        # Calculate accuracy metrics
        metrics = analytics_service.calculate_accuracy_metrics(
            predictions, actual_results, by_position=True
        )

        # Validate overall metrics
        assert "overall" in metrics
        assert "mae" in metrics["overall"]
        assert "rmse" in metrics["overall"]
        assert "correlation" in metrics["overall"]
        assert "mean_predicted" in metrics["overall"]
        assert "mean_actual" in metrics["overall"]

        # Validate metric values
        assert metrics["overall"]["mae"] >= 0
        assert metrics["overall"]["rmse"] >= 0
        assert -1 <= metrics["overall"]["correlation"] <= 1

        # Validate position-specific metrics
        assert "by_position" in metrics
        for position in ["GKP", "DEF", "MID", "FWD"]:
            if position in metrics["by_position"]:
                pos_metrics = metrics["by_position"][position]
                assert "mae" in pos_metrics
                assert "rmse" in pos_metrics
                assert "correlation" in pos_metrics
                assert pos_metrics["mae"] >= 0

    def test_accuracy_metrics_with_no_matching_players(self, analytics_service):
        """Test accuracy metrics when no players match."""
        # Create empty predictions and actuals
        predictions = pd.DataFrame(
            {
                "player_id": [9999],
                "xP": [5.0],
                "position": ["MID"],
                "web_name": ["Test"],
            }
        )

        actual_results = pd.DataFrame(
            {"player_id": [8888], "total_points": [10]}  # Different player ID
        )

        metrics = analytics_service.calculate_accuracy_metrics(
            predictions, actual_results, by_position=True
        )

        # Should return error
        assert "error" in metrics
        assert metrics["player_count"] == 0

    def test_algorithm_versions_registry(self):
        """Test that algorithm versions are properly registered."""
        from fpl_team_picker.domain.services.performance_analytics_service import (
            ALGORITHM_VERSIONS,
        )

        # Check required versions exist
        assert "current" in ALGORITHM_VERSIONS
        assert "v1.0" in ALGORITHM_VERSIONS
        assert "experimental_high_form" in ALGORITHM_VERSIONS
        assert "experimental_low_form" in ALGORITHM_VERSIONS

        # Validate algorithm configurations
        for name, algo in ALGORITHM_VERSIONS.items():
            assert algo.name == name
            assert 0.0 <= algo.form_weight <= 1.0
            assert 1 <= algo.form_window <= 10
            assert isinstance(algo.use_team_strength, bool)
            assert isinstance(algo.team_strength_params, dict)
            assert isinstance(algo.minutes_model_params, dict)
            assert isinstance(algo.statistical_estimation_params, dict)

    def test_position_specific_accuracy(self, analytics_service):
        """Test position-specific accuracy analysis."""
        from client import FPLDataClient

        client = FPLDataClient()

        # Recompute predictions for GW7
        # Note: Using include_snapshots=False since snapshots aren't available for past GWs
        predictions = analytics_service.recompute_historical_xp(
            target_gameweek=7, algorithm_version="current", include_snapshots=False
        )

        # Get actual results
        actual_results = client.get_gameweek_performance(7)

        # Calculate position-specific metrics
        metrics = analytics_service.calculate_accuracy_metrics(
            predictions, actual_results, by_position=True
        )

        # Validate we have position breakdowns
        if "by_position" in metrics:
            positions_analyzed = set(metrics["by_position"].keys())

            # Should have at least some positions
            assert len(positions_analyzed) > 0

            # Each position should have complete metrics
            for pos in positions_analyzed:
                pos_data = metrics["by_position"][pos]
                assert "mae" in pos_data
                assert "rmse" in pos_data
                assert "correlation" in pos_data
                assert "player_count" in pos_data
                assert pos_data["player_count"] > 0
