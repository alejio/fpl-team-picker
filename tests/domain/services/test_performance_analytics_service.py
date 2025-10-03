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
from fpl_team_picker.adapters.database_repositories import (
    DatabasePlayerRepository,
    DatabaseTeamRepository,
    DatabaseFixtureRepository,
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
        # Initialize repositories
        player_repo = DatabasePlayerRepository()
        team_repo = DatabaseTeamRepository()
        fixture_repo = DatabaseFixtureRepository()

        # Create services
        data_service = DataOrchestrationService(player_repo, team_repo, fixture_repo)
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
