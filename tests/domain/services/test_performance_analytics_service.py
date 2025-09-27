"""Integration tests for PerformanceAnalyticsService."""

import pytest
import pandas as pd

from fpl_team_picker.domain.services import (
    DataOrchestrationService,
    ExpectedPointsService,
)
from fpl_team_picker.domain.services.performance_analytics_service import PerformanceAnalyticsService
from fpl_team_picker.adapters.database_repositories import (
    DatabasePlayerRepository,
    DatabaseTeamRepository,
    DatabaseFixtureRepository
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
        data_result = data_service.load_gameweek_data(target_gameweek=1, form_window=3)
        assert data_result.is_success

        gameweek_data = data_result.value

        # Calculate XP
        xp_result = xp_service.calculate_combined_results(gameweek_data, use_ml_model=False)
        assert xp_result.is_success

        players_with_xp = xp_result.value
        gameweek_data["players_with_xp"] = players_with_xp

        return gameweek_data

    def test_statistical_insights(self, analytics_service, sample_gameweek_data):
        """Test statistical insights generation."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Test statistical insights (should work even without form data)
        insights_result = analytics_service.get_statistical_insights(players_with_xp)
        assert insights_result.is_success, f"Statistical insights failed: {insights_result.error.message if insights_result.error else 'Unknown'}"

        insights = insights_result.value
        assert "xp_distribution" in insights
        assert insights["xp_distribution"]["mean"] > 0
        assert insights["xp_distribution"]["max"] > insights["xp_distribution"]["min"]
        assert insights["xp_distribution"]["std"] >= 0

    def test_position_trends_analysis(self, analytics_service, sample_gameweek_data):
        """Test position trends analysis."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Test position trends for all positions
        position_result = analytics_service.analyze_position_trends(players_with_xp)
        assert position_result.is_success, f"Position trends failed: {position_result.error.message if position_result.error else 'Unknown'}"

        position_data = position_result.value
        assert "position_analysis" in position_data
        assert len(position_data["position_analysis"]) > 0

        # Check that we have data for standard positions
        positions = position_data["position_analysis"].keys()
        expected_positions = {"GKP", "DEF", "MID", "FWD"}
        assert expected_positions.intersection(positions), "Should have at least some standard positions"

    def test_specific_position_analysis(self, analytics_service, sample_gameweek_data):
        """Test analysis for a specific position."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Test specific position analysis
        mid_result = analytics_service.analyze_position_trends(players_with_xp, position="MID")
        assert mid_result.is_success, f"MID position analysis failed: {mid_result.error.message if mid_result.error else 'Unknown'}"

        mid_data = mid_result.value
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
        breakout_result = analytics_service.detect_breakout_players(
            players_with_xp, price_threshold=7.0, xp_threshold=5.0
        )
        assert breakout_result.is_success, f"Breakout detection failed: {breakout_result.error.message if breakout_result.error else 'Unknown'}"

        breakout_players = breakout_result.value
        assert isinstance(breakout_players, list)

        # Check that all breakout players meet criteria
        for player in breakout_players:
            assert player["price"] <= 7.0
            assert player["xP"] >= 5.0

    def test_form_analysis_without_form_data(self, analytics_service, sample_gameweek_data):
        """Test form analysis when form data is not available."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Remove form columns if they exist to test fallback behavior
        players_no_form = players_with_xp.drop(columns=["momentum", "form_multiplier"], errors="ignore")

        # This should fail gracefully when no form data is available
        form_result = analytics_service.analyze_player_form(players_no_form)
        assert form_result.is_failure
        assert "no form data available" in form_result.error.message.lower()

    def test_error_handling(self, analytics_service):
        """Test error handling in analytics service."""
        empty_df = pd.DataFrame()

        # Test with empty data
        result = analytics_service.get_statistical_insights(empty_df)
        assert result.is_failure
        assert "no player data" in result.error.message.lower()

        # Test with invalid position
        result = analytics_service.analyze_position_trends(empty_df, position="INVALID")
        assert result.is_failure
        assert ("invalid position" in result.error.message.lower() or
                "no player data" in result.error.message.lower())

        # Test breakout detection with empty data
        result = analytics_service.detect_breakout_players(empty_df)
        assert result.is_failure
        assert "no player data" in result.error.message.lower()

    def test_price_performance_correlation(self, analytics_service, sample_gameweek_data):
        """Test price-performance correlation analysis."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Ensure we have price column
        if "price" not in players_with_xp.columns:
            pytest.skip("No price data available for correlation test")

        insights_result = analytics_service.get_statistical_insights(players_with_xp)
        assert insights_result.is_success

        insights = insights_result.value
        if "price_performance_correlation" in insights:
            correlation = insights["price_performance_correlation"]
            # Correlation should be a valid number between -1 and 1
            assert -1 <= correlation <= 1
            assert not pd.isna(correlation)
