"""Integration tests for VisualizationService."""

import pytest
import pandas as pd

from fpl_team_picker.domain.services import (
    DataOrchestrationService,
    ExpectedPointsService,
)
from fpl_team_picker.domain.services.visualization_service import VisualizationService
from fpl_team_picker.adapters.database_repositories import (
    DatabasePlayerRepository,
    DatabaseTeamRepository,
    DatabaseFixtureRepository
)


class TestVisualizationServiceIntegration:
    """Integration tests for VisualizationService with real data."""

    @pytest.fixture
    def viz_service(self):
        """Create visualization service."""
        return VisualizationService()

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

    def test_expected_points_display(self, viz_service, sample_gameweek_data):
        """Test expected points display creation."""
        players_with_xp = sample_gameweek_data["players_with_xp"]
        target_gameweek = sample_gameweek_data["target_gameweek"]

        # Test expected points display
        xp_display_result = viz_service.create_expected_points_display(
            players_with_xp, target_gameweek
        )
        assert xp_display_result.is_success, f"XP display failed: {xp_display_result.error.message if xp_display_result.error else 'Unknown'}"

        xp_display_data = xp_display_result.value
        assert "display_data" in xp_display_data
        assert "insights" in xp_display_data
        assert xp_display_data["total_players"] > 0
        assert xp_display_data["target_gameweek"] == target_gameweek

    def test_fixture_difficulty_display(self, viz_service, sample_gameweek_data):
        """Test fixture difficulty display creation."""
        # Test fixture difficulty display
        fixture_result = viz_service.create_fixture_difficulty_display(
            sample_gameweek_data, start_gameweek=1, num_gameweeks=5
        )
        assert fixture_result.is_success, f"Fixture display failed: {fixture_result.error.message if fixture_result.error else 'Unknown'}"

        fixture_data = fixture_result.value
        assert "fixture_analysis" in fixture_data
        assert fixture_data["start_gameweek"] == 1
        assert fixture_data["num_gameweeks"] == 5
        assert fixture_data["end_gameweek"] == 5

    def test_team_strength_display(self, viz_service, sample_gameweek_data):
        """Test team strength display creation."""
        target_gameweek = sample_gameweek_data["target_gameweek"]

        # Test team strength display
        strength_result = viz_service.create_team_strength_display(
            sample_gameweek_data, target_gameweek
        )
        assert strength_result.is_success, f"Team strength display failed: {strength_result.error.message if strength_result.error else 'Unknown'}"

        strength_data = strength_result.value
        assert "team_strength_data" in strength_data
        assert strength_data["target_gameweek"] == target_gameweek
        assert strength_data["total_teams"] > 0

    def test_player_trends_display(self, viz_service, sample_gameweek_data):
        """Test player trends display creation."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Test player trends display
        trends_result = viz_service.create_player_trends_display(
            players_with_xp, trend_type="performance", top_n=10
        )
        assert trends_result.is_success, f"Player trends failed: {trends_result.error.message if trends_result.error else 'Unknown'}"

        trends_data = trends_result.value
        assert "trends_data" in trends_data
        assert trends_data["trend_type"] == "performance"
        assert trends_data["top_n"] == 10

    def test_error_handling(self, viz_service):
        """Test error handling in visualization service."""
        empty_df = pd.DataFrame()

        # Test with empty data
        result = viz_service.create_expected_points_display(empty_df, 1)
        assert result.is_failure
        assert "no expected points data" in result.error.message.lower()

        # Test with invalid gameweek
        result = viz_service.create_fixture_difficulty_display(
            {}, start_gameweek=99, num_gameweeks=5
        )
        assert result.is_failure
        assert "invalid start gameweek" in result.error.message.lower()

        # Test with invalid trend type
        result = viz_service.create_player_trends_display(
            empty_df, trend_type="invalid", top_n=10
        )
        assert result.is_failure
        assert ("invalid trend type" in result.error.message.lower() or
                "no player data" in result.error.message.lower())

    def test_data_validation(self, viz_service):
        """Test data validation in visualization service."""
        # Test with missing required columns
        invalid_df = pd.DataFrame({
            'player_id': [1, 2, 3],
            'web_name': ['P1', 'P2', 'P3']
            # Missing 'position' and 'xP' columns
        })

        result = viz_service.create_expected_points_display(invalid_df, 1)
        assert result.is_failure
        assert "missing required columns" in result.error.message.lower()
