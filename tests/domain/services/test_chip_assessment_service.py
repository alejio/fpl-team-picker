"""Integration tests for ChipAssessmentService."""

import pytest
import pandas as pd

from fpl_team_picker.domain.services import (
    DataOrchestrationService,
    ExpectedPointsService,
)
from fpl_team_picker.domain.services.chip_assessment_service import ChipAssessmentService
from fpl_team_picker.adapters.database_repositories import (
    DatabasePlayerRepository,
    DatabaseTeamRepository,
    DatabaseFixtureRepository
)


class TestChipAssessmentServiceIntegration:
    """Integration tests for ChipAssessmentService with real data."""

    @pytest.fixture
    def chip_service(self):
        """Create chip assessment service."""
        return ChipAssessmentService()

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
        assert data_result.is_success, f"Failed to load data: {data_result.error.message if data_result.error else 'Unknown'}"

        gameweek_data = data_result.value

        # Calculate XP (rule-based for reliability)
        xp_result = xp_service.calculate_combined_results(gameweek_data, use_ml_model=False)
        assert xp_result.is_success, f"Failed to calculate XP: {xp_result.error.message if xp_result.error else 'Unknown'}"

        players_with_xp = xp_result.value
        gameweek_data["players_with_xp"] = players_with_xp

        return gameweek_data

    @pytest.fixture
    def mock_squad(self, sample_gameweek_data):
        """Create a mock 15-player squad."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        squad_players = []
        for position, count in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            pos_players = players_with_xp[players_with_xp["position"] == position].nlargest(count, "xP")
            squad_players.append(pos_players)

        return pd.concat(squad_players, ignore_index=True)

    def test_assess_all_chips_integration(self, chip_service, sample_gameweek_data, mock_squad):
        """Test chip assessment with real data."""
        available_chips = ["wildcard", "bench_boost", "triple_captain"]

        # Test chip assessment
        chip_result = chip_service.assess_all_chips(
            sample_gameweek_data, mock_squad, available_chips
        )
        assert chip_result.is_success, f"Chip assessment failed: {chip_result.error.message if chip_result.error else 'Unknown'}"

        chip_data = chip_result.value
        assert "recommendations" in chip_data
        assert "summary" in chip_data
        assert "target_gameweek" in chip_data
        assert len(chip_data["recommendations"]) <= len(available_chips)

    def test_individual_chip_assessment(self, chip_service, sample_gameweek_data, mock_squad):
        """Test individual chip assessment."""
        # Test triple captain specifically
        tc_result = chip_service.get_chip_recommendation(
            "triple_captain", sample_gameweek_data, mock_squad
        )
        assert tc_result.is_success, f"Triple captain assessment failed: {tc_result.error.message if tc_result.error else 'Unknown'}"

        tc_data = tc_result.value
        assert "chip_name" in tc_data
        assert "status" in tc_data
        assert tc_data["status"] in ["🟢 RECOMMENDED", "🟡 CONSIDER", "🔴 HOLD"]

    def test_chip_timing_analysis(self, chip_service, sample_gameweek_data, mock_squad):
        """Test chip timing analysis over multiple gameweeks."""
        available_chips = ["triple_captain"]

        timing_result = chip_service.get_chip_timing_analysis(
            sample_gameweek_data, mock_squad, available_chips, gameweeks_ahead=3
        )
        assert timing_result.is_success, f"Timing analysis failed: {timing_result.error.message if timing_result.error else 'Unknown'}"

        timing_data = timing_result.value
        assert "timing_analysis" in timing_data
        assert "triple_captain" in timing_data["timing_analysis"]

    def test_error_handling(self, chip_service):
        """Test error handling in chip service."""
        empty_df = pd.DataFrame()

        # Test with invalid squad size
        result = chip_service.assess_all_chips({}, empty_df, ["wildcard"])
        assert result.is_failure
        assert "15 players" in result.error.message

        # Test with invalid chip name
        result = chip_service.get_chip_recommendation("invalid_chip", {}, empty_df)
        assert result.is_failure
        assert "invalid chip name" in result.error.message.lower()
