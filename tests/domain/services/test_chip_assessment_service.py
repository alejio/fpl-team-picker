"""Integration tests for ChipAssessmentService."""

import pytest
import pandas as pd

from fpl_team_picker.domain.services import (
    DataOrchestrationService,
    ExpectedPointsService,
)
from fpl_team_picker.domain.services.chip_assessment_service import (
    ChipAssessmentService,
)
from fpl_team_picker.adapters.database_repositories import (
    DatabasePlayerRepository,
    DatabaseTeamRepository,
    DatabaseFixtureRepository,
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
        gameweek_data = data_service.load_gameweek_data(
            target_gameweek=1, form_window=3
        )

        # Calculate XP (rule-based for reliability)
        players_with_xp = xp_service.calculate_combined_results(
            gameweek_data, use_ml_model=False
        )
        gameweek_data["players_with_xp"] = players_with_xp

        return gameweek_data

    @pytest.fixture
    def mock_squad(self, sample_gameweek_data):
        """Create a mock 15-player squad."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        squad_players = []
        for position, count in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            pos_players = players_with_xp[
                players_with_xp["position"] == position
            ].nlargest(count, "xP")
            squad_players.append(pos_players)

        return pd.concat(squad_players, ignore_index=True)

    def test_assess_all_chips_integration(
        self, chip_service, sample_gameweek_data, mock_squad
    ):
        """Test chip assessment with real data."""
        available_chips = ["wildcard", "bench_boost", "triple_captain"]

        # Test chip assessment
        chip_data = chip_service.assess_all_chips(
            sample_gameweek_data, mock_squad, available_chips
        )
        assert "recommendations" in chip_data
        assert "summary" in chip_data
        assert "target_gameweek" in chip_data
        assert len(chip_data["recommendations"]) <= len(available_chips)

    def test_individual_chip_assessment(
        self, chip_service, sample_gameweek_data, mock_squad
    ):
        """Test individual chip assessment."""
        # Test triple captain specifically
        tc_data = chip_service.get_chip_recommendation(
            "triple_captain", sample_gameweek_data, mock_squad
        )
        assert "chip_name" in tc_data
        assert "status" in tc_data
        assert tc_data["status"] in ["🟢 RECOMMENDED", "🟡 CONSIDER", "🔴 HOLD"]

    def test_chip_timing_analysis(self, chip_service, sample_gameweek_data, mock_squad):
        """Test chip timing analysis over multiple gameweeks."""
        available_chips = ["triple_captain"]

        timing_data = chip_service.get_chip_timing_analysis(
            sample_gameweek_data, mock_squad, available_chips, gameweeks_ahead=3
        )
        assert "timing_analysis" in timing_data
        assert "triple_captain" in timing_data["timing_analysis"]

    def test_error_handling(self, chip_service):
        """Test error handling in chip service."""
        empty_df = pd.DataFrame()

        # Test with invalid gameweek data (missing required keys)
        with pytest.raises(KeyError):
            chip_service.assess_all_chips({}, empty_df, ["wildcard"])

        # Test with invalid chip name - should raise ValueError
        minimal_gameweek_data = {
            "players": empty_df,
            "fixtures": empty_df,
            "target_gameweek": 1,
        }
        with pytest.raises(ValueError, match="Invalid chip name"):
            chip_service.get_chip_recommendation(
                "invalid_chip", minimal_gameweek_data, empty_df
            )
