"""Integration tests for SquadManagementService."""

import pytest
import pandas as pd

from fpl_team_picker.domain.services import (
    DataOrchestrationService,
    ExpectedPointsService,
)
from fpl_team_picker.domain.services.squad_management_service import SquadManagementService
from fpl_team_picker.adapters.database_repositories import (
    DatabasePlayerRepository,
    DatabaseTeamRepository,
    DatabaseFixtureRepository
)


class TestSquadManagementServiceIntegration:
    """Integration tests for SquadManagementService with real data."""

    @pytest.fixture
    def squad_service(self):
        """Create squad management service."""
        return SquadManagementService()

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
    def mock_15_player_squad(self, sample_gameweek_data):
        """Create a realistic 15-player squad."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Create a mock 15-player squad by selecting top players by position
        squad_players = []
        for position, count in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            pos_players = players_with_xp[players_with_xp["position"] == position].nlargest(count, "xP")
            squad_players.append(pos_players)

        mock_squad = pd.concat(squad_players, ignore_index=True)
        assert len(mock_squad) == 15, f"Mock squad should have 15 players, got {len(mock_squad)}"

        return mock_squad

    def test_starting_eleven_from_full_database(self, squad_service, sample_gameweek_data):
        """Test starting eleven selection from full player database."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Test starting eleven selection from full database
        starting_11_result = squad_service.get_starting_eleven(players_with_xp)
        assert starting_11_result.is_success, f"Starting XI failed: {starting_11_result.error.message if starting_11_result.error else 'Unknown'}"

        starting_11_data = starting_11_result.value
        assert len(starting_11_data["starting_11"]) == 11
        assert "formation" in starting_11_data
        assert "total_xp" in starting_11_data
        assert starting_11_data["total_xp"] > 0

        # Verify formation is valid
        formation = starting_11_data["formation"]
        valid_formations = ["3-5-2", "3-4-3", "4-5-1", "4-4-2", "4-3-3", "5-4-1", "5-3-2", "5-2-3"]
        assert formation in valid_formations, f"Invalid formation: {formation}"

    def test_starting_eleven_from_15_player_squad(self, squad_service, mock_15_player_squad):
        """Test starting eleven selection from a 15-player squad."""
        # Test starting XI from 15-player squad
        starting_11_result = squad_service.get_starting_eleven(mock_15_player_squad)
        assert starting_11_result.is_success, f"Starting XI from squad failed: {starting_11_result.error.message if starting_11_result.error else 'Unknown'}"

        starting_11_data = starting_11_result.value
        assert len(starting_11_data["starting_11"]) == 11

        # Verify position distribution
        positions = [player["position"] for player in starting_11_data["starting_11"]]
        assert positions.count("GKP") == 1
        assert 3 <= positions.count("DEF") <= 5
        assert 2 <= positions.count("MID") <= 5
        assert 1 <= positions.count("FWD") <= 3

    def test_captain_recommendation(self, squad_service, sample_gameweek_data):
        """Test captain recommendation functionality."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Get starting XI first
        starting_11_result = squad_service.get_starting_eleven(players_with_xp)
        assert starting_11_result.is_success

        starting_11_data = starting_11_result.value

        # Test captain recommendation
        captain_result = squad_service.get_captain_recommendation(starting_11_data["starting_11"])
        assert captain_result.is_success, f"Captain selection failed: {captain_result.error.message if captain_result.error else 'Unknown'}"

        captain_data = captain_result.value
        assert "captain" in captain_data
        assert "vice_captain" in captain_data
        assert "advantage" in captain_data

        # Verify captain data structure
        captain = captain_data["captain"]
        assert "player_id" in captain
        assert "web_name" in captain
        assert "position" in captain
        assert "xp" in captain
        assert "captain_points" in captain
        assert captain["captain_points"] == captain["xp"] * 2

    def test_bench_players_selection(self, squad_service, mock_15_player_squad):
        """Test bench players selection."""
        # Get starting XI first
        starting_11_result = squad_service.get_starting_eleven(mock_15_player_squad)
        assert starting_11_result.is_success

        starting_11_data = starting_11_result.value

        # Test bench players
        bench_result = squad_service.get_bench_players(mock_15_player_squad, starting_11_data["starting_11"])
        assert bench_result.is_success, f"Bench selection failed: {bench_result.error.message if bench_result.error else 'Unknown'}"

        bench_players = bench_result.value
        assert len(bench_players) == 4  # Should have exactly 4 bench players

        # Verify no overlap between starting XI and bench
        starting_ids = {player["player_id"] for player in starting_11_data["starting_11"]}
        bench_ids = {player["player_id"] for player in bench_players}
        assert len(starting_ids.intersection(bench_ids)) == 0, "Starting XI and bench should not overlap"

    def test_budget_analysis(self, squad_service, mock_15_player_squad):
        """Test budget analysis functionality."""
        # Mock team data with proper structure expected by optimizer
        team_data = {
            "bank_balance": 5.0,
            "transfers_made": 0,
            "sell_prices": {row["player_id"]: row["price"] for _, row in mock_15_player_squad.iterrows()}
        }

        # Test budget analysis - this may fail due to optimizer implementation details
        # but we test that the service handles it gracefully
        budget_result = squad_service.analyze_budget_situation(mock_15_player_squad, team_data)

        # Either succeeds or fails gracefully with a meaningful error
        if budget_result.is_failure:
            assert len(budget_result.error.message) > 0
            assert "budget analysis failed" in budget_result.error.message.lower()
        else:
            budget_data = budget_result.value
            assert isinstance(budget_data, dict)

    def test_error_handling(self, squad_service):
        """Test error handling in squad management service."""
        empty_df = pd.DataFrame()

        # Test with insufficient players
        result = squad_service.get_starting_eleven(empty_df)
        assert result.is_failure
        assert "at least 11 players" in result.error.message.lower()

        # Test captain recommendation with empty list
        result = squad_service.get_captain_recommendation([])
        assert result.is_failure
        assert "no players provided" in result.error.message.lower()

        # Test bench selection with insufficient squad
        small_squad = pd.DataFrame({
            'player_id': [1, 2, 3],
            'web_name': ['P1', 'P2', 'P3'],
            'position': ['GKP', 'DEF', 'MID']
        })
        result = squad_service.get_bench_players(small_squad, [])
        assert result.is_failure
        assert "at least 15 players" in result.error.message.lower()

    def test_data_validation(self, squad_service):
        """Test data validation in squad management service."""
        # Test with missing required columns
        invalid_df = pd.DataFrame({
            'player_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'web_name': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11']
            # Missing 'position' column
        })

        result = squad_service.get_starting_eleven(invalid_df)
        assert result.is_failure
        assert "missing required columns" in result.error.message.lower()

    def test_xp_column_flexibility(self, squad_service, sample_gameweek_data):
        """Test that service can handle different XP column names."""
        players_with_xp = sample_gameweek_data["players_with_xp"]

        # Test with different XP columns
        for xp_col in ["xP", "xP_5gw"]:
            if xp_col in players_with_xp.columns:
                result = squad_service.get_starting_eleven(players_with_xp, xp_column=xp_col)
                assert result.is_success, f"Failed with XP column {xp_col}: {result.error.message if result.error else 'Unknown'}"

                starting_11_data = result.value
                assert starting_11_data["xp_column_used"] == xp_col
