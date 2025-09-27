"""Integration tests for domain services."""

import pytest
import pandas as pd

from fpl_team_picker.domain.services import (
    DataOrchestrationService,
    ExpectedPointsService,
    TransferOptimizationService
)
from fpl_team_picker.adapters.database_repositories import (
    DatabasePlayerRepository,
    DatabaseTeamRepository,
    DatabaseFixtureRepository
)


class TestDomainServicesIntegration:
    """Integration tests for domain services with real data."""

    @pytest.fixture
    def repositories(self):
        """Create repository instances."""
        return {
            'player_repo': DatabasePlayerRepository(),
            'team_repo': DatabaseTeamRepository(),
            'fixture_repo': DatabaseFixtureRepository()
        }

    @pytest.fixture
    def data_service(self, repositories):
        """Create data orchestration service."""
        return DataOrchestrationService(
            repositories['player_repo'],
            repositories['team_repo'],
            repositories['fixture_repo']
        )

    @pytest.fixture
    def xp_service(self):
        """Create expected points service."""
        return ExpectedPointsService()

    @pytest.fixture
    def transfer_service(self):
        """Create transfer optimization service."""
        return TransferOptimizationService()

    def test_data_orchestration_service_integration(self, data_service):
        """Test data orchestration service loads real data."""
        # Test with gameweek 1 (should always work)
        result = data_service.load_gameweek_data(target_gameweek=1, form_window=3)

        assert result.is_success, f"Data loading failed: {result.error.message if result.error else 'Unknown error'}"

        gameweek_data = result.value
        required_keys = ["players", "teams", "fixtures", "xg_rates", "gameweek_info", "target_gameweek"]

        for key in required_keys:
            assert key in gameweek_data, f"Missing key: {key}"

        # Validate data structure
        assert isinstance(gameweek_data["players"], pd.DataFrame)
        assert isinstance(gameweek_data["teams"], pd.DataFrame)
        assert isinstance(gameweek_data["fixtures"], pd.DataFrame)
        assert len(gameweek_data["players"]) > 0
        assert len(gameweek_data["teams"]) == 20
        assert len(gameweek_data["fixtures"]) > 0

    def test_expected_points_service_integration(self, data_service, xp_service):
        """Test expected points service with real data."""
        # Load gameweek data first
        data_result = data_service.load_gameweek_data(target_gameweek=1, form_window=3)
        assert data_result.is_success

        gameweek_data = data_result.value

        # Test rule-based model (more reliable than ML)
        xp_result = xp_service.calculate_combined_results(
            gameweek_data, use_ml_model=False
        )

        assert xp_result.is_success, f"XP calculation failed: {xp_result.error.message if xp_result.error else 'Unknown error'}"

        players_with_xp = xp_result.value
        assert isinstance(players_with_xp, pd.DataFrame)
        assert len(players_with_xp) > 0

        # Check required columns exist
        required_columns = ["web_name", "position", "price", "player_id", "xP"]
        for col in required_columns:
            assert col in players_with_xp.columns, f"Missing column: {col}"

        # Check xP values are reasonable
        assert players_with_xp["xP"].min() >= 0
        assert players_with_xp["xP"].max() <= 20  # Sanity check

    def test_transfer_optimization_service_basic(self, data_service, xp_service, transfer_service):
        """Test transfer optimization service basic functionality."""
        # Load data and calculate xP
        data_result = data_service.load_gameweek_data(target_gameweek=1, form_window=3)
        assert data_result.is_success

        gameweek_data = data_result.value

        xp_result = xp_service.calculate_combined_results(
            gameweek_data, use_ml_model=False
        )
        assert xp_result.is_success

        players_with_xp = xp_result.value

        # Test starting eleven selection (doesn't require current squad)
        starting_11_result = transfer_service.get_starting_eleven(players_with_xp)
        assert starting_11_result.is_success

        starting_11 = starting_11_result.value
        assert len(starting_11) == 11

        # Test captain recommendation
        captain_result = transfer_service.get_captain_recommendation(players_with_xp)
        assert captain_result.is_success

        captain_recommendation = captain_result.value
        assert "player_id" in captain_recommendation or "web_name" in captain_recommendation

    def test_service_error_handling(self, data_service, xp_service):
        """Test that services handle errors gracefully."""
        # Test invalid gameweek
        result = data_service.load_gameweek_data(target_gameweek=99, form_window=3)
        # Should either succeed with empty data or fail gracefully
        if result.is_failure:
            assert result.error.message is not None
            assert len(result.error.message) > 0

        # Test XP service with invalid data
        empty_data = {
            "players": pd.DataFrame(),
            "teams": pd.DataFrame(),
            "fixtures": pd.DataFrame(),
            "xg_rates": pd.DataFrame(),
            "target_gameweek": 1
        }

        xp_result = xp_service.calculate_combined_results(empty_data, use_ml_model=False)
        assert xp_result.is_failure
        assert "empty" in xp_result.error.message.lower()

    def test_data_validation_service(self, data_service):
        """Test data validation functionality."""
        # Load valid data
        data_result = data_service.load_gameweek_data(target_gameweek=1, form_window=3)
        assert data_result.is_success

        gameweek_data = data_result.value

        # Test validation passes for valid data
        validation_result = data_service.validate_gameweek_data(gameweek_data)
        assert validation_result.is_success

        # Test validation fails for invalid data
        invalid_data = {"players": pd.DataFrame()}  # Missing required keys
        validation_result = data_service.validate_gameweek_data(invalid_data)
        assert validation_result.is_failure

    def test_model_info_service(self, xp_service):
        """Test model information service."""
        # Test rule-based model info
        rule_info = xp_service.get_model_info(use_ml_model=False)
        assert rule_info["type"] == "Rule-Based"
        assert "description" in rule_info

        # Test ML model info
        ml_info = xp_service.get_model_info(use_ml_model=True)
        assert ml_info["type"] == "ML"
        assert "description" in ml_info

    def test_constraint_validation_service(self, transfer_service):
        """Test constraint validation in transfer service."""
        # Create mock data
        mock_players = pd.DataFrame({
            'player_id': [1, 2, 3, 4, 5],
            'web_name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5']
        })

        # Test valid constraints
        result = transfer_service.validate_optimization_constraints(
            must_include_ids={1, 2},
            must_exclude_ids={3, 4},
            players_with_xp=mock_players
        )
        assert result.is_success

        # Test conflicting constraints
        result = transfer_service.validate_optimization_constraints(
            must_include_ids={1, 2},
            must_exclude_ids={2, 3},  # Player 2 in both
            players_with_xp=mock_players
        )
        assert result.is_failure
        assert "conflict" in result.error.message.lower()

        # Test missing players
        result = transfer_service.validate_optimization_constraints(
            must_include_ids={99},  # Player doesn't exist
            must_exclude_ids={3},
            players_with_xp=mock_players
        )
        assert result.is_failure
        assert "not found" in result.error.message.lower()
