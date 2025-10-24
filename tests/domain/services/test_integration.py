"""Integration tests for domain services."""

import pytest
import pandas as pd

from fpl_team_picker.domain.services import (
    DataOrchestrationService,
    ExpectedPointsService,
    OptimizationService,
)
from fpl_team_picker.adapters.database_repositories import DatabasePlayerRepository


class TestDomainServicesIntegration:
    """Integration tests for domain services with real data."""

    @pytest.fixture
    def repositories(self):
        """Create repository instances."""
        return {
            "player_repo": DatabasePlayerRepository(),
            # team_repo and fixture_repo not needed - using FPL client
        }

    @pytest.fixture
    def data_service(self, repositories):
        """Create data orchestration service."""
        return DataOrchestrationService(
            repositories["player_repo"],
            None,  # team_repo not needed
            None,  # fixture_repo not needed
        )

    @pytest.fixture
    def xp_service(self):
        """Create expected points service."""
        return ExpectedPointsService()

    @pytest.fixture
    def optimization_service(self):
        """Create optimization service."""
        return OptimizationService()

    def test_data_orchestration_service_integration(self, data_service):
        """Test data orchestration service loads real data."""
        # Test with gameweek 1 (should always work)
        gameweek_data = data_service.load_gameweek_data(
            target_gameweek=1, form_window=3
        )
        required_keys = [
            "players",
            "teams",
            "fixtures",
            "xg_rates",
            "gameweek_info",
            "target_gameweek",
        ]

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
        gameweek_data = data_service.load_gameweek_data(
            target_gameweek=1, form_window=3
        )

        # Test rule-based model (more reliable than ML)
        players_with_xp = xp_service.calculate_combined_results(
            gameweek_data, use_ml_model=False
        )
        assert isinstance(players_with_xp, pd.DataFrame)
        assert len(players_with_xp) > 0

        # Check required columns exist
        required_columns = ["web_name", "position", "price", "player_id", "xP"]
        for col in required_columns:
            assert col in players_with_xp.columns, f"Missing column: {col}"

        # Check xP values are reasonable
        assert players_with_xp["xP"].min() >= 0
        assert players_with_xp["xP"].max() <= 20  # Sanity check

    def test_optimization_service_basic(
        self, data_service, xp_service, optimization_service
    ):
        """Test optimization service basic functionality."""
        # Load data and calculate xP
        gameweek_data = data_service.load_gameweek_data(
            target_gameweek=1, form_window=3
        )
        assert isinstance(gameweek_data, dict)

        players_with_xp = xp_service.calculate_combined_results(
            gameweek_data, use_ml_model=False
        )
        assert not players_with_xp.empty

        # Test optimal team selection from full database (for testing/analysis)
        starting_11 = optimization_service.get_optimal_team_from_database(
            players_with_xp
        )
        assert isinstance(starting_11, list)
        assert len(starting_11) == 11

        # Test captain recommendation from full database (for testing/analysis)
        captain_data = optimization_service.get_captain_recommendation(
            players_with_xp, top_n=5
        )
        assert isinstance(captain_data, dict)
        assert "captain" in captain_data
        assert "vice_captain" in captain_data
        assert "top_candidates" in captain_data
        assert captain_data["captain"]["web_name"]  # Has a name

    def test_service_error_handling(self, data_service, xp_service):
        """Test that services handle errors gracefully."""
        # Test invalid gameweek - should still return data but might be limited
        result = data_service.load_gameweek_data(target_gameweek=99, form_window=3)
        assert isinstance(result, dict)  # Should return data structure

        # Test XP service with invalid data - should raise exceptions
        empty_data = {
            "players": pd.DataFrame(),
            "teams": pd.DataFrame(),
            "fixtures": pd.DataFrame(),
            "xg_rates": pd.DataFrame(),
            "target_gameweek": 1,
        }

        # Should raise KeyError due to missing columns in empty DataFrames
        with pytest.raises(KeyError):
            xp_service.calculate_combined_results(empty_data, use_ml_model=False)

    def test_data_validation_service(self, data_service):
        """Test data validation functionality."""
        # Load valid data
        gameweek_data = data_service.load_gameweek_data(
            target_gameweek=1, form_window=3
        )
        assert isinstance(gameweek_data, dict)

        # Test validation passes for valid data
        is_valid = data_service.validate_gameweek_data(gameweek_data)
        assert is_valid is True

        # Test validation fails for invalid data
        invalid_data = {"players": pd.DataFrame()}  # Missing required keys
        with pytest.raises(ValueError, match="Missing required data keys"):
            data_service.validate_gameweek_data(invalid_data)

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

    def test_constraint_validation_service(self, optimization_service):
        """Test constraint validation functionality."""
        # Test valid constraints
        result = optimization_service.validate_optimization_constraints(
            must_include_ids={1, 2, 3}, must_exclude_ids={4, 5, 6}, budget_limit=100.0
        )
        assert isinstance(result, dict)
        assert result["valid"] is True

        # Test conflicting constraints
        result = optimization_service.validate_optimization_constraints(
            must_include_ids={1, 2, 3},
            must_exclude_ids={2, 3, 4},  # Overlaps with must_include
            budget_limit=100.0,
        )
        assert isinstance(result, dict)
        assert result["valid"] is False
        assert len(result["conflicts"]) == 2  # Players 2 and 3
