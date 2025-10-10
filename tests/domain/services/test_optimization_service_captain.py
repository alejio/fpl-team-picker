"""Unit tests for OptimizationService.get_captain_recommendation()."""

import pytest
import pandas as pd
from fpl_team_picker.domain.services.optimization_service import OptimizationService


class TestCaptainRecommendationStructure:
    """Test that captain recommendation returns correct data structure."""

    def test_basic_structure(self):
        """Test that the method returns expected dictionary structure."""
        service = OptimizationService()

        # Create sample players
        players_df = pd.DataFrame([
            {
                "player_id": 1,
                "web_name": "Salah",
                "position": "MID",
                "xP": 8.5,
                "price": 13.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            },
            {
                "player_id": 2,
                "web_name": "Haaland",
                "position": "FWD",
                "xP": 9.2,
                "price": 14.0,
                "expected_minutes": 85,
                "fixture_outlook": "🟡 Average",
                "status": "a",
            },
            {
                "player_id": 3,
                "web_name": "De Bruyne",
                "position": "MID",
                "xP": 7.8,
                "price": 12.5,
                "expected_minutes": 80,
                "fixture_outlook": "🔴 Hard",
                "status": "a",
            },
        ])

        result = service.get_captain_recommendation(players_df, top_n=3)

        # Check top-level structure
        assert isinstance(result, dict)
        assert "captain" in result
        assert "vice_captain" in result
        assert "top_candidates" in result
        assert "captain_upside" in result
        assert "vice_upside" in result
        assert "differential" in result

        # Check captain structure
        captain = result["captain"]
        assert isinstance(captain, dict)
        assert captain["web_name"] == "Haaland"  # Highest xP
        assert captain["xP"] == 9.2
        assert captain["captain_points"] == 18.4  # xP * 2

        # Check vice captain
        vice = result["vice_captain"]
        assert isinstance(vice, dict)
        assert vice["web_name"] == "Salah"  # Second highest xP

        # Check top candidates
        assert isinstance(result["top_candidates"], list)
        assert len(result["top_candidates"]) == 3

        # Check metrics
        assert result["captain_upside"] == 18.4
        assert result["differential"] > 0

    def test_captain_candidate_structure(self):
        """Test that each candidate has all required fields."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": 1,
                "web_name": "Player1",
                "position": "FWD",
                "xP": 8.0,
                "price": 10.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            },
        ])

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["top_candidates"][0]

        # Check all required fields
        assert "rank" in candidate
        assert "player_id" in candidate
        assert "web_name" in candidate
        assert "position" in candidate
        assert "price" in candidate
        assert "xP" in candidate
        assert "captain_points" in candidate
        assert "expected_minutes" in candidate
        assert "fixture_outlook" in candidate
        assert "risk_level" in candidate
        assert "risk_factors" in candidate
        assert "risk_description" in candidate
        assert "status" in candidate


class TestCaptainRiskAnalysis:
    """Test risk factor calculations."""

    def test_injury_risk_high(self):
        """Test that injured players are marked as high risk."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": 1,
                "web_name": "Injured",
                "position": "FWD",
                "xP": 8.0,
                "price": 10.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "i",  # Injured
            },
        ])

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["captain"]

        assert candidate["risk_level"] == "🔴 High"
        assert "injury risk" in candidate["risk_factors"]

    def test_suspension_risk_high(self):
        """Test that suspended players are marked as high risk."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": 1,
                "web_name": "Suspended",
                "position": "DEF",
                "xP": 7.0,
                "price": 6.0,
                "expected_minutes": 0,
                "fixture_outlook": "🟢 Easy",
                "status": "s",  # Suspended
            },
        ])

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["captain"]

        assert candidate["risk_level"] == "🔴 High"
        assert "suspended" in candidate["risk_factors"]

    def test_hard_fixture_medium_risk(self):
        """Test that hard fixtures increase risk level."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": 1,
                "web_name": "Player1",
                "position": "MID",
                "xP": 6.0,
                "price": 8.0,
                "expected_minutes": 90,
                "fixture_outlook": "🔴 Hard",
                "status": "a",
            },
        ])

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["captain"]

        assert candidate["risk_level"] in ["🟡 Medium", "🔴 High"]
        assert "hard fixture" in candidate["risk_factors"]

    def test_rotation_risk_medium(self):
        """Test that low expected minutes increase risk."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": 1,
                "web_name": "Rotated",
                "position": "FWD",
                "xP": 5.0,
                "price": 9.0,
                "expected_minutes": 45,  # Low minutes
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            },
        ])

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["captain"]

        assert candidate["risk_level"] in ["🟡 Medium", "🔴 High"]
        assert "rotation risk" in candidate["risk_factors"]

    def test_low_risk_perfect_player(self):
        """Test that a player with no risk factors is marked as low risk."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": 1,
                "web_name": "Perfect",
                "position": "FWD",
                "xP": 8.0,
                "price": 12.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            },
        ])

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["captain"]

        assert candidate["risk_level"] == "🟢 Low"
        assert len(candidate["risk_factors"]) == 0


class TestCaptainTopCandidates:
    """Test top N candidates selection."""

    def test_top_n_parameter(self):
        """Test that top_n parameter works correctly."""
        service = OptimizationService()

        # Create 10 players
        players_df = pd.DataFrame([
            {
                "player_id": i,
                "web_name": f"Player{i}",
                "position": "MID",
                "xP": 10.0 - i * 0.5,  # Descending xP
                "price": 8.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            }
            for i in range(10)
        ])

        # Test top_n=3
        result = service.get_captain_recommendation(players_df, top_n=3)
        assert len(result["top_candidates"]) == 3

        # Test top_n=5
        result = service.get_captain_recommendation(players_df, top_n=5)
        assert len(result["top_candidates"]) == 5

        # Test top_n=10
        result = service.get_captain_recommendation(players_df, top_n=10)
        assert len(result["top_candidates"]) == 10

    def test_sorted_by_xp(self):
        """Test that candidates are sorted by xP descending."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": 1,
                "web_name": "Player1",
                "position": "MID",
                "xP": 5.0,
                "price": 8.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            },
            {
                "player_id": 2,
                "web_name": "Player2",
                "position": "FWD",
                "xP": 9.0,
                "price": 12.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            },
            {
                "player_id": 3,
                "web_name": "Player3",
                "position": "MID",
                "xP": 7.0,
                "price": 10.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            },
        ])

        result = service.get_captain_recommendation(players_df, top_n=3)
        candidates = result["top_candidates"]

        # Check ordering
        assert candidates[0]["web_name"] == "Player2"  # xP=9.0
        assert candidates[1]["web_name"] == "Player3"  # xP=7.0
        assert candidates[2]["web_name"] == "Player1"  # xP=5.0

        # Check ranks
        assert candidates[0]["rank"] == 1
        assert candidates[1]["rank"] == 2
        assert candidates[2]["rank"] == 3

    def test_captain_points_calculation(self):
        """Test that captain points are correctly calculated as xP * 2."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": 1,
                "web_name": "Player1",
                "position": "FWD",
                "xP": 7.5,
                "price": 10.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            },
        ])

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["captain"]

        assert candidate["captain_points"] == 15.0  # 7.5 * 2


class TestCaptainDataValidation:
    """Test data validation and error handling."""

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        service = OptimizationService()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="No players provided"):
            service.get_captain_recommendation(empty_df)

    def test_missing_required_columns_raises_error(self):
        """Test that missing required columns raises ValueError."""
        service = OptimizationService()

        # Missing 'web_name'
        incomplete_df = pd.DataFrame([
            {
                "player_id": 1,
                "position": "FWD",
                "xP": 8.0,
            }
        ])

        with pytest.raises(ValueError, match="Missing required columns"):
            service.get_captain_recommendation(incomplete_df)

    def test_custom_xp_column(self):
        """Test that custom xP column works."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": 1,
                "web_name": "Player1",
                "position": "MID",
                "xP_custom": 8.0,  # Custom column
                "price": 10.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            },
        ])

        result = service.get_captain_recommendation(players_df, top_n=1, xp_column="xP_custom")

        assert result["captain"]["xP"] == 8.0
        assert result["captain"]["captain_points"] == 16.0

    def test_handles_missing_optional_fields(self):
        """Test that missing optional fields use defaults."""
        service = OptimizationService()

        # Minimal player data
        players_df = pd.DataFrame([
            {
                "web_name": "Minimal",
                "position": "FWD",
                "xP": 7.0,
            }
        ])

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["captain"]

        # Should use defaults
        assert candidate["price"] == 0
        assert candidate["expected_minutes"] == 0
        assert candidate["fixture_outlook"] == "🟡 Average"
        assert candidate["status"] == "a"


class TestCaptainMetrics:
    """Test captain metrics calculations."""

    def test_differential_calculation(self):
        """Test that differential is correctly calculated."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": 1,
                "web_name": "Captain",
                "position": "FWD",
                "xP": 9.0,
                "price": 13.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            },
            {
                "player_id": 2,
                "web_name": "Vice",
                "position": "MID",
                "xP": 7.0,
                "price": 11.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            },
        ])

        result = service.get_captain_recommendation(players_df, top_n=2)

        # Captain: 9.0 * 2 = 18.0
        # Vice: 7.0 * 2 = 14.0
        # Differential: 18.0 - 14.0 = 4.0
        assert result["captain_upside"] == 18.0
        assert result["vice_upside"] == 14.0
        assert result["differential"] == 4.0

    def test_single_player_captain_and_vice_same(self):
        """Test that with only one player, captain and vice are the same."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": 1,
                "web_name": "Only",
                "position": "FWD",
                "xP": 8.0,
                "price": 10.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            },
        ])

        result = service.get_captain_recommendation(players_df, top_n=5)

        assert result["captain"]["web_name"] == "Only"
        assert result["vice_captain"]["web_name"] == "Only"
        assert result["differential"] == 0.0


class TestCaptainEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_top_n_exceeds_available_players(self):
        """Test that requesting more candidates than available works."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": i,
                "web_name": f"Player{i}",
                "position": "MID",
                "xP": 5.0,
                "price": 8.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            }
            for i in range(3)  # Only 3 players
        ])

        # Request top 10 but only have 3
        result = service.get_captain_recommendation(players_df, top_n=10)

        assert len(result["top_candidates"]) == 3  # Returns what's available

    def test_identical_xp_values(self):
        """Test behavior with players having identical xP."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": i,
                "web_name": f"Player{i}",
                "position": "MID",
                "xP": 7.0,  # All same
                "price": 8.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            }
            for i in range(5)
        ])

        result = service.get_captain_recommendation(players_df, top_n=3)

        # Should still work and return 3 candidates
        assert len(result["top_candidates"]) == 3
        assert all(c["xP"] == 7.0 for c in result["top_candidates"])

    def test_zero_xp_players(self):
        """Test that players with 0 xP can still be captains."""
        service = OptimizationService()

        players_df = pd.DataFrame([
            {
                "player_id": 1,
                "web_name": "Zero",
                "position": "DEF",
                "xP": 0.0,
                "price": 4.0,
                "expected_minutes": 90,
                "fixture_outlook": "🟢 Easy",
                "status": "a",
            },
        ])

        result = service.get_captain_recommendation(players_df, top_n=1)

        assert result["captain"]["xP"] == 0.0
        assert result["captain"]["captain_points"] == 0.0
