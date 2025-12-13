"""Unit tests for captain recommendation functionality.

Tests for:
- Captain recommendation structure and output
- Risk analysis and factors
- Top candidate selection
- Data validation
- Captain metrics calculations
- Edge cases

Enhanced 2025/26 captain selection tests:
- Template protection (20-40% boost for >50% ownership)
- Betting odds integration (team_win_probability, team_expected_goals)
- Matchup quality bonus (0-25% for all players)
- Graceful fallback when betting odds unavailable
"""

import pytest
import pandas as pd
from fpl_team_picker.domain.services.optimization_service import OptimizationService


class TestCaptainRecommendationStructure:
    """Test that captain recommendation returns correct data structure."""

    def test_basic_structure(self):
        """Test that the method returns expected dictionary structure."""
        service = OptimizationService()

        # Create sample players
        players_df = pd.DataFrame(
            [
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
            ]
        )

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
        # NOTE: With 2025/26 enhancements, Salah (Easy fixture) now beats Haaland (Average fixture)
        # despite lower xP due to matchup quality bonus (20% vs 10%)
        captain = result["captain"]
        assert isinstance(captain, dict)
        assert captain["web_name"] == "Salah"  # Best fixture-adjusted score
        assert captain["xP"] == 8.5
        assert captain["captain_points"] == 17.0  # xP * 2

        # Check vice captain
        vice = result["vice_captain"]
        assert isinstance(vice, dict)
        assert vice["web_name"] == "Haaland"  # Second best fixture-adjusted score

        # Check top candidates
        assert isinstance(result["top_candidates"], list)
        assert len(result["top_candidates"]) == 3

        # Check metrics (updated for new captain)
        assert result["captain_upside"] == 17.0
        # Note: differential can be negative when fixture quality bonus overcomes higher base xP
        # Salah (17.0 base + 20% bonus = 20.4) beats Haaland (18.4 base + 10% bonus = 20.24)
        assert abs(result["differential"] - (-1.4)) < 0.01  # Approximate equality

    def test_captain_candidate_structure(self):
        """Test that each candidate has all required fields."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
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
            ]
        )

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

        players_df = pd.DataFrame(
            [
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
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["captain"]

        assert candidate["risk_level"] == "🔴 High"
        assert "injury risk" in candidate["risk_factors"]

    def test_suspension_risk_high(self):
        """Test that suspended players are marked as high risk."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
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
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["captain"]

        assert candidate["risk_level"] == "🔴 High"
        assert "suspended" in candidate["risk_factors"]

    def test_hard_fixture_medium_risk(self):
        """Test that hard fixtures increase risk level."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
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
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["captain"]

        assert candidate["risk_level"] in ["🟡 Medium", "🔴 High"]
        assert "hard fixture" in candidate["risk_factors"]

    def test_rotation_risk_medium(self):
        """Test that low expected minutes increase risk."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
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
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["captain"]

        assert candidate["risk_level"] in ["🟡 Medium", "🔴 High"]
        assert "rotation risk" in candidate["risk_factors"]

    def test_low_risk_perfect_player(self):
        """Test that a player with no risk factors is marked as low risk."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
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
            ]
        )

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
        players_df = pd.DataFrame(
            [
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
            ]
        )

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

        players_df = pd.DataFrame(
            [
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
            ]
        )

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

        players_df = pd.DataFrame(
            [
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
            ]
        )

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
        incomplete_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "position": "FWD",
                    "xP": 8.0,
                }
            ]
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            service.get_captain_recommendation(incomplete_df)

    def test_custom_xp_column(self):
        """Test that custom xP column works."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
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
            ]
        )

        result = service.get_captain_recommendation(
            players_df, top_n=1, xp_column="xP_custom"
        )

        assert result["captain"]["xP"] == 8.0
        assert result["captain"]["captain_points"] == 16.0

    def test_handles_missing_optional_fields(self):
        """Test that missing optional fields use defaults."""
        service = OptimizationService()

        # Minimal player data
        players_df = pd.DataFrame(
            [
                {
                    "web_name": "Minimal",
                    "position": "FWD",
                    "xP": 7.0,
                }
            ]
        )

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

        players_df = pd.DataFrame(
            [
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
            ]
        )

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

        players_df = pd.DataFrame(
            [
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
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=5)

        assert result["captain"]["web_name"] == "Only"
        assert result["vice_captain"]["web_name"] == "Only"
        assert result["differential"] == 0.0


class TestCaptainEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_top_n_exceeds_available_players(self):
        """Test that requesting more candidates than available works."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
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
            ]
        )

        # Request top 10 but only have 3
        result = service.get_captain_recommendation(players_df, top_n=10)

        assert len(result["top_candidates"]) == 3  # Returns what's available

    def test_identical_xp_values(self):
        """Test behavior with players having identical xP."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
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
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=3)

        # Should still work and return 3 candidates
        assert len(result["top_candidates"]) == 3
        assert all(c["xP"] == 7.0 for c in result["top_candidates"])

    def test_zero_xp_players(self):
        """Test that players with 0 xP can still be captains."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
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
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)

        assert result["captain"]["xP"] == 0.0
        assert result["captain"]["captain_points"] == 0.0


# ============================================================================
# 2025/26 Enhanced Captain Selection Tests
# ============================================================================


class TestTemplateProtection:
    """Test template protection (20-40% boost for >50% ownership)."""

    def test_template_captain_high_ownership_good_fixture(self):
        """Test that template captain gets maximum boost in good fixture."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Haaland",
                    "position": "FWD",
                    "xP": 8.0,
                    "xP_uncertainty": 1.0,
                    "price": 15.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 60.0,  # Template (>50%)
                    "team_win_probability": 0.75,  # Strong favorite
                    "team_expected_goals": 2.5,  # High scoring expectation
                },
                {
                    "player_id": 2,
                    "web_name": "Differential",
                    "position": "MID",
                    "xP": 8.5,  # Slightly higher xP
                    "xP_uncertainty": 1.0,
                    "price": 7.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 5.0,  # Differential (<50%)
                    "team_win_probability": 0.45,
                    "team_expected_goals": 1.5,
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=2)

        # Template captain should be ranked #1 despite lower xP
        # due to template protection + matchup quality bonus
        assert result["captain"]["web_name"] == "Haaland"

        # Check that scoring components are calculated
        captain = result["top_candidates"][0]
        assert "captain_score" in captain
        assert "ownership_bonus" in captain
        assert "matchup_bonus" in captain

        # Template protection calculation:
        # ownership_factor = (60-50)/20 = 0.5
        # fixture_quality = min(0.75/0.70, 1.0) = 1.0 (capped)
        # bonus = 0.5 * 1.0 * 0.40 = 0.20 (20%)
        # Matchup quality: xg_factor = min((2.5-1.0)/1.0, 1.0) = 1.0, bonus = 1.0 * 0.25 = 0.25 (25%)
        ownership_bonus = captain["ownership_bonus"]
        matchup_bonus = captain["matchup_bonus"]

        assert abs(ownership_bonus - 0.20) < 0.05  # Should be ~20%
        assert abs(matchup_bonus - 0.25) < 0.05  # Should be ~25%

    def test_template_captain_poor_fixture_reduced_boost(self):
        """Test that template boost is reduced in poor fixtures."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Haaland",
                    "position": "FWD",
                    "xP": 6.0,
                    "xP_uncertainty": 1.0,
                    "price": 15.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🔴 Hard",
                    "status": "a",
                    "selected_by_percent": 65.0,  # High template
                    "team_win_probability": 0.35,  # Underdog
                    "team_expected_goals": 1.0,  # Low scoring expectation
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)
        captain = result["top_candidates"][0]

        # Template boost should be reduced due to poor fixture
        ownership_bonus = captain["ownership_bonus"]
        matchup_bonus = captain["matchup_bonus"]

        # Ownership bonus should be low due to fixture quality
        assert ownership_bonus < 0.30  # Reduced from 40%
        # Matchup bonus should be minimal
        assert matchup_bonus < 0.05

    def test_sub_threshold_ownership_no_template_boost(self):
        """Test that players <50% ownership don't get template boost."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Popular",
                    "position": "MID",
                    "xP": 8.0,
                    "xP_uncertainty": 1.0,
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 45.0,  # Below threshold
                    "team_win_probability": 0.70,
                    "team_expected_goals": 2.0,
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)
        captain = result["top_candidates"][0]

        # Should have NO ownership bonus (<50% threshold)
        ownership_bonus = captain["ownership_bonus"]
        assert ownership_bonus == 0.0

        # But should still have matchup bonus
        matchup_bonus = captain["matchup_bonus"]
        assert matchup_bonus > 0.0

    def test_ownership_factor_scaling(self):
        """Test that ownership factor scales correctly (50% → 0.0, 70%+ → 1.0)."""
        service = OptimizationService()

        test_cases = [
            (50.0, 0.0),  # Exactly at threshold
            (55.0, 0.25),  # 25% of range
            (60.0, 0.5),  # 50% of range
            (65.0, 0.75),  # 75% of range
            (70.0, 1.0),  # Max
            (80.0, 1.0),  # Above max (capped)
        ]

        for ownership, expected_factor in test_cases:
            players_df = pd.DataFrame(
                [
                    {
                        "player_id": 1,
                        "web_name": "Player",
                        "position": "FWD",
                        "xP": 8.0,
                        "xP_uncertainty": 1.0,
                        "price": 12.0,
                        "expected_minutes": 90,
                        "fixture_outlook": "🟢 Easy",
                        "status": "a",
                        "selected_by_percent": ownership,
                        "team_win_probability": 0.70,  # Perfect fixture (factor=1.0)
                        "team_expected_goals": 2.25,
                    },
                ]
            )

            result = service.get_captain_recommendation(players_df, top_n=1)
            captain = result["top_candidates"][0]

            ownership_bonus = captain["ownership_bonus"]

            # Ownership bonus = ownership_factor * fixture_quality * 0.40
            # With fixture_quality = 1.0, bonus = expected_factor * 0.40
            expected_bonus = expected_factor * 0.40

            assert abs(ownership_bonus - expected_bonus) < 0.01, (
                f"Ownership {ownership}%: expected bonus ~{expected_bonus:.2f}, "
                f"got {ownership_bonus:.2f}"
            )


class TestBettingOddsIntegration:
    """Test betting odds integration for matchup quality."""

    def test_matchup_quality_with_betting_odds(self):
        """Test that matchup quality uses betting odds when available."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Strong",
                    "position": "FWD",
                    "xP": 7.0,
                    "xP_uncertainty": 1.0,
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 40.0,
                    "team_win_probability": 0.75,
                    "team_expected_goals": 2.5,  # Elite matchup
                },
                {
                    "player_id": 2,
                    "web_name": "Weak",
                    "position": "MID",
                    "xP": 7.0,
                    "xP_uncertainty": 1.0,
                    "price": 10.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🔴 Hard",
                    "status": "a",
                    "selected_by_percent": 35.0,
                    "team_win_probability": 0.30,
                    "team_expected_goals": 0.8,  # Poor matchup
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=2)

        # Player with better matchup should rank higher
        assert result["captain"]["web_name"] == "Strong"

        candidates = {c["web_name"]: c for c in result["top_candidates"]}

        strong_matchup = candidates["Strong"]["matchup_bonus"]
        weak_matchup = candidates["Weak"]["matchup_bonus"]

        # Strong matchup should have higher bonus
        assert strong_matchup > weak_matchup
        assert strong_matchup > 0.15  # Should get ~25% bonus
        assert weak_matchup < 0.05  # Should get minimal bonus

    def test_matchup_quality_xg_scaling(self):
        """Test that team_expected_goals scales matchup bonus correctly."""
        service = OptimizationService()

        test_cases = [
            # (team_xg, expected_matchup_bonus)
            # Formula: xg_factor = max((xg - 1.0) / 1.0, 0.0), bonus = xg_factor * 0.25
            (0.5, 0.0),  # Very low xG → 0% bonus (clamped at 0)
            (1.0, 0.0),  # Neutral baseline → 0% bonus
            (1.25, 0.0625),  # PL average → 6.25% bonus (0.25 * 0.25)
            (1.5, 0.125),  # Above average → 12.5% bonus (0.5 * 0.25)
            (2.0, 0.25),  # High xG → 25% bonus (1.0 * 0.25)
            (2.5, 0.25),  # Very high xG → 25% bonus (capped at 1.0)
        ]

        for team_xg, expected_bonus in test_cases:
            players_df = pd.DataFrame(
                [
                    {
                        "player_id": 1,
                        "web_name": "Player",
                        "position": "FWD",
                        "xP": 7.0,
                        "xP_uncertainty": 1.0,
                        "price": 12.0,
                        "expected_minutes": 90,
                        "fixture_outlook": "🟢 Easy",
                        "status": "a",
                        "selected_by_percent": 40.0,
                        "team_win_probability": 0.60,
                        "team_expected_goals": team_xg,
                    },
                ]
            )

            result = service.get_captain_recommendation(players_df, top_n=1)
            captain = result["top_candidates"][0]

            matchup_bonus = captain["matchup_bonus"]

            assert abs(matchup_bonus - expected_bonus) < 0.02, (
                f"Team xG {team_xg}: expected bonus ~{expected_bonus:.2f}, "
                f"got {matchup_bonus:.2f}"
            )

    def test_fallback_to_fixture_outlook_without_betting_odds(self):
        """Test graceful fallback when betting odds unavailable."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "EasyFixture",
                    "position": "FWD",
                    "xP": 7.0,
                    "xP_uncertainty": 1.0,
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 60.0,
                    # NO betting odds columns
                },
                {
                    "player_id": 2,
                    "web_name": "HardFixture",
                    "position": "MID",
                    "xP": 7.0,
                    "xP_uncertainty": 1.0,
                    "price": 10.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🔴 Hard",
                    "status": "a",
                    "selected_by_percent": 60.0,
                    # NO betting odds columns
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=2)
        candidates = {c["web_name"]: c for c in result["top_candidates"]}

        # Should use fixture outlook for both template and matchup quality
        easy = candidates["EasyFixture"]
        hard = candidates["HardFixture"]

        # Easy fixture should have higher bonuses
        assert easy["ownership_bonus"] > hard["ownership_bonus"]
        assert easy["matchup_bonus"] > hard["matchup_bonus"]

        # Check fallback values: Easy → 0.20, Hard → 0.0
        assert abs(easy["matchup_bonus"] - 0.20) < 0.01
        assert abs(hard["matchup_bonus"] - 0.0) < 0.01


class TestCombinedScoring:
    """Test combined effect of template protection + matchup quality + uncertainty."""

    def test_haaland_vs_kudus_scenario(self):
        """Test the original GW10 scenario: Haaland (template) vs Kudus (differential)."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Haaland",
                    "position": "FWD",
                    "xP": 8.0,
                    "xP_uncertainty": 1.2,
                    "price": 15.0,
                    "expected_minutes": 85,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 62.0,  # Template
                    "team_win_probability": 0.72,  # Man City favorite
                    "team_expected_goals": 2.3,
                },
                {
                    "player_id": 2,
                    "web_name": "Kudus",
                    "position": "MID",
                    "xP": 7.5,
                    "xP_uncertainty": 1.5,
                    "price": 8.5,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟡 Average",
                    "status": "a",
                    "selected_by_percent": 18.0,  # Not template
                    "team_win_probability": 0.45,
                    "team_expected_goals": 1.4,
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=2)

        # Haaland should be captain (template protection dominates)
        assert result["captain"]["web_name"] == "Haaland"
        assert result["vice_captain"]["web_name"] == "Kudus"

        # Check scoring breakdown
        candidates = {c["web_name"]: c for c in result["top_candidates"]}
        haaland = candidates["Haaland"]
        kudus = candidates["Kudus"]

        # Haaland should have template + matchup bonuses
        # ownership_factor = (62-50)/20 = 0.6
        # fixture_quality = min(0.72/0.70, 1.0) = 1.0
        # ownership_bonus = 0.6 * 1.0 * 0.40 = 0.24 (24%)
        # xg_factor = (2.3-1.0)/1.0 = 1.0 (capped at 1.0), matchup = 0.25
        assert abs(haaland["ownership_bonus"] - 0.24) < 0.05
        assert abs(haaland["matchup_bonus"] - 0.25) < 0.05

        # Kudus should have NO template bonus
        assert kudus["ownership_bonus"] == 0.0
        # But some matchup bonus
        assert kudus["matchup_bonus"] > 0.0

        # Haaland's final score should be higher despite slightly lower xP
        assert haaland["captain_score"] > kudus["captain_score"]

    def test_upside_boost_for_high_uncertainty(self):
        """Test that high uncertainty INCREASES captain score (seeking ceiling)."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "LowVariance",
                    "position": "FWD",
                    "xP": 8.0,
                    "xP_uncertainty": 0.5,  # Low uncertainty
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 40.0,
                    "team_win_probability": 0.65,
                    "team_expected_goals": 2.0,
                },
                {
                    "player_id": 2,
                    "web_name": "HighVariance",
                    "position": "MID",
                    "xP": 8.0,
                    "xP_uncertainty": 3.0,  # High uncertainty
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 40.0,
                    "team_win_probability": 0.65,
                    "team_expected_goals": 2.0,
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=2)
        candidates = {c["web_name"]: c for c in result["top_candidates"]}

        low_var = candidates["LowVariance"]
        high_var = candidates["HighVariance"]

        # NEW LOGIC: High uncertainty should score HIGHER (seeking ceiling)
        assert high_var["captain_score"] > low_var["captain_score"]

        # Check upside boosts (percentile-based calculation)
        # low_var (uncertainty 0.5 <= 1.5): uses 90th percentile → 1.28 * 0.5 = 0.64
        # high_var (uncertainty 3.0 > 1.5): uses 95th percentile → 1.645 * 3.0 = 4.935
        low_var_boost = low_var["upside_boost"]
        high_var_boost = high_var["upside_boost"]

        assert abs(low_var_boost - 0.64) < 0.1
        assert abs(high_var_boost - 4.935) < 0.1
        assert high_var_boost > low_var_boost  # Higher uncertainty = higher boost

    def test_premium_differential_with_high_xp(self):
        """Test that high-xP differentials can compete if matchup is much better."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Template",
                    "position": "FWD",
                    "xP": 7.0,
                    "xP_uncertainty": 1.0,
                    "price": 15.0,
                    "expected_minutes": 85,
                    "fixture_outlook": "🟡 Average",
                    "status": "a",
                    "selected_by_percent": 55.0,
                    "team_win_probability": 0.50,  # Average fixture
                    "team_expected_goals": 1.5,
                },
                {
                    "player_id": 2,
                    "web_name": "Differential",
                    "position": "MID",
                    "xP": 9.0,  # Much higher xP
                    "xP_uncertainty": 0.8,  # Low uncertainty
                    "price": 7.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 8.0,
                    "team_win_probability": 0.70,  # Better fixture
                    "team_expected_goals": 2.2,
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=2)

        # With much higher xP + better matchup, differential should win
        assert result["captain"]["web_name"] == "Differential"


class TestScoringComponentsOutput:
    """Test that scoring components are correctly stored for analysis."""

    def test_all_scoring_components_present(self):
        """Test that all scoring components are stored in candidate."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Player",
                    "position": "FWD",
                    "xP": 8.0,
                    "xP_uncertainty": 1.0,
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 60.0,
                    "team_win_probability": 0.70,
                    "team_expected_goals": 2.0,
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["top_candidates"][0]

        # Check all scoring components are present
        assert "captain_score" in candidate
        assert "base_score" in candidate
        assert "upside_boost" in candidate
        assert "xP_upside" in candidate
        assert "ownership_bonus" in candidate
        assert "matchup_bonus" in candidate

        # Check they are numeric
        assert isinstance(candidate["captain_score"], (int, float))
        assert isinstance(candidate["base_score"], (int, float))
        assert isinstance(candidate["upside_boost"], (int, float))
        assert isinstance(candidate["xP_upside"], (int, float))
        assert isinstance(candidate["ownership_bonus"], (int, float))
        assert isinstance(candidate["matchup_bonus"], (int, float))

    def test_scoring_component_relationships(self):
        """Test mathematical relationships between scoring components."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Player",
                    "position": "FWD",
                    "xP": 8.0,
                    "xP_uncertainty": 1.0,
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 60.0,
                    "team_win_probability": 0.70,
                    "team_expected_goals": 2.0,
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["top_candidates"][0]

        xp = 8.0
        uncertainty = 1.0
        base_score = candidate["base_score"]
        upside_boost = candidate["upside_boost"]
        xp_upside = candidate["xP_upside"]
        ownership_bonus = candidate["ownership_bonus"]
        matchup_bonus = candidate["matchup_bonus"]
        final_score = candidate["captain_score"]

        # Upside boost should be 1.28 * uncertainty
        assert abs(upside_boost - (1.28 * uncertainty)) < 0.1

        # xP_upside should be xP + upside_boost
        assert abs(xp_upside - (xp + upside_boost)) < 0.1

        # Base score should be upside * 2
        assert abs(base_score - (xp_upside * 2)) < 0.1

        # Final = base * (1 + ownership_bonus + matchup_bonus)
        expected_final = base_score * (1 + ownership_bonus + matchup_bonus)

        assert abs(final_score - expected_final) < 0.1


# ============================================================================
# Hauler Strategy Tests (2025/26 Enhancement)
# ============================================================================


class TestHaulerCeilingStrategy:
    """Test ceiling-seeking captain selection based on top 1% manager analysis.

    Based on analysis of top 1% FPL managers:
    - Top managers average 18.7 captain points vs 15.0 for average
    - They capture 3.0 haulers per GW vs 2.4 for average
    - Key insight: They identify OPTIMAL haulers with highest ceilings
    """

    def test_high_uncertainty_uses_95th_percentile(self):
        """Test that high-uncertainty players use 95th percentile (1.645 * std)."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Explosive",
                    "position": "FWD",
                    "xP": 8.0,
                    "xP_uncertainty": 2.0,  # High uncertainty (> 1.5)
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 40.0,
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["top_candidates"][0]

        # Should use 95th percentile: 8.0 + 1.645 * 2.0 = 11.29
        expected_upside = 8.0 + 1.645 * 2.0

        assert abs(candidate["xP_upside"] - expected_upside) < 0.1

    def test_low_uncertainty_uses_90th_percentile(self):
        """Test that low-uncertainty players use 90th percentile (1.28 * std)."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Consistent",
                    "position": "MID",
                    "xP": 8.0,
                    "xP_uncertainty": 1.0,  # Low uncertainty (<= 1.5)
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 40.0,
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["top_candidates"][0]

        # Should use 90th percentile: 8.0 + 1.28 * 1.0 = 9.28
        expected_upside = 8.0 + 1.28 * 1.0

        assert abs(candidate["xP_upside"] - expected_upside) < 0.1

    def test_haul_bonus_for_high_ceiling_players(self):
        """Test that players with ceiling > 12 get haul bonus (up to 25%)."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "HaulPotential",
                    "position": "FWD",
                    "xP": 9.0,
                    "xP_uncertainty": 3.0,  # Ceiling = 9 + 2.33*3 = 15.99 → bonus
                    "price": 14.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 40.0,
                },
                {
                    "player_id": 2,
                    "web_name": "LowCeiling",
                    "position": "MID",
                    "xP": 6.0,
                    "xP_uncertainty": 1.0,  # Ceiling = 6 + 2.33*1 = 8.33 → no bonus
                    "price": 8.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 40.0,
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=2)
        candidates = {c["web_name"]: c for c in result["top_candidates"]}

        # HaulPotential should have haul bonus
        assert "haul_bonus" in candidates["HaulPotential"]
        assert candidates["HaulPotential"]["haul_bonus"] > 0

        # LowCeiling should have no haul bonus
        assert candidates["LowCeiling"]["haul_bonus"] == 0

    def test_haul_bonus_capped_at_25_percent(self):
        """Test that haul bonus is capped at 25% maximum."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "MassiveCeiling",
                    "position": "FWD",
                    "xP": 10.0,
                    "xP_uncertainty": 8.0,  # Ceiling = 10 + 2.33*8 = 28.64 → max bonus
                    "price": 15.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 60.0,
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["top_candidates"][0]

        # Haul bonus should be capped at 0.25 (25%)
        assert candidate["haul_bonus"] == 0.25

    def test_explosive_player_beats_consistent_with_same_xp(self):
        """Test that high-ceiling player ranks higher than consistent one with same xP."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Explosive",
                    "position": "FWD",
                    "xP": 8.0,
                    "xP_uncertainty": 3.0,  # High variance
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 40.0,
                },
                {
                    "player_id": 2,
                    "web_name": "Consistent",
                    "position": "MID",
                    "xP": 8.0,  # Same xP
                    "xP_uncertainty": 0.5,  # Low variance
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 40.0,
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=2)

        # Explosive player should be captain (seeking ceiling)
        assert result["captain"]["web_name"] == "Explosive"
        assert result["vice_captain"]["web_name"] == "Consistent"

    def test_upside_percentile_transition_at_1_5(self):
        """Test that percentile transition happens at uncertainty = 1.5."""
        service = OptimizationService()

        # Just below threshold (1.49)
        players_below = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Below",
                    "position": "MID",
                    "xP": 8.0,
                    "xP_uncertainty": 1.49,
                    "price": 10.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                },
            ]
        )

        # Just above threshold (1.51)
        players_above = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Above",
                    "position": "MID",
                    "xP": 8.0,
                    "xP_uncertainty": 1.51,
                    "price": 10.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                },
            ]
        )

        result_below = service.get_captain_recommendation(players_below, top_n=1)
        result_above = service.get_captain_recommendation(players_above, top_n=1)

        below_upside = result_below["top_candidates"][0]["xP_upside"]
        above_upside = result_above["top_candidates"][0]["xP_upside"]

        # Below threshold: 8.0 + 1.28 * 1.49 = 9.9072
        expected_below = 8.0 + 1.28 * 1.49

        # Above threshold: 8.0 + 1.645 * 1.51 = 10.484
        expected_above = 8.0 + 1.645 * 1.51

        assert abs(below_upside - expected_below) < 0.1
        assert abs(above_upside - expected_above) < 0.1

        # Higher percentile should give higher upside
        assert above_upside > below_upside


class TestHaulBonusCalculation:
    """Detailed tests for haul bonus calculation."""

    def test_haul_bonus_formula(self):
        """Test haul bonus formula: min((ceiling - 12) * 0.03, 0.25)."""
        service = OptimizationService()

        # Formula: ceiling = xP + 2.33 * uncertainty
        # haul_bonus = min((ceiling - 12) * 0.03, 0.25) if ceiling > 12 else 0
        test_cases = [
            # (xP, uncertainty, expected_haul_bonus)
            (8.0, 1.0, 0.0),  # Ceiling = 8 + 2.33 = 10.33 < 12 → no bonus
            (10.0, 1.0, 0.01),  # Ceiling = 10 + 2.33 = 12.33 → 0.33 * 0.03 = 0.0099
            (10.0, 2.0, 0.08),  # Ceiling = 10 + 4.66 = 14.66 → 2.66 * 0.03 = 0.0798
            (10.0, 4.0, 0.22),  # Ceiling = 10 + 9.32 = 19.32 → 7.32 * 0.03 = 0.2196
            (
                12.0,
                4.0,
                0.25,
            ),  # Ceiling = 12 + 9.32 = 21.32 → 9.32 * 0.03 = 0.28 → capped at 0.25
        ]

        for xp, uncertainty, expected_bonus in test_cases:
            players_df = pd.DataFrame(
                [
                    {
                        "player_id": 1,
                        "web_name": "Test",
                        "position": "FWD",
                        "xP": xp,
                        "xP_uncertainty": uncertainty,
                        "price": 10.0,
                        "expected_minutes": 90,
                        "fixture_outlook": "🟢 Easy",
                        "status": "a",
                    },
                ]
            )

            result = service.get_captain_recommendation(players_df, top_n=1)
            actual_bonus = result["top_candidates"][0]["haul_bonus"]

            assert abs(actual_bonus - expected_bonus) < 0.02, (
                f"xP={xp}, uncertainty={uncertainty}: "
                f"expected haul_bonus ~{expected_bonus:.3f}, got {actual_bonus:.3f}"
            )

    def test_haul_bonus_requires_uncertainty(self):
        """Test that haul bonus requires xP_uncertainty column."""
        service = OptimizationService()

        # Player without uncertainty data
        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "NoUncertainty",
                    "position": "FWD",
                    "xP": 10.0,
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)
        candidate = result["top_candidates"][0]

        # Should have no haul bonus without uncertainty data
        assert candidate["haul_bonus"] == 0


class TestBackwardsCompatibility:
    """Test that new functionality doesn't break existing behavior."""

    def test_works_without_ownership_data(self):
        """Test that system works when ownership data is missing."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Player",
                    "position": "FWD",
                    "xP": 8.0,
                    "xP_uncertainty": 1.0,
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    # NO ownership data
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)

        # Should work and return a captain
        assert result["captain"]["web_name"] == "Player"

        # Should have no ownership bonus
        candidate = result["top_candidates"][0]
        assert candidate["ownership_bonus"] == 0.0

    def test_works_without_betting_odds(self):
        """Test that system works when betting odds are missing."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Player",
                    "position": "FWD",
                    "xP": 8.0,
                    "xP_uncertainty": 1.0,
                    "price": 12.0,
                    "expected_minutes": 90,
                    "fixture_outlook": "🟢 Easy",
                    "status": "a",
                    "selected_by_percent": 60.0,
                    # NO betting odds
                },
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)

        # Should work and use fixture outlook fallback
        assert result["captain"]["web_name"] == "Player"

        candidate = result["top_candidates"][0]
        # Should have bonuses from fixture outlook fallback
        assert candidate["ownership_bonus"] > 0.0
        assert candidate["matchup_bonus"] > 0.0

    def test_minimal_player_data_still_works(self):
        """Test with absolute minimum player data."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "web_name": "Minimal",
                    "position": "FWD",
                    "xP": 7.0,
                }
            ]
        )

        result = service.get_captain_recommendation(players_df, top_n=1)

        # Should work with defaults
        assert result["captain"]["web_name"] == "Minimal"
        assert result["captain"]["captain_points"] == 14.0
