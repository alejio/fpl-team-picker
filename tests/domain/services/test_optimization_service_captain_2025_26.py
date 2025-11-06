"""Unit tests for enhanced captain selection (2025/26 upgrade).

Tests for Phase 1 improvements:
- Template protection (20-40% boost for >50% ownership)
- Betting odds integration (team_win_probability, team_expected_goals)
- Matchup quality bonus (0-25% for all players)
- Graceful fallback when betting odds unavailable
"""

import pandas as pd
from fpl_team_picker.domain.services.optimization_service import OptimizationService


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
            (50.0, 0.0),   # Exactly at threshold
            (55.0, 0.25),  # 25% of range
            (60.0, 0.5),   # 50% of range
            (65.0, 0.75),  # 75% of range
            (70.0, 1.0),   # Max
            (80.0, 1.0),   # Above max (capped)
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
            (0.5, 0.0),     # Very low xG → 0% bonus (clamped at 0)
            (1.0, 0.0),     # Neutral baseline → 0% bonus
            (1.25, 0.0625), # PL average → 6.25% bonus (0.25 * 0.25)
            (1.5, 0.125),   # Above average → 12.5% bonus (0.5 * 0.25)
            (2.0, 0.25),    # High xG → 25% bonus (1.0 * 0.25)
            (2.5, 0.25),    # Very high xG → 25% bonus (capped at 1.0)
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

    def test_uncertainty_penalty_preserved(self):
        """Test that uncertainty penalty is still applied correctly."""
        service = OptimizationService()

        players_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Confident",
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
                    "web_name": "Uncertain",
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

        confident = candidates["Confident"]
        uncertain = candidates["Uncertain"]

        # Confident prediction should score higher
        assert confident["captain_score"] > uncertain["captain_score"]

        # Check uncertainty penalties
        # confident: 0.5/8.0 * 0.30 = 0.01875 penalty
        # uncertain: 3.0/8.0 * 0.30 = 0.1125 penalty
        confident_penalty = confident["uncertainty_penalty"]
        uncertain_penalty = uncertain["uncertainty_penalty"]

        assert confident_penalty < uncertain_penalty
        assert uncertain_penalty > 0.10  # High uncertainty → significant penalty

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
        assert "uncertainty_penalty" in candidate
        assert "ownership_bonus" in candidate
        assert "matchup_bonus" in candidate

        # Check they are numeric
        assert isinstance(candidate["captain_score"], (int, float))
        assert isinstance(candidate["base_score"], (int, float))
        assert isinstance(candidate["uncertainty_penalty"], (int, float))
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

        base_score = candidate["base_score"]
        uncertainty_penalty = candidate["uncertainty_penalty"]
        ownership_bonus = candidate["ownership_bonus"]
        matchup_bonus = candidate["matchup_bonus"]
        final_score = candidate["captain_score"]

        # Base score should be xP * 2
        assert abs(base_score - 16.0) < 0.01

        # Risk-adjusted = base / (1 + penalty)
        risk_adjusted = base_score / (1 + uncertainty_penalty)

        # Final = risk_adjusted * (1 + ownership_bonus + matchup_bonus)
        expected_final = risk_adjusted * (1 + ownership_bonus + matchup_bonus)

        assert abs(final_score - expected_final) < 0.01


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
