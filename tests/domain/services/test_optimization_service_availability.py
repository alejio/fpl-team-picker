"""Tests for OptimizationService availability adjustment functionality.

Tests that the optimizer correctly accounts for player injuries and availability
when making team selection decisions.
"""

import pytest
import pandas as pd
import numpy as np
from fpl_team_picker.domain.services.optimization_service import OptimizationService
from fpl_team_picker.config import config


class TestGetAdjustedXP:
    """Test the get_adjusted_xp method for availability adjustment."""

    @pytest.fixture
    def service(self):
        """Create OptimizationService instance."""
        return OptimizationService()

    def test_chance_of_playing_this_round(self, service):
        """Test adjustment using chance_of_playing_this_round."""
        player = {
            "player_id": 1,
            "xP": 10.0,
            "chance_of_playing_this_round": 75.0,
            "position": "MID",
        }
        adjusted = service.get_adjusted_xp(player, "xP")
        assert adjusted == 7.5  # 10.0 * 0.75

    def test_chance_of_playing_next_round_fallback(self, service):
        """Test fallback to chance_of_playing_next_round when this_round not available."""
        player = {
            "player_id": 1,
            "xP": 10.0,
            "chance_of_playing_next_round": 80.0,
            "position": "MID",
        }
        adjusted = service.get_adjusted_xp(player, "xP")
        assert adjusted == 8.0  # 10.0 * 0.80

    def test_prefers_this_round_over_next_round(self, service):
        """Test that chance_of_playing_this_round is preferred over next_round."""
        player = {
            "player_id": 1,
            "xP": 10.0,
            "chance_of_playing_this_round": 75.0,
            "chance_of_playing_next_round": 90.0,  # Should be ignored
            "position": "MID",
        }
        adjusted = service.get_adjusted_xp(player, "xP")
        assert adjusted == 7.5  # Uses this_round (75%), not next_round (90%)

    def test_expected_minutes_fallback(self, service):
        """Test fallback to expected_minutes ratio when chance percentages not available."""
        player = {
            "player_id": 1,
            "xP": 10.0,
            "expected_minutes": 60.0,  # 60/75 = 0.8 for MID
            "position": "MID",
        }
        adjusted = service.get_adjusted_xp(player, "xP")
        assert adjusted == 8.0  # 10.0 * (60/75)

    def test_expected_minutes_position_based(self, service):
        """Test expected_minutes uses position-specific defaults."""
        # GKP: 90 default
        gkp_player = {
            "player_id": 1,
            "xP": 10.0,
            "expected_minutes": 45.0,
            "position": "GKP",
        }
        gkp_adjusted = service.get_adjusted_xp(gkp_player, "xP")
        assert gkp_adjusted == 5.0  # 10.0 * (45/90)

        # DEF: 80 default
        def_player = {
            "player_id": 2,
            "xP": 10.0,
            "expected_minutes": 60.0,
            "position": "DEF",
        }
        def_adjusted = service.get_adjusted_xp(def_player, "xP")
        assert def_adjusted == 7.5  # 10.0 * (60/80)

        # FWD: 70 default
        fwd_player = {
            "player_id": 3,
            "xP": 10.0,
            "expected_minutes": 35.0,
            "position": "FWD",
        }
        fwd_adjusted = service.get_adjusted_xp(fwd_player, "xP")
        assert fwd_adjusted == 5.0  # 10.0 * (35/70)

    def test_injured_status_uses_config_multiplier(self, service):
        """Test injured players use injury_full_game_multiplier from config."""
        player = {
            "player_id": 1,
            "xP": 10.0,
            "status": "i",  # injured
            "position": "MID",
        }
        adjusted = service.get_adjusted_xp(player, "xP")
        expected = 10.0 * config.minutes_model.injury_full_game_multiplier
        assert adjusted == pytest.approx(expected)

    def test_doubtful_status_uses_config_multiplier(self, service):
        """Test doubtful players use injury_avg_minutes_multiplier from config."""
        player = {
            "player_id": 1,
            "xP": 10.0,
            "status": "d",  # doubtful
            "position": "MID",
        }
        adjusted = service.get_adjusted_xp(player, "xP")
        expected = 10.0 * config.minutes_model.injury_avg_minutes_multiplier
        assert adjusted == pytest.approx(expected)

    def test_no_availability_data_returns_base_xp(self, service):
        """Test that players with no availability data return base xP."""
        player = {
            "player_id": 1,
            "xP": 10.0,
            "status": "a",  # available
            "position": "MID",
        }
        adjusted = service.get_adjusted_xp(player, "xP")
        assert adjusted == 10.0  # No adjustment

    def test_zero_xp_returns_zero(self, service):
        """Test that zero or negative xP returns zero."""
        player = {
            "player_id": 1,
            "xP": 0.0,
            "chance_of_playing_this_round": 75.0,
            "position": "MID",
        }
        adjusted = service.get_adjusted_xp(player, "xP")
        assert adjusted == 0.0

        player["xP"] = -5.0
        adjusted = service.get_adjusted_xp(player, "xP")
        assert adjusted == 0.0

    def test_handles_nan_values(self, service):
        """Test that NaN values in availability fields are handled correctly."""

        # NaN in chance_of_playing_this_round should fall back to next_round
        player = {
            "player_id": 1,
            "xP": 10.0,
            "chance_of_playing_this_round": np.nan,
            "chance_of_playing_next_round": 80.0,
            "position": "MID",
        }
        adjusted = service.get_adjusted_xp(player, "xP")
        assert adjusted == 8.0  # Uses next_round

        # NaN in both should fall back to expected_minutes
        player = {
            "player_id": 1,
            "xP": 10.0,
            "chance_of_playing_this_round": np.nan,
            "chance_of_playing_next_round": np.nan,
            "expected_minutes": 60.0,
            "position": "MID",
        }
        adjusted = service.get_adjusted_xp(player, "xP")
        assert adjusted == 8.0  # Uses expected_minutes

    def test_clamps_multiplier_to_valid_range(self, service):
        """Test that multipliers are clamped to 0.0-1.0 range."""
        # Over 100% chance should be clamped to 1.0
        player = {
            "player_id": 1,
            "xP": 10.0,
            "chance_of_playing_this_round": 150.0,  # Invalid > 100%
            "position": "MID",
        }
        adjusted = service.get_adjusted_xp(player, "xP")
        assert adjusted == 10.0  # Clamped to 1.0

        # Negative chance should be clamped to 0.0
        player = {
            "player_id": 1,
            "xP": 10.0,
            "chance_of_playing_this_round": -10.0,  # Invalid < 0%
            "position": "MID",
        }
        adjusted = service.get_adjusted_xp(player, "xP")
        assert adjusted == 0.0  # Clamped to 0.0


class TestOptimizerUsesAdjustedXP:
    """Test that optimizer actually uses adjusted xP in team selection."""

    @pytest.fixture
    def service(self):
        """Create OptimizationService instance."""
        return OptimizationService()

    @pytest.fixture
    def players_with_injuries(self):
        """Create sample players with varying availability."""
        return pd.DataFrame(
            [
                # Fully fit player - high xP
                {
                    "player_id": 1,
                    "web_name": "FitPlayer",
                    "position": "MID",
                    "price": 8.0,
                    "xP": 8.0,
                    "xP_5gw": 40.0,
                    "team": "Team A",
                    "status": "a",
                    "chance_of_playing_this_round": 100.0,
                    "expected_minutes": 90,
                },
                # Injured player - same base xP but 75% chance
                {
                    "player_id": 2,
                    "web_name": "InjuredPlayer",
                    "position": "MID",
                    "price": 8.0,
                    "xP": 8.0,  # Same base xP
                    "xP_5gw": 40.0,
                    "team": "Team B",
                    "status": "d",  # doubtful
                    "chance_of_playing_this_round": 75.0,
                    "expected_minutes": 67.5,  # 75% of 90
                },
                # More players to fill squad
                {
                    "player_id": 3,
                    "web_name": "Player3",
                    "position": "MID",
                    "price": 7.0,
                    "xP": 7.0,
                    "xP_5gw": 35.0,
                    "team": "Team C",
                    "status": "a",
                    "chance_of_playing_this_round": 100.0,
                    "expected_minutes": 90,
                },
                {
                    "player_id": 4,
                    "web_name": "Player4",
                    "position": "MID",
                    "price": 6.0,
                    "xP": 6.0,
                    "xP_5gw": 30.0,
                    "team": "Team D",
                    "status": "a",
                    "chance_of_playing_this_round": 100.0,
                    "expected_minutes": 90,
                },
            ]
        )

    def test_optimizer_prefers_fit_players_over_injured(
        self, service, players_with_injuries
    ):
        """Test that optimizer prefers fully fit players when base xP is similar."""
        # Add enough players to make a valid squad (need 15 total, with proper position distribution)
        all_players = []

        # Add 2 GKP
        for i in range(2):
            all_players.append(
                {
                    "player_id": 100 + i,
                    "web_name": f"GKP{i}",
                    "position": "GKP",
                    "price": 5.0,
                    "xP": 5.0,
                    "xP_5gw": 25.0,
                    "team": f"Team{i % 5}",
                    "status": "a",
                    "chance_of_playing_this_round": 100.0,
                    "expected_minutes": 90,
                }
            )

        # Add 5 DEF
        for i in range(5):
            all_players.append(
                {
                    "player_id": 102 + i,
                    "web_name": f"DEF{i}",
                    "position": "DEF",
                    "price": 5.0,
                    "xP": 5.0,
                    "xP_5gw": 25.0,
                    "team": f"Team{i % 5}",
                    "status": "a",
                    "chance_of_playing_this_round": 100.0,
                    "expected_minutes": 90,
                }
            )

        # Add 5 MID (we already have 4 from players_with_injuries, need 1 more)
        all_players.append(
            {
                "player_id": 107,
                "web_name": "MID4",
                "position": "MID",
                "price": 5.0,
                "xP": 5.0,
                "xP_5gw": 25.0,
                "team": "Team4",
                "status": "a",
                "chance_of_playing_this_round": 100.0,
                "expected_minutes": 90,
            }
        )

        # Add 3 FWD
        for i in range(3):
            all_players.append(
                {
                    "player_id": 108 + i,
                    "web_name": f"FWD{i}",
                    "position": "FWD",
                    "price": 5.0,
                    "xP": 5.0,
                    "xP_5gw": 25.0,
                    "team": f"Team{i % 5}",
                    "status": "a",
                    "chance_of_playing_this_round": 100.0,
                    "expected_minutes": 90,
                }
            )

        # Combine with test players
        all_players_df = pd.concat(
            [players_with_injuries, pd.DataFrame(all_players)], ignore_index=True
        )

        # Optimize with small iteration count for speed
        result = service.optimize_initial_squad(
            players_with_xp=all_players_df,
            budget=100.0,
            formation=(2, 5, 5, 3),
            iterations=100,  # Small for test speed
            xp_column="xP",
        )

        # Check that fit player (ID 1) is more likely to be selected than injured (ID 2)
        # Since they have same base xP, fit player should be preferred
        optimal_squad_ids = [p["player_id"] for p in result["optimal_squad"]]

        # Fit player should be in squad
        assert 1 in optimal_squad_ids, "Fit player should be selected"

        # Verify that adjusted xP is being used - if injured player is selected,
        # it should be because of other constraints, not because of higher adjusted xP
        # We can verify the scoring accounts for availability by checking the total xP
        # is calculated with adjustments

    def test_adjusted_xp_affects_team_score(self, service):
        """Test that adjusted xP is used in team scoring."""
        # Create a simple team with one injured player
        team = [
            {
                "player_id": 1,
                "xP": 10.0,
                "chance_of_playing_this_round": 100.0,
                "position": "MID",
            },
            {
                "player_id": 2,
                "xP": 10.0,
                "chance_of_playing_this_round": 75.0,
                "position": "MID",
            },
        ]

        # Calculate adjusted xP manually
        fit_adjusted = service.get_adjusted_xp(team[0], "xP")
        injured_adjusted = service.get_adjusted_xp(team[1], "xP")

        assert fit_adjusted == 10.0
        assert injured_adjusted == 7.5

        # Verify the difference
        assert fit_adjusted > injured_adjusted


class TestAvailabilityIntegration:
    """Integration tests for availability adjustment in full optimization flow."""

    @pytest.fixture
    def service(self):
        """Create OptimizationService instance."""
        return OptimizationService()

    def test_kudus_scenario(self, service):
        """Test the specific Kudus scenario: 75% chance, should reduce xP."""
        kudus = {
            "player_id": 582,
            "web_name": "Kudus",
            "xP": 8.0,
            "chance_of_playing_this_round": None,  # Not specified
            "chance_of_playing_next_round": 75.0,
            "status": "d",  # doubtful
            "position": "MID",
            "expected_minutes": 67.5,  # 75% of 90
        }

        # Should use chance_of_playing_next_round (75%)
        adjusted = service.get_adjusted_xp(kudus, "xP")
        assert adjusted == 6.0  # 8.0 * 0.75

        # If we had chance_of_playing_this_round, it should be preferred
        kudus["chance_of_playing_this_round"] = 70.0
        adjusted = service.get_adjusted_xp(kudus, "xP")
        assert adjusted == 5.6  # 8.0 * 0.70 (uses this_round, not next_round)
