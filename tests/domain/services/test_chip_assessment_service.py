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
from fpl_team_picker.adapters.database_repositories import DatabasePlayerRepository


class TestChipAssessmentServiceIntegration:
    """Integration tests for ChipAssessmentService with real data."""

    @pytest.fixture
    def chip_service(self):
        """Create chip assessment service."""
        return ChipAssessmentService()

    @pytest.fixture
    def sample_gameweek_data(self):
        """Load sample gameweek data with XP calculations."""
        # Initialize repositories (team and fixture repos not needed, using FPL client)
        player_repo = DatabasePlayerRepository()

        # Create services
        data_service = DataOrchestrationService(player_repo, None, None)
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


# =============================================================================
# Unit Test Fixtures for Lookahead Analysis
# =============================================================================


@pytest.fixture
def unit_chip_service():
    """Create a ChipAssessmentService instance for unit tests."""
    return ChipAssessmentService()


@pytest.fixture
def sample_fixtures_unit():
    """Create sample fixtures data for unit testing."""
    fixtures = []
    teams = list(range(1, 21))

    for gw in range(1, 20):
        for i in range(0, 20, 2):
            fixtures.append({
                "gameweek": gw,
                "team_h": teams[i],
                "team_a": teams[i + 1],
                "team_h_difficulty": 3,
                "team_a_difficulty": 3,
            })

    return pd.DataFrame(fixtures)


@pytest.fixture
def dgw_fixtures_unit():
    """Create fixtures with a double gameweek for unit testing."""
    fixtures = []
    teams = list(range(1, 21))

    # Normal GW11
    for i in range(0, 20, 2):
        fixtures.append({
            "gameweek": 11,
            "team_h": teams[i],
            "team_a": teams[i + 1],
            "team_h_difficulty": 3,
            "team_a_difficulty": 3,
        })

    # DGW12 - teams 1-4 play twice
    for i in range(0, 20, 2):
        fixtures.append({
            "gameweek": 12,
            "team_h": teams[i],
            "team_a": teams[i + 1],
            "team_h_difficulty": 3,
            "team_a_difficulty": 3,
        })
    # Extra fixtures for DGW teams
    fixtures.append({
        "gameweek": 12,
        "team_h": 1,
        "team_a": 3,
        "team_h_difficulty": 2,
        "team_a_difficulty": 2,
    })
    fixtures.append({
        "gameweek": 12,
        "team_h": 2,
        "team_a": 4,
        "team_h_difficulty": 2,
        "team_a_difficulty": 2,
    })

    return pd.DataFrame(fixtures)


@pytest.fixture
def sample_squad_unit():
    """Create a sample 15-player squad for unit testing."""
    return pd.DataFrame([
        {"player_id": 1, "web_name": "GK1", "position": "GKP", "team": 1, "price": 5.0, "xP": 4.0, "status": "a"},
        {"player_id": 2, "web_name": "GK2", "position": "GKP", "team": 2, "price": 4.0, "xP": 3.5, "status": "a"},
        {"player_id": 3, "web_name": "DEF1", "position": "DEF", "team": 1, "price": 6.0, "xP": 5.0, "status": "a"},
        {"player_id": 4, "web_name": "DEF2", "position": "DEF", "team": 3, "price": 5.5, "xP": 4.5, "status": "a"},
        {"player_id": 5, "web_name": "DEF3", "position": "DEF", "team": 5, "price": 5.0, "xP": 4.0, "status": "a"},
        {"player_id": 6, "web_name": "DEF4", "position": "DEF", "team": 7, "price": 4.5, "xP": 3.5, "status": "a"},
        {"player_id": 7, "web_name": "DEF5", "position": "DEF", "team": 9, "price": 4.0, "xP": 3.0, "status": "a"},
        {"player_id": 8, "web_name": "MID1", "position": "MID", "team": 1, "price": 10.0, "xP": 7.0, "status": "a"},
        {"player_id": 9, "web_name": "MID2", "position": "MID", "team": 3, "price": 8.0, "xP": 6.0, "status": "a"},
        {"player_id": 10, "web_name": "MID3", "position": "MID", "team": 5, "price": 7.0, "xP": 5.5, "status": "a"},
        {"player_id": 11, "web_name": "MID4", "position": "MID", "team": 7, "price": 6.0, "xP": 4.5, "status": "a"},
        {"player_id": 12, "web_name": "MID5", "position": "MID", "team": 9, "price": 5.0, "xP": 4.0, "status": "a"},
        {"player_id": 13, "web_name": "FWD1", "position": "FWD", "team": 1, "price": 12.0, "xP": 8.0, "status": "a"},
        {"player_id": 14, "web_name": "FWD2", "position": "FWD", "team": 3, "price": 8.0, "xP": 6.0, "status": "a"},
        {"player_id": 15, "web_name": "FWD3", "position": "FWD", "team": 5, "price": 6.0, "xP": 4.5, "status": "a"},
    ])


@pytest.fixture
def sample_players_unit():
    """Create sample all players data for unit testing."""
    players = []
    for i in range(1, 101):
        team = (i % 20) + 1
        position = ["GKP", "DEF", "DEF", "MID", "MID", "FWD"][i % 6]
        players.append({
            "player_id": i,
            "web_name": f"Player{i}",
            "position": position,
            "team": team,
            "price": 4.0 + (i % 10),
            "xP": 3.0 + (i % 8),
            "status": "a",
        })
    return pd.DataFrame(players)


# =============================================================================
# Test Chip Deadlines
# =============================================================================


class TestChipDeadlines:
    """Tests for chip deadline awareness."""

    def test_first_set_deadline_gw19(self, unit_chip_service):
        """First set of chips must be used by GW19."""
        deadline = unit_chip_service.get_chip_deadline(current_gw=10, chip_set="first")
        assert deadline == 19

    def test_second_set_deadline_gw38(self, unit_chip_service):
        """Second set of chips must be used by GW38."""
        deadline = unit_chip_service.get_chip_deadline(current_gw=20, chip_set="second")
        assert deadline == 38

    def test_remaining_gameweeks_first_set(self, unit_chip_service):
        """Calculate remaining GWs for first set chips."""
        remaining = unit_chip_service.get_remaining_gameweeks_for_chips(
            current_gw=15, chip_set="first"
        )
        assert remaining == 4  # GW16, 17, 18, 19 = 4 remaining windows

    def test_remaining_gameweeks_second_set(self, unit_chip_service):
        """Calculate remaining GWs for second set chips."""
        remaining = unit_chip_service.get_remaining_gameweeks_for_chips(
            current_gw=35, chip_set="second"
        )
        assert remaining == 3  # GW36, 37, 38 = 3 remaining windows

    def test_warns_when_chips_expiring_soon(self, unit_chip_service):
        """Warning when 3 or fewer GWs remain for chips."""
        warning = unit_chip_service.get_chip_deadline_warning(current_gw=17, chip_set="first")
        assert warning is not None
        assert "expiring" in warning.lower() or "remaining" in warning.lower()

    def test_no_warning_when_chips_not_urgent(self, unit_chip_service):
        """No warning when plenty of GWs remain."""
        warning = unit_chip_service.get_chip_deadline_warning(current_gw=5, chip_set="first")
        assert warning is None

    def test_determine_chip_set_from_gameweek(self, unit_chip_service):
        """Correctly determine which chip set based on current GW."""
        assert unit_chip_service.get_current_chip_set(current_gw=10) == "first"
        assert unit_chip_service.get_current_chip_set(current_gw=19) == "first"
        assert unit_chip_service.get_current_chip_set(current_gw=20) == "second"
        assert unit_chip_service.get_current_chip_set(current_gw=38) == "second"


# =============================================================================
# Test DGW/BGW Detection
# =============================================================================


class TestDGWDetection:
    """Tests for double/blank gameweek detection."""

    def test_detects_teams_with_double_fixtures(self, unit_chip_service, dgw_fixtures_unit):
        """Detect teams playing twice in a gameweek."""
        dgw_teams = unit_chip_service.detect_double_gameweek_teams(dgw_fixtures_unit, gameweek=12)

        assert len(dgw_teams) == 4
        assert 1 in dgw_teams
        assert 2 in dgw_teams
        assert 3 in dgw_teams
        assert 4 in dgw_teams

    def test_no_dgw_teams_in_normal_gameweek(self, unit_chip_service, sample_fixtures_unit):
        """No DGW teams in normal gameweek."""
        dgw_teams = unit_chip_service.detect_double_gameweek_teams(sample_fixtures_unit, gameweek=11)
        assert len(dgw_teams) == 0

    def test_detect_blank_gameweek_teams(self, unit_chip_service):
        """Detect teams with no fixtures (blank gameweek)."""
        fixtures = pd.DataFrame([
            {"gameweek": 12, "team_h": 1, "team_a": 2},
            {"gameweek": 12, "team_h": 3, "team_a": 4},
        ])

        bgw_teams = unit_chip_service.detect_blank_gameweek_teams(
            fixtures, gameweek=12, all_team_ids=[1, 2, 3, 4, 19, 20]
        )

        assert 19 in bgw_teams
        assert 20 in bgw_teams
        assert 1 not in bgw_teams

    def test_scan_dgw_calendar(self, unit_chip_service, dgw_fixtures_unit):
        """Scan multiple gameweeks for DGW opportunities."""
        dgw_calendar = unit_chip_service.scan_dgw_calendar(
            dgw_fixtures_unit, start_gw=11, end_gw=13
        )

        assert 11 not in dgw_calendar or len(dgw_calendar.get(11, [])) == 0
        assert 12 in dgw_calendar
        assert len(dgw_calendar[12]) == 4

    def test_handles_no_dgw_in_window(self, unit_chip_service, sample_fixtures_unit):
        """Gracefully handle no DGWs in scan window."""
        dgw_calendar = unit_chip_service.scan_dgw_calendar(
            sample_fixtures_unit, start_gw=1, end_gw=19
        )

        total_dgw_teams = sum(len(teams) for teams in dgw_calendar.values())
        assert total_dgw_teams == 0


# =============================================================================
# Test Free Hit Scoring
# =============================================================================


class TestFreeHitScoring:
    """Tests for Free Hit chip optimal gameweek scoring."""

    def test_recommends_dgw_over_normal_gw(
        self, unit_chip_service, dgw_fixtures_unit, sample_squad_unit, sample_players_unit
    ):
        """Free Hit should score DGW higher than normal GW."""
        score_gw11 = unit_chip_service.score_free_hit_for_gameweek(
            fixtures=dgw_fixtures_unit,
            current_squad=sample_squad_unit,
            all_players=sample_players_unit,
            gameweek=11,
        )

        score_gw12_dgw = unit_chip_service.score_free_hit_for_gameweek(
            fixtures=dgw_fixtures_unit,
            current_squad=sample_squad_unit,
            all_players=sample_players_unit,
            gameweek=12,
        )

        assert score_gw12_dgw > score_gw11

    def test_find_optimal_free_hit_gameweek(
        self, unit_chip_service, dgw_fixtures_unit, sample_squad_unit, sample_players_unit
    ):
        """Find the best gameweek to use Free Hit."""
        optimal_gw, score = unit_chip_service.find_optimal_chip_gameweek(
            chip_name="free_hit",
            fixtures=dgw_fixtures_unit,
            current_squad=sample_squad_unit,
            all_players=sample_players_unit,
            current_gw=11,
            deadline_gw=12,  # Limit to fixture range to test DGW detection
        )

        assert optimal_gw == 12  # Should recommend the DGW


# =============================================================================
# Test Bench Boost Scoring
# =============================================================================


class TestBenchBoostScoring:
    """Tests for Bench Boost chip optimal gameweek scoring."""

    def test_requires_minimum_bench_xp(self, unit_chip_service):
        """Bench Boost not recommended with weak bench."""
        # Create squad where bench players have very low xP (1.0 each = 4.0 total)
        weak_bench_squad = pd.DataFrame([
            {"player_id": i, "web_name": f"P{i}", "position": "MID", "team": i, "price": 4.0, "xP": 1.0}
            for i in range(1, 16)
        ])

        fixtures = pd.DataFrame([
            {"gameweek": 11, "team_h": 1, "team_a": 2, "team_h_difficulty": 2, "team_a_difficulty": 2}
        ])

        score = unit_chip_service.score_bench_boost_for_gameweek(
            fixtures=fixtures,
            current_squad=weak_bench_squad,
            gameweek=11,
        )

        # Bench xP of 4.0 is below the 8.0 threshold for Bench Boost
        assert score < 5.0


# =============================================================================
# Test Triple Captain Scoring
# =============================================================================


class TestTripleCaptainScoring:
    """Tests for Triple Captain chip optimal gameweek scoring."""

    def test_recommends_premium_easy_fixture(self, unit_chip_service, sample_squad_unit):
        """Triple Captain should target premium player with easy fixture."""
        fixtures = pd.DataFrame([
            {"gameweek": 11, "team_h": 1, "team_a": 2, "team_h_difficulty": 5, "team_a_difficulty": 5},
            {"gameweek": 12, "team_h": 1, "team_a": 20, "team_h_difficulty": 2, "team_a_difficulty": 2},
        ])

        score_gw11 = unit_chip_service.score_triple_captain_for_gameweek(
            fixtures=fixtures,
            current_squad=sample_squad_unit,
            gameweek=11,
        )

        score_gw12 = unit_chip_service.score_triple_captain_for_gameweek(
            fixtures=fixtures,
            current_squad=sample_squad_unit,
            gameweek=12,
        )

        assert score_gw12 > score_gw11


# =============================================================================
# Test Find Optimal Chip Gameweek
# =============================================================================


class TestFindOptimalChipGameweek:
    """Integration tests for finding optimal chip usage gameweek."""

    def test_find_optimal_gameweek_returns_best_option(
        self, unit_chip_service, dgw_fixtures_unit, sample_squad_unit, sample_players_unit
    ):
        """find_optimal_chip_gameweek returns GW with highest score."""
        optimal_gw, score = unit_chip_service.find_optimal_chip_gameweek(
            chip_name="free_hit",
            fixtures=dgw_fixtures_unit,
            current_squad=sample_squad_unit,
            all_players=sample_players_unit,
            current_gw=11,
            deadline_gw=15,
        )

        assert optimal_gw is not None
        assert score > 0
        assert 11 <= optimal_gw <= 15

    def test_respects_deadline_constraint(
        self, unit_chip_service, dgw_fixtures_unit, sample_squad_unit, sample_players_unit
    ):
        """Optimal GW must be within deadline."""
        optimal_gw, _ = unit_chip_service.find_optimal_chip_gameweek(
            chip_name="bench_boost",
            fixtures=dgw_fixtures_unit,
            current_squad=sample_squad_unit,
            all_players=sample_players_unit,
            current_gw=11,
            deadline_gw=13,
        )

        assert optimal_gw <= 13

    def test_returns_none_if_no_valid_gameweeks(self, unit_chip_service, sample_squad_unit, sample_players_unit):
        """Returns None if deadline already passed."""
        optimal_gw, score = unit_chip_service.find_optimal_chip_gameweek(
            chip_name="triple_captain",
            fixtures=pd.DataFrame(),
            current_squad=sample_squad_unit,
            all_players=sample_players_unit,
            current_gw=20,
            deadline_gw=19,
        )

        assert optimal_gw is None
        assert score == 0


# =============================================================================
# Test Promoted Team Bonus for TC
# =============================================================================


class TestPromotedTeamBonus:
    """Tests for promoted team bonus in Triple Captain scoring."""

    @pytest.fixture
    def promoted_team_fixtures(self):
        """Fixtures where captain faces promoted teams."""
        # 2025-26 promoted teams: Leeds (10), Burnley (4), Sunderland (17)
        return pd.DataFrame([
            # GW11: Team 1 vs non-promoted (team 5)
            {"gameweek": 11, "team_h": 1, "team_a": 5, "team_h_difficulty": 3, "team_a_difficulty": 3},
            # GW12: Team 1 (home) vs Leeds (promoted)
            {"gameweek": 12, "team_h": 1, "team_a": 10, "team_h_difficulty": 2, "team_a_difficulty": 4},
            # GW13: Team 1 (away) vs Burnley (promoted)
            {"gameweek": 13, "team_h": 4, "team_a": 1, "team_h_difficulty": 4, "team_a_difficulty": 2},
        ])

    def test_tc_scores_higher_vs_promoted_team(self, unit_chip_service, sample_squad_unit, promoted_team_fixtures):
        """TC should score higher when captain faces promoted team."""
        score_vs_normal = unit_chip_service.score_triple_captain_for_gameweek(
            fixtures=promoted_team_fixtures,
            current_squad=sample_squad_unit,
            gameweek=11,
        )

        score_vs_promoted = unit_chip_service.score_triple_captain_for_gameweek(
            fixtures=promoted_team_fixtures,
            current_squad=sample_squad_unit,
            gameweek=12,
        )

        assert score_vs_promoted > score_vs_normal

    def test_tc_home_vs_promoted_best_scenario(self, unit_chip_service, sample_squad_unit, promoted_team_fixtures):
        """Home fixture vs promoted team should be ideal TC scenario."""
        score_home_vs_promoted = unit_chip_service.score_triple_captain_for_gameweek(
            fixtures=promoted_team_fixtures,
            current_squad=sample_squad_unit,
            gameweek=12,  # Home vs Leeds
        )

        score_away_vs_promoted = unit_chip_service.score_triple_captain_for_gameweek(
            fixtures=promoted_team_fixtures,
            current_squad=sample_squad_unit,
            gameweek=13,  # Away vs Burnley
        )

        # Home should score at least as high as away vs promoted
        assert score_home_vs_promoted >= score_away_vs_promoted


# =============================================================================
# Test BGW-Aware Free Hit Scoring
# =============================================================================


class TestBGWFreeHitScoring:
    """Tests for Blank Gameweek awareness in Free Hit scoring."""

    @pytest.fixture
    def bgw_fixtures(self):
        """Fixtures with a blank gameweek for several teams."""
        return pd.DataFrame([
            # GW11: All teams play
            {"gameweek": 11, "team_h": 1, "team_a": 2},
            {"gameweek": 11, "team_h": 3, "team_a": 4},
            {"gameweek": 11, "team_h": 5, "team_a": 6},
            {"gameweek": 11, "team_h": 7, "team_a": 8},
            {"gameweek": 11, "team_h": 9, "team_a": 10},
            # GW12: Only 4 teams play (big BGW)
            {"gameweek": 12, "team_h": 1, "team_a": 2},
            {"gameweek": 12, "team_h": 3, "team_a": 4},
        ])

    @pytest.fixture
    def bgw_affected_squad(self):
        """Squad with many players from teams with no fixtures in BGW."""
        return pd.DataFrame([
            {"player_id": 1, "web_name": "GK1", "position": "GKP", "team": 5, "price": 5.0, "xP": 4.0, "status": "a"},
            {"player_id": 2, "web_name": "GK2", "position": "GKP", "team": 6, "price": 4.0, "xP": 3.5, "status": "a"},
            {"player_id": 3, "web_name": "DEF1", "position": "DEF", "team": 5, "price": 6.0, "xP": 5.0, "status": "a"},
            {"player_id": 4, "web_name": "DEF2", "position": "DEF", "team": 6, "price": 5.5, "xP": 4.5, "status": "a"},
            {"player_id": 5, "web_name": "DEF3", "position": "DEF", "team": 7, "price": 5.0, "xP": 4.0, "status": "a"},
            {"player_id": 6, "web_name": "DEF4", "position": "DEF", "team": 8, "price": 4.5, "xP": 3.5, "status": "a"},
            {"player_id": 7, "web_name": "DEF5", "position": "DEF", "team": 9, "price": 4.0, "xP": 3.0, "status": "a"},
            {"player_id": 8, "web_name": "MID1", "position": "MID", "team": 5, "price": 10.0, "xP": 7.0, "status": "a"},
            {"player_id": 9, "web_name": "MID2", "position": "MID", "team": 6, "price": 8.0, "xP": 6.0, "status": "a"},
            {"player_id": 10, "web_name": "MID3", "position": "MID", "team": 7, "price": 7.0, "xP": 5.5, "status": "a"},
            {"player_id": 11, "web_name": "MID4", "position": "MID", "team": 8, "price": 6.0, "xP": 4.5, "status": "a"},
            {"player_id": 12, "web_name": "MID5", "position": "MID", "team": 9, "price": 5.0, "xP": 4.0, "status": "a"},
            {"player_id": 13, "web_name": "FWD1", "position": "FWD", "team": 5, "price": 12.0, "xP": 8.0, "status": "a"},
            {"player_id": 14, "web_name": "FWD2", "position": "FWD", "team": 7, "price": 8.0, "xP": 6.0, "status": "a"},
            {"player_id": 15, "web_name": "FWD3", "position": "FWD", "team": 9, "price": 6.0, "xP": 4.5, "status": "a"},
        ])

    def test_fh_scores_high_when_squad_has_bgw_players(
        self, unit_chip_service, bgw_fixtures, bgw_affected_squad, sample_players_unit
    ):
        """Free Hit should score higher when many squad players have no fixture."""
        score_normal_gw = unit_chip_service.score_free_hit_for_gameweek(
            fixtures=bgw_fixtures,
            current_squad=bgw_affected_squad,
            all_players=sample_players_unit,
            gameweek=11,
        )

        score_bgw = unit_chip_service.score_free_hit_for_gameweek(
            fixtures=bgw_fixtures,
            current_squad=bgw_affected_squad,
            all_players=sample_players_unit,
            gameweek=12,
        )

        assert score_bgw > score_normal_gw


# =============================================================================
# Test Deadline Urgency Factor
# =============================================================================


class TestDeadlineUrgency:
    """Tests for deadline urgency in chip recommendations."""

    def test_urgency_increases_near_deadline(self, unit_chip_service):
        """Chip recommendation should factor in deadline urgency."""
        # GW17 = 2 GWs remaining before GW19 deadline
        urgency_gw17 = unit_chip_service.get_deadline_urgency_factor(
            current_gw=17, chip_set="first"
        )

        # GW10 = 9 GWs remaining
        urgency_gw10 = unit_chip_service.get_deadline_urgency_factor(
            current_gw=10, chip_set="first"
        )

        assert urgency_gw17 > urgency_gw10

    def test_max_urgency_at_deadline(self, unit_chip_service):
        """Maximum urgency at final gameweek before deadline."""
        urgency = unit_chip_service.get_deadline_urgency_factor(
            current_gw=19, chip_set="first"
        )

        assert urgency >= 1.5  # Should be significant boost


# =============================================================================
# Test Home Fixture Bonus
# =============================================================================


class TestHomeFixtureBonus:
    """Tests for home fixture advantage in TC scoring."""

    @pytest.fixture
    def home_away_fixtures(self):
        """Fixtures with same opponent but different venues (equal difficulty)."""
        return pd.DataFrame([
            # GW11: Team 1 plays away vs Team 20 (same difficulty as GW12)
            {"gameweek": 11, "team_h": 20, "team_a": 1, "team_h_difficulty": 3, "team_a_difficulty": 3},
            # GW12: Team 1 plays at home vs Team 20 (same difficulty as GW11)
            {"gameweek": 12, "team_h": 1, "team_a": 20, "team_h_difficulty": 3, "team_a_difficulty": 3},
        ])

    def test_tc_prefers_home_fixture(self, unit_chip_service, sample_squad_unit, home_away_fixtures):
        """TC should score higher for home fixtures vs same opponent."""
        score_away = unit_chip_service.score_triple_captain_for_gameweek(
            fixtures=home_away_fixtures,
            current_squad=sample_squad_unit,
            gameweek=11,
        )

        score_home = unit_chip_service.score_triple_captain_for_gameweek(
            fixtures=home_away_fixtures,
            current_squad=sample_squad_unit,
            gameweek=12,
        )

        assert score_home > score_away
