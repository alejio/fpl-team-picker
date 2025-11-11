"""Tests for PlayerAnalyticsService."""

import pytest
from datetime import UTC, datetime
from unittest.mock import Mock

from fpl_team_picker.domain.services.player_analytics_service import (
    PlayerAnalyticsService,
)
from fpl_team_picker.domain.models.player import EnrichedPlayerDomain, Position
from fpl_team_picker.domain.common.result import Result


@pytest.fixture
def mock_players():
    """Create mock enriched players for testing."""
    return [
        EnrichedPlayerDomain(
            player_id=1,
            web_name="Salah",
            team_id=11,
            position=Position.MID,
            price=13.0,
            selected_by_percent=45.0,
            as_of_utc=datetime.now(UTC),
            total_points_season=150,
            form_season=8.5,
            points_per_game_season=6.8,
            minutes=900,
            starts=10,
            goals_scored=12,
            assists=8,
            clean_sheets=2,
            goals_conceded=5,
            yellow_cards=1,
            red_cards=0,
            saves=0,
            bonus=15,
            bps=450,
            influence=500.0,
            creativity=400.0,
            threat=600.0,
            ict_index=1500.0,
            expected_goals=10.5,
            expected_assists=7.2,
            expected_goals_per_90=1.05,
            expected_assists_per_90=0.72,
            value_form=100.0,
            value_season=95.0,
            transfers_in=50000,
            transfers_out=10000,
            transfers_in_event=5000,
            transfers_out_event=1000,
            penalties_order=1,
            news="",
            points_per_million=11.5,
            form_per_million=0.65,
            value_score=95.0,
            value_confidence=0.95,
            form_trend="improving",
            form_momentum=0.8,
            recent_form_5gw=8.2,
            season_consistency=0.85,
            expected_points_per_game=7.0,
            points_above_expected=10.0,
            overperformance_risk=0.3,
            ownership_trend="rising",
            transfer_momentum=40000.0,
            ownership_risk=0.2,
            set_piece_priority=0.9,
            penalty_taker=True,
            corner_taker=True,
            freekick_taker=True,
            injury_risk=0.1,
            rotation_risk=0.05,
            data_quality_score=0.95,
        ),
        EnrichedPlayerDomain(
            player_id=2,
            web_name="Budget",
            team_id=1,
            position=Position.DEF,
            price=4.5,
            selected_by_percent=5.0,
            as_of_utc=datetime.now(UTC),
            total_points_season=30,
            form_season=2.0,
            points_per_game_season=1.5,
            minutes=300,
            starts=3,
            goals_scored=0,
            assists=1,
            clean_sheets=2,
            goals_conceded=5,
            yellow_cards=1,
            red_cards=0,
            saves=0,
            bonus=2,
            bps=100,
            influence=50.0,
            creativity=30.0,
            threat=20.0,
            ict_index=100.0,
            expected_goals=0.2,
            expected_assists=0.5,
            expected_goals_per_90=0.06,
            expected_assists_per_90=0.15,
            value_form=20.0,
            value_season=18.0,
            transfers_in=5000,
            transfers_out=3000,
            transfers_in_event=500,
            transfers_out_event=300,
            news="",
            points_per_million=6.7,
            form_per_million=0.44,
            value_score=50.0,
            value_confidence=0.6,
            form_trend="stable",
            form_momentum=0.0,
            recent_form_5gw=2.0,
            season_consistency=0.5,
            expected_points_per_game=1.5,
            points_above_expected=0.0,
            overperformance_risk=0.5,
            ownership_trend="stable",
            transfer_momentum=2000.0,
            ownership_risk=0.3,
            set_piece_priority=0.1,
            penalty_taker=False,
            corner_taker=False,
            freekick_taker=False,
            injury_risk=0.6,
            rotation_risk=0.7,
            data_quality_score=0.8,
        ),
        EnrichedPlayerDomain(
            player_id=3,
            web_name="Fragile",
            team_id=5,
            position=Position.FWD,
            price=7.0,
            selected_by_percent=15.0,
            as_of_utc=datetime.now(UTC),
            total_points_season=45,
            form_season=3.0,
            points_per_game_season=2.5,
            minutes=450,
            starts=5,
            goals_scored=3,
            assists=1,
            clean_sheets=0,
            goals_conceded=0,
            yellow_cards=2,
            red_cards=0,
            saves=0,
            bonus=3,
            bps=150,
            influence=100.0,
            creativity=50.0,
            threat=150.0,
            ict_index=300.0,
            expected_goals=2.5,
            expected_assists=0.8,
            expected_goals_per_90=0.5,
            expected_assists_per_90=0.16,
            value_form=30.0,
            value_season=28.0,
            transfers_in=8000,
            transfers_out=12000,
            transfers_in_event=800,
            transfers_out_event=1200,
            news="",
            points_per_million=6.4,
            form_per_million=0.43,
            value_score=40.0,
            value_confidence=0.5,
            form_trend="declining",
            form_momentum=-0.5,
            recent_form_5gw=2.5,
            season_consistency=0.4,
            expected_points_per_game=2.0,
            points_above_expected=-3.0,
            overperformance_risk=0.8,
            ownership_trend="falling",
            transfer_momentum=-4000.0,
            ownership_risk=0.6,
            set_piece_priority=0.2,
            penalty_taker=False,
            corner_taker=False,
            freekick_taker=False,
            injury_risk=0.8,
            rotation_risk=0.4,
            data_quality_score=0.65,
        ),
    ]


@pytest.fixture
def mock_repository(mock_players):
    """Create mock player repository."""
    repo = Mock()
    repo.get_enriched_players.return_value = Result.success(mock_players)
    return repo


@pytest.fixture
def service(mock_repository):
    """Create PlayerAnalyticsService with mock repository."""
    return PlayerAnalyticsService(mock_repository)


class TestPlayerAnalyticsService:
    """Test PlayerAnalyticsService methods."""

    def test_get_all_players_enriched(self, service, mock_players):
        """Test getting all enriched players."""
        players = service.get_all_players_enriched()

        assert len(players) == 3
        assert all(isinstance(p, EnrichedPlayerDomain) for p in players)
        assert players[0].web_name == "Salah"

    def test_get_penalty_takers(self, service):
        """Test filtering penalty takers."""
        penalty_takers = service.get_penalty_takers()

        assert len(penalty_takers) == 1
        assert penalty_takers[0].web_name == "Salah"
        assert penalty_takers[0].penalty_taker is True

    def test_get_high_value_players(self, service):
        """Test filtering high value players."""
        high_value = service.get_high_value_players(min_value_score=80.0)

        assert len(high_value) == 1
        assert high_value[0].web_name == "Salah"
        assert high_value[0].value_score >= 80.0

    def test_get_injury_risks(self, service):
        """Test filtering players with injury concerns."""
        injury_risks = service.get_injury_risks(min_risk=0.5)

        assert len(injury_risks) == 2  # Budget and Fragile
        assert "Salah" not in [p.web_name for p in injury_risks]

    def test_get_rotation_risks(self, service):
        """Test filtering players with rotation concerns."""
        rotation_risks = service.get_rotation_risks(min_risk=0.5)

        assert len(rotation_risks) == 1
        assert rotation_risks[0].web_name == "Budget"

    def test_get_players_by_position(self, service):
        """Test filtering players by position."""
        midfielders = service.get_players_by_position(Position.MID)

        assert len(midfielders) == 1
        assert midfielders[0].web_name == "Salah"
        assert midfielders[0].position == Position.MID

    def test_get_players_by_team(self, service):
        """Test filtering players by team."""
        liverpool_players = service.get_players_by_team(team_id=11)

        assert len(liverpool_players) == 1
        assert liverpool_players[0].web_name == "Salah"

    def test_get_form_improving_players(self, service):
        """Test filtering players with improving form."""
        improving = service.get_form_improving_players()

        assert len(improving) == 1
        assert improving[0].web_name == "Salah"
        assert improving[0].form_trend == "improving"

    def test_get_form_declining_players(self, service):
        """Test filtering players with declining form."""
        declining = service.get_form_declining_players()

        assert len(declining) == 1
        assert declining[0].web_name == "Fragile"
        assert declining[0].form_trend == "declining"

    def test_get_ownership_rising_players(self, service):
        """Test filtering players with rising ownership."""
        rising = service.get_ownership_rising_players()

        assert len(rising) == 1
        assert rising[0].web_name == "Salah"
        assert rising[0].ownership_trend == "rising"

    def test_get_overperformance_risks(self, service):
        """Test filtering players at risk of regression."""
        overperformers = service.get_overperformance_risks(min_risk=0.7)

        assert len(overperformers) == 1
        assert overperformers[0].web_name == "Fragile"

    def test_get_set_piece_takers(self, service):
        """Test filtering set piece takers."""
        set_piece_takers = service.get_set_piece_takers()

        assert len(set_piece_takers) == 1
        assert set_piece_takers[0].web_name == "Salah"

    def test_get_reliable_data_players(self, service):
        """Test filtering players with reliable data."""
        reliable = service.get_reliable_data_players(min_quality=0.7)

        assert len(reliable) == 2  # Salah and Budget
        assert "Fragile" not in [p.web_name for p in reliable]

    def test_get_player_by_id(self, service):
        """Test getting player by ID."""
        player = service.get_player_by_id(player_id=1)

        assert player is not None
        assert player.web_name == "Salah"

        # Test player not found
        not_found = service.get_player_by_id(player_id=999)
        assert not_found is None

    def test_get_top_players_by_value(self, service):
        """Test getting top players by value score."""
        top_value = service.get_top_players_by_value(limit=2)

        assert len(top_value) == 2
        assert top_value[0].web_name == "Salah"  # Highest value_score
        assert top_value[0].value_score >= top_value[1].value_score

    def test_get_top_players_by_form(self, service):
        """Test getting top players by recent form."""
        top_form = service.get_top_players_by_form(limit=2)

        assert len(top_form) == 2
        assert top_form[0].web_name == "Salah"  # Highest recent_form_5gw

    def test_get_top_players_by_points_per_million(self, service):
        """Test getting top players by efficiency."""
        top_efficient = service.get_top_players_by_points_per_million(limit=2)

        assert len(top_efficient) == 2
        assert top_efficient[0].web_name == "Salah"  # Highest points_per_million

    def test_service_with_failed_repository(self, mock_repository):
        """Test service handles repository failures."""
        from fpl_team_picker.domain.common.result import DomainError, ErrorType

        # Mock repository failure
        mock_repository.get_enriched_players.return_value = Result.failure(
            DomainError(
                error_type=ErrorType.DATA_ACCESS_ERROR,
                message="Database connection failed",
            )
        )

        service = PlayerAnalyticsService(mock_repository)

        with pytest.raises(RuntimeError, match="Failed to load players"):
            service.get_all_players_enriched()
