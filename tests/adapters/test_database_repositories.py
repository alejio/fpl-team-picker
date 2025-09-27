"""Tests for database repository implementations."""


from fpl_team_picker.adapters.database_repositories import (
    DatabaseFixtureRepository,
    DatabasePlayerRepository,
    DatabaseTeamRepository,
)
from fpl_team_picker.domain.common.result import ErrorType
from fpl_team_picker.domain.models.player import Position


class TestDatabasePlayerRepository:
    """Test DatabasePlayerRepository."""

    def test_get_current_players_success(self):
        """Test successful player data retrieval."""
        repo = DatabasePlayerRepository()
        result = repo.get_current_players()

        assert result.is_success
        players = result.value
        assert len(players) > 0

        # Test first player has required fields
        sample_player = players[0]
        assert hasattr(sample_player, "player_id")
        assert hasattr(sample_player, "web_name")
        assert hasattr(sample_player, "position")
        assert hasattr(sample_player, "price")
        assert hasattr(sample_player, "team_id")

    def test_get_player_by_id(self):
        """Test getting player by ID."""
        repo = DatabasePlayerRepository()
        players_result = repo.get_current_players()
        assert players_result.is_success

        # Get first player's ID
        first_player = players_result.value[0]
        player_id = first_player.player_id

        # Test getting specific player
        result = repo.get_player_by_id(player_id)
        assert result.is_success
        assert result.value.player_id == player_id

        # Test non-existent player
        result = repo.get_player_by_id(99999)
        assert result.is_success
        assert result.value is None

    def test_get_players_by_position(self):
        """Test getting players by position."""
        repo = DatabasePlayerRepository()

        # Test valid position
        result = repo.get_players_by_position("GKP")
        assert result.is_success
        gkps = result.value
        assert len(gkps) > 0
        assert all(p.position == Position.GKP for p in gkps)

        # Test invalid position
        result = repo.get_players_by_position("INVALID")
        assert result.is_failure
        assert result.error.error_type == ErrorType.VALIDATION_ERROR

    def test_get_players_by_team(self):
        """Test getting players by team."""
        repo = DatabasePlayerRepository()

        # Test with valid team ID
        result = repo.get_players_by_team(1)  # Arsenal
        assert result.is_success
        team_players = result.value
        assert len(team_players) > 0
        assert all(p.team_id == 1 for p in team_players)


class TestDatabaseTeamRepository:
    """Test DatabaseTeamRepository."""

    def test_get_all_teams_success(self):
        """Test successful team data retrieval."""
        repo = DatabaseTeamRepository()
        result = repo.get_all_teams()

        assert result.is_success
        teams = result.value
        assert len(teams) == 20  # Premier League teams

        # Test first team has required fields
        sample_team = teams[0]
        assert hasattr(sample_team, "team_id")
        assert hasattr(sample_team, "name")
        assert hasattr(sample_team, "short_name")

    def test_get_team_by_id(self):
        """Test getting team by ID."""
        repo = DatabaseTeamRepository()

        # Test valid team ID
        result = repo.get_team_by_id(1)
        assert result.is_success
        assert result.value.team_id == 1

        # Test invalid team ID
        result = repo.get_team_by_id(99)
        assert result.is_success
        assert result.value is None

    def test_get_team_by_name(self):
        """Test getting team by name."""
        repo = DatabaseTeamRepository()

        # Get all teams first to know valid names
        teams_result = repo.get_all_teams()
        assert teams_result.is_success
        sample_team = teams_result.value[0]

        # Test by full name
        result = repo.get_team_by_name(sample_team.name)
        assert result.is_success
        assert result.value.name == sample_team.name

        # Test by short name
        result = repo.get_team_by_name(sample_team.short_name)
        assert result.is_success
        assert result.value.short_name == sample_team.short_name

        # Test non-existent team
        result = repo.get_team_by_name("Nonexistent FC")
        assert result.is_success
        assert result.value is None


class TestDatabaseFixtureRepository:
    """Test DatabaseFixtureRepository."""

    def test_get_all_fixtures_success(self):
        """Test successful fixture data retrieval."""
        repo = DatabaseFixtureRepository()
        result = repo.get_all_fixtures()

        assert result.is_success
        fixtures = result.value
        assert len(fixtures) > 0

        # Test first fixture has required fields
        sample_fixture = fixtures[0]
        assert hasattr(sample_fixture, "fixture_id")
        assert hasattr(sample_fixture, "event")
        assert hasattr(sample_fixture, "home_team_id")
        assert hasattr(sample_fixture, "away_team_id")

    def test_get_fixtures_for_gameweek(self):
        """Test getting fixtures for gameweek."""
        repo = DatabaseFixtureRepository()

        # Test valid gameweek
        result = repo.get_fixtures_for_gameweek(1)
        assert result.is_success
        gw1_fixtures = result.value
        assert all(f.event == 1 for f in gw1_fixtures)

        # Test invalid gameweek
        result = repo.get_fixtures_for_gameweek(99)
        assert result.is_failure
        assert result.error.error_type == ErrorType.VALIDATION_ERROR

    def test_get_fixtures_for_team(self):
        """Test getting fixtures for team."""
        repo = DatabaseFixtureRepository()

        # Test valid team
        result = repo.get_fixtures_for_team(1)
        assert result.is_success
        team_fixtures = result.value
        assert all(
            f.home_team_id == 1 or f.away_team_id == 1 for f in team_fixtures
        )

        # Test invalid team ID
        result = repo.get_fixtures_for_team(25)
        assert result.is_failure
        assert result.error.error_type == ErrorType.VALIDATION_ERROR

    def test_get_upcoming_fixtures(self):
        """Test getting upcoming fixtures."""
        repo = DatabaseFixtureRepository()

        # Test getting upcoming fixtures
        result = repo.get_upcoming_fixtures(limit=5)
        assert result.is_success
        upcoming = result.value
        assert len(upcoming) <= 5

        # Test with team filter
        result = repo.get_upcoming_fixtures(team_id=1, limit=3)
        assert result.is_success
        team_upcoming = result.value
        assert len(team_upcoming) <= 3
        assert all(
            f.home_team_id == 1 or f.away_team_id == 1 for f in team_upcoming
        )
