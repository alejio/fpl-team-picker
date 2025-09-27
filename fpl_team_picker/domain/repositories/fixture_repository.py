"""Repository interface for fixture data access."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..common.result import Result
from ..models.fixture import FixtureDomain


class FixtureRepository(ABC):
    """
    Abstract repository for fixture data access.

    Provides a consistent interface for accessing fixture data regardless
    of the underlying data source.
    """

    @abstractmethod
    def get_all_fixtures(self) -> Result[List[FixtureDomain]]:
        """
        Get all fixtures for the current season.

        Returns:
            Result containing list of fixtures or error information
        """
        pass

    @abstractmethod
    def get_fixtures_for_gameweek(self, gameweek: int) -> Result[List[FixtureDomain]]:
        """
        Get all fixtures for a specific gameweek.

        Args:
            gameweek: The gameweek number (1-38)

        Returns:
            Result containing list of fixtures or error information
        """
        pass

    @abstractmethod
    def get_fixtures_for_team(self, team_id: int) -> Result[List[FixtureDomain]]:
        """
        Get all fixtures for a specific team.

        Args:
            team_id: The team's FPL ID

        Returns:
            Result containing list of fixtures or error information
        """
        pass

    @abstractmethod
    def get_upcoming_fixtures(
        self, team_id: Optional[int] = None, limit: int = 5
    ) -> Result[List[FixtureDomain]]:
        """
        Get upcoming fixtures, optionally filtered by team.

        Args:
            team_id: Optional team ID to filter by
            limit: Maximum number of fixtures to return

        Returns:
            Result containing list of upcoming fixtures or error information
        """
        pass
