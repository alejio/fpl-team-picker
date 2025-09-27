"""Repository interface for team data access."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..common.result import Result
from ..models.team import TeamDomain


class TeamRepository(ABC):
    """
    Abstract repository for team data access.

    Provides a consistent interface for accessing team data regardless
    of the underlying data source.
    """

    @abstractmethod
    def get_all_teams(self) -> Result[List[TeamDomain]]:
        """
        Get all Premier League teams for the current season.

        Returns:
            Result containing list of teams or error information
        """
        pass

    @abstractmethod
    def get_team_by_id(self, team_id: int) -> Result[Optional[TeamDomain]]:
        """
        Get a specific team by ID.

        Args:
            team_id: The team's FPL ID (1-20)

        Returns:
            Result containing team or None if not found, or error information
        """
        pass

    @abstractmethod
    def get_team_by_name(self, name: str) -> Result[Optional[TeamDomain]]:
        """
        Get a team by name (full or short name).

        Args:
            name: Team name to search for

        Returns:
            Result containing team or None if not found, or error information
        """
        pass
