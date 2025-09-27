"""Repository interface for player data access."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..common.result import Result
from ..models.player import LiveDataDomain, PlayerDomain


class PlayerRepository(ABC):
    """
    Abstract repository for player data access.

    Provides a consistent interface for accessing player data regardless
    of the underlying data source (database, API, cache, etc.).
    """

    @abstractmethod
    def get_current_players(self) -> Result[List[PlayerDomain]]:
        """
        Get all current players for the season.

        Returns:
            Result containing list of players or error information
        """
        pass

    @abstractmethod
    def get_player_by_id(self, player_id: int) -> Result[Optional[PlayerDomain]]:
        """
        Get a specific player by ID.

        Args:
            player_id: The player's FPL ID

        Returns:
            Result containing player or None if not found, or error information
        """
        pass

    @abstractmethod
    def get_players_by_team(self, team_id: int) -> Result[List[PlayerDomain]]:
        """
        Get all players for a specific team.

        Args:
            team_id: The team's FPL ID

        Returns:
            Result containing list of players for the team or error information
        """
        pass

    @abstractmethod
    def get_players_by_position(self, position: str) -> Result[List[PlayerDomain]]:
        """
        Get all players for a specific position.

        Args:
            position: The position (GKP, DEF, MID, FWD)

        Returns:
            Result containing list of players for the position or error information
        """
        pass

    @abstractmethod
    def get_live_data_for_gameweek(self, gameweek: int) -> Result[List[LiveDataDomain]]:
        """
        Get live performance data for all players in a specific gameweek.

        Args:
            gameweek: The gameweek number (1-38)

        Returns:
            Result containing list of live data or error information
        """
        pass

    @abstractmethod
    def get_player_live_data(
        self, player_id: int, gameweek: int
    ) -> Result[Optional[LiveDataDomain]]:
        """
        Get live data for a specific player in a specific gameweek.

        Args:
            player_id: The player's FPL ID
            gameweek: The gameweek number (1-38)

        Returns:
            Result containing live data or None if not found, or error information
        """
        pass
