"""Repository interfaces for data access abstraction."""

from .fixture_repository import FixtureRepository
from .player_repository import PlayerRepository
from .team_repository import TeamRepository

__all__ = [
    "PlayerRepository",
    "TeamRepository",
    "FixtureRepository",
]
