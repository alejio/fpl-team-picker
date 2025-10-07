"""Player analytics domain service for high-level player operations."""

from typing import List, Optional

from ..models.player import EnrichedPlayerDomain, Position
from ..repositories.player_repository import PlayerRepository


class PlayerAnalyticsService:
    """
    High-level player analytics operations using domain models.

    Provides clean API for player queries and analysis without exposing
    repository implementation details or DataFrame operations.
    """

    def __init__(self, player_repo: PlayerRepository):
        """
        Initialize service with player repository.

        Args:
            player_repo: PlayerRepository implementation for data access
        """
        self.player_repo = player_repo

    def get_all_players_enriched(self) -> List[EnrichedPlayerDomain]:
        """
        Get all players with complete enriched data.

        Returns:
            List of EnrichedPlayerDomain objects with 70+ validated attributes

        Raises:
            RuntimeError: If data loading fails
        """
        result = self.player_repo.get_enriched_players()
        if result.is_failure:
            raise RuntimeError(f"Failed to load players: {result.error}")
        return result.value

    def get_penalty_takers(self) -> List[EnrichedPlayerDomain]:
        """
        Get first/second choice penalty takers.

        Returns players with penalties_order <= 2 or penalty_taker flag.

        Returns:
            List of players who are likely penalty takers
        """
        players = self.get_all_players_enriched()
        return [p for p in players if p.is_penalty_taker or p.penalty_taker]

    def get_high_value_players(
        self, min_value_score: float = 80.0
    ) -> List[EnrichedPlayerDomain]:
        """
        Get players with high value scores.

        Args:
            min_value_score: Minimum value score threshold (default: 80.0)

        Returns:
            List of high-value players
        """
        players = self.get_all_players_enriched()
        return [p for p in players if p.value_score >= min_value_score]

    def get_injury_risks(self, min_risk: float = 0.5) -> List[EnrichedPlayerDomain]:
        """
        Get players with high injury risk.

        Args:
            min_risk: Minimum injury risk threshold (default: 0.5)

        Returns:
            List of players with injury concerns
        """
        players = self.get_all_players_enriched()
        return [p for p in players if p.injury_risk >= min_risk]

    def get_rotation_risks(self, min_risk: float = 0.5) -> List[EnrichedPlayerDomain]:
        """
        Get players with high rotation risk.

        Args:
            min_risk: Minimum rotation risk threshold (default: 0.5)

        Returns:
            List of players with rotation concerns
        """
        players = self.get_all_players_enriched()
        return [p for p in players if p.rotation_risk >= min_risk]

    def get_players_by_position(self, position: Position) -> List[EnrichedPlayerDomain]:
        """
        Get all players for a specific position.

        Args:
            position: Position enum (GKP, DEF, MID, FWD)

        Returns:
            List of players in the specified position
        """
        players = self.get_all_players_enriched()
        return [p for p in players if p.position == position]

    def get_players_by_team(self, team_id: int) -> List[EnrichedPlayerDomain]:
        """
        Get all players for a specific team.

        Args:
            team_id: Team ID (1-20)

        Returns:
            List of players in the specified team
        """
        players = self.get_all_players_enriched()
        return [p for p in players if p.team_id == team_id]

    def get_form_improving_players(self) -> List[EnrichedPlayerDomain]:
        """
        Get players whose form is improving.

        Returns:
            List of players with improving form trend
        """
        players = self.get_all_players_enriched()
        return [p for p in players if p.is_form_improving]

    def get_form_declining_players(self) -> List[EnrichedPlayerDomain]:
        """
        Get players whose form is declining.

        Returns:
            List of players with declining form trend
        """
        players = self.get_all_players_enriched()
        return [p for p in players if p.is_form_declining]

    def get_ownership_rising_players(self) -> List[EnrichedPlayerDomain]:
        """
        Get players whose ownership is rising.

        Returns:
            List of players with rising ownership trend
        """
        players = self.get_all_players_enriched()
        return [p for p in players if p.is_ownership_rising]

    def get_overperformance_risks(
        self, min_risk: float = 0.7
    ) -> List[EnrichedPlayerDomain]:
        """
        Get players with high overperformance risk (likely to regress).

        Args:
            min_risk: Minimum overperformance risk threshold (default: 0.7)

        Returns:
            List of players at risk of regression to mean
        """
        players = self.get_all_players_enriched()
        return [p for p in players if p.overperformance_risk >= min_risk]

    def get_set_piece_takers(self) -> List[EnrichedPlayerDomain]:
        """
        Get players who take set pieces (penalties, corners, free kicks).

        Returns:
            List of players with set piece responsibilities
        """
        players = self.get_all_players_enriched()
        return [p for p in players if p.is_set_piece_taker]

    def get_reliable_data_players(
        self, min_quality: float = 0.7
    ) -> List[EnrichedPlayerDomain]:
        """
        Get players with reliable data quality.

        Args:
            min_quality: Minimum data quality score threshold (default: 0.7)

        Returns:
            List of players with reliable data
        """
        players = self.get_all_players_enriched()
        return [p for p in players if p.data_quality_score >= min_quality]

    def get_player_by_id(self, player_id: int) -> Optional[EnrichedPlayerDomain]:
        """
        Get a specific player by ID.

        Args:
            player_id: Player's FPL ID

        Returns:
            EnrichedPlayerDomain object or None if not found
        """
        players = self.get_all_players_enriched()
        for player in players:
            if player.player_id == player_id:
                return player
        return None

    def get_top_players_by_value(self, limit: int = 10) -> List[EnrichedPlayerDomain]:
        """
        Get top N players by value score.

        Args:
            limit: Number of top players to return (default: 10)

        Returns:
            List of top value players sorted by value_score descending
        """
        players = self.get_all_players_enriched()
        return sorted(players, key=lambda p: p.value_score, reverse=True)[:limit]

    def get_top_players_by_form(self, limit: int = 10) -> List[EnrichedPlayerDomain]:
        """
        Get top N players by recent form.

        Args:
            limit: Number of top players to return (default: 10)

        Returns:
            List of top form players sorted by recent_form_5gw descending
        """
        players = self.get_all_players_enriched()
        return sorted(players, key=lambda p: p.recent_form_5gw, reverse=True)[:limit]

    def get_top_players_by_points_per_million(
        self, limit: int = 10
    ) -> List[EnrichedPlayerDomain]:
        """
        Get top N players by points per million efficiency.

        Args:
            limit: Number of top players to return (default: 10)

        Returns:
            List of most efficient players sorted by points_per_million descending
        """
        players = self.get_all_players_enriched()
        return sorted(players, key=lambda p: p.points_per_million, reverse=True)[:limit]
