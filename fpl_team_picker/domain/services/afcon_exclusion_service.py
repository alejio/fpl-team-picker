"""AFCON Player Exclusion Service.

Handles exclusion of players participating in Africa Cup of Nations tournaments
from optimization and team selection.
"""

import json
from pathlib import Path
from typing import Set, List, Dict, Optional
from loguru import logger


class AFCONExclusionService:
    """Service for managing AFCON player exclusions."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize AFCON exclusion service.

        Args:
            data_path: Path to AFCON data file. If None, uses default location.
        """
        if data_path is None:
            # Default to data/afcon_2025_players.json in project root
            project_root = Path(__file__).parent.parent.parent.parent
            data_path = project_root / "data" / "afcon_2025_players.json"

        self.data_path = data_path
        self._afcon_data: Optional[Dict] = None

    def load_afcon_data(self) -> Dict:
        """Load AFCON player data from JSON file.

        Returns:
            Dictionary with AFCON tournament and player information
        """
        if self._afcon_data is not None:
            return self._afcon_data

        try:
            with open(self.data_path, "r") as f:
                self._afcon_data = json.load(f)
            logger.info(
                f"✅ Loaded AFCON data: {len(self._afcon_data.get('players', []))} players"
            )
            return self._afcon_data
        except FileNotFoundError:
            logger.warning(f"⚠️ AFCON data file not found: {self.data_path}")
            return {"players": [], "affected_gameweeks": []}
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing AFCON data file: {e}")
            return {"players": [], "affected_gameweeks": []}

    def get_afcon_player_ids(
        self,
        impact_filter: Optional[List[str]] = None,
        country_filter: Optional[List[str]] = None,
    ) -> Set[int]:
        """Get set of AFCON player IDs with optional filtering.

        Args:
            impact_filter: Filter by impact level (e.g., ["high", "medium"])
            country_filter: Filter by country (e.g., ["Egypt", "Nigeria"])

        Returns:
            Set of player IDs going to AFCON
        """
        afcon_data = self.load_afcon_data()
        players = afcon_data.get("players", [])

        player_ids = set()
        for player in players:
            # Apply impact filter
            if impact_filter and player.get("impact") not in impact_filter:
                continue

            # Apply country filter
            if country_filter and player.get("country") not in country_filter:
                continue

            player_id = player.get("player_id")
            if player_id:
                player_ids.add(player_id)

        if player_ids:
            logger.info(
                f"🚫 AFCON exclusions: {len(player_ids)} players"
                + (f" (impact: {impact_filter})" if impact_filter else "")
                + (f" (countries: {country_filter})" if country_filter else "")
            )

        return player_ids

    def get_afcon_players_info(self) -> List[Dict]:
        """Get full information about AFCON players.

        Returns:
            List of player dictionaries with all AFCON data
        """
        afcon_data = self.load_afcon_data()
        return afcon_data.get("players", [])

    def is_gameweek_affected(self, gameweek: int) -> bool:
        """Check if a gameweek is affected by AFCON.

        Args:
            gameweek: Gameweek number to check

        Returns:
            True if gameweek is during AFCON period
        """
        afcon_data = self.load_afcon_data()
        affected_gws = afcon_data.get("affected_gameweeks", [])
        return gameweek in affected_gws

    def get_tournament_info(self) -> Dict:
        """Get AFCON tournament information.

        Returns:
            Dictionary with tournament dates and metadata
        """
        afcon_data = self.load_afcon_data()
        return {
            "name": afcon_data.get("tournament_name", ""),
            "dates": afcon_data.get("tournament_dates", {}),
            "affected_gameweeks": afcon_data.get("affected_gameweeks", []),
        }

    def get_exclusion_summary(self, player_ids: Set[int]) -> str:
        """Get a human-readable summary of excluded players.

        Args:
            player_ids: Set of excluded player IDs

        Returns:
            Formatted string with player names and countries
        """
        if not player_ids:
            return "No AFCON exclusions"

        afcon_data = self.load_afcon_data()
        players = afcon_data.get("players", [])

        excluded_players = [p for p in players if p.get("player_id") in player_ids]

        if not excluded_players:
            return f"{len(player_ids)} players excluded (AFCON)"

        # Group by impact
        by_impact = {"high": [], "medium": [], "low": []}
        for p in excluded_players:
            impact = p.get("impact", "low")
            by_impact[impact].append(p.get("web_name", p.get("full_name", "Unknown")))

        summary_parts = []
        for impact in ["high", "medium", "low"]:
            if by_impact[impact]:
                names = ", ".join(sorted(by_impact[impact]))
                summary_parts.append(f"{impact.upper()}: {names}")

        return "🚫 AFCON exclusions: " + " | ".join(summary_parts)
