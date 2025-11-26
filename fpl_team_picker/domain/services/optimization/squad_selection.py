"""Squad selection utilities for FPL optimization.

This module handles:
- Starting XI selection with formation optimization
- Bench player selection
- Budget pool calculations
"""

from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
from loguru import logger

from .optimization_base import OptimizationBaseMixin


class SquadSelectionMixin(OptimizationBaseMixin):
    """Mixin providing squad selection functionality.

    Handles starting XI selection, bench ordering, and budget calculations.
    Inherits shared utilities from OptimizationBaseMixin.
    """

    def calculate_budget_pool(
        self,
        current_squad: pd.DataFrame,
        bank_balance: float,
        players_to_keep: Optional[Set[int]] = None,
    ) -> Dict:
        """Calculate total budget pool available for transfers.

        Args:
            current_squad: Current squad DataFrame
            bank_balance: Available bank balance in millions
            players_to_keep: Set of player IDs that must be kept

        Returns:
            Dictionary with budget pool information
        """
        if current_squad.empty:
            return {
                "bank_balance": bank_balance,
                "sellable_value": 0.0,
                "total_budget": bank_balance,
                "sellable_players": 0,
                "players_to_keep": 0,
            }

        players_to_keep = players_to_keep or set()

        # Calculate sellable value (players not in must-keep list)
        sellable_players = current_squad[
            ~current_squad["player_id"].isin(players_to_keep)
        ]
        sellable_value = (
            sellable_players["price"].sum() if not sellable_players.empty else 0.0
        )

        total_budget = bank_balance + sellable_value

        return {
            "bank_balance": bank_balance,
            "sellable_value": sellable_value,
            "total_budget": total_budget,
            "sellable_players": len(sellable_players),
            "players_to_keep": len(players_to_keep),
        }

    def find_optimal_starting_11(
        self, squad_df: pd.DataFrame, xp_column: str = "xP"
    ) -> Tuple[List[Dict], str, float]:
        """Find best starting 11 from squad using specified XP column.

        Args:
            squad_df: Squad DataFrame
            xp_column: Column to use for XP sorting ('xP' for current GW, 'xP_3gw' for 3-GW, 'xP_5gw' for strategic)

        Returns:
            Tuple of (best_11_list, formation_name, total_xp)
        """
        if len(squad_df) < 11:
            return [], "", 0

        # Filter out suspended, injured, or unavailable players
        available_squad = squad_df.copy()
        if "status" in available_squad.columns:
            unavailable_mask = available_squad["status"].isin(["i", "s", "u"])
            if unavailable_mask.any():
                unavailable_players = available_squad[unavailable_mask]
                logger.debug(
                    f"🚫 Excluding {len(unavailable_players)} unavailable players from starting 11:"
                )
                for _, player in unavailable_players.iterrows():
                    status_desc = {
                        "i": "injured",
                        "s": "suspended",
                        "u": "unavailable",
                    }[player["status"]]
                    logger.debug(
                        f"   - {player.get('web_name', 'Unknown')} ({status_desc})"
                    )

                available_squad = available_squad[~unavailable_mask]

        if len(available_squad) < 11:
            logger.warning(
                f"⚠️ Warning: Only {len(available_squad)} available players in squad (need 11)"
            )
            return [], "", 0

        # Default to current gameweek XP, fallback to 3GW or 5GW if current not available
        sort_col = (
            xp_column
            if xp_column in available_squad.columns
            else (
                "xP_3gw"
                if "xP_3gw" in available_squad.columns
                else ("xP_5gw" if "xP_5gw" in available_squad.columns else "xP")
            )
        )

        # Group by position and sort by XP
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for _, player in available_squad.iterrows():
            player_dict = player.to_dict()
            by_position[player["position"]].append(player_dict)

        for pos in by_position:
            by_position[pos].sort(
                key=lambda p: p.get(sort_col, p.get("xP", 0)), reverse=True
            )

        # Use shared formation enumeration logic
        return self._enumerate_formations_for_players(by_position, sort_col)

    def find_bench_players(
        self, squad_df: pd.DataFrame, starting_11: List[Dict], xp_column: str = "xP"
    ) -> List[Dict]:
        """Get bench players (remaining 4 players) from squad ordered by XP.

        Args:
            squad_df: Squad DataFrame
            starting_11: List of starting 11 player dictionaries
            xp_column: Column to use for XP sorting ('xP' for current GW, 'xP_3gw' for 3-GW, 'xP_5gw' for strategic)

        Returns:
            List of bench player dictionaries ordered by XP (highest first)
        """
        if len(squad_df) < 15:
            return []

        # Get player IDs from starting 11
        starting_11_ids = {player["player_id"] for player in starting_11}

        # Get remaining players (bench)
        bench_players = []
        sort_col = (
            xp_column
            if xp_column in squad_df.columns
            else (
                "xP_3gw"
                if "xP_3gw" in squad_df.columns
                else ("xP_5gw" if "xP_5gw" in squad_df.columns else "xP")
            )
        )

        for _, player in squad_df.iterrows():
            if player["player_id"] not in starting_11_ids:
                bench_players.append(player.to_dict())

        # Sort bench by XP (highest first)
        bench_players.sort(key=lambda p: p.get(sort_col, 0), reverse=True)

        return bench_players[:4]  # Maximum 4 bench players
