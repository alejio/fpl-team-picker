"""Core transfer optimization entry point for FPL.

This module provides:
- Main optimize_transfers() entry point using Simulated Annealing
- Premium acquisition planning

Copy the following methods from optimization_service.py:
- optimize_transfers (lines 1216-1303)
- plan_premium_acquisition (lines 1087-1214)
"""

from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
from loguru import logger

from fpl_team_picker.config import config
from .transfer_sa import TransferSAMixin


class TransferOptimizationMixin(TransferSAMixin):
    """Mixin providing transfer optimization using Simulated Annealing."""

    def optimize_transfers(
        self,
        current_squad: pd.DataFrame,
        team_data: Dict,
        players_with_xp: pd.DataFrame,
        must_include_ids: Set[int] = None,
        must_exclude_ids: Set[int] = None,
        free_transfers_override: Optional[int] = None,
        is_free_hit: bool = False,
        is_bench_boost: bool = False,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Unified transfer optimization supporting normal gameweeks and chips.

        This is the core transfer optimization algorithm that analyzes multiple
        transfer scenarios and recommends the optimal strategy.

        **Unified Chip Support:**
        - Normal gameweek: free_transfers=1 (analyze 0-3 transfers)
        - Saved transfer: free_transfers=2 (analyze 0-4 transfers)
        - Wildcard chip: free_transfers=15 (rebuild entire squad, budget resets to £100m)
        - Free Hit chip: free_transfers=15 + is_free_hit=True (1GW optimization, squad reverts after)
        - Bench Boost chip: is_bench_boost=True (optimize all 15 players, not just starting 11)

        **Optimization Method:**
        - Simulated Annealing: Exploratory, good for non-linear objectives (~10-45s)

        Args:
            current_squad: Current squad DataFrame
            team_data: Team data dictionary with bank balance etc
            players_with_xp: All players with XP calculations
            must_include_ids: Set of player IDs that must be included
            must_exclude_ids: Set of player IDs that must be excluded
            free_transfers_override: Override free transfers (for wildcard/chips)
                - None: Use team_data['free_transfers'] (default 1)
                - 15: Wildcard chip (rebuild entire squad)
                - 2: Saved transfer from previous week
            is_free_hit: If True, optimize for 1GW only (squad reverts after deadline)
            is_bench_boost: If True, optimize for all 15 players (bench players also score)

        Returns:
            Tuple of (optimal_squad_df, best_scenario_dict, optimization_metadata_dict)
        """
        # Apply free transfers override if wildcard/chip is active
        if free_transfers_override is not None:
            team_data = team_data.copy()
            team_data["free_transfers"] = free_transfers_override

            # Wildcard/Free Hit chip: Budget resets to £100m (ignore current squad value)
            if free_transfers_override >= 15:
                team_data["bank"] = (
                    config.optimization.free_hit_budget if is_free_hit else 100.0
                )
                # Note: Selling prices will be ignored since we can replace all 15 players
                chip_name = "Free Hit" if is_free_hit else "Wildcard"
                logger.info(
                    f"🃏 {chip_name} Active: {free_transfers_override} free transfers, "
                    f"£{team_data['bank']:.1f}m budget"
                    + (" (1GW only, squad reverts)" if is_free_hit else "")
                )

        # Log Bench Boost mode
        if is_bench_boost:
            logger.info("🪑 Bench Boost Active: Optimizing for all 15 players")

        # Use Simulated Annealing for transfer optimization
        return self._optimize_transfers_sa(
            current_squad,
            team_data,
            players_with_xp,
            must_include_ids,
            must_exclude_ids,
            is_free_hit=is_free_hit,
            is_bench_boost=is_bench_boost,
        )

    def plan_premium_acquisition(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        budget_pool_info: Dict,
        top_n: int = 3,
    ) -> List[Dict]:
        """Plan premium player acquisitions with funding scenarios.

        Args:
            current_squad: Current squad DataFrame
            all_players: All available players DataFrame
            budget_pool_info: Budget pool information from calculate_budget_pool
            top_n: Number of top scenarios to return

        Returns:
            List of premium acquisition scenarios
        """
        if current_squad.empty or all_players.empty:
            return []

        scenarios = []

        # Find premium targets (players not in squad, price > bank balance)
        current_player_ids = set(current_squad["player_id"].tolist())
        available_players = all_players[
            ~all_players["player_id"].isin(current_player_ids)
        ]

        # Premium targets that cost more than available bank
        premium_targets = available_players[
            (available_players["price"] > budget_pool_info["bank_balance"])
            & (available_players["price"] <= budget_pool_info["total_budget"])
        ].nlargest(20, "xP")  # Top 20 by expected points

        if premium_targets.empty:
            return scenarios

        for _, target in premium_targets.iterrows():
            funding_gap = target["price"] - budget_pool_info["bank_balance"]

            # Find combinations of current players to sell that cover funding gap
            sellable_players = current_squad.copy()

            # Try different selling strategies
            # Strategy 1: Sell lowest XP players first (avoid selling good players)
            xp_column = self.get_optimization_xp_column()
            sort_column = xp_column if xp_column in sellable_players.columns else "xP"
            potential_sells = sellable_players.sort_values(sort_column, ascending=True)

            # Find minimum number of players to sell to cover gap
            cumulative_value = 0
            sell_players = []

            for _, player in potential_sells.iterrows():
                sell_players.append(player)
                cumulative_value += player["price"]

                if cumulative_value >= funding_gap:
                    # Check if we have enough budget for replacements
                    remaining_slots = len(sell_players)
                    min_replacement_cost = (
                        remaining_slots * 4.0
                    )  # Assume £4m minimum per replacement

                    if (
                        budget_pool_info["total_budget"] - target["price"]
                        >= min_replacement_cost
                    ):
                        # Find cheap replacements for sold players
                        replacements = []
                        remaining_budget = (
                            budget_pool_info["total_budget"] - target["price"]
                        )

                        for sell_player in sell_players:
                            # Find cheapest replacement in same position
                            position_replacements = available_players[
                                (
                                    available_players["position"]
                                    == sell_player["position"]
                                )
                                & (available_players["price"] <= remaining_budget)
                            ].nsmallest(5, "price")  # Top 5 cheapest

                            if not position_replacements.empty:
                                replacement = position_replacements.iloc[0]
                                replacements.append(replacement)
                                remaining_budget -= replacement["price"]
                            else:
                                break  # Can't find replacement, skip this scenario

                        # Calculate XP impact
                        sell_xp = sum(p.get("xP", 0) for p in sell_players)
                        replacement_xp = sum(r.get("xP", 0) for r in replacements)

                        if len(replacements) == len(sell_players):
                            # Calculate net XP gain
                            net_xp_gain = (target["xP"] + replacement_xp) - sell_xp

                            if net_xp_gain > 0.2:  # Higher threshold for multi-transfer
                                sell_names = ", ".join(
                                    [p["web_name"] for p in sell_players]
                                )
                                replace_names = ", ".join(
                                    [r["web_name"] for r in replacements]
                                )

                                scenarios.append(
                                    {
                                        "type": "premium_funded",
                                        "target_player": target,
                                        "sell_players": sell_players,
                                        "replacement_players": replacements,
                                        "funding_gap": target["price"]
                                        - budget_pool_info["bank_balance"],
                                        "xp_gain": net_xp_gain,
                                        "transfers": 1
                                        + len(sell_players)
                                        + len(replacements),
                                        "description": f"Premium Funded: {target['web_name']} (sell {sell_names}, buy {replace_names})",
                                    }
                                )
                    break

        # Sort by XP gain and return top scenarios
        scenarios.sort(key=lambda x: x["xp_gain"], reverse=True)
        return scenarios[:top_n]
