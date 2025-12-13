"""Simulated Annealing transfer optimization for FPL.

This module handles transfer optimization using Simulated Annealing:
- Exploratory, good for non-linear objectives
- Supports exhaustive search for small transfer counts
- Consensus mode for reliability

Copy the following methods from optimization_service.py:
- _optimize_transfers_sa (lines 1305-1762)
- _run_transfer_sa (lines 2550-2682)
- _swap_transfer_players (lines 2684-2850)
- _consensus_optimization (lines 3696-3813)
- _exhaustive_transfer_search (lines 3815-4075)
"""

from typing import Dict, List, Tuple, Optional, Set, Any
from itertools import combinations
import pandas as pd
import random
import math
from loguru import logger

from fpl_team_picker.config import config
from .optimization_base import OptimizationBaseMixin


class TransferSAMixin(OptimizationBaseMixin):
    """Mixin providing Simulated Annealing transfer optimization.

    Includes SA-based optimization, exhaustive search, and consensus modes.
    """

    def _optimize_transfers_sa(
        self,
        current_squad: pd.DataFrame,
        team_data: Dict,
        players_with_xp: pd.DataFrame,
        must_include_ids: Set[int] = None,
        must_exclude_ids: Set[int] = None,
        is_free_hit: bool = False,
        is_bench_boost: bool = False,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Transfer optimization using Simulated Annealing (exploration-based).

        Reuses SA logic from optimize_initial_squad but adapted for transfer context:
        - Start state: Current squad (not random)
        - Budget: Bank balance + selling prices (not £100m)
        - Objective: Maximize (squad_xp - transfer_penalties)
        - Neighbor function: Swap 1-3 players respecting budget

        Clean architecture: Returns data structures only, no UI generation.

        Args:
            current_squad: Current squad DataFrame
            team_data: Team data dictionary with bank balance etc
            players_with_xp: All players with XP calculations
            must_include_ids: Set of player IDs that must be included
            must_exclude_ids: Set of player IDs that must be excluded

        Returns:
            Tuple of (optimal_squad_df, best_scenario_dict, optimization_metadata_dict)
        """
        if len(current_squad) == 0 or players_with_xp.empty or not team_data:
            return pd.DataFrame(), {}, {"error": "Load team and calculate XP first"}

        # Set random seed for reproducibility if configured
        if config.optimization.sa_random_seed is not None:
            random.seed(config.optimization.sa_random_seed)
            logger.debug(
                f"🎲 Random seed set to {config.optimization.sa_random_seed} for reproducibility"
            )

        logger.info(
            f"🧠 Strategic Optimization: Simulated Annealing ({config.optimization.sa_iterations} iterations)..."
        )

        # Process constraints
        must_include_ids = must_include_ids or set()
        must_exclude_ids = must_exclude_ids or set()

        if must_include_ids:
            logger.debug(f"🎯 Must include {len(must_include_ids)} players")
        if must_exclude_ids:
            logger.debug(f"🚫 Must exclude {len(must_exclude_ids)} players")

        # Get optimization column based on configuration
        # Free Hit and Bench Boost always use 1GW (single-gameweek chips)
        if is_free_hit:
            xp_column = "xP"
            horizon_label = "1-GW"
            logger.info(
                "🎯 Free Hit mode: Optimizing for 1GW only (squad reverts after)"
            )
        elif is_bench_boost:
            xp_column = "xP"
            horizon_label = "1-GW (Bench Boost)"
            logger.info("🪑 Bench Boost mode: Optimizing all 15 players for 1GW")
        else:
            xp_column = self.get_optimization_xp_column()
            if xp_column == "xP":
                horizon_label = "1-GW"
            elif xp_column == "xP_3gw":
                horizon_label = "3-GW"
            else:
                horizon_label = "5-GW"

        # Update current squad with 1-GW, 3-GW, and 5-GW XP data
        merge_columns = ["player_id", "xP"]
        if "xP_3gw" in players_with_xp.columns:
            merge_columns.append("xP_3gw")
        if "xP_5gw" in players_with_xp.columns:
            merge_columns.append("xP_5gw")
        if "xP_uncertainty" in players_with_xp.columns:
            merge_columns.append("xP_uncertainty")
        if "fixture_outlook" in players_with_xp.columns:
            merge_columns.append("fixture_outlook")
        if "expected_minutes" in players_with_xp.columns:
            merge_columns.append("expected_minutes")

        # Include team information in merge to maintain data contract
        if "team" in players_with_xp.columns:
            merge_columns.append("team")
        elif "team_id" in players_with_xp.columns:
            merge_columns.append("team_id")

        current_squad_with_xp = current_squad.merge(
            players_with_xp[merge_columns],
            on="player_id",
            how="left",
            suffixes=("", "_from_xp"),
        )
        # Fill any missing XP with 0
        current_squad_with_xp["xP"] = current_squad_with_xp["xP"].fillna(0)
        if "xP_3gw" in current_squad_with_xp.columns:
            current_squad_with_xp["xP_3gw"] = current_squad_with_xp["xP_3gw"].fillna(0)
        if "xP_5gw" in current_squad_with_xp.columns:
            current_squad_with_xp["xP_5gw"] = current_squad_with_xp["xP_5gw"].fillna(0)
        if "fixture_outlook" in current_squad_with_xp.columns:
            current_squad_with_xp["fixture_outlook"] = current_squad_with_xp[
                "fixture_outlook"
            ].fillna("🟡 Average")

        # Validate team data contract
        team_col = "team" if "team" in current_squad_with_xp.columns else "team_id"
        if team_col in current_squad_with_xp.columns:
            nan_teams = current_squad_with_xp[team_col].isna().sum()
            if nan_teams > 0:
                raise ValueError(
                    f"Data contract violation: {nan_teams} players have NaN team values"
                )

        # Current squad and available budget
        available_budget = team_data["bank"]
        free_transfers = team_data.get("free_transfers", 1)

        # Calculate total budget pool
        is_wildcard = free_transfers >= 15
        if is_wildcard:
            # Wildcard: budget resets to £100m, ignore current squad value
            budget_pool_info = {
                "total_budget": 100.0,
                "sellable_value": 0.0,  # Not applicable for wildcard
                "non_sellable_value": 0.0,
                "must_include_value": 0.0,
            }
            logger.info("🃏 Wildcard Budget: £100.0m (budget reset)")
        else:
            # Normal transfers: bank + sellable squad value
            budget_pool_info = self.calculate_budget_pool(
                current_squad_with_xp, available_budget, must_include_ids
            )
            logger.info(
                f"💰 Budget: Bank £{available_budget:.1f}m | Sellable £{budget_pool_info['sellable_value']:.1f}m | Total £{budget_pool_info['total_budget']:.1f}m"
            )

        # Get all available players and filter
        all_players = players_with_xp[players_with_xp["xP"].notna()].copy()

        # Filter out unavailable players
        if "status" in all_players.columns:
            available_players_mask = ~all_players["status"].isin(["i", "s", "u"])
            all_players = all_players[available_players_mask]

            excluded_players = players_with_xp[
                players_with_xp["status"].isin(["i", "s", "u"])
                & players_with_xp["xP"].notna()
            ]
            if not excluded_players.empty:
                logger.debug(f"🚫 Filtered {len(excluded_players)} unavailable players")

        if must_exclude_ids:
            all_players = all_players[~all_players["player_id"].isin(must_exclude_ids)]

        # Get best starting 11 from current squad (needed for formation display)
        current_best_11, current_formation, starting_11_xp = (
            self.find_optimal_starting_11(current_squad_with_xp, xp_column)
        )

        # Calculate current_xp based on mode:
        # Bench Boost: ALL 15 players count, Normal: only starting 11 count
        if is_bench_boost:
            current_xp = current_squad_with_xp[xp_column].sum()
            logger.info(
                f"📊 Current squad (Bench Boost): {current_xp:.2f} {horizon_label}-xP (all 15 players) | Formation: {current_formation}"
            )
        else:
            current_xp = starting_11_xp
            logger.info(
                f"📊 Current squad: {current_xp:.2f} {horizon_label}-xP | Formation: {current_formation}"
            )

        # For wildcard/free hit (15 free transfers), use initial squad generation instead of transfer-based SA
        if is_wildcard:
            chip_name = "Free Hit" if is_free_hit else "Wildcard"
            budget = config.optimization.free_hit_budget if is_free_hit else 100.0
            iterations = (
                config.optimization.sa_free_hit_iterations
                if is_free_hit
                else config.optimization.sa_wildcard_iterations
            )
            restarts = (
                config.optimization.sa_free_hit_restarts
                if is_free_hit
                else config.optimization.sa_wildcard_restarts
            )
            use_consensus = (
                config.optimization.sa_free_hit_use_consensus
                if is_free_hit
                else config.optimization.sa_wildcard_use_consensus
            )

            logger.info(
                f"🃏 {chip_name} mode: Building optimal squad from scratch (ignoring current squad)"
            )

            # Check if consensus mode is enabled
            if use_consensus:
                logger.info(
                    f"🎯 {chip_name} Consensus: Running {restarts} restarts "
                    f"× {iterations} iterations each to find truly optimal squad..."
                )
                wildcard_result = self._consensus_wildcard_optimization(
                    players_with_xp=all_players,
                    budget=budget,
                    formation=(2, 5, 5, 3),
                    must_include_ids=must_include_ids,
                    must_exclude_ids=must_exclude_ids,
                    xp_column=xp_column,
                )
            else:
                # Single run mode (faster but less reliable)
                wildcard_result = self.optimize_initial_squad(
                    players_with_xp=all_players,
                    budget=budget,
                    formation=(2, 5, 5, 3),
                    iterations=iterations,
                    must_include_ids=must_include_ids,
                    must_exclude_ids=must_exclude_ids,
                    xp_column=xp_column,
                )

            # Convert initial squad result to transfer optimization format
            optimal_squad = wildcard_result["optimal_squad"]
            best_squad_df = pd.DataFrame(optimal_squad)

            # Calculate how many players changed
            original_ids = set(current_squad_with_xp["player_id"].tolist())
            new_ids = set(best_squad_df["player_id"].tolist())
            num_transfers = len(original_ids - new_ids)

            sa_result = {
                "optimal_squad": optimal_squad,
                "best_xp": wildcard_result["total_xp"],
                "iterations_improved": wildcard_result.get(
                    "iterations_improved", wildcard_result.get("total_improvements", 0)
                ),
                "total_iterations": wildcard_result["total_iterations"],
                "formation": wildcard_result.get("formation", "2-5-5-3"),
            }
            if "consensus_confidence" in wildcard_result:
                sa_result["consensus_confidence"] = wildcard_result[
                    "consensus_confidence"
                ]
                sa_result["consensus_runs"] = wildcard_result["consensus_runs"]
            best_sa_result = sa_result

        else:
            # Check if we should use exhaustive search for small transfer counts
            max_exhaustive = config.optimization.sa_exhaustive_search_max_transfers
            if max_exhaustive > 0 and free_transfers <= max_exhaustive:
                logger.info(
                    f"🔍 Using exhaustive search for {free_transfers} free transfer(s) (guaranteed optimal)..."
                )
                exhaustive_result = self._exhaustive_transfer_search(
                    current_squad=current_squad_with_xp,
                    all_players=all_players,
                    available_budget=available_budget,
                    free_transfers=free_transfers,
                    must_include_ids=must_include_ids,
                    must_exclude_ids=must_exclude_ids,
                    xp_column=xp_column,
                    current_xp=current_xp,
                    max_transfers=max_exhaustive,
                    is_bench_boost=is_bench_boost,
                )
                if exhaustive_result:
                    sa_result = exhaustive_result
                    best_net_xp = exhaustive_result["best_xp"]
                    logger.info(
                        f"✅ Exhaustive search complete: {best_net_xp:.2f} net xP"
                    )
                else:
                    # Fallback to SA if exhaustive fails
                    logger.warning("⚠️ Exhaustive search failed, falling back to SA")
                    sa_result = None
            else:
                sa_result = None

            # Use SA if exhaustive wasn't used or failed
            if sa_result is None:
                # Check if consensus mode is enabled
                if config.optimization.sa_use_consensus_mode:
                    logger.info(
                        f"🎯 Consensus mode: Running {config.optimization.sa_consensus_runs} full optimizations to find truly optimal solution..."
                    )
                    sa_result = self._consensus_optimization(
                        current_squad=current_squad_with_xp,
                        all_players=all_players,
                        available_budget=available_budget,
                        free_transfers=free_transfers,
                        must_include_ids=must_include_ids,
                        must_exclude_ids=must_exclude_ids,
                        xp_column=xp_column,
                        current_xp=current_xp,
                        is_bench_boost=is_bench_boost,
                    )
                    best_net_xp = sa_result["best_xp"]
                else:
                    # Normal mode: Run multiple SA restarts to find global optimum
                    num_restarts = config.optimization.sa_restarts
                    logger.info(
                        f"🔄 Running {num_restarts} SA restart(s) with {config.optimization.sa_iterations} iterations each..."
                    )

                    best_sa_result = None
                    best_net_xp = float("-inf")

                    for restart in range(num_restarts):
                        if num_restarts > 1:
                            logger.debug(f"  Restart {restart + 1}/{num_restarts}...")
                            # Use different seed for each restart if seed is set (for diversity)
                            if config.optimization.sa_random_seed is not None:
                                random.seed(
                                    config.optimization.sa_random_seed + restart
                                )

                        sa_result_restart = self._run_transfer_sa(
                            current_squad=current_squad_with_xp,
                            all_players=all_players,
                            available_budget=available_budget,
                            free_transfers=free_transfers,
                            must_include_ids=must_include_ids,
                            must_exclude_ids=must_exclude_ids,
                            xp_column=xp_column,
                            current_xp=current_xp,
                            iterations=config.optimization.sa_iterations,
                            is_bench_boost=is_bench_boost,
                        )

                        # Calculate net XP for this restart (including penalty)
                        best_squad_df = pd.DataFrame(sa_result_restart["optimal_squad"])
                        original_ids = set(current_squad_with_xp["player_id"].tolist())
                        new_ids = set(best_squad_df["player_id"].tolist())
                        num_transfers = len(original_ids - new_ids)
                        transfer_penalty = (
                            max(0, num_transfers - free_transfers)
                            * config.optimization.transfer_cost
                        )
                        net_xp = sa_result_restart["best_xp"] - transfer_penalty

                        if net_xp > best_net_xp:
                            best_net_xp = net_xp
                            best_sa_result = sa_result_restart
                            if num_restarts > 1:
                                logger.debug(
                                    f"    → New best: {net_xp:.2f} net xP ({num_transfers} transfers)"
                                )

                    sa_result = best_sa_result
                    logger.info(f"✅ Best result: {best_net_xp:.2f} net xP")

        # Create best scenario from SA result
        best_squad = sa_result["optimal_squad"]
        best_squad_df = pd.DataFrame(best_squad)

        # Calculate transfers made vs original squad
        original_ids = set(current_squad_with_xp["player_id"].tolist())
        new_ids = set(best_squad_df["player_id"].tolist())
        transfers_out = original_ids - new_ids
        transfers_in = new_ids - original_ids
        num_transfers = len(transfers_out)

        # Calculate transfer penalty
        transfer_penalty = (
            max(0, num_transfers - free_transfers) * config.optimization.transfer_cost
        )
        net_xp = sa_result["best_xp"] - transfer_penalty

        # Create transfer description
        is_wildcard = free_transfers >= 15
        if num_transfers == 0:
            description = "Keep current squad"
        elif is_wildcard:
            description = (
                f"Wildcard: Rebuild entire squad ({num_transfers} players changed)"
            )
        else:
            out_names = [
                current_squad_with_xp[current_squad_with_xp["player_id"] == pid][
                    "web_name"
                ].iloc[0]
                for pid in list(transfers_out)[:3]
            ]
            in_names = [
                best_squad_df[best_squad_df["player_id"] == pid]["web_name"].iloc[0]
                for pid in list(transfers_in)[:3]
            ]
            description = f"OUT: {', '.join(out_names)} → IN: {', '.join(in_names)}"

        # Determine scenario type
        scenario_type = "wildcard" if is_wildcard else "simulated_annealing"

        best_scenario = {
            "id": 0,
            "transfers": num_transfers,
            "type": scenario_type,
            "description": description,
            "penalty": transfer_penalty,
            "net_xp": net_xp,
            "formation": sa_result.get("formation", current_formation),
            "xp_gain": net_xp - current_xp,
            "squad": best_squad_df,
            "iterations_improved": sa_result.get("iterations_improved", 0),
            "is_wildcard": is_wildcard,
        }

        logger.info(
            f"✅ Best strategy: {num_transfers} transfers, {net_xp:.2f} net XP "
            f"({sa_result['iterations_improved']} improvements in {sa_result['total_iterations']} iterations)"
        )

        # Calculate remaining budget after squad selection
        squad_cost = best_squad_df["price"].sum()
        if is_wildcard:
            remaining_budget = 100.0 - squad_cost
        else:
            remaining_budget = available_budget - (
                squad_cost - budget_pool_info.get("non_sellable_value", 0)
            )

        # Return data structures for presentation layer to display
        optimization_metadata = {
            "method": "simulated_annealing",
            "horizon_label": horizon_label,
            "xp_column": xp_column,
            "budget_pool_info": budget_pool_info,
            "available_budget": team_data[
                "bank"
            ],  # Actual bank balance before optimization
            "remaining_budget": remaining_budget,  # Budget left after optimization
            "free_transfers": free_transfers,
            "current_xp": current_xp,
            "current_formation": current_formation,
            "sa_iterations": sa_result["total_iterations"],
            "sa_improvements": sa_result["iterations_improved"],
            "is_wildcard": is_wildcard,
            "transfers_out": [
                {
                    "web_name": current_squad_with_xp[
                        current_squad_with_xp["player_id"] == pid
                    ]["web_name"].iloc[0],
                    "position": current_squad_with_xp[
                        current_squad_with_xp["player_id"] == pid
                    ]["position"].iloc[0],
                    "price": current_squad_with_xp[
                        current_squad_with_xp["player_id"] == pid
                    ]["price"].iloc[0],
                }
                for pid in list(transfers_out)
            ],
            "transfers_in": [
                {
                    "web_name": best_squad_df[best_squad_df["player_id"] == pid][
                        "web_name"
                    ].iloc[0],
                    "position": best_squad_df[best_squad_df["player_id"] == pid][
                        "position"
                    ].iloc[0],
                    "price": best_squad_df[best_squad_df["player_id"] == pid][
                        "price"
                    ].iloc[0],
                }
                for pid in list(transfers_in)
            ],
        }

        return best_squad_df, best_scenario, optimization_metadata

    def _run_transfer_sa(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        available_budget: float,
        free_transfers: int,
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
        current_xp: float,
        iterations: int,
        is_bench_boost: bool = False,
    ) -> Dict[str, Any]:
        """Run simulated annealing for transfer optimization.

        Adapted from _simulated_annealing_squad_generation for transfer context.

        Returns:
            Dictionary with optimal_squad, best_xp, iterations_improved, etc.
        """
        # Convert current squad to list of dicts for SA processing
        original_squad = current_squad.to_dict("records")
        original_ids = {p["player_id"] for p in original_squad}

        # Check if must_include players need to be bought
        must_buy_ids = must_include_ids - original_ids
        if must_buy_ids:
            # Force-buy logic: Must-include players not in squad need to be added
            logger.info(
                f"🎯 Forcing purchase of {len(must_buy_ids)} must-include players..."
            )

            # Get must-buy players from all_players
            must_buy_players = all_players[
                all_players["player_id"].isin(must_buy_ids)
            ].to_dict("records")

            # Validate we can afford them and they fit in the squad
            must_buy_cost = sum(p["price"] for p in must_buy_players)

            # Find cheapest players to remove to make room
            swappable_by_price = sorted(
                [p for p in original_squad if p["player_id"] not in must_include_ids],
                key=lambda x: x["price"],
            )

            if len(swappable_by_price) < len(must_buy_ids):
                logger.warning(
                    "⚠️ Cannot fit all must-include players - not enough squad slots"
                )
                raise ValueError(
                    "Cannot fit all must-include players - not enough swappable players in squad"
                )

            # Remove cheapest N players (where N = number of must-buy players)
            players_to_remove = swappable_by_price[: len(must_buy_ids)]
            removed_value = sum(p["price"] for p in players_to_remove)

            if must_buy_cost > (available_budget + removed_value):
                logger.warning(
                    f"⚠️ Cannot afford must-include players: need £{must_buy_cost:.1f}m, "
                    f"have £{available_budget + removed_value:.1f}m"
                )
                raise ValueError(
                    f"Cannot afford must-include players: need £{must_buy_cost:.1f}m, "
                    f"have £{available_budget + removed_value:.1f}m"
                )

            # Build forced squad with must-buy players
            forced_squad = [
                p
                for p in original_squad
                if p["player_id"] not in {pl["player_id"] for pl in players_to_remove}
            ] + must_buy_players

            # Now use this forced squad as the baseline
            original_squad = forced_squad
            original_ids = {p["player_id"] for p in original_squad}
            available_budget = available_budget + removed_value - must_buy_cost
            logger.info(
                f"   Forced {len(must_buy_ids)} transfers to meet must-include constraints, "
                f"£{available_budget:.1f}m remaining"
            )

        # Track the max number of transfers allowed
        max_transfers = config.optimization.max_transfers  # Typically 0-3

        # Calculate initial objective value (squad xP with no transfer penalty)
        # Use availability-adjusted xP
        # Bench Boost: ALL 15 players count, Normal: only starting 11 count
        if is_bench_boost:
            # Bench Boost: ALL 15 players count
            starting_11_xp = sum(
                self.get_adjusted_xp(p, xp_column) for p in original_squad
            )
            initial_bench_penalty = 0.0  # No bench penalty - we WANT strong bench!
        else:
            # Normal: Only starting 11 count
            starting_11 = self._get_best_starting_11_from_squad(
                original_squad, xp_column
            )
            starting_11_xp = sum(
                self.get_adjusted_xp(p, xp_column) for p in starting_11
            )

            # For 1GW optimization, penalize expensive bench in objective
            initial_bench_penalty = 0.0
            if xp_column == "xP":
                starting_11_ids = {p["player_id"] for p in starting_11}
                bench_players = [
                    p for p in original_squad if p["player_id"] not in starting_11_ids
                ]
                bench_cost = sum(p["price"] for p in bench_players)
                initial_bench_penalty = bench_cost * 0.1  # Matches LP penalty

        current_objective = starting_11_xp - initial_bench_penalty

        best_team = [p.copy() for p in original_squad]
        best_objective = current_objective
        current_team = [p.copy() for p in original_squad]

        improvements = 0

        logger.debug(f"Initial squad: {current_objective:.2f} xP")

        # SA loop
        for iteration in range(iterations):
            # Linear temperature decrease
            temperature = max(0.01, 1.0 * (1 - iteration / iterations))

            # Generate neighbor by swapping 1-3 players
            new_team = self._swap_transfer_players(
                current_team,
                all_players,
                available_budget,
                must_include_ids,
                must_exclude_ids,
                xp_column,
            )

            if new_team is None:
                continue

            # CRITICAL: Check if new_team exceeds max_transfers from ORIGINAL squad
            new_ids = {p["player_id"] for p in new_team}
            num_transfers_from_original = len(original_ids - new_ids)

            if num_transfers_from_original > max_transfers:
                # Reject this move - too many transfers from original
                continue

            # Calculate new objective INCLUDING TRANSFER PENALTY
            # Use availability-adjusted xP
            # Bench Boost: ALL 15 players count, Normal: only starting 11 count
            if is_bench_boost:
                # Bench Boost: ALL 15 players count
                squad_xp = sum(self.get_adjusted_xp(p, xp_column) for p in new_team)
                bench_cost_penalty = 0.0  # No penalty - we WANT strong bench!
                # Apply ceiling bonus to all 15 players for Bench Boost
                ceiling_bonus = 0.0
                if config.optimization.ceiling_bonus_enabled:
                    for p in new_team:
                        xp = p.get(xp_column, 0)
                        uncertainty = p.get("xP_uncertainty", 0)
                        if uncertainty > 0:
                            ceiling = xp + 1.645 * uncertainty
                            if ceiling > 10:
                                ceiling_bonus += (
                                    ceiling - 10
                                ) * config.optimization.ceiling_bonus_factor
            else:
                # Normal: Only starting 11 count
                new_starting_11 = self._get_best_starting_11_from_squad(
                    new_team, xp_column
                )
                squad_xp = sum(
                    self.get_adjusted_xp(p, xp_column) for p in new_starting_11
                )

                # CEILING BONUS: Prefer players with haul potential
                # Based on top 1% manager analysis: they capture 3.0 haulers/GW vs 2.4 for average
                # Add bonus for players with high ceiling (95th percentile > 10)
                ceiling_bonus = 0.0
                if config.optimization.ceiling_bonus_enabled:
                    for p in new_starting_11:
                        xp = p.get(xp_column, 0)
                        uncertainty = p.get("xP_uncertainty", 0)
                        if uncertainty > 0:
                            # Calculate ceiling (95th percentile: mean + 1.645 * std)
                            ceiling = xp + 1.645 * uncertainty
                            # Bonus for players with ceiling > 10 (haul potential)
                            if ceiling > 10:
                                # Up to 1.5 bonus per player with 20 ceiling
                                ceiling_bonus += (
                                    ceiling - 10
                                ) * config.optimization.ceiling_bonus_factor

                # For 1GW optimization, penalize expensive bench
                bench_cost_penalty = 0.0
                if xp_column == "xP":
                    starting_11_ids = {p["player_id"] for p in new_starting_11}
                    bench_players = [
                        p for p in new_team if p["player_id"] not in starting_11_ids
                    ]
                    bench_cost = sum(p["price"] for p in bench_players)
                    bench_cost_penalty = bench_cost * 0.1  # Matches LP penalty

            # Calculate penalty for this number of transfers
            transfer_penalty = (
                max(0, num_transfers_from_original - free_transfers)
                * config.optimization.transfer_cost
            )
            new_objective = (
                squad_xp + ceiling_bonus - bench_cost_penalty - transfer_penalty
            )

            # Accept if better or with probability
            delta = new_objective - current_objective

            if delta > 0 or (
                temperature > 0 and random.random() < math.exp(delta / temperature)
            ):
                current_team = new_team
                current_objective = new_objective

                if current_objective > best_objective:
                    best_team = [p.copy() for p in current_team]
                    best_objective = current_objective
                    improvements += 1

                    if improvements % 10 == 0:
                        logger.debug(
                            f"Iteration {iteration}: New best {best_objective:.2f} xP ({num_transfers_from_original} transfers)"
                        )

        # Get best formation
        best_starting_11 = self._get_best_starting_11_from_squad(best_team, xp_column)
        _, best_formation, _ = self._enumerate_formations_for_players(
            self._group_by_position(best_starting_11), xp_column
        )

        logger.info(f"Final: {best_objective:.2f} xP ({improvements} improvements)")

        return {
            "optimal_squad": best_team,
            "best_xp": best_objective,
            "iterations_improved": improvements,
            "total_iterations": iterations,
            "formation": best_formation,
        }

    def _swap_transfer_players(
        self,
        team: List[Dict],
        all_players: pd.DataFrame,
        available_budget: float,
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
    ) -> Optional[List[Dict]]:
        """Swap 1-3 players in transfer context.

        Respects transfer budget = bank + selling prices.
        """
        new_team = [p.copy() for p in team]

        # Calculate current cost and budget
        current_cost = sum(p["price"] for p in new_team)

        # Determine how many players to swap (1-3)
        num_swaps = random.randint(
            1, min(3, config.optimization.sa_max_transfers_per_iteration)
        )

        # Get swappable players (not must-include)
        swappable_indices = [
            i for i, p in enumerate(new_team) if p["player_id"] not in must_include_ids
        ]

        if len(swappable_indices) < num_swaps:
            return None

        # Pick random players to swap out
        swap_indices = random.sample(swappable_indices, num_swaps)

        # Calculate budget from selling these players
        sold_value = sum(new_team[i]["price"] for i in swap_indices)
        swap_budget = available_budget + sold_value

        # Remove swapped players
        team_player_ids = {
            p["player_id"] for i, p in enumerate(new_team) if i not in swap_indices
        }

        # Find replacements for each position
        replacements = []
        remaining_budget = swap_budget

        for idx in swap_indices:
            old_player = new_team[idx]
            position = old_player["position"]

            # Get available replacements
            candidates = all_players[
                (all_players["position"] == position)
                & (~all_players["player_id"].isin(team_player_ids))
                & (~all_players["player_id"].isin(must_exclude_ids))
                & (all_players["price"] <= remaining_budget)
            ]

            if candidates.empty:
                return None

            # FILTER: Uncertainty-based differential strategy (2025/26 upgrade)
            # Allow differentials IF model is confident (low uncertainty)
            # This preserves asymmetric information advantage
            if "selected_by_percent" in candidates.columns:

                def is_valid_differential_target(row):
                    """
                    Smart filter using prediction uncertainty.

                    Logic:
                    1. Template/quality players (>15% owned, >£5.5m) → Always valid
                    2. Premiums (>£9m) → Always valid (proven players)
                    3. Differentials (<15% owned) → Valid IF:
                       - High xP/price (>1.0) AND
                       - Low uncertainty (<30% of xP)

                    Example:
                    - Munetsi: 6.0 xP ± 0.8, £4.5m → 1.33 xP/£, 13% uncertainty → ALLOW
                    - Cullen: 4.0 xP ± 2.0, £4.6m → 0.87 xP/£, 50% uncertainty → BLOCK
                    """
                    price = row["price"]
                    ownership = row["selected_by_percent"]
                    xp = row[xp_column]

                    # Route 1: Template/quality players (safe bets)
                    if ownership >= 15.0 and price >= 5.5:
                        return True

                    # Route 2: Premium exception (proven track record)
                    if price >= 9.0:
                        return True

                    # Route 3: Differential - require confident prediction
                    if ownership < 15.0:
                        xp_per_price = xp / max(price, 0.1)

                        # Need strong value signal (>1.0 xP per £m)
                        if xp_per_price < 1.0:
                            return False  # Insufficient value

                        # Check prediction confidence via uncertainty
                        uncertainty = row.get("xP_uncertainty", 0)
                        uncertainty_ratio = uncertainty / max(xp, 0.1)

                        # Allow if confident prediction (<30% uncertainty)
                        # Reject if uncertain (>30% uncertainty)
                        return uncertainty_ratio < 0.30

                    return True  # Default allow

                valid_mask = candidates.apply(is_valid_differential_target, axis=1)
                valid_targets = candidates[valid_mask]

                # Apply filter if it doesn't eliminate all options
                if not valid_targets.empty:
                    candidates = valid_targets

            # Prefer higher xP with randomness (or deterministic if enabled)
            topk = min(10, len(candidates))
            candidates = candidates.nlargest(topk, xp_column)

            if config.optimization.sa_deterministic_mode and len(candidates) > 1:
                # In deterministic mode, check if top candidate is significantly better
                top_xp = candidates.iloc[0][xp_column]
                second_xp = (
                    candidates.iloc[1][xp_column] if len(candidates) > 1 else top_xp
                )
                xp_gap = top_xp - second_xp

                # If top candidate is clearly better (>0.5 xP), always pick it
                if xp_gap > 0.5:
                    replacement = candidates.iloc[0].to_dict()
                else:
                    # Small gap: use weighted sampling (still some randomness for exploration)
                    replacement = candidates.sample(n=1).iloc[0].to_dict()
            else:
                # Normal mode: always use weighted sampling for exploration
                replacement = candidates.sample(n=1).iloc[0].to_dict()

            replacements.append(replacement)
            team_player_ids.add(replacement["player_id"])
            remaining_budget -= replacement["price"]

        # Build new team
        result_team = []
        swap_idx_set = set(swap_indices)
        replacement_iter = iter(replacements)

        for i, player in enumerate(new_team):
            if i in swap_idx_set:
                result_team.append(next(replacement_iter))
            else:
                result_team.append(player)

        # Validate budget
        new_cost = sum(p["price"] for p in result_team)
        if new_cost > current_cost + available_budget:
            return None

        # Validate 3-per-team constraint
        team_counts = self._count_players_per_team(result_team)
        if any(count > 3 for count in team_counts.values()):
            return None

        return result_team

    def _consensus_optimization(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        available_budget: float,
        free_transfers: int,
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
        current_xp: float,
        is_bench_boost: bool = False,
    ) -> Dict[str, Any]:
        """Run optimization multiple times and find consensus optimal solution.

        This addresses the problem where SA finds different "optimal" solutions each run.
        By running multiple times and aggregating results, we find the truly best solution.

        Args:
            Same as _run_transfer_sa

        Returns:
            Best consensus result with confidence metrics
        """
        num_runs = config.optimization.sa_consensus_runs
        num_restarts = config.optimization.sa_restarts
        iterations = config.optimization.sa_iterations

        logger.info(
            f"   Running {num_runs} full optimizations ({num_restarts} restarts × {iterations} iterations each)..."
        )

        all_results = []
        transfer_counts = {}  # Track how often each transfer combination appears

        for run in range(num_runs):
            if num_runs > 1:
                logger.debug(f"   Consensus run {run + 1}/{num_runs}...")

            # Run full SA with multiple restarts
            best_result = None
            best_net_xp = float("-inf")

            for restart in range(num_restarts):
                # Use different seed for each run+restart for diversity
                if config.optimization.sa_random_seed is not None:
                    random.seed(
                        config.optimization.sa_random_seed + run * 1000 + restart
                    )
                else:
                    # Still use some seed for this run to get different exploration
                    random.seed(run * 1000 + restart)

                sa_result = self._run_transfer_sa(
                    current_squad=current_squad,
                    all_players=all_players,
                    available_budget=available_budget,
                    free_transfers=free_transfers,
                    must_include_ids=must_include_ids,
                    must_exclude_ids=must_exclude_ids,
                    xp_column=xp_column,
                    current_xp=current_xp,
                    iterations=iterations,
                    is_bench_boost=is_bench_boost,
                )

                # Calculate net XP
                best_squad_df = pd.DataFrame(sa_result["optimal_squad"])
                original_ids = set(current_squad["player_id"].tolist())
                new_ids = set(best_squad_df["player_id"].tolist())
                transfers_out = original_ids - new_ids
                transfers_in = new_ids - original_ids
                num_transfers = len(transfers_out)

                transfer_penalty = (
                    max(0, num_transfers - free_transfers)
                    * config.optimization.transfer_cost
                )
                net_xp = sa_result["best_xp"] - transfer_penalty

                # Create transfer signature for tracking
                transfer_sig = (
                    tuple(sorted(transfers_out)),
                    tuple(sorted(transfers_in)),
                )
                transfer_counts[transfer_sig] = transfer_counts.get(transfer_sig, 0) + 1

                if net_xp > best_net_xp:
                    best_net_xp = net_xp
                    best_result = sa_result.copy()
                    best_result["net_xp"] = net_xp
                    best_result["transfers_out"] = transfers_out
                    best_result["transfers_in"] = transfers_in

            if best_result:
                all_results.append(best_result)

        if not all_results:
            raise ValueError("Consensus optimization failed - no valid results")

        # Find the best result overall
        best_overall = max(all_results, key=lambda r: r["net_xp"])

        # Calculate confidence: how often did this transfer appear?
        best_transfer_sig = (
            tuple(sorted(best_overall["transfers_out"])),
            tuple(sorted(best_overall["transfers_in"])),
        )
        confidence = transfer_counts.get(best_transfer_sig, 0) / num_runs

        logger.info(
            f"✅ Consensus: Best solution found {transfer_counts.get(best_transfer_sig, 0)}/{num_runs} times "
            f"({confidence * 100:.1f}% confidence), {best_overall['net_xp']:.2f} net xP"
        )

        # Add confidence to result
        best_overall["consensus_confidence"] = confidence
        best_overall["consensus_runs"] = num_runs
        best_overall["consensus_transfer_frequency"] = transfer_counts

        return best_overall

    def _exhaustive_transfer_search(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        available_budget: float,
        free_transfers: int,
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
        current_xp: float,
        max_transfers: int,
        is_bench_boost: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Exhaustive search for guaranteed optimal solution (0-N transfers).

        Only feasible for small transfer counts (0-2 transfers). Guarantees finding
        the truly optimal solution by evaluating all possible transfer combinations.

        Args:
            Same as _run_transfer_sa, plus max_transfers limit

        Returns:
            Optimal result or None if search is infeasible
        """

        original_squad = current_squad.to_dict("records")
        original_ids = {p["player_id"] for p in original_squad}

        # Check if must_include players need to be bought
        must_buy_ids = must_include_ids - original_ids
        if must_buy_ids:
            # Force-buy logic: Must-include players not in squad need to be added
            logger.info(
                f"🎯 Forcing purchase of {len(must_buy_ids)} must-include players..."
            )

            # Get must-buy players from all_players
            must_buy_players = all_players[
                all_players["player_id"].isin(must_buy_ids)
            ].to_dict("records")

            # Validate we can afford them and they fit in the squad
            must_buy_cost = sum(p["price"] for p in must_buy_players)

            # Find cheapest players to remove to make room
            swappable_by_price = sorted(
                [p for p in original_squad if p["player_id"] not in must_include_ids],
                key=lambda x: x["price"],
            )

            if len(swappable_by_price) < len(must_buy_ids):
                logger.warning(
                    "⚠️ Cannot fit all must-include players - not enough squad slots"
                )
                return None

            # Remove cheapest N players (where N = number of must-buy players)
            players_to_remove = swappable_by_price[: len(must_buy_ids)]
            removed_value = sum(p["price"] for p in players_to_remove)

            if must_buy_cost > (available_budget + removed_value):
                logger.warning(
                    f"⚠️ Cannot afford must-include players: need £{must_buy_cost:.1f}m, "
                    f"have £{available_budget + removed_value:.1f}m"
                )
                return None

            # Build forced squad with must-buy players
            forced_squad = [
                p
                for p in original_squad
                if p["player_id"] not in {pl["player_id"] for pl in players_to_remove}
            ] + must_buy_players

            # Now use this forced squad as the baseline
            original_squad = forced_squad
            original_ids = {p["player_id"] for p in original_squad}
            available_budget = available_budget + removed_value - must_buy_cost
            logger.info(
                f"   Forced {len(must_buy_ids)} transfers to meet must-include constraints, "
                f"£{available_budget:.1f}m remaining"
            )

        swappable_players = [
            p for p in original_squad if p["player_id"] not in must_include_ids
        ]

        if len(swappable_players) == 0:
            # All players are must-include, can't make any transfers
            logger.info("ℹ️ All players are must-include, returning current squad")
            starting_11 = self._get_best_starting_11_from_squad(
                original_squad, xp_column
            )
            squad_xp = sum(self.get_adjusted_xp(p, xp_column) for p in starting_11)
            return {
                "optimal_squad": original_squad,
                "best_xp": squad_xp,
                "net_xp": squad_xp,
                "transfers_out": must_buy_ids if must_buy_ids else set(),
                "transfers_in": must_buy_ids if must_buy_ids else set(),
                "iterations_improved": 0,
                "total_iterations": 1,
                "formation": "unknown",
            }

        best_result = None
        best_net_xp = float("-inf")
        total_combinations = 0

        # Try 0 transfers first
        # Use availability-adjusted xP for proper injury/doubtful handling
        if is_bench_boost:
            # Bench Boost: ALL 15 players count
            zero_transfer_xp = sum(
                self.get_adjusted_xp(p, xp_column) for p in original_squad
            )
            # Still need starting 11 for formation calculation
            starting_11 = self._get_best_starting_11_from_squad(
                original_squad, xp_column
            )
        else:
            # Normal: Only starting 11 count
            starting_11 = self._get_best_starting_11_from_squad(
                original_squad, xp_column
            )
            zero_transfer_xp = sum(
                self.get_adjusted_xp(p, xp_column) for p in starting_11
            )
        if zero_transfer_xp > best_net_xp:
            best_net_xp = zero_transfer_xp
            best_result = {
                "optimal_squad": original_squad,
                "best_xp": zero_transfer_xp,
                "net_xp": zero_transfer_xp,
                "transfers_out": set(),
                "transfers_in": set(),
                "iterations_improved": 0,
                "total_iterations": 1,
                "formation": self._enumerate_formations_for_players(
                    self._group_by_position(starting_11), xp_column
                )[1],
            }

        # Try 1 to max_transfers transfers
        for num_transfers in range(
            1, min(max_transfers + 1, len(swappable_players) + 1)
        ):
            # Generate all combinations of players to transfer out
            for out_players in combinations(swappable_players, num_transfers):
                out_ids = {p["player_id"] for p in out_players}
                out_value = sum(p["price"] for p in out_players)
                out_positions = [p["position"] for p in out_players]

                # Calculate available budget
                swap_budget = available_budget + out_value

                # Try to find replacements for each position
                team_player_ids = original_ids - out_ids

                # For exhaustive search, get top candidates per position
                # For 1 transfer: try top 100 candidates (truly exhaustive)
                # For 2 transfers: try top 50 candidates per position
                max_candidates_per_pos = 100 if num_transfers == 1 else 50

                # Get candidates for each unique position
                pos_candidates = {}
                for pos in set(out_positions):
                    candidates = all_players[
                        (all_players["position"] == pos)
                        & (~all_players["player_id"].isin(team_player_ids))
                        & (~all_players["player_id"].isin(must_exclude_ids))
                        & (all_players["price"] <= swap_budget)  # Can use full budget
                    ].nlargest(max_candidates_per_pos, xp_column)
                    pos_candidates[pos] = (
                        candidates.to_dict("records") if not candidates.empty else []
                    )

                # For single transfer, try ALL candidates (truly exhaustive)
                if num_transfers == 1:
                    pos = out_positions[0]
                    if pos not in pos_candidates or not pos_candidates[pos]:
                        continue

                    # Try EVERY candidate for this position
                    for candidate in pos_candidates[pos]:
                        if candidate["price"] > swap_budget:
                            continue

                        # Check 3-per-team constraint
                        candidate_team = candidate.get("team", candidate.get("team_id"))
                        existing_count = sum(
                            1
                            for p in original_squad
                            if p["player_id"] not in out_ids
                            and p.get("team", p.get("team_id")) == candidate_team
                        )
                        if existing_count >= 3:
                            continue  # Would violate 3-per-team

                        # Valid candidate - evaluate this transfer
                        replacements = [candidate]
                        total_combinations += 1
                        if total_combinations % 1000 == 0:
                            logger.debug(
                                f"   Evaluated {total_combinations} combinations..."
                            )

                        # Build new squad
                        new_squad = []
                        out_ids_set = out_ids

                        for player in original_squad:
                            if player["player_id"] in out_ids_set:
                                new_squad.append(candidate)
                            else:
                                new_squad.append(player)

                        # Validate constraints (should already be valid, but double-check)
                        team_counts = self._count_players_per_team(new_squad)
                        if any(count > 3 for count in team_counts.values()):
                            continue

                        # Calculate XP
                        # Bench Boost: ALL 15 players count, Normal: only starting 11 count
                        # Always get starting 11 for formation display
                        # Use availability-adjusted xP for proper injury/doubtful handling
                        starting_11 = self._get_best_starting_11_from_squad(
                            new_squad, xp_column
                        )
                        if is_bench_boost:
                            # Bench Boost: ALL 15 players count
                            squad_xp = sum(
                                self.get_adjusted_xp(p, xp_column) for p in new_squad
                            )
                            # Apply ceiling bonus to all 15 players for Bench Boost
                            ceiling_bonus = 0.0
                            if config.optimization.ceiling_bonus_enabled:
                                for p in new_squad:
                                    xp = p.get(xp_column, 0)
                                    uncertainty = p.get("xP_uncertainty", 0)
                                    if uncertainty > 0:
                                        ceiling = xp + 1.645 * uncertainty
                                        if ceiling > 10:
                                            ceiling_bonus += (
                                                ceiling - 10
                                            ) * config.optimization.ceiling_bonus_factor
                        else:
                            # Normal: Only starting 11 count
                            squad_xp = sum(
                                self.get_adjusted_xp(p, xp_column) for p in starting_11
                            )

                            # CEILING BONUS for exhaustive search
                            ceiling_bonus = 0.0
                            if config.optimization.ceiling_bonus_enabled:
                                for p in starting_11:
                                    xp = p.get(xp_column, 0)
                                    uncertainty = p.get("xP_uncertainty", 0)
                                    if uncertainty > 0:
                                        ceiling = xp + 1.645 * uncertainty
                                        if ceiling > 10:
                                            ceiling_bonus += (
                                                ceiling - 10
                                            ) * config.optimization.ceiling_bonus_factor

                        transfer_penalty = (
                            max(0, num_transfers - free_transfers)
                            * config.optimization.transfer_cost
                        )
                        net_xp = squad_xp + ceiling_bonus - transfer_penalty

                        if net_xp > best_net_xp:
                            best_net_xp = net_xp
                            in_ids = {candidate["player_id"]}
                            best_result = {
                                "optimal_squad": new_squad,
                                "best_xp": squad_xp,
                                "net_xp": net_xp,
                                "transfers_out": out_ids,
                                "transfers_in": in_ids,
                                "iterations_improved": 1,
                                "total_iterations": total_combinations,
                                "formation": self._enumerate_formations_for_players(
                                    self._group_by_position(starting_11), xp_column
                                )[1],
                            }
                else:
                    # For 2+ transfers, try all combinations of top candidates
                    from itertools import product

                    # Build list of candidate lists for each position
                    replacement_combos = []
                    for pos in out_positions:
                        if pos not in pos_candidates or not pos_candidates[pos]:
                            break
                        replacement_combos.append(pos_candidates[pos])
                    else:
                        # All positions have candidates - try all combinations
                        for replacement_combo in product(*replacement_combos):
                            # Check budget
                            combo_cost = sum(p["price"] for p in replacement_combo)
                            if combo_cost > swap_budget:
                                continue

                            # Check for duplicates
                            combo_ids = {p["player_id"] for p in replacement_combo}
                            if len(combo_ids) != len(replacement_combo):
                                continue

                            # Check 3-per-team constraint
                            combo_teams = {}
                            for p in replacement_combo:
                                team = p.get("team", p.get("team_id"))
                                combo_teams[team] = combo_teams.get(team, 0) + 1

                            existing_teams = {}
                            for p in original_squad:
                                if p["player_id"] not in out_ids:
                                    team = p.get("team", p.get("team_id"))
                                    existing_teams[team] = (
                                        existing_teams.get(team, 0) + 1
                                    )

                            violates_constraint = False
                            for team, count in combo_teams.items():
                                existing = existing_teams.get(team, 0)
                                if existing + count > 3:
                                    violates_constraint = True
                                    break

                            if violates_constraint:
                                continue

                            # Valid combination - evaluate
                            replacements = list(replacement_combo)
                            total_combinations += 1
                            if total_combinations % 1000 == 0:
                                logger.debug(
                                    f"   Evaluated {total_combinations} combinations..."
                                )

                            # Build new squad
                            new_squad = []
                            out_ids_set = out_ids
                            replacement_iter = iter(replacements)

                            for player in original_squad:
                                if player["player_id"] in out_ids_set:
                                    new_squad.append(next(replacement_iter))
                                else:
                                    new_squad.append(player)

                            # Calculate XP
                            # Bench Boost: ALL 15 players count, Normal: only starting 11 count
                            # Always get starting 11 for formation display
                            # Use availability-adjusted xP for proper injury/doubtful handling
                            starting_11 = self._get_best_starting_11_from_squad(
                                new_squad, xp_column
                            )
                            if is_bench_boost:
                                # Bench Boost: ALL 15 players count
                                squad_xp = sum(
                                    self.get_adjusted_xp(p, xp_column)
                                    for p in new_squad
                                )
                                # Apply ceiling bonus to all 15 players
                                ceiling_bonus = 0.0
                                if config.optimization.ceiling_bonus_enabled:
                                    for p in new_squad:
                                        xp = p.get(xp_column, 0)
                                        uncertainty = p.get("xP_uncertainty", 0)
                                        if uncertainty > 0:
                                            ceiling = xp + 1.645 * uncertainty
                                            if ceiling > 10:
                                                ceiling_bonus += (
                                                    (ceiling - 10)
                                                    * config.optimization.ceiling_bonus_factor
                                                )
                            else:
                                # Normal: Only starting 11 count
                                squad_xp = sum(
                                    self.get_adjusted_xp(p, xp_column)
                                    for p in starting_11
                                )

                                # CEILING BONUS for exhaustive search (2+ transfers)
                                ceiling_bonus = 0.0
                                if config.optimization.ceiling_bonus_enabled:
                                    for p in starting_11:
                                        xp = p.get(xp_column, 0)
                                        uncertainty = p.get("xP_uncertainty", 0)
                                        if uncertainty > 0:
                                            ceiling = xp + 1.645 * uncertainty
                                            if ceiling > 10:
                                                ceiling_bonus += (
                                                    (ceiling - 10)
                                                    * config.optimization.ceiling_bonus_factor
                                                )

                            transfer_penalty = (
                                max(0, num_transfers - free_transfers)
                                * config.optimization.transfer_cost
                            )
                            net_xp = squad_xp + ceiling_bonus - transfer_penalty

                            if net_xp > best_net_xp:
                                best_net_xp = net_xp
                                in_ids = {p["player_id"] for p in replacements}
                                best_result = {
                                    "optimal_squad": new_squad,
                                    "best_xp": squad_xp,
                                    "net_xp": net_xp,
                                    "transfers_out": out_ids,
                                    "transfers_in": in_ids,
                                    "iterations_improved": 1,
                                    "total_iterations": total_combinations,
                                    "formation": self._enumerate_formations_for_players(
                                        self._group_by_position(starting_11), xp_column
                                    )[1],
                                }

        if best_result:
            logger.info(
                f"   Exhaustive search: Evaluated {total_combinations} combinations, "
                f"best: {best_net_xp:.2f} net xP"
            )
        return best_result
