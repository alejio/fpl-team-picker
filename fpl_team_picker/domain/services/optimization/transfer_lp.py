"""Linear Programming transfer optimization for FPL.

This module handles transfer optimization using PuLP LP solver:
- Guaranteed optimal solution
- Deterministic (same input = same output)
- Fast (~1-2 seconds)

Copy the following methods from optimization_service.py:
- _optimize_transfers_lp (lines 1764-2548)
"""

from typing import Dict, Tuple, Set
import pandas as pd
import time
from loguru import logger

import pulp

from fpl_team_picker.config import config
from .optimization_base import OptimizationBaseMixin


class TransferLPMixin(OptimizationBaseMixin):
    """Mixin providing Linear Programming transfer optimization.

    Uses PuLP with CBC solver to find guaranteed optimal transfer solution.
    """

    def _optimize_transfers_lp(
        self,
        current_squad: pd.DataFrame,
        team_data: Dict,
        players_with_xp: pd.DataFrame,
        must_include_ids: Set[int] = None,
        must_exclude_ids: Set[int] = None,
        is_free_hit: bool = False,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Transfer optimization using Linear Programming (optimal solution).

        Uses PuLP with CBC solver to find guaranteed optimal transfer solution.
        Advantages over SA:
        - Provably optimal solution
        - Deterministic (same input = same output)
        - Faster (~1-2 seconds vs 10-45 seconds)

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

        start_time = time.time()

        logger.info("🎯 Linear Programming Optimization: Finding optimal solution...")

        # Process constraints
        must_include_ids = must_include_ids or set()
        must_exclude_ids = must_exclude_ids or set()

        if must_include_ids:
            logger.debug(f"🎯 Must include {len(must_include_ids)} players")
        if must_exclude_ids:
            logger.debug(f"🚫 Must exclude {len(must_exclude_ids)} players")

        # Get optimization column based on configuration
        # Free Hit always uses 1GW (squad reverts after deadline)
        if is_free_hit:
            xp_column = "xP"
            horizon_label = "1-GW"
            logger.info(
                "🎯 Free Hit mode: Optimizing for 1GW only (squad reverts after)"
            )
        else:
            xp_column = self.get_optimization_xp_column()
            if xp_column == "xP":
                horizon_label = "1-GW"
            elif xp_column == "xP_3gw":
                horizon_label = "3-GW"
            else:
                horizon_label = "5-GW"

        # Update current squad with XP data (same as SA)
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
        if "status" in players_with_xp.columns:
            merge_columns.append("status")

        # Include team information in merge
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

        # Get budget and transfer info
        available_budget = team_data["bank"]
        free_transfers = team_data.get("free_transfers", 1)

        # Calculate current squad xP for baseline
        current_squad_list = current_squad_with_xp.to_dict("records")
        current_starting_11 = self._get_best_starting_11_from_squad(
            current_squad_list, xp_column
        )
        current_xp = sum(
            self.get_adjusted_xp(p, xp_column) for p in current_starting_11
        )

        logger.info(f"📋 Current squad: {len(current_squad_list)} players")
        logger.info(f"   Baseline starting 11 xP: {current_xp:.2f}")
        # Get current formation for metadata
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for player in current_squad_list:
            by_position[player["position"]].append(player)
        for pos in by_position:
            by_position[pos].sort(
                key=lambda p: self.get_adjusted_xp(p, xp_column), reverse=True
            )
        _, current_formation, _ = self._enumerate_formations_for_players(
            by_position, xp_column
        )

        # Calculate total budget pool
        is_wildcard = free_transfers >= 15
        if is_wildcard:
            budget_pool_info = {
                "total_budget": 100.0,
                "sellable_value": 0.0,
                "non_sellable_value": 0.0,
                "must_include_value": 0.0,
            }
            total_budget = 100.0
            logger.info("🃏 Wildcard Budget: £100.0m (budget reset)")
        else:
            budget_pool_info = self.calculate_budget_pool(
                current_squad_with_xp, available_budget, must_include_ids
            )
            total_budget = budget_pool_info["total_budget"]
            # Calculate current squad total price for debugging
            current_squad_total_price = sum(p["price"] for p in current_squad_list)
            logger.info(
                f"💰 Budget: Bank £{available_budget:.1f}m | Sellable £{budget_pool_info['sellable_value']:.1f}m | Total £{total_budget:.1f}m"
            )
            logger.info(
                f"   Current squad price: £{current_squad_total_price:.1f}m "
                f"({'fits' if current_squad_total_price <= total_budget else 'EXCEEDS'} budget)"
            )

        # Get all available players and filter
        all_players = players_with_xp[players_with_xp["xP"].notna()].copy()

        # Track current squad player IDs for transfer penalty calculations
        current_player_ids = set(current_squad_with_xp["player_id"].tolist())

        # Filter out unavailable players (injured, suspended, unavailable)
        # NOTE: This includes current squad players - if they're injured, the LP
        # will be forced to transfer them out. The safety check at the end ensures
        # we don't recommend transfers if keeping current squad is better overall.
        if "status" in all_players.columns:
            unavailable_mask = all_players["status"].isin(["i", "s", "u"])

            # Log which current players are unavailable (will force transfers)
            unavailable_current = all_players[
                unavailable_mask & all_players["player_id"].isin(current_player_ids)
            ]
            if not unavailable_current.empty:
                logger.warning(
                    f"⚠️ Current squad has {len(unavailable_current)} unavailable players "
                    f"({unavailable_current['web_name'].tolist()}) - LP will consider transferring them"
                )

            # Filter out unavailable players
            all_players = all_players[~unavailable_mask]

        # Filter must_exclude
        if must_exclude_ids:
            all_players = all_players[~all_players["player_id"].isin(must_exclude_ids)]

        # Track if any current squad players were filtered out
        available_current_ids = current_player_ids & set(
            all_players["player_id"].tolist()
        )
        missing_current = current_player_ids - available_current_ids
        if missing_current:
            logger.warning(
                f"⚠️ {len(missing_current)} current squad players are unavailable - "
                f"LP must transfer them out to find valid squad"
            )

        logger.debug(f"Available players for LP: {len(all_players)}")

        # DEBUG: Show top potential upgrades by position for diagnostics
        if xp_column in all_players.columns:
            for position in ["GKP", "DEF", "MID", "FWD"]:
                pos_players = all_players[all_players["position"] == position]
                current_pos_players = current_squad_with_xp[
                    current_squad_with_xp["position"] == position
                ]
                if not current_pos_players.empty and not pos_players.empty:
                    worst_current = current_pos_players.loc[
                        current_pos_players[xp_column].idxmin()
                    ]
                    min_current_xp = worst_current[xp_column]
                    # Find better alternatives not in current squad
                    better_alternatives = pos_players[
                        (~pos_players["player_id"].isin(current_player_ids))
                        & (pos_players[xp_column] > min_current_xp)
                    ].nlargest(3, xp_column)
                    if not better_alternatives.empty:
                        best_alt = better_alternatives.iloc[0]
                        xp_gain = best_alt[xp_column] - min_current_xp
                        logger.info(
                            f"💡 Potential {position} upgrade: {worst_current['web_name']} "
                            f"({min_current_xp:.2f} xP, £{worst_current['price']:.1f}m) → "
                            f"{best_alt['web_name']} ({best_alt[xp_column]:.2f} xP, "
                            f"£{best_alt['price']:.1f}m) = +{xp_gain:.2f} xP"
                        )

        # === LINEAR PROGRAMMING FORMULATION ===

        # Initialize LP problem
        prob = pulp.LpProblem("FPL_Transfer_Optimization", pulp.LpMaximize)

        # Decision variables: x[player_id] = 1 if in squad, 0 otherwise
        player_vars = {}
        for _, player in all_players.iterrows():
            pid = player["player_id"]
            player_vars[pid] = pulp.LpVariable(f"player_{pid}", cat="Binary")

        # For 1GW optimization, maximize starting XI xP (not total squad)
        # For 3GW/5GW, keep total squad xP (bench players rotate in)
        optimize_starting_xi = xp_column == "xP"

        if optimize_starting_xi:
            # Add starter variables: s[pid] = 1 if starting, 0 otherwise
            starter_vars = {}
            for _, player in all_players.iterrows():
                pid = player["player_id"]
                starter_vars[pid] = pulp.LpVariable(f"starter_{pid}", cat="Binary")

            # Constraint: Can only start if in squad (s[i] <= x[i])
            for pid in player_vars:
                prob += starter_vars[pid] <= player_vars[pid], f"Starter_In_Squad_{pid}"

            # Constraint: Exactly 11 starters
            prob += (
                pulp.lpSum([starter_vars[pid] for pid in starter_vars]) == 11,
                "Starting_XI_Size",
            )

            # Formation constraints for starters (valid FPL formations)
            # GKP: exactly 1 starter
            gkp_players = all_players[all_players["position"] == "GKP"][
                "player_id"
            ].tolist()
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in gkp_players if pid in starter_vars]
                )
                == 1,
                "Starting_GKP",
            )

            # DEF: 3-5 starters
            def_players = all_players[all_players["position"] == "DEF"][
                "player_id"
            ].tolist()
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in def_players if pid in starter_vars]
                )
                >= 3,
                "Starting_DEF_Min",
            )
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in def_players if pid in starter_vars]
                )
                <= 5,
                "Starting_DEF_Max",
            )

            # MID: 2-5 starters
            mid_players = all_players[all_players["position"] == "MID"][
                "player_id"
            ].tolist()
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in mid_players if pid in starter_vars]
                )
                >= 2,
                "Starting_MID_Min",
            )
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in mid_players if pid in starter_vars]
                )
                <= 5,
                "Starting_MID_Max",
            )

            # FWD: 1-3 starters
            fwd_players = all_players[all_players["position"] == "FWD"][
                "player_id"
            ].tolist()
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in fwd_players if pid in starter_vars]
                )
                >= 1,
                "Starting_FWD_Min",
            )
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in fwd_players if pid in starter_vars]
                )
                <= 3,
                "Starting_FWD_Max",
            )

            # Objective: Maximize starting XI xP, penalize expensive bench
            # Bench cost penalty ensures we don't waste budget on bench players
            bench_cost_penalty = 0.1  # Penalize £1m bench cost as 0.1 xP
            objective_terms = []
            for _, player in all_players.iterrows():
                pid = player["player_id"]
                adjusted_xp = self.get_adjusted_xp(player.to_dict(), xp_column)
                player_price = player["price"]
                # Starting XI gets full xP weight
                objective_terms.append(adjusted_xp * starter_vars[pid])
                # Bench gets cost penalty (prefer cheap bench to free budget for starters)
                # bench_var = player_vars[pid] - starter_vars[pid] (1 if bench, 0 otherwise)
                objective_terms.append(
                    -bench_cost_penalty
                    * player_price
                    * (player_vars[pid] - starter_vars[pid])
                )

            prob += pulp.lpSum(objective_terms), "Starting_XI_XP"
            logger.info("🎯 1GW mode: Optimizing starting XI xP (not total squad)")
        else:
            # Multi-GW: Maximize total squad xP (bench players rotate in)
            objective_terms = []
            for _, player in all_players.iterrows():
                pid = player["player_id"]
                adjusted_xp = self.get_adjusted_xp(player.to_dict(), xp_column)
                objective_terms.append(adjusted_xp * player_vars[pid])

            prob += pulp.lpSum(objective_terms), "Total_Squad_XP"

        # CONSTRAINT 1: Squad size (exactly 15 players)
        prob += (
            pulp.lpSum([player_vars[pid] for pid in player_vars]) == 15,
            "Squad_Size",
        )

        # CONSTRAINT 2: Budget
        # CRITICAL: Use SELLING prices for current squad players, MARKET prices for new players
        # total_budget = bank + selling_value, so we must be consistent
        current_selling_prices = {
            p["player_id"]: p["price"] for p in current_squad_list
        }

        def get_player_cost(pid: int) -> float:
            """Get cost for budget constraint - selling price for current, market for new."""
            if pid in current_selling_prices:
                return current_selling_prices[
                    pid
                ]  # Use selling price for current players
            # Use market price for new players
            player_row = all_players[all_players["player_id"] == pid]
            if len(player_row) > 0:
                return player_row["price"].iloc[0]
            return 0.0

        budget_terms = [get_player_cost(pid) * player_vars[pid] for pid in player_vars]
        prob += pulp.lpSum(budget_terms) <= total_budget, "Budget_Limit"

        # DEBUG: Log budget constraint details
        current_squad_market_price = sum(
            all_players[all_players["player_id"] == p["player_id"]]["price"].iloc[0]
            for p in current_squad_list
            if len(all_players[all_players["player_id"] == p["player_id"]]) > 0
        )
        current_squad_selling_price = sum(p["price"] for p in current_squad_list)
        if current_squad_market_price > current_squad_selling_price + 0.5:
            logger.warning(
                f"⚠️ Price difference: Market £{current_squad_market_price:.1f}m vs Selling £{current_squad_selling_price:.1f}m "
                f"(diff: £{current_squad_market_price - current_squad_selling_price:.1f}m)"
            )

        # CONSTRAINT 3: Position requirements
        for position, count in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            position_players = all_players[all_players["position"] == position][
                "player_id"
            ].tolist()
            prob += (
                pulp.lpSum(
                    [player_vars[pid] for pid in position_players if pid in player_vars]
                )
                == count,
                f"Position_{position}",
            )

        # CONSTRAINT 4: Team limit (max 3 players per team)
        for team in all_players[team_col].unique():
            if pd.isna(team):
                continue
            team_players = all_players[all_players[team_col] == team][
                "player_id"
            ].tolist()
            prob += (
                pulp.lpSum(
                    [player_vars[pid] for pid in team_players if pid in player_vars]
                )
                <= 3,
                f"Team_Limit_{team}",
            )

        # CONSTRAINT 5: Transfer limit and penalty
        # We need to track transfers and penalize those beyond free_transfers
        current_ids = set(current_squad_with_xp["player_id"].tolist())

        if not is_wildcard:
            max_transfers = config.optimization.max_transfers

            # Limit max transfers
            prob += (
                pulp.lpSum(
                    [player_vars[pid] for pid in current_ids if pid in player_vars]
                )
                >= 15 - max_transfers,
                "Transfer_Limit",
            )

            # Add transfer penalty to objective function
            # penalty_transfers = max(0, num_transfers - free_transfers)
            #                   = max(0, (15 - num_kept) - free_transfers)
            # We use a continuous variable to model this
            penalty_transfers = pulp.LpVariable(
                "penalty_transfers", lowBound=0, cat="Continuous"
            )

            # num_kept = sum of current squad players selected
            num_kept = pulp.lpSum(
                [player_vars[pid] for pid in current_ids if pid in player_vars]
            )
            # num_transfers = 15 - num_kept
            # penalty_transfers >= num_transfers - free_transfers
            # penalty_transfers >= (15 - num_kept) - free_transfers
            # penalty_transfers >= 15 - free_transfers - num_kept
            prob += (
                penalty_transfers >= 15 - free_transfers - num_kept,
                "Penalty_Transfer_Lower_Bound",
            )

            # Subtract transfer penalty from objective (modify objective in-place)
            # penalty = penalty_transfers * transfer_cost (default 4 points)
            transfer_cost = config.optimization.transfer_cost

            # DEBUG: Log objective before adding penalty
            logger.debug(f"LP objective before penalty: {prob.objective}")

            prob.objective += -transfer_cost * penalty_transfers

            # DEBUG: Log objective after adding penalty
            logger.debug(f"LP objective after penalty: {prob.objective}")

            logger.info(
                f"🔄 Transfer penalty: -{transfer_cost} pts per transfer beyond {free_transfers} free"
            )

        # CONSTRAINT 6: Must include players
        for pid in must_include_ids:
            if pid in player_vars:
                prob += player_vars[pid] == 1, f"Must_Include_{pid}"

        # CONSTRAINT 7: Must exclude players (already filtered from all_players, but add for safety)
        for pid in must_exclude_ids:
            if pid in player_vars:
                prob += player_vars[pid] == 0, f"Must_Exclude_{pid}"

        # Solve with CBC solver (default, open-source)
        logger.debug("Solving LP problem with CBC solver...")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        solve_time = time.time() - start_time

        # Check solver status
        status = pulp.LpStatus[prob.status]

        # DEBUG: Log LP solution details
        if status == "Optimal" and not is_wildcard:
            kept_count = sum(
                1
                for pid in current_ids
                if pid in player_vars and pulp.value(player_vars[pid]) == 1
            )
            dropped_players = [
                current_squad_with_xp[current_squad_with_xp["player_id"] == pid][
                    "web_name"
                ].iloc[0]
                for pid in current_ids
                if pid in player_vars and pulp.value(player_vars[pid]) == 0
            ]
            penalty_val = pulp.value(penalty_transfers) if penalty_transfers else 0
            logger.info(
                f"🔍 LP DEBUG: kept={kept_count}/14 current players, "
                f"penalty_transfers={penalty_val:.1f}, "
                f"LP objective={pulp.value(prob.objective):.2f}"
            )
            if dropped_players:
                logger.info(
                    f"🔍 LP DEBUG: Dropped available players: {dropped_players}"
                )

            # What would keeping all 14 look like?
            if kept_count < 14:
                # Calculate what objective would be if we kept all 14
                theoretical_penalty = max(
                    0, 1 - free_transfers
                )  # 1 forced transfer (Gabriel)
                logger.info(
                    f"🔍 LP DEBUG: If kept all 14, penalty would be {theoretical_penalty} "
                    f"(1 forced transfer for Gabriel)"
                )
        if status != "Optimal":
            logger.error(f"❌ LP solver failed with status: {status}")
            return (
                pd.DataFrame(),
                {},
                {
                    "error": f"LP solver failed: {status}",
                    "method": "linear_programming",
                },
            )

        # Extract solution
        selected_player_ids = [
            pid for pid in player_vars if pulp.value(player_vars[pid]) == 1
        ]

        # For 1GW, also extract which players the LP chose as starters
        lp_starter_ids = None
        if optimize_starting_xi:
            lp_starter_ids = set(
                pid for pid in starter_vars if pulp.value(starter_vars[pid]) == 1
            )
            logger.debug(f"LP selected {len(lp_starter_ids)} starters")

        optimal_squad_df = all_players[
            all_players["player_id"].isin(selected_player_ids)
        ].copy()

        # Mark starters in DataFrame for UI display
        if lp_starter_ids is not None:
            optimal_squad_df["is_starter"] = optimal_squad_df["player_id"].isin(
                lp_starter_ids
            )
        else:
            optimal_squad_df["is_starter"] = True  # Will be determined by UI

        logger.debug(f"✅ LP solved in {solve_time:.2f}s - found optimal squad")

        # Calculate transfers
        original_ids = set(current_squad_with_xp["player_id"].tolist())
        new_ids = set(selected_player_ids)
        transfers_out = original_ids - new_ids
        transfers_in = new_ids - original_ids
        num_transfers = len(transfers_out)

        # Calculate optimal squad xP (for best starting 11)
        optimal_squad_list = optimal_squad_df.to_dict("records")

        # For 1GW, use LP's starter decisions; otherwise compute best 11
        if lp_starter_ids is not None:
            best_starting_11 = [
                p for p in optimal_squad_list if p["player_id"] in lp_starter_ids
            ]
            # Determine formation from LP starters
            pos_counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
            for p in best_starting_11:
                pos_counts[p["position"]] += 1
            best_formation = (
                f"{pos_counts['DEF']}-{pos_counts['MID']}-{pos_counts['FWD']}"
            )
        else:
            best_starting_11 = self._get_best_starting_11_from_squad(
                optimal_squad_list, xp_column
            )
            # Get best formation
            by_position_best = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
            for player in optimal_squad_list:
                by_position_best[player["position"]].append(player)
            for pos in by_position_best:
                by_position_best[pos].sort(
                    key=lambda p: self.get_adjusted_xp(p, xp_column), reverse=True
                )
            _, best_formation, _ = self._enumerate_formations_for_players(
                by_position_best, xp_column
            )

        optimal_squad_xp = sum(
            self.get_adjusted_xp(p, xp_column) for p in best_starting_11
        )

        # Calculate transfer penalty
        transfer_penalty = (
            max(0, num_transfers - free_transfers) * config.optimization.transfer_cost
        )
        net_xp = optimal_squad_xp - transfer_penalty

        # Check if unavailable players forced the LP to make transfers
        # (If current squad has unavailable players, they're filtered from all_players,
        # so LP cannot "keep" them - it must transfer them out)
        forced_transfers = len(missing_current) if missing_current else 0

        # Only apply safety check if LP wasn't forced to make transfers
        # If unavailable players exist, the LP solution is valid even if worse than
        # the theoretical "keep everyone" option (which isn't actually possible)
        if (
            net_xp < current_xp
            and num_transfers > 0
            and not is_wildcard
            and forced_transfers == 0
        ):
            logger.warning(
                f"⚠️ SAFETY CHECK: LP solution ({net_xp:.2f} net xP) is worse than "
                f"keeping current squad ({current_xp:.2f} xP). Difference: {net_xp - current_xp:+.2f}"
            )
            logger.warning(
                f"   LP found {num_transfers} transfers with squad xP {optimal_squad_xp:.2f} "
                f"- penalty {transfer_penalty:.0f} = {net_xp:.2f} net xP"
            )
            logger.warning("   Overriding to recommend 0 transfers.")

            # Override to "no transfers" recommendation
            num_transfers = 0
            transfer_penalty = 0
            net_xp = current_xp
            optimal_squad_df = current_squad_with_xp.copy()
            optimal_squad_xp = current_xp
            transfers_out = set()
            transfers_in = set()
            best_formation = current_formation
            best_starting_11 = current_starting_11
        elif forced_transfers > 0:
            logger.info(
                f"ℹ️ {forced_transfers} unavailable player(s) forced LP to make transfers - "
                f"accepting LP solution even though net xP ({net_xp:.2f}) < current xP ({current_xp:.2f})"
            )

        # Create transfer description
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
                optimal_squad_df[optimal_squad_df["player_id"] == pid]["web_name"].iloc[
                    0
                ]
                for pid in list(transfers_in)[:3]
            ]
            description = f"OUT: {', '.join(out_names)} → IN: {', '.join(in_names)}"

        # Determine scenario type
        scenario_type = "wildcard" if is_wildcard else "linear_programming"

        best_scenario = {
            "id": 0,
            "transfers": num_transfers,
            "type": scenario_type,
            "description": description,
            "penalty": transfer_penalty,
            "net_xp": net_xp,
            "formation": best_formation,
            "xp_gain": net_xp - current_xp,
            "squad": optimal_squad_df,
            "is_wildcard": is_wildcard,
            "lp_objective_value": pulp.value(prob.objective),
            "lp_solve_time": solve_time,
        }

        # Log detailed diagnostics for debugging
        logger.info(
            f"✅ LP solved in {solve_time:.2f}s - "
            f"Transfers: {num_transfers}, Squad XP: {optimal_squad_xp:.2f}, "
            f"Penalty: {transfer_penalty:.0f}, Net XP: {net_xp:.2f}, "
            f"Current XP: {current_xp:.2f}, Gain: {net_xp - current_xp:+.2f}"
        )
        if transfers_out and transfers_in:
            out_details = [
                f"{current_squad_with_xp[current_squad_with_xp['player_id'] == pid]['web_name'].iloc[0]} "
                f"({current_squad_with_xp[current_squad_with_xp['player_id'] == pid][xp_column].iloc[0]:.2f} xP)"
                for pid in list(transfers_out)[:3]
            ]
            in_details = [
                f"{optimal_squad_df[optimal_squad_df['player_id'] == pid]['web_name'].iloc[0]} "
                f"({optimal_squad_df[optimal_squad_df['player_id'] == pid][xp_column].iloc[0]:.2f} xP)"
                for pid in list(transfers_in)[:3]
            ]
            logger.info(f"📤 OUT: {', '.join(out_details)}")
            logger.info(f"📥 IN: {', '.join(in_details)}")

        # Calculate remaining budget
        squad_cost = optimal_squad_df["price"].sum()
        if is_wildcard:
            remaining_budget = 100.0 - squad_cost
        else:
            remaining_budget = available_budget - (
                squad_cost - budget_pool_info.get("non_sellable_value", 0)
            )

        # Return data structures matching SA format
        optimization_metadata = {
            "method": "linear_programming",
            "horizon_label": horizon_label,
            "xp_column": xp_column,
            "budget_pool_info": budget_pool_info,
            "available_budget": team_data["bank"],
            "remaining_budget": remaining_budget,
            "free_transfers": free_transfers,
            "current_xp": current_xp,
            "current_formation": current_formation,
            "lp_solve_time": solve_time,
            "lp_objective_value": pulp.value(prob.objective),
            "lp_status": status,
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
                    "web_name": optimal_squad_df[optimal_squad_df["player_id"] == pid][
                        "web_name"
                    ].iloc[0],
                    "position": optimal_squad_df[optimal_squad_df["player_id"] == pid][
                        "position"
                    ].iloc[0],
                    "price": optimal_squad_df[optimal_squad_df["player_id"] == pid][
                        "price"
                    ].iloc[0],
                }
                for pid in list(transfers_in)
            ],
        }

        return optimal_squad_df, best_scenario, optimization_metadata
