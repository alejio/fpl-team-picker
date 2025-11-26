"""Squad generation optimization for FPL.

This module handles initial squad and wildcard optimization:
- Building optimal 15-player squads from scratch
- Simulated annealing for squad generation
- Wildcard optimization
- Consensus wildcard optimization

Copy the following methods from optimization_service.py:
- optimize_wildcard_squad (lines 2859-2898)
- optimize_initial_squad (lines 2900-3039)
- _simulated_annealing_squad_generation (lines 3041-3420)
- _optimize_bench_for_1gw (lines 3450-3528)
- _consensus_wildcard_optimization (lines 3607-3694)
- get_optimal_team_from_database (lines 3530-3547)
- _select_optimal_team_from_all_players (lines 3578-3605)
"""

from typing import Dict, List, Tuple, Optional, Set, Any
import pandas as pd
import random
import math
from loguru import logger

from fpl_team_picker.config import config
from .optimization_base import InitialSquadOptimizationInput
from .transfer_core import TransferOptimizationMixin


class SquadGenerationMixin(TransferOptimizationMixin):
    """Mixin providing squad generation and wildcard optimization.

    Handles building optimal squads from scratch using simulated annealing.
    """

    def optimize_wildcard_squad(
        self,
        current_squad: pd.DataFrame,
        team_data: Dict,
        players_with_xp: pd.DataFrame,
        must_include_ids: Optional[Set[int]] = None,
        must_exclude_ids: Optional[Set[int]] = None,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Optimize squad for wildcard chip usage (15 free transfers).

        **UNIFIED API**: This method delegates to `optimize_transfers()` with
        `free_transfers_override=15`. The unified approach treats wildcard as
        15 free transfers with £100m budget reset.

        Args:
            current_squad: Current squad (will be replaced entirely for wildcard)
            team_data: Team data (bank will be overridden to £100m)
            players_with_xp: All players with XP calculations
            must_include_ids: Set of player IDs that must be included
            must_exclude_ids: Set of player IDs that must be excluded

        Returns:
            Tuple of (optimal_squad_df, result_summary_dict, optimization_metadata_dict)
        """
        logger.info(
            "🃏 Wildcard Chip: Using unified optimization with 15 free transfers..."
        )

        # Delegate to unified transfer optimization with 15 free transfers
        return self.optimize_transfers(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=players_with_xp,
            must_include_ids=must_include_ids,
            must_exclude_ids=must_exclude_ids,
            free_transfers_override=15,  # Wildcard = 15 free transfers
        )

    def optimize_initial_squad(
        self,
        players_with_xp: pd.DataFrame,
        budget: float = 100.0,
        formation: Tuple[int, int, int, int] = (2, 5, 5, 3),
        iterations: int = 5000,
        must_include_ids: Optional[Set[int]] = None,
        must_exclude_ids: Optional[Set[int]] = None,
        xp_column: str = "xP",
    ) -> Dict[str, Any]:
        """
        Generate optimal 15-player squad from scratch using simulated annealing.

        This is fundamentally different from transfer optimization - it starts with
        no existing squad and builds a complete team from all available players.

        Args:
            players_with_xp: DataFrame with player data and expected points (must have xP, price, position, player_id, team columns)
            budget: Total budget in millions (default 100.0)
            formation: Tuple of (GKP, DEF, MID, FWD) counts (default (2,5,5,3) = 15 players)
            iterations: Number of simulated annealing iterations (default 5000)
            must_include_ids: Set of player IDs that must be in final squad
            must_exclude_ids: Set of player IDs to exclude from consideration
            xp_column: Column name for expected points (default "xP")

        Returns:
            Dictionary with:
                - optimal_squad: List of 15 player dicts
                - remaining_budget: Unused budget
                - total_xp: Total expected points
                - iterations_improved: Number of improvements found
                - constraint_violations: Any violations detected

        Raises:
            ValueError: If data validation fails or constraints are impossible
        """
        # Validate inputs using data contract
        validation = InitialSquadOptimizationInput(
            budget=budget,
            formation=formation,
            iterations=iterations,
            must_include_ids=list(must_include_ids or []),
            must_exclude_ids=list(must_exclude_ids or []),
            xp_column=xp_column,
        )

        must_include_ids = set(validation.must_include_ids)
        must_exclude_ids = set(validation.must_exclude_ids)

        # Validate DataFrame has required columns
        required_cols = [xp_column, "price", "position", "player_id", "team"]
        missing_cols = [
            col for col in required_cols if col not in players_with_xp.columns
        ]
        if missing_cols:
            raise ValueError(
                f"players_with_xp DataFrame missing required columns: {missing_cols}"
            )

        # Validate no NaN values in critical columns - fail fast
        for col in required_cols:
            nan_count = players_with_xp[col].isna().sum()
            if nan_count > 0:
                raise ValueError(
                    f"Data quality issue: {nan_count} players have missing {col}. "
                    f"Fix upstream data processing - all players must have complete data."
                )

        # Filter valid players (apply exclusions)
        valid_players = players_with_xp[
            ~players_with_xp["player_id"].isin(must_exclude_ids)
        ].copy()

        if len(valid_players) < 15:
            raise ValueError(
                f"Only {len(valid_players)} players available after exclusions. "
                f"Cannot form 15-player squad."
            )

        # Validate must-include players exist and have complete data
        if must_include_ids:
            must_include_players = players_with_xp[
                players_with_xp["player_id"].isin(must_include_ids)
            ]

            if len(must_include_players) != len(must_include_ids):
                missing_ids = must_include_ids - set(must_include_players["player_id"])
                raise ValueError(
                    f"Must-include player IDs not found in dataset: {missing_ids}"
                )

            # Validate position constraints
            gkp_count, def_count, mid_count, fwd_count = formation
            position_requirements = {
                "GKP": gkp_count,
                "DEF": def_count,
                "MID": mid_count,
                "FWD": fwd_count,
            }

            must_include_positions = (
                must_include_players.groupby("position").size().to_dict()
            )
            for pos, required in position_requirements.items():
                included_count = must_include_positions.get(pos, 0)
                if included_count > required:
                    raise ValueError(
                        f"Must-include constraint violation: {included_count} {pos} players "
                        f"required but formation only allows {required}"
                    )

            # Validate budget constraint
            must_include_cost = must_include_players["price"].sum()
            if must_include_cost > budget:
                raise ValueError(
                    f"Must-include players cost £{must_include_cost:.1f}m, "
                    f"exceeding budget of £{budget:.1f}m"
                )

            # Validate 3-per-team constraint
            team_counts = must_include_players["team"].value_counts()
            violations = team_counts[team_counts > 3]
            if not violations.empty:
                raise ValueError(
                    f"Must-include players violate 3-per-team rule: "
                    f"{violations.to_dict()}"
                )

        # Run simulated annealing optimization
        result = self._simulated_annealing_squad_generation(
            valid_players=valid_players,
            budget=budget,
            formation=formation,
            iterations=iterations,
            must_include_ids=must_include_ids,
            must_exclude_ids=must_exclude_ids,
            xp_column=xp_column,
        )

        return result

    def _simulated_annealing_squad_generation(
        self,
        valid_players: pd.DataFrame,
        budget: float,
        formation: Tuple[int, int, int, int],
        iterations: int,
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
    ) -> Dict[str, Any]:
        """
        Internal method: Simulated annealing algorithm for squad generation.

        This implements the proven algorithm from season_planner.py with
        improvements: removed debug code, proper error handling, clean structure.
        """
        # Set random seed for reproducibility if configured
        if config.optimization.sa_random_seed is not None:
            random.seed(config.optimization.sa_random_seed)
        gkp_count, def_count, mid_count, fwd_count = formation
        position_requirements = {
            "GKP": gkp_count,
            "DEF": def_count,
            "MID": mid_count,
            "FWD": fwd_count,
        }

        def generate_random_team() -> Optional[List[Dict]]:
            """Generate a random valid 15-player squad."""
            team = []
            remaining_budget = budget
            team_counts = {}

            # Start with must-include players
            for player_id in must_include_ids:
                player = valid_players[valid_players["player_id"] == player_id].iloc[0]
                team_name = player["team"]

                # Verify 3-per-team constraint
                if team_counts.get(team_name, 0) >= 3:
                    return None

                team.append(player.to_dict())
                remaining_budget -= player["price"]
                team_counts[team_name] = team_counts.get(team_name, 0) + 1

            # Get minimum costs per position for budget feasibility
            min_costs = {}
            for position in ["GKP", "DEF", "MID", "FWD"]:
                pos_players = valid_players[valid_players["position"] == position]
                if len(pos_players) == 0:
                    return None
                min_costs[position] = pos_players["price"].min()

            # Fill each position
            for position in ["GKP", "DEF", "MID", "FWD"]:
                count = position_requirements[position]
                already_have = sum(1 for p in team if p["position"] == position)
                remaining_needed = count - already_have

                for _ in range(remaining_needed):
                    team_player_ids = {p["player_id"] for p in team}

                    # Calculate budget reserve for other positions
                    remaining_slots_cost = 0.0
                    for other_pos in ["GKP", "DEF", "MID", "FWD"]:
                        if other_pos == position:
                            continue
                        slots_needed = position_requirements[other_pos] - sum(
                            1 for p in team if p["position"] == other_pos
                        )
                        remaining_slots_cost += (
                            max(0, slots_needed) * min_costs[other_pos]
                        )

                    # Get affordable candidates respecting all constraints
                    max_affordable_price = remaining_budget - remaining_slots_cost
                    candidates = valid_players[
                        (valid_players["position"] == position)
                        & (~valid_players["player_id"].isin(team_player_ids))
                        & (valid_players["price"] <= max_affordable_price)
                    ].copy()

                    # Apply 3-per-team constraint
                    team_ok_mask = candidates["team"].map(
                        lambda t: team_counts.get(t, 0) < 3
                    )
                    candidates = candidates[team_ok_mask]

                    if len(candidates) == 0:
                        return None

                    # Prefer higher availability-adjusted xP with randomness
                    # Calculate adjusted xP for all candidates
                    candidates_list = candidates.to_dict("records")
                    for candidate in candidates_list:
                        candidate["_adjusted_xp"] = self.get_adjusted_xp(
                            candidate, xp_column
                        )
                    candidates_df = pd.DataFrame(candidates_list)

                    topk = min(8, len(candidates_df))
                    candidates_df = candidates_df.nlargest(topk, "_adjusted_xp")
                    player = candidates_df.sample(n=1).iloc[0]

                    team.append(player.to_dict())
                    remaining_budget -= player["price"]
                    team_counts[player["team"]] = team_counts.get(player["team"], 0) + 1

            return team if len(team) == 15 else None

        def calculate_team_xp(team: List[Dict]) -> float:
            """Calculate total xP for team's best starting 11, accounting for availability.

            For 1GW optimization (xp_column == 'xP'), adds bench efficiency penalty
            to avoid wasting budget on expensive bench players.
            """
            if len(team) != 15:
                return sum(self.get_adjusted_xp(p, xp_column) for p in team)

            starting_11 = self._get_best_starting_11_from_squad(team, xp_column)
            starting_11_ids = {p["player_id"] for p in starting_11}

            # Use availability-adjusted xP for scoring
            base_xp = sum(self.get_adjusted_xp(p, xp_column) for p in starting_11)

            # Penalty for constraint violations using shared utility
            team_counts = self._count_players_per_team(team)
            constraint_penalty = sum(
                (count - 3) * -10.0 for count in team_counts.values() if count > 3
            )

            # For 1GW optimization, penalize expensive bench players
            # This prevents selecting £14.9m Haaland just to bench him
            bench_cost_penalty = 0.0
            if xp_column == "xP":
                bench_players = [
                    p for p in team if p["player_id"] not in starting_11_ids
                ]
                bench_cost = sum(p["price"] for p in bench_players)
                # Penalize £1m bench cost as 0.1 xP (matches LP penalty)
                bench_cost_penalty = bench_cost * 0.1

            return base_xp + constraint_penalty - bench_cost_penalty

        def calculate_team_cost(team: List[Dict]) -> float:
            """Calculate total cost of team."""
            return sum(p["price"] for p in team)

        def is_valid_team(team: List[Dict]) -> bool:
            """Validate team satisfies all constraints."""
            if len(team) != 15:
                return False

            if calculate_team_cost(team) > budget:
                return False

            # Check formation
            position_counts = {}
            for player in team:
                pos = player["position"]
                position_counts[pos] = position_counts.get(pos, 0) + 1

            for pos, required in position_requirements.items():
                if position_counts.get(pos, 0) != required:
                    return False

            # Check 3-per-team using shared utility
            team_counts = self._count_players_per_team(team)
            for count in team_counts.values():
                if count > 3:
                    return False

            return True

        def swap_player(team: List[Dict]) -> Optional[List[Dict]]:
            """Generate neighbor solution by swapping one player."""
            new_team = [p.copy() for p in team]
            current_cost = calculate_team_cost(new_team)
            remaining_budget = budget - current_cost

            # Get swappable players (not must-include)
            swappable_indices = [
                i
                for i, p in enumerate(new_team)
                if p["player_id"] not in must_include_ids
            ]

            if not swappable_indices:
                return None

            # Pick random player to replace
            replace_idx = random.choice(swappable_indices)
            old_player = new_team[replace_idx]
            position = old_player["position"]

            # Get available replacements
            team_player_ids = {p["player_id"] for p in new_team}
            available = valid_players[
                (valid_players["position"] == position)
                & (~valid_players["player_id"].isin(team_player_ids))
                & (~valid_players["player_id"].isin(must_exclude_ids))
            ]

            if len(available) == 0:
                return None

            # Budget constraint with flexibility
            max_new_cost = old_player["price"] + remaining_budget + 5.0
            affordable = available[available["price"] <= max_new_cost]

            if len(affordable) == 0:
                affordable = available  # Try without budget constraint
                if len(affordable) == 0:
                    return None

            # Prefer higher availability-adjusted xP with weighting (or deterministic if enabled)
            if remaining_budget > 1.0 and len(affordable) > 1:
                # Calculate adjusted xP for all candidates
                affordable_list = affordable.to_dict("records")
                for candidate in affordable_list:
                    candidate["_adjusted_xp"] = self.get_adjusted_xp(
                        candidate, xp_column
                    )
                affordable_df = pd.DataFrame(affordable_list)

                if config.optimization.sa_deterministic_mode:
                    # In deterministic mode, check if top candidate is significantly better
                    affordable_sorted = affordable_df.sort_values(
                        "_adjusted_xp", ascending=False
                    )
                    top_xp = affordable_sorted.iloc[0]["_adjusted_xp"]
                    second_xp = (
                        affordable_sorted.iloc[1]["_adjusted_xp"]
                        if len(affordable_sorted) > 1
                        else top_xp
                    )
                    xp_gap = top_xp - second_xp

                    # If top candidate is clearly better (>0.5 xP), always pick it
                    if xp_gap > 0.5:
                        new_player = affordable_sorted.iloc[0].to_dict()
                    else:
                        # Small gap: use weighted sampling with adjusted xP
                        weights = affordable_df["_adjusted_xp"] * (
                            1 + affordable_df["price"] / 20
                        )
                        min_weight = weights.min()
                        if min_weight < 0:
                            weights = weights - min_weight + 0.1
                        weights_array = weights.values
                        if weights_array.sum() > 0:
                            weights_array = weights_array / weights_array.sum()
                            new_player = (
                                affordable_df.sample(n=1, weights=weights_array)
                                .iloc[0]
                                .to_dict()
                            )
                        else:
                            new_player = affordable_df.sample(n=1).iloc[0].to_dict()
                else:
                    # Normal mode: use weighted sampling with adjusted xP
                    weights = affordable_df["_adjusted_xp"] * (
                        1 + affordable_df["price"] / 20
                    )
                    min_weight = weights.min()
                    if min_weight < 0:
                        weights = weights - min_weight + 0.1

                    weights_array = weights.values
                    if weights_array.sum() > 0:
                        weights_array = weights_array / weights_array.sum()
                        new_player = (
                            affordable_df.sample(n=1, weights=weights_array)
                            .iloc[0]
                            .to_dict()
                        )
                    else:
                        new_player = affordable_df.sample(n=1).iloc[0].to_dict()
            else:
                # Budget too tight or only one option - still use adjusted xP if possible
                if len(affordable) > 0:
                    affordable_list = affordable.to_dict("records")
                    for candidate in affordable_list:
                        candidate["_adjusted_xp"] = self.get_adjusted_xp(
                            candidate, xp_column
                        )
                    affordable_df = pd.DataFrame(affordable_list)
                    new_player = (
                        affordable_df.nlargest(1, "_adjusted_xp").iloc[0].to_dict()
                    )
                else:
                    new_player = affordable.sample(n=1).iloc[0].to_dict()

            new_team[replace_idx] = new_player

            # Validate budget
            if calculate_team_cost(new_team) > budget:
                return None

            return new_team if is_valid_team(new_team) else None

        # Generate initial team
        current_team = None
        for _ in range(1000):
            current_team = generate_random_team()
            if current_team and is_valid_team(current_team):
                break

        if not current_team:
            raise ValueError(
                "Could not generate valid initial team. Check data quality and constraints."
            )

        best_team = [p.copy() for p in current_team]
        current_xp = calculate_team_xp(current_team)
        best_xp = current_xp

        # Simulated annealing loop
        improvements = 0

        for iteration in range(iterations):
            # Linear temperature decrease
            temperature = max(0.01, 1.0 * (1 - iteration / iterations))

            # Generate neighbor
            new_team = swap_player(current_team)
            if new_team is None:
                continue

            new_xp = calculate_team_xp(new_team)
            new_cost = calculate_team_cost(new_team)
            current_cost = calculate_team_cost(current_team)

            # Budget utilization bonus
            budget_util_bonus = 0
            if new_cost > current_cost:
                budget_util_bonus = (new_cost - current_cost) * 0.1

            delta = new_xp - current_xp + budget_util_bonus

            # Accept if better or with probability
            if delta > 0 or (
                temperature > 0 and random.random() < math.exp(delta / temperature)
            ):
                current_team = new_team
                current_xp = new_xp

                if current_xp > best_xp:
                    best_team = [p.copy() for p in current_team]
                    best_xp = current_xp
                    improvements += 1

        # Final validation
        if must_exclude_ids:
            final_team_ids = {p["player_id"] for p in best_team}
            violations = final_team_ids.intersection(must_exclude_ids)
            if violations:
                raise ValueError(
                    f"Algorithm error: Excluded players found in final team: {violations}"
                )

        # For 1GW optimization, replace expensive bench with cheapest alternatives
        if xp_column == "xP":
            best_team = self._optimize_bench_for_1gw(
                best_team, valid_players, must_include_ids, must_exclude_ids, xp_column
            )
            # Recalculate xP after bench optimization
            best_xp = calculate_team_xp(best_team)

        remaining_budget = budget - calculate_team_cost(best_team)

        return {
            "optimal_squad": best_team,
            "remaining_budget": remaining_budget,
            "total_xp": best_xp,
            "iterations_improved": improvements,
            "total_iterations": iterations,
            "final_cost": calculate_team_cost(best_team),
        }

    def _optimize_bench_for_1gw(
        self,
        squad: List[Dict],
        valid_players: pd.DataFrame,
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
    ) -> List[Dict]:
        """Replace expensive bench players with cheapest valid alternatives for 1GW.

        For 1GW optimization, bench players don't contribute points, so we want
        the cheapest valid bench to maximize budget for starters.

        Args:
            squad: 15-player squad
            valid_players: All available players
            must_include_ids: Players that must stay in squad
            must_exclude_ids: Players to exclude
            xp_column: XP column name

        Returns:
            Optimized squad with cheap bench
        """
        if len(squad) != 15:
            return squad

        # Identify starting 11 and bench
        starting_11 = self._get_best_starting_11_from_squad(squad, xp_column)
        starting_11_ids = {p["player_id"] for p in starting_11}
        bench_players = [p for p in squad if p["player_id"] not in starting_11_ids]

        # Get team counts from starting 11 (bench replacements must respect 3-per-team)
        team_counts = {}
        for p in starting_11:
            team = p.get("team", p.get("team_id"))
            team_counts[team] = team_counts.get(team, 0) + 1

        # For each bench position, find cheapest valid replacement
        new_squad = [p.copy() for p in starting_11]
        squad_ids = starting_11_ids.copy()

        for bench_player in bench_players:
            position = bench_player["position"]
            old_team = bench_player.get("team", bench_player.get("team_id"))

            # Skip must-include players
            if bench_player["player_id"] in must_include_ids:
                new_squad.append(bench_player.copy())
                squad_ids.add(bench_player["player_id"])
                team_counts[old_team] = team_counts.get(old_team, 0) + 1
                continue

            # Find cheapest valid replacement
            candidates = valid_players[
                (valid_players["position"] == position)
                & (~valid_players["player_id"].isin(squad_ids))
                & (~valid_players["player_id"].isin(must_exclude_ids))
            ].copy()

            # Apply 3-per-team constraint
            def team_ok(team):
                return team_counts.get(team, 0) < 3

            candidates = candidates[candidates["team"].apply(team_ok)]

            if len(candidates) == 0:
                # No replacement found, keep original
                new_squad.append(bench_player.copy())
                squad_ids.add(bench_player["player_id"])
                team_counts[old_team] = team_counts.get(old_team, 0) + 1
            else:
                # Pick cheapest
                cheapest = candidates.nsmallest(1, "price").iloc[0]
                new_squad.append(cheapest.to_dict())
                squad_ids.add(cheapest["player_id"])
                new_team = cheapest["team"]
                team_counts[new_team] = team_counts.get(new_team, 0) + 1

        return new_squad

    def _consensus_wildcard_optimization(
        self,
        players_with_xp: pd.DataFrame,
        budget: float,
        formation: Tuple[int, int, int, int],
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
    ) -> Dict[str, Any]:
        """Run wildcard optimization multiple times and find consensus optimal squad.

        This addresses the problem where SA finds different "optimal" squads each run.
        By running multiple times and aggregating results, we find the truly best squad.

        Args:
            players_with_xp: All available players with XP
            budget: Budget (always £100m for wildcard)
            formation: Formation tuple (GKP, DEF, MID, FWD)
            must_include_ids: Players that must be included
            must_exclude_ids: Players to exclude
            xp_column: Column name for XP values

        Returns:
            Best consensus result with confidence metrics
        """
        num_restarts = config.optimization.sa_wildcard_restarts
        iterations = config.optimization.sa_wildcard_iterations

        logger.info(
            f"   Running {num_restarts} wildcard optimizations ({iterations} iterations each)..."
        )

        all_results = []
        squad_signatures = {}  # Track how often each squad appears

        for restart in range(num_restarts):
            if num_restarts > 1:
                logger.debug(f"   Wildcard restart {restart + 1}/{num_restarts}...")

            # Use different seed for each restart for diversity
            if config.optimization.sa_random_seed is not None:
                random.seed(config.optimization.sa_random_seed + restart * 1000)
            else:
                # Still use some seed for this restart to get different exploration
                random.seed(restart * 1000)

            result = self.optimize_initial_squad(
                players_with_xp=players_with_xp,
                budget=budget,
                formation=formation,
                iterations=iterations,
                must_include_ids=must_include_ids,
                must_exclude_ids=must_exclude_ids,
                xp_column=xp_column,
            )

            # Create squad signature (sorted player IDs) for tracking
            squad_ids = tuple(sorted([p["player_id"] for p in result["optimal_squad"]]))
            squad_signatures[squad_ids] = squad_signatures.get(squad_ids, 0) + 1

            all_results.append(result)

        if not all_results:
            raise ValueError(
                "Wildcard consensus optimization failed - no valid results"
            )

        # Find the best result overall (highest xP)
        best_overall = max(all_results, key=lambda r: r["total_xp"])

        # Calculate confidence: how often did this exact squad appear?
        best_squad_sig = tuple(
            sorted([p["player_id"] for p in best_overall["optimal_squad"]])
        )
        confidence = squad_signatures.get(best_squad_sig, 0) / num_restarts

        logger.info(
            f"✅ Wildcard Consensus: Best squad found {squad_signatures.get(best_squad_sig, 0)}/{num_restarts} times "
            f"({confidence * 100:.1f}% confidence), {best_overall['total_xp']:.2f} xP"
        )

        # Add consensus metrics to result
        best_overall["consensus_confidence"] = confidence
        best_overall["consensus_runs"] = num_restarts
        best_overall["consensus_squad_frequency"] = squad_signatures
        best_overall["total_improvements"] = best_overall.get("iterations_improved", 0)

        return best_overall

    def get_optimal_team_from_database(
        self, players_with_xp: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Build optimal 11-player team from full player database.

        For analysis and testing purposes - builds theoretically best team
        without squad constraints.

        Args:
            players_with_xp: DataFrame with all available players

        Returns:
            Optimal starting eleven
        """
        available_players = players_with_xp[
            ~players_with_xp["status"].isin(["i", "s", "u"])
        ].copy()
        return self._select_optimal_team_from_all_players(available_players)

    def _select_optimal_team_from_all_players(
        self, players: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Select optimal starting 11 from full player database.

        Args:
            players: DataFrame of available players

        Returns:
            List of 11 player dicts forming optimal starting team
        """
        players_sorted = players.sort_values("xP", ascending=False)

        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for _, player in players_sorted.iterrows():
            position = player["position"]
            if position in by_position:
                by_position[position].append(player.to_dict())

        # Formation: 1 GKP, 4 DEF, 4 MID, 2 FWD
        formation = {"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2}

        starting_11 = []
        for position, needed in formation.items():
            available = by_position[position][:needed]
            starting_11.extend(available)

        return starting_11
