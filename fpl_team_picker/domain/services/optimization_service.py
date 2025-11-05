"""Optimization service for FPL squad and transfer optimization.

This service contains core FPL optimization algorithms including:
- Starting XI selection with formation optimization
- Bench player selection
- Budget pool calculations
- Transfer scenario analysis (0-3 transfers)
- Premium player acquisition planning
- Captain selection
- Initial squad generation using simulated annealing
"""

from typing import Dict, List, Tuple, Optional, Set, Any
import pandas as pd
import random
import math
from pydantic import BaseModel, Field, field_validator
from loguru import logger
from fpl_team_picker.config import config


class OptimizationService:
    """Service for FPL optimization algorithms and constraint satisfaction."""

    def __init__(self, optimization_config: Optional[Dict[str, Any]] = None):
        """Initialize optimization service.

        Args:
            optimization_config: Optional optimization configuration override
        """
        self.config = optimization_config or {}

    def get_optimization_xp_column(self) -> str:
        """Get the XP column to use for optimization based on configuration.

        Returns:
            'xP' for 1-gameweek optimization, 'xP_5gw' for 5-gameweek optimization
        """
        if config.optimization.optimization_horizon == "1gw":
            return "xP"
        else:  # "5gw"
            return "xP_5gw"

    def _count_players_per_team(self, team: List[Dict]) -> Dict[str, int]:
        """Count players per team.

        Args:
            team: List of player dictionaries with 'team' field

        Returns:
            Dictionary mapping team names to player counts
        """
        team_counts = {}
        for player in team:
            team_name = player["team"]
            team_counts[team_name] = team_counts.get(team_name, 0) + 1
        return team_counts

    def _enumerate_formations_for_players(
        self, by_position: Dict[str, List[Dict]], xp_column: str = "xP"
    ) -> Tuple[List[Dict], str, float]:
        """Core formation enumeration logic.

        Evaluates all valid FPL formations and returns the one with highest XP.

        Args:
            by_position: Players grouped by position (GKP, DEF, MID, FWD)
            xp_column: Column name for XP values

        Returns:
            Tuple of (best_11_players, formation_name, total_xp)
        """
        formations = [
            (1, 3, 5, 2),
            (1, 3, 4, 3),
            (1, 4, 5, 1),
            (1, 4, 4, 2),
            (1, 4, 3, 3),
            (1, 5, 4, 1),
            (1, 5, 3, 2),
            (1, 5, 2, 3),
        ]
        formation_names = {
            "(1, 3, 5, 2)": "3-5-2",
            "(1, 3, 4, 3)": "3-4-3",
            "(1, 4, 5, 1)": "4-5-1",
            "(1, 4, 4, 2)": "4-4-2",
            "(1, 4, 3, 3)": "4-3-3",
            "(1, 5, 4, 1)": "5-4-1",
            "(1, 5, 3, 2)": "5-3-2",
            "(1, 5, 2, 3)": "5-2-3",
        }

        best_11, best_xp, best_formation = [], 0, ""

        for gkp, def_count, mid, fwd in formations:
            if (
                gkp <= len(by_position["GKP"])
                and def_count <= len(by_position["DEF"])
                and mid <= len(by_position["MID"])
                and fwd <= len(by_position["FWD"])
            ):
                formation_11 = (
                    by_position["GKP"][:gkp]
                    + by_position["DEF"][:def_count]
                    + by_position["MID"][:mid]
                    + by_position["FWD"][:fwd]
                )
                formation_xp = sum(p.get(xp_column, 0) for p in formation_11)

                if formation_xp > best_xp:
                    best_xp = formation_xp
                    best_11 = formation_11
                    best_formation = formation_names.get(
                        str((gkp, def_count, mid, fwd)),
                        f"{gkp}-{def_count}-{mid}-{fwd}",
                    )

        return best_11, best_formation, best_xp

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
            xp_column: Column to use for XP sorting ('xP' for current GW, 'xP_5gw' for strategic)

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

        # Default to current gameweek XP, fallback to 5GW if current not available
        sort_col = (
            xp_column
            if xp_column in available_squad.columns
            else ("xP_5gw" if "xP_5gw" in available_squad.columns else "xP")
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
            xp_column: Column to use for XP sorting ('xP' for current GW, 'xP_5gw' for strategic)

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
            else ("xP_5gw" if "xP_5gw" in squad_df.columns else "xP")
        )

        for _, player in squad_df.iterrows():
            if player["player_id"] not in starting_11_ids:
                bench_players.append(player.to_dict())

        # Sort bench by XP (highest first)
        bench_players.sort(key=lambda p: p.get(sort_col, 0), reverse=True)

        return bench_players[:4]  # Maximum 4 bench players

    def get_captain_recommendation(
        self, players: pd.DataFrame, top_n: int = 5, xp_column: str = "xP"
    ) -> Dict[str, Any]:
        """Get captain recommendation with risk-adjusted analysis.

        Returns structured data only - no UI generation.

        Implements uncertainty-aware + ownership-adjusted captain selection:
        1. Considers prediction uncertainty (xP_uncertainty) to prefer reliable picks
        2. Template protection: High ownership (>50%) players favored if within 10% xP of leader
        3. Risk-adjusted scoring: xP / (1 + uncertainty_penalty)

        Args:
            players: DataFrame of players to consider (squad or all players)
            top_n: Number of top candidates to analyze (default 5)
            xp_column: Column to use for xP (default "xP")

        Returns:
            {
                "captain": {dict with recommended captain},
                "vice_captain": {dict with vice captain},
                "top_candidates": [list of top 5 candidates with analysis],
                "captain_upside": float (xP * 2),
                "differential": float (captain vs vice diff)
            }
        """
        if players.empty:
            raise ValueError("No players provided for captain selection")

        # Ensure required columns exist
        required_cols = ["web_name", "position", xp_column]
        missing_cols = [col for col in required_cols if col not in players.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert to list of dicts for easier processing
        players_list = players.to_dict("records")

        # Enhanced captain scoring with uncertainty and ownership
        has_uncertainty = "xP_uncertainty" in players.columns
        has_ownership = "selected_by_percent" in players.columns

        for player in players_list:
            xp_value = player.get(xp_column, 0)
            uncertainty = player.get("xP_uncertainty", 0)
            ownership_pct = player.get("selected_by_percent", 0)

            # Base captain score (xP * 2)
            base_score = xp_value * 2

            # Apply uncertainty penalty if available
            # Higher uncertainty = lower score (prefer reliable players)
            uncertainty_penalty = 0
            if has_uncertainty and uncertainty > 0:
                # Penalty scales with uncertainty relative to xP
                # If uncertainty is 50% of xP, reduce score by ~15%
                uncertainty_ratio = uncertainty / max(xp_value, 0.1)
                uncertainty_penalty = uncertainty_ratio * 0.3  # Max 30% penalty

            risk_adjusted_score = base_score / (1 + uncertainty_penalty)

            # Template protection: Boost high ownership players (>50%)
            ownership_bonus = 0
            if has_ownership and ownership_pct > 50:
                # Template captain (>50% owned) gets 5-10% bonus
                # This protects against big rank swings if they haul
                ownership_bonus = min((ownership_pct - 50) / 100, 0.10)  # Max 10% bonus

            final_score = risk_adjusted_score * (1 + ownership_bonus)

            # Store scoring components for analysis
            player["_captain_score"] = final_score
            player["_base_score"] = base_score
            player["_uncertainty_penalty"] = uncertainty_penalty
            player["_ownership_bonus"] = ownership_bonus

        # Sort by enhanced captain score
        captain_candidates = sorted(
            players_list, key=lambda p: p.get("_captain_score", 0), reverse=True
        )

        if not captain_candidates:
            raise ValueError("No valid captain candidates found")

        # Analyze top N candidates
        top_candidates = []

        for i, player in enumerate(captain_candidates[:top_n]):
            xp_value = player.get(xp_column, 0)
            fixture_outlook = player.get("fixture_outlook", "🟡 Average")
            uncertainty = player.get("xP_uncertainty", 0)
            ownership_pct = player.get("selected_by_percent", 0)

            # Enhanced risk assessment
            risk_factors = []
            risk_level = "🟢 Low"

            # Prediction uncertainty risk
            if has_uncertainty and uncertainty > 0:
                uncertainty_ratio = uncertainty / max(xp_value, 0.1)
                if uncertainty_ratio > 0.4:  # High uncertainty (>40% of xP)
                    risk_factors.append(f"high variance ({uncertainty:.2f})")
                    if risk_level == "🟢 Low":
                        risk_level = "🟡 Medium"

            # Injury/availability risk
            if player.get("status") in ["i", "d"]:
                risk_factors.append("injury risk")
                risk_level = "🔴 High"
            elif player.get("status") == "s":  # Suspended
                risk_factors.append("suspended")
                risk_level = "🔴 High"

            # Fixture difficulty risk
            if "Hard" in fixture_outlook or "🔴" in fixture_outlook:
                risk_factors.append("hard fixture")
                if risk_level == "🟢 Low":
                    risk_level = "🟡 Medium"

            # Minutes certainty
            expected_mins = player.get("expected_minutes", 0)
            if expected_mins < 60:
                risk_factors.append("rotation risk")
                risk_level = "🟡 Medium" if risk_level == "🟢 Low" else risk_level

            # Combine risk description
            if risk_factors:
                risk_desc = f"{risk_level} ({', '.join(risk_factors)})"
            else:
                risk_desc = risk_level

            # Calculate captaincy potential (XP * 2 for double points)
            captain_potential = xp_value * 2

            # Build candidate dict with new scoring components
            candidate = {
                "rank": i + 1,
                "player_id": player.get("player_id"),
                "web_name": player["web_name"],
                "position": player["position"],
                "price": player.get("price", 0),
                "xP": xp_value,
                "captain_points": captain_potential,
                "expected_minutes": expected_mins,
                "fixture_outlook": fixture_outlook,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "risk_description": risk_desc,
                "status": player.get("status", "a"),
            }

            # Add uncertainty and ownership metrics if available
            if has_uncertainty:
                candidate["xP_uncertainty"] = uncertainty
                candidate["uncertainty_penalty"] = player.get("_uncertainty_penalty", 0)

            if has_ownership:
                candidate["ownership_pct"] = ownership_pct
                candidate["ownership_bonus"] = player.get("_ownership_bonus", 0)
                # Add template flag
                candidate["is_template"] = ownership_pct > 50

            # Add risk-adjusted score
            candidate["captain_score"] = player.get("_captain_score", captain_potential)

            top_candidates.append(candidate)

        # Select captain and vice captain
        captain = top_candidates[0]
        vice_captain = top_candidates[1] if len(top_candidates) > 1 else captain

        # Calculate captaincy metrics
        captain_upside = captain["captain_points"]
        vice_upside = vice_captain["captain_points"]
        differential = captain_upside - vice_upside

        return {
            "captain": captain,
            "vice_captain": vice_captain,
            "top_candidates": top_candidates,
            "captain_upside": captain_upside,
            "vice_upside": vice_upside,
            "differential": differential,
        }

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

    def optimize_transfers(
        self,
        current_squad: pd.DataFrame,
        team_data: Dict,
        players_with_xp: pd.DataFrame,
        must_include_ids: Set[int] = None,
        must_exclude_ids: Set[int] = None,
        free_transfers_override: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Unified transfer optimization supporting normal gameweeks and chips.

        This is the core transfer optimization algorithm that analyzes multiple
        transfer scenarios and recommends the optimal strategy.

        **Unified Chip Support:**
        - Normal gameweek: free_transfers=1 (analyze 0-3 transfers)
        - Saved transfer: free_transfers=2 (analyze 0-4 transfers)
        - Wildcard chip: free_transfers=15 (rebuild entire squad, budget resets to £100m)
        - Free Hit chip: free_transfers=15 + revert_after_gw flag (future extension)

        Uses simulated annealing for transfer optimization.

        Clean architecture: Returns data structures only, no UI generation.

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

        Returns:
            Tuple of (optimal_squad_df, best_scenario_dict, optimization_metadata_dict)
            where optimization_metadata contains budget_pool_info, scenarios, etc for display
        """
        # Apply free transfers override if wildcard/chip is active
        if free_transfers_override is not None:
            team_data = team_data.copy()
            team_data["free_transfers"] = free_transfers_override

            # Wildcard chip: Budget resets to £100m (ignore current squad value)
            if free_transfers_override >= 15:
                team_data["bank"] = 100.0
                # Note: Selling prices will be ignored since we can replace all 15 players
                logger.info(
                    f"🃏 Wildcard/Chip Active: {free_transfers_override} free transfers, "
                    f"£{team_data['bank']:.1f}m budget"
                )

        # Use simulated annealing for transfer optimization
        return self._optimize_transfers_sa(
            current_squad,
            team_data,
            players_with_xp,
            must_include_ids,
            must_exclude_ids,
        )

    def _optimize_transfers_sa(
        self,
        current_squad: pd.DataFrame,
        team_data: Dict,
        players_with_xp: pd.DataFrame,
        must_include_ids: Set[int] = None,
        must_exclude_ids: Set[int] = None,
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
        xp_column = self.get_optimization_xp_column()
        horizon_label = "1-GW" if xp_column == "xP" else "5-GW"

        # Update current squad with both 1-GW and 5-GW XP data
        merge_columns = ["player_id", "xP", "xP_5gw"]
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

        # Get best starting 11 from current squad
        current_best_11, current_formation, current_xp = self.find_optimal_starting_11(
            current_squad_with_xp, xp_column
        )

        logger.info(
            f"📊 Current squad: {current_xp:.2f} {horizon_label}-xP | Formation: {current_formation}"
        )

        # For wildcard (15 free transfers), use initial squad generation instead of transfer-based SA
        if is_wildcard:
            logger.info(
                "🃏 Wildcard mode: Building optimal squad from scratch (ignoring current squad)"
            )
            wildcard_result = self.optimize_initial_squad(
                players_with_xp=all_players,
                budget=100.0,  # Wildcard always £100m
                formation=(2, 5, 5, 3),
                iterations=config.optimization.sa_iterations,
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
                "iterations_improved": wildcard_result["iterations_improved"],
                "total_iterations": wildcard_result["total_iterations"],
            }
            best_sa_result = sa_result

        else:
            # Normal transfers: Run multiple SA restarts to find global optimum
            num_restarts = config.optimization.sa_restarts
            logger.info(
                f"🔄 Running {num_restarts} SA restart(s) with {config.optimization.sa_iterations} iterations each..."
            )

            best_sa_result = None
            best_net_xp = float("-inf")

            for restart in range(num_restarts):
                if num_restarts > 1:
                    logger.debug(f"  Restart {restart + 1}/{num_restarts}...")

                sa_result = self._run_transfer_sa(
                    current_squad=current_squad_with_xp,
                    all_players=all_players,
                    available_budget=available_budget,
                    free_transfers=free_transfers,
                    must_include_ids=must_include_ids,
                    must_exclude_ids=must_exclude_ids,
                    xp_column=xp_column,
                    current_xp=current_xp,
                    iterations=config.optimization.sa_iterations,
                )

                # Calculate net XP for this restart (including penalty)
                best_squad_df = pd.DataFrame(sa_result["optimal_squad"])
                original_ids = set(current_squad_with_xp["player_id"].tolist())
                new_ids = set(best_squad_df["player_id"].tolist())
                num_transfers = len(original_ids - new_ids)
                transfer_penalty = (
                    max(0, num_transfers - free_transfers)
                    * config.optimization.transfer_cost
                )
                net_xp = sa_result["best_xp"] - transfer_penalty

                if net_xp > best_net_xp:
                    best_net_xp = net_xp
                    best_sa_result = sa_result
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
    ) -> Dict[str, Any]:
        """Run simulated annealing for transfer optimization.

        Adapted from _simulated_annealing_squad_generation for transfer context.

        Returns:
            Dictionary with optimal_squad, best_xp, iterations_improved, etc.
        """
        # Convert current squad to list of dicts for SA processing
        original_squad = current_squad.to_dict("records")
        original_ids = {p["player_id"] for p in original_squad}

        # Track the max number of transfers allowed
        max_transfers = config.optimization.max_transfers  # Typically 0-3

        # Calculate initial objective value (squad xP with no transfer penalty)
        starting_11 = self._get_best_starting_11_from_squad(original_squad, xp_column)
        current_objective = sum(p[xp_column] for p in starting_11)

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
            new_starting_11 = self._get_best_starting_11_from_squad(new_team, xp_column)
            squad_xp = sum(p[xp_column] for p in new_starting_11)

            # Calculate penalty for this number of transfers
            transfer_penalty = (
                max(0, num_transfers_from_original - free_transfers)
                * config.optimization.transfer_cost
            )
            new_objective = squad_xp - transfer_penalty  # Net XP after penalty

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

            # Prefer higher xP with randomness
            topk = min(10, len(candidates))
            candidates = candidates.nlargest(topk, xp_column)
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

    def _group_by_position(self, players: List[Dict]) -> Dict[str, List[Dict]]:
        """Group players by position for formation enumeration."""
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for player in players:
            by_position[player["position"]].append(player)
        return by_position

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

        **Backward Compatibility**: Maintains the same return format as before.

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

                    # Prefer higher xP with randomness
                    topk = min(8, len(candidates))
                    candidates = candidates.nlargest(topk, xp_column)
                    player = candidates.sample(n=1).iloc[0]

                    team.append(player.to_dict())
                    remaining_budget -= player["price"]
                    team_counts[player["team"]] = team_counts.get(player["team"], 0) + 1

            return team if len(team) == 15 else None

        def calculate_team_xp(team: List[Dict]) -> float:
            """Calculate total xP for team's best starting 11."""
            if len(team) != 15:
                return sum(p[xp_column] for p in team)

            starting_11 = self._get_best_starting_11_from_squad(team, xp_column)
            base_xp = sum(p[xp_column] for p in starting_11)

            # Penalty for constraint violations using shared utility
            team_counts = self._count_players_per_team(team)
            constraint_penalty = sum(
                (count - 3) * -10.0 for count in team_counts.values() if count > 3
            )

            return base_xp + constraint_penalty

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

            # Prefer higher xP with weighting
            if remaining_budget > 1.0 and len(affordable) > 1:
                weights = affordable[xp_column] * (1 + affordable["price"] / 20)
                min_weight = weights.min()
                if min_weight < 0:
                    weights = weights - min_weight + 0.1

                weights_array = weights.values
                if weights_array.sum() > 0:
                    weights_array = weights_array / weights_array.sum()
                    new_player = (
                        affordable.sample(n=1, weights=weights_array).iloc[0].to_dict()
                    )
                else:
                    new_player = affordable.sample(n=1).iloc[0].to_dict()
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

        remaining_budget = budget - calculate_team_cost(best_team)

        return {
            "optimal_squad": best_team,
            "remaining_budget": remaining_budget,
            "total_xp": best_xp,
            "iterations_improved": improvements,
            "total_iterations": iterations,
            "final_cost": calculate_team_cost(best_team),
        }

    def _get_best_starting_11_from_squad(
        self, squad: List[Dict], xp_column: str = "xP"
    ) -> List[Dict]:
        """Get best starting 11 from 15-player squad.

        Args:
            squad: List of 15 player dictionaries
            xp_column: Column name for XP values

        Returns:
            List of 11 player dicts forming best starting team
        """
        if len(squad) != 15:
            return []

        # Group by position
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for player in squad:
            by_position[player["position"]].append(player)

        # Sort by xP
        for pos in by_position:
            by_position[pos].sort(key=lambda p: p[xp_column], reverse=True)

        # Use shared formation enumeration logic (return just the players, ignore formation name)
        best_11, _, _ = self._enumerate_formations_for_players(by_position, xp_column)
        return best_11

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

    def validate_optimization_constraints(
        self,
        must_include_ids: Optional[Set[int]] = None,
        must_exclude_ids: Optional[Set[int]] = None,
        budget_limit: float = 100.0,
    ) -> Dict[str, Any]:
        """Validate optimization constraints for conflicts.

        Args:
            must_include_ids: Player IDs that must be included
            must_exclude_ids: Player IDs that must be excluded
            budget_limit: Budget limit in millions

        Returns:
            Validation result with conflicts detected
        """
        must_include_ids = must_include_ids or set()
        must_exclude_ids = must_exclude_ids or set()

        conflicts = must_include_ids.intersection(must_exclude_ids)

        return {
            "valid": len(conflicts) == 0,
            "conflicts": list(conflicts),
            "must_include_count": len(must_include_ids),
            "must_exclude_count": len(must_exclude_ids),
            "budget_limit": budget_limit,
        }

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


class InitialSquadOptimizationInput(BaseModel):
    """Data contract for initial squad optimization inputs."""

    budget: float = Field(ge=0, le=200, description="Budget in millions")
    formation: Tuple[int, int, int, int] = Field(
        description="Formation as (GKP, DEF, MID, FWD)"
    )
    iterations: int = Field(ge=100, le=50000, description="SA iterations")
    must_include_ids: List[int] = Field(default_factory=list)
    must_exclude_ids: List[int] = Field(default_factory=list)
    xp_column: str = Field(default="xP")

    @field_validator("formation")
    @classmethod
    def validate_formation(
        cls, v: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Validate formation sums to 15 players."""
        if len(v) != 4:
            raise ValueError("Formation must have 4 elements (GKP, DEF, MID, FWD)")
        if sum(v) != 15:
            raise ValueError(f"Formation must sum to 15 players, got {sum(v)}")
        if v[0] < 1:  # GKP
            raise ValueError("Formation must include at least 1 GKP")
        if v[1] < 3:  # DEF
            raise ValueError("Formation must include at least 3 DEF")
        if v[2] < 2:  # MID
            raise ValueError("Formation must include at least 2 MID")
        if v[3] < 1:  # FWD
            raise ValueError("Formation must include at least 1 FWD")
        return v

    @field_validator("must_include_ids", "must_exclude_ids")
    @classmethod
    def validate_no_overlap(cls, v: List[int], info) -> List[int]:
        """Ensure no duplicate IDs."""
        if len(v) != len(set(v)):
            raise ValueError("Duplicate player IDs found in constraints")
        return v
