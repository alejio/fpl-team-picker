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
                print(
                    f"🚫 Excluding {len(unavailable_players)} unavailable players from starting 11:"
                )
                for _, player in unavailable_players.iterrows():
                    status_desc = {
                        "i": "injured",
                        "s": "suspended",
                        "u": "unavailable",
                    }[player["status"]]
                    print(f"   - {player.get('web_name', 'Unknown')} ({status_desc})")

                available_squad = available_squad[~unavailable_mask]

        if len(available_squad) < 11:
            print(
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

        # Try formations and pick best
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
                formation_xp = sum(
                    p.get(sort_col, p.get("xP", 0)) for p in formation_11
                )

                if formation_xp > best_xp:
                    best_xp = formation_xp
                    best_11 = formation_11
                    best_formation = formation_names.get(
                        str((gkp, def_count, mid, fwd)),
                        f"{gkp}-{def_count}-{mid}-{fwd}",
                    )

        return best_11, best_formation, best_xp

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

        # Sort by XP (immediate gameweek captain choice)
        captain_candidates = sorted(
            players_list, key=lambda p: p.get(xp_column, 0), reverse=True
        )

        if not captain_candidates:
            raise ValueError("No valid captain candidates found")

        # Analyze top N candidates
        top_candidates = []

        for i, player in enumerate(captain_candidates[:top_n]):
            xp_value = player.get(xp_column, 0)
            fixture_outlook = player.get("fixture_outlook", "🟡 Average")

            # Enhanced risk assessment
            risk_factors = []
            risk_level = "🟢 Low"

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

            top_candidates.append(
                {
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
            )

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
    ) -> Tuple[object, pd.DataFrame, Dict]:
        """Comprehensive strategic transfer optimization analyzing 0-3 transfers.

        This is the core transfer optimization algorithm that analyzes multiple
        transfer scenarios and recommends the optimal strategy.

        Args:
            current_squad: Current squad DataFrame
            team_data: Team data dictionary with bank balance etc
            players_with_xp: All players with XP calculations
            must_include_ids: Set of player IDs that must be included
            must_exclude_ids: Set of player IDs that must be excluded

        Returns:
            Tuple of (marimo_component, optimal_squad_df, optimization_results)
        """
        import marimo as mo

        if len(current_squad) == 0 or players_with_xp.empty or not team_data:
            return mo.md("Load team and calculate XP first"), pd.DataFrame(), {}

        print("🧠 Strategic Optimization: Analyzing 0-3 transfer scenarios...")

        # Process constraints
        must_include_ids = must_include_ids or set()
        must_exclude_ids = must_exclude_ids or set()

        if must_include_ids:
            print(f"🎯 Must include {len(must_include_ids)} players")
        if must_exclude_ids:
            print(f"🚫 Must exclude {len(must_exclude_ids)} players")

        # Get optimization column based on configuration
        xp_column = self.get_optimization_xp_column()
        horizon_label = "1-GW" if xp_column == "xP" else "5-GW"

        # Update current squad with both 1-GW and 5-GW XP data
        merge_columns = ["player_id", "xP", "xP_5gw", "fixture_outlook"]
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
        current_player_ids = set(current_squad_with_xp["player_id"].tolist())
        available_budget = team_data["bank"]
        free_transfers = team_data.get("free_transfers", 1)

        # Calculate total budget pool
        budget_pool_info = self.calculate_budget_pool(
            current_squad_with_xp, available_budget, must_include_ids
        )
        print(
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
                print(f"🚫 Filtered {len(excluded_players)} unavailable players")

        if must_exclude_ids:
            all_players = all_players[~all_players["player_id"].isin(must_exclude_ids)]

        # Get best starting 11 from current squad
        current_best_11, current_formation, current_xp = self.find_optimal_starting_11(
            current_squad_with_xp, xp_column
        )

        print(
            f"📊 Current squad: {current_xp:.2f} {horizon_label}-xP | Formation: {current_formation}"
        )

        # Analyze transfer scenarios
        scenarios = self._analyze_transfer_scenarios(
            current_squad_with_xp,
            all_players,
            current_player_ids,
            available_budget,
            free_transfers,
            budget_pool_info,
            must_include_ids,
            xp_column,
            current_xp,
            current_formation,
        )

        # Get best scenario
        best_scenario = scenarios[0]
        print(
            f"✅ Best strategy: {best_scenario['transfers']} transfers, {best_scenario['net_xp']:.2f} net XP"
        )

        # Create display component
        display_component = self._create_optimization_display(
            scenarios,
            best_scenario,
            budget_pool_info,
            available_budget,
            horizon_label,
            xp_column,
            mo,
        )

        optimal_squad = best_scenario.get("squad", current_squad_with_xp)

        return display_component, optimal_squad, best_scenario

    def _analyze_transfer_scenarios(
        self,
        current_squad_with_xp: pd.DataFrame,
        all_players: pd.DataFrame,
        current_player_ids: Set[int],
        available_budget: float,
        free_transfers: int,
        budget_pool_info: Dict,
        must_include_ids: Set[int],
        xp_column: str,
        current_xp: float,
        current_formation: str,
    ) -> List[Dict]:
        """Analyze 0-3 transfer scenarios and return sorted list by net XP."""
        scenarios = []

        # Scenario 0: No transfers (baseline)
        scenarios.append(
            {
                "id": 0,
                "transfers": 0,
                "type": "standard",
                "description": "Keep current squad",
                "penalty": 0,
                "net_xp": current_xp,
                "formation": current_formation,
                "xp_gain": 0.0,
                "squad": current_squad_with_xp.copy(),
            }
        )

        # Prepare squad for transfer analysis
        squad_sorted = self._prepare_squad_for_transfers(
            current_squad_with_xp, xp_column
        )

        # 1-Transfer scenarios
        self._analyze_1_transfer_scenarios(
            scenarios,
            squad_sorted,
            must_include_ids,
            all_players,
            current_player_ids,
            available_budget,
            free_transfers,
            current_squad_with_xp,
            xp_column,
            current_xp,
        )

        # 2-Transfer scenarios
        self._analyze_2_transfer_scenarios(
            scenarios,
            squad_sorted,
            must_include_ids,
            all_players,
            current_player_ids,
            available_budget,
            free_transfers,
            current_squad_with_xp,
            xp_column,
            current_xp,
        )

        # 3-Transfer scenarios
        self._analyze_3_transfer_scenarios(
            scenarios,
            squad_sorted,
            must_include_ids,
            all_players,
            current_player_ids,
            available_budget,
            free_transfers,
            current_squad_with_xp,
            xp_column,
            current_xp,
        )

        # Premium acquisition scenarios
        self._analyze_premium_scenarios(
            scenarios,
            current_squad_with_xp,
            all_players,
            budget_pool_info,
            free_transfers,
            current_xp,
        )

        # Sort by net XP
        scenarios.sort(key=lambda x: x["net_xp"], reverse=True)
        return scenarios

    def _prepare_squad_for_transfers(
        self, squad: pd.DataFrame, xp_column: str
    ) -> pd.DataFrame:
        """Prepare squad for transfer analysis by sorting unavailable players first."""
        if "status" in squad.columns:
            unavailable_in_squad = squad[squad["status"].isin(["i", "s", "u"])]
            available_in_squad = squad[~squad["status"].isin(["i", "s", "u"])]

            if not unavailable_in_squad.empty:
                print(
                    f"🚨 PRIORITY: {len(unavailable_in_squad)} unavailable players in squad"
                )

            # Sort: unavailable first, then worst XP
            unavailable_sorted = unavailable_in_squad.sort_values(
                ["status", xp_column], ascending=[True, True]
            )
            available_sorted = available_in_squad.sort_values(xp_column, ascending=True)

            return pd.concat([unavailable_sorted, available_sorted], ignore_index=True)
        else:
            return squad.sort_values(xp_column, ascending=True)

    def _analyze_1_transfer_scenarios(
        self,
        scenarios: List[Dict],
        squad_sorted: pd.DataFrame,
        must_include_ids: Set[int],
        all_players: pd.DataFrame,
        current_player_ids: Set[int],
        available_budget: float,
        free_transfers: int,
        current_squad: pd.DataFrame,
        xp_column: str,
        current_xp: float,
    ) -> None:
        """Analyze 1-transfer scenarios."""
        print("🔍 Analyzing 1-transfer scenarios...")

        for idx, out_player in squad_sorted.iterrows():
            if out_player["player_id"] in must_include_ids:
                continue

            # Find best replacements for this position
            position_replacements = all_players[
                (all_players["position"] == out_player["position"])
                & (all_players["price"] <= available_budget + out_player["price"])
                & (~all_players["player_id"].isin(current_player_ids))
            ]

            if not position_replacements.empty:
                top_replacements = position_replacements.nlargest(3, xp_column)

                for _, replacement in top_replacements.iterrows():
                    if replacement[xp_column] > out_player.get(xp_column, 0):
                        penalty = (
                            config.optimization.transfer_cost
                            if free_transfers < 1
                            else 0
                        )

                        # Create new squad
                        new_squad = current_squad.copy()
                        new_squad = new_squad[
                            new_squad["player_id"] != out_player["player_id"]
                        ]
                        new_squad = pd.concat(
                            [new_squad, replacement.to_frame().T], ignore_index=True
                        )

                        # Get best starting 11 from new squad
                        new_best_11, new_formation, new_xp = (
                            self.find_optimal_starting_11(new_squad, xp_column)
                        )
                        net_xp = new_xp - penalty
                        xp_gain = net_xp - current_xp

                        if xp_gain > 0.1:
                            scenarios.append(
                                {
                                    "id": len(scenarios),
                                    "transfers": 1,
                                    "type": "standard",
                                    "description": f"OUT: {out_player['web_name']} → IN: {replacement['web_name']}",
                                    "penalty": penalty,
                                    "net_xp": net_xp,
                                    "formation": new_formation,
                                    "xp_gain": xp_gain,
                                    "squad": new_squad,
                                }
                            )

    def _analyze_2_transfer_scenarios(
        self,
        scenarios: List[Dict],
        squad_sorted: pd.DataFrame,
        must_include_ids: Set[int],
        all_players: pd.DataFrame,
        current_player_ids: Set[int],
        available_budget: float,
        free_transfers: int,
        current_squad: pd.DataFrame,
        xp_column: str,
        current_xp: float,
    ) -> None:
        """Analyze 2-transfer scenarios."""
        print("🔍 Analyzing 2-transfer scenarios...")
        worst_2_players = squad_sorted.head(2)

        if len(worst_2_players) >= 2:
            out1, out2 = worst_2_players.iloc[0], worst_2_players.iloc[1]

            if (
                out1["player_id"] not in must_include_ids
                and out2["player_id"] not in must_include_ids
            ):
                pos1_replacements = all_players[
                    (all_players["position"] == out1["position"])
                    & (~all_players["player_id"].isin(current_player_ids))
                ].nlargest(2, xp_column)

                pos2_replacements = all_players[
                    (all_players["position"] == out2["position"])
                    & (~all_players["player_id"].isin(current_player_ids))
                ].nlargest(2, xp_column)

                for _, rep1 in pos1_replacements.iterrows():
                    for _, rep2 in pos2_replacements.iterrows():
                        if rep1["player_id"] == rep2["player_id"]:
                            continue

                        total_cost = rep1["price"] + rep2["price"]
                        total_funds = available_budget + out1["price"] + out2["price"]

                        if total_cost <= total_funds:
                            penalty = (
                                config.optimization.transfer_cost
                                if free_transfers < 2
                                else 0
                            )

                            new_squad = current_squad.copy()
                            new_squad = new_squad[
                                ~new_squad["player_id"].isin(
                                    [out1["player_id"], out2["player_id"]]
                                )
                            ]
                            new_squad = pd.concat(
                                [new_squad, rep1.to_frame().T, rep2.to_frame().T],
                                ignore_index=True,
                            )

                            new_best_11, new_formation, new_xp = (
                                self.find_optimal_starting_11(new_squad, xp_column)
                            )
                            net_xp = new_xp - penalty
                            xp_gain = net_xp - current_xp

                            if xp_gain > 0.2:
                                scenarios.append(
                                    {
                                        "id": len(scenarios),
                                        "transfers": 2,
                                        "type": "standard",
                                        "description": f"OUT: {out1['web_name']}, {out2['web_name']} → IN: {rep1['web_name']}, {rep2['web_name']}",
                                        "penalty": penalty,
                                        "net_xp": net_xp,
                                        "formation": new_formation,
                                        "xp_gain": xp_gain,
                                        "squad": new_squad,
                                    }
                                )

    def _analyze_3_transfer_scenarios(
        self,
        scenarios: List[Dict],
        squad_sorted: pd.DataFrame,
        must_include_ids: Set[int],
        all_players: pd.DataFrame,
        current_player_ids: Set[int],
        available_budget: float,
        free_transfers: int,
        current_squad: pd.DataFrame,
        xp_column: str,
        current_xp: float,
    ) -> None:
        """Analyze 3-transfer scenarios."""
        print("🔍 Analyzing 3-transfer scenarios...")
        worst_3_players = squad_sorted.head(3)

        if len(worst_3_players) >= 3:
            out_players = worst_3_players
            valid_outs = [
                p
                for _, p in out_players.iterrows()
                if p["player_id"] not in must_include_ids
            ]

            if len(valid_outs) >= 3:
                out1, out2, out3 = valid_outs[0], valid_outs[1], valid_outs[2]

                replacements = []
                replacement_ids = set()
                total_out_value = out1["price"] + out2["price"] + out3["price"]
                remaining_budget = available_budget + total_out_value

                for out_player in [out1, out2, out3]:
                    best_rep = all_players[
                        (all_players["position"] == out_player["position"])
                        & (all_players["price"] <= remaining_budget / 3)
                        & (~all_players["player_id"].isin(current_player_ids))
                        & (~all_players["player_id"].isin(replacement_ids))
                    ].nlargest(1, xp_column)

                    if not best_rep.empty:
                        replacement = best_rep.iloc[0]
                        replacements.append(replacement)
                        replacement_ids.add(replacement["player_id"])
                        remaining_budget -= replacement["price"]
                    else:
                        break

                if len(replacements) == 3:
                    rep1, rep2, rep3 = replacements[0], replacements[1], replacements[2]
                    total_replacement_cost = (
                        rep1["price"] + rep2["price"] + rep3["price"]
                    )

                    if total_replacement_cost <= available_budget + total_out_value:
                        penalty = (
                            config.optimization.transfer_cost * 2
                            if free_transfers < 3
                            else 0
                        )

                        out_xp = (
                            out1.get(xp_column, 0)
                            + out2.get(xp_column, 0)
                            + out3.get(xp_column, 0)
                        )
                        in_xp = rep1[xp_column] + rep2[xp_column] + rep3[xp_column]
                        estimated_gain = in_xp - out_xp
                        net_xp = current_xp + estimated_gain - penalty
                        xp_gain = net_xp - current_xp

                        if xp_gain > 0.3:
                            out_names = f"{out1['web_name']}, {out2['web_name']}, {out3['web_name']}"
                            in_names = f"{rep1['web_name']}, {rep2['web_name']}, {rep3['web_name']}"

                            scenarios.append(
                                {
                                    "id": len(scenarios),
                                    "transfers": 3,
                                    "type": "standard",
                                    "description": f"OUT: {out_names} → IN: {in_names}",
                                    "penalty": penalty,
                                    "net_xp": net_xp,
                                    "formation": "3-4-3",
                                    "xp_gain": xp_gain,
                                    "squad": current_squad,
                                }
                            )

    def _analyze_premium_scenarios(
        self,
        scenarios: List[Dict],
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        budget_pool_info: Dict,
        free_transfers: int,
        current_xp: float,
    ) -> None:
        """Analyze premium player acquisition scenarios."""
        print("🔍 Analyzing premium acquisition scenarios...")
        premium_scenarios = self.plan_premium_acquisition(
            current_squad, all_players, budget_pool_info, 3
        )

        for premium_scenario in premium_scenarios:
            if premium_scenario["xp_gain"] > 0.5:
                penalty = (
                    premium_scenario["transfers"] * 4
                    if premium_scenario["transfers"] > free_transfers
                    else 0
                )
                net_xp = current_xp + premium_scenario["xp_gain"] - penalty

                scenarios.append(
                    {
                        "id": len(scenarios),
                        "transfers": premium_scenario["transfers"],
                        "type": "premium_acquisition",
                        "description": premium_scenario["description"],
                        "penalty": penalty,
                        "net_xp": net_xp,
                        "formation": "3-4-3",
                        "xp_gain": premium_scenario["xp_gain"],
                        "squad": current_squad,
                    }
                )

    def _create_optimization_display(
        self,
        scenarios: List[Dict],
        best_scenario: Dict,
        budget_pool_info: Dict,
        available_budget: float,
        horizon_label: str,
        xp_column: str,
        mo_ref,
    ) -> object:
        """Create marimo display component for optimization results."""
        scenario_data = []
        for s in scenarios[:7]:  # Top 7 scenarios
            scenario_data.append(
                {
                    "Transfers": s["transfers"],
                    "Type": s["type"],
                    "Description": s["description"],
                    "Penalty": -s["penalty"] if s["penalty"] > 0 else 0,
                    "Net XP": round(s["net_xp"], 2),
                    "Formation": s["formation"],
                    "XP Gain": round(s["xp_gain"], 2),
                }
            )

        scenarios_df = pd.DataFrame(scenario_data)

        max_single_acquisition = min(budget_pool_info["total_budget"], 15.0)

        strategic_summary = f"""
## 🏆 Strategic {horizon_label} Decision: {best_scenario["transfers"]} Transfer(s) Optimal

**Recommended Strategy:** {best_scenario["description"]}

**Expected Net {horizon_label} XP:** {best_scenario["net_xp"]:.2f} | **Formation:** {best_scenario["formation"]}

*Decisions based on {horizon_label.lower()} horizon*

### 💰 Budget Pool Analysis:
- **Bank:** £{available_budget:.1f}m | **Sellable Value:** £{budget_pool_info["sellable_value"]:.1f}m | **Total Pool:** £{budget_pool_info["total_budget"]:.1f}m
- **Max Single Acquisition:** £{max_single_acquisition:.1f}m

### 🔄 Transfer Analysis:
- **{len(scenarios)} total scenarios analyzed**

**All Scenarios (sorted by {horizon_label} Net XP):**
"""

        return mo_ref.vstack(
            [
                mo_ref.md(strategic_summary),
                mo_ref.ui.table(
                    scenarios_df, page_size=config.visualization.scenario_page_size
                ),
                mo_ref.md("---"),
                mo_ref.md("### 🏆 Optimal Starting 11 (Strategic):"),
            ]
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

            # Penalty for constraint violations
            team_counts = {}
            for player in team:
                team_name = player["team"]
                team_counts[team_name] = team_counts.get(team_name, 0) + 1

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

            # Check 3-per-team
            team_counts = {}
            for player in team:
                team_name = player["team"]
                team_counts[team_name] = team_counts.get(team_name, 0) + 1
                if team_counts[team_name] > 3:
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
        """Get best starting 11 from 15-player squad."""
        if len(squad) != 15:
            return []

        # Group by position
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for player in squad:
            by_position[player["position"]].append(player)

        # Sort by xP
        for pos in by_position:
            by_position[pos].sort(key=lambda p: p[xp_column], reverse=True)

        # Try valid formations
        valid_formations = [
            (1, 3, 5, 2),
            (1, 3, 4, 3),
            (1, 4, 5, 1),
            (1, 4, 4, 2),
            (1, 4, 3, 3),
            (1, 5, 4, 1),
            (1, 5, 3, 2),
            (1, 5, 2, 3),
        ]

        best_11 = []
        best_xp = 0

        for gkp, def_count, mid, fwd in valid_formations:
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

                formation_xp = sum(p[xp_column] for p in formation_11)

                if formation_xp > best_xp:
                    best_xp = formation_xp
                    best_11 = formation_11

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
