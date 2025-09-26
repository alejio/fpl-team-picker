"""
FPL Optimization Module

Handles all optimization and transfer logic for FPL gameweek management including:
- Smart transfer optimization (0-3 transfers)
- Budget pool calculations
- Premium acquisition planning
- Formation and starting 11 selection
- Transfer scenario analysis
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from fpl_team_picker.config import config


def calculate_total_budget_pool(
    current_squad: pd.DataFrame,
    bank_balance: float,
    players_to_keep: Optional[Set[int]] = None,
) -> Dict:
    """
    Calculate total budget pool available for transfers

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
    sellable_players = current_squad[~current_squad["player_id"].isin(players_to_keep)]
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


def premium_acquisition_planner(
    current_squad: pd.DataFrame,
    all_players: pd.DataFrame,
    budget_pool_info: Dict,
    top_n: int = 3,
) -> List[Dict]:
    """
    Plan premium player acquisitions with funding scenarios

    Args:
        current_squad: Current squad DataFrame
        all_players: All available players DataFrame
        budget_pool_info: Budget pool information from calculate_total_budget_pool
        top_n: Number of top scenarios to return

    Returns:
        List of premium acquisition scenarios
    """
    if current_squad.empty or all_players.empty:
        return []

    scenarios = []

    # Find premium targets (players not in squad, price > bank balance)
    current_player_ids = set(current_squad["player_id"].tolist())
    available_players = all_players[~all_players["player_id"].isin(current_player_ids)]

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
        sort_column = "xP_5gw" if "xP_5gw" in sellable_players.columns else "xP"
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
                            (available_players["position"] == sell_player["position"])
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


def get_best_starting_11(
    squad_df: pd.DataFrame, xp_column: str = "xP"
) -> Tuple[List[Dict], str, float]:
    """
    Get best starting 11 from squad using specified XP column

    Args:
        squad_df: Squad DataFrame
        xp_column: Column to use for XP sorting ('xP' for current GW, 'xP_5gw' for strategic)

    Returns:
        Tuple of (best_11_list, formation_name, total_xp)
    """
    if len(squad_df) < 11:
        return [], "", 0

    # Filter out suspended, injured, or unavailable players from starting 11 selection
    available_squad = squad_df.copy()
    if "status" in available_squad.columns:
        unavailable_mask = available_squad["status"].isin(["i", "s", "u"])
        if unavailable_mask.any():
            unavailable_players = available_squad[unavailable_mask]
            print(
                f"🚫 Excluding {len(unavailable_players)} unavailable players from starting 11:"
            )
            for _, player in unavailable_players.iterrows():
                status_desc = {"i": "injured", "s": "suspended", "u": "unavailable"}[
                    player["status"]
                ]
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
    formations = [(1, 3, 5, 2), (1, 3, 4, 3), (1, 4, 5, 1), (1, 4, 4, 2), (1, 4, 3, 3)]
    formation_names = {
        "(1, 3, 5, 2)": "3-5-2",
        "(1, 3, 4, 3)": "3-4-3",
        "(1, 4, 5, 1)": "4-5-1",
        "(1, 4, 4, 2)": "4-4-2",
        "(1, 4, 3, 3)": "4-3-3",
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
            formation_xp = sum(p.get(sort_col, p.get("xP", 0)) for p in formation_11)

            if formation_xp > best_xp:
                best_xp = formation_xp
                best_11 = formation_11
                best_formation = formation_names.get(
                    str((gkp, def_count, mid, fwd)), f"{gkp}-{def_count}-{mid}-{fwd}"
                )

    return best_11, best_formation, best_xp


def get_bench_players(
    squad_df: pd.DataFrame, starting_11: List[Dict], xp_column: str = "xP"
) -> List[Dict]:
    """
    Get bench players (remaining 4 players) from squad ordered by XP

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


def optimize_team_with_transfers(
    current_squad: pd.DataFrame,
    team_data: Dict,
    players_with_xp: pd.DataFrame,
    must_include_ids: Set[int] = None,
    must_exclude_ids: Set[int] = None,
) -> Tuple[object, pd.DataFrame, Dict]:
    """
    Comprehensive strategic transfer optimization analyzing 0-3 transfers with detailed scenario breakdown

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

    # Update current squad with both 1-GW and 5-GW XP data for strategic decisions
    current_squad_with_xp = current_squad.merge(
        players_with_xp[["player_id", "xP", "xP_5gw", "fixture_outlook"]],
        on="player_id",
        how="left",
    )
    # Fill any missing XP with 0
    current_squad_with_xp["xP"] = current_squad_with_xp["xP"].fillna(0)
    current_squad_with_xp["xP_5gw"] = current_squad_with_xp["xP_5gw"].fillna(0)
    current_squad_with_xp["fixture_outlook"] = current_squad_with_xp[
        "fixture_outlook"
    ].fillna("🟡 Average")

    # Current squad and available budget
    current_player_ids = set(current_squad_with_xp["player_id"].tolist())
    available_budget = team_data["bank"]
    free_transfers = team_data.get("free_transfers", 1)

    # Calculate total budget pool for advanced transfer scenarios
    budget_pool_info = calculate_total_budget_pool(
        current_squad_with_xp, available_budget, must_include_ids
    )
    print(
        f"💰 Budget Analysis: Bank £{available_budget:.1f}m | Sellable Value £{budget_pool_info['sellable_value']:.1f}m | Total Pool £{budget_pool_info['total_budget']:.1f}m"
    )

    # Get all players with XP data and apply exclusion constraints
    all_players = players_with_xp[players_with_xp["xP"].notna()].copy()

    # Filter out suspended, injured, or unavailable players
    if "status" in all_players.columns:
        available_players_mask = ~all_players["status"].isin(["i", "s", "u"])
        all_players = all_players[available_players_mask]

        # Log any filtered players for debugging
        excluded_players = players_with_xp[
            players_with_xp["status"].isin(["i", "s", "u"])
            & players_with_xp["xP"].notna()
        ]
        if not excluded_players.empty:
            print(f"🚫 Filtered {len(excluded_players)} unavailable players:")
            for _, player in excluded_players.iterrows():
                status_desc = {"i": "injured", "s": "suspended", "u": "unavailable"}[
                    player["status"]
                ]
                print(f"   - {player.get('web_name', 'Unknown')} ({status_desc})")

    else:
        print("⚠️ No 'status' column found in players data")

    if must_exclude_ids:
        all_players = all_players[~all_players["player_id"].isin(must_exclude_ids)]

    # Get best starting 11 from current squad (use 5GW for strategic transfer planning)
    current_best_11, current_formation, current_xp = get_best_starting_11(
        current_squad_with_xp, "xP_5gw"
    )

    print(f"📊 Current squad: {current_xp:.2f} 5GW-xP | Formation: {current_formation}")

    # Comprehensive scenario analysis
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

    # 1-Transfer scenarios
    print("🔍 Analyzing 1-transfer scenarios...")

    # Prioritize suspended/injured players for transfer out, then worst performers
    squad_for_transfer = current_squad_with_xp.copy()

    # Check for unavailable players in current squad that should be prioritized for transfer
    if "status" in squad_for_transfer.columns:
        unavailable_in_squad = squad_for_transfer[
            squad_for_transfer["status"].isin(["i", "s", "u"])
        ]
        available_in_squad = squad_for_transfer[
            ~squad_for_transfer["status"].isin(["i", "s", "u"])
        ]

        if not unavailable_in_squad.empty:
            print(
                f"🚨 PRIORITY: {len(unavailable_in_squad)} unavailable players in current squad:"
            )
            for _, player in unavailable_in_squad.iterrows():
                status_desc = {"i": "injured", "s": "suspended", "u": "unavailable"}[
                    player["status"]
                ]
                print(
                    f"   - {player.get('web_name', 'Unknown')} ({status_desc}) - MUST TRANSFER OUT"
                )

        # Sort: unavailable players first (by status priority), then available players by worst XP
        unavailable_sorted = unavailable_in_squad.sort_values(
            ["status", "xP_5gw"], ascending=[True, True]
        )  # 's' comes before 'i' and 'u' alphabetically
        available_sorted = available_in_squad.sort_values("xP_5gw", ascending=True)

        # Combine: unavailable first, then worst performers
        squad_sorted = pd.concat(
            [unavailable_sorted, available_sorted], ignore_index=True
        )
    else:
        # Fallback to original logic if no status column
        squad_sorted = squad_for_transfer.sort_values("xP_5gw", ascending=True)

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
            # Get top 3 replacements
            top_replacements = position_replacements.nlargest(3, "xP_5gw")

            for _, replacement in top_replacements.iterrows():
                if replacement["xP_5gw"] > out_player.get("xP_5gw", 0):
                    # Calculate transfer cost
                    penalty = (
                        config.optimization.transfer_cost if free_transfers < 1 else 0
                    )

                    # Create new squad
                    new_squad = current_squad_with_xp.copy()
                    new_squad = new_squad[
                        new_squad["player_id"] != out_player["player_id"]
                    ]
                    new_squad = pd.concat(
                        [new_squad, replacement.to_frame().T], ignore_index=True
                    )

                    # Get best starting 11 from new squad
                    new_best_11, new_formation, new_xp = get_best_starting_11(
                        new_squad, "xP_5gw"
                    )
                    net_xp = new_xp - penalty
                    xp_gain = net_xp - current_xp

                    if xp_gain > 0.1:  # Only consider if meaningful gain
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
                                "changes": [{"out": out_player, "in": replacement}],
                            }
                        )

    # 2-Transfer scenarios
    print("🔍 Analyzing 2-transfer scenarios...")
    worst_2_players = squad_sorted.head(2)

    if len(worst_2_players) >= 2:
        out1, out2 = worst_2_players.iloc[0], worst_2_players.iloc[1]

        if (
            out1["player_id"] not in must_include_ids
            and out2["player_id"] not in must_include_ids
        ):
            # Find replacements for both positions
            pos1_replacements = all_players[
                (all_players["position"] == out1["position"])
                & (~all_players["player_id"].isin(current_player_ids))
            ].nlargest(2, "xP_5gw")

            pos2_replacements = all_players[
                (all_players["position"] == out2["position"])
                & (~all_players["player_id"].isin(current_player_ids))
            ].nlargest(2, "xP_5gw")

            for _, rep1 in pos1_replacements.iterrows():
                for _, rep2 in pos2_replacements.iterrows():
                    # Ensure we don't buy the same player twice
                    if rep1["player_id"] == rep2["player_id"]:
                        continue

                    total_cost = rep1["price"] + rep2["price"]
                    total_funds = available_budget + out1["price"] + out2["price"]

                    if total_cost <= total_funds:
                        penalty = (
                            config.optimization.transfer_cost
                            if free_transfers < 2
                            else 0
                            if free_transfers >= 2
                            else config.optimization.transfer_cost
                        )

                        # Create new squad
                        new_squad = current_squad_with_xp.copy()
                        new_squad = new_squad[
                            ~new_squad["player_id"].isin(
                                [out1["player_id"], out2["player_id"]]
                            )
                        ]
                        new_squad = pd.concat(
                            [new_squad, rep1.to_frame().T, rep2.to_frame().T],
                            ignore_index=True,
                        )

                        new_best_11, new_formation, new_xp = get_best_starting_11(
                            new_squad, "xP_5gw"
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

    # 3-Transfer scenarios
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

            # Find best replacement for each of the 3 positions
            replacements = []
            replacement_ids = (
                set()
            )  # Track selected replacement IDs to avoid duplicates
            total_out_value = out1["price"] + out2["price"] + out3["price"]
            remaining_budget = available_budget + total_out_value

            for out_player in [out1, out2, out3]:
                best_rep = all_players[
                    (all_players["position"] == out_player["position"])
                    & (
                        all_players["price"] <= remaining_budget / 3
                    )  # Rough budget allocation
                    & (~all_players["player_id"].isin(current_player_ids))
                    & (
                        ~all_players["player_id"].isin(replacement_ids)
                    )  # Avoid duplicate replacements
                ].nlargest(1, "xP_5gw")

                if not best_rep.empty:
                    replacement = best_rep.iloc[0]
                    replacements.append(replacement)
                    replacement_ids.add(replacement["player_id"])
                    remaining_budget -= replacement["price"]
                else:
                    # Can't find replacement, skip this scenario
                    break

            # Only create scenario if we found all 3 replacements
            if len(replacements) == 3:
                rep1, rep2, rep3 = replacements[0], replacements[1], replacements[2]
                total_replacement_cost = rep1["price"] + rep2["price"] + rep3["price"]

                if total_replacement_cost <= available_budget + total_out_value:
                    penalty = (
                        config.optimization.transfer_cost * 2
                        if free_transfers < 3
                        else 0
                        if free_transfers >= 3
                        else config.optimization.transfer_cost
                    )

                    # Calculate XP gain
                    out_xp = (
                        out1.get("xP_5gw", 0)
                        + out2.get("xP_5gw", 0)
                        + out3.get("xP_5gw", 0)
                    )
                    in_xp = rep1["xP_5gw"] + rep2["xP_5gw"] + rep3["xP_5gw"]
                    estimated_gain = in_xp - out_xp
                    net_xp = current_xp + estimated_gain - penalty
                    xp_gain = net_xp - current_xp

                    if xp_gain > 0.3:
                        # Create proper description with all player names
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
                                "formation": "3-4-3",  # Estimated
                                "xp_gain": xp_gain,
                                "squad": current_squad_with_xp,  # Placeholder
                            }
                        )

    # Premium acquisition scenarios
    print("🔍 Analyzing premium acquisition scenarios...")
    premium_scenarios = premium_acquisition_planner(
        current_squad_with_xp, all_players, budget_pool_info, 3
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
                    "formation": "3-4-3",  # Estimated
                    "xp_gain": premium_scenario["xp_gain"],
                    "squad": current_squad_with_xp,  # Placeholder
                }
            )

    # Sort all scenarios by net XP
    scenarios.sort(key=lambda x: x["net_xp"], reverse=True)

    # Get the best scenario
    best_scenario = scenarios[0]
    print(
        f"✅ Best strategy: {best_scenario['transfers']} transfers, {best_scenario['net_xp']:.2f} net XP"
    )

    # Create comprehensive display
    scenario_data = []
    for s in scenarios[:7]:  # Top 7 scenarios to match original functionality
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

    # Enhanced budget pool analysis
    max_single_acquisition = min(
        budget_pool_info["total_budget"], 15.0
    )  # Practical maximum
    sellable_count = (
        len(current_squad_with_xp) - len(must_include_ids) if must_include_ids else 15
    )

    strategic_summary = f"""
## 🏆 Strategic 5-GW Decision: {best_scenario["transfers"]} Transfer(s) Optimal

**Recommended Strategy:** {best_scenario["description"]}

**Expected Net 5-GW XP:** {best_scenario["net_xp"]:.2f} | **Formation:** {best_scenario["formation"]}

*Decisions based on 5-gameweek horizon with temporal weighting and fixture analysis*

### 💰 Budget Pool Analysis:
- **Bank:** £{available_budget:.1f}m | **Sellable Value:** £{budget_pool_info["sellable_value"]:.1f}m | **Total Pool:** £{budget_pool_info["total_budget"]:.1f}m
- **Max Single Acquisition:** £{max_single_acquisition:.1f}m | **Budget Utilization:** {((budget_pool_info["total_budget"] - available_budget) / budget_pool_info["total_budget"] * 100) if budget_pool_info["total_budget"] > 0 else 0:.0f}% - {sellable_count} sellable players | **Min squad cost:** £{46.0:.1f}m

### 🔄 Enhanced Transfer Analysis:
- **{len(scenarios)} total scenarios analyzed** ({len([s for s in scenarios if s["type"] == "premium_acquisition"])} premium acquisition scenarios) - Premium acquisition planner targets high-value players with smart funding strategies - Enhanced budget pool calculation enables complex multi-player funding scenarios

**All Scenarios (sorted by 5-GW Net XP):**
"""

    display_component = mo.vstack(
        [
            mo.md(strategic_summary),
            mo.ui.table(
                scenarios_df, page_size=config.visualization.scenario_page_size
            ),
            mo.md("---"),
            mo.md("### 🏆 Optimal Starting 11 (Strategic):"),
            mo.md("*Shows both 1-GW and 5-GW XP with fixture outlook*"),
        ]
    )

    optimal_squad = best_scenario.get("squad", current_squad_with_xp)

    return display_component, optimal_squad, best_scenario


def select_captain(starting_11: List[Dict], mo_ref=None) -> object:
    """
    Select captain from starting 11 based on current gameweek XP and risk factors

    Args:
        starting_11: List of starting 11 player dictionaries
        mo_ref: Marimo reference for UI components

    Returns:
        Marimo UI component with captain recommendations
    """
    if not starting_11:
        return (
            mo_ref.md("⚠️ **No starting 11 available for captain selection**")
            if mo_ref
            else None
        )

    # Sort by 1-GW XP (immediate gameweek captain choice)
    captain_candidates = sorted(starting_11, key=lambda p: p.get("xP", 0), reverse=True)

    captain_analysis = []

    for i, player in enumerate(captain_candidates[:5]):  # Top 5 candidates
        xp_1gw = player.get("xP", 0)
        fixture_outlook = player.get("fixture_outlook", "🟡 Average")

        # Enhanced risk assessment for immediate gameweek
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
        expected_mins = player.get("expected_minutes", 90)
        if expected_mins < 60:
            risk_factors.append("rotation risk")
            risk_level = "🟡 Medium" if risk_level == "🟢 Low" else risk_level

        # Combine risk description
        if risk_factors:
            risk_desc = f"{risk_level} ({', '.join(risk_factors)})"
        else:
            risk_desc = risk_level

        # Calculate captaincy potential (XP * 2 for double points)
        captain_potential = xp_1gw * 2

        captain_analysis.append(
            {
                "Rank": i + 1,
                "Player": player["web_name"],
                "Position": player["position"],
                "Price": f"£{player['price']:.1f}m",
                "GW XP": f"{xp_1gw:.2f}",
                "Captain Pts": f"{captain_potential:.1f}",
                "Minutes": f"{expected_mins:.0f}",
                "Fixture": fixture_outlook.replace("🟢 Easy", "🟢")
                .replace("🟡 Average", "🟡")
                .replace("🔴 Hard", "🔴"),
                "Risk": risk_desc,
                "Recommendation": "👑 Captain" if i == 0 else "(VC)" if i == 1 else "",
            }
        )

    captain_df = pd.DataFrame(captain_analysis)

    captain_pick = captain_candidates[0]
    vice_pick = (
        captain_candidates[1] if len(captain_candidates) > 1 else captain_candidates[0]
    )

    # Calculate captaincy upside
    captain_upside = captain_pick.get("xP", 0) * 2
    vice_upside = vice_pick.get("xP", 0) * 2
    differential = captain_upside - vice_upside

    summary = f"""
### 👑 Captain Selection (Current Gameweek Focus)

**Recommended Captain:** {captain_pick["web_name"]} ({captain_pick["position"]})
- **Expected Points:** {captain_pick.get("xP", 0):.2f} → **{captain_upside:.1f} as captain**
- **Minutes:** {captain_pick.get("expected_minutes", 90):.0f}' expected
- **Fixture:** {captain_pick.get("fixture_outlook", "Unknown")}

**Vice Captain:** {vice_pick["web_name"]} ({vice_pick["position"]})
- **Expected Points:** {vice_pick.get("xP", 0):.2f} → **{vice_upside:.1f} as captain**

**Captain Advantage:** +{differential:.1f} points over vice captain

**Current Gameweek Analysis:**
*Captain selection optimized for immediate fixture only - can be changed weekly*
"""

    return (
        mo_ref.vstack([mo_ref.md(summary), mo_ref.ui.table(captain_df, page_size=5)])
        if mo_ref
        else captain_df
    )
