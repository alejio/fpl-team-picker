"""Detailed gameweek-by-gameweek analysis for 2024-25 season (GW1-9)."""

import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "fpl-dataset-builder"))

from client import FPLDataClient

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 50)


def analyze_gameweek_by_gameweek():
    """Detailed GW-by-GW analysis."""

    client = FPLDataClient()

    print("=" * 100)
    print("GAMEWEEK-BY-GAMEWEEK ANALYSIS (2024-25 Season)")
    print("=" * 100)

    # Get data
    manager_data = client.get_my_manager_data()
    picks_history = client.get_my_picks_history()
    players = client.get_current_players()

    if picks_history.empty:
        print("❌ No picks history available")
        return

    manager_name = (
        manager_data["player_first_name"].iloc[0]
        + " "
        + manager_data["player_last_name"].iloc[0]
    )
    print(f"\nManager: {manager_name}")
    print(f"Overall Rank: {manager_data['summary_overall_rank'].iloc[0]:,}")
    print(f"Total Points: {manager_data['summary_overall_points'].iloc[0]}")

    # Get player performance data for each GW
    print("\n" + "=" * 100)
    print("GAMEWEEK BREAKDOWN")
    print("=" * 100)

    gameweeks = sorted(picks_history["event"].unique())

    gw_summary = []
    prev_squad = None

    for gw in gameweeks:
        print(f"\n{'─' * 100}")
        print(f"GAMEWEEK {gw}")
        print(f"{'─' * 100}")

        # Get picks for this GW
        gw_picks = picks_history[picks_history["event"] == gw].copy()

        # Merge with player data
        gw_picks = gw_picks.merge(
            players[["player_id", "web_name", "position", "price_gbp"]],
            on="player_id",
            how="left",
            suffixes=("", "_player"),
        )

        # Get actual performance for this GW
        gw_performance = client.get_gameweek_performance(gw)

        # Merge performance data
        gw_picks = gw_picks.merge(
            gw_performance, on="player_id", how="left", suffixes=("_pick", "")
        )

        # Fill NaN values for players who didn't play
        gw_picks["total_points"] = gw_picks["total_points"].fillna(0)
        gw_picks["minutes"] = gw_picks["minutes"].fillna(0)

        # Calculate points with multipliers
        gw_picks["actual_points"] = gw_picks["total_points"] * gw_picks["multiplier"]

        # Define position order for sorting
        position_order = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
        gw_picks["position_order"] = gw_picks["position"].map(position_order)

        # Identify transfers
        current_squad = set(gw_picks["player_id"].values)
        if prev_squad is not None:
            players_in = current_squad - prev_squad
            players_out = prev_squad - current_squad
            transfers_made = len(players_in)
        else:
            players_in = set()
            players_out = set()
            transfers_made = 0

        # Squad composition
        print("\n📊 SQUAD:")

        # Starting XI
        starting_xi = gw_picks[gw_picks["multiplier"] > 0].sort_values("position_order")
        print("\n  Starting XI:")
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            pos_players = starting_xi[starting_xi["position"] == pos]
            if not pos_players.empty:
                print(f"\n  {pos}:")
                for _, player in pos_players.iterrows():
                    captain_mark = (
                        " (C)"
                        if player["is_captain"]
                        else " (VC)"
                        if player["is_vice_captain"]
                        else ""
                    )
                    pts = player.get("actual_points", 0)
                    mins = player.get("minutes", 0)
                    print(
                        f"    • {player['web_name']:<20} {captain_mark:<5} → {pts:.0f} pts ({mins} mins)"
                    )

        # Bench
        bench = gw_picks[gw_picks["multiplier"] == 0].sort_values("position_order")
        print("\n  Bench:")
        for _, player in bench.iterrows():
            pts = player.get("total_points", 0)
            mins = player.get("minutes", 0)
            print(f"    • {player['web_name']:<20} → {pts:.0f} pts ({mins} mins)")

        # Transfers
        if transfers_made > 0:
            print(f"\n🔄 TRANSFERS ({transfers_made}):")
            for player_id in players_out:
                player_name = players[players["player_id"] == player_id][
                    "web_name"
                ].values
                if len(player_name) > 0:
                    print(f"    OUT: {player_name[0]}")
            for player_id in players_in:
                player_name = players[players["player_id"] == player_id][
                    "web_name"
                ].values
                if len(player_name) > 0:
                    print(f"    IN:  {player_name[0]}")

        # Points summary
        total_gw_points = gw_picks["actual_points"].sum()
        captain = starting_xi[starting_xi["is_captain"]]
        captain_pts = captain["actual_points"].iloc[0] if not captain.empty else 0
        bench_pts = bench["total_points"].sum()

        print("\n📈 POINTS:")
        print(f"    Total GW Points: {total_gw_points:.0f}")
        print(f"    Captain Points: {captain_pts:.0f}")
        print(f"    Points on Bench: {bench_pts:.0f}")
        if transfers_made > 1:
            transfer_cost = (transfers_made - 1) * 4
            print(f"    Transfer Cost: -{transfer_cost}")
            total_gw_points -= transfer_cost
            print(f"    NET Points: {total_gw_points:.0f}")

        gw_summary.append(
            {
                "gameweek": gw,
                "points": total_gw_points,
                "transfers": transfers_made,
                "captain_points": captain_pts,
                "bench_points": bench_pts,
            }
        )

        prev_squad = current_squad

    # Overall summary
    print("\n" + "=" * 100)
    print("SEASON SUMMARY (GW1-GW9)")
    print("=" * 100)

    summary_df = pd.DataFrame(gw_summary)

    print(f"\nTotal Points: {summary_df['points'].sum():.0f}")
    print(f"Average per GW: {summary_df['points'].mean():.1f}")
    print(
        f"Best GW: GW{summary_df.loc[summary_df['points'].idxmax(), 'gameweek']:.0f} ({summary_df['points'].max():.0f} pts)"
    )
    print(
        f"Worst GW: GW{summary_df.loc[summary_df['points'].idxmin(), 'gameweek']:.0f} ({summary_df['points'].min():.0f} pts)"
    )
    print(f"\nTotal Transfers: {summary_df['transfers'].sum():.0f}")
    print(
        f"Transfer Cost: -{(summary_df['transfers'].sum() - 9) * 4:.0f} pts"
    )  # 1 free per GW
    print(f"\nTotal Bench Points: {summary_df['bench_points'].sum():.0f}")
    print(f"Average Bench pts/GW: {summary_df['bench_points'].mean():.1f}")

    # Trend analysis
    print("\n📊 POINTS TREND:")
    for _, row in summary_df.iterrows():
        bar_length = int(row["points"] / 5)
        bar = "█" * bar_length
        print(f"  GW{row['gameweek']:.0f}: {bar} {row['points']:.0f} pts")

    # Target comparison
    print("\n🎯 PERFORMANCE vs TARGETS:")
    current_avg = summary_df["points"].mean()
    target_avg = 74  # Winner's average
    print(f"  Current pace: {current_avg:.1f} pts/GW")
    print(f"  Winner's pace: {target_avg:.1f} pts/GW")
    print(f"  Gap: {target_avg - current_avg:.1f} pts/GW")

    projected_total = current_avg * 38
    winner_total = target_avg * 38
    print(f"\n  Projected season total: {projected_total:.0f} pts")
    print(f"  Winner's total (2024-25): {winner_total:.0f} pts")
    print(f"  Points behind pace: {winner_total - projected_total:.0f} pts")


if __name__ == "__main__":
    analyze_gameweek_by_gameweek()
