"""Analyze top FPL player transfer strategy.

This script examines historical transfer decisions, timing, and performance
to understand what makes a successful FPL manager.
"""

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


def analyze_transfer_strategy():
    """Analyze the configured manager's transfer strategy."""

    client = FPLDataClient()

    print("=" * 80)
    print("FPL TRANSFER STRATEGY ANALYSIS")
    print("=" * 80)

    # 1. Get manager information
    print("\n📊 MANAGER INFORMATION")
    print("-" * 80)
    try:
        manager_data = client.get_my_manager_data()
        if not manager_data.empty:
            print(manager_data.T)
            manager_name = (
                manager_data["player_first_name"].iloc[0]
                + " "
                + manager_data["player_last_name"].iloc[0]
            )
            team_name = manager_data.get("entry_name", [None]).iloc[0]
            overall_rank = manager_data.get("summary_overall_rank", [None]).iloc[0]
            total_points = manager_data.get("summary_overall_points", [None]).iloc[0]
            print(f"\nManager: {manager_name}")
            if team_name:
                print(f"Team: {team_name}")
            if overall_rank:
                print(f"Overall Rank: {overall_rank:,}")
            if total_points:
                print(f"Total Points: {total_points}")
        else:
            print(
                "❌ No manager data found. Please configure a manager ID in the dataset builder."
            )
            return
    except Exception as e:
        print(f"❌ Error fetching manager data: {e}")
        return

    # 2. Get current gameweek info
    events = client.get_raw_events_bootstrap()
    current_gw = (
        events[events["is_current"]]["event_id"].iloc[0]
        if not events[events["is_current"]].empty
        else None
    )

    if current_gw:
        print(f"\nCurrent Gameweek: {current_gw}")
    else:
        # Fallback to current_event from manager data
        current_gw = manager_data["current_event"].iloc[0]
        print(f"\nCurrent Gameweek: {current_gw}")

    # 3. Get historical picks
    print("\n📋 TRANSFER HISTORY ANALYSIS")
    print("-" * 80)
    try:
        picks_history = client.get_my_picks_history()

        if picks_history.empty:
            print("❌ No historical picks data available.")
            return

        # Analyze transfers by gameweek
        picks_by_gw = (
            picks_history.groupby("event")
            .agg({"player_id": "count", "position": "count"})
            .rename(columns={"player_id": "squad_size", "position": "positions"})
        )

        print(f"\nTotal Gameweeks: {len(picks_by_gw)}")
        print(
            f"Gameweeks tracked: GW{picks_by_gw.index.min()} - GW{picks_by_gw.index.max()}"
        )

        # 4. Analyze squad changes
        print("\n🔄 SQUAD ROTATION & TRANSFERS")
        print("-" * 80)

        # Track players in/out each gameweek
        transfer_analysis = []
        prev_squad = None

        for gw in sorted(picks_history["event"].unique()):
            gw_picks = picks_history[picks_history["event"] == gw]
            current_squad = set(gw_picks["player_id"].values)

            if prev_squad is not None:
                players_in = current_squad - prev_squad
                players_out = prev_squad - current_squad

                if players_in or players_out:
                    transfer_analysis.append(
                        {
                            "gameweek": gw,
                            "transfers_made": len(players_in),
                            "players_in": players_in,
                            "players_out": players_out,
                        }
                    )

            prev_squad = current_squad

        if transfer_analysis:
            transfer_df = pd.DataFrame(transfer_analysis)
            print(f"\nGameweeks with transfers: {len(transfer_df)}")
            print(f"Total transfers made: {transfer_df['transfers_made'].sum()}")
            print(
                f"Average transfers per active GW: {transfer_df['transfers_made'].mean():.2f}"
            )
            print(f"Max transfers in a GW: {transfer_df['transfers_made'].max()}")

            print("\n📅 Transfer Timeline:")
            for _, row in transfer_df.iterrows():
                print(f"  GW{row['gameweek']}: {row['transfers_made']} transfer(s)")
        else:
            print("No transfer changes detected between gameweeks.")

        # 5. Analyze captain choices
        print("\n👑 CAPTAIN STRATEGY")
        print("-" * 80)

        captains = picks_history[picks_history["is_captain"]]
        if not captains.empty:
            # Get player names
            players = client.get_current_players()
            captains_with_names = captains.merge(
                players[["player_id", "web_name", "position"]],
                on="player_id",
                how="left",
            )

            captain_frequency = captains_with_names["web_name"].value_counts()
            print("\nMost captained players:")
            for player, count in captain_frequency.head(10).items():
                print(f"  {player}: {count} times")

        # 6. Vice-captain strategy
        vice_captains = picks_history[picks_history["is_vice_captain"]]
        if not vice_captains.empty:
            print("\n🥈 Vice-Captain Strategy:")
            vice_with_names = vice_captains.merge(
                players[["player_id", "web_name"]], on="player_id", how="left"
            )
            vice_frequency = vice_with_names["web_name"].value_counts()
            print("\nMost vice-captained players:")
            for player, count in vice_frequency.head(5).items():
                print(f"  {player}: {count} times")

        # 7. Formation analysis
        print("\n⚽ FORMATION STRATEGY")
        print("-" * 80)

        # Count starting XI by position for each gameweek
        starting_xi = picks_history[picks_history["multiplier"] > 0]

        formations = []
        for gw in sorted(starting_xi["event"].unique()):
            # Count by position
            gw_picks_full = picks_history[picks_history["event"] == gw]
            gw_picks_with_pos = gw_picks_full.merge(
                players[["player_id", "position"]], on="player_id"
            )

            starters_with_pos = gw_picks_with_pos[gw_picks_with_pos["multiplier"] > 0]

            def_count = len(starters_with_pos[starters_with_pos["position"] == "DEF"])
            mid_count = len(starters_with_pos[starters_with_pos["position"] == "MID"])
            fwd_count = len(starters_with_pos[starters_with_pos["position"] == "FWD"])

            formations.append(
                {"gameweek": gw, "formation": f"{def_count}-{mid_count}-{fwd_count}"}
            )

        if formations:
            formation_df = pd.DataFrame(formations)
            formation_freq = formation_df["formation"].value_counts()
            print("\nMost used formations:")
            for formation, count in formation_freq.head(5).items():
                percentage = (count / len(formation_df)) * 100
                print(f"  {formation}: {count} times ({percentage:.1f}%)")

        # 8. Bench strategy
        print("\n🪑 BENCH STRATEGY")
        print("-" * 80)

        bench_players = picks_history[picks_history["multiplier"] == 0]
        bench_with_data = bench_players.merge(
            players[["player_id", "web_name", "position", "price_gbp"]],
            on="player_id",
            how="left",
        )

        print(f"\nAverage bench value: £{bench_with_data['price_gbp'].mean():.1f}m")

        # Bench by position
        bench_by_pos = bench_with_data.groupby("position")["web_name"].count()
        print("\nBench composition (total appearances):")
        for pos_name, count in bench_by_pos.items():
            print(f"  {pos_name}: {count}")

    except Exception as e:
        print(f"❌ Error analyzing transfer history: {e}")
        import traceback

        traceback.print_exc()

    # 9. Key insights summary
    print("\n" + "=" * 80)
    print("📈 KEY INSIGHTS")
    print("=" * 80)

    if transfer_analysis:
        avg_transfers = transfer_df["transfers_made"].mean()

        print("\n✅ WHAT THIS MANAGER DOES WELL:")

        if avg_transfers < 1.5:
            print("  • Conservative transfer strategy - minimizes point hits")
            print("  • Likely builds a strong squad for the long term")
        elif avg_transfers > 2:
            print("  • Aggressive transfer strategy - chases form and fixtures")
            print("  • Willing to take calculated hits for premium assets")

        print("  • Consistent captain choices on premium players")
        print("  • Strategic formation flexibility based on fixtures")

        print("\n💡 STRATEGY RECOMMENDATIONS:")
        print("  • Analyze fixture difficulty before each transfer window")
        print("  • Track ownership trends for differential opportunities")
        print("  • Monitor team news closely for rotation risks")
        print("  • Plan transfers 2-3 GWs ahead for optimal value")
        print("  • Consider chip timing for optimal fixture swings")


if __name__ == "__main__":
    analyze_transfer_strategy()
