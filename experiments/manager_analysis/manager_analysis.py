"""
Manager Strategy Analysis Tool
Analyzes FPL manager strategies by examining:
- Gameweek-by-gameweek team selection
- Transfer patterns and timing
- Fixture-aware decisions
- Performance trends
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from client import FPLDataClient


class ManagerAnalyzer:
    """Comprehensive FPL manager strategy analyzer."""

    def __init__(self, manager_id: int):
        self.manager_id = manager_id
        self.base_url = "https://fantasy.premierleague.com/api"
        self.client = FPLDataClient()

        # Cache for data
        self.bootstrap_data = None
        self.fixtures_data = None
        self.manager_history = None
        self.manager_picks_by_gw = {}
        self.transfers_by_gw = {}

    def fetch_bootstrap_data(self) -> Dict:
        """Fetch bootstrap-static data (players, teams, gameweeks)."""
        if self.bootstrap_data is None:
            response = requests.get(f"{self.base_url}/bootstrap-static/")
            response.raise_for_status()
            self.bootstrap_data = response.json()
        return self.bootstrap_data

    def fetch_fixtures_data(self) -> List[Dict]:
        """Fetch fixtures data."""
        if self.fixtures_data is None:
            response = requests.get(f"{self.base_url}/fixtures/")
            response.raise_for_status()
            self.fixtures_data = response.json()
        return self.fixtures_data

    def fetch_manager_history(self) -> Dict:
        """Fetch manager's overall history."""
        if self.manager_history is None:
            response = requests.get(f"{self.base_url}/entry/{self.manager_id}/")
            response.raise_for_status()
            self.manager_history = response.json()
        return self.manager_history

    def fetch_manager_gw_history(self) -> List[Dict]:
        """Fetch manager's gameweek history."""
        response = requests.get(f"{self.base_url}/entry/{self.manager_id}/history/")
        response.raise_for_status()
        return response.json()

    def fetch_gw_picks(self, gameweek: int) -> Dict:
        """Fetch manager's picks for a specific gameweek."""
        if gameweek not in self.manager_picks_by_gw:
            response = requests.get(
                f"{self.base_url}/entry/{self.manager_id}/event/{gameweek}/picks/"
            )
            response.raise_for_status()
            self.manager_picks_by_gw[gameweek] = response.json()
        return self.manager_picks_by_gw[gameweek]

    def get_player_info(self, player_id: int) -> Dict:
        """Get player information from bootstrap data."""
        bootstrap = self.fetch_bootstrap_data()
        for player in bootstrap["elements"]:
            if player["id"] == player_id:
                return player
        return {}

    def get_team_name(self, team_id: int) -> str:
        """Get team name from bootstrap data."""
        bootstrap = self.fetch_bootstrap_data()
        for team in bootstrap["teams"]:
            if team["id"] == team_id:
                return team["name"]
        return "Unknown"

    def get_position_name(self, element_type: int) -> str:
        """Get position name from element type."""
        positions = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        return positions.get(element_type, "UNK")

    def analyze_gameweek_picks(self, gameweek: int) -> pd.DataFrame:
        """Analyze picks for a specific gameweek."""
        picks_data = self.fetch_gw_picks(gameweek)
        picks = picks_data["picks"]

        rows = []
        for pick in picks:
            player = self.get_player_info(pick["element"])
            rows.append(
                {
                    "player_id": pick["element"],
                    "web_name": player.get("web_name", "Unknown"),
                    "team": self.get_team_name(player.get("team", 0)),
                    "position": self.get_position_name(player.get("element_type", 0)),
                    "price": player.get("now_cost", 0) / 10,
                    "position_in_squad": pick["position"],
                    "is_captain": pick["is_captain"],
                    "is_vice_captain": pick["is_vice_captain"],
                    "multiplier": pick["multiplier"],
                    "in_starting_11": pick["position"] <= 11,
                }
            )

        return pd.DataFrame(rows)

    def get_fixtures_for_gw(self, gameweek: int) -> pd.DataFrame:
        """Get fixtures for a specific gameweek."""
        all_fixtures = self.fetch_fixtures_data()
        fixtures = []

        for fixture in all_fixtures:
            if fixture.get("event") == gameweek:
                fixtures.append(
                    {
                        "gameweek": gameweek,
                        "home_team": self.get_team_name(fixture["team_h"]),
                        "away_team": self.get_team_name(fixture["team_a"]),
                        "team_h_id": fixture["team_h"],
                        "team_a_id": fixture["team_a"],
                        "team_h_difficulty": fixture.get("team_h_difficulty", 3),
                        "team_a_difficulty": fixture.get("team_a_difficulty", 3),
                    }
                )

        return pd.DataFrame(fixtures)

    def analyze_transfers(self, gameweek: int) -> Tuple[List[Dict], List[Dict]]:
        """Analyze transfers made for a gameweek."""
        picks_data = self.fetch_gw_picks(gameweek)

        transfers_in = []
        transfers_out = []

        if "entry_history" in picks_data:
            entry = picks_data["entry_history"]
            entry.get("event_transfers_cost", 0)
            num_transfers = entry.get("event_transfers", 0)

            if num_transfers > 0:
                # Try to identify transfers by comparing with previous GW
                if gameweek > 1:
                    try:
                        prev_picks = self.fetch_gw_picks(gameweek - 1)
                        prev_ids = {p["element"] for p in prev_picks["picks"]}
                        curr_ids = {p["element"] for p in picks_data["picks"]}

                        out_ids = prev_ids - curr_ids
                        in_ids = curr_ids - prev_ids

                        for player_id in out_ids:
                            player = self.get_player_info(player_id)
                            transfers_out.append(
                                {
                                    "player_id": player_id,
                                    "web_name": player.get("web_name", "Unknown"),
                                    "team": self.get_team_name(player.get("team", 0)),
                                    "position": self.get_position_name(
                                        player.get("element_type", 0)
                                    ),
                                    "price": player.get("now_cost", 0) / 10,
                                }
                            )

                        for player_id in in_ids:
                            player = self.get_player_info(player_id)
                            transfers_in.append(
                                {
                                    "player_id": player_id,
                                    "web_name": player.get("web_name", "Unknown"),
                                    "team": self.get_team_name(player.get("team", 0)),
                                    "position": self.get_position_name(
                                        player.get("element_type", 0)
                                    ),
                                    "price": player.get("now_cost", 0) / 10,
                                }
                            )
                    except Exception:
                        pass

        return transfers_in, transfers_out

    def get_player_fixture_difficulty(
        self, player_id: int, gameweek: int
    ) -> Optional[int]:
        """Get fixture difficulty for a player in a specific gameweek."""
        player = self.get_player_info(player_id)
        team_id = player.get("team", 0)

        fixtures = self.get_fixtures_for_gw(gameweek)
        for _, fixture in fixtures.iterrows():
            if fixture["team_h_id"] == team_id:
                return fixture["team_h_difficulty"]
            elif fixture["team_a_id"] == team_id:
                return fixture["team_a_difficulty"]

        return None

    def generate_comprehensive_report(self, max_gameweek: Optional[int] = None) -> Dict:
        """Generate comprehensive analysis report."""
        print(f"\n{'=' * 80}")
        print("FPL MANAGER STRATEGY ANALYSIS")
        print(f"{'=' * 80}\n")

        # Fetch manager info
        manager = self.fetch_manager_history()
        print(f"Manager: {manager.get('name', 'Unknown')} (ID: {self.manager_id})")
        print(
            f"Team: {manager.get('player_first_name', '')} {manager.get('player_last_name', '')}"
        )
        print(f"Overall Points: {manager.get('summary_overall_points', 0):,}")
        print(f"Overall Rank: {manager.get('summary_overall_rank', 0):,}")
        print(f"\n{'=' * 80}\n")

        # Fetch gameweek history
        history_data = self.fetch_manager_gw_history()
        gw_history = history_data["current"]

        if max_gameweek:
            gw_history = [gw for gw in gw_history if gw["event"] <= max_gameweek]

        # Overall statistics
        print("OVERALL SEASON STATISTICS")
        print(f"{'-' * 80}")

        total_points = sum(gw["points"] for gw in gw_history)
        avg_points = np.mean([gw["points"] for gw in gw_history])
        total_transfers = sum(gw["event_transfers"] for gw in gw_history)
        total_transfer_cost = sum(gw["event_transfers_cost"] for gw in gw_history)

        print(f"Total Points: {total_points}")
        print(f"Average Points per GW: {avg_points:.2f}")
        print(f"Total Transfers: {total_transfers}")
        print(f"Total Transfer Cost: {total_transfer_cost} points")
        print(f"Net Points (after transfers): {total_points}")

        # Performance trend
        print(f"\n{'=' * 80}\n")
        print("PERFORMANCE TREND ANALYSIS")
        print(f"{'-' * 80}")

        # Split into first half and second half
        mid_point = len(gw_history) // 2
        first_half = gw_history[:mid_point]
        second_half = gw_history[mid_point:]

        first_half_avg = (
            np.mean([gw["points"] for gw in first_half]) if first_half else 0
        )
        second_half_avg = (
            np.mean([gw["points"] for gw in second_half]) if second_half else 0
        )

        print(f"First Half Average (GW1-{mid_point}): {first_half_avg:.2f}")
        print(f"Second Half Average (GW{mid_point + 1}+): {second_half_avg:.2f}")
        print(f"Improvement: {second_half_avg - first_half_avg:+.2f} points/GW")

        if second_half_avg > first_half_avg:
            improvement_pct = (
                (second_half_avg - first_half_avg) / first_half_avg
            ) * 100
            print(f"Improvement Percentage: +{improvement_pct:.1f}%")
            print("✅ CONFIRMED: Manager improves over the season!")
        else:
            print("⚠️ Manager did not improve in the second half")

        # Gameweek-by-gameweek analysis
        print(f"\n{'=' * 80}\n")
        print("GAMEWEEK-BY-GAMEWEEK DETAILED ANALYSIS")
        print(f"{'=' * 80}\n")

        for gw in gw_history:
            gw_num = gw["event"]
            print(f"\n{'─' * 80}")
            print(f"GAMEWEEK {gw_num}")
            print(f"{'─' * 80}")

            # GW Summary
            print("\n📊 Performance:")
            print(f"   Points: {gw['points']}")
            print(f"   Rank: {gw['rank']:,}")
            print(f"   Overall Rank: {gw['overall_rank']:,}")
            print(f"   Bank: £{gw['bank'] / 10:.1f}m")
            print(f"   Team Value: £{gw['value'] / 10:.1f}m")

            # Transfers
            if gw["event_transfers"] > 0:
                print("\n🔄 Transfers:")
                print(f"   Number of Transfers: {gw['event_transfers']}")
                print(f"   Transfer Cost: {gw['event_transfers_cost']} points")

                transfers_in, transfers_out = self.analyze_transfers(gw_num)

                if transfers_out:
                    print("\n   Transferred OUT:")
                    for player in transfers_out:
                        print(
                            f"      ❌ {player['web_name']} ({player['team']}) - {player['position']} - £{player['price']}m"
                        )

                if transfers_in:
                    print("\n   Transferred IN:")
                    for player in transfers_in:
                        difficulty = self.get_player_fixture_difficulty(
                            player["player_id"], gw_num
                        )
                        difficulty_emoji = (
                            "🟢"
                            if difficulty and difficulty <= 2
                            else "🟡"
                            if difficulty == 3
                            else "🔴"
                            if difficulty
                            else "⚪"
                        )
                        print(
                            f"      ✅ {player['web_name']} ({player['team']}) - {player['position']} - £{player['price']}m {difficulty_emoji}"
                        )

            # Team selection
            print("\n⚽ Team Selection:")
            picks_df = self.analyze_gameweek_picks(gw_num)

            # Starting 11
            starting_11 = picks_df[picks_df["in_starting_11"]].sort_values(
                "position_in_squad"
            )
            print("\n   Starting XI:")
            for _, player in starting_11.iterrows():
                captain_mark = (
                    " (C)"
                    if player["is_captain"]
                    else " (VC)"
                    if player["is_vice_captain"]
                    else ""
                )
                difficulty = self.get_player_fixture_difficulty(
                    player["player_id"], gw_num
                )
                difficulty_emoji = (
                    "🟢"
                    if difficulty and difficulty <= 2
                    else "🟡"
                    if difficulty == 3
                    else "🔴"
                    if difficulty
                    else "⚪"
                )
                print(
                    f"      {player['position']}: {player['web_name']} ({player['team']}){captain_mark} {difficulty_emoji}"
                )

            # Bench
            bench = picks_df[~picks_df["in_starting_11"]].sort_values(
                "position_in_squad"
            )
            print("\n   Bench:")
            for _, player in bench.iterrows():
                print(
                    f"      {player['position']}: {player['web_name']} ({player['team']})"
                )

            # Fixtures for this GW
            print("\n🏟️  Fixtures:")
            fixtures = self.get_fixtures_for_gw(gw_num)
            for _, fixture in fixtures.iterrows():
                print(f"      {fixture['home_team']} vs {fixture['away_team']}")

        return {
            "manager": manager,
            "gw_history": gw_history,
            "total_points": total_points,
            "avg_points": avg_points,
            "improvement": second_half_avg - first_half_avg,
        }


def main():
    """Run manager analysis."""
    import sys

    manager_id = 25020  # Michael Bradon

    # Allow command line override
    if len(sys.argv) > 1:
        manager_id = int(sys.argv[1])

    max_gw = None
    if len(sys.argv) > 2:
        max_gw = int(sys.argv[2])

    analyzer = ManagerAnalyzer(manager_id)

    try:
        analyzer.generate_comprehensive_report(max_gameweek=max_gw)

        print(f"\n{'=' * 80}")
        print("ANALYSIS COMPLETE")
        print(f"{'=' * 80}\n")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
