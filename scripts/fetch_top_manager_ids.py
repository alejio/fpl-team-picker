"""
Fetch Top 1% Manager IDs from FPL API
Helper script to get manager IDs for analysis
"""

import requests
import json
from typing import List, Dict


def get_top_managers_from_league(
    league_id: int = 314, num_managers: int = 100
) -> List[Dict]:
    """
    Get top managers from a global league.
    League 314 is the overall FPL league.
    """
    base_url = "https://fantasy.premierleague.com/api"

    managers = []
    page = 1
    page_size = 50

    print(f"Fetching top {num_managers} managers from league {league_id}...")

    while len(managers) < num_managers:
        try:
            url = f"{base_url}/leagues-classic/{league_id}/standings/"
            params = {"page": page, "page_standings": page}

            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                break

            data = response.json()
            standings = data.get("standings", {}).get("results", [])

            if not standings:
                break

            for entry in standings:
                managers.append(
                    {
                        "manager_id": entry["entry"],
                        "rank": entry["rank"],
                        "total_points": entry["total"],
                        "team_name": entry["entry_name"],
                        "player_name": entry.get("player_name", ""),
                    }
                )

                if len(managers) >= num_managers:
                    break

            page += 1

            # Check if there are more pages
            if len(standings) < page_size:
                break

        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break

    return managers[:num_managers]


def get_top_managers_by_rank_range(
    start_rank: int = 1, end_rank: int = 100
) -> List[Dict]:
    """
    Get managers in a specific rank range.
    Note: This requires iterating through pages, which may be slow.
    """
    # For top 1% (assuming ~10M players), that's rank < 100k
    # We'll sample from different rank ranges
    managers = []

    # Sample from different rank tiers
    rank_tiers = [
        (1, 1000),  # Top 1k
        (1000, 10000),  # Top 10k
        (10000, 50000),  # Top 50k
        (50000, 100000),  # Top 100k (1%)
    ]

    print("Fetching managers from different rank tiers...")

    for start, end in rank_tiers:
        # Get a sample from this tier
        sample_size = min(25, (end - start) // 100)
        print(f"  Sampling {sample_size} managers from rank {start}-{end}")

        # In practice, you'd need to iterate through pages
        # For now, we'll use the league endpoint
        pass

    return managers


def main():
    """Fetch and save top manager IDs."""
    import sys

    num_managers = 100
    if len(sys.argv) > 1:
        num_managers = int(sys.argv[1])

    print("=" * 80)
    print("FETCH TOP 1% FPL MANAGER IDs")
    print("=" * 80)
    print()

    # Try to get from overall league
    managers = get_top_managers_from_league(league_id=314, num_managers=num_managers)

    if not managers:
        print("⚠️  Could not fetch managers from league endpoint")
        print("   You may need to manually collect manager IDs from:")
        print("   - fantasy.premierleague.com/leagues/global")
        print("   - Or use known top manager IDs")
        return

    print(f"\n✅ Fetched {len(managers)} managers")
    print("\nTop 10 managers:")
    for i, m in enumerate(managers[:10], 1):
        print(
            f"  {i}. {m['team_name']} (ID: {m['manager_id']}, Rank: {m['rank']:,}, Points: {m['total_points']})"
        )

    # Save to file
    output_file = "experiments/top_manager_ids.json"
    with open(output_file, "w") as f:
        json.dump(managers, f, indent=2)

    print(f"\n💾 Saved to: {output_file}")

    # Also save just IDs for easy use
    ids_file = "experiments/top_manager_ids.txt"
    with open(ids_file, "w") as f:
        for m in managers:
            f.write(f"{m['manager_id']}\n")

    print(f"💾 Manager IDs saved to: {ids_file}")
    print("\nTo analyze these managers, run:")
    print(
        f"  python scripts/analyze_top_managers_strategy.py {' '.join([str(m['manager_id']) for m in managers[:20]])}"
    )


if __name__ == "__main__":
    main()
