#!/usr/bin/env python3
"""
Demonstration script for Dynamic Team Strength improvements

Shows the difference between static vs dynamic team strength calculations
and demonstrates the GW8 transition logic.
"""

from dynamic_team_strength import DynamicTeamStrength, load_historical_gameweek_data
from client import get_current_teams
import pandas as pd

def demonstrate_dynamic_team_strength():
    """Demonstrate key improvements in dynamic team strength calculation"""
    
    print("🏟️  Dynamic Team Strength Demonstration")
    print("=" * 60)
    
    # Load data
    teams = get_current_teams()
    calculator = DynamicTeamStrength(debug=False)
    current_data = load_historical_gameweek_data(start_gw=1, end_gw=1)
    
    print(f"📊 Data loaded: {len(teams)} teams, {len(current_data)} gameweeks")
    
    # Static baseline for comparison (2023-24 table)
    static_ratings = {
        'Manchester City': 1.3, 'Arsenal': 1.269, 'Liverpool': 1.237, 'Aston Villa': 1.205,
        'Tottenham': 1.174, 'Chelsea': 1.142, 'Newcastle': 1.11, 'Manchester Utd': 1.079,
        'West Ham': 1.047, 'Crystal Palace': 1.016, 'Brighton': 0.984, 'Bournemouth': 0.953,
        'Fulham': 0.921, 'Wolves': 0.889, 'Everton': 0.858, 'Brentford': 0.826,
        'Nottingham Forest': 0.795, 'Luton': 0.763, 'Burnley': 0.732, 'Sheffield Utd': 0.7,
        'Man City': 1.3, 'Man Utd': 1.079, "Nott'm Forest": 0.795, 'Spurs': 1.174,
        'Leicester': 0.826, 'Southampton': 0.763, 'Ipswich': 0.7
    }
    
    # Early season (weighted)
    gw2_ratings = calculator.get_team_strength(2, teams, current_data)
    
    # Late season (current only)  
    gw10_ratings = calculator.get_team_strength(10, teams, current_data)
    
    # Create comparison DataFrame
    comparison = []
    for team_name in static_ratings.keys():
        comparison.append({
            'Team': team_name,
            'Static_2023_24': static_ratings[team_name],
            'GW2_Weighted': gw2_ratings.get(team_name, static_ratings[team_name]),
            'GW10_Current': gw10_ratings.get(team_name, static_ratings[team_name])
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Calculate changes
    comparison_df['Change_GW2'] = comparison_df['GW2_Weighted'] - comparison_df['Static_2023_24']
    comparison_df['Change_GW10'] = comparison_df['GW10_Current'] - comparison_df['Static_2023_24']
    comparison_df['GW2_to_GW10'] = comparison_df['GW10_Current'] - comparison_df['GW2_Weighted']
    
    # Sort by static rating for comparison
    comparison_df = comparison_df.sort_values('Static_2023_24', ascending=False)
    
    print("\n🔍 Team Strength Evolution Analysis")
    print("-" * 60)
    print("Static 2023-24 vs Weighted GW2 vs Current Season GW10")
    print("-" * 60)
    
    for _, row in comparison_df.head(10).iterrows():  # Top 10 teams
        team = row['Team'][:12].ljust(12)  # Truncate long names
        static = f"{row['Static_2023_24']:.3f}"
        gw2 = f"{row['GW2_Weighted']:.3f}"
        gw10 = f"{row['GW10_Current']:.3f}"
        change_gw2 = f"{row['Change_GW2']:+.3f}" if abs(row['Change_GW2']) > 0.001 else " 0.000"
        change_gw10 = f"{row['Change_GW10']:+.3f}" if abs(row['Change_GW10']) > 0.001 else " 0.000"
        
        print(f"{team} | {static} → {gw2} ({change_gw2}) → {gw10} ({change_gw10})")
    
    print("\n📈 Key Insights:")
    
    # Biggest gainers/losers
    biggest_gainer = comparison_df.loc[comparison_df['Change_GW10'].idxmax()]
    biggest_loser = comparison_df.loc[comparison_df['Change_GW10'].idxmin()]
    
    print(f"🔥 Biggest strength gainer: {biggest_gainer['Team']} ({biggest_gainer['Change_GW10']:+.3f})")
    print(f"❄️  Biggest strength loser: {biggest_loser['Team']} ({biggest_loser['Change_GW10']:+.3f})")
    
    # Transition impact
    transition_impact = abs(comparison_df['GW2_to_GW10']).mean()
    print(f"⚡ Average rating change GW2→GW10: {transition_impact:.3f}")
    
    # Teams that would be affected most by GW8+ transition
    high_change_teams = comparison_df[abs(comparison_df['GW2_to_GW10']) > 0.1]
    print(f"🎯 Teams significantly affected by GW8+ transition: {len(high_change_teams)}")
    
    print("\n🏆 Methodology Benefits:")
    print("✅ Early season stability with 2024-25 historical baseline")
    print("✅ Gradual transition incorporating current season performance")  
    print("✅ GW8+ pure current season focus (no historical bias)")
    print("✅ Promoted teams get fair assessment based on actual performance")
    print("✅ Form teams properly recognized vs outdated table positions")
    
    print(f"\n🔄 Transition Schedule:")
    print(f"   GW1-3: 80% historical + 20% current")
    print(f"   GW4-7: 60% historical + 40% current")
    print(f"   GW8+:  100% current season (ignoring historical)")
    
    return comparison_df

if __name__ == "__main__":
    comparison_results = demonstrate_dynamic_team_strength()
    print(f"\n✅ Dynamic team strength demonstration complete!")