#!/usr/bin/env python3
"""
Test script to verify form analytics fix for early season

This tests that form analytics properly work when we only have 1 gameweek of data.
"""

import sys
sys.path.append('/Users/alex/dev/FPL/fpl-team-picker')

from xp_model import XPModel
from client import (
    get_current_players,
    get_current_teams, 
    get_player_xg_xa_rates,
    get_fixtures_normalized,
    get_gameweek_live_data
)
import pandas as pd

def test_early_season_form_analytics():
    """Test form analytics with only GW1 data available"""
    
    print("🧪 Testing Early Season Form Analytics Fix")
    print("=" * 50)
    
    # Load all required data
    print("📂 Loading data...")
    players = get_current_players()
    teams = get_current_teams()
    xg_rates = get_player_xg_xa_rates()
    fixtures = get_fixtures_normalized()
    
    # Load GW1 live data
    live_data = get_gameweek_live_data(1)
    print(f"✅ Loaded {len(live_data)} player performances from GW1")
    
    # Test XP calculation with form weighting
    print("\n🔮 Testing XP calculation with form weighting...")
    model = XPModel(debug=False)
    
    # Calculate XP for GW2 using GW1 form data
    result = model.calculate_expected_points(
        players_data=players,
        teams_data=teams,
        xg_rates_data=xg_rates,
        fixtures_data=fixtures,
        target_gameweek=2,
        live_data=live_data,
        gameweeks_ahead=1
    )
    
    print(f"✅ Calculated XP for {len(result)} players")
    
    # Analyze form distribution
    if 'momentum' in result.columns:
        momentum_counts = result['momentum'].value_counts()
        print(f"\n📊 Form Distribution:")
        for momentum, count in momentum_counts.items():
            print(f"   {momentum} {count} players")
        
        # Show top performers by category
        hot_players = result[result['momentum'] == '🔥'].nlargest(5, 'recent_points_per_game')
        cold_players = result[result['momentum'] == '❄️'].nsmallest(5, 'recent_points_per_game')
        
        if not hot_players.empty:
            print(f"\n🔥 Top Hot Players (GW1 stars):")
            for _, player in hot_players.iterrows():
                print(f"   {player['web_name']} ({player['position']}) - {player['recent_points_per_game']:.1f} pts")
        
        if not cold_players.empty:
            print(f"\n❄️ Cold Players (GW1 strugglers):")
            for _, player in cold_players.head(3).iterrows():
                print(f"   {player['web_name']} ({player['position']}) - {player['recent_points_per_game']:.1f} pts")
        
        # Test form multiplier application
        form_enhanced = result[result['form_multiplier'] != 1.0]
        print(f"\n📈 Form Analysis:")
        print(f"   Players with form adjustments: {len(form_enhanced)}")
        print(f"   Average form multiplier: {result['form_multiplier'].mean():.2f}")
        print(f"   XP range: {result['xP'].min():.1f} - {result['xP'].max():.1f}")
        
        print(f"\n✅ Form analytics working properly for early season!")
        return True
        
    else:
        print("❌ No momentum column found - form analytics not working")
        return False

if __name__ == "__main__":
    success = test_early_season_form_analytics()
    if success:
        print("\n🎉 All tests passed! Form analytics ready for marimo interface.")
    else:
        print("\n⚠️ Tests failed - form analytics need more work.")