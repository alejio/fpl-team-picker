#!/usr/bin/env python3
"""
Debug script to check why current squad form analysis is empty
"""

import sys
sys.path.append('/Users/alex/dev/FPL/fpl-team-picker')

from client import (
    get_current_players,
    get_current_teams, 
    get_player_xg_xa_rates,
    get_fixtures_normalized,
    get_gameweek_live_data,
    get_fpl_team
)
from xp_model import XPModel
import pandas as pd

def debug_squad_form_analysis():
    """Debug the squad form analysis issue"""
    
    print("🔍 Debugging Squad Form Analysis")
    print("=" * 50)
    
    # Test with a sample team ID (you can change this to your actual ID)
    team_id = 123456  # Replace with actual team ID if known
    target_gw = 2
    
    try:
        # Load team data
        print(f"📂 Loading FPL team {team_id}...")
        team_data = get_fpl_team(team_id)
        
        if not team_data:
            print("❌ Could not load team data - using sample squad")
            # Create a sample squad for testing
            sample_squad = pd.DataFrame({
                'player_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                'web_name': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5', 
                           'Player6', 'Player7', 'Player8', 'Player9', 'Player10',
                           'Player11', 'Player12', 'Player13', 'Player14', 'Player15'],
                'position': ['GKP', 'GKP', 'DEF', 'DEF', 'DEF', 'DEF', 'DEF',
                           'MID', 'MID', 'MID', 'MID', 'MID', 'FWD', 'FWD', 'FWD'],
                'price': [4.5, 4.0, 5.0, 5.5, 6.0, 4.5, 4.0, 6.5, 7.0, 8.0, 5.5, 5.0, 7.5, 9.0, 6.5]
            })
            current_squad = sample_squad
        else:
            # Load actual squad
            players = get_current_players()
            teams = get_current_teams()
            
            player_ids = [pick['element'] for pick in team_data['picks']]
            current_squad = players[players['player_id'].isin(player_ids)].copy()
            current_squad = current_squad.merge(teams[['id', 'name']], left_on='team', right_on='id', how='left')
            
            print(f"✅ Loaded squad with {len(current_squad)} players")
        
        # Load players and calculate XP with form data
        print("📊 Calculating XP with form data...")
        players = get_current_players()
        teams = get_current_teams()
        xg_rates = get_player_xg_xa_rates()
        fixtures = get_fixtures_normalized()
        live_data = get_gameweek_live_data(1)
        
        model = XPModel(debug=False)
        players_with_xp = model.calculate_expected_points(
            players_data=players,
            teams_data=teams,
            xg_rates_data=xg_rates,
            fixtures_data=fixtures,
            target_gameweek=target_gw,
            live_data=live_data,
            gameweeks_ahead=1
        )
        
        print(f"✅ Calculated XP for {len(players_with_xp)} players")
        print(f"XP data columns: {list(players_with_xp.columns)}")
        print(f"Has momentum column: {'momentum' in players_with_xp.columns}")
        
        # Debug the merge operation
        print(f"\n🔍 Debug merge operation:")
        print(f"Squad shape: {current_squad.shape}")
        print(f"Squad player_id column: {'player_id' in current_squad.columns}")
        print(f"XP data shape: {players_with_xp.shape}")
        print(f"XP player_id column: {'player_id' in players_with_xp.columns}")
        
        if 'player_id' in current_squad.columns and 'player_id' in players_with_xp.columns:
            # Test the merge
            squad_with_form = current_squad.merge(
                players_with_xp[['player_id', 'xP', 'momentum', 'form_multiplier', 'recent_points_per_game']], 
                on='player_id', 
                how='left'
            )
            
            print(f"✅ Merge successful: {squad_with_form.shape}")
            print(f"Merged columns: {list(squad_with_form.columns)}")
            
            if not squad_with_form.empty and 'momentum' in squad_with_form.columns:
                momentum_counts = squad_with_form['momentum'].value_counts()
                print(f"Squad momentum distribution: {dict(momentum_counts)}")
                
                # Show sample squad with form data
                print(f"\n📋 Sample squad with form data:")
                sample_cols = ['web_name', 'position', 'momentum', 'form_multiplier', 'recent_points_per_game']
                available_cols = [col for col in sample_cols if col in squad_with_form.columns]
                print(squad_with_form[available_cols].head(10))
                
                return True
            else:
                print("❌ Merge resulted in empty data or missing momentum column")
                return False
        else:
            print("❌ Missing player_id column in one of the dataframes")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_squad_form_analysis()
    if success:
        print("\n✅ Squad form analysis should work - check marimo cell dependencies")
    else:
        print("\n❌ Squad form analysis has issues that need fixing")