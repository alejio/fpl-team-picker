#!/usr/bin/env python3
"""
Test script to validate XP model improvements

Compares old vs new model predictions and validates form weighting functionality.
"""

import pandas as pd
import numpy as np
from xp_model import XPModel
from client import (
    get_current_players,
    get_current_teams, 
    get_gameweek_live_data,
    get_player_xg_xa_rates,
    get_fixtures_normalized
)

def test_xp_model_improvements():
    """Test the improved XP model against baseline"""
    
    print("🧪 Testing XP Model Improvements")
    print("=" * 50)
    
    # Load test data
    print("📥 Loading test data...")
    try:
        players = get_current_players()
        teams = get_current_teams()
        xg_rates = get_player_xg_xa_rates()
        fixtures = get_fixtures_normalized()
        
        print(f"✅ Loaded {len(players)} players, {len(teams)} teams")
        
        # Standardize column names
        players = players.rename(columns={
            'price_gbp': 'price',
            'team_id': 'team',
            'selected_by_percentage': 'selected_by_percent',
            'availability_status': 'status'
        })
        
        teams = teams.rename(columns={'team_id': 'id'})
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
    
    # Test 1: Basic model functionality
    print("\n🔧 Test 1: Basic Model Functionality")
    try:
        model = XPModel(debug=True)
        
        # Test without form data first
        result_basic = model.calculate_expected_points(
            players_data=players.head(50),  # Test subset for speed
            teams_data=teams,
            xg_rates_data=xg_rates,
            fixtures_data=fixtures,
            target_gameweek=10,
            live_data=None,
            gameweeks_ahead=1
        )
        
        print(f"✅ Basic XP calculation successful for {len(result_basic)} players")
        print(f"📊 Average XP: {result_basic['xP'].mean():.2f}")
        print(f"📈 Top XP: {result_basic['xP'].max():.2f}")
        
    except Exception as e:
        print(f"❌ Basic model test failed: {e}")
        return False
    
    # Test 2: Form data integration
    print("\n📈 Test 2: Form Data Integration")
    try:
        # Create mock form data
        mock_form_data = []
        test_players = players.head(10)['player_id'].tolist()
        
        for gw in range(5, 10):  # GW 5-9 for form window
            for i, player_id in enumerate(test_players):
                # Create varying performance patterns
                if i < 3:  # Hot players
                    points = np.random.randint(8, 15)  # High scoring
                elif i < 6:  # Average players  
                    points = np.random.randint(2, 6)   # Average scoring
                else:  # Cold players
                    points = np.random.randint(0, 3)   # Low scoring
                
                mock_form_data.append({
                    'player_id': player_id,
                    'event': gw,
                    'total_points': points
                })
        
        mock_live_data = pd.DataFrame(mock_form_data)
        
        # Test with form data - get DataFrame of test players
        test_players_df = players[players['player_id'].isin(test_players)].head(10)
        
        result_with_form = model.calculate_expected_points(
            players_data=test_players_df,
            teams_data=teams,
            xg_rates_data=xg_rates,
            fixtures_data=fixtures,
            target_gameweek=10,
            live_data=mock_live_data,
            gameweeks_ahead=1
        )
        
        # Check if form columns are present
        form_columns = ['form_multiplier', 'momentum', 'recent_points_per_game']
        form_present = all(col in result_with_form.columns for col in form_columns)
        
        if form_present:
            print("✅ Form data integration successful")
            
            # Show form distribution
            hot_players = len(result_with_form[result_with_form['momentum'] == '🔥'])
            cold_players = len(result_with_form[result_with_form['momentum'] == '❄️'])
            avg_multiplier = result_with_form['form_multiplier'].mean()
            
            print(f"🔥 Hot players: {hot_players}")
            print(f"❄️ Cold players: {cold_players}")
            print(f"📊 Average form multiplier: {avg_multiplier:.2f}")
            
        else:
            print(f"⚠️ Form columns missing: {[col for col in form_columns if col not in result_with_form.columns]}")
        
    except Exception as e:
        print(f"❌ Form data test failed: {e}")
        return False
    
    # Test 3: Statistical xG/xA estimation
    print("\n🎯 Test 3: Statistical xG/xA Estimation")
    try:
        # Test with players missing xG/xA data
        test_subset = players.head(20).copy()
        
        # Artificially remove some xG/xA data to test estimation
        xg_rates_limited = xg_rates.head(10)  # Only keep 10 players with xG/xA data
        
        result_estimation = model.calculate_expected_points(
            players_data=test_subset,
            teams_data=teams,
            xg_rates_data=xg_rates_limited,
            fixtures_data=fixtures,
            target_gameweek=10,
            live_data=None,
            gameweeks_ahead=1
        )
        
        # Check that all players have xG/xA values
        missing_xg = result_estimation['xG90'].isna().sum()
        missing_xa = result_estimation['xA90'].isna().sum()
        
        if missing_xg == 0 and missing_xa == 0:
            print("✅ Statistical xG/xA estimation successful - no missing values")
        else:
            print(f"⚠️ Still missing xG90: {missing_xg}, xA90: {missing_xa}")
        
        # Show xG/xA distribution by position
        position_stats = result_estimation.groupby('position')[['xG90', 'xA90']].mean()
        print("📊 Average xG/xA by position:")
        print(position_stats.round(3))
        
    except Exception as e:
        print(f"❌ xG/xA estimation test failed: {e}")
        return False
    
    # Test 4: Performance comparison
    print("\n⚡ Test 4: Performance Impact Analysis")
    try:
        # Compare XP with and without form weighting
        if 'result_basic' in locals() and 'result_with_form' in locals():
            # Find common players
            common_players = set(result_basic['player_id']).intersection(
                set(result_with_form['player_id'])
            )
            
            basic_xp = result_basic[result_basic['player_id'].isin(common_players)].set_index('player_id')['xP']
            form_xp = result_with_form[result_with_form['player_id'].isin(common_players)].set_index('player_id')['xP']
            
            # Calculate differences
            xp_diff = form_xp - basic_xp
            
            print(f"📊 XP Differences (Form vs Basic):")
            print(f"   Mean difference: {xp_diff.mean():.3f}")
            print(f"   Max increase: {xp_diff.max():.3f}")
            print(f"   Max decrease: {xp_diff.min():.3f}")
            print(f"   Players affected: {len(xp_diff[xp_diff != 0])}/{len(xp_diff)}")
            
    except Exception as e:
        print(f"⚠️ Performance comparison failed: {e}")
    
    print("\n🎉 XP Model Testing Complete!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_xp_model_improvements()
    exit(0 if success else 1)