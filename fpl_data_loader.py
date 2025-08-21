"""
FPL Data Loader Module

Handles all data loading operations for FPL gameweek management including:
- FPL player, team, and fixture data from database
- Historical form data collection
- Manager team data fetching
- Data preprocessing and standardization
"""

import pandas as pd
from typing import Dict, Tuple, Optional


def fetch_fpl_data(target_gameweek: int, form_window: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, pd.DataFrame]:
    """
    Fetch FPL data from database for specified gameweek with historical form data
    
    Args:
        target_gameweek: The gameweek to optimize for
        form_window: Number of previous gameweeks to include for form analysis
        
    Returns:
        Tuple of (players, teams, xg_rates, fixtures, target_gameweek, live_data_historical)
    """
    from client import (
        get_current_players, 
        get_current_teams, 
        get_gameweek_live_data,
        get_player_xg_xa_rates,
        get_fixtures_normalized
    )
    
    print(f"🔄 Loading FPL data from database for gameweek {target_gameweek}...")
    
    # Get base data from database
    players_base = get_current_players()
    teams_df = get_current_teams()
    
    # Get historical live data for form calculation
    historical_data = []
    
    for gw in range(max(1, target_gameweek - form_window), target_gameweek):
        try:
            gw_data = get_gameweek_live_data(gw)
            if not gw_data.empty:
                historical_data.append(gw_data)
                print(f"✅ Loaded form data for GW{gw}")
        except:
            print(f"⚠️  No form data for GW{gw}")
            continue
    
    # Combine historical data for form analysis
    if historical_data:
        live_data_historical = pd.concat(historical_data, ignore_index=True)
        print(f"📊 Combined form data from {len(historical_data)} gameweeks")
    else:
        live_data_historical = pd.DataFrame()
        print("⚠️  No historical form data available")
    
    # Try to get current gameweek live data
    try:
        current_live_data = get_gameweek_live_data(target_gameweek)
        print(f"✅ Found current live data for GW{target_gameweek}")
    except:
        print(f"⚠️  No current live data for GW{target_gameweek}")
        current_live_data = pd.DataFrame()
    
    # Merge players with current gameweek stats if available
    if not current_live_data.empty:
        players = players_base.merge(
            current_live_data, 
            on='player_id', 
            how='left'
        )
    else:
        # Use baseline data without live stats
        players = players_base.copy()
        players['total_points'] = 0  # Reset for new gameweek
        players['event'] = target_gameweek
    
    # Get additional data for XP calculation
    xg_rates = get_player_xg_xa_rates()
    fixtures = get_fixtures_normalized()
    
    # Standardize column names for compatibility
    players = players.rename(columns={
        'price_gbp': 'price',
        'team_id': 'team', 
        'selected_by_percentage': 'selected_by_percent',
        'availability_status': 'status'
    })
    
    # Add calculated fields
    players['points_per_game'] = players['total_points'] / players['event'].fillna(1)
    players['form'] = players['total_points'].fillna(0).astype(float)
    
    teams = teams_df.rename(columns={'team_id': 'id'})
    
    print(f"✅ Loaded {len(players)} players, {len(teams)} teams from database")
    print(f"📅 Target GW: {target_gameweek}")
    
    return players, teams, xg_rates, fixtures, target_gameweek, live_data_historical


def fetch_manager_team(previous_gameweek: int) -> Optional[Dict]:
    """
    Fetch manager's team from previous gameweek
    
    Args:
        previous_gameweek: The gameweek to fetch team data from
        
    Returns:
        Dictionary with team info including picks, bank balance, etc., or None if failed
    """
    print(f"🔄 Fetching your team from GW{previous_gameweek}...")
    
    try:
        from client import FPLDataClient
        client = FPLDataClient()
        
        # Get manager data
        manager_data = client.get_my_manager_data()
        if manager_data.empty:
            print("❌ No manager data found in database")
            return None
        
        # Get picks from previous gameweek
        try:
            # Try to get historical picks for previous gameweek
            current_picks = client.get_my_current_picks()
            if current_picks.empty:
                print("❌ No current picks found in database")
                return None
        except:
            print("⚠️ Using current picks as fallback")
            current_picks = client.get_my_current_picks()
        
        # Convert picks to the expected format
        picks = []
        for _, pick in current_picks.iterrows():
            picks.append({
                'element': pick['player_id'],
                'is_captain': pick.get('is_captain', False),
                'is_vice_captain': pick.get('is_vice_captain', False),
                'multiplier': pick.get('multiplier', 1)
            })
        
        # Get manager info from first row
        manager_info = manager_data.iloc[0]
        
        team_info = {
            "manager_id": manager_info.get('manager_id', 0),
            "entry_name": manager_info.get('entry_name', 'My Team'),
            "total_points": manager_info.get('summary_overall_points', 0),
            "bank": manager_info.get('bank', 0),
            "team_value": manager_info.get('value', 100.0),
            "picks": picks,
            "free_transfers": 1  # Default to 1 free transfer
        }
        
        print(f"✅ Loaded team from GW{previous_gameweek}: {team_info['entry_name']}")
        return team_info
        
    except Exception as e:
        print(f"❌ Error fetching team from database: {e}")
        print("💡 Make sure manager data is available in the database")
        return None


def process_current_squad(team_data: Dict, players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """
    Process current squad data by merging team picks with player details
    
    Args:
        team_data: Team data dictionary from fetch_manager_team
        players: Players DataFrame
        teams: Teams DataFrame
        
    Returns:
        Processed current squad DataFrame with captain info and team names
    """
    if not team_data:
        return pd.DataFrame()
    
    # Get player details and merge with current data
    player_ids = [pick['element'] for pick in team_data['picks']]
    current_squad = players[players['player_id'].isin(player_ids)].copy()
    current_squad = current_squad.merge(teams[['id', 'name']], left_on='team', right_on='id', how='left')
    
    # Add captain info
    captain_id = next((pick['element'] for pick in team_data['picks'] if pick['is_captain']), None)
    vice_captain_id = next((pick['element'] for pick in team_data['picks'] if pick['is_vice_captain']), None)
    
    current_squad['role'] = current_squad['player_id'].apply(
        lambda x: '(C)' if x == captain_id else '(VC)' if x == vice_captain_id else ''
    )
    
    return current_squad


def load_gameweek_datasets(target_gameweek: int) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Comprehensive data loading for gameweek analysis
    
    Args:
        target_gameweek: The gameweek to optimize for
        
    Returns:
        Tuple of (current_squad, team_data, players, teams, xg_rates, fixtures, live_data_historical)
    """
    # Load FPL data
    players, teams, xg_rates, fixtures, _, live_data_historical = fetch_fpl_data(target_gameweek)
    
    # Load manager team from previous gameweek
    previous_gw = target_gameweek - 1
    team_data = fetch_manager_team(previous_gw)
    
    # Process current squad
    current_squad = process_current_squad(team_data, players, teams)
    
    return current_squad, team_data, players, teams, xg_rates, fixtures, live_data_historical