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
    from client import FPLDataClient
    
    # Initialize robust client
    client = FPLDataClient()
    
    print(f"🔄 Loading FPL data from database for gameweek {target_gameweek}...")
    
    # Get base data from database using enhanced client
    players_base = client.get_current_players()
    teams_df = client.get_current_teams()
    
    # Get historical live data for form calculation using enhanced methods
    historical_data = []
    
    for gw in range(max(1, target_gameweek - form_window), target_gameweek):
        try:
            # Try gameweek-specific performance data first (more detailed)
            gw_data = client.get_gameweek_performance(gw)
            if not gw_data.empty:
                # Standardize column names for compatibility
                if 'gameweek' in gw_data.columns and 'event' not in gw_data.columns:
                    gw_data['event'] = gw_data['gameweek']
                elif 'event' not in gw_data.columns:
                    gw_data['event'] = gw
                    
                historical_data.append(gw_data)
                print(f"✅ Loaded detailed form data for GW{gw} ({len(gw_data)} players)")
            else:
                # Fallback to legacy live data
                gw_data = client.get_gameweek_live_data(gw)
                if not gw_data.empty:
                    # Ensure event column exists
                    if 'event' not in gw_data.columns:
                        gw_data['event'] = gw
                    historical_data.append(gw_data)
                    print(f"✅ Loaded legacy form data for GW{gw}")
        except Exception as e:
            print(f"⚠️  No form data for GW{gw}: {str(e)[:50]}")
            continue
    
    # Combine historical data for form analysis
    if historical_data:
        live_data_historical = pd.concat(historical_data, ignore_index=True)
        print(f"📊 Combined form data from {len(historical_data)} gameweeks")
    else:
        live_data_historical = pd.DataFrame()
        print("⚠️  No historical form data available")
    
    # Try to get current gameweek live data with enhanced methods
    try:
        # Try detailed gameweek performance first
        current_live_data = client.get_gameweek_performance(target_gameweek)
        if not current_live_data.empty:
            print(f"✅ Found detailed live data for GW{target_gameweek} ({len(current_live_data)} players)")
        else:
            # Fallback to legacy live data
            current_live_data = client.get_gameweek_live_data(target_gameweek)
            if not current_live_data.empty:
                print(f"✅ Found legacy live data for GW{target_gameweek}")
    except Exception as e:
        print(f"⚠️  No current live data for GW{target_gameweek}: {str(e)[:50]}")
        current_live_data = pd.DataFrame()
    
    # Merge players with current gameweek stats if available
    if not current_live_data.empty:
        # Standardize column names for compatibility
        if 'gameweek' in current_live_data.columns and 'event' not in current_live_data.columns:
            current_live_data['event'] = current_live_data['gameweek']
        
        players = players_base.merge(
            current_live_data, 
            on='player_id', 
            how='left'
        )
        
        # Ensure event column exists
        if 'event' not in players.columns:
            players['event'] = target_gameweek
    else:
        # Use baseline data without live stats
        players = players_base.copy()
        players['total_points'] = 0  # Reset for new gameweek
        players['event'] = target_gameweek
    
    # Get additional data for XP calculation using enhanced client
    xg_rates = client.get_player_xg_xa_rates()
    fixtures = client.get_fixtures_normalized()
    
    # Standardize column names for compatibility
    rename_dict = {
        'price_gbp': 'price',
        'selected_by_percentage': 'selected_by_percent',
        'availability_status': 'status'
    }
    
    # Handle team column naming - keep original name to avoid conflicts
    if 'team_id' in players.columns and 'team' not in players.columns:
        rename_dict['team_id'] = 'team'
    
    players = players.rename(columns=rename_dict)
    
    # Add calculated fields with robust handling
    # Ensure event column exists and has valid values
    if 'event' not in players.columns:
        players['event'] = target_gameweek
    
    players['points_per_game'] = players['total_points'].fillna(0) / players['event'].fillna(target_gameweek).replace(0, 1)
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
        
        # Get picks from previous gameweek with enhanced methods
        try:
            # Try to get historical picks for specific gameweek
            historical_picks = client.get_my_picks_history(start_gw=previous_gameweek, end_gw=previous_gameweek)
            if not historical_picks.empty:
                current_picks = historical_picks
                print(f"✅ Found historical picks for GW{previous_gameweek}")
            else:
                # Fallback to current picks
                current_picks = client.get_my_current_picks()
                if not current_picks.empty:
                    print("⚠️ Using current picks as fallback")
                else:
                    print("❌ No picks found in database")
                    return None
        except Exception as e:
            print(f"⚠️ Error getting picks: {str(e)[:50]}")
            # Last resort fallback to current picks
            current_picks = client.get_my_current_picks()
            if current_picks.empty:
                print("❌ No current picks found in database")
                return None
        
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
            "bank": manager_info.get('bank', 0) / 10.0,  # Convert from 0.1M units to millions
            "team_value": manager_info.get('team_value', 1000) / 10.0,  # Convert from 0.1M units to millions
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
    
    # Handle flexible team column naming
    team_col = None
    for col in ['team', 'team_id', 'team_id_x', 'team_id_y']:
        if col in current_squad.columns:
            team_col = col
            break
    
    if team_col:
        current_squad = current_squad.merge(teams[['id', 'name']], left_on=team_col, right_on='id', how='left')
    else:
        print(f"⚠️ No team column found in squad data. Available columns: {list(current_squad.columns)}")
        # Add placeholder team name
        current_squad['name'] = 'Unknown Team'
    
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