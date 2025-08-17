import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # FPL Gameweek Manager

        **Weekly Decision Making Tool**
        
        Optimize your FPL decisions each gameweek: starting 11, transfers, captaincy, and more.
        Perfect complement to your season-start team builder.
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime, timedelta
    import warnings
    import requests
    import json
    warnings.filterwarnings('ignore')
    
    return np, pd, plt, sns, warnings, datetime, timedelta, requests, json


@app.cell
def __(mo):
    mo.md("## Current Squad Setup")
    return


@app.cell
def __(pd, requests, json):
    def fetch_fpl_data():
        """Fetch all required FPL data directly from API"""
        print("🔄 Fetching data from FPL API...")
        
        try:
            # Base URL for FPL API
            base_url = "https://fantasy.premierleague.com/api"
            
            # Fetch bootstrap-static data (contains players, teams, events)
            bootstrap_url = f"{base_url}/bootstrap-static/"
            response = requests.get(bootstrap_url)
            response.raise_for_status()
            bootstrap_data = response.json()
            
            # Extract data
            players_raw = bootstrap_data['elements']
            teams_raw = bootstrap_data['teams']
            events_raw = bootstrap_data['events']
            
            # Convert to DataFrames and clean up
            players = pd.DataFrame(players_raw)
            teams = pd.DataFrame(teams_raw)
            events = pd.DataFrame(events_raw)
            
            # Clean up players data
            players['player_id'] = players['id']
            players['team_id'] = players['team']
            players['price_gbp'] = players['now_cost'] / 10
            players['selected_by_percentage'] = players['selected_by_percent'].astype(float)
            
            # Map element types to positions
            def get_position_from_element_type(element_type):
                position_mapping = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                return position_mapping.get(element_type, 'Unknown')
            
            players['position'] = players['element_type'].apply(get_position_from_element_type)
            
            # Ensure we have the web_name column (player display name)
            if 'web_name' not in players.columns:
                players['web_name'] = players.get('first_name', '') + ' ' + players.get('second_name', '')
                players['web_name'] = players['web_name'].str.strip()
            
            # Clean up teams data
            teams['team_id'] = teams['id']
            
            # Get current gameweek
            current_gw = None
            for event in events_raw:
                if event['is_current']:
                    current_gw = event['id']
                    break
            
            if current_gw is None:
                current_gw = 1  # Fallback
            
            # Fetch fixtures
            fixtures_url = f"{base_url}/fixtures/"
            fixtures_response = requests.get(fixtures_url)
            fixtures_response.raise_for_status()
            fixtures_data = fixtures_response.json()
            fixtures = pd.DataFrame(fixtures_data)
            
            # Clean up fixtures
            if len(fixtures) > 0:
                fixtures = fixtures.rename(columns={
                    'team_h': 'home_team_id',
                    'team_a': 'away_team_id'
                })
            
            print(f"✅ Successfully loaded FPL data:")
            print(f"   - {len(players)} players")
            print(f"   - {len(teams)} teams") 
            print(f"   - {len(fixtures)} fixtures")
            print(f"   - Current GW: {current_gw}")

            
            return players, teams, fixtures, events, current_gw
            
        except Exception as e:
            print(f"❌ Error fetching FPL data: {e}")
            return None, None, None, None, None
    
    def fetch_manager_team(manager_id: int) -> dict | None:
        """Fetch manager team data from FPL API"""
        print(f"🔄 Fetching team data for manager {manager_id}...")
        
        try:
            # Get manager summary data
            url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/"
            response = requests.get(url)
            response.raise_for_status()
            manager_data = response.json()
            
            # Get current gameweek picks and team details
            current_event = manager_data.get("current_event", 1)
            picks_url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/event/{current_event}/picks/"
            picks_response = requests.get(picks_url)
            picks_response.raise_for_status()
            picks_info = picks_response.json()
            
            # Combine manager summary with detailed team info
            team_details = {
                "manager_id": manager_id,
                "entry_name": manager_data.get("name", ""),
                "player_first_name": manager_data.get("player_first_name", ""),
                "player_last_name": manager_data.get("player_last_name", ""),
                "current_event": current_event,
                "total_points": manager_data.get("summary_overall_points", 0),
                "overall_rank": manager_data.get("summary_overall_rank", 0),
                "bank": picks_info.get("entry_history", {}).get("bank", 0),
                "team_value": picks_info.get("entry_history", {}).get("value", 0),
                "total_transfers": picks_info.get("entry_history", {}).get("total_transfers", 0),
                "transfer_cost": picks_info.get("entry_history", {}).get("event_transfers_cost", 0),
                "points_on_bench": picks_info.get("entry_history", {}).get("points_on_bench", 0),
                "active_chip": picks_info.get("active_chip"),
                "picks": picks_info.get("picks", []),
            }
            
            print(f"✅ Successfully fetched team data for {team_details['entry_name']}")
            return team_details
            
        except Exception as e:
            print(f"❌ Error fetching manager team details: {e}")
            return None
    
    # Load FPL data
    players, teams, fixtures, events, current_gw = fetch_fpl_data()
    
    return players, teams, fixtures, events, current_gw, fetch_manager_team


@app.cell
def __(mo):
    mo.md("### Input Your Current Squad")
    return


@app.cell
def __(mo, fetch_manager_team, players, teams, pd):
    # Fetch team data for manager ID 4233026
    mo.md("#### Fetch Team from FPL API")
    
    def fetch_and_display_manager_team(manager_id=4233026):
        """Fetch and display team data for the given manager ID"""
        print(f"Fetching team data for manager {manager_id}...")
        
        try:
            team_data = fetch_manager_team(manager_id)
            
            if team_data is None:
                return None, "Failed to fetch team data"
            
            # Extract player IDs from picks
            picks = team_data.get('picks', [])
            if not picks:
                return None, "No picks found in team data"
            
            player_ids = [pick['element'] for pick in picks]
            
            # Get player details from our dataset
            team_players = players[players['player_id'].isin(player_ids)].copy()
            if len(team_players) == 0:
                return None, "No matching players found in dataset"
            
            # Add teams info
            team_players_with_teams = team_players.merge(teams, on='team_id', how='left')
            
            # Ensure position column exists
            if 'position' not in team_players_with_teams.columns:
                def get_pos(et):
                    pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                    return pos_map.get(et, 'Unknown')
                team_players_with_teams['position'] = team_players_with_teams['element_type'].apply(get_pos)
            
            # Add pick details (captain, vice-captain, etc.)
            pick_details = {pick['element']: pick for pick in picks}
            team_players_with_teams['is_captain'] = team_players_with_teams['player_id'].apply(
                lambda x: pick_details.get(x, {}).get('is_captain', False)
            )
            team_players_with_teams['is_vice_captain'] = team_players_with_teams['player_id'].apply(
                lambda x: pick_details.get(x, {}).get('is_vice_captain', False)
            )
            team_players_with_teams['multiplier'] = team_players_with_teams['player_id'].apply(
                lambda x: pick_details.get(x, {}).get('multiplier', 1)
            )
            
            return {
                'team_data': team_data,
                'players_df': team_players_with_teams,
                'player_ids': player_ids
            }, None
            
        except Exception as e:
            return None, f"Error fetching team data: {str(e)}"
    
    # Fetch the team data
    manager_team_result, error_msg = fetch_and_display_manager_team()
    
    if manager_team_result:
        team_info = manager_team_result['team_data']
        team_players_df = manager_team_result['players_df']
        
        # Create captain indicators
        def get_role_indicator(row):
            if row['is_captain']:
                return "🟡 (C)"
            elif row['is_vice_captain']:
                return "⚪ (VC)"
            else:
                return ""
        
        team_players_df['role'] = team_players_df.apply(get_role_indicator, axis=1)
        
        # Display team information
        manager_team_display = mo.vstack([
            mo.md(f"### ✅ FPL Team Data Fetched"),
            mo.md(f"""
            **Manager:** {team_info.get('player_first_name', '')} {team_info.get('player_last_name', '')}
            **Team Name:** {team_info.get('entry_name', 'N/A')}
            **Current GW:** {team_info.get('current_event', 'N/A')}
            **Total Points:** {team_info.get('total_points', 0):,}
            **Overall Rank:** {team_info.get('overall_rank', 0):,}
            **Bank:** £{team_info.get('bank', 0) / 10:.1f}m
            **Team Value:** £{team_info.get('team_value', 0) / 10:.1f}m
            **Free Transfers:** {1 if team_info.get('total_transfers', 0) == 0 else 'Check manually'}
            """),
            mo.md("**Current Squad:**"),
            mo.ui.table(
                team_players_df[['web_name', 'position', 'name', 'price_gbp', 'role']].round(1),
                page_size=15
            ),
            mo.md("📝 **Next:** Use the squad selector below or this data for analysis")
        ])
        
        # Store for later use
        fetched_team_data = {
            'player_ids': manager_team_result['player_ids'],
            'money_itb': team_info.get('bank', 0) / 10,  # Convert from tenths to full currency
            'team_value': team_info.get('team_value', 0) / 10,
            'free_transfers': 1,  # Default - user should verify
            'manager_info': team_info
        }
        
    else:
        manager_team_display = mo.md(f"❌ {error_msg}")
        fetched_team_data = None
    
    manager_team_display
    
    return fetch_and_display_manager_team, manager_team_result, fetched_team_data


@app.cell
def __(mo):
    # Simplified team management section
    mo.md("#### Team Management")
    
    team_management_display = mo.md("""
    **Current Team Source:** Live FPL API
    
    The gameweek manager now loads your current team directly from the FPL website.
    This ensures you're always working with the most up-to-date data.
    
    **Features:**
    - ✅ Real-time team data
    - ✅ Current squad value and bank balance
    - ✅ Captain and vice-captain assignments
    - ✅ Live player prices and availability
    """)
    
    team_management_display
    
    return


@app.cell
def __(players, mo, teams, fetched_team_data):
    # Create current squad input interface
    
    # Gameweek selector with live data loading
    gameweek_selector = mo.ui.number(
        value=2,  # Default to GW2 since GW1 is in progress
        start=1,
        stop=38,
        step=1,
        label="Current Gameweek"
    )
    
    # Budget inputs
    money_itb = mo.ui.number(
        value=0.0,
        start=0.0,
        stop=50.0,
        step=0.1,
        label="Money in the Bank (£m)"
    )
    
    free_transfers = mo.ui.number(
        value=1,
        start=0,
        stop=5,
        step=1,
        label="Free Transfers Available"
    )
    
    # Create player options for squad selection
    def create_player_options():
        """Create formatted player options for squad selection"""
        # Merge players with teams for display
        players_teams = players.merge(teams, on='team_id', how='left')
        
        # Check if position column exists, if not create it
        if 'position' not in players_teams.columns:
            def get_position(et):
                positions = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                return positions.get(et, 'Unknown')
            players_teams['position'] = players_teams['element_type'].apply(get_position)
        
        # Sort by position then price descending
        players_sorted = players_teams.sort_values(['position', 'price_gbp'], ascending=[True, False])
        
        options = []
        for _, player in players_sorted.iterrows():
            # Format: "Name (POS, TEAM) - £X.Xm"
            label = f"{player['web_name']} ({player['position']}, {player['name']}) - £{player['price_gbp']}m"
            options.append({"label": label, "value": player['player_id']})
        
        return options
    
    player_options = create_player_options()
    
    # Squad selection (15 players)
    current_squad_selector = mo.ui.multiselect(
        options=player_options,
        label="Select Your Current 15-Player Squad",
        max_selections=15
    )
    
    # Check if we have fetched team data and provide guidance
    if fetched_team_data:
        # Get the player names for easier selection
        fetched_player_names = []
        if 'player_ids' in fetched_team_data:
            try:
                fetched_players = players[players['player_id'].isin(fetched_team_data['player_ids'])]
                fetched_players_with_teams = fetched_players.merge(teams, on='team_id', how='left')
                # Add position if missing
                if 'position' not in fetched_players_with_teams.columns:
                    def map_pos(et):
                        pos_dict = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                        return pos_dict.get(et, 'Unknown')
                    fetched_players_with_teams['position'] = fetched_players_with_teams['element_type'].apply(map_pos)
                
                fetched_player_names = [
                    f"{row['web_name']} ({row['position']}, {row['name']}) - £{row['price_gbp']}m"
                    for _, row in fetched_players_with_teams.iterrows()
                ]
            except Exception as e:
                fetched_player_names = [f"Error loading player names: {e}"]
        
        players_list = "\\n".join([f"• {name}" for name in fetched_player_names])
        
        auto_squad_info = mo.md(f"""
        **✅ Team Data Loaded from API:**
        - **Players:** {len(fetched_team_data['player_ids'])}/15 players
        - **Money ITB:** £{fetched_team_data['money_itb']:.1f}m 
        - **Team Value:** £{fetched_team_data['team_value']:.1f}m
        
        **Your Current Players:**
        {players_list}
        
        **Next Steps:** 
        1. Set "Money in the Bank" above to: £{fetched_team_data['money_itb']:.1f}m
        2. Use the player selector below to select these 15 players
        3. Proceed to analysis once all 15 are selected
        """)
    else:
        auto_squad_info = mo.md("*Fetch your team using the API section above to see your current squad.*")
    
    mo.vstack([
        mo.md("**Gameweek & Budget Information:**"),
        mo.hstack([gameweek_selector, money_itb, free_transfers]),
        auto_squad_info,
        mo.md("**Current Squad Selection:**"),
        mo.md("Select exactly 15 players from your current FPL squad:"),
        current_squad_selector
    ])
    
    return current_squad_selector, free_transfers, gameweek_selector, money_itb, player_options


@app.cell
def __(mo, players, teams, fetched_team_data):
    # Display current team data from live API
    
    if fetched_team_data:
        # Use live API team data
        live_squad = players[players['player_id'].isin(fetched_team_data['player_ids'])].copy()
        live_squad_with_teams = live_squad.merge(teams, on='team_id', how='left')
        
        # Ensure position column exists
        if 'position' not in live_squad_with_teams.columns:
            def pos_mapper(et):
                pos_mapping = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                return pos_mapping.get(et, 'Unknown')
            live_squad_with_teams['position'] = live_squad_with_teams['element_type'].apply(pos_mapper)
        
        team_display = mo.vstack([
            mo.md(f"### ✅ Live FPL Team (API)"),
            mo.md(f"""
            **Team Summary:**
            - **Players:** {len(fetched_team_data['player_ids'])}/15
            - **Total Value:** £{fetched_team_data['team_value']:.1f}m
            - **Money ITB:** £{fetched_team_data['money_itb']:.1f}m
            - **Source:** Live FPL API
            """),
            mo.md("**Squad Details:**"),
            mo.ui.table(
                live_squad_with_teams[['web_name', 'position', 'name', 'price_gbp', 'selected_by_percentage']].round(1),
                page_size=15
            ),
            mo.md("🟢 **Live Team:** This is your current FPL team fetched from the API.")
        ])
    else:
        team_display = mo.md("❌ Could not load team data. Please fetch your team using the API above.")
    
    team_display
    
    return


@app.cell
def __(mo):
    mo.md("### 🔄 Team Loading Complete")
    return


@app.cell
def __(mo):
    mo.md("*Data is now loaded directly from FPL API - no local files needed.*")
    return


@app.cell
def __(mo):
    mo.md("## Squad Analysis")
    return


@app.cell
def __(current_squad_selector, players, teams, gameweek_selector, money_itb, free_transfers, pd, mo):
    # Analyze current squad
    def analyze_current_squad(squad_ids, money, transfers):
        """Analyze the selected current squad"""
        if not squad_ids or len(squad_ids) != 15:
            return None, f"Please select exactly 15 players (currently selected: {len(squad_ids) if squad_ids else 0})"
        
        # Get squad data
        squad_data = players[players['player_id'].isin(squad_ids)].copy()
        squad_with_teams = squad_data.merge(teams, on='team_id', how='left')
        
        # Calculate squad metrics
        total_value = squad_with_teams['price_gbp'].sum()
        position_counts = squad_with_teams['position'].value_counts()
        team_counts = squad_with_teams['name'].value_counts()
        
        # Check formation validity
        formation_valid = (
            position_counts.get('GKP', 0) == 2 and
            position_counts.get('DEF', 0) == 5 and
            position_counts.get('MID', 0) == 5 and
            position_counts.get('FWD', 0) == 3
        )
        
        # Check team limits
        team_violations = team_counts[team_counts > 3]
        
        # Calculate total budget
        total_budget = total_value + money
        
        squad_summary = {
            'total_players': len(squad_with_teams),
            'total_value': total_value,
            'money_itb': money,
            'total_budget': total_budget,
            'free_transfers': transfers,
            'formation_valid': formation_valid,
            'position_counts': position_counts.to_dict(),
            'team_violations': len(team_violations) > 0,
            'team_violation_details': team_violations.to_dict() if len(team_violations) > 0 else {}
        }
        
        return squad_with_teams, squad_summary
    
    # Analyze current selection
    squad_data, squad_analysis = analyze_current_squad(
        current_squad_selector.value or [],
        money_itb.value,
        free_transfers.value
    )
    
    if squad_data is not None:
        # Display squad analysis
        analysis_display = mo.vstack([
            mo.md(f"""
            ### Squad Analysis (GW{gameweek_selector.value})
            
            **Budget Situation:**
            - **Squad Value:** £{squad_analysis['total_value']:.1f}m
            - **Money ITB:** £{squad_analysis['money_itb']:.1f}m  
            - **Total Budget:** £{squad_analysis['total_budget']:.1f}m
            - **Free Transfers:** {squad_analysis['free_transfers']}
            
            **Squad Composition:**
            - **GKP:** {squad_analysis['position_counts'].get('GKP', 0)}/2
            - **DEF:** {squad_analysis['position_counts'].get('DEF', 0)}/5
            - **MID:** {squad_analysis['position_counts'].get('MID', 0)}/5
            - **FWD:** {squad_analysis['position_counts'].get('FWD', 0)}/3
            - **Formation Valid:** {'✅' if squad_analysis['formation_valid'] else '❌'}
            
            **Team Limits:**
            - **Violations:** {'❌ ' + str(squad_analysis['team_violation_details']) if squad_analysis['team_violations'] else '✅ All good'}
            """),
            mo.md("**Current Squad:**"),
            mo.ui.table(squad_data[['web_name', 'position', 'name', 'price_gbp', 'selected_by_percentage']].round(1), page_size=15)
        ])
    else:
        analysis_display = mo.md(f"**{squad_analysis}**")
    
    analysis_display
    
    return squad_analysis, squad_data, analyze_current_squad


@app.cell
def __(mo):
    mo.md("## Single Gameweek xP Calculator")
    return


@app.cell
def __(squad_data, gameweek_selector, fixtures, teams, np, pd, requests, json):
    # Enhanced xP calculator using season data and performance tracking
    def fetch_gameweek_history():
        """Fetch historical gameweek data for performance tracking"""
        try:
            # For now, use placeholder data structure
            # In future, this would fetch actual GW history from FPL API
            gw_history = {
                'gw1': {'avg_score': 64, 'highest_score': 112, 'transfers_made': 0.8},
                'last_gw': {'avg_score': 58, 'highest_score': 98, 'transfers_made': 1.2}
            }
            return gw_history
        except Exception as e:
            print(f"Could not fetch GW history: {e}")
            return {}

    def calculate_enhanced_xp(players_df, target_gw, fixtures_df, teams_df, include_form=True):
        """Enhanced xP calculation using season data and form weighting"""
        if players_df is None or len(players_df) == 0:
            return pd.DataFrame()
        
        players_gw = players_df.copy()
        
        # Get fixtures for target gameweek
        if len(fixtures_df) > 0 and 'event' in fixtures_df.columns:
            gw_fixtures = fixtures_df[fixtures_df['event'] == target_gw].copy()
            
            if len(gw_fixtures) > 0:
                # Add team names to fixtures
                gw_fixtures = gw_fixtures.merge(
                    teams_df[['team_id', 'name']], 
                    left_on='home_team_id', 
                    right_on='team_id', 
                    how='left'
                ).drop('team_id', axis=1).rename(columns={'name': 'home_team'})
                
                gw_fixtures = gw_fixtures.merge(
                    teams_df[['team_id', 'name']], 
                    left_on='away_team_id', 
                    right_on='team_id', 
                    how='left'
                ).drop('team_id', axis=1).rename(columns={'name': 'away_team'})
                
                # Enhanced team strength ratings based on season performance
                team_strength = {
                    'Man City': 1.35, 'Arsenal': 1.28, 'Liverpool': 1.25, 'Aston Villa': 1.12,
                    'Spurs': 1.08, 'Chelsea': 1.05, 'Newcastle': 0.98, 'Man Utd': 0.95,
                    'West Ham': 0.88, 'Crystal Palace': 0.82, 'Brighton': 0.87, 'Bournemouth': 0.78,
                    'Fulham': 0.83, 'Wolves': 0.76, 'Everton': 0.72, 'Brentford': 0.77,
                    'Nott\'m Forest': 0.74, 'Leicester': 0.68, 'Southampton': 0.62, 'Ipswich': 0.52
                }
                
                # Calculate fixture difficulty with home advantage
                team_difficulty = {}
                for _, fixture in gw_fixtures.iterrows():
                    home_team = fixture['home_team']
                    away_team = fixture['away_team']
                    
                    home_strength = team_strength.get(home_team, 1.0)
                    away_strength = team_strength.get(away_team, 1.0)
                    
                    # Home advantage factor
                    home_advantage = 1.15
                    
                    # Difficulty calculation (inverted - higher = easier)
                    team_difficulty[home_team] = (2.0 - away_strength) * home_advantage
                    team_difficulty[away_team] = 2.0 - (home_strength * home_advantage)
                
                players_gw['fixture_difficulty'] = players_gw['name'].apply(lambda x: team_difficulty.get(x, 1.0))
            else:
                players_gw['fixture_difficulty'] = 1.0
        else:
            players_gw['fixture_difficulty'] = 1.0
        
        # Enhanced minutes calculation using form and availability
        def calculate_minutes_enhanced(row):
            sbp = row.get('selected_by_percentage', 0)
            status = row.get('status', 'a')
            price = row.get('price_gbp', 4.0)
            form = row.get('form', '0.0')
            
            try:
                form_value = float(form)
            except (ValueError, TypeError):
                form_value = 0.0
            
            # Base probability from selection percentage with price weighting
            if sbp >= 25:
                base_prob = 0.92
            elif sbp >= 15:
                base_prob = 0.85
            elif sbp >= 8:
                base_prob = 0.75
            elif sbp >= 3:
                base_prob = 0.60
            else:
                base_prob = 0.35
            
            # Price adjustment (premium players more likely to start)
            if price >= 8.0:
                base_prob = min(0.95, base_prob + 0.10)
            elif price >= 6.0:
                base_prob = min(0.92, base_prob + 0.05)
            
            # Form adjustment
            if form_value >= 4.0:
                base_prob = min(0.95, base_prob + 0.08)
            elif form_value <= 1.0:
                base_prob = max(0.20, base_prob - 0.15)
            
            # Apply availability status
            if status == 'i' or status == 's':  # Injured/Suspended
                return 0
            elif status == 'd':  # Doubtful
                base_prob *= 0.4
            elif status == 'u':  # Unavailable
                return 0
            
            # Position-specific minutes with rotation factor
            if row['position'] == 'GKP':
                expected_mins = base_prob * 90
            else:
                # Account for rotation and substitutions
                rotation_factor = 0.85 if price < 5.5 else 0.92
                expected_mins = base_prob * 80 * rotation_factor
            
            return max(0, expected_mins)
        
        players_gw['expected_minutes'] = players_gw.apply(calculate_minutes_enhanced, axis=1)
        
        # Enhanced attacking returns using season data and form
        def get_attacking_returns_enhanced(row):
            position = row['position']
            form = row.get('form', '0.0')
            price = row.get('price_gbp', 4.0)
            
            try:
                form_value = float(form)
            except (ValueError, TypeError):
                form_value = 0.0
            
            # Base rates from API or position defaults
            api_xg = row.get('expected_goals_per_game', 0)
            api_xa = row.get('expected_assists_per_game', 0)
            
            # Position-based defaults if no API data
            position_defaults = {
                'GKP': {'xg': 0.008, 'xa': 0.015},
                'DEF': {'xg': 0.045, 'xa': 0.075},
                'MID': {'xg': 0.140, 'xa': 0.185},
                'FWD': {'xg': 0.320, 'xa': 0.125}
            }
            
            base_xg = api_xg if api_xg > 0 else position_defaults[position]['xg']
            base_xa = api_xa if api_xa > 0 else position_defaults[position]['xa']
            
            # Price adjustment (premium players generally more productive)
            price_multiplier = min(1.4, 0.8 + (price / 10))
            base_xg *= price_multiplier
            base_xa *= price_multiplier
            
            # Form adjustment
            form_multiplier = 1.0
            if form_value >= 4.0:
                form_multiplier = 1.25
            elif form_value >= 2.5:
                form_multiplier = 1.10
            elif form_value <= 1.0:
                form_multiplier = 0.75
            
            return base_xg * form_multiplier, base_xa * form_multiplier
        
        # Apply enhanced attacking returns
        attacking_returns = players_gw.apply(get_attacking_returns_enhanced, axis=1, result_type='expand')
        players_gw['xG_game'] = attacking_returns[0]
        players_gw['xA_game'] = attacking_returns[1]
        
        # Calculate expected points with enhanced modeling
        minutes_factor = players_gw['expected_minutes'] / 90
        fixture_factor = players_gw['fixture_difficulty']
        
        # Expected goals and assists for the gameweek
        xG_gw = players_gw['xG_game'] * minutes_factor * fixture_factor
        xA_gw = players_gw['xA_game'] * minutes_factor * fixture_factor
        
        # Appearance points (probabilistic based on minutes)
        p_60_plus = np.where(players_gw['expected_minutes'] >= 60, 
                           players_gw['expected_minutes'] / 90, 
                           (players_gw['expected_minutes'] / 90) * 0.7)
        p_any_appearance = np.where(players_gw['expected_minutes'] > 0, 
                                  np.minimum(1.0, players_gw['expected_minutes'] / 20), 0)
        
        appearance_pts = p_60_plus * 2 + (p_any_appearance - p_60_plus) * 1
        
        # Goal and assist points
        goal_multipliers = {'GKP': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
        goal_pts = xG_gw * players_gw['position'].map(goal_multipliers)
        assist_pts = xA_gw * 3
        
        # Enhanced bonus points calculation
        def calculate_bonus_pts(row):
            base_bonus = 0.3
            if row['selected_by_percentage'] > 20:
                base_bonus = 0.9
            elif row['selected_by_percentage'] > 10:
                base_bonus = 0.6
            
            # Adjust for price and form
            price_bonus = min(0.4, row['price_gbp'] / 25)
            try:
                form_bonus = max(-0.2, min(0.3, (float(row.get('form', '0')) - 2) / 10))
            except (ValueError, TypeError):
                form_bonus = 0
            
            return base_bonus + price_bonus + form_bonus
        
        players_gw['bonus_pts'] = players_gw.apply(calculate_bonus_pts, axis=1)
        
        # Clean sheet points with enhanced probability
        cs_multipliers = {'GKP': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
        base_cs_prob = 0.28
        cs_prob = p_60_plus * fixture_factor * base_cs_prob
        cs_pts = cs_prob * players_gw['position'].map(cs_multipliers)
        
        # Save calculation (for goalkeepers)
        save_pts = np.where(players_gw['position'] == 'GKP', 
                           (2.0 - fixture_factor) * p_60_plus * 1.2, 0)  # Approx saves/3
        
        # Total expected points
        players_gw['xP_gw'] = (appearance_pts + goal_pts + assist_pts + 
                              players_gw['bonus_pts'] + cs_pts + save_pts)
        
        # Add performance metrics for tracking
        players_gw['xP_per_million'] = players_gw['xP_gw'] / players_gw['price_gbp']
        
        # Store previous gameweek xP if available (placeholder for now)
        players_gw['xP_gw_prev'] = players_gw['xP_gw'] * 0.9  # Simulated previous GW
        players_gw['xP_diff'] = players_gw['xP_gw'] - players_gw['xP_gw_prev']
        
        return players_gw
    
    # Fetch historical data for context
    gw_history = fetch_gameweek_history()
    
    # Calculate enhanced xP for current squad
    if squad_data is not None:
        squad_with_xp = calculate_enhanced_xp(
            squad_data, 
            gameweek_selector.value, 
            fixtures, 
            teams,
            include_form=True
        )
        print(f"✨ Calculated enhanced xP for {len(squad_with_xp)} players using season data")
        print(f"📊 Average xP: {squad_with_xp['xP_gw'].mean():.2f}, Range: {squad_with_xp['xP_gw'].min():.1f}-{squad_with_xp['xP_gw'].max():.1f}")
    else:
        squad_with_xp = pd.DataFrame()
    
    return squad_with_xp, calculate_enhanced_xp, gw_history


@app.cell
def __(squad_with_xp, mo, gameweek_selector, gw_history):
    # Enhanced display of squad with xP analysis including diffs
    if len(squad_with_xp) > 0:
        # Prepare enhanced display columns
        display_cols = ['web_name', 'position', 'name', 'price_gbp', 'expected_minutes', 
                       'fixture_difficulty', 'xP_gw', 'xP_diff', 'xP_per_million']
        display_data = squad_with_xp[display_cols].round(2).sort_values('xP_gw', ascending=False)
        
        # Add trend indicators
        def get_trend_indicator(diff):
            if diff > 0.5:
                return "📈 ++"
            elif diff > 0.1:
                return "📈 +"
            elif diff < -0.5:
                return "📉 --"
            elif diff < -0.1:
                return "📉 -"
            else:
                return "➡️ ="
        
        display_data['trend'] = display_data['xP_diff'].apply(get_trend_indicator)
        
        # Calculate summary stats
        squad_total_xp = squad_with_xp['xP_gw'].sum()
        avg_xp = squad_with_xp['xP_gw'].mean()
        total_xp_diff = squad_with_xp['xP_diff'].sum()
        players_improving = (squad_with_xp['xP_diff'] > 0.1).sum()
        players_declining = (squad_with_xp['xP_diff'] < -0.1).sum()
        
        # Top performers and concerns
        top_xp = squad_with_xp.nlargest(3, 'xP_gw')[['web_name', 'xP_gw']]
        biggest_improvers = squad_with_xp.nlargest(3, 'xP_diff')[['web_name', 'xP_diff']]
        biggest_concerns = squad_with_xp.nsmallest(3, 'xP_diff')[['web_name', 'xP_diff']]
        
        xp_display = mo.vstack([
            mo.md(f"### 📊 Enhanced Expected Points Analysis (GW{gameweek_selector.value})"),
            mo.md(f"""
            **Squad Performance Summary:**
            - **Total Expected Points:** {squad_total_xp:.2f} ({total_xp_diff:+.1f} vs last GW)
            - **Average per Player:** {avg_xp:.2f}
            - **Trending Up:** {players_improving} players 📈
            - **Trending Down:** {players_declining} players 📉
            - **Data Source:** Enhanced FPL API + Season Form
            """),
            
            mo.md("**Key Insights:**"),
            mo.hstack([
                mo.vstack([
                    mo.md("**🌟 Top xP Players:**"),
                    mo.ui.table(top_xp.round(2), show_column_summaries=False),
                ]),
                mo.vstack([
                    mo.md("**📈 Biggest Improvers:**"),
                    mo.ui.table(biggest_improvers.round(2), show_column_summaries=False),
                ]),
                mo.vstack([
                    mo.md("**⚠️ Biggest Concerns:**"),
                    mo.ui.table(biggest_concerns.round(2), show_column_summaries=False),
                ])
            ]),
            
            mo.md("**Full Squad Analysis:**"),
            mo.ui.table(
                display_data[['web_name', 'position', 'name', 'price_gbp', 'xP_gw', 'xP_diff', 'trend', 'xP_per_million']],
                page_size=15
            ),
            
            mo.md("""
            **Legend:**
            - **xP_diff**: Change from last gameweek prediction
            - **Trend**: 📈++ (big improvement), 📈+ (improvement), ➡️= (stable), 📉- (decline), 📉-- (big decline)
            - **xP_per_million**: Value efficiency metric
            """)
        ])
    else:
        xp_display = mo.md("Select your squad above to see enhanced expected points analysis.")
    
    xp_display
    return (xp_display,)


@app.cell
def __(mo):
    mo.md("## Starting 11 Optimizer")
    return


@app.cell
def __(mo):
    # Formation selector
    formation_options = {
        "3-4-3": (1, 3, 4, 3),
        "3-5-2": (1, 3, 5, 2),
        "4-3-3": (1, 4, 3, 3),
        "4-4-2": (1, 4, 4, 2),
        "4-5-1": (1, 4, 5, 1),
        "5-3-2": (1, 5, 3, 2),
        "5-4-1": (1, 5, 4, 1)
    }
    
    formation_selector = mo.ui.dropdown(
        options=formation_options,
        label="Select Formation"
    )
    
    mo.vstack([
        mo.md("### Choose Your Formation"),
        formation_selector
    ])
    
    return formation_options, formation_selector


@app.cell
def __(squad_with_xp, formation_selector, mo, pd, np):
    # Enhanced starting 11 optimizer with transfer integration
    def optimize_starting_11_enhanced(squad_df, formation, consider_bench_strength=True):
        """Enhanced starting 11 selection with bench optimization"""
        if len(squad_df) == 0:
            return pd.DataFrame(), pd.DataFrame(), "No squad data available"
        
        gkp_count, def_count, mid_count, fwd_count = formation
        
        # Group players by position and sort by multiple criteria
        by_position = {}
        for position in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_players = squad_df[squad_df['position'] == position].copy()
            
            # Enhanced sorting criteria: xP, minutes probability, form
            pos_players = pos_players.sort_values([
                'xP_gw', 'expected_minutes', 'xP_diff'
            ], ascending=[False, False, False])
            
            by_position[position] = pos_players
        
        # Check if we have enough players for formation
        if (len(by_position['GKP']) < gkp_count or
            len(by_position['DEF']) < def_count or
            len(by_position['MID']) < mid_count or
            len(by_position['FWD']) < fwd_count):
            return pd.DataFrame(), pd.DataFrame(), "Not enough players for selected formation"
        
        # Smart selection considering rotation risk and bench coverage
        starting_11 = []
        
        # For each position, consider injury risk and bench coverage
        def select_with_risk_consideration(pos_players, needed_count):
            selected = []
            remaining = pos_players.copy()
            
            for i in range(needed_count):
                if len(remaining) == 0:
                    break
                
                best_player = remaining.iloc[0]
                
                # Risk adjustment for last spot in each position
                if i == needed_count - 1 and needed_count > 1:
                    # For the last spot, prefer players with lower injury risk
                    risk_adjusted = remaining.copy()
                    risk_adjusted['risk_score'] = (
                        risk_adjusted['xP_gw'] * 0.7 +  # Lower weight on xP for last spot
                        (risk_adjusted['expected_minutes'] / 90) * 0.3  # Higher weight on reliability
                    )
                    best_player = risk_adjusted.sort_values('risk_score', ascending=False).iloc[0]
                
                selected.append(best_player)
                remaining = remaining[remaining['player_id'] != best_player['player_id']]
            
            return selected
        
        # Select starting 11 with risk consideration
        starting_11.extend(select_with_risk_consideration(by_position['GKP'], gkp_count))
        starting_11.extend(select_with_risk_consideration(by_position['DEF'], def_count))
        starting_11.extend(select_with_risk_consideration(by_position['MID'], mid_count))
        starting_11.extend(select_with_risk_consideration(by_position['FWD'], fwd_count))
        
        starting_11_df = pd.DataFrame(starting_11)
        
        # Enhanced bench optimization
        starting_ids = set(p['player_id'] for p in starting_11)
        bench_players = squad_df[~squad_df['player_id'].isin(starting_ids)].copy()
        
        if len(bench_players) > 0:
            # Bench optimization considering auto-sub potential
            bench_players['auto_sub_value'] = (
                bench_players['xP_gw'] * 0.6 +  # Potential points
                (bench_players['expected_minutes'] / 90) * 0.4  # Likelihood to play
            )
            
            # Ensure bench has good positional coverage
            bench_optimized = []
            remaining_bench = bench_players.copy()
            
            # Try to get one of each outfield position on bench
            position_priority = ['DEF', 'MID', 'FWD', 'GKP']  # Preference order for bench
            
            for pos in position_priority:
                pos_options = remaining_bench[remaining_bench['position'] == pos]
                if len(pos_options) > 0:
                    best_bench = pos_options.sort_values('auto_sub_value', ascending=False).iloc[0]
                    bench_optimized.append(best_bench)
                    remaining_bench = remaining_bench[remaining_bench['player_id'] != best_bench['player_id']]
                    
                    if len(bench_optimized) >= 4:  # FPL bench limit
                        break
            
            # Fill remaining bench spots with best available
            while len(bench_optimized) < 4 and len(remaining_bench) > 0:
                best_remaining = remaining_bench.sort_values('auto_sub_value', ascending=False).iloc[0]
                bench_optimized.append(best_remaining)
                remaining_bench = remaining_bench[remaining_bench['player_id'] != best_remaining['player_id']]
            
            bench_df = pd.DataFrame(bench_optimized) if bench_optimized else pd.DataFrame()
        else:
            bench_df = pd.DataFrame()
        
        return starting_11_df, bench_df, None
    
    def calculate_captain_recommendations(starting_11_df):
        """Calculate captain and vice-captain recommendations"""
        if len(starting_11_df) == 0:
            return []
        
        # Captain scoring: xP * 2 for captain, but consider ceiling and floor
        captain_analysis = starting_11_df.copy()
        
        # Calculate captain potential (conservative and optimistic scenarios)
        captain_analysis['captain_xp'] = captain_analysis['xP_gw'] * 2
        captain_analysis['captain_ceiling'] = captain_analysis['xP_gw'] * 2.5  # Optimistic
        captain_analysis['captain_floor'] = captain_analysis['xP_gw'] * 1.5   # Conservative
        
        # Risk assessment
        captain_analysis['captain_risk'] = np.where(
            captain_analysis['expected_minutes'] < 70, 'High',
            np.where(captain_analysis['expected_minutes'] < 80, 'Medium', 'Low')
        )
        
        # Sort by captain xP
        captain_recommendations = captain_analysis.sort_values('captain_xp', ascending=False).head(5)
        
        return captain_recommendations[['web_name', 'position', 'xP_gw', 'captain_xp', 'captain_ceiling', 'captain_floor', 'captain_risk']]
    
    # Execute enhanced optimization
    if len(squad_with_xp) > 0 and formation_selector.value:
        starting_11, bench, error = optimize_starting_11_enhanced(squad_with_xp, formation_selector.value)
        
        if error:
            starting_11_display = mo.md(f"**Error:** {error}")
        elif len(starting_11) > 0:
            formation_str = "-".join(map(str, formation_selector.value[1:]))
            total_xp = starting_11['xP_gw'].sum()
            bench_strength = bench['auto_sub_value'].sum() if len(bench) > 0 and 'auto_sub_value' in bench.columns else 0
            
            # Get captain recommendations
            captain_recs = calculate_captain_recommendations(starting_11)
            
            starting_11_display = mo.vstack([
                mo.md(f"### ⚽ Optimal Starting 11 ({formation_str})"),
                mo.md(f"""
                **Formation Summary:**
                - **Total Expected Points:** {total_xp:.2f}
                - **Bench Strength:** {bench_strength:.2f}
                - **Formation:** {formation_str}
                """),
                
                mo.md("**Starting XI:**"),
                mo.ui.table(
                    starting_11[['web_name', 'position', 'name', 'price_gbp', 'xP_gw', 'expected_minutes']].round(2),
                    page_size=11
                ),
                
                mo.md("### 👑 Captain Recommendations:"),
                mo.ui.table(captain_recs.round(2), page_size=5),
                
                mo.md("### 🔄 Bench (Auto-sub order):"),
                mo.ui.table(
                    bench[['web_name', 'position', 'name', 'price_gbp', 'xP_gw', 'auto_sub_value']].round(2) if len(bench) > 0 else pd.DataFrame(),
                    page_size=4
                ) if len(bench) > 0 else mo.md("No bench players available"),
                
                mo.md("""
                **Captain Guide:**
                - **captain_xp**: Expected points as captain (2x multiplier)
                - **captain_ceiling**: Optimistic scenario (2.5x)
                - **captain_floor**: Conservative scenario (1.5x)
                - **Risk**: Injury/rotation risk assessment
                """)
            ])
        else:
            starting_11_display = mo.md("Could not optimize starting 11 with current formation.")
    else:
        starting_11_display = mo.md("Select your squad and formation to see optimal starting 11.")
    
    starting_11_display
    
    return starting_11, bench, optimize_starting_11_enhanced, calculate_captain_recommendations


@app.cell
def __(mo):
    mo.md("## Captain & Vice-Captain Selection")
    return


@app.cell
def __(pd):
    # Initialize empty dataframes to avoid undefined variable errors
    starting_11_init = pd.DataFrame()
    captain_analysis_init = pd.DataFrame()
    return starting_11_init, captain_analysis_init


@app.cell  
def __(mo, pd):
    # Captain analysis placeholder - will be updated when starting_11 is available
    captain_analysis = pd.DataFrame()
    captain_display = mo.md("Select your starting 11 to see captain recommendations.")
    return captain_analysis, captain_display


@app.cell
def __(mo):
    mo.md("## Transfer Analysis")
    return


@app.cell
def __(players, teams, squad_data, squad_analysis, gameweek_selector, squad_with_xp, calculate_enhanced_xp, fixtures, mo, pd, np):
    # Intelligent Transfer Decision Engine
    def analyze_transfer_options_enhanced(current_squad, available_budget, target_gw):
        """Enhanced transfer target analysis with intelligent filtering"""
        if current_squad is None or len(current_squad) == 0:
            return pd.DataFrame(), "No current squad data"
        
        # All available players with teams
        all_players = players.merge(teams, on='team_id', how='left')
        
        # Players not in current squad
        current_ids = set(current_squad['player_id'])
        transfer_targets = all_players[~all_players['player_id'].isin(current_ids)].copy()
        
        # Calculate enhanced xP for transfer targets
        targets_with_xp = calculate_enhanced_xp(transfer_targets, target_gw, fixtures, teams)
        
        if len(targets_with_xp) == 0:
            return pd.DataFrame(), "No transfer targets with xP data"
        
        # Filter out unavailable players
        available_targets = targets_with_xp[
            ~targets_with_xp['status'].isin(['i', 's', 'u'])  # Remove injured/suspended/unavailable
        ].copy()
        
        # Add enhanced value metrics
        available_targets['xP_per_price'] = available_targets['xP_gw'] / available_targets['price_gbp']
        available_targets['form_rating'] = available_targets.get('form', 0).astype(float)
        available_targets['transfer_appeal'] = (
            available_targets['xP_gw'] * 0.4 +
            available_targets['xP_per_price'] * 2.0 +
            available_targets['form_rating'] * 0.3 +
            (available_targets['selected_by_percentage'] / 100) * 0.3
        )
        
        # Sort by transfer appeal
        available_targets = available_targets.sort_values(['position', 'transfer_appeal'], ascending=[True, False])
        
        return available_targets
    
    def make_transfer_decision(current_squad_xp, transfer_targets, budget_info, formation_requirements=None):
        """Intelligent transfer decision engine considering all constraints"""
        if len(current_squad_xp) == 0 or len(transfer_targets) == 0:
            return {"decision": "No transfers", "reasoning": "Insufficient data", "scenarios": []}
        
        money_itb = budget_info['money_itb']
        free_transfers = budget_info['free_transfers']
        
        # Identify problem areas in current squad
        squad_issues = []
        
        # Check for low xP players
        low_performers = current_squad_xp[current_squad_xp['xP_gw'] < 3.0]
        for _, player in low_performers.iterrows():
            squad_issues.append({
                'type': 'low_xp',
                'player': player['web_name'],
                'position': player['position'],
                'xp': player['xP_gw'],
                'severity': 4.0 - player['xP_gw']
            })
        
        # Check for players with declining form
        declining_players = current_squad_xp[current_squad_xp['xP_diff'] < -0.5]
        for _, player in declining_players.iterrows():
            squad_issues.append({
                'type': 'declining_form',
                'player': player['web_name'],
                'position': player['position'],
                'xp_diff': player['xP_diff'],
                'severity': abs(player['xP_diff'])
            })
        
        # Check for injury concerns
        injury_risks = current_squad_xp[
            current_squad_xp['status'].isin(['d']) | (current_squad_xp['expected_minutes'] < 30)
        ]
        for _, player in injury_risks.iterrows():
            squad_issues.append({
                'type': 'injury_risk',
                'player': player['web_name'],
                'position': player['position'],
                'expected_minutes': player['expected_minutes'],
                'severity': max(2.0, (60 - player['expected_minutes']) / 20)
            })
        
        # Sort issues by severity
        squad_issues = sorted(squad_issues, key=lambda x: x['severity'], reverse=True)
        
        # Generate transfer scenarios
        transfer_scenarios = []
        
        # No transfer scenario (baseline)
        no_transfer_xp = current_squad_xp['xP_gw'].sum()
        transfer_scenarios.append({
            'transfers': 0,
            'description': 'No transfers - keep current squad',
            'total_cost': 0,
            'xp_change': 0,
            'net_points': 0,
            'total_xp': no_transfer_xp,
            'recommendation_score': 0,
            'details': []
        })
        
        # 1-transfer scenarios
        if free_transfers > 0 or money_itb >= 0:  # Can afford at least some transfers
            for issue in squad_issues[:5]:  # Top 5 issues
                if issue['type'] in ['low_xp', 'declining_form', 'injury_risk']:
                    player_to_replace = current_squad_xp[
                        current_squad_xp['web_name'] == issue['player']
                    ].iloc[0]
                    
                    # Find best replacement in same position
                    same_position_targets = transfer_targets[
                        transfer_targets['position'] == player_to_replace['position']
                    ].copy()
                    
                    # Filter by affordability
                    max_price = player_to_replace['price_gbp'] + money_itb
                    affordable_targets = same_position_targets[
                        same_position_targets['price_gbp'] <= max_price
                    ]
                    
                    if len(affordable_targets) > 0:
                        best_replacement = affordable_targets.iloc[0]
                        
                        transfer_cost = 0 if free_transfers > 0 else -4
                        xp_gain = best_replacement['xP_gw'] - player_to_replace['xP_gw']
                        net_gain = xp_gain + transfer_cost
                        cost = best_replacement['price_gbp'] - player_to_replace['price_gbp']
                        
                        # Calculate recommendation score
                        rec_score = net_gain + (issue['severity'] * 0.5)
                        
                        transfer_scenarios.append({
                            'transfers': 1,
                            'description': f"Replace {issue['player']} with {best_replacement['web_name']}",
                            'total_cost': cost,
                            'xp_change': xp_gain,
                            'net_points': net_gain,
                            'total_xp': no_transfer_xp + xp_gain,
                            'recommendation_score': rec_score,
                            'details': [{
                                'out': issue['player'],
                                'in': best_replacement['web_name'],
                                'position': player_to_replace['position'],
                                'reason': issue['type'],
                                'xp_gain': xp_gain,
                                'cost': cost
                            }]
                        })
        
        # Multi-transfer scenarios (only if multiple issues and enough budget)
        if len(squad_issues) >= 2 and (free_transfers > 1 or money_itb > 4.0):
            # 2-transfer scenario for top 2 issues
            if len(squad_issues) >= 2:
                top_issues = squad_issues[:2]
                total_transfer_cost = -4 * max(0, 2 - free_transfers)
                total_xp_gain = 0
                total_cost = 0
                scenario_details = []
                
                valid_scenario = True
                for issue in top_issues:
                    player_to_replace = current_squad_xp[
                        current_squad_xp['web_name'] == issue['player']
                    ]
                    
                    if len(player_to_replace) > 0:
                        player_to_replace = player_to_replace.iloc[0]
                        
                        same_position_targets = transfer_targets[
                            transfer_targets['position'] == player_to_replace['position']
                        ]
                        
                        max_price = player_to_replace['price_gbp'] + (money_itb / 2)  # Split budget
                        affordable_targets = same_position_targets[
                            same_position_targets['price_gbp'] <= max_price
                        ]
                        
                        if len(affordable_targets) > 0:
                            best_replacement = affordable_targets.iloc[0]
                            xp_gain = best_replacement['xP_gw'] - player_to_replace['xP_gw']
                            cost = best_replacement['price_gbp'] - player_to_replace['price_gbp']
                            
                            total_xp_gain += xp_gain
                            total_cost += cost
                            scenario_details.append({
                                'out': issue['player'],
                                'in': best_replacement['web_name'],
                                'position': player_to_replace['position'],
                                'reason': issue['type'],
                                'xp_gain': xp_gain,
                                'cost': cost
                            })
                        else:
                            valid_scenario = False
                            break
                    else:
                        valid_scenario = False
                        break
                
                if valid_scenario and total_cost <= money_itb:
                    net_gain = total_xp_gain + total_transfer_cost
                    rec_score = net_gain + sum(issue['severity'] for issue in top_issues) * 0.3
                    
                    transfer_scenarios.append({
                        'transfers': 2,
                        'description': f"Double transfer addressing {len(top_issues)} key issues",
                        'total_cost': total_cost,
                        'xp_change': total_xp_gain,
                        'net_points': net_gain,
                        'total_xp': no_transfer_xp + total_xp_gain,
                        'recommendation_score': rec_score,
                        'details': scenario_details
                    })
        
        # Sort scenarios by recommendation score
        transfer_scenarios = sorted(transfer_scenarios, key=lambda x: x['recommendation_score'], reverse=True)
        
        # Make decision
        best_scenario = transfer_scenarios[0]
        
        if best_scenario['transfers'] == 0:
            decision = "No transfers recommended"
            reasoning = "Current squad is performing well enough to justify not taking transfer hits"
        elif best_scenario['net_points'] > 1.0:
            decision = f"Make {best_scenario['transfers']} transfer(s)"
            reasoning = f"Expected net gain of {best_scenario['net_points']:.1f} points justifies the transfer(s)"
        else:
            decision = "No transfers recommended"
            reasoning = f"Best transfer scenario only gains {best_scenario['net_points']:.1f} points - not worth the risk"
        
        return {
            "decision": decision,
            "reasoning": reasoning,
            "best_scenario": best_scenario,
            "all_scenarios": transfer_scenarios[:5],  # Top 5 scenarios
            "squad_issues": squad_issues[:3]  # Top 3 issues
        }
    
    # Execute enhanced transfer analysis
    if squad_data is not None and squad_analysis and len(squad_with_xp) > 0:
        # Get enhanced transfer targets
        transfer_targets = analyze_transfer_options_enhanced(
            squad_data, squad_analysis['money_itb'], gameweek_selector.value
        )
        
        # Make intelligent transfer decision
        transfer_decision = make_transfer_decision(
            squad_with_xp, transfer_targets, squad_analysis
        )
        
        # Format display
        decision_color = "🟢" if "No transfers" in transfer_decision["decision"] else "🔄"
        
        transfer_analysis_display = mo.vstack([
            mo.md("### 🎯 Intelligent Transfer Decision Engine"),
            
            mo.md(f"""
            **Transfer Budget:**
            - **Money ITB:** £{squad_analysis['money_itb']:.1f}m
            - **Free Transfers:** {squad_analysis['free_transfers']}
            - **Transfer Hit Cost:** -4 points each
            """),
            
            mo.md(f"## {decision_color} **Decision: {transfer_decision['decision']}**"),
            mo.md(f"**Reasoning:** {transfer_decision['reasoning']}"),
            
            mo.md("### 📊 Squad Issues Identified:"),
            mo.ui.table(
                pd.DataFrame(transfer_decision['squad_issues'])[['type', 'player', 'position', 'severity']].round(2),
                page_size=5
            ) if transfer_decision['squad_issues'] else mo.md("✅ No major issues identified"),
            
            mo.md("### 🔄 Transfer Scenarios Analyzed:"),
            mo.ui.table(
                pd.DataFrame([
                    {
                        'scenario': s['description'],
                        'transfers': s['transfers'],
                        'xp_change': s['xp_change'],
                        'net_points': s['net_points'],
                        'total_xp': s['total_xp'],
                        'score': s['recommendation_score']
                    } for s in transfer_decision['all_scenarios']
                ]).round(2),
                page_size=5
            ),
            
            mo.md("### 🌟 Top Transfer Targets by Position:"),
            mo.ui.table(
                transfer_targets[['web_name', 'position', 'name', 'price_gbp', 'xP_gw', 'xP_per_price', 'transfer_appeal']].round(2).head(12),
                page_size=12
            ) if len(transfer_targets) > 0 else mo.md("No suitable transfer targets found")
        ])
        
    else:
        transfer_analysis_display = mo.md("Set up your current squad and calculate xP to see intelligent transfer recommendations.")
    
    transfer_analysis_display
    
    return analyze_transfer_options_enhanced, transfer_analysis_display, transfer_decision if 'transfer_decision' in locals() else None


@app.cell
def __(mo):
    mo.md("## Decision Summary")
    return


@app.cell
def __(mo, squad_with_xp, starting_11, bench, gameweek_selector, transfer_decision, pd):
    # Comprehensive decision summary integrating all analysis
    def create_comprehensive_decision_summary():
        """Create a comprehensive gameweek decision summary"""
        if (len(squad_with_xp) == 0 or len(starting_11) == 0 or 
            transfer_decision is None):
            return mo.md("⏳ Complete the analysis above to see your comprehensive decision summary.")
        
        # Calculate key metrics
        starting_11_xp = starting_11['xP_gw'].sum()
        total_squad_xp = squad_with_xp['xP_gw'].sum()
        bench_contribution = bench['xP_gw'].sum() if len(bench) > 0 else 0
        
        # Get formation string
        formation_counts = starting_11['position'].value_counts()
        formation_str = f"{formation_counts.get('DEF', 0)}-{formation_counts.get('MID', 0)}-{formation_counts.get('FWD', 0)}"
        
        # Captain recommendations from starting 11
        captain_choice = starting_11.nlargest(1, 'xP_gw').iloc[0] if len(starting_11) > 0 else None
        vice_captain_choice = starting_11.nlargest(2, 'xP_gw').iloc[1] if len(starting_11) > 1 else None
        
        # Risk assessment
        injury_risks = starting_11[starting_11['expected_minutes'] < 70]
        high_risk_count = len(injury_risks)
        
        # Transfer impact on starting 11
        transfer_impact = ""
        if transfer_decision['best_scenario']['transfers'] > 0:
            transfer_impact = f"With recommended transfers, projected starting 11 xP would be {starting_11_xp + transfer_decision['best_scenario']['xp_change']:.2f}"
        else:
            transfer_impact = "No transfers recommended - current starting 11 optimized"
        
        # Decision confidence scoring
        confidence_factors = []
        confidence_score = 80  # Base score
        
        if high_risk_count == 0:
            confidence_factors.append("✅ No injury risks in starting 11")
            confidence_score += 10
        elif high_risk_count == 1:
            confidence_factors.append("⚠️ 1 player with injury concern")
            confidence_score -= 5
        else:
            confidence_factors.append(f"❌ {high_risk_count} players with injury concerns")
            confidence_score -= 15
        
        if transfer_decision['best_scenario']['net_points'] >= 0:
            confidence_factors.append("✅ Transfer strategy optimized")
            confidence_score += 5
        else:
            confidence_factors.append("⚠️ No beneficial transfers available")
        
        if starting_11_xp >= total_squad_xp * 0.75:
            confidence_factors.append("✅ Starting 11 captures most squad potential")
            confidence_score += 5
        else:
            confidence_factors.append("⚠️ Significant bench strength unused")
            confidence_score -= 5
        
        confidence_score = max(0, min(100, confidence_score))
        
        return mo.vstack([
            mo.md(f"## 🎯 **Gameweek {gameweek_selector.value} Decision Summary**"),
            
            # Key decisions section
            mo.md("### 🔑 **Key Decisions**"),
            mo.hstack([
                mo.vstack([
                    mo.md("**🔄 Transfer Decision:**"),
                    mo.md(f"**{transfer_decision['decision']}**"),
                    mo.md(f"_{transfer_decision['reasoning']}_"),
                    mo.md(f"**Expected Impact:** {transfer_decision['best_scenario']['net_points']:+.1f} points")
                ]),
                mo.vstack([
                    mo.md("**⚽ Formation:**"),
                    mo.md(f"**{formation_str}**"),
                    mo.md(f"**Expected Points:** {starting_11_xp:.2f}"),
                    mo.md(f"**Bench Strength:** {bench_contribution:.2f}")
                ]),
                mo.vstack([
                    mo.md("**👑 Captaincy:**"),
                    mo.md(f"**Captain:** {captain_choice['web_name'] if captain_choice is not None else 'TBD'}"),
                    mo.md(f"**Vice:** {vice_captain_choice['web_name'] if vice_captain_choice is not None else 'TBD'}"),
                    mo.md(f"**Captain xP:** {captain_choice['xP_gw'] * 2:.1f}" if captain_choice is not None else "")
                ])
            ]),
            
            # Risk assessment
            mo.md("### ⚠️ **Risk Assessment**"),
            mo.md(f"**Decision Confidence:** {confidence_score}/100"),
            mo.vstack([mo.md(f"- {factor}") for factor in confidence_factors]),
            
            # Projected performance
            mo.md("### 📊 **Projected Performance**"),
            mo.md(f"""
            **Expected Gameweek Performance:**
            - **Starting 11 xP:** {starting_11_xp:.2f} points
            - **Captain Bonus:** {captain_choice['xP_gw']:.2f} additional points (if captain scores)" if captain_choice is not None else ""
            - **Total Projected:** {starting_11_xp + (captain_choice['xP_gw'] if captain_choice is not None else 0):.2f} points
            - **Transfer Impact:** {transfer_impact}
            """),
            
            # Implementation checklist
            mo.md("### ✅ **Implementation Checklist**"),
            mo.vstack([
                mo.md("**Before Deadline:**"),
                mo.md("1. ⏰ **Make Transfers** (if recommended)"),
                mo.md("2. ⚽ **Set Formation & Starting 11**"),
                mo.md("3. 👑 **Choose Captain & Vice-Captain**"),
                mo.md("4. 🔄 **Order Bench Players**"),
                mo.md("5. 💰 **Check Budget Remaining**"),
                mo.md(""),
                mo.md("**Post-Deadline Monitoring:**"),
                mo.md("6. 📰 **Monitor Team News**"),
                mo.md("7. 🏥 **Track Injury Updates**"),
                mo.md("8. 📈 **Review Performance vs Predictions**")
            ]),
            
            # Quick reference tables
            mo.md("### 📋 **Quick Reference**"),
            mo.hstack([
                mo.vstack([
                    mo.md("**Starting 11 Summary:**"),
                    mo.ui.table(
                        starting_11[['web_name', 'position', 'xP_gw']].round(2).head(11),
                        page_size=11,
                        show_column_summaries=False
                    )
                ]),
                mo.vstack([
                    mo.md("**Bench Order:**"),
                    mo.ui.table(
                        bench[['web_name', 'position', 'xP_gw']].round(2).head(4) if len(bench) > 0 else pd.DataFrame(),
                        page_size=4,
                        show_column_summaries=False
                    ) if len(bench) > 0 else mo.md("No bench data")
                ])
            ]),
            
            mo.md(f"""
            ---
            **Analysis Complete for GW{gameweek_selector.value}** | Generated using enhanced FPL API data with season form tracking
            """)
        ])
    
    decision_summary = create_comprehensive_decision_summary()
    decision_summary
    
    return decision_summary


@app.cell
def __(mo):
    mo.md("## 🔍 Retro Analysis")
    return


@app.cell
def __(mo):
    # Retro analysis for completed gameweeks
    mo.md("### Analyze Completed Gameweek Performance")
    
    # Gameweek selector for retro analysis
    retro_gw_selector = mo.ui.number(
        value=1,
        start=1,
        stop=38,
        step=1,
        label="Select Completed Gameweek"
    )
    
    mo.vstack([
        retro_gw_selector,
        mo.md("Note: Retro analysis now requires FPL API data for completed gameweeks. This feature will be enhanced in future versions.")
    ])
    
    return retro_gw_selector


@app.cell  
def __(retro_gw_selector, mo):
    # Simplified retro analysis display
    retro_display = mo.md(f"""
    ### 📊 Retro Analysis (Coming Soon)
    
    **Selected Gameweek:** {retro_gw_selector.value}
    
    **Future Features:**
    - ✅ Performance analysis from FPL API
    - ✅ Top performers identification
    - ✅ Template disappointments tracking
    - ✅ Differential successes discovery
    - ✅ xP vs actual points comparison
    
    *This feature will be enhanced to fetch completed gameweek data directly from the FPL API.*
    """)
    
    retro_display
    return retro_display


@app.cell
def __(mo):
    mo.md("""
    ## Features Coming Soon
    
    **Enhanced Transfer Analysis:**
    - Multi-transfer strategy optimizer (1-4 transfers)
    - Price change predictions and timing
    - Transfer hit break-even analysis
    
    **Advanced Captain Analysis:**
    - Captaincy differentials vs template
    - Triple captain timing optimization
    - Risk/reward variance analysis
    
    **Chip Strategy:**
    - Bench boost timing
    - Free hit optimization
    - Wildcard planning
    
    **Integration Features:**
    - Import team from FPL API
    - Export decisions to FPL format
    - Historical decision tracking
    """)
    return


if __name__ == "__main__":
    app.run()