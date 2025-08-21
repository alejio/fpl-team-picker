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
        # FPL Gameweek Manager (XP-Optimized)

        **Advanced Weekly Decision Making Tool with Form Analytics**
        
        1. Load team for previous gameweek from database
        2. Calculate form-weighted expected XP for all players
        3. **NEW**: Form analytics dashboard with hot/cold player insights
        4. **NEW**: Current squad form analysis and transfer recommendations
        5. Set constraints and run smart optimization (auto-selects 0-3 transfers)
        6. Analyze captain and vice-captain options
        
        *Built with advanced XP modeling, form weighting, and optimization algorithms for competitive advantage!*
        
        **🔥 New Features:** Hot/Cold player detection, form multipliers, transfer risk analysis
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import warnings
    warnings.filterwarnings('ignore')
    
    return pd, np, datetime, warnings


@app.cell
def __(mo):
    mo.md("## 1. Configure Gameweek")
    return


@app.cell
def __(pd):
    def fetch_fpl_data(target_gameweek):
        """Fetch FPL data from database for specified gameweek with historical form data"""
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
        
        # Get historical live data for form calculation (last 5 gameweeks)
        form_window = 5
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

    return fetch_fpl_data


@app.cell
def __(mo):
    gameweek_input = mo.ui.number(
        value=2,
        start=2,
        stop=38,
        label="Target Gameweek (we'll optimize for this GW using data from GW-1)"
    )
    
    mo.vstack([
        mo.md("**Select Target Gameweek:**"),
        mo.md("_We'll load your team from the previous gameweek and optimize for the target gameweek_"),
        gameweek_input
    ])
    
    return (gameweek_input,)


@app.cell
def __(fetch_fpl_data, gameweek_input, mo, pd):
    # Load data when gameweek is provided
    if gameweek_input.value:
        # Load data for target gameweek
        target_gw = gameweek_input.value
        previous_gw = target_gw - 1
        
        # Get FPL data and calculate XP for target gameweek
        players, teams, xg_rates, fixtures, _, live_data_historical = fetch_fpl_data(target_gw)
        
        # Load manager team from previous gameweek
        def fetch_manager_team(previous_gameweek):
            """Fetch manager's team from previous gameweek"""
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
        
        team_data = fetch_manager_team(previous_gw)
        
        if team_data:
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
            
            team_display = mo.vstack([
                mo.md(f"### ✅ {team_data['entry_name']} (from GW{previous_gw})"),
                mo.md(f"**Previous Points:** {team_data['total_points']:,} | **Bank:** £{team_data['bank']:.1f}m | **Value:** £{team_data['team_value']:.1f}m | **Free Transfers:** {team_data['free_transfers']}"),
                mo.md(f"**Optimizing for GW{target_gw}**"),
                mo.md("**Current Squad:**"),
                mo.ui.table(
                    current_squad[['web_name', 'position', 'name', 'price', 'role', 'status']].round(2),
                    page_size=15
                )
            ])
        else:
            team_display = mo.md("❌ Could not load team data")
            current_squad = pd.DataFrame()
            team_data = None
            players = pd.DataFrame()
            teams = pd.DataFrame() 
            xg_rates = pd.DataFrame()
            fixtures = pd.DataFrame()
    else:
        team_display = mo.md("Select target gameweek above to load team")
        current_squad = pd.DataFrame()
        team_data = None
        players = pd.DataFrame()
        teams = pd.DataFrame() 
        xg_rates = pd.DataFrame()
        fixtures = pd.DataFrame()
        live_data_historical = pd.DataFrame()
    
    team_display
    
    return current_squad, team_data, players, teams, xg_rates, fixtures, live_data_historical


@app.cell
def __(mo):
    mo.md("## 2. Team Strength Analysis")
    return


@app.cell
def __(gameweek_input, mo, pd):
    def create_team_strength_visualization(target_gameweek):
        """Create team strength visualization showing current ratings"""
        try:
            from client import get_current_teams
            from dynamic_team_strength import DynamicTeamStrength, load_historical_gameweek_data
            
            # Load team data
            teams = get_current_teams()
            
            # Get dynamic team strength ratings for the target gameweek
            calculator = DynamicTeamStrength(debug=False)
            current_season_data = load_historical_gameweek_data(start_gw=1, end_gw=target_gameweek-1)
            team_strength = calculator.get_team_strength(
                target_gameweek=target_gameweek,
                teams_data=teams,
                current_season_data=current_season_data
            )
            
            # Create team strength dataframe
            strength_data = []
            for team_name, strength in team_strength.items():
                team_id = teams[teams['name'] == team_name]['team_id'].iloc[0] if len(teams[teams['name'] == team_name]) > 0 else None
                
                # Categorize strength
                if strength >= 1.15:
                    category = "🔴 Very Strong"
                    difficulty_as_opponent = "Very Hard"
                elif strength >= 1.05:
                    category = "🟠 Strong" 
                    difficulty_as_opponent = "Hard"
                elif strength >= 0.95:
                    category = "🟡 Average"
                    difficulty_as_opponent = "Average"
                elif strength >= 0.85:
                    category = "🟢 Weak"
                    difficulty_as_opponent = "Easy"
                else:
                    category = "🟢 Very Weak"
                    difficulty_as_opponent = "Very Easy"
                
                strength_data.append({
                    'Team': team_name,
                    'Strength': round(strength, 3),
                    'Category': category,
                    'As Opponent': difficulty_as_opponent,
                    'Attack Rating': round(strength * 1.0, 3),  # Simplified for display
                    'Defense Rating': round(2.0 - strength, 3)  # Inverse relationship
                })
            
            # Sort by strength (strongest first)
            strength_df = pd.DataFrame(strength_data).sort_values('Strength', ascending=False)
            strength_df['Rank'] = range(1, len(strength_df) + 1)
            
            # Reorder columns
            display_df = strength_df[['Rank', 'Team', 'Strength', 'Category', 'As Opponent', 'Attack Rating', 'Defense Rating']]
            
            # Create summary stats
            strongest_team = strength_df.iloc[0]
            weakest_team = strength_df.iloc[-1]
            avg_strength = strength_df['Strength'].mean()
            
            return mo.vstack([
                mo.md(f"### 🏆 Team Strength Ratings - GW{target_gameweek}"),
                mo.md(f"**Strongest:** {strongest_team['Team']} ({strongest_team['Strength']:.3f}) | **Weakest:** {weakest_team['Team']} ({weakest_team['Strength']:.3f}) | **Average:** {avg_strength:.3f}"),
                mo.md("*Higher strength = better team = harder opponent when playing against them*"),
                mo.md("**📊 Dynamic Team Strength Table:**"),
                mo.ui.table(display_df, page_size=20),
                mo.md("""
                **💡 How to Use This:**
                - **🔴 Very Strong teams**: Avoid their opponents (hard fixtures) 
                - **🟢 Weak teams**: Target their opponents (easy fixtures)
                - **Attack Rating**: Expected goals scoring ability
                - **Defense Rating**: Expected goals conceded (lower = better defense)
                """)
            ])
            
        except Exception as e:
            return mo.md(f"❌ **Could not create team strength analysis:** {e}")
    
    # Only show if gameweek is selected
    if gameweek_input.value:
        team_strength_analysis = create_team_strength_visualization(gameweek_input.value)
    else:
        team_strength_analysis = mo.md("Select target gameweek to see team strength analysis")
    
    team_strength_analysis
    
    return (team_strength_analysis,)


@app.cell
def __(mo):
    mo.md("## 3. Calculate Expected Points (XP) for All Players")
    return


@app.cell
def __(players, teams, xg_rates, fixtures, live_data_historical, gameweek_input, mo, pd):
    def calculate_expected_points_all_players(players_data, teams_data, xg_rates_data, fixtures_data, target_gameweek, live_data_hist=None):
        """Calculate expected points for all players using improved XP model with form weighting"""
        if players_data.empty or not target_gameweek:
            empty_df = pd.DataFrame(columns=['web_name', 'position', 'name', 'price', 'player_id', 'xP', 'xP_5gw'])
            return mo.md("Select gameweek and load data first"), empty_df
        
        print(f"🧠 Calculating expected points using improved XP model for GW{target_gameweek}...")
        print(f"📊 Input data: {len(players_data)} players, {len(teams_data)} teams, {len(fixtures_data)} fixtures")
        
        try:
            # Import and use the new XP model
            print("🔄 Importing XP model...")
            from xp_model import XPModel
            
            # Initialize model with form weighting enabled
            print("🔄 Initializing XP model...")
            xp_model = XPModel(
                form_weight=0.7,  # 70% recent form, 30% season average
                form_window=5,    # Last 5 gameweeks for form
                debug=True
            )
            print("✅ XP model initialized successfully")
            
            # Calculate both single-GW and 5-GW XP for strategic comparison
            print(f"🔮 Calculating single-gameweek XP for tactical analysis...")
            players_1gw = xp_model.calculate_expected_points(
                players_data=players_data,
                teams_data=teams_data,
                xg_rates_data=xg_rates_data,
                fixtures_data=fixtures_data,
                target_gameweek=target_gameweek,
                live_data=live_data_hist,  # Historical form data
                gameweeks_ahead=1
            )
            
            print(f"🎯 Calculating 5-gameweek strategic horizon XP...")
            players_5gw = xp_model.calculate_expected_points(
                players_data=players_data,
                teams_data=teams_data,
                xg_rates_data=xg_rates_data,
                fixtures_data=fixtures_data,
                target_gameweek=target_gameweek,
                live_data=live_data_hist,  # Historical form data
                gameweeks_ahead=5
            )
            
            try:
                # Debug: Check available columns and data integrity
                print(f"🔍 1-GW columns: {list(players_1gw.columns)}")
                print(f"🔍 5-GW columns: {list(players_5gw.columns)}")
                print(f"🔍 1-GW shape: {players_1gw.shape}")
                print(f"🔍 5-GW shape: {players_5gw.shape}")
                
                # Merge both calculations for comparison
                # First check if player_id column exists in both dataframes
                if 'player_id' in players_1gw.columns and 'player_id' in players_5gw.columns:
                    # Check which columns are actually available for merging
                    merge_cols = ['player_id']
                    for col in ['xP', 'xP_per_price', 'fixture_difficulty']:
                        if col in players_5gw.columns:
                            merge_cols.append(col)
                    
                    print(f"🔗 Merging columns: {merge_cols}")
                    
                    # Create suffix mapping to avoid conflicts
                    suffix_data = players_5gw[merge_cols].copy()
                    for col in merge_cols:
                        if col != 'player_id':
                            suffix_data[f"{col}_5gw"] = suffix_data[col]
                            suffix_data = suffix_data.drop(col, axis=1)
                    
                    players_xp = players_1gw.merge(
                        suffix_data,
                        on='player_id',
                        how='left'
                    )
                    print(f"✅ Successfully merged data for {len(players_xp)} players")
                    
                else:
                    print("⚠️ Warning: player_id column missing, using single-GW data only")
                    players_xp = players_1gw.copy()
                    # Add placeholder 5-GW columns
                    players_xp['xP_5gw'] = players_xp['xP'] * 4.0  # Approximate scaling
                    players_xp['xP_per_price_5gw'] = players_xp.get('xP_per_price', players_xp['xP']) * 4.0
                    players_xp['fixture_difficulty_5gw'] = players_xp.get('fixture_difficulty', 1.0)
                    
            except Exception as e:
                print(f"⚠️ Error merging 1-GW and 5-GW data: {e}")
                print("🔧 Using single-GW data with estimated 5-GW projections")
                players_xp = players_1gw.copy()
                # Add placeholder 5-GW columns
                players_xp['xP_5gw'] = players_xp['xP'] * 4.0  # Approximate scaling
                players_xp['xP_per_price_5gw'] = players_xp.get('xP_per_price', players_xp['xP']) * 4.0
                players_xp['fixture_difficulty_5gw'] = players_xp.get('fixture_difficulty', 1.0)
            
            # Add strategic metrics with error handling
            try:
                players_xp['xP_horizon_advantage'] = players_xp['xP_5gw'] - (players_xp['xP'] * 5)  # 5-GW advantage over simple scaling
                players_xp['fixture_outlook'] = players_xp['fixture_difficulty_5gw'].apply(
                    lambda x: '🟢 Easy' if x >= 1.15 else '🟡 Average' if x >= 0.85 else '🔴 Hard'
                )
                print(f"✅ Added strategic metrics to {len(players_xp)} players")
            except Exception as e:
                print(f"⚠️ Error adding strategic metrics: {e}")
                players_xp['xP_horizon_advantage'] = 0
                players_xp['fixture_outlook'] = '🟡 Average'
            
            # Add comprehensive player attributes for display
            # Start with core player info
            display_columns = ['web_name', 'position', 'name', 'price', 'selected_by_percent']
            
            # Add XP model outputs (both 1-GW and 5-GW)
            xp_columns = ['xP', 'xP_5gw', 'xP_per_price', 'xP_per_price_5gw', 'expected_minutes', 'fixture_difficulty', 'fixture_difficulty_5gw', 'xP_horizon_advantage', 'fixture_outlook']
            display_columns.extend(xp_columns)
            
            # Add XP component breakdown if available
            xp_components = ['xP_appearance', 'xP_goals', 'xP_assists', 'xP_clean_sheets']
            for component in xp_components:
                if component in players_xp.columns:
                    display_columns.append(component)
            
            # Add form indicators if they exist
            form_columns = ['momentum', 'form_multiplier', 'recent_points_per_game']
            for form_col in form_columns:
                if form_col in players_xp.columns:
                    display_columns.append(form_col)
            
            # Add statistical data if available
            stats_columns = ['xG90', 'xA90', 'minutes', 'p_60_plus_mins']
            for stat_col in stats_columns:
                if stat_col in players_xp.columns:
                    display_columns.append(stat_col)
            
            # Add availability status at the end
            if 'status' in players_xp.columns:
                display_columns.append('status')
            
            # Filter to only include columns that actually exist in the dataframe
            form_columns = [col for col in display_columns if col in players_xp.columns]
            
            # Create display table (sorted by 5-GW XP for strategic decisions)
            display_df = players_xp[form_columns].sort_values('xP_5gw', ascending=False).round(3)
            
            # Count form-enhanced players
            form_enhanced_players = 0
            avg_form_multiplier = 1.0
            if 'form_multiplier' in players_xp.columns:
                form_enhanced_players = len(players_xp[players_xp['form_multiplier'] != 1.0])
                avg_form_multiplier = players_xp['form_multiplier'].mean()
            
            # Enhanced display with form information
            form_info = ""
            if form_enhanced_players > 0:
                hot_players = len(players_xp[players_xp.get('momentum', '') == '🔥'])
                cold_players = len(players_xp[players_xp.get('momentum', '') == '❄️'])
                form_info = f"**Form Analysis:** {form_enhanced_players} players with form data | {hot_players} 🔥 Hot | {cold_players} ❄️ Cold | Avg multiplier: {avg_form_multiplier:.2f}"
            
            # Strategic analysis with error handling
            try:
                fixture_advantage_players = len(players_xp[players_xp['xP_horizon_advantage'] > 0.5])
                fixture_disadvantage_players = len(players_xp[players_xp['xP_horizon_advantage'] < -0.5])
                
                # Check if we have enough data for comparison
                leaders_different = False
                if not display_df.empty and 'xP' in players_xp.columns:
                    top_5gw_player = display_df.iloc[0]['web_name']
                    top_1gw_players = players_xp.nlargest(1, 'xP')
                    if not top_1gw_players.empty:
                        top_1gw_player = top_1gw_players.iloc[0]['web_name']
                        leaders_different = top_5gw_player != top_1gw_player
                
                strategic_info = f"""
                **🎯 Strategic 5-Gameweek Analysis:**
                - Players with fixture advantage (5-GW > 1-GW): {fixture_advantage_players}
                - Players with fixture disadvantage (tough upcoming): {fixture_disadvantage_players}
                - 5-GW leaders different from 1-GW: {leaders_different}
                """
            except Exception as e:
                print(f"⚠️ Error in strategic analysis: {e}")
                strategic_info = "**🎯 Strategic 5-Gameweek Analysis:** Analysis unavailable"
            
            return mo.vstack([
                mo.md(f"### ✅ Strategic XP Model - GW{target_gameweek} + 5 Horizon"),
                mo.md(f"**Players analyzed:** {len(players_xp)}"),
                mo.md(f"**Average 1-GW XP:** {players_xp['xP'].mean():.2f} | **Average 5-GW XP:** {players_xp['xP_5gw'].mean():.2f}"),
                mo.md(f"**Top 1-GW:** {players_xp['xP'].max():.2f} | **Top 5-GW:** {players_xp['xP_5gw'].max():.2f}"),
                mo.md(strategic_info),
                mo.md(form_info) if form_info else mo.md(""),
                mo.md("**All Players - Strategic Comparison (Sorted by 5-GW XP):**"),
                mo.md("*Showing: 1-GW vs 5-GW XP, fixture outlook, horizon advantage, form data*"),
                mo.ui.table(display_df, page_size=25)
            ]), players_xp
            
        except ImportError as e:
            print(f"⚠️ New XP model not available: {e}")
            # Return empty dataframe with basic structure to prevent downstream errors
            empty_df = pd.DataFrame(columns=['web_name', 'position', 'name', 'price', 'player_id', 'xP', 'xP_5gw'])
            return mo.md("❌ Could not load improved XP model - check xp_model.py"), empty_df
        except Exception as e:
            print(f"❌ Error calculating XP: {e}")
            # Return empty dataframe with basic structure to prevent downstream errors
            empty_df = pd.DataFrame(columns=['web_name', 'position', 'name', 'price', 'player_id', 'xP', 'xP_5gw'])
            return mo.md(f"❌ Error calculating expected points: {e}"), empty_df
    
    # Debug current state
    print(f"🔍 XP Calculation Debug:")
    print(f"  - players.empty: {players.empty if hasattr(players, 'empty') else 'not available'}")
    print(f"  - gameweek_input.value: {gameweek_input.value if hasattr(gameweek_input, 'value') else 'not available'}")
    print(f"  - players shape: {players.shape if hasattr(players, 'shape') else 'not available'}")
    
    # Only calculate XP if we have valid data
    try:
        if not players.empty and gameweek_input.value:
            print(f"🚀 Starting XP calculation for {len(players)} players...")
            xp_result, players_with_xp = calculate_expected_points_all_players(players, teams, xg_rates, fixtures, gameweek_input.value, live_data_historical)
            print(f"✅ XP calculation completed, returned {len(players_with_xp)} players with XP data")
            print(f"📊 players_with_xp type: {type(players_with_xp)}")
            print(f"📊 players_with_xp shape: {players_with_xp.shape if hasattr(players_with_xp, 'shape') else 'no shape'}")
        else:
            print(f"❌ XP calculation skipped - players empty: {players.empty}, gameweek: {gameweek_input.value}")
            xp_result = mo.md("Load gameweek data first")
            # Create empty dataframe with expected structure to prevent downstream errors
            players_with_xp = pd.DataFrame(columns=['web_name', 'position', 'name', 'price', 'player_id', 'xP', 'xP_5gw', 'fixture_outlook'])
            print("⚠️ No data loaded, using empty dataframe")
    except Exception as e:
        print(f"❌ CRITICAL ERROR in XP calculation cell: {e}")
        import traceback
        traceback.print_exc()
        xp_result = mo.md(f"❌ Critical error in XP calculation: {str(e)}")
        # Ensure we always return a valid dataframe
        players_with_xp = pd.DataFrame(columns=['web_name', 'position', 'name', 'price', 'player_id', 'xP', 'xP_5gw', 'fixture_outlook'])
    
    # Display the result
    xp_result
    
    return players_with_xp


@app.cell
def __(mo):
    mo.md("## 4. Form Analytics Dashboard")
    return


@app.cell
def __(players_with_xp, mo):
    # Form Analytics Dashboard - Fixed for marimo rendering
    
    # Check if we have form data
    if not players_with_xp.empty and 'momentum' in players_with_xp.columns:
        # Hot players analysis
        hot_players = players_with_xp[players_with_xp['momentum'] == '🔥'].nlargest(8, 'xP')
        cold_players = players_with_xp[players_with_xp['momentum'] == '❄️'].nsmallest(8, 'form_multiplier')
        
        # Value analysis with form
        value_players = players_with_xp[
            (players_with_xp['momentum'].isin(['🔥', '📈'])) & 
            (players_with_xp['price'] <= 7.5)
        ].nlargest(10, 'xP_per_price')
        
        # Expensive underperformers
        expensive_poor = players_with_xp[
            (players_with_xp['price'] >= 8.0) & 
            (players_with_xp['momentum'].isin(['❄️', '📉']))
        ].nsmallest(6, 'form_multiplier')
        
        insights = []
        
        if len(hot_players) > 0:
            insights.extend([
                mo.md("### 🔥 Hot Players (Prime Transfer Targets)"),
                mo.md(f"**{len(hot_players)} players in excellent recent form**"),
                mo.ui.table(
                    hot_players[['web_name', 'position', 'name', 'price', 'xP', 'recent_points_per_game', 'momentum']].round(2),
                    page_size=8
                )
            ])
        
        if len(value_players) > 0:
            insights.extend([
                mo.md("### 💎 Form + Value Players (Budget-Friendly Options)"),
                mo.md(f"**{len(value_players)} players with good form and great value**"),
                mo.ui.table(
                    value_players[['web_name', 'position', 'name', 'price', 'xP', 'xP_per_price', 'momentum']].round(3),
                    page_size=10
                )
            ])
        
        if len(cold_players) > 0:
            insights.extend([
                mo.md("### ❄️ Cold Players (Sell Candidates)"),
                mo.md(f"**{len(cold_players)} players in poor recent form - consider selling**"),
                mo.ui.table(
                    cold_players[['web_name', 'position', 'name', 'price', 'xP', 'recent_points_per_game', 'momentum']].round(2),
                    page_size=8
                )
            ])
        
        if len(expensive_poor) > 0:
            insights.extend([
                mo.md("### 💸 Expensive Underperformers (Priority Sells)"),
                mo.md(f"**{len(expensive_poor)} expensive players in poor form - sell to fund transfers**"),
                mo.ui.table(
                    expensive_poor[['web_name', 'position', 'name', 'price', 'xP', 'recent_points_per_game', 'momentum']].round(2),
                    page_size=6
                )
            ])
        
        # Summary stats
        if 'form_multiplier' in players_with_xp.columns:
            momentum_counts = players_with_xp['momentum'].value_counts()
            avg_multiplier = players_with_xp['form_multiplier'].mean()
            
            insights.append(mo.md(f"""
            ### 📊 Form Summary
            **Player Distribution:** 🔥 {momentum_counts.get('🔥', 0)} | 📈 {momentum_counts.get('📈', 0)} | ➡️ {momentum_counts.get('➡️', 0)} | 📉 {momentum_counts.get('📉', 0)} | ❄️ {momentum_counts.get('❄️', 0)}
            
            **Average Form Multiplier:** {avg_multiplier:.2f}
            
            **Transfer Strategy:** Target 🔥 hot and 📈 rising players, avoid ❄️ cold and 📉 declining players
            """))
        
        form_insights_display = mo.vstack(insights)
    else:
        form_insights_display = mo.md("⚠️ **No form data available** - load historical data first")
    
    form_insights_display


@app.cell
def __(current_squad, players_with_xp, mo):
    # Current Squad Form Analysis - Fixed for marimo rendering
    
    # Check data availability without early returns
    squad_available = hasattr(current_squad, 'empty') and not current_squad.empty
    players_available = hasattr(players_with_xp, 'empty') and not players_with_xp.empty
    momentum_available = hasattr(players_with_xp, 'columns') and 'momentum' in players_with_xp.columns
    
    # Create content based on data availability
    if squad_available and players_available and momentum_available:
        # Merge squad with form data
        squad_with_form = current_squad.merge(
            players_with_xp[['player_id', 'xP', 'momentum', 'form_multiplier', 'recent_points_per_game']], 
            on='player_id', 
            how='left'
        )
        
        if not squad_with_form.empty:
            # Squad form analysis
            squad_insights = []
            
            # Count momentum distribution in squad
            squad_momentum = squad_with_form['momentum'].value_counts()
            squad_avg_form = squad_with_form['form_multiplier'].mean()
            
            # Identify problem players in squad
            problem_players = squad_with_form[
                squad_with_form['momentum'].isin(['❄️', '📉'])
            ].sort_values('form_multiplier')
            
            # Identify top performers in squad  
            top_performers = squad_with_form[
                squad_with_form['momentum'].isin(['🔥', '📈'])
            ].sort_values('xP', ascending=False)
            
            squad_insights.append(mo.md("### 🔍 Current Squad Form Analysis"))
            
            # Squad overview
            squad_insights.append(mo.md(f"""
            **Your Squad Form Distribution:**
            - 🔥 Hot: {squad_momentum.get('🔥', 0)} players - 📈 Rising: {squad_momentum.get('📈', 0)} players
            - ➡️ Stable: {squad_momentum.get('➡️', 0)} players - 📉 Declining: {squad_momentum.get('📉', 0)} players - ❄️ Cold: {squad_momentum.get('❄️', 0)} players
            
            **Squad Average Form Multiplier:** {squad_avg_form:.2f}
            """))
            
            # Squad health assessment
            hot_count = squad_momentum.get('🔥', 0) + squad_momentum.get('📈', 0)
            cold_count = squad_momentum.get('❄️', 0) + squad_momentum.get('📉', 0)
            
            if hot_count >= 8:
                health_status = "🔥 EXCELLENT - Squad is in great form, minimal transfers needed"
                transfer_priority = "Low"
            elif hot_count >= 5:
                health_status = "📈 GOOD - Squad form is solid, consider tactical transfers"
                transfer_priority = "Medium"
            elif cold_count >= 8:
                health_status = "❄️ POOR - Squad struggling, multiple transfers recommended"
                transfer_priority = "High"
            else:
                health_status = "➡️ AVERAGE - Squad form is stable, monitor for improvements"
                transfer_priority = "Low"
            
            squad_insights.append(mo.md("### 🎯 Squad Health Assessment"))
            squad_insights.append(mo.md(f"""
            {health_status}
            
            **Transfer Priority:** {transfer_priority}
            """))
            
            # Show top performers if any
            if not top_performers.empty:
                squad_insights.append(mo.md("### ⭐ Squad Stars (Keep!)"))
                squad_insights.append(mo.ui.table(
                    top_performers[['web_name', 'position', 'momentum', 'recent_points_per_game', 'xP']].round(2),
                    page_size=5
                ))
            
            # Show problem players if any
            if not problem_players.empty:
                squad_insights.append(mo.md("### ⚠️ Problem Players (Consider Selling)"))
                squad_insights.append(mo.ui.table(
                    problem_players[['web_name', 'position', 'momentum', 'recent_points_per_game', 'xP']].round(2),
                    page_size=5
                ))
            
            squad_form_content = mo.vstack(squad_insights)
        else:
            squad_form_content = mo.md("⚠️ **Could not merge squad with form data**")
    else:
        # Debug information for missing data
        debug_parts = []
        if not squad_available:
            debug_parts.append("No squad loaded")
        if not players_available:
            debug_parts.append("No player XP data")
        if not momentum_available:
            debug_parts.append("No form/momentum data")
        
        debug_msg = ", ".join(debug_parts)
        squad_form_content = mo.md(f"⚠️ **Squad form analysis unavailable**\n\n_{debug_msg}_")
    
    squad_form_content


@app.cell
def __(mo):
    mo.md("## 5. Player Performance Trends")
    return


@app.cell
def __(mo, pd, players_with_xp):
    def create_player_trends_visualization(players_data):
        """Create interactive player trends visualization"""
        try:
            # Get historical data for trends
            from client import get_gameweek_live_data
            import plotly.graph_objects as go
            import plotly.express as px
            
            # Load historical gameweek data (check available gameweeks)
            historical_data = []
            max_gw = 10  # Adjust based on season progress
            
            print(f"🔍 Loading historical data for trends...")
            
            for gw in range(1, max_gw + 1):
                try:
                    gw_data = get_gameweek_live_data(gw)
                    if not gw_data.empty:
                        gw_data['gameweek'] = gw
                        historical_data.append(gw_data)
                        print(f"✅ GW{gw}: {len(gw_data)} player records")
                    else:
                        print(f"❌ GW{gw}: No data")
                        break  # Stop if we hit missing data
                except Exception as e:
                    print(f"❌ GW{gw}: Error - {e}")
                    break
            
            if not historical_data:
                return [], [], pd.DataFrame()
            
            print(f"📊 Total historical datasets: {len(historical_data)}")
            
            # Combine all historical data
            all_data = pd.concat(historical_data, ignore_index=True)
            print(f"📊 Combined data shape: {all_data.shape}")
            
            # CRITICAL: Merge with current player names 
            # The live data only has player_id, we need names from current players
            if not players_data.empty:
                print(f"🔗 Merging with players data to get names...")
                player_names = players_data[['player_id', 'web_name', 'position', 'team', 'name']].drop_duplicates('player_id')
                print(f"📋 Player names data shape: {player_names.shape}")
                
                # Include current price info too (handle missing columns gracefully)
                base_cols = ['player_id', 'web_name', 'position', 'team', 'name']
                optional_cols = ['now_cost', 'selected_by_percent']
                
                # Only include columns that exist
                available_cols = base_cols + [col for col in optional_cols if col in players_data.columns]
                player_info = players_data[available_cols].drop_duplicates('player_id')
                
                print(f"📊 Using player info columns: {available_cols}")
                
                all_data = all_data.merge(
                    player_info,
                    on='player_id',
                    how='left'
                )
                print(f"📊 After merge shape: {all_data.shape}")
                print(f"📊 Players with names: {all_data['web_name'].notna().sum()}")
                print(f"📊 Sample merged data columns: {list(all_data.columns)}")
            else:
                print("⚠️ No players data available for name merge!")
            
            # Calculate derived metrics (handle missing columns gracefully)
            all_data['points_per_game'] = all_data['total_points'] / all_data['gameweek']
            all_data['xg_per_90'] = (all_data.get('expected_goals', 0) / all_data.get('minutes', 1)) * 90
            all_data['xa_per_90'] = (all_data.get('expected_assists', 0) / all_data.get('minutes', 1)) * 90
            
            # Value ratio - only calculate if we have price data
            if 'now_cost' in all_data.columns:
                all_data['value_ratio'] = all_data['total_points'] / (all_data['now_cost'] / 10)  # Points per £1m
            else:
                print("⚠️ No price data available, skipping value ratio calculation")
                all_data['value_ratio'] = 0
            
            # Debug: Check for specific players (try different name variations)
            ekitike_data = all_data[all_data['web_name'].str.contains('Ekitike', case=False, na=False)]
            if not ekitike_data.empty:
                print(f"🎯 Found Ekitike data: {ekitike_data[['web_name', 'gameweek', 'total_points', 'goals_scored']].to_dict('records')}")
            else:
                # Try other variations
                ekit_data = all_data[all_data['web_name'].str.contains('Ekit', case=False, na=False)]
                if not ekit_data.empty:
                    print(f"🎯 Found Ekit* player: {ekit_data[['web_name', 'gameweek', 'total_points', 'goals_scored']].to_dict('records')}")
                else:
                    print(f"❌ No Ekitike/Ekit player found")
                    # Show players with high goals_scored to see if he's there under different name
                    high_scorers = all_data[all_data['goals_scored'] >= 2]
                    if not high_scorers.empty:
                        print(f"🥅 Players with 2+ goals: {high_scorers[['web_name', 'goals_scored']].to_dict('records')}")
                    print(f"📝 Sample player names: {all_data['web_name'].dropna().head(10).tolist()}")
            
            # Create player options (filter to players with data)
            player_options = []
            
            # Get players with the most total points and data availability
            valid_players = all_data[all_data['web_name'].notna()]
            print(f"📊 Valid players with names: {len(valid_players)}")
            
            if valid_players.empty:
                print("❌ No players with valid names found!")
                return [], [], pd.DataFrame()
                
            player_stats = valid_players.groupby('player_id').agg({
                'total_points': 'max',
                'web_name': 'first',
                'position': 'first', 
                'name': 'first',
                'gameweek': 'count'  # Number of gameweeks with data
            }).reset_index()
            
            # Filter to players with at least some data and sort by points
            active_players = player_stats[player_stats['gameweek'] >= 1].nlargest(100, 'total_points')
            
            for _, player_row in active_players.iterrows():
                label = f"{player_row['web_name']} ({player_row['position']}, {player_row['name']}) - {player_row['total_points']} pts"
                player_options.append({"label": label, "value": int(player_row['player_id'])})
            
            print(f"📊 Created {len(player_options)} player options for trends")
            
            # Create attribute options (only include available columns)
            base_attributes = [
                {"label": "Total Points", "value": "total_points"},
                {"label": "Points Per Game", "value": "points_per_game"},
                {"label": "Minutes Played", "value": "minutes"},
                {"label": "Goals Scored", "value": "goals_scored"},
                {"label": "Assists", "value": "assists"},
                {"label": "Expected Goals (xG)", "value": "expected_goals"},
                {"label": "Expected Assists (xA)", "value": "expected_assists"},
                {"label": "xG per 90", "value": "xg_per_90"},
                {"label": "xA per 90", "value": "xa_per_90"},
                {"label": "Clean Sheets", "value": "clean_sheets"},
                {"label": "Yellow Cards", "value": "yellow_cards"},
                {"label": "Red Cards", "value": "red_cards"},
                {"label": "Bonus Points", "value": "bonus"},
                {"label": "ICT Index", "value": "ict_index"}
            ]
            
            # Add optional attributes if they exist in the data
            optional_attributes = [
                {"label": "Price (£m)", "value": "now_cost"},
                {"label": "Selected By %", "value": "selected_by_percent"},
                {"label": "Value Ratio (pts/£1m)", "value": "value_ratio"}
            ]
            
            attribute_options = base_attributes.copy()
            for attr in optional_attributes:
                if attr["value"] in all_data.columns:
                    attribute_options.append(attr)
                    
            print(f"📊 Available attributes: {[attr['value'] for attr in attribute_options]}")
            
            return player_options, attribute_options, all_data
            
        except Exception as e:
            print(f"Error creating trends data: {e}")
            return [], [], pd.DataFrame()
    
    # Create the trend analysis components
    if hasattr(players_with_xp, 'empty') and not players_with_xp.empty:
        try:
            print(f"🔍 Creating trends visualization with {len(players_with_xp)} players")
            player_opts, attr_opts, trends_data = create_player_trends_visualization(players_with_xp)
            print(f"✅ Trends created: {len(player_opts)} players, {len(attr_opts)} attributes")
        except Exception as e:
            print(f"❌ Error creating trends: {e}")
            player_opts, attr_opts, trends_data = [], [], pd.DataFrame()
    else:
        print("❌ No players_with_xp data available - calculate XP first")
        player_opts, attr_opts, trends_data = [], [], pd.DataFrame()
    
    return player_opts, attr_opts, trends_data


@app.cell
def __(mo, player_opts, attr_opts):
    # Player and attribute selectors
    if player_opts and attr_opts:
        player_selector = mo.ui.dropdown(
            options=player_opts[:50],  # Limit to top 50 for performance
            label="Select Player:",
            value=None  # Don't set default value to avoid validation issues
        )
        
        attribute_selector = mo.ui.dropdown(
            options=attr_opts,
            label="Select Attribute:",
            value=None  # Don't set default value to avoid validation issues
        )
        
        multi_player_selector = mo.ui.multiselect(
            options=player_opts[:20],  # Top 20 for multi-select
            label="Compare Multiple Players (optional):",
            value=[]
        )
        
        trends_ui = mo.vstack([
            mo.md("### 📈 Player Performance Trends"),
            mo.md("*Track how players' attributes change over gameweeks*"),
            mo.hstack([player_selector, attribute_selector]),
            multi_player_selector,
            mo.md("---")
        ])
    else:
        trends_ui = mo.vstack([
            mo.md("### 📈 Player Performance Trends"),
            mo.md("⚠️ **Calculate XP first in section 2 to enable trends analysis**"),
            mo.md("*After calculating XP, this section will show interactive player performance charts over gameweeks*")
        ])
        player_selector = None
        attribute_selector = None
        multi_player_selector = None
    
    trends_ui
    
    return player_selector, attribute_selector, multi_player_selector


@app.cell
def __(mo, player_selector, attribute_selector, multi_player_selector, trends_data, pd):
    # Generate the trends chart
    def create_trends_chart(data, selected_player, selected_attr, multi_players=None):
        """Create interactive trends chart"""
        if data.empty:
            return mo.md("⚠️ **No data available for trends**")
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            fig = go.Figure()
            
            # Extract values if they're still dict objects
            if isinstance(selected_player, dict):
                player_id = selected_player.get('value')
            else:
                player_id = selected_player
                
            if isinstance(selected_attr, dict):
                attr_name = selected_attr.get('value')
            else:
                attr_name = selected_attr
            
            # Main player
            if player_id:
                # Debug player selection
                print(f"🔍 Looking for player_id: {player_id} (type: {type(player_id)})")
                print(f"🔍 Attribute: {attr_name} (type: {type(attr_name)})")
                print(f"🔍 Unique player_ids in data: {sorted(data['player_id'].unique())[:10]}...")
                print(f"🔍 Data player_id type: {type(data['player_id'].iloc[0]) if not data.empty else 'no data'}")
                    
                player_data = data[data['player_id'] == player_id].sort_values('gameweek')
                print(f"🔍 Player data shape: {player_data.shape}")
                print(f"🔍 Available columns: {list(player_data.columns)}")
                
                if not player_data.empty and attr_name in player_data.columns:
                    player_name = player_data['web_name'].iloc[0]
                    
                    print(f"🔍 Raw data for {player_name} ({attr_name}): {player_data[attr_name].tolist()}")
                    
                    # Handle price conversion and ensure numeric data
                    y_values = pd.to_numeric(player_data[attr_name], errors='coerce')
                    if attr_name == 'now_cost':
                        y_values = y_values / 10  # Convert to £m
                    
                    print(f"🔍 Processed y_values: {y_values.tolist()}")
                    
                    # Remove any NaN values
                    valid_data = pd.DataFrame({
                        'gameweek': player_data['gameweek'],
                        'y_values': y_values
                    }).dropna()
                    
                    print(f"🔍 Final valid_data:\n{valid_data}")
                    print(f"🔍 All NaN check - y_values: {y_values.isna().sum()}, gameweeks: {player_data['gameweek'].isna().sum()}")
                    
                    print(f"📊 Valid data points: {len(valid_data)}")
                    if not valid_data.empty:
                        print(f"📊 Y-values range: {valid_data['y_values'].min():.2f} to {valid_data['y_values'].max():.2f}")
                        print(f"📊 X-values range: {valid_data['gameweek'].min()} to {valid_data['gameweek'].max()}")
                        
                        fig.add_trace(go.Scatter(
                            x=valid_data['gameweek'],
                            y=valid_data['y_values'],
                            mode='lines+markers',
                            name=f"{player_name}",
                            line=dict(width=3),
                            marker=dict(size=8)
                        ))
                else:
                    print(f"⚠️ No data found for player {player_id} or column {attr_name}")
            
            # Multiple players for comparison
            if multi_players:
                colors = px.colors.qualitative.Set3
                for i, multi_player_id in enumerate(multi_players):
                    if multi_player_id != player_id:  # Avoid duplicate
                        player_data = data[data['player_id'] == multi_player_id].sort_values('gameweek')
                        if not player_data.empty:
                            player_name = player_data['web_name'].iloc[0]
                            
                            # Handle price conversion
                            y_values = player_data[attr_name]
                            if attr_name == 'now_cost':
                                y_values = y_values / 10
                            
                            fig.add_trace(go.Scatter(
                                x=player_data['gameweek'],
                                y=y_values,
                                mode='lines+markers',
                                name=f"{player_name}",
                                line=dict(width=2, color=colors[i % len(colors)]),
                                marker=dict(size=6)
                            ))
            
            # Customize layout
            attr_labels = {
                'total_points': 'Total Points',
                'now_cost': 'Price (£m)',
                'selected_by_percent': 'Selected By (%)',
                'points_per_game': 'Points Per Game',
                'minutes': 'Minutes',
                'goals_scored': 'Goals',
                'assists': 'Assists',
                'expected_goals': 'Expected Goals (xG)',
                'expected_assists': 'Expected Assists (xA)',
                'xg_per_90': 'xG per 90 min',
                'xa_per_90': 'xA per 90 min',
                'value_ratio': 'Points per £1m',
                'clean_sheets': 'Clean Sheets',
                'bonus': 'Bonus Points',
                'ict_index': 'ICT Index'
            }
            
            # Handle case where attr_name might not be a string
            if isinstance(attr_name, str):
                y_label = attr_labels.get(attr_name, attr_name.replace('_', ' ').title())
            else:
                y_label = str(attr_name).replace('_', ' ').title()
            
            fig.update_layout(
                title=f'Player Trends: {y_label} Over Time',
                xaxis_title='Gameweek',
                yaxis_title=y_label,
                hovermode='x unified',
                width=800,
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Ensure axes auto-scale to data
            fig.update_xaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='lightgray',
                autorange=True
            )
            fig.update_yaxes(
                showgrid=True, 
                gridwidth=1, 
                gridcolor='lightgray',
                autorange=True,
                type='linear'
            )
            
            # Summary stats for selected player
            if player_id and not data[data['player_id'] == player_id].empty:
                player_stats = data[data['player_id'] == player_id]
                current_value = player_stats[attr_name].iloc[-1] if not player_stats.empty else 0
                if attr_name == 'now_cost':
                    current_value = current_value / 10
                
                trend_direction = "📈" if len(player_stats) >= 2 and player_stats[attr_name].iloc[-1] > player_stats[attr_name].iloc[0] else "📉"
                
                summary = mo.md(f"""
                **📊 Current {y_label}:** {current_value:.2f} {trend_direction}
                
                **📈 Trend Insights:**
                - Data points: {len(player_stats)} gameweeks
                - Showing progression of {y_label.lower()} over time
                """)
            else:
                summary = mo.md("")
            
            return mo.vstack([
                mo.as_html(fig),
                summary
            ])
            
        except Exception as e:
            return mo.md(f"❌ **Error creating chart:** {str(e)}")
    
    # Generate chart when selectors have values
    if (player_selector and hasattr(player_selector, 'value') and player_selector.value and 
        attribute_selector and hasattr(attribute_selector, 'value') and attribute_selector.value):
        
        multi_values = multi_player_selector.value if multi_player_selector else []
        # Extract actual values from selectors
        selected_player_id = player_selector.value
        selected_attr_val = attribute_selector.value
        
        print(f"🔍 Chart inputs: player={selected_player_id}, attr={selected_attr_val}")
        print(f"🔍 Player type: {type(selected_player_id)}, Attr type: {type(selected_attr_val)}")
        print(f"🔍 Trends data shape: {trends_data.shape}")
        print(f"🔍 Trends data columns: {list(trends_data.columns)}")
        
        trends_chart = create_trends_chart(
            trends_data, 
            selected_player_id, 
            selected_attr_val,
            multi_values
        )
    else:
        trends_chart = mo.md("👆 **Select a player and attribute above to view trends**")
    
    trends_chart
    
    return (trends_chart,)


@app.cell
def __(mo):
    mo.md("## 6. Fixture Difficulty Analysis")
    return


@app.cell
def __(gameweek_input, mo, pd):
    # Fixture Difficulty Heatmap
    def create_fixture_difficulty_visualization(start_gw, num_gws=5):
        """Create fixture difficulty heatmap for next 5 gameweeks"""
        try:
            from client import get_current_teams, get_fixtures_normalized
            from dynamic_team_strength import DynamicTeamStrength, load_historical_gameweek_data
            
            # Load data
            teams = get_current_teams()
            fixtures = get_fixtures_normalized()
            
            # Get dynamic team strength ratings
            calculator = DynamicTeamStrength(debug=False)
            current_season_data = load_historical_gameweek_data(start_gw=1, end_gw=start_gw-1)
            team_strength = calculator.get_team_strength(
                target_gameweek=start_gw,
                teams_data=teams,
                current_season_data=current_season_data
            )
            
            # Create team mapping
            team_id_to_name = dict(zip(teams['team_id'], teams['name']))
            
            # Initialize difficulty matrix
            team_names = sorted(team_strength.keys())
            difficulty_matrix = pd.DataFrame(
                index=team_names,
                columns=[f'GW{gw}' for gw in range(start_gw, start_gw + num_gws)]
            )
            
            # Calculate fixture difficulty for each team and gameweek
            for gw in range(start_gw, start_gw + num_gws):
                gw_fixtures = fixtures[fixtures['event'] == gw].copy()
                
                if gw_fixtures.empty:
                    continue
                
                for _, fixture in gw_fixtures.iterrows():
                    home_team_id = fixture['home_team_id']
                    away_team_id = fixture['away_team_id']
                    
                    home_team_name = team_id_to_name.get(home_team_id)
                    away_team_name = team_id_to_name.get(away_team_id)
                    
                    if home_team_name and away_team_name:
                        # Home team difficulty = opponent strength
                        away_strength = team_strength.get(away_team_name, 1.0)
                        home_difficulty = away_strength
                        
                        # Away team difficulty = opponent strength + away disadvantage
                        home_strength = team_strength.get(home_team_name, 1.0)
                        away_difficulty = home_strength * 1.1  # 10% away disadvantage
                        
                        difficulty_matrix.loc[home_team_name, f'GW{gw}'] = home_difficulty
                        difficulty_matrix.loc[away_team_name, f'GW{gw}'] = away_difficulty
            
            # Convert to float and fill missing values
            difficulty_matrix = difficulty_matrix.astype(float)
            difficulty_matrix = difficulty_matrix.fillna(1.0)
            
            # Calculate average difficulty for analysis
            avg_difficulty = difficulty_matrix.mean(axis=1).sort_values()
            
            # Best and worst fixture runs
            best_fixtures = avg_difficulty.head(5)
            worst_fixtures = avg_difficulty.tail(5)
            
            # Create analysis summary
            analysis_md = f"""
            ### 🏟️ Fixture Difficulty Analysis (GW{start_gw}-{start_gw + num_gws - 1})
            
            **🟢 EASIEST FIXTURE RUNS:**
            """
            
            for i, (team, difficulty) in enumerate(best_fixtures.items(), 1):
                analysis_md += f"\n{i}. **{team}**: {difficulty:.3f} avg difficulty"
            
            analysis_md += "\n\n**🔴 HARDEST FIXTURE RUNS:**"
            
            for i, (team, difficulty) in enumerate(worst_fixtures.items(), 1):
                analysis_md += f"\n{i}. **{team}**: {difficulty:.3f} avg difficulty"
            
            analysis_md += """
            
            **💡 Transfer Strategy:**
            - 🎯 Target players from teams with green fixtures (easy run)
            - ⚠️ Avoid players from teams with red fixtures (tough run)  
            - 📊 Difficulty: 0.7=Very Easy, 1.0=Average, 1.4=Very Hard
            """
            
            # Create interactive plotly heatmap
            import plotly.graph_objects as go
            
            # Get opponent info for hover text
            opponent_info = {}
            for gw in range(start_gw, start_gw + num_gws):
                gw_fixtures = fixtures[fixtures['event'] == gw].copy()
                for _, fixture in gw_fixtures.iterrows():
                    home_team_name = team_id_to_name.get(fixture['home_team_id'])
                    away_team_name = team_id_to_name.get(fixture['away_team_id'])
                    
                    if home_team_name and away_team_name:
                        opponent_info[(home_team_name, f'GW{gw}')] = f"vs {away_team_name} (H)"
                        opponent_info[(away_team_name, f'GW{gw}')] = f"vs {home_team_name} (A)"
            
            # Reshape hover text to match matrix
            hover_matrix = []
            for i, team in enumerate(difficulty_matrix.index):
                team_row = []
                for gw_col in difficulty_matrix.columns:
                    difficulty = difficulty_matrix.loc[team, gw_col]
                    opponent = opponent_info.get((team, gw_col), "No fixture")
                    
                    if difficulty < 0.85:
                        diff_desc = "Very Easy"
                    elif difficulty < 0.95:
                        diff_desc = "Easy"
                    elif difficulty < 1.05:
                        diff_desc = "Average"
                    elif difficulty < 1.15:
                        diff_desc = "Hard"
                    else:
                        diff_desc = "Very Hard"
                    
                    hover_info = (
                        f"<b>{team}</b><br>"
                        f"{gw_col}: {opponent}<br>"
                        f"Difficulty: {difficulty:.2f}<br>"
                        f"Rating: {diff_desc}"
                    )
                    team_row.append(hover_info)
                hover_matrix.append(team_row)
            
            # Create the interactive heatmap
            fig = go.Figure(data=go.Heatmap(
                z=difficulty_matrix.values,
                x=difficulty_matrix.columns,
                y=difficulty_matrix.index,
                colorscale=[
                    [0.0, '#2E8B57'],   # Dark green (very easy)
                    [0.25, '#90EE90'],  # Light green (easy)
                    [0.5, '#FFFF99'],   # Yellow (average)
                    [0.75, '#FFA500'],  # Orange (hard)
                    [1.0, '#FF4500']    # Red (very hard)
                ],
                zmid=1.0,  # Center on average difficulty
                zmin=0.7,
                zmax=1.4,
                text=difficulty_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover_matrix,
                colorbar=dict(
                    title=dict(
                        text="Fixture Difficulty<br>(0.7=Very Easy, 1.4=Very Hard)",
                        side="right"
                    )
                )
            ))
            
            fig.update_layout(
                title=f'Fixture Difficulty Heatmap: GW{start_gw}-{start_gw + num_gws - 1}<br><sub>Interactive - Hover for details</sub>',
                title_x=0.5,
                xaxis_title="Gameweek",
                yaxis_title="Team",
                font=dict(size=12),
                height=600,
                width=800,
                margin=dict(l=120, r=100, t=80, b=50)
            )
            
            # Update axes
            fig.update_xaxes(side="bottom")
            fig.update_yaxes(autorange="reversed")  # Teams from top to bottom
            
            # Create table as backup
            display_df = difficulty_matrix.round(2).reset_index()
            display_df = display_df.rename(columns={'index': 'Team'})
            
            return mo.vstack([
                mo.md(analysis_md),
                mo.md("### 📊 Interactive Fixture Difficulty Heatmap"),
                mo.md("*🟢 Green = Easy, 🟡 Yellow = Average, 🔴 Red = Hard | Hover for opponent details*"),
                mo.as_html(fig),
                mo.md("### 📋 Detailed Values Table"),
                mo.ui.table(display_df, page_size=20)
            ])
            
        except Exception as e:
            return mo.md(f"❌ **Could not create fixture analysis:** {e}")
    
    # Only show if gameweek is selected
    if gameweek_input.value:
        fixture_analysis = create_fixture_difficulty_visualization(gameweek_input.value)
    else:
        fixture_analysis = mo.md("Select target gameweek to see fixture difficulty analysis")
    
    fixture_analysis


@app.cell
def __(mo):
    mo.md("## 7. Team Optimization & Constraints")
    return


@app.cell
def __(players_with_xp, mo):
    # Create player constraint UI and optimization button with error handling
    try:
        if not players_with_xp.empty:
            # Create options for dropdowns
            player_options = []
            # Use 5-GW XP if available, otherwise fall back to regular XP
            sort_column = 'xP_5gw' if 'xP_5gw' in players_with_xp.columns else 'xP'
            
            for _, player in players_with_xp.sort_values(['position', sort_column], ascending=[True, False]).iterrows():
                xp_display = player.get('xP_5gw', player.get('xP', 0))
                label = f"{player['web_name']} ({player['position']}, {player['name']}) - £{player['price']:.1f}m, {xp_display:.2f} 5GW-xP"
                player_options.append({"label": label, "value": player['player_id']})
            
            must_include_dropdown = mo.ui.multiselect(
                options=player_options,
                label="Must Include Players (force these players to be bought/kept)",
                value=[]
            )
            
            must_exclude_dropdown = mo.ui.multiselect(
                options=player_options,
                label="Must Exclude Players (never consider these players)", 
                value=[]
            )
            
            optimize_button = mo.ui.run_button(
                label="🚀 Run Strategic 5-GW Optimization (Auto-selects 0-3 transfers)",
                kind="success"
            )
            
            constraints_ui = mo.vstack([
                mo.md("### 🎯 Player Constraints"),
                mo.md("**Set constraints before optimization:**"),
                must_include_dropdown,
                must_exclude_dropdown,
                mo.md("---"),
                mo.md("### 🧠 Strategic 5-GW Optimization"),
                mo.md("**The optimizer will automatically decide the optimal number of transfers (0-3) based on 5-gameweek net expected points after penalties.**"),
                optimize_button
            ])
        else:
            constraints_ui = mo.md("Calculate XP first to enable optimization")
            must_include_dropdown = None
            must_exclude_dropdown = None
            optimize_button = None
            
    except Exception as e:
        print(f"⚠️ Error creating optimization UI: {e}")
        constraints_ui = mo.md("⚠️ Error creating optimization interface - calculate XP first")
        must_include_dropdown = None
        must_exclude_dropdown = None
        optimize_button = None
    
    constraints_ui
    
    return must_include_dropdown, must_exclude_dropdown, optimize_button


@app.cell
def __(current_squad, team_data, players_with_xp, mo, pd, optimize_button, must_include_dropdown, must_exclude_dropdown):
    def calculate_total_budget_pool(current_squad, bank_balance, players_to_keep=None):
        """
        Calculate total budget available for transfers including sellable squad value.
        
        Args:
            current_squad: DataFrame of current 15 players with price column
            bank_balance: Available money in bank (float)
            players_to_keep: Optional set of player_ids that must be retained
            
        Returns:
            dict with budget analysis for advanced transfer scenarios
        """
        if players_to_keep is None:
            players_to_keep = set()
        
        # Calculate sellable value (exclude must-keep players)
        sellable_players = current_squad[~current_squad['player_id'].isin(players_to_keep)]
        sellable_value = sellable_players['price'].sum()
        
        # Total theoretical budget pool
        total_budget = bank_balance + sellable_value
        
        # Constrained budget (keeping minimum viable squad)
        # Must keep at least: 1 GKP, 3 DEF, 3 MID, 1 FWD = 8 players minimum for valid squad
        min_squad_positions = {'GKP': 1, 'DEF': 3, 'MID': 3, 'FWD': 1}
        
        # Find cheapest players per position to calculate minimum squad cost
        min_squad_cost = 0
        for pos, min_count in min_squad_positions.items():
            pos_players = current_squad[current_squad['position'] == pos]
            if len(pos_players) >= min_count:
                cheapest = pos_players.nsmallest(min_count, 'price')['price'].sum()
                min_squad_cost += cheapest
            else:
                # If we don't have enough players in this position, use their total cost
                min_squad_cost += pos_players['price'].sum()
        
        # Constrained budget = total available - minimum squad cost to maintain validity
        constrained_budget = total_budget - min_squad_cost
        
        # Calculate potential for premium acquisitions (high-value single player purchases)
        max_single_acquisition = constrained_budget
        
        return {
            'total_budget': total_budget,
            'bank_balance': bank_balance,
            'sellable_value': sellable_value,
            'constrained_budget': max(0, constrained_budget),  # Can't go negative
            'min_squad_cost': min_squad_cost,
            'sellable_players_count': len(sellable_players),
            'max_single_acquisition': max(0, max_single_acquisition),
            'budget_utilization_pct': (bank_balance / total_budget * 100) if total_budget > 0 else 0
        }



    def premium_acquisition_planner(current_squad, all_players, budget_pool_info, top_n=3):
        """
        Work backwards from premium targets to find optimal funding strategies.
        
        Args:
            current_squad: DataFrame of current 15 players
            all_players: DataFrame of all available players  
            budget_pool_info: Dict from calculate_total_budget_pool()
            top_n: Number of top premium targets to analyze per position
            
        Returns:
            List of premium acquisition scenarios with funding strategies
        """
        scenarios = []
        
        # Define premium player thresholds (top tier by position)
        premium_thresholds = {
            'GKP': 5.5,  # Premium GKPs are 5.5m+
            'DEF': 6.0,  # Premium DEFs are 6.0m+ 
            'MID': 8.0,  # Premium MIDs are 8.0m+
            'FWD': 9.0   # Premium FWDs are 9.0m+
        }
        
        # Find premium targets not in current squad
        for position, min_price in premium_thresholds.items():
            premium_targets = all_players[
                (all_players['position'] == position) &
                (all_players['price'] >= min_price) &
                (~all_players['player_id'].isin(current_squad['player_id'])) &
                (all_players['xP'] > 0)  # Must have valid XP
            ].nlargest(top_n, 'xP')  # Top N by expected points
            
            print(f"🏆 Found {len(premium_targets)} premium {position} targets (threshold £{min_price:.1f}m+)")
            
            for _, target in premium_targets.iterrows():
                # Check if we can afford this target with available budget  
                max_affordable = budget_pool_info['max_single_acquisition']
                print(f"🎯 Testing premium target {target['web_name']} (£{target['price']:.1f}m) vs max affordable £{max_affordable:.1f}m")
                
                if target['price'] <= max_affordable:
                    
                    # Strategy 1: Direct replacement (same position)
                    current_position_players = current_squad[
                        current_squad['position'] == position
                    ].nsmallest(3, 'xP')  # Consider replacing weakest 3
                    
                    for _, weak_player in current_position_players.iterrows():
                        funding_gap = target['price'] - weak_player['price']
                        xp_gain = target['xP'] - weak_player['xP']
                        
                        # For direct swaps, check if we can afford with available budget
                        # We have bank + selling the weak player = bank + weak_player_price total available
                        total_available = budget_pool_info['bank_balance'] + weak_player['price']
                        
                        print(f"  📊 Direct swap: {weak_player['web_name']} (£{weak_player['price']:.1f}m, {weak_player['xP']:.2f}xP) → {target['web_name']} (£{target['price']:.1f}m, {target['xP']:.2f}xP)")
                        print(f"  💰 Need £{target['price']:.1f}m vs available £{budget_pool_info['bank_balance']:.1f}m + £{weak_player['price']:.1f}m = £{total_available:.1f}m, XP gain: {xp_gain:.2f}")
                        can_afford = target['price'] <= total_available
                        
                        if can_afford:
                            # Simple same-position upgrade
                            if xp_gain > 0.05:  # Meaningful upgrade threshold (lowered for early season)
                                print(f"  ✅ Direct premium scenario viable!")
                                scenarios.append({
                                    'type': 'premium_direct',
                                    'target_player': target,
                                    'sell_players': [weak_player],
                                    'funding_gap': funding_gap,
                                    'xp_gain': xp_gain,
                                    'transfers': 1,
                                    'description': f"Premium Direct: {target['web_name']} for {weak_player['web_name']}"
                                })
                            else:
                                print(f"  ❌ XP gain too small ({xp_gain:.2f} ≤ 0.05)")
                        else:
                            print(f"  ❌ Cannot afford: need £{target['price']:.1f}m but only have £{total_available:.1f}m available")
                    
                    # Strategy 2: Multi-player funding (sell 2-3 players to fund 1 premium)
                    if target['price'] > budget_pool_info['bank_balance']:
                        funding_needed = target['price'] - budget_pool_info['bank_balance']
                        
                        # Find combinations of players to sell (excluding same position to maintain squad balance)
                        other_positions = [pos for pos in ['DEF', 'MID', 'FWD'] if pos != position]
                        potential_sales = current_squad[
                            current_squad['position'].isin(other_positions)
                        ].nsmallest(5, 'xP')  # Consider weakest 5 from other positions
                        
                        # Try 2-player funding combinations
                        from itertools import combinations
                        for sell_combo in combinations(range(len(potential_sales)), 2):
                            sell_players = [potential_sales.iloc[i] for i in sell_combo]
                            sell_value = sum(p['price'] for p in sell_players)
                            sell_xp = sum(p['xP'] for p in sell_players)
                            
                            if sell_value >= funding_needed:
                                # Need to find replacement(s) for sold players
                                remaining_budget = budget_pool_info['bank_balance'] + sell_value - target['price']
                                
                                # Find budget replacements for sold players
                                replacements = []
                                replacement_xp = 0
                                
                                for sell_player in sell_players:
                                    budget_options = all_players[
                                        (all_players['position'] == sell_player['position']) &
                                        (all_players['price'] <= remaining_budget / len(sell_players)) &  # Split budget
                                        (~all_players['player_id'].isin(current_squad['player_id'])) &
                                        (~all_players['player_id'].isin([r['player_id'] for r in replacements]))
                                    ]
                                    
                                    if len(budget_options) > 0:
                                        best_budget = budget_options.nlargest(1, 'xP').iloc[0]
                                        replacements.append(best_budget)
                                        replacement_xp += best_budget['xP']
                                        remaining_budget -= best_budget['price']
                                
                                if len(replacements) == len(sell_players):
                                    # Calculate net XP gain
                                    net_xp_gain = (target['xP'] + replacement_xp) - sell_xp
                                    
                                    if net_xp_gain > 0.2:  # Higher threshold for multi-transfer
                                        sell_names = ", ".join([p['web_name'] for p in sell_players])
                                        replace_names = ", ".join([r['web_name'] for r in replacements])
                                        
                                        scenarios.append({
                                            'type': 'premium_funded',
                                            'target_player': target,
                                            'sell_players': sell_players,
                                            'replacement_players': replacements,
                                            'funding_gap': target['price'] - budget_pool_info['bank_balance'],
                                            'xp_gain': net_xp_gain,
                                            'transfers': 1 + len(sell_players) + len(replacements),
                                            'description': f"Premium Funded: {target['web_name']} (sell {sell_names}, buy {replace_names})"
                                        })
        
        # Sort by XP gain and return top scenarios
        scenarios.sort(key=lambda x: x['xp_gain'], reverse=True)
        return scenarios[:8]  # Return top 8 premium acquisition scenarios

    def optimize_team_with_transfers():
        """Smart optimization that decides optimal number of transfers (0-3) based on net XP"""
        if len(current_squad) == 0 or players_with_xp.empty or not team_data:
            return mo.md("Load team and calculate XP first"), pd.DataFrame(), {}
        
        print(f"🧠 Smart optimization: Finding optimal number of transfers...")
        
        # Process constraints
        must_include_ids = set(must_include_dropdown.value if must_include_dropdown else [])
        must_exclude_ids = set(must_exclude_dropdown.value if must_exclude_dropdown else [])
        
        if must_include_ids:
            print(f"🎯 Must include {len(must_include_ids)} players")
        if must_exclude_ids:
            print(f"🚫 Must exclude {len(must_exclude_ids)} players")
        
        # Update current squad with both 1-GW and 5-GW XP data for strategic decisions
        current_squad_with_xp = current_squad.merge(
            players_with_xp[['player_id', 'xP', 'xP_5gw', 'fixture_outlook']], 
            on='player_id', 
            how='left'
        )
        # Fill any missing XP with 0
        current_squad_with_xp['xP'] = current_squad_with_xp['xP'].fillna(0)
        current_squad_with_xp['xP_5gw'] = current_squad_with_xp['xP_5gw'].fillna(0)
        current_squad_with_xp['fixture_outlook'] = current_squad_with_xp['fixture_outlook'].fillna('🟡 Average')
        
        # Current squad and available budget
        current_player_ids = set(current_squad_with_xp['player_id'].tolist())
        available_budget = team_data['bank']
        free_transfers = team_data.get('free_transfers', 1)
        
        # Calculate total budget pool for advanced transfer scenarios
        budget_pool_info = calculate_total_budget_pool(current_squad_with_xp, available_budget, must_include_ids)
        print(f"💰 Budget Analysis: Bank £{available_budget:.1f}m | Sellable Value £{budget_pool_info['sellable_value']:.1f}m | Total Pool £{budget_pool_info['total_budget']:.1f}m")
        
        # Get all players with XP data and apply exclusion constraints
        all_players = players_with_xp[players_with_xp['xP'].notna()].copy()
        if must_exclude_ids:
            all_players = all_players[~all_players['player_id'].isin(must_exclude_ids)]
        
        def get_best_starting_11(squad_df):
            """Get best starting 11 from squad using 5-GW strategic XP"""
            if len(squad_df) < 11:
                return [], "", 0
            
            # Group by position and sort by 5-GW XP for strategic decisions
            by_position = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
            for _, player in squad_df.iterrows():
                by_position[player['position']].append(player)
            
            for pos in by_position:
                by_position[pos].sort(key=lambda p: p.get('xP_5gw', p.get('xP', 0)), reverse=True)
            
            # Try formations and pick best
            formations = [(1, 3, 5, 2), (1, 3, 4, 3), (1, 4, 5, 1), (1, 4, 4, 2), (1, 4, 3, 3)]
            formation_names = {"(1, 3, 5, 2)": "3-5-2", "(1, 3, 4, 3)": "3-4-3", "(1, 4, 5, 1)": "4-5-1", 
                             "(1, 4, 4, 2)": "4-4-2", "(1, 4, 3, 3)": "4-3-3"}
            
            best_11, best_xp, best_formation = [], 0, ""
            
            for gkp, def_count, mid, fwd in formations:
                if (gkp <= len(by_position['GKP']) and def_count <= len(by_position['DEF']) and
                    mid <= len(by_position['MID']) and fwd <= len(by_position['FWD'])):
                    
                    formation_11 = (by_position['GKP'][:gkp] + by_position['DEF'][:def_count] +
                                  by_position['MID'][:mid] + by_position['FWD'][:fwd])
                    formation_xp = sum(p.get('xP_5gw', p.get('xP', 0)) for p in formation_11)
                    
                    if formation_xp > best_xp:
                        best_xp = formation_xp
                        best_11 = formation_11
                        best_formation = formation_names.get(str((gkp, def_count, mid, fwd)), f"{gkp}-{def_count}-{mid}-{fwd}")
            
            return best_11, best_formation, best_xp
        
        # Scenario 0: No transfers
        current_11, current_formation, current_xp = get_best_starting_11(current_squad_with_xp)
        
        scenarios = [{
            'transfers': 0, 'cost': 0, 'penalty': 0, 'net_xp': current_xp,
            'formation': current_formation, 'starting_11': current_11,
            'description': 'Keep current squad'
        }]
        
        # Scenario 1: 1 Transfer (strategic - based on 5-GW XP)
        best_1_transfer_xp = current_xp
        # Target worst 5-GW performers, considering fixture outlook
        worst_5gw_players = current_squad_with_xp.nsmallest(5, 'xP_5gw')
        for _, out_player in worst_5gw_players.iterrows():
            same_pos = all_players[(all_players['position'] == out_player['position']) &
                                 (~all_players['player_id'].isin(current_player_ids))]
            affordable = same_pos[same_pos['price'] <= out_player['price'] + available_budget]
            
            if len(affordable) > 0:
                # Prioritize players with best 5-GW prospects
                for _, in_player in affordable.nlargest(3, 'xP_5gw').iterrows():
                    new_squad = current_squad_with_xp[current_squad_with_xp['player_id'] != out_player['player_id']].copy()
                    new_squad = pd.concat([new_squad, pd.DataFrame([in_player])], ignore_index=True)
                    
                    if new_squad['name'].value_counts().max() <= 3:  # Team constraint
                        new_11, new_formation, new_xp = get_best_starting_11(new_squad)
                        penalty = 0 if free_transfers >= 1 else -4
                        net_xp = new_xp + penalty
                        
                        if net_xp > best_1_transfer_xp:
                            best_1_transfer_xp = net_xp
                            scenarios.append({
                                'transfers': 1, 'cost': in_player['price'] - out_player['price'],
                                'penalty': penalty, 'net_xp': net_xp, 'formation': new_formation,
                                'starting_11': new_11, 'description': f"OUT: {out_player['web_name']} → IN: {in_player['web_name']}"
                            })
        
        # Scenario 2: 2 Transfers (strategic - target worst 5-GW performers)
        best_2_transfer_xp = current_xp
        worst_players = current_squad_with_xp.nsmallest(3, 'xP_5gw')
        
        for i, (_, out1) in enumerate(worst_players.iterrows()):
            for k, (_, out2) in enumerate(worst_players.iterrows()):
                if i >= k:  # Avoid duplicates and self-comparison
                    continue
                    
                # Find replacements for both positions
                temp_squad = current_squad_with_xp[
                    ~current_squad_with_xp['player_id'].isin([out1['player_id'], out2['player_id']])
                ].copy()
                temp_player_ids = set(temp_squad['player_id'].tolist())
                
                # Find best replacement for position 1
                pos1_options = all_players[
                    (all_players['position'] == out1['position']) &
                    (~all_players['player_id'].isin(temp_player_ids))
                ]
                
                for _, in1 in pos1_options.nlargest(2, 'xP_5gw').iterrows():
                    remaining_budget = available_budget - (in1['price'] - out1['price'])
                    
                    # Find best replacement for position 2
                    pos2_options = all_players[
                        (all_players['position'] == out2['position']) &
                        (~all_players['player_id'].isin(temp_player_ids.union({in1['player_id']}))) &
                        (all_players['price'] <= out2['price'] + remaining_budget)
                    ]
                    
                    if len(pos2_options) > 0:
                        in2 = pos2_options.nlargest(1, 'xP_5gw').iloc[0]
                        
                        # Create new squad
                        new_squad = temp_squad.copy()
                        new_squad = pd.concat([new_squad, pd.DataFrame([in1, in2])], ignore_index=True)
                        
                        if new_squad['name'].value_counts().max() <= 3:  # Team constraint
                            new_11, new_formation, new_xp = get_best_starting_11(new_squad)
                            penalty = -4 if free_transfers < 2 else -4 * max(0, 2 - free_transfers)
                            net_xp = new_xp + penalty
                            
                            if net_xp > best_2_transfer_xp:
                                best_2_transfer_xp = net_xp
                                total_cost = (in1['price'] + in2['price']) - (out1['price'] + out2['price'])
                                scenarios.append({
                                    'transfers': 2, 'cost': total_cost,
                                    'penalty': penalty, 'net_xp': net_xp, 'formation': new_formation,
                                    'starting_11': new_11, 
                                    'description': f"OUT: {out1['web_name']}, {out2['web_name']} → IN: {in1['web_name']}, {in2['web_name']}"
                                })
        
        # Scenario 3: 3 Transfers (simplified - replace 3 worst performing players)
        best_3_transfer_xp = current_xp
        if len(worst_players) >= 3:
            out_players = worst_players.iloc[:3]
            
            # Calculate remaining squad
            temp_squad = current_squad_with_xp[
                ~current_squad_with_xp['player_id'].isin(out_players['player_id'])
            ].copy()
            temp_player_ids = set(temp_squad['player_id'].tolist())
            
            # Simple greedy replacement for 3 transfers
            replacements = []
            remaining_budget = available_budget
            
            for _, out_player in out_players.iterrows():
                remaining_budget += out_player['price']
                
                # Find best replacement for this position within budget
                pos_options = all_players[
                    (all_players['position'] == out_player['position']) &
                    (~all_players['player_id'].isin(temp_player_ids.union(set(r['player_id'] for r in replacements))))
                ]
                affordable = pos_options[pos_options['price'] <= remaining_budget]
                
                if len(affordable) > 0:
                    best_replacement = affordable.nlargest(1, 'xP_5gw').iloc[0]
                    replacements.append(best_replacement)
                    remaining_budget -= best_replacement['price']
            
            if len(replacements) == 3:
                # Create new squad
                new_squad = temp_squad.copy()
                new_squad = pd.concat([new_squad, pd.DataFrame(replacements)], ignore_index=True)
                
                if new_squad['name'].value_counts().max() <= 3:  # Team constraint
                    new_11, new_formation, new_xp = get_best_starting_11(new_squad)
                    penalty = -4 * max(0, 3 - free_transfers)
                    net_xp = new_xp + penalty
                    
                    if net_xp > best_3_transfer_xp:
                        best_3_transfer_xp = net_xp
                        total_cost = sum(r['price'] for r in replacements) - sum(out_players['price'])
                        out_names = ", ".join(out_players['web_name'])
                        in_names = ", ".join([r['web_name'] for r in replacements])
                        scenarios.append({
                            'transfers': 3, 'cost': total_cost,
                            'penalty': penalty, 'net_xp': net_xp, 'formation': new_formation,
                            'starting_11': new_11,
                            'description': f"OUT: {out_names} → IN: {in_names}"
                        })
        
        # Note: Cross-position transfers are not possible in FPL (only like-for-like position swaps allowed)
        print(f"📊 Squad composition: {current_squad_with_xp['position'].value_counts().to_dict()}")
        print(f"💰 Budget pool: Bank £{budget_pool_info['bank_balance']:.1f}m, Max acquisition £{budget_pool_info['max_single_acquisition']:.1f}m")
        
        # Scenario 4+: Premium Acquisition Planning
        print(f"🎯 Generating premium acquisition scenarios...")
        premium_scenarios = premium_acquisition_planner(
            current_squad_with_xp, all_players, budget_pool_info, top_n=2
        )
        
        # Convert premium scenarios to standard format
        for prem_scenario in premium_scenarios:
            num_transfers = prem_scenario['transfers']
            penalty = -4 * max(0, num_transfers - free_transfers)
            
            # For premium scenarios, we need to build the new squad
            if prem_scenario['type'] == 'premium_direct':
                # Simple 1:1 replacement
                new_squad = current_squad_with_xp[
                    current_squad_with_xp['player_id'] != prem_scenario['sell_players'][0]['player_id']
                ].copy()
                new_squad = pd.concat([new_squad, pd.DataFrame([prem_scenario['target_player']])], ignore_index=True)
                
            elif prem_scenario['type'] == 'premium_funded':
                # Multi-player funding scenario
                sell_ids = [p['player_id'] for p in prem_scenario['sell_players']]
                new_squad = current_squad_with_xp[
                    ~current_squad_with_xp['player_id'].isin(sell_ids)
                ].copy()
                
                # Add target player and replacements
                new_players = [prem_scenario['target_player']] + prem_scenario['replacement_players']
                new_squad = pd.concat([new_squad, pd.DataFrame(new_players)], ignore_index=True)
            
            # Validate team constraints
            if new_squad['name'].value_counts().max() <= 3:
                # Get best starting 11 from new squad
                new_11, new_formation, new_xp = get_best_starting_11(new_squad)
                net_xp = new_xp + penalty
                
                # Only add if net XP is better than current
                if net_xp > current_xp:
                    scenarios.append({
                        'transfers': num_transfers,
                        'cost': prem_scenario['funding_gap'], 
                        'penalty': penalty,
                        'net_xp': net_xp,
                        'formation': new_formation,
                        'starting_11': new_11,
                        'description': f"🎯 {prem_scenario['description']} (+{prem_scenario['xp_gain']:.1f} raw XP)",
                        'type': 'premium_acquisition',
                        'target': prem_scenario['target_player']['web_name']
                    })
        
        print(f"✅ Found {len(premium_scenarios)} premium acquisition scenarios, {len([s for s in premium_scenarios if s['transfers'] <= 5])} viable with transfer limits")
        
        # Smart decision: Find best scenario based on net XP
        best_scenario = max(scenarios, key=lambda x: x['net_xp'])
        
        # Create comparison table sorted by net XP
        comparison_df = pd.DataFrame([{
            'Transfers': s['transfers'], 
            'Type': s.get('type', 'standard'),
            'Description': s['description'][:55] + '...' if len(s['description']) > 55 else s['description'],
            'Penalty': s['penalty'], 
            'Net XP': round(s['net_xp'], 2), 
            'Formation': s['formation'],
            'XP Gain': round(s['net_xp'] - scenarios[0]['net_xp'], 2)  # Gain vs no transfers
        } for s in scenarios]).sort_values('Net XP', ascending=False)
        
        # Best starting 11 with strategic information
        best_11_df = pd.DataFrame(best_scenario['starting_11'])
        if not best_11_df.empty:
            # Show both 1-GW and 5-GW XP for strategic insight
            display_cols = ['web_name', 'position', 'name', 'price', 'xP', 'xP_5gw']
            if 'fixture_outlook' in best_11_df.columns:
                display_cols.append('fixture_outlook')
            if 'status' in best_11_df.columns:
                display_cols.append('status')
            
            best_11_df = best_11_df[[col for col in display_cols if col in best_11_df.columns]].round(2)
        
        # Constraint summary
        constraint_info = ""
        if must_include_ids or must_exclude_ids:
            constraint_info = f"\n**Constraints Applied:** {len(must_include_ids)} must include, {len(must_exclude_ids)} must exclude"
        
        # Count scenario types for display
        premium_count = len([s for s in scenarios if s.get('type') == 'premium_acquisition'])
        
        # Create budget analysis display
        budget_analysis = f"""
        **💰 Budget Pool Analysis:**
        - Bank: £{budget_pool_info['bank_balance']:.1f}m | Sellable Value: £{budget_pool_info['sellable_value']:.1f}m | **Total Pool: £{budget_pool_info['total_budget']:.1f}m**
        - Max Single Acquisition: £{budget_pool_info['max_single_acquisition']:.1f}m | Budget Utilization: {budget_pool_info['budget_utilization_pct']:.1f}%
        - {budget_pool_info['sellable_players_count']} sellable players | Min squad cost: £{budget_pool_info['min_squad_cost']:.1f}m
        
        **🎯 Enhanced Transfer Analysis:**
        - {len(scenarios)} total scenarios analyzed ({premium_count} premium acquisition scenarios)
        - Premium acquisition planner targets high-value players with smart funding strategies
        - Enhanced budget pool calculation enables complex multi-player funding scenarios
        """
        
        return mo.vstack([
            mo.md(f"### 🏆 Strategic 5-GW Decision: {best_scenario['transfers']} Transfer(s) Optimal"),
            mo.md(f"**Recommended Strategy:** {best_scenario['description']}"),
            mo.md(f"**Expected Net 5-GW XP:** {best_scenario['net_xp']:.2f} | **Formation:** {best_scenario['formation']}{constraint_info}"),
            mo.md("*Decisions based on 5-gameweek horizon with temporal weighting and fixture analysis*"),
            mo.md(budget_analysis),
            mo.md("**All Scenarios (sorted by 5-GW Net XP):**"),
            mo.ui.table(comparison_df, page_size=6),
            mo.md("**Optimal Starting 11 (Strategic):**"),
            mo.md("*Shows both 1-GW and 5-GW XP with fixture outlook*"),
            mo.ui.table(best_11_df, page_size=11) if not best_11_df.empty else mo.md("No data")
        ]), best_11_df, best_scenario
    
    # Only run optimization when button is clicked with error handling
    try:
        if optimize_button and hasattr(optimize_button, 'value') and optimize_button.value:
            optimization_display, optimal_starting_11, optimal_scenario = optimize_team_with_transfers()
        else:
            optimization_display = mo.md("👆 Click the optimization button above to analyze 5-gameweek strategic transfer scenarios")
            optimal_starting_11 = pd.DataFrame()
            optimal_scenario = {}
    except Exception as e:
        print(f"⚠️ Error in optimization: {e}")
        optimization_display = mo.md(f"❌ Optimization error: {str(e)}")
        optimal_starting_11 = pd.DataFrame()
        optimal_scenario = {}
    
    # Display the results
    optimization_display
    
    return optimal_starting_11, optimization_display, optimal_scenario


@app.cell
def __(mo):
    mo.md("## 8. Captain Selection")
    return


@app.cell
def __(optimal_starting_11, mo):
    # Captain Selection
    def select_captain():
        if optimal_starting_11.empty:
            return mo.md("Optimize team first to select captain")
        
        # Captain selection based on XP
        captain_candidates = optimal_starting_11.copy()
        captain_candidates['captain_score'] = captain_candidates['xP'] * 2  # Double points consideration
        
        top_3 = captain_candidates.nlargest(3, 'captain_score')
        captain = top_3.iloc[0]
        vice_captain = top_3.iloc[1] if len(top_3) > 1 else captain
        
        return mo.vstack([
            mo.md("### 👑 Captain Selection"),
            mo.md(f"**Captain:** {captain['web_name']} ({captain['position']}) - {captain['xP']:.2f} xP"),
            mo.md(f"**Vice-Captain:** {vice_captain['web_name']} ({vice_captain['position']}) - {vice_captain['xP']:.2f} xP"),
            mo.md("**Top Captain Options:**"),
            mo.ui.table(
                top_3[['web_name', 'position', 'name', 'price', 'xP']].round(2),
                page_size=3
            )
        ])
    
    captain_display = select_captain()
    captain_display
    
    return (captain_display,)


@app.cell
def __(mo, current_squad, team_data, optimal_scenario):
    # Summary
    if len(current_squad) > 0 and team_data:
        optimal_info = ""
        if optimal_scenario:
            optimal_info = f"""
        **Recommended Strategy:** {optimal_scenario.get('description', 'N/A')}
        **Expected XP:** {optimal_scenario.get('net_xp', 0):.2f}
        **Formation:** {optimal_scenario.get('formation', 'N/A')}
        """
        
        mo.md(f"""
        ## ✅ Summary
        
        **Team:** {team_data['entry_name']}
        **Current Squad:** {len(current_squad)}/15 players
        **Available Budget:** £{team_data['bank']:.1f}m
        **Free Transfers:** {team_data.get('free_transfers', 1)}
        {optimal_info}
        
        **Next Steps:**
        1. Review optimization recommendations above
        2. Apply any transfer constraints if needed
        3. Confirm captain and vice-captain selection
        4. Execute transfers and lineup in FPL app before deadline
        """)
    else:
        mo.md("Select target gameweek above to get started")
    
    return


if __name__ == "__main__":
    app.run()