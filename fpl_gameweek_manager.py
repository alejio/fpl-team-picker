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
    mo.md("## 2. Calculate Expected Points (XP) for All Players")
    return


@app.cell
def __(players, teams, xg_rates, fixtures, live_data_historical, gameweek_input, mo, pd):
    def calculate_expected_points_all_players(players_data, teams_data, xg_rates_data, fixtures_data, target_gameweek, live_data_hist=None):
        """Calculate expected points for all players using improved XP model with form weighting"""
        if players_data.empty or not target_gameweek:
            return mo.md("Select gameweek and load data first"), pd.DataFrame()
        
        print(f"🧠 Calculating expected points using improved XP model for GW{target_gameweek}...")
        
        try:
            # Import and use the new XP model
            from xp_model import XPModel
            
            # Initialize model with form weighting enabled
            xp_model = XPModel(
                form_weight=0.7,  # 70% recent form, 30% season average
                form_window=5,    # Last 5 gameweeks for form
                debug=True
            )
            
            # Calculate XP using the new model
            players_xp = xp_model.calculate_expected_points(
                players_data=players_data,
                teams_data=teams_data,
                xg_rates_data=xg_rates_data,
                fixtures_data=fixtures_data,
                target_gameweek=target_gameweek,
                live_data=live_data_hist,  # Historical form data
                gameweeks_ahead=1
            )
            
            # Add form indicators to display if available
            form_columns = ['web_name', 'position', 'name', 'price', 'xP', 'xP_per_price', 'fixture_difficulty', 'status']
            
            # Add form indicators if they exist
            if 'momentum' in players_xp.columns:
                form_columns.insert(-1, 'momentum')  # Insert before status
            if 'form_multiplier' in players_xp.columns:
                form_columns.insert(-1, 'form_multiplier')
            if 'recent_points_per_game' in players_xp.columns:
                form_columns.insert(-1, 'recent_points_per_game')
            
            # Create display table
            display_df = players_xp[form_columns].sort_values('xP', ascending=False).round(3)
            
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
            
            return mo.vstack([
                mo.md(f"### ✅ Improved XP Model - GW{target_gameweek}"),
                mo.md(f"**Players analyzed:** {len(players_xp)}"),
                mo.md(f"**Average XP:** {players_xp['xP'].mean():.2f}"),
                mo.md(f"**Top performing player XP:** {players_xp['xP'].max():.2f}"),
                mo.md(form_info) if form_info else mo.md(""),
                mo.md("**All Players by Expected Points (with form analysis):**"),
                mo.ui.table(display_df, page_size=25)
            ]), players_xp
            
        except ImportError as e:
            print(f"⚠️ New XP model not available: {e}")
            return mo.md("❌ Could not load improved XP model - check xp_model.py"), pd.DataFrame()
        except Exception as e:
            print(f"❌ Error calculating XP: {e}")
            return mo.md(f"❌ Error calculating expected points: {e}"), pd.DataFrame()
    
    # Only calculate XP if we have valid data
    if not players.empty and gameweek_input.value:
        xp_display, players_with_xp = calculate_expected_points_all_players(players, teams, xg_rates, fixtures, gameweek_input.value, live_data_historical)
    else:
        xp_display = mo.md("Load gameweek data first")
        players_with_xp = pd.DataFrame()
    
    xp_display
    
    return players_with_xp, xp_display


@app.cell
def __(mo):
    mo.md("## 3. Form Analytics Dashboard")
    return


@app.cell
def __(players_with_xp, mo):
    # Form Analytics Dashboard
    def create_form_insights(players_df):
        """Create form-based insights for transfer decisions"""
        if players_df.empty or 'momentum' not in players_df.columns:
            return mo.md("⚠️ **No form data available** - load historical data first")
        
        # Hot players analysis
        hot_players = players_df[players_df['momentum'] == '🔥'].nlargest(8, 'xP')
        cold_players = players_df[players_df['momentum'] == '❄️'].nsmallest(8, 'form_multiplier')
        
        # Value analysis with form
        value_players = players_df[
            (players_df['momentum'].isin(['🔥', '📈'])) & 
            (players_df['price'] <= 7.5)
        ].nlargest(10, 'xP_per_price')
        
        # Expensive underperformers
        expensive_poor = players_df[
            (players_df['price'] >= 8.0) & 
            (players_df['momentum'].isin(['❄️', '📉']))
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
        if 'form_multiplier' in players_df.columns:
            momentum_counts = players_df['momentum'].value_counts()
            avg_multiplier = players_df['form_multiplier'].mean()
            
            insights.append(mo.md(f"""
            ### 📊 Form Summary
            **Player Distribution:** 🔥 {momentum_counts.get('🔥', 0)} | 📈 {momentum_counts.get('📈', 0)} | ➡️ {momentum_counts.get('➡️', 0)} | 📉 {momentum_counts.get('📉', 0)} | ❄️ {momentum_counts.get('❄️', 0)}
            
            **Average Form Multiplier:** {avg_multiplier:.2f}
            
            **Transfer Strategy:** Target 🔥 hot and 📈 rising players, avoid ❄️ cold and 📉 declining players
            """))
        
        return mo.vstack(insights)
    
    # Display form insights
    form_insights_display = create_form_insights(players_with_xp)
    form_insights_display
    
    return (form_insights_display,)


@app.cell
def __(current_squad, players_with_xp, mo):
    # Current Squad Form Analysis
    def analyze_current_squad_form(squad_df, all_players_df):
        """Analyze the form of current squad players"""
        if squad_df.empty or all_players_df.empty or 'momentum' not in all_players_df.columns:
            return mo.md("⚠️ **No squad or form data available**")
        
        # Merge squad with form data
        squad_with_form = squad_df.merge(
            all_players_df[['player_id', 'xP', 'momentum', 'form_multiplier', 'recent_points_per_game']], 
            on='player_id', 
            how='left'
        )
        
        if squad_with_form.empty:
            return mo.md("⚠️ **Could not analyze squad form**")
        
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
        - 🔥 Hot: {squad_momentum.get('🔥', 0)} players
        - 📈 Rising: {squad_momentum.get('📈', 0)} players  
        - ➡️ Stable: {squad_momentum.get('➡️', 0)} players
        - 📉 Declining: {squad_momentum.get('📉', 0)} players
        - ❄️ Cold: {squad_momentum.get('❄️', 0)} players
        
        **Squad Average Form Multiplier:** {squad_avg_form:.2f}
        """))
        
        # Show problem players if any
        if len(problem_players) > 0:
            squad_insights.extend([
                mo.md("### 🚨 Squad Players in Poor Form (Consider Selling)"),
                mo.ui.table(
                    problem_players[['web_name', 'position', 'name', 'price', 'xP', 'momentum', 'form_multiplier']].round(2),
                    page_size=10
                )
            ])
        
        # Show top performers
        if len(top_performers) > 0:
            squad_insights.extend([
                mo.md("### ⭐ Squad Players in Great Form (Keep/Captain)"),
                mo.ui.table(
                    top_performers[['web_name', 'position', 'name', 'price', 'xP', 'momentum', 'form_multiplier']].round(2),
                    page_size=10
                )
            ])
        
        # Overall squad health assessment
        poor_form_count = len(problem_players)
        good_form_count = len(top_performers)
        
        if poor_form_count >= 3:
            health_status = "🚨 **POOR** - Multiple players in poor form, urgent transfers needed"
        elif poor_form_count >= 2:
            health_status = "⚠️ **CONCERNING** - Some players in poor form, consider transfers"
        elif good_form_count >= 5:
            health_status = "🔥 **EXCELLENT** - Squad in great form, minimal changes needed"
        elif good_form_count >= 3:
            health_status = "✅ **GOOD** - Solid squad form, selective improvements possible"
        else:
            health_status = "➡️ **AVERAGE** - Squad form is stable, monitor for improvements"
        
        squad_insights.append(mo.md(f"""
        ### 🎯 Squad Health Assessment
        {health_status}
        
        **Transfer Priority:** {"High" if poor_form_count >= 2 else "Medium" if poor_form_count == 1 else "Low"}
        """))
        
        return mo.vstack(squad_insights)
    
    # Only show if we have squad and form data
    if not current_squad.empty and not players_with_xp.empty:
        squad_form_analysis = analyze_current_squad_form(current_squad, players_with_xp)
    else:
        squad_form_analysis = mo.md("Load your team first to analyze squad form")
    
    squad_form_analysis
    
    return (squad_form_analysis,)


@app.cell
def __(mo):
    mo.md("## 5. Team Optimization & Constraints")
    return


@app.cell
def __(players_with_xp, mo):
    # Create player constraint UI and optimization button
    if not players_with_xp.empty:
        # Create options for dropdowns
        player_options = []
        for _, player in players_with_xp.sort_values(['position', 'xP'], ascending=[True, False]).iterrows():
            label = f"{player['web_name']} ({player['position']}, {player['name']}) - £{player['price']:.1f}m, {player['xP']:.2f} xP"
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
            label="🚀 Run Smart Optimization (Auto-selects 0-3 transfers)",
            kind="success"
        )
        
        constraints_ui = mo.vstack([
            mo.md("### 🎯 Player Constraints"),
            mo.md("**Set constraints before optimization:**"),
            must_include_dropdown,
            must_exclude_dropdown,
            mo.md("---"),
            mo.md("### 🧠 Smart Optimization"),
            mo.md("**The optimizer will automatically decide the optimal number of transfers (0-3) based on net expected points after penalties.**"),
            optimize_button
        ])
    else:
        constraints_ui = mo.md("Calculate XP first to enable optimization")
        must_include_dropdown = None
        must_exclude_dropdown = None
        optimize_button = None
    
    constraints_ui
    
    return must_include_dropdown, must_exclude_dropdown, optimize_button


@app.cell
def __(current_squad, team_data, players_with_xp, mo, pd, optimize_button, must_include_dropdown, must_exclude_dropdown):
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
        
        # Update current squad with XP data
        current_squad_with_xp = current_squad.merge(
            players_with_xp[['player_id', 'xP']], 
            on='player_id', 
            how='left'
        )
        # Fill any missing XP with 0
        current_squad_with_xp['xP'] = current_squad_with_xp['xP'].fillna(0)
        
        # Current squad and available budget
        current_player_ids = set(current_squad_with_xp['player_id'].tolist())
        available_budget = team_data['bank']
        free_transfers = team_data.get('free_transfers', 1)
        
        # Get all players with XP data and apply exclusion constraints
        all_players = players_with_xp[players_with_xp['xP'].notna()].copy()
        if must_exclude_ids:
            all_players = all_players[~all_players['player_id'].isin(must_exclude_ids)]
        
        def get_best_starting_11(squad_df):
            """Get best starting 11 from squad using XP"""
            if len(squad_df) < 11:
                return [], "", 0
            
            # Group by position and sort by XP
            by_position = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
            for _, player in squad_df.iterrows():
                by_position[player['position']].append(player)
            
            for pos in by_position:
                by_position[pos].sort(key=lambda p: p['xP'], reverse=True)
            
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
                    formation_xp = sum(p['xP'] for p in formation_11)
                    
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
        
        # Scenario 1: 1 Transfer (simplified)
        best_1_transfer_xp = current_xp
        for _, out_player in current_squad_with_xp.nsmallest(5, 'xP').iterrows():
            same_pos = all_players[(all_players['position'] == out_player['position']) &
                                 (~all_players['player_id'].isin(current_player_ids))]
            affordable = same_pos[same_pos['price'] <= out_player['price'] + available_budget]
            
            if len(affordable) > 0:
                for _, in_player in affordable.nlargest(3, 'xP').iterrows():
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
        
        # Scenario 2: 2 Transfers (simplified - best 2 single transfers)
        best_2_transfer_xp = current_xp
        worst_players = current_squad_with_xp.nsmallest(3, 'xP')
        
        for i, (_, out1) in enumerate(worst_players.iterrows()):
            for j, (_, out2) in enumerate(worst_players.iterrows()):
                if i >= j:  # Avoid duplicates and self-comparison
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
                
                for _, in1 in pos1_options.nlargest(2, 'xP').iterrows():
                    remaining_budget = available_budget - (in1['price'] - out1['price'])
                    
                    # Find best replacement for position 2
                    pos2_options = all_players[
                        (all_players['position'] == out2['position']) &
                        (~all_players['player_id'].isin(temp_player_ids.union({in1['player_id']}))) &
                        (all_players['price'] <= out2['price'] + remaining_budget)
                    ]
                    
                    if len(pos2_options) > 0:
                        in2 = pos2_options.nlargest(1, 'xP').iloc[0]
                        
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
                    best_replacement = affordable.nlargest(1, 'xP').iloc[0]
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
        
        # Smart decision: Find best scenario based on net XP
        best_scenario = max(scenarios, key=lambda x: x['net_xp'])
        
        # Create comparison table sorted by net XP
        comparison_df = pd.DataFrame([{
            'Transfers': s['transfers'], 
            'Description': s['description'][:50] + '...' if len(s['description']) > 50 else s['description'],
            'Penalty': s['penalty'], 
            'Net XP': round(s['net_xp'], 2), 
            'Formation': s['formation'],
            'XP Gain': round(s['net_xp'] - scenarios[0]['net_xp'], 2)  # Gain vs no transfers
        } for s in scenarios]).sort_values('Net XP', ascending=False)
        
        # Best starting 11
        best_11_df = pd.DataFrame(best_scenario['starting_11'])
        if not best_11_df.empty:
            best_11_df = best_11_df[['web_name', 'position', 'name', 'price', 'xP', 'status']].round(2)
        
        # Constraint summary
        constraint_info = ""
        if must_include_ids or must_exclude_ids:
            constraint_info = f"\n**Constraints Applied:** {len(must_include_ids)} must include, {len(must_exclude_ids)} must exclude"
        
        return mo.vstack([
            mo.md(f"### 🏆 Smart Decision: {best_scenario['transfers']} Transfer(s) Optimal"),
            mo.md(f"**Recommended Strategy:** {best_scenario['description']}"),
            mo.md(f"**Expected Net XP:** {best_scenario['net_xp']:.2f} | **Formation:** {best_scenario['formation']}{constraint_info}"),
            mo.md("**All Scenarios (sorted by Net XP):**"),
            mo.ui.table(comparison_df, page_size=6),
            mo.md("**Optimal Starting 11:**"),
            mo.ui.table(best_11_df, page_size=11) if not best_11_df.empty else mo.md("No data")
        ]), best_11_df, best_scenario
    
    # Only run optimization when button is clicked
    if optimize_button.value:
        optimization_display, optimal_starting_11, optimal_scenario = optimize_team_with_transfers()
    else:
        optimization_display = mo.md("👆 Click the optimization button above to analyze transfer scenarios")
        optimal_starting_11 = pd.DataFrame()
        optimal_scenario = {}
    
    # Display the results
    optimization_display
    
    return optimal_starting_11, optimization_display, optimal_scenario


@app.cell
def __(mo):
    mo.md("## 6. Captain Selection")
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