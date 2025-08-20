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
    mo.md("## 5. Fixture Difficulty Analysis")
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
                for j, gw_col in enumerate(difficulty_matrix.columns):
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
    mo.md("## 6. Team Optimization & Constraints")
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
    mo.md("## 7. Captain Selection")
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