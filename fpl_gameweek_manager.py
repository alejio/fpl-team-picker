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
    # Import data loading functions from new module
    from fpl_data_loader import fetch_fpl_data, fetch_manager_team, process_current_squad, load_gameweek_datasets
    
    return fetch_fpl_data, fetch_manager_team, process_current_squad, load_gameweek_datasets


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
def __(fetch_fpl_data, fetch_manager_team, process_current_squad, gameweek_input, mo, pd):
    # Load data when gameweek is provided
    if gameweek_input.value:
        # Load data for target gameweek
        target_gw = gameweek_input.value
        previous_gw = target_gw - 1
        
        # Get FPL data and calculate XP for target gameweek
        players, teams, xg_rates, fixtures, _, live_data_historical = fetch_fpl_data(target_gw)
        
        # Load manager team from previous gameweek
        team_data = fetch_manager_team(previous_gw)
        
        if team_data:
            # Process current squad using the imported function
            current_squad = process_current_squad(team_data, players, teams)
            
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
    # Import visualization function from new module
    from fpl_visualization import create_team_strength_visualization
    
    # Only show if gameweek is selected
    if gameweek_input.value:
        team_strength_analysis = create_team_strength_visualization(gameweek_input.value, mo)
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
    # Import visualization function from new module
    from fpl_visualization import create_player_trends_visualization
    
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
    # Import trends chart function from visualization module
    from fpl_visualization import create_trends_chart
    
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
            multi_values,
            mo
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
    # Import fixture difficulty visualization from new module
    from fpl_visualization import create_fixture_difficulty_visualization
    
    # Only show if gameweek is selected
    if gameweek_input.value:
        fixture_analysis = create_fixture_difficulty_visualization(gameweek_input.value, 5, mo)
    else:
        fixture_analysis = mo.md("Select target gameweek to see fixture difficulty analysis")
    
    fixture_analysis
    
    return (fixture_analysis,)



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