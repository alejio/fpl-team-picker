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
def __(gameweek_input, mo):
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
    # Import optimization functions from new module
    from fpl_optimization import optimize_team_with_transfers
    
    # Run optimization when button is pressed
    if optimize_button and hasattr(optimize_button, 'value') and optimize_button.value:
        try:
            # Get constraint values
            must_include_ids = set(must_include_dropdown.value if must_include_dropdown and hasattr(must_include_dropdown, 'value') else [])
            must_exclude_ids = set(must_exclude_dropdown.value if must_exclude_dropdown and hasattr(must_exclude_dropdown, 'value') else [])
            
            # Run optimization
            optimization_display, optimal_squad, optimal_scenario = optimize_team_with_transfers(
                current_squad=current_squad,
                team_data=team_data,
                players_with_xp=players_with_xp,
                must_include_ids=must_include_ids,
                must_exclude_ids=must_exclude_ids
            )
            
            # Get starting 11 from optimal squad
            from fpl_optimization import get_best_starting_11
            optimal_starting_11, formation, total_xp = get_best_starting_11(optimal_squad)
            
        except Exception as e:
            optimization_display = mo.md(f"❌ **Optimization Error:** {e}")
            optimal_starting_11 = []
            optimal_scenario = {}
    else:
        optimization_display = mo.md("👆 **Click 'Optimize Team' button above to run transfer analysis**")
        optimal_starting_11 = []
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
    # Import captain selection from optimization module
    from fpl_optimization import select_captain
    
    # Use optimization module function for captain selection
    if optimal_starting_11 and len(optimal_starting_11) > 0:
        captain_display = select_captain(optimal_starting_11, mo)
    else:
        captain_display = mo.md("Optimize team first to select captain")
    
    captain_display
    
    return (captain_display,)


@app.cell
def __(mo, current_squad, team_data, optimal_scenario):
    # Summary
    if len(current_squad) > 0 and team_data:
        optimal_info = ""
        if optimal_scenario and isinstance(optimal_scenario, dict) and optimal_scenario:
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