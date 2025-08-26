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
        # 🎯 FPL Gameweek Manager
        ## Advanced Weekly Decision Making Tool with Form Analytics

        Transform your FPL strategy with data-driven weekly optimization and comprehensive form analysis.
        
        ### 📋 Workflow Overview
        1. **Configure Gameweek** - Select target gameweek for optimization
        2. **Team Strength Analysis** - Dynamic strength ratings and venue adjustments  
        3. **Expected Points Calculation** - Form-weighted XP with 1-GW vs 5-GW comparison
        4. **Form Analytics Dashboard** - Hot/cold player insights and transfer targets
        5. **Squad Health Assessment** - Current team form analysis
        6. **Performance Trends** - Interactive player tracking over time
        7. **Fixture Analysis** - 5-gameweek difficulty heatmaps
        8. **Transfer Optimization** - Smart 0-3 transfer scenarios with budget analysis
        9. **Captain Selection** - Risk-adjusted captaincy recommendations
        
        ### 🔥 Key Features
        - **Form-weighted predictions**: 70% recent form + 30% season baseline
        - **Hot/Cold detection**: Player momentum indicators (🔥📈➡️📉❄️)
        - **Strategic horizon**: Compare 1-GW tactical vs 5-GW strategic decisions
        - **Premium acquisition planning**: Multi-transfer scenarios for expensive targets
        - **Budget pool analysis**: Total available funds including sellable squad value
        
        ---
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
    mo.md(
        r"""
        ## 1️⃣ Configure Gameweek
        
        **Start by selecting your target gameweek for optimization.**
        
        The system will load your team from the previous gameweek and analyze transfer opportunities for the upcoming fixtures.
        
        ---
        """
    )
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
        start=1,
        stop=38,
        label="Target Gameweek (we'll optimize for this GW using data from previous GW)"
    )
    
    mo.vstack([
        mo.md("### 📅 Select Target Gameweek"),
        mo.md("*We'll load your team from the previous gameweek and optimize for the target gameweek*"),
        mo.md(""),
        gameweek_input,
        mo.md("---")
    ])
    
    return (gameweek_input,)


@app.cell
def __(fetch_fpl_data, fetch_manager_team, process_current_squad, gameweek_input, mo, pd):
    # Load data when gameweek is provided
    if gameweek_input.value:
        # Load data for target gameweek
        target_gw = gameweek_input.value
        
        # Handle different gameweek scenarios
        if target_gw == 1:
            # GW1 - no previous gameweek data available
            previous_gw = None
            team_data = None
            current_squad = pd.DataFrame()
        else:
            previous_gw = target_gw - 1
            # Load manager team from previous gameweek
            team_data = fetch_manager_team(previous_gw)
            
            # Special handling for GW2 when it's in progress
            if target_gw == 2 and not team_data:
                print("⚠️ No GW1 team data found - GW2 may be in progress")
                print("💡 Try loading GW1 first to see your initial team")
        
        # Get FPL data and calculate XP for target gameweek
        players, teams, xg_rates, fixtures, _, live_data_historical = fetch_fpl_data(target_gw)
        
        # Check what data is available for better guidance
        def check_data_availability():
            """Check what gameweek data is currently available"""
            try:
                from client import FPLDataClient
                client = FPLDataClient()
                
                available_gws = []
                for gw in range(1, 6):  # Check first 5 gameweeks
                    try:
                        gw_data = client.get_gameweek_live_data(gw)
                        if not gw_data.empty:
                            available_gws.append(gw)
                    except:
                        continue
                
                return available_gws
            except:
                return []
        
        available_gameweeks = check_data_availability()
        if available_gameweeks:
            print(f"📊 Available gameweek data: {available_gameweeks}")
        else:
            print("⚠️ No gameweek data found in database")
        
        if target_gw == 1:
            # GW1 - no previous team data available
            # Check what columns are available in players DataFrame
            available_columns = list(players.columns)
            print(f"🔍 Available columns in players DataFrame: {available_columns}")
            
            # Select columns that exist for display
            display_columns = []
            for col_name in ['web_name', 'position', 'name', 'price', 'selected_by_percent']:
                if col_name in available_columns:
                    display_columns.append(col_name)
            
            # Fallback columns if primary ones don't exist
            if not display_columns:
                display_columns = available_columns[:5]  # Take first 5 columns as fallback
            
            print(f"📊 Using columns for display: {display_columns}")
            
            # Show data availability info for GW1
            data_info = ""
            if available_gameweeks:
                if 1 in available_gameweeks:
                    data_info = "✅ GW1 data available for analysis"
                else:
                    data_info = f"⚠️ GW1 data not found. Available: GW{', GW'.join(map(str, available_gameweeks))}"
            else:
                data_info = "⚠️ No gameweek data found in database"
            
            team_display = mo.vstack([
                mo.md(f"### 🚀 GW1 Team Selection"),
                mo.md(f"**Optimizing for GW{target_gw}**"),
                mo.md(data_info),
                mo.md("**Note:** No previous team data available for GW1. This will help you select your initial squad."),
                mo.md("**Available Players:**"),
                mo.ui.table(
                    players[display_columns].head(20).round(2),
                    page_size=20
                )
            ])
        elif team_data:
            # Process current squad using the imported function
            current_squad = process_current_squad(team_data, players, teams)
            
            team_display = mo.vstack([
                mo.md(f"### ✅ {team_data['entry_name']} (from GW{previous_gw})"),
                mo.md(f"**Previous Points:** {team_data['total_points']:,} | **Bank:** £{team_data['bank']:.1f}m | **Value:** £{team_data['team_value']:.1f}m | **Free Transfers:** {team_data['free_transfers']}"),
                mo.md(f"**Optimizing for GW{target_gw}**"),
                mo.md("**Current Squad:**"),
                mo.ui.table(
                    current_squad[['web_name', 'position', 'price']].round(2),  # Simplified columns to avoid missing ones
                    page_size=15
                )
            ])
        else:
            # Handle case where team data couldn't be loaded
            if target_gw == 2:
                available_info = ""
                if available_gameweeks:
                    available_info = f"**Available data:** GW{', GW'.join(map(str, available_gameweeks))}"
                else:
                    available_info = "**Available data:** None found"
                
                team_display = mo.vstack([
                    mo.md("### ⚠️ GW2 Team Data Unavailable"),
                    mo.md(available_info),
                    mo.md(""),
                    mo.md("**Possible reasons:**"),
                    mo.md("• GW2 is currently in progress"),
                    mo.md("• No GW1 team data has been saved yet"),
                    mo.md("• Database connection issue"),
                    mo.md(""),
                    mo.md("**Suggestions:**"),
                    mo.md("• Try selecting GW1 to see available players"),
                    mo.md("• Check if your GW1 team has been saved"),
                    mo.md("• Wait for GW2 to complete and data to be updated")
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
    mo.md(
        r"""
        ## 2️⃣ Team Strength Analysis
        
        **Dynamic team strength ratings with seasonal transitions and venue adjustments.**
        
        Our model evolves throughout the season:
        - **Early Season**: Blend 2024-25 baseline with current performance
        - **GW8+ Focus**: Pure current season form and results
        - **Home/Away**: Contextual difficulty scaling by venue
        
        ---
        """
    )
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
    mo.md(
        r"""
        ## 3️⃣ Expected Points Engine
        
        **Form-weighted XP calculations with strategic 1-GW vs 5-GW comparison.**
        
        ### Model Features:
        - **Form Weighting**: 70% recent form + 30% season baseline for responsive predictions
        - **Statistical Estimation**: Advanced modeling for new transfers and missing data
        - **Dual Horizons**: Compare immediate (1-GW) vs strategic (5-GW) value
        - **Enhanced Minutes**: SBP + availability + price-based durability modeling
        - **Fixture Scaling**: Dynamic difficulty multipliers by opponent strength
        
        ---
        """
    )
    return


@app.cell
def __(players, teams, xg_rates, fixtures, live_data_historical, gameweek_input, mo, pd):
    # Import visualization functions from new module
    from fpl_visualization import create_xp_results_display
    
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
            print("🔮 Calculating single-gameweek XP for tactical analysis...")
            players_1gw = xp_model.calculate_expected_points(
                players_data=players_data,
                teams_data=teams_data,
                xg_rates_data=xg_rates_data,
                fixtures_data=fixtures_data,
                target_gameweek=target_gameweek,
                live_data=live_data_hist,  # Historical form data
                gameweeks_ahead=1
            )
            
            print("🎯 Calculating 5-gameweek strategic horizon XP...")
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
                    for col_key in ['xP', 'xP_per_price', 'fixture_difficulty']:
                        if col_key in players_5gw.columns:
                            merge_cols.append(col_key)
                    
                    print(f"🔗 Merging columns: {merge_cols}")
                    
                    # Create suffix mapping to avoid conflicts
                    suffix_data = players_5gw[merge_cols].copy()
                    for merge_col in merge_cols:
                        if merge_col != 'player_id':
                            suffix_data[f"{merge_col}_5gw"] = suffix_data[merge_col]
                            suffix_data = suffix_data.drop(merge_col, axis=1)
                    
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
            
            # XP calculation complete - display logic now handled by visualization module
            
            # Use the new visualization function
            return create_xp_results_display(players_xp, target_gameweek, mo), players_xp
            
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
    print("🔍 XP Calculation Debug:")
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
    mo.md(
        r"""
        ## 4️⃣ Form Analytics Dashboard
        
        **Advanced form detection with momentum indicators and transfer insights.**
        
        ### Player Classification:
        - 🔥 **Hot Players**: Excellent recent form - prime transfer targets
        - 📈 **Rising Players**: Improving trend - good value opportunities  
        - ➡️ **Stable Players**: Consistent performance - reliable options
        - 📉 **Declining Players**: Concerning trend - monitor closely
        - ❄️ **Cold Players**: Poor recent form - consider selling
        
        ### Analysis Categories:
        - **Prime Transfer Targets**: Hot players with strong XP projections
        - **Budget-Friendly Options**: Form + value combinations under £7.5m
        - **Sell Candidates**: Cold players in poor form
        - **Priority Sells**: Expensive underperformers (£8m+ in poor form)
        
        ---
        """
    )
    return


@app.cell
def __(players_with_xp, mo):
    # Form Analytics Dashboard - Using visualization module
    from fpl_visualization import create_form_analytics_display
    
    form_insights_display = create_form_analytics_display(players_with_xp, mo)
    form_insights_display


@app.cell
def __(current_squad, players_with_xp, mo):
    # Current Squad Form Analysis - Using visualization module
    from fpl_visualization import create_squad_form_analysis
    
    squad_form_content = create_squad_form_analysis(current_squad, players_with_xp, mo)
    squad_form_content
    
    def get_safe_columns(df, preferred_columns):
        """Get columns that exist in the DataFrame"""
        available_columns = list(df.columns)
        safe_columns = []
        for pref_col in preferred_columns:
            if pref_col in available_columns:
                safe_columns.append(pref_col)
        return safe_columns if safe_columns else available_columns[:3]  # Fallback to first 3 columns
    
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
                top_columns = get_safe_columns(top_performers, ['web_name', 'position', 'momentum', 'recent_points_per_game', 'xP'])
                squad_insights.append(mo.md("### ⭐ Squad Stars (Keep!)"))
                squad_insights.append(mo.ui.table(
                    top_performers[top_columns].round(2),
                    page_size=5
                ))
            
            # Show problem players if any
            if not problem_players.empty:
                problem_columns = get_safe_columns(problem_players, ['web_name', 'position', 'momentum', 'recent_points_per_game', 'xP'])
                squad_insights.append(mo.md("### ⚠️ Problem Players (Consider Selling)"))
                squad_insights.append(mo.ui.table(
                    problem_players[problem_columns].round(2),
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
    mo.md(
        r"""
        ## 5️⃣ Player Performance Trends
        
        **Interactive historical performance tracking and multi-player comparisons.**
        
        Visualize how key attributes change over gameweeks:
        - **Points per gameweek**: Overall performance trends
        - **Expected goals & assists**: Underlying attacking threat
        - **Minutes played**: Rotation risk and game time trends
        - **Value ratios**: Points per £1m efficiency over time
        
        *Use the dropdowns below to explore individual players or compare multiple options.*
        
        ---
        """
    )
    return


@app.cell
def __(mo, pd, players_with_xp):
    # Import visualization function from new module
    from fpl_visualization import create_player_trends_visualization
    
    # Create the trend analysis components - trends are independent of XP calculations
    # They use live historical data from the API directly
    try:
        print(f"🔍 Creating trends visualization (loads historical data independently)")
        
        # Use players_with_xp if available for names, otherwise use empty DataFrame
        player_data = players_with_xp if hasattr(players_with_xp, 'empty') and not players_with_xp.empty else pd.DataFrame()
        
        player_opts, attr_opts, trends_data = create_player_trends_visualization(player_data)
        print(f"✅ Trends created: {len(player_opts)} players, {len(attr_opts)} attributes")
        
    except Exception as e:
        print(f"❌ Error creating trends: {e}")
        player_opts, attr_opts, trends_data = [], [], pd.DataFrame()
    
    return player_opts, attr_opts, trends_data


@app.cell
def __(mo, player_opts, attr_opts):
    # Player and attribute selectors
    if player_opts and attr_opts:
        player_selector = mo.ui.dropdown(
            options=player_opts,  # All players available
            label="Select Player:",
            value=None  # Don't set default value to avoid validation issues
        )
        
        attribute_selector = mo.ui.dropdown(
            options=attr_opts,
            label="Select Attribute:",
            value=None  # Don't set default value to avoid validation issues
        )
        
        multi_player_selector = mo.ui.multiselect(
            options=player_opts,  # All players available for comparison
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
            mo.md("⚠️ **Loading historical data...**"),
            mo.md("*This section loads live gameweek data directly from the API to show trends*")
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
    if (player_selector is not None and hasattr(player_selector, 'value') and player_selector.value and 
        attribute_selector is not None and hasattr(attribute_selector, 'value') and attribute_selector.value):
        
        multi_values = multi_player_selector.value if multi_player_selector is not None and hasattr(multi_player_selector, 'value') else []
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
    mo.md(
        r"""
        ## 6️⃣ Fixture Difficulty Analysis
        
        **5-gameweek fixture heatmaps with dynamic team strength integration.**
        
        ### Difficulty Indicators:
        - 🟢 **Easy Fixtures** (>1.15): Favorable matchups for attacking returns
        - 🟡 **Average Fixtures** (0.85-1.15): Neutral difficulty expectations
        - 🔴 **Hard Fixtures** (<0.85): Challenging opponents, defensive focus
        
        *Difficulty scores update dynamically based on current team strength ratings*
        
        ---
        """
    )
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
    mo.md(
        r"""
        ## 7️⃣ Strategic Transfer Optimization
        
        **Smart 0-3 transfer analysis with advanced budget pool calculations.**
        
        ### Optimization Features:
        - **Intelligent Transfer Count**: Auto-selects optimal 0-3 transfers based on net XP after penalties
        - **Premium Acquisition Planning**: Multi-transfer scenarios for expensive targets
        - **Budget Pool Analysis**: Total available funds including sellable squad value
        - **Constraint Support**: Force include/exclude specific players
        - **5-GW Strategic Focus**: Decisions based on fixture outlook and form trends
        
        ### Transfer Scenarios Analyzed:
        1. **No Transfers**: Keep current squad (baseline)
        2. **1 Transfer**: Replace worst 5-GW performer
        3. **2 Transfers**: Target two weakest links
        4. **3 Transfers**: Major squad overhaul
        5. **Premium Scenarios**: Direct upgrades and funded acquisitions
        
        ---
        """
    )
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
                mo.md("### 🎯 Transfer Constraints"),
                mo.md("*Optional: Set player constraints before running optimization*"),
                mo.md(""),
                must_include_dropdown,
                mo.md(""),
                must_exclude_dropdown,
                mo.md(""),
                mo.md("---"),
                mo.md("### 🚀 Run Optimization"),
                mo.md("*The optimizer analyzes all transfer scenarios (0-3 transfers) and recommends the strategy with highest net expected points after penalties.*"),
                mo.md(""),
                optimize_button,
                mo.md("---")
            ])
        else:
            constraints_ui = mo.vstack([
                mo.md("### ⚠️ Optimization Unavailable"),
                mo.md("*Please complete the expected points calculation first to enable transfer optimization.*"),
                mo.md(""),
                mo.md("**Required Steps:**"),
                mo.md("1. Select a target gameweek above"),
                mo.md("2. Wait for XP calculations to complete"),
                mo.md("3. Return here to run optimization"),
                mo.md("---")
            ])
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
    # Import optimization functions from dedicated module
    from fpl_optimization import optimize_team_with_transfers
    
    # Execute optimization when button is clicked
    optimal_starting_11 = []
    optimization_display = None
    optimal_scenario = {}
    
    if optimize_button is not None and optimize_button.value:
        # Get constraint sets
        must_include_ids = set(must_include_dropdown.value) if must_include_dropdown is not None and must_include_dropdown.value else set()
        must_exclude_ids = set(must_exclude_dropdown.value) if must_exclude_dropdown is not None and must_exclude_dropdown.value else set()
        
        print(f"🎯 Starting transfer optimization with {len(must_include_ids)} must-include, {len(must_exclude_ids)} must-exclude")
        
        try:
            # Run optimization using the module
            result = optimize_team_with_transfers(
                current_squad=current_squad,
                team_data=team_data, 
                players_with_xp=players_with_xp,
                must_include_ids=must_include_ids,
                must_exclude_ids=must_exclude_ids
            )
            
            # Unpack the results
            if isinstance(result, tuple) and len(result) == 3:
                optimization_display, optimal_squad_df, optimal_scenario = result
                
                # Extract starting 11 from optimal squad
                if not optimal_squad_df.empty:
                    from fpl_optimization import get_best_starting_11
                    optimal_starting_11, formation, xp_total = get_best_starting_11(optimal_squad_df)
                    
                    # Create starting 11 display
                    if optimal_starting_11:
                        starting_11_df = pd.DataFrame(optimal_starting_11)
                        
                        # Display columns that are likely to exist
                        display_cols = []
                        for disp_col in ['web_name', 'position', 'name', 'price', 'xP', 'xP_5gw', 'fixture_outlook']:
                            if disp_col in starting_11_df.columns:
                                display_cols.append(disp_col)
                        
                        starting_11_display = mo.vstack([
                            optimization_display,
                            mo.md("---"),
                            mo.md(f"### 🏆 Optimal Starting 11 ({formation})"),
                            mo.md(f"**Total 5-GW XP:** {xp_total:.2f}"),
                            mo.ui.table(starting_11_df[display_cols].round(2) if display_cols else starting_11_df, page_size=11)
                        ])
                        optimization_display = starting_11_display
                    else:
                        optimal_starting_11 = []
                else:
                    optimal_starting_11 = []
            else:
                # Fallback handling
                optimization_display = result if result else mo.md("⚠️ Optimization completed but no results returned")
                optimal_starting_11 = []
                optimal_scenario = {}
                
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            optimization_display = mo.md(f"❌ Optimization failed: {str(e)}")
            optimal_starting_11 = []
            optimal_scenario = {}
    else:
        optimization_display = mo.md("👆 **Click the optimization button above to analyze transfer scenarios**")
    
    # Display the results
    optimization_display
    
    return optimal_starting_11, optimization_display, optimal_scenario


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 8️⃣ Captain Selection
        
        **Risk-adjusted captaincy recommendations based on expected points analysis.**
        
        Captain selection considers:
        - **Double Points Potential**: XP × 2 for captain scoring
        - **Fixture Difficulty**: Opponent strength and venue
        - **Recent Form**: Hot/cold momentum indicators
        - **Minutes Certainty**: Start probability and injury risk
        
        ---
        """
    )
    return


@app.cell
def __(optimal_starting_11, mo):
    # Import captain selection function from dedicated module
    from fpl_optimization import select_captain
    
    # Captain selection with proper header
    if isinstance(optimal_starting_11, list) and len(optimal_starting_11) > 0:
        try:
            # Add section header
            header = mo.md(
                r"""
                ## 8️⃣ Captain Selection
                
                **Risk-adjusted captaincy recommendations based on expected points analysis.**
                
                ---
                """
            )
            captain_recommendations = select_captain(optimal_starting_11, mo)
            
            # Combine header and recommendations
            mo.vstack([header, captain_recommendations])
        except Exception as e:
            print(f"⚠️ Captain selection error: {e}")
            mo.vstack([
                mo.md("## 8️⃣ Captain Selection"),
                mo.md(f"⚠️ **Captain selection error:** {str(e)}")
            ])
    else:
        mo.vstack([
            mo.md("## 8️⃣ Captain Selection"),
            mo.md(
                r"""
                **Please run transfer optimization first to enable captain selection.**
                
                Once you have an optimal starting 11, captain recommendations will appear here based on:
                - Double points potential (XP × 2)  
                - Fixture difficulty and opponent strength
                - Recent form and momentum indicators
                - Minutes certainty and injury risk
                
                ---
                """
            )
        ])
        
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 🚀 Ready to Optimize
        
        **Transfer optimization is ready to use.**
        
        Click the optimization button below to analyze transfer scenarios.
        
        ---
        """
    )
    return


@app.cell
def __(mo):
    # Summary
    mo.md(
        r"""
        ## ✅ Summary
        
        **FPL Gameweek Manager** - Advanced weekly decision making tool
        
        **Current Status:** All features restored and operational
        
        **Available Features:**
        - ✅ Gameweek data loading and team analysis
        - ✅ Team strength analysis with dynamic ratings
        - ✅ Expected points calculations with form weighting
        - ✅ Form analytics dashboard with momentum indicators
        - ✅ Player performance trends visualization
        - ✅ Fixture difficulty analysis
        - ✅ Transfer optimization (0-3 transfer scenarios)
        - ✅ Captain selection (risk-adjusted recommendations)
        
        **Next Steps:**
        1. Select your target gameweek above
        2. Review XP calculations and form analytics
        3. Set any transfer constraints if needed
        4. Run optimization to get transfer recommendations
        5. Review captain selection suggestions
        6. Execute decisions in the FPL app before deadline
        
        ---
        """
    )
    return


if __name__ == "__main__":
    app.run()