import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Consolidated FPL visualization imports
    from fpl_team_picker.visualization.charts import (
        create_xp_results_display,
        create_form_analytics_display,
        create_player_trends_visualization,
        create_trends_chart,
        create_fixture_difficulty_visualization
    )
    return (
        create_fixture_difficulty_visualization,
        create_form_analytics_display,
        create_player_trends_visualization,
        create_trends_chart,
        create_xp_results_display,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # 🎯 FPL Gameweek Manager
    ## Advanced Weekly Decision Making Tool with Form Analytics

    Transform your FPL strategy with data-driven weekly optimization and comprehensive form analysis.


    ---
    """
    )
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(mo):
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
def _():
    from fpl_team_picker.core.data_loader import fetch_fpl_data, fetch_manager_team, process_current_squad
    return fetch_fpl_data, fetch_manager_team, process_current_squad


@app.cell
def _(mo):
    from fpl_team_picker.core.data_loader import get_current_gameweek_info

    # Automatically detect current gameweek
    gw_info = get_current_gameweek_info()
    current_gw = gw_info['current_gameweek']
    status_message = gw_info['message']
    available_data = gw_info['available_data']

    # Still provide manual override option
    gameweek_input = mo.ui.number(
        value=current_gw,
        start=1,
        stop=38,
        label="Target Gameweek (auto-detected, but you can override)"
    )

    # Display status information
    status_display = mo.vstack([
        mo.md("### 📅 Current Gameweek Status"),
        mo.md(status_message),
        mo.md(f"**Available data:** GW{', GW'.join(map(str, available_data)) if available_data else 'None'}"),
        mo.md(""),
        mo.md("**Target Gameweek:**"),
        gameweek_input,
        mo.md("---")
    ])

    status_display
    return (gameweek_input, available_data, gw_info)


@app.cell
def _(
    available_data,
    fetch_fpl_data,
    fetch_manager_team,
    gameweek_input,
    gw_info,
    mo,
    pd,
    process_current_squad,
):

    # Initialize all variables with defaults first to avoid marimo dependency issues
    target_gw = gameweek_input.value if gameweek_input.value else None
    team_data = None
    current_squad = pd.DataFrame()
    players = pd.DataFrame()
    teams = pd.DataFrame() 
    xg_rates = pd.DataFrame()
    fixtures = pd.DataFrame()
    live_data_historical = pd.DataFrame()

    if target_gw:

        # Load core FPL data first
        players, teams, xg_rates, fixtures, _, live_data_historical = fetch_fpl_data(target_gw)

        # Handle team data based on gameweek
        if target_gw == 1:
            previous_gw = None
            # team_data remains None from initialization
            # current_squad remains empty from initialization
        else:
            previous_gw = target_gw - 1
            team_data = fetch_manager_team(previous_gw)

    # Create team display based on gameweek and data availability
    team_display = mo.md("Configure target gameweek above to load team data")

    if target_gw == 1:
        # GW1 - Season start scenario
        available_columns = list(players.columns)

        display_columns = []
        for col_name in ['web_name', 'position', 'name', 'price', 'selected_by_percent']:
            if col_name in available_columns:
                display_columns.append(col_name)

        if not display_columns:
            display_columns = available_columns[:5]

        team_display = mo.vstack([
            mo.md("### 🚀 GW1 - Season Start"),
            mo.md(f"**Optimizing for GW{target_gw}**"),
            mo.md("**Note:** This is the season start - no previous team data is expected."),
            mo.md("Use this interface to analyze player options for your initial squad selection."),
            mo.md("**Available Players:**"),
            mo.ui.table(
                players[display_columns].head(20).round(2) if not players.empty else pd.DataFrame(),
                page_size=20
            )
        ])

    elif target_gw and team_data:
        # Successfully loaded previous team data
        current_squad = process_current_squad(team_data, players, teams)

        team_display = mo.vstack([
            mo.md(f"### ✅ {team_data['entry_name']} (from GW{previous_gw})"),
            mo.md(f"**Previous Points:** {team_data['total_points']:,} | **Bank:** £{team_data['bank']:.1f}m | **Value:** £{team_data['team_value']:.1f}m | **Free Transfers:** {team_data['free_transfers']}"),
            mo.md(f"**Optimizing for GW{target_gw}**"),
            mo.md("**Current Squad:**"),
            mo.ui.table(
                current_squad[['web_name', 'position', 'price']].round(2) if not current_squad.empty else pd.DataFrame(),
                page_size=15
            )
        ])

    elif target_gw and not team_data:
        # No team data available for previous gameweek
        # Use already imported function and data from gameweek input cell
        # gw_info and available_data should be available from that cell

        # Provide context-specific messaging
        if target_gw == 2:
            team_display = mo.vstack([
                mo.md("### ⚠️ GW2 - Previous Team Data Missing"),
                mo.md(f"**Available data:** GW{', GW'.join(map(str, available_data)) if available_data else 'None'}"),
                mo.md(""),
                mo.md("**This usually happens when:**"),
                mo.md("• GW1 hasn't completed yet, or"),
                mo.md("• Your GW1 team data hasn't been saved to the database"),
                mo.md(""),
                mo.md("**Next steps:**"),
                mo.md("• Wait for GW1 to complete and data to be updated, or"),
                mo.md("• Use GW1 mode to analyze initial team selection")
            ])
        else:
            team_display = mo.vstack([
                mo.md(f"### ⚠️ GW{target_gw} - Previous Team Data Missing"),
                mo.md(f"**Available data:** GW{', GW'.join(map(str, available_data)) if available_data else 'None'}"),
                mo.md(""),
                mo.md(f"Cannot load team data from GW{previous_gw}. This may be because:"),
                mo.md(f"• GW{previous_gw} data isn't available yet, or"),
                mo.md("• Database hasn't been updated with your team information"),
                mo.md(""),
                mo.md("**Suggestion:** Try selecting a gameweek where your previous team data is available.")
            ])

    team_display
    return (
        current_squad,
        fixtures,
        live_data_historical,
        players,
        team_data,
        teams,
        xg_rates,
    )


@app.cell
def _(mo):
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
def _(gameweek_input, mo):
    from fpl_team_picker.visualization.charts import create_team_strength_visualization

    # Initialize once
    team_strength_analysis = None

    if gameweek_input.value:
        team_strength_analysis = create_team_strength_visualization(gameweek_input.value, mo)
    else:
        team_strength_analysis = mo.md("Select target gameweek to see team strength analysis")

    team_strength_analysis
    return


@app.cell
def _(mo):
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
def _(
    create_xp_results_display,
    fixtures,
    gameweek_input,
    live_data_historical,
    mo,
    pd,
    players,
    teams,
    xg_rates,
):
    # Import global configuration
    from fpl_team_picker.config import config

    # Initialize variables
    players_with_xp = pd.DataFrame(columns=['web_name', 'position', 'name', 'price', 'player_id', 'xP', 'xP_5gw', 'fixture_outlook'])

    try:
        if not players.empty and gameweek_input.value:
            if config.xp_model.use_ml_model:
                # Use ML XP Model as primary
                from fpl_team_picker.core.ml_xp_model import MLXPModel, merge_1gw_5gw_results
                from fpl_team_picker.core.xp_model import XPModel

                ml_xp_model = MLXPModel(
                    min_training_gameweeks=config.xp_model.ml_min_training_gameweeks,
                    training_gameweeks=config.xp_model.ml_training_gameweeks,
                    position_min_samples=config.xp_model.ml_position_min_samples,
                    ensemble_rule_weight=config.xp_model.ml_ensemble_rule_weight,
                    debug=config.xp_model.debug
                )
                
                # Create rule-based model for ensemble
                rule_xp_model = XPModel(
                    form_weight=config.xp_model.form_weight,
                    form_window=config.xp_model.form_window,
                    debug=config.xp_model.debug
                )
                
                model_type = "ML"
                xp_model = ml_xp_model
                rule_model_for_ensemble = rule_xp_model
            else:
                # Use traditional rule-based model
                from fpl_team_picker.core.xp_model import XPModel, merge_1gw_5gw_results
                
                xp_model = XPModel(
                    form_weight=config.xp_model.form_weight,
                    form_window=config.xp_model.form_window,
                    debug=config.xp_model.debug
                )
                
                model_type = "Rule-Based"
                rule_model_for_ensemble = None

            # Calculate 1GW and 5GW expected points
            if config.xp_model.use_ml_model:
                players_1gw = xp_model.calculate_expected_points(
                    players_data=players,
                    teams_data=teams,
                    xg_rates_data=xg_rates,
                    fixtures_data=fixtures,
                    target_gameweek=gameweek_input.value,
                    live_data=live_data_historical,
                    gameweeks_ahead=1,
                    rule_based_model=rule_model_for_ensemble
                )

                players_5gw = xp_model.calculate_expected_points(
                    players_data=players,
                    teams_data=teams,
                    xg_rates_data=xg_rates,
                    fixtures_data=fixtures,
                    target_gameweek=gameweek_input.value,
                    live_data=live_data_historical,
                    gameweeks_ahead=5,
                    rule_based_model=rule_model_for_ensemble
                )
            else:
                players_1gw = xp_model.calculate_expected_points(
                    players_data=players,
                    teams_data=teams,
                    xg_rates_data=xg_rates,
                    fixtures_data=fixtures,
                    target_gameweek=gameweek_input.value,
                    live_data=live_data_historical,
                    gameweeks_ahead=1
                )

                players_5gw = xp_model.calculate_expected_points(
                    players_data=players,
                    teams_data=teams,
                    xg_rates_data=xg_rates,
                    fixtures_data=fixtures,
                    target_gameweek=gameweek_input.value,
                    live_data=live_data_historical,
                    gameweeks_ahead=5
                )

            # Merge 1GW and 5GW results with derived metrics
            players_with_xp = merge_1gw_5gw_results(players_1gw, players_5gw)

            # Create results display with model information
            model_info = mo.md(f"**Model Type:** {model_type} {'(with rule-based ensemble)' if config.xp_model.use_ml_model and config.xp_model.ml_ensemble_rule_weight > 0 else ''}")
            xp_results = create_xp_results_display(players_with_xp, gameweek_input.value, mo)
            xp_result = mo.vstack([model_info, xp_results])
        else:
            xp_result = mo.md("Load gameweek data first")
    except ImportError as ie:
        xp_result = mo.md(f"❌ Could not load ML XP Model - {str(ie)}")
    except ValueError as ve:
        # Handle insufficient training data gracefully
        if "Need at least" in str(ve):
            xp_result = mo.vstack([
                mo.md("⚠️ **Insufficient Training Data for ML Model**"),
                mo.md(f"Error: {str(ve)}"),
                mo.md("**Falling back to rule-based model...**")
            ])
            try:
                # Fallback to rule-based model
                from fpl_team_picker.core.xp_model import XPModel, merge_1gw_5gw_results
                
                xp_model = XPModel(
                    form_weight=config.xp_model.form_weight,
                    form_window=config.xp_model.form_window,
                    debug=config.xp_model.debug
                )
                
                players_1gw = xp_model.calculate_expected_points(
                    players_data=players,
                    teams_data=teams,
                    xg_rates_data=xg_rates,
                    fixtures_data=fixtures,
                    target_gameweek=gameweek_input.value,
                    live_data=live_data_historical,
                    gameweeks_ahead=1
                )

                players_5gw = xp_model.calculate_expected_points(
                    players_data=players,
                    teams_data=teams,
                    xg_rates_data=xg_rates,
                    fixtures_data=fixtures,
                    target_gameweek=gameweek_input.value,
                    live_data=live_data_historical,
                    gameweeks_ahead=5
                )
                
                players_with_xp = merge_1gw_5gw_results(players_1gw, players_5gw)
                xp_result = mo.vstack([
                    mo.md("✅ **Using Rule-Based Model (ML fallback)**"),
                    create_xp_results_display(players_with_xp, gameweek_input.value, mo)
                ])
            except Exception as fallback_e:
                xp_result = mo.md(f"❌ Both ML and rule-based models failed: {str(fallback_e)}")
        else:
            xp_result = mo.md(f"❌ ML XP Model error: {str(ve)}")
    except Exception as e:
        xp_result = mo.md(f"❌ Critical error in ML XP calculation: {str(e)}")
        # Add debug information
        import traceback
        error_details = traceback.format_exc()
        xp_result = mo.vstack([
            xp_result,
            mo.md("**Debug Information:**"),
            mo.md(f"```\n{error_details}\n```")
        ])

    xp_result
    return (players_with_xp,)


@app.cell
def _(mo):
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
def _(create_form_analytics_display, mo, players_with_xp):
    form_insights_display = create_form_analytics_display(players_with_xp, mo)
    form_insights_display
    return


@app.cell
def _(current_squad, mo, players_with_xp):
    from fpl_team_picker.visualization.charts import create_squad_form_analysis

    squad_form_content = create_squad_form_analysis(current_squad, players_with_xp, mo)
    squad_form_content
    return


@app.cell
def _(mo):
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
def _(create_player_trends_visualization, pd, players_with_xp):

    # Initialize defaults first to avoid marimo dependency issues
    player_opts = []
    attr_opts = []
    trends_data = pd.DataFrame()

    try:
        player_data = players_with_xp if hasattr(players_with_xp, 'empty') and not players_with_xp.empty else pd.DataFrame()

        if not player_data.empty:
            player_opts, attr_opts, trends_data = create_player_trends_visualization(player_data)

    except Exception:
        # Defaults already set above
        pass
    return attr_opts, player_opts, trends_data


@app.cell
def _(attr_opts, mo, player_opts):
    # Initialize variables with safe defaults to avoid marimo dependency issues
    player_selector = mo.ui.dropdown(options=[], value=None)
    attribute_selector = mo.ui.dropdown(options=[], value=None)
    multi_player_selector = mo.ui.multiselect(options=[], value=[])

    if player_opts and attr_opts:
        player_selector = mo.ui.dropdown(
            options=player_opts,
            label="Select Player:",
            value=None
        )

        attribute_selector = mo.ui.dropdown(
            options=attr_opts,
            label="Select Attribute:",
            value=None
        )

        multi_player_selector = mo.ui.multiselect(
            options=player_opts,
            label="Compare Multiple Players (optional):",
            value=[]
        )

        trends_content = [
            mo.md("### 📈 Player Performance Trends"),
            mo.md("*Track how players' attributes change over gameweeks*"),
            mo.hstack([player_selector, attribute_selector]),
            multi_player_selector,
            mo.md("---")
        ]
    else:
        trends_content = [
            mo.md("### 📈 Player Performance Trends"),
            mo.md("⚠️ **Loading historical data...**"),
            mo.md("*This section loads live gameweek data directly from the API to show trends*")
        ]

    trends_ui = mo.vstack(trends_content)

    trends_ui
    return attribute_selector, multi_player_selector, player_selector


@app.cell
def _(
    attribute_selector,
    create_trends_chart,
    mo,
    multi_player_selector,
    player_selector,
    trends_data,
):
    # Initialize once
    trends_chart = None

    if (player_selector is not None and hasattr(player_selector, 'value') and player_selector.value and 
        attribute_selector is not None and hasattr(attribute_selector, 'value') and attribute_selector.value):

        multi_values = multi_player_selector.value if multi_player_selector is not None and hasattr(multi_player_selector, 'value') else []
        selected_player_id = player_selector.value
        selected_attr_val = attribute_selector.value

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
    return


@app.cell
def _(mo):
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
def _(create_fixture_difficulty_visualization, gameweek_input, mo):

    # Initialize once
    fixture_analysis = None

    if gameweek_input.value:
        fixture_analysis = create_fixture_difficulty_visualization(gameweek_input.value, 5, mo)
    else:
        fixture_analysis = mo.md("Select target gameweek to see fixture difficulty analysis")

    fixture_analysis
    return


@app.cell
def _(mo):
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
def _(mo, players_with_xp):
    # Initialize variables with safe defaults to avoid marimo dependency issues
    must_include_dropdown = mo.ui.multiselect(options=[], value=[])
    must_exclude_dropdown = mo.ui.multiselect(options=[], value=[]) 
    optimize_button = mo.ui.run_button(label="Loading...", disabled=True)

    try:
        if not players_with_xp.empty:
            player_options = []
            sort_column = 'xP_5gw' if 'xP_5gw' in players_with_xp.columns else 'xP'

            for _, player in players_with_xp.sort_values(['position', sort_column], ascending=[True, False]).iterrows():
                xp_display = player.get('xP_5gw', player.get('xP', 0))
                
                # Handle different team name column possibilities
                team_name = ""
                if 'name' in player and pd.notna(player['name']):
                    team_name = player['name']
                elif 'team' in player and pd.notna(player['team']):
                    team_name = player['team']
                elif 'team_name' in player and pd.notna(player['team_name']):
                    team_name = player['team_name']
                
                label = f"{player['web_name']} ({player['position']}, {team_name}) - £{player['price']:.1f}m, {xp_display:.2f} 5GW-xP"
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

    except Exception:
        constraints_ui = mo.md("⚠️ Error creating optimization interface - calculate XP first")
        # Ensure variables are never None to avoid dependency issues
        must_include_dropdown = mo.ui.multiselect(options=[], value=[])
        must_exclude_dropdown = mo.ui.multiselect(options=[], value=[])
        optimize_button = mo.ui.run_button(label="Optimization Unavailable", disabled=True)

    constraints_ui
    return must_exclude_dropdown, must_include_dropdown, optimize_button


@app.cell
def _(
    current_squad,
    mo,
    must_exclude_dropdown,
    must_include_dropdown,
    optimize_button,
    pd,
    players_with_xp,
    team_data,
):
    from fpl_team_picker.optimization.optimizer import optimize_team_with_transfers

    optimal_starting_11 = []
    optimization_display = None  # Initialize once

    if optimize_button is not None and optimize_button.value:
        must_include_ids = set(must_include_dropdown.value) if must_include_dropdown is not None and must_include_dropdown.value else set()
        must_exclude_ids = set(must_exclude_dropdown.value) if must_exclude_dropdown is not None and must_exclude_dropdown.value else set()

        try:
            result = optimize_team_with_transfers(
                current_squad=current_squad,
                team_data=team_data, 
                players_with_xp=players_with_xp,
                must_include_ids=must_include_ids,
                must_exclude_ids=must_exclude_ids
            )

            if isinstance(result, tuple) and len(result) == 3:
                optimization_display, optimal_squad_df, best_scenario = result
                # Convert best_scenario to the expected optimal_scenario format for compatibility

                if not optimal_squad_df.empty:
                    from fpl_team_picker.optimization.optimizer import get_best_starting_11, get_bench_players
                    # Use current gameweek XP for starting 11 selection (can change weekly)
                    optimal_starting_11, formation, xp_total = get_best_starting_11(optimal_squad_df, 'xP')

                    if optimal_starting_11:
                        starting_11_df = pd.DataFrame(optimal_starting_11)

                        display_cols = []
                        for disp_col in ['web_name', 'position', 'name', 'price', 'xP', 'xP_5gw', 'fixture_outlook']:
                            if disp_col in starting_11_df.columns:
                                display_cols.append(disp_col)

                        # Get bench players
                        bench_players = get_bench_players(optimal_squad_df, optimal_starting_11, 'xP')
                        bench_components = []

                        if bench_players:
                            bench_df = pd.DataFrame(bench_players)
                            bench_xp_total = sum(p.get('xP', 0) for p in bench_players)

                            bench_display_cols = []
                            for disp_col in ['web_name', 'position', 'name', 'price', 'xP', 'xP_5gw', 'fixture_outlook']:
                                if disp_col in bench_df.columns:
                                    bench_display_cols.append(disp_col)

                            bench_components.extend([
                                mo.md("---"),
                                mo.md("### 🪑 Bench - Current Gameweek"),
                                mo.md(f"**Total Bench GW XP:** {bench_xp_total:.2f} | *Ordered by expected points*"),
                                mo.ui.table(bench_df[bench_display_cols].round(2) if bench_display_cols else bench_df, page_size=4)
                            ])

                        starting_11_display = mo.vstack([
                            optimization_display,
                            mo.md("---"),
                            mo.md(f"### 🏆 Optimal Starting 11 - Current Gameweek ({formation})"),
                            mo.md(f"**Total Current GW XP:** {xp_total:.2f} | *Optimized for this gameweek only*"),
                            mo.ui.table(starting_11_df[display_cols].round(2) if display_cols else starting_11_df, page_size=11)
                        ] + bench_components)
                        optimization_display = starting_11_display
                    else:
                        optimal_starting_11 = []
                else:
                    optimal_starting_11 = []
            else:
                optimization_display = result if result else mo.md("⚠️ Optimization completed but no results returned")
                optimal_starting_11 = []

        except Exception as e:
            optimization_display = mo.md(f"❌ Optimization failed: {str(e)}")
            optimal_starting_11 = []

    if optimization_display is None:
        optimization_display = mo.md("👆 **Click the optimization button above to analyze transfer scenarios**")

    optimization_display
    return (optimal_starting_11,)


@app.cell
def _(mo):
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
def _(mo, optimal_starting_11):
    from fpl_team_picker.optimization.optimizer import select_captain

    if isinstance(optimal_starting_11, list) and len(optimal_starting_11) > 0:
        try:
            header = mo.md(
                r"""
                ## 8️⃣ Captain Selection

                **Risk-adjusted captaincy recommendations based on expected points analysis.**

                ---
                """
            )
            captain_recommendations = select_captain(optimal_starting_11, mo)

            captain_display = mo.vstack([header, captain_recommendations])
        except Exception as e:
            captain_display = mo.vstack([
                mo.md("## 8️⃣ Captain Selection"),
                mo.md(f"⚠️ **Captain selection error:** {str(e)}")
            ])
    else:
        captain_display = mo.vstack([
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

    captain_display
    return


@app.cell
def _(mo):
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
def _(mo):
    mo.md(
        r"""
    ## ✅ Summary

    **FPL Gameweek Manager** - Advanced weekly decision making tool

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
