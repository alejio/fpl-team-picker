"""Clean gameweek manager interface using domain services."""

import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell
def _(mo):
    # Transfer Deadline Display with enhanced formatting
    try:
        from client import get_raw_events_bootstrap
        from datetime import datetime

        events = get_raw_events_bootstrap()
        now = datetime.now()

        if not events.empty:
            # Find current/next deadline
            upcoming_events = events[events["deadline_time"] > now].sort_values(
                "deadline_time"
            )

            if not upcoming_events.empty:
                next_event = upcoming_events.iloc[0]
                deadline_time = next_event["deadline_time"]
                event_name = next_event["name"]

                # Calculate time until deadline
                time_diff = deadline_time - now
                total_hours = time_diff.total_seconds() / 3600

                # Format time remaining with urgency indicators
                if total_hours < 1:
                    minutes = int(time_diff.total_seconds() / 60)
                    time_remaining = f"{minutes} minutes"
                    urgency_color = "#ef4444"  # Red
                    urgency_emoji = "🚨"
                elif total_hours < 6:
                    time_remaining = f"{total_hours:.1f} hours"
                    urgency_color = "#f59e0b"  # Yellow
                    urgency_emoji = "⚠️"
                elif total_hours < 24:
                    time_remaining = f"{total_hours:.1f} hours"
                    urgency_color = "#3b82f6"  # Blue
                    urgency_emoji = "⏰"
                else:
                    days = total_hours / 24
                    time_remaining = f"{days:.1f} days"
                    urgency_color = "#22c55e"  # Green
                    urgency_emoji = "📅"

                # Format deadline display
                deadline_formatted = deadline_time.strftime("%A, %B %d at %I:%M %p")

                deadline_content = f"""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: white; padding: 16px; margin: 12px 0; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <div style="font-size: 1.1em; margin-bottom: 4px;">
    <strong>{urgency_emoji} {event_name} Transfer Deadline</strong>
    </div>
    <div style="font-size: 1.3em; color: {urgency_color}; font-weight: bold; margin-bottom: 4px;">
    {time_remaining} remaining
    </div>
    <div style="font-size: 0.9em; color: #cbd5e1;">
    {deadline_formatted}
    </div>
    </div>
    """
                deadline_display = mo.md(deadline_content)
            else:
                deadline_display = mo.md("""
    <div style="background: #374151; color: white; padding: 16px; margin: 12px 0; border-radius: 8px; text-align: center;">
    <strong>📋 No upcoming transfer deadlines found</strong>
    </div>
    """)
        else:
            deadline_display = mo.md("""
    <div style="background: #374151; color: white; padding: 16px; margin: 12px 0; border-radius: 8px; text-align: center;">
    <strong>📋 Could not fetch deadline information</strong>
    </div>
    """)
    except Exception as e:
        deadline_display = mo.md(f"""
    <div style="background: #fee2e2; border: 1px solid #ef4444; padding: 12px; margin: 8px 0; border-radius: 4px; text-align: center;">
    🔴 <strong>Unable to load transfer deadline</strong><br/>
    <small>Error: {str(e)}</small>
    </div>
    """)

    deadline_display
    return


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
def _(mo):
    # Data Freshness Display with enhanced formatting
    try:
        from client import get_data_freshness

        freshness = get_data_freshness()

        # Get status info
        status = freshness["overall_freshness"]["status"]
        age_desc = freshness["data_age_description"]
        _recommendations_freshness = freshness["recommendations"]

        # Color coding based on status
        if status == "very_fresh":
            status_emoji = "🟢"
            status_color = "#22c55e"
        elif status == "fresh":
            status_emoji = "🟢"
            status_color = "#22c55e"
        elif status == "stale":
            status_emoji = "🟡"
            status_color = "#f59e0b"
        else:  # very_stale
            status_emoji = "🔴"
            status_color = "#ef4444"

        # Create freshness display
        freshness_content = f"""
    ## 📊 Data Status

    <div style="background: #f8fafc; border-left: 4px solid {status_color}; padding: 12px; margin: 8px 0; border-radius: 4px;">
    <strong>{status_emoji} {age_desc}</strong><br/>
    <small style="color: #64748b;">
    Players: {freshness["raw_data_freshness"]["raw_players_bootstrap"]["age_description"]} |
    Events: {freshness["raw_data_freshness"]["raw_events_bootstrap"]["age_description"]} |
    Fixtures: {freshness["raw_data_freshness"]["raw_fixtures"]["age_description"]}
    </small>
    </div>
    """

        # Add recommendations if data is stale
        if status in ["stale", "very_stale"]:
            rec_text = "<br/>".join([f"• {rec}" for rec in _recommendations_freshness])
            freshness_content += f"""
    <div style="background: #fef3c7; border: 1px solid #f59e0b; padding: 8px; margin: 4px 0; border-radius: 4px; font-size: 0.9em;">
    <strong>💡 Recommendations:</strong><br/>
    {rec_text}
    </div>
    """

        data_freshness_display = mo.md(freshness_content)

    except Exception as e:
        data_freshness_display = mo.md(f"""
    ## 📊 Data Status
    <div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 12px; margin: 8px 0; border-radius: 4px;">
    🔴 <strong>Unable to check data freshness</strong><br/>
    <small>Error: {str(e)}</small>
    </div>
    """)

    data_freshness_display
    return


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
def _(mo):
    # Gameweek input with current gameweek detection
    try:
        from fpl_team_picker.domain.services import (
            DataOrchestrationService as _DataOrchestrationService,
        )

        _orchestration_service = _DataOrchestrationService()
        current_gw_info = _orchestration_service.get_current_gameweek_info()
        current_gw = current_gw_info.get("current_gameweek", 1)
        current_gw_info.get("next_gameweek", current_gw + 1)
        default_gw = current_gw  # Use current gameweek as default, not next
    except Exception:
        default_gw = 1

    gameweek_input = mo.ui.number(
        start=1, stop=38, value=default_gw, step=1, label="Target Gameweek"
    )

    mo.vstack(
        [
            mo.md(f"**Current Default:** GW{default_gw}"),
            gameweek_input,
            mo.md("*Select the gameweek you want to analyze and optimize for.*"),
        ]
    )
    return (gameweek_input,)


@app.cell
def _(gameweek_input, mo):
    # Load gameweek data using domain service
    gameweek_data = None

    if gameweek_input.value:
        try:
            from fpl_team_picker.domain.services import (
                DataOrchestrationService as _DataOrchestrationService2,
            )

            _orchestration_service_2 = _DataOrchestrationService2()
            gameweek_data = _orchestration_service_2.load_gameweek_data(
                gameweek_input.value
            )
            status_display = mo.vstack(
                [
                    mo.md("### 📊 Data Loading Status"),
                    mo.md(
                        f"✅ **Gameweek {gameweek_input.value} data loaded successfully**"
                    ),
                    mo.md(f"- **Players:** {len(gameweek_data.get('players', []))}"),
                    mo.md(f"- **Teams:** {len(gameweek_data.get('teams', []))}"),
                    mo.md(f"- **Fixtures:** {len(gameweek_data.get('fixtures', []))}"),
                    mo.md(
                        f"- **Current Squad:** {'✅ Found' if gameweek_data.get('current_squad') is not None else '❌ Not found'}"
                    ),
                    mo.md(
                        f"- **Manager Team:** {'✅ Found' if gameweek_data.get('manager_team') is not None else '❌ Not found'}"
                    ),
                ]
            )
        except Exception as e:
            status_display = mo.md(f"❌ **Error loading data:** {str(e)}")
    else:
        status_display = mo.md("⚠️ **Select a gameweek to load data**")

    status_display
    return (gameweek_data,)


@app.cell
def _(gameweek_data, gameweek_input, mo):
    # Squad Performance Review - Simple Implementation
    if not gameweek_data or not gameweek_input.value:
        performance_review = mo.md(
            "📊 Select gameweek and load data to see performance analysis"
        )
    else:
        try:
            from fpl_team_picker.domain.services import (
                DataOrchestrationService as _DataServicePerf,
                ExpectedPointsService as _XPServicePerf,
                PerformanceAnalyticsService as _AnalyticsServicePerf,
            )
            from client import FPLDataClient as _ClientPerf

            _current_gw_perf = gameweek_input.value
            previous_gw = max(1, _current_gw_perf - 1)

            # Get team data
            _team_data_perf = gameweek_data.get("manager_team")

            if _team_data_perf and "picks" in _team_data_perf:
                # Load previous gameweek predictions
                _data_svc = _DataServicePerf()
                prev_data = _data_svc.load_gameweek_data(
                    target_gameweek=previous_gw, form_window=5
                )

                _xp_svc = _XPServicePerf()
                prev_predictions = _xp_svc.calculate_combined_results(
                    prev_data, use_ml_model=False
                )

                # Get actual results
                _client = _ClientPerf()
                actual_results = _client.get_gameweek_performance(previous_gw)

                if not actual_results.empty:
                    # Analyze performance
                    _analytics_svc = _AnalyticsServicePerf()
                    analysis = _analytics_svc.analyze_squad_performance(
                        prev_predictions,
                        actual_results,
                        _team_data_perf["picks"],
                        previous_gw,
                    )

                    if "error" not in analysis:
                        sa = analysis["squad_analysis"]
                        performance_review = mo.md(
                            f"""
## 🎯 GW{previous_gw} Squad Performance

**Total:** {sa["total_predicted"]} xP → {sa["total_actual"]} pts ({sa["difference"]:+.1f})
**Accuracy:** {sa["accuracy_percentage"]:.1f}% • **Players:** {sa["players_analyzed"]}

**Top 5 Performers:**
"""
                            + "\n".join(
                                [
                                    f"• {p['web_name']} ({p['position']}): {p['xP']:.1f} → {p['total_points']} ({p['xP_diff']:+.1f})"
                                    for p in analysis["individual_performance"][:5]
                                ]
                            )
                        )
                    else:
                        performance_review = mo.md(f"❌ {analysis['error']}")
                else:
                    performance_review = mo.md(
                        f"❌ No actual data available for GW{previous_gw}"
                    )
            else:
                performance_review = mo.md(
                    "📊 Load your team data to see squad performance"
                )

        except Exception as e:
            performance_review = mo.md(f"❌ Error: {str(e)}")

    performance_review
    return


@app.cell
def _(mo):
    mo.md(r"""## 🎯 Last Gameweek Performance Review

**Compare actual player performance vs predicted xP from the previous gameweek**

This section analyzes how well the expected points model predicted actual results, including your specific team's performance.
""")
    return


@app.cell
def _(mo):
    # Simple test to ensure this cell always shows
    test_display = mo.md("""
🔄 **Performance Review Section**
""")
    test_display
    return (test_display,)


@app.cell
def _(mo):
    # Performance review cell with minimal dependencies
    performance_review_status = mo.md("""
📊 **Ready for Performance Analysis**

To see performance review:
1. Select a gameweek (GW2 or higher)
2. Load gameweek data in the section above
3. The analysis will appear here automatically

*Features:*
- Model accuracy metrics
- Your team's specific performance
- Best/worst predictions
- Detailed player comparisons
""")
    performance_review_status
    return


@app.cell
def _(mo):
    mo.md(r"""## 2️⃣ Team Strength Analysis""")
    return


@app.cell
def _(gameweek_input, mo):
    # Team strength analysis using visualization service
    from fpl_team_picker.visualization.charts import create_team_strength_visualization

    # Initialize once
    team_strength_analysis = None

    if gameweek_input.value:
        team_strength_analysis = create_team_strength_visualization(
            gameweek_input.value, mo
        )
    else:
        team_strength_analysis = mo.md(
            "Select target gameweek to see team strength analysis"
        )

    team_strength_analysis
    return


@app.cell
def _(mo):
    mo.md(r"""## 3️⃣ Expected Points Engine""")
    return


@app.cell
def _(gameweek_data, mo):
    # Calculate expected points using domain service
    if gameweek_data:
        from fpl_team_picker.domain.services import ExpectedPointsService
        from fpl_team_picker.config import config
        from fpl_team_picker.visualization.charts import create_xp_results_display

        xp_service = ExpectedPointsService()
        players_with_xp = xp_service.calculate_combined_results(
            gameweek_data, use_ml_model=config.xp_model.use_ml_model
        )
        model_info = xp_service.get_model_info(config.xp_model.use_ml_model)

        xp_results_display = create_xp_results_display(
            players_with_xp, gameweek_data["target_gameweek"], mo
        )

        xp_section_display = mo.vstack(
            [
                mo.md(f"**Model Type:** {model_info['type']}"),
                xp_results_display,
            ]
        )
    else:
        import pandas as _pd

        players_with_xp = _pd.DataFrame()
        xp_section_display = mo.md("### 🎯 Expected Points\nLoad gameweek data first")

    xp_section_display
    return (players_with_xp,)


@app.cell
def _(mo):
    mo.md(r"""## 4️⃣ Form Analytics Dashboard""")
    return


@app.cell
def _(mo, players_with_xp):
    # Form Analytics Dashboard using visualization function
    if not players_with_xp.empty:
        from fpl_team_picker.visualization.charts import create_form_analytics_display

        form_analytics_display = create_form_analytics_display(players_with_xp, mo)
    else:
        form_analytics_display = mo.md(
            "⚠️ **Calculate expected points first to enable form analytics**"
        )

    form_analytics_display
    return


@app.cell
def _(gameweek_data, mo, players_with_xp):
    # Squad Form Analysis using visualization function
    if gameweek_data and not players_with_xp.empty:
        _current_squad = gameweek_data.get("current_squad")
        if _current_squad is not None and not _current_squad.empty:
            from fpl_team_picker.visualization.charts import create_squad_form_analysis

            squad_form_content = create_squad_form_analysis(
                _current_squad, players_with_xp, mo
            )
        else:
            squad_form_content = mo.md(
                "⚠️ Load team data first to enable squad form analysis"
            )
    else:
        squad_form_content = mo.md(
            "⚠️ Calculate expected points and load team data first to enable squad form analysis"
        )

    squad_form_content
    return


@app.cell
def _(mo):
    mo.md(r"""## 5️⃣ Player Performance Trends""")
    return


@app.cell
def _(gameweek_data, mo, players_with_xp):
    # Player Performance Trends using visualization function
    if not players_with_xp.empty and gameweek_data:
        from fpl_team_picker.visualization.charts import (
            create_player_trends_visualization,
        )

        player_opts, attr_opts, trends_data = create_player_trends_visualization(
            players_with_xp
        )

        trends_display = mo.vstack(
            [
                mo.md("### 📈 Player Performance Trends"),
                mo.md("*Track how players' attributes change over gameweeks*"),
                mo.md("**🎯 Top Performers (by Expected Points):**"),
                mo.ui.table(
                    players_with_xp.nlargest(15, "xP")[
                        ["web_name", "position", "xP", "price"]
                    ].round(2),
                    page_size=10,
                ),
                mo.md(
                    "*Use the performance analytics service for detailed historical trends*"
                ),
            ]
        )
    else:
        trends_display = mo.md(
            "### 📈 Player Performance Trends\n⚠️ **Calculate expected points first to enable trends analysis**"
        )

    trends_display
    return


@app.cell
def _(mo):
    mo.md(r"""## 6️⃣ Fixture Difficulty Analysis""")
    return


@app.cell
def _(gameweek_data, gameweek_input, mo):
    # Fixture Difficulty Analysis using visualization function
    if gameweek_input.value:
        from fpl_team_picker.visualization.charts import (
            create_fixture_difficulty_visualization,
        )

        fixture_analysis = create_fixture_difficulty_visualization(
            gameweek_input.value, 5, mo
        )
    else:
        fixture_analysis = mo.md(
            "Select target gameweek to see fixture difficulty analysis"
        )

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
    - **Configurable Horizon**: Choose between 1-GW immediate focus or 5-GW strategic planning

    ### Transfer Scenarios Analyzed:
    1. **No Transfers**: Keep current squad (baseline)
    2. **1 Transfer**: Replace worst performer
    3. **2 Transfers**: Target two weakest links
    4. **3 Transfers**: Major squad overhaul
    5. **Premium Scenarios**: Direct upgrades and funded acquisitions

    ---
    """
    )
    return


@app.cell
def _(mo):
    # Optimization horizon toggle
    optimization_horizon_toggle = mo.ui.radio(
        options=["5gw", "1gw"], value="5gw", label="Optimization Horizon:"
    )

    mo.vstack(
        [
            mo.md("### ⚖️ Choose Optimization Strategy"),
            optimization_horizon_toggle,
            mo.md(
                "**5GW (Strategic)**: Optimizes for 5-gameweek fixture outlook and form trends"
            ),
            mo.md("**1GW (Immediate)**: Focuses only on current gameweek performance"),
            mo.md("---"),
        ]
    )
    return (optimization_horizon_toggle,)


@app.cell
def _(mo, optimization_horizon_toggle, players_with_xp):
    # Transfer Constraints UI
    must_include_dropdown = mo.ui.multiselect(options=[], value=[])
    must_exclude_dropdown = mo.ui.multiselect(options=[], value=[])
    optimize_button = mo.ui.run_button(label="Loading...", disabled=True)

    if not players_with_xp.empty:
        player_options = []
        # Use the selected horizon to determine sort column and display
        horizon = (
            optimization_horizon_toggle.value
            if optimization_horizon_toggle.value
            else "5gw"
        )
        sort_column = (
            "xP_5gw"
            if horizon == "5gw" and "xP_5gw" in players_with_xp.columns
            else "xP"
        )
        horizon_label = "5GW-xP" if horizon == "5gw" else "1GW-xP"

        for _, player in players_with_xp.sort_values(
            ["position", sort_column], ascending=[True, False]
        ).iterrows():
            xp_display = player.get(sort_column, 0)
            team_name = player.get(
                "name", player.get("team", player.get("team_name", ""))
            )

            label = f"{player['web_name']} ({player['position']}, {team_name}) - £{player['price']:.1f}m, {xp_display:.2f} {horizon_label}"
            player_options.append({"label": label, "value": player["player_id"]})

        must_include_dropdown = mo.ui.multiselect(
            options=player_options,
            label="Must Include Players (force these players to be bought/kept)",
            value=[],
        )

        must_exclude_dropdown = mo.ui.multiselect(
            options=player_options,
            label="Must Exclude Players (never consider these players)",
            value=[],
        )

        button_label = (
            f"🚀 Run {horizon.upper()} Optimization (Auto-selects 0-3 transfers)"
        )
        optimize_button = mo.ui.run_button(label=button_label, kind="success")

        constraints_ui = mo.vstack(
            [
                mo.md("### 🎯 Transfer Constraints"),
                mo.md("*Optional: Set player constraints before running optimization*"),
                mo.md(""),
                must_include_dropdown,
                mo.md(""),
                must_exclude_dropdown,
                mo.md(""),
                mo.md("---"),
                mo.md("### 🚀 Run Optimization"),
                mo.md(
                    f"*The optimizer analyzes all transfer scenarios (0-3 transfers) using {horizon.upper()} strategy and recommends the approach with highest net expected points after penalties.*"
                ),
                mo.md(""),
                optimize_button,
                mo.md("---"),
            ]
        )
    else:
        constraints_ui = mo.vstack(
            [
                mo.md("### ⚠️ Optimization Unavailable"),
                mo.md(
                    "*Please complete the expected points calculation first to enable transfer optimization.*"
                ),
                mo.md(""),
                mo.md("**Required Steps:**"),
                mo.md("1. Select a target gameweek above"),
                mo.md("2. Wait for XP calculations to complete"),
                mo.md("3. Return here to run optimization"),
                mo.md("---"),
            ]
        )

    constraints_ui
    return must_exclude_dropdown, must_include_dropdown, optimize_button


@app.cell
def _(
    gameweek_data,
    mo,
    must_exclude_dropdown,
    must_include_dropdown,
    optimize_button,
    optimization_horizon_toggle,
    players_with_xp,
):
    # Transfer optimization using interactive optimization engine
    optimal_starting_11 = []

    if optimize_button is not None and optimize_button.value:
        if not players_with_xp.empty and gameweek_data:
            current_squad = gameweek_data.get("current_squad")
            team_data = gameweek_data.get("manager_team")
            teams = gameweek_data.get("teams")

            if current_squad is not None and not current_squad.empty:
                from fpl_team_picker.domain.services import (
                    TransferOptimizationService as _TransferOptimizationService,
                )
                import pandas as _pd

                must_include_ids = (
                    set(must_include_dropdown.value)
                    if must_include_dropdown.value
                    else set()
                )
                must_exclude_ids = (
                    set(must_exclude_dropdown.value)
                    if must_exclude_dropdown.value
                    else set()
                )

                # Determine optimization horizon without mutating global config
                selected_horizon = (
                    optimization_horizon_toggle.value
                    if optimization_horizon_toggle.value
                    else "5gw"
                )

                # Use domain service for optimization
                _transfer_service = _TransferOptimizationService()

                try:
                    optimization_result = _transfer_service.optimize_transfers(
                        players_with_xp=players_with_xp,
                        current_squad=current_squad,
                        team_data=team_data,
                        teams=teams,
                        optimization_horizon=selected_horizon,
                        must_include_ids=must_include_ids,
                        must_exclude_ids=must_exclude_ids,
                    )

                    # Extract results from domain service
                    optimization_display = optimization_result.get("display_component")
                    optimal_squad_df = optimization_result.get("optimal_squad")

                    # Generate starting 11 using domain service
                    if optimal_squad_df is not None and not optimal_squad_df.empty:
                        optimal_starting_11 = (
                            _transfer_service.get_starting_eleven_from_squad(
                                optimal_squad_df
                            )
                        )
                        _formation = "4-4-2"  # Default formation from domain service
                        xp_total = (
                            sum(p.get("xP", 0) for p in optimal_starting_11)
                            if optimal_starting_11
                            else 0
                        )

                        if optimal_starting_11:
                            _starting_11_df = _pd.DataFrame(optimal_starting_11)

                            # Fix team names - enforce data contract
                            if (
                                "team" in _starting_11_df.columns
                                and teams is not None
                                and not teams.empty
                            ):
                                # Ensure consistent column naming in data contract
                                team_id_col = (
                                    "id" if "id" in teams.columns else "team_id"
                                )
                                team_map = dict(zip(teams[team_id_col], teams["name"]))

                                # Check if mapping will work - fail fast if data contract violated
                                team_values = _starting_11_df["team"].unique()
                                available_teams = set(teams[team_id_col])

                                if not set(team_values).issubset(available_teams):
                                    missing_teams = set(team_values) - available_teams
                                    raise ValueError(
                                        f"Data contract violation: team IDs {missing_teams} not found in teams data. Expected teams: {available_teams}"
                                    )

                                _starting_11_df["name"] = _starting_11_df["team"].map(
                                    team_map
                                )

                            display_cols = [
                                col
                                for col in [
                                    "web_name",
                                    "position",
                                    "name",
                                    "price",
                                    "xP",
                                    "xP_5gw",
                                    "fixture_outlook",
                                ]
                                if col in _starting_11_df.columns
                            ]

                            # Get bench players (remaining players from optimal squad)
                            starting_11_ids = {
                                p.get("player_id") for p in optimal_starting_11
                            }
                            bench_players = optimal_squad_df[
                                ~optimal_squad_df["player_id"].isin(starting_11_ids)
                            ].to_dict("records")
                            bench_components = []

                            if bench_players:
                                bench_df = _pd.DataFrame(bench_players)
                                bench_xp_total = sum(
                                    p.get("xP", 0) for p in bench_players
                                )

                                # Fix team names for bench - enforce data contract
                                if (
                                    "team" in bench_df.columns
                                    and teams is not None
                                    and not teams.empty
                                ):
                                    team_id_col = (
                                        "id" if "id" in teams.columns else "team_id"
                                    )
                                    team_map = dict(
                                        zip(teams[team_id_col], teams["name"])
                                    )

                                    # Check if mapping will work - fail fast if data contract violated
                                    team_values = bench_df["team"].unique()
                                    available_teams = set(teams[team_id_col])

                                    if not set(team_values).issubset(available_teams):
                                        missing_teams = (
                                            set(team_values) - available_teams
                                        )
                                        raise ValueError(
                                            f"Data contract violation: bench team IDs {missing_teams} not found in teams data. Expected teams: {available_teams}"
                                        )

                                    bench_df["name"] = bench_df["team"].map(team_map)

                                bench_display_cols = [
                                    col
                                    for col in [
                                        "web_name",
                                        "position",
                                        "name",
                                        "price",
                                        "xP",
                                        "xP_5gw",
                                        "fixture_outlook",
                                    ]
                                    if col in bench_df.columns
                                ]

                                bench_components.extend(
                                    [
                                        mo.md("---"),
                                        mo.md("### 🪑 Bench - Current Gameweek"),
                                        mo.md(
                                            f"**Total Bench GW XP:** {bench_xp_total:.2f} | *Ordered by expected points*"
                                        ),
                                        mo.ui.table(
                                            bench_df[bench_display_cols].round(2)
                                            if bench_display_cols
                                            else bench_df,
                                            page_size=4,
                                        ),
                                    ]
                                )

                            starting_11_display = mo.vstack(
                                [
                                    optimization_display,
                                    mo.md("---"),
                                    mo.md(
                                        f"### 🏆 Optimal Starting 11 - Current Gameweek ({_formation})"
                                    ),
                                    mo.md(
                                        f"**Total Current GW XP:** {xp_total:.2f} | *Optimized for this gameweek only*"
                                    ),
                                    mo.ui.table(
                                        _starting_11_df[display_cols].round(2)
                                        if display_cols
                                        else _starting_11_df,
                                        page_size=11,
                                    ),
                                ]
                                + bench_components
                            )
                            optimization_display = starting_11_display
                    else:
                        optimization_display = (
                            optimization_display
                            if optimization_display
                            else mo.md(
                                "⚠️ Optimization completed but no results returned"
                            )
                        )

                except Exception as e:
                    optimization_display = mo.md(f"⚠️ Optimization failed: {str(e)}")
            else:
                optimization_display = mo.md(
                    "❌ **Current squad not available** - cannot optimize transfers"
                )
        else:
            optimization_display = mo.md(
                "⚠️ **Calculate expected points first** to enable transfer optimization"
            )
    else:
        optimization_display = mo.md(
            "👆 **Click the optimization button above to analyze transfer scenarios**"
        )

    optimization_display
    return (optimal_starting_11,)


@app.cell
def _(mo):
    mo.md(r"""## 8️⃣ Captain Selection""")
    return


@app.cell
def _(mo, optimal_starting_11):
    # Captain Selection using domain service - clean architecture
    if isinstance(optimal_starting_11, list) and len(optimal_starting_11) > 0:
        from fpl_team_picker.domain.services import (
            TransferOptimizationService as _TransferOptimizationServiceCaptain,
        )
        import pandas as _pd

        _captain_service = _TransferOptimizationServiceCaptain()
        squad_df = _pd.DataFrame(optimal_starting_11)

        captain_recommendation = _captain_service.get_captain_recommendation_from_squad(
            squad_df
        )

        captain_display = mo.vstack(
            [
                mo.md(
                    "**Risk-adjusted captaincy recommendations based on expected points analysis.**"
                ),
                mo.md("---"),
                mo.md(f"""
### 🏆 **Recommended Captain**

**{captain_recommendation["web_name"]}** ({captain_recommendation["position"]})
- **Expected Points:** {captain_recommendation["xP"]:.2f}
- **Reasoning:** {captain_recommendation["reason"]}
"""),
            ]
        )
    else:
        captain_display = mo.md("""
**Please run transfer optimization first to enable captain selection.**

Once you have an optimal starting 11, captain recommendations will appear here based on:
- Double points potential (XP × 2)
- Fixture difficulty and opponent strength
- Recent form and momentum indicators
- Minutes certainty and injury risk

---
""")

    captain_display
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 9️⃣ Chip Assessment

    **Smart chip timing recommendations with traffic light system.**

    ### Chip Assessment Features:
    - **Wildcard Analysis**: Transfer opportunity analysis and fixture run quality evaluation
    - **Free Hit Evaluation**: Double gameweek detection and temporary squad improvement potential
    - **Bench Boost Assessment**: Bench strength calculation and rotation risk assessment
    - **Triple Captain Identification**: Premium captain candidate analysis and fixture quality review
    - **Traffic Light System**: 🟢 RECOMMENDED, 🟡 CONSIDER, 🔴 HOLD status indicators

    ---
    """
    )
    return


@app.cell
def _(gameweek_data, gameweek_input, mo, players_with_xp):
    # Chip Assessment using domain service - clean architecture
    def _format_chip_metrics(metrics: dict) -> str:
        """Format chip metrics for display"""
        formatted_lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_lines.append(
                    f"- **{key.replace('_', ' ').title()}:** {value:.2f}"
                )
            elif isinstance(value, int):
                formatted_lines.append(
                    f"- **{key.replace('_', ' ').title()}:** {value}"
                )
            elif isinstance(value, str):
                formatted_lines.append(
                    f"- **{key.replace('_', ' ').title()}:** {value}"
                )
            else:
                formatted_lines.append(
                    f"- **{key.replace('_', ' ').title()}:** {str(value)}"
                )
        return "\n".join(formatted_lines)

    if gameweek_input.value and not players_with_xp.empty and gameweek_data:
        _current_squad = gameweek_data.get("current_squad")
        _team_data = gameweek_data.get("manager_team")

        if _current_squad is not None and not _current_squad.empty and _team_data:
            from fpl_team_picker.domain.services import ChipAssessmentService

            # Get available chips from team data
            available_chips = _team_data.get(
                "chips_available",
                ["wildcard", "bench_boost", "triple_captain", "free_hit"],
            )
            used_chips = _team_data.get("chips_used", [])

            if available_chips:
                # Use domain service for chip assessment
                chip_service = ChipAssessmentService()

                # Merge current squad with xP data for chip assessment
                current_squad_with_xp = _current_squad.merge(
                    players_with_xp[["player_id", "xP", "xP_5gw"]],
                    on="player_id",
                    how="left",
                )

                # Update gameweek_data with squad and target gameweek for domain service
                chip_gameweek_data = gameweek_data.copy()
                chip_gameweek_data["target_gameweek"] = gameweek_input.value
                chip_gameweek_data["team_data"] = _team_data

                # Run chip assessments using domain service
                try:
                    chip_assessment_result = chip_service.assess_all_chips(
                        gameweek_data=chip_gameweek_data,
                        current_squad=current_squad_with_xp,
                        available_chips=available_chips,
                        target_gameweek=gameweek_input.value,
                    )

                    chip_recommendations = chip_assessment_result.get(
                        "recommendations", {}
                    )

                    # Create display components for each chip
                    chip_displays = []

                    # Chip status overview
                    chip_displays.append(
                        mo.md(f"""
### 🎯 Chip Status Overview
- **Available:** {", ".join(available_chips) if available_chips else "None"}
- **Used this season:** {", ".join(used_chips) if used_chips else "None"}
""")
                    )

                    # Individual chip recommendations
                    for chip_name, recommendation in chip_recommendations.items():
                        _status = recommendation.get("status", "❓")
                        chip_title = recommendation.get("chip_name", chip_name.title())
                        reasoning = recommendation.get(
                            "reasoning", "No reasoning provided"
                        )
                        key_metrics = recommendation.get("key_metrics", {})

                        chip_displays.append(
                            mo.md(f"""
### {_status} {chip_title}
**{reasoning}**

**Key Metrics:**
{_format_chip_metrics(key_metrics)}
""")
                        )

                    chip_assessment_display = mo.vstack(chip_displays)

                except Exception as e:
                    chip_assessment_display = mo.md(
                        f"⚠️ **Chip assessment failed:** {str(e)}"
                    )
            else:
                chip_assessment_display = mo.md(
                    "✅ **All chips used** - No chips available for this season"
                )
        else:
            chip_assessment_display = mo.md(
                "⚠️ **Load team data and calculate xP first** to enable chip assessment"
            )
    else:
        chip_assessment_display = mo.md(
            "⚠️ **Select gameweek and calculate expected points** to enable chip assessment"
        )

    chip_assessment_display
    return


if __name__ == "__main__":
    app.run()
