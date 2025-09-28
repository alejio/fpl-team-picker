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
        next_gw = current_gw_info.get("next_gameweek", current_gw + 1)
        default_gw = next_gw
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
def _(mo):
    mo.md(r"""## 2️⃣ Team Strength Analysis""")
    return


@app.cell
def _(gameweek_data, gameweek_input, mo):
    # Team strength analysis - simplified direct display
    team_strength_analysis = None

    if gameweek_input.value and gameweek_data:
        teams_data = gameweek_data.get("teams")
        if teams_data is not None and not teams_data.empty:
            # Display team data directly
            display_cols = []
            for team_col in [
                "name",
                "short_name",
                "code",
                "strength",
                "strength_overall_home",
                "strength_overall_away",
            ]:
                if team_col in teams_data.columns:
                    display_cols.append(team_col)

            if not display_cols and len(teams_data.columns) > 0:
                display_cols = list(teams_data.columns)

            if display_cols:
                team_strength_analysis = mo.vstack(
                    [
                        mo.md(
                            f"### 🏆 Team Strength Analysis - GW{gameweek_input.value}"
                        ),
                        mo.md(f"**Teams Analyzed:** {len(teams_data)} teams"),
                        mo.md(
                            f"**Available Columns:** {', '.join(teams_data.columns.tolist())}"
                        ),
                        mo.md("**📊 Team Data:**"),
                        mo.ui.table(teams_data[display_cols], page_size=20),
                    ]
                )
            else:
                team_strength_analysis = mo.md(
                    "❌ No displayable team data columns found"
                )
        else:
            team_strength_analysis = mo.md("❌ No teams data available")
    else:
        team_strength_analysis = mo.md(
            "Select target gameweek and load data to see team strength analysis"
        )

    team_strength_analysis
    return


@app.cell
def _(mo):
    mo.md(r"""## 3️⃣ Expected Points Engine""")
    return


@app.cell
def _(gameweek_data, mo, pd):
    # Calculate expected points using domain service
    players_with_xp = pd.DataFrame()

    try:
        xp_status = "Load gameweek data first"
        if gameweek_data:
            try:
                from fpl_team_picker.domain.services import ExpectedPointsService
                from fpl_team_picker.config import config

                xp_service = ExpectedPointsService()
                xp_result = xp_service.calculate_combined_results(
                    gameweek_data, use_ml_model=config.xp_model.use_ml_model
                )

                players_with_xp = xp_result
                model_info = xp_service.get_model_info(config.xp_model.use_ml_model)
                xp_status = (
                    f"✅ {model_info['type']} Model: {len(players_with_xp)} players"
                )

                # Simple display of top players
                available_cols = []
                for xp_col in [
                    "web_name",
                    "position",
                    "name",
                    "price",
                    "xP",
                    "xP_5gw",
                ]:
                    if xp_col in players_with_xp.columns:
                        available_cols.append(xp_col)

                if available_cols and not players_with_xp.empty:
                    top_players = players_with_xp.nlargest(20, "xP")[available_cols]
                    xp_results_display = mo.vstack(
                        [
                            mo.md("**🎯 Top Expected Points (Current GW):**"),
                            mo.ui.table(top_players.round(2), page_size=10),
                            mo.md(
                                f"**📊 Summary:** {len(players_with_xp)} players analyzed | Average XP: {players_with_xp['xP'].mean():.2f}"
                            ),
                        ]
                    )
                else:
                    xp_results_display = mo.md(
                        "❌ No player data available for display"
                    )

                xp_section_display = mo.vstack(
                    [
                        mo.md("### 🎯 Expected Points"),
                        mo.md(f"**Status:** {xp_status}"),
                        mo.md(f"**Model Type:** {model_info['type']}"),
                        xp_results_display,
                    ]
                )
            except Exception as e:
                xp_status = f"❌ Service error: {str(e)}"
                xp_section_display = mo.md(
                    f"### 🎯 Expected Points\n**Status:** {xp_status}"
                )
        else:
            xp_section_display = mo.md(
                f"### 🎯 Expected Points\n**Status:** {xp_status}"
            )

        xp_section_display

    except Exception as critical_error:
        xp_section_display = mo.md(
            f"### 🎯 Expected Points\n❌ Critical error: {str(critical_error)}"
        )
        xp_section_display
    return (players_with_xp,)


@app.cell
def _(mo):
    mo.md(r"""## 4️⃣ Form Analytics Dashboard""")
    return


@app.cell
def _(gameweek_data, mo, pd, players_with_xp):
    # Form Analytics using PerformanceAnalyticsService
    form_analytics_display = None

    if not players_with_xp.empty and gameweek_data:
        try:
            from fpl_team_picker.domain.services import PerformanceAnalyticsService

            _analytics_service_form = PerformanceAnalyticsService()

            # Get form analytics
            form_data = _analytics_service_form.analyze_player_form(
                players_with_xp, gameweek_data.get("live_data_historical")
            )
            form_analysis = form_data.get("form_analysis", {})
            hot_players = form_analysis.get("hot_players", [])
            cold_players = form_analysis.get("cold_players", [])

            form_analytics_display = mo.vstack(
                [
                    mo.md("### 📈 Form Analytics Dashboard"),
                    mo.md(
                        f"**Analysis Period:** GW{gameweek_data['target_gameweek']} | **Players Analyzed:** {len(players_with_xp)}"
                    ),
                    mo.md("**🔥 Hot Players (Excellent Recent Form):**"),
                    (
                        mo.ui.table(
                            pd.DataFrame(hot_players)[
                                ["web_name", "position", "xP"]
                            ].round(2)
                            if hot_players
                            and isinstance(hot_players, list)
                            and len(hot_players) > 0
                            else pd.DataFrame(
                                {
                                    "Player": ["No hot players found"],
                                    "Position": [""],
                                    "xP": [0],
                                }
                            ),
                            page_size=8,
                        )
                        if hot_players
                        else mo.md("No hot players identified")
                    ),
                    mo.md("**🧊 Cold Players (Poor Recent Form):**"),
                    (
                        mo.ui.table(
                            pd.DataFrame(cold_players)[
                                ["web_name", "position", "xP"]
                            ].round(2)
                            if cold_players
                            and isinstance(cold_players, list)
                            and len(cold_players) > 0
                            else pd.DataFrame(
                                {
                                    "Player": ["No cold players found"],
                                    "Position": [""],
                                    "xP": [0],
                                }
                            ),
                            page_size=8,
                        )
                        if cold_players
                        else mo.md("No cold players identified")
                    ),
                    mo.md(
                        f"**📊 Form Summary:** Analysis complete - {len(hot_players)} hot players, {len(cold_players)} cold players identified"
                    ),
                ]
            )
        except Exception as e:
            form_analytics_display = mo.md(f"❌ **Form analytics error:** {str(e)}")
    else:
        form_analytics_display = mo.md(
            "⚠️ **Calculate expected points first to enable form analytics**"
        )

    form_analytics_display
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7️⃣ Squad Management Dashboard

    **Analyze your current squad with enhanced starting XI selection and bench analysis.**

    ### Squad Analysis Features:
    - **Formation-Flexible Selection**: Automatically selects optimal starting XI from your 15-player squad
    - **Expected Points Integration**: Uses latest XP calculations for lineup decisions
    - **Bench Value Analysis**: Shows expected points from bench players for rotation planning
    - **Captain Integration**: Starting XI feeds into captain selection recommendations

    ---
    """
    )
    return


@app.cell
def _(gameweek_data, mo, pd, players_with_xp):
    # Squad Management using domain services
    squad_display_components = []

    if not players_with_xp.empty and gameweek_data:
        user_squad = gameweek_data.get("current_squad")

        squad_display_components.append(mo.md("### 👥 Squad Management"))

        if user_squad is not None and not user_squad.empty:
            try:
                from fpl_team_picker.domain.services.squad_management_service import (
                    SquadManagementService as _SquadManagementService,
                )

                _squad_service = _SquadManagementService()

                # Merge squad with XP data
                if "xP" not in user_squad.columns and not players_with_xp.empty:
                    user_squad_with_xp = user_squad.merge(
                        players_with_xp[["player_id", "xP", "xP_5gw"]],
                        on="player_id",
                        how="left",
                    )
                else:
                    user_squad_with_xp = user_squad.copy()

                squad_result = _squad_service.get_starting_eleven(user_squad_with_xp)

                if squad_result:
                    starting_11 = squad_result["starting_11"]
                    formation = squad_result["formation"]
                    total_xp = squad_result["total_xp"]

                    starting_11_df = pd.DataFrame(starting_11)

                    squad_display_components.extend(
                        [
                            mo.md("**Squad Type:** Your Current Squad"),
                            mo.md(f"**Formation:** {formation}"),
                            mo.md(f"**Total Expected Points:** {total_xp:.2f}"),
                            mo.md("**Starting XI:**"),
                            mo.ui.table(
                                starting_11_df[["web_name", "position", "xP"]].round(2),
                                page_size=11,
                            ),
                        ]
                    )
                else:
                    squad_display_components.append(
                        mo.md(
                            "❌ Squad analysis failed: No starting eleven data available"
                        )
                    )
            except Exception as e:
                squad_display_components.append(
                    mo.md(f"⚠️ Squad management error: {str(e)}")
                )
        else:
            squad_display_components.append(mo.md("⚠️ No current squad data found"))
    else:
        squad_display_components.extend(
            [
                mo.md("### 👥 Squad Management"),
                mo.md("⚠️ Calculate expected points first to enable squad management"),
            ]
        )

    mo.vstack(squad_display_components) if squad_display_components else mo.md("")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8️⃣ Strategic Transfer Optimization

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
def _(gameweek_data, mo, optimization_horizon_toggle, pd, players_with_xp):
    # Transfer optimization using TransferOptimizationService
    transfer_section_display = None

    if not players_with_xp.empty and gameweek_data:
        try:
            from fpl_team_picker.domain.services import TransferOptimizationService

            _transfer_service = TransferOptimizationService()

            # Get current squad if available
            current_squad = gameweek_data.get("current_squad")
            if current_squad is not None and not current_squad.empty:
                # Use 5GW horizon based on toggle
                use_5gw = optimization_horizon_toggle.value == "5gw"

                # Get transfer optimization
                transfer_results = _transfer_service.optimize_transfers_simple(
                    players_with_xp=players_with_xp,
                    current_squad=current_squad,
                    gameweek_data=gameweek_data,
                    use_5gw_horizon=use_5gw,
                )

                # Display transfer recommendations
                _recommendations_transfer = transfer_results.get(
                    "transfer_recommendations", []
                )
                horizon = transfer_results.get("optimization_horizon", "1gw")

                if _recommendations_transfer:
                    _rec_transfer = _recommendations_transfer[
                        0
                    ]  # Get first recommendation
                    transfer_content = f"""
**Optimization Horizon:** {horizon.upper()}
**Transfer Count:** {_rec_transfer.get("transfer_count", 0)}
**Expected Gain:** {_rec_transfer.get("net_xp_gain", 0.0):.2f} xP
**Total Cost:** {_rec_transfer.get("total_cost", 0.0):.1f}m
**Summary:** {_rec_transfer.get("summary", "No description")}
"""
                else:
                    transfer_content = f"**Optimization Horizon:** {horizon.upper()}\n**Status:** No beneficial transfers found"

                transfer_section_display = mo.md(
                    f"### 🔄 Transfer Optimization\n{transfer_content}"
                )
            else:
                transfer_section_display = mo.md(
                    "### 🔄 Transfer Optimization\n**Status:** Current squad not available - cannot optimize transfers"
                )

        except Exception as e:
            transfer_section_display = mo.md(
                f"### 🔄 Transfer Optimization\n❌ Transfer optimization failed: {str(e)}"
            )
    else:
        transfer_section_display = mo.md(
            "### 🔄 Transfer Optimization\n**Status:** Waiting for expected points data"
        )

    transfer_section_display
    return


@app.cell
def _(gameweek_data, mo, pd, players_with_xp):
    # Captain Selection using SquadManagementService
    captain_selection_display = None

    if not players_with_xp.empty and gameweek_data:
        try:
            from fpl_team_picker.domain.services import SquadManagementService

            squad_service = SquadManagementService()

            # Get top players as potential captains
            top_captains = players_with_xp.nlargest(10, "xP")
            captain_data = squad_service.get_captain_recommendation_from_database(
                players_with_xp
            )

            captain_selection_display = mo.vstack(
                [
                    mo.md("### 🎖️ Captain Selection"),
                    mo.md(
                        f"**Recommended Captain:** {captain_data['web_name']} ({captain_data['position']})"
                    ),
                    mo.md(f"**Expected Points:** {captain_data['xP']:.2f}"),
                    mo.md(f"**Reasoning:** {captain_data['reason']}"),
                    mo.md("**Top Captain Options:**"),
                    mo.ui.table(
                        top_captains[["web_name", "position", "xP", "price"]].round(2),
                        page_size=5,
                    ),
                ]
            )

        except Exception as e:
            captain_selection_display = mo.md(
                f"❌ **Captain selection error:** {str(e)}"
            )
    else:
        captain_selection_display = mo.md(
            "⚠️ **Calculate expected points first to enable captain selection**"
        )

    captain_selection_display
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 🔟 Chip Assessment

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
def _(gameweek_data, mo, players_with_xp):
    # Chip Assessment using ChipAssessmentService
    chip_assessment_display = None

    try:
        if not players_with_xp.empty and gameweek_data:
            _current_squad_chip = gameweek_data.get("current_squad")

            if _current_squad_chip is not None and not _current_squad_chip.empty:
                from fpl_team_picker.domain.services import ChipAssessmentService

                chip_service = ChipAssessmentService()

                # Assess all available chips (assuming all chips are available for demo)
                available_chips = [
                    "wildcard",
                    "free_hit",
                    "bench_boost",
                    "triple_captain",
                ]

                assessment_result = chip_service.assess_all_chips(
                    gameweek_data,
                    _current_squad_chip,
                    available_chips,
                    gameweek_data["target_gameweek"],
                )

                assessment_data = assessment_result
                _recommendations_chip = assessment_data.get("recommendations", {})
                summary = assessment_data.get("summary", "No summary available")

                if _recommendations_chip:
                    chip_displays = []
                    chip_displays.extend(
                        [
                            mo.md("### 🔟 Chip Recommendations"),
                            mo.md(
                                "**Smart chip timing recommendations with traffic light system**"
                            ),
                            mo.md(f"**📊 Summary:** {summary}"),
                            mo.md("---"),
                        ]
                    )

                    # Display each chip recommendation
                    for chip_name, rec in _recommendations_chip.items():
                        _chip_status = rec.get("status", "🟡 UNKNOWN")
                        reasoning = rec.get("reasoning", "No reasoning provided")
                        chip_display_name = chip_name.replace("_", " ").title()

                        chip_displays.extend(
                            [
                                mo.md(f"### {_chip_status} {chip_display_name}"),
                                mo.md(f"**Reasoning:** {reasoning}"),
                                mo.md("---"),
                            ]
                        )

                    chip_assessment_display = mo.vstack(chip_displays)
                else:
                    chip_assessment_display = mo.md(
                        "### 🔟 Chip Recommendations\n✅ **No chip recommendations available**"
                    )
            else:
                chip_assessment_display = mo.md(
                    "### 🔟 Chip Recommendations\n❌ **Chip assessment failed:** Current squad not available"
                )
        else:
            chip_assessment_display = mo.md(
                "### 🔟 Chip Recommendations\n⚠️ **No current squad data found** - load gameweek data first"
            )

    except Exception as e:
        chip_assessment_display = mo.md(
            f"### 🔟 Chip Recommendations\n❌ **Chip assessment error:** {str(e)}"
        )

    chip_assessment_display
    return


@app.cell
def _(mo):
    mo.md(r"""## 5️⃣ Player Performance Trends""")
    return


@app.cell
def _(gameweek_data, mo, players_with_xp):
    # Player Performance Trends using domain services
    try:
        from fpl_team_picker.domain.services import (
            PerformanceAnalyticsService as _PerformanceAnalyticsService,
        )

        trends_display = None
        player_opts = []

        if not players_with_xp.empty and gameweek_data:
            _analytics_service_trends = _PerformanceAnalyticsService()

            # Get performance trends using domain service
            trends_result = _analytics_service_trends.analyze_performance_trends(
                players_with_xp, gameweek_data["target_gameweek"]
            )

            trends_data = trends_result
            top_performers = trends_data.get(
                "top_performers", players_with_xp.nlargest(15, "xP")
            )
            performance_insights = trends_data.get("insights", {})

            # Create player options for dropdown
            if not players_with_xp.empty:
                for _, player in players_with_xp.iterrows():
                    if "web_name" in player and "player_id" in player:
                        player_opts.append(
                            {
                                "label": f"{player['web_name']} ({player.get('position', 'N/A')})",
                                "value": player["player_id"],
                            }
                        )

            # Attribute options for trends

            trends_display = mo.vstack(
                [
                    mo.md("### 📈 Player Performance Trends"),
                    mo.md(
                        f"**Analysis:** Performance trends analysis for GW{gameweek_data['target_gameweek']} | **Players Analyzed:** {len(players_with_xp)}"
                    ),
                    mo.md("**🎯 Top Performers (by Expected Points):**"),
                    mo.ui.table(
                        top_performers[["web_name", "position", "xP", "price"]].round(2)
                        if all(
                            col in top_performers.columns
                            for col in ["web_name", "position", "xP", "price"]
                        )
                        else top_performers.head(15),
                        page_size=10,
                    ),
                    mo.md(
                        f"**📊 Performance Insights:** {performance_insights.get('summary', 'Trends analysis complete using domain services')}"
                    ),
                    mo.md(
                        "*Track how players' attributes change over gameweeks using the performance analytics service*"
                    ),
                ]
            )
        else:
            trends_display = mo.md(
                "### 📈 Player Performance Trends\n⚠️ **Calculate expected points first to enable trends analysis**"
            )

    except Exception as e:
        trends_display = mo.md(
            f"### 📈 Player Performance Trends\n❌ **Error:** {str(e)}"
        )
        player_opts = []

    trends_display
    return


@app.cell
def _(mo):
    mo.md(r"""## 6️⃣ Fixture Difficulty Analysis""")
    return


@app.cell
def _(gameweek_data, mo, pd, players_with_xp):
    # Fixture Difficulty Analysis using domain services
    try:
        from fpl_team_picker.domain.services import FixtureAnalysisService

        fixture_display = None

        if not players_with_xp.empty and gameweek_data:
            fixture_service = FixtureAnalysisService()

            # Analyze fixture difficulty using domain service
            fixture_result = fixture_service.analyze_fixture_difficulty(
                gameweek_data, gameweek_data["target_gameweek"], gameweeks_ahead=5
            )

            fixture_data = fixture_result
            upcoming_fixtures = fixture_data.get("upcoming_fixtures", [])
            difficulty_analysis = fixture_data.get("difficulty_analysis", {})
            team_outlook = fixture_data.get("team_outlook", {})

            fixture_components = [
                mo.md("### 🎯 Fixture Difficulty Analysis"),
                mo.md(
                    f"**Analysis Period:** GW{gameweek_data['target_gameweek']} | **Using Domain Services Architecture**"
                ),
            ]

            if upcoming_fixtures:
                fixtures_df = pd.DataFrame(upcoming_fixtures)
                _display_cols_fixture = [
                    col
                    for col in [
                        "home_team",
                        "away_team",
                        "kickoff_time",
                        "difficulty_home",
                        "difficulty_away",
                    ]
                    if col in fixtures_df.columns
                ]

                fixture_components.extend(
                    [
                        mo.md("**📅 Upcoming Fixtures with Difficulty Ratings:**"),
                        mo.ui.table(
                            fixtures_df[_display_cols_fixture].head(10),
                            page_size=10,
                        )
                        if _display_cols_fixture
                        else mo.md("Fixture data processing..."),
                    ]
                )

            if team_outlook:
                fixture_components.extend(
                    [
                        mo.md("**🏆 Team Fixture Outlook (5GW):**"),
                        mo.md(
                            f"- **Easiest Fixtures:** {', '.join(team_outlook.get('easiest_teams', ['N/A'])[:3])}"
                        ),
                        mo.md(
                            f"- **Hardest Fixtures:** {', '.join(team_outlook.get('hardest_teams', ['N/A'])[:3])}"
                        ),
                        mo.md(
                            f"- **Double Gameweeks:** {', '.join(team_outlook.get('double_gameweeks', ['None']))}"
                        ),
                    ]
                )

            fixture_components.append(
                mo.md(
                    f"**📊 Summary:** {difficulty_analysis.get('summary', 'Fixture difficulty analysis complete using domain services')}"
                )
            )

            fixture_display = mo.vstack(fixture_components)
        else:
            fixture_display = mo.md(
                "### 🎯 Fixture Difficulty Analysis\n⚠️ **Calculate expected points first to enable fixture analysis**"
            )

    except Exception as e:
        fixture_display = mo.md(
            f"### 🎯 Fixture Difficulty Analysis\n❌ **Error:** {str(e)}"
        )

    fixture_display
    return


if __name__ == "__main__":
    app.run()
