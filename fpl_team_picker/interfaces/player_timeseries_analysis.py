import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # 📈 Player Stats Timeseries Analysis

        This notebook allows you to analyze individual player performance over time with:
        - **Player selector** with search functionality
        - **Multiple stat tracking** across gameweeks
        - **Interactive timeseries charts** for trend analysis
        - **Position-specific metrics** comparison

        **Use this for:** Player research, form analysis, and transfer decisions
        """
    )
    return


@app.cell
def _():
    # Core imports
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from client import FPLDataClient

    # Initialize client
    client = FPLDataClient()
    return client, go, make_subplots, pd, px


@app.cell
def _(mo):
    mo.md("## ⚙️ Configuration")
    return


@app.cell
def _(mo):
    # Gameweek range selection
    start_gw = mo.ui.number(
        start=1,
        stop=38,
        step=1,
        value=1,
        label="Start Gameweek:",
    )

    end_gw = mo.ui.number(
        start=1,
        stop=38,
        step=1,
        value=10,
        label="End Gameweek:",
    )

    mo.hstack([start_gw, end_gw], justify="start")
    return end_gw, start_gw


@app.cell
def _(client, mo, pd):
    # Load players using PlayerAnalyticsService with domain models
    try:
        from fpl_team_picker.domain.services import PlayerAnalyticsService
        from fpl_team_picker.adapters.database_repositories import (
            DatabasePlayerRepository,
        )

        # Initialize analytics service
        _player_repo = DatabasePlayerRepository()
        _analytics_service = PlayerAnalyticsService(_player_repo)

        # Get all enriched players with 70+ validated attributes
        enriched_players = _analytics_service.get_all_players_enriched()

        if not enriched_players:
            status_message = mo.md("❌ **No players loaded from database**")
            player_selector = mo.ui.multiselect(
                options=[],
                value=[],
                label="Select Players:",
            )
        else:
            # Get team names for display
            teams_df = client.get_current_teams()
            team_map = dict(zip(teams_df["team_id"], teams_df["short_name"]))

            # Create player options from domain models with type safety
            player_options = []
            for player in enriched_players:
                team_name = team_map.get(player.team_id, "Unknown")
                # Type-safe access to domain model properties
                display_name = f"{player.web_name} ({team_name}) - {player.position}"

                # Add indicators for quick identification
                indicators = []
                if player.is_premium:
                    indicators.append("💎")
                if player.is_penalty_taker:
                    indicators.append("⚽")
                if player.is_differential:
                    indicators.append("🔥")
                if player.has_injury_concern:
                    indicators.append("🤕")

                if indicators:
                    display_name += f" {' '.join(indicators)}"

                player_options.append((display_name, player.player_id))

            player_options.sort(key=lambda x: x[0])  # Sort alphabetically

            # Use multiselect for multiple player comparison
            player_selector = mo.ui.multiselect(
                options=player_options,
                value=[],
                label="Select Players (💎=Premium ⚽=PK taker 🔥=Differential 🤕=Injury):",
            )

            status_message = mo.md(
                f"✅ **Loaded {len(player_options)} players with type-safe domain models** (70+ attributes per player)"
            )

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        status_message = mo.md(
            f"❌ **Error loading players:** {str(e)}\n\n```\n{error_details}\n```"
        )
        player_selector = mo.ui.multiselect(
            options=[],
            value=[],
            label="Select Players:",
        )

    mo.vstack([status_message, player_selector])
    return (player_selector,)


@app.cell
def _(mo, player_selector):
    # Display player selection info
    mo.md(
        f"**Selected Players:** {len(player_selector.value)} / {len(player_selector.options) if hasattr(player_selector, 'options') else 0} available"
    )
    return


@app.cell
def _(mo):
    mo.md("## 📊 Stat Selection")
    return


@app.cell
def _(mo):
    # Comprehensive stats from multiple data sources

    # Stats from gameweek_performance (31 columns)
    gameweek_stats = [
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        "yellow_cards",
        "red_cards",
        "saves",
        "bonus",
        "bps",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "value",
    ]

    # Enhanced stats from get_players_enhanced() (current season aggregates)
    enhanced_stats = [
        "form",
        "points_per_game",
        "selected_by_percent",
        "expected_goals_per_90",
        "expected_assists_per_90",
        "starts",
        "tackles",
        "recoveries",
        "defensive_contribution",
        "value_form",
        "value_season",
        "cost_change_event",
        "cost_change_start",
        "transfers_in",
        "transfers_out",
        "transfers_in_event",
        "transfers_out_event",
        "chance_of_playing_this_round",
        "chance_of_playing_next_round",
    ]

    # Derived metrics from get_derived_player_metrics() (advanced analytics)
    derived_stats = [
        "points_per_million",
        "form_per_million",
        "value_score",
        "value_confidence",
        "form_trend",
        "form_momentum",
        "recent_form_5gw",
        "season_consistency",
        "expected_points_per_game",
        "points_above_expected",
        "overperformance_risk",
        "ownership_trend",
        "ownership_trend_numeric",
        "ownership_risk",
        "transfer_momentum",
        "injury_risk",
        "rotation_risk",
        "set_piece_priority",
        "data_quality_score",
    ]

    # Rankings (from get_players_enhanced())
    ranking_stats = [
        "form_rank",
        "ict_index_rank",
        "points_per_game_rank",
        "influence_rank",
        "creativity_rank",
        "threat_rank",
    ]

    # Set piece indicators (from get_players_enhanced())
    set_piece_stats = [
        "corners_and_indirect_freekicks_order",
        "direct_freekicks_order",
        "penalties_order",
        "penalty_taker",
        "corner_taker",
        "freekick_taker",
    ]

    # Basic player attributes (from PlayerDomain)
    basic_attributes = [
        "price",
        "availability_status",
        "first_name",
        "last_name",
        "team_id",
        "news",
    ]

    # All available stats
    available_stats = (
        gameweek_stats
        + enhanced_stats
        + derived_stats
        + ranking_stats
        + set_piece_stats
        + basic_attributes
    )

    # Stat categories for better organization
    stat_categories = {
        "⭐ Key Metrics": [
            "total_points",
            "expected_points_per_game",
            "form",
            "points_per_game",
            "value_score",
        ],
        "⚽ Attacking": [
            "goals_scored",
            "assists",
            "expected_goals",
            "expected_assists",
            "expected_goal_involvements",
            "expected_goals_per_90",
            "expected_assists_per_90",
        ],
        "🛡️ Defensive": [
            "clean_sheets",
            "goals_conceded",
            "expected_goals_conceded",
            "saves",
            "tackles",
            "recoveries",
            "defensive_contribution",
        ],
        "🎯 Set Pieces": [
            "penalties_order",
            "corners_and_indirect_freekicks_order",
            "direct_freekicks_order",
            "penalties_saved",
            "penalties_missed",
            "penalty_taker",
            "corner_taker",
            "freekick_taker",
            "set_piece_priority",
        ],
        "📊 Advanced": ["bps", "influence", "creativity", "threat", "ict_index"],
        "📈 Form & Consistency": [
            "form",
            "form_trend",
            "form_momentum",
            "recent_form_5gw",
            "season_consistency",
            "points_above_expected",
        ],
        "💰 Value": [
            "value",
            "value_season",
            "points_per_million",
            "form_per_million",
            "value_form",
            "value_score",
            "value_confidence",
            "cost_change_event",
            "cost_change_start",
        ],
        "👥 Ownership": [
            "selected_by_percent",
            "transfers_in",
            "transfers_out",
            "transfers_in_event",
            "transfers_out_event",
            "transfer_momentum",
            "ownership_trend",
            "ownership_trend_numeric",
            "ownership_risk",
        ],
        "🏆 Rankings": [
            "form_rank",
            "ict_index_rank",
            "points_per_game_rank",
            "influence_rank",
            "creativity_rank",
            "threat_rank",
        ],
        "⚠️ Risk Factors": [
            "injury_risk",
            "rotation_risk",
            "overperformance_risk",
            "ownership_risk",
            "minutes",
        ],
        "🃏 Rare Events": ["own_goals", "yellow_cards", "red_cards"],
        "📅 Playing Time": ["minutes", "starts"],
        "🏥 Availability": [
            "availability_status",
            "chance_of_playing_this_round",
            "chance_of_playing_next_round",
        ],
        "📰 News & Info": ["news", "first_name", "last_name", "team_id"],
        "📊 Data Quality": ["data_quality_score"],
    }

    # Create stat selector with categories
    stat_selector = mo.ui.multiselect(
        options=available_stats,
        value=["total_points", "goals_scored", "assists", "ict_index"],
        label="Select Stats to Display:",
    )

    # Display stat categories
    category_info = []
    for category, stats in stat_categories.items():
        category_info.append(f"**{category}:** {', '.join(stats)}")

    mo.vstack([stat_selector, mo.md("\n".join(category_info))])
    return (stat_selector,)


@app.cell
def _(mo):
    mo.md("## 📈 Player Performance Timeseries")
    return


@app.cell
def _(
    client,
    end_gw,
    go,
    make_subplots,
    mo,
    pd,
    player_selector,
    px,
    start_gw,
    stat_selector,
):
    # Load historical data for selected players
    if not player_selector.value or len(player_selector.value) == 0:
        timeseries_viz = mo.md("⚠️ **Please select at least one player first**")
    else:
        try:
            # Load historical gameweek data for all selected players
            _all_historical_data = []
            _player_info_map = {}

            # Get player info for all selected players
            _players_df = client.get_current_players()
            _teams_df = client.get_current_teams()
            _players_with_teams = _players_df.merge(
                _teams_df[["team_id", "short_name"]], on="team_id", how="left"
            )

            # Ensure player_selector.value is a list of integers
            # player_selector.value should already be a list of player_ids (second element of tuple)
            # If it's somehow tuples, extract the second element
            selected_player_ids = []
            for _pid in player_selector.value:
                if isinstance(_pid, tuple):
                    selected_player_ids.append(
                        int(_pid[1])
                    )  # (display_name, player_id)
                else:
                    selected_player_ids.append(int(_pid))

            # Build player info map
            for _player_id in selected_player_ids:
                # Convert to int to ensure type consistency
                _player_id = int(_player_id)
                # Filter using .loc to avoid broadcasting issues
                _player_info = _players_with_teams.loc[
                    _players_with_teams["player_id"] == _player_id
                ]
                if not _player_info.empty:
                    _player_name = _player_info["web_name"].iloc[0]
                    _player_team = _player_info["short_name"].iloc[0]
                    _player_position = _player_info["position"].iloc[0]
                    _player_info_map[_player_id] = (
                        f"{_player_name} ({_player_team}, {_player_position})"
                    )
                else:
                    _player_info_map[_player_id] = f"Player {_player_id}"

            # Load data for each selected player
            for _player_id in selected_player_ids:
                # Ensure it's an int
                _player_id = int(_player_id)
                _player_historical_data = []

                for _gw in range(start_gw.value, end_gw.value + 1):
                    try:
                        _gw_data = client.get_gameweek_performance(_gw)
                        if not _gw_data.empty:
                            # Filter using .isin for safer comparison
                            _player_gw_data = _gw_data.loc[
                                _gw_data["player_id"].isin([_player_id])
                            ].copy()
                            if not _player_gw_data.empty:
                                _player_gw_data["gameweek"] = _gw
                                _player_gw_data["player_name"] = _player_info_map.get(
                                    _player_id, f"Player {_player_id}"
                                )
                                _player_historical_data.append(_player_gw_data)
                    except Exception:
                        continue

                if _player_historical_data:
                    _all_historical_data.extend(_player_historical_data)

            if not _all_historical_data:
                timeseries_viz = mo.md(
                    "❌ **No historical data found for selected players in the selected gameweek range**"
                )
            else:
                # Combine all historical data
                _combined_df = pd.concat(_all_historical_data, ignore_index=True)

                # Create subplots for selected stats
                selected_stats = stat_selector.value
                if not selected_stats:
                    timeseries_viz = mo.md(
                        "⚠️ **Please select at least one stat to display**"
                    )
                else:
                    # Create subplots
                    n_stats = len(selected_stats)
                    cols = min(2, n_stats)
                    rows = (n_stats + 1) // 2

                    fig = make_subplots(
                        rows=rows,
                        cols=cols,
                        subplot_titles=selected_stats,
                        vertical_spacing=0.1,
                    )

                    # Get unique players for color mapping
                    unique_players = _combined_df["player_name"].unique()
                    colors = px.colors.qualitative.Set1[: len(unique_players)]

                    # Add traces for each stat and each player
                    for i, stat in enumerate(selected_stats):
                        _row = (i // cols) + 1
                        _col = (i % cols) + 1

                        if stat in _combined_df.columns:
                            for j, _player_name in enumerate(unique_players):
                                # Filter using .loc to avoid broadcasting issues
                                _player_data = _combined_df.loc[
                                    _combined_df["player_name"] == _player_name
                                ]

                                fig.add_trace(
                                    go.Scatter(
                                        x=_player_data["gameweek"],
                                        y=_player_data[stat],
                                        mode="lines+markers",
                                        name=_player_name,
                                        line=dict(width=2, color=colors[j]),
                                        marker=dict(size=6),
                                        showlegend=(
                                            i == 0
                                        ),  # Only show legend for first subplot
                                    ),
                                    row=_row,
                                    col=_col,
                                )

                    # Update layout
                    selected_players_text = ", ".join(
                        [
                            _player_info_map.get(_pid, f"Player {_pid}")
                            for _pid in selected_player_ids
                        ]
                    )
                    fig.update_layout(
                        title=f"Player Comparison: {selected_players_text}",
                        height=300 * rows,
                        showlegend=True,
                        template="plotly_white",
                    )

                    # Update x-axis labels
                    fig.update_xaxes(title_text="Gameweek")

                    timeseries_viz = mo.ui.plotly(fig)

        except Exception as e:
            timeseries_viz = mo.md(f"❌ **Error loading data:** {str(e)}")

    timeseries_viz
    return


@app.cell
def _(mo):
    mo.md("## 📋 Player Summary")
    return


@app.cell
def _(client, end_gw, mo, pd, player_selector, start_gw):
    # Create player summary table for multiple players
    if not player_selector.value or len(player_selector.value) == 0:
        summary_viz = mo.md("⚠️ **Please select at least one player first**")
    else:
        try:
            # Load historical data for all selected players
            _summary_all_data = []
            _players_df = client.get_current_players()
            _teams_df = client.get_current_teams()
            _players_with_teams = _players_df.merge(
                _teams_df[["team_id", "short_name"]], on="team_id", how="left"
            )

            # Ensure player_selector.value is a list of integers
            # Handle both tuple and scalar cases
            _selected_player_ids = []
            for _pid in player_selector.value:
                if isinstance(_pid, tuple):
                    _selected_player_ids.append(
                        int(_pid[1])
                    )  # (display_name, player_id)
                else:
                    _selected_player_ids.append(int(_pid))

            for _summary_player_id in _selected_player_ids:
                # Ensure it's an int
                _summary_player_id = int(_summary_player_id)
                _summary_historical_data = []
                for _summary_gw in range(start_gw.value, end_gw.value + 1):
                    try:
                        _summary_gw_data = client.get_gameweek_performance(_summary_gw)
                        if not _summary_gw_data.empty:
                            # Filter using .isin for safer comparison
                            _summary_player_gw_data = _summary_gw_data.loc[
                                _summary_gw_data["player_id"].isin([_summary_player_id])
                            ].copy()
                            if not _summary_player_gw_data.empty:
                                # Add gameweek as a new column - use assignment on the copy
                                _summary_player_gw_data["gameweek"] = _summary_gw
                                _summary_historical_data.append(_summary_player_gw_data)
                    except Exception:
                        continue

                if _summary_historical_data:
                    _summary_all_data.extend(_summary_historical_data)

            if not _summary_all_data:
                summary_viz = mo.md("❌ **No data available for summary**")
            else:
                # Combine data and create summary
                # Use pandas concat with proper column alignment
                _summary_combined_df = pd.concat(
                    _summary_all_data, ignore_index=True, sort=False
                )

                # Ensure numeric columns are actually numeric (fix any string contamination)
                numeric_columns = [
                    "total_points",
                    "minutes",
                    "goals_scored",
                    "assists",
                    "clean_sheets",
                    "goals_conceded",
                    "saves",
                    "bonus",
                    "bps",
                    "influence",
                    "creativity",
                    "threat",
                    "ict_index",
                    "expected_goals",
                    "expected_assists",
                    "value",
                ]
                for _ncol in numeric_columns:
                    if _ncol in _summary_combined_df.columns:
                        _summary_combined_df[_ncol] = pd.to_numeric(
                            _summary_combined_df[_ncol], errors="coerce"
                        )

                # Get player names
                _summary_player_names = {}
                for _, _summary_player in _players_with_teams.iterrows():
                    _summary_player_names[_summary_player["player_id"]] = (
                        f"{_summary_player['web_name']} ({_summary_player['short_name']}, {_summary_player['position']})"
                    )

                # Create summary for each player
                summary_rows = []
                for _summary_player_id in _selected_player_ids:
                    # Ensure it's an int
                    _summary_player_id = int(_summary_player_id)
                    # Filter using .isin for safer comparison
                    _summary_player_data = _summary_combined_df.loc[
                        _summary_combined_df["player_id"].isin([_summary_player_id])
                    ]
                    if not _summary_player_data.empty:
                        _summary_player_name = _summary_player_names.get(
                            _summary_player_id, f"Player {_summary_player_id}"
                        )
                        # Convert value from tenths (e.g., 55) to millions (e.g., 5.5)
                        _current_value = "N/A"
                        if (
                            "value" in _summary_player_data.columns
                            and not _summary_player_data["value"].empty
                        ):
                            _value_tenths = _summary_player_data["value"].iloc[-1]
                            _current_value = f"£{_value_tenths / 10:.1f}m"

                        summary_rows.append(
                            {
                                "Player": _summary_player_name,
                                "Gameweeks": len(_summary_player_data),
                                "Total Points": _summary_player_data[
                                    "total_points"
                                ].sum(),
                                "Avg Points": round(
                                    _summary_player_data["total_points"].mean(), 2
                                ),
                                "Goals": _summary_player_data["goals_scored"].sum(),
                                "Assists": _summary_player_data["assists"].sum(),
                                "Clean Sheets": _summary_player_data[
                                    "clean_sheets"
                                ].sum(),
                                "Avg ICT": round(
                                    _summary_player_data["ict_index"].mean(), 2
                                ),
                                "Current Value": _current_value,
                            }
                        )

                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows)
                    # Convert numeric columns to proper types for display
                    numeric_cols = [
                        "Gameweeks",
                        "Total Points",
                        "Avg Points",
                        "Goals",
                        "Assists",
                        "Clean Sheets",
                        "Avg ICT",
                    ]
                    for _col in numeric_cols:
                        if _col in summary_df.columns:
                            summary_df[_col] = pd.to_numeric(
                                summary_df[_col], errors="coerce"
                            )
                    summary_viz = mo.ui.table(summary_df)
                else:
                    summary_viz = mo.md("❌ **No summary data available**")

        except Exception as e:
            import traceback as _traceback

            _error_details = _traceback.format_exc()
            summary_viz = mo.md(
                f"❌ **Error creating summary:** {str(e)}\n\n```\n{_error_details}\n```"
            )

    summary_viz
    return


if __name__ == "__main__":
    app.run()
