"""Clean gameweek manager interface using domain services."""

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


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
    # Gameweek Points Timeseries
    from fpl_team_picker.visualization.charts import create_gameweek_points_timeseries

    # TODO: would be nice to overlay average points per week across FPL
    points_timeline = create_gameweek_points_timeseries(mo)
    points_timeline
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
    # TODO: where are these XP coming from? I don't remember storing them
    # Squad Performance Review - Simple Implementation
    # TODO: This is the last gw retro. We need to improve visuals. We should
    # particularly evaluate the quality of the transfers we did
    # it should be moved after the XP calculation
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
                # TODO: this on-the-fly prediction is definitely false
                # we should implement xp storing and manually trigger when
                #  we have finalised our team selection in FPL site
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
    mo.md(r"""## 2️⃣ Team Strength Analysis""")
    return


@app.cell
def _(gameweek_input, mo):
    # TODO: I think this section is out of place. It should be moved to the
    # fixture difficulty section
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
    # TODO: we should move this further up in the notebook
    # so downstream analyses can benefit
    # Calculate expected points using ML pipeline (NEW!)
    if gameweek_data:
        from fpl_team_picker.domain.services import (
            MLExpectedPointsService,
            ExpectedPointsService,
        )
        from fpl_team_picker.config import config
        from fpl_team_picker.visualization.charts import create_xp_results_display
        from pathlib import Path
        import pandas as _pd

        # Choose ML or rule-based model based on config
        use_ml = config.xp_model.use_ml_model
        if config.xp_model.debug:
            print(
                f"🔧 Config: use_ml_model={use_ml}, ml_model_path={config.xp_model.ml_model_path}"
            )

        if use_ml:
            # Use pre-trained ML model specified in config
            # Default: Custom pipeline optimizer trained model
            # Override via config.json or env var: FPL_XP_MODEL_ML_MODEL_PATH
            model_path = Path(config.xp_model.ml_model_path)

            # If relative path, check common model directories
            if not model_path.is_absolute():
                original_path = model_path
                # Check if path already includes models/ directory
                if str(original_path).startswith("models/"):
                    # Already has models/ prefix, just check if it exists
                    if not original_path.exists():
                        # Try as-is for error message
                        model_path = original_path
                else:
                    # Try models/custom first (where custom_pipeline_optimizer saves)
                    custom_path = Path("models/custom") / original_path
                    if custom_path.exists():
                        model_path = custom_path
                    # Then try models/tpot (for TPOT models)
                    elif not original_path.exists():
                        tpot_path = Path("models/tpot") / original_path
                        if tpot_path.exists():
                            model_path = tpot_path
                    # Finally try current directory
                    elif original_path.exists():
                        model_path = original_path
                    else:
                        # Keep original for error message
                        model_path = original_path

            if not model_path.exists():
                raise FileNotFoundError(
                    f"❌ ML model not found: {model_path}\n"
                    f"   Train a model first using:\n"
                    f"   - Custom Pipeline Optimizer (recommended): "
                    f"uv run python scripts/custom_pipeline_optimizer.py train --end-gw <gw> --use-best-params-from <json>\n"
                    f"   - Or fresh search: uv run python scripts/custom_pipeline_optimizer.py train --regressor random-forest --end-gw <gw>\n"
                    f"   - Or TPOT: uv run python scripts/tpot_pipeline_optimizer.py --start-gw 1 --end-gw 8\n"
                    f"   Then set config.xp_model.ml_model_path to the trained .joblib file (e.g., models/custom/random-forest_gw1-10_*.joblib)"
                )

            # Determine model type from path for display
            # TODO: I don't like this if/elif/else but can live with it
            if "tpot" in model_path.name.lower():
                model_type_label = "ML Pipeline (TPOT Auto-Optimized)"
            elif "lgbm" in model_path.name.lower():
                model_type_label = "ML Pipeline (LightGBM)"
            elif "rf" in model_path.name.lower():
                model_type_label = "ML Pipeline (RandomForest)"
            else:
                model_type_label = f"ML Pipeline ({model_path.stem})"

            # Initialize ML service with configured model
            if config.xp_model.debug:
                print(f"🤖 Initializing ML XP Service: {model_path.name}")
            ml_xp_service = MLExpectedPointsService(
                model_path=str(model_path),
                ensemble_rule_weight=config.xp_model.ml_ensemble_rule_weight,
                debug=config.xp_model.debug,
            )
            if config.xp_model.debug:
                print(
                    f"✅ ML XP Service initialized with {len(ml_xp_service.pipeline.named_steps) if hasattr(ml_xp_service, 'pipeline') and ml_xp_service.pipeline else 'N/A'} pipeline steps"
                )

            # Calculate xP using ML service (no fallback - fail explicitly)
            if config.xp_model.debug:
                print(
                    f"🚀 Calculating ML xP for GW{gameweek_data['target_gameweek']} with betting features..."
                )

            # Calculate 1GW predictions
            players_with_xp = ml_xp_service.calculate_expected_points(
                players_data=gameweek_data.get("players", _pd.DataFrame()),
                teams_data=gameweek_data.get("teams", _pd.DataFrame()),
                xg_rates_data=gameweek_data.get("xg_rates", _pd.DataFrame()),
                fixtures_data=gameweek_data.get("fixtures", _pd.DataFrame()),
                target_gameweek=gameweek_data["target_gameweek"],
                live_data=gameweek_data.get("live_data_historical", _pd.DataFrame()),
                gameweeks_ahead=1,
                ownership_trends_df=gameweek_data.get("ownership_trends"),
                value_analysis_df=gameweek_data.get("value_analysis"),
                fixture_difficulty_df=gameweek_data.get("fixture_difficulty"),
                raw_players_df=gameweek_data.get("raw_players"),
                betting_features_df=gameweek_data.get("betting_features"),
            )

            # Calculate 3GW predictions using cascading predictions
            # This properly accounts for fixture difficulty changes across gameweeks
            if config.xp_model.debug:
                print("🔄 Calculating 3GW ML xP using cascading predictions...")
            players_3gw = ml_xp_service.calculate_3gw_expected_points(
                players_data=gameweek_data.get("players", _pd.DataFrame()),
                teams_data=gameweek_data.get("teams", _pd.DataFrame()),
                xg_rates_data=gameweek_data.get("xg_rates", _pd.DataFrame()),
                fixtures_data=gameweek_data.get("fixtures", _pd.DataFrame()),
                target_gameweek=gameweek_data["target_gameweek"],
                live_data=gameweek_data.get("live_data_historical", _pd.DataFrame()),
                ownership_trends_df=gameweek_data.get("ownership_trends"),
                value_analysis_df=gameweek_data.get("value_analysis"),
                fixture_difficulty_df=gameweek_data.get("fixture_difficulty"),
                raw_players_df=gameweek_data.get("raw_players"),
                betting_features_df=gameweek_data.get("betting_features"),
            )

            # Calculate 5GW predictions using cascading predictions
            # This properly accounts for fixture difficulty changes across gameweeks
            if config.xp_model.debug:
                print("🔄 Calculating 5GW ML xP using cascading predictions...")
            players_5gw = ml_xp_service.calculate_5gw_expected_points(
                players_data=gameweek_data.get("players", _pd.DataFrame()),
                teams_data=gameweek_data.get("teams", _pd.DataFrame()),
                xg_rates_data=gameweek_data.get("xg_rates", _pd.DataFrame()),
                fixtures_data=gameweek_data.get("fixtures", _pd.DataFrame()),
                target_gameweek=gameweek_data["target_gameweek"],
                live_data=gameweek_data.get("live_data_historical", _pd.DataFrame()),
                ownership_trends_df=gameweek_data.get("ownership_trends"),
                value_analysis_df=gameweek_data.get("value_analysis"),
                fixture_difficulty_df=gameweek_data.get("fixture_difficulty"),
                raw_players_df=gameweek_data.get("raw_players"),
                betting_features_df=gameweek_data.get("betting_features"),
            )

            # Merge 3GW predictions into main dataframe
            if "player_id" in players_3gw.columns:
                merge_cols = ["player_id", "ml_xP", "xP_uncertainty"]
                # Also include per-gameweek predictions if available
                for col in players_3gw.columns:
                    if col.startswith("xP_gw") or col.startswith("uncertainty_gw"):
                        merge_cols.append(col)

                players_3gw_subset = players_3gw[merge_cols].copy()
                players_3gw_subset = players_3gw_subset.rename(
                    columns={"ml_xP": "xP_3gw", "xP_uncertainty": "xP_uncertainty_3gw"}
                )
                players_with_xp = players_with_xp.merge(
                    players_3gw_subset,
                    on="player_id",
                    how="left",
                    suffixes=("", "_3gw"),
                )
                # Fill any missing 3gw values with 1gw * 3 as fallback
                players_with_xp["xP_3gw"] = players_with_xp["xP_3gw"].fillna(
                    players_with_xp["xP"] * 3
                )
            else:
                # Fallback if merge fails
                players_with_xp["xP_3gw"] = players_with_xp["xP"] * 3

            # Merge 5GW predictions into main dataframe
            if "player_id" in players_5gw.columns:
                merge_cols = ["player_id", "ml_xP", "xP_uncertainty"]
                # Also include per-gameweek predictions if available
                for col in players_5gw.columns:
                    if col.startswith("xP_gw") or col.startswith("uncertainty_gw"):
                        merge_cols.append(col)

                players_5gw_subset = players_5gw[merge_cols].copy()
                players_5gw_subset = players_5gw_subset.rename(
                    columns={"ml_xP": "xP_5gw", "xP_uncertainty": "xP_uncertainty_5gw"}
                )
                players_with_xp = players_with_xp.merge(
                    players_5gw_subset,
                    on="player_id",
                    how="left",
                    suffixes=("", "_5gw"),
                )
                # Fill any missing 5gw values with 1gw * 5 as fallback
                players_with_xp["xP_5gw"] = players_with_xp["xP_5gw"].fillna(
                    players_with_xp["xP"] * 5
                )
            else:
                # Fallback if merge fails
                players_with_xp["xP_5gw"] = players_with_xp["xP"] * 5

            # Enrich with additional season statistics FIRST
            rule_service = ExpectedPointsService()
            players_with_xp = rule_service.enrich_players_with_season_stats(
                players_with_xp
            )
            players_with_xp["xP_per_price"] = (
                players_with_xp["xP"] / players_with_xp["price"]
            )
            players_with_xp["xP_per_price_3gw"] = (
                players_with_xp["xP_3gw"] / players_with_xp["price"]
            )
            players_with_xp["xP_per_price_5gw"] = (
                players_with_xp["xP_5gw"] / players_with_xp["price"]
            )
            # Calculate horizon advantage: difference between 5GW total and 1GW * 5
            # Positive means 5GW fixtures are better on average, negative means worse
            players_with_xp["xP_horizon_advantage"] = players_with_xp["xP_5gw"] - (
                players_with_xp["xP"] * 5
            )

            # Add form analytics columns
            # Calculate form multiplier from recent performance (use form_season if available)
            # TODO: we have clear data models so i don't understand the if/else
            if "form_season" in players_with_xp.columns:
                players_with_xp["form_multiplier"] = (
                    1.0 + (players_with_xp["form_season"].astype(float) - 3.0) / 10.0
                )
                players_with_xp["form_multiplier"] = players_with_xp[
                    "form_multiplier"
                ].clip(0.5, 1.5)
            else:
                players_with_xp["form_multiplier"] = 1.0

            # Calculate recent points per game
            # TODO: we have clear data models so i don't understand the if/else
            # TODO: I also don't understand the "recent" condition. It seems like
            # we are just using the full season for ppg
            if "points_per_game_season" in players_with_xp.columns:
                players_with_xp["recent_points_per_game"] = players_with_xp[
                    "points_per_game_season"
                ]
            elif "points_per_game" in players_with_xp.columns:
                players_with_xp["recent_points_per_game"] = players_with_xp[
                    "points_per_game"
                ]
            else:
                players_with_xp["recent_points_per_game"] = 0.0

            # Assign momentum indicators based on form
            def assign_momentum(row):
                ppg = row.get("recent_points_per_game", 0)
                if ppg >= 6.0:
                    return "🔥"  # Hot
                elif ppg >= 4.5:
                    return "📈"  # Rising
                elif ppg >= 3.0:
                    return "➡️"  # Stable
                elif ppg >= 2.0:
                    return "📉"  # Declining
                else:
                    return "❄️"  # Cold

            players_with_xp["momentum"] = players_with_xp.apply(assign_momentum, axis=1)

            model_info = {
                "type": model_type_label,
                # TODO: these features are hardcoded. Should be a variable count
                "features": "99 features (FPLFeatureEngineer: 5GW rolling, team context, fixtures, betting odds)",
                "status": "✅ ML predictions generated",
            }
        else:
            # Use rule-based model
            # TODO: We need a clear log here that explains we are using rules-based
            # model.
            xp_service = ExpectedPointsService()
            # Respect config setting (should be False here since use_ml is False, but be explicit)
            players_with_xp = xp_service.calculate_combined_results(
                gameweek_data, use_ml_model=False
            )
            players_with_xp = xp_service.enrich_players_with_season_stats(
                players_with_xp
            )
            model_info = xp_service.get_model_info(False)

        # Extract fixture difficulty from ML service output or gameweek_data
        # ML service should have already added fixture_difficulty and fixture_outlook
        if "fixture_difficulty" not in players_with_xp.columns:
            # Extract from gameweek_data if ML service didn't provide it
            fixture_difficulty_df = gameweek_data.get("fixture_difficulty")
            if fixture_difficulty_df is None or fixture_difficulty_df.empty:
                raise ValueError(
                    "fixture_difficulty not found in ML service output and gameweek_data['fixture_difficulty'] "
                    "is missing or empty. DataOrchestrationService should provide fixture_difficulty_df."
                )

            target_gw = gameweek_data["target_gameweek"]
            target_fixture_difficulty = fixture_difficulty_df[
                fixture_difficulty_df["gameweek"] == target_gw
            ].copy()

            if target_fixture_difficulty.empty:
                raise ValueError(
                    f"No fixture difficulty data found for gameweek {target_gw}. "
                    f"fixture_difficulty_df must contain data for the target gameweek."
                )

            if "overall_difficulty" not in target_fixture_difficulty.columns:
                raise ValueError(
                    f"fixture_difficulty_df missing required 'overall_difficulty' column. "
                    f"Available columns: {list(target_fixture_difficulty.columns)}"
                )

            # Rename to fixture_difficulty for consistency in output
            players_with_xp = players_with_xp.merge(
                target_fixture_difficulty[["team_id", "overall_difficulty"]].rename(
                    columns={"overall_difficulty": "fixture_difficulty"}
                ),
                left_on="team",
                right_on="team_id",
                how="left",
                suffixes=("", "_from_fixture"),
            )

            # Validate that all players have fixture difficulty data
            missing_difficulty = players_with_xp["fixture_difficulty"].isna()
            if missing_difficulty.any():
                missing_players = players_with_xp.loc[
                    missing_difficulty, ["player_id", "web_name", "team"]
                ]
                raise ValueError(
                    f"Missing fixture difficulty data for {missing_difficulty.sum()} players. "
                    f"Missing players: {missing_players['web_name'].tolist()}. "
                    f"This indicates a data quality issue - fixture_difficulty_df should contain "
                    f"data for all teams in the target gameweek."
                )

            # Drop the team_id column from merge if it was added
            if "team_id_from_fixture" in players_with_xp.columns:
                players_with_xp = players_with_xp.drop(columns=["team_id_from_fixture"])

        # Calculate fixture_difficulty_5gw as average of next 5 gameweeks
        fixture_difficulty_df = gameweek_data.get("fixture_difficulty")
        if fixture_difficulty_df is None or fixture_difficulty_df.empty:
            raise ValueError(
                "gameweek_data['fixture_difficulty'] is required for calculating fixture_difficulty_5gw. "
                "DataOrchestrationService should provide fixture_difficulty_df."
            )

        target_gw = gameweek_data["target_gameweek"]
        # Get fixture difficulties for next 5 gameweeks
        future_gws = list(range(target_gw, target_gw + 5))
        future_fixture_difficulty = fixture_difficulty_df[
            fixture_difficulty_df["gameweek"].isin(future_gws)
        ].copy()

        if future_fixture_difficulty.empty:
            raise ValueError(
                f"No fixture difficulty data found for gameweeks {future_gws}. "
                f"fixture_difficulty_df must contain data for the next 5 gameweeks."
            )

        if "overall_difficulty" not in future_fixture_difficulty.columns:
            raise ValueError(
                f"fixture_difficulty_df missing required 'overall_difficulty' column. "
                f"Available columns: {list(future_fixture_difficulty.columns)}"
            )

        # Calculate average difficulty per team across the 5 gameweeks
        avg_difficulty = (
            future_fixture_difficulty.groupby("team_id")["overall_difficulty"]
            .mean()
            .reset_index()
        )
        avg_difficulty.columns = ["team_id", "fixture_difficulty_5gw"]

        players_with_xp = players_with_xp.merge(
            avg_difficulty,
            left_on="team",
            right_on="team_id",
            how="left",
            suffixes=("", "_5gw"),
        )

        # Validate that all players have 5GW fixture difficulty data
        missing_difficulty_5gw = players_with_xp["fixture_difficulty_5gw"].isna()
        if missing_difficulty_5gw.any():
            missing_players = players_with_xp.loc[
                missing_difficulty_5gw, ["player_id", "web_name", "team"]
            ]
            raise ValueError(
                f"Missing fixture_difficulty_5gw data for {missing_difficulty_5gw.sum()} players. "
                f"Missing players: {missing_players['web_name'].tolist()}. "
                f"This indicates a data quality issue - fixture_difficulty_df should contain "
                f"data for all teams across the next 5 gameweeks."
            )

        # Drop the team_id column from merge if it was added
        if "team_id_5gw" in players_with_xp.columns:
            players_with_xp = players_with_xp.drop(columns=["team_id_5gw"])

        # Calculate fixture_outlook if not already set by ML service
        # NOTE: Higher fixture_difficulty = easier fixture (inverse of opponent strength)
        # Use configurable thresholds that match the actual distribution
        if "fixture_outlook" not in players_with_xp.columns:
            from fpl_team_picker.config import config

            easy_threshold = config.fixture_difficulty.easy_fixture_threshold
            average_min = config.fixture_difficulty.average_fixture_min

            def get_fixture_outlook(diff):
                if _pd.isna(diff):
                    raise ValueError(
                        "fixture_difficulty is NaN - this should not happen after validation"
                    )
                elif diff >= easy_threshold:
                    return "🟢 Easy"  # High value = easy fixture
                elif diff <= average_min:
                    return "🔴 Hard"  # Low value = hard fixture
                else:
                    return "🟡 Average"

            players_with_xp["fixture_outlook"] = players_with_xp[
                "fixture_difficulty"
            ].apply(get_fixture_outlook)

        xp_results_display = create_xp_results_display(
            players_with_xp, gameweek_data["target_gameweek"], mo
        )

        xp_section_display = mo.vstack(
            [
                mo.md(f"**Model Type:** {model_info['type']}"),
                mo.md(
                    f"_{model_info.get('status', '')}_{model_info.get('features', '')}"
                ),
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
    # TODO: I think this section should have analytics related to all players across FPL
    # TODO: I don't understand this form multiplier. It doesn't seem useful from first
    # glance
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
    # TODO: this section should probably be merged with 4
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
                        [
                            "web_name",
                            "position",
                            "xP",
                            "xP_3gw",
                            "xP_5gw",
                            "xP_uncertainty",
                            "price",
                        ]
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
def _(gameweek_input, mo):
    # TODO: we can include the team strength visualisation in this section
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
    ## 7️⃣ Transfer & Chip Optimization

    **Unified optimization for normal gameweeks and chip usage.**

    ### Optimization Modes:
    - **1 Free Transfer**: Normal gameweek (analyze 0-3 transfers, -4pts per extra transfer)
    - **2 Free Transfers**: Saved transfer from previous week (analyze 0-4 transfers)
    - **15 Free Transfers (Wildcard)**: Rebuild entire squad, £100m budget reset, no penalties

    ---
    """
    )
    return


@app.cell
def _(mo):
    # TODO: I want to always show at the top the "ideal team" for the target gw
    # Free transfer selector and optimization horizon toggle
    free_transfer_selector = mo.ui.dropdown(
        options=["1", "2", "15"],
        value="1",
        label="Free Transfers:",
    )

    optimization_horizon_toggle = mo.ui.radio(
        options=["5gw", "3gw", "1gw"], value="5gw", label="Optimization Horizon:"
    )

    mo.vstack(
        [
            mo.md("### ⚖️ Optimization Configuration"),
            mo.md("**Free Transfers:**"),
            mo.md("- **1**: Normal gameweek (1 free transfer, -4pts per extra)"),
            mo.md("- **2**: Saved transfer from previous week"),
            mo.md("- **15**: Wildcard chip (rebuild entire squad, £100m reset)"),
            free_transfer_selector,
            mo.md(""),
            mo.md("**Optimization Horizon:**"),
            optimization_horizon_toggle,
            mo.md(
                "**5GW (Strategic)**: Optimizes for 5-gameweek fixture outlook and form trends"
            ),
            mo.md(
                "**3GW (Medium-term)**: Optimizes for 3-gameweek fixture outlook and form trends"
            ),
            mo.md("**1GW (Immediate)**: Focuses only on current gameweek performance"),
            mo.md("---"),
        ]
    )
    return free_transfer_selector, optimization_horizon_toggle


@app.cell
def _(
    free_transfer_selector,
    mo,
    optimization_horizon_toggle,
    players_with_xp,
):
    # Transfer Constraints UI - using PlayerAnalyticsService
    must_include_dropdown = mo.ui.multiselect(options=[], value=[])
    must_exclude_dropdown = mo.ui.multiselect(options=[], value=[])
    optimize_button = mo.ui.run_button(label="Loading...", disabled=True)

    if not players_with_xp.empty:
        # Use PlayerAnalyticsService for type-safe player operations
        from fpl_team_picker.domain.services import PlayerAnalyticsService
        from fpl_team_picker.adapters.database_repositories import (
            DatabasePlayerRepository,
        )

        try:
            # Initialize analytics service with repository
            _player_repo = DatabasePlayerRepository()
            _analytics_service = PlayerAnalyticsService(_player_repo)

            # Get all enriched players with 70+ validated attributes
            # TODO: Validate: does this include all features,
            # including feature engineered ones?
            enriched_players = _analytics_service.get_all_players_enriched()

            # Determine horizon for display
            horizon = (
                optimization_horizon_toggle.value
                if optimization_horizon_toggle.value
                else "5gw"
            )
            if horizon == "5gw":
                horizon_label = "5GW-xP"
            elif horizon == "3gw":
                horizon_label = "3GW-xP"
            else:
                horizon_label = "1GW-xP"

            # Get xP values from DataFrame for sorting (temporary bridge)
            if horizon == "5gw" and "xP_5gw" in players_with_xp.columns:
                xp_lookup = players_with_xp.set_index("player_id")["xP_5gw"]
            elif horizon == "3gw" and "xP_3gw" in players_with_xp.columns:
                xp_lookup = players_with_xp.set_index("player_id")["xP_3gw"]
            else:
                xp_lookup = players_with_xp.set_index("player_id")["xP"]

            # Generate UI options from domain models with type safety
            player_options = []
            for player in enriched_players:
                # TODO: this might need an update below, looks obsolete.
                # Get xP from lookup (bridge until xP is in domain model)
                xp_value = xp_lookup.get(player.player_id, 0.0)

                # Type-safe access to domain model properties
                # Note: position is already a string due to use_enum_values=True
                label = f"{player.web_name} ({player.position}) - £{player.price:.1f}m | {xp_value:.1f} {horizon_label}"

                # Add risk indicators using computed properties
                indicators = []
                if player.is_penalty_taker:
                    indicators.append("⚽")
                if player.has_injury_concern:
                    indicators.append("🤕")
                if player.has_rotation_concern:
                    indicators.append("🔄")
                if player.is_high_value:
                    indicators.append("💎")

                if indicators:
                    label += f" {' '.join(indicators)}"

                player_options.append({"label": label, "value": player.player_id})

            # Sort by position and xP
            player_options.sort(
                key=lambda x: (
                    x["label"].split("(")[1].split(")")[0],  # Position
                    -float(x["label"].split("|")[1].split()[0]),  # xP descending
                )
            )

        except Exception as e:
            print(f"⚠️ PlayerAnalyticsService failed: {e}")

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

        # Get free transfer count and create appropriate button label
        free_transfers_count = int(free_transfer_selector.value)
        if free_transfers_count >= 15:
            button_label = f"🃏 Run Wildcard Optimization ({horizon.upper()})"
            button_kind = "warn"  # Yellow for wildcard
            description = f"*Wildcard chip: Rebuild entire squad from scratch using {horizon.upper()} strategy. £100m budget reset, no transfer penalties.*"
        else:
            button_label = f"🚀 Run {horizon.upper()} Optimization ({free_transfers_count} free transfer{'s' if free_transfers_count > 1 else ''})"
            button_kind = "success"  # Green for normal
            max_transfers = min(free_transfers_count + 3, 15)
            description = f"*The optimizer analyzes 0-{max_transfers} transfer scenarios using {horizon.upper()} strategy and recommends the approach with highest net expected points after penalties.*"

        optimize_button = mo.ui.run_button(label=button_label, kind=button_kind)

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
                mo.md(description),
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
    free_transfer_selector,
    gameweek_data,
    mo,
    must_exclude_dropdown,
    must_include_dropdown,
    optimization_horizon_toggle,
    optimize_button,
    players_with_xp,
):
    # before we do the transfer optimisation analysis, so we have a visual comparison.
    # Transfer optimization using interactive optimization engine
    from fpl_team_picker.interfaces.display_utils import (
        resolve_team_names_pydantic,
        DataContractError as _DataContractError2,
    )
    from fpl_team_picker.domain.services import OptimizationService
    from fpl_team_picker.config import config as _config
    import pandas as _pd

    def _create_optimization_summary(best_scenario, optimization_metadata):
        """Create UI display for optimization results (presentation layer)."""
        # Defensive check: ensure we have dictionaries
        # TODO: remove defensive check
        if not isinstance(best_scenario, dict):
            return mo.md(
                f"❌ Error: best_scenario is {type(best_scenario).__name__}, expected dict"
            )
        if not isinstance(optimization_metadata, dict):
            return mo.md(
                f"❌ Error: optimization_metadata is {type(optimization_metadata).__name__}, expected dict"
            )

        # TODO: validate what is optimization_metadata
        method = optimization_metadata.get("method", "unknown")
        horizon_label = optimization_metadata.get("horizon_label", "N/A")
        budget_pool_info = optimization_metadata.get("budget_pool_info", {})
        available_budget = optimization_metadata.get("available_budget", 0.0)

        # Calculate max single acquisition
        max_single_acquisition = min(budget_pool_info.get("total_budget", 0.0), 15.0)

        # Only simulated annealing is supported
        if method == "simulated_annealing":
            sa_iterations = optimization_metadata.get("sa_iterations", 0)
            sa_improvements = optimization_metadata.get("sa_improvements", 0)
            free_transfers = optimization_metadata.get("free_transfers", 1)
            transfers_out = optimization_metadata.get("transfers_out", [])
            transfers_in = optimization_metadata.get("transfers_in", [])
            current_xp = optimization_metadata.get("current_xp", 0.0)
            num_transfers = len(transfers_out)

            # Calculate penalty
            transfer_penalty = (
                max(0, num_transfers - free_transfers)
                * _config.optimization.transfer_cost
            )
            new_squad_xp = best_scenario["net_xp"] + transfer_penalty
            net_xp = best_scenario["net_xp"]
            xp_gain = best_scenario["xp_gain"]

            # Create before/after comparison
            comparison_data = [
                {
                    "Option": "❌ No Transfers",
                    "Squad XP": round(current_xp, 2),
                    "Transfer Penalty": 0,
                    "Net XP": round(current_xp, 2),
                    "vs Current": 0.0,
                },
                {
                    "Option": f"✅ {num_transfers} Transfer(s)",
                    "Squad XP": round(new_squad_xp, 2),
                    "Transfer Penalty": -round(transfer_penalty, 2)
                    if transfer_penalty > 0
                    else 0,
                    "Net XP": round(net_xp, 2),
                    "vs Current": round(xp_gain, 2),
                },
            ]
            comparison_df = _pd.DataFrame(comparison_data)

            strategic_summary = f"""
    ## 🏆 Strategic {horizon_label} Decision: {best_scenario["transfers"]} Transfer(s) Optimal (Simulated Annealing)

    **Recommended Strategy:** {best_scenario["description"]}

    *Decisions based on {horizon_label.lower()} horizon using simulated annealing with {_config.optimization.sa_restarts} restart(s) × {sa_iterations:,} iterations each ({sa_improvements} improvements found in best run)*

    ### 📊 Impact Analysis:
    """

            budget_summary = f"""
    ### 💰 Budget Pool Analysis:
    - **Bank:** £{available_budget:.1f}m | **Sellable Value:** £{budget_pool_info.get("sellable_value", 0.0):.1f}m | **Total Pool:** £{budget_pool_info.get("total_budget", 0.0):.1f}m
    - **Max Single Acquisition:** £{max_single_acquisition:.1f}m
    - **Free Transfers Available:** {free_transfers}
    """

            components = [
                mo.md(strategic_summary),
                mo.ui.table(comparison_df, page_size=5),
                mo.md(budget_summary),
            ]

            # Add transfer details if applicable
            if transfers_out and transfers_in:
                transfer_details = "### 🔄 Recommended Transfers:\n\n"
                for out_player, in_player in zip(transfers_out, transfers_in):
                    transfer_details += f"- **OUT:** {out_player['web_name']} ({out_player['position']}, £{out_player['price']:.1f}m)\n"
                    transfer_details += f"  **IN:** {in_player['web_name']} ({in_player['position']}, £{in_player['price']:.1f}m)\n\n"

                components.append(mo.md(transfer_details))

            return mo.vstack(components)

        else:
            return mo.md(f"⚠️ Unknown optimization method: {method}")

    optimal_starting_11 = []

    if optimize_button is not None and optimize_button.value:
        if not players_with_xp.empty and gameweek_data:
            current_squad = gameweek_data.get("current_squad")
            team_data = gameweek_data.get("manager_team")
            _teams2 = gameweek_data.get("teams")

            if current_squad is not None and not current_squad.empty:
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

                # Use domain service for optimization
                # Update config based on UI toggle
                _config.optimization.optimization_horizon = (
                    optimization_horizon_toggle.value
                    if optimization_horizon_toggle.value
                    else "5gw"
                )
                _optimization_service = OptimizationService()

                try:
                    # Get free transfer count from selector
                    _free_transfers_count = int(free_transfer_selector.value)
                    _free_transfers_override = (
                        _free_transfers_count if _free_transfers_count != 1 else None
                    )

                    optimal_squad_df, best_scenario, optimization_metadata = (
                        _optimization_service.optimize_transfers(
                            players_with_xp=players_with_xp,
                            current_squad=current_squad,
                            team_data=team_data,
                            must_include_ids=must_include_ids,
                            must_exclude_ids=must_exclude_ids,
                            free_transfers_override=_free_transfers_override,
                        )
                    )

                    # Create optimization display in presentation layer
                    optimization_display = _create_optimization_summary(
                        best_scenario=best_scenario,
                        optimization_metadata=optimization_metadata,
                    )

                    # TODO: There is lots of business logic here that we should put elsewhere
                    # Generate starting 11 using domain service
                    if optimal_squad_df is not None and not optimal_squad_df.empty:
                        starting_11_list, _formation, xp_total = (
                            _optimization_service.find_optimal_starting_11(
                                optimal_squad_df
                            )
                        )
                        optimal_starting_11 = starting_11_list

                        if optimal_starting_11:
                            _starting_11_df = _pd.DataFrame(optimal_starting_11)

                            # Fix team names - enforce data contract using Pydantic validation
                            if (
                                "team" in _starting_11_df.columns
                                and _teams2 is not None
                                and not _teams2.empty
                            ):
                                _starting_11_df = resolve_team_names_pydantic(
                                    _starting_11_df,
                                    _teams2,
                                    context="starting 11 display",
                                )

                            display_cols = [
                                col
                                for col in [
                                    "web_name",
                                    "position",
                                    "name",
                                    "price",
                                    "xP",
                                    "xP_uncertainty",
                                    "xP_3gw",
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

                                # Fix team names for bench - enforce data contract using Pydantic validation
                                if (
                                    "team" in bench_df.columns
                                    and _teams2 is not None
                                    and not _teams2.empty
                                ):
                                    bench_df = resolve_team_names_pydantic(
                                        bench_df, _teams2, context="bench display"
                                    )

                                bench_display_cols = [
                                    col
                                    for col in [
                                        "web_name",
                                        "position",
                                        "name",
                                        "price",
                                        "xP",
                                        "xP_uncertainty",
                                        "xP_3gw",
                                        "xP_5gw",
                                        "fixture_outlook",
                                    ]
                                    if col in bench_df.columns
                                ]

                                # Get horizon for bench display
                                bench_horizon = optimization_metadata.get(
                                    "xp_column", "xP"
                                )
                                if bench_horizon == "xP":
                                    bench_xp_label = "1-GW"
                                elif bench_horizon == "xP_3gw":
                                    bench_xp_label = "3-GW"
                                else:
                                    bench_xp_label = "5-GW"

                                bench_components.extend(
                                    [
                                        mo.md("### 🪑 Bench"),
                                        mo.md(
                                            f"**Total Bench XP ({bench_xp_label}):** {bench_xp_total:.2f}"
                                        ),
                                        mo.ui.table(
                                            bench_df[bench_display_cols].round(2)
                                            if bench_display_cols
                                            else bench_df,
                                            page_size=4,
                                        ),
                                    ]
                                )

                            # Determine horizon for display label
                            _horizon_label = optimization_metadata.get(
                                "horizon_label", "Current GW"
                            )

                            starting_11_display = mo.vstack(
                                [
                                    optimization_display,
                                    mo.md("---"),
                                    mo.md(f"## 📋 Optimized Squad ({_horizon_label})"),
                                    mo.md(f"### 🏆 Starting 11 ({_formation})"),
                                    mo.md(
                                        f"**Total {_horizon_label} XP:** {xp_total:.2f}"
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

                except (ValueError, _DataContractError2) as e:
                    # Data contract violations - actionable error messages
                    optimization_display = mo.md(
                        f"❌ **Data contract violation:** {str(e)}"
                    )
                except (KeyError, TypeError) as e:
                    # Missing fields or type mismatches - indicates upstream data issues
                    import traceback

                    error_details = traceback.format_exc()
                    optimization_display = mo.md(
                        f"❌ **Data structure error:** {str(e)}\n\n```\n{error_details}\n```"
                    )
                except Exception as e:
                    # Unexpected errors - preserve for debugging but minimize scope
                    optimization_display = mo.md(
                        f"⚠️ **Unexpected optimization error:** {str(e)}"
                    )
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
            OptimizationService as _OptimizationServiceCaptain,
        )
        from fpl_team_picker.visualization.charts import (
            create_captain_selection_display as _create_captain_display,
        )
        import pandas as _pd

        _captain_service = _OptimizationServiceCaptain()
        squad_df = _pd.DataFrame(optimal_starting_11)

        # Get captain recommendation data (domain logic)
        captain_data = _captain_service.get_captain_recommendation(squad_df, top_n=5)

        # Create UI display (visualization layer)
        captain_display = _create_captain_display(captain_data, mo)
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

    ---
    """
    )
    return


@app.cell
def _(gameweek_data, gameweek_input, mo, players_with_xp):
    # TODO: Wildcard and free hit assessment should be included
    # right after we run optimization leveraging the yet-to-be-created
    # best squad analysis.
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
                merge_xp_cols = ["player_id", "xP"]
                if "xP_3gw" in players_with_xp.columns:
                    merge_xp_cols.append("xP_3gw")
                if "xP_5gw" in players_with_xp.columns:
                    merge_xp_cols.append("xP_5gw")
                current_squad_with_xp = _current_squad.merge(
                    players_with_xp[merge_xp_cols],
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

                except ValueError as e:
                    # Data contract violations in chip assessment
                    chip_assessment_display = mo.md(
                        f"❌ **Chip assessment data error:** {str(e)}"
                    )
                except (KeyError, TypeError, AttributeError) as e:
                    # Missing fields or structure issues in chip data
                    chip_assessment_display = mo.md(
                        f"❌ **Chip data structure error:** {str(e)} - Fix data upstream"
                    )
                except Exception as e:
                    # Unexpected chip assessment errors
                    chip_assessment_display = mo.md(
                        f"⚠️ **Unexpected chip assessment error:** {str(e)}"
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
