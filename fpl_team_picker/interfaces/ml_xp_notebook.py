"""
ML Expected Points (xP) Deep Dive Analysis Notebook

This notebook provides comprehensive analysis of pre-trained TPOT ML models for FPL xP prediction.
The goal is to understand model behavior, identify strengths/weaknesses, and validate performance.

Key Analysis Features:
- Automatic loading of latest pre-trained TPOT model
- Temporal cross-validation for proper time-series evaluation
- Feature importance analysis (which features drive predictions)
- Position-specific error breakdown (where model struggles)
- Prediction bias detection (systematic over/under-prediction)
- Model comparison against rule-based baseline
- Comprehensive feature engineering (117 features) with all enhanced data sources

Analysis Strategy:
- Load historical gameweek data (GW6+ for rolling features)
- Engineer 117 features using production FPLFeatureEngineer
- Load latest TPOT model automatically
- Use temporal walk-forward validation (test on future gameweeks)
- Deep dive into predictions, errors, and model behavior
"""

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px

    # ML libraries
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMRegressor

    # FPL services
    from fpl_team_picker.domain.services import (
        DataOrchestrationService,
        ExpectedPointsService,
    )

    # Production ML modules - USE THESE instead of custom functions
    from fpl_team_picker.domain.services.ml_feature_engineering import (
        FPLFeatureEngineer,
    )
    from fpl_team_picker.domain.services.ml_pipeline_factory import (
        get_team_strength_ratings,
    )
    from client import FPLDataClient

    # Initialize services
    data_service = DataOrchestrationService()
    xp_service = ExpectedPointsService()
    client = FPLDataClient()
    return (
        FPLFeatureEngineer,
        GradientBoostingRegressor,
        GroupKFold,
        LGBMRegressor,
        Pipeline,
        RandomForestRegressor,
        Ridge,
        StandardScaler,
        client,
        cross_val_score,
        data_service,
        get_team_strength_ratings,
        mo,
        np,
        pd,
        px,
        xp_service,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # 🔬 ML Expected Points (xP) Deep Dive Analysis

    **Goal:** Comprehensive analysis of pre-trained TPOT models to understand model behavior,
    validate performance, and identify areas for improvement.

    This notebook automatically loads the latest TPOT model and provides detailed analysis including:
    - Feature importance analysis
    - Position-specific error breakdown
    - Prediction bias detection
    - Temporal cross-validation metrics
    - Comparison with rule-based baseline

    ---
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 1️⃣ Data Loading - Historical Gameweeks""")
    return


@app.cell
def _(mo):
    # Fixed gameweek range for analysis
    # NOTE: Latest TPOT model was trained on GW1-9
    # We load GW1-9 for feature engineering, then use GW6-9 for temporal CV analysis
    start_gw = 1
    end_gw = 9

    mo.vstack(
        [
            mo.md("### 📊 Analysis Configuration"),
            mo.md(f"**Model Training Range:** GW{start_gw}-{end_gw}"),
            mo.md(
                f"**Temporal CV Range:** GW6-{end_gw} (GW6+ required for rolling features)"
            ),
            mo.md("---"),
        ]
    )
    return end_gw, start_gw


@app.cell
def _(client, end_gw, mo, pd, start_gw):
    # Load historical gameweek performance data + fixtures & teams for opponent features
    # Also load enhanced data sources for 117-feature FPLFeatureEngineer
    # NOTE: TPOT model was trained on GW1-9, so we load data in that range for analysis
    historical_data = []
    data_load_status = []
    fixtures_df = pd.DataFrame()
    teams_df = pd.DataFrame()
    ownership_trends_df = pd.DataFrame()
    value_analysis_df = pd.DataFrame()
    fixture_difficulty_df = pd.DataFrame()
    betting_features_df = pd.DataFrame()
    raw_players_df = pd.DataFrame()

    try:
        # Load actual performance data for each gameweek (GW1-9)
        for gw in range(start_gw, end_gw + 1):
            gw_performance = client.get_gameweek_performance(gw)
            if not gw_performance.empty:
                gw_performance["gameweek"] = gw
                historical_data.append(gw_performance)
                data_load_status.append(f"✅ GW{gw}: {len(gw_performance)} players")
            else:
                data_load_status.append(f"⚠️ GW{gw}: No data available")

        if historical_data:
            historical_df = pd.concat(historical_data, ignore_index=True)

            # Enrich with player position data (not in gameweek_performance)
            # Need to join from current_players to get position
            try:
                players_data = client.get_current_players()
                if (
                    "position" in players_data.columns
                    and "player_id" in players_data.columns
                ):
                    # Merge position data
                    historical_df = historical_df.merge(
                        players_data[["player_id", "position"]],
                        on="player_id",
                        how="left",
                    )
                    # Validate all players have position
                    if historical_df["position"].isna().any():
                        missing_count = historical_df["position"].isna().sum()
                        data_load_status.append(
                            f"⚠️ Warning: {missing_count} records missing position data"
                        )
                else:
                    raise ValueError(
                        f"Position column not found in current_players data. "
                        f"Available columns: {list(players_data.columns)}"
                    )
            except Exception as e:
                raise ValueError(
                    f"Failed to enrich historical data with position information: {str(e)}"
                )

            # Load fixtures and teams data for fixture-specific features
            try:
                fixtures_df = client.get_fixtures_normalized()
                teams_df = client.get_current_teams()

                data_load_status.append(
                    f"✅ Fixtures: {len(fixtures_df)} | Teams: {len(teams_df)}"
                )
            except Exception as e:
                data_load_status.append(
                    f"⚠️ Warning: Could not load fixtures/teams: {str(e)}"
                )

            # Load enhanced data sources for 117-feature FPLFeatureEngineer
            try:
                ownership_trends_df = client.get_derived_ownership_trends()
                value_analysis_df = client.get_derived_value_analysis()
                fixture_difficulty_df = client.get_derived_fixture_difficulty()
                data_load_status.append(
                    f"✅ Enhanced: Ownership={len(ownership_trends_df)}, Value={len(value_analysis_df)}, Fixture Difficulty={len(fixture_difficulty_df)}"
                )
            except Exception as e:
                data_load_status.append(
                    f"⚠️ Warning: Could not load enhanced data: {str(e)}"
                )

            # Load betting odds features
            try:
                betting_features_df = client.get_derived_betting_features()
                data_load_status.append(
                    f"✅ Betting features: {len(betting_features_df)} records"
                )
            except (AttributeError, Exception) as e:
                data_load_status.append(
                    f"⚠️ Betting features unavailable: {str(e)} (continuing with defaults)"
                )
                betting_features_df = pd.DataFrame()

            # Load raw players data for penalty/set-piece features
            try:
                raw_players_df = client.get_raw_players_bootstrap()
                required_cols = [
                    "penalties_order",
                    "corners_and_indirect_freekicks_order",
                    "direct_freekicks_order",
                ]
                missing_cols = [
                    col for col in required_cols if col not in raw_players_df.columns
                ]
                if missing_cols:
                    data_load_status.append(
                        f"⚠️ Missing penalty columns: {missing_cols}"
                    )
                    raw_players_df = pd.DataFrame()
                else:
                    data_load_status.append(
                        f"✅ Penalty/set-piece data: {len(raw_players_df)} players"
                    )
            except Exception as e:
                data_load_status.append(
                    f"⚠️ Penalty data unavailable: {str(e)} (continuing without)"
                )
                raw_players_df = pd.DataFrame()

            load_summary = mo.vstack(
                [
                    mo.md(
                        f"### ✅ Loaded {len(historical_data)} Gameweeks (GW{start_gw}-{end_gw})"
                    ),
                    mo.md("\n".join(data_load_status)),
                    mo.md(f"**Total records:** {len(historical_df):,}"),
                    mo.md(
                        f"**Unique players:** {historical_df['player_id'].nunique():,}"
                    ),
                    mo.md("---"),
                ]
            )
        else:
            historical_df = pd.DataFrame()
            load_summary = mo.md("❌ **No data loaded**")
    except Exception as e:
        historical_df = pd.DataFrame()
        load_summary = mo.md(f"❌ **Data loading error:** {str(e)}")

    load_summary
    return (
        betting_features_df,
        fixture_difficulty_df,
        fixtures_df,
        historical_df,
        ownership_trends_df,
        raw_players_df,
        teams_df,
        value_analysis_df,
    )


@app.cell
def _(historical_df, mo):
    # Interactive table to explore loaded data
    if not historical_df.empty:
        # Select key columns for display
        _hist_display_cols = [
            "gameweek",
            "player_id",
            "web_name",
            "position",
            "team",
            "total_points",
            "minutes",
            "goals_scored",
            "assists",
            "clean_sheets",
            "expected_goals",
            "expected_assists",
            "bps",
            "ict_index",
            "value",
        ]
        _hist_available_cols = [
            col for col in _hist_display_cols if col in historical_df.columns
        ]

        historical_data_table = mo.ui.table(
            historical_df[_hist_available_cols],
            selection=None,
            pagination=True,
            page_size=20,
            label="Historical Gameweek Performance Data",
        )
    else:
        historical_data_table = mo.md("_No data loaded yet_")

    historical_data_table
    return


@app.cell
def _(mo):
    mo.md(r"""## 2️⃣ Team Strength Ratings - Opponent Quality Context""")
    return


@app.cell
def _(get_team_strength_ratings, mo, teams_df):
    # ✅ USE PRODUCTION CODE: Dynamic team strength using TeamAnalyticsService
    # Multi-factor: position (25%), quality (35%), reputation (20%), form (20%)
    # Using GW1 baseline for training consistency - actual predictions use target_gameweek
    team_strength = get_team_strength_ratings(target_gameweek=1, teams_df=teams_df)

    # Display team strength summary
    if not teams_df.empty:
        strength_summary = []
        for team_name, strength in sorted(
            team_strength.items(), key=lambda x: x[1], reverse=True
        )[:20]:
            strength_summary.append(f"- **{team_name}**: {strength}")

        team_strength_display = mo.vstack(
            [
                mo.md("### ✅ Team Strength Ratings Calculated"),
                mo.md(
                    f"**Total teams:** {len(team_strength)} | **Range:** [0.65, 1.30]"
                ),
                mo.md("**Top 5 Teams:**"),
                mo.md("\n".join(strength_summary[:5])),
                mo.md("---"),
            ]
        )
    else:
        team_strength_display = mo.md("⚠️ **Load teams data first**")

    team_strength_display
    return (team_strength,)


@app.cell
def _(mo):
    mo.md(r"""## 3️⃣ Feature Engineering - Leak-Free Historical + Fixture Features""")
    return


@app.cell
def _(
    FPLFeatureEngineer,
    betting_features_df,
    fixture_difficulty_df,
    fixtures_df,
    historical_df,
    mo,
    ownership_trends_df,
    pd,
    raw_players_df,
    team_strength,
    teams_df,
    value_analysis_df,
):
    # ===================================================================
    # ✅ USE PRODUCTION CODE: FPLFeatureEngineer from ml_feature_engineering.py
    # ===================================================================
    # This ensures notebook and gameweek_manager.py use IDENTICAL feature engineering
    # Any changes to feature logic propagate automatically to production!
    #
    # NEW: FPLFeatureEngineer class (scroll down to see usage)
    # ===================================================================
    # ===================================================================
    # ✅ PRODUCTION INTEGRATION: Feature Engineering
    # ===================================================================
    # This section uses FPLFeatureEngineer from ml_feature_engineering.py
    # Benefits:
    #   - Single source of truth for feature logic
    #   - Changes here automatically propagate to gameweek_manager.py
    #   - No code duplication or drift between notebook and production
    #   - Easier to test and maintain
    # ===================================================================

    if not historical_df.empty:
        # Initialize production feature engineer with ALL data sources for 117 features
        # This matches what TPOT models were trained with
        feature_engineer = FPLFeatureEngineer(
            fixtures_df=fixtures_df if not fixtures_df.empty else None,
            teams_df=teams_df if not teams_df.empty else None,
            team_strength=team_strength if team_strength else None,
            ownership_trends_df=ownership_trends_df
            if not ownership_trends_df.empty
            else None,
            value_analysis_df=value_analysis_df
            if not value_analysis_df.empty
            else None,
            fixture_difficulty_df=fixture_difficulty_df
            if not fixture_difficulty_df.empty
            else None,
            raw_players_df=raw_players_df if not raw_players_df.empty else None,
            betting_features_df=betting_features_df
            if not betting_features_df.empty
            else None,
        )

        # Transform historical data to features (same as production!)
        features_df = feature_engineer.fit_transform(
            historical_df, historical_df["total_points"]
        )

        # Add back metadata for notebook analysis
        # (FPLFeatureEngineer sorts by player_id, gameweek internally)
        historical_df_sorted = historical_df.sort_values(
            ["player_id", "gameweek"]
        ).reset_index(drop=True)
        features_df["player_id"] = historical_df_sorted["player_id"].values
        features_df["gameweek"] = historical_df_sorted["gameweek"].values
        features_df["total_points"] = historical_df_sorted["total_points"].values

        # Add position and other metadata for analysis sections
        if "position" in historical_df_sorted.columns:
            features_df["position"] = historical_df_sorted["position"].values
        if "web_name" in historical_df_sorted.columns:
            features_df["web_name"] = historical_df_sorted["web_name"].values
        if "team" in historical_df_sorted.columns:
            features_df["team"] = historical_df_sorted["team"].values
        if "value" in historical_df_sorted.columns:
            # FPL API returns prices in tenths (e.g., 90 = £9.0m)
            features_df["price"] = historical_df_sorted["value"].values / 10.0

        # Get feature names from production code (to be used in CV/training)
        production_feature_cols = list(feature_engineer.get_feature_names_out())

        # Show feature engineering results
        feature_summary = mo.vstack(
            [
                mo.md(
                    "### ✅ Features Engineered (5-Gameweek Rolling Windows + Fixtures)"
                ),
                mo.md("**Static Features:**"),
                mo.md("- `price` - Player price (£M)"),
                mo.md(
                    "- `position_encoded` - Position encoding (GKP=0, DEF=1, MID=2, FWD=3)"
                ),
                mo.md("- `games_played` - Games played this season"),
                mo.md(""),
                mo.md("**Cumulative Season Statistics (up to GW N-1):**"),
                mo.md(
                    "- `cumulative_minutes`, `cumulative_goals`, `cumulative_assists`, `cumulative_points`"
                ),
                mo.md(
                    "- `cumulative_bonus`, `cumulative_clean_sheets`, `cumulative_xg`, `cumulative_xa`, `cumulative_bps`"
                ),
                mo.md(""),
                mo.md("**Cumulative Per-90 Rates (season-long efficiency):**"),
                mo.md("- `goals_per_90`, `assists_per_90`, `points_per_90`"),
                mo.md("- `xg_per_90`, `xa_per_90`, `bps_per_90`, `clean_sheet_rate`"),
                mo.md(""),
                mo.md("**Rolling 5GW Form Features (GW N-6 to N-1):**"),
                mo.md(
                    "- `rolling_5gw_points`, `rolling_5gw_minutes`, `rolling_5gw_goals`, `rolling_5gw_assists`"
                ),
                mo.md(
                    "- `rolling_5gw_xg`, `rolling_5gw_xa`, `rolling_5gw_bps`, `rolling_5gw_bonus`"
                ),
                mo.md(
                    "- `rolling_5gw_ict_index`, `rolling_5gw_influence`, `rolling_5gw_creativity`, `rolling_5gw_threat`"
                ),
                mo.md(""),
                mo.md("**Rolling 5GW Per-90 Rates (recent efficiency):**"),
                mo.md(
                    "- `rolling_5gw_goals_per_90`, `rolling_5gw_assists_per_90`, `rolling_5gw_points_per_90`"
                ),
                mo.md(""),
                mo.md("**Consistency & Volatility:**"),
                mo.md(
                    "- `rolling_5gw_points_std`, `rolling_5gw_minutes_std` - Volatility measures"
                ),
                mo.md("- `minutes_played_rate` - % of available minutes in last 5GW"),
                mo.md(
                    "- `form_trend` - Linear trend coefficient (improving/declining)"
                ),
                mo.md(""),
                mo.md("**Team Context Features (leak-free):**"),
                mo.md("- `team_encoded` - Team identity (categorical)"),
                mo.md(
                    "- `team_rolling_5gw_goals_scored/conceded` - Team attacking/defensive form"
                ),
                mo.md("- `team_rolling_5gw_xg/xgc` - Team underlying quality"),
                mo.md("- `team_rolling_5gw_clean_sheets` - Defensive consistency"),
                mo.md(
                    "- `team_cumulative_goals_scored/conceded` - Season-long team performance"
                ),
                mo.md(
                    "- `team_rolling_5gw_goal_diff/xg_diff` - Team form differentials"
                ),
                mo.md(""),
                mo.md("**Fixture-Specific Features (NEW!):**"),
                mo.md("- `is_home` - Home (1), Away (0), Neutral (0.5)"),
                mo.md("- `opponent_strength` - Opponent team strength [0.65, 1.30]"),
                mo.md(
                    "- `fixture_difficulty` - Higher = easier (inverse of opponent strength with home advantage)"
                ),
                mo.md(
                    "- `opponent_rolling_5gw_goals_conceded` - Opponent's defensive weakness"
                ),
                mo.md(
                    "- `opponent_rolling_5gw_clean_sheets` - Opponent's defensive strength"
                ),
                mo.md(
                    "- `opponent_rolling_5gw_xgc` - Opponent's expected goals conceded"
                ),
                mo.md(""),
                mo.md("**Enhanced Features (Issue #37):**"),
                mo.md(
                    "- `ownership_tier`, `transfer_momentum`, `bandwagon_score` - Ownership trends (7 features)"
                ),
                mo.md(
                    "- `points_per_pound`, `value_vs_position`, `predicted_price_change` - Value analysis (5 features)"
                ),
                mo.md(
                    "- `congestion_difficulty`, `form_adjusted_difficulty` - Enhanced fixture difficulty (3 features)"
                ),
                mo.md(""),
                mo.md("**Penalty & Set-Piece Taker Features:**"),
                mo.md(
                    "- `is_primary_penalty_taker`, `is_penalty_taker` - Penalty taker indicators"
                ),
                mo.md(
                    "- `is_corner_taker`, `is_fk_taker` - Set-piece taker indicators"
                ),
                mo.md(""),
                mo.md("**Betting Odds Features (Issue #38):**"),
                mo.md(
                    "- `team_win_probability`, `opponent_win_probability`, `draw_probability` - Match outcome probabilities"
                ),
                mo.md(
                    "- `implied_clean_sheet_probability`, `implied_total_goals` - Implied statistics"
                ),
                mo.md(
                    "- `market_consensus_strength`, `odds_movement_team` - Market confidence indicators"
                ),
                mo.md(
                    "- `asian_handicap_line`, `expected_goal_difference` - Handicap and expectation features"
                ),
                mo.md(""),
                mo.md(
                    "💡 **Team features are safe with player-based GroupKFold:** We test the model's ability to predict NEW players on KNOWN teams, not future outcomes."
                ),
                mo.md(""),
                mo.md(
                    f"**Total features created:** {len(production_feature_cols)} (117 when all data sources available)"
                ),
                mo.md(
                    "**Features using 5GW windows:** All rolling features use 5-gameweek lookback"
                ),
                mo.md(
                    "**Fixture features:** Match-specific context known before gameweek starts"
                ),
                mo.md("---"),
            ]
        )
    else:
        features_df = pd.DataFrame()
        production_feature_cols = []
        feature_summary = mo.md("⚠️ **Load data first**")

    feature_summary
    return features_df, production_feature_cols


@app.cell
def _(features_df, mo):
    # Interactive table to explore feature-engineered data
    if not features_df.empty:
        # Select key columns for display - mix of original and engineered features
        _feat_display_cols = [
            "gameweek",
            "player_id",
            "web_name",
            "position",
            "team",
            "total_points",  # Target variable
            "price",
            "games_played",
            # Cumulative stats
            "cumulative_points",
            "cumulative_minutes",
            "cumulative_goals",
            "cumulative_assists",
            # Per-90 rates
            "goals_per_90",
            "assists_per_90",
            "points_per_90",
            # Rolling 5GW features
            "rolling_5gw_points",
            "rolling_5gw_minutes",
            "rolling_5gw_goals_per_90",
            "rolling_5gw_assists_per_90",
            # Consistency
            "rolling_5gw_points_std",
            "form_trend",
            "minutes_played_rate",
        ]
        _feat_available_cols = [
            col for col in _feat_display_cols if col in features_df.columns
        ]

        features_table = mo.ui.table(
            features_df[_feat_available_cols],
            selection=None,
            pagination=True,
            page_size=20,
            label="Feature-Engineered Data (Leak-Free)",
        )
    else:
        features_table = mo.md("_No features engineered yet_")

    features_table
    return


@app.cell
def _(mo):
    mo.md(r"""## 4️⃣ Analysis Data Preparation - Temporal Validation Setup""")
    return


@app.cell
def _(
    end_gw,
    features_df,
    mo,
    pd,
    production_feature_cols,
):
    # Prepare data for analysis
    # Strategy: Use GW6-9 for temporal validation (GW6+ required for rolling features)
    if not features_df.empty:
        # Use GW6-9 for temporal CV analysis (GW6+ required for rolling features)
        # Model was trained on GW1-9, so we analyze on GW6-9 for proper temporal validation
        cv_data = features_df[
            (features_df["gameweek"] >= 6) & (features_df["gameweek"] <= end_gw)
        ].copy()

        # ✅ USE PRODUCTION FEATURE LIST from FPLFeatureEngineer
        # This ensures notebook uses same features as gameweek_manager.py
        feature_cols = production_feature_cols

        # Validate all feature columns exist in the data
        missing_features = [col for col in feature_cols if col not in cv_data.columns]
        if missing_features:
            raise ValueError(
                f"Missing feature columns: {missing_features}. "
                f"Available columns: {list(cv_data.columns)}"
            )

        # Prepare X and y for cross-validation
        X_cv = cv_data[feature_cols].fillna(0)
        y_cv = cv_data["total_points"]

        # Calculate gameweek coverage
        _cv_gws = sorted(cv_data["gameweek"].unique())
        _gw_range = (
            f"GW{min(_cv_gws)}-{max(_cv_gws)}"
            if len(_cv_gws) > 1
            else f"GW{_cv_gws[0]}"
        )

        cv_summary = mo.vstack(
            [
                mo.md("### ✅ Analysis Data Prepared"),
                mo.md(f"**Data Range:** {_gw_range} ({len(cv_data):,} total records)"),
                mo.md(
                    f"**Players:** {cv_data['player_id'].nunique():,} unique players"
                ),
                mo.md(
                    f"**Features:** {len(feature_cols)} columns (117-feature TPOT model)"
                ),
                mo.md(""),
                mo.md("**✅ No Data Leakage:**"),
                mo.md("- All features use `.shift(1)` to exclude current gameweek"),
                mo.md("- Cumulative stats: up to GW N-1 only"),
                mo.md("- Rolling features: GW N-6 to N-1 window"),
                mo.md("- Temporal validation will test on future gameweeks"),
                mo.md(""),
                mo.md(
                    f"**Total: {len(feature_cols)} features** (65 base + 15 enhanced + 4 penalty + 15 betting + 18 new)"
                ),
                mo.md("---"),
            ]
        )
    else:
        X_cv = pd.DataFrame()
        y_cv = pd.Series()
        cv_data = pd.DataFrame()
        feature_cols = []
        cv_summary = mo.md("⚠️ **Engineer features first**")

    cv_summary
    return X_cv, cv_data, feature_cols, y_cv


@app.cell
def _(X_cv, cv_data, mo, y_cv):
    # Display sample training data for inspection
    if not cv_data.empty:
        sample_display = mo.vstack(
            [
                mo.md("### 🔍 Training Data Sample (First 10 Rows)"),
                mo.md(f"**Shape:** {X_cv.shape[0]} samples × {X_cv.shape[1]} features"),
                mo.md(""),
                mo.md("**Feature Values + Target:**"),
                mo.ui.table(
                    cv_data[
                        [
                            "player_id",
                            "gameweek",
                            "total_points",
                            "price",
                            "position_encoded",
                            "rolling_5gw_points",
                            "rolling_5gw_minutes",
                            "fixture_difficulty",
                        ]
                    ].head(10)
                ),
                mo.md(""),
                mo.md("**Full Feature Matrix (X_cv) - First 5 rows:**"),
                mo.ui.table(X_cv.head(5)),
                mo.md(""),
                mo.md("**Target Variable (y_cv) - First 10 values:**"),
                mo.md(f"`{y_cv.head(10).tolist()}`"),
            ]
        )
    else:
        sample_display = mo.md("")

    sample_display
    return


@app.cell
def _(mo):
    mo.md(r"""## 5️⃣ Load Pre-Trained TPOT Model""")
    return


@app.cell
def _(
    X_cv,
    cross_val_score,
    cv_data,
    feature_cols,
    mo,
    np,
    y_cv,
):
    # Load pre-trained TPOT model and perform temporal cross-validation analysis
    import joblib
    from pathlib import Path

    tpot_model = None
    cv_results = {}
    model_info = {}

    if not X_cv.empty and feature_cols:
        # Find most recent TPOT model
        _tpot_models_dir = Path("models/tpot")
        if _tpot_models_dir.exists():
            _tpot_joblib_files = sorted(
                _tpot_models_dir.glob("tpot_pipeline_*.joblib"), reverse=True
            )

            if _tpot_joblib_files:
                _tpot_model_path = _tpot_joblib_files[0]
                print(f"📦 Loading TPOT pipeline from: {_tpot_model_path.name}")

                # Load pre-trained TPOT pipeline
                tpot_model = joblib.load(_tpot_model_path)
                model_info["path"] = str(_tpot_model_path)
                model_info["name"] = _tpot_model_path.name

                # Inspect pipeline structure
                print("\n🔍 Pipeline structure:")
                for i, (name, step) in enumerate(tpot_model.steps, 1):
                    print(f"   {i}. {name}: {type(step).__name__}")

                # Validate feature count matches (TPOT models expect 117 features)
                try:
                    first_step = tpot_model.steps[0][1]
                    if hasattr(first_step, "n_features_in_"):
                        expected_features = first_step.n_features_in_
                        actual_features = len(feature_cols)
                        if expected_features != actual_features:
                            print(
                                f"\n⚠️ Feature count mismatch: Model expects {expected_features}, got {actual_features}"
                            )
                            print(
                                "   Model was likely trained with different feature set."
                            )
                            print(
                                "   Ensure all enhanced data sources are loaded (ownership, value, fixture_difficulty, betting, penalties)."
                            )
                        else:
                            print(
                                f"\n✅ Feature count validated: {actual_features} features match model expectations"
                            )
                except Exception as e:
                    print(
                        f"\n⚠️ Could not validate feature count: {e} (continuing anyway)"
                    )

                # Create temporal CV splits for analysis
                _cv_gws = sorted(cv_data["gameweek"].unique())
                print(f"\n📊 Available gameweeks for analysis: {_cv_gws}")

                if len(_cv_gws) >= 2:
                    # Create temporal splits: train on all GWs up to N-1, test on GW N
                    cv_data_reset = cv_data.reset_index(drop=True)
                    _cv_splits = []

                    for i in range(len(_cv_gws) - 1):
                        train_gws = _cv_gws[: i + 1]  # GW6 to GW(6+i)
                        test_gw = _cv_gws[i + 1]  # GW(6+i+1)

                        train_mask = cv_data_reset["gameweek"].isin(train_gws)
                        test_mask = cv_data_reset["gameweek"] == test_gw

                        train_idx = np.where(train_mask)[0]
                        test_idx = np.where(test_mask)[0]

                        _cv_splits.append((train_idx, test_idx))
                        print(
                            f"  Fold {i + 1}: Train GW{min(train_gws)}-{max(train_gws)} ({len(train_idx)} samples) → Test GW{test_gw} ({len(test_idx)} samples)"
                        )

                    print(
                        f"\n✅ Temporal CV: {len(_cv_splits)} folds created for analysis"
                    )

                    # Perform temporal cross-validation
                    print("\n🔄 Running temporal cross-validation analysis...")
                    _tpot_mae_scores = -cross_val_score(
                        tpot_model,
                        X_cv,
                        y_cv,
                        cv=_cv_splits,
                        scoring="neg_mean_absolute_error",
                        n_jobs=-1,
                    )
                    _tpot_rmse_scores = np.sqrt(
                        -cross_val_score(
                            tpot_model,
                            X_cv,
                            y_cv,
                            cv=_cv_splits,
                            scoring="neg_mean_squared_error",
                            n_jobs=-1,
                        )
                    )
                    _tpot_r2_scores = cross_val_score(
                        tpot_model,
                        X_cv,
                        y_cv,
                        cv=_cv_splits,
                        scoring="r2",
                        n_jobs=-1,
                    )

                    cv_results["tpot"] = {
                        "MAE_mean": _tpot_mae_scores.mean(),
                        "MAE_std": _tpot_mae_scores.std(),
                        "RMSE_mean": _tpot_rmse_scores.mean(),
                        "RMSE_std": _tpot_rmse_scores.std(),
                        "R²_mean": _tpot_r2_scores.mean(),
                        "R²_std": _tpot_r2_scores.std(),
                        "fold_scores": _tpot_mae_scores,
                        "fold_rmse": _tpot_rmse_scores,
                        "fold_r2": _tpot_r2_scores,
                    }

                    print(
                        f"\n✅ Analysis complete: MAE = {_tpot_mae_scores.mean():.3f} ± {_tpot_mae_scores.std():.3f}"
                    )
                    print(
                        f"   RMSE = {_tpot_rmse_scores.mean():.3f} ± {_tpot_rmse_scores.std():.3f}"
                    )
                    print(
                        f"   R² = {_tpot_r2_scores.mean():.3f} ± {_tpot_r2_scores.std():.3f}"
                    )

                    # Create summary display
                    model_summary = mo.vstack(
                        [
                            mo.md("### ✅ TPOT Model Loaded & Analyzed"),
                            mo.md(f"**Model:** `{model_info['name']}`"),
                            mo.md(""),
                            mo.md("**Temporal Cross-Validation Results:**"),
                            mo.md(
                                f"- **MAE:** {_tpot_mae_scores.mean():.3f} ± {_tpot_mae_scores.std():.3f} points"
                            ),
                            mo.md(
                                f"- **RMSE:** {_tpot_rmse_scores.mean():.3f} ± {_tpot_rmse_scores.std():.3f} points"
                            ),
                            mo.md(
                                f"- **R²:** {_tpot_r2_scores.mean():.3f} ± {_tpot_r2_scores.std():.3f}"
                            ),
                            mo.md(""),
                            mo.md(f"**Temporal Folds:** {len(_cv_splits)} folds"),
                            mo.md(
                                f"**Fold MAE scores:** {', '.join([f'{s:.2f}' for s in _tpot_mae_scores])}"
                            ),
                            mo.md("---"),
                        ]
                    )
                else:
                    model_summary = mo.md(
                        "⚠️ **Need at least 2 gameweeks for temporal CV analysis**"
                    )
            else:
                model_summary = mo.md(
                    "⚠️ **No TPOT models found in models/tpot/.**\n\nRun `uv run python scripts/tpot_pipeline_optimizer.py` to generate models."
                )
        else:
            model_summary = mo.md(
                "⚠️ **models/tpot/ directory not found.**\n\nRun `uv run python scripts/tpot_pipeline_optimizer.py` to generate models."
            )
    else:
        model_summary = mo.md("⚠️ **Prepare analysis data first**")

    model_summary
    return cv_results, model_info, tpot_model


@app.cell
def _(mo):
    mo.md(r"""## 6️⃣ Cross-Validation Results Visualization""")
    return


@app.cell
def _(cv_results, mo, pd, px):
    # Visualize cross-validation fold scores
    if cv_results:
        # Create fold scores DataFrame for plotting
        _fold_data = []
        for _model_name, _cv_metrics in cv_results.items():
            for _fold_idx, _mae in enumerate(_cv_metrics["fold_scores"], 1):
                _fold_data.append(
                    {
                        "Model": _model_name.upper(),
                        "Fold": _fold_idx,
                        "MAE": _mae,
                    }
                )

        _fold_df = pd.DataFrame(_fold_data)

        # Box plot of fold scores
        _fig_box = px.box(
            _fold_df,
            x="Model",
            y="MAE",
            points="all",
            title="Cross-Validation MAE Scores by Fold",
            labels={"MAE": "Mean Absolute Error (points)"},
        )

        # Bar chart comparing mean scores
        _mean_df = pd.DataFrame(
            [
                {
                    "Model": _m.upper(),
                    "MAE": _cv["MAE_mean"],
                    "RMSE": _cv["RMSE_mean"],
                    "R²": _cv["R²_mean"],
                }
                for _m, _cv in cv_results.items()
            ]
        )

        _fig_comparison = px.bar(
            _mean_df,
            x="Model",
            y=["MAE", "RMSE"],
            title="Model Comparison (Mean CV Scores)",
            barmode="group",
            labels={"value": "Score", "variable": "Metric"},
        )

        eval_display = mo.vstack(
            [
                mo.md("### 📊 Cross-Validation Results"),
                mo.ui.plotly(_fig_box),
                mo.md("---"),
                mo.ui.plotly(_fig_comparison),
                mo.md("---"),
                mo.md("### 📋 Detailed Metrics"),
                mo.ui.table(_mean_df.round(3), page_size=10),
            ]
        )
    else:
        eval_display = mo.md("⚠️ **Train models first**")

    eval_display
    return


@app.cell
def _(mo):
    mo.md(r"""## 7️⃣ Feature Importance - What Drives xP?""")
    return


@app.cell
def _(feature_cols, mo, pd, px, tpot_model):
    # Feature importance analysis for TPOT model
    # Note: TPOT models use complex pipelines with feature selection (RFE), so we need to
    # trace through the pipeline to find which features were actually used
    if tpot_model and feature_cols:
        importance_data = []

        # Try to extract feature importance from TPOT pipeline
        # TPOT pipelines typically end with an AdaBoostRegressor or similar
        final_estimator = tpot_model.steps[-1][1]

        if hasattr(final_estimator, "feature_importances_"):
            # TPOT pipeline includes RFE which reduces features
            # Need to trace through to find which features were selected
            selected_features = feature_cols.copy()

            # Step 1: Find RFE step and get selected features
            for step_name, step_transformer in tpot_model.steps:
                if hasattr(step_transformer, "support_"):
                    # RFE has support_ attribute indicating selected features
                    selected_mask = step_transformer.support_
                    selected_features = [
                        f
                        for f, selected in zip(selected_features, selected_mask)
                        if selected
                    ]
                    print(
                        f"   🔍 {step_name} selected {len(selected_features)}/{len(selected_mask)} features"
                    )
                    break

            # Step 2: Handle FeatureUnion steps (they may add features)
            # For now, assume FeatureUnion doesn't change feature order or count significantly
            # Final estimator importances should match features after all transformations

            n_importances = len(final_estimator.feature_importances_)
            n_features = len(selected_features)

            if n_importances == n_features:
                # Perfect match - use selected features
                feature_names = selected_features
                print(
                    f"   ✅ Feature importance extraction: {len(feature_names)} features"
                )
            elif n_importances < n_features:
                # Importances fewer than selected - pipeline further reduced features
                # This could happen if FeatureUnion or other steps reduced features
                # Use first N selected features (approximation)
                feature_names = selected_features[:n_importances]
                print(
                    f"   ⚠️ Pipeline further reduced features: {n_importances} vs {n_features} after RFE"
                )
                print(
                    f"   ℹ️  Showing importance for first {n_importances} selected features"
                )
            else:
                # Importances more than selected - FeatureUnion may have expanded features
                # Pad with placeholder names
                feature_names = selected_features + [
                    f"transformed_feature_{i}"
                    for i in range(len(selected_features), n_importances)
                ]
                print(
                    f"   ⚠️ Pipeline expanded features: {n_importances} vs {n_features} after RFE"
                )
                print("   ℹ️  Some features may be transformed/combined versions")

            # Create DataFrame with matching lengths
            try:
                tpot_importance = pd.DataFrame(
                    {
                        "feature": feature_names[:n_importances],
                        "importance": final_estimator.feature_importances_,
                        "model": "TPOT",
                    }
                ).nlargest(20, "importance")
                importance_data.append(tpot_importance)
            except ValueError as e:
                print(f"   ❌ Error creating importance DataFrame: {e}")
                print(
                    f"      Feature names length: {len(feature_names[:n_importances])}"
                )
                print(f"      Importances length: {n_importances}")

        if importance_data:
            all_importance = pd.concat(importance_data, ignore_index=True)

            # Plot feature importance
            fig_importance = px.bar(
                all_importance,
                x="importance",
                y="feature",
                color="model",
                orientation="h",
                title="Top 15 Features by Model",
                labels={"importance": "Importance", "feature": "Feature"},
                barmode="group",
            )

            importance_display = mo.vstack(
                [
                    mo.md("### 🔍 Feature Importance Analysis"),
                    mo.ui.plotly(fig_importance),
                    mo.md("---"),
                    mo.md("### 📋 Feature Importance Table"),
                    mo.ui.table(all_importance.round(4), page_size=15),
                ]
            )
        else:
            importance_display = mo.md("⚠️ **No feature importance available**")
    else:
        all_importance = pd.DataFrame()
        importance_display = mo.md("⚠️ **Train models first**")

    importance_display
    return


@app.cell
def _(mo):
    mo.md(r"""## 8️⃣ Position-Specific Error Analysis - Where Are We Losing Points?""")
    return


@app.cell
def _(X_cv, cv_data, cv_results, feature_cols, mo, np, px, tpot_model):
    # Position-specific error breakdown to identify which positions need improvement
    if tpot_model and not cv_data.empty and feature_cols and cv_results:
        # Use TPOT model for predictions
        _best_model_name = "tpot"
        _best_model = tpot_model

        # Generate predictions on all CV data
        _y_pred = _best_model.predict(X_cv)

        # Calculate errors by position
        _position_errors = cv_data.copy()
        _position_errors["predicted_points"] = _y_pred
        _position_errors["actual_points"] = cv_data["total_points"]
        _position_errors["error"] = abs(
            _position_errors["predicted_points"] - _position_errors["actual_points"]
        )
        _position_errors["signed_error"] = (
            _position_errors["predicted_points"] - _position_errors["actual_points"]
        )

        # Need position data - check if available
        if "position" in _position_errors.columns:
            # Aggregate by position
            _position_stats = (
                _position_errors.groupby("position")
                .agg(
                    {
                        "error": ["mean", "std", "count"],
                        "signed_error": "mean",
                        "actual_points": "mean",
                        "predicted_points": "mean",
                    }
                )
                .reset_index()
            )

            # Flatten column names
            _position_stats.columns = [
                "position",
                "MAE",
                "MAE_std",
                "count",
                "bias",
                "actual_avg",
                "predicted_avg",
            ]

            # Calculate RMSE by position
            _position_rmse = (
                _position_errors.groupby("position")
                .apply(
                    lambda x: np.sqrt(
                        ((x["predicted_points"] - x["actual_points"]) ** 2).mean()
                    )
                )
                .reset_index(name="RMSE")
            )

            _position_stats = _position_stats.merge(_position_rmse, on="position")

            # Sort by MAE descending (worst positions first)
            _position_stats = _position_stats.sort_values("MAE", ascending=False)

            # Visualize position-specific errors
            _fig_position_mae = px.bar(
                _position_stats,
                x="position",
                y="MAE",
                error_y="MAE_std",
                title=f"Position-Specific MAE ({_best_model_name.upper()} model)",
                labels={
                    "MAE": "Mean Absolute Error (points)",
                    "position": "Position",
                },
                text="MAE",
            )
            _fig_position_mae.update_traces(
                texttemplate="%{text:.2f}", textposition="outside"
            )

            # Bias analysis (systematic over/under-prediction)
            _fig_position_bias = px.bar(
                _position_stats,
                x="position",
                y="bias",
                title=f"Position-Specific Prediction Bias ({_best_model_name.upper()} model)",
                labels={
                    "bias": "Mean Signed Error (+ = over-predict, - = under-predict)",
                    "position": "Position",
                },
                text="bias",
                color="bias",
                color_continuous_scale=["red", "white", "blue"],
                color_continuous_midpoint=0,
            )
            _fig_position_bias.update_traces(
                texttemplate="%{text:.2f}", textposition="outside"
            )

            position_analysis_display = mo.vstack(
                [
                    mo.md("### 🎯 Position-Specific Error Breakdown"),
                    mo.md(
                        f"**Model:** {_best_model_name.upper()} (MAE: {cv_results[_best_model_name]['MAE_mean']:.2f})"
                    ),
                    mo.md(""),
                    mo.ui.plotly(_fig_position_mae),
                    mo.md("---"),
                    mo.ui.plotly(_fig_position_bias),
                    mo.md("---"),
                    mo.md("### 📊 Position-Specific Metrics"),
                    mo.ui.table(_position_stats.round(2), page_size=10),
                    mo.md(""),
                    mo.md("**Insights:**"),
                    mo.md(
                        f"- **Worst Position:** {_position_stats.iloc[0]['position']} (MAE: {_position_stats.iloc[0]['MAE']:.2f})"
                    ),
                    mo.md(
                        f"- **Best Position:** {_position_stats.iloc[-1]['position']} (MAE: {_position_stats.iloc[-1]['MAE']:.2f})"
                    ),
                    mo.md(
                        f"- **Most Over-Predicted:** {_position_stats.loc[_position_stats['bias'].idxmax(), 'position']} (+{_position_stats['bias'].max():.2f} pts)"
                    ),
                    mo.md(
                        f"- **Most Under-Predicted:** {_position_stats.loc[_position_stats['bias'].idxmin(), 'position']} ({_position_stats['bias'].min():.2f} pts)"
                    ),
                ]
            )
        else:
            position_analysis_display = mo.md(
                "❌ **Position data not available in CV data**"
            )
    else:
        position_analysis_display = mo.md("⚠️ **Load TPOT model first**")

    position_analysis_display
    return


@app.cell
def _(mo):
    mo.md(r"""## 9️⃣ Prediction Bias Detection - Systematic Over/Under-Prediction""")
    return


@app.cell
def _(X_cv, cv_data, cv_results, feature_cols, mo, pd, px, tpot_model):
    # Detect systematic biases in predictions (price bands, team strength, etc.)
    if tpot_model and not cv_data.empty and feature_cols and cv_results:
        _best_model_name_bias = "tpot"
        _best_model_bias = tpot_model

        # Generate predictions
        _y_pred_bias = _best_model_bias.predict(X_cv)

        _bias_data = cv_data.copy()
        _bias_data["predicted_points"] = _y_pred_bias
        _bias_data["actual_points"] = cv_data["total_points"]

        # Scatter plot: predicted vs actual (overall)
        _fig_scatter = px.scatter(
            _bias_data.sample(min(1000, len(_bias_data))),  # Sample for performance
            x="actual_points",
            y="predicted_points",
            color="position" if "position" in _bias_data.columns else None,
            title=f"Predicted vs Actual Points ({_best_model_name_bias.upper()} model)",
            labels={
                "actual_points": "Actual Points",
                "predicted_points": "Predicted Points",
            },
            trendline="ols",
            opacity=0.6,
        )
        # Add perfect prediction line
        _fig_scatter.add_scatter(
            x=[0, _bias_data["actual_points"].max()],
            y=[0, _bias_data["actual_points"].max()],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="red", dash="dash"),
        )

        # Price band analysis (if price available)
        if "price" in _bias_data.columns:
            # Create price bands
            _bias_data["price_band"] = pd.cut(
                _bias_data["price"],
                bins=[0, 5, 7, 9, 15],
                labels=[
                    "Budget (<£5M)",
                    "Mid (£5-7M)",
                    "Premium (£7-9M)",
                    "Elite (£9M+)",
                ],
            )

            _price_band_errors = (
                _bias_data.groupby("price_band", observed=True)
                .agg(
                    {
                        "predicted_points": "mean",
                        "actual_points": "mean",
                    }
                )
                .reset_index()
            )
            _price_band_errors["error"] = abs(
                _price_band_errors["predicted_points"]
                - _price_band_errors["actual_points"]
            )
            _price_band_errors["bias"] = (
                _price_band_errors["predicted_points"]
                - _price_band_errors["actual_points"]
            )

            _fig_price_bands = px.bar(
                _price_band_errors,
                x="price_band",
                y=["actual_points", "predicted_points"],
                title="Average Points by Price Band (Predicted vs Actual)",
                labels={"value": "Average Points", "price_band": "Price Band"},
                barmode="group",
            )

            _price_band_display = mo.vstack(
                [
                    mo.md("### 💰 Price Band Analysis"),
                    mo.ui.plotly(_fig_price_bands),
                    mo.md("---"),
                    mo.ui.table(_price_band_errors.round(2), page_size=10),
                ]
            )
        else:
            _price_band_display = mo.md("_Price data not available_")

        bias_display = mo.vstack(
            [
                mo.md("### 🔍 Prediction Bias Analysis"),
                mo.md(f"**Model:** {_best_model_name_bias.upper()}"),
                mo.md(""),
                mo.md("#### Overall Predicted vs Actual"),
                mo.ui.plotly(_fig_scatter),
                mo.md(""),
                mo.md("**Interpretation:**"),
                mo.md("- Points above red line = over-prediction"),
                mo.md("- Points below red line = under-prediction"),
                mo.md("- Clustered around line = good calibration"),
                mo.md("---"),
                _price_band_display,
            ]
        )
    else:
        bias_display = mo.md("⚠️ **Train models first**")

    bias_display
    return


@app.cell
def _(mo):
    mo.md(r"""## 🔟 Rule-Based vs ML Comparison - Is ML Actually Better?""")
    return


@app.cell
def _(
    X_cv,
    cv_data,
    cv_results,
    data_service,
    feature_cols,
    mo,
    np,
    pd,
    px,
    tpot_model,
    xp_service,
):
    # Compare ML model against rule-based ExpectedPointsService
    if tpot_model and not cv_data.empty and feature_cols and cv_results:
        _best_model_name_comp = "tpot"
        _best_model_comp = tpot_model

        # Generate ML predictions
        _ml_predictions = _best_model_comp.predict(X_cv)

        # Now generate rule-based predictions for the SAME data
        # Need to reconstruct gameweek data for each GW in cv_data
        _unique_gws = sorted(cv_data["gameweek"].unique())

        try:
            _rule_based_predictions = []
            _rule_based_errors = []

            for _gw in _unique_gws:
                # Get players for this gameweek from cv_data
                _gw_mask = cv_data["gameweek"] == _gw
                _gw_player_ids = cv_data.loc[_gw_mask, "player_id"].unique()

                # Load gameweek data using data orchestration service
                try:
                    _gameweek_data = data_service.load_gameweek_data(
                        target_gameweek=_gw, load_live_data_history=True
                    )

                    # Calculate rule-based xP
                    _rule_xp = xp_service.calculate_expected_points(
                        gameweek_data=_gameweek_data,
                        use_ml_model=False,
                        gameweeks_ahead=1,
                    )

                    # Extract predictions for players in CV data
                    for _pid in _gw_player_ids:
                        _player_rule_xp = _rule_xp[_rule_xp["player_id"] == _pid][
                            "xP"
                        ].values
                        if len(_player_rule_xp) > 0:
                            _rule_based_predictions.append(_player_rule_xp[0])
                        else:
                            _rule_based_predictions.append(
                                0
                            )  # Fallback if player not found

                except Exception as e:
                    # If rule-based calculation fails for this GW, skip comparison
                    _rule_based_errors.append(f"GW{_gw}: {str(e)}")
                    # Fill with zeros for this gameweek
                    _rule_based_predictions.extend([0] * _gw_mask.sum())

            if len(_rule_based_predictions) == len(_ml_predictions):
                # Calculate errors
                _actual_points = cv_data["total_points"].values
                _ml_errors = np.abs(_ml_predictions - _actual_points)
                _rule_errors = np.abs(
                    np.array(_rule_based_predictions) - _actual_points
                )

                _ml_mae = _ml_errors.mean()
                _rule_mae = _rule_errors.mean()
                _ml_rmse = np.sqrt(((_ml_predictions - _actual_points) ** 2).mean())
                _rule_rmse = np.sqrt(
                    ((np.array(_rule_based_predictions) - _actual_points) ** 2).mean()
                )

                # Create comparison DataFrame
                _comparison_df = pd.DataFrame(
                    {
                        "Model": ["ML Model", "Rule-Based"],
                        "MAE": [_ml_mae, _rule_mae],
                        "RMSE": [_ml_rmse, _rule_rmse],
                        "MAE_Improvement": [0, _ml_mae - _rule_mae],
                    }
                )

                _fig_comparison = px.bar(
                    _comparison_df,
                    x="Model",
                    y=["MAE", "RMSE"],
                    title="ML vs Rule-Based Model Comparison",
                    barmode="group",
                    labels={"value": "Error (points)", "variable": "Metric"},
                    text_auto=".2f",
                )

                # Position-specific comparison if position available
                if "position" in cv_data.columns:
                    _position_comparison = []
                    for _pos in cv_data["position"].unique():
                        _pos_mask = cv_data["position"] == _pos
                        _pos_ml_mae = _ml_errors[_pos_mask].mean()
                        _pos_rule_mae = _rule_errors[_pos_mask].mean()
                        _position_comparison.append(
                            {
                                "Position": _pos,
                                "ML MAE": _pos_ml_mae,
                                "Rule-Based MAE": _pos_rule_mae,
                                "Improvement": _pos_rule_mae - _pos_ml_mae,
                            }
                        )

                    _pos_comp_df = pd.DataFrame(_position_comparison)

                    _fig_position_comp = px.bar(
                        _pos_comp_df,
                        x="Position",
                        y=["ML MAE", "Rule-Based MAE"],
                        title="Position-Specific: ML vs Rule-Based",
                        barmode="group",
                        labels={"value": "MAE (points)"},
                        text_auto=".2f",
                    )

                    _position_comp_display = mo.vstack(
                        [
                            mo.md("### 📊 Position-Specific Comparison"),
                            mo.ui.plotly(_fig_position_comp),
                            mo.md("---"),
                            mo.ui.table(_pos_comp_df.round(2), page_size=10),
                        ]
                    )
                else:
                    _position_comp_display = mo.md("_Position data not available_")

                _improvement_pct = (_rule_mae - _ml_mae) / _rule_mae * 100

                rule_comparison_display = mo.vstack(
                    [
                        mo.md("### ⚔️ ML vs Rule-Based Model"),
                        mo.md(f"**ML Model:** {_best_model_name_comp.upper()}"),
                        mo.md(""),
                        mo.ui.plotly(_fig_comparison),
                        mo.md("---"),
                        mo.md("### 📈 Performance Summary"),
                        mo.ui.table(_comparison_df.round(3), page_size=10),
                        mo.md(""),
                        mo.md("**Key Insights:**"),
                        mo.md(f"- **ML MAE:** {_ml_mae:.2f} points"),
                        mo.md(f"- **Rule-Based MAE:** {_rule_mae:.2f} points"),
                        mo.md(
                            f"- **Improvement:** {_improvement_pct:+.1f}% {'✅ ML WINS' if _improvement_pct > 0 else '❌ Rule-Based WINS'}"
                        ),
                        mo.md(
                            f"- **Absolute Improvement:** {_rule_mae - _ml_mae:+.2f} points per prediction"
                        ),
                        mo.md("---"),
                        _position_comp_display,
                        mo.md("---") if _rule_based_errors else mo.md(""),
                        mo.md(
                            "**Errors during rule-based calculation:**\n\n"
                            + "\n".join([f"- {e}" for e in _rule_based_errors])
                        )
                        if _rule_based_errors
                        else mo.md(""),
                    ]
                )
            else:
                rule_comparison_display = mo.md(
                    f"❌ **Prediction length mismatch:** ML={len(_ml_predictions)}, Rule={len(_rule_based_predictions)}"
                )

        except Exception as e:
            rule_comparison_display = mo.md(
                f"❌ **Rule-based comparison failed:** {str(e)}\n\nThis is expected if historical gameweek data is not available for all CV gameweeks."
            )
    else:
        rule_comparison_display = mo.md("⚠️ **Train models first**")

    rule_comparison_display
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1️⃣1️⃣ Summary & Next Steps

    **Cross-validation provides robust model evaluation with confidence intervals.**

    The CV metrics above show mean ± std across all folds, giving you confidence in model performance.

    ### Diagnostic Insights:
    1. **Position-Specific Errors** - Identify which positions need model improvements (FWD vs DEF)
    2. **Prediction Bias** - Detect systematic over/under-prediction patterns
    3. **ML vs Rule-Based** - Validate ML actually improves over current production model

    ### Next Steps:
    1. Review position-specific errors to focus improvement efforts
    2. Check for systematic biases in price bands or team strength
    3. Compare ML vs Rule-Based to confirm ML provides value
    4. If ML wins: Export trained model for production use
    5. If Rule-Based wins: Focus on improving rule-based parameters instead
    6. Integrate best model with `MLExpectedPointsService` in gameweek_manager.py
    """
    )
    return


if __name__ == "__main__":
    app.run()
