"""
ML Expected Points (xP) Experimentation Notebook

This notebook provides a structured environment for developing, testing, and validating
ML-based xP prediction models. The goal is to improve upon the current rule-based model
used in gameweek_manager.py.

Key Features:
- Leak-free player-based validation with 5-gameweek rolling windows
- Comprehensive feature engineering:
  * Cumulative season statistics (up to GW N-1)
  * Rolling 5GW form features (GW N-6 to N-1)
  * Season-long and rolling per-90 efficiency metrics
  * Consistency and volatility indicators
  * Team context features (rolling 5GW team form, cumulative team stats)
- Position-specific model evaluation
- Model comparison against rule-based baseline
- Production-ready model export

Modeling Strategy:
- Use 5 preceding gameweeks (GW N-5 to N-1) to predict GW N performance
- Player-based train/test split using GroupKFold (ensures each player's data stays together)
- All features properly lagged to prevent data leakage (shift(1) applied to all rolling stats)
- Team features are safe: Testing "can we predict NEW players on KNOWN teams?" not future outcomes
- Minimum requirement: GW6+ for complete rolling feature coverage
- Tests model's ability to generalize to unseen players (not unseen time periods)
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
    # 🤖 ML Expected Points (xP) Experimentation Lab

    **Goal:** Develop and validate ML models to improve xP predictions beyond the current rule-based approach.

    ---
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 1️⃣ Data Loading - Historical Gameweeks""")
    return


@app.cell
def _(data_service, mo):
    # Get current gameweek
    gw_info = data_service.get_current_gameweek_info()
    current_gw = gw_info.get("current_gameweek", 1)

    # Gameweek range selector
    start_gw_input = mo.ui.number(
        start=1,
        stop=38,
        value=1,
        step=1,
        label="Training Start Gameweek",
    )

    end_gw_input = mo.ui.number(
        start=2,
        stop=38,
        value=min(6, current_gw),
        step=1,
        label="Training End Gameweek (predict this GW)",
    )

    mo.vstack(
        [
            mo.md(f"**Current Season Gameweek:** GW{current_gw}"),
            mo.md("---"),
            mo.md("### 📅 Select Training Window"),
            mo.md("*Training data: GW{start} to GW{end-1} → Predict GW{end}*"),
            start_gw_input,
            end_gw_input,
            mo.md("*Example: GW1-5 data predicts GW6 (no future leakage)*"),
        ]
    )
    return end_gw_input, start_gw_input


@app.cell
def _(client, end_gw_input, mo, pd, start_gw_input):
    # Load historical gameweek performance data + fixtures & teams for opponent features
    historical_data = []
    data_load_status = []
    fixtures_df = pd.DataFrame()
    teams_df = pd.DataFrame()

    if start_gw_input.value and end_gw_input.value:
        if end_gw_input.value <= start_gw_input.value:
            load_summary = mo.md("❌ **End gameweek must be after start gameweek**")
        else:
            try:
                # Load actual performance data for each gameweek
                for gw in range(start_gw_input.value, end_gw_input.value + 1):
                    gw_performance = client.get_gameweek_performance(gw)
                    if not gw_performance.empty:
                        gw_performance["gameweek"] = gw
                        historical_data.append(gw_performance)
                        data_load_status.append(
                            f"✅ GW{gw}: {len(gw_performance)} players"
                        )
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

                    load_summary = mo.vstack(
                        [
                            mo.md(f"### ✅ Loaded {len(historical_data)} Gameweeks"),
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
    else:
        historical_df = pd.DataFrame()
        load_summary = mo.md("⚠️ **Select gameweek range to load data**")

    load_summary
    return fixtures_df, historical_df, teams_df


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
    # ✅ USE PRODUCTION CODE: Team strength ratings from ml_pipeline_factory
    # This ensures notebook and gameweek_manager.py use IDENTICAL logic
    team_strength = get_team_strength_ratings()

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
    fixtures_df,
    historical_df,
    mo,
    pd,
    team_strength,
    teams_df,
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
        # Initialize production feature engineer
        feature_engineer = FPLFeatureEngineer(
            fixtures_df=fixtures_df if not fixtures_df.empty else None,
            teams_df=teams_df if not teams_df.empty else None,
            team_strength=team_strength if team_strength else None,
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
                mo.md(
                    "💡 **Team features are safe with player-based GroupKFold:** We test the model's ability to predict NEW players on KNOWN teams, not future outcomes."
                ),
                mo.md(""),
                mo.md(f"**Total features created:** {len(features_df.columns)}"),
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
    mo.md(r"""## 4️⃣ Cross-Validation Setup - Player-Based Stratified K-Fold""")
    return


@app.cell
def _(mo):
    # Number of folds for cross-validation
    n_folds_input = mo.ui.slider(
        start=3,
        stop=10,
        value=5,
        step=1,
        label="Number of Cross-Validation Folds",
        show_value=True,
    )
    n_folds_input
    return (n_folds_input,)


@app.cell
def _(
    end_gw_input,
    features_df,
    mo,
    n_folds_input,
    pd,
    production_feature_cols,
):
    # Prepare data for cross-validation
    # Strategy: Use ALL available gameweeks (GW6+) with player-based stratified k-fold
    if not features_df.empty and end_gw_input.value and n_folds_input.value:
        # Validate sufficient training history for 5GW rolling features
        # Need at least GW6 to have 5 preceding gameweeks of data
        if end_gw_input.value < 6:
            cv_summary = mo.md(
                f"⚠️ **Target gameweek must be GW6 or later for 5GW rolling features.**\n\n"
                f"Current selection: GW{end_gw_input.value}\n\n"
                f"Rolling 5GW features require data from GW N-5 to N-1 to predict GW N."
            )
            X_cv = pd.DataFrame()
            y_cv = pd.Series()
            cv_data = pd.DataFrame()
            feature_cols = []
            cv_groups = None
        else:
            # Use all completed gameweeks from GW6 up to end_gw (inclusive)
            # This provides more training data across multiple gameweeks
            cv_data = features_df[
                (features_df["gameweek"] >= 6)
                & (features_df["gameweek"] <= end_gw_input.value)
            ].copy()

            # ✅ USE PRODUCTION FEATURE LIST from FPLFeatureEngineer
            # This ensures notebook training uses same features as gameweek_manager.py
            # OLD: Hardcoded list of 63 features (could drift from production)
            # NEW: Dynamic list from production code (always in sync)
            feature_cols = production_feature_cols

            # Validate all feature columns exist in the data
            missing_features = [
                col for col in feature_cols if col not in cv_data.columns
            ]
            if missing_features:
                raise ValueError(
                    f"Missing feature columns: {missing_features}. "
                    f"Available columns: {list(cv_data.columns)}"
                )

            # Prepare X and y for cross-validation
            X_cv = cv_data[feature_cols].fillna(0)
            y_cv = cv_data["total_points"]

            # Create groups for GroupKFold based on player_id
            # This ensures each player's data stays together in folds
            cv_groups = cv_data["player_id"].values

            # Calculate gameweek coverage
            _cv_gws = sorted(cv_data["gameweek"].unique())
            _gw_range = (
                f"GW{min(_cv_gws)}-{max(_cv_gws)}"
                if len(_cv_gws) > 1
                else f"GW{_cv_gws[0]}"
            )

            cv_summary = mo.vstack(
                [
                    mo.md("### ✅ Cross-Validation Setup Complete"),
                    mo.md(
                        f"**Strategy:** {n_folds_input.value}-Fold Player-Based Cross-Validation"
                    ),
                    mo.md(f"**Data:** {_gw_range} ({len(cv_data):,} total records)"),
                    mo.md(
                        f"**Players:** {cv_data['player_id'].nunique():,} unique players"
                    ),
                    mo.md(
                        f"**Features:** {len(feature_cols)} columns (5GW rolling windows)"
                    ),
                    mo.md(""),
                    mo.md("**✅ No Data Leakage:**"),
                    mo.md("- All features use `.shift(1)` to exclude current gameweek"),
                    mo.md("- Cumulative stats: up to GW N-1 only"),
                    mo.md("- Rolling features: GW N-6 to N-1 window"),
                    mo.md("- Player-based CV ensures generalization to unseen players"),
                    mo.md(""),
                    mo.md("**Cross-Validation Details:**"),
                    mo.md(
                        f"- Each fold tests on ~{100 // n_folds_input.value}% of players"
                    ),
                    mo.md(
                        f"- {n_folds_input.value} independent evaluations for robust metrics"
                    ),
                    mo.md("- GroupKFold ensures player data stays within single fold"),
                    mo.md(""),
                    mo.md("**Feature Categories:**"),
                    mo.md("- Static: 3 features (price, position, games_played)"),
                    mo.md("- Cumulative Season Stats: 9 features"),
                    mo.md("- Cumulative Per-90 Rates: 7 features"),
                    mo.md("- Rolling 5GW Form: 13 features"),
                    mo.md("- Rolling 5GW Per-90 Rates: 3 features"),
                    mo.md("- Defensive Metrics: 4 features"),
                    mo.md("- Consistency & Volatility: 4 features"),
                    mo.md("- Team Context: 14 features"),
                    mo.md(
                        "- **Fixture Features (NEW!): 6 features** (opponent strength, home/away, opponent defense)"
                    ),
                    mo.md(""),
                    mo.md(f"**Total: {len(feature_cols)} features**"),
                    mo.md("---"),
                ]
            )
    else:
        X_cv = pd.DataFrame()
        y_cv = pd.Series()
        cv_data = pd.DataFrame()
        feature_cols = []
        cv_groups = None
        cv_summary = mo.md("⚠️ **Engineer features first**")

    cv_summary
    return X_cv, cv_data, cv_groups, feature_cols, y_cv


@app.cell
def _(cv_data, mo, X_cv, y_cv):
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
    return (sample_display,)


@app.cell
def _(mo):
    mo.md(r"""## 5️⃣ Model Training - Multiple Algorithms""")
    return


@app.cell
def _(mo):
    # Model selection
    model_selector = mo.ui.dropdown(
        options={
            "Ridge Regression": "ridge",
            "Random Forest": "rf",
            "Gradient Boosting": "gb",
            "Ensemble (All Models)": "ensemble",
        },
        value="Ridge Regression",
        label="Select ML Algorithm",
    )

    train_button = mo.ui.run_button(
        label="🚀 Train Model",
        kind="success",
    )

    # CV Strategy Selection
    cv_strategy_selector = mo.ui.dropdown(
        options={
            "player_based": "Player-Based GroupKFold (test on NEW players)",
            "temporal": "Temporal Walk-Forward (test on NEXT gameweek)",
        },
        value="temporal",
        label="Cross-Validation Strategy",
    )

    mo.vstack(
        [
            mo.md("### 🎯 Model Selection"),
            model_selector,
            mo.md(""),
            mo.md("### 📊 Cross-Validation Strategy"),
            cv_strategy_selector,
            mo.md(
                "- **Player-Based:** Tests generalization to unseen players (mixes gameweeks)"
            ),
            mo.md(
                "- **Temporal:** Tests prediction of next future gameweek (proper time-series validation) ⭐"
            ),
            mo.md(""),
            train_button,
            mo.md("---"),
        ]
    )
    return cv_strategy_selector, model_selector, train_button


@app.cell
def _(
    GradientBoostingRegressor,
    GroupKFold,
    Pipeline,
    RandomForestRegressor,
    Ridge,
    StandardScaler,
    TimeSeriesSplit,
    X_cv,
    cross_val_score,
    cv_data,
    cv_groups,
    cv_strategy_selector,
    mo,
    model_selector,
    n_folds_input,
    np,
    train_button,
    y_cv,
):
    # Train models with cross-validation
    trained_models = {}
    cv_results = {}
    _final_metrics = {}

    if train_button.value and not X_cv.empty and cv_groups is not None:
        # Initialize cross-validator based on selected strategy
        _use_temporal = False  # Track which CV strategy is actually being used

        # DEBUG: Print which CV strategy was selected
        print(f"🔍 CV Strategy Selected: {cv_strategy_selector.value}")

        # marimo dropdown returns the display label, not the key
        if "temporal" in cv_strategy_selector.value.lower():
            # Temporal walk-forward validation
            # Train on GW6 to N-1, test on GW N (iterate forward)
            _cv_splits = []
            _cv_gws = sorted(cv_data["gameweek"].unique())

            print(f"🔍 Available gameweeks: {_cv_gws}")

            # Need at least 2 gameweeks for temporal CV (train on 1, test on 1)
            if len(_cv_gws) < 2:
                print(
                    "⚠️ Need at least 2 gameweeks for temporal CV. Using player-based instead."
                )
                _gkf = GroupKFold(n_splits=n_folds_input.value)
                _cv_strategy_used = "player_based (fallback)"
                _use_temporal = False
            else:
                # Create temporal splits: train on all GWs up to N-1, test on GW N
                # IMPORTANT: sklearn needs positional indices (0, 1, 2...), not DataFrame index values
                # Reset index to ensure alignment with X_cv and y_cv
                cv_data_reset = cv_data.reset_index(drop=True)

                for i in range(len(_cv_gws) - 1):
                    train_gws = _cv_gws[: i + 1]  # GW6 to GW(6+i)
                    test_gw = _cv_gws[i + 1]  # GW(6+i+1)

                    # Get positional indices (not DataFrame .index values)
                    train_mask = cv_data_reset["gameweek"].isin(train_gws)
                    test_mask = cv_data_reset["gameweek"] == test_gw

                    train_idx = np.where(train_mask)[0]
                    test_idx = np.where(test_mask)[0]

                    _cv_splits.append((train_idx, test_idx))
                    print(
                        f"  Fold {i + 1}: Train on GW{min(train_gws)}-{max(train_gws)} ({len(train_idx)} samples) → Test on GW{test_gw} ({len(test_idx)} samples)"
                    )

                # Use custom CV splits (pass list directly to sklearn)
                _gkf = _cv_splits
                _cv_strategy_used = f"temporal ({len(_cv_splits)} folds)"
                _use_temporal = True
                print(f"✅ Temporal CV: {len(_cv_splits)} folds created")
        else:
            # Player-based GroupKFold
            print(f"✅ Player-Based CV: {n_folds_input.value} folds (GroupKFold)")
            _gkf = GroupKFold(n_splits=n_folds_input.value)
            _cv_strategy_used = "player_based"
            _use_temporal = False

        # Prepare CV arguments
        # For temporal CV (custom splits), groups parameter should be omitted
        # For player-based CV (GroupKFold), groups parameter is required
        _cv_kwargs = {
            "cv": _gkf,
            "n_jobs": -1,
        }
        if not _use_temporal:
            _cv_kwargs["groups"] = cv_groups

        # Train selected model(s)
        if model_selector.value == "ridge" or model_selector.value == "ensemble":
            # Ridge Regression with StandardScaler pipeline
            ridge_pipeline = Pipeline(
                [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
            )

            # Cross-validation scoring
            _ridge_mae_scores = -cross_val_score(
                ridge_pipeline,
                X_cv,
                y_cv,
                scoring="neg_mean_absolute_error",
                **_cv_kwargs,
            )
            _ridge_rmse_scores = np.sqrt(
                -cross_val_score(
                    ridge_pipeline,
                    X_cv,
                    y_cv,
                    scoring="neg_mean_squared_error",
                    **_cv_kwargs,
                )
            )
            _ridge_r2_scores = cross_val_score(
                ridge_pipeline,
                X_cv,
                y_cv,
                scoring="r2",
                **_cv_kwargs,
            )

            # Train final model on all data
            ridge_pipeline.fit(X_cv, y_cv)
            trained_models["ridge"] = ridge_pipeline

            cv_results["ridge"] = {
                "MAE_mean": _ridge_mae_scores.mean(),
                "MAE_std": _ridge_mae_scores.std(),
                "RMSE_mean": _ridge_rmse_scores.mean(),
                "RMSE_std": _ridge_rmse_scores.std(),
                "R²_mean": _ridge_r2_scores.mean(),
                "R²_std": _ridge_r2_scores.std(),
                "fold_scores": _ridge_mae_scores,
            }

        if model_selector.value == "rf" or model_selector.value == "ensemble":
            # Random Forest (no scaling needed)
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1,
            )

            # Cross-validation scoring
            _rf_mae_scores = -cross_val_score(
                rf_model,
                X_cv,
                y_cv,
                scoring="neg_mean_absolute_error",
                **_cv_kwargs,
            )
            _rf_rmse_scores = np.sqrt(
                -cross_val_score(
                    rf_model,
                    X_cv,
                    y_cv,
                    scoring="neg_mean_squared_error",
                    **_cv_kwargs,
                )
            )
            _rf_r2_scores = cross_val_score(
                rf_model,
                X_cv,
                y_cv,
                scoring="r2",
                **_cv_kwargs,
            )

            # Train final model on all data
            rf_model.fit(X_cv, y_cv)
            trained_models["rf"] = rf_model

            cv_results["rf"] = {
                "MAE_mean": _rf_mae_scores.mean(),
                "MAE_std": _rf_mae_scores.std(),
                "RMSE_mean": _rf_rmse_scores.mean(),
                "RMSE_std": _rf_rmse_scores.std(),
                "R²_mean": _rf_r2_scores.mean(),
                "R²_std": _rf_r2_scores.std(),
                "fold_scores": _rf_mae_scores,
            }

        if model_selector.value == "gb" or model_selector.value == "ensemble":
            # Gradient Boosting (no scaling needed)
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )

            # Cross-validation scoring
            _gb_mae_scores = -cross_val_score(
                gb_model,
                X_cv,
                y_cv,
                scoring="neg_mean_absolute_error",
                **_cv_kwargs,
            )
            _gb_rmse_scores = np.sqrt(
                -cross_val_score(
                    gb_model,
                    X_cv,
                    y_cv,
                    scoring="neg_mean_squared_error",
                    **_cv_kwargs,
                )
            )
            _gb_r2_scores = cross_val_score(
                gb_model,
                X_cv,
                y_cv,
                scoring="r2",
                **_cv_kwargs,
            )

            # Train final model on all data
            gb_model.fit(X_cv, y_cv)
            trained_models["gb"] = gb_model

            cv_results["gb"] = {
                "MAE_mean": _gb_mae_scores.mean(),
                "MAE_std": _gb_mae_scores.std(),
                "RMSE_mean": _gb_rmse_scores.mean(),
                "RMSE_std": _gb_rmse_scores.std(),
                "R²_mean": _gb_r2_scores.mean(),
                "R²_std": _gb_r2_scores.std(),
                "fold_scores": _gb_mae_scores,
            }

        # Display cross-validation results
        _metrics_display = [
            mo.md(f"### ✅ Cross-Validation Complete ({n_folds_input.value} folds)")
        ]
        for _model_name, _cv_metrics in cv_results.items():
            _metrics_display.extend(
                [
                    mo.md(f"**{_model_name.upper()} Cross-Validation Metrics:**"),
                    mo.md(
                        f"- MAE: {_cv_metrics['MAE_mean']:.3f} ± {_cv_metrics['MAE_std']:.3f} points"
                    ),
                    mo.md(
                        f"- RMSE: {_cv_metrics['RMSE_mean']:.3f} ± {_cv_metrics['RMSE_std']:.3f} points"
                    ),
                    mo.md(
                        f"- R²: {_cv_metrics['R²_mean']:.3f} ± {_cv_metrics['R²_std']:.3f}"
                    ),
                    mo.md(
                        f"- Fold MAE scores: {', '.join([f'{s:.2f}' for s in _cv_metrics['fold_scores']])}"
                    ),
                    mo.md(""),
                ]
            )

        training_summary = mo.vstack(_metrics_display + [mo.md("---")])
    else:
        training_summary = mo.md("👆 **Click train button after preparing data**")

    training_summary
    return cv_results, trained_models


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
def _(feature_cols, mo, pd, px, trained_models):
    # Feature importance analysis
    if trained_models and feature_cols:
        importance_data = []

        # Ridge coefficients (absolute values) - extract from pipeline
        if "ridge" in trained_models:
            ridge_importance = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": abs(
                        trained_models["ridge"].named_steps["ridge"].coef_
                    ),
                    "model": "Ridge",
                }
            ).nlargest(15, "importance")
            importance_data.append(ridge_importance)

        # Random Forest feature importance
        if "rf" in trained_models:
            rf_importance = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": trained_models["rf"].feature_importances_,
                    "model": "Random Forest",
                }
            ).nlargest(15, "importance")
            importance_data.append(rf_importance)

        # Gradient Boosting feature importance
        if "gb" in trained_models:
            gb_importance = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": trained_models["gb"].feature_importances_,
                    "model": "Gradient Boosting",
                }
            ).nlargest(15, "importance")
            importance_data.append(gb_importance)

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
def _(X_cv, cv_data, cv_results, feature_cols, mo, np, px, trained_models):
    # Position-specific error breakdown to identify which positions need improvement
    if trained_models and not cv_data.empty and feature_cols:
        # Generate predictions from best model (lowest MAE)
        if cv_results:
            _best_model_name = min(cv_results.items(), key=lambda x: x[1]["MAE_mean"])[
                0
            ]
            _best_model = trained_models[_best_model_name]

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
                            f"**Best Model:** {_best_model_name.upper()} (MAE: {cv_results[_best_model_name]['MAE_mean']:.2f})"
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
            position_analysis_display = mo.md("⚠️ **No CV results available**")
    else:
        position_analysis_display = mo.md("⚠️ **Train models first**")

    position_analysis_display
    return


@app.cell
def _(mo):
    mo.md(r"""## 9️⃣ Prediction Bias Detection - Systematic Over/Under-Prediction""")
    return


@app.cell
def _(X_cv, cv_data, cv_results, feature_cols, mo, pd, px, trained_models):
    # Detect systematic biases in predictions (price bands, team strength, etc.)
    if trained_models and not cv_data.empty and feature_cols and cv_results:
        _best_model_name_bias = min(cv_results.items(), key=lambda x: x[1]["MAE_mean"])[
            0
        ]
        _best_model_bias = trained_models[_best_model_name_bias]

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
    trained_models,
    xp_service,
):
    # Compare ML model against rule-based ExpectedPointsService
    if trained_models and not cv_data.empty and feature_cols and cv_results:
        _best_model_name_comp = min(cv_results.items(), key=lambda x: x[1]["MAE_mean"])[
            0
        ]
        _best_model_comp = trained_models[_best_model_name_comp]

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
