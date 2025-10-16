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
        create_fpl_pipeline,
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
        create_fpl_pipeline,
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
    return (historical_data_table,)


@app.cell
def _(mo):
    mo.md(r"""## 2️⃣ Team Strength Ratings - Opponent Quality Context""")
    return


@app.cell
def _(teams_df, mo, get_team_strength_ratings):
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
    fixtures_df, historical_df, mo, np, pd, team_strength, teams_df, FPLFeatureEngineer
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
    return (features_df, production_feature_cols)


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
    return (features_table,)


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
def _(end_gw_input, features_df, mo, n_folds_input, np, pd, production_feature_cols):
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

    mo.vstack(
        [
            mo.md("### 🎯 Model Selection"),
            model_selector,
            mo.md(""),
            train_button,
            mo.md("---"),
        ]
    )
    return model_selector, train_button


@app.cell
def _(
    GradientBoostingRegressor,
    GroupKFold,
    Pipeline,
    RandomForestRegressor,
    Ridge,
    StandardScaler,
    X_cv,
    cross_val_score,
    cv_groups,
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
        # Initialize GroupKFold cross-validator
        _gkf = GroupKFold(n_splits=n_folds_input.value)

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
                groups=cv_groups,
                cv=_gkf,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            _ridge_rmse_scores = np.sqrt(
                -cross_val_score(
                    ridge_pipeline,
                    X_cv,
                    y_cv,
                    groups=cv_groups,
                    cv=_gkf,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )
            )
            _ridge_r2_scores = cross_val_score(
                ridge_pipeline,
                X_cv,
                y_cv,
                groups=cv_groups,
                cv=_gkf,
                scoring="r2",
                n_jobs=-1,
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
                groups=cv_groups,
                cv=_gkf,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            _rf_rmse_scores = np.sqrt(
                -cross_val_score(
                    rf_model,
                    X_cv,
                    y_cv,
                    groups=cv_groups,
                    cv=_gkf,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )
            )
            _rf_r2_scores = cross_val_score(
                rf_model,
                X_cv,
                y_cv,
                groups=cv_groups,
                cv=_gkf,
                scoring="r2",
                n_jobs=-1,
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
                groups=cv_groups,
                cv=_gkf,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            _gb_rmse_scores = np.sqrt(
                -cross_val_score(
                    gb_model,
                    X_cv,
                    y_cv,
                    groups=cv_groups,
                    cv=_gkf,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )
            )
            _gb_r2_scores = cross_val_score(
                gb_model, X_cv, y_cv, groups=cv_groups, cv=_gkf, scoring="r2", n_jobs=-1
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
    return (eval_display,)


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
    mo.md(
        r"""
    ## 8️⃣ Summary & Next Steps

    **Cross-validation provides robust model evaluation with confidence intervals.**

    The CV metrics above show mean ± std across all folds, giving you confidence in model performance.

    ### Next Steps:
    1. Compare Ridge, Random Forest, and Gradient Boosting models
    2. Select the best performer based on CV MAE scores
    3. Export the trained model for production use
    4. Integrate with `MLExpectedPointsService` in the main codebase
    """
    )
    return


if __name__ == "__main__":
    app.run()
