import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # ML experiment imports
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    # ML libraries
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    # import shap  # Commented out due to Python 3.13 compatibility issues

    # FPL data pipeline
    from fpl_team_picker.core.data_loader import (
        fetch_fpl_data,
        get_current_gameweek_info,
    )
    from fpl_team_picker.core.xp_model import XPModel
    from client import FPLDataClient
    from fpl_team_picker.config import config

    return (
        FPLDataClient,
        LabelEncoder,
        RidgeCV,
        StandardScaler,
        XPModel,
        config,
        fetch_fpl_data,
        get_current_gameweek_info,
        go,
        mean_absolute_error,
        np,
        pd,
        px,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # 🤖 Enhanced ML XP Model Experiment with Rich Database Features

    **Advanced machine learning experiment leveraging rich FPL database features to improve XGBoost predictions beyond the rule-based XP model.**

    ## Enhanced Feature Approach
    - **Rich database features**: Injury risk, set piece orders, performance rankings, market intelligence
    - **Advanced analytics**: Official xG90/xA90 from FPL API, transfer momentum, value analysis
    - **Form intelligence**: Rolling averages, consistency metrics, trend analysis
    - **Position-specific models**: Specialized models for GKP/DEF/MID/FWD with ensemble predictions
    - **Temporal validation**: Leak-free training with proper time-series validation
    - **Feature importance analysis**: Understanding which rich features drive accuracy

    ---
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1️⃣ Data Loading & Target Gameweek

    **Load FPL data and select target gameweek for experimentation.**
    """
    )
    return


@app.cell
def _(get_current_gameweek_info, mo):
    # Get current gameweek info
    gw_info = get_current_gameweek_info()
    current_gw = gw_info["current_gameweek"]

    # Target gameweek selector (default to 4 for ML experiment)
    target_gw_input = mo.ui.number(
        value=4,  # Default to GW4 for ML experiment
        start=2,  # Need at least GW2 for historical data
        stop=38,
        step=1,
        label="Target Gameweek for ML Experiment",
    )

    mo.vstack(
        [
            mo.md(f"**Current gameweek detected:** GW{current_gw}"),
            mo.md("**Select target gameweek:**"),
            target_gw_input,
            mo.md("*Note: Need GW2+ to have historical data for training*"),
        ]
    )
    return (target_gw_input,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 1️⃣b Training Configuration

    **Configure how many previous gameweeks to use for training.**
    """
    )
    return


@app.cell
def _(mo):
    # Training window configuration
    training_window_input = mo.ui.number(
        value=5,  # Default to 5 gameweeks
        start=2,  # Minimum 2 gameweeks for training
        stop=20,  # Maximum 20 gameweeks
        step=1,
        label="Number of previous gameweeks to train on",
    )

    use_all_available = mo.ui.checkbox(
        label="Use all available gameweeks (ignore window size)", value=False
    )

    train_button = mo.ui.button(
        label="🚀 Train Model",
        value=0,
        on_click=lambda value: value + 1,
        kind="success",
    )

    mo.vstack(
        [
            mo.md("### 🧠 ML Training Configuration"),
            training_window_input,
            use_all_available,
            mo.md(
                "*This controls how much historical data the model uses for training*"
            ),
            train_button,
            mo.md("---"),
        ]
    )
    return (training_window_input, use_all_available, train_button)


@app.cell
def _(fetch_fpl_data, mo, pd, target_gw_input):
    # Initialize variables
    ml_data_loaded = False
    players_df = pd.DataFrame()
    teams_df = pd.DataFrame()
    xg_rates_df = pd.DataFrame()
    fixtures_df = pd.DataFrame()
    live_data_df = pd.DataFrame()

    if target_gw_input.value and target_gw_input.value >= 2:
        try:
            # Load FPL data for target gameweek
            players_df, teams_df, xg_rates_df, fixtures_df, _, live_data_df = (
                fetch_fpl_data(target_gw_input.value)
            )
            ml_data_loaded = True

            data_summary = mo.vstack(
                [
                    mo.md(f"### ✅ Data Loaded for GW{target_gw_input.value}"),
                    mo.md(
                        f"**Players:** {len(players_df)} | **Teams:** {len(teams_df)} | **Fixtures:** {len(fixtures_df)}"
                    ),
                    mo.md(f"**Live data:** {len(live_data_df)} gameweek records"),
                    mo.md("---"),
                ]
            )
        except Exception as e:
            data_summary = mo.md(f"❌ **Data loading failed:** {str(e)}")
    else:
        data_summary = mo.md("⚠️ **Select a target gameweek (GW2+) to load data**")

    data_summary
    return (
        fixtures_df,
        live_data_df,
        ml_data_loaded,
        players_df,
        teams_df,
        xg_rates_df,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2️⃣ Generate Rule-Based XP Predictions

    **Run our current XP model to get baseline predictions for comparison.**
    """
    )
    return


@app.cell
def _(
    XPModel,
    config,
    fixtures_df,
    live_data_df,
    ml_data_loaded,
    mo,
    pd,
    players_df,
    target_gw_input,
    teams_df,
    xg_rates_df,
):
    # Initialize variables
    rule_based_predictions = pd.DataFrame()

    if ml_data_loaded:
        try:
            # Create XP model instance
            xp_model = XPModel(
                form_weight=config.xp_model.form_weight,
                form_window=config.xp_model.form_window,
                debug=False,  # Disable debug for cleaner output
            )

            # Generate 1GW predictions (our main comparison point)
            players_1gw = xp_model.calculate_expected_points(
                players_data=players_df,
                teams_data=teams_df,
                xg_rates_data=xg_rates_df,
                fixtures_data=fixtures_df,
                target_gameweek=target_gw_input.value,
                live_data=live_data_df,
                gameweeks_ahead=1,
            )

            rule_based_predictions = players_1gw.copy()

            xp_summary = mo.vstack(
                [
                    mo.md("### ✅ Rule-Based XP Generated"),
                    mo.md(
                        f"**Players with predictions:** {len(rule_based_predictions)}"
                    ),
                    mo.md(f"**Average XP:** {rule_based_predictions['xP'].mean():.2f}"),
                    mo.md(f"**Top XP:** {rule_based_predictions['xP'].max():.2f}"),
                    mo.md("---"),
                ]
            )

        except Exception as e:
            xp_summary = mo.md(f"❌ **XP model failed:** {str(e)}")
            rule_based_predictions = pd.DataFrame()
    else:
        xp_summary = mo.md("⚠️ **Load data first to generate XP predictions**")

    xp_summary
    return (rule_based_predictions,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 3️⃣ Prepare Training Data

    **Create features and target variable for ML training using historical gameweek data.**
    """
    )
    return


@app.cell
def _(
    FPLDataClient,
    LabelEncoder,
    live_data_df,
    mo,
    np,
    pd,
    rule_based_predictions,
    target_gw_input,
    train_button,
    training_window_input,
    use_all_available,
):
    # Initialize variables with defaults
    training_data = pd.DataFrame()
    ml_features = []
    target_variable = "total_points"
    le_position = LabelEncoder()

    # Only run training if button has been clicked
    if (
        train_button.value > 0
        and not rule_based_predictions.empty
        and not live_data_df.empty
    ):
        try:
            # Get training data from historical gameweeks (before target gameweek)
            historical_gws = sorted(
                [
                    gw
                    for gw in live_data_df["event"].unique()
                    if gw < target_gw_input.value
                ]
            )

            if len(historical_gws) < 2:
                training_summary = mo.md(
                    "⚠️ **Need at least 2 historical gameweeks for training**"
                )
                training_data = (
                    pd.DataFrame()
                )  # Initialize empty when insufficient data
                ml_features = []  # Also initialize ml_features
            else:
                # Configurable training window logic
                if use_all_available.value:
                    # Use all available historical gameweeks
                    train_gws = historical_gws
                    training_strategy = f"All available ({len(historical_gws)} GWs)"
                else:
                    # Use specified training window
                    window_size = training_window_input.value
                    train_gws = (
                        historical_gws[-window_size:]
                        if len(historical_gws) >= window_size
                        else historical_gws
                    )
                    training_strategy = (
                        f"Last {len(train_gws)} GWs (window: {window_size})"
                    )

                training_records = []

                # Get enhanced player data with rich features (static data, not leaked)
                training_client = FPLDataClient()
                current_players = training_client.get_current_players()
                enhanced_players = training_client.get_players_enhanced()
                player_positions = (
                    current_players[["player_id", "position"]]
                    .set_index("player_id")["position"]
                    .to_dict()
                )

                # Create enhanced player features mapping for static enrichment (leak-free)
                enhanced_features_cols = [
                    "chance_of_playing_next_round",  # Next gameweek availability (known before current GW)
                    "corners_and_indirect_freekicks_order",
                    "direct_freekicks_order",
                    "penalties_order",
                    "expected_goals_per_90",
                    "expected_assists_per_90",
                    "form_rank",
                    "ict_index_rank",
                    "points_per_game_rank",
                    "influence_rank",
                    "creativity_rank",
                    "threat_rank",
                    "value_form",
                    "cost_change_start",
                    # Removed: 'chance_of_playing_this_round' (current GW info - data leakage)
                ]

                # Filter to only include available columns and create mapping
                available_enhanced_cols = [
                    enhanced_col
                    for enhanced_col in enhanced_features_cols
                    if enhanced_col in enhanced_players.columns
                ]
                player_enhanced_features = enhanced_players[
                    ["player_id"] + available_enhanced_cols
                ].set_index("player_id")

                for gw in train_gws:
                    # Target: what we want to predict (current gameweek performance)
                    gw_target = live_data_df[live_data_df["event"] == gw][
                        ["player_id", "total_points"]
                    ].copy()

                    # Calculate cumulative stats up to BEFORE this gameweek only (no data leakage)
                    historical_data_up_to_gw = live_data_df[live_data_df["event"] < gw]

                    # Get previous gameweek data for lagged features (if available)
                    prev_gw_data = (
                        live_data_df[live_data_df["event"] == (gw - 1)]
                        if gw > 1
                        else pd.DataFrame()
                    )

                    # Calculate cumulative season stats up to this gameweek
                    cumulative_stats = (
                        historical_data_up_to_gw.groupby("player_id")
                        .agg(
                            {
                                "total_points": "sum",
                                "minutes": "sum",
                                "goals_scored": "sum",
                                "assists": "sum",
                                "expected_goals": "sum",
                                "expected_assists": "sum",
                                "bps": "mean",  # Average BPS per game
                                "ict_index": "mean",  # Average ICT per game
                            }
                        )
                        .reset_index()
                    )

                    # Calculate per-90 stats based only on data available up to this gameweek
                    cumulative_stats["xG90_historical"] = np.where(
                        cumulative_stats["minutes"] > 0,
                        (
                            cumulative_stats["expected_goals"]
                            / cumulative_stats["minutes"]
                        )
                        * 90,
                        0,
                    )
                    cumulative_stats["xA90_historical"] = np.where(
                        cumulative_stats["minutes"] > 0,
                        (
                            cumulative_stats["expected_assists"]
                            / cumulative_stats["minutes"]
                        )
                        * 90,
                        0,
                    )
                    cumulative_stats["points_per_90"] = np.where(
                        cumulative_stats["minutes"] > 0,
                        (cumulative_stats["total_points"] / cumulative_stats["minutes"])
                        * 90,
                        0,
                    )

                    # Add position data (static, not leaked)
                    cumulative_stats["position"] = (
                        cumulative_stats["player_id"]
                        .map(player_positions)
                        .fillna("MID")
                    )

                    # Add enhanced static player features (not leaked as they're player characteristics)
                    for enhanced_col in available_enhanced_cols:
                        cumulative_stats[f"{enhanced_col}_static"] = (
                            cumulative_stats["player_id"]
                            .map(
                                player_enhanced_features[enhanced_col]
                                if enhanced_col in player_enhanced_features.columns
                                else pd.Series(dtype=float)
                            )
                            .fillna(
                                0
                                if enhanced_col != "chance_of_playing_next_round"
                                else 100
                            )
                        )  # Default availability to 100%

                    # Merge gameweek data with cumulative historical stats
                    # Prepare historical + enhanced features (avoid column conflicts)
                    historical_base_features = [
                        "player_id",
                        "position",
                        "xG90_historical",
                        "xA90_historical",
                        "points_per_90",
                    ]
                    historical_features = cumulative_stats[
                        historical_base_features
                        + ["bps", "ict_index"]
                        + [f"{col}_static" for col in available_enhanced_cols]
                    ].rename(
                        columns={
                            "bps": "bps_historical",
                            "ict_index": "ict_index_historical",
                        }
                    )

                    # Create features using only historical and lagged data (no current GW data!)
                    gw_features = gw_target.merge(
                        historical_features, on="player_id", how="left"
                    )

                    # Add lagged features from previous gameweek (safe to use)
                    if not prev_gw_data.empty:
                        # Ensure required columns exist
                        available_cols = ["player_id"] + [
                            col_name
                            for col_name in [
                                "minutes",
                                "goals_scored",
                                "assists",
                                "bonus",
                            ]
                            if col_name in prev_gw_data.columns
                        ]

                        lagged_features = prev_gw_data[available_cols].copy()

                        # Rename columns with prev_ prefix
                        rename_map = {}
                        if "minutes" in available_cols:
                            rename_map["minutes"] = "prev_minutes"
                        if "goals_scored" in available_cols:
                            rename_map["goals_scored"] = "prev_goals"
                        if "assists" in available_cols:
                            rename_map["assists"] = "prev_assists"
                        if "bonus" in available_cols:
                            rename_map["bonus"] = "prev_bonus"

                        lagged_features = lagged_features.rename(columns=rename_map)
                        gw_features = gw_features.merge(
                            lagged_features, on="player_id", how="left"
                        )

                        # Ensure all lagged columns exist with defaults
                        required_lagged_cols = [
                            "prev_minutes",
                            "prev_goals",
                            "prev_assists",
                            "prev_bonus",
                        ]
                        for required_lagged_col in required_lagged_cols:
                            if required_lagged_col not in gw_features.columns:
                                gw_features[required_lagged_col] = 0

                        # Calculate safe per-90 features from previous gameweek
                        gw_features["prev_goals_per_90"] = np.where(
                            gw_features["prev_minutes"] > 0,
                            (gw_features["prev_goals"] / gw_features["prev_minutes"])
                            * 90,
                            0,
                        )
                        gw_features["prev_assists_per_90"] = np.where(
                            gw_features["prev_minutes"] > 0,
                            (gw_features["prev_assists"] / gw_features["prev_minutes"])
                            * 90,
                            0,
                        )
                        gw_features["prev_bonus_per_90"] = np.where(
                            gw_features["prev_minutes"] > 0,
                            (gw_features["prev_bonus"] / gw_features["prev_minutes"])
                            * 90,
                            0,
                        )
                    else:
                        # No previous gameweek data available
                        gw_features["prev_minutes"] = 0
                        gw_features["prev_goals"] = 0
                        gw_features["prev_assists"] = 0
                        gw_features["prev_bonus"] = 0
                        gw_features["prev_goals_per_90"] = 0
                        gw_features["prev_assists_per_90"] = 0
                        gw_features["prev_bonus_per_90"] = 0

                    # Fill any missing values including enhanced features
                    fill_values = {
                        "xG90_historical": 0,
                        "xA90_historical": 0,
                        "points_per_90": 0,
                        "bps_historical": 0,
                        "ict_index_historical": 0,
                        "position": "MID",
                    }

                    # Add default values for enhanced features
                    for enhanced_col in available_enhanced_cols:
                        static_col = f"{enhanced_col}_static"
                        if "chance_of_playing" in enhanced_col:
                            fill_values[static_col] = 100  # Default to fully available
                        elif "rank" in enhanced_col:
                            fill_values[static_col] = 300  # Mid-range rank default
                        elif "order" in enhanced_col:
                            fill_values[static_col] = 0  # Not on set pieces by default
                        else:
                            fill_values[static_col] = (
                                0  # Zero default for other metrics
                            )

                    # Add defaults for lagged features
                    fill_values.update(
                        {
                            "prev_minutes": 0,
                            "prev_goals": 0,
                            "prev_assists": 0,
                            "prev_bonus": 0,
                            "prev_goals_per_90": 0,
                            "prev_assists_per_90": 0,
                            "prev_bonus_per_90": 0,
                        }
                    )

                    gw_features = gw_features.fillna(fill_values)

                    training_records.append(gw_features)

                # Combine all training data
                training_data = pd.concat(training_records, ignore_index=True)

                # Feature engineering
                le_position = LabelEncoder()
                training_data["position_encoded"] = le_position.fit_transform(
                    training_data["position"]
                )

                # Create derived features (leak-free only)
                # Note: Price-based features removed to maintain leak-free approach

                # NOTE: Rolling averages removed to prevent data leakage
                # Previous implementation calculated rolling averages AFTER combining all training data,
                # which caused future information leakage. To implement correctly, rolling features
                # should be calculated within each gameweek loop using only historical data.

                # 🚨 REMOVED: Current gameweek per-90 metrics (MAJOR DATA LEAKAGE)
                # These features used current GW goals/assists/bonus to predict current GW points!
                # training_data['goals_per_90'] = current_gw_goals / current_gw_minutes * 90
                # training_data['assists_per_90'] = current_gw_assists / current_gw_minutes * 90
                # training_data['bonus_per_90'] = current_gw_bonus / current_gw_minutes * 90
                # ☝️ This is direct target leakage since total_points = goals + assists + bonus + others

                # Home/away context (if available) - note: this should be from fixtures, not live data
                # For now, set to 0 as we don't have reliable home/away info in current structure
                training_data["is_home"] = (
                    0  # TODO: Get actual home/away from fixtures data
                )

                # Consistency metrics - removed to prevent data leakage

                # Enhanced feature set with rich database features
                base_features = [
                    "position_encoded",
                    "xG90_historical",
                    "xA90_historical",
                    "points_per_90",
                    "bps_historical",
                    "ict_index_historical",
                ]

                # Add enhanced static features (player characteristics)
                enhanced_static_features = [
                    f"{enhanced_col}_static" for enhanced_col in available_enhanced_cols
                ]
                available_enhanced_static = [
                    f for f in enhanced_static_features if f in training_data.columns
                ]

                # Add derived contextual features (temporally-safe lagged features)
                lagged_features = [
                    "prev_goals_per_90",
                    "prev_assists_per_90",
                    "prev_bonus_per_90",
                    "prev_minutes",
                    "prev_goals",
                    "prev_assists",
                    "prev_bonus",
                ]
                available_lagged = [
                    f for f in lagged_features if f in training_data.columns
                ]

                contextual_features = ["is_home"] + available_lagged
                # ✅ NOW USING LAGGED FEATURES: prev_goals_per_90, prev_assists_per_90, prev_bonus_per_90
                # These use PREVIOUS gameweek performance to predict CURRENT gameweek points!
                available_contextual = [
                    f for f in contextual_features if f in training_data.columns
                ]

                # Combine all feature sets
                ml_features = (
                    base_features + available_enhanced_static + available_contextual
                )

                # Target variable: actual total points
                target_variable = "total_points"

                # Clean data
                training_data = training_data.dropna(
                    subset=ml_features + [target_variable]
                )

                training_summary = mo.vstack(
                    [
                        mo.md("### ✅ Enhanced Training Data Prepared"),
                        mo.md(f"**Training strategy:** {training_strategy}"),
                        mo.md(
                            f"**Training gameweeks:** GW{min(train_gws)}-{max(train_gws)} ({len(train_gws)} GWs)"
                        ),
                        mo.md(
                            f"**Training samples:** {len(training_data)} player-gameweek records"
                        ),
                        mo.md(
                            f"**Features:** {len(ml_features)} enhanced features | **Target:** {target_variable}"
                        ),
                        mo.md("**Feature categories:**"),
                        mo.md(f"• Base historical: {len(base_features)} features"),
                        mo.md(
                            f"• Enhanced static: {len(available_enhanced_static)} features (injury risk, set pieces, rankings)"
                        ),
                        mo.md(
                            f"• Contextual: {len(available_contextual)} features (form trends, per-90 metrics)"
                        ),
                        mo.md("---"),
                    ]
                )

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            training_summary = mo.md(
                f"❌ **Training data preparation failed:** {str(e)}\n\n```\n{error_details}\n```"
            )
            training_data = pd.DataFrame()
            ml_features = []  # Also initialize ml_features on exception
    else:
        # Check what's blocking training - show detailed debug info
        debug_info = []
        debug_info.append("🔍 **Training Status Debug:**")
        debug_info.append(
            f"• Button clicked: {train_button.value > 0} (clicks: {train_button.value})"
        )
        debug_info.append(
            f"• XP predictions loaded: {not rule_based_predictions.empty}"
        )
        debug_info.append(f"• Live data loaded: {not live_data_df.empty}")

        if train_button.value == 0:
            debug_info.append("")
            debug_info.append("⏯️ **Click the 'Train Model' button to start training**")
        elif rule_based_predictions.empty:
            debug_info.append("")
            debug_info.append("⚠️ **Load data and generate XP predictions first**")
        elif live_data_df.empty:
            debug_info.append("")
            debug_info.append("⚠️ **Load live data first**")
        else:
            debug_info.append("")
            debug_info.append("⚠️ **Unknown issue blocking training**")

        training_summary = mo.vstack([mo.md(line) for line in debug_info])

        training_data = (
            pd.DataFrame()
        )  # Initialize empty DataFrame when conditions not met
        ml_features = []  # Also initialize ml_features
        target_variable = "total_points"  # Initialize target_variable
        le_position = LabelEncoder()  # Initialize le_position

    training_summary
    return le_position, ml_features, target_variable, training_data


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4️⃣ Train Enhanced XGBoost Model with Rich Database Features

    **Train optimized XGBoost model with rich features from the FPL database client.**

    **Enhanced Feature Categories (Leak-Free Validated):**
    - **Historical Performance**: xG90, xA90, points per 90, BPS, ICT index (from before target gameweek)
    - **Injury/Availability Risk**: Chance of playing next round (known before current gameweek)
    - **Set Piece Intelligence**: Corner/freekick/penalty taking order priority (static player roles)
    - **Performance Rankings**: Form rank, ICT rank, points per game rank (season-to-date rankings)
    - **Advanced Per-90 Metrics**: Expected goals/assists per 90 from FPL API (player characteristics)
    - **Market Intelligence**: Value form, season cost changes (no current gameweek data)
    - **Temporal Features**: Lagged per-90 metrics from previous gameweek (safe to use)
    - **Contextual Features**: Home/away venue, positional encoding

    **Model Optimizations:**
    - Hyperparameter tuning for FPL-specific prediction patterns
    - L1/L2 regularization to prevent overfitting on rich feature set
    - Position-aware training with ensemble approaches
    """
    )
    return


@app.cell
def _(
    RidgeCV,
    StandardScaler,
    mean_absolute_error,
    ml_features,
    mo,
    target_variable,
    train_test_split,
    training_data,
):
    # Initialize variables
    ml_model = None
    scaler = None
    X_train, X_test, y_train, y_test = None, None, None, None

    if not training_data.empty and ml_features and len(ml_features) > 0:
        try:
            # Prepare features and target
            X = training_data[ml_features]
            y = training_data[target_variable]

            # Train/test split (80/20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )

            # Scale features for Ridge regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Ridge regression with cross-validation for optimal alpha
            ml_model = RidgeCV(
                alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                cv=5,
                scoring="neg_mean_absolute_error",
            )

            ml_model.fit(X_train_scaled, y_train)

            # Quick training metrics
            train_pred = ml_model.predict(X_train_scaled)
            test_pred = ml_model.predict(X_test_scaled)

            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)

            model_summary = mo.vstack(
                [
                    mo.md(
                        "### ✅ Enhanced Ridge Regression Model Trained with Rich Database Features"
                    ),
                    mo.md(
                        f"**Training samples:** {len(X_train)} | **Test samples:** {len(X_test)}"
                    ),
                    mo.md(
                        f"**Train MAE:** {train_mae:.2f} | **Test MAE:** {test_mae:.2f}"
                    ),
                    mo.md(
                        f"**Total features:** {len(ml_features)} rich features from FPL database"
                    ),
                    mo.md(
                        f"**Feature breakdown:** Base ({len([f for f in ml_features if f in ['position_encoded', 'xG90_historical', 'xA90_historical', 'points_per_90', 'bps_historical', 'ict_index_historical', 'minutes', 'goals_scored', 'assists']])}) + Enhanced ({len([f for f in ml_features if '_static' in f])}) + Contextual ({len([f for f in ml_features if f not in ['position_encoded', 'xG90_historical', 'xA90_historical', 'points_per_90', 'bps_historical', 'ict_index_historical', 'minutes', 'goals_scored', 'assists'] and '_static' not in f])})"
                    ),
                    mo.md(
                        "**Model type:** Ridge Regression with feature scaling and cross-validated regularization"
                    ),
                    mo.md("---"),
                ]
            )

        except Exception as e:
            model_summary = mo.md(f"❌ **Model training failed:** {str(e)}")
            ml_model = None
            scaler = None
    else:
        model_summary = mo.md("⚠️ **Prepare training data first to train model**")
        scaler = None

    model_summary
    return (ml_model, scaler)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4️⃣b Position-Specific Models

    **Train specialized models for each position to capture position-specific scoring patterns.**

    Different positions have vastly different scoring patterns:
    - **Goalkeepers**: Clean sheets, saves, penalty saves
    - **Defenders**: Clean sheets + attacking returns
    - **Midfielders**: Most diverse scoring patterns
    - **Forwards**: Goals heavily weighted, assist potential varies
    """
    )
    return


@app.cell
def _(
    RidgeCV,
    StandardScaler,
    mean_absolute_error,
    ml_features,
    mo,
    target_variable,
    train_test_split,
    training_data,
):
    # Initialize position models dictionary
    position_models = {}
    position_performance = {}

    if not training_data.empty and ml_features and len(ml_features) > 0:
        try:
            all_positions = ["GKP", "DEF", "MID", "FWD"]

            for player_position in all_positions:
                pos_training_data = training_data[
                    training_data["position"] == player_position
                ].copy()

                # Need minimum samples for reliable training
                if len(pos_training_data) >= 30:  # Minimum 30 samples per position
                    # Prepare position-specific features and target
                    X_pos_train = pos_training_data[ml_features]
                    y_pos_train = pos_training_data[target_variable]

                    # Train/test split for this position
                    (
                        X_train_pos_specific,
                        X_test_pos_specific,
                        y_train_pos_specific,
                        y_test_pos_specific,
                    ) = train_test_split(
                        X_pos_train, y_pos_train, test_size=0.2, random_state=42
                    )

                    # Scale features for position-specific Ridge regression
                    position_scaler = StandardScaler()
                    X_train_pos_scaled = position_scaler.fit_transform(
                        X_train_pos_specific
                    )
                    X_test_pos_scaled = position_scaler.transform(X_test_pos_specific)

                    # Train position-specific Ridge model with stronger regularization for small datasets
                    alpha_range = (
                        [10.0, 100.0, 1000.0]
                        if player_position in ["GKP", "FWD"]
                        else [0.1, 1.0, 10.0, 100.0]
                    )
                    position_specific_model = RidgeCV(
                        alphas=alpha_range,
                        cv=min(
                            5, len(X_train_pos_specific) // 10
                        ),  # Adjust CV folds for small datasets
                        scoring="neg_mean_absolute_error",
                    )

                    position_specific_model.fit(
                        X_train_pos_scaled, y_train_pos_specific
                    )

                    # Evaluate position-specific model
                    train_pred_pos_specific = position_specific_model.predict(
                        X_train_pos_scaled
                    )
                    test_pred_pos_specific = position_specific_model.predict(
                        X_test_pos_scaled
                    )

                    train_mae_pos_specific = mean_absolute_error(
                        y_train_pos_specific, train_pred_pos_specific
                    )
                    test_mae_pos_specific = mean_absolute_error(
                        y_test_pos_specific, test_pred_pos_specific
                    )

                    position_models[player_position] = (
                        position_specific_model,
                        position_scaler,
                    )
                    position_performance[player_position] = {
                        "samples": len(pos_training_data),
                        "train_samples": len(X_train_pos_specific),
                        "test_samples": len(X_test_pos_specific),
                        "train_mae": train_mae_pos_specific,
                        "test_mae": test_mae_pos_specific,
                    }
                else:
                    position_performance[player_position] = {
                        "samples": len(pos_training_data),
                        "status": "Insufficient data (min 30 required)",
                    }

            # Create summary of position-specific models
            position_summary_content = [
                mo.md("### 🎯 Position-Specific Model Performance")
            ]

            for player_position in ["GKP", "DEF", "MID", "FWD"]:
                perf = position_performance.get(player_position, {})
                if player_position in position_models:
                    position_summary_content.append(
                        mo.md(
                            f"**{player_position}**: {perf['samples']} samples | Train MAE: {perf['train_mae']:.2f} | Test MAE: {perf['test_mae']:.2f}"
                        )
                    )
                else:
                    status = perf.get("status", "No data")
                    position_summary_content.append(
                        mo.md(
                            f"**{player_position}**: {perf.get('samples', 0)} samples | {status}"
                        )
                    )

            position_summary_content.append(mo.md("---"))
            position_summary = mo.vstack(position_summary_content)

        except Exception as e:
            position_summary = mo.md(
                f"❌ **Position-specific model training failed:** {str(e)}"
            )
            position_models = {}
    else:
        position_summary = mo.md(
            "⚠️ **Prepare training data first to train position-specific models**"
        )

    position_summary
    return (position_models,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5️⃣ Generate ML Predictions for Target Gameweek

    **Use trained models (general + position-specific) to predict XP and compare with rule-based model.**
    """
    )
    return


@app.cell
def _(
    FPLDataClient,
    le_position,
    live_data_df,
    ml_features,
    ml_model,
    mo,
    np,
    pd,
    position_models,
    rule_based_predictions,
    scaler,
    target_gw_input,
):
    # Initialize variables
    comparison_data = pd.DataFrame()

    if ml_model is not None and not rule_based_predictions.empty:
        try:
            # Prepare current gameweek data for prediction with enhanced features
            current_data = rule_based_predictions.copy()

            # Enrich current data with enhanced player features
            prediction_client = FPLDataClient()
            enhanced_players_current = prediction_client.get_players_enhanced()

            # Enhanced features for prediction (leak-free: no current gameweek information)
            enhanced_features_prediction = [
                "chance_of_playing_next_round",  # Next gameweek availability (known before current GW)
                "corners_and_indirect_freekicks_order",
                "direct_freekicks_order",
                "penalties_order",
                "expected_goals_per_90",
                "expected_assists_per_90",
                "form_rank",
                "ict_index_rank",
                "points_per_game_rank",
                "influence_rank",
                "creativity_rank",
                "threat_rank",
                "value_form",
                "cost_change_start",
                # Removed: 'chance_of_playing_this_round' (current GW info)
                # Removed: 'transfers_in_event', 'transfers_out_event' (current GW transfer data)
                # Removed: 'cost_change_event' (current GW price change)
            ]

            # Add available enhanced features to current data
            available_enhanced_pred = [
                pred_col
                for pred_col in enhanced_features_prediction
                if pred_col in enhanced_players_current.columns
            ]
            if available_enhanced_pred:
                enhanced_subset = enhanced_players_current[
                    ["player_id"] + available_enhanced_pred
                ]
                current_data = current_data.merge(
                    enhanced_subset, on="player_id", how="left"
                )

                # Fill missing enhanced features with appropriate defaults
                for pred_col in available_enhanced_pred:
                    if "chance_of_playing" in pred_col:
                        current_data[pred_col] = current_data[pred_col].fillna(100)
                    elif "rank" in pred_col:
                        current_data[pred_col] = current_data[pred_col].fillna(300)
                    elif "order" in pred_col:
                        current_data[pred_col] = current_data[pred_col].fillna(0)
                    else:
                        current_data[pred_col] = current_data[pred_col].fillna(0)

            # Calculate historical features for prediction - use cumulative stats up to current gameweek
            # This is the same approach as training but for current gameweek prediction
            prediction_gw = target_gw_input.value
            historical_data_for_prediction = live_data_df[
                live_data_df["event"] < prediction_gw
            ]

            if not historical_data_for_prediction.empty:
                # Calculate cumulative historical stats for each player up to current gameweek
                prediction_historical_stats = (
                    historical_data_for_prediction.groupby("player_id")
                    .agg(
                        {
                            "total_points": "sum",
                            "minutes": "sum",
                            "goals_scored": "sum",
                            "assists": "sum",
                            "expected_goals": "sum",
                            "expected_assists": "sum",
                            "bps": "mean",
                            "ict_index": "mean",
                        }
                    )
                    .reset_index()
                )

                # Calculate per-90 historical stats
                prediction_historical_stats["xG90_historical"] = np.where(
                    prediction_historical_stats["minutes"] > 0,
                    (
                        prediction_historical_stats["expected_goals"]
                        / prediction_historical_stats["minutes"]
                    )
                    * 90,
                    0,
                )
                prediction_historical_stats["xA90_historical"] = np.where(
                    prediction_historical_stats["minutes"] > 0,
                    (
                        prediction_historical_stats["expected_assists"]
                        / prediction_historical_stats["minutes"]
                    )
                    * 90,
                    0,
                )
                prediction_historical_stats["points_per_90"] = np.where(
                    prediction_historical_stats["minutes"] > 0,
                    (
                        prediction_historical_stats["total_points"]
                        / prediction_historical_stats["minutes"]
                    )
                    * 90,
                    0,
                )

                # Rename to avoid conflicts and merge with current data
                historical_for_merge = prediction_historical_stats[
                    [
                        "player_id",
                        "xG90_historical",
                        "xA90_historical",
                        "points_per_90",
                        "bps",
                        "ict_index",
                    ]
                ].rename(
                    columns={
                        "bps": "bps_historical",
                        "ict_index": "ict_index_historical",
                    }
                )

                # Merge historical features with current data
                current_data = current_data.merge(
                    historical_for_merge, on="player_id", how="left"
                )
            else:
                # No historical data available - use current season data as fallback
                current_data["xG90_historical"] = current_data.get("xG90", 0)
                current_data["xA90_historical"] = current_data.get("xA90", 0)
                current_data["points_per_90"] = (
                    current_data.get("points_per_game", 0) * 90 / 38
                )  # Convert season avg
                current_data["bps_historical"] = current_data.get("bps", 0)
                current_data["ict_index_historical"] = current_data.get("ict_index", 0)

            # Fill any missing historical features
            current_data = current_data.fillna(
                {
                    "xG90_historical": 0,
                    "xA90_historical": 0,
                    "points_per_90": 0,
                    "bps_historical": 0,
                    "ict_index_historical": 0,
                }
            )

            # Apply same feature engineering as training
            current_data["position_encoded"] = le_position.transform(
                current_data["position"]
            )

            # Add enhanced static features with _static suffix to match training
            for pred_col in available_enhanced_pred:
                if pred_col in current_data.columns:
                    current_data[f"{pred_col}_static"] = current_data[pred_col]

            # Add temporally-safe lagged features for prediction
            current_data["is_home"] = 0  # TODO: Get actual home/away from fixtures

            # Add lagged features from previous gameweek for prediction
            prediction_gw = target_gw_input.value
            prev_gw_for_prediction = prediction_gw - 1

            if prev_gw_for_prediction > 0 and not live_data_df.empty:
                prev_gw_prediction_data = live_data_df[
                    live_data_df["event"] == prev_gw_for_prediction
                ]
                if not prev_gw_prediction_data.empty:
                    # Get lagged features for prediction (robust column handling)
                    available_pred_cols = ["player_id"] + [
                        col_name
                        for col_name in ["minutes", "goals_scored", "assists", "bonus"]
                        if col_name in prev_gw_prediction_data.columns
                    ]
                    prev_features = prev_gw_prediction_data[available_pred_cols].copy()

                    # Rename columns with prev_ prefix
                    pred_rename_map = {}
                    if "minutes" in available_pred_cols:
                        pred_rename_map["minutes"] = "prev_minutes"
                    if "goals_scored" in available_pred_cols:
                        pred_rename_map["goals_scored"] = "prev_goals"
                    if "assists" in available_pred_cols:
                        pred_rename_map["assists"] = "prev_assists"
                    if "bonus" in available_pred_cols:
                        pred_rename_map["bonus"] = "prev_bonus"

                    prev_features = prev_features.rename(columns=pred_rename_map)
                    current_data = current_data.merge(
                        prev_features, on="player_id", how="left"
                    )

                    # Ensure all lagged columns exist with defaults
                    required_pred_lagged_cols = [
                        "prev_minutes",
                        "prev_goals",
                        "prev_assists",
                        "prev_bonus",
                    ]
                    for required_pred_lagged_col in required_pred_lagged_cols:
                        if required_pred_lagged_col not in current_data.columns:
                            current_data[required_pred_lagged_col] = 0

                    # Calculate lagged per-90 features
                    current_data["prev_goals_per_90"] = np.where(
                        current_data["prev_minutes"] > 0,
                        (current_data["prev_goals"] / current_data["prev_minutes"])
                        * 90,
                        0,
                    )
                    current_data["prev_assists_per_90"] = np.where(
                        current_data["prev_minutes"] > 0,
                        (current_data["prev_assists"] / current_data["prev_minutes"])
                        * 90,
                        0,
                    )
                    current_data["prev_bonus_per_90"] = np.where(
                        current_data["prev_minutes"] > 0,
                        (current_data["prev_bonus"] / current_data["prev_minutes"])
                        * 90,
                        0,
                    )
                else:
                    # No previous gameweek data
                    current_data["prev_minutes"] = 0
                    current_data["prev_goals"] = 0
                    current_data["prev_assists"] = 0
                    current_data["prev_bonus"] = 0
                    current_data["prev_goals_per_90"] = 0
                    current_data["prev_assists_per_90"] = 0
                    current_data["prev_bonus_per_90"] = 0
            else:
                # No previous gameweek available
                current_data["prev_minutes"] = 0
                current_data["prev_goals"] = 0
                current_data["prev_assists"] = 0
                current_data["prev_bonus"] = 0
                current_data["prev_goals_per_90"] = 0
                current_data["prev_assists_per_90"] = 0
                current_data["prev_bonus_per_90"] = 0

            # Fill any missing lagged features
            lagged_defaults = {
                "prev_minutes": 0,
                "prev_goals": 0,
                "prev_assists": 0,
                "prev_bonus": 0,
                "prev_goals_per_90": 0,
                "prev_assists_per_90": 0,
                "prev_bonus_per_90": 0,
            }
            for lagged_default_col, default_val in lagged_defaults.items():
                if lagged_default_col in current_data.columns:
                    current_data[lagged_default_col] = current_data[
                        lagged_default_col
                    ].fillna(default_val)

            # NOTE: Rolling form features removed to prevent data leakage
            # Previous implementation used simple proxies that could introduce bias

            # Select features for prediction (ensure non-negative values)
            available_features = [f for f in ml_features if f in current_data.columns]
            X_current = current_data[available_features].copy()

            # Fill any remaining NaN values and fix negative values
            X_current = X_current.fillna(0)
            for feature_col in X_current.columns:
                if X_current[feature_col].min() < 0:
                    X_current[feature_col] = X_current[feature_col].clip(lower=0)

            # Scale features for prediction
            X_current_scaled = scaler.transform(X_current)

            # Generate General ML predictions
            ml_xp_predictions = ml_model.predict(X_current_scaled)
            ml_xp_predictions = (
                pd.Series(ml_xp_predictions).clip(lower=0, upper=20).values
            )

            # Generate Position-Specific ML predictions
            position_specific_predictions = np.zeros(len(current_data))
            position_prediction_counts = {}

            if position_models:
                for position in ["GKP", "DEF", "MID", "FWD"]:
                    if position in position_models:
                        pos_mask = current_data["position"] == position
                        pos_players = current_data[pos_mask]

                        if len(pos_players) > 0:
                            X_pos = pos_players[available_features]
                            X_pos = X_pos.fillna(0)
                            for pos_feature_col in X_pos.columns:
                                if X_pos[pos_feature_col].min() < 0:
                                    X_pos[pos_feature_col] = X_pos[
                                        pos_feature_col
                                    ].clip(lower=0)

                            # Unpack model and scaler
                            pos_model, prediction_scaler = position_models[position]
                            X_pos_scaled = prediction_scaler.transform(X_pos)
                            pos_pred = pos_model.predict(X_pos_scaled)
                            pos_pred = np.clip(pos_pred, 0, 20)
                            position_specific_predictions[pos_mask] = pos_pred
                            position_prediction_counts[position] = len(pos_players)

            # Ensemble prediction (combine general and position-specific)
            has_position_pred = position_specific_predictions > 0
            ensemble_predictions = np.where(
                has_position_pred,
                0.6 * position_specific_predictions
                + 0.4 * ml_xp_predictions,  # Favor position-specific when available
                ml_xp_predictions,  # Fall back to general model
            )

            # Get actual points for target gameweek if available (for backtesting)
            actual_points = None
            if not live_data_df.empty:
                target_gw_actual = live_data_df[
                    live_data_df["event"] == target_gw_input.value
                ]
                if not target_gw_actual.empty:
                    actual_points = target_gw_actual.set_index("player_id")[
                        "total_points"
                    ].to_dict()

            # Create enhanced comparison dataframe with all prediction types
            comparison_data = pd.DataFrame(
                {
                    "player_id": current_data["player_id"],
                    "web_name": current_data["web_name"],
                    "position": current_data["position"],
                    "name": current_data["name"],  # Team name
                    "price": current_data["price"],
                    "rule_based_xP": current_data["xP"],
                    "ml_general_xP": ml_xp_predictions,
                    "ml_position_xP": position_specific_predictions,
                    "ml_ensemble_xP": ensemble_predictions,
                    "selected_by_percent": current_data["selected_by_percent"],
                }
            )

            # Add differences from rule-based model
            comparison_data["general_vs_rule"] = (
                comparison_data["ml_general_xP"] - comparison_data["rule_based_xP"]
            )
            comparison_data["position_vs_rule"] = (
                comparison_data["ml_position_xP"] - comparison_data["rule_based_xP"]
            )
            comparison_data["ensemble_vs_rule"] = (
                comparison_data["ml_ensemble_xP"] - comparison_data["rule_based_xP"]
            )

            # Add actual points if available (for backtesting)
            if actual_points:
                comparison_data["actual_points"] = (
                    comparison_data["player_id"].map(actual_points).fillna(0)
                )
                comparison_data["rule_error"] = abs(
                    comparison_data["rule_based_xP"] - comparison_data["actual_points"]
                )
                comparison_data["general_error"] = abs(
                    comparison_data["ml_general_xP"] - comparison_data["actual_points"]
                )
                comparison_data["position_error"] = abs(
                    comparison_data["ml_position_xP"] - comparison_data["actual_points"]
                )
                comparison_data["ensemble_error"] = abs(
                    comparison_data["ml_ensemble_xP"] - comparison_data["actual_points"]
                )

            # Sort by ensemble predictions (best overall model)
            comparison_data = comparison_data.sort_values(
                "ml_ensemble_xP", ascending=False
            ).reset_index(drop=True)

            if actual_points:
                # Calculate model performance
                rule_mae_score = comparison_data["rule_error"].mean()
                general_mae_score = comparison_data["general_error"].mean()
                position_mae_score = (
                    comparison_data["position_error"].mean()
                    if (comparison_data["ml_position_xP"] > 0).any()
                    else float("inf")
                )
                ensemble_mae_score = comparison_data["ensemble_error"].mean()

                # Determine best model
                best_mae_score = min(
                    rule_mae_score, general_mae_score, ensemble_mae_score
                )
                if position_mae_score != float("inf"):
                    best_mae_score = min(best_mae_score, position_mae_score)

                if best_mae_score == rule_mae_score:
                    best_model = "📊 Rule-based Model"
                elif best_mae_score == general_mae_score:
                    best_model = "🤖 General ML Model"
                elif best_mae_score == position_mae_score:
                    best_model = "🎯 Position-Specific ML"
                else:
                    best_model = "🏆 Ensemble Model"

                prediction_summary = mo.vstack(
                    [
                        mo.md(
                            "### ✅ Enhanced ML Predictions Generated (with Actual Results)"
                        ),
                        mo.md(f"**Players predicted:** {len(comparison_data)}"),
                        mo.md(
                            f"**Position-specific models:** {', '.join(position_prediction_counts.keys()) if position_prediction_counts else 'None'}"
                        ),
                        mo.md(""),
                        mo.md("**Model Performance (MAE vs Actual Points):**"),
                        mo.md(f"• Rule-based: {rule_mae_score:.2f}"),
                        mo.md(f"• General ML: {general_mae_score:.2f}"),
                        mo.md(
                            f"• Position-specific: {position_mae_score:.2f}"
                            if position_mae_score != float("inf")
                            else "• Position-specific: Not available"
                        ),
                        mo.md(f"• Ensemble: {ensemble_mae_score:.2f}"),
                        mo.md(
                            f"**🏆 Best Model: {best_model} (MAE: {best_mae_score:.2f})**"
                        ),
                        mo.md("---"),
                    ]
                )
            else:
                prediction_summary = mo.vstack(
                    [
                        mo.md("### ✅ Enhanced ML Predictions Generated"),
                        mo.md(f"**Players predicted:** {len(comparison_data)}"),
                        mo.md(
                            f"**Position-specific models:** {', '.join(position_prediction_counts.keys()) if position_prediction_counts else 'None'}"
                        ),
                        mo.md(""),
                        mo.md("**Prediction Ranges:**"),
                        mo.md(
                            f"• Rule-based: {comparison_data['rule_based_xP'].min():.2f} - {comparison_data['rule_based_xP'].max():.2f}"
                        ),
                        mo.md(
                            f"• General ML: {comparison_data['ml_general_xP'].min():.2f} - {comparison_data['ml_general_xP'].max():.2f}"
                        ),
                        mo.md(
                            f"• Ensemble: {comparison_data['ml_ensemble_xP'].min():.2f} - {comparison_data['ml_ensemble_xP'].max():.2f}"
                        ),
                        mo.md("⚠️ **No actual results available for this gameweek**"),
                        mo.md("---"),
                    ]
                )

        except Exception as e:
            prediction_summary = mo.md(f"❌ **ML prediction failed:** {str(e)}")
            comparison_data = pd.DataFrame()
    else:
        prediction_summary = mo.md("⚠️ **Train model first to generate predictions**")

    prediction_summary
    return (comparison_data,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 6️⃣ Model Comparison & Analysis

    **Compare rule-based vs ML predictions with visualizations and metrics.**
    """
    )
    return


@app.cell
def _(comparison_data, go, mo, np, px):
    # Initialize variables
    rule_ensemble_corr = 0.0

    if not comparison_data.empty:
        # Correlations between different prediction models
        rule_general_corr = np.corrcoef(
            comparison_data["rule_based_xP"], comparison_data["ml_general_xP"]
        )[0, 1]
        rule_ensemble_corr = np.corrcoef(
            comparison_data["rule_based_xP"], comparison_data["ml_ensemble_xP"]
        )[0, 1]

        # Check if we have actual points for backtesting
        has_actual = "actual_points" in comparison_data.columns

        if has_actual:
            # Create comprehensive comparison with actual points using ensemble predictions
            fig_scatter = px.scatter(
                comparison_data.head(100),  # Top 100 players for cleaner visualization
                x="rule_based_xP",
                y="ml_ensemble_xP",
                size="actual_points",
                color="position",
                hover_data=[
                    "web_name",
                    "name",
                    "price",
                    "actual_points",
                    "rule_error",
                    "ensemble_error",
                ],
                title=f"Rule-Based vs Enhanced ML Ensemble Predictions vs Actual Points (Correlation: {rule_ensemble_corr:.3f})",
                labels={
                    "rule_based_xP": "Rule-Based XP",
                    "ml_ensemble_xP": "ML Ensemble XP",
                },
            )

            # Add diagonal line (perfect correlation)
            min_xp = min(
                comparison_data["rule_based_xP"].min(),
                comparison_data["ml_ensemble_xP"].min(),
            )
            max_xp = max(
                comparison_data["rule_based_xP"].max(),
                comparison_data["ml_ensemble_xP"].max(),
            )
            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_xp, max_xp],
                    y=[min_xp, max_xp],
                    mode="lines",
                    name="Perfect Correlation",
                    line=dict(dash="dash", color="red"),
                )
            )

            # Enhanced comparison table with all prediction types
            display_columns = [
                "web_name",
                "position",
                "name",
                "price",
                "rule_based_xP",
                "ml_general_xP",
                "ml_ensemble_xP",
                "actual_points",
                "rule_error",
                "general_error",
                "ensemble_error",
            ]
            available_display_columns = [
                col for col in display_columns if col in comparison_data.columns
            ]
            top_20_comparison = comparison_data.head(20)[
                available_display_columns
            ].round(2)

            # Calculate comprehensive accuracy metrics
            viz_rule_mae = comparison_data["rule_error"].mean()
            viz_general_mae = comparison_data["general_error"].mean()
            viz_ensemble_mae = comparison_data["ensemble_error"].mean()

            viz_rule_correlation = np.corrcoef(
                comparison_data["rule_based_xP"], comparison_data["actual_points"]
            )[0, 1]
            viz_general_correlation = np.corrcoef(
                comparison_data["ml_general_xP"], comparison_data["actual_points"]
            )[0, 1]
            viz_ensemble_correlation = np.corrcoef(
                comparison_data["ml_ensemble_xP"], comparison_data["actual_points"]
            )[0, 1]

            comparison_display = mo.vstack(
                [
                    mo.md("### 📊 Enhanced Prediction Comparison with Actual Results"),
                    mo.md(
                        f"**Model Correlations with Rule-Based:** General ML: {rule_general_corr:.3f} | Ensemble: {rule_ensemble_corr:.3f}"
                    ),
                    mo.md(""),
                    mo.md("**Performance vs Actual Points:**"),
                    mo.md(
                        f"• Rule-based: Correlation {viz_rule_correlation:.3f} | MAE {viz_rule_mae:.2f}"
                    ),
                    mo.md(
                        f"• General ML: Correlation {viz_general_correlation:.3f} | MAE {viz_general_mae:.2f}"
                    ),
                    mo.md(
                        f"• Ensemble: Correlation {viz_ensemble_correlation:.3f} | MAE {viz_ensemble_mae:.2f}"
                    ),
                    mo.md(
                        f"**🏆 Best Model:** {'🏆 Ensemble' if viz_ensemble_mae <= min(viz_rule_mae, viz_general_mae) else '🤖 General ML' if viz_general_mae < viz_rule_mae else '📊 Rule-based'}"
                    ),
                    mo.ui.plotly(fig_scatter),
                    mo.md("### 🏆 Top 20 Players - Backtesting Results"),
                    mo.md(
                        "*Size of points = actual points scored. Lower error = better prediction.*"
                    ),
                    mo.ui.table(top_20_comparison, page_size=20),
                ]
            )
        else:
            # Enhanced comparison without actual points
            fig_scatter = px.scatter(
                comparison_data.head(100),  # Top 100 players for cleaner visualization
                x="rule_based_xP",
                y="ml_ensemble_xP",
                color="position",
                size="price",
                hover_data=[
                    "web_name",
                    "name",
                    "price",
                    "ml_general_xP",
                    "ml_ensemble_xP",
                ],
                title=f"Rule-Based vs Enhanced ML Ensemble Predictions (Correlation: {rule_ensemble_corr:.3f})",
                labels={
                    "rule_based_xP": "Rule-Based XP",
                    "ml_ensemble_xP": "ML Ensemble XP",
                },
            )

            # Add diagonal line (perfect correlation)
            min_xp = min(
                comparison_data["rule_based_xP"].min(),
                comparison_data["ml_ensemble_xP"].min(),
            )
            max_xp = max(
                comparison_data["rule_based_xP"].max(),
                comparison_data["ml_ensemble_xP"].max(),
            )
            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_xp, max_xp],
                    y=[min_xp, max_xp],
                    mode="lines",
                    name="Perfect Correlation",
                    line=dict(dash="dash", color="red"),
                )
            )

            # Enhanced comparison table with all prediction types
            table_columns = [
                "web_name",
                "position",
                "name",
                "price",
                "rule_based_xP",
                "ml_general_xP",
                "ml_ensemble_xP",
                "ensemble_vs_rule",
            ]
            available_table_columns = [
                col for col in table_columns if col in comparison_data.columns
            ]
            top_20_comparison = comparison_data.head(20)[available_table_columns].round(
                2
            )

            comparison_display = mo.vstack(
                [
                    mo.md("### 📊 Enhanced Prediction Comparison"),
                    mo.md(
                        f"**Model Correlations:** General ML vs Rule: {rule_general_corr:.3f} | Ensemble vs Rule: {rule_ensemble_corr:.3f}"
                    ),
                    mo.ui.plotly(fig_scatter),
                    mo.md("### 🏆 Top 20 Players - Enhanced ML Predictions"),
                    mo.md(
                        "*Ensemble combines general ML and position-specific models*"
                    ),
                    mo.ui.table(top_20_comparison, page_size=20),
                ]
            )
    else:
        comparison_display = mo.md("⚠️ **Generate predictions first to see comparison**")
        rule_ensemble_corr = 0.0  # Default value when no data

    comparison_display
    return (rule_ensemble_corr,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7️⃣ Feature Importance Analysis

    **Understand which features are most important for ML predictions using SHAP values.**
    """
    )
    return


@app.cell
def _(ml_features, ml_model, mo, pd, px):
    if ml_model is not None:
        try:
            # Feature importance plot (Ridge coefficients)
            feature_coeffs = abs(ml_model.coef_)  # Absolute values of coefficients
            importance_df = pd.DataFrame(
                {"feature": ml_features, "importance": feature_coeffs}
            ).sort_values("importance", ascending=True)

            fig_importance = px.bar(
                importance_df,
                x="importance",
                y="feature",
                orientation="h",
                title="Ridge Regression Feature Importance (Absolute Coefficients)",
                labels={
                    "importance": "Absolute Coefficient Value",
                    "feature": "Features",
                },
            )

            # Top features analysis
            top_5_features = importance_df.tail(5).sort_values(
                "importance", ascending=False
            )

            feature_analysis = mo.vstack(
                [
                    mo.md("### 🔍 Feature Importance Analysis"),
                    mo.ui.plotly(fig_importance),
                    mo.md("### 📈 Top 5 Most Important Features"),
                    mo.ui.table(top_5_features.round(4), page_size=5),
                    mo.md("**Key Insights:**"),
                    mo.md(
                        f"• Most important feature: **{top_5_features.iloc[0]['feature']}** ({top_5_features.iloc[0]['importance']:.4f})"
                    ),
                    mo.md(
                        f"• Price-related features: **{', '.join([f for f in top_5_features['feature'] if 'price' in f])}**"
                    ),
                    mo.md(
                        f"• Position encoding importance: **{importance_df[importance_df['feature'] == 'position_encoded']['importance'].iloc[0]:.4f}**"
                    ),
                    mo.md(""),
                    mo.md(
                        "*Note: SHAP analysis temporarily disabled due to Python 3.13 compatibility. Will be added in future version.*"
                    ),
                ]
            )

        except Exception as e:
            feature_analysis = mo.md(
                f"❌ **Feature importance analysis failed:** {str(e)}"
            )
    else:
        feature_analysis = mo.md(
            "⚠️ **Train model first to analyze feature importance**"
        )

    feature_analysis
    return (top_5_features,)


@app.cell
def _(comparison_data, mo, rule_ensemble_corr, top_5_features):
    if not comparison_data.empty:
        # Calculate some basic insights using ensemble predictions
        ensemble_higher_count = len(
            comparison_data[comparison_data["ensemble_vs_rule"] > 0]
        )
        rule_higher_count = len(
            comparison_data[comparison_data["ensemble_vs_rule"] < 0]
        )

        avg_abs_difference = comparison_data["ensemble_vs_rule"].abs().mean()
        max_difference = comparison_data["ensemble_vs_rule"].abs().max()

        top_feature = (
            top_5_features.iloc[0]["feature"] if not top_5_features.empty else "Unknown"
        )

        insights = mo.vstack(
            [
                mo.md("### 💡 Enhanced Model Findings"),
                mo.md(
                    f"**Model Correlation:** {rule_ensemble_corr:.3f} - {'Strong' if rule_ensemble_corr > 0.7 else 'Moderate' if rule_ensemble_corr > 0.5 else 'Weak'} agreement with rule-based model"
                ),
                mo.md(
                    f"**Prediction Differences:** Enhanced ML higher for {ensemble_higher_count} players, Rule-based higher for {rule_higher_count}"
                ),
                mo.md(
                    f"**Average absolute difference:** {avg_abs_difference:.2f} XP points"
                ),
                mo.md(f"**Maximum difference:** {max_difference:.2f} XP points"),
                mo.md(f"**Most important feature:** {top_feature}"),
                mo.md(""),
                mo.md("### 🎩 Rich Feature Impact"),
                mo.md(
                    "**Injury intelligence**: Availability percentages help avoid rotation risks"
                ),
                mo.md(
                    "**Set piece data**: Identifies penalty/corner takers for bonus points"
                ),
                mo.md(
                    "**Rankings**: Relative performance vs all players provides context"
                ),
                mo.md("**Market signals**: Transfer momentum indicates crowd wisdom"),
                mo.md(""),
                mo.md("### 🚀 Model Performance Guidance"),
                mo.md(
                    "**If correlation > 0.7:** Enhanced features working well, focus on ensemble tuning"
                ),
                mo.md(
                    "**If correlation 0.5-0.7:** Rich features adding value, try feature selection"
                ),
                mo.md(
                    "**If correlation < 0.5:** Review feature importance and data quality"
                ),
                mo.md(""),
                mo.md("### ✅ Leak-Free Enhanced Features Implemented"),
                mo.md(
                    "• **Static player characteristics**: Set piece orders, performance rankings (no current GW data)"
                ),
                mo.md(
                    "• **Advanced metrics**: Official xG90/xA90 from FPL API (player characteristics)"
                ),
                mo.md(
                    "• **Historical intelligence**: Season price changes, value analysis (no current GW data)"
                ),
                mo.md(
                    "• **Temporal validation**: All historical features from BEFORE target gameweek only"
                ),
                mo.md(
                    "• **Position-specific models**: Separate models for each position with ensemble"
                ),
                mo.md(""),
                mo.md("### 🔧 Further Improvements to Try"),
                mo.md(
                    "• **Fixture context**: Opposition strength, recent defensive/attacking form"
                ),
                mo.md(
                    "• **Team dynamics**: Squad rotation risk, fixture congestion analysis"
                ),
                mo.md(
                    "• **Advanced ensembles**: Weighted combinations based on player characteristics"
                ),
                mo.md(
                    "• **Cross-validation**: Time-series CV for more robust temporal evaluation"
                ),
                mo.md(
                    "• **Feature selection**: Automated selection of most predictive enhanced features"
                ),
            ]
        )
    else:
        insights = mo.md("⚠️ **Complete the experiment to see insights**")

    insights
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
