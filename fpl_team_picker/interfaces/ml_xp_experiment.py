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
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import LabelEncoder
    # import shap  # Commented out due to Python 3.13 compatibility issues

    # FPL data pipeline
    from fpl_team_picker.core.data_loader import fetch_fpl_data, get_current_gameweek_info
    from fpl_team_picker.core.xp_model import XPModel
    from client import FPLDataClient
    from fpl_team_picker.config import config
    return (
        FPLDataClient,
        LabelEncoder,
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
        xgb,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # 🤖 ML XP Model Experiment

    **Lightweight machine learning experiment to compare XGBoost predictions against our current rule-based XP model.**

    ## Quick Start Approach
    - Use existing data pipeline and features
    - Simple XGBoost model with default parameters
    - Side-by-side comparison with rule-based predictions
    - Feature importance analysis with SHAP

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
    current_gw = gw_info['current_gameweek']

    # Target gameweek selector (default to 4 for ML experiment)
    target_gw_input = mo.ui.number(
        value=4,  # Default to GW4 for ML experiment
        start=2,  # Need at least GW2 for historical data
        stop=38,
        step=1,
        label="Target Gameweek for ML Experiment"
    )

    mo.vstack([
        mo.md(f"**Current gameweek detected:** GW{current_gw}"),
        mo.md("**Select target gameweek:**"),
        target_gw_input,
        mo.md("*Note: Need GW2+ to have historical data for training*")
    ])
    return (target_gw_input,)


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
            players_df, teams_df, xg_rates_df, fixtures_df, _, live_data_df = fetch_fpl_data(target_gw_input.value)
            ml_data_loaded = True

            data_summary = mo.vstack([
                mo.md(f"### ✅ Data Loaded for GW{target_gw_input.value}"),
                mo.md(f"**Players:** {len(players_df)} | **Teams:** {len(teams_df)} | **Fixtures:** {len(fixtures_df)}"),
                mo.md(f"**Live data:** {len(live_data_df)} gameweek records"),
                mo.md("---")
            ])
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
                debug=False  # Disable debug for cleaner output
            )

            # Generate 1GW predictions (our main comparison point)
            players_1gw = xp_model.calculate_expected_points(
                players_data=players_df,
                teams_data=teams_df,
                xg_rates_data=xg_rates_df,
                fixtures_data=fixtures_df,
                target_gameweek=target_gw_input.value,
                live_data=live_data_df,
                gameweeks_ahead=1
            )

            rule_based_predictions = players_1gw.copy()

            xp_summary = mo.vstack([
                mo.md("### ✅ Rule-Based XP Generated"),
                mo.md(f"**Players with predictions:** {len(rule_based_predictions)}"),
                mo.md(f"**Average XP:** {rule_based_predictions['xP'].mean():.2f}"),
                mo.md(f"**Top XP:** {rule_based_predictions['xP'].max():.2f}"),
                mo.md("---")
            ])

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
):
    # Initialize variables with defaults
    training_data = pd.DataFrame()
    ml_features = []
    target_variable = 'total_points'
    le_position = LabelEncoder()


    if not rule_based_predictions.empty and not live_data_df.empty:
        try:
            # Get training data from historical gameweeks (before target gameweek)
            historical_gws = sorted([gw for gw in live_data_df['event'].unique() if gw < target_gw_input.value])

            if len(historical_gws) < 3:
                training_summary = mo.md("⚠️ **Need at least 3 historical gameweeks for training**")
            else:
                # Use last 5 gameweeks as training data (or all available if less)
                train_gws = historical_gws[-5:] if len(historical_gws) >= 5 else historical_gws

                training_records = []

                # Get position mapping from current players (this is static data, not leaked)
                client = FPLDataClient()
                current_players = client.get_current_players()
                player_positions = current_players[['player_id', 'position']].set_index('player_id')['position'].to_dict()

                for gw in train_gws:
                    gw_data = live_data_df[live_data_df['event'] == gw].copy()

                    # Calculate cumulative stats up to this gameweek only (no data leakage)
                    historical_data_up_to_gw = live_data_df[live_data_df['event'] <= gw]

                    # Calculate cumulative season stats up to this gameweek
                    cumulative_stats = historical_data_up_to_gw.groupby('player_id').agg({
                        'total_points': 'sum',
                        'minutes': 'sum',
                        'goals_scored': 'sum',
                        'assists': 'sum',
                        'expected_goals': 'sum',
                        'expected_assists': 'sum',
                        'bps': 'mean',  # Average BPS per game
                        'ict_index': 'mean'  # Average ICT per game
                    }).reset_index()

                    # Calculate per-90 stats based only on data available up to this gameweek
                    cumulative_stats['xG90_historical'] = np.where(
                        cumulative_stats['minutes'] > 0,
                        (cumulative_stats['expected_goals'] / cumulative_stats['minutes']) * 90,
                        0
                    )
                    cumulative_stats['xA90_historical'] = np.where(
                        cumulative_stats['minutes'] > 0,
                        (cumulative_stats['expected_assists'] / cumulative_stats['minutes']) * 90,
                        0
                    )
                    cumulative_stats['points_per_90'] = np.where(
                        cumulative_stats['minutes'] > 0,
                        (cumulative_stats['total_points'] / cumulative_stats['minutes']) * 90,
                        0
                    )

                    # Add position data (static, not leaked)
                    cumulative_stats['position'] = cumulative_stats['player_id'].map(player_positions).fillna('MID')

                    # Merge gameweek data with cumulative historical stats
                    # Rename historical stats to avoid column conflicts
                    historical_features = cumulative_stats[['player_id', 'position', 'xG90_historical', 'xA90_historical',
                                                          'points_per_90', 'bps', 'ict_index']].rename(columns={
                        'bps': 'bps_historical',
                        'ict_index': 'ict_index_historical'
                    })

                    gw_features = gw_data.merge(historical_features, on='player_id', how='left')

                    # Fill any missing values
                    gw_features = gw_features.fillna({
                        'xG90_historical': 0,
                        'xA90_historical': 0,
                        'points_per_90': 0,
                        'bps_historical': 0,
                        'ict_index_historical': 0,
                        'position': 'MID'
                    })

                    training_records.append(gw_features)

                # Combine all training data
                training_data = pd.concat(training_records, ignore_index=True)

                # Feature engineering
                le_position = LabelEncoder()
                training_data['position_encoded'] = le_position.fit_transform(training_data['position'])

                # Create derived features (leak-free only)
                # Note: Price-based features removed to maintain leak-free approach

                # Enhanced Feature Engineering: Rolling Averages and Form Trends
                training_data = training_data.sort_values(['player_id', 'event']).reset_index(drop=True)

                # Rolling averages for recent form (3-gameweek window)
                training_data['points_last_3'] = training_data.groupby('player_id')['total_points'].rolling(
                    window=3, min_periods=1
                ).mean().reset_index(0, drop=True)

                training_data['minutes_last_3'] = training_data.groupby('player_id')['minutes'].rolling(
                    window=3, min_periods=1
                ).mean().reset_index(0, drop=True)

                # Form trend (rate of change in recent performance)
                training_data['form_trend'] = training_data.groupby('player_id')['total_points'].rolling(
                    window=3, min_periods=2
                ).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0).reset_index(0, drop=True)

                # Position-specific per-90 metrics (handle division by zero)
                training_data['goals_per_90'] = np.where(
                    training_data['minutes'] > 0,
                    (training_data['goals_scored'] / training_data['minutes']) * 90,
                    0
                )
                training_data['assists_per_90'] = np.where(
                    training_data['minutes'] > 0,
                    (training_data['assists'] / training_data['minutes']) * 90,
                    0
                )
                training_data['bonus_per_90'] = np.where(
                    training_data['minutes'] > 0,
                    (training_data['bonus'] / training_data['minutes']) * 90,
                    0
                )

                # Home/away context (if available)
                training_data['is_home'] = training_data.get('was_home', 0).fillna(0)

                # Consistency metrics
                training_data['points_consistency'] = training_data.groupby('player_id')['total_points'].rolling(
                    window=3, min_periods=2
                ).std().reset_index(0, drop=True).fillna(0)

                # Enhanced feature set including new rolling and per-90 features
                ml_features = [
                    # Leak-free historical features only
                    'position_encoded', 'xG90_historical', 'xA90_historical',
                    'points_per_90', 'bps_historical', 'ict_index_historical', 'minutes', 'goals_scored', 'assists'
                ]

                # Target variable: actual total points
                target_variable = 'total_points'

                # Clean data
                training_data = training_data.dropna(subset=ml_features + [target_variable])

                training_summary = mo.vstack([
                    mo.md("### ✅ Training Data Prepared"),
                    mo.md(f"**Training gameweeks:** GW{min(train_gws)}-{max(train_gws)} ({len(train_gws)} GWs)"),
                    mo.md(f"**Training samples:** {len(training_data)} player-gameweek records"),
                    mo.md(f"**Features:** {len(ml_features)} | **Target:** {target_variable}"),
                    mo.md(f"**Features used:** {', '.join(ml_features)}"),
                    mo.md("---")
                ])

        except Exception as e:
            training_summary = mo.md(f"❌ **Training data preparation failed:** {str(e)}")
            training_data = pd.DataFrame()
    else:
        # Debug info to help diagnose the issue
        debug_info = []
        debug_info.append("⚠️ **Training data preparation blocked. Debug info:**")
        debug_info.append(f"• rule_based_predictions empty: {rule_based_predictions.empty if hasattr(rule_based_predictions, 'empty') else 'Not a DataFrame'}")
        debug_info.append(f"• live_data_df empty: {live_data_df.empty if hasattr(live_data_df, 'empty') else 'Not a DataFrame'}")
        debug_info.append(f"• target_gw_input value: {target_gw_input.value if hasattr(target_gw_input, 'value') else 'No value'}")
        debug_info.append("**Next steps:** Ensure target gameweek is selected and XP predictions are generated first.")

        training_summary = mo.vstack([mo.md(line) for line in debug_info])

    training_summary
    return le_position, ml_features, target_variable, training_data


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 4️⃣ Train Enhanced XGBoost Model

    **Train optimized XGBoost model with tuned hyperparameters and enhanced features.**

    **Model Enhancements:**
    - Optimized hyperparameters for better generalization
    - Enhanced feature set with rolling averages and form trends
    - Regularization to prevent overfitting
    - Position-specific performance metrics
    """
    )
    return


@app.cell
def _(
    mean_absolute_error,
    ml_features,
    mo,
    target_variable,
    train_test_split,
    training_data,
    xgb,
):
    # Initialize variables
    ml_model = None
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

            # Train XGBoost with optimized hyperparameters
            ml_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1,  # Use all cores
                # Optimized hyperparameters for better performance
                max_depth=6,           # Prevent overfitting
                learning_rate=0.1,     # More conservative learning
                n_estimators=200,      # More trees for better learning
                subsample=0.8,         # Sample 80% of data per tree
                colsample_bytree=0.8,  # Sample 80% of features per tree
                reg_alpha=0.1,         # L1 regularization
                reg_lambda=1.0,        # L2 regularization
                min_child_weight=3,    # Minimum samples in leaf nodes
                gamma=0.1              # Minimum loss reduction for splits
            )

            ml_model.fit(X_train, y_train)

            # Quick training metrics
            train_pred = ml_model.predict(X_train)
            test_pred = ml_model.predict(X_test)

            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)

            model_summary = mo.vstack([
                mo.md("### ✅ Enhanced XGBoost Model Trained"),
                mo.md(f"**Training samples:** {len(X_train)} | **Test samples:** {len(X_test)}"),
                mo.md(f"**Train MAE:** {train_mae:.2f} | **Test MAE:** {test_mae:.2f}"),
                mo.md(f"**Features:** {len(ml_features)} enhanced features with rolling averages"),
                mo.md("**Model type:** XGBRegressor (optimized hyperparameters with regularization)"),
                mo.md("---")
            ])

        except Exception as e:
            model_summary = mo.md(f"❌ **Model training failed:** {str(e)}")
            ml_model = None
    else:
        model_summary = mo.md("⚠️ **Prepare training data first to train model**")

    model_summary
    return (ml_model,)


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
    mean_absolute_error,
    ml_features,
    mo,
    target_variable,
    train_test_split,
    training_data,
    xgb,
):
    # Initialize position models dictionary
    position_models = {}
    position_performance = {}

    if not training_data.empty and ml_features and len(ml_features) > 0:
        try:
            all_positions = ['GKP', 'DEF', 'MID', 'FWD']

            for player_position in all_positions:
                pos_training_data = training_data[training_data['position'] == player_position].copy()

                # Need minimum samples for reliable training
                if len(pos_training_data) >= 30:  # Minimum 30 samples per position
                    # Prepare position-specific features and target
                    X_pos_train = pos_training_data[ml_features]
                    y_pos_train = pos_training_data[target_variable]

                    # Train/test split for this position
                    X_train_pos_specific, X_test_pos_specific, y_train_pos_specific, y_test_pos_specific = train_test_split(
                        X_pos_train, y_pos_train, test_size=0.2, random_state=42
                    )

                    # Train position-specific XGBoost model
                    position_specific_model = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        random_state=42,
                        n_jobs=-1,
                        # Slightly different params per position
                        max_depth=5 if player_position in ['GKP', 'DEF'] else 6,  # Simpler for defensive positions
                        learning_rate=0.1,
                        n_estimators=150 if player_position == 'GKP' else 200,  # Fewer trees for GKP
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        min_child_weight=2 if player_position in ['MID', 'FWD'] else 3,  # More flexible for attacking positions
                        gamma=0.1
                    )

                    position_specific_model.fit(X_train_pos_specific, y_train_pos_specific)

                    # Evaluate position-specific model
                    train_pred_pos_specific = position_specific_model.predict(X_train_pos_specific)
                    test_pred_pos_specific = position_specific_model.predict(X_test_pos_specific)

                    train_mae_pos_specific = mean_absolute_error(y_train_pos_specific, train_pred_pos_specific)
                    test_mae_pos_specific = mean_absolute_error(y_test_pos_specific, test_pred_pos_specific)

                    position_models[player_position] = position_specific_model
                    position_performance[player_position] = {
                        'samples': len(pos_training_data),
                        'train_samples': len(X_train_pos_specific),
                        'test_samples': len(X_test_pos_specific),
                        'train_mae': train_mae_pos_specific,
                        'test_mae': test_mae_pos_specific
                    }
                else:
                    position_performance[player_position] = {
                        'samples': len(pos_training_data),
                        'status': 'Insufficient data (min 30 required)'
                    }

            # Create summary of position-specific models
            position_summary_content = [mo.md("### 🎯 Position-Specific Model Performance")]

            for player_position in ['GKP', 'DEF', 'MID', 'FWD']:
                perf = position_performance.get(player_position, {})
                if player_position in position_models:
                    position_summary_content.append(
                        mo.md(f"**{player_position}**: {perf['samples']} samples | Train MAE: {perf['train_mae']:.2f} | Test MAE: {perf['test_mae']:.2f}")
                    )
                else:
                    status = perf.get('status', 'No data')
                    position_summary_content.append(
                        mo.md(f"**{player_position}**: {perf.get('samples', 0)} samples | {status}")
                    )

            position_summary_content.append(mo.md("---"))
            position_summary = mo.vstack(position_summary_content)

        except Exception as e:
            position_summary = mo.md(f"❌ **Position-specific model training failed:** {str(e)}")
            position_models = {}
    else:
        position_summary = mo.md("⚠️ **Prepare training data first to train position-specific models**")

    position_summary
    return (position_models, position_performance)


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
    le_position,
    live_data_df,
    ml_features,
    ml_model,
    mo,
    np,
    pd,
    position_models,
    rule_based_predictions,
    target_gw_input,
):
    # Initialize variables
    comparison_data = pd.DataFrame()

    if ml_model is not None and not rule_based_predictions.empty:
        try:
            # Prepare current gameweek data for prediction
            current_data = rule_based_predictions.copy()

            # Calculate historical features for prediction - use cumulative stats up to current gameweek
            # This is the same approach as training but for current gameweek prediction
            prediction_gw = target_gw_input.value
            historical_data_for_prediction = live_data_df[live_data_df['event'] < prediction_gw]

            if not historical_data_for_prediction.empty:
                # Calculate cumulative historical stats for each player up to current gameweek
                prediction_historical_stats = historical_data_for_prediction.groupby('player_id').agg({
                    'total_points': 'sum',
                    'minutes': 'sum',
                    'goals_scored': 'sum',
                    'assists': 'sum',
                    'expected_goals': 'sum',
                    'expected_assists': 'sum',
                    'bps': 'mean',
                    'ict_index': 'mean'
                }).reset_index()

                # Calculate per-90 historical stats
                prediction_historical_stats['xG90_historical'] = np.where(
                    prediction_historical_stats['minutes'] > 0,
                    (prediction_historical_stats['expected_goals'] / prediction_historical_stats['minutes']) * 90,
                    0
                )
                prediction_historical_stats['xA90_historical'] = np.where(
                    prediction_historical_stats['minutes'] > 0,
                    (prediction_historical_stats['expected_assists'] / prediction_historical_stats['minutes']) * 90,
                    0
                )
                prediction_historical_stats['points_per_90'] = np.where(
                    prediction_historical_stats['minutes'] > 0,
                    (prediction_historical_stats['total_points'] / prediction_historical_stats['minutes']) * 90,
                    0
                )

                # Rename to avoid conflicts and merge with current data
                historical_for_merge = prediction_historical_stats[['player_id', 'xG90_historical', 'xA90_historical',
                                                                  'points_per_90', 'bps', 'ict_index']].rename(columns={
                    'bps': 'bps_historical',
                    'ict_index': 'ict_index_historical'
                })

                # Merge historical features with current data
                current_data = current_data.merge(historical_for_merge, on='player_id', how='left')
            else:
                # No historical data available - use current season data as fallback
                current_data['xG90_historical'] = current_data.get('xG90', 0)
                current_data['xA90_historical'] = current_data.get('xA90', 0)
                current_data['points_per_90'] = current_data.get('points_per_game', 0) * 90 / 38  # Convert season avg
                current_data['bps_historical'] = current_data.get('bps', 0)
                current_data['ict_index_historical'] = current_data.get('ict_index', 0)

            # Fill any missing historical features
            current_data = current_data.fillna({
                'xG90_historical': 0,
                'xA90_historical': 0,
                'points_per_90': 0,
                'bps_historical': 0,
                'ict_index_historical': 0
            })

            # Apply same feature engineering as training
            current_data['position_encoded'] = le_position.transform(current_data['position'])

            # Select features for prediction (ensure non-negative values)
            available_features = [f for f in ml_features if f in current_data.columns]
            X_current = current_data[available_features].copy()

            # Fill any remaining NaN values and fix negative values
            X_current = X_current.fillna(0)
            for col in X_current.columns:
                if X_current[col].min() < 0:
                    X_current[col] = X_current[col].clip(lower=0)

            # Generate General ML predictions
            ml_xp_predictions = ml_model.predict(X_current)
            ml_xp_predictions = pd.Series(ml_xp_predictions).clip(lower=0, upper=20).values

            # Generate Position-Specific ML predictions
            position_specific_predictions = np.zeros(len(current_data))
            position_prediction_counts = {}

            if position_models:
                for position in ['GKP', 'DEF', 'MID', 'FWD']:
                    if position in position_models:
                        pos_mask = current_data['position'] == position
                        pos_players = current_data[pos_mask]

                        if len(pos_players) > 0:
                            X_pos = pos_players[available_features]
                            X_pos = X_pos.fillna(0)
                            for col in X_pos.columns:
                                if X_pos[col].min() < 0:
                                    X_pos[col] = X_pos[col].clip(lower=0)

                            pos_pred = position_models[position].predict(X_pos)
                            pos_pred = np.clip(pos_pred, 0, 20)
                            position_specific_predictions[pos_mask] = pos_pred
                            position_prediction_counts[position] = len(pos_players)

            # Ensemble prediction (combine general and position-specific)
            has_position_pred = position_specific_predictions > 0
            ensemble_predictions = np.where(
                has_position_pred,
                0.6 * position_specific_predictions + 0.4 * ml_xp_predictions,  # Favor position-specific when available
                ml_xp_predictions  # Fall back to general model
            )

            # Get actual points for target gameweek if available (for backtesting)
            actual_points = None
            if not live_data_df.empty:
                target_gw_actual = live_data_df[live_data_df['event'] == target_gw_input.value]
                if not target_gw_actual.empty:
                    actual_points = target_gw_actual.set_index('player_id')['total_points'].to_dict()


            # Create enhanced comparison dataframe with all prediction types
            comparison_data = pd.DataFrame({
                'player_id': current_data['player_id'],
                'web_name': current_data['web_name'],
                'position': current_data['position'],
                'name': current_data['name'],  # Team name
                'price': current_data['price'],
                'rule_based_xP': current_data['xP'],
                'ml_general_xP': ml_xp_predictions,
                'ml_position_xP': position_specific_predictions,
                'ml_ensemble_xP': ensemble_predictions,
                'selected_by_percent': current_data['selected_by_percent']
            })

            # Add differences from rule-based model
            comparison_data['general_vs_rule'] = comparison_data['ml_general_xP'] - comparison_data['rule_based_xP']
            comparison_data['position_vs_rule'] = comparison_data['ml_position_xP'] - comparison_data['rule_based_xP']
            comparison_data['ensemble_vs_rule'] = comparison_data['ml_ensemble_xP'] - comparison_data['rule_based_xP']

            # Add actual points if available (for backtesting)
            if actual_points:
                comparison_data['actual_points'] = comparison_data['player_id'].map(actual_points).fillna(0)
                comparison_data['rule_error'] = abs(comparison_data['rule_based_xP'] - comparison_data['actual_points'])
                comparison_data['general_error'] = abs(comparison_data['ml_general_xP'] - comparison_data['actual_points'])
                comparison_data['position_error'] = abs(comparison_data['ml_position_xP'] - comparison_data['actual_points'])
                comparison_data['ensemble_error'] = abs(comparison_data['ml_ensemble_xP'] - comparison_data['actual_points'])

            # Sort by ensemble predictions (best overall model)
            comparison_data = comparison_data.sort_values('ml_ensemble_xP', ascending=False).reset_index(drop=True)

            if actual_points:
                # Calculate model performance
                rule_mae_score = comparison_data['rule_error'].mean()
                general_mae_score = comparison_data['general_error'].mean()
                position_mae_score = comparison_data['position_error'].mean() if (comparison_data['ml_position_xP'] > 0).any() else float('inf')
                ensemble_mae_score = comparison_data['ensemble_error'].mean()

                # Determine best model
                best_mae_score = min(rule_mae_score, general_mae_score, ensemble_mae_score)
                if position_mae_score != float('inf'):
                    best_mae_score = min(best_mae_score, position_mae_score)

                if best_mae_score == rule_mae_score:
                    best_model = "📊 Rule-based Model"
                elif best_mae_score == general_mae_score:
                    best_model = "🤖 General ML Model"
                elif best_mae_score == position_mae_score:
                    best_model = "🎯 Position-Specific ML"
                else:
                    best_model = "🏆 Ensemble Model"

                prediction_summary = mo.vstack([
                    mo.md("### ✅ Enhanced ML Predictions Generated (with Actual Results)"),
                    mo.md(f"**Players predicted:** {len(comparison_data)}"),
                    mo.md(f"**Position-specific models:** {', '.join(position_prediction_counts.keys()) if position_prediction_counts else 'None'}"),
                    mo.md(""),
                    mo.md("**Model Performance (MAE vs Actual Points):**"),
                    mo.md(f"• Rule-based: {rule_mae_score:.2f}"),
                    mo.md(f"• General ML: {general_mae_score:.2f}"),
                    mo.md(f"• Position-specific: {position_mae_score:.2f}" if position_mae_score != float('inf') else "• Position-specific: Not available"),
                    mo.md(f"• Ensemble: {ensemble_mae_score:.2f}"),
                    mo.md(f"**🏆 Best Model: {best_model} (MAE: {best_mae_score:.2f})**"),
                    mo.md("---")
                ])
            else:
                prediction_summary = mo.vstack([
                    mo.md("### ✅ Enhanced ML Predictions Generated"),
                    mo.md(f"**Players predicted:** {len(comparison_data)}"),
                    mo.md(f"**Position-specific models:** {', '.join(position_prediction_counts.keys()) if position_prediction_counts else 'None'}"),
                    mo.md(""),
                    mo.md("**Prediction Ranges:**"),
                    mo.md(f"• Rule-based: {comparison_data['rule_based_xP'].min():.2f} - {comparison_data['rule_based_xP'].max():.2f}"),
                    mo.md(f"• General ML: {comparison_data['ml_general_xP'].min():.2f} - {comparison_data['ml_general_xP'].max():.2f}"),
                    mo.md(f"• Ensemble: {comparison_data['ml_ensemble_xP'].min():.2f} - {comparison_data['ml_ensemble_xP'].max():.2f}"),
                    mo.md("⚠️ **No actual results available for this gameweek**"),
                    mo.md("---")
                ])

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
    if not comparison_data.empty:
        # Correlations between different prediction models
        rule_general_corr = np.corrcoef(comparison_data['rule_based_xP'], comparison_data['ml_general_xP'])[0, 1]
        rule_ensemble_corr = np.corrcoef(comparison_data['rule_based_xP'], comparison_data['ml_ensemble_xP'])[0, 1]

        # Check if we have actual points for backtesting
        has_actual = 'actual_points' in comparison_data.columns

        if has_actual:
            # Create comprehensive comparison with actual points using ensemble predictions
            fig_scatter = px.scatter(
                comparison_data.head(100),  # Top 100 players for cleaner visualization
                x='rule_based_xP',
                y='ml_ensemble_xP',
                size='actual_points',
                color='position',
                hover_data=['web_name', 'name', 'price', 'actual_points', 'rule_error', 'ensemble_error'],
                title=f'Rule-Based vs Enhanced ML Ensemble Predictions vs Actual Points (Correlation: {rule_ensemble_corr:.3f})',
                labels={'rule_based_xP': 'Rule-Based XP', 'ml_ensemble_xP': 'ML Ensemble XP'}
            )

            # Add diagonal line (perfect correlation)
            min_xp = min(comparison_data['rule_based_xP'].min(), comparison_data['ml_ensemble_xP'].min())
            max_xp = max(comparison_data['rule_based_xP'].max(), comparison_data['ml_ensemble_xP'].max())
            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_xp, max_xp],
                    y=[min_xp, max_xp],
                    mode='lines',
                    name='Perfect Correlation',
                    line=dict(dash='dash', color='red')
                )
            )

            # Enhanced comparison table with all prediction types
            display_columns = ['web_name', 'position', 'name', 'price', 'rule_based_xP', 'ml_general_xP', 'ml_ensemble_xP', 'actual_points', 'rule_error', 'general_error', 'ensemble_error']
            available_display_columns = [col for col in display_columns if col in comparison_data.columns]
            top_20_comparison = comparison_data.head(20)[available_display_columns].round(2)

            # Calculate comprehensive accuracy metrics
            viz_rule_mae = comparison_data['rule_error'].mean()
            viz_general_mae = comparison_data['general_error'].mean()
            viz_ensemble_mae = comparison_data['ensemble_error'].mean()

            viz_rule_correlation = np.corrcoef(comparison_data['rule_based_xP'], comparison_data['actual_points'])[0, 1]
            viz_general_correlation = np.corrcoef(comparison_data['ml_general_xP'], comparison_data['actual_points'])[0, 1]
            viz_ensemble_correlation = np.corrcoef(comparison_data['ml_ensemble_xP'], comparison_data['actual_points'])[0, 1]

            comparison_display = mo.vstack([
                mo.md("### 📊 Enhanced Prediction Comparison with Actual Results"),
                mo.md(f"**Model Correlations with Rule-Based:** General ML: {rule_general_corr:.3f} | Ensemble: {rule_ensemble_corr:.3f}"),
                mo.md(""),
                mo.md("**Performance vs Actual Points:**"),
                mo.md(f"• Rule-based: Correlation {viz_rule_correlation:.3f} | MAE {viz_rule_mae:.2f}"),
                mo.md(f"• General ML: Correlation {viz_general_correlation:.3f} | MAE {viz_general_mae:.2f}"),
                mo.md(f"• Ensemble: Correlation {viz_ensemble_correlation:.3f} | MAE {viz_ensemble_mae:.2f}"),
                mo.md(f"**🏆 Best Model:** {'🏆 Ensemble' if viz_ensemble_mae <= min(viz_rule_mae, viz_general_mae) else '🤖 General ML' if viz_general_mae < viz_rule_mae else '📊 Rule-based'}"),
                mo.ui.plotly(fig_scatter),
                mo.md("### 🏆 Top 20 Players - Backtesting Results"),
                mo.md("*Size of points = actual points scored. Lower error = better prediction.*"),
                mo.ui.table(top_20_comparison, page_size=20)
            ])
        else:
            # Enhanced comparison without actual points
            fig_scatter = px.scatter(
                comparison_data.head(100),  # Top 100 players for cleaner visualization
                x='rule_based_xP',
                y='ml_ensemble_xP',
                color='position',
                size='price',
                hover_data=['web_name', 'name', 'price', 'ml_general_xP', 'ml_ensemble_xP'],
                title=f'Rule-Based vs Enhanced ML Ensemble Predictions (Correlation: {rule_ensemble_corr:.3f})',
                labels={'rule_based_xP': 'Rule-Based XP', 'ml_ensemble_xP': 'ML Ensemble XP'}
            )

            # Add diagonal line (perfect correlation)
            min_xp = min(comparison_data['rule_based_xP'].min(), comparison_data['ml_ensemble_xP'].min())
            max_xp = max(comparison_data['rule_based_xP'].max(), comparison_data['ml_ensemble_xP'].max())
            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_xp, max_xp],
                    y=[min_xp, max_xp],
                    mode='lines',
                    name='Perfect Correlation',
                    line=dict(dash='dash', color='red')
                )
            )

            # Enhanced comparison table with all prediction types
            table_columns = ['web_name', 'position', 'name', 'price', 'rule_based_xP', 'ml_general_xP', 'ml_ensemble_xP', 'ensemble_vs_rule']
            available_table_columns = [col for col in table_columns if col in comparison_data.columns]
            top_20_comparison = comparison_data.head(20)[available_table_columns].round(2)

            comparison_display = mo.vstack([
                mo.md("### 📊 Enhanced Prediction Comparison"),
                mo.md(f"**Model Correlations:** General ML vs Rule: {rule_general_corr:.3f} | Ensemble vs Rule: {rule_ensemble_corr:.3f}"),
                mo.ui.plotly(fig_scatter),
                mo.md("### 🏆 Top 20 Players - Enhanced ML Predictions"),
                mo.md("*Ensemble combines general ML and position-specific models*"),
                mo.ui.table(top_20_comparison, page_size=20)
            ])
    else:
        comparison_display = mo.md("⚠️ **Generate predictions first to see comparison**")

    comparison_display
    return


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
            # Feature importance plot (XGBoost built-in)
            feature_importance = ml_model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': ml_features,
                'importance': feature_importance
            }).sort_values('importance', ascending=True)

            fig_importance = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title='XGBoost Feature Importance',
                labels={'importance': 'Feature Importance', 'feature': 'Features'}
            )

            # Top features analysis
            top_5_features = importance_df.tail(5).sort_values('importance', ascending=False)

            feature_analysis = mo.vstack([
                mo.md("### 🔍 Feature Importance Analysis"),
                mo.ui.plotly(fig_importance),
                mo.md("### 📈 Top 5 Most Important Features"),
                mo.ui.table(top_5_features.round(4), page_size=5),
                mo.md("**Key Insights:**"),
                mo.md(f"• Most important feature: **{top_5_features.iloc[0]['feature']}** ({top_5_features.iloc[0]['importance']:.4f})"),
                mo.md(f"• Price-related features: **{', '.join([f for f in top_5_features['feature'] if 'price' in f])}**"),
                mo.md(f"• Position encoding importance: **{importance_df[importance_df['feature'] == 'position_encoded']['importance'].iloc[0]:.4f}**"),
                mo.md(""),
                mo.md("*Note: SHAP analysis temporarily disabled due to Python 3.13 compatibility. Will be added in future version.*")
            ])

        except Exception as e:
            feature_analysis = mo.md(f"❌ **Feature importance analysis failed:** {str(e)}")
    else:
        feature_analysis = mo.md("⚠️ **Train model first to analyze feature importance**")

    feature_analysis
    return (top_5_features,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8️⃣ Next Steps & Insights

    **Summary of findings and recommendations for further development.**
    """
    )
    return


@app.cell
def _(comparison_data, rule_ensemble_corr, mo, top_5_features):
    if not comparison_data.empty:
        # Calculate some basic insights using ensemble predictions
        ensemble_higher_count = len(comparison_data[comparison_data['ensemble_vs_rule'] > 0])
        rule_higher_count = len(comparison_data[comparison_data['ensemble_vs_rule'] < 0])

        avg_abs_difference = comparison_data['ensemble_vs_rule'].abs().mean()
        max_difference = comparison_data['ensemble_vs_rule'].abs().max()

        top_feature = top_5_features.iloc[0]['feature'] if not top_5_features.empty else "Unknown"

        insights = mo.vstack([
            mo.md("### 💡 Key Findings"),
            mo.md(f"**Model Correlation:** {rule_ensemble_corr:.3f} - {'Strong' if rule_ensemble_corr > 0.7 else 'Moderate' if rule_ensemble_corr > 0.5 else 'Weak'} agreement"),
            mo.md(f"**Prediction Differences:** Ensemble higher for {ensemble_higher_count} players, Rule-based higher for {rule_higher_count}"),
            mo.md(f"**Average absolute difference:** {avg_abs_difference:.2f} XP points"),
            mo.md(f"**Maximum difference:** {max_difference:.2f} XP points"),
            mo.md(f"**Most important feature:** {top_feature}"),
            mo.md(""),
            mo.md("### 🚀 Recommended Next Steps"),
            mo.md("**If correlation > 0.7:** ML shows promise, try hyperparameter tuning"),
            mo.md("**If correlation 0.5-0.7:** Add more features (historical averages, recent form)"),
            mo.md("**If correlation < 0.5:** Review feature engineering and data quality"),
            mo.md(""),
            mo.md("### 🔧 Quick Improvements to Try"),
            mo.md("• Add rolling averages (3, 5, 10 GW) for points, minutes, goals"),
            mo.md("• Include fixture congestion and recent form trends"),
            mo.md("• Try position-specific models (separate for GKP, DEF, MID, FWD)"),
            mo.md("• Experiment with ensemble of rule-based + ML predictions"),
            mo.md("• Add cross-validation for more robust evaluation")
        ])
    else:
        insights = mo.md("⚠️ **Complete the experiment to see insights**")

    insights
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
