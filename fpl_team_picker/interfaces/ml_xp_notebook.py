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
- Position-specific model evaluation
- Model comparison against rule-based baseline
- Production-ready model export

Modeling Strategy:
- Use 5 preceding gameweeks (GW N-5 to N-1) to predict GW N performance
- Player-based train/test split: 70% players for training, 30% for testing
- All features properly lagged to prevent data leakage
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
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.preprocessing import StandardScaler

    # FPL services
    from fpl_team_picker.domain.services import (
        DataOrchestrationService,
        ExpectedPointsService,
    )
    from client import FPLDataClient

    # Initialize services
    data_service = DataOrchestrationService()
    xp_service = ExpectedPointsService()
    client = FPLDataClient()
    return (
        GradientBoostingRegressor,
        RandomForestRegressor,
        Ridge,
        StandardScaler,
        client,
        data_service,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        px,
        r2_score,
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
    # Load historical gameweek performance data
    historical_data = []
    data_load_status = []

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
    return (historical_df,)


@app.cell
def _(mo):
    mo.md(r"""## 2️⃣ Feature Engineering - Leak-Free Historical Features""")
    return


@app.cell
def _(historical_df, mo, np, pd):
    # Feature engineering with temporal validation
    def create_features_leak_free(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features without data leakage - only use past data.

        For predicting gameweek N, uses:
        - Cumulative stats up to GW N-1 (season-long totals)
        - Rolling 5GW stats from GW N-6 to N-1 (recent form)
        - Static features (price, position) known before GW N starts
        """
        if df.empty:
            return df

        df = df.copy()

        # Sort by player and gameweek to ensure temporal ordering
        df = df.sort_values(["player_id", "gameweek"]).reset_index(drop=True)

        # Validate required columns exist - fail fast
        required_cols = [
            "player_id",
            "gameweek",
            "position",  # Required for position-specific analysis
            "minutes",
            "goals_scored",
            "assists",
            "total_points",
            "bonus",
            "bps",  # Bonus Points System (will be lagged)
            "clean_sheets",
            "expected_goals",
            "expected_assists",
            "ict_index",
            "influence",
            "creativity",
            "threat",
            "value",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for feature engineering: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        # Convert numeric columns to proper types (handle mixed types from data source)
        numeric_cols = [
            "minutes",
            "goals_scored",
            "assists",
            "total_points",
            "bonus",
            "bps",
            "clean_sheets",
            "expected_goals",
            "expected_assists",
            "ict_index",
            "influence",
            "creativity",
            "threat",
            "value",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Add optional columns if they exist
        optional_cols = [
            "yellow_cards",
            "red_cards",
            "goals_conceded",
            "saves",
            "expected_goal_involvements",
            "expected_goals_conceded",
        ]
        for col in optional_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Group by player for all temporal operations
        grouped = df.groupby("player_id", group_keys=False)

        # ===================================================================
        # STATIC FEATURES (known before gameweek starts)
        # ===================================================================
        df["price"] = df["value"] / 10.0  # Convert to millions

        # Position encoding for ML models
        position_map = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
        df["position_encoded"] = df["position"].map(position_map).fillna(-1).astype(int)

        # Games played (up to previous gameweek only - prevent leakage)
        # cumcount() starts at 0, so it naturally gives us games played BEFORE current GW
        df["games_played"] = grouped.cumcount()

        # ===================================================================
        # CUMULATIVE SEASON STATISTICS (up to GW N-1)
        # ===================================================================
        # Use transform with shift to get proper per-player cumulative sums
        df["cumulative_minutes"] = grouped["minutes"].transform(
            lambda x: x.shift(1).fillna(0).cumsum()
        )
        df["cumulative_goals"] = grouped["goals_scored"].transform(
            lambda x: x.shift(1).fillna(0).cumsum()
        )
        df["cumulative_assists"] = grouped["assists"].transform(
            lambda x: x.shift(1).fillna(0).cumsum()
        )
        df["cumulative_points"] = grouped["total_points"].transform(
            lambda x: x.shift(1).fillna(0).cumsum()
        )
        df["cumulative_bonus"] = grouped["bonus"].transform(
            lambda x: x.shift(1).fillna(0).cumsum()
        )
        df["cumulative_clean_sheets"] = grouped["clean_sheets"].transform(
            lambda x: x.shift(1).fillna(0).cumsum()
        )
        df["cumulative_xg"] = grouped["expected_goals"].transform(
            lambda x: x.shift(1).fillna(0).cumsum()
        )
        df["cumulative_xa"] = grouped["expected_assists"].transform(
            lambda x: x.shift(1).fillna(0).cumsum()
        )
        df["cumulative_bps"] = grouped["bps"].transform(
            lambda x: x.shift(1).fillna(0).cumsum()
        )

        # Optional cumulative stats
        if "yellow_cards" in df.columns:
            df["cumulative_yellow_cards"] = grouped["yellow_cards"].transform(
                lambda x: x.shift(1).fillna(0).cumsum()
            )
        else:
            df["cumulative_yellow_cards"] = 0

        if "red_cards" in df.columns:
            df["cumulative_red_cards"] = grouped["red_cards"].transform(
                lambda x: x.shift(1).fillna(0).cumsum()
            )
        else:
            df["cumulative_red_cards"] = 0

        # ===================================================================
        # CUMULATIVE PER-90 RATES (season-long efficiency)
        # ===================================================================
        df["goals_per_90"] = np.where(
            df["cumulative_minutes"] > 0,
            (df["cumulative_goals"] / df["cumulative_minutes"]) * 90,
            0,
        )
        df["assists_per_90"] = np.where(
            df["cumulative_minutes"] > 0,
            (df["cumulative_assists"] / df["cumulative_minutes"]) * 90,
            0,
        )
        df["points_per_90"] = np.where(
            df["cumulative_minutes"] > 0,
            (df["cumulative_points"] / df["cumulative_minutes"]) * 90,
            0,
        )
        df["xg_per_90"] = np.where(
            df["cumulative_minutes"] > 0,
            (df["cumulative_xg"] / df["cumulative_minutes"]) * 90,
            0,
        )
        df["xa_per_90"] = np.where(
            df["cumulative_minutes"] > 0,
            (df["cumulative_xa"] / df["cumulative_minutes"]) * 90,
            0,
        )
        df["bps_per_90"] = np.where(
            df["cumulative_minutes"] > 0,
            (df["cumulative_bps"] / df["cumulative_minutes"]) * 90,
            0,
        )
        df["clean_sheet_rate"] = np.where(
            df["games_played"] > 0,
            df["cumulative_clean_sheets"] / df["games_played"],
            0,
        )

        # ===================================================================
        # ROLLING 5GW FORM FEATURES (GW N-6 to N-1)
        # ===================================================================
        df["rolling_5gw_points"] = (
            grouped["total_points"].shift(1).rolling(window=5, min_periods=1).mean()
        )
        df["rolling_5gw_minutes"] = (
            grouped["minutes"].shift(1).rolling(window=5, min_periods=1).mean()
        )
        df["rolling_5gw_goals"] = (
            grouped["goals_scored"].shift(1).rolling(window=5, min_periods=1).mean()
        )
        df["rolling_5gw_assists"] = (
            grouped["assists"].shift(1).rolling(window=5, min_periods=1).mean()
        )
        df["rolling_5gw_xg"] = (
            grouped["expected_goals"].shift(1).rolling(window=5, min_periods=1).mean()
        )
        df["rolling_5gw_xa"] = (
            grouped["expected_assists"].shift(1).rolling(window=5, min_periods=1).mean()
        )
        df["rolling_5gw_bps"] = (
            grouped["bps"].shift(1).rolling(window=5, min_periods=1).mean()
        )
        df["rolling_5gw_bonus"] = (
            grouped["bonus"].shift(1).rolling(window=5, min_periods=1).mean()
        )
        df["rolling_5gw_clean_sheets"] = (
            grouped["clean_sheets"].shift(1).rolling(window=5, min_periods=1).mean()
        )

        # ICT components (rolling 5GW)
        df["rolling_5gw_ict_index"] = (
            grouped["ict_index"].shift(1).rolling(window=5, min_periods=1).mean()
        )
        df["rolling_5gw_influence"] = (
            grouped["influence"].shift(1).rolling(window=5, min_periods=1).mean()
        )
        df["rolling_5gw_creativity"] = (
            grouped["creativity"].shift(1).rolling(window=5, min_periods=1).mean()
        )
        df["rolling_5gw_threat"] = (
            grouped["threat"].shift(1).rolling(window=5, min_periods=1).mean()
        )

        # ===================================================================
        # ROLLING 5GW PER-90 RATES (recent efficiency)
        # ===================================================================
        # Calculate rolling sums first, then divide by rolling minutes
        rolling_5gw_goals_sum = (
            grouped["goals_scored"].shift(1).rolling(window=5, min_periods=1).sum()
        )
        rolling_5gw_assists_sum = (
            grouped["assists"].shift(1).rolling(window=5, min_periods=1).sum()
        )
        rolling_5gw_points_sum = (
            grouped["total_points"].shift(1).rolling(window=5, min_periods=1).sum()
        )
        rolling_5gw_minutes_sum = (
            grouped["minutes"].shift(1).rolling(window=5, min_periods=1).sum()
        )

        df["rolling_5gw_goals_per_90"] = np.where(
            rolling_5gw_minutes_sum > 0,
            (rolling_5gw_goals_sum / rolling_5gw_minutes_sum) * 90,
            0,
        )
        df["rolling_5gw_assists_per_90"] = np.where(
            rolling_5gw_minutes_sum > 0,
            (rolling_5gw_assists_sum / rolling_5gw_minutes_sum) * 90,
            0,
        )
        df["rolling_5gw_points_per_90"] = np.where(
            rolling_5gw_minutes_sum > 0,
            (rolling_5gw_points_sum / rolling_5gw_minutes_sum) * 90,
            0,
        )

        # ===================================================================
        # DEFENSIVE METRICS (rolling 5GW)
        # ===================================================================
        if "goals_conceded" in df.columns:
            df["rolling_5gw_goals_conceded"] = (
                grouped["goals_conceded"]
                .shift(1)
                .rolling(window=5, min_periods=1)
                .mean()
            )
        else:
            df["rolling_5gw_goals_conceded"] = 0

        if "saves" in df.columns:
            df["rolling_5gw_saves"] = (
                grouped["saves"].shift(1).rolling(window=5, min_periods=1).mean()
            )
        else:
            df["rolling_5gw_saves"] = 0

        if "expected_goal_involvements" in df.columns:
            df["rolling_5gw_xgi"] = (
                grouped["expected_goal_involvements"]
                .shift(1)
                .rolling(window=5, min_periods=1)
                .mean()
            )
        else:
            df["rolling_5gw_xgi"] = 0

        if "expected_goals_conceded" in df.columns:
            df["rolling_5gw_xgc"] = (
                grouped["expected_goals_conceded"]
                .shift(1)
                .rolling(window=5, min_periods=1)
                .mean()
            )
        else:
            df["rolling_5gw_xgc"] = 0

        # ===================================================================
        # CONSISTENCY & VOLATILITY FEATURES (rolling 5GW)
        # ===================================================================
        df["rolling_5gw_points_std"] = (
            grouped["total_points"]
            .shift(1)
            .rolling(window=5, min_periods=2)
            .std()
            .fillna(0)
        )
        df["rolling_5gw_minutes_std"] = (
            grouped["minutes"].shift(1).rolling(window=5, min_periods=2).std().fillna(0)
        )

        # Minutes played rate (% of available minutes in last 5GW)
        # Assuming 90 minutes per game
        rolling_5gw_games = (
            grouped["gameweek"].shift(1).rolling(window=5, min_periods=1).count()
        )
        df["minutes_played_rate"] = np.where(
            rolling_5gw_games > 0,
            rolling_5gw_minutes_sum / (rolling_5gw_games * 90),
            0,
        )

        # Form trend: linear regression slope of points over last 5 GW
        # Simplified as correlation between GW number and points (approximation)
        def calc_form_trend(series):
            """Calculate linear trend coefficient for rolling window"""
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            if series.std() == 0:
                return 0
            correlation = np.corrcoef(x, series)[0, 1]
            return correlation if not np.isnan(correlation) else 0

        df["form_trend"] = (
            grouped["total_points"]
            .shift(1)
            .rolling(window=5, min_periods=2)
            .apply(calc_form_trend, raw=True)
            .fillna(0)
        )

        # ===================================================================
        # FILL MISSING VALUES
        # ===================================================================
        # List all engineered feature columns
        engineered_feature_cols = [
            # Static
            "price",
            "position_encoded",
            "games_played",
            # Cumulative season stats
            "cumulative_minutes",
            "cumulative_goals",
            "cumulative_assists",
            "cumulative_points",
            "cumulative_bonus",
            "cumulative_clean_sheets",
            "cumulative_xg",
            "cumulative_xa",
            "cumulative_bps",
            "cumulative_yellow_cards",
            "cumulative_red_cards",
            # Cumulative per-90 rates
            "goals_per_90",
            "assists_per_90",
            "points_per_90",
            "xg_per_90",
            "xa_per_90",
            "bps_per_90",
            "clean_sheet_rate",
            # Rolling 5GW form
            "rolling_5gw_points",
            "rolling_5gw_minutes",
            "rolling_5gw_goals",
            "rolling_5gw_assists",
            "rolling_5gw_xg",
            "rolling_5gw_xa",
            "rolling_5gw_bps",
            "rolling_5gw_bonus",
            "rolling_5gw_clean_sheets",
            "rolling_5gw_ict_index",
            "rolling_5gw_influence",
            "rolling_5gw_creativity",
            "rolling_5gw_threat",
            # Rolling 5GW per-90 rates
            "rolling_5gw_goals_per_90",
            "rolling_5gw_assists_per_90",
            "rolling_5gw_points_per_90",
            # Defensive metrics
            "rolling_5gw_goals_conceded",
            "rolling_5gw_saves",
            "rolling_5gw_xgi",
            "rolling_5gw_xgc",
            # Consistency & volatility
            "rolling_5gw_points_std",
            "rolling_5gw_minutes_std",
            "minutes_played_rate",
            "form_trend",
        ]
        df[engineered_feature_cols] = df[engineered_feature_cols].fillna(0)

        return df

    if not historical_df.empty:
        features_df = create_features_leak_free(historical_df)

        # Show feature engineering results
        feature_summary = mo.vstack(
            [
                mo.md("### ✅ Features Engineered (5-Gameweek Rolling Windows)"),
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
                mo.md(f"**Total features created:** {len(features_df.columns)}"),
                mo.md(
                    "**Features using 5GW windows:** All rolling features now use 5-gameweek lookback"
                ),
                mo.md("---"),
            ]
        )
    else:
        features_df = pd.DataFrame()
        feature_summary = mo.md("⚠️ **Load data first**")

    feature_summary
    return (features_df,)


@app.cell
def _(mo):
    mo.md(r"""## 3️⃣ Train/Test Split - Player-Based Validation""")
    return


@app.cell
def _(end_gw_input, features_df, mo, np, pd):
    # Split data: 70% players for training, 30% for testing
    # Strategy: Use ALL available gameweeks (GW6+) to maximize training data
    # Player-based split ensures model generalizes to unseen players
    if not features_df.empty and end_gw_input.value:
        # Validate sufficient training history for 5GW rolling features
        # Need at least GW6 to have 5 preceding gameweeks of data
        if end_gw_input.value < 6:
            split_summary = mo.md(
                f"⚠️ **Target gameweek must be GW6 or later for 5GW rolling features.**\n\n"
                f"Current selection: GW{end_gw_input.value}\n\n"
                f"Rolling 5GW features require data from GW N-5 to N-1 to predict GW N."
            )
            X_train = pd.DataFrame()
            y_train = pd.Series()
            X_test = pd.DataFrame()
            y_test = pd.Series()
            feature_cols = []
            train_df = pd.DataFrame()
            test_df = pd.DataFrame()
        else:
            # Use all completed gameweeks from GW6 up to end_gw (inclusive)
            # This provides more training data across multiple gameweeks
            training_gws = features_df[
                (features_df["gameweek"] >= 6)
                & (features_df["gameweek"] <= end_gw_input.value)
            ].copy()

            # Player-based split: 70% train, 30% test (consistent across all gameweeks)
            # Ensures model is tested on unseen players, not unseen time periods
            all_players = training_gws["player_id"].unique()
            np.random.seed(42)  # Reproducibility
            np.random.shuffle(all_players)

            split_idx = int(len(all_players) * 0.7)
            train_players = set(all_players[:split_idx])
            test_players = set(all_players[split_idx:])

            train_df = training_gws[
                training_gws["player_id"].isin(train_players)
            ].copy()
            test_df = training_gws[training_gws["player_id"].isin(test_players)].copy()

            # Define feature columns - EXPLICIT list to prevent leakage
            # Only use engineered features (lagged historical data) and pre-gameweek info
            feature_cols = [
                # Static features (known before gameweek)
                "price",
                "position_encoded",
                "games_played",
                # Cumulative season statistics (up to GW N-1)
                "cumulative_minutes",
                "cumulative_goals",
                "cumulative_assists",
                "cumulative_points",
                "cumulative_bonus",
                "cumulative_clean_sheets",
                "cumulative_xg",
                "cumulative_xa",
                "cumulative_bps",
                # Cumulative per-90 rates (season-long efficiency)
                "goals_per_90",
                "assists_per_90",
                "points_per_90",
                "xg_per_90",
                "xa_per_90",
                "bps_per_90",
                "clean_sheet_rate",
                # Rolling 5GW form features (GW N-6 to N-1)
                "rolling_5gw_points",
                "rolling_5gw_minutes",
                "rolling_5gw_goals",
                "rolling_5gw_assists",
                "rolling_5gw_xg",
                "rolling_5gw_xa",
                "rolling_5gw_bps",
                "rolling_5gw_bonus",
                "rolling_5gw_clean_sheets",
                "rolling_5gw_ict_index",
                "rolling_5gw_influence",
                "rolling_5gw_creativity",
                "rolling_5gw_threat",
                # Rolling 5GW per-90 rates (recent efficiency)
                "rolling_5gw_goals_per_90",
                "rolling_5gw_assists_per_90",
                "rolling_5gw_points_per_90",
                # Defensive metrics (rolling 5GW)
                "rolling_5gw_goals_conceded",
                "rolling_5gw_saves",
                "rolling_5gw_xgi",
                "rolling_5gw_xgc",
                # Consistency & volatility (rolling 5GW)
                "rolling_5gw_points_std",
                "rolling_5gw_minutes_std",
                "minutes_played_rate",
                "form_trend",
            ]

            # Validate all feature columns exist in the data
            missing_features = [
                col for col in feature_cols if col not in train_df.columns
            ]
            if missing_features:
                raise ValueError(
                    f"Missing feature columns: {missing_features}. "
                    f"Available columns: {list(train_df.columns)}"
                )

            # Prepare X and y for training
            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df["total_points"]

            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df["total_points"]

            # Calculate gameweek coverage
            train_gws = sorted(train_df["gameweek"].unique())
            test_gws = sorted(test_df["gameweek"].unique())
            gw_range = (
                f"GW{min(train_gws)}-{max(train_gws)}"
                if len(train_gws) > 1
                else f"GW{train_gws[0]}"
            )

            split_summary = mo.vstack(
                [
                    mo.md("### ✅ Train/Test Split Complete (Player-Based)"),
                    mo.md(f"**Strategy:** 70/30 player split across {gw_range}"),
                    mo.md(
                        f"**Training:** {len(train_players)} players × {len(train_gws)} gameweeks = {len(train_df):,} records"
                    ),
                    mo.md(
                        f"**Testing:** {len(test_players)} players × {len(test_gws)} gameweeks = {len(test_df):,} records"
                    ),
                    mo.md(
                        f"**Features:** {len(feature_cols)} columns (5GW rolling windows)"
                    ),
                    mo.md(""),
                    mo.md("**✅ No Data Leakage:**"),
                    mo.md("- All features use `.shift(1)` to exclude current gameweek"),
                    mo.md("- Cumulative stats: up to GW N-1 only"),
                    mo.md("- Rolling features: GW N-6 to N-1 window"),
                    mo.md(
                        "- Player-based split ensures generalization to unseen players"
                    ),
                    mo.md(""),
                    mo.md("**Feature Categories:**"),
                    mo.md("- Static: 3 features (price, position, games_played)"),
                    mo.md("- Cumulative Season Stats: 9 features"),
                    mo.md("- Cumulative Per-90 Rates: 7 features"),
                    mo.md("- Rolling 5GW Form: 13 features"),
                    mo.md("- Rolling 5GW Per-90 Rates: 3 features"),
                    mo.md("- Defensive Metrics: 4 features"),
                    mo.md("- Consistency & Volatility: 4 features"),
                    mo.md(""),
                    mo.md(f"**Total: {len(feature_cols)} features**"),
                    mo.md("---"),
                ]
            )
    else:
        X_train = pd.DataFrame()
        y_train = pd.Series()
        X_test = pd.DataFrame()
        y_test = pd.Series()
        feature_cols = []
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        split_summary = mo.md("⚠️ **Engineer features first**")

    split_summary
    return X_test, X_train, feature_cols, test_df, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""## 4️⃣ Model Training - Multiple Algorithms""")
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
    RandomForestRegressor,
    Ridge,
    StandardScaler,
    X_test,
    X_train,
    mean_absolute_error,
    mean_squared_error,
    mo,
    model_selector,
    np,
    r2_score,
    train_button,
    y_test,
    y_train,
):
    # Train models
    trained_models = {}
    predictions = {}
    metrics = {}

    if train_button.value and not X_train.empty:
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train selected model(s)
        if model_selector.value == "ridge" or model_selector.value == "ensemble":
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X_train_scaled, y_train)
            ridge_pred = ridge_model.predict(X_test_scaled)
            ridge_pred = np.clip(ridge_pred, 0, None)  # No negative points

            trained_models["ridge"] = ridge_model
            predictions["ridge"] = ridge_pred
            metrics["ridge"] = {
                "MAE": mean_absolute_error(y_test, ridge_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, ridge_pred)),
                "R²": r2_score(y_test, ridge_pred),
            }

        if model_selector.value == "rf" or model_selector.value == "ensemble":
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1,
            )
            rf_model.fit(X_train, y_train)  # RF doesn't need scaling
            rf_pred = rf_model.predict(X_test)
            rf_pred = np.clip(rf_pred, 0, None)

            trained_models["rf"] = rf_model
            predictions["rf"] = rf_pred
            metrics["rf"] = {
                "MAE": mean_absolute_error(y_test, rf_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
                "R²": r2_score(y_test, rf_pred),
            }

        if model_selector.value == "gb" or model_selector.value == "ensemble":
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
            gb_model.fit(X_train, y_train)
            gb_pred = gb_model.predict(X_test)
            gb_pred = np.clip(gb_pred, 0, None)

            trained_models["gb"] = gb_model
            predictions["gb"] = gb_pred
            metrics["gb"] = {
                "MAE": mean_absolute_error(y_test, gb_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, gb_pred)),
                "R²": r2_score(y_test, gb_pred),
            }

        # Ensemble predictions (average of all models)
        if model_selector.value == "ensemble" and len(predictions) > 1:
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            predictions["ensemble"] = ensemble_pred
            metrics["ensemble"] = {
                "MAE": mean_absolute_error(y_test, ensemble_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, ensemble_pred)),
                "R²": r2_score(y_test, ensemble_pred),
            }

        # Display results
        metrics_display = [mo.md("### ✅ Training Complete")]
        for _model_name, _model_metrics in metrics.items():
            metrics_display.extend(
                [
                    mo.md(f"**{_model_name.upper()} Metrics:**"),
                    mo.md(f"- MAE: {_model_metrics['MAE']:.3f} points"),
                    mo.md(f"- RMSE: {_model_metrics['RMSE']:.3f} points"),
                    mo.md(f"- R²: {_model_metrics['R²']:.3f}"),
                    mo.md(""),
                ]
            )

        training_summary = mo.vstack(metrics_display + [mo.md("---")])
    else:
        scaler = None
        training_summary = mo.md("👆 **Click train button after preparing data**")

    training_summary
    return metrics, predictions, scaler, trained_models


@app.cell
def _(mo):
    mo.md(r"""## 5️⃣ Model Evaluation - Predictions vs Actuals""")
    return


@app.cell
def _(metrics, mo, pd, predictions, px, test_df, y_test):
    # Visualize predictions vs actuals
    if predictions and not test_df.empty:
        # Create comparison DataFrame
        comparison_data = []
        for _model_name, _model_predictions in predictions.items():
            temp_df = test_df[["player_id"]].copy()
            temp_df["predicted_points"] = _model_predictions
            temp_df["actual_points"] = y_test.values
            temp_df["model"] = _model_name.upper()
            temp_df["error"] = temp_df["actual_points"] - temp_df["predicted_points"]
            temp_df["abs_error"] = abs(temp_df["error"])
            comparison_data.append(temp_df)

        comparison_df = pd.concat(comparison_data, ignore_index=True)

        # Scatter plot: predicted vs actual
        fig_scatter = px.scatter(
            comparison_df,
            x="predicted_points",
            y="actual_points",
            color="model",
            hover_data=["player_id", "error"],
            title="Predicted vs Actual Points",
            labels={
                "predicted_points": "Predicted Points",
                "actual_points": "Actual Points",
            },
            opacity=0.6,
        )
        fig_scatter.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max(
                comparison_df["actual_points"].max(),
                comparison_df["predicted_points"].max(),
            ),
            y1=max(
                comparison_df["actual_points"].max(),
                comparison_df["predicted_points"].max(),
            ),
            line=dict(color="red", dash="dash"),
        )

        # Error distribution
        fig_error = px.histogram(
            comparison_df,
            x="error",
            color="model",
            title="Prediction Error Distribution",
            labels={"error": "Prediction Error (Actual - Predicted)"},
            barmode="overlay",
            opacity=0.7,
        )

        # Top/bottom performers
        best_model = min(metrics.items(), key=lambda x: x[1]["MAE"])[0]
        model_comp_df = comparison_df[comparison_df["model"] == best_model.upper()]
        top_10_errors = model_comp_df.nlargest(10, "abs_error")[
            ["player_id", "predicted_points", "actual_points", "error", "abs_error"]
        ]

        eval_display = mo.vstack(
            [
                mo.md("### 📊 Model Performance Visualization"),
                mo.ui.plotly(fig_scatter),
                mo.md("---"),
                mo.ui.plotly(fig_error),
                mo.md("---"),
                mo.md(f"### 🎯 Largest Prediction Errors ({best_model.upper()})"),
                mo.ui.table(top_10_errors.round(2), page_size=10),
            ]
        )
    else:
        comparison_df = pd.DataFrame()
        eval_display = mo.md("⚠️ **Train models first**")

    eval_display
    return (best_model,)


@app.cell
def _(mo):
    mo.md(r"""## 6️⃣ Feature Importance - What Drives xP?""")
    return


@app.cell
def _(feature_cols, mo, pd, px, trained_models):
    # Feature importance analysis
    if trained_models and feature_cols:
        importance_data = []

        # Ridge coefficients (absolute values)
        if "ridge" in trained_models:
            ridge_importance = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": abs(trained_models["ridge"].coef_),
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
    ## 7️⃣ Comparison with Rule-Based Model

    **Compare ML predictions against the current rule-based xP model.**
    """
    )
    return


@app.cell
def _(data_service, end_gw_input, mo, test_df, xp_service):
    # Load rule-based predictions for comparison
    if not test_df.empty and end_gw_input.value:
        try:
            # Load gameweek data for rule-based model
            gw_data = data_service.load_gameweek_data(
                target_gameweek=end_gw_input.value, form_window=5
            )

            # Calculate rule-based xP
            rule_based_xp = xp_service.calculate_combined_results(
                gw_data, use_ml_model=False
            )

            # Merge with test data
            # Note: position column should already exist in test_df from feature engineering
            test_with_rule_based = test_df.merge(
                rule_based_xp[["player_id", "xP"]],
                on="player_id",
                how="left",
            )
            test_with_rule_based = test_with_rule_based.rename(
                columns={"xP": "rule_based_xp"}
            )

            # Validate position column exists for position-specific analysis
            if "position" not in test_with_rule_based.columns:
                raise ValueError(
                    f"Position column missing from test data. Available columns: {list(test_with_rule_based.columns)}"
                )

            comparison_status = mo.md(
                f"✅ **Rule-based xP loaded for {len(test_with_rule_based)} players**"
            )
        except Exception as e:
            test_with_rule_based = test_df.copy()
            comparison_status = mo.md(f"❌ **Rule-based comparison failed:** {str(e)}")
    else:
        test_with_rule_based = test_df.copy()
        comparison_status = mo.md("⚠️ **Train ML models first**")

    comparison_status
    return (test_with_rule_based,)


@app.cell
def _(
    best_model,
    mean_absolute_error,
    mean_squared_error,
    metrics,
    mo,
    np,
    predictions,
    px,
    r2_score,
    test_with_rule_based,
):
    # Compare ML vs Rule-Based
    if "rule_based_xp" in test_with_rule_based.columns and predictions:
        # Calculate rule-based metrics
        rule_based_metrics = {
            "MAE": mean_absolute_error(
                test_with_rule_based["total_points"],
                test_with_rule_based["rule_based_xp"],
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    test_with_rule_based["total_points"],
                    test_with_rule_based["rule_based_xp"],
                )
            ),
            "R²": r2_score(
                test_with_rule_based["total_points"],
                test_with_rule_based["rule_based_xp"],
            ),
        }

        # Add rule-based to comparison
        test_comparison_df = test_with_rule_based.copy()
        test_comparison_df["ml_prediction"] = predictions[best_model]

        # Scatter plot comparison
        fig_comparison = px.scatter(
            test_comparison_df,
            x="rule_based_xp",
            y="ml_prediction",
            color="total_points",
            hover_data=["player_id"],
            title="ML vs Rule-Based Predictions (colored by actual points)",
            labels={
                "rule_based_xp": "Rule-Based xP",
                "ml_prediction": f"ML Prediction ({best_model.upper()})",
            },
            color_continuous_scale="Viridis",
        )

        # Add diagonal line
        max_val = max(
            test_comparison_df["rule_based_xp"].max(),
            test_comparison_df["ml_prediction"].max(),
        )
        fig_comparison.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_val,
            y1=max_val,
            line=dict(color="red", dash="dash"),
        )

        # Display comparison
        rule_comparison_display = mo.vstack(
            [
                mo.md("### 🎯 ML vs Rule-Based Comparison"),
                mo.md("**Rule-Based Model Metrics:**"),
                mo.md(f"- MAE: {rule_based_metrics['MAE']:.3f} points"),
                mo.md(f"- RMSE: {rule_based_metrics['RMSE']:.3f} points"),
                mo.md(f"- R²: {rule_based_metrics['R²']:.3f}"),
                mo.md(""),
                mo.md(f"**Best ML Model ({best_model.upper()}) Improvement:**"),
                mo.md(
                    f"- MAE: {((rule_based_metrics['MAE'] - metrics[best_model]['MAE']) / rule_based_metrics['MAE'] * 100):.1f}% better"
                ),
                mo.md(
                    f"- RMSE: {((rule_based_metrics['RMSE'] - metrics[best_model]['RMSE']) / rule_based_metrics['RMSE'] * 100):.1f}% better"
                ),
                mo.md("---"),
                mo.ui.plotly(fig_comparison),
            ]
        )
    else:
        rule_based_metrics = {}
        rule_comparison_display = mo.md("⚠️ **Load rule-based predictions first**")

    rule_comparison_display
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 8️⃣ Position-Specific Analysis

    **Analyze model performance by player position.**
    """
    )
    return


@app.cell
def _(
    best_model,
    mean_absolute_error,
    mo,
    pd,
    predictions,
    px,
    test_with_rule_based,
):
    # Position-specific evaluation
    if (
        not test_with_rule_based.empty
        and predictions
        and "rule_based_xp" in test_with_rule_based.columns
    ):
        position_metrics = []

        for position in ["GKP", "DEF", "MID", "FWD"]:
            pos_data = test_with_rule_based[
                test_with_rule_based["position"] == position
            ]

            if len(pos_data) > 0:
                # Get indices for this position's predictions
                pos_indices = pos_data.index

                # ML metrics
                ml_mae = mean_absolute_error(
                    pos_data["total_points"],
                    [
                        predictions[best_model][i]
                        for i in range(len(predictions[best_model]))
                        if test_with_rule_based.index[i] in pos_indices
                    ],
                )

                # Rule-based metrics
                rb_mae = mean_absolute_error(
                    pos_data["total_points"],
                    pos_data["rule_based_xp"],
                )

                position_metrics.append(
                    {
                        "Position": position,
                        "ML MAE": ml_mae,
                        "Rule-Based MAE": rb_mae,
                        "Improvement %": ((rb_mae - ml_mae) / rb_mae * 100)
                        if rb_mae > 0
                        else 0,
                        "Players": len(pos_data),
                    }
                )

        position_metrics_df = pd.DataFrame(position_metrics)

        # Plot position comparison
        fig_position = px.bar(
            position_metrics_df,
            x="Position",
            y=["ML MAE", "Rule-Based MAE"],
            title="MAE by Position: ML vs Rule-Based",
            labels={"value": "Mean Absolute Error", "variable": "Model"},
            barmode="group",
        )

        position_display = mo.vstack(
            [
                mo.md("### 📊 Position-Specific Performance"),
                mo.ui.plotly(fig_position),
                mo.md("---"),
                mo.ui.table(position_metrics_df.round(2), page_size=4),
            ]
        )
    else:
        position_metrics_df = pd.DataFrame()
        position_display = mo.md(
            "⚠️ **Complete model training and rule-based comparison first**"
        )

    position_display
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 9️⃣ Next Steps & Production Deployment

    **How to integrate the best ML model into gameweek_manager.py:**

    1. **Export Best Model**: Save the trained model and scaler as pickle files
    2. **Update MLExpectedPointsService**: Load the model in the service
    3. **Enable in Config**: Set `use_ml_model=True` in gameweek_manager.py
    4. **Monitor Performance**: Track live predictions vs actual results

    ---
    """
    )
    return


@app.cell
def _(best_model, mo):
    # Model export button
    export_button = mo.ui.run_button(
        label=f"💾 Export {best_model.upper() if 'best_model' in dir() else 'BEST'} Model",
        kind="success",
    )

    export_button
    return (export_button,)


@app.cell
def _(best_model, export_button, mo, scaler, trained_models):
    # Handle model export
    if export_button.value and trained_models:
        import pickle
        from pathlib import Path

        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        # Save model and scaler
        model_path = model_dir / f"ml_xp_{best_model}_model.pkl"
        scaler_path = model_dir / "ml_xp_scaler.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(trained_models[best_model], f)

        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        export_status = mo.md(
            f"""
        ✅ **Model Exported Successfully**

        - Model: `{model_path}`
        - Scaler: `{scaler_path}`

        **Next:** Update `MLExpectedPointsService` to load these files.
        """
        )
    else:
        export_status = mo.md("👆 **Click to export model after training**")

    export_status
    return


if __name__ == "__main__":
    app.run()
