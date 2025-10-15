"""
Scikit-learn compatible feature engineering transformers for ML xP models.

This module provides production-ready sklearn transformers that wrap the complex
feature engineering logic from the ML notebook into reusable pipeline components.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Optional


class FPLFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer for FPL feature engineering.

    Handles all temporal feature creation:
    - Cumulative season statistics (up to GW N-1)
    - Rolling 5GW form features
    - Per-90 efficiency metrics
    - Team context features
    - Fixture-specific features

    All features are leak-free (only use past data).
    """

    def __init__(
        self,
        fixtures_df: Optional[pd.DataFrame] = None,
        teams_df: Optional[pd.DataFrame] = None,
        team_strength: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize feature engineer.

        Args:
            fixtures_df: Fixture data with [event, home_team_id, away_team_id]
            teams_df: Teams data with [team_id, name]
            team_strength: Dict mapping team name to strength rating [0.65, 1.30]
        """
        self.fixtures_df = fixtures_df if fixtures_df is not None else pd.DataFrame()
        self.teams_df = teams_df if teams_df is not None else pd.DataFrame()
        self.team_strength = team_strength if team_strength is not None else {}

        # Store feature names for reference
        self.feature_names_ = None

    def fit(self, X, y=None):
        """
        Fit the transformer (no-op for feature engineering).

        Args:
            X: Input DataFrame with player performance data
            y: Target variable (unused)

        Returns:
            self
        """
        # Feature engineering is stateless - no fitting required
        # We just store the feature names after first transform
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw player data into engineered features.

        Args:
            X: DataFrame with columns [player_id, gameweek, position, team_id,
               minutes, goals_scored, assists, total_points, etc.]

        Returns:
            DataFrame with 63 engineered features per player-gameweek
        """
        if X.empty:
            return X

        df = X.copy()

        # Sort by player and gameweek to ensure temporal ordering
        df = df.sort_values(["player_id", "gameweek"]).reset_index(drop=True)

        # Validate required columns
        required_cols = [
            "player_id",
            "gameweek",
            "position",
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
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert numeric columns
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

        # CRITICAL: Sort by player_id and gameweek BEFORE calculating temporal features
        # This ensures shift(1) gets the previous gameweek, not a random row
        df = df.sort_values(["player_id", "gameweek"]).reset_index(drop=True)

        # Group by player for temporal operations
        grouped = df.groupby("player_id", group_keys=False)

        # ===== STATIC FEATURES =====
        df["price"] = df["value"] / 10.0
        position_map = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
        df["position_encoded"] = df["position"].map(position_map).fillna(-1).astype(int)
        df["games_played"] = grouped.cumcount()

        # ===== CUMULATIVE SEASON STATISTICS =====
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
        df["cumulative_yellow_cards"] = (
            grouped["yellow_cards"].transform(lambda x: x.shift(1).fillna(0).cumsum())
            if "yellow_cards" in df.columns
            else 0
        )
        df["cumulative_red_cards"] = (
            grouped["red_cards"].transform(lambda x: x.shift(1).fillna(0).cumsum())
            if "red_cards" in df.columns
            else 0
        )

        # ===== CUMULATIVE PER-90 RATES =====
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

        # ===== ROLLING 5GW FORM FEATURES =====
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

        # ===== ROLLING 5GW PER-90 RATES =====
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

        # ===== DEFENSIVE METRICS =====
        df["rolling_5gw_goals_conceded"] = (
            grouped["goals_conceded"].shift(1).rolling(window=5, min_periods=1).mean()
            if "goals_conceded" in df.columns
            else 0
        )
        df["rolling_5gw_saves"] = (
            grouped["saves"].shift(1).rolling(window=5, min_periods=1).mean()
            if "saves" in df.columns
            else 0
        )
        df["rolling_5gw_xgi"] = (
            grouped["expected_goal_involvements"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            if "expected_goal_involvements" in df.columns
            else 0
        )
        df["rolling_5gw_xgc"] = (
            grouped["expected_goals_conceded"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            if "expected_goals_conceded" in df.columns
            else 0
        )

        # ===== CONSISTENCY & VOLATILITY =====
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

        rolling_5gw_games = (
            grouped["gameweek"].shift(1).rolling(window=5, min_periods=1).count()
        )
        df["minutes_played_rate"] = np.where(
            rolling_5gw_games > 0,
            rolling_5gw_minutes_sum / (rolling_5gw_games * 90),
            0,
        )

        def calc_form_trend(series):
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

        # ===== TEAM-LEVEL FEATURES =====
        df = self._add_team_features(df)

        # ===== FIXTURE-SPECIFIC FEATURES =====
        df = self._add_fixture_features(df)

        # Fill missing values
        df = df.fillna(0)

        # Store feature names on first call
        if self.feature_names_ is None:
            self.feature_names_ = self._get_feature_columns()

        return df[self.feature_names_]

    def _add_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team-level context features."""
        if "team_id" not in df.columns:
            # No team data - add zero columns
            for col in self._get_team_feature_columns():
                df[col] = 0
            return df

        df["team_id"] = (
            pd.to_numeric(df["team_id"], errors="coerce").fillna(-1).astype(int)
        )
        df_sorted = df.sort_values(["team_id", "gameweek"]).reset_index(drop=True)

        # Create team-level aggregates per gameweek
        team_stats = (
            df_sorted.groupby(["team_id", "gameweek"])
            .agg(
                {
                    "goals_scored": "sum",
                    "goals_conceded": "first"
                    if "goals_conceded" in df.columns
                    else lambda x: 0,
                    "clean_sheets": "max",
                    "expected_goals": "sum",
                    "expected_goals_conceded": "first"
                    if "expected_goals_conceded" in df.columns
                    else lambda x: 0,
                    "total_points": "sum",
                }
            )
            .reset_index()
        )

        # Ensure team_stats is sorted by team_id and gameweek before rolling calculations
        team_stats = team_stats.sort_values(["team_id", "gameweek"]).reset_index(
            drop=True
        )

        team_grouped = team_stats.groupby("team_id", group_keys=False)

        # Rolling 5GW team form
        team_stats["team_rolling_5gw_goals_scored"] = (
            team_grouped["goals_scored"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .fillna(0)
        )
        team_stats["team_rolling_5gw_goals_conceded"] = (
            team_grouped["goals_conceded"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .fillna(0)
        )
        team_stats["team_rolling_5gw_xg"] = (
            team_grouped["expected_goals"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .fillna(0)
        )
        team_stats["team_rolling_5gw_xgc"] = (
            team_grouped["expected_goals_conceded"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .fillna(0)
        )
        team_stats["team_rolling_5gw_clean_sheets"] = (
            team_grouped["clean_sheets"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .fillna(0)
        )
        team_stats["team_rolling_5gw_points"] = (
            team_grouped["total_points"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .fillna(0)
        )

        # Cumulative team season stats
        team_stats["team_cumulative_goals_scored"] = (
            team_grouped["goals_scored"].shift(1).fillna(0).cumsum()
        )
        team_stats["team_cumulative_goals_conceded"] = (
            team_grouped["goals_conceded"].shift(1).fillna(0).cumsum()
        )
        team_stats["team_cumulative_clean_sheets"] = (
            team_grouped["clean_sheets"].shift(1).fillna(0).cumsum()
        )
        team_stats["team_season_points"] = (
            team_grouped["total_points"].shift(1).fillna(0).cumsum()
        )

        # Derived team metrics
        team_stats["team_rolling_5gw_goal_diff"] = (
            team_stats["team_rolling_5gw_goals_scored"]
            - team_stats["team_rolling_5gw_goals_conceded"]
        )
        team_stats["team_rolling_5gw_xg_diff"] = (
            team_stats["team_rolling_5gw_xg"] - team_stats["team_rolling_5gw_xgc"]
        )

        # Merge back to player data
        team_feature_cols = [
            "team_id",
            "gameweek",
            "team_rolling_5gw_goals_scored",
            "team_rolling_5gw_goals_conceded",
            "team_rolling_5gw_xg",
            "team_rolling_5gw_xgc",
            "team_rolling_5gw_clean_sheets",
            "team_rolling_5gw_points",
            "team_cumulative_goals_scored",
            "team_cumulative_goals_conceded",
            "team_cumulative_clean_sheets",
            "team_season_points",
            "team_rolling_5gw_goal_diff",
            "team_rolling_5gw_xg_diff",
        ]

        df = df.merge(
            team_stats[team_feature_cols], on=["team_id", "gameweek"], how="left"
        )

        for col in team_feature_cols[2:]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        df["team_encoded"] = df["team_id"].astype("category").cat.codes

        return df

    def _add_fixture_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fixture-specific features (opponent strength, home/away, opponent defense)."""
        if self.fixtures_df.empty or self.teams_df.empty or "team_id" not in df.columns:
            # No fixture data - add zero/neutral columns
            df["is_home"] = 0.5
            df["opponent_strength"] = 1.0
            df["fixture_difficulty"] = 1.0
            df["opponent_rolling_5gw_goals_conceded"] = 0
            df["opponent_rolling_5gw_clean_sheets"] = 0
            df["opponent_rolling_5gw_xgc"] = 0
            return df

        # Data contract: fixtures has [event, home_team_id, away_team_id]
        # Validate fixtures has required columns
        required_fixture_cols = ["event", "home_team_id", "away_team_id"]
        missing_fixture_cols = [
            c for c in required_fixture_cols if c not in self.fixtures_df.columns
        ]
        if missing_fixture_cols:
            raise ValueError(
                f"Fixtures DataFrame missing required columns: {missing_fixture_cols}. "
                f"Available columns: {list(self.fixtures_df.columns)}. "
                f"Data contract violation - fixtures must have [event, home_team_id, away_team_id]"
            )

        # Reset index to avoid duplicate index issues during concat
        # Select only the columns we need to avoid duplicate column issues
        fixtures_clean = self.fixtures_df[
            ["event", "home_team_id", "away_team_id"]
        ].copy()
        fixtures_clean = fixtures_clean.rename(
            columns={"event": "gameweek"}
        ).reset_index(drop=True)

        # Create opponent mapping using vectorized operations (much faster and cleaner)
        # Ensure consistent data types (int) for merge keys
        home_fixtures = pd.DataFrame(
            {
                "gameweek": fixtures_clean["gameweek"].astype(int),
                "team_id": fixtures_clean["home_team_id"].astype(int),
                "opponent_team_id": fixtures_clean["away_team_id"].astype(int),
                "is_home": 1,
            }
        ).reset_index(drop=True)

        away_fixtures = pd.DataFrame(
            {
                "gameweek": fixtures_clean["gameweek"].astype(int),
                "team_id": fixtures_clean["away_team_id"].astype(int),
                "opponent_team_id": fixtures_clean["home_team_id"].astype(int),
                "is_home": 0,
            }
        ).reset_index(drop=True)

        all_team_fixtures = pd.concat([home_fixtures, away_fixtures], ignore_index=True)

        # Merge opponent team names
        all_team_fixtures = all_team_fixtures.merge(
            self.teams_df[["team_id", "name"]],
            left_on="opponent_team_id",
            right_on="team_id",
            how="left",
            suffixes=("", "_opponent"),
        )
        all_team_fixtures = all_team_fixtures.rename(columns={"name": "opponent_name"})
        all_team_fixtures = all_team_fixtures.drop(columns=["team_id_opponent"])

        # Add opponent strength
        all_team_fixtures["opponent_strength"] = (
            all_team_fixtures["opponent_name"].map(self.team_strength).fillna(1.0)
        )

        # Calculate fixture difficulty
        all_team_fixtures["fixture_difficulty"] = np.where(
            all_team_fixtures["is_home"] == 1,
            (2.0 - all_team_fixtures["opponent_strength"]) * 1.1,
            2.0 - all_team_fixtures["opponent_strength"],
        )

        # Merge fixture features to player data
        df = df.merge(
            all_team_fixtures[
                [
                    "gameweek",
                    "team_id",
                    "is_home",
                    "opponent_strength",
                    "fixture_difficulty",
                ]
            ],
            on=["gameweek", "team_id"],
            how="left",
        )

        df["is_home"] = df["is_home"].fillna(0.5)
        df["opponent_strength"] = df["opponent_strength"].fillna(1.0)
        df["fixture_difficulty"] = df["fixture_difficulty"].fillna(1.0)

        # Opponent defensive metrics
        opponent_defensive_stats = (
            df.groupby(["team_id", "gameweek"])
            .agg(
                {
                    "goals_conceded": "first"
                    if "goals_conceded" in df.columns
                    else lambda x: 0,
                    "clean_sheets": "max",
                    "expected_goals_conceded": "first"
                    if "expected_goals_conceded" in df.columns
                    else lambda x: 0,
                }
            )
            .reset_index()
        )

        opponent_defensive_stats = opponent_defensive_stats.sort_values(
            ["team_id", "gameweek"]
        ).reset_index(drop=True)

        opp_grouped = opponent_defensive_stats.groupby("team_id", group_keys=False)

        opponent_defensive_stats["opponent_rolling_5gw_goals_conceded"] = (
            opp_grouped["goals_conceded"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .fillna(0)
        )
        opponent_defensive_stats["opponent_rolling_5gw_clean_sheets"] = (
            opp_grouped["clean_sheets"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .fillna(0)
        )
        opponent_defensive_stats["opponent_rolling_5gw_xgc"] = (
            opp_grouped["expected_goals_conceded"]
            .shift(1)
            .rolling(window=5, min_periods=1)
            .mean()
            .fillna(0)
        )

        # Merge opponent_team_id
        df = df.merge(
            all_team_fixtures[["gameweek", "team_id", "opponent_team_id"]],
            on=["gameweek", "team_id"],
            how="left",
        )

        # Merge opponent defensive metrics
        df = df.merge(
            opponent_defensive_stats[
                [
                    "team_id",
                    "gameweek",
                    "opponent_rolling_5gw_goals_conceded",
                    "opponent_rolling_5gw_clean_sheets",
                    "opponent_rolling_5gw_xgc",
                ]
            ],
            left_on=["opponent_team_id", "gameweek"],
            right_on=["team_id", "gameweek"],
            how="left",
            suffixes=("", "_opp"),
        )

        df = df.drop(columns=["team_id_opp"])
        df["opponent_rolling_5gw_goals_conceded"] = df[
            "opponent_rolling_5gw_goals_conceded"
        ].fillna(0)
        df["opponent_rolling_5gw_clean_sheets"] = df[
            "opponent_rolling_5gw_clean_sheets"
        ].fillna(0)
        df["opponent_rolling_5gw_xgc"] = df["opponent_rolling_5gw_xgc"].fillna(0)

        return df

    def _get_feature_columns(self) -> list:
        """Get list of all feature columns (63 features)."""
        return [
            # Static (3)
            "price",
            "position_encoded",
            "games_played",
            # Cumulative season stats (11)
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
            # Cumulative per-90 rates (7)
            "goals_per_90",
            "assists_per_90",
            "points_per_90",
            "xg_per_90",
            "xa_per_90",
            "bps_per_90",
            "clean_sheet_rate",
            # Rolling 5GW form (13)
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
            # Rolling 5GW per-90 rates (3)
            "rolling_5gw_goals_per_90",
            "rolling_5gw_assists_per_90",
            "rolling_5gw_points_per_90",
            # Defensive metrics (4)
            "rolling_5gw_goals_conceded",
            "rolling_5gw_saves",
            "rolling_5gw_xgi",
            "rolling_5gw_xgc",
            # Consistency & volatility (4)
            "rolling_5gw_points_std",
            "rolling_5gw_minutes_std",
            "minutes_played_rate",
            "form_trend",
            # Team features (14)
            "team_encoded",
            "team_rolling_5gw_goals_scored",
            "team_rolling_5gw_goals_conceded",
            "team_rolling_5gw_xg",
            "team_rolling_5gw_xgc",
            "team_rolling_5gw_clean_sheets",
            "team_rolling_5gw_points",
            "team_cumulative_goals_scored",
            "team_cumulative_goals_conceded",
            "team_cumulative_clean_sheets",
            "team_season_points",
            "team_rolling_5gw_goal_diff",
            "team_rolling_5gw_xg_diff",
            # Fixture features (6)
            "is_home",
            "opponent_strength",
            "fixture_difficulty",
            "opponent_rolling_5gw_goals_conceded",
            "opponent_rolling_5gw_clean_sheets",
            "opponent_rolling_5gw_xgc",
        ]

    def _get_team_feature_columns(self) -> list:
        """Get list of team feature columns."""
        return [
            "team_encoded",
            "team_rolling_5gw_goals_scored",
            "team_rolling_5gw_goals_conceded",
            "team_rolling_5gw_xg",
            "team_rolling_5gw_xgc",
            "team_rolling_5gw_clean_sheets",
            "team_rolling_5gw_points",
            "team_cumulative_goals_scored",
            "team_cumulative_goals_conceded",
            "team_cumulative_clean_sheets",
            "team_season_points",
            "team_rolling_5gw_goal_diff",
            "team_rolling_5gw_xg_diff",
        ]

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for sklearn compatibility."""
        if self.feature_names_ is None:
            return self._get_feature_columns()
        return self.feature_names_
