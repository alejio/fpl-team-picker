"""
Scikit-learn compatible feature engineering transformers for ML xP models.

This module provides production-ready sklearn transformers that wrap the complex
feature engineering logic from the ML notebook into reusable pipeline components.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Optional


def calculate_per_gameweek_team_strength(
    start_gw: int,
    end_gw: int,
    teams_df: pd.DataFrame,
) -> Dict[int, Dict[str, float]]:
    """
    Calculate team strength for each gameweek using only prior gameweeks.

    For GW N, uses data from GW1 to GW(N-1) to calculate strength.
    This ensures no data leakage during ML training.

    Args:
        start_gw: First gameweek to calculate strength for (typically 6, first trainable GW)
        end_gw: Last gameweek to calculate strength for
        teams_df: Teams DataFrame with [team_id, name] columns

    Returns:
        Dict mapping gameweek to team strength dict:
        {6: {"Arsenal": 1.2, "Liverpool": 1.25, ...},
         7: {"Arsenal": 1.21, "Liverpool": 1.24, ...},
         ...}

    Example:
        >>> team_strength = calculate_per_gameweek_team_strength(6, 10, teams_df)
        >>> # GW6 predictions use strength calculated from GW1-5
        >>> # GW7 predictions use strength calculated from GW1-6
        >>> feature_engineer = FPLFeatureEngineer(
        ...     fixtures_df=fixtures_df,
        ...     teams_df=teams_df,
        ...     team_strength=team_strength  # Per-gameweek format
        ... )
    """
    from fpl_team_picker.domain.services.team_analytics_service import (
        TeamAnalyticsService,
    )

    team_analytics = TeamAnalyticsService(debug=False)
    per_gw_strength = {}

    for gw in range(start_gw, end_gw + 1):
        # Use data up to (gw - 1) to predict gw
        # For GW6, use GW5 data (which itself uses GW1-5)
        # For GW7, use GW6 data (which itself uses GW1-6)
        target_gw = gw - 1 if gw > 1 else 1
        per_gw_strength[gw] = team_analytics.get_team_strength(
            target_gameweek=target_gw,
            teams_data=teams_df,
            current_season_data=None,
        )

    return per_gw_strength


class FPLFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer for FPL feature engineering.

    Generates 99 features (65 base + 15 enhanced + 4 penalty/set-piece + 15 betting odds):

    Base features (65):
    - Cumulative season statistics (up to GW N-1)
    - Rolling 5GW form features
    - Per-90 efficiency metrics
    - Team context features
    - Fixture-specific features
    - Price band categorization (ordinal)

    Enhanced features (15) - Issue #37:
    - Ownership trends (7): selected_by_percent, ownership_tier, transfer_momentum,
      net_transfers, bandwagon_score, ownership_velocity
    - Value analysis (5): points_per_pound, value_vs_position, predicted_price_change,
      price_volatility, price_risk
    - Enhanced fixture difficulty (3): congestion_difficulty, form_adjusted_difficulty,
      clean_sheet_probability_enhanced

    Penalty & set-piece taker features (4):
    - is_primary_penalty_taker: Primary penalty taker (order=1)
    - is_penalty_taker: Any penalty taker (order>=1, includes backups)
    - is_corner_taker: Primary corner taker (order=1)
    - is_fk_taker: Primary free-kick taker (order=1)

    Betting odds features (15) - Issue #38:
    - Implied probabilities (6): team_win_probability, opponent_win_probability, draw_probability,
      implied_clean_sheet_probability, implied_total_goals, team_expected_goals
    - Market confidence (4): market_consensus_strength, odds_movement_team, odds_movement_magnitude,
      favorite_status
    - Asian Handicap (3): asian_handicap_line, handicap_team_odds, expected_goal_difference
    - Match context (2): over_under_signal, referee_encoded

    All features are leak-free (only use past data).
    """

    def __init__(
        self,
        fixtures_df: Optional[pd.DataFrame] = None,
        teams_df: Optional[pd.DataFrame] = None,
        team_strength: Optional[Dict] = None,
        ownership_trends_df: Optional[pd.DataFrame] = None,
        value_analysis_df: Optional[pd.DataFrame] = None,
        fixture_difficulty_df: Optional[pd.DataFrame] = None,
        raw_players_df: Optional[pd.DataFrame] = None,
        betting_features_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize feature engineer.

        Args:
            fixtures_df: Fixture data with [event, home_team_id, away_team_id]
            teams_df: Teams data with [team_id, name]
            team_strength: Team strength ratings. Supports two formats:
                - Dict[str, float]: Single static dict {team_name: strength} (backward compatible)
                - Dict[int, Dict[str, float]]: Per-gameweek dict {gameweek: {team_name: strength}}
                  Recommended format to avoid data leakage (GW N uses strength from GW 1 to N-1)
            ownership_trends_df: Ownership trends from get_derived_ownership_trends() (NEW - Issue #37)
            value_analysis_df: Value analysis from get_derived_value_analysis() (NEW - Issue #37)
            fixture_difficulty_df: Enhanced fixture difficulty from get_derived_fixture_difficulty() (NEW - Issue #37)
            raw_players_df: Raw FPL players bootstrap data with penalty/set-piece order columns
                [player_id, penalties_order, corners_and_indirect_freekicks_order, direct_freekicks_order]
            betting_features_df: Betting odds features from get_derived_betting_features() (NEW - Issue #38)
                [gameweek, player_id, team_win_probability, ..., referee_encoded]
        """
        self.fixtures_df = fixtures_df if fixtures_df is not None else pd.DataFrame()
        self.teams_df = teams_df if teams_df is not None else pd.DataFrame()
        self.team_strength = team_strength if team_strength is not None else {}

        # Detect team_strength format for efficient lookup
        self._is_per_gameweek_strength = self._detect_strength_format()

        # NEW: Enhanced data sources for differential strategy (Issue #37)
        self.ownership_trends_df = (
            ownership_trends_df if ownership_trends_df is not None else pd.DataFrame()
        )
        self.value_analysis_df = (
            value_analysis_df if value_analysis_df is not None else pd.DataFrame()
        )
        self.fixture_difficulty_df = (
            fixture_difficulty_df
            if fixture_difficulty_df is not None
            else pd.DataFrame()
        )

        # Raw players data for penalty/set-piece taker features
        self.raw_players_df = (
            raw_players_df if raw_players_df is not None else pd.DataFrame()
        )

        # NEW: Betting odds features (Issue #38)
        self.betting_features_df = (
            betting_features_df if betting_features_df is not None else pd.DataFrame()
        )

        # Store feature names for reference
        self.feature_names_ = None

    def _detect_strength_format(self) -> bool:
        """
        Detect if team_strength is per-gameweek format.

        Returns:
            True if team_strength is Dict[int, Dict[str, float]] (per-gameweek)
            False if team_strength is Dict[str, float] (static) or empty
        """
        if not self.team_strength:
            return False

        # Check first key - if it's an int, assume per-gameweek format
        first_key = next(iter(self.team_strength.keys()))
        if isinstance(first_key, int):
            # Validate structure: {int: {str: float}}
            first_value = self.team_strength[first_key]
            if not isinstance(first_value, dict):
                raise ValueError(
                    f"Invalid team_strength format. Per-gameweek format requires "
                    f"Dict[int, Dict[str, float]], got {type(first_value)} for gameweek {first_key}"
                )
            return True
        elif isinstance(first_key, str):
            # Static format: {str: float}
            return False
        else:
            raise ValueError(
                f"Invalid team_strength format. Expected Dict[str, float] or Dict[int, Dict[str, float]], "
                f"got key type {type(first_key)}"
            )

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
            DataFrame with 84 engineered features per player-gameweek
            (65 base + 15 enhanced + 4 penalty/set-piece taker features)
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

        # Price band feature (ordinal: 0=Budget, 1=Mid, 2=Premium, 3=Elite)
        # Bins: [0, 5) = Budget, [5, 7) = Mid, [7, 9) = Premium, [9, inf) = Elite
        df["price_band"] = pd.cut(
            df["price"],
            bins=[0, 5.0, 7.0, 9.0, float("inf")],
            labels=[0, 1, 2, 3],
            right=False,
        ).astype(int)

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

        # ===== ENHANCED FEATURES (Issue #37) =====
        df = self._add_ownership_features(df)
        df = self._add_value_features(df)
        df = self._add_enhanced_fixture_features(df)

        # ===== PENALTY & SET-PIECE TAKER FEATURES =====
        df = self._add_penalty_set_piece_features(df)

        # ===== BETTING ODDS FEATURES (Issue #38) =====
        df = self._add_betting_odds_features(df)

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

        # Add opponent strength (per-gameweek or static)
        if self._is_per_gameweek_strength:
            # Per-gameweek format: {gw: {team: strength}}
            def map_opponent_strength(row):
                gw = int(row["gameweek"])
                opponent = row["opponent_name"]
                return self.team_strength.get(gw, {}).get(opponent, 1.0)

            all_team_fixtures["opponent_strength"] = all_team_fixtures.apply(
                map_opponent_strength, axis=1
            )
        else:
            # Static format: {team: strength} (backward compatible)
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

    def _add_ownership_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ownership and transfer momentum features (Issue #37)."""
        if self.ownership_trends_df.empty:
            raise ValueError(
                "Ownership trends data is required but was not provided. "
                "Pass ownership_trends_df=client.get_derived_ownership_trends() to constructor."
            )

        # Merge ownership data by player_id and gameweek
        ownership_cols = [
            "player_id",
            "gameweek",
            "selected_by_percent",
            "net_transfers_gw",
            "avg_net_transfers_5gw",
            "transfer_momentum",
            "ownership_velocity",
            "ownership_tier",
            "bandwagon_score",
        ]

        # Validate required columns exist
        missing_cols = [
            col
            for col in ownership_cols
            if col not in self.ownership_trends_df.columns
            and col not in ["player_id", "gameweek"]
        ]
        if missing_cols:
            raise ValueError(
                f"Ownership trends data missing required columns: {missing_cols}. "
                f"Available columns: {list(self.ownership_trends_df.columns)}"
            )

        available_cols = [
            col for col in ownership_cols if col in self.ownership_trends_df.columns
        ]
        ownership_df = self.ownership_trends_df[available_cols].copy()

        # Debug: Check merge keys exist in both dataframes
        if "player_id" not in df.columns:
            raise ValueError("player_id column missing from main dataframe")
        if "gameweek" not in df.columns:
            raise ValueError("gameweek column missing from main dataframe")

        # Drop any existing ownership columns to avoid _x _y suffixes during merge
        ownership_feature_cols = [
            col
            for col in ownership_cols
            if col not in ["player_id", "gameweek"] and col in df.columns
        ]
        if ownership_feature_cols:
            df = df.drop(columns=ownership_feature_cols)

        # Shift ownership by 1 gameweek (consistent with .shift(1) on performance features)
        # This means: Use GW7 ownership to predict GW8, GW8 ownership to predict GW9, etc.
        ownership_df_shifted = ownership_df.copy()
        ownership_df_shifted["gameweek"] = ownership_df_shifted["gameweek"] + 1

        df = df.merge(ownership_df_shifted, on=["player_id", "gameweek"], how="left")

        # Debug: Check if merge succeeded
        if "selected_by_percent" not in df.columns:
            raise ValueError(
                f"Merge failed - selected_by_percent not in result. "
                f"Main df had {len(df)} rows with gameweeks {df['gameweek'].unique() if 'gameweek' in df.columns else 'N/A'}. "
                f"Ownership df had {len(ownership_df)} rows with gameweeks {ownership_df['gameweek'].unique() if 'gameweek' in ownership_df.columns else 'N/A'}. "
                f"Result columns: {list(df.columns)}"
            )

        # Validate no missing data (except GW1 and new players)
        if (
            "selected_by_percent" in df.columns
            and df["selected_by_percent"].isna().any()
        ):
            missing_gws = sorted(
                df[df["selected_by_percent"].isna()]["gameweek"].unique()
            )
            max_ownership_gw = (
                ownership_df["gameweek"].max()
                if "gameweek" in ownership_df.columns
                else "unknown"
            )

            # GW1 naturally has no ownership (no GW0), fill with neutral values
            if 1 in missing_gws:
                gw1_mask = (df["gameweek"] == 1) & df["selected_by_percent"].isna()
                df.loc[gw1_mask, "selected_by_percent"] = 5.0  # Median ownership
                df.loc[gw1_mask, "net_transfers_gw"] = 0
                df.loc[gw1_mask, "avg_net_transfers_5gw"] = 0
                df.loc[gw1_mask, "bandwagon_score"] = 0
                df.loc[gw1_mask, "ownership_velocity"] = 0
                missing_gws = [gw for gw in missing_gws if gw != 1]

            # Check if there are still missing gameweeks (after handling GW1)
            # Note: With dataset-builder backfill, this should never happen
            if missing_gws:
                raise ValueError(
                    f"Missing ownership data for gameweeks {missing_gws}. "
                    f"Ownership trends data available through GW{max_ownership_gw}. "
                    f"With shift(1), can predict up to GW{max_ownership_gw + 1}. "
                    f"\n\nCannot predict GW{missing_gws[0]} - need GW{missing_gws[0] - 1} ownership data. "
                    f"\n\nSolution: Regenerate ownership trends with backfill enabled in dataset-builder."
                )

        # Encode categorical ownership_tier
        ownership_tier_map = {"punt": 0, "budget": 1, "popular": 2, "template": 3}
        df["ownership_tier_encoded"] = (
            df["ownership_tier"].map(ownership_tier_map).fillna(1)
        )

        # Encode transfer_momentum
        momentum_map = {
            "neutral": 0,
            "slow_in": 1,
            "accelerating_in": 2,
            "slow_out": -1,
            "accelerating_out": -2,
        }
        df["transfer_momentum_encoded"] = (
            df["transfer_momentum"].map(momentum_map).fillna(0)
        )

        return df

    def _add_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add value and price change features (Issue #37)."""
        if self.value_analysis_df.empty:
            raise ValueError(
                "Value analysis data is required but was not provided. "
                "Pass value_analysis_df=client.get_derived_value_analysis() to constructor."
            )

        # Merge value data
        value_cols = [
            "player_id",
            "gameweek",
            "points_per_pound",
            "value_vs_position",
            "predicted_price_change_1gw",
            "price_volatility",
            "price_risk",
        ]

        available_cols = [
            col for col in value_cols if col in self.value_analysis_df.columns
        ]
        value_df = self.value_analysis_df[available_cols].copy()

        # Drop any existing value columns to avoid _x _y suffixes during merge
        value_feature_cols = [
            col
            for col in value_cols
            if col not in ["player_id", "gameweek"] and col in df.columns
        ]
        if value_feature_cols:
            df = df.drop(columns=value_feature_cols)

        # Shift value by 1 gameweek (consistent with .shift(1) on performance features)
        value_df_shifted = value_df.copy()
        value_df_shifted["gameweek"] = value_df_shifted["gameweek"] + 1

        df = df.merge(value_df_shifted, on=["player_id", "gameweek"], how="left")

        # Validate no missing data (except GW1 and new players)
        if df["points_per_pound"].isna().any():
            missing_gws = sorted(df[df["points_per_pound"].isna()]["gameweek"].unique())
            max_value_gw = (
                value_df["gameweek"].max()
                if "gameweek" in value_df.columns
                else "unknown"
            )

            # GW1 naturally has no value (no GW0), fill with neutral values
            if 1 in missing_gws:
                gw1_mask = (df["gameweek"] == 1) & df["points_per_pound"].isna()
                df.loc[gw1_mask, "points_per_pound"] = 0.5  # Neutral value
                df.loc[gw1_mask, "value_vs_position"] = 1.0  # Average
                df.loc[gw1_mask, "predicted_price_change_1gw"] = 0
                df.loc[gw1_mask, "price_volatility"] = 0
                df.loc[gw1_mask, "price_risk"] = 0
                missing_gws = [gw for gw in missing_gws if gw != 1]

            # Check if there are still missing gameweeks (after handling GW1)
            # Note: With dataset-builder backfill, this should never happen
            if missing_gws:
                raise ValueError(
                    f"Missing value analysis data for gameweeks {missing_gws}. "
                    f"Value analysis data available through GW{max_value_gw}. "
                    f"With shift(1), can predict up to GW{max_value_gw + 1}. "
                    f"\n\nSolution: Regenerate value analysis with backfill enabled in dataset-builder."
                )

        return df

    def _add_enhanced_fixture_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced fixture difficulty features (Issue #37)."""
        if self.fixture_difficulty_df.empty:
            raise ValueError(
                "Enhanced fixture difficulty data is required but was not provided. "
                "Pass fixture_difficulty_df=client.get_derived_fixture_difficulty() to constructor."
            )

        if "team_id" not in df.columns:
            raise ValueError(
                "team_id column is required in data to merge fixture difficulty features."
            )

        # Start with required columns, add optional ones if they exist
        required_cols = ["team_id", "gameweek"]
        optional_cols = {
            "congestion_difficulty": "congestion_difficulty",
            "form_difficulty": "form_adjusted_difficulty",
            "clean_sheet_probability": "clean_sheet_probability_enhanced",
        }

        # Get columns that exist in fixture_difficulty_df
        cols_to_merge = required_cols + [
            col
            for col in optional_cols.keys()
            if col in self.fixture_difficulty_df.columns
        ]

        if len(cols_to_merge) == len(required_cols):
            raise ValueError(
                f"No optional fixture difficulty columns found. "
                f"Expected at least one of: {list(optional_cols.keys())}"
            )

        fixture_df = self.fixture_difficulty_df[cols_to_merge].copy()

        # Rename columns
        rename_map = {
            old: new for old, new in optional_cols.items() if old in fixture_df.columns
        }
        if rename_map:
            fixture_df = fixture_df.rename(columns=rename_map)

        df = df.merge(fixture_df, on=["team_id", "gameweek"], how="left")

        # Validate at least one feature was added
        added_features = [new_name for new_name in rename_map.values()]
        if added_features and df[added_features[0]].isna().all():
            raise ValueError(
                "All fixture difficulty data is missing after merge. "
                "Check that team_id and gameweek values match between datasets."
            )

        # Ensure all expected features exist (fill with 0 if column doesn't exist from optional merge)
        for expected_feature in [
            "congestion_difficulty",
            "form_adjusted_difficulty",
            "clean_sheet_probability_enhanced",
        ]:
            if expected_feature not in df.columns:
                df[expected_feature] = 0

        return df

    def _add_penalty_set_piece_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add penalty and set-piece taker features from raw FPL players bootstrap data.

        Creates 4 binary features:
        - is_primary_penalty_taker: penalties_order == 1
        - is_penalty_taker: penalties_order >= 1 (includes backups)
        - is_corner_taker: corners_and_indirect_freekicks_order == 1
        - is_fk_taker: direct_freekicks_order == 1

        All features default to 0 if raw_players_df is not provided.

        Args:
            df: Player performance DataFrame with player_id column

        Returns:
            DataFrame with 4 additional binary features
        """
        # If no raw players data, return with default zero values
        if self.raw_players_df.empty:
            df["is_primary_penalty_taker"] = 0
            df["is_penalty_taker"] = 0
            df["is_corner_taker"] = 0
            df["is_fk_taker"] = 0
            return df

        # Validate required columns exist in raw_players_df
        required_cols = [
            "player_id",
            "penalties_order",
            "corners_and_indirect_freekicks_order",
            "direct_freekicks_order",
        ]
        missing_cols = [
            col for col in required_cols if col not in self.raw_players_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"raw_players_df is missing required columns: {missing_cols}"
            )

        # Create subset with only needed columns (and gameweek if available)
        merge_keys = ["player_id"]
        extra_cols = []
        if "gameweek" in self.raw_players_df.columns:
            # Support per-gameweek penalty/set-piece orders
            merge_keys = ["player_id", "gameweek"]
            extra_cols = ["gameweek"]

        penalty_data = self.raw_players_df[extra_cols + required_cols].copy()

        # Create binary features based on order values
        penalty_data["is_primary_penalty_taker"] = (
            penalty_data["penalties_order"] == 1
        ).astype(int)
        penalty_data["is_penalty_taker"] = (
            penalty_data["penalties_order"] >= 1
        ).astype(int)
        penalty_data["is_corner_taker"] = (
            penalty_data["corners_and_indirect_freekicks_order"] == 1
        ).astype(int)
        penalty_data["is_fk_taker"] = (
            penalty_data["direct_freekicks_order"] == 1
        ).astype(int)

        # Merge with main DataFrame on player_id (and gameweek if available)
        # Use left merge to keep all rows, features default to 0 for unmatched players
        df = df.merge(
            penalty_data[
                [
                    *merge_keys,
                    "is_primary_penalty_taker",
                    "is_penalty_taker",
                    "is_corner_taker",
                    "is_fk_taker",
                ]
            ],
            on=merge_keys,
            how="left",
        )

        # Handle any NaN values from merge (players not in raw_players_df)
        df["is_primary_penalty_taker"] = (
            df["is_primary_penalty_taker"].fillna(0).astype(int)
        )
        df["is_penalty_taker"] = df["is_penalty_taker"].fillna(0).astype(int)
        df["is_corner_taker"] = df["is_corner_taker"].fillna(0).astype(int)
        df["is_fk_taker"] = df["is_fk_taker"].fillna(0).astype(int)

        return df

    def _add_betting_odds_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add betting odds features from derived betting features data.

        Creates 15 features from betting odds:
        - Implied probabilities (6): team_win_probability, opponent_win_probability, draw_probability,
          implied_clean_sheet_probability, implied_total_goals, team_expected_goals
        - Market confidence (4): market_consensus_strength, odds_movement_team, odds_movement_magnitude,
          favorite_status
        - Asian Handicap (3): asian_handicap_line, handicap_team_odds, expected_goal_difference
        - Match context (2): over_under_signal, referee_encoded

        All features default to neutral values if betting_features_df is not provided.

        Args:
            df: Player performance DataFrame with [player_id, gameweek] columns

        Returns:
            DataFrame with 15 additional betting odds features
        """
        # Define neutral defaults for missing betting odds data
        # These are used for GW1 or when betting data is unavailable
        betting_defaults = {
            "team_win_probability": 0.33,  # neutral 3-way split
            "opponent_win_probability": 0.33,
            "draw_probability": 0.33,
            "implied_clean_sheet_probability": 0.35,  # PL average
            "implied_total_goals": 2.5,  # PL average
            "team_expected_goals": 1.25,  # neutral split
            "market_consensus_strength": 0.5,  # unknown confidence
            "odds_movement_team": 0.0,  # no movement
            "odds_movement_magnitude": 0.0,
            "favorite_status": 0.5,  # unknown
            "asian_handicap_line": 0.0,  # neutral handicap
            "handicap_team_odds": 2.0,  # neutral odds
            "expected_goal_difference": 0.0,
            "over_under_signal": 0.0,  # neutral tempo
            "referee_encoded": -1,  # unknown referee
        }

        # If no betting features data, use neutral defaults
        if self.betting_features_df.empty:
            for feature, default_value in betting_defaults.items():
                df[feature] = default_value
            return df

        # Validate required columns in betting_features_df
        required_cols = ["gameweek", "player_id"] + list(betting_defaults.keys())
        missing_cols = [
            col for col in required_cols if col not in self.betting_features_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"betting_features_df is missing required columns: {missing_cols}. "
                f"Expected columns from get_derived_betting_features(): {required_cols}"
            )

        # NO TEMPORAL SHIFT NEEDED - betting odds are forward-looking
        # Odds for GW N are available before GW N kickoff (before points are known)
        # Unlike player performance data (which requires shift(1)), odds are pre-match
        # This is leak-free: odds exist BEFORE the match outcome

        # Merge betting features on player_id + gameweek
        # Left join keeps all player-gameweek rows
        df = df.merge(
            self.betting_features_df[required_cols],
            on=["player_id", "gameweek"],
            how="left",
        )

        # Fill missing values (GW1 or missing odds) with neutral defaults
        for feature, default_value in betting_defaults.items():
            df[feature] = df[feature].fillna(default_value)

        return df

    def _get_feature_columns(self) -> list:
        """Get list of all feature columns (99 features: 65 base + 15 enhanced + 4 penalty/set-piece + 15 betting odds)."""
        return [
            # Static (4)
            "price",
            "position_encoded",
            "games_played",
            "price_band",
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
            # Team features (13)
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
            # Enhanced ownership features (7) - Issue #37
            "selected_by_percent",
            "ownership_tier_encoded",
            "transfer_momentum_encoded",
            "net_transfers_gw",
            "avg_net_transfers_5gw",
            "bandwagon_score",
            "ownership_velocity",
            # Enhanced value features (5) - Issue #37
            "points_per_pound",
            "value_vs_position",
            "predicted_price_change_1gw",
            "price_volatility",
            "price_risk",
            # Enhanced fixture features (3) - Issue #37
            "congestion_difficulty",
            "form_adjusted_difficulty",
            "clean_sheet_probability_enhanced",
            # Penalty & set-piece taker features (4)
            "is_primary_penalty_taker",
            "is_penalty_taker",
            "is_corner_taker",
            "is_fk_taker",
            # Betting odds features (15) - Issue #38
            # Implied probabilities (6)
            "team_win_probability",
            "opponent_win_probability",
            "draw_probability",
            "implied_clean_sheet_probability",
            "implied_total_goals",
            "team_expected_goals",
            # Market confidence (4)
            "market_consensus_strength",
            "odds_movement_team",
            "odds_movement_magnitude",
            "favorite_status",
            # Asian Handicap (3)
            "asian_handicap_line",
            "handicap_team_odds",
            "expected_goal_difference",
            # Match context (2)
            "over_under_signal",
            "referee_encoded",
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
