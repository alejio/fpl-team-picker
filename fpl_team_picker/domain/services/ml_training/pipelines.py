"""
Pipeline construction and hyperparameter spaces for ML training.

Centralizes all regressor definitions, parameter distributions, and
preprocessing strategies in one place.
"""

from typing import Any, Dict, List, Optional, Tuple

from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from fpl_team_picker.domain.ml import FeatureSelector


# =============================================================================
# REGRESSOR DEFINITIONS
# =============================================================================

REGRESSOR_MAP = {
    "lightgbm": "LGBMRegressor",
    "xgboost": "XGBRegressor",
    "random-forest": "RandomForestRegressor",
    "gradient-boost": "GradientBoostingRegressor",
    "adaboost": "AdaBoostRegressor",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "ElasticNet",
}


def get_regressor(regressor_name: str, random_seed: int = 42) -> Any:
    """
    Get regressor instance by name.

    Args:
        regressor_name: Name of regressor (lightgbm, xgboost, etc.)
        random_seed: Random seed for reproducibility

    Returns:
        Regressor instance
    """
    if regressor_name == "xgboost":
        try:
            import xgboost as xgb

            return xgb.XGBRegressor(
                random_state=random_seed,
                n_jobs=1,
                tree_method="hist",
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
            )
        except ImportError:
            raise ImportError("XGBoost not installed. Run: uv add xgboost")

    elif regressor_name == "lightgbm":
        try:
            import lightgbm as lgb

            return lgb.LGBMRegressor(
                random_state=random_seed,
                n_jobs=1,
                verbose=-1,
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
            )
        except ImportError:
            raise ImportError("LightGBM not installed. Run: uv add lightgbm")

    elif regressor_name == "random-forest":
        return RandomForestRegressor(random_state=random_seed, n_jobs=1)

    elif regressor_name == "gradient-boost":
        return GradientBoostingRegressor(
            random_state=random_seed,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
        )

    elif regressor_name == "adaboost":
        return AdaBoostRegressor(random_state=random_seed)

    elif regressor_name == "ridge":
        return Ridge(random_state=random_seed)

    elif regressor_name == "lasso":
        return Lasso(random_state=random_seed, max_iter=10000)

    elif regressor_name == "elasticnet":
        return ElasticNet(random_state=random_seed, max_iter=10000)

    else:
        raise ValueError(f"Unknown regressor: {regressor_name}")


def get_param_space(regressor_name: str) -> Dict[str, Any]:
    """
    Get hyperparameter search space for regressor.

    All keys are prefixed with 'regressor__' for sklearn Pipeline compatibility.

    Args:
        regressor_name: Name of regressor

    Returns:
        Dictionary of parameter distributions for RandomizedSearchCV
    """
    try:
        from scipy.stats import loguniform
    except ImportError:
        loguniform = None

    if regressor_name == "xgboost":
        return {
            "regressor__n_estimators": randint(200, 1500),
            "regressor__max_depth": randint(3, 10),
            "regressor__learning_rate": loguniform(0.03, 0.3)
            if loguniform
            else uniform(0.03, 0.27),
            "regressor__subsample": uniform(0.6, 0.4),
            "regressor__colsample_bytree": uniform(0.6, 0.4),
            "regressor__min_child_weight": randint(1, 10),
            "regressor__gamma": uniform(0, 0.5),
            "regressor__reg_alpha": uniform(0, 1.0),
            "regressor__reg_lambda": uniform(0.0, 2.0),
        }

    elif regressor_name == "lightgbm":
        return {
            "regressor__n_estimators": randint(200, 1500),
            "regressor__max_depth": randint(3, 10),
            "regressor__learning_rate": loguniform(0.03, 0.3)
            if loguniform
            else uniform(0.03, 0.27),
            "regressor__num_leaves": randint(20, 100),
            "regressor__subsample": uniform(0.6, 0.4),
            "regressor__colsample_bytree": uniform(0.6, 0.4),
            "regressor__min_child_samples": randint(10, 50),
            "regressor__reg_alpha": uniform(0, 1.0),
            "regressor__reg_lambda": uniform(0.0, 2.0),
        }

    elif regressor_name == "random-forest":
        return {
            "regressor__n_estimators": randint(100, 500),
            "regressor__max_depth": randint(5, 30),
            "regressor__min_samples_split": randint(2, 20),
            "regressor__min_samples_leaf": randint(1, 10),
            "regressor__max_features": uniform(0.3, 0.7),
        }

    elif regressor_name == "gradient-boost":
        return {
            "regressor__n_estimators": randint(200, 1000),
            "regressor__max_depth": randint(3, 8),
            "regressor__learning_rate": loguniform(0.03, 0.2)
            if loguniform
            else uniform(0.03, 0.17),
            "regressor__subsample": uniform(0.6, 0.4),
            "regressor__min_samples_split": randint(2, 20),
            "regressor__min_samples_leaf": randint(1, 10),
        }

    elif regressor_name == "adaboost":
        return {
            "regressor__n_estimators": randint(50, 300),
            "regressor__learning_rate": uniform(0.01, 1.0),
            "regressor__loss": ["linear", "square", "exponential"],
        }

    elif regressor_name == "ridge":
        return {
            "regressor__alpha": uniform(0.001, 10.0),
            "regressor__solver": ["auto", "svd", "cholesky", "lsqr"],
        }

    elif regressor_name == "lasso":
        return {
            "regressor__alpha": uniform(0.001, 5.0),
            "regressor__selection": ["cyclic", "random"],
        }

    elif regressor_name == "elasticnet":
        return {
            "regressor__alpha": uniform(0.001, 5.0),
            "regressor__l1_ratio": uniform(0.0, 1.0),
            "regressor__selection": ["cyclic", "random"],
        }

    else:
        raise ValueError(f"Unknown regressor: {regressor_name}")


# =============================================================================
# FEATURE GROUPS FOR PREPROCESSING
# =============================================================================


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Define feature groups for grouped preprocessing.

    Categorizes all 155 FPL features into appropriate preprocessing groups.
    """
    return {
        # Count features (Poisson-like, many zeros) - Use Robust Scaler
        "count_features": [
            "games_played",
            "cumulative_goals",
            "cumulative_assists",
            "cumulative_clean_sheets",
            "cumulative_yellow_cards",
            "cumulative_red_cards",
            "rolling_5gw_goals",
            "rolling_5gw_assists",
            "rolling_5gw_clean_sheets",
            "rolling_5gw_saves",
            "rolling_5gw_goals_conceded",
            "net_transfers_gw",
            "team_rolling_5gw_goals_scored",
            "team_rolling_5gw_goals_conceded",
            "team_rolling_5gw_clean_sheets",
            "team_cumulative_goals_scored",
            "team_cumulative_goals_conceded",
            "team_cumulative_clean_sheets",
            "opponent_rolling_5gw_goals_conceded",
            "opponent_rolling_5gw_clean_sheets",
        ],
        # Binary features (0/1) - No scaling needed
        "binary_features": [
            "is_home",
            "is_primary_penalty_taker",
            "is_penalty_taker",
            "is_corner_taker",
            "is_fk_taker",
        ],
        # Percentage features (0-100 range) - Use MinMaxScaler
        "percentage_features": [
            "selected_by_percent",
            "ownership_velocity",
            "minutes_played_rate",
            "clean_sheet_rate",
            "clean_sheet_probability_enhanced",
        ],
        # Probability features (0-1 range) - Use MinMaxScaler
        "probability_features": [
            "team_win_probability",
            "opponent_win_probability",
            "draw_probability",
            "implied_clean_sheet_probability",
            "market_consensus_strength",
            "favorite_status",
        ],
        # Continuous/rate features - Use StandardScaler
        "continuous_features": [
            "goals_per_90",
            "assists_per_90",
            "points_per_90",
            "xg_per_90",
            "xa_per_90",
            "bps_per_90",
            "rolling_5gw_goals_per_90",
            "rolling_5gw_assists_per_90",
            "rolling_5gw_points_per_90",
            "cumulative_xg",
            "cumulative_xa",
            "rolling_5gw_xg",
            "rolling_5gw_xa",
            "rolling_5gw_xgi",
            "rolling_5gw_xgc",
            "team_rolling_5gw_xg",
            "team_rolling_5gw_xgc",
            "team_rolling_5gw_xg_diff",
            "opponent_rolling_5gw_xgc",
            "cumulative_points",
            "cumulative_bonus",
            "cumulative_bps",
            "rolling_5gw_points",
            "rolling_5gw_bps",
            "rolling_5gw_bonus",
            "team_rolling_5gw_points",
            "team_season_points",
            "team_rolling_5gw_goal_diff",
            "rolling_5gw_ict_index",
            "rolling_5gw_influence",
            "rolling_5gw_creativity",
            "rolling_5gw_threat",
            "rolling_5gw_points_std",
            "rolling_5gw_minutes_std",
            "form_trend",
            "opponent_strength",
            "fixture_difficulty",
            "congestion_difficulty",
            "form_adjusted_difficulty",
            "implied_total_goals",
            "team_expected_goals",
            "asian_handicap_line",
            "handicap_team_odds",
            "expected_goal_difference",
            "over_under_signal",
            "odds_movement_team",
            "odds_movement_magnitude",
            "points_per_pound",
            "value_vs_position",
            "predicted_price_change_1gw",
            "price_volatility",
            "price_risk",
            "bandwagon_score",
            "avg_net_transfers_5gw",
            "cumulative_minutes",
            "rolling_5gw_minutes",
        ],
        # Encoded/ordinal features - Use StandardScaler
        "encoded_features": [
            "position_encoded",
            "team_encoded",
            "ownership_tier_encoded",
            "transfer_momentum_encoded",
            "price_band",
            "referee_encoded",
        ],
        # Price features - Use StandardScaler
        "price_features": ["price"],
    }


# =============================================================================
# PREPROCESSOR CONSTRUCTION
# =============================================================================


def create_preprocessor(strategy: str, feature_names: List[str]) -> Any:
    """
    Create a preprocessor based on the specified strategy.

    Args:
        strategy: Preprocessing strategy ("standard", "grouped", or "robust")
        feature_names: List of feature names (after feature selection)

    Returns:
        Preprocessor (StandardScaler, RobustScaler, or ColumnTransformer)
    """
    if strategy == "standard":
        return StandardScaler()

    elif strategy == "robust":
        return RobustScaler()

    elif strategy == "grouped":
        feature_groups = get_feature_groups()
        transformers = []

        # Count features -> RobustScaler
        count_feats = [
            f for f in feature_groups["count_features"] if f in feature_names
        ]
        if count_feats:
            transformers.append(("count_scaler", RobustScaler(), count_feats))

        # Binary features -> passthrough
        binary_feats = [
            f for f in feature_groups["binary_features"] if f in feature_names
        ]
        if binary_feats:
            transformers.append(("binary_passthrough", "passthrough", binary_feats))

        # Percentage features -> MinMaxScaler
        pct_feats = [
            f for f in feature_groups["percentage_features"] if f in feature_names
        ]
        if pct_feats:
            transformers.append(("percentage_minmax", MinMaxScaler(), pct_feats))

        # Probability features -> MinMaxScaler
        prob_feats = [
            f for f in feature_groups["probability_features"] if f in feature_names
        ]
        if prob_feats:
            transformers.append(("probability_minmax", MinMaxScaler(), prob_feats))

        # Continuous features -> StandardScaler
        cont_feats = [
            f for f in feature_groups["continuous_features"] if f in feature_names
        ]
        if cont_feats:
            transformers.append(("continuous_standard", StandardScaler(), cont_feats))

        # Encoded features -> StandardScaler
        encoded_feats = [
            f for f in feature_groups["encoded_features"] if f in feature_names
        ]
        if encoded_feats:
            transformers.append(("encoded_standard", StandardScaler(), encoded_feats))

        # Price features -> StandardScaler
        price_feats = [
            f for f in feature_groups["price_features"] if f in feature_names
        ]
        if price_feats:
            transformers.append(("price_standard", StandardScaler(), price_feats))

        return ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

    else:
        raise ValueError(f"Unknown preprocessing strategy: {strategy}")


# =============================================================================
# PIPELINE CONSTRUCTION
# =============================================================================


def build_pipeline(
    regressor_name: str,
    feature_names: List[str],
    preprocessing: str = "standard",
    random_seed: int = 42,
    params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """
    Build a complete sklearn Pipeline.

    Pipeline structure:
    1. FeatureSelector - selects and orders features (makes pipeline self-contained)
    2. Preprocessor - scales/transforms features
    3. Regressor - the ML model

    Args:
        regressor_name: Name of regressor
        feature_names: List of feature names to use
        preprocessing: Preprocessing strategy
        random_seed: Random seed
        params: Optional hyperparameters to set on pipeline

    Returns:
        Configured sklearn Pipeline
    """
    regressor = get_regressor(regressor_name, random_seed)
    preprocessor = create_preprocessor(preprocessing, feature_names)

    pipeline = Pipeline(
        [
            ("feature_selector", FeatureSelector(feature_names)),
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    if params:
        pipeline.set_params(**params)

    return pipeline


def get_regressor_and_param_grid(
    regressor_name: str, random_seed: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Get regressor instance and its parameter grid.

    Convenience function that returns both together.

    Args:
        regressor_name: Name of regressor
        random_seed: Random seed

    Returns:
        Tuple of (regressor_instance, param_distributions)
    """
    return get_regressor(regressor_name, random_seed), get_param_space(regressor_name)
