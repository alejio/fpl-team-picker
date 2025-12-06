"""
Reusable ML training utilities for FPL expected points prediction.

This module extracts common data loading, preprocessing, and evaluation logic
used across different modeling approaches (TPOT, custom pipelines, experiments).

Ensures "apples to apples" comparison by standardizing:
- Data loading (GW range, features, target)
- Feature engineering (FPLFeatureEngineer with 117 features)
- Temporal cross-validation splits (walk-forward validation)
- Evaluation metrics (MAE, RMSE, Spearman correlation)
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client import FPLDataClient  # noqa: E402
from fpl_team_picker.domain.services.ml_feature_engineering import (  # noqa: E402
    FPLFeatureEngineer,
    calculate_per_gameweek_team_strength,
)


class TemporalCVSplitter(BaseCrossValidator):
    """
    Custom CV splitter for temporal walk-forward validation.
    Wraps list of (train_idx, test_idx) tuples for sklearn compatibility.
    """

    def __init__(self, splits):
        """
        Args:
            splits: List of (train_idx, test_idx) tuples
        """
        self.splits = splits

    def split(self, X, y=None, groups=None):
        """Yield train/test indices."""
        for train_idx, test_idx in self.splits:
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        return len(self.splits)


def load_training_data(
    start_gw: int, end_gw: int, verbose: bool = True
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load all data required for ML training.

    Args:
        start_gw: Starting gameweek (typically 1)
        end_gw: Ending gameweek (inclusive, typically 8-9)
        verbose: Print loading progress

    Returns:
        Tuple of (historical_df, fixtures_df, teams_df, ownership_trends_df,
                  value_analysis_df, fixture_difficulty_df, betting_features_df, raw_players_df,
                  derived_player_metrics_df, player_availability_snapshot_df, derived_team_form_df, players_enhanced_df)
    """
    client = FPLDataClient()

    if verbose:
        logger.info(f"\n📥 Loading training data (GW{start_gw} to GW{end_gw})...")

    # Load historical gameweek performance
    historical_data = []
    for gw in range(start_gw, end_gw + 1):
        gw_performance = client.get_gameweek_performance(gw)
        if not gw_performance.empty:
            gw_performance["gameweek"] = gw
            historical_data.append(gw_performance)
            if verbose:
                logger.debug(f"   ✅ GW{gw}: {len(gw_performance)} players")
        else:
            if verbose:
                logger.warning(f"   ⚠️  GW{gw}: No data available")

    if not historical_data:
        raise ValueError("No historical data loaded. Check gameweek range.")

    historical_df = pd.concat(historical_data, ignore_index=True)

    # Enrich with position data
    if verbose:
        logger.info("\n📊 Enriching with position data...")

    players_data = client.get_current_players()
    if (
        "position" not in players_data.columns
        or "player_id" not in players_data.columns
    ):
        raise ValueError(
            f"Position column not found in current_players data. "
            f"Available columns: {list(players_data.columns)}"
        )

    historical_df = historical_df.merge(
        players_data[["player_id", "position"]], on="player_id", how="left"
    )

    missing_count = historical_df["position"].isna().sum()
    if missing_count > 0 and verbose:
        logger.warning(f"   ⚠️  Warning: {missing_count} records missing position data")

    # Load fixtures and teams
    if verbose:
        logger.info("\n🏟️  Loading fixtures and teams...")
    fixtures_df = client.get_fixtures_normalized()
    teams_df = client.get_current_teams()

    if verbose:
        logger.info(f"   ✅ Fixtures: {len(fixtures_df)} | Teams: {len(teams_df)}")

    # Load enhanced data sources
    if verbose:
        logger.info("\n📊 Loading enhanced data sources...")
    ownership_trends_df = client.get_derived_ownership_trends()
    value_analysis_df = client.get_derived_value_analysis()
    fixture_difficulty_df = client.get_derived_fixture_difficulty()

    if verbose:
        logger.info(f"   ✅ Ownership trends: {len(ownership_trends_df)} records")
        logger.info(f"   ✅ Value analysis: {len(value_analysis_df)} records")
        logger.info(f"   ✅ Fixture difficulty: {len(fixture_difficulty_df)} records")

    # Load betting odds features
    if verbose:
        logger.info("\n🎲 Loading betting odds features...")
    try:
        betting_features_df = client.get_derived_betting_features()
        if verbose:
            logger.info(f"   ✅ Betting features: {len(betting_features_df)} records")
    except (AttributeError, Exception) as e:
        if verbose:
            logger.warning(f"   ⚠️  Betting features unavailable: {e}")
            logger.info(
                "   ℹ️  Continuing with neutral defaults (features will be 0/neutral)"
            )
        betting_features_df = pd.DataFrame()

    # Load raw players data for penalty/set-piece features
    if verbose:
        logger.info("\n⚽ Loading penalty/set-piece taker data...")
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
            if verbose:
                logger.warning(f"   ⚠️  Missing penalty columns: {missing_cols}")
            raw_players_df = pd.DataFrame()
        else:
            if verbose:
                logger.info(f"   ✅ Loaded penalty data: {len(raw_players_df)} players")
    except Exception as e:
        if verbose:
            logger.error(f"   ❌ Error loading penalty data: {e}")
        raw_players_df = pd.DataFrame()

    # Load Phase 1: Injury & rotation risk data sources
    if verbose:
        logger.info("\n🏥 Loading injury & rotation risk data sources...")
    try:
        derived_player_metrics_df = client.get_derived_player_metrics()
        if verbose:
            logger.info(
                f"   ✅ Derived player metrics: {len(derived_player_metrics_df)} records"
            )
    except (AttributeError, Exception) as e:
        if verbose:
            logger.warning(f"   ⚠️  Derived player metrics unavailable: {e}")
        derived_player_metrics_df = pd.DataFrame()

    # Load player availability snapshots for historical training
    # For each gameweek in range, load snapshot
    player_availability_snapshot_df = pd.DataFrame()
    try:
        availability_snapshots = []
        for gw in range(start_gw, end_gw + 1):
            try:
                snapshot = client.get_player_availability_snapshot(
                    gameweek=gw, include_backfilled=True
                )
                if not snapshot.empty:
                    snapshot["gameweek"] = gw
                    availability_snapshots.append(snapshot)
            except Exception:
                # Skip if snapshot not available for this gameweek
                pass

        if availability_snapshots:
            player_availability_snapshot_df = pd.concat(
                availability_snapshots, ignore_index=True
            )
            if verbose:
                logger.info(
                    f"   ✅ Player availability snapshots: {len(player_availability_snapshot_df)} records"
                )
        else:
            if verbose:
                logger.warning("   ⚠️  No player availability snapshots available")
    except (AttributeError, Exception) as e:
        if verbose:
            logger.warning(f"   ⚠️  Player availability snapshots unavailable: {e}")
        player_availability_snapshot_df = pd.DataFrame()

    # Load Phase 2: Venue-specific team strength
    if verbose:
        logger.info("\n🏟️  Loading venue-specific team strength data...")
    try:
        derived_team_form_df = client.get_derived_team_form()
        if verbose:
            logger.info(f"   ✅ Derived team form: {len(derived_team_form_df)} records")
    except (AttributeError, Exception) as e:
        if verbose:
            logger.warning(f"   ⚠️  Derived team form unavailable: {e}")
        derived_team_form_df = pd.DataFrame()

    # Load Phase 3: Player rankings & context
    if verbose:
        logger.info("\n📊 Loading player rankings & context data...")
    try:
        players_enhanced_df = client.get_players_enhanced()
        if verbose:
            logger.info(f"   ✅ Players enhanced: {len(players_enhanced_df)} records")
    except (AttributeError, Exception) as e:
        if verbose:
            logger.warning(f"   ⚠️  Players enhanced unavailable: {e}")
        players_enhanced_df = pd.DataFrame()

    if verbose:
        logger.info(f"\n✅ Total records: {len(historical_df):,}")
        logger.info(f"   Unique players: {historical_df['player_id'].nunique():,}")
        logger.info(f"   Gameweeks: {sorted(historical_df['gameweek'].unique())}")

    return (
        historical_df,
        fixtures_df,
        teams_df,
        ownership_trends_df,
        value_analysis_df,
        fixture_difficulty_df,
        betting_features_df,
        raw_players_df,
        derived_player_metrics_df,
        player_availability_snapshot_df,
        derived_team_form_df,
        players_enhanced_df,
    )


def engineer_features(
    historical_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    ownership_trends_df: pd.DataFrame,
    value_analysis_df: pd.DataFrame,
    fixture_difficulty_df: pd.DataFrame,
    betting_features_df: pd.DataFrame,
    raw_players_df: pd.DataFrame,
    derived_player_metrics_df: pd.DataFrame,
    player_availability_snapshot_df: pd.DataFrame,
    derived_team_form_df: pd.DataFrame,
    players_enhanced_df: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Engineer features using production FPLFeatureEngineer.

    Args:
        historical_df: Historical gameweek performance with position column
        fixtures_df: Fixture data
        teams_df: Team data
        ownership_trends_df: Ownership trends data
        value_analysis_df: Value analysis data
        fixture_difficulty_df: Enhanced fixture difficulty data
        betting_features_df: Betting odds features data
        raw_players_df: Raw players data with penalty orders
        derived_player_metrics_df: Derived player metrics data (Phase 1)
        player_availability_snapshot_df: Player availability snapshot data (Phase 1)
        derived_team_form_df: Derived team form data (Phase 2)
        players_enhanced_df: Enhanced players data (Phase 3)
        verbose: Print progress messages

    Returns:
        Tuple of (features_df, target_array, feature_column_names)
    """
    if verbose:
        logger.info(
            "\n🔧 Engineering features (FPLFeatureEngineer with 117 features)..."
        )

    # Calculate per-gameweek team strength (no data leakage)
    if verbose:
        logger.info("   📊 Calculating per-gameweek team strength (leak-free)...")

    start_gw = 6  # First trainable gameweek (needs GW1-5 for rolling features)
    end_gw = historical_df["gameweek"].max()
    team_strength = calculate_per_gameweek_team_strength(
        start_gw=start_gw,
        end_gw=end_gw,
        teams_df=teams_df,
    )

    if verbose:
        logger.info(f"   ✅ Team strength calculated for GW{start_gw}-{end_gw}")

    # Initialize feature engineer
    feature_engineer = FPLFeatureEngineer(
        fixtures_df=fixtures_df if not fixtures_df.empty else None,
        teams_df=teams_df if not teams_df.empty else None,
        team_strength=team_strength if team_strength else None,
        ownership_trends_df=ownership_trends_df
        if not ownership_trends_df.empty
        else None,
        value_analysis_df=value_analysis_df if not value_analysis_df.empty else None,
        fixture_difficulty_df=fixture_difficulty_df
        if not fixture_difficulty_df.empty
        else None,
        betting_features_df=betting_features_df
        if not betting_features_df.empty
        else None,
        raw_players_df=raw_players_df if not raw_players_df.empty else None,
        derived_player_metrics_df=derived_player_metrics_df
        if not derived_player_metrics_df.empty
        else None,
        player_availability_snapshot_df=player_availability_snapshot_df
        if not player_availability_snapshot_df.empty
        else None,
        derived_team_form_df=derived_team_form_df
        if not derived_team_form_df.empty
        else None,
        players_enhanced_df=players_enhanced_df
        if not players_enhanced_df.empty
        else None,
    )

    # Transform
    if verbose:
        logger.info("   🔄 Transforming data...")

    features_df = feature_engineer.fit_transform(historical_df)
    target = historical_df["total_points"].values
    feature_names = feature_engineer.get_feature_names_out()

    if verbose:
        logger.info(f"   ✅ Features shape: {features_df.shape}")
        logger.info(f"   ✅ Target shape: {target.shape}")
        logger.info(f"   ✅ Feature count: {len(feature_names)}")

    # Add metadata columns back for CV splitting and evaluation
    # CRITICAL: Sort historical_df to match feature_df order (FPLFeatureEngineer sorts internally)
    historical_df_sorted = historical_df.sort_values(
        ["player_id", "gameweek"]
    ).reset_index(drop=True)
    # Use pd.concat instead of multiple frame.insert to avoid fragmentation warnings
    metadata_df = pd.DataFrame(
        {
            "gameweek": historical_df_sorted["gameweek"].values,
            "position": historical_df_sorted["position"].values,
            "player_id": historical_df_sorted["player_id"].values,
        },
        index=features_df.index,
    )
    features_df = pd.concat([features_df, metadata_df], axis=1)

    # Update target to match sorted order
    target = historical_df_sorted["total_points"].values

    return features_df, target, feature_names


def create_temporal_cv_splits(
    features_df: pd.DataFrame, max_folds: Optional[int] = None, verbose: bool = True
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """
    Create temporal walk-forward cross-validation splits.

    Train on GW6 to N-1, test on GW N (iterate forward).

    Args:
        features_df: Feature-engineered DataFrame with 'gameweek' column
        max_folds: Maximum number of folds (default: None = all available)
        verbose: Print split details

    Returns:
        Tuple of (cv_splits, cv_data)
        - cv_splits: List of (train_idx, test_idx) tuples
        - cv_data: Filtered DataFrame (GW6+)
    """
    if verbose:
        logger.info("\n📊 Creating temporal CV splits (walk-forward validation)...")

    # Filter to GW6+ (need 5 preceding GWs for features)
    cv_data = features_df[features_df["gameweek"] >= 6].copy()

    # Store original index before reset (for target alignment)
    cv_data["_original_index"] = cv_data.index
    cv_data = cv_data.reset_index(drop=True)
    cv_gws = sorted(cv_data["gameweek"].unique())

    if len(cv_gws) < 2:
        raise ValueError(
            f"Need at least 2 gameweeks for temporal CV. Found: {cv_gws}. "
            "Ensure end_gw >= 7 (GW6 train, GW7 test minimum)."
        )

    cv_splits = []
    n_folds = len(cv_gws) - 1
    if max_folds is not None:
        n_folds = min(n_folds, max_folds)

    for i in range(n_folds):
        train_gws = cv_gws[: i + 1]  # GW6 to GW(6+i)
        test_gw = cv_gws[i + 1]  # GW(6+i+1)

        # Get positional indices
        train_mask = cv_data["gameweek"].isin(train_gws)
        test_mask = cv_data["gameweek"] == test_gw

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        cv_splits.append((train_idx, test_idx))

        if verbose:
            logger.debug(
                f"   Fold {i + 1}: Train on GW{min(train_gws)}-{max(train_gws)} "
                f"({len(train_idx)} samples) → Test on GW{test_gw} ({len(test_idx)} samples)"
            )

    if verbose:
        logger.info(f"\n✅ Created {len(cv_splits)} temporal CV folds")

    return cv_splits, cv_data


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    position: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate predictions using FPL-relevant metrics.

    Args:
        y_true: Actual total points
        y_pred: Predicted total points
        position: Player positions (for per-position metrics)
        verbose: Print evaluation results

    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    correlation, _ = spearmanr(y_true, y_pred)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "spearman_correlation": correlation,
    }

    if verbose:
        logger.info("\n📊 Evaluation Metrics:")
        logger.info(f"   MAE: {mae:.3f}")
        logger.info(f"   RMSE: {rmse:.3f}")
        logger.info(f"   Spearman Correlation: {correlation:.3f}")

    # Per-position metrics
    if position is not None:
        if verbose:
            logger.info("\n📊 Per-Position Metrics:")

        for pos in ["GKP", "DEF", "MID", "FWD"]:
            mask = position == pos
            if mask.sum() > 0:
                pos_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                pos_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                pos_corr, _ = spearmanr(y_true[mask], y_pred[mask])

                metrics[f"{pos}_mae"] = pos_mae
                metrics[f"{pos}_rmse"] = pos_rmse
                metrics[f"{pos}_correlation"] = pos_corr

                if verbose:
                    logger.info(
                        f"   {pos}: MAE={pos_mae:.3f}, RMSE={pos_rmse:.3f}, Corr={pos_corr:.3f}"
                    )

    return metrics


def spearman_scorer(y_true, y_pred):
    """Spearman correlation scorer for sklearn (higher is better)."""
    correlation, _ = spearmanr(y_true, y_pred)
    return correlation


# Create sklearn-compatible scorer
spearman_correlation_scorer = make_scorer(spearman_scorer, greater_is_better=True)


# ==============================================================================
# FPL-SPECIFIC SCORERS
# ==============================================================================


def fpl_weighted_huber_scorer(y_true, y_pred, sample_weight=None):
    """
    Custom FPL scorer: Weighted Huber Loss with value-aware penalties.

    Optimized for FPL optimization use case:
    1. Huber loss balances MAE/MSE (robust to outliers, penalizes large errors)
    2. Value-based weighting (penalize errors on high xP players - captaincy candidates)
    3. Asymmetric loss (underestimation worse than overestimation)

    Args:
        y_true: Actual total points
        y_pred: Predicted xP
        sample_weight: Optional position weights

    Returns:
        Negative loss (higher is better for sklearn compatibility)
    """
    errors = y_true - y_pred

    # 1. Huber loss (delta=2.0 tuned for FPL point scale)
    #    Acts like MAE for small errors (<2 pts), MSE for large errors (>2 pts)
    #    Prevents catastrophic misses on explosive hauls
    delta = 2.0
    huber_loss = np.where(
        np.abs(errors) <= delta, 0.5 * errors**2, delta * (np.abs(errors) - 0.5 * delta)
    )

    # 2. Value-based weighting (penalize errors on high scorers)
    #    AGGRESSIVE: Focus on top scorers (actual points, not predicted)
    #    Captain candidates (10+ actual points) matter most for FPL strategy
    value_weights = np.where(
        y_true >= 10.0,
        5.0,  # Explosive hauls (10+ pts): 5x penalty - MUST get these right!
        np.where(
            y_true >= 6.0,
            3.0,  # Good returns (6-9 pts): 3x penalty
            np.where(
                y_true >= 3.0,
                1.5,  # Decent (3-5 pts): 1.5x penalty
                1.0,  # Blanks (0-2 pts): 1x penalty
            ),
        ),
    )

    # 3. Asymmetric penalty (underestimation is MUCH worse for FPL)
    #    Missing a 15pt haul (underestimate) costs WAY more than overestimating a 2pt return
    #    AGGRESSIVE: penalize underestimation 3.0x vs overestimation 1.0x
    asymmetric_weights = np.where(errors > 0, 3.0, 1.0)

    # 4. Combine weights
    combined_weights = value_weights * asymmetric_weights

    # Apply sample weights if provided (e.g., position-based from cross-validation)
    if sample_weight is not None:
        combined_weights *= sample_weight

    # Weighted Huber loss
    weighted_loss = (huber_loss * combined_weights).mean()

    return -weighted_loss  # Negative for sklearn (higher is better)


def fpl_top_k_ranking_scorer(y_true, y_pred, k=50, sample_weight=None):
    """
    Optimize for top-K player ranking accuracy (most important for FPL).

    FPL optimization cares about:
    - Getting the top 50 players ranked correctly (your squad + bench)
    - Captain pick (top 1-3)
    - Differentials (high xP, low ownership)

    This scorer:
    1. Only penalizes errors in top-K predictions
    2. Uses ranking loss (Spearman-like)
    3. Extra weight on top-3 (captain candidates)

    Args:
        y_true: Actual points
        y_pred: Predicted xP
        k: Number of top players to focus on (default: 50)
        sample_weight: Optional position weights

    Returns:
        Combined ranking + captain score (0-1, higher is better)
    """
    # Get top K players by actual points
    top_k_indices = np.argsort(y_true)[-k:]

    # Extract top K actual and predicted
    y_true_topk = y_true[top_k_indices]
    y_pred_topk = y_pred[top_k_indices]

    # Calculate ranking correlation for top K
    if len(y_true_topk) < 3:
        return 0.0

    correlation, _ = spearmanr(y_true_topk, y_pred_topk)

    # Penalize MAE on top-3 (captain candidates) more heavily
    top_3_indices = np.argsort(y_true)[-3:]
    y_true_top3 = y_true[top_3_indices]
    y_pred_top3 = y_pred[top_3_indices]
    mae_top3 = np.abs(y_true_top3 - y_pred_top3).mean()

    # Combined score: 80% ranking correlation, 20% captain accuracy
    ranking_score = (correlation + 1) / 2  # Scale to 0-1
    captain_score = 1.0 / (1.0 + mae_top3)  # Lower MAE = higher score

    combined_score = 0.8 * ranking_score + 0.2 * captain_score

    return combined_score


def fpl_captain_pick_scorer(y_true, y_pred, sample_weight=None):
    """
    Optimize purely for captain pick accuracy.

    In FPL, getting the captain right is worth 2x the points.
    This scorer only cares about correctly identifying the top scorer.

    Args:
        y_true: Actual points
        y_pred: Predicted xP
        sample_weight: Ignored

    Returns:
        1.0 if top predicted = top actual, else overlap score (0-0.9)
    """
    # Get top scorer (captain candidate)
    actual_captain = np.argmax(y_true)
    predicted_captain = np.argmax(y_pred)

    # Binary: did we get it right?
    if actual_captain == predicted_captain:
        return 1.0

    # If wrong, score based on top-3 overlap
    top_3_actual = np.argsort(y_true)[-3:]
    top_3_predicted = np.argsort(y_pred)[-3:]

    overlap = len(set(top_3_actual) & set(top_3_predicted))
    return (overlap / 3.0) * 0.9  # 0.0 (no overlap) to 0.9 (2/3 overlap)


def fpl_hauler_capture_scorer(y_true, y_pred, sample_weight=None):
    """
    FPL Hauler Capture Rate - optimizes for the real FPL objective:
    "Find the players who will score big."

    Components:
    1. Hauler Precision@15: Of top-15 predicted, how many hauled (8+ pts)?
    2. Point Efficiency: What % of optimal points did top-15 capture?
    3. Captain Proximity: How close was your #1 to the actual best?

    All components are smooth (not binary) for better optimization.

    Args:
        y_true: Actual points
        y_pred: Predicted xP
        sample_weight: Optional (ignored, for sklearn compatibility)

    Returns:
        Combined score (0-1, higher is better)
    """
    # Define haulers (8+ points - realistic "good return" threshold)
    HAULER_THRESHOLD = 8
    K = 15  # Squad size

    hauler_mask = y_true >= HAULER_THRESHOLD
    hauler_indices = set(np.where(hauler_mask)[0])
    n_haulers = len(hauler_indices)

    # Top-K predictions
    top_k_pred = np.argsort(y_pred)[-K:]
    top_k_pred_set = set(top_k_pred)

    # 1. HAULER PRECISION (40%)
    # What fraction of your top-15 were actual haulers?
    if n_haulers > 0:
        found_haulers = len(top_k_pred_set & hauler_indices)
        hauler_precision = found_haulers / min(K, n_haulers)
        hauler_recall = found_haulers / n_haulers
        # F0.5 (precision-weighted F-score, we care more about precision)
        if hauler_precision + hauler_recall > 0:
            hauler_f05 = (
                1.25
                * hauler_precision
                * hauler_recall
                / (0.25 * hauler_precision + hauler_recall)
            )
        else:
            hauler_f05 = 0.0
    else:
        # No haulers this batch - fall back to ranking correlation
        corr, _ = spearmanr(y_true, y_pred)
        hauler_f05 = (corr + 1) / 2  # Scale to 0-1

    # 2. POINT EFFICIENCY (40%)
    # How close to optimal points did your top-15 capture?
    captured_points = y_true[top_k_pred].sum()
    optimal_points = np.sort(y_true)[-K:].sum()
    point_efficiency = captured_points / optimal_points if optimal_points > 0 else 0.0

    # 3. CAPTAIN PROXIMITY (20%)
    # Smooth score based on how your #1 pick ranked in actuals
    captain_pred_idx = np.argmax(y_pred)
    actual_ranks = np.argsort(np.argsort(y_true))  # Rank of each player
    captain_actual_rank = actual_ranks[captain_pred_idx]
    n_players = len(y_true)

    # Normalize: rank 0 (worst) = 0, rank n-1 (best) = 1
    captain_score = captain_actual_rank / (n_players - 1) if n_players > 1 else 0.0

    # Combine (higher is better)
    combined = 0.40 * hauler_f05 + 0.40 * point_efficiency + 0.20 * captain_score

    return combined


def fpl_hauler_ceiling_scorer(y_true, y_pred, sample_weight=None):
    """
    Hauler-optimized scorer with variance preservation.

    Based on empirical analysis of top 1% FPL managers:
    - Top managers average 18.7 captain points vs 15.0 for average
    - They capture 3.0 haulers per GW vs 2.4 for average
    - Key insight: They don't just find haulers, they find the OPTIMAL haulers

    This scorer prevents the common ML problem of "variance compression" where
    models predict ~5-7 xP for everyone, missing explosive hauls.

    Components:
    1. Hauler capture (50%): Identify the 8+ point scorers
    2. Captain accuracy (30%): Get the top scorer right
    3. Variance preservation (20%): Don't compress predictions to the mean

    Args:
        y_true: Actual points
        y_pred: Predicted xP
        sample_weight: Optional (ignored, for sklearn compatibility)

    Returns:
        Combined score (0-1, higher is better)
    """
    # 1. HAULER CAPTURE (50%)
    # Use existing hauler capture scorer
    hauler_score = fpl_hauler_capture_scorer(y_true, y_pred)

    # 2. CAPTAIN ACCURACY (30%)
    # Use existing captain scorer
    captain_score = fpl_captain_pick_scorer(y_true, y_pred)

    # 3. VARIANCE PRESERVATION (20%)
    # Penalize models that compress predictions to the mean
    # Top 1% managers find the spread - they need models that preserve variance
    actual_variance = np.var(y_true)
    pred_variance = np.var(y_pred)

    if actual_variance > 0:
        # Ratio of predicted to actual variance
        variance_ratio = pred_variance / actual_variance

        # Optimal: variance_ratio = 1.0 (perfect preservation)
        # Penalize both under-variance (compression) and over-variance
        # Under-variance is worse (missing haulers), so asymmetric penalty
        if variance_ratio < 1.0:
            # Under-variance: compression problem (common with MAE-trained models)
            # Score drops quickly as variance decreases
            variance_score = variance_ratio**0.5  # Sqrt makes it less punishing
        else:
            # Over-variance: predictions too spread out
            # Less severe penalty (better to predict some 15s than all 6s)
            variance_score = min(1.0, 2.0 - variance_ratio)
    else:
        variance_score = 0.5  # Default if actual has no variance

    # Clamp to [0, 1]
    variance_score = max(0.0, min(1.0, variance_score))

    # Combine with weights emphasizing hauler identification
    # 50% hauler capture + 30% captain + 20% variance = prioritize finding haulers
    combined = 0.50 * hauler_score + 0.30 * captain_score + 0.20 * variance_score

    return combined


# Create sklearn-compatible FPL scorers
fpl_weighted_huber_scorer_sklearn = make_scorer(
    fpl_weighted_huber_scorer, greater_is_better=True
)
fpl_topk_scorer_sklearn = make_scorer(fpl_top_k_ranking_scorer, greater_is_better=True)
fpl_captain_scorer_sklearn = make_scorer(
    fpl_captain_pick_scorer, greater_is_better=True
)
fpl_hauler_capture_scorer_sklearn = make_scorer(
    fpl_hauler_capture_scorer, greater_is_better=True
)
fpl_hauler_ceiling_scorer_sklearn = make_scorer(
    fpl_hauler_ceiling_scorer, greater_is_better=True
)


# ==============================================================================
# POSITION-AWARE FPL SCORERS
# ==============================================================================


def make_position_aware_scorer_sklearn(
    scorer_func, cv_data: pd.DataFrame, formation: str = "4-4-2"
):
    """
    Create sklearn-compatible scorer for position-aware FPL scorers.

    Position-aware scorers need access to position labels from cv_data.
    This wrapper extracts position based on the indices in X during CV splits.

    Args:
        scorer_func: Position-aware scorer function (e.g., fpl_position_aware_scorer)
        cv_data: DataFrame with 'position' column (must have index matching X)
        formation: Formation string for XI-based scorers (default: "4-4-2")

    Returns:
        sklearn-compatible scorer function
    """

    def scorer(estimator, X, y):
        # X is a subset of cv_data[feature_names] after CV split
        # Extract position labels from cv_data using X's index
        if isinstance(X, pd.DataFrame):
            # Get indices from X's index (should match cv_data index for the split)
            position_labels = cv_data.loc[X.index, "position"].values
        else:
            raise ValueError(
                "Position-aware scorers require DataFrame input with index matching cv_data"
            )

        # Get predictions
        y_pred = estimator.predict(X)

        # Call the position-aware scorer
        return scorer_func(y, y_pred, position_labels, formation=formation)

    return scorer


# Factory functions for position-aware scorers (to be called with cv_data)
def create_fpl_position_aware_scorer_sklearn(cv_data: pd.DataFrame):
    """Create sklearn-compatible position-aware top-K overlap scorer."""
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent))
    from position_aware_scorer import fpl_position_aware_scorer  # noqa: E402

    return make_position_aware_scorer_sklearn(fpl_position_aware_scorer, cv_data)


def create_fpl_starting_xi_scorer_sklearn(
    cv_data: pd.DataFrame, formation: str = "4-4-2"
):
    """Create sklearn-compatible starting XI efficiency scorer."""
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent))
    from position_aware_scorer import fpl_starting_xi_scorer  # noqa: E402

    return make_position_aware_scorer_sklearn(
        fpl_starting_xi_scorer, cv_data, formation
    )


def create_fpl_comprehensive_scorer_sklearn(
    cv_data: pd.DataFrame, formation: str = "4-4-2"
):
    """Create sklearn-compatible comprehensive team scorer."""
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent))
    from position_aware_scorer import fpl_comprehensive_team_scorer  # noqa: E402

    return make_position_aware_scorer_sklearn(
        fpl_comprehensive_team_scorer, cv_data, formation
    )


# ==============================================================================
# COMPREHENSIVE FPL EVALUATION
# ==============================================================================


def evaluate_fpl_comprehensive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cv_data: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Comprehensive FPL-specific evaluation with strategic metrics.

    Includes:
    - Overall MAE/RMSE/R²
    - Ranking correlation (Spearman)
    - Top-15 selection overlap per gameweek (squad building)
    - Captain accuracy per gameweek (top-1 identification)
    - Position-specific MAE
    - Gameweek analysis with chaos detection

    Args:
        y_true: Actual total points
        y_pred: Predicted xP
        cv_data: DataFrame with 'gameweek', 'player_id', 'position' columns
        verbose: Print detailed evaluation

    Returns:
        Dictionary of comprehensive metrics
    """
    # Create evaluation dataframe
    eval_df = cv_data.copy()
    eval_df["predicted"] = y_pred
    eval_df["actual"] = y_true

    metrics = {}

    # Overall metrics
    mae = np.abs(y_pred - y_true).mean()
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
    r2 = 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()

    metrics["mae"] = mae
    metrics["rmse"] = rmse
    metrics["r2"] = r2

    if verbose:
        logger.info("\n📈 Overall Metrics:")
        logger.info(f"   MAE:  {mae:.3f} points")
        logger.info(f"   RMSE: {rmse:.3f} points")
        logger.info(f"   R²:   {r2:.3f}")

    # FPL-specific strategic metrics
    rank_corr, _ = spearmanr(eval_df["actual"], eval_df["predicted"])
    metrics["spearman_correlation"] = rank_corr

    if verbose:
        logger.info("\n🎯 FPL Strategic Metrics:")
        logger.info(f"   Spearman correlation (ranking): {rank_corr:.3f}")

    # Top-15 selection and captain accuracy per gameweek
    top15_overlaps = []
    captain_hits = []

    if verbose:
        logger.info("\n📅 Per-Gameweek Strategic Performance:")

    for gw in sorted(eval_df["gameweek"].unique()):
        gw_data = eval_df[eval_df["gameweek"] == gw]

        # Top-15 overlap
        top_15_actual = set(gw_data.nlargest(15, "actual")["player_id"])
        top_15_pred = set(gw_data.nlargest(15, "predicted")["player_id"])
        overlap = len(top_15_actual & top_15_pred)
        top15_overlaps.append(overlap)

        # Captain accuracy
        best_actual_id = gw_data.loc[gw_data["actual"].idxmax(), "player_id"]
        best_pred_id = gw_data.loc[gw_data["predicted"].idxmax(), "player_id"]
        captain_correct = 1 if best_actual_id == best_pred_id else 0
        captain_hits.append(captain_correct)

        if verbose:
            logger.info(
                f"   GW{gw}: Top-15 overlap {overlap}/15 ({100 * overlap / 15:.0f}%), "
                f"Captain {'✓' if captain_correct else '✗'}"
            )

    metrics["avg_top15_overlap"] = np.mean(top15_overlaps)
    metrics["captain_accuracy"] = np.mean(captain_hits)

    # Position-specific metrics
    if "position" in eval_df.columns:
        if verbose:
            logger.info("\n📊 Position-Specific MAE:")

        for position in ["GKP", "DEF", "MID", "FWD"]:
            pos_mask = eval_df["position"] == position
            if pos_mask.sum() > 0:
                pos_mae = np.abs(
                    eval_df.loc[pos_mask, "predicted"] - eval_df.loc[pos_mask, "actual"]
                ).mean()
                metrics[f"{position}_mae"] = pos_mae

                if verbose:
                    logger.info(f"   {position}: {pos_mae:.3f} points")

    # Gameweek-specific metrics with chaos detection
    if verbose:
        logger.info("\n📅 Gameweek Analysis (with chaos detection):")

    gw_metrics = []
    for gw in sorted(eval_df["gameweek"].unique()):
        gw_mask = eval_df["gameweek"] == gw
        gw_data = eval_df.loc[gw_mask]

        gw_mae = np.abs(gw_data["predicted"] - gw_data["actual"]).mean()

        # Chaos indicators
        actual_std = gw_data["actual"].std()
        actual_max = gw_data["actual"].max()
        high_scorers = (gw_data["actual"] >= 10).sum()
        zero_pct = 100 * (gw_data["actual"] == 0).sum() / len(gw_data)

        gw_metrics.append(
            {
                "gameweek": gw,
                "mae": gw_mae,
                "std": actual_std,
                "max": actual_max,
                "high_scorers": high_scorers,
                "zero_pct": zero_pct,
            }
        )

        # Flag chaos weeks
        chaos_flags = []
        if actual_std > 2.5:
            chaos_flags.append("high_variance")
        if actual_max >= 20:
            chaos_flags.append("extreme_haul")
        if high_scorers >= 15:
            chaos_flags.append("many_hauls")

        chaos_str = f" ⚠️ [{', '.join(chaos_flags)}]" if chaos_flags else ""

        if verbose:
            logger.info(
                f"   GW{gw}: MAE {gw_mae:.3f} | Std {actual_std:.2f} | "
                f"Max {actual_max:.0f} | 10+ pts: {high_scorers} | "
                f"0 pts: {zero_pct:.0f}%{chaos_str}"
            )

    metrics["gameweek_metrics"] = gw_metrics

    return metrics
