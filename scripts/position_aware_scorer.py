"""
Position-aware FPL scorers that respect positional constraints.

FPL team building requires selecting top players WITHIN each position:
- 1 GK (always)
- 3-5 DEF (formation dependent)
- 2-5 MID (formation dependent)
- 1-3 FWD (formation dependent)

Total: 11 starters (not 15)

This module provides scorers that evaluate model performance for realistic
FPL team selection scenarios.
"""

import numpy as np
import pandas as pd


def fpl_position_aware_scorer(
    y_true,
    y_pred,
    position_labels,
    top_k_per_position={"GKP": 5, "DEF": 10, "MID": 10, "FWD": 10},
    sample_weight=None,
):
    """
    Evaluate model for position-specific top-K accuracy.

    FPL team selection requires:
    1. Identify top GKs (need 2 total, 1 starter)
    2. Identify top DEFs (need 5 total, 3-5 starters)
    3. Identify top MIDs (need 5 total, 2-5 starters)
    4. Identify top FWDs (need 3 total, 1-3 starters)

    This scorer measures overlap in top-K players per position.

    Args:
        y_true: Actual points (1D array)
        y_pred: Predicted points (1D array)
        position_labels: Position for each player (1D array: "GKP", "DEF", "MID", "FWD")
        top_k_per_position: Dict of how many top players to evaluate per position
        sample_weight: Optional sample weights (ignored)

    Returns:
        Weighted average of position-specific top-K overlaps (0-1, higher better)
    """
    if len(y_true) != len(y_pred) != len(position_labels):
        raise ValueError("y_true, y_pred, and position_labels must have same length")

    position_scores = []
    position_weights = {
        "GKP": 0.10,  # 1 starter, less important
        "DEF": 0.30,  # 3-5 starters, very important
        "MID": 0.35,  # 2-5 starters, most important (captains often MID)
        "FWD": 0.25,  # 1-3 starters, important (captains often FWD)
    }

    for position in ["GKP", "DEF", "MID", "FWD"]:
        # Get indices for this position
        pos_mask = position_labels == position
        pos_indices = np.where(pos_mask)[0]

        if len(pos_indices) == 0:
            continue  # No players in this position

        # Extract position-specific values
        y_true_pos = y_true[pos_mask]
        y_pred_pos = y_pred[pos_mask]

        # Get top-K for this position
        k = min(top_k_per_position.get(position, 10), len(pos_indices))

        # Top-K by actual points
        top_k_actual_pos_indices = np.argsort(y_true_pos)[-k:]
        top_k_actual = set(pos_indices[top_k_actual_pos_indices])

        # Top-K by predicted points
        top_k_pred_pos_indices = np.argsort(y_pred_pos)[-k:]
        top_k_pred = set(pos_indices[top_k_pred_pos_indices])

        # Overlap
        overlap = len(top_k_actual & top_k_pred)
        overlap_score = overlap / k

        position_scores.append((position, overlap_score, position_weights[position]))

    # Weighted average
    total_weight = sum(weight for _, _, weight in position_scores)
    weighted_score = (
        sum(score * weight for _, score, weight in position_scores) / total_weight
    )

    return weighted_score


def fpl_starting_xi_scorer(
    y_true,
    y_pred,
    position_labels,
    formation="4-4-2",
    sample_weight=None,
):
    """
    Evaluate model for selecting optimal starting XI given a formation.

    Simulates actual FPL team selection:
    1. Pick 1 GK
    2. Pick N DEF (based on formation)
    3. Pick N MID (based on formation)
    4. Pick N FWD (based on formation)
    5. Calculate total points from actual starters

    Args:
        y_true: Actual points
        y_pred: Predicted points
        position_labels: Position for each player
        formation: String like "4-4-2", "3-5-2", etc.
        sample_weight: Optional sample weights (ignored)

    Returns:
        Ratio of achieved points to optimal points (0-1, higher better)
    """
    # Parse formation
    def_count, mid_count, fwd_count = map(int, formation.split("-"))

    selected_indices = []

    # Select best players per position based on predictions
    for position, count in [
        ("GKP", 1),
        ("DEF", def_count),
        ("MID", mid_count),
        ("FWD", fwd_count),
    ]:
        pos_mask = position_labels == position
        pos_indices = np.where(pos_mask)[0]

        if len(pos_indices) == 0:
            return 0.0  # Can't form valid team

        y_pred_pos = y_pred[pos_mask]

        # Get top N by prediction
        top_n_pos_indices = np.argsort(y_pred_pos)[-count:]
        selected_indices.extend(pos_indices[top_n_pos_indices])

    # Calculate actual points from selected team
    actual_points_selected = y_true[selected_indices].sum()

    # Calculate optimal points (if we had perfect predictions)
    optimal_indices = []
    for position, count in [
        ("GKP", 1),
        ("DEF", def_count),
        ("MID", mid_count),
        ("FWD", fwd_count),
    ]:
        pos_mask = position_labels == position
        pos_indices = np.where(pos_mask)[0]

        y_true_pos = y_true[pos_mask]

        # Get top N by actual
        top_n_pos_indices = np.argsort(y_true_pos)[-count:]
        optimal_indices.extend(pos_indices[top_n_pos_indices])

    optimal_points = y_true[optimal_indices].sum()

    # Return ratio (how close to optimal)
    if optimal_points == 0:
        return 0.0

    return actual_points_selected / optimal_points


def fpl_captain_within_xi_scorer(
    y_true,
    y_pred,
    position_labels,
    formation="4-4-2",
    sample_weight=None,
):
    """
    Evaluate captain selection accuracy within a realistic starting XI.

    FPL captain selection is constrained:
    - Must be in your starting XI
    - Gets 2x points

    This scorer:
    1. Forms optimal starting XI based on predictions
    2. Captains highest predicted xP within that XI
    3. Checks if captain is the actual top scorer in that XI

    Args:
        y_true: Actual points
        y_pred: Predicted points
        position_labels: Position for each player
        formation: Starting XI formation
        sample_weight: Optional sample weights (ignored)

    Returns:
        1.0 if captain is top scorer in XI, else 0.0-0.8 based on rank
    """
    # Parse formation
    def_count, mid_count, fwd_count = map(int, formation.split("-"))

    # Select XI based on predictions
    selected_indices = []
    for position, count in [
        ("GKP", 1),
        ("DEF", def_count),
        ("MID", mid_count),
        ("FWD", fwd_count),
    ]:
        pos_mask = position_labels == position
        pos_indices = np.where(pos_mask)[0]

        if len(pos_indices) == 0:
            return 0.0

        y_pred_pos = y_pred[pos_mask]
        top_n_pos_indices = np.argsort(y_pred_pos)[-count:]
        selected_indices.extend(pos_indices[top_n_pos_indices])

    # Captain = highest predicted xP in XI
    xi_pred = y_pred[selected_indices]
    captain_xi_idx = np.argmax(xi_pred)
    captain_idx = selected_indices[captain_xi_idx]

    # Actual top scorer in XI
    xi_actual = y_true[selected_indices]
    actual_best_xi_idx = np.argmax(xi_actual)
    actual_best_idx = selected_indices[actual_best_xi_idx]

    # Perfect captain pick
    if captain_idx == actual_best_idx:
        return 1.0

    # Partial credit based on how high captain actually scored
    captain_actual_rank = np.where(np.argsort(xi_actual) == captain_xi_idx)[0][0]
    # Rank 10 (top) = 1.0, rank 9 = 0.9, ..., rank 0 (worst) = 0.0
    return captain_actual_rank / 10.0 * 0.8  # Max 0.8 if not perfect


def fpl_comprehensive_team_scorer(
    y_true,
    y_pred,
    position_labels,
    formation="4-4-2",
    sample_weight=None,
):
    """
    Combined FPL scorer that evaluates:
    1. Position-aware top-10 overlap (40%)
    2. Starting XI point efficiency (40%)
    3. Captain accuracy within XI (20%)

    This is the most realistic FPL evaluation metric.

    Args:
        y_true: Actual points
        y_pred: Predicted points
        position_labels: Position for each player
        formation: Formation for XI selection
        sample_weight: Optional sample weights (ignored)

    Returns:
        Weighted combination of all FPL-relevant metrics (0-1, higher better)
    """
    # Component 1: Position-aware top-K overlap
    top_k_score = fpl_position_aware_scorer(
        y_true,
        y_pred,
        position_labels,
        top_k_per_position={"GKP": 5, "DEF": 10, "MID": 10, "FWD": 10},
    )

    # Component 2: Starting XI efficiency
    xi_score = fpl_starting_xi_scorer(
        y_true, y_pred, position_labels, formation=formation
    )

    # Component 3: Captain accuracy
    captain_score = fpl_captain_within_xi_scorer(
        y_true, y_pred, position_labels, formation=formation
    )

    # Weighted combination
    combined_score = 0.40 * top_k_score + 0.40 * xi_score + 0.20 * captain_score

    return combined_score


# Sklearn-compatible scorers (require position_labels to be passed via cv_data)
# These need to be wrapped differently since they require position_labels


def make_position_aware_scorer(scorer_func, formation="4-4-2"):
    """
    Create sklearn-compatible scorer that extracts position labels from cv_data.

    Usage in GridSearchCV:
        scorer = make_position_aware_scorer(fpl_position_aware_scorer)
        GridSearchCV(..., scoring=scorer)

    Note: This requires the cv_data DataFrame to have a 'position' column.
    """

    def scorer_wrapper(estimator, X, y):
        # X should be a DataFrame with 'position' column
        if isinstance(X, pd.DataFrame) and "position" in X.columns:
            position_labels = X["position"].values
            # Remove position column before prediction
            X_features = X.drop(columns=["position"])
        else:
            raise ValueError(
                "X must be DataFrame with 'position' column for position-aware scoring"
            )

        y_pred = estimator.predict(X_features)
        return scorer_func(y, y_pred, position_labels, formation=formation)

    return scorer_wrapper


# For manual evaluation (not sklearn GridSearch)
def evaluate_position_aware_metrics(y_true, y_pred, position_labels, formations=None):
    """
    Evaluate all position-aware metrics for a trained model.

    Args:
        y_true: Actual points
        y_pred: Predicted points
        position_labels: Position array
        formations: List of formations to test (default: common FPL formations)

    Returns:
        Dict of metric scores
    """
    if formations is None:
        formations = ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1"]

    metrics = {}

    # Position-aware top-K overlap
    metrics["position_topk_overlap"] = fpl_position_aware_scorer(
        y_true,
        y_pred,
        position_labels,
        top_k_per_position={"GKP": 5, "DEF": 10, "MID": 10, "FWD": 10},
    )

    # Per-formation metrics
    for formation in formations:
        metrics[f"xi_efficiency_{formation}"] = fpl_starting_xi_scorer(
            y_true, y_pred, position_labels, formation=formation
        )
        metrics[f"captain_accuracy_{formation}"] = fpl_captain_within_xi_scorer(
            y_true, y_pred, position_labels, formation=formation
        )

    # Comprehensive score (using most common formation 4-4-2)
    metrics["comprehensive_442"] = fpl_comprehensive_team_scorer(
        y_true, y_pred, position_labels, formation="4-4-2"
    )

    return metrics


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Simulate 100 players
    n_players = 100
    positions = np.array(["GKP"] * 10 + ["DEF"] * 30 + ["MID"] * 35 + ["FWD"] * 25)

    # Simulate actual points (position-dependent)
    y_true = np.concatenate(
        [
            np.random.poisson(3, 10),  # GKP: ~3 pts avg
            np.random.poisson(4, 30),  # DEF: ~4 pts avg
            np.random.poisson(5, 35),  # MID: ~5 pts avg
            np.random.poisson(6, 25),  # FWD: ~6 pts avg
        ]
    ).astype(float)

    # Simulate predictions (with noise)
    y_pred = y_true + np.random.normal(0, 1.5, n_players)

    # Evaluate
    print("Position-Aware FPL Scorer Demo")
    print("=" * 60)

    metrics = evaluate_position_aware_metrics(y_true, y_pred, positions)

    for metric, score in metrics.items():
        print(f"{metric:30s}: {score:.3f}")
