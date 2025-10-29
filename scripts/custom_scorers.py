"""
Custom scoring functions for FPL ML training.

Different scorers optimize for different FPL objectives:
1. fpl_weighted_huber_scorer: Balance accuracy with captain/premium focus
2. fpl_top_k_ranking_scorer: Optimize top-50 player ranking (best for team selection)
3. fpl_captain_pick_scorer: Pure captain identification focus
"""

import numpy as np
from sklearn.metrics import make_scorer


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
        Negative ranking loss (higher is better)
    """
    # Get top K players by actual points
    top_k_indices = np.argsort(y_true)[-k:]

    # Extract top K actual and predicted
    y_true_topk = y_true[top_k_indices]
    y_pred_topk = y_pred[top_k_indices]

    # Calculate ranking correlation for top K
    # Higher correlation = better ranking
    from scipy.stats import spearmanr

    if len(y_true_topk) < 3:
        # Not enough samples for correlation
        return 0.0

    correlation, _ = spearmanr(y_true_topk, y_pred_topk)

    # Penalize MAE on top-3 (captain candidates) more heavily
    top_3_indices = np.argsort(y_true)[-3:]
    y_true_top3 = y_true[top_3_indices]
    y_pred_top3 = y_pred[top_3_indices]
    mae_top3 = np.abs(y_true_top3 - y_pred_top3).mean()

    # Combined score: 80% ranking correlation, 20% captain accuracy
    # Correlation is -1 to 1, we want 0 to 1, so (correlation + 1) / 2
    ranking_score = (correlation + 1) / 2
    captain_score = 1.0 / (1.0 + mae_top3)  # Lower MAE = higher score

    combined_score = 0.8 * ranking_score + 0.2 * captain_score

    return combined_score  # Already positive, higher is better


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
        1.0 if top predicted = top actual, else negative MAE on top-3
    """
    # Get top scorer (captain candidate)
    actual_captain = np.argmax(y_true)
    predicted_captain = np.argmax(y_pred)

    # Binary: did we get it right?
    if actual_captain == predicted_captain:
        return 1.0

    # If wrong, penalize by how far off we were in top-3
    top_3_actual = np.argsort(y_true)[-3:]
    top_3_predicted_ranks = np.argsort(y_pred)[-3:]

    # How many top-3 actual are in top-3 predicted?
    overlap = len(set(top_3_actual) & set(top_3_predicted_ranks))

    # Score: 0.0 (no overlap) to 0.9 (2/3 overlap)
    return (overlap / 3.0) * 0.9


# Create sklearn-compatible scorers
fpl_topk_scorer = make_scorer(fpl_top_k_ranking_scorer, greater_is_better=True)
fpl_captain_scorer = make_scorer(fpl_captain_pick_scorer, greater_is_better=True)
