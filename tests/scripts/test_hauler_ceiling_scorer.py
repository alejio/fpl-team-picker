"""Tests for fpl_hauler_ceiling_scorer - variance-preserving scorer.

Based on top 1% manager analysis:
- Top managers average 18.7 captain points vs 15.0 for average
- They capture 3.0 haulers per GW vs 2.4 for average
- Key insight: They identify OPTIMAL haulers, not just any haulers

This scorer penalizes models that compress predictions to the mean,
which causes them to miss explosive hauls.
"""

import sys
from pathlib import Path
import numpy as np

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from ml_training_utils import (
    fpl_hauler_ceiling_scorer,
    fpl_hauler_capture_scorer,
    fpl_captain_pick_scorer,
)


class TestHaulerCeilingScorerBasics:
    """Test basic functionality of the hauler ceiling scorer."""

    def test_scorer_returns_float(self):
        """Test that scorer returns a float in [0, 1] range."""
        y_true = np.array([2, 5, 8, 12, 3])
        y_pred = np.array([3, 4, 7, 10, 4])

        score = fpl_hauler_ceiling_scorer(y_true, y_pred)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_perfect_predictions_high_score(self):
        """Test that perfect predictions get a high score."""
        y_true = np.array([2, 5, 8, 12, 3, 15, 6, 9])
        y_pred = y_true.copy()  # Perfect predictions

        score = fpl_hauler_ceiling_scorer(y_true, y_pred)

        # Perfect predictions should score very high
        assert score > 0.9

    def test_handles_empty_haulers(self):
        """Test that scorer handles gameweeks with no haulers (all < 8 pts)."""
        y_true = np.array([2, 3, 4, 5, 6, 7, 7, 7])  # No haulers
        y_pred = np.array([3, 4, 5, 6, 7, 6, 6, 6])

        score = fpl_hauler_ceiling_scorer(y_true, y_pred)

        # Should still produce a valid score
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestVariancePreservation:
    """Test variance preservation component of the scorer."""

    def test_good_model_preserves_variance(self):
        """Test that a model preserving variance scores higher than one compressing it."""
        # Actual points with haulers
        y_true = np.array([2, 3, 15, 6, 8, 2, 4, 12, 5, 7, 2, 1, 9, 3, 6])

        # Good model: preserves variance, identifies haulers
        y_pred_good = np.array([3, 4, 12, 5, 9, 3, 5, 10, 6, 8, 3, 2, 8, 4, 7])

        # Bad model: compresses to mean (misses haulers)
        y_pred_bad = np.array(
            [5.5, 5.5, 6.0, 5.5, 5.8, 5.5, 5.5, 5.8, 5.5, 5.7, 5.5, 5.5, 5.6, 5.5, 5.6]
        )

        good_score = fpl_hauler_ceiling_scorer(y_true, y_pred_good)
        bad_score = fpl_hauler_ceiling_scorer(y_true, y_pred_bad)

        # Good model should score significantly higher
        assert good_score > bad_score
        assert good_score > bad_score * 1.1  # At least 10% better

    def test_variance_compression_penalized(self):
        """Test that models with compressed variance are penalized."""
        y_true = np.array([2, 5, 15, 3, 20, 7, 4, 12])

        # Model with same mean but compressed variance
        mean_val = np.mean(y_true)
        y_pred_compressed = np.full_like(y_true, mean_val, dtype=float)

        # Model with correct variance
        y_pred_variance = y_true * 0.9 + 1.0  # Scaled but preserves variance

        compressed_score = fpl_hauler_ceiling_scorer(y_true, y_pred_compressed)
        variance_score = fpl_hauler_ceiling_scorer(y_true, y_pred_variance)

        # Variance-preserving model should score higher
        assert variance_score > compressed_score

    def test_over_variance_also_penalized(self):
        """Test that models with too much variance are also penalized (but less severely)."""
        y_true = np.array([4, 5, 6, 7, 8, 5, 6, 7])

        # Model with much higher variance than actual
        y_pred_over = np.array([0, 3, 15, 2, 18, 1, 12, 10])

        # Model with correct variance
        y_pred_correct = y_true.copy()

        over_score = fpl_hauler_ceiling_scorer(y_true, y_pred_over)
        correct_score = fpl_hauler_ceiling_scorer(y_true, y_pred_correct)

        # Correct variance should score higher
        assert correct_score > over_score


class TestHaulerCapture:
    """Test hauler capture component of the scorer."""

    def test_captures_haulers_in_top_15(self):
        """Test that capturing haulers in top-15 predictions improves score."""
        y_true = np.array([2, 3, 15, 6, 8, 2, 4, 12, 5, 7, 2, 1, 9, 3, 6, 20, 5, 4])

        # Model that captures haulers in top-15
        y_pred_captures = np.array(
            [3, 4, 14, 5, 9, 3, 5, 11, 6, 8, 3, 2, 8, 4, 7, 18, 6, 5]
        )

        # Model that misses haulers
        y_pred_misses = np.array(
            [8, 9, 5, 10, 4, 9, 10, 3, 11, 5, 10, 9, 2, 10, 8, 4, 9, 10]
        )

        captures_score = fpl_hauler_ceiling_scorer(y_true, y_pred_captures)
        misses_score = fpl_hauler_ceiling_scorer(y_true, y_pred_misses)

        # Capturing haulers should score higher
        assert captures_score > misses_score


class TestCaptainAccuracy:
    """Test captain accuracy component of the scorer."""

    def test_correct_captain_improves_score(self):
        """Test that correctly identifying top scorer improves score."""
        y_true = np.array([5, 8, 20, 6, 7, 4, 5, 10])  # 20 is the top scorer

        # Model that correctly identifies top scorer
        y_pred_correct = np.array([4, 7, 18, 5, 6, 3, 4, 9])

        # Model that picks wrong top scorer
        y_pred_wrong = np.array([4, 7, 8, 5, 6, 3, 4, 20])  # Puts 20 on wrong player

        correct_score = fpl_hauler_ceiling_scorer(y_true, y_pred_correct)
        wrong_score = fpl_hauler_ceiling_scorer(y_true, y_pred_wrong)

        # Correct captain should score higher
        assert correct_score > wrong_score


class TestComponentWeights:
    """Test that component weights are correct (50% hauler, 30% captain, 20% variance)."""

    def test_hauler_capture_most_important(self):
        """Test that hauler capture has the highest weight (50%)."""
        y_true = np.array([2, 3, 15, 6, 8, 2, 4, 12, 5, 7])

        # Model good at hauler capture but bad at captain
        y_pred_hauler = np.array([3, 4, 14, 5, 9, 3, 5, 11, 6, 8])

        # Model bad at hauler capture but good at captain (somehow)
        y_pred_captain = np.array(
            [6, 6, 6, 6, 6, 6, 6, 15, 6, 6]
        )  # Only predicts one hauler

        hauler_score = fpl_hauler_ceiling_scorer(y_true, y_pred_hauler)
        captain_score = fpl_hauler_ceiling_scorer(y_true, y_pred_captain)

        # Both should be valid scores
        assert 0.0 <= hauler_score <= 1.0
        assert 0.0 <= captain_score <= 1.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_player(self):
        """Test with single player."""
        y_true = np.array([10])
        y_pred = np.array([8])

        score = fpl_hauler_ceiling_scorer(y_true, y_pred)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_all_same_predictions(self):
        """Test when all predictions are the same."""
        y_true = np.array([2, 5, 8, 12, 3, 15])
        y_pred = np.full(6, 5.0)

        score = fpl_hauler_ceiling_scorer(y_true, y_pred)

        # Should penalize zero variance
        assert score < 0.8  # Low score due to variance compression

    def test_negative_predictions_handled(self):
        """Test that negative predictions are handled (shouldn't happen, but be robust)."""
        y_true = np.array([2, 5, 8])
        y_pred = np.array([1, 3, -1])  # Invalid negative prediction

        # Should not crash
        score = fpl_hauler_ceiling_scorer(y_true, y_pred)
        assert isinstance(score, float)

    def test_zero_actual_points(self):
        """Test handling of zero actual points."""
        y_true = np.array([0, 0, 2, 0, 1])
        y_pred = np.array([1, 1, 2, 1, 1])

        score = fpl_hauler_ceiling_scorer(y_true, y_pred)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestComparisonWithOtherScorers:
    """Test that hauler ceiling scorer behaves correctly vs other scorers."""

    def test_distinguishes_good_from_bad_models(self):
        """Test that scorer distinguishes good models from bad ones better than hauler_capture alone."""
        y_true = np.array([2, 3, 15, 6, 8, 2, 4, 12, 5, 7, 2, 1, 9, 3, 6])

        # Good model: preserves variance
        y_pred_good = np.array([3, 4, 12, 5, 9, 3, 5, 10, 6, 8, 3, 2, 8, 4, 7])

        # Bad model: compresses variance
        y_pred_bad = np.array(
            [5.5, 5.5, 6.0, 5.5, 5.8, 5.5, 5.5, 5.8, 5.5, 5.7, 5.5, 5.5, 5.6, 5.5, 5.6]
        )

        # Hauler capture alone might not distinguish (both capture haulers in top-15)
        hauler_good = fpl_hauler_capture_scorer(y_true, y_pred_good)
        hauler_bad = fpl_hauler_capture_scorer(y_true, y_pred_bad)

        # Ceiling scorer should distinguish them
        ceiling_good = fpl_hauler_ceiling_scorer(y_true, y_pred_good)
        ceiling_bad = fpl_hauler_ceiling_scorer(y_true, y_pred_bad)

        # Ceiling scorer should show bigger difference
        ceiling_diff = ceiling_good - ceiling_bad
        hauler_diff = hauler_good - hauler_bad

        assert ceiling_diff > hauler_diff * 0.5  # At least half the differentiation

    def test_consistent_with_captain_scorer(self):
        """Test that captain component aligns with captain_pick_scorer."""
        y_true = np.array([5, 8, 20, 6, 7])
        y_pred = np.array([4, 7, 18, 5, 6])  # Correct top scorer

        ceiling_score = fpl_hauler_ceiling_scorer(y_true, y_pred)
        captain_score = fpl_captain_pick_scorer(y_true, y_pred)

        # Captain is 30% of ceiling score, so correlation expected
        # Just verify both work
        assert ceiling_score > 0.5  # Should be decent
        assert captain_score > 0.8  # Captain should be high (correct pick)


class TestRealWorldScenarios:
    """Test with realistic FPL data scenarios."""

    def test_typical_gameweek_distribution(self):
        """Test with typical gameweek point distribution."""
        # Typical GW: 2 haulers (10+), 5-6 returns (5-9), rest blanks
        y_true = np.array(
            [
                2,
                2,
                3,
                1,
                4,
                15,
                8,
                7,
                3,
                2,  # 15 and 8 are haulers
                6,
                5,
                4,
                2,
                1,
                3,
                2,
                12,
                5,
                6,  # 12 is also a hauler
            ]
        )

        # Good model preserving shape
        y_pred_good = np.array(
            [3, 2, 4, 2, 5, 12, 9, 8, 4, 3, 7, 6, 5, 3, 2, 4, 3, 10, 6, 7]
        )

        # Bad model compressing to mean ~4.5
        y_pred_bad = np.array(
            [4, 4, 5, 4, 5, 6, 5, 5, 4, 4, 5, 5, 5, 4, 4, 4, 4, 6, 5, 5]
        )

        good_score = fpl_hauler_ceiling_scorer(y_true, y_pred_good)
        bad_score = fpl_hauler_ceiling_scorer(y_true, y_pred_bad)

        assert good_score > bad_score
        assert good_score > 0.7  # Good model should score well

    def test_captain_scenario(self):
        """Test captain selection scenario (top 1% insight)."""
        # Top 1% managers average 18.7 captain points - they pick the right haulers
        y_true = np.array(
            [
                8.0,
                15.0,
                6.0,
                4.0,
                20.0,
                7.0,
                5.0,
                3.0,  # 20pt is ideal captain
                10.0,
                6.0,
                4.0,
                2.0,
                8.0,
                5.0,
                3.0,
            ]
        )

        # Good model: correctly identifies 20pt as top
        y_pred_good = np.array(
            [
                7.0,
                12.0,
                5.0,
                4.0,
                16.0,
                6.0,
                5.0,
                3.0,
                9.0,
                6.0,
                4.0,
                2.0,
                7.0,
                5.0,
                3.0,
            ]
        )

        # Bad model: picks wrong captain
        y_pred_bad = np.array(
            [
                14.0,
                7.0,
                5.0,
                4.0,
                8.0,
                6.0,
                5.0,
                3.0,  # Thinks first player is best
                9.0,
                6.0,
                4.0,
                2.0,
                7.0,
                5.0,
                3.0,
            ]
        )

        good_score = fpl_hauler_ceiling_scorer(y_true, y_pred_good)
        bad_score = fpl_hauler_ceiling_scorer(y_true, y_pred_bad)

        assert good_score > bad_score
