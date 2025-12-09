"""Simple probabilistic calibration of ML xP predictions.

Uses empirical distributions to adjust ML predictions based on
player quality × fixture difficulty interaction.

NOTE: This service uses simple blend weighting (NOT precision weighting) because:
- ML uncertainty (tree variance) measures model disagreement, not prediction bias
- Distribution std measures outcome variance (FPL points are inherently noisy)
- These are fundamentally different quantities - precision weighting would give
  ML ~99% weight, making calibration useless
- Simple blend weighting allows controllable calibration strength
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from pathlib import Path
from loguru import logger


class XPCalibrationService:
    """Simple probabilistic calibration of ML xP predictions.

    Uses empirical distributions to adjust ML predictions based on
    player quality × fixture difficulty interaction.

    The calibration addresses a specific problem: ML models underweight
    fixture difficulty for premium players because fixture difficulty
    is just 1 of 150+ features. This service uses historical data to
    learn the (tier × fixture) interaction and applies a correction.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize calibration service.

        Args:
            config: Optional configuration dictionary. If not provided, uses defaults.
                Keys:
                - premium_price_threshold: Price threshold for premium tier (default: 8.0)
                - mid_price_threshold: Price threshold for mid tier (default: 6.0)
                - easy_fixture_threshold: Fixture difficulty threshold for easy (default: 1.538)
                    Note: This should match config.fixture_difficulty.easy_fixture_threshold
                - distributions_path: Path to distributions.json (default: data/calibration/distributions.json)
                - minimum_sample_size: Minimum samples required (default: 30)
                - empirical_blend_weight: Weight for empirical distribution (default: 0.2)
                - risk_adjustment_multiplier: Std multiplier for risk profiles (default: 0.67)
                - debug: Enable debug logging (default: False)
        """
        self.config = config or {}

        # Load configuration with defaults
        self.premium_price_threshold = self.config.get("premium_price_threshold", 8.0)
        self.mid_price_threshold = self.config.get("mid_price_threshold", 6.0)
        # easy_fixture_threshold should be sourced from fixture_difficulty config
        # but we accept it here for backwards compatibility and flexibility
        self.easy_fixture_threshold = self.config.get("easy_fixture_threshold", 1.538)
        self.minimum_sample_size = self.config.get("minimum_sample_size", 30)

        # Blend weight: how much of the fixture effect to apply
        # 0.5 = apply 50% of the historical (tier × fixture) effect
        # This creates meaningful adjustments while being conservative
        self.empirical_blend_weight = self.config.get("empirical_blend_weight", 0.5)

        # Risk adjustment: how many std deviations for conservative/risk-taking
        self.risk_adjustment_multiplier = self.config.get(
            "risk_adjustment_multiplier", 0.67
        )

        self.debug = self.config.get("debug", False)

        # Get distributions path (relative to project root)
        distributions_path = self.config.get(
            "distributions_path", "data/calibration/distributions.json"
        )

        # Resolve path relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        self.distributions_path = project_root / distributions_path

        # Load distributions and priors
        self.distributions = self._load_distributions()
        self.priors = self._get_priors()

        if self.debug:
            logger.info(
                f"✅ XPCalibrationService initialized - "
                f"Loaded {len(self.distributions)} distributions from {self.distributions_path}, "
                f"empirical_blend_weight={self.empirical_blend_weight:.2f}"
            )

    def _load_distributions(self) -> Dict[str, Dict[str, Any]]:
        """Load fitted distributions from JSON file.

        Returns:
            Dictionary mapping combination keys (e.g., "premium_easy") to distribution parameters.
        """
        try:
            with open(self.distributions_path, "r") as f:
                distributions = json.load(f)

            if self.debug:
                logger.debug(f"Loaded distributions: {list(distributions.keys())}")

            return distributions
        except FileNotFoundError:
            logger.warning(
                f"⚠️  Distributions file not found at {self.distributions_path}. "
                f"Calibration will use priors only."
            )
            return {}
        except json.JSONDecodeError as e:
            logger.error(
                f"❌ Error parsing distributions file: {e}. "
                f"Calibration will use priors only."
            )
            return {}

    def _get_priors(self) -> Dict[str, Any]:
        """Get prior distributions based on domain knowledge.

        Returns:
            Dictionary with tier means and fixture boosts.
        """
        return {
            "tier_means": {
                "premium": 8.0,  # Premium players average ~8 points
                "mid": 6.0,  # Mid players average ~6 points
                "budget": 4.5,  # Budget players average ~4.5 points
            },
            "fixture_boosts": {
                "easy": 2.0,  # Easy fixtures: +2 points
                "hard": -1.0,  # Hard fixtures: -1 point
            },
            "dispersion": 2.0,  # Moderate overdispersion
            "default_std": 3.0,  # High uncertainty for fallback
        }

    def _get_tier_baseline(self, tier: str) -> float:
        """Calculate the baseline (average) for a tier across fixtures.

        This is used for additive fixture effect correction:
        - fixture_effect = distribution_mean - tier_baseline
        - The effect captures how much better/worse easy/hard fixtures are

        Args:
            tier: Player tier ("premium", "mid", "budget")

        Returns:
            Tier baseline (average of easy and hard distribution means)
        """
        easy_key = f"{tier}_easy"
        hard_key = f"{tier}_hard"

        easy_dist = self.distributions.get(easy_key, {})
        hard_dist = self.distributions.get(hard_key, {})

        # Use distribution means if available, else fall back to priors
        easy_mean = easy_dist.get("mean") if easy_dist else None
        hard_mean = hard_dist.get("mean") if hard_dist else None

        if easy_mean is not None and hard_mean is not None:
            return (easy_mean + hard_mean) / 2
        else:
            # Fallback to prior
            return self.priors["tier_means"].get(tier, 5.0)

    def _get_player_tier(self, price: float) -> str:
        """Determine player tier from price.

        Args:
            price: Player price in millions (£)

        Returns:
            Tier string: "premium", "mid", or "budget"
        """
        if price >= self.premium_price_threshold:
            return "premium"
        elif price >= self.mid_price_threshold:
            return "mid"
        else:
            return "budget"

    def _get_fixture_category(self, fixture_difficulty: float) -> str:
        """Determine fixture category (2-tier only for simplicity).

        Args:
            fixture_difficulty: Fixture difficulty score (higher = easier)

        Returns:
            Fixture category: "easy" or "hard"

        Raises:
            ValueError: If fixture_difficulty is NaN or missing
        """
        if pd.isna(fixture_difficulty):
            raise ValueError(
                "Missing fixture_difficulty value. This indicates a data problem upstream."
            )

        if fixture_difficulty >= self.easy_fixture_threshold:
            return "easy"
        else:
            return "hard"

    def calibrate_predictions(
        self,
        players_df: pd.DataFrame,
        risk_profile: str = "balanced",
        fixture_difficulty_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Calibrate ML predictions using additive fixture effect correction.

        Approach:
        1. Map player to (tier, fixture) combination
        2. Get fitted distribution for that combination
        3. Apply risk adjustment (using percentiles for skewed FPL data)
        4. Calculate fixture effect = (distribution_mean - tier_baseline)
        5. Apply additive correction: calibrated_xp = ml_xp + weight * fixture_effect

        Why ADDITIVE correction (not blending):
        - Blending toward distribution means creates weak adjustments when ML is close
        - Additive correction preserves ML as baseline while adding fixture effect
        - This captures the full historical easy/hard differential that ML underweights

        Example with premium player (empirical_weight=0.5):
        - tier_baseline = (premium_easy_mean + premium_hard_mean) / 2 = 4.0
        - easy fixture_effect = 4.62 - 4.0 = +0.62
        - hard fixture_effect = 3.37 - 4.0 = -0.63
        - ML predicts 5.0 for both → after calibration:
          - easy: 5.0 + 0.5 * 0.62 = 5.31
          - hard: 5.0 + 0.5 * (-0.63) = 4.69
        - Creates 0.62 point differential (half of 1.25 historical difference)

        Args:
            players_df: DataFrame with player data including:
                - "price": Player price in millions
                - "xP": ML predicted expected points
                - "fixture_difficulty": Fixture difficulty (can be in players_df or separate)
                - "xP_uncertainty": ML prediction uncertainty (optional, used for output)
            risk_profile: Risk profile for calibration:
                - "conservative": Use lower credible interval (~25th percentile)
                - "balanced": Use mean (default)
                - "risk-taking": Use upper credible interval (~75th percentile)
            fixture_difficulty_df: Optional DataFrame with fixture difficulty data.
                If provided, will merge with players_df. If fixture_difficulty is already
                in players_df, this is ignored.

        Returns:
            DataFrame with calibrated predictions:
                - "xP": Calibrated expected points
                - "xP_uncertainty": Combined uncertainty estimate
                - "xP_calibrated": Boolean flag indicating calibration was applied
                - "xP_raw": Original ML prediction (for comparison)
        """
        result = players_df.copy()

        # Ensure we have fixture_difficulty column
        if (
            "fixture_difficulty" not in result.columns
            and fixture_difficulty_df is not None
        ):
            # Try to merge fixture difficulty if provided separately
            if (
                "team_id" in result.columns
                and "team_id" in fixture_difficulty_df.columns
            ):
                result = result.merge(
                    fixture_difficulty_df[["team_id", "fixture_difficulty"]],
                    on="team_id",
                    how="left",
                    suffixes=("", "_merged"),
                )
                # Use merged column if original was missing
                if "fixture_difficulty_merged" in result.columns:
                    result["fixture_difficulty"] = result["fixture_difficulty_merged"]
                    result = result.drop("fixture_difficulty_merged", axis=1)

        # Validate required columns exist
        required_columns = ["price", "xP", "fixture_difficulty"]
        missing_columns = [col for col in required_columns if col not in result.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"This indicates a data problem upstream."
            )

        # Initialize calibration columns
        if "xP_calibrated" not in result.columns:
            result["xP_calibrated"] = False
        if "xP_raw" not in result.columns:
            result["xP_raw"] = result["xP"]

        # xP_uncertainty is optional - if missing, we'll calculate it from distribution std
        has_ml_uncertainty = "xP_uncertainty" in result.columns

        calibrated_count = 0
        fallback_count = 0

        # Get blend weights
        empirical_weight = self.empirical_blend_weight
        1.0 - empirical_weight
        risk_multiplier = self.risk_adjustment_multiplier

        for idx, player in result.iterrows():
            # Get player attributes (direct access - will raise KeyError if missing)
            price = player["price"]
            tier = self._get_player_tier(price)

            fixture_difficulty = player["fixture_difficulty"]
            fixture = self._get_fixture_category(fixture_difficulty)

            # Get ML prediction
            ml_xp = player["xP"]
            ml_std = player["xP_uncertainty"] if has_ml_uncertainty else 1.5

            # Get fitted distribution
            combination_key = f"{tier}_{fixture}"
            distribution = self.distributions.get(combination_key)

            if (
                not distribution
                or distribution.get("sample_size", 0) < self.minimum_sample_size
            ):
                # Fallback: use prior only
                prior_mean = (
                    self.priors["tier_means"][tier]
                    + self.priors["fixture_boosts"][fixture]
                )
                distribution_mean = prior_mean
                distribution_std = self.priors["default_std"]
                fallback_count += 1

                if self.debug and idx < 5:  # Log first few for debugging
                    logger.debug(
                        f"Fallback to prior for {combination_key}: "
                        f"mean={distribution_mean:.2f}, std={distribution_std:.2f}"
                    )
            else:
                # Use fitted distribution
                distribution_mean = distribution["mean"]
                distribution_std = distribution["std"]
                calibrated_count += 1

                if self.debug and idx < 5:  # Log first few for debugging
                    logger.debug(
                        f"Using fitted distribution for {combination_key}: "
                        f"mean={distribution_mean:.2f}, std={distribution_std:.2f}, "
                        f"sample_size={distribution['sample_size']}"
                    )

            # Apply risk adjustment BEFORE blending
            # Use actual percentiles when available (more robust for skewed FPL data)
            # Fall back to mean ± k*std only when percentiles aren't available
            if risk_profile == "conservative":
                # Conservative: use lower bound (25th percentile or interpolated)
                if distribution and "percentile_25" in distribution:
                    # Use actual 25th percentile from data
                    p25 = distribution["percentile_25"]
                    # Interpolate between mean and p25 based on risk_multiplier
                    # At multiplier=0.67, we get ~60% of the way from mean to p25
                    risk_adjusted_distribution_mean = (
                        distribution_mean - risk_multiplier * (distribution_mean - p25)
                    )
                else:
                    # Fallback: mean - k*std, but capped at 50% of mean
                    adjustment = min(
                        risk_multiplier * distribution_std, 0.5 * distribution_mean
                    )
                    risk_adjusted_distribution_mean = distribution_mean - adjustment
            elif risk_profile == "risk-taking":
                # Risk-taking: use upper bound (75th percentile or interpolated)
                if distribution and "percentile_75" in distribution:
                    # Use actual 75th percentile from data
                    p75 = distribution["percentile_75"]
                    # Interpolate between mean and p75 based on risk_multiplier
                    # At multiplier=0.67, we get ~60% of the way from mean to p75
                    risk_adjusted_distribution_mean = (
                        distribution_mean + risk_multiplier * (p75 - distribution_mean)
                    )
                else:
                    # Fallback: mean + k*std, but capped at 2x mean
                    adjustment = min(
                        risk_multiplier * distribution_std,
                        distribution_mean,  # Cap at doubling the mean
                    )
                    risk_adjusted_distribution_mean = distribution_mean + adjustment
            else:  # balanced
                risk_adjusted_distribution_mean = distribution_mean

            # Ensure non-negative
            risk_adjusted_distribution_mean = max(0.0, risk_adjusted_distribution_mean)

            # ADDITIVE FIXTURE EFFECT CORRECTION
            # Instead of blending toward distribution mean, we ADD the fixture effect
            # that ML underweights. This preserves ML as the baseline while correcting
            # for the (tier × fixture) interaction.
            #
            # fixture_effect = risk_adjusted_distribution_mean - tier_baseline
            # calibrated_xp = ml_xp + empirical_weight * fixture_effect
            #
            # Example with premium player:
            # - tier_baseline = (4.62 + 3.37) / 2 = 4.0
            # - easy fixture_effect = 4.62 - 4.0 = +0.62
            # - hard fixture_effect = 3.37 - 4.0 = -0.63
            # - With empirical_weight=0.5: easy gets +0.31, hard gets -0.32
            # - This creates a 0.63 point differential (half the historical difference)
            tier_baseline = self._get_tier_baseline(tier)
            fixture_effect = risk_adjusted_distribution_mean - tier_baseline

            calibrated_xp = ml_xp + empirical_weight * fixture_effect

            # Calculate uncertainty: ML uncertainty + scaled fixture effect uncertainty
            # Fixture effect uncertainty comes from distribution std
            calibrated_std = np.sqrt(
                ml_std**2 + (empirical_weight**2) * (distribution_std**2)
            )

            # Ensure non-negative prediction
            calibrated_xp = max(0.0, calibrated_xp)

            # Update player
            result.at[idx, "xP"] = calibrated_xp
            result.at[idx, "xP_uncertainty"] = calibrated_std
            result.at[idx, "xP_calibrated"] = True
            result.at[idx, "xP_raw"] = ml_xp  # Keep original for comparison

        if self.debug:
            logger.info(
                f"✅ Calibration complete: {calibrated_count} players calibrated, "
                f"{fallback_count} players used priors, "
                f"risk_profile={risk_profile}, "
                f"empirical_weight={empirical_weight:.2f}"
            )

        return result
