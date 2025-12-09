"""Simple probabilistic calibration of ML xP predictions.

Uses regularized Negative Binomial distributions to adjust
ML predictions based on player quality × fixture difficulty.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from pathlib import Path
from loguru import logger


class XPCalibrationService:
    """Simple probabilistic calibration of ML xP predictions.

    Uses regularized Negative Binomial distributions to adjust
    ML predictions based on player quality × fixture difficulty.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize calibration service.

        Args:
            config: Optional configuration dictionary. If not provided, uses defaults.
                Keys:
                - premium_price_threshold: Price threshold for premium tier (default: 8.0)
                - mid_price_threshold: Price threshold for mid tier (default: 6.0)
                - easy_fixture_threshold: Fixture difficulty threshold for easy (default: 1.538)
                - distributions_path: Path to distributions.json (default: data/calibration/distributions.json)
                - minimum_sample_size: Minimum samples required (default: 30)
                - debug: Enable debug logging (default: False)
        """
        self.config = config or {}

        # Load configuration with defaults
        self.premium_price_threshold = self.config.get("premium_price_threshold", 8.0)
        self.mid_price_threshold = self.config.get("mid_price_threshold", 6.0)
        self.easy_fixture_threshold = self.config.get("easy_fixture_threshold", 1.538)
        self.minimum_sample_size = self.config.get("minimum_sample_size", 30)
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
                f"Loaded {len(self.distributions)} distributions from {self.distributions_path}"
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
        """Calibrate ML predictions using fitted distributions.

        Simple approach:
        1. Map player to (tier, fixture) combination
        2. Get fitted distribution for that combination
        3. Combine ML prediction with distribution mean (precision-weighted)
        4. Extract risk-adjusted value

        Args:
            players_df: DataFrame with player data including:
                - "price": Player price in millions
                - "xP": ML predicted expected points
                - "fixture_difficulty": Fixture difficulty (optional, can be in players_df or separate)
                - "xP_uncertainty": ML prediction uncertainty (optional, defaults to 1.5)
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
                - "xP_uncertainty": Calibrated uncertainty
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

        # Validate xP_uncertainty exists (required for precision weighting)
        if "xP_uncertainty" not in result.columns:
            raise ValueError(
                "Missing required column: xP_uncertainty. "
                "This is required for precision-weighted calibration."
            )

        calibrated_count = 0
        fallback_count = 0

        for idx, player in result.iterrows():
            # Get player attributes (direct access - will raise KeyError if missing)
            price = player["price"]
            tier = self._get_player_tier(price)

            fixture_difficulty = player["fixture_difficulty"]
            fixture = self._get_fixture_category(fixture_difficulty)

            # Get ML prediction (direct access - will raise KeyError if missing)
            ml_xp = player["xP"]
            ml_std = player["xP_uncertainty"]

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
                posterior_mean = prior_mean
                posterior_std = self.priors["default_std"]
                fallback_count += 1

                if self.debug and idx < 5:  # Log first few for debugging
                    logger.debug(
                        f"Fallback to prior for {combination_key}: "
                        f"mean={posterior_mean:.2f}, std={posterior_std:.2f}"
                    )
            else:
                # Use fitted distribution
                posterior_mean = distribution["mean"]
                posterior_std = distribution["std"]
                calibrated_count += 1

                if self.debug and idx < 5:  # Log first few for debugging
                    logger.debug(
                        f"Using fitted distribution for {combination_key}: "
                        f"mean={posterior_mean:.2f}, std={posterior_std:.2f}, "
                        f"sample_size={distribution['sample_size']}"
                    )

            # Precision-weighted combination (Bayesian update)
            # Higher precision = lower variance = more weight
            if ml_std <= 0:
                raise ValueError(
                    f"Invalid xP_uncertainty value: {ml_std} for player {player.get('player_id', idx)}. "
                    f"Uncertainty must be positive."
                )
            if posterior_std <= 0:
                raise ValueError(
                    f"Invalid posterior_std value: {posterior_std} for combination {combination_key}. "
                    f"Standard deviation must be positive."
                )

            ml_precision = 1.0 / (ml_std**2)
            posterior_precision = 1.0 / (posterior_std**2)
            total_precision = ml_precision + posterior_precision

            calibrated_mean = (ml_precision / total_precision) * ml_xp + (
                posterior_precision / total_precision
            ) * posterior_mean
            calibrated_std = np.sqrt(1.0 / total_precision)

            # Risk-adjusted value
            if risk_profile == "conservative":
                # Use lower credible interval (~25th percentile, -0.67 std)
                calibrated_xp = calibrated_mean - 0.67 * calibrated_std
            elif risk_profile == "risk-taking":
                # Use upper credible interval (~75th percentile, +0.67 std)
                calibrated_xp = calibrated_mean + 0.67 * calibrated_std
            else:  # balanced
                calibrated_xp = calibrated_mean

            # Ensure non-negative
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
                f"risk_profile={risk_profile}"
            )

        return result
