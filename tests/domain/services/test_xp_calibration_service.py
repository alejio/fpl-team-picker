"""Tests for XPCalibrationService."""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path

from fpl_team_picker.domain.services.xp_calibration_service import XPCalibrationService


class TestXPCalibrationServiceInitialization:
    """Tests for service initialization and configuration."""

    def test_default_initialization(self):
        """Test service initializes with default configuration."""
        service = XPCalibrationService()

        assert service.premium_price_threshold == 8.0
        assert service.mid_price_threshold == 6.0
        assert service.easy_fixture_threshold == 1.538
        assert service.minimum_sample_size == 30
        assert service.empirical_blend_weight == 0.5  # Default is 50% of fixture effect
        assert service.risk_adjustment_multiplier == 0.67
        assert service.debug is False

    def test_custom_configuration(self):
        """Test service initializes with custom configuration."""
        config = {
            "premium_price_threshold": 9.0,
            "mid_price_threshold": 7.0,
            "easy_fixture_threshold": 2.0,
            "minimum_sample_size": 50,
            "empirical_blend_weight": 0.3,
            "risk_adjustment_multiplier": 1.0,
            "debug": True,
        }
        service = XPCalibrationService(config=config)

        assert service.premium_price_threshold == 9.0
        assert service.mid_price_threshold == 7.0
        assert service.easy_fixture_threshold == 2.0
        assert service.minimum_sample_size == 50
        assert service.empirical_blend_weight == 0.3
        assert service.risk_adjustment_multiplier == 1.0
        assert service.debug is True

    def test_loads_distributions_from_file(self):
        """Test service loads distributions from JSON file."""
        # Create temporary distributions file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_distributions = {
                "premium_easy": {
                    "mean": 9.2,
                    "dispersion": 2.3,
                    "std": 4.1,
                    "sample_size": 65,
                },
                "premium_hard": {
                    "mean": 6.5,
                    "dispersion": 2.0,
                    "std": 3.5,
                    "sample_size": 100,
                },
            }
            json.dump(test_distributions, f)
            temp_path = f.name

        try:
            config = {"distributions_path": temp_path}
            service = XPCalibrationService(config=config)

            assert "premium_easy" in service.distributions
            assert "premium_hard" in service.distributions
            assert service.distributions["premium_easy"]["mean"] == 9.2
        finally:
            Path(temp_path).unlink()

    def test_handles_missing_distributions_file(self):
        """Test service handles missing distributions file gracefully."""
        config = {"distributions_path": "nonexistent/path/distributions.json"}
        service = XPCalibrationService(config=config)

        # Should use empty distributions and fall back to priors
        assert service.distributions == {}

    def test_handles_invalid_json_file(self):
        """Test service handles invalid JSON file gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content {")
            temp_path = f.name

        try:
            config = {"distributions_path": temp_path}
            service = XPCalibrationService(config=config)

            # Should use empty distributions
            assert service.distributions == {}
        finally:
            Path(temp_path).unlink()


class TestTierAndFixtureCategorization:
    """Tests for player tier and fixture categorization."""

    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return XPCalibrationService()

    def test_get_player_tier_premium(self, service):
        """Test premium tier categorization."""
        assert service._get_player_tier(8.0) == "premium"
        assert service._get_player_tier(10.0) == "premium"
        assert service._get_player_tier(15.0) == "premium"

    def test_get_player_tier_mid(self, service):
        """Test mid tier categorization."""
        assert service._get_player_tier(6.0) == "mid"
        assert service._get_player_tier(7.0) == "mid"
        assert service._get_player_tier(7.99) == "mid"

    def test_get_player_tier_budget(self, service):
        """Test budget tier categorization."""
        assert service._get_player_tier(5.99) == "budget"
        assert service._get_player_tier(4.0) == "budget"
        assert service._get_player_tier(1.0) == "budget"

    def test_get_fixture_category_easy(self, service):
        """Test easy fixture categorization."""
        assert service._get_fixture_category(1.538) == "easy"
        assert service._get_fixture_category(2.0) == "easy"
        assert service._get_fixture_category(5.0) == "easy"

    def test_get_fixture_category_hard(self, service):
        """Test hard fixture categorization."""
        assert service._get_fixture_category(1.537) == "hard"
        assert service._get_fixture_category(1.0) == "hard"
        assert service._get_fixture_category(0.5) == "hard"

    def test_get_fixture_category_missing(self, service):
        """Test fixture categorization raises error with missing data."""
        with pytest.raises(ValueError, match="Missing fixture_difficulty"):
            service._get_fixture_category(np.nan)
        with pytest.raises(ValueError, match="Missing fixture_difficulty"):
            service._get_fixture_category(pd.NA)

    def test_custom_thresholds(self):
        """Test categorization with custom thresholds."""
        config = {
            "premium_price_threshold": 9.0,
            "mid_price_threshold": 7.0,
            "easy_fixture_threshold": 2.0,
        }
        service = XPCalibrationService(config=config)

        assert service._get_player_tier(8.5) == "mid"  # Below 9.0
        assert service._get_player_tier(9.0) == "premium"
        assert service._get_fixture_category(1.9) == "hard"  # Below 2.0
        assert service._get_fixture_category(2.0) == "easy"


class TestCalibrationLogic:
    """Tests for calibration logic and risk profiles."""

    @pytest.fixture
    def mock_distributions(self):
        """Create mock distributions for testing."""
        return {
            "premium_easy": {
                "mean": 9.2,
                "dispersion": 2.3,
                "std": 4.1,
                "sample_size": 65,
            },
            "premium_hard": {
                "mean": 6.5,
                "dispersion": 2.0,
                "std": 3.5,
                "sample_size": 100,
            },
            "mid_easy": {
                "mean": 7.0,
                "dispersion": 1.8,
                "std": 3.2,
                "sample_size": 200,
            },
            "mid_hard": {
                "mean": 5.0,
                "dispersion": 1.5,
                "std": 2.8,
                "sample_size": 300,
            },
            "budget_easy": {
                "mean": 4.5,
                "dispersion": 1.2,
                "std": 2.5,
                "sample_size": 500,
            },
            "budget_hard": {
                "mean": 3.5,
                "dispersion": 1.0,
                "std": 2.0,
                "sample_size": 600,
            },
        }

    @pytest.fixture
    def service_with_distributions(self, mock_distributions):
        """Create service with mock distributions."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(mock_distributions, f)
            temp_path = f.name

        try:
            config = {"distributions_path": temp_path, "debug": False}
            service = XPCalibrationService(config=config)
            yield service
        finally:
            Path(temp_path).unlink()

    @pytest.fixture
    def sample_players(self):
        """Create sample players DataFrame."""
        return pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Premium Easy",
                    "price": 10.0,
                    "xP": 8.0,
                    "xP_uncertainty": 1.5,
                    "fixture_difficulty": 2.0,  # Easy
                },
                {
                    "player_id": 2,
                    "web_name": "Premium Hard",
                    "price": 9.0,
                    "xP": 7.0,
                    "xP_uncertainty": 1.5,
                    "fixture_difficulty": 1.0,  # Hard
                },
                {
                    "player_id": 3,
                    "web_name": "Mid Easy",
                    "price": 7.0,
                    "xP": 6.0,
                    "xP_uncertainty": 1.5,
                    "fixture_difficulty": 2.0,  # Easy
                },
                {
                    "player_id": 4,
                    "web_name": "Budget Hard",
                    "price": 5.0,
                    "xP": 4.0,
                    "xP_uncertainty": 1.5,
                    "fixture_difficulty": 1.0,  # Hard
                },
            ]
        )

    def test_calibration_applies_to_all_players(
        self, service_with_distributions, sample_players
    ):
        """Test calibration is applied to all players."""
        result = service_with_distributions.calibrate_predictions(
            sample_players, risk_profile="balanced"
        )

        assert len(result) == len(sample_players)
        assert "xP_calibrated" in result.columns
        assert all(result["xP_calibrated"])
        assert "xP_raw" in result.columns

    def test_calibration_balanced_profile(
        self, service_with_distributions, sample_players
    ):
        """Test balanced risk profile uses mean."""
        result = service_with_distributions.calibrate_predictions(
            sample_players, risk_profile="balanced"
        )

        # Calibrated xP should be between ML and distribution mean (precision-weighted)
        premium_easy = result[result["player_id"] == 1].iloc[0]
        assert premium_easy["xP"] > 0
        assert premium_easy["xP"] != premium_easy["xP_raw"]  # Should be calibrated
        assert premium_easy["xP_raw"] == 8.0  # Original ML prediction preserved

    def test_calibration_conservative_profile(
        self, service_with_distributions, sample_players
    ):
        """Test conservative risk profile uses lower credible interval."""
        result_balanced = service_with_distributions.calibrate_predictions(
            sample_players, risk_profile="balanced"
        )
        result_conservative = service_with_distributions.calibrate_predictions(
            sample_players, risk_profile="conservative"
        )

        # Conservative should be lower than balanced
        for idx in range(len(sample_players)):
            conservative_xp = result_conservative.iloc[idx]["xP"]
            balanced_xp = result_balanced.iloc[idx]["xP"]
            assert conservative_xp <= balanced_xp

    def test_calibration_risk_taking_profile(
        self, service_with_distributions, sample_players
    ):
        """Test risk-taking profile uses upper credible interval."""
        result_balanced = service_with_distributions.calibrate_predictions(
            sample_players, risk_profile="balanced"
        )
        result_risk_taking = service_with_distributions.calibrate_predictions(
            sample_players, risk_profile="risk-taking"
        )

        # Risk-taking should be higher than balanced
        for idx in range(len(sample_players)):
            risk_xp = result_risk_taking.iloc[idx]["xP"]
            balanced_xp = result_balanced.iloc[idx]["xP"]
            assert risk_xp >= balanced_xp

    def test_calibration_non_negative(self, service_with_distributions, sample_players):
        """Test calibrated xP is never negative."""
        result = service_with_distributions.calibrate_predictions(
            sample_players, risk_profile="conservative"
        )

        assert all(result["xP"] >= 0)

    def test_calibration_additive_fixture_effect(
        self, service_with_distributions, sample_players
    ):
        """Test calibration uses additive fixture effect correction."""
        result = service_with_distributions.calibrate_predictions(
            sample_players, risk_profile="balanced"
        )

        # For premium_easy: ML=8.0, distribution_mean=9.2
        # tier_baseline = (premium_easy + premium_hard) / 2 = (9.2 + 6.5) / 2 = 7.85
        # fixture_effect = 9.2 - 7.85 = 1.35
        # With default 50% empirical weight: calibrated = 8.0 + 0.5 * 1.35 = 8.675
        premium_easy = result[result["player_id"] == 1].iloc[0]
        ml_xp = premium_easy["xP_raw"]

        # Calculate expected using additive correction
        easy_mean = service_with_distributions.distributions["premium_easy"]["mean"]
        hard_mean = service_with_distributions.distributions["premium_hard"]["mean"]
        tier_baseline = (easy_mean + hard_mean) / 2
        fixture_effect = easy_mean - tier_baseline
        expected_xp = ml_xp + 0.5 * fixture_effect  # Default weight is 0.5

        assert abs(premium_easy["xP"] - expected_xp) < 0.01

    def test_calibration_uncertainty_updated(
        self, service_with_distributions, sample_players
    ):
        """Test calibration updates uncertainty."""
        result = service_with_distributions.calibrate_predictions(
            sample_players, risk_profile="balanced"
        )

        assert "xP_uncertainty" in result.columns
        # Uncertainty should be positive
        assert all(result["xP_uncertainty"] > 0)

    def test_blend_weight_affects_calibration(self, mock_distributions, sample_players):
        """Test that different blend weights produce different fixture effect adjustments."""
        # Create services with different blend weights
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(mock_distributions, f)
            temp_path = f.name

        try:
            # Low blend weight (25% of fixture effect)
            config_low = {
                "distributions_path": temp_path,
                "empirical_blend_weight": 0.25,
            }
            service_low = XPCalibrationService(config=config_low)

            # High blend weight (100% of fixture effect)
            config_high = {
                "distributions_path": temp_path,
                "empirical_blend_weight": 1.0,
            }
            service_high = XPCalibrationService(config=config_high)

            result_low = service_low.calibrate_predictions(sample_players.copy())
            result_high = service_high.calibrate_predictions(sample_players.copy())

            # Results should be different - higher weight applies more fixture effect
            for idx in range(len(sample_players)):
                low_adjustment = abs(
                    result_low.iloc[idx]["xP"] - result_low.iloc[idx]["xP_raw"]
                )
                high_adjustment = abs(
                    result_high.iloc[idx]["xP"] - result_high.iloc[idx]["xP_raw"]
                )
                # High weight should produce larger adjustments (or equal if fixture effect is 0)
                assert high_adjustment >= low_adjustment - 0.001  # Small tolerance

        finally:
            Path(temp_path).unlink()


class TestFallbackBehavior:
    """Tests for fallback behavior when distributions are missing."""

    @pytest.fixture
    def service_no_distributions(self):
        """Create service without distributions (uses priors only)."""
        config = {"distributions_path": "nonexistent/path.json", "debug": False}
        return XPCalibrationService(config=config)

    @pytest.fixture
    def service_insufficient_samples(self):
        """Create service with distributions but insufficient samples."""
        mock_distributions = {
            "premium_easy": {
                "mean": 9.2,
                "std": 4.1,
                "sample_size": 20,  # Below minimum of 30
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(mock_distributions, f)
            temp_path = f.name

        try:
            config = {"distributions_path": temp_path, "minimum_sample_size": 30}
            service = XPCalibrationService(config=config)
            yield service
        finally:
            Path(temp_path).unlink()

    @pytest.fixture
    def sample_players(self):
        """Create sample players DataFrame."""
        return pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Test Player",
                    "price": 10.0,
                    "xP": 8.0,
                    "xP_uncertainty": 1.5,
                    "fixture_difficulty": 2.0,
                },
            ]
        )

    def test_fallback_to_prior_when_no_distributions(
        self, service_no_distributions, sample_players
    ):
        """Test fallback to prior when distributions file missing."""
        result = service_no_distributions.calibrate_predictions(
            sample_players, risk_profile="balanced"
        )

        # Should still calibrate (using priors)
        assert len(result) == 1
        assert result.iloc[0]["xP_calibrated"]
        assert result.iloc[0]["xP"] >= 0

    def test_fallback_to_prior_when_insufficient_samples(
        self, service_insufficient_samples, sample_players
    ):
        """Test fallback to prior when sample size too small."""
        result = service_insufficient_samples.calibrate_predictions(
            sample_players, risk_profile="balanced"
        )

        # Should still calibrate (using priors)
        assert len(result) == 1
        assert result.iloc[0]["xP_calibrated"]

    def test_fallback_to_prior_when_combination_missing(
        self, service_no_distributions, sample_players
    ):
        """Test fallback to prior when specific combination missing."""
        result = service_no_distributions.calibrate_predictions(
            sample_players, risk_profile="balanced"
        )

        # Should use prior: premium (8.0) + easy (+2.0) = 10.0
        # Blended with ML (8.0): 0.8 * 8.0 + 0.2 * 10.0 = 8.4
        assert result.iloc[0]["xP"] > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return XPCalibrationService()

    def test_missing_price_column(self, service):
        """Test that missing price column raises error."""
        players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "xP": 5.0,
                    "xP_uncertainty": 1.5,
                    "fixture_difficulty": 2.0,
                },
            ]
        )

        # Should raise ValueError indicating data problem
        with pytest.raises(ValueError, match="Missing required columns"):
            service.calibrate_predictions(players)

    def test_missing_xp_column(self, service):
        """Test that missing xP column raises error."""
        players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "price": 8.0,
                    "xP_uncertainty": 1.5,
                    "fixture_difficulty": 2.0,
                },
            ]
        )

        # Should raise ValueError indicating data problem
        with pytest.raises(ValueError, match="Missing required columns"):
            service.calibrate_predictions(players)

    def test_missing_xp_uncertainty_column(self, service):
        """Test that missing xP_uncertainty column uses default value."""
        players = pd.DataFrame(
            [
                {"player_id": 1, "price": 8.0, "xP": 5.0, "fixture_difficulty": 2.0},
            ]
        )

        # Should work - xP_uncertainty is now optional (uses default 1.5)
        result = service.calibrate_predictions(players)
        assert len(result) == 1
        assert result.iloc[0]["xP_calibrated"]
        # Should have calculated uncertainty from distribution
        assert "xP_uncertainty" in result.columns
        assert result.iloc[0]["xP_uncertainty"] > 0

    def test_missing_fixture_difficulty(self, service):
        """Test that missing fixture difficulty raises error."""
        players = pd.DataFrame(
            [
                {"player_id": 1, "price": 8.0, "xP": 5.0, "xP_uncertainty": 1.5},
            ]
        )

        # Should raise ValueError indicating data problem
        with pytest.raises(ValueError, match="Missing required columns"):
            service.calibrate_predictions(players)

    def test_fixture_difficulty_from_separate_df(self, service):
        """Test merging fixture difficulty from separate DataFrame."""
        players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "price": 8.0,
                    "xP": 5.0,
                    "xP_uncertainty": 1.5,
                    "team_id": 1,
                },
            ]
        )
        fixture_df = pd.DataFrame(
            [
                {"team_id": 1, "fixture_difficulty": 2.0},
            ]
        )

        result = service.calibrate_predictions(
            players, fixture_difficulty_df=fixture_df
        )
        assert len(result) == 1
        assert result.iloc[0]["xP_calibrated"]

    def test_fixture_difficulty_from_separate_df_missing(self, service):
        """Test that missing fixture difficulty after merge raises error."""
        players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "price": 8.0,
                    "xP": 5.0,
                    "xP_uncertainty": 1.5,
                    "team_id": 1,
                },
            ]
        )
        fixture_df = pd.DataFrame(
            [
                {"team_id": 999, "fixture_difficulty": 2.0},  # No match for team_id=1
            ]
        )

        # Should raise error because fixture_difficulty will be NaN after merge
        with pytest.raises(ValueError, match="Missing fixture_difficulty"):
            service.calibrate_predictions(players, fixture_difficulty_df=fixture_df)

    def test_nan_fixture_difficulty_in_dataframe(self, service):
        """Test that NaN fixture_difficulty values in DataFrame raise error."""
        players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "price": 8.0,
                    "xP": 5.0,
                    "xP_uncertainty": 1.5,
                    "fixture_difficulty": np.nan,  # NaN value
                },
            ]
        )

        # Should raise error when processing the NaN value
        with pytest.raises(ValueError, match="Missing fixture_difficulty"):
            service.calibrate_predictions(players)

    def test_zero_uncertainty_handled_gracefully(self, service):
        """Test that zero uncertainty is handled gracefully with default value."""
        players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "price": 8.0,
                    "xP": 5.0,
                    "xP_uncertainty": 0.0,
                    "fixture_difficulty": 2.0,
                },
            ]
        )

        # Should handle gracefully - new implementation uses default 1.5 for missing/invalid
        result = service.calibrate_predictions(players)
        assert len(result) == 1
        assert result.iloc[0]["xP_calibrated"]
        assert result.iloc[0]["xP"] >= 0

    def test_negative_uncertainty_handled_gracefully(self, service):
        """Test that negative uncertainty is handled gracefully with default value."""
        players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "price": 8.0,
                    "xP": 5.0,
                    "xP_uncertainty": -1.0,
                    "fixture_difficulty": 2.0,
                },
            ]
        )

        # Should handle gracefully - new implementation uses default 1.5 for missing/invalid
        result = service.calibrate_predictions(players)
        assert len(result) == 1
        assert result.iloc[0]["xP_calibrated"]
        assert result.iloc[0]["xP"] >= 0

    def test_empty_dataframe(self, service):
        """Test that empty DataFrame raises error for missing columns."""
        players = pd.DataFrame()

        # Should raise ValueError for missing required columns
        with pytest.raises(ValueError, match="Missing required columns"):
            service.calibrate_predictions(players)

    def test_very_low_ml_xp(self, service):
        """Test calibration with very low ML prediction."""
        players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "price": 8.0,
                    "xP": 0.1,
                    "xP_uncertainty": 1.5,
                    "fixture_difficulty": 2.0,
                },
            ]
        )

        result = service.calibrate_predictions(players)
        # Should still be non-negative
        assert result.iloc[0]["xP"] >= 0

    def test_very_high_ml_xp(self, service):
        """Test calibration with very high ML prediction."""
        players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "price": 8.0,
                    "xP": 20.0,
                    "xP_uncertainty": 1.5,
                    "fixture_difficulty": 2.0,
                },
            ]
        )

        result = service.calibrate_predictions(players)
        # Should handle gracefully
        assert result.iloc[0]["xP"] >= 0


class TestIntegrationWithRealDistributions:
    """Integration tests with actual distributions file."""

    def test_loads_real_distributions_file(self):
        """Test service can load actual distributions.json file."""
        # Use the actual distributions file path
        # tests/domain/services/test_xp_calibration_service.py -> project root
        project_root = Path(__file__).parent.parent.parent.parent
        distributions_path = (
            project_root / "data" / "calibration" / "distributions.json"
        )

        if not distributions_path.exists():
            pytest.skip("Distributions file not found")

        service = XPCalibrationService()

        # Should have loaded distributions
        assert len(service.distributions) > 0
        assert any("premium" in key for key in service.distributions.keys())
        assert any("mid" in key for key in service.distributions.keys())
        assert any("budget" in key for key in service.distributions.keys())

    def test_calibration_with_real_distributions(self):
        """Test calibration using real distributions."""
        # tests/domain/services/test_xp_calibration_service.py -> project root
        project_root = Path(__file__).parent.parent.parent.parent
        distributions_path = (
            project_root / "data" / "calibration" / "distributions.json"
        )

        if not distributions_path.exists():
            pytest.skip("Distributions file not found")

        service = XPCalibrationService()

        players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Test Premium",
                    "price": 10.0,
                    "xP": 8.0,
                    "xP_uncertainty": 1.5,
                    "fixture_difficulty": 2.0,  # Easy
                },
                {
                    "player_id": 2,
                    "web_name": "Test Budget",
                    "price": 5.0,
                    "xP": 3.0,
                    "xP_uncertainty": 1.5,
                    "fixture_difficulty": 1.0,  # Hard
                },
            ]
        )

        result = service.calibrate_predictions(players, risk_profile="balanced")

        assert len(result) == 2
        assert all(result["xP_calibrated"])
        assert all(result["xP"] >= 0)
        assert all(result["xP_uncertainty"] > 0)
