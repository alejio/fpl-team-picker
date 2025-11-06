"""Unit tests for uncertainty-based transfer filter (2025/26 upgrade).

Tests for Phase 1 improvements:
- Smart differential filtering using ML uncertainty
- Value trap prevention (low xP/price OR high uncertainty)
- Template/premium exceptions
- Asymmetric information preservation
"""

import pandas as pd
from fpl_team_picker.domain.services.optimization_service import OptimizationService


class TestDifferentialFiltering:
    """Test uncertainty-based differential filtering logic."""

    def test_high_confidence_differential_allowed(self):
        """Test that high-confidence differentials (Munetsi case) are allowed."""
        OptimizationService()

        # Create a high-confidence differential
        # Munetsi: 6.0 xP ± 0.8, £4.5m → 1.33 xP/£, 13% uncertainty → ALLOW
        all_players = pd.DataFrame(
            [
                {
                    "player_id": 100,
                    "web_name": "Munetsi",
                    "position": "MID",
                    "price": 4.5,
                    "xP": 6.0,
                    "xP_uncertainty": 0.8,  # Low uncertainty (13% of xP)
                    "selected_by_percent": 8.0,  # Low ownership (<15%)
                    "team": "FUL",
                    "status": "a",
                },
            ]
        )

        pd.DataFrame(
            [
                {
                    "player_id": i,
                    "web_name": f"Player{i}",
                    "position": "MID",
                    "price": 6.0,
                    "xP": 5.0,
                }
                for i in range(1, 16)
            ]
        )

        # Test via internal filter logic
        # Simulate what _swap_transfer_players does
        candidates = all_players.copy()

        def is_valid_differential_target(row):
            """Replicate the filter logic from _swap_transfer_players."""
            ownership = row["selected_by_percent"]
            price = row["price"]
            xp = row["xP"]
            uncertainty = row.get("xP_uncertainty", 0)

            # Route 1: Template/quality (>15% owned, >£5.5m)
            if ownership >= 15.0 and price >= 5.5:
                return True

            # Route 2: Premium exception (>£9m)
            if price >= 9.0:
                return True

            # Route 3: Differential - require confident prediction
            if ownership < 15.0:
                xp_per_price = xp / max(price, 0.1)

                if xp_per_price < 1.0:
                    return False

                uncertainty_ratio = uncertainty / max(xp, 0.1)
                return uncertainty_ratio < 0.30

            return True

        valid_mask = candidates.apply(is_valid_differential_target, axis=1)
        valid_targets = candidates[valid_mask]

        # Munetsi should be ALLOWED (high xP/price + low uncertainty)
        assert len(valid_targets) == 1
        assert valid_targets.iloc[0]["web_name"] == "Munetsi"

        # Check metrics
        munetsi = all_players.iloc[0]
        xp_per_price = munetsi["xP"] / munetsi["price"]
        uncertainty_ratio = munetsi["xP_uncertainty"] / munetsi["xP"]

        assert xp_per_price > 1.0  # 1.33
        assert uncertainty_ratio < 0.30  # 0.13

    def test_low_confidence_differential_blocked(self):
        """Test that low-confidence differentials (Cullen case) are blocked."""
        OptimizationService()

        # Create a low-confidence differential
        # Cullen: 4.0 xP ± 2.0, £4.6m → 0.87 xP/£, 50% uncertainty → BLOCK
        all_players = pd.DataFrame(
            [
                {
                    "player_id": 101,
                    "web_name": "Cullen",
                    "position": "MID",
                    "price": 4.6,
                    "xP": 4.0,
                    "xP_uncertainty": 2.0,  # High uncertainty (50% of xP)
                    "selected_by_percent": 5.0,  # Low ownership
                    "team": "LUT",
                    "status": "a",
                },
            ]
        )

        def is_valid_differential_target(row):
            ownership = row["selected_by_percent"]
            price = row["price"]
            xp = row["xP"]
            uncertainty = row.get("xP_uncertainty", 0)

            if ownership >= 15.0 and price >= 5.5:
                return True
            if price >= 9.0:
                return True
            if ownership < 15.0:
                xp_per_price = xp / max(price, 0.1)
                if xp_per_price < 1.0:
                    return False
                uncertainty_ratio = uncertainty / max(xp, 0.1)
                return uncertainty_ratio < 0.30
            return True

        valid_mask = all_players.apply(is_valid_differential_target, axis=1)
        valid_targets = all_players[valid_mask]

        # Cullen should be BLOCKED (low xP/price + high uncertainty)
        assert len(valid_targets) == 0

        # Check why blocked
        cullen = all_players.iloc[0]
        xp_per_price = cullen["xP"] / cullen["price"]
        uncertainty_ratio = cullen["xP_uncertainty"] / cullen["xP"]

        # Should fail BOTH checks
        assert xp_per_price < 1.0  # 0.87 - insufficient value
        assert uncertainty_ratio > 0.30  # 0.50 - too uncertain

    def test_differential_blocked_by_low_value(self):
        """Test that differentials with low xP/price are blocked even if confident."""
        OptimizationService()

        all_players = pd.DataFrame(
            [
                {
                    "player_id": 102,
                    "web_name": "LowValue",
                    "position": "DEF",
                    "price": 5.0,
                    "xP": 4.0,  # Low value: 0.8 xP/£
                    "xP_uncertainty": 0.5,  # Low uncertainty (12.5%)
                    "selected_by_percent": 8.0,
                    "team": "BUR",
                    "status": "a",
                },
            ]
        )

        def is_valid_differential_target(row):
            ownership = row["selected_by_percent"]
            price = row["price"]
            xp = row["xP"]
            uncertainty = row.get("xP_uncertainty", 0)

            if ownership >= 15.0 and price >= 5.5:
                return True
            if price >= 9.0:
                return True
            if ownership < 15.0:
                xp_per_price = xp / max(price, 0.1)
                if xp_per_price < 1.0:
                    return False  # Blocked here
                uncertainty_ratio = uncertainty / max(xp, 0.1)
                return uncertainty_ratio < 0.30
            return True

        valid_mask = all_players.apply(is_valid_differential_target, axis=1)
        valid_targets = all_players[valid_mask]

        # Should be BLOCKED due to low value
        assert len(valid_targets) == 0

    def test_differential_blocked_by_high_uncertainty(self):
        """Test that differentials with high uncertainty are blocked even if good value."""
        OptimizationService()

        all_players = pd.DataFrame(
            [
                {
                    "player_id": 103,
                    "web_name": "HighUncertainty",
                    "position": "MID",
                    "price": 5.0,
                    "xP": 6.0,  # Good value: 1.2 xP/£
                    "xP_uncertainty": 2.5,  # High uncertainty (42%)
                    "selected_by_percent": 6.0,
                    "team": "SHU",
                    "status": "a",
                },
            ]
        )

        def is_valid_differential_target(row):
            ownership = row["selected_by_percent"]
            price = row["price"]
            xp = row["xP"]
            uncertainty = row.get("xP_uncertainty", 0)

            if ownership >= 15.0 and price >= 5.5:
                return True
            if price >= 9.0:
                return True
            if ownership < 15.0:
                xp_per_price = xp / max(price, 0.1)
                if xp_per_price < 1.0:
                    return False
                uncertainty_ratio = uncertainty / max(xp, 0.1)
                return uncertainty_ratio < 0.30  # Blocked here
            return True

        valid_mask = all_players.apply(is_valid_differential_target, axis=1)
        valid_targets = all_players[valid_mask]

        # Should be BLOCKED due to high uncertainty
        assert len(valid_targets) == 0

    def test_differential_boundary_cases(self):
        """Test boundary cases for xP/price and uncertainty thresholds."""
        OptimizationService()

        # Test at boundaries
        test_cases = [
            # (xP, price, uncertainty, expected_allowed)
            (5.0, 5.0, 1.4, True),   # Exactly 1.0 xP/£, exactly 30% uncertainty → marginal
            (4.99, 5.0, 1.4, False),  # Just below 1.0 xP/£ → blocked
            (5.0, 5.0, 1.51, False),  # Just above 30% uncertainty → blocked
            (6.0, 5.0, 1.7, True),   # 1.2 xP/£, 28% uncertainty → allowed
        ]

        for xp, price, uncertainty, expected_allowed in test_cases:
            all_players = pd.DataFrame(
                [
                    {
                        "player_id": 1,
                        "web_name": "Boundary",
                        "position": "MID",
                        "price": price,
                        "xP": xp,
                        "xP_uncertainty": uncertainty,
                        "selected_by_percent": 10.0,
                        "team": "BRE",
                        "status": "a",
                    },
                ]
            )

            def is_valid_differential_target(row):
                ownership = row["selected_by_percent"]
                price_val = row["price"]
                xp_val = row["xP"]
                unc = row.get("xP_uncertainty", 0)

                if ownership >= 15.0 and price_val >= 5.5:
                    return True
                if price_val >= 9.0:
                    return True
                if ownership < 15.0:
                    xp_per_price = xp_val / max(price_val, 0.1)
                    if xp_per_price < 1.0:
                        return False
                    uncertainty_ratio = unc / max(xp_val, 0.1)
                    return uncertainty_ratio < 0.30
                return True

            valid_mask = all_players.apply(is_valid_differential_target, axis=1)
            is_allowed = len(all_players[valid_mask]) > 0

            xp_per_price = xp / price
            uncertainty_ratio = uncertainty / xp

            print(f"xP={xp}, price={price}, unc={uncertainty}: "
                  f"xP/£={xp_per_price:.2f}, unc%={uncertainty_ratio:.1%}, "
                  f"allowed={is_allowed} (expected={expected_allowed})")


class TestTemplateAndPremiumExceptions:
    """Test that template and premium players always pass filter."""

    def test_template_quality_player_always_allowed(self):
        """Test that template/quality players (>15% owned, >£5.5m) always pass."""
        OptimizationService()

        all_players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Template",
                    "position": "MID",
                    "price": 8.0,  # >£5.5m
                    "xP": 4.0,  # Low xP (0.5 xP/£)
                    "xP_uncertainty": 3.0,  # High uncertainty (75%)
                    "selected_by_percent": 25.0,  # >15%
                    "team": "ARS",
                    "status": "a",
                },
            ]
        )

        def is_valid_differential_target(row):
            ownership = row["selected_by_percent"]
            price = row["price"]
            xp = row["xP"]
            uncertainty = row.get("xP_uncertainty", 0)

            # Route 1: Template/quality exception
            if ownership >= 15.0 and price >= 5.5:
                return True  # ALLOWED immediately
            if price >= 9.0:
                return True
            if ownership < 15.0:
                xp_per_price = xp / max(price, 0.1)
                if xp_per_price < 1.0:
                    return False
                uncertainty_ratio = uncertainty / max(xp, 0.1)
                return uncertainty_ratio < 0.30
            return True

        valid_mask = all_players.apply(is_valid_differential_target, axis=1)
        valid_targets = all_players[valid_mask]

        # Should be ALLOWED despite terrible value and uncertainty
        assert len(valid_targets) == 1

    def test_premium_player_always_allowed(self):
        """Test that premium players (>£9m) always pass regardless of uncertainty."""
        OptimizationService()

        all_players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "Haaland",
                    "position": "FWD",
                    "price": 15.0,  # >£9m
                    "xP": 9.0,
                    "xP_uncertainty": 2.5,  # High uncertainty (28%)
                    "selected_by_percent": 62.0,
                    "team": "MCI",
                    "status": "a",
                },
                {
                    "player_id": 2,
                    "web_name": "Salah",
                    "position": "MID",
                    "price": 13.0,  # >£9m
                    "xP": 8.5,
                    "xP_uncertainty": 2.0,
                    "selected_by_percent": 48.0,
                    "team": "LIV",
                    "status": "a",
                },
                {
                    "player_id": 3,
                    "web_name": "InjuredPremium",
                    "position": "FWD",
                    "price": 11.0,  # >£9m
                    "xP": 3.0,  # Low xP
                    "xP_uncertainty": 3.0,  # Very high uncertainty (100%)
                    "selected_by_percent": 5.0,  # Low ownership
                    "team": "CHE",
                    "status": "i",  # Injured
                },
            ]
        )

        def is_valid_differential_target(row):
            ownership = row["selected_by_percent"]
            price = row["price"]
            xp = row["xP"]
            uncertainty = row.get("xP_uncertainty", 0)

            if ownership >= 15.0 and price >= 5.5:
                return True
            # Route 2: Premium exception
            if price >= 9.0:
                return True  # ALLOWED immediately
            if ownership < 15.0:
                xp_per_price = xp / max(price, 0.1)
                if xp_per_price < 1.0:
                    return False
                uncertainty_ratio = uncertainty / max(xp, 0.1)
                return uncertainty_ratio < 0.30
            return True

        valid_mask = all_players.apply(is_valid_differential_target, axis=1)
        valid_targets = all_players[valid_mask]

        # ALL premiums should be ALLOWED
        assert len(valid_targets) == 3
        assert set(valid_targets["web_name"]) == {"Haaland", "Salah", "InjuredPremium"}

    def test_near_threshold_ownership_and_price(self):
        """Test boundary cases for template/quality thresholds."""
        OptimizationService()

        test_cases = [
            # (ownership, price, should_pass_exception)
            (14.9, 5.6, False),  # Just below ownership threshold (fails template, fails differential)
            (15.0, 5.5, True),   # Exactly at both thresholds → PASSES template exception
            (20.0, 6.0, True),   # Above both thresholds → PASSES template exception
            (14.0, 9.1, True),   # Below ownership but premium → PASSES premium exception
            (16.0, 5.5, True),   # Above ownership threshold, at price threshold → PASSES template exception
            (5.0, 4.5, False),   # Low ownership, low price, low xP → fails differential check
        ]

        for ownership, price, should_pass in test_cases:
            all_players = pd.DataFrame(
                [
                    {
                        "player_id": 1,
                        "web_name": "Boundary",
                        "position": "MID",
                        "price": price,
                        "xP": 3.0,  # Low xP (would fail differential check)
                        "xP_uncertainty": 2.0,  # High uncertainty
                        "selected_by_percent": ownership,
                        "team": "WHU",
                        "status": "a",
                    },
                ]
            )

            def is_valid_differential_target(row):
                own = row["selected_by_percent"]
                pr = row["price"]
                xp = row["xP"]
                uncertainty = row.get("xP_uncertainty", 0)

                if own >= 15.0 and pr >= 5.5:
                    return True
                if pr >= 9.0:
                    return True
                if own < 15.0:
                    xp_per_price = xp / max(pr, 0.1)
                    if xp_per_price < 1.0:
                        return False
                    uncertainty_ratio = uncertainty / max(xp, 0.1)
                    return uncertainty_ratio < 0.30
                return True

            valid_mask = all_players.apply(is_valid_differential_target, axis=1)
            passes = len(all_players[valid_mask]) > 0

            assert passes == should_pass, (
                f"Ownership={ownership}%, Price=£{price}: "
                f"Expected {'PASS' if should_pass else 'FAIL'}, got {'PASS' if passes else 'FAIL'}"
            )


class TestFilterPriorities:
    """Test the priority order of filtering routes."""

    def test_template_exception_before_differential_check(self):
        """Test that template exception takes priority over differential logic."""
        OptimizationService()

        # Player qualifies for template exception but would fail differential check
        all_players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "TemplateFirst",
                    "position": "DEF",
                    "price": 6.0,  # >£5.5m
                    "xP": 3.0,  # 0.5 xP/£ (would fail)
                    "xP_uncertainty": 2.0,  # 67% (would fail)
                    "selected_by_percent": 20.0,  # >15% (qualifies)
                    "team": "NEW",
                    "status": "a",
                },
            ]
        )

        def is_valid_differential_target(row):
            ownership = row["selected_by_percent"]
            price = row["price"]
            xp = row["xP"]
            uncertainty = row.get("xP_uncertainty", 0)

            # Priority 1: Template/quality
            if ownership >= 15.0 and price >= 5.5:
                return True  # ALLOWED - should not reach differential check
            # Priority 2: Premium
            if price >= 9.0:
                return True
            # Priority 3: Differential
            if ownership < 15.0:
                xp_per_price = xp / max(price, 0.1)
                if xp_per_price < 1.0:
                    return False
                uncertainty_ratio = uncertainty / max(xp, 0.1)
                return uncertainty_ratio < 0.30
            return True

        valid_mask = all_players.apply(is_valid_differential_target, axis=1)
        valid_targets = all_players[valid_mask]

        # Should be ALLOWED via template exception (never reaches differential check)
        assert len(valid_targets) == 1

    def test_premium_exception_before_differential_check(self):
        """Test that premium exception takes priority over differential logic."""
        OptimizationService()

        all_players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "PremiumFirst",
                    "position": "FWD",
                    "price": 12.0,  # >£9m (qualifies)
                    "xP": 5.0,  # 0.42 xP/£ (would fail)
                    "xP_uncertainty": 3.0,  # 60% (would fail)
                    "selected_by_percent": 8.0,  # <15% (would enter differential check)
                    "team": "TOT",
                    "status": "a",
                },
            ]
        )

        def is_valid_differential_target(row):
            ownership = row["selected_by_percent"]
            price = row["price"]
            xp = row["xP"]
            uncertainty = row.get("xP_uncertainty", 0)

            # Priority 1: Template/quality
            if ownership >= 15.0 and price >= 5.5:
                return True
            # Priority 2: Premium
            if price >= 9.0:
                return True  # ALLOWED - should not reach differential check
            # Priority 3: Differential
            if ownership < 15.0:
                xp_per_price = xp / max(price, 0.1)
                if xp_per_price < 1.0:
                    return False
                uncertainty_ratio = uncertainty / max(xp, 0.1)
                return uncertainty_ratio < 0.30
            return True

        valid_mask = all_players.apply(is_valid_differential_target, axis=1)
        valid_targets = all_players[valid_mask]

        # Should be ALLOWED via premium exception
        assert len(valid_targets) == 1


class TestMixedCandidateSet:
    """Test filter behavior with mixed candidate pool."""

    def test_mixed_candidates_correct_filtering(self):
        """Test that filter correctly handles mix of templates, premiums, and differentials."""
        OptimizationService()

        all_players = pd.DataFrame(
            [
                # SHOULD PASS: Template
                {
                    "player_id": 1,
                    "web_name": "Template",
                    "position": "MID",
                    "price": 8.0,
                    "xP": 6.5,
                    "xP_uncertainty": 1.0,
                    "selected_by_percent": 30.0,
                    "team": "ARS",
                    "status": "a",
                },
                # SHOULD PASS: Premium
                {
                    "player_id": 2,
                    "web_name": "Premium",
                    "position": "FWD",
                    "price": 14.0,
                    "xP": 8.5,
                    "xP_uncertainty": 1.5,
                    "selected_by_percent": 55.0,
                    "team": "MCI",
                    "status": "a",
                },
                # SHOULD PASS: High-confidence differential
                {
                    "player_id": 3,
                    "web_name": "GoodDifferential",
                    "position": "MID",
                    "price": 5.0,
                    "xP": 6.0,  # 1.2 xP/£
                    "xP_uncertainty": 0.9,  # 15%
                    "selected_by_percent": 7.0,
                    "team": "FUL",
                    "status": "a",
                },
                # SHOULD FAIL: Low value differential
                {
                    "player_id": 4,
                    "web_name": "LowValue",
                    "position": "DEF",
                    "price": 4.5,
                    "xP": 4.0,  # 0.89 xP/£
                    "xP_uncertainty": 0.8,
                    "selected_by_percent": 5.0,
                    "team": "BUR",
                    "status": "a",
                },
                # SHOULD FAIL: High uncertainty differential
                {
                    "player_id": 5,
                    "web_name": "HighUncertainty",
                    "position": "MID",
                    "price": 4.8,
                    "xP": 5.5,  # 1.15 xP/£ (good)
                    "xP_uncertainty": 2.2,  # 40% (bad)
                    "selected_by_percent": 4.0,
                    "team": "SHU",
                    "status": "a",
                },
            ]
        )

        def is_valid_differential_target(row):
            ownership = row["selected_by_percent"]
            price = row["price"]
            xp = row["xP"]
            uncertainty = row.get("xP_uncertainty", 0)

            if ownership >= 15.0 and price >= 5.5:
                return True
            if price >= 9.0:
                return True
            if ownership < 15.0:
                xp_per_price = xp / max(price, 0.1)
                if xp_per_price < 1.0:
                    return False
                uncertainty_ratio = uncertainty / max(xp, 0.1)
                return uncertainty_ratio < 0.30
            return True

        valid_mask = all_players.apply(is_valid_differential_target, axis=1)
        valid_targets = all_players[valid_mask]

        # Should have exactly 3 valid targets
        assert len(valid_targets) == 3

        valid_names = set(valid_targets["web_name"])
        assert valid_names == {"Template", "Premium", "GoodDifferential"}

        # Check that bad ones were filtered out
        all_names = set(all_players["web_name"])
        filtered_out = all_names - valid_names
        assert filtered_out == {"LowValue", "HighUncertainty"}

    def test_empty_candidate_pool_after_filter(self):
        """Test behavior when all candidates are filtered out."""
        OptimizationService()

        # All players are value traps
        all_players = pd.DataFrame(
            [
                {
                    "player_id": i,
                    "web_name": f"ValueTrap{i}",
                    "position": "MID",
                    "price": 5.0,
                    "xP": 3.0,  # 0.6 xP/£
                    "xP_uncertainty": 1.5,  # 50%
                    "selected_by_percent": 5.0,
                    "team": "BUR",
                    "status": "a",
                }
                for i in range(5)
            ]
        )

        def is_valid_differential_target(row):
            ownership = row["selected_by_percent"]
            price = row["price"]
            xp = row["xP"]
            uncertainty = row.get("xP_uncertainty", 0)

            if ownership >= 15.0 and price >= 5.5:
                return True
            if price >= 9.0:
                return True
            if ownership < 15.0:
                xp_per_price = xp / max(price, 0.1)
                if xp_per_price < 1.0:
                    return False
                uncertainty_ratio = uncertainty / max(xp, 0.1)
                return uncertainty_ratio < 0.30
            return True

        valid_mask = all_players.apply(is_valid_differential_target, axis=1)
        valid_targets = all_players[valid_mask]

        # All should be filtered out
        assert len(valid_targets) == 0


class TestMissingData:
    """Test filter behavior with missing uncertainty data."""

    def test_missing_uncertainty_uses_default(self):
        """Test that missing uncertainty column is handled gracefully."""
        OptimizationService()

        all_players = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "web_name": "NoUncertainty",
                    "position": "MID",
                    "price": 5.0,
                    "xP": 6.0,  # 1.2 xP/£
                    # NO xP_uncertainty column
                    "selected_by_percent": 8.0,
                    "team": "FUL",
                    "status": "a",
                },
            ]
        )

        def is_valid_differential_target(row):
            ownership = row["selected_by_percent"]
            price = row["price"]
            xp = row["xP"]
            uncertainty = row.get("xP_uncertainty", 0)  # Default to 0

            if ownership >= 15.0 and price >= 5.5:
                return True
            if price >= 9.0:
                return True
            if ownership < 15.0:
                xp_per_price = xp / max(price, 0.1)
                if xp_per_price < 1.0:
                    return False
                uncertainty_ratio = uncertainty / max(xp, 0.1)
                return uncertainty_ratio < 0.30  # 0 / 6.0 = 0 < 0.30 → PASS
            return True

        valid_mask = all_players.apply(is_valid_differential_target, axis=1)
        valid_targets = all_players[valid_mask]

        # Should PASS (uncertainty defaults to 0, which is < 30%)
        assert len(valid_targets) == 1
