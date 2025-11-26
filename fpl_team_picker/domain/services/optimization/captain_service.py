"""Captain selection service for FPL optimization.

This module handles:
- Captain and vice-captain recommendations
- Upside-focused captain analysis (ceiling-seeking)
- Situation-aware intelligent captain selection
- Template protection and matchup quality scoring
- Haul probability calculations
- Rank impact estimation
"""

from typing import Dict, Optional, Any
import pandas as pd
from loguru import logger
from fpl_team_picker.config import config

from .optimization_base import OptimizationBaseMixin


class CaptainServiceMixin(OptimizationBaseMixin):
    """Mixin providing captain selection functionality.

    Implements ceiling-seeking captain selection that prioritizes explosive upside
    over safety. Includes situation-aware analysis for rank-based strategy.
    """

    def get_captain_recommendation(
        self, players: pd.DataFrame, top_n: int = 5, xp_column: str = "xP"
    ) -> Dict[str, Any]:
        """Get captain recommendation with upside-focused analysis.

        Returns structured data only - no UI generation.

        Implements ceiling-seeking captain selection (prioritizes explosive upside):
        1. Uses 90th percentile (xP + 1.28*uncertainty) instead of mean - seeks high ceiling
        2. Template protection: High ownership (>50%) players get 20-40% boost in good fixtures
        3. Matchup quality: Betting odds integration for explosive game identification

        Philosophy: Captaincy should maximize ceiling (potential hauls), not minimize risk.
        High uncertainty = high upside potential (15+ point hauls vs weak opposition).

        Args:
            players: DataFrame of players to consider (squad or all players)
            top_n: Number of top candidates to analyze (default 5)
            xp_column: Column to use for xP (default "xP")

        Returns:
            {
                "captain": {dict with recommended captain},
                "vice_captain": {dict with vice captain},
                "top_candidates": [list of top 5 candidates with analysis],
                "captain_upside": float (upside * 2),
                "differential": float (captain vs vice diff)
            }
        """
        if players.empty:
            raise ValueError("No players provided for captain selection")

        # Ensure required columns exist
        required_cols = ["web_name", "position", xp_column]
        missing_cols = [col for col in required_cols if col not in players.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert to list of dicts for easier processing
        players_list = players.to_dict("records")

        # Enhanced captain scoring with uncertainty and ownership
        has_uncertainty = "xP_uncertainty" in players.columns
        has_ownership = "selected_by_percent" in players.columns

        for player in players_list:
            xp_value = player.get(xp_column, 0)
            uncertainty = player.get("xP_uncertainty", 0)
            ownership_pct = player.get("selected_by_percent", 0)

            # Calculate upside (90th percentile) instead of penalizing uncertainty
            # Philosophy: Captaincy seeks ceiling/explosive potential, not safety
            # 90th percentile = mean + 1.28 * std_dev (normal distribution)
            xp_upside = xp_value
            if has_uncertainty and uncertainty > 0:
                # Players with high uncertainty have higher ceiling potential
                # Example: Haaland vs Leeds (8.5 ± 2.2 xP) → upside = 11.3 xP
                xp_upside = xp_value + (1.28 * uncertainty)

            # Base captain score uses upside (seeking ceiling, not mean)
            base_score = xp_upside * 2

            # Store upside for transparency (used in candidate analysis)
            upside_boost = xp_upside - xp_value

            # Template protection + Matchup quality: Boost high ownership in good fixtures
            # UPGRADED 2025/26: Increased from 5-10% to 20-40% + betting odds integration
            # Rationale: Template captains (Haaland 60%) vs weak teams MUST be captained
            ownership_bonus = 0
            matchup_bonus = 0

            # Check if betting odds available
            has_betting_odds = (
                "team_win_probability" in player and "team_expected_goals" in player
            )

            if has_ownership and ownership_pct > 50:
                # TEMPLATE PROTECTION (20-40% boost)
                # Ownership factor: 50% = 0.0, 70%+ = 1.0
                ownership_factor = min((ownership_pct - 50) / 20, 1.0)

                # Fixture quality from betting odds (preferred) or fixture outlook (fallback)
                if has_betting_odds:
                    # Use win probability (0.33 = neutral, 0.70+ = strong favorite)
                    win_prob = player.get("team_win_probability", 0.33)
                    fixture_quality = min(win_prob / 0.70, 1.0)  # Normalize [0, 1]
                else:
                    # Fallback to fixture outlook
                    fixture_outlook_str = player.get("fixture_outlook", "")
                    if "Easy" in fixture_outlook_str or "🟢" in fixture_outlook_str:
                        fixture_quality = 1.0  # Easy fixture
                    elif "Hard" in fixture_outlook_str or "🔴" in fixture_outlook_str:
                        fixture_quality = 0.2  # Hard fixture
                    else:
                        fixture_quality = 0.5  # Average fixture

                # Template protection: up to 40% boost
                # Example: Haaland 60% owned, 75% win prob → 40% boost
                ownership_bonus = ownership_factor * fixture_quality * 0.40

            # MATCHUP QUALITY (0-25% boost for ALL players)
            # Rewards attacking potential regardless of ownership
            if has_betting_odds:
                # Expected goals as proxy for attacking potential
                team_xg = player.get("team_expected_goals", 1.25)
                # Normalize: 1.25 = neutral (PL average), 2.0+ = elite matchup
                xg_factor = min(max((team_xg - 1.0) / 1.0, 0.0), 1.0)  # [0, 1]
                matchup_bonus = xg_factor * 0.25  # Up to 25% boost
            else:
                # Fallback: use fixture outlook for matchup quality
                fixture_outlook_str = player.get("fixture_outlook", "")
                if "Easy" in fixture_outlook_str or "🟢" in fixture_outlook_str:
                    matchup_bonus = 0.20  # Good matchup
                elif "Hard" in fixture_outlook_str or "🔴" in fixture_outlook_str:
                    matchup_bonus = 0.0  # Poor matchup
                else:
                    matchup_bonus = 0.10  # Average matchup

            # Apply bonuses to upside-based score
            final_score = base_score * (1 + ownership_bonus + matchup_bonus)

            # Store scoring components for analysis
            player["_captain_score"] = final_score
            player["_base_score"] = base_score
            player["_upside_boost"] = upside_boost
            player["_xp_upside"] = xp_upside
            player["_ownership_bonus"] = ownership_bonus
            player["_matchup_bonus"] = matchup_bonus

        # Sort by enhanced captain score
        captain_candidates = sorted(
            players_list, key=lambda p: p.get("_captain_score", 0), reverse=True
        )

        if not captain_candidates:
            raise ValueError("No valid captain candidates found")

        # Analyze top N candidates
        top_candidates = []

        for i, player in enumerate(captain_candidates[:top_n]):
            xp_value = player.get(xp_column, 0)
            fixture_outlook = player.get("fixture_outlook", "🟡 Average")
            uncertainty = player.get("xP_uncertainty", 0)
            ownership_pct = player.get("selected_by_percent", 0)

            # Enhanced risk assessment
            risk_factors = []
            risk_level = "🟢 Low"

            # Prediction uncertainty risk
            if has_uncertainty and uncertainty > 0:
                uncertainty_ratio = uncertainty / max(xp_value, 0.1)
                if uncertainty_ratio > 0.4:  # High uncertainty (>40% of xP)
                    risk_factors.append(f"high variance ({uncertainty:.2f})")
                    if risk_level == "🟢 Low":
                        risk_level = "🟡 Medium"

            # Injury/availability risk
            if player.get("status") in ["i", "d"]:
                risk_factors.append("injury risk")
                risk_level = "🔴 High"
            elif player.get("status") == "s":  # Suspended
                risk_factors.append("suspended")
                risk_level = "🔴 High"

            # Fixture difficulty risk
            if "Hard" in fixture_outlook or "🔴" in fixture_outlook:
                risk_factors.append("hard fixture")
                if risk_level == "🟢 Low":
                    risk_level = "🟡 Medium"

            # Minutes certainty
            expected_mins = player.get("expected_minutes", 0)
            if expected_mins < 60:
                risk_factors.append("rotation risk")
                risk_level = "🟡 Medium" if risk_level == "🟢 Low" else risk_level

            # Combine risk description
            if risk_factors:
                risk_desc = f"{risk_level} ({', '.join(risk_factors)})"
            else:
                risk_desc = risk_level

            # Calculate captaincy potential (XP * 2 for double points)
            captain_potential = xp_value * 2

            # Build candidate dict with new scoring components
            candidate = {
                "rank": i + 1,
                "player_id": player.get("player_id"),
                "web_name": player["web_name"],
                "position": player["position"],
                "price": player.get("price", 0),
                "xP": xp_value,
                "captain_points": captain_potential,
                "expected_minutes": expected_mins,
                "fixture_outlook": fixture_outlook,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "risk_description": risk_desc,
                "status": player.get("status", "a"),
            }

            # Add uncertainty and upside metrics if available
            if has_uncertainty:
                candidate["xP_uncertainty"] = uncertainty
                candidate["upside_boost"] = player.get("_upside_boost", 0)
                candidate["xP_upside"] = player.get("_xp_upside", xp_value)

            if has_ownership:
                candidate["ownership_pct"] = ownership_pct
                candidate["ownership_bonus"] = player.get("_ownership_bonus", 0)
                # Add template flag
                candidate["is_template"] = ownership_pct > 50

            # Add risk-adjusted score and scoring components (always present)
            candidate["captain_score"] = player.get("_captain_score", captain_potential)
            candidate["base_score"] = player.get("_base_score", captain_potential)
            candidate["matchup_bonus"] = player.get("_matchup_bonus", 0)
            # Add ownership_bonus even if 0 (for consistent API)
            if not has_ownership:
                candidate["ownership_bonus"] = 0

            top_candidates.append(candidate)

        # Select captain and vice captain
        captain = top_candidates[0]
        vice_captain = top_candidates[1] if len(top_candidates) > 1 else captain

        # Calculate captaincy metrics
        captain_upside = captain["captain_points"]
        vice_upside = vice_captain["captain_points"]
        differential = captain_upside - vice_upside

        return {
            "captain": captain,
            "vice_captain": vice_captain,
            "top_candidates": top_candidates,
            "captain_upside": captain_upside,
            "vice_upside": vice_upside,
            "differential": differential,
        }

    def get_intelligent_captain_recommendation(
        self,
        players: pd.DataFrame,
        manager_context: Optional[Dict[str, Any]] = None,
        top_n: int = 5,
        xp_column: str = "xP",
    ) -> Dict[str, Any]:
        """Get intelligent captain recommendation with situation-aware analysis.

        This enhanced method provides:
        1. Situation analysis based on current rank and season phase
        2. Strategy recommendation (auto-detected or user-specified)
        3. Haul probability matrix for each candidate
        4. Expected rank impact for each captain choice
        5. Risk-adjusted recommendations based on strategy

        Args:
            players: DataFrame of players to consider (squad or all players)
            manager_context: Optional context dict with:
                - overall_rank: Current overall rank
                - gameweek: Current gameweek number
                - chip_active: Active chip name (e.g., "freehit", "wildcard")
                - rank_trend: Recent rank changes (positive = improving)
                - strategy_override: Force a specific strategy mode
            top_n: Number of top candidates to analyze (default 5)
            xp_column: Column to use for xP (default "xP")

        Returns:
            Enhanced captain recommendation with situation analysis and rank impact
        """

        # Get base recommendation first
        base_recommendation = self.get_captain_recommendation(
            players, top_n=top_n, xp_column=xp_column
        )

        # Extract context with defaults
        context = manager_context or {}
        overall_rank = context.get("overall_rank", 2_000_000)  # Default to mid-table
        gameweek = context.get("gameweek", 15)  # Default to mid-season
        chip_active = context.get("chip_active", None)
        rank_trend = context.get("rank_trend", 0)  # 0 = stable
        strategy_override = context.get("strategy_override", None)

        # Analyze situation
        situation_analysis = self._analyze_captain_situation(
            overall_rank=overall_rank,
            gameweek=gameweek,
            chip_active=chip_active,
            rank_trend=rank_trend,
        )

        # Determine strategy (override or auto-detected)
        if strategy_override and strategy_override != "auto":
            strategy = strategy_override
            strategy_source = "user_override"
            logger.info(f"🎯 Captain strategy OVERRIDE: {strategy}")
        else:
            strategy = situation_analysis["recommended_strategy"]
            strategy_source = "auto_detected"
            logger.info(f"🎯 Captain strategy AUTO-DETECTED: {strategy}")

        # Enhance candidates with haul probabilities and rank impact
        enhanced_candidates = []

        for candidate in base_recommendation["top_candidates"]:
            enhanced = candidate.copy()

            # Calculate haul probabilities
            xp_value = candidate.get("xP", 0)
            uncertainty = candidate.get("xP_uncertainty", 1.5)  # Default uncertainty

            haul_probs = self._calculate_haul_probabilities(
                xp_value=xp_value,
                uncertainty=uncertainty,
            )
            enhanced["haul_probabilities"] = haul_probs

            # Calculate rank impact estimates
            ownership_pct = candidate.get("ownership_pct", 10.0)
            rank_impact = self._estimate_rank_impact(
                ownership_pct=ownership_pct,
                current_rank=overall_rank,
                haul_probs=haul_probs,
            )
            enhanced["rank_impact"] = rank_impact

            # Apply strategy-based scoring adjustment
            strategy_score = self._apply_strategy_scoring(
                candidate=enhanced,
                strategy=strategy,
                base_score=candidate.get("captain_score", xp_value * 2),
            )
            enhanced["strategy_score"] = strategy_score
            enhanced["strategy_applied"] = strategy

            enhanced_candidates.append(enhanced)

        # Re-sort by strategy score
        enhanced_candidates.sort(key=lambda x: x.get("strategy_score", 0), reverse=True)

        # Update rankings
        for i, candidate in enumerate(enhanced_candidates):
            candidate["strategy_rank"] = i + 1

        # Select captain and vice captain based on strategy
        strategy_captain = enhanced_candidates[0]
        strategy_vice = (
            enhanced_candidates[1] if len(enhanced_candidates) > 1 else strategy_captain
        )

        # Compare with template choice (if different)
        template_captain = None
        for candidate in enhanced_candidates:
            if (
                candidate.get("is_template", False)
                or candidate.get("ownership_pct", 0) > 50
            ):
                template_captain = candidate
                break

        # Build comparison insight
        template_comparison = None
        if (
            template_captain
            and template_captain["player_id"] != strategy_captain["player_id"]
        ):
            template_comparison = {
                "template_choice": template_captain["web_name"],
                "template_ownership": template_captain.get("ownership_pct", 0),
                "recommended_choice": strategy_captain["web_name"],
                "recommended_ownership": strategy_captain.get("ownership_pct", 0),
                "xp_difference": strategy_captain.get("xP", 0)
                - template_captain.get("xP", 0),
                "rank_upside_if_hauls": strategy_captain["rank_impact"].get(
                    "haul_rank_change", 0
                ),
                "rank_downside_if_blanks": strategy_captain["rank_impact"].get(
                    "blank_rank_change", 0
                ),
            }

        return {
            "captain": strategy_captain,
            "vice_captain": strategy_vice,
            "top_candidates": enhanced_candidates,
            "situation_analysis": situation_analysis,
            "strategy": {
                "mode": strategy,
                "source": strategy_source,
                "description": self._get_strategy_description(strategy),
            },
            "template_comparison": template_comparison,
            "captain_upside": strategy_captain.get("captain_points", 0),
            "vice_upside": strategy_vice.get("captain_points", 0),
            "differential": strategy_captain.get("captain_points", 0)
            - strategy_vice.get("captain_points", 0),
        }

    def _analyze_captain_situation(
        self,
        overall_rank: int,
        gameweek: int,
        chip_active: Optional[str],
        rank_trend: int,
    ) -> Dict[str, Any]:
        """Analyze the manager's current situation for captain strategy.

        Returns situation assessment and recommended strategy.
        """
        captain_config = config.captain_selection

        # Determine rank category
        if overall_rank <= captain_config.elite_rank_threshold:
            rank_category = "elite"
            rank_description = f"Top {captain_config.elite_rank_threshold // 1000}k - protect your position"
        elif overall_rank <= captain_config.comfortable_rank_threshold:
            rank_category = "comfortable"
            rank_description = f"Top {captain_config.comfortable_rank_threshold // 1000}k - balanced approach"
        elif overall_rank <= captain_config.chasing_rank_threshold:
            rank_category = "chasing"
            rank_description = f"Rank {overall_rank:,} - need differentials to climb"
        else:
            rank_category = "trailing"
            rank_description = f"Rank {overall_rank:,} - maximum upside needed"

        # Determine season phase
        if gameweek <= 10:
            season_phase = "early"
            phase_description = "Early season - can take calculated risks"
        elif gameweek <= 28:
            season_phase = "mid"
            phase_description = "Mid season - balance risk and reward"
        else:
            season_phase = "late"
            phase_description = "Late season - every point matters"

        # Determine momentum
        if rank_trend > 200_000:
            momentum = "surging"
            momentum_description = "Strong upward momentum - strategy is working"
        elif rank_trend > 0:
            momentum = "improving"
            momentum_description = "Positive trend - maintain approach"
        elif rank_trend > -200_000:
            momentum = "stable"
            momentum_description = "Stable rank - consider adjustments"
        else:
            momentum = "declining"
            momentum_description = "Losing ground - may need to take risks"

        # Chip influence
        chip_influence = None
        if chip_active:
            if chip_active.lower() == "freehit":
                chip_influence = "aggressive"
                chip_description = "Free Hit active - maximize single GW, take risks"
            elif chip_active.lower() == "wildcard":
                chip_influence = "balanced"
                chip_description = "Wildcard active - optimize for medium term"
            elif chip_active.lower() == "3xc":
                chip_influence = "ceiling_seeking"
                chip_description = "Triple Captain - need maximum ceiling pick"
            else:
                chip_description = f"{chip_active} active"
        else:
            chip_description = "No chip active"

        # Determine recommended strategy based on all factors
        if chip_active and chip_active.lower() == "3xc":
            recommended_strategy = "maximum_upside"
        elif chip_active and chip_active.lower() == "freehit":
            recommended_strategy = "chase_rank"
        elif rank_category == "elite":
            if season_phase == "late":
                recommended_strategy = "template_lock"
            else:
                recommended_strategy = "protect_rank"
        elif rank_category == "comfortable":
            recommended_strategy = "balanced"
        elif rank_category == "chasing":
            if momentum in ["surging", "improving"]:
                recommended_strategy = "balanced"  # Don't fix what's working
            else:
                recommended_strategy = "chase_rank"
        else:  # trailing
            recommended_strategy = "maximum_upside"

        return {
            "overall_rank": overall_rank,
            "gameweek": gameweek,
            "rank_category": rank_category,
            "rank_description": rank_description,
            "season_phase": season_phase,
            "phase_description": phase_description,
            "momentum": momentum,
            "momentum_description": momentum_description,
            "chip_active": chip_active,
            "chip_description": chip_description,
            "chip_influence": chip_influence,
            "recommended_strategy": recommended_strategy,
            "reasoning": self._build_strategy_reasoning(
                rank_category, season_phase, momentum, chip_active
            ),
        }

    def _calculate_haul_probabilities(
        self,
        xp_value: float,
        uncertainty: float,
    ) -> Dict[str, float]:
        """Calculate probability of blank, return, and haul outcomes.

        Uses normal distribution assumption with xP as mean and uncertainty as std.
        """
        from scipy import stats as scipy_stats

        captain_config = config.captain_selection

        # Ensure reasonable values
        xp_value = max(xp_value, 1.0)
        uncertainty = max(uncertainty, 0.5)

        # Create normal distribution
        dist = scipy_stats.norm(loc=xp_value, scale=uncertainty)

        # Calculate probabilities
        # P(blank) = P(X <= blank_threshold)
        blank_prob = dist.cdf(captain_config.blank_threshold)

        # P(return) = P(blank_threshold < X <= return_threshold)
        return_prob = dist.cdf(captain_config.return_threshold) - blank_prob

        # P(haul) = P(X > return_threshold)
        haul_prob = 1 - dist.cdf(captain_config.return_threshold)

        # Expected points for each outcome (conditional expectations)
        # Simplified: use threshold midpoints
        blank_expected = captain_config.blank_threshold / 2
        return_expected = (
            captain_config.blank_threshold + captain_config.return_threshold
        ) / 2
        haul_expected = captain_config.return_threshold + uncertainty  # Above threshold

        return {
            "blank_prob": round(blank_prob * 100, 1),  # As percentage
            "return_prob": round(return_prob * 100, 1),
            "haul_prob": round(haul_prob * 100, 1),
            "blank_threshold": captain_config.blank_threshold,
            "return_threshold": captain_config.return_threshold,
            "blank_expected": round(blank_expected, 1),
            "return_expected": round(return_expected, 1),
            "haul_expected": round(haul_expected, 1),
        }

    def _estimate_rank_impact(
        self,
        ownership_pct: float,
        current_rank: int,
        haul_probs: Dict[str, float],
    ) -> Dict[str, Any]:
        """Estimate rank impact for different captain outcomes.

        Higher ownership = less rank movement (everyone has them)
        Lower ownership = more rank movement (differential)
        """
        # Normalize ownership to 0-1 scale
        ownership_factor = ownership_pct / 100.0

        # Base rank impact scales inversely with ownership
        # Low ownership (5%) = high differential potential
        # High ownership (60%) = minimal rank movement
        differential_potential = 1.0 - ownership_factor

        # Estimate rank changes (rough heuristics based on FPL dynamics)
        # These are illustrative - actual rank changes depend on many factors

        # Blank scenario: lose ground relative to ownership
        # If 60% own player and they blank, you don't lose much
        # If 5% own player and they blank, you lose to the 95% who didn't captain them
        blank_rank_change = int(
            current_rank * ownership_factor * 0.05  # Small loss relative to ownership
        )

        # Return scenario: modest gain inversely proportional to ownership
        return_rank_change = int(
            -current_rank
            * differential_potential
            * 0.03  # Gain (negative = better rank)
        )

        # Haul scenario: significant gain inversely proportional to ownership
        haul_rank_change = int(
            -current_rank * differential_potential * 0.10  # Big gain
        )

        # Cap the changes to reasonable bounds
        max_change = current_rank // 2
        blank_rank_change = min(blank_rank_change, max_change)
        return_rank_change = max(return_rank_change, -max_change)
        haul_rank_change = max(haul_rank_change, -max_change)

        # Expected rank change (weighted by probabilities)
        expected_rank_change = (
            (haul_probs["blank_prob"] / 100) * blank_rank_change
            + (haul_probs["return_prob"] / 100) * return_rank_change
            + (haul_probs["haul_prob"] / 100) * haul_rank_change
        )

        return {
            "blank_rank_change": blank_rank_change,  # Positive = rank worsens
            "return_rank_change": return_rank_change,  # Negative = rank improves
            "haul_rank_change": haul_rank_change,
            "expected_rank_change": int(expected_rank_change),
            "differential_potential": round(differential_potential * 100, 1),
            "is_differential": ownership_pct < 20,
            "is_template": ownership_pct > 50,
            "ownership_category": (
                "template"
                if ownership_pct > 50
                else "popular"
                if ownership_pct > 20
                else "differential"
            ),
        }

    def _apply_strategy_scoring(
        self,
        candidate: Dict[str, Any],
        strategy: str,
        base_score: float,
    ) -> float:
        """Apply strategy-specific scoring adjustments.

        Returns adjusted captain score based on selected strategy.
        Uses aggressive multipliers to ensure strategy actually changes recommendations.
        """
        captain_config = config.captain_selection

        ownership_pct = candidate.get("ownership_pct", 10.0)
        haul_probs = candidate.get("haul_probabilities", {})
        rank_impact = candidate.get("rank_impact", {})

        score = base_score

        if strategy == "template_lock":
            # STRONGLY prefer highest-owned player
            if ownership_pct > captain_config.template_ownership_threshold:
                score *= 2.0  # Double score for template picks
            elif ownership_pct > captain_config.high_ownership_threshold:
                score *= 1.2  # Slight bonus for popular picks
            else:
                score *= 0.5  # Heavily penalize differentials

        elif strategy == "protect_rank":
            # Minimize downside, prefer safe template picks
            if ownership_pct > captain_config.high_ownership_threshold:
                score *= 1.5  # Strong bonus for safety
            # Penalize high-variance picks
            blank_prob = haul_probs.get("blank_prob", 30)
            if blank_prob > 35:
                score *= 0.6  # Significant penalty for high blank risk

        elif strategy == "balanced":
            # Current xP-weighted approach with template protection
            # Already applied in base score
            pass

        elif strategy == "chase_rank":
            # Differential focus - STRONGLY reward low ownership with upside
            differential_potential = rank_impact.get("differential_potential", 50) / 100
            haul_prob = haul_probs.get("haul_prob", 20) / 100

            # Big boost for differentials with haul potential
            # Low ownership (90% differential) + high haul (30%) = 1 + (0.9 * 0.3 * 3) = 1.81x
            diff_bonus = 1.0 + (differential_potential * haul_prob * 3.0)
            score *= diff_bonus

            # Strong penalty for template picks - we want differentials!
            if ownership_pct > captain_config.template_ownership_threshold:
                score *= 0.6
            elif ownership_pct > captain_config.high_ownership_threshold:
                score *= 0.8

        elif strategy == "maximum_upside":
            # Pure ceiling-seeking - maximize haul probability and rank gain
            haul_prob = haul_probs.get("haul_prob", 20) / 100
            differential_potential = rank_impact.get("differential_potential", 50) / 100

            # Heavily weight haul probability and differential potential
            # Player with 30% haul + 90% differential = 1 + 0.3 + 0.9 = 2.2x
            upside_bonus = 1.0 + haul_prob + differential_potential
            score *= upside_bonus

        logger.debug(
            f"Strategy scoring: {candidate.get('web_name')} | "
            f"strategy={strategy} | base={base_score:.1f} | final={score:.1f} | "
            f"ownership={ownership_pct:.1f}%"
        )

        return round(score, 2)

    def _get_strategy_description(self, strategy: str) -> str:
        """Get human-readable description for captain strategy."""
        descriptions = {
            "template_lock": "🔒 Template Lock: Always captain highest-owned player (>50%) to protect rank",
            "protect_rank": "🛡️ Protect Rank: Prefer safe picks with low blank probability",
            "balanced": "⚖️ Balanced: xP-weighted with template protection for good fixtures",
            "chase_rank": "🏃 Chase Rank: Target differentials with haul potential to climb",
            "maximum_upside": "🚀 Maximum Upside: Pure ceiling-seeking for maximum rank gains",
        }
        return descriptions.get(strategy, f"Unknown strategy: {strategy}")

    def _build_strategy_reasoning(
        self,
        rank_category: str,
        season_phase: str,
        momentum: str,
        chip_active: Optional[str],
    ) -> str:
        """Build reasoning text for strategy recommendation."""
        parts = []

        if chip_active:
            if chip_active.lower() == "3xc":
                parts.append("Triple Captain active - need maximum ceiling pick")
            elif chip_active.lower() == "freehit":
                parts.append("Free Hit active - one-week punt, can take risks")

        if rank_category == "elite":
            parts.append("At elite rank, protecting position is priority")
        elif rank_category == "trailing":
            parts.append("Behind target rank, need aggressive differentials")
        elif rank_category == "chasing":
            if momentum in ["surging", "improving"]:
                parts.append("Good momentum - maintain current approach")
            else:
                parts.append("Need differentials to make up ground")

        if season_phase == "late":
            parts.append("Late season - every point counts")
        elif season_phase == "early":
            parts.append("Early season - can afford calculated risks")

        return ". ".join(parts) if parts else "Standard captain selection"
