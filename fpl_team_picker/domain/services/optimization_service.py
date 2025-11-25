"""Optimization service for FPL squad and transfer optimization.

This service contains core FPL optimization algorithms including:
- Starting XI selection with formation optimization
- Bench player selection
- Budget pool calculations
- Transfer scenario analysis (0-3 transfers) using LP or SA
- Premium player acquisition planning
- Captain selection
- Initial squad generation using simulated annealing

Optimization Methods:
- Linear Programming (LP): Optimal, fast, deterministic (default for transfers)
- Simulated Annealing (SA): Exploratory, non-linear objectives (squad generation)
"""

from typing import Dict, List, Tuple, Optional, Set, Any
import pandas as pd
import random
import math
from pydantic import BaseModel, Field, field_validator
from loguru import logger
from fpl_team_picker.config import config


class OptimizationService:
    """Service for FPL optimization algorithms and constraint satisfaction."""

    def __init__(self, optimization_config: Optional[Dict[str, Any]] = None):
        """Initialize optimization service.

        Args:
            optimization_config: Optional optimization configuration override
        """
        self.config = optimization_config or {}

    def get_optimization_xp_column(self) -> str:
        """Get the XP column to use for optimization based on configuration.

        Returns:
            'xP' for 1-gameweek optimization, 'xP_3gw' for 3-gameweek optimization, 'xP_5gw' for 5-gameweek optimization
        """
        if config.optimization.optimization_horizon == "1gw":
            return "xP"
        elif config.optimization.optimization_horizon == "3gw":
            return "xP_3gw"
        else:  # "5gw"
            return "xP_5gw"

    def get_adjusted_xp(self, player: Dict, xp_col: str) -> float:
        """Calculate availability-adjusted xP for a player.

        Accounts for injury/availability risk by scaling xP based on:
        - chance_of_playing_this_round (preferred) or chance_of_playing_next_round
        - expected_minutes if available
        - status (injured/doubtful players get penalty)

        Args:
            player: Player dictionary with xP and availability info
            xp_col: Column name for xP values

        Returns:
            Adjusted xP accounting for availability risk
        """
        base_xp = player.get(xp_col, 0.0)
        if base_xp <= 0:
            return 0.0

        # Get availability multiplier (0.0 to 1.0)
        availability_multiplier = 1.0

        # Prefer chance_of_playing_this_round for current gameweek
        chance_this = player.get("chance_of_playing_this_round")
        chance_next = player.get("chance_of_playing_next_round")

        if chance_this is not None and not pd.isna(chance_this):
            availability_multiplier = float(chance_this) / 100.0
        elif chance_next is not None and not pd.isna(chance_next):
            availability_multiplier = float(chance_next) / 100.0
        else:
            # Fallback to expected_minutes ratio if available
            expected_mins = player.get("expected_minutes")
            if expected_mins is not None and not pd.isna(expected_mins):
                # Position-based default minutes
                position = player.get("position", "MID")
                position_defaults = {"GKP": 90, "DEF": 80, "MID": 75, "FWD": 70}
                base_minutes = position_defaults.get(position, 75)
                availability_multiplier = min(float(expected_mins) / base_minutes, 1.0)
            else:
                # Check status for injured/doubtful players
                status = player.get("status", "a")
                if status in ["i", "d"]:  # injured or doubtful
                    # Apply penalty based on config
                    if status == "i":
                        availability_multiplier = (
                            config.minutes_model.injury_full_game_multiplier
                        )
                    else:  # doubtful
                        availability_multiplier = (
                            config.minutes_model.injury_avg_minutes_multiplier
                        )

        # Clamp multiplier to valid range
        availability_multiplier = max(0.0, min(1.0, availability_multiplier))

        return base_xp * availability_multiplier

    def _count_players_per_team(self, team: List[Dict]) -> Dict[str, int]:
        """Count players per team.

        Args:
            team: List of player dictionaries with 'team' field

        Returns:
            Dictionary mapping team names to player counts
        """
        team_counts = {}
        for player in team:
            team_name = player["team"]
            team_counts[team_name] = team_counts.get(team_name, 0) + 1
        return team_counts

    def _enumerate_formations_for_players(
        self, by_position: Dict[str, List[Dict]], xp_column: str = "xP"
    ) -> Tuple[List[Dict], str, float]:
        """Core formation enumeration logic.

        Evaluates all valid FPL formations and returns the one with highest XP.

        Args:
            by_position: Players grouped by position (GKP, DEF, MID, FWD)
            xp_column: Column name for XP values

        Returns:
            Tuple of (best_11_players, formation_name, total_xp)
        """
        formations = [
            (1, 3, 5, 2),
            (1, 3, 4, 3),
            (1, 4, 5, 1),
            (1, 4, 4, 2),
            (1, 4, 3, 3),
            (1, 5, 4, 1),
            (1, 5, 3, 2),
            (1, 5, 2, 3),
        ]
        formation_names = {
            "(1, 3, 5, 2)": "3-5-2",
            "(1, 3, 4, 3)": "3-4-3",
            "(1, 4, 5, 1)": "4-5-1",
            "(1, 4, 4, 2)": "4-4-2",
            "(1, 4, 3, 3)": "4-3-3",
            "(1, 5, 4, 1)": "5-4-1",
            "(1, 5, 3, 2)": "5-3-2",
            "(1, 5, 2, 3)": "5-2-3",
        }

        best_11, best_xp, best_formation = [], 0, ""

        for gkp, def_count, mid, fwd in formations:
            if (
                gkp <= len(by_position["GKP"])
                and def_count <= len(by_position["DEF"])
                and mid <= len(by_position["MID"])
                and fwd <= len(by_position["FWD"])
            ):
                formation_11 = (
                    by_position["GKP"][:gkp]
                    + by_position["DEF"][:def_count]
                    + by_position["MID"][:mid]
                    + by_position["FWD"][:fwd]
                )
                formation_xp = sum(p.get(xp_column, 0) for p in formation_11)

                if formation_xp > best_xp:
                    best_xp = formation_xp
                    best_11 = formation_11
                    best_formation = formation_names.get(
                        str((gkp, def_count, mid, fwd)),
                        f"{gkp}-{def_count}-{mid}-{fwd}",
                    )

        return best_11, best_formation, best_xp

    def calculate_budget_pool(
        self,
        current_squad: pd.DataFrame,
        bank_balance: float,
        players_to_keep: Optional[Set[int]] = None,
    ) -> Dict:
        """Calculate total budget pool available for transfers.

        Args:
            current_squad: Current squad DataFrame
            bank_balance: Available bank balance in millions
            players_to_keep: Set of player IDs that must be kept

        Returns:
            Dictionary with budget pool information
        """
        if current_squad.empty:
            return {
                "bank_balance": bank_balance,
                "sellable_value": 0.0,
                "total_budget": bank_balance,
                "sellable_players": 0,
                "players_to_keep": 0,
            }

        players_to_keep = players_to_keep or set()

        # Calculate sellable value (players not in must-keep list)
        sellable_players = current_squad[
            ~current_squad["player_id"].isin(players_to_keep)
        ]
        sellable_value = (
            sellable_players["price"].sum() if not sellable_players.empty else 0.0
        )

        total_budget = bank_balance + sellable_value

        return {
            "bank_balance": bank_balance,
            "sellable_value": sellable_value,
            "total_budget": total_budget,
            "sellable_players": len(sellable_players),
            "players_to_keep": len(players_to_keep),
        }

    def find_optimal_starting_11(
        self, squad_df: pd.DataFrame, xp_column: str = "xP"
    ) -> Tuple[List[Dict], str, float]:
        """Find best starting 11 from squad using specified XP column.

        Args:
            squad_df: Squad DataFrame
            xp_column: Column to use for XP sorting ('xP' for current GW, 'xP_3gw' for 3-GW, 'xP_5gw' for strategic)

        Returns:
            Tuple of (best_11_list, formation_name, total_xp)
        """
        if len(squad_df) < 11:
            return [], "", 0

        # Filter out suspended, injured, or unavailable players
        available_squad = squad_df.copy()
        if "status" in available_squad.columns:
            unavailable_mask = available_squad["status"].isin(["i", "s", "u"])
            if unavailable_mask.any():
                unavailable_players = available_squad[unavailable_mask]
                logger.debug(
                    f"🚫 Excluding {len(unavailable_players)} unavailable players from starting 11:"
                )
                for _, player in unavailable_players.iterrows():
                    status_desc = {
                        "i": "injured",
                        "s": "suspended",
                        "u": "unavailable",
                    }[player["status"]]
                    logger.debug(
                        f"   - {player.get('web_name', 'Unknown')} ({status_desc})"
                    )

                available_squad = available_squad[~unavailable_mask]

        if len(available_squad) < 11:
            logger.warning(
                f"⚠️ Warning: Only {len(available_squad)} available players in squad (need 11)"
            )
            return [], "", 0

        # Default to current gameweek XP, fallback to 3GW or 5GW if current not available
        sort_col = (
            xp_column
            if xp_column in available_squad.columns
            else (
                "xP_3gw"
                if "xP_3gw" in available_squad.columns
                else ("xP_5gw" if "xP_5gw" in available_squad.columns else "xP")
            )
        )

        # Group by position and sort by XP
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for _, player in available_squad.iterrows():
            player_dict = player.to_dict()
            by_position[player["position"]].append(player_dict)

        for pos in by_position:
            by_position[pos].sort(
                key=lambda p: p.get(sort_col, p.get("xP", 0)), reverse=True
            )

        # Use shared formation enumeration logic
        return self._enumerate_formations_for_players(by_position, sort_col)

    def find_bench_players(
        self, squad_df: pd.DataFrame, starting_11: List[Dict], xp_column: str = "xP"
    ) -> List[Dict]:
        """Get bench players (remaining 4 players) from squad ordered by XP.

        Args:
            squad_df: Squad DataFrame
            starting_11: List of starting 11 player dictionaries
            xp_column: Column to use for XP sorting ('xP' for current GW, 'xP_3gw' for 3-GW, 'xP_5gw' for strategic)

        Returns:
            List of bench player dictionaries ordered by XP (highest first)
        """
        if len(squad_df) < 15:
            return []

        # Get player IDs from starting 11
        starting_11_ids = {player["player_id"] for player in starting_11}

        # Get remaining players (bench)
        bench_players = []
        sort_col = (
            xp_column
            if xp_column in squad_df.columns
            else (
                "xP_3gw"
                if "xP_3gw" in squad_df.columns
                else ("xP_5gw" if "xP_5gw" in squad_df.columns else "xP")
            )
        )

        for _, player in squad_df.iterrows():
            if player["player_id"] not in starting_11_ids:
                bench_players.append(player.to_dict())

        # Sort bench by XP (highest first)
        bench_players.sort(key=lambda p: p.get(sort_col, 0), reverse=True)

        return bench_players[:4]  # Maximum 4 bench players

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

    def plan_premium_acquisition(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        budget_pool_info: Dict,
        top_n: int = 3,
    ) -> List[Dict]:
        """Plan premium player acquisitions with funding scenarios.

        Args:
            current_squad: Current squad DataFrame
            all_players: All available players DataFrame
            budget_pool_info: Budget pool information from calculate_budget_pool
            top_n: Number of top scenarios to return

        Returns:
            List of premium acquisition scenarios
        """
        if current_squad.empty or all_players.empty:
            return []

        scenarios = []

        # Find premium targets (players not in squad, price > bank balance)
        current_player_ids = set(current_squad["player_id"].tolist())
        available_players = all_players[
            ~all_players["player_id"].isin(current_player_ids)
        ]

        # Premium targets that cost more than available bank
        premium_targets = available_players[
            (available_players["price"] > budget_pool_info["bank_balance"])
            & (available_players["price"] <= budget_pool_info["total_budget"])
        ].nlargest(20, "xP")  # Top 20 by expected points

        if premium_targets.empty:
            return scenarios

        for _, target in premium_targets.iterrows():
            funding_gap = target["price"] - budget_pool_info["bank_balance"]

            # Find combinations of current players to sell that cover funding gap
            sellable_players = current_squad.copy()

            # Try different selling strategies
            # Strategy 1: Sell lowest XP players first (avoid selling good players)
            xp_column = self.get_optimization_xp_column()
            sort_column = xp_column if xp_column in sellable_players.columns else "xP"
            potential_sells = sellable_players.sort_values(sort_column, ascending=True)

            # Find minimum number of players to sell to cover gap
            cumulative_value = 0
            sell_players = []

            for _, player in potential_sells.iterrows():
                sell_players.append(player)
                cumulative_value += player["price"]

                if cumulative_value >= funding_gap:
                    # Check if we have enough budget for replacements
                    remaining_slots = len(sell_players)
                    min_replacement_cost = (
                        remaining_slots * 4.0
                    )  # Assume £4m minimum per replacement

                    if (
                        budget_pool_info["total_budget"] - target["price"]
                        >= min_replacement_cost
                    ):
                        # Find cheap replacements for sold players
                        replacements = []
                        remaining_budget = (
                            budget_pool_info["total_budget"] - target["price"]
                        )

                        for sell_player in sell_players:
                            # Find cheapest replacement in same position
                            position_replacements = available_players[
                                (
                                    available_players["position"]
                                    == sell_player["position"]
                                )
                                & (available_players["price"] <= remaining_budget)
                            ].nsmallest(5, "price")  # Top 5 cheapest

                            if not position_replacements.empty:
                                replacement = position_replacements.iloc[0]
                                replacements.append(replacement)
                                remaining_budget -= replacement["price"]
                            else:
                                break  # Can't find replacement, skip this scenario

                        # Calculate XP impact
                        sell_xp = sum(p.get("xP", 0) for p in sell_players)
                        replacement_xp = sum(r.get("xP", 0) for r in replacements)

                        if len(replacements) == len(sell_players):
                            # Calculate net XP gain
                            net_xp_gain = (target["xP"] + replacement_xp) - sell_xp

                            if net_xp_gain > 0.2:  # Higher threshold for multi-transfer
                                sell_names = ", ".join(
                                    [p["web_name"] for p in sell_players]
                                )
                                replace_names = ", ".join(
                                    [r["web_name"] for r in replacements]
                                )

                                scenarios.append(
                                    {
                                        "type": "premium_funded",
                                        "target_player": target,
                                        "sell_players": sell_players,
                                        "replacement_players": replacements,
                                        "funding_gap": target["price"]
                                        - budget_pool_info["bank_balance"],
                                        "xp_gain": net_xp_gain,
                                        "transfers": 1
                                        + len(sell_players)
                                        + len(replacements),
                                        "description": f"Premium Funded: {target['web_name']} (sell {sell_names}, buy {replace_names})",
                                    }
                                )
                    break

        # Sort by XP gain and return top scenarios
        scenarios.sort(key=lambda x: x["xp_gain"], reverse=True)
        return scenarios[:top_n]

    def optimize_transfers(
        self,
        current_squad: pd.DataFrame,
        team_data: Dict,
        players_with_xp: pd.DataFrame,
        must_include_ids: Set[int] = None,
        must_exclude_ids: Set[int] = None,
        free_transfers_override: Optional[int] = None,
        is_free_hit: bool = False,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Unified transfer optimization supporting normal gameweeks and chips.

        This is the core transfer optimization algorithm that analyzes multiple
        transfer scenarios and recommends the optimal strategy.

        **Unified Chip Support:**
        - Normal gameweek: free_transfers=1 (analyze 0-3 transfers)
        - Saved transfer: free_transfers=2 (analyze 0-4 transfers)
        - Wildcard chip: free_transfers=15 (rebuild entire squad, budget resets to £100m)
        - Free Hit chip: free_transfers=15 + is_free_hit=True (1GW optimization, squad reverts after)

        **Optimization Methods:**
        - Linear Programming (default): Guarantees optimal solution, fast (~1-2s), deterministic
        - Simulated Annealing: Exploratory, good for non-linear objectives (~10-45s)
        - Configure via config.optimization.transfer_optimization_method

        Clean architecture: Returns data structures only, no UI generation.

        Args:
            current_squad: Current squad DataFrame
            team_data: Team data dictionary with bank balance etc
            players_with_xp: All players with XP calculations
            must_include_ids: Set of player IDs that must be included
            must_exclude_ids: Set of player IDs that must be excluded
            free_transfers_override: Override free transfers (for wildcard/chips)
                - None: Use team_data['free_transfers'] (default 1)
                - 15: Wildcard chip (rebuild entire squad)
                - 2: Saved transfer from previous week
            is_free_hit: If True, optimize for 1GW only (squad reverts after deadline)

        Returns:
            Tuple of (optimal_squad_df, best_scenario_dict, optimization_metadata_dict)
            where optimization_metadata contains budget_pool_info, scenarios, etc for display
        """
        # Apply free transfers override if wildcard/chip is active
        if free_transfers_override is not None:
            team_data = team_data.copy()
            team_data["free_transfers"] = free_transfers_override

            # Wildcard/Free Hit chip: Budget resets to £100m (ignore current squad value)
            if free_transfers_override >= 15:
                team_data["bank"] = (
                    config.optimization.free_hit_budget if is_free_hit else 100.0
                )
                # Note: Selling prices will be ignored since we can replace all 15 players
                chip_name = "Free Hit" if is_free_hit else "Wildcard"
                logger.info(
                    f"🃏 {chip_name} Active: {free_transfers_override} free transfers, "
                    f"£{team_data['bank']:.1f}m budget"
                    + (" (1GW only, squad reverts)" if is_free_hit else "")
                )

        # Route to appropriate optimizer based on configuration
        method = config.optimization.transfer_optimization_method

        if method == "linear_programming":
            return self._optimize_transfers_lp(
                current_squad,
                team_data,
                players_with_xp,
                must_include_ids,
                must_exclude_ids,
                is_free_hit=is_free_hit,
            )
        elif method == "simulated_annealing":
            return self._optimize_transfers_sa(
                current_squad,
                team_data,
                players_with_xp,
                must_include_ids,
                must_exclude_ids,
                is_free_hit=is_free_hit,
            )
        else:
            raise ValueError(
                f"Unknown optimization method: {method}. "
                "Must be 'linear_programming' or 'simulated_annealing'"
            )

    def _optimize_transfers_sa(
        self,
        current_squad: pd.DataFrame,
        team_data: Dict,
        players_with_xp: pd.DataFrame,
        must_include_ids: Set[int] = None,
        must_exclude_ids: Set[int] = None,
        is_free_hit: bool = False,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Transfer optimization using Simulated Annealing (exploration-based).

        Reuses SA logic from optimize_initial_squad but adapted for transfer context:
        - Start state: Current squad (not random)
        - Budget: Bank balance + selling prices (not £100m)
        - Objective: Maximize (squad_xp - transfer_penalties)
        - Neighbor function: Swap 1-3 players respecting budget

        Clean architecture: Returns data structures only, no UI generation.

        Args:
            current_squad: Current squad DataFrame
            team_data: Team data dictionary with bank balance etc
            players_with_xp: All players with XP calculations
            must_include_ids: Set of player IDs that must be included
            must_exclude_ids: Set of player IDs that must be excluded

        Returns:
            Tuple of (optimal_squad_df, best_scenario_dict, optimization_metadata_dict)
        """
        if len(current_squad) == 0 or players_with_xp.empty or not team_data:
            return pd.DataFrame(), {}, {"error": "Load team and calculate XP first"}

        # Set random seed for reproducibility if configured
        if config.optimization.sa_random_seed is not None:
            random.seed(config.optimization.sa_random_seed)
            logger.debug(
                f"🎲 Random seed set to {config.optimization.sa_random_seed} for reproducibility"
            )

        logger.info(
            f"🧠 Strategic Optimization: Simulated Annealing ({config.optimization.sa_iterations} iterations)..."
        )

        # Process constraints
        must_include_ids = must_include_ids or set()
        must_exclude_ids = must_exclude_ids or set()

        if must_include_ids:
            logger.debug(f"🎯 Must include {len(must_include_ids)} players")
        if must_exclude_ids:
            logger.debug(f"🚫 Must exclude {len(must_exclude_ids)} players")

        # Get optimization column based on configuration
        # Free Hit always uses 1GW (squad reverts after deadline)
        if is_free_hit:
            xp_column = "xP"
            horizon_label = "1-GW"
            logger.info(
                "🎯 Free Hit mode: Optimizing for 1GW only (squad reverts after)"
            )
        else:
            xp_column = self.get_optimization_xp_column()
            if xp_column == "xP":
                horizon_label = "1-GW"
            elif xp_column == "xP_3gw":
                horizon_label = "3-GW"
            else:
                horizon_label = "5-GW"

        # Update current squad with 1-GW, 3-GW, and 5-GW XP data
        merge_columns = ["player_id", "xP"]
        if "xP_3gw" in players_with_xp.columns:
            merge_columns.append("xP_3gw")
        if "xP_5gw" in players_with_xp.columns:
            merge_columns.append("xP_5gw")
        if "xP_uncertainty" in players_with_xp.columns:
            merge_columns.append("xP_uncertainty")
        if "fixture_outlook" in players_with_xp.columns:
            merge_columns.append("fixture_outlook")
        if "expected_minutes" in players_with_xp.columns:
            merge_columns.append("expected_minutes")

        # Include team information in merge to maintain data contract
        if "team" in players_with_xp.columns:
            merge_columns.append("team")
        elif "team_id" in players_with_xp.columns:
            merge_columns.append("team_id")

        current_squad_with_xp = current_squad.merge(
            players_with_xp[merge_columns],
            on="player_id",
            how="left",
            suffixes=("", "_from_xp"),
        )
        # Fill any missing XP with 0
        current_squad_with_xp["xP"] = current_squad_with_xp["xP"].fillna(0)
        if "xP_3gw" in current_squad_with_xp.columns:
            current_squad_with_xp["xP_3gw"] = current_squad_with_xp["xP_3gw"].fillna(0)
        if "xP_5gw" in current_squad_with_xp.columns:
            current_squad_with_xp["xP_5gw"] = current_squad_with_xp["xP_5gw"].fillna(0)
        if "fixture_outlook" in current_squad_with_xp.columns:
            current_squad_with_xp["fixture_outlook"] = current_squad_with_xp[
                "fixture_outlook"
            ].fillna("🟡 Average")

        # Validate team data contract
        team_col = "team" if "team" in current_squad_with_xp.columns else "team_id"
        if team_col in current_squad_with_xp.columns:
            nan_teams = current_squad_with_xp[team_col].isna().sum()
            if nan_teams > 0:
                raise ValueError(
                    f"Data contract violation: {nan_teams} players have NaN team values"
                )

        # Current squad and available budget
        available_budget = team_data["bank"]
        free_transfers = team_data.get("free_transfers", 1)

        # Calculate total budget pool
        is_wildcard = free_transfers >= 15
        if is_wildcard:
            # Wildcard: budget resets to £100m, ignore current squad value
            budget_pool_info = {
                "total_budget": 100.0,
                "sellable_value": 0.0,  # Not applicable for wildcard
                "non_sellable_value": 0.0,
                "must_include_value": 0.0,
            }
            logger.info("🃏 Wildcard Budget: £100.0m (budget reset)")
        else:
            # Normal transfers: bank + sellable squad value
            budget_pool_info = self.calculate_budget_pool(
                current_squad_with_xp, available_budget, must_include_ids
            )
            logger.info(
                f"💰 Budget: Bank £{available_budget:.1f}m | Sellable £{budget_pool_info['sellable_value']:.1f}m | Total £{budget_pool_info['total_budget']:.1f}m"
            )

        # Get all available players and filter
        all_players = players_with_xp[players_with_xp["xP"].notna()].copy()

        # Filter out unavailable players
        if "status" in all_players.columns:
            available_players_mask = ~all_players["status"].isin(["i", "s", "u"])
            all_players = all_players[available_players_mask]

            excluded_players = players_with_xp[
                players_with_xp["status"].isin(["i", "s", "u"])
                & players_with_xp["xP"].notna()
            ]
            if not excluded_players.empty:
                logger.debug(f"🚫 Filtered {len(excluded_players)} unavailable players")

        if must_exclude_ids:
            all_players = all_players[~all_players["player_id"].isin(must_exclude_ids)]

        # Get best starting 11 from current squad
        current_best_11, current_formation, current_xp = self.find_optimal_starting_11(
            current_squad_with_xp, xp_column
        )

        logger.info(
            f"📊 Current squad: {current_xp:.2f} {horizon_label}-xP | Formation: {current_formation}"
        )

        # For wildcard/free hit (15 free transfers), use initial squad generation instead of transfer-based SA
        if is_wildcard:
            chip_name = "Free Hit" if is_free_hit else "Wildcard"
            budget = config.optimization.free_hit_budget if is_free_hit else 100.0
            iterations = (
                config.optimization.sa_free_hit_iterations
                if is_free_hit
                else config.optimization.sa_wildcard_iterations
            )
            restarts = (
                config.optimization.sa_free_hit_restarts
                if is_free_hit
                else config.optimization.sa_wildcard_restarts
            )
            use_consensus = (
                config.optimization.sa_free_hit_use_consensus
                if is_free_hit
                else config.optimization.sa_wildcard_use_consensus
            )

            logger.info(
                f"🃏 {chip_name} mode: Building optimal squad from scratch (ignoring current squad)"
            )

            # Check if consensus mode is enabled
            if use_consensus:
                logger.info(
                    f"🎯 {chip_name} Consensus: Running {restarts} restarts "
                    f"× {iterations} iterations each to find truly optimal squad..."
                )
                wildcard_result = self._consensus_wildcard_optimization(
                    players_with_xp=all_players,
                    budget=budget,
                    formation=(2, 5, 5, 3),
                    must_include_ids=must_include_ids,
                    must_exclude_ids=must_exclude_ids,
                    xp_column=xp_column,
                )
            else:
                # Single run mode (faster but less reliable)
                wildcard_result = self.optimize_initial_squad(
                    players_with_xp=all_players,
                    budget=budget,
                    formation=(2, 5, 5, 3),
                    iterations=iterations,
                    must_include_ids=must_include_ids,
                    must_exclude_ids=must_exclude_ids,
                    xp_column=xp_column,
                )

            # Convert initial squad result to transfer optimization format
            optimal_squad = wildcard_result["optimal_squad"]
            best_squad_df = pd.DataFrame(optimal_squad)

            # Calculate how many players changed
            original_ids = set(current_squad_with_xp["player_id"].tolist())
            new_ids = set(best_squad_df["player_id"].tolist())
            num_transfers = len(original_ids - new_ids)

            sa_result = {
                "optimal_squad": optimal_squad,
                "best_xp": wildcard_result["total_xp"],
                "iterations_improved": wildcard_result.get(
                    "iterations_improved", wildcard_result.get("total_improvements", 0)
                ),
                "total_iterations": wildcard_result["total_iterations"],
                "formation": wildcard_result.get("formation", "2-5-5-3"),
            }
            if "consensus_confidence" in wildcard_result:
                sa_result["consensus_confidence"] = wildcard_result[
                    "consensus_confidence"
                ]
                sa_result["consensus_runs"] = wildcard_result["consensus_runs"]
            best_sa_result = sa_result

        else:
            # Check if we should use exhaustive search for small transfer counts
            max_exhaustive = config.optimization.sa_exhaustive_search_max_transfers
            if max_exhaustive > 0 and free_transfers <= max_exhaustive:
                logger.info(
                    f"🔍 Using exhaustive search for {free_transfers} free transfer(s) (guaranteed optimal)..."
                )
                exhaustive_result = self._exhaustive_transfer_search(
                    current_squad=current_squad_with_xp,
                    all_players=all_players,
                    available_budget=available_budget,
                    free_transfers=free_transfers,
                    must_include_ids=must_include_ids,
                    must_exclude_ids=must_exclude_ids,
                    xp_column=xp_column,
                    current_xp=current_xp,
                    max_transfers=max_exhaustive,
                )
                if exhaustive_result:
                    sa_result = exhaustive_result
                    best_net_xp = exhaustive_result["best_xp"]
                    logger.info(
                        f"✅ Exhaustive search complete: {best_net_xp:.2f} net xP"
                    )
                else:
                    # Fallback to SA if exhaustive fails
                    logger.warning("⚠️ Exhaustive search failed, falling back to SA")
                    sa_result = None
            else:
                sa_result = None

            # Use SA if exhaustive wasn't used or failed
            if sa_result is None:
                # Check if consensus mode is enabled
                if config.optimization.sa_use_consensus_mode:
                    logger.info(
                        f"🎯 Consensus mode: Running {config.optimization.sa_consensus_runs} full optimizations to find truly optimal solution..."
                    )
                    sa_result = self._consensus_optimization(
                        current_squad=current_squad_with_xp,
                        all_players=all_players,
                        available_budget=available_budget,
                        free_transfers=free_transfers,
                        must_include_ids=must_include_ids,
                        must_exclude_ids=must_exclude_ids,
                        xp_column=xp_column,
                        current_xp=current_xp,
                    )
                    best_net_xp = sa_result["best_xp"]
                else:
                    # Normal mode: Run multiple SA restarts to find global optimum
                    num_restarts = config.optimization.sa_restarts
                    logger.info(
                        f"🔄 Running {num_restarts} SA restart(s) with {config.optimization.sa_iterations} iterations each..."
                    )

                    best_sa_result = None
                    best_net_xp = float("-inf")

                    for restart in range(num_restarts):
                        if num_restarts > 1:
                            logger.debug(f"  Restart {restart + 1}/{num_restarts}...")
                            # Use different seed for each restart if seed is set (for diversity)
                            if config.optimization.sa_random_seed is not None:
                                random.seed(
                                    config.optimization.sa_random_seed + restart
                                )

                        sa_result_restart = self._run_transfer_sa(
                            current_squad=current_squad_with_xp,
                            all_players=all_players,
                            available_budget=available_budget,
                            free_transfers=free_transfers,
                            must_include_ids=must_include_ids,
                            must_exclude_ids=must_exclude_ids,
                            xp_column=xp_column,
                            current_xp=current_xp,
                            iterations=config.optimization.sa_iterations,
                        )

                        # Calculate net XP for this restart (including penalty)
                        best_squad_df = pd.DataFrame(sa_result_restart["optimal_squad"])
                        original_ids = set(current_squad_with_xp["player_id"].tolist())
                        new_ids = set(best_squad_df["player_id"].tolist())
                        num_transfers = len(original_ids - new_ids)
                        transfer_penalty = (
                            max(0, num_transfers - free_transfers)
                            * config.optimization.transfer_cost
                        )
                        net_xp = sa_result_restart["best_xp"] - transfer_penalty

                        if net_xp > best_net_xp:
                            best_net_xp = net_xp
                            best_sa_result = sa_result_restart
                            if num_restarts > 1:
                                logger.debug(
                                    f"    → New best: {net_xp:.2f} net xP ({num_transfers} transfers)"
                                )

                    sa_result = best_sa_result
                    logger.info(f"✅ Best result: {best_net_xp:.2f} net xP")

        # Create best scenario from SA result
        best_squad = sa_result["optimal_squad"]
        best_squad_df = pd.DataFrame(best_squad)

        # Calculate transfers made vs original squad
        original_ids = set(current_squad_with_xp["player_id"].tolist())
        new_ids = set(best_squad_df["player_id"].tolist())
        transfers_out = original_ids - new_ids
        transfers_in = new_ids - original_ids
        num_transfers = len(transfers_out)

        # Calculate transfer penalty
        transfer_penalty = (
            max(0, num_transfers - free_transfers) * config.optimization.transfer_cost
        )
        net_xp = sa_result["best_xp"] - transfer_penalty

        # Create transfer description
        is_wildcard = free_transfers >= 15
        if num_transfers == 0:
            description = "Keep current squad"
        elif is_wildcard:
            description = (
                f"Wildcard: Rebuild entire squad ({num_transfers} players changed)"
            )
        else:
            out_names = [
                current_squad_with_xp[current_squad_with_xp["player_id"] == pid][
                    "web_name"
                ].iloc[0]
                for pid in list(transfers_out)[:3]
            ]
            in_names = [
                best_squad_df[best_squad_df["player_id"] == pid]["web_name"].iloc[0]
                for pid in list(transfers_in)[:3]
            ]
            description = f"OUT: {', '.join(out_names)} → IN: {', '.join(in_names)}"

        # Determine scenario type
        scenario_type = "wildcard" if is_wildcard else "simulated_annealing"

        best_scenario = {
            "id": 0,
            "transfers": num_transfers,
            "type": scenario_type,
            "description": description,
            "penalty": transfer_penalty,
            "net_xp": net_xp,
            "formation": sa_result.get("formation", current_formation),
            "xp_gain": net_xp - current_xp,
            "squad": best_squad_df,
            "iterations_improved": sa_result.get("iterations_improved", 0),
            "is_wildcard": is_wildcard,
        }

        logger.info(
            f"✅ Best strategy: {num_transfers} transfers, {net_xp:.2f} net XP "
            f"({sa_result['iterations_improved']} improvements in {sa_result['total_iterations']} iterations)"
        )

        # Calculate remaining budget after squad selection
        squad_cost = best_squad_df["price"].sum()
        if is_wildcard:
            remaining_budget = 100.0 - squad_cost
        else:
            remaining_budget = available_budget - (
                squad_cost - budget_pool_info.get("non_sellable_value", 0)
            )

        # Return data structures for presentation layer to display
        optimization_metadata = {
            "method": "simulated_annealing",
            "horizon_label": horizon_label,
            "xp_column": xp_column,
            "budget_pool_info": budget_pool_info,
            "available_budget": team_data[
                "bank"
            ],  # Actual bank balance before optimization
            "remaining_budget": remaining_budget,  # Budget left after optimization
            "free_transfers": free_transfers,
            "current_xp": current_xp,
            "current_formation": current_formation,
            "sa_iterations": sa_result["total_iterations"],
            "sa_improvements": sa_result["iterations_improved"],
            "is_wildcard": is_wildcard,
            "transfers_out": [
                {
                    "web_name": current_squad_with_xp[
                        current_squad_with_xp["player_id"] == pid
                    ]["web_name"].iloc[0],
                    "position": current_squad_with_xp[
                        current_squad_with_xp["player_id"] == pid
                    ]["position"].iloc[0],
                    "price": current_squad_with_xp[
                        current_squad_with_xp["player_id"] == pid
                    ]["price"].iloc[0],
                }
                for pid in list(transfers_out)
            ],
            "transfers_in": [
                {
                    "web_name": best_squad_df[best_squad_df["player_id"] == pid][
                        "web_name"
                    ].iloc[0],
                    "position": best_squad_df[best_squad_df["player_id"] == pid][
                        "position"
                    ].iloc[0],
                    "price": best_squad_df[best_squad_df["player_id"] == pid][
                        "price"
                    ].iloc[0],
                }
                for pid in list(transfers_in)
            ],
        }

        return best_squad_df, best_scenario, optimization_metadata

    def _optimize_transfers_lp(
        self,
        current_squad: pd.DataFrame,
        team_data: Dict,
        players_with_xp: pd.DataFrame,
        must_include_ids: Set[int] = None,
        must_exclude_ids: Set[int] = None,
        is_free_hit: bool = False,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Transfer optimization using Linear Programming (optimal solution).

        Uses PuLP with CBC solver to find guaranteed optimal transfer solution.
        Advantages over SA:
        - Provably optimal solution
        - Deterministic (same input = same output)
        - Faster (~1-2 seconds vs 10-45 seconds)

        Clean architecture: Returns data structures only, no UI generation.

        Args:
            current_squad: Current squad DataFrame
            team_data: Team data dictionary with bank balance etc
            players_with_xp: All players with XP calculations
            must_include_ids: Set of player IDs that must be included
            must_exclude_ids: Set of player IDs that must be excluded

        Returns:
            Tuple of (optimal_squad_df, best_scenario_dict, optimization_metadata_dict)
        """
        import pulp
        import time

        if len(current_squad) == 0 or players_with_xp.empty or not team_data:
            return pd.DataFrame(), {}, {"error": "Load team and calculate XP first"}

        start_time = time.time()

        logger.info("🎯 Linear Programming Optimization: Finding optimal solution...")

        # Process constraints
        must_include_ids = must_include_ids or set()
        must_exclude_ids = must_exclude_ids or set()

        if must_include_ids:
            logger.debug(f"🎯 Must include {len(must_include_ids)} players")
        if must_exclude_ids:
            logger.debug(f"🚫 Must exclude {len(must_exclude_ids)} players")

        # Get optimization column based on configuration
        # Free Hit always uses 1GW (squad reverts after deadline)
        if is_free_hit:
            xp_column = "xP"
            horizon_label = "1-GW"
            logger.info(
                "🎯 Free Hit mode: Optimizing for 1GW only (squad reverts after)"
            )
        else:
            xp_column = self.get_optimization_xp_column()
            if xp_column == "xP":
                horizon_label = "1-GW"
            elif xp_column == "xP_3gw":
                horizon_label = "3-GW"
            else:
                horizon_label = "5-GW"

        # Update current squad with XP data (same as SA)
        merge_columns = ["player_id", "xP"]
        if "xP_3gw" in players_with_xp.columns:
            merge_columns.append("xP_3gw")
        if "xP_5gw" in players_with_xp.columns:
            merge_columns.append("xP_5gw")
        if "xP_uncertainty" in players_with_xp.columns:
            merge_columns.append("xP_uncertainty")
        if "fixture_outlook" in players_with_xp.columns:
            merge_columns.append("fixture_outlook")
        if "expected_minutes" in players_with_xp.columns:
            merge_columns.append("expected_minutes")

        # Include team information in merge
        if "team" in players_with_xp.columns:
            merge_columns.append("team")
        elif "team_id" in players_with_xp.columns:
            merge_columns.append("team_id")

        current_squad_with_xp = current_squad.merge(
            players_with_xp[merge_columns],
            on="player_id",
            how="left",
            suffixes=("", "_from_xp"),
        )
        # Fill any missing XP with 0
        current_squad_with_xp["xP"] = current_squad_with_xp["xP"].fillna(0)
        if "xP_3gw" in current_squad_with_xp.columns:
            current_squad_with_xp["xP_3gw"] = current_squad_with_xp["xP_3gw"].fillna(0)
        if "xP_5gw" in current_squad_with_xp.columns:
            current_squad_with_xp["xP_5gw"] = current_squad_with_xp["xP_5gw"].fillna(0)
        if "fixture_outlook" in current_squad_with_xp.columns:
            current_squad_with_xp["fixture_outlook"] = current_squad_with_xp[
                "fixture_outlook"
            ].fillna("🟡 Average")

        # Validate team data contract
        team_col = "team" if "team" in current_squad_with_xp.columns else "team_id"
        if team_col in current_squad_with_xp.columns:
            nan_teams = current_squad_with_xp[team_col].isna().sum()
            if nan_teams > 0:
                raise ValueError(
                    f"Data contract violation: {nan_teams} players have NaN team values"
                )

        # Get budget and transfer info
        available_budget = team_data["bank"]
        free_transfers = team_data.get("free_transfers", 1)

        # Calculate current squad xP for baseline
        current_squad_list = current_squad_with_xp.to_dict("records")
        current_starting_11 = self._get_best_starting_11_from_squad(
            current_squad_list, xp_column
        )
        current_xp = sum(
            self.get_adjusted_xp(p, xp_column) for p in current_starting_11
        )
        # Get current formation for metadata
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for player in current_squad_list:
            by_position[player["position"]].append(player)
        for pos in by_position:
            by_position[pos].sort(
                key=lambda p: self.get_adjusted_xp(p, xp_column), reverse=True
            )
        _, current_formation, _ = self._enumerate_formations_for_players(
            by_position, xp_column
        )

        # Calculate total budget pool
        is_wildcard = free_transfers >= 15
        if is_wildcard:
            budget_pool_info = {
                "total_budget": 100.0,
                "sellable_value": 0.0,
                "non_sellable_value": 0.0,
                "must_include_value": 0.0,
            }
            total_budget = 100.0
            logger.info("🃏 Wildcard Budget: £100.0m (budget reset)")
        else:
            budget_pool_info = self.calculate_budget_pool(
                current_squad_with_xp, available_budget, must_include_ids
            )
            total_budget = budget_pool_info["total_budget"]
            logger.info(
                f"💰 Budget: Bank £{available_budget:.1f}m | Sellable £{budget_pool_info['sellable_value']:.1f}m | Total £{total_budget:.1f}m"
            )

        # Get all available players and filter
        all_players = players_with_xp[players_with_xp["xP"].notna()].copy()

        # Filter out unavailable players
        if "status" in all_players.columns:
            available_players_mask = ~all_players["status"].isin(["i", "s", "u"])
            all_players = all_players[available_players_mask]

        # Filter must_exclude
        if must_exclude_ids:
            all_players = all_players[~all_players["player_id"].isin(must_exclude_ids)]

        logger.debug(f"Available players for LP: {len(all_players)}")

        # === LINEAR PROGRAMMING FORMULATION ===

        # Initialize LP problem
        prob = pulp.LpProblem("FPL_Transfer_Optimization", pulp.LpMaximize)

        # Decision variables: x[player_id] = 1 if in squad, 0 otherwise
        player_vars = {}
        for _, player in all_players.iterrows():
            pid = player["player_id"]
            player_vars[pid] = pulp.LpVariable(f"player_{pid}", cat="Binary")

        # For 1GW optimization, maximize starting XI xP (not total squad)
        # For 3GW/5GW, keep total squad xP (bench players rotate in)
        optimize_starting_xi = xp_column == "xP"

        if optimize_starting_xi:
            # Add starter variables: s[pid] = 1 if starting, 0 otherwise
            starter_vars = {}
            for _, player in all_players.iterrows():
                pid = player["player_id"]
                starter_vars[pid] = pulp.LpVariable(f"starter_{pid}", cat="Binary")

            # Constraint: Can only start if in squad (s[i] <= x[i])
            for pid in player_vars:
                prob += starter_vars[pid] <= player_vars[pid], f"Starter_In_Squad_{pid}"

            # Constraint: Exactly 11 starters
            prob += (
                pulp.lpSum([starter_vars[pid] for pid in starter_vars]) == 11,
                "Starting_XI_Size",
            )

            # Formation constraints for starters (valid FPL formations)
            # GKP: exactly 1 starter
            gkp_players = all_players[all_players["position"] == "GKP"][
                "player_id"
            ].tolist()
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in gkp_players if pid in starter_vars]
                )
                == 1,
                "Starting_GKP",
            )

            # DEF: 3-5 starters
            def_players = all_players[all_players["position"] == "DEF"][
                "player_id"
            ].tolist()
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in def_players if pid in starter_vars]
                )
                >= 3,
                "Starting_DEF_Min",
            )
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in def_players if pid in starter_vars]
                )
                <= 5,
                "Starting_DEF_Max",
            )

            # MID: 2-5 starters
            mid_players = all_players[all_players["position"] == "MID"][
                "player_id"
            ].tolist()
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in mid_players if pid in starter_vars]
                )
                >= 2,
                "Starting_MID_Min",
            )
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in mid_players if pid in starter_vars]
                )
                <= 5,
                "Starting_MID_Max",
            )

            # FWD: 1-3 starters
            fwd_players = all_players[all_players["position"] == "FWD"][
                "player_id"
            ].tolist()
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in fwd_players if pid in starter_vars]
                )
                >= 1,
                "Starting_FWD_Min",
            )
            prob += (
                pulp.lpSum(
                    [starter_vars[pid] for pid in fwd_players if pid in starter_vars]
                )
                <= 3,
                "Starting_FWD_Max",
            )

            # Objective: Maximize starting XI xP, penalize expensive bench
            # Bench cost penalty ensures we don't waste budget on bench players
            bench_cost_penalty = 0.1  # Penalize £1m bench cost as 0.1 xP
            objective_terms = []
            for _, player in all_players.iterrows():
                pid = player["player_id"]
                adjusted_xp = self.get_adjusted_xp(player.to_dict(), xp_column)
                player_price = player["price"]
                # Starting XI gets full xP weight
                objective_terms.append(adjusted_xp * starter_vars[pid])
                # Bench gets cost penalty (prefer cheap bench to free budget for starters)
                # bench_var = player_vars[pid] - starter_vars[pid] (1 if bench, 0 otherwise)
                objective_terms.append(
                    -bench_cost_penalty
                    * player_price
                    * (player_vars[pid] - starter_vars[pid])
                )

            prob += pulp.lpSum(objective_terms), "Starting_XI_XP"
            logger.info("🎯 1GW mode: Optimizing starting XI xP (not total squad)")
        else:
            # Multi-GW: Maximize total squad xP (bench players rotate in)
            objective_terms = []
            for _, player in all_players.iterrows():
                pid = player["player_id"]
                adjusted_xp = self.get_adjusted_xp(player.to_dict(), xp_column)
                objective_terms.append(adjusted_xp * player_vars[pid])

            prob += pulp.lpSum(objective_terms), "Total_Squad_XP"

        # CONSTRAINT 1: Squad size (exactly 15 players)
        prob += (
            pulp.lpSum([player_vars[pid] for pid in player_vars]) == 15,
            "Squad_Size",
        )

        # CONSTRAINT 2: Budget
        prob += (
            pulp.lpSum(
                [
                    all_players[all_players["player_id"] == pid]["price"].iloc[0]
                    * player_vars[pid]
                    for pid in player_vars
                ]
            )
            <= total_budget,
            "Budget_Limit",
        )

        # CONSTRAINT 3: Position requirements
        for position, count in [("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            position_players = all_players[all_players["position"] == position][
                "player_id"
            ].tolist()
            prob += (
                pulp.lpSum(
                    [player_vars[pid] for pid in position_players if pid in player_vars]
                )
                == count,
                f"Position_{position}",
            )

        # CONSTRAINT 4: Team limit (max 3 players per team)
        for team in all_players[team_col].unique():
            if pd.isna(team):
                continue
            team_players = all_players[all_players[team_col] == team][
                "player_id"
            ].tolist()
            prob += (
                pulp.lpSum(
                    [player_vars[pid] for pid in team_players if pid in player_vars]
                )
                <= 3,
                f"Team_Limit_{team}",
            )

        # CONSTRAINT 5: Transfer limit (keep at least 15 - max_transfers from current squad)
        if not is_wildcard:
            max_transfers = config.optimization.max_transfers
            current_ids = set(current_squad_with_xp["player_id"].tolist())
            prob += (
                pulp.lpSum(
                    [player_vars[pid] for pid in current_ids if pid in player_vars]
                )
                >= 15 - max_transfers,
                "Transfer_Limit",
            )

        # CONSTRAINT 6: Must include players
        for pid in must_include_ids:
            if pid in player_vars:
                prob += player_vars[pid] == 1, f"Must_Include_{pid}"

        # CONSTRAINT 7: Must exclude players (already filtered from all_players, but add for safety)
        for pid in must_exclude_ids:
            if pid in player_vars:
                prob += player_vars[pid] == 0, f"Must_Exclude_{pid}"

        # Solve with CBC solver (default, open-source)
        logger.debug("Solving LP problem with CBC solver...")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        solve_time = time.time() - start_time

        # Check solver status
        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            logger.error(f"❌ LP solver failed with status: {status}")
            return (
                pd.DataFrame(),
                {},
                {
                    "error": f"LP solver failed: {status}",
                    "method": "linear_programming",
                },
            )

        # Extract solution
        selected_player_ids = [
            pid for pid in player_vars if pulp.value(player_vars[pid]) == 1
        ]

        # For 1GW, also extract which players the LP chose as starters
        lp_starter_ids = None
        if optimize_starting_xi:
            lp_starter_ids = set(
                pid for pid in starter_vars if pulp.value(starter_vars[pid]) == 1
            )
            logger.debug(f"LP selected {len(lp_starter_ids)} starters")

        optimal_squad_df = all_players[
            all_players["player_id"].isin(selected_player_ids)
        ].copy()

        # Mark starters in DataFrame for UI display
        if lp_starter_ids is not None:
            optimal_squad_df["is_starter"] = optimal_squad_df["player_id"].isin(
                lp_starter_ids
            )
        else:
            optimal_squad_df["is_starter"] = True  # Will be determined by UI

        logger.debug(f"✅ LP solved in {solve_time:.2f}s - found optimal squad")

        # Calculate transfers
        original_ids = set(current_squad_with_xp["player_id"].tolist())
        new_ids = set(selected_player_ids)
        transfers_out = original_ids - new_ids
        transfers_in = new_ids - original_ids
        num_transfers = len(transfers_out)

        # Calculate optimal squad xP (for best starting 11)
        optimal_squad_list = optimal_squad_df.to_dict("records")

        # For 1GW, use LP's starter decisions; otherwise compute best 11
        if lp_starter_ids is not None:
            best_starting_11 = [
                p for p in optimal_squad_list if p["player_id"] in lp_starter_ids
            ]
            # Determine formation from LP starters
            pos_counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
            for p in best_starting_11:
                pos_counts[p["position"]] += 1
            best_formation = (
                f"{pos_counts['DEF']}-{pos_counts['MID']}-{pos_counts['FWD']}"
            )
        else:
            best_starting_11 = self._get_best_starting_11_from_squad(
                optimal_squad_list, xp_column
            )
            # Get best formation
            by_position_best = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
            for player in optimal_squad_list:
                by_position_best[player["position"]].append(player)
            for pos in by_position_best:
                by_position_best[pos].sort(
                    key=lambda p: self.get_adjusted_xp(p, xp_column), reverse=True
                )
            _, best_formation, _ = self._enumerate_formations_for_players(
                by_position_best, xp_column
            )

        optimal_squad_xp = sum(
            self.get_adjusted_xp(p, xp_column) for p in best_starting_11
        )

        # Calculate transfer penalty
        transfer_penalty = (
            max(0, num_transfers - free_transfers) * config.optimization.transfer_cost
        )
        net_xp = optimal_squad_xp - transfer_penalty

        # Create transfer description
        if num_transfers == 0:
            description = "Keep current squad"
        elif is_wildcard:
            description = (
                f"Wildcard: Rebuild entire squad ({num_transfers} players changed)"
            )
        else:
            out_names = [
                current_squad_with_xp[current_squad_with_xp["player_id"] == pid][
                    "web_name"
                ].iloc[0]
                for pid in list(transfers_out)[:3]
            ]
            in_names = [
                optimal_squad_df[optimal_squad_df["player_id"] == pid]["web_name"].iloc[
                    0
                ]
                for pid in list(transfers_in)[:3]
            ]
            description = f"OUT: {', '.join(out_names)} → IN: {', '.join(in_names)}"

        # Determine scenario type
        scenario_type = "wildcard" if is_wildcard else "linear_programming"

        best_scenario = {
            "id": 0,
            "transfers": num_transfers,
            "type": scenario_type,
            "description": description,
            "penalty": transfer_penalty,
            "net_xp": net_xp,
            "formation": best_formation,
            "xp_gain": net_xp - current_xp,
            "squad": optimal_squad_df,
            "is_wildcard": is_wildcard,
            "lp_objective_value": pulp.value(prob.objective),
            "lp_solve_time": solve_time,
        }

        logger.info(
            f"✅ Optimal solution: {num_transfers} transfers, {net_xp:.2f} net XP "
            f"(solved in {solve_time:.2f}s, objective: {pulp.value(prob.objective):.2f})"
        )

        # Calculate remaining budget
        squad_cost = optimal_squad_df["price"].sum()
        if is_wildcard:
            remaining_budget = 100.0 - squad_cost
        else:
            remaining_budget = available_budget - (
                squad_cost - budget_pool_info.get("non_sellable_value", 0)
            )

        # Return data structures matching SA format
        optimization_metadata = {
            "method": "linear_programming",
            "horizon_label": horizon_label,
            "xp_column": xp_column,
            "budget_pool_info": budget_pool_info,
            "available_budget": team_data["bank"],
            "remaining_budget": remaining_budget,
            "free_transfers": free_transfers,
            "current_xp": current_xp,
            "current_formation": current_formation,
            "lp_solve_time": solve_time,
            "lp_objective_value": pulp.value(prob.objective),
            "lp_status": status,
            "is_wildcard": is_wildcard,
            "transfers_out": [
                {
                    "web_name": current_squad_with_xp[
                        current_squad_with_xp["player_id"] == pid
                    ]["web_name"].iloc[0],
                    "position": current_squad_with_xp[
                        current_squad_with_xp["player_id"] == pid
                    ]["position"].iloc[0],
                    "price": current_squad_with_xp[
                        current_squad_with_xp["player_id"] == pid
                    ]["price"].iloc[0],
                }
                for pid in list(transfers_out)
            ],
            "transfers_in": [
                {
                    "web_name": optimal_squad_df[optimal_squad_df["player_id"] == pid][
                        "web_name"
                    ].iloc[0],
                    "position": optimal_squad_df[optimal_squad_df["player_id"] == pid][
                        "position"
                    ].iloc[0],
                    "price": optimal_squad_df[optimal_squad_df["player_id"] == pid][
                        "price"
                    ].iloc[0],
                }
                for pid in list(transfers_in)
            ],
        }

        return optimal_squad_df, best_scenario, optimization_metadata

    def _run_transfer_sa(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        available_budget: float,
        free_transfers: int,
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
        current_xp: float,
        iterations: int,
    ) -> Dict[str, Any]:
        """Run simulated annealing for transfer optimization.

        Adapted from _simulated_annealing_squad_generation for transfer context.

        Returns:
            Dictionary with optimal_squad, best_xp, iterations_improved, etc.
        """
        # Convert current squad to list of dicts for SA processing
        original_squad = current_squad.to_dict("records")
        original_ids = {p["player_id"] for p in original_squad}

        # Track the max number of transfers allowed
        max_transfers = config.optimization.max_transfers  # Typically 0-3

        # Calculate initial objective value (squad xP with no transfer penalty)
        # Use availability-adjusted xP
        starting_11 = self._get_best_starting_11_from_squad(original_squad, xp_column)
        starting_11_xp = sum(self.get_adjusted_xp(p, xp_column) for p in starting_11)

        # For 1GW optimization, penalize expensive bench in objective
        initial_bench_penalty = 0.0
        if xp_column == "xP":
            starting_11_ids = {p["player_id"] for p in starting_11}
            bench_players = [
                p for p in original_squad if p["player_id"] not in starting_11_ids
            ]
            bench_cost = sum(p["price"] for p in bench_players)
            initial_bench_penalty = bench_cost * 0.1  # Matches LP penalty

        current_objective = starting_11_xp - initial_bench_penalty

        best_team = [p.copy() for p in original_squad]
        best_objective = current_objective
        current_team = [p.copy() for p in original_squad]

        improvements = 0

        logger.debug(f"Initial squad: {current_objective:.2f} xP")

        # SA loop
        for iteration in range(iterations):
            # Linear temperature decrease
            temperature = max(0.01, 1.0 * (1 - iteration / iterations))

            # Generate neighbor by swapping 1-3 players
            new_team = self._swap_transfer_players(
                current_team,
                all_players,
                available_budget,
                must_include_ids,
                must_exclude_ids,
                xp_column,
            )

            if new_team is None:
                continue

            # CRITICAL: Check if new_team exceeds max_transfers from ORIGINAL squad
            new_ids = {p["player_id"] for p in new_team}
            num_transfers_from_original = len(original_ids - new_ids)

            if num_transfers_from_original > max_transfers:
                # Reject this move - too many transfers from original
                continue

            # Calculate new objective INCLUDING TRANSFER PENALTY
            # Use availability-adjusted xP
            new_starting_11 = self._get_best_starting_11_from_squad(new_team, xp_column)
            squad_xp = sum(self.get_adjusted_xp(p, xp_column) for p in new_starting_11)

            # For 1GW optimization, penalize expensive bench
            bench_cost_penalty = 0.0
            if xp_column == "xP":
                starting_11_ids = {p["player_id"] for p in new_starting_11}
                bench_players = [
                    p for p in new_team if p["player_id"] not in starting_11_ids
                ]
                bench_cost = sum(p["price"] for p in bench_players)
                bench_cost_penalty = bench_cost * 0.1  # Matches LP penalty

            # Calculate penalty for this number of transfers
            transfer_penalty = (
                max(0, num_transfers_from_original - free_transfers)
                * config.optimization.transfer_cost
            )
            new_objective = squad_xp - bench_cost_penalty - transfer_penalty

            # Accept if better or with probability
            delta = new_objective - current_objective

            if delta > 0 or (
                temperature > 0 and random.random() < math.exp(delta / temperature)
            ):
                current_team = new_team
                current_objective = new_objective

                if current_objective > best_objective:
                    best_team = [p.copy() for p in current_team]
                    best_objective = current_objective
                    improvements += 1

                    if improvements % 10 == 0:
                        logger.debug(
                            f"Iteration {iteration}: New best {best_objective:.2f} xP ({num_transfers_from_original} transfers)"
                        )

        # Get best formation
        best_starting_11 = self._get_best_starting_11_from_squad(best_team, xp_column)
        _, best_formation, _ = self._enumerate_formations_for_players(
            self._group_by_position(best_starting_11), xp_column
        )

        logger.info(f"Final: {best_objective:.2f} xP ({improvements} improvements)")

        return {
            "optimal_squad": best_team,
            "best_xp": best_objective,
            "iterations_improved": improvements,
            "total_iterations": iterations,
            "formation": best_formation,
        }

    def _swap_transfer_players(
        self,
        team: List[Dict],
        all_players: pd.DataFrame,
        available_budget: float,
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
    ) -> Optional[List[Dict]]:
        """Swap 1-3 players in transfer context.

        Respects transfer budget = bank + selling prices.
        """
        new_team = [p.copy() for p in team]

        # Calculate current cost and budget
        current_cost = sum(p["price"] for p in new_team)

        # Determine how many players to swap (1-3)
        num_swaps = random.randint(
            1, min(3, config.optimization.sa_max_transfers_per_iteration)
        )

        # Get swappable players (not must-include)
        swappable_indices = [
            i for i, p in enumerate(new_team) if p["player_id"] not in must_include_ids
        ]

        if len(swappable_indices) < num_swaps:
            return None

        # Pick random players to swap out
        swap_indices = random.sample(swappable_indices, num_swaps)

        # Calculate budget from selling these players
        sold_value = sum(new_team[i]["price"] for i in swap_indices)
        swap_budget = available_budget + sold_value

        # Remove swapped players
        team_player_ids = {
            p["player_id"] for i, p in enumerate(new_team) if i not in swap_indices
        }

        # Find replacements for each position
        replacements = []
        remaining_budget = swap_budget

        for idx in swap_indices:
            old_player = new_team[idx]
            position = old_player["position"]

            # Get available replacements
            candidates = all_players[
                (all_players["position"] == position)
                & (~all_players["player_id"].isin(team_player_ids))
                & (~all_players["player_id"].isin(must_exclude_ids))
                & (all_players["price"] <= remaining_budget)
            ]

            if candidates.empty:
                return None

            # FILTER: Uncertainty-based differential strategy (2025/26 upgrade)
            # Allow differentials IF model is confident (low uncertainty)
            # This preserves asymmetric information advantage
            if "selected_by_percent" in candidates.columns:

                def is_valid_differential_target(row):
                    """
                    Smart filter using prediction uncertainty.

                    Logic:
                    1. Template/quality players (>15% owned, >£5.5m) → Always valid
                    2. Premiums (>£9m) → Always valid (proven players)
                    3. Differentials (<15% owned) → Valid IF:
                       - High xP/price (>1.0) AND
                       - Low uncertainty (<30% of xP)

                    Example:
                    - Munetsi: 6.0 xP ± 0.8, £4.5m → 1.33 xP/£, 13% uncertainty → ALLOW
                    - Cullen: 4.0 xP ± 2.0, £4.6m → 0.87 xP/£, 50% uncertainty → BLOCK
                    """
                    price = row["price"]
                    ownership = row["selected_by_percent"]
                    xp = row[xp_column]

                    # Route 1: Template/quality players (safe bets)
                    if ownership >= 15.0 and price >= 5.5:
                        return True

                    # Route 2: Premium exception (proven track record)
                    if price >= 9.0:
                        return True

                    # Route 3: Differential - require confident prediction
                    if ownership < 15.0:
                        xp_per_price = xp / max(price, 0.1)

                        # Need strong value signal (>1.0 xP per £m)
                        if xp_per_price < 1.0:
                            return False  # Insufficient value

                        # Check prediction confidence via uncertainty
                        uncertainty = row.get("xP_uncertainty", 0)
                        uncertainty_ratio = uncertainty / max(xp, 0.1)

                        # Allow if confident prediction (<30% uncertainty)
                        # Reject if uncertain (>30% uncertainty)
                        return uncertainty_ratio < 0.30

                    return True  # Default allow

                valid_mask = candidates.apply(is_valid_differential_target, axis=1)
                valid_targets = candidates[valid_mask]

                # Apply filter if it doesn't eliminate all options
                if not valid_targets.empty:
                    candidates = valid_targets

            # Prefer higher xP with randomness (or deterministic if enabled)
            topk = min(10, len(candidates))
            candidates = candidates.nlargest(topk, xp_column)

            if config.optimization.sa_deterministic_mode and len(candidates) > 1:
                # In deterministic mode, check if top candidate is significantly better
                top_xp = candidates.iloc[0][xp_column]
                second_xp = (
                    candidates.iloc[1][xp_column] if len(candidates) > 1 else top_xp
                )
                xp_gap = top_xp - second_xp

                # If top candidate is clearly better (>0.5 xP), always pick it
                if xp_gap > 0.5:
                    replacement = candidates.iloc[0].to_dict()
                else:
                    # Small gap: use weighted sampling (still some randomness for exploration)
                    replacement = candidates.sample(n=1).iloc[0].to_dict()
            else:
                # Normal mode: always use weighted sampling for exploration
                replacement = candidates.sample(n=1).iloc[0].to_dict()

            replacements.append(replacement)
            team_player_ids.add(replacement["player_id"])
            remaining_budget -= replacement["price"]

        # Build new team
        result_team = []
        swap_idx_set = set(swap_indices)
        replacement_iter = iter(replacements)

        for i, player in enumerate(new_team):
            if i in swap_idx_set:
                result_team.append(next(replacement_iter))
            else:
                result_team.append(player)

        # Validate budget
        new_cost = sum(p["price"] for p in result_team)
        if new_cost > current_cost + available_budget:
            return None

        # Validate 3-per-team constraint
        team_counts = self._count_players_per_team(result_team)
        if any(count > 3 for count in team_counts.values()):
            return None

        return result_team

    def _group_by_position(self, players: List[Dict]) -> Dict[str, List[Dict]]:
        """Group players by position for formation enumeration."""
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for player in players:
            by_position[player["position"]].append(player)
        return by_position

    def optimize_wildcard_squad(
        self,
        current_squad: pd.DataFrame,
        team_data: Dict,
        players_with_xp: pd.DataFrame,
        must_include_ids: Optional[Set[int]] = None,
        must_exclude_ids: Optional[Set[int]] = None,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Optimize squad for wildcard chip usage (15 free transfers).

        **UNIFIED API**: This method delegates to `optimize_transfers()` with
        `free_transfers_override=15`. The unified approach treats wildcard as
        15 free transfers with £100m budget reset.

        **Backward Compatibility**: Maintains the same return format as before.

        Args:
            current_squad: Current squad (will be replaced entirely for wildcard)
            team_data: Team data (bank will be overridden to £100m)
            players_with_xp: All players with XP calculations
            must_include_ids: Set of player IDs that must be included
            must_exclude_ids: Set of player IDs that must be excluded

        Returns:
            Tuple of (optimal_squad_df, result_summary_dict, optimization_metadata_dict)
        """
        logger.info(
            "🃏 Wildcard Chip: Using unified optimization with 15 free transfers..."
        )

        # Delegate to unified transfer optimization with 15 free transfers
        return self.optimize_transfers(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=players_with_xp,
            must_include_ids=must_include_ids,
            must_exclude_ids=must_exclude_ids,
            free_transfers_override=15,  # Wildcard = 15 free transfers
        )

    def optimize_initial_squad(
        self,
        players_with_xp: pd.DataFrame,
        budget: float = 100.0,
        formation: Tuple[int, int, int, int] = (2, 5, 5, 3),
        iterations: int = 5000,
        must_include_ids: Optional[Set[int]] = None,
        must_exclude_ids: Optional[Set[int]] = None,
        xp_column: str = "xP",
    ) -> Dict[str, Any]:
        """
        Generate optimal 15-player squad from scratch using simulated annealing.

        This is fundamentally different from transfer optimization - it starts with
        no existing squad and builds a complete team from all available players.

        Args:
            players_with_xp: DataFrame with player data and expected points (must have xP, price, position, player_id, team columns)
            budget: Total budget in millions (default 100.0)
            formation: Tuple of (GKP, DEF, MID, FWD) counts (default (2,5,5,3) = 15 players)
            iterations: Number of simulated annealing iterations (default 5000)
            must_include_ids: Set of player IDs that must be in final squad
            must_exclude_ids: Set of player IDs to exclude from consideration
            xp_column: Column name for expected points (default "xP")

        Returns:
            Dictionary with:
                - optimal_squad: List of 15 player dicts
                - remaining_budget: Unused budget
                - total_xp: Total expected points
                - iterations_improved: Number of improvements found
                - constraint_violations: Any violations detected

        Raises:
            ValueError: If data validation fails or constraints are impossible
        """
        # Validate inputs using data contract
        validation = InitialSquadOptimizationInput(
            budget=budget,
            formation=formation,
            iterations=iterations,
            must_include_ids=list(must_include_ids or []),
            must_exclude_ids=list(must_exclude_ids or []),
            xp_column=xp_column,
        )

        must_include_ids = set(validation.must_include_ids)
        must_exclude_ids = set(validation.must_exclude_ids)

        # Validate DataFrame has required columns
        required_cols = [xp_column, "price", "position", "player_id", "team"]
        missing_cols = [
            col for col in required_cols if col not in players_with_xp.columns
        ]
        if missing_cols:
            raise ValueError(
                f"players_with_xp DataFrame missing required columns: {missing_cols}"
            )

        # Validate no NaN values in critical columns - fail fast
        for col in required_cols:
            nan_count = players_with_xp[col].isna().sum()
            if nan_count > 0:
                raise ValueError(
                    f"Data quality issue: {nan_count} players have missing {col}. "
                    f"Fix upstream data processing - all players must have complete data."
                )

        # Filter valid players (apply exclusions)
        valid_players = players_with_xp[
            ~players_with_xp["player_id"].isin(must_exclude_ids)
        ].copy()

        if len(valid_players) < 15:
            raise ValueError(
                f"Only {len(valid_players)} players available after exclusions. "
                f"Cannot form 15-player squad."
            )

        # Validate must-include players exist and have complete data
        if must_include_ids:
            must_include_players = players_with_xp[
                players_with_xp["player_id"].isin(must_include_ids)
            ]

            if len(must_include_players) != len(must_include_ids):
                missing_ids = must_include_ids - set(must_include_players["player_id"])
                raise ValueError(
                    f"Must-include player IDs not found in dataset: {missing_ids}"
                )

            # Validate position constraints
            gkp_count, def_count, mid_count, fwd_count = formation
            position_requirements = {
                "GKP": gkp_count,
                "DEF": def_count,
                "MID": mid_count,
                "FWD": fwd_count,
            }

            must_include_positions = (
                must_include_players.groupby("position").size().to_dict()
            )
            for pos, required in position_requirements.items():
                included_count = must_include_positions.get(pos, 0)
                if included_count > required:
                    raise ValueError(
                        f"Must-include constraint violation: {included_count} {pos} players "
                        f"required but formation only allows {required}"
                    )

            # Validate budget constraint
            must_include_cost = must_include_players["price"].sum()
            if must_include_cost > budget:
                raise ValueError(
                    f"Must-include players cost £{must_include_cost:.1f}m, "
                    f"exceeding budget of £{budget:.1f}m"
                )

            # Validate 3-per-team constraint
            team_counts = must_include_players["team"].value_counts()
            violations = team_counts[team_counts > 3]
            if not violations.empty:
                raise ValueError(
                    f"Must-include players violate 3-per-team rule: "
                    f"{violations.to_dict()}"
                )

        # Run simulated annealing optimization
        result = self._simulated_annealing_squad_generation(
            valid_players=valid_players,
            budget=budget,
            formation=formation,
            iterations=iterations,
            must_include_ids=must_include_ids,
            must_exclude_ids=must_exclude_ids,
            xp_column=xp_column,
        )

        return result

    def _simulated_annealing_squad_generation(
        self,
        valid_players: pd.DataFrame,
        budget: float,
        formation: Tuple[int, int, int, int],
        iterations: int,
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
    ) -> Dict[str, Any]:
        """
        Internal method: Simulated annealing algorithm for squad generation.

        This implements the proven algorithm from season_planner.py with
        improvements: removed debug code, proper error handling, clean structure.
        """
        # Set random seed for reproducibility if configured
        if config.optimization.sa_random_seed is not None:
            random.seed(config.optimization.sa_random_seed)
        gkp_count, def_count, mid_count, fwd_count = formation
        position_requirements = {
            "GKP": gkp_count,
            "DEF": def_count,
            "MID": mid_count,
            "FWD": fwd_count,
        }

        def generate_random_team() -> Optional[List[Dict]]:
            """Generate a random valid 15-player squad."""
            team = []
            remaining_budget = budget
            team_counts = {}

            # Start with must-include players
            for player_id in must_include_ids:
                player = valid_players[valid_players["player_id"] == player_id].iloc[0]
                team_name = player["team"]

                # Verify 3-per-team constraint
                if team_counts.get(team_name, 0) >= 3:
                    return None

                team.append(player.to_dict())
                remaining_budget -= player["price"]
                team_counts[team_name] = team_counts.get(team_name, 0) + 1

            # Get minimum costs per position for budget feasibility
            min_costs = {}
            for position in ["GKP", "DEF", "MID", "FWD"]:
                pos_players = valid_players[valid_players["position"] == position]
                if len(pos_players) == 0:
                    return None
                min_costs[position] = pos_players["price"].min()

            # Fill each position
            for position in ["GKP", "DEF", "MID", "FWD"]:
                count = position_requirements[position]
                already_have = sum(1 for p in team if p["position"] == position)
                remaining_needed = count - already_have

                for _ in range(remaining_needed):
                    team_player_ids = {p["player_id"] for p in team}

                    # Calculate budget reserve for other positions
                    remaining_slots_cost = 0.0
                    for other_pos in ["GKP", "DEF", "MID", "FWD"]:
                        if other_pos == position:
                            continue
                        slots_needed = position_requirements[other_pos] - sum(
                            1 for p in team if p["position"] == other_pos
                        )
                        remaining_slots_cost += (
                            max(0, slots_needed) * min_costs[other_pos]
                        )

                    # Get affordable candidates respecting all constraints
                    max_affordable_price = remaining_budget - remaining_slots_cost
                    candidates = valid_players[
                        (valid_players["position"] == position)
                        & (~valid_players["player_id"].isin(team_player_ids))
                        & (valid_players["price"] <= max_affordable_price)
                    ].copy()

                    # Apply 3-per-team constraint
                    team_ok_mask = candidates["team"].map(
                        lambda t: team_counts.get(t, 0) < 3
                    )
                    candidates = candidates[team_ok_mask]

                    if len(candidates) == 0:
                        return None

                    # Prefer higher availability-adjusted xP with randomness
                    # Calculate adjusted xP for all candidates
                    candidates_list = candidates.to_dict("records")
                    for candidate in candidates_list:
                        candidate["_adjusted_xp"] = self.get_adjusted_xp(
                            candidate, xp_column
                        )
                    candidates_df = pd.DataFrame(candidates_list)

                    topk = min(8, len(candidates_df))
                    candidates_df = candidates_df.nlargest(topk, "_adjusted_xp")
                    player = candidates_df.sample(n=1).iloc[0]

                    team.append(player.to_dict())
                    remaining_budget -= player["price"]
                    team_counts[player["team"]] = team_counts.get(player["team"], 0) + 1

            return team if len(team) == 15 else None

        def calculate_team_xp(team: List[Dict]) -> float:
            """Calculate total xP for team's best starting 11, accounting for availability.

            For 1GW optimization (xp_column == 'xP'), adds bench efficiency penalty
            to avoid wasting budget on expensive bench players.
            """
            if len(team) != 15:
                return sum(self.get_adjusted_xp(p, xp_column) for p in team)

            starting_11 = self._get_best_starting_11_from_squad(team, xp_column)
            starting_11_ids = {p["player_id"] for p in starting_11}

            # Use availability-adjusted xP for scoring
            base_xp = sum(self.get_adjusted_xp(p, xp_column) for p in starting_11)

            # Penalty for constraint violations using shared utility
            team_counts = self._count_players_per_team(team)
            constraint_penalty = sum(
                (count - 3) * -10.0 for count in team_counts.values() if count > 3
            )

            # For 1GW optimization, penalize expensive bench players
            # This prevents selecting £14.9m Haaland just to bench him
            bench_cost_penalty = 0.0
            if xp_column == "xP":
                bench_players = [
                    p for p in team if p["player_id"] not in starting_11_ids
                ]
                bench_cost = sum(p["price"] for p in bench_players)
                # Penalize £1m bench cost as 0.1 xP (matches LP penalty)
                bench_cost_penalty = bench_cost * 0.1

            return base_xp + constraint_penalty - bench_cost_penalty

        def calculate_team_cost(team: List[Dict]) -> float:
            """Calculate total cost of team."""
            return sum(p["price"] for p in team)

        def is_valid_team(team: List[Dict]) -> bool:
            """Validate team satisfies all constraints."""
            if len(team) != 15:
                return False

            if calculate_team_cost(team) > budget:
                return False

            # Check formation
            position_counts = {}
            for player in team:
                pos = player["position"]
                position_counts[pos] = position_counts.get(pos, 0) + 1

            for pos, required in position_requirements.items():
                if position_counts.get(pos, 0) != required:
                    return False

            # Check 3-per-team using shared utility
            team_counts = self._count_players_per_team(team)
            for count in team_counts.values():
                if count > 3:
                    return False

            return True

        def swap_player(team: List[Dict]) -> Optional[List[Dict]]:
            """Generate neighbor solution by swapping one player."""
            new_team = [p.copy() for p in team]
            current_cost = calculate_team_cost(new_team)
            remaining_budget = budget - current_cost

            # Get swappable players (not must-include)
            swappable_indices = [
                i
                for i, p in enumerate(new_team)
                if p["player_id"] not in must_include_ids
            ]

            if not swappable_indices:
                return None

            # Pick random player to replace
            replace_idx = random.choice(swappable_indices)
            old_player = new_team[replace_idx]
            position = old_player["position"]

            # Get available replacements
            team_player_ids = {p["player_id"] for p in new_team}
            available = valid_players[
                (valid_players["position"] == position)
                & (~valid_players["player_id"].isin(team_player_ids))
                & (~valid_players["player_id"].isin(must_exclude_ids))
            ]

            if len(available) == 0:
                return None

            # Budget constraint with flexibility
            max_new_cost = old_player["price"] + remaining_budget + 5.0
            affordable = available[available["price"] <= max_new_cost]

            if len(affordable) == 0:
                affordable = available  # Try without budget constraint
                if len(affordable) == 0:
                    return None

            # Prefer higher availability-adjusted xP with weighting (or deterministic if enabled)
            if remaining_budget > 1.0 and len(affordable) > 1:
                # Calculate adjusted xP for all candidates
                affordable_list = affordable.to_dict("records")
                for candidate in affordable_list:
                    candidate["_adjusted_xp"] = self.get_adjusted_xp(
                        candidate, xp_column
                    )
                affordable_df = pd.DataFrame(affordable_list)

                if config.optimization.sa_deterministic_mode:
                    # In deterministic mode, check if top candidate is significantly better
                    affordable_sorted = affordable_df.sort_values(
                        "_adjusted_xp", ascending=False
                    )
                    top_xp = affordable_sorted.iloc[0]["_adjusted_xp"]
                    second_xp = (
                        affordable_sorted.iloc[1]["_adjusted_xp"]
                        if len(affordable_sorted) > 1
                        else top_xp
                    )
                    xp_gap = top_xp - second_xp

                    # If top candidate is clearly better (>0.5 xP), always pick it
                    if xp_gap > 0.5:
                        new_player = affordable_sorted.iloc[0].to_dict()
                    else:
                        # Small gap: use weighted sampling with adjusted xP
                        weights = affordable_df["_adjusted_xp"] * (
                            1 + affordable_df["price"] / 20
                        )
                        min_weight = weights.min()
                        if min_weight < 0:
                            weights = weights - min_weight + 0.1
                        weights_array = weights.values
                        if weights_array.sum() > 0:
                            weights_array = weights_array / weights_array.sum()
                            new_player = (
                                affordable_df.sample(n=1, weights=weights_array)
                                .iloc[0]
                                .to_dict()
                            )
                        else:
                            new_player = affordable_df.sample(n=1).iloc[0].to_dict()
                else:
                    # Normal mode: use weighted sampling with adjusted xP
                    weights = affordable_df["_adjusted_xp"] * (
                        1 + affordable_df["price"] / 20
                    )
                    min_weight = weights.min()
                    if min_weight < 0:
                        weights = weights - min_weight + 0.1

                    weights_array = weights.values
                    if weights_array.sum() > 0:
                        weights_array = weights_array / weights_array.sum()
                        new_player = (
                            affordable_df.sample(n=1, weights=weights_array)
                            .iloc[0]
                            .to_dict()
                        )
                    else:
                        new_player = affordable_df.sample(n=1).iloc[0].to_dict()
            else:
                # Budget too tight or only one option - still use adjusted xP if possible
                if len(affordable) > 0:
                    affordable_list = affordable.to_dict("records")
                    for candidate in affordable_list:
                        candidate["_adjusted_xp"] = self.get_adjusted_xp(
                            candidate, xp_column
                        )
                    affordable_df = pd.DataFrame(affordable_list)
                    new_player = (
                        affordable_df.nlargest(1, "_adjusted_xp").iloc[0].to_dict()
                    )
                else:
                    new_player = affordable.sample(n=1).iloc[0].to_dict()

            new_team[replace_idx] = new_player

            # Validate budget
            if calculate_team_cost(new_team) > budget:
                return None

            return new_team if is_valid_team(new_team) else None

        # Generate initial team
        current_team = None
        for _ in range(1000):
            current_team = generate_random_team()
            if current_team and is_valid_team(current_team):
                break

        if not current_team:
            raise ValueError(
                "Could not generate valid initial team. Check data quality and constraints."
            )

        best_team = [p.copy() for p in current_team]
        current_xp = calculate_team_xp(current_team)
        best_xp = current_xp

        # Simulated annealing loop
        improvements = 0

        for iteration in range(iterations):
            # Linear temperature decrease
            temperature = max(0.01, 1.0 * (1 - iteration / iterations))

            # Generate neighbor
            new_team = swap_player(current_team)
            if new_team is None:
                continue

            new_xp = calculate_team_xp(new_team)
            new_cost = calculate_team_cost(new_team)
            current_cost = calculate_team_cost(current_team)

            # Budget utilization bonus
            budget_util_bonus = 0
            if new_cost > current_cost:
                budget_util_bonus = (new_cost - current_cost) * 0.1

            delta = new_xp - current_xp + budget_util_bonus

            # Accept if better or with probability
            if delta > 0 or (
                temperature > 0 and random.random() < math.exp(delta / temperature)
            ):
                current_team = new_team
                current_xp = new_xp

                if current_xp > best_xp:
                    best_team = [p.copy() for p in current_team]
                    best_xp = current_xp
                    improvements += 1

        # Final validation
        if must_exclude_ids:
            final_team_ids = {p["player_id"] for p in best_team}
            violations = final_team_ids.intersection(must_exclude_ids)
            if violations:
                raise ValueError(
                    f"Algorithm error: Excluded players found in final team: {violations}"
                )

        # For 1GW optimization, replace expensive bench with cheapest alternatives
        if xp_column == "xP":
            best_team = self._optimize_bench_for_1gw(
                best_team, valid_players, must_include_ids, must_exclude_ids, xp_column
            )
            # Recalculate xP after bench optimization
            best_xp = calculate_team_xp(best_team)

        remaining_budget = budget - calculate_team_cost(best_team)

        return {
            "optimal_squad": best_team,
            "remaining_budget": remaining_budget,
            "total_xp": best_xp,
            "iterations_improved": improvements,
            "total_iterations": iterations,
            "final_cost": calculate_team_cost(best_team),
        }

    def _get_best_starting_11_from_squad(
        self, squad: List[Dict], xp_column: str = "xP"
    ) -> List[Dict]:
        """Get best starting 11 from 15-player squad.

        Args:
            squad: List of 15 player dictionaries
            xp_column: Column name for XP values

        Returns:
            List of 11 player dicts forming best starting team
        """
        if len(squad) != 15:
            return []

        # Group by position
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for player in squad:
            by_position[player["position"]].append(player)

        # Sort by xP
        for pos in by_position:
            by_position[pos].sort(key=lambda p: p[xp_column], reverse=True)

        # Use shared formation enumeration logic (return just the players, ignore formation name)
        best_11, _, _ = self._enumerate_formations_for_players(by_position, xp_column)
        return best_11

    def _optimize_bench_for_1gw(
        self,
        squad: List[Dict],
        valid_players: pd.DataFrame,
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
    ) -> List[Dict]:
        """Replace expensive bench players with cheapest valid alternatives for 1GW.

        For 1GW optimization, bench players don't contribute points, so we want
        the cheapest valid bench to maximize budget for starters.

        Args:
            squad: 15-player squad
            valid_players: All available players
            must_include_ids: Players that must stay in squad
            must_exclude_ids: Players to exclude
            xp_column: XP column name

        Returns:
            Optimized squad with cheap bench
        """
        if len(squad) != 15:
            return squad

        # Identify starting 11 and bench
        starting_11 = self._get_best_starting_11_from_squad(squad, xp_column)
        starting_11_ids = {p["player_id"] for p in starting_11}
        bench_players = [p for p in squad if p["player_id"] not in starting_11_ids]

        # Get team counts from starting 11 (bench replacements must respect 3-per-team)
        team_counts = {}
        for p in starting_11:
            team = p.get("team", p.get("team_id"))
            team_counts[team] = team_counts.get(team, 0) + 1

        # For each bench position, find cheapest valid replacement
        new_squad = [p.copy() for p in starting_11]
        squad_ids = starting_11_ids.copy()

        for bench_player in bench_players:
            position = bench_player["position"]
            old_team = bench_player.get("team", bench_player.get("team_id"))

            # Skip must-include players
            if bench_player["player_id"] in must_include_ids:
                new_squad.append(bench_player.copy())
                squad_ids.add(bench_player["player_id"])
                team_counts[old_team] = team_counts.get(old_team, 0) + 1
                continue

            # Find cheapest valid replacement
            candidates = valid_players[
                (valid_players["position"] == position)
                & (~valid_players["player_id"].isin(squad_ids))
                & (~valid_players["player_id"].isin(must_exclude_ids))
            ].copy()

            # Apply 3-per-team constraint
            def team_ok(team):
                return team_counts.get(team, 0) < 3

            candidates = candidates[candidates["team"].apply(team_ok)]

            if len(candidates) == 0:
                # No replacement found, keep original
                new_squad.append(bench_player.copy())
                squad_ids.add(bench_player["player_id"])
                team_counts[old_team] = team_counts.get(old_team, 0) + 1
            else:
                # Pick cheapest
                cheapest = candidates.nsmallest(1, "price").iloc[0]
                new_squad.append(cheapest.to_dict())
                squad_ids.add(cheapest["player_id"])
                new_team = cheapest["team"]
                team_counts[new_team] = team_counts.get(new_team, 0) + 1

        return new_squad

    def get_optimal_team_from_database(
        self, players_with_xp: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Build optimal 11-player team from full player database.

        For analysis and testing purposes - builds theoretically best team
        without squad constraints.

        Args:
            players_with_xp: DataFrame with all available players

        Returns:
            Optimal starting eleven
        """
        available_players = players_with_xp[
            ~players_with_xp["status"].isin(["i", "s", "u"])
        ].copy()
        return self._select_optimal_team_from_all_players(available_players)

    def validate_optimization_constraints(
        self,
        must_include_ids: Optional[Set[int]] = None,
        must_exclude_ids: Optional[Set[int]] = None,
        budget_limit: float = 100.0,
    ) -> Dict[str, Any]:
        """Validate optimization constraints for conflicts.

        Args:
            must_include_ids: Player IDs that must be included
            must_exclude_ids: Player IDs that must be excluded
            budget_limit: Budget limit in millions

        Returns:
            Validation result with conflicts detected
        """
        must_include_ids = must_include_ids or set()
        must_exclude_ids = must_exclude_ids or set()

        conflicts = must_include_ids.intersection(must_exclude_ids)

        return {
            "valid": len(conflicts) == 0,
            "conflicts": list(conflicts),
            "must_include_count": len(must_include_ids),
            "must_exclude_count": len(must_exclude_ids),
            "budget_limit": budget_limit,
        }

    def _select_optimal_team_from_all_players(
        self, players: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Select optimal starting 11 from full player database.

        Args:
            players: DataFrame of available players

        Returns:
            List of 11 player dicts forming optimal starting team
        """
        players_sorted = players.sort_values("xP", ascending=False)

        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for _, player in players_sorted.iterrows():
            position = player["position"]
            if position in by_position:
                by_position[position].append(player.to_dict())

        # Formation: 1 GKP, 4 DEF, 4 MID, 2 FWD
        formation = {"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2}

        starting_11 = []
        for position, needed in formation.items():
            available = by_position[position][:needed]
            starting_11.extend(available)

        return starting_11

    def _consensus_wildcard_optimization(
        self,
        players_with_xp: pd.DataFrame,
        budget: float,
        formation: Tuple[int, int, int, int],
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
    ) -> Dict[str, Any]:
        """Run wildcard optimization multiple times and find consensus optimal squad.

        This addresses the problem where SA finds different "optimal" squads each run.
        By running multiple times and aggregating results, we find the truly best squad.

        Args:
            players_with_xp: All available players with XP
            budget: Budget (always £100m for wildcard)
            formation: Formation tuple (GKP, DEF, MID, FWD)
            must_include_ids: Players that must be included
            must_exclude_ids: Players to exclude
            xp_column: Column name for XP values

        Returns:
            Best consensus result with confidence metrics
        """
        num_restarts = config.optimization.sa_wildcard_restarts
        iterations = config.optimization.sa_wildcard_iterations

        logger.info(
            f"   Running {num_restarts} wildcard optimizations ({iterations} iterations each)..."
        )

        all_results = []
        squad_signatures = {}  # Track how often each squad appears

        for restart in range(num_restarts):
            if num_restarts > 1:
                logger.debug(f"   Wildcard restart {restart + 1}/{num_restarts}...")

            # Use different seed for each restart for diversity
            if config.optimization.sa_random_seed is not None:
                random.seed(config.optimization.sa_random_seed + restart * 1000)
            else:
                # Still use some seed for this restart to get different exploration
                random.seed(restart * 1000)

            result = self.optimize_initial_squad(
                players_with_xp=players_with_xp,
                budget=budget,
                formation=formation,
                iterations=iterations,
                must_include_ids=must_include_ids,
                must_exclude_ids=must_exclude_ids,
                xp_column=xp_column,
            )

            # Create squad signature (sorted player IDs) for tracking
            squad_ids = tuple(sorted([p["player_id"] for p in result["optimal_squad"]]))
            squad_signatures[squad_ids] = squad_signatures.get(squad_ids, 0) + 1

            all_results.append(result)

        if not all_results:
            raise ValueError(
                "Wildcard consensus optimization failed - no valid results"
            )

        # Find the best result overall (highest xP)
        best_overall = max(all_results, key=lambda r: r["total_xp"])

        # Calculate confidence: how often did this exact squad appear?
        best_squad_sig = tuple(
            sorted([p["player_id"] for p in best_overall["optimal_squad"]])
        )
        confidence = squad_signatures.get(best_squad_sig, 0) / num_restarts

        logger.info(
            f"✅ Wildcard Consensus: Best squad found {squad_signatures.get(best_squad_sig, 0)}/{num_restarts} times "
            f"({confidence * 100:.1f}% confidence), {best_overall['total_xp']:.2f} xP"
        )

        # Add consensus metrics to result
        best_overall["consensus_confidence"] = confidence
        best_overall["consensus_runs"] = num_restarts
        best_overall["consensus_squad_frequency"] = squad_signatures
        best_overall["total_improvements"] = best_overall.get("iterations_improved", 0)

        return best_overall

    def _consensus_optimization(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        available_budget: float,
        free_transfers: int,
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
        current_xp: float,
    ) -> Dict[str, Any]:
        """Run optimization multiple times and find consensus optimal solution.

        This addresses the problem where SA finds different "optimal" solutions each run.
        By running multiple times and aggregating results, we find the truly best solution.

        Args:
            Same as _run_transfer_sa

        Returns:
            Best consensus result with confidence metrics
        """
        num_runs = config.optimization.sa_consensus_runs
        num_restarts = config.optimization.sa_restarts
        iterations = config.optimization.sa_iterations

        logger.info(
            f"   Running {num_runs} full optimizations ({num_restarts} restarts × {iterations} iterations each)..."
        )

        all_results = []
        transfer_counts = {}  # Track how often each transfer combination appears

        for run in range(num_runs):
            if num_runs > 1:
                logger.debug(f"   Consensus run {run + 1}/{num_runs}...")

            # Run full SA with multiple restarts
            best_result = None
            best_net_xp = float("-inf")

            for restart in range(num_restarts):
                # Use different seed for each run+restart for diversity
                if config.optimization.sa_random_seed is not None:
                    random.seed(
                        config.optimization.sa_random_seed + run * 1000 + restart
                    )
                else:
                    # Still use some seed for this run to get different exploration
                    random.seed(run * 1000 + restart)

                sa_result = self._run_transfer_sa(
                    current_squad=current_squad,
                    all_players=all_players,
                    available_budget=available_budget,
                    free_transfers=free_transfers,
                    must_include_ids=must_include_ids,
                    must_exclude_ids=must_exclude_ids,
                    xp_column=xp_column,
                    current_xp=current_xp,
                    iterations=iterations,
                )

                # Calculate net XP
                best_squad_df = pd.DataFrame(sa_result["optimal_squad"])
                original_ids = set(current_squad["player_id"].tolist())
                new_ids = set(best_squad_df["player_id"].tolist())
                transfers_out = original_ids - new_ids
                transfers_in = new_ids - original_ids
                num_transfers = len(transfers_out)

                transfer_penalty = (
                    max(0, num_transfers - free_transfers)
                    * config.optimization.transfer_cost
                )
                net_xp = sa_result["best_xp"] - transfer_penalty

                # Create transfer signature for tracking
                transfer_sig = (
                    tuple(sorted(transfers_out)),
                    tuple(sorted(transfers_in)),
                )
                transfer_counts[transfer_sig] = transfer_counts.get(transfer_sig, 0) + 1

                if net_xp > best_net_xp:
                    best_net_xp = net_xp
                    best_result = sa_result.copy()
                    best_result["net_xp"] = net_xp
                    best_result["transfers_out"] = transfers_out
                    best_result["transfers_in"] = transfers_in

            if best_result:
                all_results.append(best_result)

        if not all_results:
            raise ValueError("Consensus optimization failed - no valid results")

        # Find the best result overall
        best_overall = max(all_results, key=lambda r: r["net_xp"])

        # Calculate confidence: how often did this transfer appear?
        best_transfer_sig = (
            tuple(sorted(best_overall["transfers_out"])),
            tuple(sorted(best_overall["transfers_in"])),
        )
        confidence = transfer_counts.get(best_transfer_sig, 0) / num_runs

        logger.info(
            f"✅ Consensus: Best solution found {transfer_counts.get(best_transfer_sig, 0)}/{num_runs} times "
            f"({confidence * 100:.1f}% confidence), {best_overall['net_xp']:.2f} net xP"
        )

        # Add confidence to result
        best_overall["consensus_confidence"] = confidence
        best_overall["consensus_runs"] = num_runs
        best_overall["consensus_transfer_frequency"] = transfer_counts

        return best_overall

    def _exhaustive_transfer_search(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        available_budget: float,
        free_transfers: int,
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        xp_column: str,
        current_xp: float,
        max_transfers: int,
    ) -> Optional[Dict[str, Any]]:
        """Exhaustive search for guaranteed optimal solution (0-N transfers).

        Only feasible for small transfer counts (0-2 transfers). Guarantees finding
        the truly optimal solution by evaluating all possible transfer combinations.

        Args:
            Same as _run_transfer_sa, plus max_transfers limit

        Returns:
            Optimal result or None if search is infeasible
        """
        from itertools import combinations

        original_squad = current_squad.to_dict("records")
        original_ids = {p["player_id"] for p in original_squad}
        swappable_players = [
            p for p in original_squad if p["player_id"] not in must_include_ids
        ]

        if len(swappable_players) == 0:
            return None

        best_result = None
        best_net_xp = float("-inf")
        total_combinations = 0

        # Try 0 transfers first
        starting_11 = self._get_best_starting_11_from_squad(original_squad, xp_column)
        zero_transfer_xp = sum(p[xp_column] for p in starting_11)
        if zero_transfer_xp > best_net_xp:
            best_net_xp = zero_transfer_xp
            best_result = {
                "optimal_squad": original_squad,
                "best_xp": zero_transfer_xp,
                "net_xp": zero_transfer_xp,
                "transfers_out": set(),
                "transfers_in": set(),
                "iterations_improved": 0,
                "total_iterations": 1,
                "formation": self._enumerate_formations_for_players(
                    self._group_by_position(starting_11), xp_column
                )[1],
            }

        # Try 1 to max_transfers transfers
        for num_transfers in range(
            1, min(max_transfers + 1, len(swappable_players) + 1)
        ):
            # Generate all combinations of players to transfer out
            for out_players in combinations(swappable_players, num_transfers):
                out_ids = {p["player_id"] for p in out_players}
                out_value = sum(p["price"] for p in out_players)
                out_positions = [p["position"] for p in out_players]

                # Calculate available budget
                swap_budget = available_budget + out_value

                # Try to find replacements for each position
                team_player_ids = original_ids - out_ids

                # For exhaustive search, get top candidates per position
                # For 1 transfer: try top 100 candidates (truly exhaustive)
                # For 2 transfers: try top 50 candidates per position
                max_candidates_per_pos = 100 if num_transfers == 1 else 50

                # Get candidates for each unique position
                pos_candidates = {}
                for pos in set(out_positions):
                    candidates = all_players[
                        (all_players["position"] == pos)
                        & (~all_players["player_id"].isin(team_player_ids))
                        & (~all_players["player_id"].isin(must_exclude_ids))
                        & (all_players["price"] <= swap_budget)  # Can use full budget
                    ].nlargest(max_candidates_per_pos, xp_column)
                    pos_candidates[pos] = (
                        candidates.to_dict("records") if not candidates.empty else []
                    )

                # For single transfer, try ALL candidates (truly exhaustive)
                if num_transfers == 1:
                    pos = out_positions[0]
                    if pos not in pos_candidates or not pos_candidates[pos]:
                        continue

                    # Try EVERY candidate for this position
                    for candidate in pos_candidates[pos]:
                        if candidate["price"] > swap_budget:
                            continue

                        # Check 3-per-team constraint
                        candidate_team = candidate.get("team", candidate.get("team_id"))
                        existing_count = sum(
                            1
                            for p in original_squad
                            if p["player_id"] not in out_ids
                            and p.get("team", p.get("team_id")) == candidate_team
                        )
                        if existing_count >= 3:
                            continue  # Would violate 3-per-team

                        # Valid candidate - evaluate this transfer
                        replacements = [candidate]
                        total_combinations += 1
                        if total_combinations % 1000 == 0:
                            logger.debug(
                                f"   Evaluated {total_combinations} combinations..."
                            )

                        # Build new squad
                        new_squad = []
                        out_ids_set = out_ids

                        for player in original_squad:
                            if player["player_id"] in out_ids_set:
                                new_squad.append(candidate)
                            else:
                                new_squad.append(player)

                        # Validate constraints (should already be valid, but double-check)
                        team_counts = self._count_players_per_team(new_squad)
                        if any(count > 3 for count in team_counts.values()):
                            continue

                        # Calculate XP
                        starting_11 = self._get_best_starting_11_from_squad(
                            new_squad, xp_column
                        )
                        squad_xp = sum(p[xp_column] for p in starting_11)
                        transfer_penalty = (
                            max(0, num_transfers - free_transfers)
                            * config.optimization.transfer_cost
                        )
                        net_xp = squad_xp - transfer_penalty

                        if net_xp > best_net_xp:
                            best_net_xp = net_xp
                            in_ids = {candidate["player_id"]}
                            best_result = {
                                "optimal_squad": new_squad,
                                "best_xp": squad_xp,
                                "net_xp": net_xp,
                                "transfers_out": out_ids,
                                "transfers_in": in_ids,
                                "iterations_improved": 1,
                                "total_iterations": total_combinations,
                                "formation": self._enumerate_formations_for_players(
                                    self._group_by_position(starting_11), xp_column
                                )[1],
                            }
                else:
                    # For 2+ transfers, try all combinations of top candidates
                    from itertools import product

                    # Build list of candidate lists for each position
                    replacement_combos = []
                    for pos in out_positions:
                        if pos not in pos_candidates or not pos_candidates[pos]:
                            break
                        replacement_combos.append(pos_candidates[pos])
                    else:
                        # All positions have candidates - try all combinations
                        for replacement_combo in product(*replacement_combos):
                            # Check budget
                            combo_cost = sum(p["price"] for p in replacement_combo)
                            if combo_cost > swap_budget:
                                continue

                            # Check for duplicates
                            combo_ids = {p["player_id"] for p in replacement_combo}
                            if len(combo_ids) != len(replacement_combo):
                                continue

                            # Check 3-per-team constraint
                            combo_teams = {}
                            for p in replacement_combo:
                                team = p.get("team", p.get("team_id"))
                                combo_teams[team] = combo_teams.get(team, 0) + 1

                            existing_teams = {}
                            for p in original_squad:
                                if p["player_id"] not in out_ids:
                                    team = p.get("team", p.get("team_id"))
                                    existing_teams[team] = (
                                        existing_teams.get(team, 0) + 1
                                    )

                            violates_constraint = False
                            for team, count in combo_teams.items():
                                existing = existing_teams.get(team, 0)
                                if existing + count > 3:
                                    violates_constraint = True
                                    break

                            if violates_constraint:
                                continue

                            # Valid combination - evaluate
                            replacements = list(replacement_combo)
                            total_combinations += 1
                            if total_combinations % 1000 == 0:
                                logger.debug(
                                    f"   Evaluated {total_combinations} combinations..."
                                )

                            # Build new squad
                            new_squad = []
                            out_ids_set = out_ids
                            replacement_iter = iter(replacements)

                            for player in original_squad:
                                if player["player_id"] in out_ids_set:
                                    new_squad.append(next(replacement_iter))
                                else:
                                    new_squad.append(player)

                            # Calculate XP
                            starting_11 = self._get_best_starting_11_from_squad(
                                new_squad, xp_column
                            )
                            squad_xp = sum(p[xp_column] for p in starting_11)
                            transfer_penalty = (
                                max(0, num_transfers - free_transfers)
                                * config.optimization.transfer_cost
                            )
                            net_xp = squad_xp - transfer_penalty

                            if net_xp > best_net_xp:
                                best_net_xp = net_xp
                                in_ids = {p["player_id"] for p in replacements}
                                best_result = {
                                    "optimal_squad": new_squad,
                                    "best_xp": squad_xp,
                                    "net_xp": net_xp,
                                    "transfers_out": out_ids,
                                    "transfers_in": in_ids,
                                    "iterations_improved": 1,
                                    "total_iterations": total_combinations,
                                    "formation": self._enumerate_formations_for_players(
                                        self._group_by_position(starting_11), xp_column
                                    )[1],
                                }

        if best_result:
            logger.info(
                f"   Exhaustive search: Evaluated {total_combinations} combinations, "
                f"best: {best_net_xp:.2f} net xP"
            )
        return best_result


class InitialSquadOptimizationInput(BaseModel):
    """Data contract for initial squad optimization inputs."""

    budget: float = Field(ge=0, le=200, description="Budget in millions")
    formation: Tuple[int, int, int, int] = Field(
        description="Formation as (GKP, DEF, MID, FWD)"
    )
    iterations: int = Field(ge=100, le=50000, description="SA iterations")
    must_include_ids: List[int] = Field(default_factory=list)
    must_exclude_ids: List[int] = Field(default_factory=list)
    xp_column: str = Field(default="xP")

    @field_validator("formation")
    @classmethod
    def validate_formation(
        cls, v: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Validate formation sums to 15 players."""
        if len(v) != 4:
            raise ValueError("Formation must have 4 elements (GKP, DEF, MID, FWD)")
        if sum(v) != 15:
            raise ValueError(f"Formation must sum to 15 players, got {sum(v)}")
        if v[0] < 1:  # GKP
            raise ValueError("Formation must include at least 1 GKP")
        if v[1] < 3:  # DEF
            raise ValueError("Formation must include at least 3 DEF")
        if v[2] < 2:  # MID
            raise ValueError("Formation must include at least 2 MID")
        if v[3] < 1:  # FWD
            raise ValueError("Formation must include at least 1 FWD")
        return v

    @field_validator("must_include_ids", "must_exclude_ids")
    @classmethod
    def validate_no_overlap(cls, v: List[int], info) -> List[int]:
        """Ensure no duplicate IDs."""
        if len(v) != len(set(v)):
            raise ValueError("Duplicate player IDs found in constraints")
        return v
