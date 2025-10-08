"""Chip assessment service for FPL chip timing recommendations.

Provides heuristic-based analysis for optimal chip usage timing:
- Wildcard: Analyze transfer opportunities and fixture runs
- Free Hit: Detect double gameweeks and squad improvement potential
- Bench Boost: Calculate bench strength for target gameweek
- Triple Captain: Identify optimal captaincy candidates

Uses simple heuristics rather than complex optimization for fast analysis.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
import pandas as pd


class ChipRecommendation(BaseModel):
    """Structured chip recommendation with reasoning."""

    chip_name: str = Field(..., description="Name of the chip being assessed")
    status: str = Field(
        ..., description="Recommendation status: 🟢 RECOMMENDED, 🟡 CONSIDER, 🔴 HOLD"
    )
    reasoning: str = Field(..., description="Explanation for the recommendation")
    key_metrics: Dict[str, Union[str, float, int]] = Field(
        default_factory=dict,
        description="Metrics supporting the recommendation (can be numeric or text)",
    )
    optimal_gameweek: Optional[int] = Field(
        None, description="Optimal gameweek to use this chip"
    )


class ChipAssessmentService:
    """Service for assessing optimal chip usage timing with structured recommendations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the service with configuration.

        Args:
            config: Configuration dictionary for chip assessment thresholds
        """
        self.config = config or {
            "wildcard_min_transfers": 4,
            "bench_boost_min_points": 8.0,
            "triple_captain_min_xp": 8.0,
            "free_hit_min_improvement": 15.0,
            "double_gameweek_bonus": 10.0,
        }

    def assess_all_chips(
        self,
        gameweek_data: Dict[str, Any],
        current_squad: pd.DataFrame,
        available_chips: List[str],
        target_gameweek: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Assess all available chips and return structured recommendations.

        Args:
            gameweek_data: Complete gameweek data from DataOrchestrationService - guaranteed clean
            current_squad: Current squad DataFrame (15 players) - guaranteed clean
            available_chips: List of available chip names
            target_gameweek: Target gameweek for assessment (defaults to current)

        Returns:
            Chip recommendations data
        """
        if not available_chips:
            return {
                "recommendations": {},
                "summary": "No chips available for assessment",
                "target_gameweek": target_gameweek,
            }

        # Extract required data - trust it's clean
        players = gameweek_data["players"]
        fixtures = gameweek_data["fixtures"]
        target_gw = target_gameweek or gameweek_data["target_gameweek"]
        team_data = gameweek_data.get("team_data", {})

        # Assess each available chip using internal methods
        recommendations = {}

        if "wildcard" in available_chips:
            recommendations["wildcard"] = self._assess_wildcard_internal(
                current_squad, players, fixtures, target_gw, team_data
            )

        if "free_hit" in available_chips:
            recommendations["free_hit"] = self._assess_free_hit_internal(
                current_squad, players, fixtures, target_gw, team_data
            )

        if "bench_boost" in available_chips:
            recommendations["bench_boost"] = self._assess_bench_boost_internal(
                current_squad, fixtures, target_gw
            )

        if "triple_captain" in available_chips:
            recommendations["triple_captain"] = self._assess_triple_captain_internal(
                current_squad, players, fixtures, target_gw
            )

        # Generate summary
        summary = self._generate_chip_summary(recommendations)

        return {
            "recommendations": recommendations,
            "summary": summary,
            "target_gameweek": target_gw,
            "available_chips": available_chips,
        }

    def get_chip_recommendation(
        self,
        chip_name: str,
        gameweek_data: Dict[str, Any],
        current_squad: pd.DataFrame,
        target_gameweek: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get recommendation for a specific chip.

        Args:
            chip_name: Name of the chip to assess
            gameweek_data: Complete gameweek data - guaranteed clean
            current_squad: Current squad DataFrame - guaranteed clean
            target_gameweek: Target gameweek for assessment

        Returns:
            Specific chip recommendation
        """
        valid_chips = ["wildcard", "free_hit", "bench_boost", "triple_captain"]
        if chip_name not in valid_chips:
            raise ValueError(
                f"Invalid chip name: {chip_name}. Must be one of {valid_chips}"
            )

        # Assess the specific chip
        all_chips_result = self.assess_all_chips(
            gameweek_data, current_squad, [chip_name], target_gameweek
        )

        recommendations = all_chips_result["recommendations"]
        return recommendations[chip_name]

    def _assess_wildcard_internal(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
        team_data: Dict,
    ) -> Dict[str, Any]:
        """Assess wildcard chip using transfer opportunity analysis."""
        # Count beneficial transfer opportunities
        transfer_opportunities = self._count_transfer_opportunities(
            current_squad, all_players
        )

        # Analyze fixture run quality over next 5 gameweeks
        fixture_outlook = self._analyze_fixture_run(
            current_squad, fixtures, target_gameweek, horizon=5
        )

        # Calculate squad freshness
        squad_freshness = self._calculate_squad_freshness(current_squad)

        # Budget utilization analysis
        budget_available = team_data.get("bank", 0)
        budget_efficiency = self._analyze_budget_efficiency(
            current_squad, budget_available
        )

        key_metrics = {
            "transfer_opportunities": transfer_opportunities,
            "fixture_outlook": fixture_outlook,
            "squad_freshness": squad_freshness,
            "budget_available": budget_available,
            "budget_efficiency": budget_efficiency,
        }

        # Recommendation logic
        if (
            transfer_opportunities >= self.config["wildcard_min_transfers"]
            and fixture_outlook < 0.9
        ):
            status = "🟢 RECOMMENDED"
            reasoning = f"{transfer_opportunities} strong transfers available with poor fixture run ahead"
        elif transfer_opportunities >= self.config["wildcard_min_transfers"]:
            status = "🟡 CONSIDER"
            reasoning = f"{transfer_opportunities} transfers available, but fixtures look decent"
        elif fixture_outlook < 0.8:
            status = "🟡 CONSIDER"
            reasoning = "Very poor fixture run ahead, consider wildcard for fresh start"
        else:
            status = "🔴 HOLD"
            reasoning = "Few beneficial transfers available, squad looks strong"

        recommendation = ChipRecommendation(
            chip_name="Wildcard",
            status=status,
            reasoning=reasoning,
            key_metrics=key_metrics,
        )
        return self._convert_chip_recommendation(recommendation)

    def _assess_free_hit_internal(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
        team_data: Dict,
    ) -> Dict[str, Any]:
        """Assess free hit chip focusing on double gameweeks and squad gaps."""
        # Detect double gameweeks
        double_gw_teams, double_gw_strength = self._detect_double_gameweek(
            fixtures, target_gameweek
        )

        # Count missing/unavailable key players
        missing_players = self._count_unavailable_players(current_squad)

        # Estimate temporary squad improvement potential
        temp_squad_improvement = self._estimate_temp_squad_improvement(
            current_squad, all_players, target_gameweek, team_data.get("bank", 0)
        )

        # Analyze current squad's gameweek outlook
        current_outlook = self._analyze_single_gw_outlook(
            current_squad, fixtures, target_gameweek
        )

        key_metrics = {
            "double_gw_teams": len(double_gw_teams),
            "double_gw_strength": double_gw_strength,
            "missing_players": missing_players,
            "temp_squad_improvement": temp_squad_improvement,
            "current_outlook": current_outlook,
        }

        # Recommendation logic
        if double_gw_teams and double_gw_strength > 1.2:
            status = "🟢 RECOMMENDED"
            reasoning = f"Strong double gameweek opportunity - {len(double_gw_teams)} quality teams playing twice"
        elif temp_squad_improvement > self.config["free_hit_min_improvement"]:
            status = "🟢 RECOMMENDED"
            reasoning = (
                f"Major squad improvement possible (+{temp_squad_improvement:.1f} xP)"
            )
        elif missing_players >= 3 or current_outlook < 0.7:
            status = "🟡 CONSIDER"
            reasoning = f"Squad issues present ({missing_players} missing players, poor outlook)"
        else:
            status = "🔴 HOLD"
            reasoning = "Current squad looks strong for this gameweek"

        recommendation = ChipRecommendation(
            chip_name="Free Hit",
            status=status,
            reasoning=reasoning,
            key_metrics=key_metrics,
        )
        return self._convert_chip_recommendation(recommendation)

    def _assess_bench_boost_internal(
        self, current_squad: pd.DataFrame, fixtures: pd.DataFrame, target_gameweek: int
    ) -> Dict[str, Any]:
        """Assess bench boost chip based on bench strength."""
        # Check if xP column exists
        if current_squad.empty or "xP" not in current_squad.columns:
            recommendation = ChipRecommendation(
                chip_name="Bench Boost",
                status="🔴 HOLD",
                reasoning="Expected points data not available",
                key_metrics={"bench_xp": 0, "data_available": False},
            )
            return self._convert_chip_recommendation(recommendation)

        # Identify bench players (lowest 4 xP players typically)
        if len(current_squad) >= 15:
            bench_players = current_squad.nsmallest(4, "xP")
        else:
            bench_players = pd.DataFrame()

        # Calculate bench expected points
        bench_xp = bench_players["xP"].sum() if not bench_players.empty else 0

        # Analyze bench player fixtures
        bench_fixture_quality = self._analyze_bench_fixtures(
            bench_players, fixtures, target_gameweek
        )

        # Check for bench player rotation risks
        bench_rotation_risk = self._assess_bench_rotation_risk(bench_players)

        key_metrics = {
            "bench_xp": bench_xp,
            "bench_fixture_quality": bench_fixture_quality,
            "bench_rotation_risk": bench_rotation_risk,
            "bench_size": len(bench_players),
        }

        # Recommendation logic
        if (
            bench_xp >= self.config["bench_boost_min_points"]
            and bench_fixture_quality > 1.0
        ):
            status = "🟢 RECOMMENDED"
            reasoning = f"Strong bench ({bench_xp:.1f} xP) with good fixtures"
        elif bench_xp >= self.config["bench_boost_min_points"]:
            status = "🟡 CONSIDER"
            reasoning = f"Decent bench strength ({bench_xp:.1f} xP) but mixed fixtures"
        else:
            status = "🔴 HOLD"
            reasoning = f"Weak bench ({bench_xp:.1f} xP) - not worth boosting"

        recommendation = ChipRecommendation(
            chip_name="Bench Boost",
            status=status,
            reasoning=reasoning,
            key_metrics=key_metrics,
        )
        return self._convert_chip_recommendation(recommendation)

    def _assess_triple_captain_internal(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
    ) -> Dict[str, Any]:
        """Assess triple captain chip based on high xP candidates."""
        # Check if xP column exists
        if current_squad.empty or "xP" not in current_squad.columns:
            recommendation = ChipRecommendation(
                chip_name="Triple Captain",
                status="🔴 HOLD",
                reasoning="Expected points data not available",
                key_metrics={"top_candidate_xp": 0, "data_available": False},
            )
            return self._convert_chip_recommendation(recommendation)

        # Find best captain candidates in current squad
        captain_candidates = current_squad.nlargest(5, "xP")

        # Get the top candidate
        if not captain_candidates.empty:
            top_candidate = captain_candidates.iloc[0]
            top_xp = top_candidate["xP"]
            top_name = top_candidate["web_name"]
        else:
            top_xp = 0
            top_name = "None"

        # Check for fixture quality of top candidates
        captain_fixture_quality = self._analyze_captain_fixtures(
            captain_candidates, fixtures, target_gameweek
        )

        # Assess rotation risk for top candidates
        rotation_risk = self._assess_captain_rotation_risk(captain_candidates)

        # Compare with premium alternatives not in squad
        premium_alternatives = self._find_premium_captain_alternatives(
            current_squad, all_players
        )

        key_metrics = {
            "top_candidate_xp": top_xp,
            "top_candidate_name": top_name,
            "captain_fixture_quality": captain_fixture_quality,
            "rotation_risk": rotation_risk,
            "premium_alternatives": len(premium_alternatives),
        }

        # Recommendation logic
        if (
            top_xp >= self.config["triple_captain_min_xp"]
            and captain_fixture_quality > 1.1
            and rotation_risk < 0.3
        ):
            status = "🟢 RECOMMENDED"
            reasoning = (
                f"{top_name} excellent candidate ({top_xp:.1f} xP, great fixture)"
            )
        elif top_xp >= self.config["triple_captain_min_xp"]:
            status = "🟡 CONSIDER"
            reasoning = f"{top_name} good candidate ({top_xp:.1f} xP) but check fixtures/rotation"
        else:
            status = "🔴 HOLD"
            reasoning = f"No premium candidates in squad (best: {top_xp:.1f} xP)"

        recommendation = ChipRecommendation(
            chip_name="Triple Captain",
            status=status,
            reasoning=reasoning,
            key_metrics=key_metrics,
        )
        return self._convert_chip_recommendation(recommendation)

    def _convert_chip_recommendation(self, recommendation) -> Dict[str, Any]:
        """Convert ChipRecommendation object to dictionary."""
        if hasattr(recommendation, "__dict__"):
            return {
                "chip_name": recommendation.chip_name,
                "status": recommendation.status,
                "reasoning": getattr(
                    recommendation, "reasoning", "No reasoning provided"
                ),
                "key_metrics": recommendation.key_metrics,
                "optimal_gameweek": recommendation.optimal_gameweek,
            }
        else:
            # Fallback for dict-like objects
            return {
                "chip_name": recommendation.get("chip_name", "Unknown"),
                "status": recommendation.get("status", "🟡 UNKNOWN"),
                "reasoning": recommendation.get("reasoning", "No reasoning provided"),
                "key_metrics": recommendation.get("key_metrics", {}),
                "optimal_gameweek": recommendation.get("optimal_gameweek"),
            }

    def _generate_chip_summary(self, recommendations: Dict[str, Any]) -> str:
        """Generate a summary of all chip recommendations."""
        if not recommendations:
            return "No chip recommendations available"

        summary_parts = []

        # Count recommendations by status
        recommended = []
        consider = []
        hold = []

        for chip_name, rec in recommendations.items():
            status = rec.get("status", "🟡 UNKNOWN")
            if "🟢" in status:
                recommended.append(chip_name.replace("_", " ").title())
            elif "🟡" in status:
                consider.append(chip_name.replace("_", " ").title())
            else:
                hold.append(chip_name.replace("_", " ").title())

        if recommended:
            summary_parts.append(f"🟢 RECOMMENDED: {', '.join(recommended)}")
        if consider:
            summary_parts.append(f"🟡 CONSIDER: {', '.join(consider)}")
        if hold:
            summary_parts.append(f"🔴 HOLD: {', '.join(hold)}")

        return " | ".join(summary_parts) if summary_parts else "All chips assessed"

    def get_chip_timing_analysis(
        self,
        gameweek_data: Dict[str, Any],
        current_squad: pd.DataFrame,
        available_chips: List[str],
        gameweeks_ahead: int = 5,
    ) -> Dict[str, Any]:
        """Analyze optimal timing for chip usage over multiple gameweeks.

        Args:
            gameweek_data: Complete gameweek data - guaranteed clean
            current_squad: Current squad DataFrame - guaranteed clean
            available_chips: List of available chips
            gameweeks_ahead: Number of gameweeks to analyze ahead

        Returns:
            Timing analysis for each chip
        """
        current_gw = gameweek_data["target_gameweek"]
        timing_analysis = {}

        # Analyze each chip across multiple gameweeks
        for chip in available_chips:
            chip_timing = []

            for gw_offset in range(gameweeks_ahead):
                target_gw = current_gw + gw_offset
                if target_gw > 38:  # Don't go beyond season end
                    break

                rec = self.get_chip_recommendation(
                    chip, gameweek_data, current_squad, target_gw
                )

                chip_timing.append(
                    {
                        "gameweek": target_gw,
                        "status": rec.get("status", "🟡 UNKNOWN"),
                        "key_metrics": rec.get("key_metrics", {}),
                    }
                )

            timing_analysis[chip] = {
                "gameweek_analysis": chip_timing,
                "optimal_gameweek": self._find_optimal_gameweek(chip_timing),
            }

        return {
            "timing_analysis": timing_analysis,
            "analysis_period": f"GW{current_gw}-{min(current_gw + gameweeks_ahead - 1, 38)}",
            "available_chips": available_chips,
        }

    def _find_optimal_gameweek(self, chip_timing: List[Dict]) -> Optional[int]:
        """Find the optimal gameweek for a chip based on timing analysis."""
        if not chip_timing:
            return None

        # Prefer RECOMMENDED status, then CONSIDER
        for timing in chip_timing:
            if "🟢" in timing.get("status", ""):
                return timing["gameweek"]

        for timing in chip_timing:
            if "🟡" in timing.get("status", ""):
                return timing["gameweek"]

        # Fallback to first gameweek
        return chip_timing[0]["gameweek"]

    # Helper methods for chip assessment analysis

    def _count_transfer_opportunities(
        self, current_squad: pd.DataFrame, all_players: pd.DataFrame
    ) -> int:
        """Count how many beneficial transfers are available."""
        if current_squad.empty or all_players.empty:
            return 0

        # Check if required columns exist
        if "xP" not in current_squad.columns or "xP" not in all_players.columns:
            return 0

        current_player_ids = set(current_squad["player_id"].tolist())
        available_players = all_players[
            ~all_players["player_id"].isin(current_player_ids)
        ]

        transfer_count = 0

        for position in ["GKP", "DEF", "MID", "FWD"]:
            current_pos = current_squad[current_squad["position"] == position]
            available_pos = available_players[available_players["position"] == position]

            if current_pos.empty or available_pos.empty:
                continue

            # Find upgrades (better xP for similar or higher price)
            for _, current_player in current_pos.iterrows():
                current_xp = current_player.get("xP", 0)
                current_price = current_player.get("price", 0)

                # Look for upgrades within reasonable price range
                upgrades = available_pos[
                    (available_pos["xP"] > current_xp + 2)  # Significant improvement
                    & (
                        available_pos["price"] <= current_price + 2
                    )  # Affordable upgrade
                ]

                if not upgrades.empty:
                    transfer_count += 1

        return min(transfer_count, 11)  # Cap at reasonable number

    def _analyze_fixture_run(
        self,
        current_squad: pd.DataFrame,
        fixtures: pd.DataFrame,
        start_gw: int,
        horizon: int = 5,
    ) -> float:
        """Analyze fixture difficulty over next N gameweeks."""
        if current_squad.empty:
            return 1.0

        total_difficulty = 0
        team_count = 0

        for _, player in current_squad.iterrows():
            team_name = player.get("name", "")
            if not team_name:
                continue

            team_difficulties = []
            for gw in range(start_gw, start_gw + horizon):
                # This is simplified - in practice, you'd match team fixtures
                # For now, use a default moderate difficulty
                team_difficulties.append(1.0)

            if team_difficulties:
                avg_difficulty = sum(team_difficulties) / len(team_difficulties)
                total_difficulty += avg_difficulty
                team_count += 1

        return total_difficulty / team_count if team_count > 0 else 1.0

    def _calculate_squad_freshness(self, current_squad: pd.DataFrame) -> float:
        """Calculate what percentage of squad might need refreshing."""
        if current_squad.empty:
            return 0.0

        # Simple heuristic: players with low xP per price ratio
        if "price" in current_squad.columns and "xP" in current_squad.columns:
            current_squad = current_squad.copy()
            current_squad["xp_per_price"] = current_squad["xP"] / current_squad[
                "price"
            ].clip(lower=4.0)

            # Count players in bottom quartile of efficiency
            threshold = current_squad["xp_per_price"].quantile(0.25)
            stale_players = (current_squad["xp_per_price"] <= threshold).sum()

            return stale_players / len(current_squad)

        return 0.0

    def _analyze_budget_efficiency(
        self, current_squad: pd.DataFrame, budget_available: float
    ) -> float:
        """Analyze how efficiently current budget could be used."""
        if budget_available < 1.0:
            return 1.0  # No budget to improve

        # Simple heuristic: more budget = more improvement potential
        return min(budget_available / 10.0, 1.0)

    def _detect_double_gameweek(
        self, fixtures: pd.DataFrame, target_gw: int
    ) -> Tuple[List[str], float]:
        """Detect if any teams have double gameweeks."""
        if fixtures.empty:
            return [], 0.0

        # Count fixtures per team for target gameweek
        # Note: DataOrchestrationService renames "event" to "gameweek"
        gameweek_col = "gameweek" if "gameweek" in fixtures.columns else "event"
        gw_fixtures = fixtures[fixtures[gameweek_col] == target_gw]

        if gw_fixtures.empty:
            return [], 0.0

        # Count home and away appearances
        team_fixture_counts = {}

        # Note: DataOrchestrationService renames "home_team_id" to "team_h" and "away_team_id" to "team_a"
        home_col = "team_h" if "team_h" in gw_fixtures.columns else "home_team_id"
        away_col = "team_a" if "team_a" in gw_fixtures.columns else "away_team_id"

        for _, fixture in gw_fixtures.iterrows():
            home_team = fixture.get(home_col, "")
            away_team = fixture.get(away_col, "")

            if home_team:  # Only count if team ID exists
                team_fixture_counts[home_team] = (
                    team_fixture_counts.get(home_team, 0) + 1
                )
            if away_team:  # Only count if team ID exists
                team_fixture_counts[away_team] = (
                    team_fixture_counts.get(away_team, 0) + 1
                )

        # Find teams with 2+ fixtures
        double_gw_teams = [
            team for team, count in team_fixture_counts.items() if count >= 2
        ]

        # Estimate strength of double gameweek (simplified)
        avg_strength = 1.0 + (len(double_gw_teams) * 0.1)

        return double_gw_teams, avg_strength

    def _count_unavailable_players(self, current_squad: pd.DataFrame) -> int:
        """Count players who are injured, suspended, or doubtful."""
        if current_squad.empty:
            return 0

        unavailable_count = 0

        for _, player in current_squad.iterrows():
            availability = player.get("status", "a")
            if availability in [
                "i",
                "s",
                "u",
                "d",
            ]:  # injured, suspended, unavailable, doubtful
                unavailable_count += 1

        return unavailable_count

    def _estimate_temp_squad_improvement(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        target_gw: int,
        budget: float,
    ) -> float:
        """Estimate xP improvement possible with temporary squad."""
        if current_squad.empty or all_players.empty:
            return 0.0

        # Check if required columns exist
        if "xP" not in current_squad.columns or "xP" not in all_players.columns:
            return 0.0

        current_best_11_xp = current_squad.nlargest(11, "xP")["xP"].sum()

        # Simple estimate: could we build a much better team for one week?
        # This is a simplified calculation
        total_budget = budget + current_squad["price"].sum()

        # Estimate what top 11 players we could afford
        affordable_players = all_players[all_players["price"] <= total_budget / 11]

        if not affordable_players.empty:
            temp_best_11_xp = affordable_players.nlargest(11, "xP")["xP"].sum()
            improvement = temp_best_11_xp - current_best_11_xp
            return max(0, improvement)

        return 0.0

    def _analyze_single_gw_outlook(
        self, current_squad: pd.DataFrame, fixtures: pd.DataFrame, target_gw: int
    ) -> float:
        """Analyze how good current squad looks for single gameweek."""
        if current_squad.empty:
            return 0.5

        # Check if xP column exists
        if "xP" not in current_squad.columns:
            return 0.5

        # Simplified: average xP of starting 11
        starting_11_xp = current_squad.nlargest(11, "xP")["xP"].mean()

        # Normalize to 0-1 scale (assuming 4-8 xP range per player)
        return min(max((starting_11_xp - 4) / 4, 0), 1)

    def _analyze_bench_fixtures(
        self, bench_players: pd.DataFrame, fixtures: pd.DataFrame, target_gw: int
    ) -> float:
        """Analyze fixture quality for bench players."""
        if bench_players.empty:
            return 1.0

        # Simplified: assume average fixture quality
        return 1.0

    def _assess_bench_rotation_risk(self, bench_players: pd.DataFrame) -> float:
        """Assess rotation risk for bench players."""
        if bench_players.empty:
            return 0.0

        # Simplified: lower priced players have higher rotation risk
        if "price" in bench_players.columns:
            avg_price = bench_players["price"].mean()
            # Lower price = higher rotation risk
            return max(0, min(1, (7 - avg_price) / 3))

        return 0.3  # Default moderate risk

    def _analyze_captain_fixtures(
        self, captain_candidates: pd.DataFrame, fixtures: pd.DataFrame, target_gw: int
    ) -> float:
        """Analyze fixture quality for captain candidates."""
        if captain_candidates.empty:
            return 1.0

        # Simplified: assume good fixtures for top players
        return 1.1

    def _assess_captain_rotation_risk(self, captain_candidates: pd.DataFrame) -> float:
        """Assess rotation risk for captain candidates."""
        if captain_candidates.empty:
            return 1.0

        # Simplified: premium players have low rotation risk
        if "price" in captain_candidates.columns:
            avg_price = captain_candidates["price"].mean()
            # Higher price = lower rotation risk
            return max(0, min(1, (10 - avg_price) / 5))

        return 0.2  # Default low risk for top candidates

    def _find_premium_captain_alternatives(
        self, current_squad: pd.DataFrame, all_players: pd.DataFrame
    ) -> pd.DataFrame:
        """Find premium captain alternatives not in current squad."""
        if current_squad.empty or all_players.empty:
            return pd.DataFrame()

        # Check if required columns exist
        if "xP" not in all_players.columns:
            return pd.DataFrame()

        current_player_ids = set(current_squad["player_id"].tolist())
        premium_players = all_players[
            (~all_players["player_id"].isin(current_player_ids))
            & (all_players["price"] >= 9.0)
            & (all_players["xP"] >= 8.0)
        ]

        return premium_players.nlargest(5, "xP")
