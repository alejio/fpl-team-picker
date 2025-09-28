"""Chip assessment service for FPL chip timing recommendations."""

from typing import Dict, Any, List, Optional
import pandas as pd


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

        # Use existing chip assessment engine
        from fpl_team_picker.core.chip_assessment import ChipAssessmentEngine

        chip_engine = ChipAssessmentEngine(self.config)

        # Assess each available chip
        recommendations = {}

        if "wildcard" in available_chips:
            recommendations["wildcard"] = self._assess_wildcard(
                chip_engine, current_squad, players, fixtures, target_gw, team_data
            )

        if "free_hit" in available_chips:
            recommendations["free_hit"] = self._assess_free_hit(
                chip_engine, current_squad, players, fixtures, target_gw, team_data
            )

        if "bench_boost" in available_chips:
            recommendations["bench_boost"] = self._assess_bench_boost(
                chip_engine, current_squad, fixtures, target_gw
            )

        if "triple_captain" in available_chips:
            recommendations["triple_captain"] = self._assess_triple_captain(
                chip_engine, current_squad, players, fixtures, target_gw
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

    def _assess_wildcard(
        self,
        chip_engine,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
        team_data: Dict,
    ) -> Dict[str, Any]:
        """Assess wildcard chip."""
        recommendation = chip_engine.assess_wildcard(
            current_squad, all_players, fixtures, target_gameweek, team_data
        )
        return self._convert_chip_recommendation(recommendation)

    def _assess_free_hit(
        self,
        chip_engine,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
        team_data: Dict,
    ) -> Dict[str, Any]:
        """Assess free hit chip."""
        recommendation = chip_engine.assess_free_hit(
            current_squad, all_players, fixtures, target_gameweek, team_data
        )
        return self._convert_chip_recommendation(recommendation)

    def _assess_bench_boost(
        self,
        chip_engine,
        current_squad: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
    ) -> Dict[str, Any]:
        """Assess bench boost chip."""
        recommendation = chip_engine.assess_bench_boost(
            current_squad, fixtures, target_gameweek
        )
        return self._convert_chip_recommendation(recommendation)

    def _assess_triple_captain(
        self,
        chip_engine,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
    ) -> Dict[str, Any]:
        """Assess triple captain chip."""
        recommendation = chip_engine.assess_triple_captain(
            current_squad, all_players, fixtures, target_gameweek
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
