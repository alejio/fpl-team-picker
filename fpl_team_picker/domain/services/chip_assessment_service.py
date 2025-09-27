"""Chip assessment service for FPL chip timing recommendations."""

from typing import Dict, Any, List, Optional
import pandas as pd

from fpl_team_picker.domain.common.result import Result, DomainError, ErrorType


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
    ) -> Result[Dict[str, Any]]:
        """Assess all available chips and return structured recommendations.

        Args:
            gameweek_data: Complete gameweek data from DataOrchestrationService
            current_squad: Current squad DataFrame (15 players)
            available_chips: List of available chip names
            target_gameweek: Target gameweek for assessment (defaults to current)

        Returns:
            Result containing chip recommendations or error
        """
        try:
            # Validate inputs
            if not available_chips:
                return Result(
                    value={
                        "recommendations": {},
                        "summary": "No chips available for assessment",
                        "target_gameweek": target_gameweek,
                    }
                )

            if len(current_squad) != 15:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Current squad must contain exactly 15 players, got {len(current_squad)}",
                    )
                )

            # Extract required data
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
                wildcard_result = self._assess_wildcard_safe(
                    chip_engine, current_squad, players, fixtures, target_gw, team_data
                )
                if wildcard_result.is_success:
                    recommendations["wildcard"] = wildcard_result.value

            if "free_hit" in available_chips:
                free_hit_result = self._assess_free_hit_safe(
                    chip_engine, current_squad, players, fixtures, target_gw, team_data
                )
                if free_hit_result.is_success:
                    recommendations["free_hit"] = free_hit_result.value

            if "bench_boost" in available_chips:
                bench_boost_result = self._assess_bench_boost_safe(
                    chip_engine, current_squad, fixtures, target_gw
                )
                if bench_boost_result.is_success:
                    recommendations["bench_boost"] = bench_boost_result.value

            if "triple_captain" in available_chips:
                triple_captain_result = self._assess_triple_captain_safe(
                    chip_engine, current_squad, players, fixtures, target_gw
                )
                if triple_captain_result.is_success:
                    recommendations["triple_captain"] = triple_captain_result.value

            # Generate summary
            summary = self._generate_chip_summary(recommendations)

            return Result(
                value={
                    "recommendations": recommendations,
                    "summary": summary,
                    "target_gameweek": target_gw,
                    "available_chips": available_chips,
                }
            )

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Chip assessment failed: {str(e)}",
                )
            )

    def get_chip_recommendation(
        self,
        chip_name: str,
        gameweek_data: Dict[str, Any],
        current_squad: pd.DataFrame,
        target_gameweek: Optional[int] = None,
    ) -> Result[Dict[str, Any]]:
        """Get recommendation for a specific chip.

        Args:
            chip_name: Name of the chip to assess
            gameweek_data: Complete gameweek data
            current_squad: Current squad DataFrame
            target_gameweek: Target gameweek for assessment

        Returns:
            Result containing specific chip recommendation
        """
        try:
            # Validate chip name
            valid_chips = ["wildcard", "free_hit", "bench_boost", "triple_captain"]
            if chip_name not in valid_chips:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Invalid chip name: {chip_name}. Must be one of {valid_chips}",
                    )
                )

            # Assess the specific chip
            all_chips_result = self.assess_all_chips(
                gameweek_data, current_squad, [chip_name], target_gameweek
            )

            if all_chips_result.is_failure:
                return all_chips_result

            recommendations = all_chips_result.value["recommendations"]
            if chip_name not in recommendations:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.CALCULATION_ERROR,
                        message=f"Failed to generate recommendation for {chip_name}",
                    )
                )

            return Result(value=recommendations[chip_name])

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Chip recommendation failed: {str(e)}",
                )
            )

    def _assess_wildcard_safe(
        self,
        chip_engine,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
        team_data: Dict,
    ) -> Result[Dict[str, Any]]:
        """Safely assess wildcard chip."""
        try:
            recommendation = chip_engine.assess_wildcard(
                current_squad, all_players, fixtures, target_gameweek, team_data
            )
            return Result(value=self._convert_chip_recommendation(recommendation))
        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Wildcard assessment failed: {str(e)}",
                )
            )

    def _assess_free_hit_safe(
        self,
        chip_engine,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
        team_data: Dict,
    ) -> Result[Dict[str, Any]]:
        """Safely assess free hit chip."""
        try:
            recommendation = chip_engine.assess_free_hit(
                current_squad, all_players, fixtures, target_gameweek, team_data
            )
            return Result(value=self._convert_chip_recommendation(recommendation))
        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Free hit assessment failed: {str(e)}",
                )
            )

    def _assess_bench_boost_safe(
        self,
        chip_engine,
        current_squad: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
    ) -> Result[Dict[str, Any]]:
        """Safely assess bench boost chip."""
        try:
            recommendation = chip_engine.assess_bench_boost(
                current_squad, fixtures, target_gameweek
            )
            return Result(value=self._convert_chip_recommendation(recommendation))
        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Bench boost assessment failed: {str(e)}",
                )
            )

    def _assess_triple_captain_safe(
        self,
        chip_engine,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
    ) -> Result[Dict[str, Any]]:
        """Safely assess triple captain chip."""
        try:
            recommendation = chip_engine.assess_triple_captain(
                current_squad, all_players, fixtures, target_gameweek
            )
            return Result(value=self._convert_chip_recommendation(recommendation))
        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Triple captain assessment failed: {str(e)}",
                )
            )

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
    ) -> Result[Dict[str, Any]]:
        """Analyze optimal timing for chip usage over multiple gameweeks.

        Args:
            gameweek_data: Complete gameweek data
            current_squad: Current squad DataFrame
            available_chips: List of available chips
            gameweeks_ahead: Number of gameweeks to analyze ahead

        Returns:
            Result containing timing analysis for each chip
        """
        try:
            current_gw = gameweek_data["target_gameweek"]
            timing_analysis = {}

            # Analyze each chip across multiple gameweeks
            for chip in available_chips:
                chip_timing = []

                for gw_offset in range(gameweeks_ahead):
                    target_gw = current_gw + gw_offset
                    if target_gw > 38:  # Don't go beyond season end
                        break

                    chip_result = self.get_chip_recommendation(
                        chip, gameweek_data, current_squad, target_gw
                    )

                    if chip_result.is_success:
                        rec = chip_result.value
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

            return Result(
                value={
                    "timing_analysis": timing_analysis,
                    "analysis_period": f"GW{current_gw}-{min(current_gw + gameweeks_ahead - 1, 38)}",
                    "available_chips": available_chips,
                }
            )

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Chip timing analysis failed: {str(e)}",
                )
            )

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
