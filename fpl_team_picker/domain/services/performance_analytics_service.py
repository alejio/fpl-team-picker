"""Performance analytics service for FPL form analysis and trend detection."""

from typing import Dict, Any, List, Optional
import pandas as pd

from fpl_team_picker.domain.common.result import Result, DomainError, ErrorType


class PerformanceAnalyticsService:
    """Service for analyzing player performance, form, and trends."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the service with configuration.

        Args:
            config: Configuration dictionary for performance analytics
        """
        self.config = config or {
            "hot_threshold": 1.2,  # Form multiplier threshold for "hot" players
            "cold_threshold": 0.8,  # Form multiplier threshold for "cold" players
            "momentum_window": 5,  # Gameweeks to consider for momentum
            "min_minutes_threshold": 60,  # Minimum minutes for analysis
        }

    def analyze_player_form(
        self,
        players_with_xp: pd.DataFrame,
        live_data_historical: Optional[pd.DataFrame] = None,
    ) -> Result[Dict[str, Any]]:
        """Analyze player form and momentum indicators.

        Args:
            players_with_xp: DataFrame with current player data and XP calculations
            live_data_historical: Optional historical performance data

        Returns:
            Result containing comprehensive form analysis
        """
        try:
            if players_with_xp.empty:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No player data available for form analysis",
                    )
                )

            # Check for required form columns
            form_columns = ["momentum", "form_multiplier"]
            available_form_columns = [
                col for col in form_columns if col in players_with_xp.columns
            ]

            if not available_form_columns:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No form data available - requires momentum or form_multiplier columns",
                    )
                )

            # Analyze different aspects of form
            form_analysis = {}

            # Hot and cold players analysis
            hot_cold_analysis = self._analyze_hot_cold_players(players_with_xp)
            form_analysis.update(hot_cold_analysis)

            # Momentum trends
            momentum_analysis = self._analyze_momentum_trends(players_with_xp)
            form_analysis.update(momentum_analysis)

            # Value players with good form
            value_form_analysis = self._analyze_value_form_players(players_with_xp)
            form_analysis.update(value_form_analysis)

            # Expensive underperformers
            underperformer_analysis = self._analyze_underperformers(players_with_xp)
            form_analysis.update(underperformer_analysis)

            # Historical trends if data available
            if live_data_historical is not None and not live_data_historical.empty:
                historical_analysis = self._analyze_historical_trends(
                    players_with_xp, live_data_historical
                )
                form_analysis.update(historical_analysis)

            return Result(
                value={
                    "form_analysis": form_analysis,
                    "total_players_analyzed": len(players_with_xp),
                    "available_form_columns": available_form_columns,
                    "has_historical_data": live_data_historical is not None
                    and not live_data_historical.empty,
                }
            )

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Form analysis failed: {str(e)}",
                )
            )

    def detect_breakout_players(
        self,
        players_with_xp: pd.DataFrame,
        price_threshold: float = 7.0,
        xp_threshold: float = 6.0,
    ) -> Result[List[Dict[str, Any]]]:
        """Detect potential breakout players based on form and value.

        Args:
            players_with_xp: DataFrame with player data
            price_threshold: Maximum price for breakout consideration
            xp_threshold: Minimum expected points threshold

        Returns:
            Result containing list of potential breakout players
        """
        try:
            if players_with_xp.empty:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No player data available for breakout analysis",
                    )
                )

            # Filter potential breakout players
            breakout_candidates = players_with_xp[
                (players_with_xp["price"] <= price_threshold)
                & (players_with_xp["xP"] >= xp_threshold)
            ].copy()

            # Add form indicators if available
            if "momentum" in breakout_candidates.columns:
                breakout_candidates = breakout_candidates[
                    breakout_candidates["momentum"].isin(["🔥", "📈"])
                ]

            # Sort by expected points and value
            if "xP_per_price" in breakout_candidates.columns:
                breakout_candidates = breakout_candidates.sort_values(
                    ["xP_per_price", "xP"], ascending=[False, False]
                )
            else:
                breakout_candidates = breakout_candidates.sort_values(
                    "xP", ascending=False
                )

            # Convert to list of dictionaries
            breakout_players = breakout_candidates.head(10).to_dict("records")

            return Result(value=breakout_players)

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Breakout player detection failed: {str(e)}",
                )
            )

    def analyze_position_trends(
        self,
        players_with_xp: pd.DataFrame,
        position: Optional[str] = None,
    ) -> Result[Dict[str, Any]]:
        """Analyze performance trends by position.

        Args:
            players_with_xp: DataFrame with player data
            position: Specific position to analyze (None for all positions)

        Returns:
            Result containing position-specific trend analysis
        """
        try:
            if players_with_xp.empty:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No player data available for position analysis",
                    )
                )

            # Validate position if specified
            valid_positions = ["GKP", "DEF", "MID", "FWD"]
            if position and position not in valid_positions:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Invalid position: {position}. Must be one of {valid_positions}",
                    )
                )

            position_analysis = {}

            # Analyze specific position or all positions
            positions_to_analyze = [position] if position else valid_positions

            for pos in positions_to_analyze:
                pos_players = players_with_xp[players_with_xp["position"] == pos]

                if pos_players.empty:
                    continue

                pos_stats = {
                    "total_players": len(pos_players),
                    "avg_xp": pos_players["xP"].mean(),
                    "max_xp": pos_players["xP"].max(),
                    "avg_price": pos_players["price"].mean(),
                    "top_player": pos_players.loc[
                        pos_players["xP"].idxmax(), "web_name"
                    ],
                }

                # Add form statistics if available
                if "momentum" in pos_players.columns:
                    pos_stats["hot_players"] = len(
                        pos_players[pos_players["momentum"] == "🔥"]
                    )
                    pos_stats["cold_players"] = len(
                        pos_players[pos_players["momentum"] == "❄️"]
                    )

                # Add value statistics if available
                if "xP_per_price" in pos_players.columns:
                    pos_stats["avg_value"] = pos_players["xP_per_price"].mean()
                    pos_stats["best_value_player"] = pos_players.loc[
                        pos_players["xP_per_price"].idxmax(), "web_name"
                    ]

                position_analysis[pos] = pos_stats

            return Result(
                value={
                    "position_analysis": position_analysis,
                    "analyzed_positions": positions_to_analyze,
                    "total_players": len(players_with_xp),
                }
            )

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Position trend analysis failed: {str(e)}",
                )
            )

    def get_statistical_insights(
        self,
        players_with_xp: pd.DataFrame,
        live_data_historical: Optional[pd.DataFrame] = None,
    ) -> Result[Dict[str, Any]]:
        """Generate statistical insights from player performance data.

        Args:
            players_with_xp: DataFrame with current player data
            live_data_historical: Optional historical performance data

        Returns:
            Result containing statistical insights and metrics
        """
        try:
            if players_with_xp.empty:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No player data available for statistical analysis",
                    )
                )

            insights = {}

            # Basic distribution statistics
            insights["xp_distribution"] = {
                "mean": players_with_xp["xP"].mean(),
                "median": players_with_xp["xP"].median(),
                "std": players_with_xp["xP"].std(),
                "min": players_with_xp["xP"].min(),
                "max": players_with_xp["xP"].max(),
            }

            # Price vs performance correlation
            if "price" in players_with_xp.columns:
                correlation = players_with_xp["xP"].corr(players_with_xp["price"])
                insights["price_performance_correlation"] = correlation

            # Form distribution if available
            if "form_multiplier" in players_with_xp.columns:
                insights["form_distribution"] = {
                    "mean": players_with_xp["form_multiplier"].mean(),
                    "hot_players": len(
                        players_with_xp[
                            players_with_xp["form_multiplier"]
                            >= self.config["hot_threshold"]
                        ]
                    ),
                    "cold_players": len(
                        players_with_xp[
                            players_with_xp["form_multiplier"]
                            <= self.config["cold_threshold"]
                        ]
                    ),
                }

            # Minutes analysis if available
            if "expected_minutes" in players_with_xp.columns:
                insights["minutes_analysis"] = {
                    "avg_expected_minutes": players_with_xp["expected_minutes"].mean(),
                    "nailed_on_players": len(
                        players_with_xp[players_with_xp["expected_minutes"] >= 75]
                    ),
                    "rotation_risk_players": len(
                        players_with_xp[
                            players_with_xp["expected_minutes"]
                            < self.config["min_minutes_threshold"]
                        ]
                    ),
                }

            # Historical comparison if available
            if live_data_historical is not None and not live_data_historical.empty:
                historical_insights = self._generate_historical_insights(
                    players_with_xp, live_data_historical
                )
                insights["historical_comparison"] = historical_insights

            return Result(value=insights)

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Statistical insights generation failed: {str(e)}",
                )
            )

    def _analyze_hot_cold_players(
        self, players_with_xp: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze hot and cold players based on momentum."""
        analysis = {}

        if "momentum" in players_with_xp.columns:
            hot_players = players_with_xp[players_with_xp["momentum"] == "🔥"].nlargest(
                8, "xP"
            )

            cold_players = players_with_xp[
                players_with_xp["momentum"] == "❄️"
            ].nsmallest(
                8,
                "form_multiplier"
                if "form_multiplier" in players_with_xp.columns
                else "xP",
            )

            analysis["hot_players"] = hot_players.to_dict("records")
            analysis["cold_players"] = cold_players.to_dict("records")
            analysis["hot_count"] = len(
                players_with_xp[players_with_xp["momentum"] == "🔥"]
            )
            analysis["cold_count"] = len(
                players_with_xp[players_with_xp["momentum"] == "❄️"]
            )

        return analysis

    def _analyze_momentum_trends(self, players_with_xp: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum trends across all players."""
        analysis = {}

        if "momentum" in players_with_xp.columns:
            momentum_counts = players_with_xp["momentum"].value_counts()
            analysis["momentum_distribution"] = momentum_counts.to_dict()

            # Trending players
            trending_up = players_with_xp[
                players_with_xp["momentum"].isin(["🔥", "📈"])
            ].nlargest(10, "xP")

            trending_down = players_with_xp[
                players_with_xp["momentum"].isin(["❄️", "📉"])
            ].nsmallest(10, "xP")

            analysis["trending_up"] = trending_up.to_dict("records")
            analysis["trending_down"] = trending_down.to_dict("records")

        return analysis

    def _analyze_value_form_players(
        self, players_with_xp: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze value players with good form."""
        analysis = {}

        if (
            "xP_per_price" in players_with_xp.columns
            and "momentum" in players_with_xp.columns
        ):
            value_players = players_with_xp[
                (players_with_xp["momentum"].isin(["🔥", "📈"]))
                & (players_with_xp["price"] <= 7.5)
            ].nlargest(10, "xP_per_price")

            analysis["value_form_players"] = value_players.to_dict("records")
            analysis["value_form_count"] = len(value_players)

        return analysis

    def _analyze_underperformers(self, players_with_xp: pd.DataFrame) -> Dict[str, Any]:
        """Analyze expensive underperforming players."""
        analysis = {}

        if "price" in players_with_xp.columns and "momentum" in players_with_xp.columns:
            expensive_poor = players_with_xp[
                (players_with_xp["price"] >= 8.0)
                & (players_with_xp["momentum"].isin(["❄️", "📉"]))
            ].nsmallest(
                6,
                "form_multiplier"
                if "form_multiplier" in players_with_xp.columns
                else "xP",
            )

            analysis["expensive_underperformers"] = expensive_poor.to_dict("records")
            analysis["underperformer_count"] = len(expensive_poor)

        return analysis

    def _analyze_historical_trends(
        self, players_with_xp: pd.DataFrame, live_data_historical: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze historical performance trends."""
        analysis = {}

        try:
            # Basic historical statistics
            if "total_points" in live_data_historical.columns:
                recent_avg = live_data_historical.groupby("player_id")[
                    "total_points"
                ].mean()
                analysis["historical_avg_points"] = recent_avg.mean()

            # Add more historical analysis as needed
            analysis["historical_gameweeks"] = (
                len(live_data_historical["gameweek"].unique())
                if "gameweek" in live_data_historical.columns
                else 0
            )

        except Exception:
            # If historical analysis fails, return empty analysis
            analysis["historical_analysis_error"] = "Could not process historical data"

        return analysis

    def _generate_historical_insights(
        self, players_with_xp: pd.DataFrame, live_data_historical: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate insights comparing current and historical performance."""
        insights = {}

        try:
            # Compare current XP with historical averages
            if (
                "total_points" in live_data_historical.columns
                and "player_id" in live_data_historical.columns
            ):
                historical_avg = live_data_historical.groupby("player_id")[
                    "total_points"
                ].mean()

                # Merge with current data
                comparison_data = players_with_xp.merge(
                    historical_avg.reset_index().rename(
                        columns={"total_points": "historical_avg"}
                    ),
                    on="player_id",
                    how="left",
                )

                if not comparison_data.empty:
                    insights["players_with_historical_data"] = len(
                        comparison_data.dropna(subset=["historical_avg"])
                    )
                    insights["avg_historical_performance"] = comparison_data[
                        "historical_avg"
                    ].mean()

        except Exception:
            insights["comparison_error"] = "Could not compare with historical data"

        return insights

    def analyze_performance_trends(
        self,
        players_with_xp: pd.DataFrame,
        target_gameweek: int,
        live_data_historical: Optional[pd.DataFrame] = None,
    ) -> Result[Dict[str, Any]]:
        """Analyze performance trends for players.

        Args:
            players_with_xp: DataFrame with current player data and XP calculations
            target_gameweek: Target gameweek for analysis
            live_data_historical: Optional historical performance data

        Returns:
            Result containing comprehensive performance trends analysis
        """
        try:
            if players_with_xp.empty:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No player data available for trends analysis",
                    )
                )

            # Get top performers
            top_performers = players_with_xp.nlargest(15, "xP")

            # Generate performance insights
            insights = {
                "summary": f"Performance trends analysis for GW{target_gameweek}",
                "total_players_analyzed": len(players_with_xp),
                "top_performers_count": len(top_performers),
                "analysis_gameweek": target_gameweek,
            }

            # Add form-based insights if available
            if "momentum" in players_with_xp.columns:
                insights["momentum_summary"] = {
                    "hot_players": len(
                        players_with_xp[players_with_xp["momentum"] == "🔥"]
                    ),
                    "trending_up": len(
                        players_with_xp[players_with_xp["momentum"] == "📈"]
                    ),
                    "stable": len(players_with_xp[players_with_xp["momentum"] == "➡️"]),
                    "trending_down": len(
                        players_with_xp[players_with_xp["momentum"] == "📉"]
                    ),
                    "cold_players": len(
                        players_with_xp[players_with_xp["momentum"] == "❄️"]
                    ),
                }

            # Add historical insights if available
            if live_data_historical is not None and not live_data_historical.empty:
                historical_insights = self._analyze_historical_trends(
                    players_with_xp, live_data_historical
                )
                insights["historical_trends"] = historical_insights

            return Result(
                value={
                    "top_performers": top_performers,
                    "insights": insights,
                    "analysis_period": f"GW{target_gameweek}",
                }
            )

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Performance trends analysis failed: {str(e)}",
                )
            )
