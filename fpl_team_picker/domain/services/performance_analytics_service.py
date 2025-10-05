"""Performance analytics service for FPL form analysis and trend detection."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class AlgorithmVersion(BaseModel):
    """Configuration for a specific xP algorithm version.

    This enables retroactive testing of different algorithm parameters
    on historical data to optimize model accuracy.
    """

    name: str = Field(description="Algorithm version identifier")
    form_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight for recent form (0-1)"
    )
    form_window: int = Field(
        default=5, ge=1, le=10, description="Number of gameweeks for form calculation"
    )
    use_team_strength: bool = Field(
        default=True, description="Whether to use dynamic team strength"
    )
    team_strength_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "historical_transition_gw": 8,
            "rolling_window": 6,
        },
        description="Team strength calculation parameters",
    )
    minutes_model_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "use_sbp": True,
            "use_availability": True,
            "use_form_adjustment": True,
        },
        description="Minutes prediction model parameters",
    )
    statistical_estimation_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "use_position_averages": True,
            "use_team_context": True,
        },
        description="xG/xA statistical estimation parameters",
    )

    class Config:
        frozen = True  # Make immutable


# Algorithm version registry - add new versions here for testing
ALGORITHM_VERSIONS: Dict[str, AlgorithmVersion] = {
    "v1.0": AlgorithmVersion(
        name="v1.0",
        form_weight=0.7,
        form_window=5,
        use_team_strength=True,
        team_strength_params={
            "historical_transition_gw": 8,
            "rolling_window": 6,
        },
    ),
    "current": AlgorithmVersion(
        name="current",
        form_weight=0.7,
        form_window=5,
        use_team_strength=True,
    ),
    "experimental_high_form": AlgorithmVersion(
        name="experimental_high_form",
        form_weight=0.9,
        form_window=3,
        use_team_strength=True,
    ),
    "experimental_low_form": AlgorithmVersion(
        name="experimental_low_form",
        form_weight=0.5,
        form_window=8,
        use_team_strength=True,
    ),
}


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
    ) -> Dict[str, Any]:
        """Analyze player form and momentum indicators.

        Args:
            players_with_xp: DataFrame with current player data and XP calculations - guaranteed clean
            live_data_historical: Optional historical performance data

        Returns:
            Comprehensive form analysis
        """
        # Check for required form columns
        form_columns = ["momentum", "form_multiplier"]
        available_form_columns = [
            col for col in form_columns if col in players_with_xp.columns
        ]

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

        return {
            "form_analysis": form_analysis,
            "total_players_analyzed": len(players_with_xp),
            "available_form_columns": available_form_columns,
            "has_historical_data": live_data_historical is not None
            and not live_data_historical.empty,
        }

    def detect_breakout_players(
        self,
        players_with_xp: pd.DataFrame,
        price_threshold: float = 7.0,
        xp_threshold: float = 6.0,
    ) -> List[Dict[str, Any]]:
        """Detect potential breakout players based on form and value.

        Args:
            players_with_xp: DataFrame with player data - guaranteed clean
            price_threshold: Maximum price for breakout consideration
            xp_threshold: Minimum expected points threshold

        Returns:
            List of potential breakout players
        """
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
            breakout_candidates = breakout_candidates.sort_values("xP", ascending=False)

        # Convert to list of dictionaries
        return breakout_candidates.head(10).to_dict("records")

    def analyze_position_trends(
        self,
        players_with_xp: pd.DataFrame,
        position: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze performance trends by position.

        Args:
            players_with_xp: DataFrame with player data - guaranteed clean
            position: Specific position to analyze (None for all positions)

        Returns:
            Position-specific trend analysis
        """
        valid_positions = ["GKP", "DEF", "MID", "FWD"]
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
                "top_player": pos_players.loc[pos_players["xP"].idxmax(), "web_name"],
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

        return {
            "position_analysis": position_analysis,
            "analyzed_positions": positions_to_analyze,
            "total_players": len(players_with_xp),
        }

    def get_statistical_insights(
        self,
        players_with_xp: pd.DataFrame,
        live_data_historical: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Generate statistical insights from player performance data.

        Args:
            players_with_xp: DataFrame with current player data - guaranteed clean
            live_data_historical: Optional historical performance data

        Returns:
            Statistical insights and metrics
        """
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

        return insights

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

    def recompute_historical_xp(
        self,
        target_gameweek: int,
        algorithm_version: str = "current",
        include_snapshots: bool = True,
    ) -> pd.DataFrame:
        """Recompute historical xP for a specific gameweek using specified algorithm.

        This enables retroactive testing of algorithm variants on historical data
        to optimize model accuracy and validate improvements.

        Args:
            target_gameweek: Which gameweek to recompute (1-38)
            algorithm_version: Algorithm version to use (key from ALGORITHM_VERSIONS)
            include_snapshots: Whether to use availability snapshots (if available)

        Returns:
            DataFrame with recomputed xP predictions using only data available at that time
            Columns: player_id, web_name, position, xP, xP_5gw, algorithm_version, computed_at

        Raises:
            ValueError: If algorithm_version not found or target_gameweek invalid
        """
        from fpl_team_picker.domain.services.data_orchestration_service import (
            DataOrchestrationService,
        )
        from fpl_team_picker.domain.services.expected_points_service import (
            ExpectedPointsService,
        )

        # Validate algorithm version
        if algorithm_version not in ALGORITHM_VERSIONS:
            available = list(ALGORITHM_VERSIONS.keys())
            raise ValueError(
                f"Unknown algorithm version: {algorithm_version}. Available: {available}"
            )

        algo_config = ALGORITHM_VERSIONS[algorithm_version]

        # Load historical gameweek state
        orchestration_service = DataOrchestrationService()
        historical_data = orchestration_service.load_historical_gameweek_state(
            target_gameweek=target_gameweek,
            form_window=algo_config.form_window,
            include_snapshots=include_snapshots,
        )

        # Configure xP service with algorithm parameters
        # Note: Current ExpectedPointsService uses global config, but we can override
        # by creating a modified config dict
        algo_params = {
            "form_weight": algo_config.form_weight,
            "form_window": algo_config.form_window,
            "use_team_strength": algo_config.use_team_strength,
            "team_strength_params": algo_config.team_strength_params,
            "minutes_model_params": algo_config.minutes_model_params,
            "statistical_estimation_params": algo_config.statistical_estimation_params,
        }

        xp_service = ExpectedPointsService(config=algo_params)

        # Calculate xP using historical data
        players_with_xp = xp_service.calculate_combined_results(
            historical_data, use_ml_model=False
        )

        # Add metadata
        players_with_xp["algorithm_version"] = algorithm_version
        players_with_xp["computed_at"] = pd.Timestamp.now()
        players_with_xp["target_gameweek"] = target_gameweek

        return players_with_xp

    def batch_recompute_season(
        self,
        start_gw: int,
        end_gw: int,
        algorithm_versions: Optional[List[str]] = None,
        include_snapshots: bool = True,
    ) -> pd.DataFrame:
        """Recompute xP across multiple gameweeks and algorithm versions.

        This enables comprehensive algorithm comparison and optimization across
        historical data to identify best-performing parameter sets.

        Args:
            start_gw: Starting gameweek (inclusive, 1-38)
            end_gw: Ending gameweek (inclusive, 1-38)
            algorithm_versions: List of algorithm versions to test (None = all registered)
            include_snapshots: Whether to use availability snapshots (if available)

        Returns:
            Multi-index DataFrame with columns:
            (gameweek, player_id, algorithm_version) → xP, xP_5gw, web_name, position

        Raises:
            ValueError: If gameweek range invalid
        """
        if start_gw < 1 or end_gw > 38 or start_gw > end_gw:
            raise ValueError(
                f"Invalid gameweek range: {start_gw}-{end_gw}. Must be 1-38 and start <= end."
            )

        # Use all versions if not specified
        if algorithm_versions is None:
            algorithm_versions = list(ALGORITHM_VERSIONS.keys())

        # Validate all algorithm versions exist
        invalid_versions = [
            v for v in algorithm_versions if v not in ALGORITHM_VERSIONS
        ]
        if invalid_versions:
            raise ValueError(f"Unknown algorithm versions: {invalid_versions}")

        # Batch recomputation
        all_results = []
        for gw in range(start_gw, end_gw + 1):
            for algo_version in algorithm_versions:
                try:
                    gw_results = self.recompute_historical_xp(
                        target_gameweek=gw,
                        algorithm_version=algo_version,
                        include_snapshots=include_snapshots,
                    )
                    all_results.append(gw_results)
                except Exception as e:
                    # Log error but continue with other gameweeks/algorithms
                    print(f"⚠️ Failed to recompute GW{gw} with {algo_version}: {e}")
                    continue

        if not all_results:
            raise ValueError(
                f"No results computed for GW{start_gw}-{end_gw} with algorithms {algorithm_versions}"
            )

        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)

        # Create multi-index for efficient lookups
        combined_df = combined_df.set_index(
            ["target_gameweek", "player_id", "algorithm_version"]
        ).sort_index()

        return combined_df

    def calculate_accuracy_metrics(
        self,
        predictions_df: pd.DataFrame,
        actual_results_df: pd.DataFrame,
        by_position: bool = True,
    ) -> Dict[str, Any]:
        """Calculate accuracy metrics comparing predicted vs actual points.

        Args:
            predictions_df: DataFrame with predicted xP (requires: player_id, xP)
            actual_results_df: DataFrame with actual points (requires: player_id, total_points)
            by_position: Whether to calculate position-specific metrics

        Returns:
            Dictionary with accuracy metrics:
            - overall: MAE, RMSE, correlation
            - by_position: Position-specific metrics (if by_position=True)
            - player_count: Number of players analyzed
        """
        # Merge predictions with actual results
        comparison_df = pd.merge(
            predictions_df[["player_id", "xP", "position", "web_name"]],
            actual_results_df[["player_id", "total_points"]],
            on="player_id",
            how="inner",
        )

        if comparison_df.empty:
            return {
                "error": "No matching players between predictions and actual results",
                "player_count": 0,
            }

        # Calculate error metrics
        errors = comparison_df["total_points"] - comparison_df["xP"]
        abs_errors = errors.abs()

        # Overall metrics
        mae = abs_errors.mean()
        rmse = np.sqrt((errors**2).mean())
        correlation = comparison_df["xP"].corr(comparison_df["total_points"])

        metrics = {
            "overall": {
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "correlation": round(correlation, 3),
                "mean_predicted": round(comparison_df["xP"].mean(), 2),
                "mean_actual": round(comparison_df["total_points"].mean(), 2),
            },
            "player_count": len(comparison_df),
        }

        # Position-specific metrics
        if by_position and "position" in comparison_df.columns:
            position_metrics = {}
            for position in ["GKP", "DEF", "MID", "FWD"]:
                pos_df = comparison_df[comparison_df["position"] == position]
                if not pos_df.empty:
                    pos_errors = pos_df["total_points"] - pos_df["xP"]
                    pos_abs_errors = pos_errors.abs()

                    position_metrics[position] = {
                        "mae": round(pos_abs_errors.mean(), 2),
                        "rmse": round(np.sqrt((pos_errors**2).mean()), 2),
                        "correlation": round(
                            pos_df["xP"].corr(pos_df["total_points"]), 3
                        ),
                        "player_count": len(pos_df),
                    }

            metrics["by_position"] = position_metrics

        return metrics

    def analyze_performance_trends(
        self,
        players_with_xp: pd.DataFrame,
        target_gameweek: int,
        live_data_historical: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Analyze performance trends for players.

        Args:
            players_with_xp: DataFrame with current player data and XP calculations - guaranteed clean
            target_gameweek: Target gameweek for analysis
            live_data_historical: Optional historical performance data

        Returns:
            Comprehensive performance trends analysis
        """
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

        return {
            "top_performers": top_performers,
            "insights": insights,
            "analysis_period": f"GW{target_gameweek}",
        }

    def analyze_squad_performance(
        self,
        previous_predictions: pd.DataFrame,
        actual_results: pd.DataFrame,
        squad_picks: List[Dict[str, Any]],
        previous_gameweek: int,
    ) -> Dict[str, Any]:
        """Analyze squad performance comparing predicted vs actual results.

        Args:
            previous_predictions: DataFrame with xP predictions - guaranteed clean
            actual_results: DataFrame with actual gameweek results - guaranteed clean
            squad_picks: List of squad player picks with element IDs
            previous_gameweek: Gameweek number for analysis

        Returns:
            Squad performance analysis with xP vs actual comparison
        """
        import pandas as pd

        try:
            # Get squad player IDs
            squad_player_ids = [pick["element"] for pick in squad_picks]

            # Merge predictions with actual results
            comparison_df = pd.merge(
                previous_predictions[["player_id", "web_name", "position", "xP"]],
                actual_results[
                    ["player_id", "total_points", "minutes", "goals_scored", "assists"]
                ],
                on="player_id",
                how="inner",
            )

            # Filter for squad players only
            squad_comparison = comparison_df[
                comparison_df["player_id"].isin(squad_player_ids)
            ].copy()

            if squad_comparison.empty:
                return {
                    "error": "No matching players found between predictions and actual results",
                    "gameweek": previous_gameweek,
                    "squad_size": len(squad_player_ids),
                }

            # Calculate performance metrics
            squad_comparison["xP_diff"] = (
                squad_comparison["total_points"] - squad_comparison["xP"]
            )
            squad_comparison["accuracy_pct"] = (
                1
                - abs(squad_comparison["xP_diff"])
                / squad_comparison["xP"].clip(lower=0.1)
            ) * 100

            # Summary statistics
            total_predicted = squad_comparison["xP"].sum()
            total_actual = squad_comparison["total_points"].sum()
            total_difference = total_actual - total_predicted
            avg_accuracy = squad_comparison["accuracy_pct"].mean()

            # Best and worst performers
            best_performer = squad_comparison.loc[squad_comparison["xP_diff"].idxmax()]
            worst_performer = squad_comparison.loc[squad_comparison["xP_diff"].idxmin()]

            return {
                "gameweek": previous_gameweek,
                "squad_analysis": {
                    "total_predicted": round(total_predicted, 1),
                    "total_actual": int(total_actual),
                    "difference": round(total_difference, 1),
                    "accuracy_percentage": round(avg_accuracy, 1),
                    "players_analyzed": len(squad_comparison),
                },
                "individual_performance": squad_comparison[
                    ["web_name", "position", "xP", "total_points", "xP_diff"]
                ]
                .sort_values("xP_diff", ascending=False)
                .to_dict("records"),
                "best_performer": {
                    "name": best_performer["web_name"],
                    "position": best_performer["position"],
                    "predicted": round(best_performer["xP"], 1),
                    "actual": int(best_performer["total_points"]),
                    "difference": round(best_performer["xP_diff"], 1),
                },
                "worst_performer": {
                    "name": worst_performer["web_name"],
                    "position": worst_performer["position"],
                    "predicted": round(worst_performer["xP"], 1),
                    "actual": int(worst_performer["total_points"]),
                    "difference": round(worst_performer["xP_diff"], 1),
                },
            }

        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "gameweek": previous_gameweek,
            }
