"""Visualization service for FPL charts and analytics displays."""

from typing import Dict, Any, Optional
import pandas as pd

from fpl_team_picker.domain.common.result import Result, DomainError, ErrorType


class VisualizationService:
    """Service for creating FPL visualizations and analytics displays."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the service with configuration.

        Args:
            config: Configuration dictionary for visualization settings
        """
        self.config = config or {}

    def create_expected_points_display(
        self,
        players_with_xp: pd.DataFrame,
        target_gameweek: int,
        display_format: str = "comprehensive",
    ) -> Result[Dict[str, Any]]:
        """Create expected points results display.

        Args:
            players_with_xp: DataFrame with calculated XP data
            target_gameweek: Target gameweek number
            display_format: Format type ('comprehensive', 'summary', 'table_only')

        Returns:
            Result containing display data and metadata
        """
        try:
            if players_with_xp.empty:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No expected points data available",
                    )
                )

            # Validate required columns
            required_columns = {"web_name", "position", "xP"}
            missing_columns = required_columns - set(players_with_xp.columns)
            if missing_columns:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Players data missing required columns: {missing_columns}",
                    )
                )

            # Generate display data
            display_data = self._prepare_xp_display_data(
                players_with_xp, target_gameweek
            )

            # Generate analysis insights
            insights = self._generate_xp_insights(players_with_xp)

            return Result(
                value={
                    "display_data": display_data,
                    "insights": insights,
                    "target_gameweek": target_gameweek,
                    "total_players": len(players_with_xp),
                    "display_format": display_format,
                }
            )

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Expected points display creation failed: {str(e)}",
                )
            )

    def create_form_analytics_display(
        self,
        players_with_xp: pd.DataFrame,
        analysis_type: str = "comprehensive",
    ) -> Result[Dict[str, Any]]:
        """Create form analytics display with hot/cold player detection.

        Args:
            players_with_xp: DataFrame with form and XP data
            analysis_type: Type of analysis ('comprehensive', 'hot_cold', 'value_focus')

        Returns:
            Result containing form analytics data
        """
        try:
            if players_with_xp.empty:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No player data available for form analysis",
                    )
                )

            # Check for form data availability
            form_columns = ["momentum", "form_multiplier"]
            available_form_columns = [
                col for col in form_columns if col in players_with_xp.columns
            ]

            if not available_form_columns:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No form data available - momentum or form_multiplier columns required",
                    )
                )

            # Generate form analytics
            form_analytics = self._analyze_player_form(players_with_xp, analysis_type)

            return Result(
                value={
                    "form_analytics": form_analytics,
                    "analysis_type": analysis_type,
                    "available_form_columns": available_form_columns,
                    "total_players": len(players_with_xp),
                }
            )

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Form analytics creation failed: {str(e)}",
                )
            )

    def create_fixture_difficulty_display(
        self,
        gameweek_data: Dict[str, Any],
        start_gameweek: int,
        num_gameweeks: int = 5,
    ) -> Result[Dict[str, Any]]:
        """Create fixture difficulty visualization.

        Args:
            gameweek_data: Complete gameweek data including fixtures and teams
            start_gameweek: Starting gameweek for analysis
            num_gameweeks: Number of gameweeks to analyze

        Returns:
            Result containing fixture difficulty data
        """
        try:
            # Validate inputs
            if start_gameweek < 1 or start_gameweek > 38:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Invalid start gameweek: {start_gameweek}. Must be between 1 and 38",
                    )
                )

            if num_gameweeks < 1 or num_gameweeks > 10:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Invalid number of gameweeks: {num_gameweeks}. Must be between 1 and 10",
                    )
                )

            # Extract required data
            fixtures = gameweek_data.get("fixtures")
            teams = gameweek_data.get("teams")

            if fixtures is None or fixtures.empty:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No fixtures data available",
                    )
                )

            if teams is None or teams.empty:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No teams data available",
                    )
                )

            # Generate fixture difficulty analysis
            fixture_analysis = self._analyze_fixture_difficulty(
                fixtures, teams, start_gameweek, num_gameweeks
            )

            return Result(
                value={
                    "fixture_analysis": fixture_analysis,
                    "start_gameweek": start_gameweek,
                    "num_gameweeks": num_gameweeks,
                    "end_gameweek": min(start_gameweek + num_gameweeks - 1, 38),
                }
            )

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Fixture difficulty display creation failed: {str(e)}",
                )
            )

    def create_team_strength_display(
        self,
        gameweek_data: Dict[str, Any],
        target_gameweek: int,
    ) -> Result[Dict[str, Any]]:
        """Create team strength visualization.

        Args:
            gameweek_data: Complete gameweek data
            target_gameweek: Target gameweek for strength analysis

        Returns:
            Result containing team strength data
        """
        try:
            # Extract teams data
            teams = gameweek_data.get("teams")
            if teams is None or teams.empty:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No teams data available",
                    )
                )

            # Generate team strength analysis
            team_strength_data = self._analyze_team_strength(teams, target_gameweek)

            return Result(
                value={
                    "team_strength_data": team_strength_data,
                    "target_gameweek": target_gameweek,
                    "total_teams": len(teams),
                }
            )

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Team strength display creation failed: {str(e)}",
                )
            )

    def create_player_trends_display(
        self,
        players_with_xp: pd.DataFrame,
        trend_type: str = "performance",
        top_n: int = 20,
    ) -> Result[Dict[str, Any]]:
        """Create player trends visualization.

        Args:
            players_with_xp: DataFrame with player performance data
            trend_type: Type of trend analysis ('performance', 'value', 'form')
            top_n: Number of top players to include

        Returns:
            Result containing player trends data
        """
        try:
            if players_with_xp.empty:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No player data available for trends analysis",
                    )
                )

            # Validate trend type
            valid_trend_types = ["performance", "value", "form"]
            if trend_type not in valid_trend_types:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Invalid trend type: {trend_type}. Must be one of {valid_trend_types}",
                    )
                )

            # Generate trends analysis
            trends_data = self._analyze_player_trends(
                players_with_xp, trend_type, top_n
            )

            return Result(
                value={
                    "trends_data": trends_data,
                    "trend_type": trend_type,
                    "top_n": top_n,
                    "total_players": len(players_with_xp),
                }
            )

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Player trends display creation failed: {str(e)}",
                )
            )

    def _prepare_xp_display_data(
        self, players_with_xp: pd.DataFrame, target_gameweek: int
    ) -> Dict[str, Any]:
        """Prepare expected points display data."""
        # Core display columns
        display_columns = [
            "web_name",
            "position",
            "name",
            "price",
            "selected_by_percent",
        ]

        # XP columns
        xp_columns = ["xP", "expected_minutes", "fixture_difficulty"]
        if "xP_5gw" in players_with_xp.columns:
            xp_columns.extend(["xP_5gw", "xP_per_price_5gw", "xP_horizon_advantage"])

        # Add available columns
        available_columns = [
            col
            for col in display_columns + xp_columns
            if col in players_with_xp.columns
        ]

        # Create display dataframe
        from fpl_team_picker.utils.helpers import create_display_dataframe

        sort_column = "xP_5gw" if "xP_5gw" in players_with_xp.columns else "xP"
        display_df = create_display_dataframe(
            players_with_xp,
            core_columns=available_columns,
            sort_by=sort_column,
            ascending=False,
            round_decimals=3,
        )

        return {
            "display_dataframe": display_df,
            "sort_column": sort_column,
            "available_columns": available_columns,
        }

    def _generate_xp_insights(self, players_with_xp: pd.DataFrame) -> Dict[str, Any]:
        """Generate insights from expected points data."""
        insights = {}

        # Basic statistics
        insights["avg_1gw_xp"] = players_with_xp["xP"].mean()
        insights["max_1gw_xp"] = players_with_xp["xP"].max()
        insights["top_1gw_player"] = players_with_xp.loc[
            players_with_xp["xP"].idxmax(), "web_name"
        ]

        # 5GW statistics if available
        if "xP_5gw" in players_with_xp.columns:
            insights["avg_5gw_xp"] = players_with_xp["xP_5gw"].mean()
            insights["max_5gw_xp"] = players_with_xp["xP_5gw"].max()
            insights["top_5gw_player"] = players_with_xp.loc[
                players_with_xp["xP_5gw"].idxmax(), "web_name"
            ]

        # Value insights
        if "xP_per_price" in players_with_xp.columns:
            insights["best_value_player"] = players_with_xp.loc[
                players_with_xp["xP_per_price"].idxmax(), "web_name"
            ]
            insights["max_value_ratio"] = players_with_xp["xP_per_price"].max()

        return insights

    def _analyze_player_form(
        self, players_with_xp: pd.DataFrame, analysis_type: str
    ) -> Dict[str, Any]:
        """Analyze player form data."""
        form_analysis = {}

        # Hot players (if momentum column exists)
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

            form_analysis["hot_players"] = hot_players.to_dict("records")
            form_analysis["cold_players"] = cold_players.to_dict("records")

        # Value players with good form
        if (
            "xP_per_price" in players_with_xp.columns
            and "momentum" in players_with_xp.columns
        ):
            value_players = players_with_xp[
                (players_with_xp["momentum"].isin(["🔥", "📈"]))
                & (players_with_xp["price"] <= 7.5)
            ].nlargest(10, "xP_per_price")

            form_analysis["value_form_players"] = value_players.to_dict("records")

        return form_analysis

    def _analyze_fixture_difficulty(
        self, fixtures: pd.DataFrame, teams: pd.DataFrame, start_gw: int, num_gws: int
    ) -> Dict[str, Any]:
        """Analyze fixture difficulty for teams."""
        # This is a simplified implementation
        # In a full implementation, you'd use team strength data
        fixture_analysis = {
            "analysis_period": f"GW{start_gw}-{start_gw + num_gws - 1}",
            "total_fixtures": len(
                fixtures[
                    (fixtures["event"] >= start_gw)
                    & (fixtures["event"] < start_gw + num_gws)
                ]
            ),
            "teams_analyzed": len(teams),
        }

        return fixture_analysis

    def _analyze_team_strength(
        self, teams: pd.DataFrame, target_gameweek: int
    ) -> Dict[str, Any]:
        """Analyze team strength data."""
        # Simplified team strength analysis
        team_strength_data = {
            "target_gameweek": target_gameweek,
            "total_teams": len(teams),
            "teams": teams.to_dict("records") if not teams.empty else [],
        }

        return team_strength_data

    def _analyze_player_trends(
        self, players_with_xp: pd.DataFrame, trend_type: str, top_n: int
    ) -> Dict[str, Any]:
        """Analyze player trends based on type."""
        trends_data = {"trend_type": trend_type}

        if trend_type == "performance":
            # Sort by expected points
            top_performers = players_with_xp.nlargest(top_n, "xP")
            trends_data["top_performers"] = top_performers.to_dict("records")

        elif trend_type == "value" and "xP_per_price" in players_with_xp.columns:
            # Sort by value (xP per price)
            top_value = players_with_xp.nlargest(top_n, "xP_per_price")
            trends_data["top_value"] = top_value.to_dict("records")

        elif trend_type == "form" and "momentum" in players_with_xp.columns:
            # Focus on form trends
            trending_up = players_with_xp[
                players_with_xp["momentum"].isin(["🔥", "📈"])
            ].nlargest(top_n, "xP")
            trends_data["trending_up"] = trending_up.to_dict("records")

        return trends_data
