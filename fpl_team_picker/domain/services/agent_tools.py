"""Agent tools - thin wrappers around domain services for LLM agent use.

These tools provide a simplified, JSON-serializable interface for the
transfer planning agent, wrapping existing domain services.
"""

import logging
from typing import Any, Dict

import pandas as pd
from pydantic_ai import RunContext

from fpl_team_picker.config import config
from fpl_team_picker.domain.services.ml_expected_points_service import (
    MLExpectedPointsService,
)

logger = logging.getLogger(__name__)


# Type alias for agent dependencies (data passed to agent at runtime)
class AgentDeps:
    """Dependencies injected into agent tools."""

    def __init__(
        self,
        players_data: pd.DataFrame,
        teams_data: pd.DataFrame,
        fixtures_data: pd.DataFrame,
        live_data: pd.DataFrame,
        ownership_trends: pd.DataFrame | None = None,
        value_analysis: pd.DataFrame | None = None,
        fixture_difficulty: pd.DataFrame | None = None,
        betting_features: pd.DataFrame | None = None,
        player_metrics: pd.DataFrame | None = None,
        player_availability: pd.DataFrame | None = None,
        team_form: pd.DataFrame | None = None,
        players_enhanced: pd.DataFrame | None = None,
        xg_rates: pd.DataFrame | None = None,
    ):
        self.players_data = players_data
        self.teams_data = teams_data
        self.fixtures_data = fixtures_data
        self.live_data = live_data
        self.ownership_trends = ownership_trends
        self.value_analysis = value_analysis
        self.fixture_difficulty = fixture_difficulty
        self.betting_features = betting_features
        self.player_metrics = player_metrics
        self.player_availability = player_availability
        self.team_form = team_form
        self.players_enhanced = players_enhanced
        self.xg_rates = xg_rates


def get_multi_gw_xp_predictions(
    ctx: RunContext[AgentDeps], start_gameweek: int, num_gameweeks: int
) -> Dict[str, Any]:
    """
    Get per-gameweek xP predictions for all players over multiple gameweeks.

    Args:
        ctx: Agent context with data dependencies
        start_gameweek: Starting gameweek number
        num_gameweeks: Number of gameweeks to predict (2-5)

    Returns:
        Dictionary with:
        - players: List of player dicts with per-GW xP (xP_gw1, xP_gw2, etc.)
        - summary: Aggregate statistics
        - metadata: Prediction metadata
    """
    logger.info(f"🔮 Predicting xP for GW{start_gameweek}+{num_gameweeks}")

    deps = ctx.deps

    try:
        # Initialize ML service with configured model
        ml_service = MLExpectedPointsService(
            model_path=config.xp_model.ml_model_path,
            ensemble_rule_weight=config.xp_model.ml_ensemble_rule_weight,
            debug=config.xp_model.debug,
        )

        # Calculate multi-GW predictions
        if num_gameweeks == 3:
            predictions_df = ml_service.calculate_3gw_expected_points(
                players_data=deps.players_data,
                teams_data=deps.teams_data,
                xg_rates_data=deps.xg_rates or pd.DataFrame(),
                fixtures_data=deps.fixtures_data,
                target_gameweek=start_gameweek,
                live_data=deps.live_data,
                ownership_trends_df=deps.ownership_trends,
                value_analysis_df=deps.value_analysis,
                fixture_difficulty_df=deps.fixture_difficulty,
                betting_features_df=deps.betting_features,
                player_metrics_df=deps.player_metrics,
                player_availability_df=deps.player_availability,
                team_form_df=deps.team_form,
                players_enhanced_df=deps.players_enhanced,
            )
        elif num_gameweeks == 5:
            predictions_df = ml_service.calculate_5gw_expected_points(
                players_data=deps.players_data,
                teams_data=deps.teams_data,
                xg_rates_data=deps.xg_rates or pd.DataFrame(),
                fixtures_data=deps.fixtures_data,
                target_gameweek=start_gameweek,
                live_data=deps.live_data,
                ownership_trends_df=deps.ownership_trends,
                value_analysis_df=deps.value_analysis,
                fixture_difficulty_df=deps.fixture_difficulty,
                betting_features_df=deps.betting_features,
                player_metrics_df=deps.player_metrics,
                player_availability_df=deps.player_availability,
                team_form_df=deps.team_form,
                players_enhanced_df=deps.players_enhanced,
            )
        else:
            raise ValueError(f"Unsupported horizon: {num_gameweeks} (must be 3 or 5)")

        # Format for agent consumption - select relevant columns
        relevant_cols = [
            "player_id",
            "web_name",
            "position",
            "team_name",
            "price",
            "selected_by_percent",
            "ml_xP",  # Total xP over horizon
        ]

        # Add per-GW columns
        for i in range(1, num_gameweeks + 1):
            gw_col = f"xP_gw{i}"
            if gw_col in predictions_df.columns:
                relevant_cols.append(gw_col)

        # Add uncertainty if available
        if "xP_uncertainty" in predictions_df.columns:
            relevant_cols.append("xP_uncertainty")

        # Filter to available columns only
        available_cols = [c for c in relevant_cols if c in predictions_df.columns]
        result_df = predictions_df[available_cols].copy()

        # Sort by total xP
        result_df = result_df.sort_values("ml_xP", ascending=False)

        # Convert to list of dicts for JSON serialization
        players_list = result_df.to_dict("records")

        # Calculate summary statistics
        summary = {
            "total_players": len(players_list),
            "avg_xp_total": float(result_df["ml_xP"].mean()),
            "max_xp_total": float(result_df["ml_xP"].max()),
            "min_xp_total": float(result_df["ml_xP"].min()),
        }

        # Add per-GW averages
        for i in range(1, num_gameweeks + 1):
            gw_col = f"xP_gw{i}"
            if gw_col in result_df.columns:
                summary[f"avg_xp_gw{i}"] = float(result_df[gw_col].mean())

        logger.info(
            f"✅ Predictions complete: {len(players_list)} players, "
            f"avg xP={summary['avg_xp_total']:.2f}"
        )

        return {
            "players": players_list[:100],  # Top 100 to limit token usage
            "summary": summary,
            "metadata": {
                "start_gameweek": start_gameweek,
                "num_gameweeks": num_gameweeks,
                "model_path": config.xp_model.ml_model_path,
            },
        }

    except Exception as e:
        logger.error(f"❌ Error in get_multi_gw_xp_predictions: {e}")
        raise


# Tool metadata for pydantic_ai
get_multi_gw_xp_predictions.__doc__ = """
Get expected points predictions for multiple future gameweeks.

Use this tool to see how players are projected to perform over the next 2-5 gameweeks.
Returns per-gameweek breakdowns (xP_gw1, xP_gw2, etc.) and total xP across the horizon.

Example usage:
- To see 3-gameweek projections: get_multi_gw_xp_predictions(start_gameweek=17, num_gameweeks=3)
- To see 5-gameweek projections: get_multi_gw_xp_predictions(start_gameweek=17, num_gameweeks=5)

The results include:
- Top 100 players sorted by total xP
- Per-gameweek xP breakdowns
- Uncertainty estimates (if available)
- Team, position, price context
"""


# Registry of all available tools
AGENT_TOOLS = [
    get_multi_gw_xp_predictions,
]
