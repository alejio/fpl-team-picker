"""Agent tools - thin wrappers around domain services for LLM agent use.

These tools provide a simplified, JSON-serializable interface for the
transfer planning agent, wrapping existing domain services.

Design follows Anthropic's "Agent-Computer Interface (ACI)" principles:
- Comprehensive docstrings with examples, edge cases, performance hints
- Poka-yoke (error prevention) via structured arguments
- Clear boundaries and token limits
- JSON-serializable outputs only (no pandas DataFrames)
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

    Use this tool to see how players are projected to perform over the next 1, 3, or 5 gameweeks.
    Returns per-gameweek breakdowns (xP_gw1, xP_gw2, etc.) and total xP across the horizon.

    Args:
        ctx: Agent context with data dependencies
        start_gameweek: Starting gameweek number (1-38)
        num_gameweeks: Number of gameweeks to predict (1, 3, or 5)

    Returns:
        {
            "players": [  # Top 100 players sorted by total xP
                {
                    "player_id": int,
                    "web_name": str,
                    "position": str,
                    "team_name": str,
                    "price": float,
                    "selected_by_percent": float,
                    "ml_xP": float,  # Total xP over horizon
                    "xP_gw1": float,  # Per-GW breakdowns
                    "xP_gw2": float,  # (if num_gameweeks >= 2)
                    "xP_gw3": float,  # (if num_gameweeks >= 3)
                    "xP_uncertainty": float  # Prediction uncertainty (if available)
                }
            ],
            "summary": {
                "total_players": int,
                "avg_xp_total": float,
                "max_xp_total": float,
                "min_xp_total": float,
                "avg_xp_gw1": float,  # Per-GW averages
                "avg_xp_gw2": float,  # (if applicable)
                "avg_xp_gw3": float
            },
            "metadata": {
                "start_gameweek": int,
                "num_gameweeks": int,
                "model_path": str
            }
        }

    Example usage:
        # Get 1-gameweek projections
        xp_1gw = get_multi_gw_xp_predictions(start_gameweek=18, num_gameweeks=1)
        top_player = xp_1gw["players"][0]  # Highest xP player
        print(f"{top_player['web_name']}: {top_player['xP_gw1']} xP")

        # Get 3-gameweek projections for strategic planning
        xp_3gw = get_multi_gw_xp_predictions(start_gameweek=18, num_gameweeks=3)
        for player in xp_3gw["players"][:10]:
            print(f"{player['web_name']}: GW18={player['xP_gw1']}, GW19={player['xP_gw2']}, GW20={player['xP_gw3']}")

    Edge cases:
        - Returns top 100 players only (sorted by ml_xP) to control token usage
        - num_gameweeks must be 1, 3, or 5 (validation enforced)
        - xP_uncertainty only present for ensemble models (RandomForest, XGBoost, etc.)
        - Players with no data return 0.0 xP

    Performance:
        - 1 GW: ~500ms
        - 3 GW: ~1.5s
        - 5 GW: ~2.5s
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
        if num_gameweeks == 1:
            predictions_df = ml_service.calculate_expected_points(
                players_data=deps.players_data,
                teams_data=deps.teams_data,
                xg_rates_data=deps.xg_rates
                if deps.xg_rates is not None
                else pd.DataFrame(),
                fixtures_data=deps.fixtures_data,
                target_gameweek=start_gameweek,
                live_data=deps.live_data,
                gameweeks_ahead=1,
                ownership_trends_df=deps.ownership_trends,
                value_analysis_df=deps.value_analysis,
                fixture_difficulty_df=deps.fixture_difficulty,
                betting_features_df=deps.betting_features,
                derived_player_metrics_df=deps.player_metrics,
                player_availability_snapshot_df=deps.player_availability,
                derived_team_form_df=deps.team_form,
                players_enhanced_df=deps.players_enhanced,
            )
            # Rename ml_xP to match expected format and add xP_gw1 column
            if "ml_xP" in predictions_df.columns:
                predictions_df["xP_gw1"] = predictions_df["ml_xP"]
        elif num_gameweeks == 3:
            predictions_df = ml_service.calculate_3gw_expected_points(
                players_data=deps.players_data,
                teams_data=deps.teams_data,
                xg_rates_data=deps.xg_rates
                if deps.xg_rates is not None
                else pd.DataFrame(),
                fixtures_data=deps.fixtures_data,
                target_gameweek=start_gameweek,
                live_data=deps.live_data,
                ownership_trends_df=deps.ownership_trends,
                value_analysis_df=deps.value_analysis,
                fixture_difficulty_df=deps.fixture_difficulty,
                betting_features_df=deps.betting_features,
                derived_player_metrics_df=deps.player_metrics,
                player_availability_snapshot_df=deps.player_availability,
                derived_team_form_df=deps.team_form,
                players_enhanced_df=deps.players_enhanced,
            )
        elif num_gameweeks == 5:
            predictions_df = ml_service.calculate_5gw_expected_points(
                players_data=deps.players_data,
                teams_data=deps.teams_data,
                xg_rates_data=deps.xg_rates
                if deps.xg_rates is not None
                else pd.DataFrame(),
                fixtures_data=deps.fixtures_data,
                target_gameweek=start_gameweek,
                live_data=deps.live_data,
                ownership_trends_df=deps.ownership_trends,
                value_analysis_df=deps.value_analysis,
                fixture_difficulty_df=deps.fixture_difficulty,
                betting_features_df=deps.betting_features,
                derived_player_metrics_df=deps.player_metrics,
                player_availability_snapshot_df=deps.player_availability,
                derived_team_form_df=deps.team_form,
                players_enhanced_df=deps.players_enhanced,
            )
        else:
            raise ValueError(
                f"Unsupported horizon: {num_gameweeks} (must be 1, 3, or 5)"
            )

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


def analyze_fixture_context(
    ctx: RunContext[AgentDeps],
    start_gameweek: int,
    num_gameweeks: int = 3,
) -> Dict[str, Any]:
    """
    Analyze fixture context for next N gameweeks: DGWs, BGWs, and difficulty runs.

    This tool combines DGW/BGW detection with fixture difficulty analysis to provide
    complete strategic context for transfer planning. Use this to identify teams with
    favorable/unfavorable fixture runs and double gameweek opportunities.

    Args:
        ctx: Agent context with fixture data
        start_gameweek: Starting gameweek (e.g., 18)
        num_gameweeks: Lookahead window (default: 3)

    Returns:
        {
            "dgw_bgw_calendar": {
                "dgw_teams": {  # Gameweek -> team names with DGW
                    17: ["Arsenal", "Chelsea"]
                },
                "bgw_teams": {  # Gameweek -> team names with BGW
                    19: ["Liverpool"]
                },
                "has_dgw": bool,  # Whether any DGW detected
                "has_bgw": bool,  # Whether any BGW detected
                "next_dgw_gameweek": int | None,  # Next DGW gameweek number
                "next_bgw_gameweek": int | None
            },
            "fixture_runs": {
                "easy_runs": [  # Top 5 teams with easiest fixtures
                    {
                        "team": "Arsenal",
                        "team_id": 1,
                        "avg_difficulty": 1.8,
                        "fixtures": ["Ipswich(H)", "Wolves(A)", "Burnley(H)"]
                    }
                ],
                "hard_runs": [  # Top 5 teams with hardest fixtures
                    {
                        "team": "Newcastle",
                        "team_id": 4,
                        "avg_difficulty": 4.2,
                        "fixtures": ["Man City(A)", "Liverpool(H)", "Arsenal(A)"]
                    }
                ],
                "fixture_swings": [  # Teams with significant difficulty changes
                    {
                        "team": "Chelsea",
                        "team_id": 3,
                        "swing_type": "hard_to_easy",  # or "easy_to_hard"
                        "current_difficulty": 4.0,  # Current GW
                        "upcoming_difficulty": 1.9,  # Next N-1 GW average
                        "swing_magnitude": 2.1  # Absolute difference
                    }
                ]
            }
        }

    Example usage:
        # Check for DGWs and fixture runs for next 3 gameweeks
        context = analyze_fixture_context(start_gameweek=18, num_gameweeks=3)

        # Identify DGW opportunities
        if context["dgw_bgw_calendar"]["has_dgw"]:
            dgw_gw = context["dgw_bgw_calendar"]["next_dgw_gameweek"]
            dgw_teams = context["dgw_bgw_calendar"]["dgw_teams"][dgw_gw]
            print(f"DGW{dgw_gw} teams: {dgw_teams}")

        # Find teams with easy fixture runs
        for run in context["fixture_runs"]["easy_runs"]:
            print(f"{run['team']}: {run['avg_difficulty']:.1f} difficulty")

        # Identify fixture swings for transfer timing
        for swing in context["fixture_runs"]["fixture_swings"]:
            if swing["swing_type"] == "hard_to_easy":
                print(f"Transfer IN {swing['team']} players (fixtures improving)")

    Edge cases:
        - Returns empty lists if no DGWs/BGWs detected
        - Fixture swings require >1.5 difficulty change to trigger
        - Limited to top 5 teams per category (easy/hard runs) to control token usage
        - If fewer than 5 teams qualify, returns all qualifying teams
        - Handles cases where fixture_difficulty data is missing (returns basic counts only)

    Performance:
        - Typical execution: ~200-500ms
        - Depends on number of fixtures and teams in horizon
    """
    logger.info(f"🔍 Analyzing fixture context for GW{start_gameweek}+{num_gameweeks}")

    deps = ctx.deps
    end_gameweek = start_gameweek + num_gameweeks - 1

    try:
        # Filter fixtures for the horizon
        fixtures_in_horizon = deps.fixtures_data[
            (deps.fixtures_data["event"] >= start_gameweek)
            & (deps.fixtures_data["event"] <= end_gameweek)
        ].copy()

        # 1. DGW/BGW Detection
        dgw_bgw_result = _detect_dgw_bgw(
            fixtures_in_horizon, deps.teams_data, start_gameweek, end_gameweek
        )

        # 2. Fixture Difficulty Analysis
        fixture_runs_result = _analyze_fixture_runs(
            fixtures_in_horizon,
            deps.teams_data,
            deps.fixture_difficulty,
            start_gameweek,
            num_gameweeks,
        )

        result = {
            "dgw_bgw_calendar": dgw_bgw_result,
            "fixture_runs": fixture_runs_result,
        }

        logger.info(
            f"✅ Fixture context analyzed: "
            f"DGW={dgw_bgw_result['has_dgw']}, "
            f"Easy runs={len(fixture_runs_result['easy_runs'])}, "
            f"Swings={len(fixture_runs_result['fixture_swings'])}"
        )

        return result

    except Exception as e:
        logger.error(f"❌ Error in analyze_fixture_context: {e}")
        raise


def _detect_dgw_bgw(
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    start_gw: int,
    end_gw: int,
) -> Dict[str, Any]:
    """Helper function to detect DGWs and BGWs."""
    dgw_teams = {}  # GW -> list of team names
    bgw_teams = {}  # GW -> list of team names

    # Count fixtures per team per gameweek
    for gw in range(start_gw, end_gw + 1):
        gw_fixtures = fixtures[fixtures["event"] == gw]

        # Count home and away fixtures per team
        home_counts = gw_fixtures["team_h"].value_counts()
        away_counts = gw_fixtures["team_a"].value_counts()
        total_counts = home_counts.add(away_counts, fill_value=0)

        # Find teams with DGW (2+ fixtures)
        dgw_team_ids = total_counts[total_counts >= 2].index.tolist()
        if dgw_team_ids:
            team_names = teams[teams["id"].isin(dgw_team_ids)]["name"].tolist()
            dgw_teams[gw] = team_names

        # Find teams with BGW (0 fixtures)
        all_team_ids = set(teams["id"].tolist())
        playing_team_ids = set(total_counts.index.tolist())
        bgw_team_ids = all_team_ids - playing_team_ids
        if bgw_team_ids:
            team_names = teams[teams["id"].isin(bgw_team_ids)]["name"].tolist()
            bgw_teams[gw] = team_names

    has_dgw = len(dgw_teams) > 0
    has_bgw = len(bgw_teams) > 0
    next_dgw_gw = min(dgw_teams.keys()) if has_dgw else None
    next_bgw_gw = min(bgw_teams.keys()) if has_bgw else None

    return {
        "dgw_teams": dgw_teams,
        "bgw_teams": bgw_teams,
        "has_dgw": has_dgw,
        "has_bgw": has_bgw,
        "next_dgw_gameweek": next_dgw_gw,
        "next_bgw_gameweek": next_bgw_gw,
    }


def _analyze_fixture_runs(
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    fixture_difficulty_df: pd.DataFrame | None,
    start_gw: int,
    num_gameweeks: int,
) -> Dict[str, Any]:
    """Helper function to analyze fixture difficulty runs."""
    if fixture_difficulty_df is None or fixture_difficulty_df.empty:
        # Return basic structure if no difficulty data
        return {"easy_runs": [], "hard_runs": [], "fixture_swings": []}

    # Calculate average difficulty per team over the horizon
    team_difficulties = []

    for _, team in teams.iterrows():
        team_id = team["id"]
        team_name = team["name"]

        # Get this team's fixtures
        team_fixtures = fixtures[
            (fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)
        ]

        if len(team_fixtures) == 0:
            continue

        # Get fixture difficulty scores
        difficulties = []
        fixture_details = []

        for _, fixture in team_fixtures.iterrows():
            is_home = fixture["team_h"] == team_id
            opponent_id = fixture["team_a"] if is_home else fixture["team_h"]
            opponent_name = teams[teams["id"] == opponent_id]["name"].values[0]

            # Get difficulty from fixture_difficulty_df
            # This should have columns: player_id, gameweek, fixture_difficulty
            # For team-level, we can use an aggregate or lookup
            # Simplified: use average difficulty from all players
            gw_difficulty = fixture_difficulty_df[
                fixture_difficulty_df["gameweek"] == fixture["event"]
            ]

            if not gw_difficulty.empty:
                avg_difficulty = gw_difficulty["fixture_difficulty"].mean()
            else:
                avg_difficulty = 3.0  # Neutral default

            difficulties.append(avg_difficulty)
            venue = "H" if is_home else "A"
            fixture_details.append(f"{opponent_name}({venue})")

        if len(difficulties) > 0:
            avg_difficulty = sum(difficulties) / len(difficulties)
            team_difficulties.append(
                {
                    "team": team_name,
                    "team_id": team_id,
                    "avg_difficulty": avg_difficulty,
                    "fixtures": fixture_details,
                    "current_difficulty": difficulties[0]
                    if len(difficulties) > 0
                    else avg_difficulty,
                    "upcoming_difficulty": (
                        sum(difficulties[1:]) / len(difficulties[1:])
                        if len(difficulties) > 1
                        else avg_difficulty
                    ),
                }
            )

    # Sort by difficulty
    team_difficulties.sort(key=lambda x: x["avg_difficulty"])

    # Top 5 easiest and hardest
    easy_runs = team_difficulties[:5]
    hard_runs = team_difficulties[-5:][::-1]  # Reverse for hardest first

    # Detect fixture swings (>1.5 difficulty change)
    fixture_swings = []
    for team_diff in team_difficulties:
        swing_magnitude = abs(
            team_diff["upcoming_difficulty"] - team_diff["current_difficulty"]
        )
        if swing_magnitude > 1.5:
            swing_type = (
                "hard_to_easy"
                if team_diff["current_difficulty"] > team_diff["upcoming_difficulty"]
                else "easy_to_hard"
            )
            fixture_swings.append(
                {
                    "team": team_diff["team"],
                    "team_id": team_diff["team_id"],
                    "swing_type": swing_type,
                    "current_difficulty": team_diff["current_difficulty"],
                    "upcoming_difficulty": team_diff["upcoming_difficulty"],
                    "swing_magnitude": swing_magnitude,
                }
            )

    return {
        "easy_runs": easy_runs,
        "hard_runs": hard_runs,
        "fixture_swings": fixture_swings,
    }


# Registry of all available tools
AGENT_TOOLS = [
    get_multi_gw_xp_predictions,
    analyze_fixture_context,
]
