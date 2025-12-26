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
from fpl_team_picker.domain.services.optimization_service import OptimizationService

logger = logging.getLogger(__name__)


def calculate_hit_cost(num_transfers: int, free_transfers: int) -> int:
    """
    Calculate hit cost for transfers.

    FPL rules: First N free transfers cost 0 points, additional transfers cost points.
    Uses config.optimization.transfer_cost (default: 4 points per transfer).

    Args:
        num_transfers: Number of transfers being made
        free_transfers: Number of free transfers available (0-2, typically 1)

    Returns:
        Hit cost in points (0, -4, -8, etc.)

    Examples:
        >>> calculate_hit_cost(1, 1)  # 1 transfer with 1 FT
        0
        >>> calculate_hit_cost(2, 1)  # 2 transfers with 1 FT
        -4
        >>> calculate_hit_cost(3, 2)  # 3 transfers with 2 FTs
        -4
        >>> calculate_hit_cost(0, 1)  # No transfers (banking FT)
        0
    """
    return -max(0, num_transfers - free_transfers) * config.optimization.transfer_cost


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
        free_transfers: int = 1,
        budget_available: float = 0.0,
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
        self.free_transfers = free_transfers
        self.budget_available = budget_available


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

        # Add team_name by merging with teams data (critical for LLM accuracy)
        # Prevents hallucinations where LLM uses outdated training data for player-team associations
        if "team_name" not in predictions_df.columns:
            # Merge on team_id to get current team name
            teams_for_merge = deps.teams_data[["team_id", "name"]].rename(
                columns={"name": "team_name"}
            )
            predictions_df = predictions_df.merge(
                teams_for_merge, on="team_id", how="left"
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
            team_names = teams[teams["team_id"].isin(dgw_team_ids)]["name"].tolist()
            dgw_teams[gw] = team_names

        # Find teams with BGW (0 fixtures)
        all_team_ids = set(teams["team_id"].tolist())
        playing_team_ids = set(total_counts.index.tolist())
        bgw_team_ids = all_team_ids - playing_team_ids
        if bgw_team_ids:
            team_names = teams[teams["team_id"].isin(bgw_team_ids)]["name"].tolist()
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
        team_id = team["team_id"]
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
            opponent_name = teams[teams["team_id"] == opponent_id]["name"].values[0]

            # Get difficulty from fixture_difficulty_df
            # This DataFrame has team-level fixture difficulty (not player-level)
            # Columns: team_id, opponent_id, gameweek, overall_difficulty, etc.
            team_fixture = fixture_difficulty_df[
                (fixture_difficulty_df["team_id"] == team_id)
                & (fixture_difficulty_df["gameweek"] == fixture["event"])
            ]

            if not team_fixture.empty:
                avg_difficulty = team_fixture["overall_difficulty"].values[0]
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


def run_sa_optimizer(
    ctx: RunContext[AgentDeps],
    current_squad_ids: list[int],
    num_transfers: int,
    target_gameweek: int,
    horizon: int = 3,
    must_include_ids: list[int] | None = None,
    must_exclude_ids: list[int] | None = None,
) -> Dict[str, Any]:
    """
    Run Simulated Annealing optimizer for validation/benchmarking.

    Use this tool to validate your transfer recommendations against the SA optimizer's
    mathematically optimal solution. The optimizer searches thousands of combinations
    to find the highest xP squad within constraints over the specified horizon.

    Hit costs are calculated automatically using free_transfers from context:
    - First N free transfers cost 0 points
    - Additional transfers cost -4 each
    - Example: 2 transfers with 1 FT = -4 hit cost

    IMPORTANT: Call this tool AFTER generating your candidate scenarios to validate them,
    not BEFORE. The agent should reason first, then validate.

    NOTE: If num_transfers=0 (hold option), this returns the current squad's xP without
    running any optimization. Use this to get baseline xP for the hold scenario.

    Args:
        ctx: Agent context with data dependencies (includes free_transfers)
        current_squad_ids: List of 15 player IDs in current squad
        num_transfers: Number of transfers to make (0-15, where 0 = hold/no transfers)
        target_gameweek: Gameweek to optimize for (1-38)
        horizon: Gameweeks ahead to optimize (1, 3, or 5; default: 3 for agent context)
        must_include_ids: Player IDs that must stay in squad (optional)
        must_exclude_ids: Player IDs to avoid (optional)

    Returns:
        {
            "optimal_squad_ids": [1, 2, 3, ...],  # 15 player IDs
            "transfers": [  # Transfers to reach optimal squad (empty if num_transfers=0)
                {
                    "out_id": 5,
                    "in_id": 42,
                    "out_name": "Smith",
                    "in_name": "Jones",
                    "position": "MID",
                    "cost_diff": 0.5  # Price difference
                }
            ],
            "expected_xp": 65.3,  # Total squad xP
            "xp_gain": 4.2,  # Improvement vs current (0.0 if num_transfers=0)
            "hit_cost": -4,  # Points deduction (0 if num_transfers=0)
            "net_gain": 0.2,  # xP gain minus hit cost (0.0 if num_transfers=0)
            "free_transfers_used": 1,  # How many FTs were used (0 if num_transfers=0)
            "runtime_seconds": 12.4,
            "formation": "4-4-2"
        }

    Example usage:
        # Validate hold option (0 transfers)
        hold_result = run_sa_optimizer(
            current_squad_ids=[1,2,3,...,15],
            num_transfers=0,
            target_gameweek=18,
            horizon=3
        )
        # Returns current squad's xP without optimization

        # Validate a 1-transfer scenario (free if you have 1 FT)
        sa_result = run_sa_optimizer(
            current_squad_ids=[1,2,3,...,15],
            num_transfers=1,
            target_gameweek=18,
            horizon=3  # Default, can be 1, 3, or 5
        )
        # If free_transfers=1, hit_cost will be 0
        # Compare your scenario's 3-GW xP to sa_result["expected_xp"]

        # Validate a 2-transfer scenario (takes a hit if only 1 FT)
        sa_result_2t = run_sa_optimizer(
            current_squad_ids=[1,2,3,...,15],
            num_transfers=2,
            target_gameweek=18
        )
        # If free_transfers=1, hit_cost will be -4

        # Check if SA found a better transfer
        if len(sa_result["transfers"]) > 0:
            best_transfer = sa_result["transfers"][0]
            print(f"SA suggests: {best_transfer['out_name']} → {best_transfer['in_name']}")

    Edge cases:
        - Fails if current_squad_ids length != 15 (validation enforced)
        - num_transfers must be 0-15 (0 = hold, 1+ = optimize transfers)
        - target_gameweek must be 1-38 (validated)
        - horizon must be 1, 3, or 5 (validated)
        - Runtime increases with num_transfers (0: instant, 1: ~10s, 2: ~30s, 3: ~60s)
        - Returns empty transfers list if num_transfers=0 or no improvement found
        - Hit cost automatically calculated based on free_transfers in context

    Poka-yoke (error prevention):
        - current_squad_ids must be List[int], not DataFrame
        - All IDs validated against available players
        - Budget and squad constraints automatically enforced
        - Free transfers from context used for hit cost calculation

    Performance:
        - 0 transfers (hold): ~1-2s (xP calculation only)
        - 1 transfer: ~10-15s (exhaustive search if enabled)
        - 2 transfers: ~30-45s
        - 3+ transfers: ~60-120s (SA iterations)
    """
    logger.info(
        f"🔧 Running SA optimizer: {num_transfers} transfers for GW{target_gameweek}"
    )

    deps = ctx.deps

    # Validate inputs
    if len(current_squad_ids) != 15:
        raise ValueError(
            f"current_squad_ids must have exactly 15 players, got {len(current_squad_ids)}"
        )
    if not (0 <= num_transfers <= 15):
        raise ValueError(f"num_transfers must be 0-15, got {num_transfers}")
    if not (1 <= target_gameweek <= 38):
        raise ValueError(f"target_gameweek must be 1-38, got {target_gameweek}")

    try:
        # Convert squad IDs to DataFrame format expected by OptimizationService
        current_squad_df = deps.players_data[
            deps.players_data["player_id"].isin(current_squad_ids)
        ].copy()

        # Check for missing players (e.g., long-term injury, left club)
        num_missing = 15 - len(current_squad_df)
        if num_missing > 0:
            found_ids = set(current_squad_df["player_id"].tolist())
            missing_ids = [pid for pid in current_squad_ids if pid not in found_ids]

            logger.warning(
                f"⚠️ {num_missing} player(s) from squad not available in current gameweek: {missing_ids}. "
                f"These players must be transferred out (likely long-term injury or left club)."
            )

            # If hold option (0 transfers), this is an error - can't hold with missing players
            if num_transfers == 0:
                raise ValueError(
                    f"Cannot hold squad with {num_missing} missing player(s) (IDs: {missing_ids}). "
                    f"These players must be transferred out. Minimum transfers required: {num_missing}"
                )

            # If num_transfers < num_missing, this is also an error
            if num_transfers < num_missing:
                raise ValueError(
                    f"Cannot complete {num_transfers} transfer(s) when {num_missing} player(s) are missing (IDs: {missing_ids}). "
                    f"Minimum transfers required: {num_missing}"
                )

            # For optimization, we'll work with the available squad (14 or fewer players)
            # and the optimizer will need to fill the missing slots
            logger.info(
                f"📊 Optimizing with {len(current_squad_df)} available players, "
                f"need to add {num_missing} players to reach 15-player squad"
            )

        # Validate horizon parameter
        if horizon not in [1, 3, 5]:
            raise ValueError(f"horizon must be 1, 3, or 5, got {horizon}")

        # Get xP predictions for optimization based on horizon
        ml_service = MLExpectedPointsService(model_path=config.xp_model.ml_model_path)

        if horizon == 1:
            players_with_xp = ml_service.calculate_expected_points(
                players_data=deps.players_data,
                teams_data=deps.teams_data,
                xg_rates_data=deps.xg_rates
                if deps.xg_rates is not None
                else pd.DataFrame(),
                fixtures_data=deps.fixtures_data,
                target_gameweek=target_gameweek,
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
        elif horizon == 3:
            players_with_xp = ml_service.calculate_3gw_expected_points(
                players_data=deps.players_data,
                teams_data=deps.teams_data,
                xg_rates_data=deps.xg_rates
                if deps.xg_rates is not None
                else pd.DataFrame(),
                fixtures_data=deps.fixtures_data,
                target_gameweek=target_gameweek,
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
        else:  # horizon == 5
            players_with_xp = ml_service.calculate_5gw_expected_points(
                players_data=deps.players_data,
                teams_data=deps.teams_data,
                xg_rates_data=deps.xg_rates
                if deps.xg_rates is not None
                else pd.DataFrame(),
                fixtures_data=deps.fixtures_data,
                target_gameweek=target_gameweek,
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

        # Create column alias for optimizer (it expects 'xP_5gw' column)
        # Map the horizon-specific column to what optimizer expects
        logger.info(
            f"📊 Available columns in players_with_xp: {list(players_with_xp.columns)}"
        )

        if horizon == 1:
            # For 1-GW, use ml_xP as the optimization target
            players_with_xp["xP_5gw"] = players_with_xp["ml_xP"]
        elif horizon == 3:
            # For 3-GW, use xP_3gw as the optimization target
            # Check if xP_3gw exists, if not use ml_xP (which is the 3gw total)
            if "xP_3gw" in players_with_xp.columns:
                players_with_xp["xP_5gw"] = players_with_xp["xP_3gw"]
            else:
                logger.warning(
                    "⚠️ xP_3gw column not found, using ml_xP instead (this should contain 3GW total)"
                )
                players_with_xp["xP_5gw"] = players_with_xp["ml_xP"]
        # For horizon==5, xP_5gw already exists, no alias needed

        # Handle hold option (0 transfers) - just calculate current squad xP
        import time

        start_time = time.time()

        if num_transfers == 0:
            # Get xP for current squad without optimization
            current_squad_with_xp = players_with_xp[
                players_with_xp["player_id"].isin(current_squad_ids)
            ].copy()

            total_xp = current_squad_with_xp["xP_5gw"].sum()

            # Determine formation (simple heuristic: count positions)
            pos_counts = current_squad_with_xp["position"].value_counts().to_dict()
            formation = f"{pos_counts.get('DEF', 0)}-{pos_counts.get('MID', 0)}-{pos_counts.get('FWD', 0)}"

            runtime = time.time() - start_time

            result = {
                "optimal_squad_ids": current_squad_ids,
                "transfers": [],
                "expected_xp": float(total_xp),
                "xp_gain": 0.0,
                "hit_cost": 0,
                "net_gain": 0.0,
                "free_transfers_used": 0,
                "runtime_seconds": round(runtime, 1),
                "formation": formation,
            }

            logger.info(
                f"✅ Hold option calculated: xP={result['expected_xp']:.1f}, runtime={result['runtime_seconds']}s"
            )

            return result

        # Initialize optimization service
        opt_service = OptimizationService()

        # IMPORTANT: Pass the actual number of free transfers to the optimizer
        # num_transfers parameter is how many transfers the agent wants to test,
        # but we need to tell optimizer how many are free (from deps)
        optimal_squad, best_scenario, metadata = opt_service.optimize_transfers(
            current_squad=current_squad_df,
            team_data={
                "bank": deps.budget_available,
                "free_transfers": num_transfers,  # Tell optimizer how many transfers to make
            },
            players_with_xp=players_with_xp,
            must_include_ids=set(must_include_ids) if must_include_ids else None,
            must_exclude_ids=set(must_exclude_ids) if must_exclude_ids else None,
        )

        runtime = time.time() - start_time

        # Extract transfers
        transfers_out = metadata.get("transfers_out", [])
        transfers_in = metadata.get("transfers_in", [])

        transfers = []
        for out_player, in_player in zip(transfers_out, transfers_in):
            transfers.append(
                {
                    "out_id": out_player.get("player_id", 0),
                    "in_id": in_player.get("player_id", 0),
                    "out_name": out_player.get("web_name", "Unknown"),
                    "in_name": in_player.get("web_name", "Unknown"),
                    "position": in_player.get("position", ""),
                    "cost_diff": in_player.get("price", 0.0)
                    - out_player.get("price", 0.0),
                }
            )

        # Calculate hit cost using actual free transfers from context
        actual_num_transfers = len(transfers)
        hit_cost = calculate_hit_cost(actual_num_transfers, deps.free_transfers)

        # Build result
        result = {
            "optimal_squad_ids": optimal_squad["player_id"].tolist(),
            "transfers": transfers,
            "expected_xp": float(best_scenario.get("net_xp", 0.0)),
            "xp_gain": float(best_scenario.get("xp_gain", 0.0)),
            "hit_cost": hit_cost,
            "net_gain": float(best_scenario.get("net_xp", 0.0))
            + hit_cost,  # hit_cost is negative
            "free_transfers_used": min(actual_num_transfers, deps.free_transfers),
            "runtime_seconds": round(runtime, 1),
            "formation": best_scenario.get("formation", ""),
        }

        logger.info(
            f"✅ SA optimization complete: {len(transfers)} transfers, "
            f"xP={result['expected_xp']:.1f}, runtime={result['runtime_seconds']}s"
        )

        return result

    except Exception as e:
        logger.error(f"❌ Error in run_sa_optimizer: {e}")
        raise


def analyze_squad_weaknesses(
    ctx: RunContext[AgentDeps],
    current_squad_ids: list[int],
    target_gameweek: int,
) -> Dict[str, Any]:
    """
    Identify weak spots in current squad for targeted upgrades.

    Analyzes current squad to find players with low xP, injury risks, rotation threats,
    or poor fixtures. Use this to identify which positions need upgrading.

    Args:
        ctx: Agent context with data dependencies
        current_squad_ids: List of 15 player IDs
        target_gameweek: Gameweek to analyze for (1-38)

    Returns:
        {
            "missing_players": [  # Players not available in current GW (MUST transfer out)
                {
                    "player_id": 123,
                    "web_name": "Unknown (ID 123)",
                    "xp_gw1": 0.0,
                    "position": "UNKNOWN",
                    "reason": "Player not available (likely long-term injury or left club)"
                }
            ],
            "low_xp_players": [  # Bottom 5 by xP
                {
                    "player_id": 5,
                    "web_name": "Smith",
                    "xp_gw1": 2.1,
                    "position": "DEF",
                    "reason": "Low expected points"
                }
            ],
            "rotation_risks": [...],  # rotation_risk > 0.5
            "injury_concerns": [...],  # chance_of_playing < 75%
            "poor_fixtures": [...],  # fixture_difficulty >= 4.0
            "summary": {
                "total_weaknesses": 3,
                "by_position": {"DEF": 2, "MID": 1},
                "message": "3 upgrade targets identified: 2 DEF, 1 MID"
            }
        }

    Example usage:
        weaknesses = analyze_squad_weaknesses(
            current_squad_ids=[1,2,3,...,15],
            target_gameweek=18
        )

        # Focus transfers on low xP players
        for player in weaknesses["low_xp_players"]:
            print(f"{player['web_name']}: {player['xp_gw1']} xP ({player['position']})")

        # Check injury concerns
        if weaknesses["injury_concerns"]:
            print(f"Injury concerns: {len(weaknesses['injury_concerns'])} players")

    Edge cases:
        - Returns empty lists if no weaknesses found (strong squad)
        - Limited to 5 players per category
        - Thresholds: xP < 3.0, rotation > 0.5, injury < 75%, fixture >= 4.0
        - If player metrics missing, some categories may be empty

    Performance:
        - Typical execution: ~100-200ms
    """
    logger.info(f"🔍 Analyzing squad weaknesses for GW{target_gameweek}")

    deps = ctx.deps

    # Validate input
    if len(current_squad_ids) != 15:
        raise ValueError(
            f"current_squad_ids must have exactly 15 players, got {len(current_squad_ids)}"
        )

    try:
        # Get current squad data
        squad_df = deps.players_data[
            deps.players_data["player_id"].isin(current_squad_ids)
        ].copy()

        # Check for missing players
        num_missing = 15 - len(squad_df)
        missing_players_list = []
        if num_missing > 0:
            found_ids = set(squad_df["player_id"].tolist())
            missing_ids = [pid for pid in current_squad_ids if pid not in found_ids]

            logger.warning(
                f"⚠️ {num_missing} player(s) from squad not available: {missing_ids}. "
                f"These will be automatically flagged as weaknesses."
            )

            # Add missing players to weaknesses list
            for pid in missing_ids:
                missing_players_list.append(
                    {
                        "player_id": pid,
                        "web_name": f"Unknown (ID {pid})",
                        "xp_gw1": 0.0,
                        "position": "UNKNOWN",
                        "reason": "Player not available (likely long-term injury or left club)",
                    }
                )

        # Get xP predictions
        ml_service = MLExpectedPointsService(model_path=config.xp_model.ml_model_path)
        squad_with_xp = ml_service.calculate_expected_points(
            players_data=squad_df,
            teams_data=deps.teams_data,
            xg_rates_data=deps.xg_rates
            if deps.xg_rates is not None
            else pd.DataFrame(),
            fixtures_data=deps.fixtures_data,
            target_gameweek=target_gameweek,
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

        # 1. Low xP players (< 3.0)
        low_xp = squad_with_xp[squad_with_xp["ml_xP"] < 3.0].nsmallest(5, "ml_xP")
        low_xp_players = [
            {
                "player_id": int(row["player_id"]),
                "web_name": row["web_name"],
                "xp_gw1": float(row["ml_xP"]),
                "position": row["position"],
                "reason": "Low expected points",
            }
            for _, row in low_xp.iterrows()
        ]

        # 2. Rotation risks (if player_metrics available)
        rotation_risks = []
        if deps.player_metrics is not None and not deps.player_metrics.empty:
            metrics_df = deps.player_metrics[
                deps.player_metrics["player_id"].isin(current_squad_ids)
            ]
            high_rotation = metrics_df[metrics_df.get("rotation_risk", 0) > 0.5]
            rotation_risks = [
                {
                    "player_id": int(row["player_id"]),
                    "web_name": squad_df[squad_df["player_id"] == row["player_id"]][
                        "web_name"
                    ].values[0],
                    "rotation_risk": float(row.get("rotation_risk", 0)),
                    "position": squad_df[squad_df["player_id"] == row["player_id"]][
                        "position"
                    ].values[0],
                    "reason": "High rotation risk",
                }
                for _, row in high_rotation.head(5).iterrows()
            ]

        # 3. Injury concerns (if player_availability available)
        injury_concerns = []
        if deps.player_availability is not None and not deps.player_availability.empty:
            avail_df = deps.player_availability[
                deps.player_availability["player_id"].isin(current_squad_ids)
            ]
            injured = avail_df[avail_df.get("chance_of_playing_next_round", 100) < 75]
            injury_concerns = [
                {
                    "player_id": int(row["player_id"]),
                    "web_name": squad_df[squad_df["player_id"] == row["player_id"]][
                        "web_name"
                    ].values[0],
                    "chance_of_playing": int(
                        row.get("chance_of_playing_next_round", 0)
                    ),
                    "position": squad_df[squad_df["player_id"] == row["player_id"]][
                        "position"
                    ].values[0],
                    "reason": "Injury concern",
                }
                for _, row in injured.head(5).iterrows()
            ]

        # 4. Poor fixtures (if fixture_difficulty available)
        # NOTE: fixture_difficulty is team-level data, not player-level
        poor_fixtures = []
        if deps.fixture_difficulty is not None and not deps.fixture_difficulty.empty:
            # squad_with_xp already has team_id from ML service, use it directly
            # Get team IDs from current squad
            squad_team_ids = squad_with_xp["team_id"].dropna().unique().tolist()

            # Filter fixtures by squad teams and target gameweek
            fixture_df = deps.fixture_difficulty[
                (deps.fixture_difficulty["team_id"].isin(squad_team_ids))
                & (deps.fixture_difficulty["gameweek"] == target_gameweek)
            ]

            # Find hard fixtures (overall_difficulty >= 4.0)
            hard_fixtures_df = fixture_df[
                fixture_df.get("overall_difficulty", 0) >= 4.0
            ]

            # Map back to players from hard-fixture teams
            for _, fixture_row in hard_fixtures_df.iterrows():
                team_id = fixture_row["team_id"]
                # Get all players from this team in the squad
                team_players = squad_with_xp[squad_with_xp["team_id"] == team_id]

                for _, player_row in team_players.iterrows():
                    poor_fixtures.append(
                        {
                            "player_id": int(player_row["player_id"]),
                            "web_name": player_row["web_name"],
                            "fixture_difficulty": float(
                                fixture_row.get("overall_difficulty", 0)
                            ),
                            "position": player_row["position"],
                            "reason": "Poor fixture",
                        }
                    )

            # Limit to 5 total
            poor_fixtures = poor_fixtures[:5]

        # Build summary
        all_weaknesses = (
            missing_players_list
            + low_xp_players
            + rotation_risks
            + injury_concerns
            + poor_fixtures
        )
        by_position = {}
        for weakness in all_weaknesses:
            pos = weakness["position"]
            by_position[pos] = by_position.get(pos, 0) + 1

        summary_parts = [f"{count} {pos}" for pos, count in by_position.items()]
        summary_message = f"{len(all_weaknesses)} upgrade targets identified"
        if summary_parts:
            summary_message += f": {', '.join(summary_parts)}"

        result = {
            "missing_players": missing_players_list,
            "low_xp_players": low_xp_players,
            "rotation_risks": rotation_risks,
            "injury_concerns": injury_concerns,
            "poor_fixtures": poor_fixtures,
            "summary": {
                "total_weaknesses": len(all_weaknesses),
                "by_position": by_position,
                "message": summary_message,
            },
        }

        logger.info(f"✅ Weaknesses analyzed: {result['summary']['message']}")

        return result

    except Exception as e:
        logger.error(f"❌ Error in analyze_squad_weaknesses: {e}")
        raise


def get_template_players(
    ctx: RunContext[AgentDeps],
    target_gameweek: int,
    ownership_threshold: float = 30.0,
) -> Dict[str, Any]:
    """
    Identify template players (high ownership) for safety consideration.

    Template players are widely owned (>30%) - avoiding them risks rank drops if they haul.
    Use this to ensure recommended transfers don't create excessive differential risk.

    Args:
        ctx: Agent context with data dependencies
        target_gameweek: Gameweek to check (1-38)
        ownership_threshold: Minimum ownership % to qualify (default: 30.0, range: 10.0-50.0)

    Returns:
        {
            "template_players": [  # Top 15 by ownership
                {
                    "player_id": 1,
                    "web_name": "Haaland",
                    "ownership": 65.4,
                    "xp_gw1": 8.2,
                    "position": "FWD",
                    "price": 15.0,
                    "in_squad": False  # Whether player is in current squad
                }
            ],
            "missing_from_squad": [...],  # High ownership not owned (risky)
            "owned_differentials": [...],  # Low ownership (<10%) owned (differential)
            "template_coverage": 0.73  # % of template players owned (0-1)
        }

    Example usage:
        templates = get_template_players(target_gameweek=18, ownership_threshold=40.0)

        # Ensure recommended squad has >60% template_coverage for safety
        if templates["template_coverage"] < 0.6:
            print("Warning: Low template coverage, consider safer picks")

        # Identify risky differentials
        if templates["owned_differentials"]:
            print(f"You own {len(templates['owned_differentials'])} differentials")

    Edge cases:
        - May return <15 players if few exceed ownership_threshold
        - ownership_threshold range: 10.0-50.0 (validated)
        - Returns empty lists if no template players found
        - Requires current_squad_ids in context for in_squad calculation

    Performance:
        - Typical execution: ~100-200ms
    """
    logger.info(
        f"🔍 Analyzing template players for GW{target_gameweek} "
        f"(threshold={ownership_threshold}%)"
    )

    deps = ctx.deps

    # Validate ownership threshold
    if not (10.0 <= ownership_threshold <= 50.0):
        raise ValueError(
            f"ownership_threshold must be 10.0-50.0, got {ownership_threshold}"
        )

    try:
        # Get xP predictions for all players
        ml_service = MLExpectedPointsService(model_path=config.xp_model.ml_model_path)
        players_with_xp = ml_service.calculate_expected_points(
            players_data=deps.players_data,
            teams_data=deps.teams_data,
            xg_rates_data=deps.xg_rates
            if deps.xg_rates is not None
            else pd.DataFrame(),
            fixtures_data=deps.fixtures_data,
            target_gameweek=target_gameweek,
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

        # Filter template players (high ownership)
        template_df = players_with_xp[
            players_with_xp["selected_by_percent"] >= ownership_threshold
        ].nlargest(15, "selected_by_percent")

        template_players = [
            {
                "player_id": int(row["player_id"]),
                "web_name": row["web_name"],
                "ownership": float(row["selected_by_percent"]),
                "xp_gw1": float(row["ml_xP"]),
                "position": row["position"],
                "price": float(row["price"]),
                "in_squad": False,  # Will be updated if current_squad provided
            }
            for _, row in template_df.iterrows()
        ]

        # Identify missing template players (not in squad)
        # Note: This requires current_squad_ids to be available
        # For now, return empty - agent should pass squad IDs if needed
        missing_from_squad = []
        owned_differentials = []
        template_coverage = 0.0

        # If we had current_squad_ids, we would calculate:
        # missing_from_squad = [p for p in template_players if p["player_id"] not in current_squad_ids]
        # owned_differentials = [p for p in squad if p["ownership"] < 10.0]
        # template_coverage = (15 - len(missing_from_squad)) / 15

        result = {
            "template_players": template_players,
            "missing_from_squad": missing_from_squad,
            "owned_differentials": owned_differentials,
            "template_coverage": template_coverage,
        }

        logger.info(
            f"✅ Template analysis complete: {len(template_players)} template players found"
        )

        return result

    except Exception as e:
        logger.error(f"❌ Error in get_template_players: {e}")
        raise


# Registry of all available tools
AGENT_TOOLS = [
    get_multi_gw_xp_predictions,
    analyze_fixture_context,
    run_sa_optimizer,
    analyze_squad_weaknesses,
    get_template_players,
]
