"""Run transfer planning agent evaluations using pydantic-evals.

This script evaluates the FPL transfer planning agent across multiple scenarios,
assessing structural validity, strategic quality, and decision-making.

Usage:
    # Run all evaluations
    uv run python evals/run_transfer_evals.py

    # Run specific dataset (with short option)
    uv run python evals/run_transfer_evals.py -d basic

    # Run with specific model
    uv run python evals/run_transfer_evals.py -m claude-haiku-4-5

    # Quick test with 1 case (combined short options)
    uv run python evals/run_transfer_evals.py -d basic -n 1

    # Enable Logfire reporting
    uv run python evals/run_transfer_evals.py --enable-logfire

    # Get help
    uv run python evals/run_transfer_evals.py --help
"""

import asyncio
import json
import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import typer
from pydantic_evals import Dataset

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evals.datasets.transfer_planning_scenarios import (  # noqa: E402
    get_basic_scenarios,
    get_constraint_scenarios,
    get_strategic_scenarios,
    transfer_planning_dataset,
)
from evals.evaluators.transfer_quality import (  # noqa: E402
    CompositeTransferQualityEvaluator,
    HitAnalysisEvaluator,
    LLMReasoningQualityEvaluator,
    OwnershipStrategyEvaluator,
    ScenarioCoverageEvaluator,
    StrategicQualityEvaluator,
    StructuralValidityEvaluator,
)
from fpl_team_picker.domain.models.transfer_plan import StrategyMode  # noqa: E402
from fpl_team_picker.domain.models.transfer_recommendation import (  # noqa: E402
    SingleGWRecommendation,
)
from fpl_team_picker.domain.services.transfer_planning_agent_service import (  # noqa: E402
    TransferPlanningAgentService,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_gameweek_data(
    gameweek: int, scenario_context: str
) -> Dict[str, pd.DataFrame]:
    """Create mock gameweek data for evaluation.

    In a production eval setup, this would load real historical data.
    For now, creates minimal valid data structures.

    Args:
        gameweek: Target gameweek number
        scenario_context: Scenario description for context

    Returns:
        Dict with all required gameweek data structures
    """
    # Create minimal valid data
    # In production, load from historical FPL data matching the scenario
    players_df = pd.DataFrame(
        {
            "player_id": list(range(1, 101)),
            "web_name": [f"Player{i}" for i in range(1, 101)],
            "name": [f"Team{i % 20}" for i in range(1, 101)],
            "position": (["GKP"] * 10 + ["DEF"] * 30 + ["MID"] * 40 + ["FWD"] * 20),
            "price": [5.0 + (i % 10) for i in range(1, 101)],
            "selected_by_percent": [10.0 + (i % 50) for i in range(1, 101)],
            "team_id": [(i % 20) + 1 for i in range(1, 101)],
        }
    )

    teams_df = pd.DataFrame(
        {"team_id": list(range(1, 21)), "name": [f"Team{i}" for i in range(1, 21)]}
    )

    # Add 'team' column (SA optimizer expects 'team', not just 'team_id')
    # Map team_id to team name using teams_df
    players_df = players_df.merge(
        teams_df[["team_id", "name"]].rename(columns={"name": "team"}),
        on="team_id",
        how="left",
    )

    current_squad_df = players_df.head(15).copy()

    fixtures_df = pd.DataFrame(
        {
            "id": list(range(1, 11)),
            "event": [gameweek] * 10,
            "home_team_id": list(range(1, 11)),
            "away_team_id": list(range(11, 21)),
        }
    )

    # Create minimal historical live data for ML predictions
    # Start from GW1 to ensure sufficient history for any lookback operations
    historical_data = []
    for player_id in range(1, 101):
        for gw in range(1, gameweek):  # Full history from GW1
            historical_data.append(
                {
                    "player_id": player_id,
                    "gameweek": gw,
                    "minutes": 60 + (player_id % 30),  # Varied minutes
                    "total_points": 2 + (player_id % 8),  # Varied points
                    "goals_scored": (player_id % 10) // 5,  # Some goals
                    "assists": (player_id % 8) // 4,  # Some assists
                }
            )

    live_data_historical = (
        pd.DataFrame(historical_data) if historical_data else pd.DataFrame()
    )

    # Enrich historical data with position (required for ML feature engineering)
    if (
        not live_data_historical.empty
        and "position" not in live_data_historical.columns
    ):
        if "position" in players_df.columns and "player_id" in players_df.columns:
            live_data_historical = live_data_historical.merge(
                players_df[["player_id", "position"]],
                on="player_id",
                how="left",
            )
            # Fill any missing positions with a default (shouldn't happen, but safety check)
            if live_data_historical["position"].isna().any():
                live_data_historical["position"] = live_data_historical[
                    "position"
                ].fillna("MID")

    # Create minimal ownership trends data for ML feature engineering
    # Start from GW1 and extend beyond current GW for predictions
    # This ensures shift(1) operations have all required historical data
    ownership_data = []
    for player_id in range(1, 101):
        for gw in range(1, gameweek + 10):  # GW1 to current+10
            ownership_data.append(
                {
                    "player_id": player_id,
                    "gameweek": gw,
                    "selected_by_percent": 10.0 + (player_id % 50),  # Varied ownership
                    "net_transfers_gw": (player_id % 10) - 5,  # Varied net transfers
                    "transfer_momentum": 0.5 + (player_id % 3) * 0.2,  # Momentum score
                    "ownership_tier": "mid",  # Simple tier classification
                    "bandwagon_score": 0.3 + (player_id % 5) * 0.1,  # Bandwagon metric
                    "ownership_vs_price": 1.0 + (player_id % 7) * 0.1,  # Value metric
                }
            )

    ownership_trends = (
        pd.DataFrame(ownership_data) if ownership_data else pd.DataFrame()
    )

    # Create mock value analysis data for ML service
    # Required columns based on DerivedValueAnalysisSchema
    # CRITICAL: Feature engineer shifts by +1 gameweek, so we need value_analysis
    # for all historical gameweeks (1 to gameweek+1) to support the shift operation
    # #region debug log
    with open("/Users/alex/dev/FPL/fpl-team-picker/.cursor/debug.log", "a") as f:
        f.write(
            json.dumps(
                {
                    "sessionId": "debug-session",
                    "runId": "pre-fix",
                    "hypothesisId": "A",
                    "location": "evals/run_transfer_evals.py:create_mock_gameweek_data",
                    "message": "Creating value_analysis - checking required gameweeks",
                    "data": {
                        "target_gameweek": gameweek,
                        "live_data_gws": sorted(
                            live_data_historical["gameweek"].unique().tolist()
                        )
                        if not live_data_historical.empty
                        and "gameweek" in live_data_historical.columns
                        else [],
                        "expected_value_gws": f"1 to {gameweek + 1}",
                    },
                    "timestamp": pd.Timestamp.now().isoformat(),
                }
            )
            + "\n"
        )
    # #endregion

    value_analysis_list = []
    # Create value_analysis for all gameweeks from 1 to gameweek+1
    # (gameweek+1 accounts for the shift(1) operation in feature engineering)
    for gw in range(1, gameweek + 2):  # +2 because range is exclusive
        for _, player in players_df.iterrows():
            player_id = player["player_id"]
            position_map = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
            position_id = position_map.get(player["position"], 2)

            # Create realistic mock values (can vary by gameweek if needed)
            total_points = (
                50 + (player_id % 100) + (gw * 2)
            )  # Slight progression over time
            current_price = player["price"]
            points_per_pound = total_points / current_price if current_price > 0 else 0

            value_analysis_row = {
                "player_id": player_id,
                "gameweek": gw,
                "web_name": player["web_name"],
                "position_id": position_id,
                "current_price": current_price,
                "total_points": total_points,
                "points_per_pound": points_per_pound,
                "expected_points_per_pound": points_per_pound
                * 0.9,  # Slightly lower expected
                "value_vs_position": 50.0 + (player_id % 50),  # Percentile 0-100
                "value_vs_price_tier": 50.0 + (player_id % 50),  # Percentile 0-100
                "predicted_price_change_1gw": (player_id % 3) - 1,  # -1, 0, or 1
                "predicted_price_change_5gw": (player_id % 5) - 2,  # -2 to 2
                "price_volatility": 0.1 + (player_id % 10) * 0.05,  # 0.1 to 0.55
                "buy_rating": 5.0 + (player_id % 5),  # 5.0 to 9.0
                "sell_rating": 2.0 + (player_id % 4),  # 2.0 to 5.0
                "hold_rating": 5.0 + (player_id % 5),  # 5.0 to 9.0
                "ownership_risk": 0.2 + (player_id % 5) * 0.1,  # 0.2 to 0.6
                "price_risk": 0.1 + (player_id % 5) * 0.1,  # 0.1 to 0.5
                "performance_risk": 0.2 + (player_id % 5) * 0.1,  # 0.2 to 0.6
                "recommendation": ["hold", "buy", "hold", "sell", "hold"][
                    player_id % 5
                ],
                "confidence": 0.6 + (player_id % 4) * 0.1,  # 0.6 to 0.9
                "analysis_date": pd.Timestamp.now(),
            }
            value_analysis_list.append(value_analysis_row)

    value_analysis = pd.DataFrame(value_analysis_list)

    # #region debug log
    try:
        with open("/Users/alex/dev/FPL/fpl-team-picker/.cursor/debug.log", "a") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "A",
                        "location": "evals/run_transfer_evals.py:create_mock_gameweek_data",
                        "message": "Value_analysis created",
                        "data": {
                            "total_rows": len(value_analysis),
                            "gameweeks_created": sorted(
                                value_analysis["gameweek"].unique().tolist()
                            )
                            if not value_analysis.empty
                            and "gameweek" in value_analysis.columns
                            else [],
                            "players_in_value_analysis": sorted(
                                value_analysis["player_id"].unique().tolist()[:20]
                            )
                            if not value_analysis.empty
                            and "player_id" in value_analysis.columns
                            else [],
                            "players_per_gw": len(players_df),
                            "players_in_players_df": sorted(
                                players_df["player_id"].unique().tolist()[:20]
                            )
                            if "player_id" in players_df.columns
                            else [],
                            "live_data_players": sorted(
                                live_data_historical["player_id"].unique().tolist()[:20]
                            )
                            if not live_data_historical.empty
                            and "player_id" in live_data_historical.columns
                            else [],
                            "live_data_gameweeks": sorted(
                                live_data_historical["gameweek"].unique().tolist()
                            )
                            if not live_data_historical.empty
                            and "gameweek" in live_data_historical.columns
                            else [],
                        },
                        "timestamp": pd.Timestamp.now().isoformat(),
                    }
                )
                + "\n"
            )
    except Exception:
        pass  # Don't fail on logging
    # #endregion

    # Create mock fixture difficulty data for ML service
    # Required columns based on DerivedFixtureDifficultySchema
    # Need data for all historical gameweeks (1 to gameweek+1) to support feature engineering
    fixture_difficulty_list = []
    # Create fixture difficulty for all gameweeks from 1 to gameweek+1
    for gw in range(1, gameweek + 2):  # +2 because range is exclusive
        # Create fixtures for this gameweek (10 fixtures = 20 teams, similar to fixtures_df)
        # Use a simple pairing: team 1 vs 11, 2 vs 12, etc.
        for i in range(10):
            home_team_id = (i % 10) + 1
            away_team_id = ((i + 10) % 20) + 1
            fixture_id = (gw - 1) * 100 + i + 1

            # Create two rows per fixture (home and away perspectives)
            for is_home, team_id, opponent_id in [
                (True, home_team_id, away_team_id),
                (False, away_team_id, home_team_id),
            ]:
                # Create realistic mock difficulty values
                base_difficulty = 2.5 + (team_id % 3)  # 2.5 to 4.5
                fixture_difficulty_row = {
                    "fixture_id": fixture_id,
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "gameweek": gw,
                    "is_home": is_home,
                    "kickoff_time": pd.Timestamp.now(),
                    "opponent_strength_difficulty": base_difficulty,
                    "venue_difficulty": base_difficulty + (0.5 if is_home else -0.5),
                    "congestion_difficulty": 2.0 + (team_id % 2),
                    "form_difficulty": 2.5 + (team_id % 2) * 0.5,
                    "overall_difficulty": base_difficulty,
                    "difficulty_tier": min(5, max(1, int(base_difficulty))),
                    "difficulty_confidence": 0.7 + (team_id % 3) * 0.1,
                    "expected_goals_for": 1.0 + (team_id % 3) * 0.5,
                    "expected_goals_against": 1.0 + (opponent_id % 3) * 0.5,
                    "expected_points": 1.0 + (team_id % 2) * 0.5,
                    "clean_sheet_probability": 0.2 + (team_id % 4) * 0.1,
                    "calculation_date": pd.Timestamp.now(),
                    "factors_included": '["opponent_strength", "venue", "form"]',
                }
                fixture_difficulty_list.append(fixture_difficulty_row)

    fixture_difficulty = pd.DataFrame(fixture_difficulty_list)

    # #region debug log
    try:
        with open("/Users/alex/dev/FPL/fpl-team-picker/.cursor/debug.log", "a") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "A",
                        "location": "evals/run_transfer_evals.py:create_mock_gameweek_data",
                        "message": "Fixture_difficulty created",
                        "data": {
                            "total_rows": len(fixture_difficulty),
                            "gameweeks_created": sorted(
                                fixture_difficulty["gameweek"].unique().tolist()
                            )
                            if not fixture_difficulty.empty
                            else [],
                        },
                        "timestamp": pd.Timestamp.now().isoformat(),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion

    # Create mock betting features for ML service
    # Required columns based on DerivedBettingFeaturesSchema
    # Need data for all historical gameweeks (1 to gameweek+1) to support feature engineering
    betting_features_list = []
    # Create betting features for all gameweeks from 1 to gameweek+1
    for gw in range(1, gameweek + 2):  # +2 because range is exclusive
        # Create fixtures for this gameweek (10 fixtures = 20 teams)
        for i in range(10):
            fixture_id = (gw - 1) * 100 + i + 1
            home_team_id = (i % 10) + 1
            away_team_id = ((i + 10) % 20) + 1

            # Get all players for both teams
            home_players = players_df[players_df["team_id"] == home_team_id][
                "player_id"
            ].tolist()
            away_players = players_df[players_df["team_id"] == away_team_id][
                "player_id"
            ].tolist()

            # Create betting features for all players in both teams
            for player_id in home_players + away_players:
                is_home = player_id in home_players
                team_id = home_team_id if is_home else away_team_id

                # Create realistic mock betting values
                betting_row = {
                    "gameweek": gw,
                    "fixture_id": fixture_id,
                    "player_id": player_id,
                    "is_home": is_home,
                    "team_win_probability": 0.3 + (team_id % 3) * 0.1,  # 0.3 to 0.5
                    "opponent_win_probability": 0.3
                    + ((away_team_id if is_home else home_team_id) % 3) * 0.1,
                    "draw_probability": 0.25 + (fixture_id % 2) * 0.05,  # 0.25 to 0.3
                    "implied_clean_sheet_probability": 0.3
                    + (team_id % 3) * 0.1,  # 0.3 to 0.5
                    "implied_total_goals": 2.0 + (fixture_id % 3) * 0.5,  # 2.0 to 3.5
                    "team_expected_goals": 1.0 + (team_id % 3) * 0.3,  # 1.0 to 1.9
                    "market_consensus_strength": 0.5
                    + (team_id % 2) * 0.2,  # 0.5 to 0.7
                    "odds_movement_team": (team_id % 5) - 2,  # -2 to 2
                    "odds_movement_magnitude": 0.1 + (team_id % 3) * 0.1,  # 0.1 to 0.3
                    "favorite_status": 0.3 + (team_id % 3) * 0.2,  # 0.3 to 0.7
                    "asian_handicap_line": (team_id % 3) - 1,  # -1 to 1
                    "handicap_team_odds": 1.5 + (team_id % 3) * 0.3,  # 1.5 to 2.4
                    "expected_goal_difference": (team_id % 3) - 1,  # -1 to 1
                    "over_under_signal": (fixture_id % 3) - 1,  # -1 to 1
                    "referee_encoded": fixture_id % 20,  # 0 to 19
                    "as_of_utc": pd.Timestamp.now(),
                }
                betting_features_list.append(betting_row)

    betting_features = pd.DataFrame(betting_features_list)

    return {
        "players": players_df,
        "teams": teams_df,
        "fixtures": fixtures_df,
        "current_squad": current_squad_df,
        "live_data_historical": live_data_historical,
        "ownership_trends": ownership_trends,
        "value_analysis": value_analysis,
        "fixture_difficulty": fixture_difficulty,
        "betting_features": betting_features,
        "player_metrics": pd.DataFrame(),
        "player_availability": pd.DataFrame(),
        "team_form": pd.DataFrame(),
        "players_enhanced": pd.DataFrame(),
        "xg_rates": pd.DataFrame(),
        "manager_team": {"bank": 20, "transfers": {"limit": 1}},  # £2.0m ITB, 1 FT
    }


def run_transfer_planning_eval(
    inputs: Dict[str, Any],
    model: str = "claude-sonnet-4-5",
    enable_logfire: bool = True,
) -> SingleGWRecommendation:
    """Wrapper function to run transfer planning agent for evaluation.

    This function adapts the agent service to the pydantic-evals interface.
    Note: This is sync (not async) because the agent service uses run_sync internally.

    Args:
        inputs: Input dictionary with scenario, context, expected_behavior
        model: Claude model to use
        enable_logfire: Whether to enable Logfire observability

    Returns:
        SingleGWRecommendation from agent
    """
    # Parse inputs
    scenario = inputs.get("scenario", "GW18, 1 FT, £1.0m ITB")
    context = inputs.get("context", "")

    # Extract gameweek from scenario (default to 18)
    try:
        gameweek = int(scenario.split("GW")[1].split(",")[0])
    except (IndexError, ValueError):
        gameweek = 18

    # Extract strategy mode from scenario (default to balanced)
    strategy_mode = StrategyMode.BALANCED
    if "CONSERVATIVE" in scenario.upper():
        strategy_mode = StrategyMode.CONSERVATIVE
    elif "AGGRESSIVE" in scenario.upper():
        strategy_mode = StrategyMode.AGGRESSIVE

    # Create mock gameweek data
    gameweek_data = create_mock_gameweek_data(gameweek, context)

    # Initialize agent service
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    service = TransferPlanningAgentService(
        model=model, api_key=api_key, enable_logfire=enable_logfire
    )

    # Run agent
    logger.info(f"Running eval for: {scenario}")
    recommendation = service.generate_single_gw_recommendations(
        target_gameweek=gameweek,
        current_squad=gameweek_data["current_squad"],
        gameweek_data=gameweek_data,
        strategy_mode=strategy_mode,
        hit_roi_threshold=5.0,
        num_recommendations=5,
    )

    return recommendation


async def run_evaluations(
    dataset: Dataset,
    model: str = "claude-sonnet-4-5",
    enable_logfire: bool = False,
    max_cases: int | None = None,
    use_llm_judge: bool = False,
) -> None:
    """Run evaluations on the transfer planning dataset.

    Args:
        dataset: Dataset to evaluate
        model: Claude model to use
        enable_logfire: Whether to enable Logfire observability
        max_cases: Maximum number of cases to run (None = all)
        use_llm_judge: Whether to use LLM-as-judge evaluator (slower, costs extra)
    """
    logger.info(f"Starting evaluations for dataset: {dataset.name}")
    logger.info(f"Total cases: {len(dataset.cases)}")
    logger.info(f"Model: {model}")
    logger.info(f"Logfire enabled: {enable_logfire}")
    logger.info(f"LLM judge enabled: {use_llm_judge}")

    # Limit cases if requested
    if max_cases:
        dataset = Dataset(
            cases=dataset.cases[:max_cases],
        )
        logger.info(f"Limited to {max_cases} cases")

    # Create wrapper function with model and logfire settings
    def eval_wrapper(inputs: Dict[str, Any]) -> SingleGWRecommendation:
        return run_transfer_planning_eval(inputs, model, enable_logfire)

    # Run evaluations with all evaluators
    logger.info("\n" + "=" * 80)
    logger.info("Running evaluations with all quality checks...")
    logger.info("=" * 80 + "\n")

    # Create evaluators dict for tracking
    evaluators_dict = {
        "structural_validity": StructuralValidityEvaluator(),
        "scenario_coverage": ScenarioCoverageEvaluator(),
        "strategic_quality": StrategicQualityEvaluator(),
        "hit_analysis": HitAnalysisEvaluator(),
        "ownership_strategy": OwnershipStrategyEvaluator(),
        "composite_score": CompositeTransferQualityEvaluator(),
    }

    # Add LLM judge if enabled (optional, costs extra)
    if use_llm_judge:
        evaluators_dict["llm_reasoning_quality"] = LLMReasoningQualityEvaluator(
            model="claude-haiku-4-5"
        )
        logger.info(
            "✨ LLM-as-judge evaluator enabled for reasoning quality assessment"
        )

    # Re-create dataset with evaluators
    dataset_with_evals = Dataset(
        cases=dataset.cases,
        evaluators=list(evaluators_dict.values()),
    )

    report = await dataset_with_evals.evaluate(eval_wrapper)

    # Print detailed report
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80 + "\n")

    # Print report with default settings
    report.print()

    # Print summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80 + "\n")

    logger.info("Evaluation complete. See report above for details.")
    logger.info("=" * 80 + "\n")


# Create Typer app
app = typer.Typer(
    name="fpl-transfer-evals",
    help="Run FPL transfer planning agent evaluations using Pydantic AI Evals",
    add_completion=False,
)


class DatasetChoice(str, Enum):
    """Dataset selection options."""

    ALL = "all"
    BASIC = "basic"
    STRATEGIC = "strategic"
    CONSTRAINTS = "constraints"


@app.command()
def main(
    dataset: DatasetChoice = typer.Option(
        DatasetChoice.ALL,
        "--dataset",
        "-d",
        help="Dataset to evaluate",
    ),
    model: str = typer.Option(
        "claude-sonnet-4-5",
        "--model",
        "-m",
        help="Claude model to use",
    ),
    enable_logfire: bool = typer.Option(
        True,
        "--enable-logfire",
        help="Enable Logfire observability",
    ),
    max_cases: Optional[int] = typer.Option(
        None,
        "--max-cases",
        "-n",
        help="Maximum number of cases to run (default: all)",
    ),
    use_llm_judge: bool = typer.Option(
        False,
        "--use-llm-judge",
        help="Enable LLM-as-judge for reasoning quality (slower, costs extra API calls)",
    ),
):
    """Run FPL transfer planning agent evaluations.

    Examples:

        # Run all evaluations
        $ python evals/run_transfer_evals.py

        # Run basic scenarios with Haiku model
        $ python evals/run_transfer_evals.py --dataset basic --model claude-haiku-4-5

        # Quick test with 1 case
        $ python evals/run_transfer_evals.py -d basic -n 1

        # Enable Logfire
        $ python evals/run_transfer_evals.py --enable-logfire

        # Use LLM-as-judge for subjective quality assessment
        $ python evals/run_transfer_evals.py -d basic --use-llm-judge
    """
    # Select dataset
    if dataset == DatasetChoice.BASIC:
        selected_dataset = get_basic_scenarios()
    elif dataset == DatasetChoice.STRATEGIC:
        selected_dataset = get_strategic_scenarios()
    elif dataset == DatasetChoice.CONSTRAINTS:
        selected_dataset = get_constraint_scenarios()
    else:
        selected_dataset = transfer_planning_dataset

    # Run evaluations
    asyncio.run(
        run_evaluations(
            dataset=selected_dataset,
            model=model,
            enable_logfire=enable_logfire,
            max_cases=max_cases,
            use_llm_judge=use_llm_judge,
        )
    )


if __name__ == "__main__":
    app()
