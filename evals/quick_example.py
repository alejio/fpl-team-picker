"""Quick example: Run a single transfer planning eval.

This demonstrates the basic eval workflow with a single test case.

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    uv run python evals/quick_example.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from pydantic_evals import Case

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evals.evaluators.transfer_quality import (  # noqa: E402
    CompositeTransferQualityEvaluator,
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


def create_mock_data() -> Dict[str, pd.DataFrame]:
    """Create minimal mock data for the example."""
    players_df = pd.DataFrame(
        {
            "player_id": list(range(1, 51)),
            "web_name": [f"Player{i}" for i in range(1, 51)],
            "name": [f"Team{i % 10}" for i in range(1, 51)],
            "position": (["GKP"] * 5 + ["DEF"] * 15 + ["MID"] * 20 + ["FWD"] * 10),
            "price": [5.0 + (i % 8) for i in range(1, 51)],
            "selected_by_percent": [10.0 + (i % 40) for i in range(1, 51)],
            "team_id": [(i % 10) + 1 for i in range(1, 51)],
        }
    )

    # Create minimal historical live data for ML predictions
    # Start from GW1 for full history
    historical_data = []
    for player_id in range(1, 51):
        for gw in range(1, 18):  # GW1-17
            historical_data.append(
                {
                    "player_id": player_id,
                    "gameweek": gw,
                    "minutes": 60 + (player_id % 30),
                    "total_points": 2 + (player_id % 8),
                    "goals_scored": (player_id % 10) // 5,
                    "assists": (player_id % 8) // 4,
                }
            )

    live_data_historical = pd.DataFrame(historical_data)

    # Create minimal ownership trends data for ML feature engineering
    # Start from GW1, extend beyond current for predictions
    ownership_data = []
    for player_id in range(1, 51):
        for gw in range(1, 28):  # GW1-27 (18+10)
            ownership_data.append(
                {
                    "player_id": player_id,
                    "gameweek": gw,
                    "selected_by_percent": 10.0 + (player_id % 40),
                    "net_transfers_gw": (player_id % 10) - 5,
                    "transfer_momentum": 0.5 + (player_id % 3) * 0.2,
                    "ownership_tier": "mid",
                    "bandwagon_score": 0.3 + (player_id % 5) * 0.1,
                    "ownership_vs_price": 1.0 + (player_id % 7) * 0.1,
                }
            )

    ownership_trends = pd.DataFrame(ownership_data)

    return {
        "players": players_df,
        "teams": pd.DataFrame(
            {"team_id": list(range(1, 11)), "name": [f"Team{i}" for i in range(1, 11)]}
        ),
        "fixtures": pd.DataFrame(
            {
                "id": [1, 2],
                "event": [18, 18],
                "home_team_id": [1, 2],
                "away_team_id": [3, 4],
            }
        ),
        "current_squad": players_df.head(15),
        "live_data_historical": live_data_historical,
        "ownership_trends": ownership_trends,
        "value_analysis": pd.DataFrame(),
        "fixture_difficulty": pd.DataFrame(),
        "betting_features": pd.DataFrame(),
        "player_metrics": pd.DataFrame(),
        "player_availability": pd.DataFrame(),
        "team_form": pd.DataFrame(),
        "players_enhanced": pd.DataFrame(),
        "xg_rates": pd.DataFrame(),
        "manager_team": {"bank": 20, "transfers": {"limit": 1}},
    }


async def run_single_eval() -> SingleGWRecommendation:
    """Run agent on the example scenario."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    service = TransferPlanningAgentService(
        model="claude-sonnet-4-5", api_key=api_key, debug=False
    )

    gameweek_data = create_mock_data()

    logger.info("Running transfer planning agent...")
    recommendation = service.generate_single_gw_recommendations(
        target_gameweek=18,
        current_squad=gameweek_data["current_squad"],
        gameweek_data=gameweek_data,
        strategy_mode=StrategyMode.BALANCED,
        hit_roi_threshold=5.0,
        num_recommendations=5,
    )

    return recommendation


async def main():
    """Run a simple eval example."""
    logger.info("=" * 80)
    logger.info("Transfer Planning Agent - Quick Eval Example")
    logger.info("=" * 80 + "\n")

    # Create a simple test case
    test_case = Case(
        name="premium_upgrade_example",
        inputs={
            "scenario": "GW18, 1 FT, £2.0m ITB",
            "context": "Considering premium upgrade",
            "expected_behavior": "Provide 3-5 ranked options with hold baseline",
        },
        expected_output={
            "should_include_hold": True,
            "min_scenarios": 3,
            "max_scenarios": 5,
        },
    )

    # Run evaluation
    logger.info("Running agent on test case...")
    result = await run_single_eval()

    # Manual evaluation (instead of dataset.evaluate for clarity)
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80 + "\n")

    # Check structural validity
    structural_eval = StructuralValidityEvaluator()
    from pydantic_evals.evaluators import EvaluatorContext

    ctx = EvaluatorContext(
        case=test_case,
        inputs=test_case.inputs,
        output=result,
        expected_output=test_case.expected_output,
    )

    structural_score = await structural_eval.evaluate(ctx)
    coverage_eval = ScenarioCoverageEvaluator()
    coverage_score = await coverage_eval.evaluate(ctx)
    strategic_eval = StrategicQualityEvaluator()
    strategic_score = await strategic_eval.evaluate(ctx)
    composite_eval = CompositeTransferQualityEvaluator()
    composite_score = await composite_eval.evaluate(ctx)

    logger.info(f"Structural Validity: {structural_score:.2%}")
    logger.info(f"Scenario Coverage:   {coverage_score:.2%}")
    logger.info(f"Strategic Quality:   {strategic_score:.2%}")
    logger.info(f"Composite Score:     {composite_score:.2%}")

    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATION SUMMARY")
    logger.info("=" * 80 + "\n")

    logger.info(f"Target Gameweek: {result.target_gameweek}")
    logger.info(f"Free Transfers: {result.current_free_transfers}")
    logger.info(f"Budget Available: £{result.budget_available:.1f}m")
    logger.info(f"\nTotal Scenarios: {len(result.recommended_scenarios)}")
    logger.info(f"Top Recommendation: {result.top_recommendation_id}")
    logger.info(f"\nFinal Reasoning:\n{result.final_reasoning}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Evaluation complete!")
    logger.info("=" * 80 + "\n")

    logger.info("To run full eval suite:")
    logger.info("  uv run python evals/run_transfer_evals.py")


if __name__ == "__main__":
    asyncio.run(main())
