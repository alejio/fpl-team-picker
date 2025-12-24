#!/usr/bin/env python3
"""Multi-GW Transfer Planning Agent CLI

Test script for the LLM-based multi-gameweek transfer planning agent.

Usage:
    python scripts/multi_gw_agent.py --gameweek 18 --horizon 3 --strategy balanced

Dependencies:
    - typer (install with: uv add typer)
    - loguru (install with: uv add loguru)
"""

from fpl_team_picker.domain.services import (
    DataOrchestrationService,
)
from fpl_team_picker.domain.services.transfer_planning_agent_service import (
    TransferPlanningAgentService,
)
from fpl_team_picker.domain.models.transfer_plan import StrategyMode

import typer
from pathlib import Path
import sys
from loguru import logger

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Reuse data loading utilities
sys.path.insert(0, str(Path(__file__).parent))

app = typer.Typer(
    help="Multi-GW Transfer Planning Agent",
    add_completion=True,
)


@app.command()
def main(
    gameweek: int = typer.Option(18, help="Gameweek to plan for"),
    horizon: int = typer.Option(3, help="Planning horizon"),
    strategy: str = typer.Option("balanced", help="Planning strategy"),
):
    logger.info("🤖 Multi-GW Transfer Planning Agent\n")

    # 1. Load data
    logger.info(f"Loading GW{gameweek} data...")
    data_service = DataOrchestrationService()
    gw_data = data_service.load_gameweek_data(gameweek)
    logger.info(f"✅ Data loaded: {len(gw_data['players'])} players\n")

    # 2. Run agent
    logger.info(f"Generating {horizon}-GW plan ({strategy} strategy)...")
    agent_service = TransferPlanningAgentService(model="claude-sonnet-4-5")

    plan = agent_service.generate_multi_gw_plan(
        start_gameweek=gameweek,
        planning_horizon=horizon,
        current_squad=gw_data["current_squad"],
        strategy_mode=StrategyMode(strategy),
        gameweek_data=gw_data,
    )

    # 3. Print results
    logger.info(
        f"\n📅 Recommended Plan (GW{plan.start_gameweek}-{plan.end_gameweek})\n"
    )

    for weekly_plan in plan.weekly_plans:
        logger.info(
            f"Week {weekly_plan.gameweek - plan.start_gameweek + 1}: GW{weekly_plan.gameweek}"
        )
        if weekly_plan.transfers:
            for t in weekly_plan.transfers:
                logger.info(
                    f"  Transfer: {t.player_out_name} → {t.player_in_name} (£{t.cost:+.1f}m)"
                )
        else:
            logger.info("  Transfers: Hold FT")
        logger.info(
            f"  Expected xP: {weekly_plan.expected_xp:.1f} (baseline: {weekly_plan.baseline_xp:.1f})"
        )
        logger.info(f"  Reasoning: {weekly_plan.reasoning}\n")

    logger.info("Summary:")
    logger.info(f"  Total xP Gain: +{plan.total_xp_gain:.1f} points")
    logger.info(f"  Total Hits: {plan.total_hit_cost}")
    logger.info(f"  Net ROI: +{plan.net_roi:.1f} points")


if __name__ == "__main__":
    app()
