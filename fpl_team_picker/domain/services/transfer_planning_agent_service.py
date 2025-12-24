"""Transfer planning agent service using pydantic-ai for multi-GW strategic planning."""

import logging
import os
from typing import Dict

import pandas as pd
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from fpl_team_picker.config import config
from fpl_team_picker.domain.models.transfer_plan import (
    MultiGWPlan,
    StrategyMode,
)
from fpl_team_picker.domain.services.agent_tools import (
    AGENT_TOOLS,
    AgentDeps,
)

logger = logging.getLogger(__name__)


# System prompts for different strategy modes
SYSTEM_PROMPTS = {
    StrategyMode.BALANCED: """You are an expert FPL (Fantasy Premier League) transfer planner with deep knowledge of player performance, fixtures, and strategic planning.

Your task is to analyze the next {horizon} gameweeks and create a multi-gameweek transfer plan that maximizes expected points (xP) while managing risk.

Strategy: BALANCED
- Prefer free transfers over hits unless ROI clearly exceeds {roi_threshold} points
- Consider both high-ceiling players (haul potential) and consistent performers
- Factor in fixture difficulty - prioritize players with favorable upcoming fixtures
- Be aware of template players (>30% ownership) for safety
- Use expected points + uncertainty to identify haul opportunities

Key Principles:
1. Analyze xP predictions across all gameweeks in the horizon
2. Plan sequential transfers - consider banking free transfers (FT) for better opportunities
3. Maximize total xP minus hit costs over the planning horizon
4. Provide clear, data-driven reasoning for each decision
5. Consider the full context: budget, squad composition, upcoming fixtures

Available Tools:
- get_multi_gw_xp_predictions: Get xP projections for all players across multiple gameweeks

Output Format:
Return a structured plan with:
- Week-by-week transfer recommendations (or "hold FT" if banking)
- Expected xP for each week with and without transfers
- Clear reasoning for each decision
- Summary of total xP gain, hit costs, and net ROI
- Identification of key opportunities (good fixtures, fixture swings, etc.)
""",
    StrategyMode.CONSERVATIVE: """You are a conservative FPL transfer planner focused on minimizing risk and protecting rank.

Strategy: CONSERVATIVE
- Rarely take hits unless absolutely necessary for injury/suspension coverage
- Prefer template players (>40% ownership) to avoid rank drops
- Prioritize defensive players and goalkeepers with good fixtures
- Avoid differential picks unless extremely high confidence
- Bank free transfers when no clear opportunities exist

ROI Threshold: Only take -4 hit if expected gain > {roi_threshold} points (conservative: 6+)
""",
    StrategyMode.AGGRESSIVE: """You are an aggressive FPL transfer planner focused on rank climbing through differentials.

Strategy: AGGRESSIVE
- Willing to take multiple hits (-4, -8) for high-upside moves
- Target differential players (<10% ownership) with ceiling potential
- Prioritize attacking players (forwards, midfielders) over defenders
- Use uncertainty/variance to identify explosive haul candidates
- Front-load transfers for immediate gains rather than banking

ROI Threshold: Take -4 hit if expected gain > {roi_threshold} points (aggressive: 4+)
""",
    StrategyMode.DGW_STACKER: """You are a DGW (Double Gameweek) specialist focused on maximizing points in weeks where teams play twice.

Strategy: DGW STACKER
- Identify double gameweeks in the planning horizon
- Plan multi-week buildup to reach 3 players from DGW teams
- Prioritize players with highest DGW xP (playing twice = 2x points)
- Bank free transfers strategically to have 2 FTs available for final DGW week
- Consider bench boost chip timing if DGW detected

DGW Planning:
1. Use tools to detect any DGWs in horizon
2. Calculate xP for DGW teams (sum of both fixtures)
3. Plan progressive transfers: Week 1: +1 DGW player, Week 2: +2 more
4. Ensure squad has 3 players from top DGW teams before the double gameweek
""",
}


class TransferPlanningAgentService:
    """Service for generating multi-gameweek transfer plans using LLM reasoning."""

    def __init__(
        self, model: str | None = None, api_key: str | None = None, debug: bool = False
    ):
        """
        Initialize transfer planning agent service.

        Args:
            model: Anthropic model to use (default: from config)
            api_key: Anthropic API key (default: from config or ANTHROPIC_API_KEY env var)
            debug: Enable debug logging
        """
        self.model_name = model or config.transfer_planning_agent.model

        # API key priority: param > config > env var
        self.api_key = (
            api_key
            or config.transfer_planning_agent.api_key
            or os.getenv("ANTHROPIC_API_KEY")
        )

        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Set it via:\n"
                "  1. ANTHROPIC_API_KEY environment variable, or\n"
                "  2. FPL_TRANSFER_PLANNING_AGENT_API_KEY environment variable, or\n"
                "  3. config.json: transfer_planning_agent.api_key, or\n"
                "  4. Pass api_key parameter to TransferPlanningAgentService()"
            )

        self.debug = debug

        if debug:
            logger.setLevel(logging.DEBUG)

    def generate_multi_gw_plan(
        self,
        start_gameweek: int,
        planning_horizon: int,
        current_squad: pd.DataFrame,
        gameweek_data: Dict[str, pd.DataFrame],
        strategy_mode: StrategyMode = StrategyMode.BALANCED,
        hit_roi_threshold: float = 5.0,
        must_include: list[str] | None = None,
        must_exclude: list[str] | None = None,
    ) -> MultiGWPlan:
        """
        Generate multi-gameweek transfer plan using LLM agent.

        Args:
            start_gameweek: Starting gameweek number
            planning_horizon: Number of gameweeks to plan (1, 3, or 5)
            current_squad: DataFrame with current 15-player squad
            gameweek_data: Dict containing all required data (players, teams, fixtures, etc.)
            strategy_mode: Planning strategy to use
            hit_roi_threshold: Minimum xP gain required for taking -4 hit
            must_include: Player names that must stay in squad
            must_exclude: Player names to avoid transferring in

        Returns:
            MultiGWPlan with weekly recommendations and reasoning
        """
        logger.info(
            f"🤖 Generating {planning_horizon}-GW plan for GW{start_gameweek} ({strategy_mode.value} strategy)"
        )

        # Prepare agent dependencies (data to inject into tools)
        deps = AgentDeps(
            players_data=gameweek_data["players"],
            teams_data=gameweek_data["teams"],
            fixtures_data=gameweek_data["fixtures"],
            live_data=gameweek_data.get("live_data_historical", pd.DataFrame()),
            ownership_trends=gameweek_data.get("ownership_trends"),
            value_analysis=gameweek_data.get("value_analysis"),
            fixture_difficulty=gameweek_data.get("fixture_difficulty"),
            betting_features=gameweek_data.get("betting_features"),
            player_metrics=gameweek_data.get("player_metrics"),
            player_availability=gameweek_data.get("player_availability"),
            team_form=gameweek_data.get("team_form"),
            players_enhanced=gameweek_data.get("players_enhanced"),
            xg_rates=gameweek_data.get("xg_rates"),
        )

        # Get system prompt for strategy
        system_prompt = SYSTEM_PROMPTS[strategy_mode].format(
            horizon=planning_horizon,
            roi_threshold=hit_roi_threshold,
        )

        # Initialize agent with Anthropic model and tools
        # Create provider with API key, then pass to model
        provider = AnthropicProvider(api_key=self.api_key)
        agent = Agent(
            model=AnthropicModel(self.model_name, provider=provider),
            system_prompt=system_prompt,
            deps_type=AgentDeps,
        )

        # Register tools
        for tool in AGENT_TOOLS:
            agent.tool(tool)

        # Build user prompt with current context
        squad_summary = self._format_squad(current_squad)
        budget = (
            gameweek_data.get("manager_team", {}).get("bank", 0.0) / 10.0
        )  # Convert to millions
        free_transfers = (
            gameweek_data.get("manager_team", {}).get("transfers", {}).get("limit", 1)
        )

        user_prompt = f"""Generate a {planning_horizon}-gameweek transfer plan for GW{start_gameweek}-{start_gameweek + planning_horizon - 1}.

Current Squad:
{squad_summary}

Budget: £{budget:.1f}m in the bank
Free Transfers: {free_transfers}
Hit ROI Threshold: Require +{hit_roi_threshold} xP gain to justify -4 hit

Instructions:
1. Use get_multi_gw_xp_predictions to analyze player projections
2. Identify key opportunities (fixture runs, potential hauls)
3. Plan week-by-week transfers considering:
   - When to make transfers vs. hold FT
   - Hit costs vs. xP gains
   - Building toward better future gameweeks
4. Return a complete MultiGWPlan with all required fields

Think step-by-step and explain your reasoning clearly.
"""

        if must_include:
            user_prompt += f"\nMust keep: {', '.join(must_include)}"
        if must_exclude:
            user_prompt += f"\nAvoid: {', '.join(must_exclude)}"

        # Run agent
        logger.info("🧠 Agent thinking...")
        result = agent.run_sync(user_prompt, deps=deps, output_type=MultiGWPlan)

        logger.info(
            f"✅ Plan generated: {result.output.total_transfers} transfers, ROI={result.output.net_roi:.1f}"
        )

        return result.output

    def _format_squad(self, squad_df: pd.DataFrame) -> str:
        """Format current squad for display in prompt."""
        if squad_df.empty:
            return "No current squad data available"

        lines = []
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            pos_players = squad_df[squad_df["position"] == pos]
            if not pos_players.empty:
                players_str = ", ".join(
                    f"{row['web_name']} ({row['name']}, £{row['price']:.1f}m)"
                    for _, row in pos_players.iterrows()
                )
                lines.append(f"{pos}: {players_str}")

        return "\n".join(lines)
