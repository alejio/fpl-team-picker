"""Transfer planning agent service using pydantic-ai for single-GW strategic planning.

This service implements an orchestrator-workers pattern following Anthropic's
"Building Effective Agents" best practices:
- Start simple: Core 5 tools + orchestrator (Phase 1)
- ACI investment: Comprehensive tool docstrings
- Thinking space: 6-step workflow in system prompt
- SA as validator: Agent reasons first, SA validates

The agent recommends transfer options for a single gameweek while considering
3 gameweeks ahead for strategic context (DGWs, fixture swings, chip timing).
"""

import logging
import os
from typing import Dict, Set

import pandas as pd
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from fpl_team_picker.config import config
from fpl_team_picker.domain.models.transfer_plan import StrategyMode
from fpl_team_picker.domain.models.transfer_recommendation import (
    SingleGWRecommendation,
)
from fpl_team_picker.domain.services.agent_tools import AGENT_TOOLS, AgentDeps

logger = logging.getLogger(__name__)


# System prompt for single-GW recommendations
SINGLE_GW_SYSTEM_PROMPT = """You are an expert FPL (Fantasy Premier League) transfer advisor.

Your task: Recommend TOP 3-5 RANKED TRANSFER OPTIONS for gameweek {target_gw}, considering the next 3 gameweeks for strategic context.

## Core Principles

1. **Single-GW Focus**: Recommend transfers for GW{target_gw} ONLY
2. **Multiple Options**: Provide 3-5 ranked scenarios (NOT a single recommendation)
3. **Always Include Hold**: Baseline option of making no transfers
4. **Strategic Context**: Look ahead to GW{target_gw}+1 and GW{target_gw}+2 for DGWs, fixture runs, chip timing
5. **SA Validation**: Use run_sa_optimizer to benchmark your top recommendations
6. **Data Accuracy**: ALWAYS use the team_name field from tool outputs when referencing player teams. NEVER rely on your training data or memory for player-team associations, as transfers happen frequently. When writing reasoning about fixtures, verify the team name matches the data you received.

## Strategy: {strategy_mode}

{strategy_specific_guidance}

## Hit Threshold

Current ROI threshold: +{roi_threshold} xP over 3 gameweeks
- Only recommend -4 hit if 3GW net ROI > threshold
- Show both hit and no-hit options for comparison

## Step-by-Step Workflow (THINK BEFORE ACTING)

Before making ANY tool calls, think through your approach:

1. ANALYZE: What are the key questions to answer?
   - Who are the weak players in current squad?
   - Are there any DGWs or fixture swings coming?
   - What's the template landscape?

2. PLAN: Which tools do I need and in what order?
   - Start with analyze_squad_weaknesses (identify targets)
   - Then get_multi_gw_xp_predictions (find replacements)
   - Then analyze_fixture_context (strategic context)
   - Then get_template_players (safety check)
   - Finally run_sa_optimizer (validate top 2-3 scenarios)

3. EXECUTE: Call tools in logical sequence

4. SYNTHESIZE: Generate 3-5 scenarios ranked by 3GW net ROI
   - When writing reasoning, ONLY reference team names from tool outputs
   - Never use training data for player-team associations

5. VALIDATE: Compare top scenarios against SA benchmark

6. RECOMMEND: Return SingleGWRecommendation with clear reasoning

## Output Requirements

Return a SingleGWRecommendation with:
- hold_option: Baseline no-transfer scenario
- recommended_scenarios: 3-5 transfer scenarios ranked by net_roi_3gw
- context_analysis: DGW opportunities, fixture swings identified
- sa_benchmark: SA optimizer results
- top_recommendation_id: Best overall option (can be "hold")
- final_reasoning: Strategic summary (2-3 sentences)

Think carefully, use tools strategically, and provide data-driven reasoning.
"""

# Strategy-specific guidance
STRATEGY_GUIDANCE = {
    StrategyMode.BALANCED: """
- Balance immediate (GW1) vs strategic (3GW) value
- Prefer free transfers, only take hits for +{roi_threshold} 3GW ROI
- Consider template safety (>30% ownership) for rank protection
- Look for fixture swings and DGW opportunities
""",
    StrategyMode.CONSERVATIVE: """
- Minimize risk: prefer hold option unless clear upgrade
- Require +{roi_threshold} 3GW ROI for any hit (higher threshold)
- Heavily favor template players (>40% ownership)
- Avoid differentials unless extremely high confidence
""",
    StrategyMode.AGGRESSIVE: """
- Maximize upside: willing to take -4 hits for +{roi_threshold} 3GW ROI (lower threshold)
- Target differentials (<10% ownership) with haul potential
- Prioritize high-ceiling players (xP + 2×uncertainty > 12)
- Front-load transfers for immediate gains
""",
}


class TransferPlanningAgentService:
    """Service for generating single-gameweek transfer recommendations using LLM reasoning."""

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

    def generate_single_gw_recommendations(
        self,
        target_gameweek: int,
        current_squad: pd.DataFrame,
        gameweek_data: Dict[str, pd.DataFrame],
        strategy_mode: StrategyMode = StrategyMode.BALANCED,
        hit_roi_threshold: float = 5.0,
        must_include_ids: Set[int] | None = None,
        must_exclude_ids: Set[int] | None = None,
        num_recommendations: int = 5,
    ) -> SingleGWRecommendation:
        """
        Generate single-GW transfer recommendations with multi-GW context awareness.

        This method uses an LLM agent with orchestrator-workers pattern to analyze
        the current squad, identify opportunities, and generate 3-5 ranked transfer
        scenarios validated against the SA optimizer.

        Args:
            target_gameweek: Gameweek to recommend for (1-38)
            current_squad: Current 15-player squad DataFrame
            gameweek_data: Dict with all required data (players, teams, fixtures, etc.)
            strategy_mode: Recommendation strategy to use
            hit_roi_threshold: Min 3GW xP gain for -4 hit (default: 5.0)
            must_include_ids: Player IDs that must stay in squad (optional)
            must_exclude_ids: Player IDs to avoid (optional)
            num_recommendations: Number of scenarios to return (3-5, default: 5)

        Returns:
            SingleGWRecommendation with ranked options + hold baseline
        """
        logger.info(
            f"🤖 Generating single-GW recommendations for GW{target_gameweek} "
            f"({strategy_mode.value} strategy, top {num_recommendations})"
        )

        # Validate inputs
        if not (3 <= num_recommendations <= 5):
            raise ValueError(
                f"num_recommendations must be 3-5, got {num_recommendations}"
            )
        if not (1 <= target_gameweek <= 38):
            raise ValueError(f"target_gameweek must be 1-38, got {target_gameweek}")
        if len(current_squad) != 15:
            raise ValueError(
                f"current_squad must have 15 players, got {len(current_squad)}"
            )

        # Extract free transfers and budget from manager data
        budget = gameweek_data.get("manager_team", {}).get("bank", 0.0) / 10.0
        free_transfers = (
            gameweek_data.get("manager_team", {}).get("transfers", {}).get("limit", 1)
        )

        # 1. Prepare agent dependencies (includes free_transfers and budget)
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
            free_transfers=free_transfers,
            budget_available=budget,
        )

        # 2. Build system prompt with strategy guidance
        strategy_guidance = STRATEGY_GUIDANCE.get(strategy_mode, "")
        system_prompt = SINGLE_GW_SYSTEM_PROMPT.format(
            target_gw=target_gameweek,
            strategy_mode=strategy_mode.value,
            strategy_specific_guidance=strategy_guidance,
            roi_threshold=hit_roi_threshold,
        )

        # 3. Initialize agent with Anthropic model
        provider = AnthropicProvider(api_key=self.api_key)
        agent = Agent(
            model=AnthropicModel(self.model_name, provider=provider),
            system_prompt=system_prompt,
            deps_type=AgentDeps,
        )

        # 4. Register all 5 tools
        for tool in AGENT_TOOLS:
            agent.tool(tool)

        # 5. Build user prompt
        squad_summary = self._format_squad(current_squad)

        user_prompt = f"""Generate transfer recommendations for GW{target_gameweek}.

Current Squad:
{squad_summary}

Budget: £{budget:.1f}m in the bank
Free Transfers: {free_transfers}

Instructions:
1. Think through your approach before calling tools
2. Use tools in logical order (weaknesses → xP → context → templates → SA)
3. Generate 5-7 candidate scenarios (hold + various transfers)
4. Validate top 3 scenarios with run_sa_optimizer
5. Rank by 3GW net ROI and return top {num_recommendations}

Remember: Think step-by-step, use tools strategically, validate with SA.
"""

        if must_include_ids:
            user_prompt += (
                f"\nMust keep: {', '.join(str(id) for id in must_include_ids)}"
            )
        if must_exclude_ids:
            user_prompt += f"\nAvoid: {', '.join(str(id) for id in must_exclude_ids)}"

        # 6. Run agent
        logger.info("🧠 Agent analyzing transfer options...")
        result = agent.run_sync(
            user_prompt, deps=deps, output_type=SingleGWRecommendation
        )

        logger.info(
            f"✅ Recommendations generated: {len(result.output.recommended_scenarios)} scenarios, "
            f"top pick = {result.output.top_recommendation_id}"
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
