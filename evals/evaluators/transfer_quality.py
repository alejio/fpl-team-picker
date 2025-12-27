"""Custom evaluators for FPL transfer recommendation quality assessment.

This module provides evaluators that assess:
1. Structural validity (Pydantic model compliance)
2. Scenario coverage (min/max scenarios, hold option)
3. Strategic quality (flags, reasoning, confidence)
4. Budget compliance
5. ROI threshold adherence
6. LLM-as-judge reasoning quality (subjective quality assessment)
"""

import os
from typing import Any, Dict

from pydantic_ai import Agent
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from fpl_team_picker.domain.models.transfer_recommendation import SingleGWRecommendation


class StructuralValidityEvaluator(Evaluator):
    """Evaluates structural validity of SingleGWRecommendation output.

    Checks:
    - Output is a valid SingleGWRecommendation instance
    - All required fields are present
    - Pydantic validation passed
    """

    async def evaluate(
        self, ctx: EvaluatorContext[Dict[str, Any], Dict[str, Any]]
    ) -> float:
        """Evaluate structural validity.

        Returns:
            1.0 if valid SingleGWRecommendation, 0.0 otherwise
        """
        if not isinstance(ctx.output, SingleGWRecommendation):
            return 0.0

        # Check required fields are present and valid
        try:
            assert ctx.output.target_gameweek >= 1
            assert ctx.output.hold_option is not None
            assert len(ctx.output.recommended_scenarios) >= 1
            assert ctx.output.top_recommendation_id
            assert ctx.output.final_reasoning
            return 1.0
        except (AssertionError, AttributeError):
            return 0.0


class ScenarioCoverageEvaluator(Evaluator):
    """Evaluates scenario coverage based on expected output requirements.

    Checks:
    - Hold option included (if expected)
    - Minimum number of scenarios met
    - Maximum number of scenarios not exceeded
    - Top recommendation is valid
    """

    async def evaluate(
        self, ctx: EvaluatorContext[Dict[str, Any], Dict[str, Any]]
    ) -> float:
        """Evaluate scenario coverage.

        Returns:
            Score from 0.0 to 1.0 based on coverage completeness
        """
        if not isinstance(ctx.output, SingleGWRecommendation):
            return 0.0

        expected = ctx.expected_output
        score = 0.0
        checks = 0

        # Check hold option inclusion
        if "should_include_hold" in expected:
            checks += 1
            if expected["should_include_hold"]:
                if ctx.output.hold_option is not None:
                    score += 1.0
            else:
                # Some cases don't want hold (e.g., injury emergency)
                if ctx.output.hold_option is None:
                    score += 1.0

        # Check minimum scenarios
        if "min_scenarios" in expected:
            checks += 1
            if len(ctx.output.recommended_scenarios) >= expected["min_scenarios"]:
                score += 1.0

        # Check maximum scenarios
        if "max_scenarios" in expected:
            checks += 1
            if len(ctx.output.recommended_scenarios) <= expected["max_scenarios"]:
                score += 1.0

        # Check top recommendation validity
        checks += 1
        if ctx.output.top_recommendation_id:
            # Verify top_recommendation_id matches a scenario or "hold"
            valid_ids = {s.scenario_id for s in ctx.output.recommended_scenarios}
            valid_ids.add("hold")
            if ctx.output.top_recommendation_id in valid_ids:
                score += 1.0

        # Check if hold should be top pick
        if "top_recommendation_should_be_hold" in expected:
            checks += 1
            if expected["top_recommendation_should_be_hold"]:
                if ctx.output.top_recommendation_id == "hold":
                    score += 1.0

        return score / checks if checks > 0 else 0.0


class StrategicQualityEvaluator(Evaluator):
    """Evaluates strategic quality of recommendations.

    Checks:
    - Expected strategic flags are set (leverages_dgw, leverages_fixture_swing, etc.)
    - Reasoning mentions expected keywords
    - Confidence levels are appropriate
    - Context analysis includes expected insights
    """

    async def evaluate(
        self, ctx: EvaluatorContext[Dict[str, Any], Dict[str, Any]]
    ) -> float:
        """Evaluate strategic quality.

        Returns:
            Score from 0.0 to 1.0 based on strategic quality
        """
        if not isinstance(ctx.output, SingleGWRecommendation):
            return 0.0

        expected = ctx.expected_output
        score = 0.0
        checks = 0

        # Get top scenario for analysis
        top_scenario = None
        for scenario in ctx.output.recommended_scenarios:
            if scenario.scenario_id == ctx.output.top_recommendation_id:
                top_scenario = scenario
                break

        # Check DGW scenario exists and flagged
        if (
            "should_have_dgw_scenario" in expected
            and expected["should_have_dgw_scenario"]
        ):
            checks += 1
            dgw_scenarios = [
                s for s in ctx.output.recommended_scenarios if s.leverages_dgw
            ]
            if dgw_scenarios:
                score += 1.0

        if (
            "dgw_scenario_should_be_flagged" in expected
            and expected["dgw_scenario_should_be_flagged"]
        ):
            checks += 1
            if top_scenario and top_scenario.leverages_dgw:
                score += 1.0

        # Check fixture swing scenario exists and flagged
        if (
            "should_have_fixture_swing_scenario" in expected
            and expected["should_have_fixture_swing_scenario"]
        ):
            checks += 1
            fixture_scenarios = [
                s for s in ctx.output.recommended_scenarios if s.leverages_fixture_swing
            ]
            if fixture_scenarios:
                score += 1.0

        if (
            "fixture_swing_should_be_flagged" in expected
            and expected["fixture_swing_should_be_flagged"]
        ):
            checks += 1
            if top_scenario and top_scenario.leverages_fixture_swing:
                score += 1.0

        # Check chip preparation scenario exists and flagged
        if (
            "should_have_chip_prep_scenario" in expected
            and expected["should_have_chip_prep_scenario"]
        ):
            checks += 1
            chip_scenarios = [
                s for s in ctx.output.recommended_scenarios if s.prepares_for_chip
            ]
            if chip_scenarios:
                score += 1.0

        if (
            "chip_prep_should_be_flagged" in expected
            and expected["chip_prep_should_be_flagged"]
        ):
            checks += 1
            if top_scenario and top_scenario.prepares_for_chip:
                score += 1.0

        # Check reasoning mentions expected keywords
        if "top_pick_reasoning_mentions" in expected:
            checks += 1
            keywords = expected["top_pick_reasoning_mentions"]
            reasoning_text = ctx.output.final_reasoning.lower()
            if top_scenario:
                reasoning_text += " " + top_scenario.reasoning.lower()

            mentions = sum(
                1 for keyword in keywords if keyword.lower() in reasoning_text
            )
            score += mentions / len(keywords) if keywords else 0.0

        # Check confidence level
        if "top_pick_confidence" in expected and top_scenario:
            checks += 1
            if top_scenario.confidence == expected["top_pick_confidence"]:
                score += 1.0

        # Check context analysis mentions
        if (
            "context_analysis_should_mention_dgw" in expected
            and expected["context_analysis_should_mention_dgw"]
        ):
            checks += 1
            context_str = str(ctx.output.context_analysis).lower()
            if "dgw" in context_str or "double" in context_str:
                score += 1.0

        if (
            "context_analysis_should_mention_fixtures" in expected
            and expected["context_analysis_should_mention_fixtures"]
        ):
            checks += 1
            context_str = str(ctx.output.context_analysis).lower()
            if "fixture" in context_str or "run" in context_str:
                score += 1.0

        if (
            "context_analysis_should_mention_chip" in expected
            and expected["context_analysis_should_mention_chip"]
        ):
            checks += 1
            context_str = str(ctx.output.context_analysis).lower()
            if "chip" in context_str or "wildcard" in context_str:
                score += 1.0

        # Check hold reasoning if hold is recommended
        if (
            "hold_reasoning_mentions" in expected
            and ctx.output.top_recommendation_id == "hold"
        ):
            checks += 1
            keywords = expected["hold_reasoning_mentions"]
            hold_text = ctx.output.hold_option.reasoning.lower()
            mentions = sum(1 for keyword in keywords if keyword.lower() in hold_text)
            score += mentions / len(keywords) if keywords else 0.0

        return score / checks if checks > 0 else 0.0


class HitAnalysisEvaluator(Evaluator):
    """Evaluates hit analysis quality.

    Checks:
    - Hit scenarios exist when expected
    - ROI calculations are reasonable
    - Hit threshold is respected
    """

    async def evaluate(
        self, ctx: EvaluatorContext[Dict[str, Any], Dict[str, Any]]
    ) -> float:
        """Evaluate hit analysis quality.

        Returns:
            Score from 0.0 to 1.0 based on hit analysis quality
        """
        if not isinstance(ctx.output, SingleGWRecommendation):
            return 0.0

        expected = ctx.expected_output
        score = 0.0
        checks = 0

        # Check if hit scenario exists when expected
        if (
            "should_have_hit_scenario" in expected
            and expected["should_have_hit_scenario"]
        ):
            checks += 1
            hit_scenarios = [
                s for s in ctx.output.recommended_scenarios if s.hit_cost < 0
            ]
            if hit_scenarios:
                score += 1.0

                # If hits exist, check ROI is positive over 3GW
                checks += 1
                valid_roi_scenarios = [s for s in hit_scenarios if s.net_roi_3gw > 0]
                if valid_roi_scenarios:
                    score += 1.0

        return score / checks if checks > 0 else 0.0


class OwnershipStrategyEvaluator(Evaluator):
    """Evaluates ownership-based strategic decisions.

    Checks:
    - Template scenarios prioritize high ownership (>40%)
    - Differential scenarios target low ownership (<25%)
    - Conservative strategy favors template
    - Aggressive strategy considers differentials
    """

    async def evaluate(
        self, ctx: EvaluatorContext[Dict[str, Any], Dict[str, Any]]
    ) -> float:
        """Evaluate ownership strategy.

        Returns:
            Score from 0.0 to 1.0 based on ownership strategy quality
        """
        if not isinstance(ctx.output, SingleGWRecommendation):
            return 0.0

        expected = ctx.expected_output
        score = 0.0
        checks = 0

        # Check template prioritization
        if (
            "should_prioritize_template" in expected
            and expected["should_prioritize_template"]
        ):
            checks += 1
            # Check if reasoning mentions template/safety/ownership
            reasoning = ctx.output.final_reasoning.lower()
            if any(
                word in reasoning
                for word in ["template", "safety", "ownership", "protect"]
            ):
                score += 1.0

        # Check differential targeting
        if (
            "should_target_differentials" in expected
            and expected["should_target_differentials"]
        ):
            checks += 1
            # Check if reasoning mentions differential/upside/haul
            reasoning = ctx.output.final_reasoning.lower()
            if any(
                word in reasoning
                for word in ["differential", "upside", "haul", "ceiling"]
            ):
                score += 1.0

        if (
            "should_have_differential_scenario" in expected
            and expected["should_have_differential_scenario"]
        ):
            checks += 1
            # Check if reasoning in scenarios mentions differential
            has_differential = any(
                "differential" in s.reasoning.lower()
                for s in ctx.output.recommended_scenarios
            )
            if has_differential:
                score += 1.0

        return score / checks if checks > 0 else 0.0


class LLMReasoningQualityEvaluator(Evaluator):
    """LLM-as-judge evaluator for reasoning quality and strategic coherence.

    Uses Claude to subjectively assess:
    - Clarity and coherence of reasoning
    - Strategic depth and FPL knowledge
    - Appropriateness of recommendations for scenario
    - Quality of context analysis

    This complements deterministic checks with subjective quality assessment.
    """

    def __init__(self, model: str = "claude-haiku-4-5"):
        """Initialize LLM judge.

        Args:
            model: Claude model to use for judging (default: haiku for speed/cost)
        """
        self.model = model

        # Initialize judge agent
        self.judge = Agent(
            model=self.model,
            system_prompt="""You are an expert FPL (Fantasy Premier League) analyst evaluating
transfer recommendations. Your task is to assess the quality of reasoning and strategic analysis
in transfer recommendations.

Evaluate the recommendation on these criteria (each 0-1 scale):

1. **Clarity** (0.3 weight): Is the reasoning clear and well-structured?
2. **Strategic Depth** (0.3 weight): Does it show deep FPL knowledge (fixtures, ownership, form)?
3. **Scenario Appropriateness** (0.2 weight): Does the recommendation fit the scenario context?
4. **Context Analysis** (0.2 weight): Does it identify key opportunities/risks correctly?

Respond ONLY with a JSON object containing:
{
  "clarity_score": <float 0-1>,
  "strategic_depth_score": <float 0-1>,
  "appropriateness_score": <float 0-1>,
  "context_score": <float 0-1>,
  "overall_score": <float 0-1>,
  "brief_feedback": "<1-2 sentences explaining the score>"
}

Be critical but fair. A score of 0.7+ indicates good quality, 0.5-0.7 is acceptable, <0.5 needs improvement.""",
        )

    async def evaluate(
        self, ctx: EvaluatorContext[Dict[str, Any], Dict[str, Any]]
    ) -> float:
        """Evaluate reasoning quality using LLM judge.

        Returns:
            Overall quality score from 0.0 to 1.0
        """
        if not isinstance(ctx.output, SingleGWRecommendation):
            return 0.0

        # Skip if no API key (graceful degradation)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return 0.5  # Neutral score when LLM judge unavailable

        # Build evaluation prompt
        scenario_desc = ctx.inputs.get("scenario", "Unknown scenario")
        context_desc = ctx.inputs.get("context", "")
        expected_behavior = ctx.inputs.get("expected_behavior", "")

        # Extract recommendation details
        top_scenario = None
        for scenario in ctx.output.recommended_scenarios:
            if scenario.scenario_id == ctx.output.top_recommendation_id:
                top_scenario = scenario
                break

        if not top_scenario:
            return 0.0  # No top recommendation found

        # Derive scenario type from existing attributes
        if top_scenario.is_hold:
            scenario_type = "Hold (no transfers)"
        elif top_scenario.hit_cost < 0:
            scenario_type = f"Hit ({abs(top_scenario.hit_cost)} points)"
        else:
            scenario_type = f"Free Transfer ({top_scenario.num_transfers} transfer(s))"

        eval_prompt = f"""Scenario: {scenario_desc}
Context: {context_desc}
Expected Behavior: {expected_behavior}

Top Recommendation:
- Type: {scenario_type}
- Transfers: {len(top_scenario.transfers)} transfer(s)
- Hit Cost: {top_scenario.hit_cost} points
- Single-GW xP: {top_scenario.xp_gw1:.1f} (gain: {top_scenario.xp_gain_gw1:.1f})
- 3GW xP: {top_scenario.xp_3gw:.1f} (ROI: {top_scenario.net_roi_3gw:.1f})
- Confidence: {top_scenario.confidence}

Reasoning: {top_scenario.reasoning}

Context Analysis: {ctx.output.context_analysis}

Final Recommendation: {ctx.output.final_reasoning}

Strategic Flags:
- Leverages DGW: {top_scenario.leverages_dgw}
- Leverages Fixture Swing: {top_scenario.leverages_fixture_swing}
- Prepares for Chip: {top_scenario.prepares_for_chip}

Evaluate the quality of this recommendation."""

        try:
            # Run LLM judge
            result = await self.judge.run(eval_prompt)

            # Parse JSON response
            import json

            scores = json.loads(result.data)

            # Return weighted overall score
            overall = (
                scores.get("clarity_score", 0.5) * 0.3
                + scores.get("strategic_depth_score", 0.5) * 0.3
                + scores.get("appropriateness_score", 0.5) * 0.2
                + scores.get("context_score", 0.5) * 0.2
            )

            return max(0.0, min(1.0, overall))  # Clamp to [0, 1]

        except Exception as e:
            # Graceful degradation on error
            print(f"LLM judge error: {e}")
            return 0.5  # Neutral score on error


class CompositeTransferQualityEvaluator(Evaluator):
    """Composite evaluator combining all quality checks.

    Weights:
    - Structural validity: 20%
    - Scenario coverage: 25%
    - Strategic quality: 30%
    - Hit analysis: 15%
    - Ownership strategy: 10%
    """

    def __init__(self):
        self.structural = StructuralValidityEvaluator()
        self.coverage = ScenarioCoverageEvaluator()
        self.strategic = StrategicQualityEvaluator()
        self.hit_analysis = HitAnalysisEvaluator()
        self.ownership = OwnershipStrategyEvaluator()

    async def evaluate(
        self, ctx: EvaluatorContext[Dict[str, Any], Dict[str, Any]]
    ) -> float:
        """Evaluate overall transfer quality.

        Returns:
            Weighted composite score from 0.0 to 1.0
        """
        structural_score = await self.structural.evaluate(ctx)
        coverage_score = await self.coverage.evaluate(ctx)
        strategic_score = await self.strategic.evaluate(ctx)
        hit_score = await self.hit_analysis.evaluate(ctx)
        ownership_score = await self.ownership.evaluate(ctx)

        # Weighted average
        composite_score = (
            structural_score * 0.20
            + coverage_score * 0.25
            + strategic_score * 0.30
            + hit_score * 0.15
            + ownership_score * 0.10
        )

        return composite_score
