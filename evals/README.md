# FPL Transfer Planning Agent Evaluations

Comprehensive evaluation framework for the FPL transfer planning agent using [Pydantic AI Evals](https://ai.pydantic.dev/evals/).

## Overview

This evaluation suite systematically tests the transfer planning agent across 10 realistic FPL scenarios, assessing:

- **Structural validity**: Pydantic model compliance
- **Scenario coverage**: Hold option, min/max scenarios, top pick validity
- **Strategic quality**: DGW/fixture swing/chip prep flags, reasoning quality
- **Hit analysis**: ROI calculations, threshold adherence
- **Ownership strategy**: Template vs differential decisions

## Quick Start

```bash
# Install dependencies
uv sync

# Set API key
export ANTHROPIC_API_KEY=your_key_here

# Run all evaluations (10 scenarios)
uv run python evals/run_transfer_evals.py

# Run specific dataset subset (short option)
uv run python evals/run_transfer_evals.py -d basic

# Test with cheaper/faster Haiku model (short option)
uv run python evals/run_transfer_evals.py -m claude-haiku-4-5

# Quick test with 1 case (combined short options)
uv run python evals/run_transfer_evals.py -d basic -n 1

# Enable LLM-as-judge for reasoning quality assessment (costs extra)
uv run python evals/run_transfer_evals.py -d basic --use-llm-judge

# Enable Logfire reporting for visualization
export LOGFIRE_TOKEN=your_token_here
uv run python evals/run_transfer_evals.py --enable-logfire

# Get help (shows all options)
uv run python evals/run_transfer_evals.py --help
```

**Short Options**:
- `-d, --dataset`: Dataset choice (all/basic/strategic/constraints)
- `-m, --model`: Claude model name
- `-n, --max-cases`: Number of cases to run
- `--enable-logfire`: Enable Logfire observability
- `--use-llm-judge`: Enable LLM-as-judge evaluator (slower, costs extra)

## Datasets

### Full Dataset (10 scenarios)
```bash
uv run python evals/run_transfer_evals.py --dataset all
```

Includes:
1. **Premium Upgrade**: Should take -4 hit if ROI > threshold
2. **Differential Punt**: Low ownership player with good fixtures
3. **Template Safety**: Conservative strategy favors high ownership
4. **DGW Opportunity**: Identify and leverage double gameweeks
5. **Fixture Swing**: Exploit difficulty changes
6. **Budget Constraint**: Respect tight budget limits
7. **Injury Emergency**: Urgent replacement, don't bank transfer
8. **Aggressive Ceiling**: High-variance picks for upside
9. **Bank Transfer**: Hold when squad is strong
10. **Chip Preparation**: Position for upcoming Wildcard

### Basic Scenarios (3 scenarios)
```bash
uv run python evals/run_transfer_evals.py --dataset basic
```

Covers: Premium upgrades, differentials, template safety

### Strategic Scenarios (3 scenarios)
```bash
uv run python evals/run_transfer_evals.py --dataset strategic
```

Covers: DGW exploitation, fixture swings, chip preparation

### Constraint Scenarios (3 scenarios)
```bash
uv run python evals/run_transfer_evals.py --dataset constraints
```

Covers: Budget constraints, injuries, hold decisions

## Evaluators

### 1. StructuralValidityEvaluator (20% weight)
- Valid `SingleGWRecommendation` instance
- All required fields present
- Pydantic validation passed

### 2. ScenarioCoverageEvaluator (25% weight)
- Hold option included (when expected)
- Min/max scenario counts met
- Top recommendation is valid

### 3. StrategicQualityEvaluator (30% weight)
- Strategic flags set correctly (leverages_dgw, leverages_fixture_swing, etc.)
- Reasoning mentions expected keywords
- Confidence levels appropriate
- Context analysis includes insights

### 4. HitAnalysisEvaluator (15% weight)
- Hit scenarios exist when expected
- ROI calculations reasonable
- Threshold respected

### 5. OwnershipStrategyEvaluator (10% weight)
- Template scenarios prioritize high ownership
- Differential scenarios target low ownership
- Strategy-appropriate decisions

### 6. LLMReasoningQualityEvaluator (Optional)
**LLM-as-judge** for subjective quality assessment:
- **Clarity** (30%): Reasoning is clear and well-structured
- **Strategic Depth** (30%): Shows deep FPL knowledge (fixtures, ownership, form)
- **Scenario Appropriateness** (20%): Recommendation fits the context
- **Context Analysis** (20%): Identifies key opportunities/risks correctly

Uses Claude Haiku to evaluate reasoning quality. Enable with `--use-llm-judge` flag.
Costs ~$0.001-0.002 per evaluation (extra API calls).

### 7. CompositeTransferQualityEvaluator
Weighted combination of deterministic evaluators for overall score.

## Output

Example output:
```
================================================================================
EVALUATION RESULTS
================================================================================

Case: premium_upgrade_with_hit
  structural_validity      : 1.00 (100%)
  scenario_coverage        : 1.00 (100%)
  strategic_quality        : 0.85 (85%)
  hit_analysis            : 1.00 (100%)
  ownership_strategy      : 0.80 (80%)
  composite_score         : 0.91 (91%)
  Status: ✅ PASSED

Case: differential_punt_opportunity
  structural_validity      : 1.00 (100%)
  scenario_coverage        : 1.00 (100%)
  strategic_quality        : 0.90 (90%)
  hit_analysis            : 0.75 (75%)
  ownership_strategy      : 1.00 (100%)
  composite_score         : 0.93 (93%)
  Status: ✅ PASSED

...

================================================================================
SUMMARY STATISTICS
================================================================================

structural_validity      : 100.00%
scenario_coverage        : 98.00%
strategic_quality        : 87.50%
hit_analysis            : 85.00%
ownership_strategy      : 90.00%
composite_score         : 90.10%

================================================================================
Overall Pass Rate: 90.00%
Total Cases: 10
Passed: 9
Failed: 1
================================================================================
```

## Extending Evaluations

### Adding New Test Cases

Edit `evals/datasets/transfer_planning_scenarios.py`:

```python
def create_my_scenario_case() -> Case[Dict[str, Any], Dict[str, Any]]:
    """Test case: Description of what should happen."""
    return Case(
        name="my_scenario_name",
        inputs={
            "scenario": "GW20, 2 FTs, £3.0m ITB",
            "context": "Specific situation details",
            "expected_behavior": "What the agent should do",
        },
        expected_output={
            "should_include_hold": True,
            "min_scenarios": 3,
            "top_pick_reasoning_mentions": ["keyword1", "keyword2"],
        },
        metadata={
            "difficulty": "medium",
            "category": "my_category",
        },
    )

# Add to dataset
transfer_planning_dataset.cases.append(create_my_scenario_case())
```

### Adding Custom Evaluators

Create new evaluator in `evals/evaluators/transfer_quality.py`:

```python
class MyCustomEvaluator(Evaluator):
    """Evaluates custom aspect of recommendations."""

    async def evaluate(
        self, ctx: EvaluatorContext[Dict[str, Any], Dict[str, Any]]
    ) -> float:
        """Evaluate custom criteria.

        Returns:
            Score from 0.0 to 1.0
        """
        if not isinstance(ctx.output, SingleGWRecommendation):
            return 0.0

        # Your evaluation logic here
        score = 0.0
        # ...
        return score
```

Then add to `run_transfer_evals.py`:

```python
evaluators = {
    "my_custom_check": MyCustomEvaluator(),
    # ... existing evaluators
}
```

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
name: Transfer Agent Evals

on: [push, pull_request]

jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: Run evals
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          uv sync
          uv run python evals/run_transfer_evals.py --max-cases 3
```

## Using with Logfire

Enable Logfire for visualization and collaborative analysis:

```bash
# Set token
export LOGFIRE_TOKEN=your_token_here

# Run evals with Logfire
uv run python evals/run_transfer_evals.py --enable-logfire
```

View results at [logfire.pydantic.dev](https://logfire.pydantic.dev)

## Performance Notes

- **Full dataset (10 cases)**: ~5-10 minutes with Sonnet 4.5
- **Basic dataset (3 cases)**: ~2-3 minutes
- **Single case**: ~20-40 seconds

Use `--model claude-haiku-4-5` for faster/cheaper evals during development.

## Troubleshooting

### "Anthropic API key is required"
Set `ANTHROPIC_API_KEY` environment variable:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### "Failed to initialize Logfire"
Ensure `LOGFIRE_TOKEN` is set if using `--enable-logfire`.

### Low scores on strategic_quality
Agent may not be setting strategic flags correctly. Check:
- `leverages_dgw`, `leverages_fixture_swing`, `prepares_for_chip` flags
- Reasoning mentions expected keywords
- Context analysis includes DGW/fixture/chip insights

### Mock data limitations
Current implementation uses minimal mock data. For production evals:
1. Load real historical FPL data matching scenarios
2. Use actual fixture difficulties, ownership, prices
3. Include real player performance data

See `create_mock_gameweek_data()` in `run_transfer_evals.py` for extension.

## Resources

- [Pydantic AI Evals Documentation](https://ai.pydantic.dev/evals/)
- [Transfer Planning Agent Service](../fpl_team_picker/domain/services/transfer_planning_agent_service.py)
- [Transfer Recommendation Models](../fpl_team_picker/domain/models/transfer_recommendation.py)
