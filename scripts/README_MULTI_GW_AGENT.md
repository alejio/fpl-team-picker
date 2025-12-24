# Multi-Gameweek Transfer Planning Agent

LLM-based reasoning agent for intelligent multi-gameweek transfer planning using Claude Agent SDK.

## Quick Start

1. **Set your Anthropic API key** (choose one method):
   ```bash
   # Option 1: Environment variable (recommended)
   export ANTHROPIC_API_KEY=your_api_key_here

   # Option 2: Config-specific environment variable
   export FPL_TRANSFER_PLANNING_AGENT_API_KEY=your_api_key_here

   # Option 3: Add to config.json
   # {
   #   "transfer_planning_agent": {
   #     "api_key": "your_api_key_here"
   #   }
   # }
   ```

2. **Run the agent:**
   ```bash
   # Default: GW18, 3-week horizon, balanced strategy
   uv run python scripts/multi_gw_agent.py

   # Custom options
   uv run python scripts/multi_gw_agent.py --gameweek 17 --horizon 5 --strategy aggressive
   ```

3. **Available options:**
   ```bash
   uv run python scripts/multi_gw_agent.py --help
   ```

## Strategy Modes

- **balanced** (default): Optimize xP while managing risk, prefer free transfers over hits
- **conservative**: Minimize risks and hits, favor template players
- **aggressive**: Willing to take multiple hits for high-upside differentials
- **dgw_stacker**: Specialize in building for Double Gameweeks

## Example Usage

```bash
# Conservative approach for top 100k managers
uv run python scripts/multi_gw_agent.py --gameweek 18 --horizon 3 --strategy conservative

# Aggressive differential hunting for rank climbing
uv run python scripts/multi_gw_agent.py --gameweek 18 --horizon 5 --strategy aggressive

# DGW preparation
uv run python scripts/multi_gw_agent.py --gameweek 17 --horizon 5 --strategy dgw_stacker
```

## Output

The agent will:
1. Load your current squad from the database
2. Analyze multi-GW xP predictions
3. Generate a week-by-week transfer plan with reasoning
4. Display:
   - Transfer recommendations for each week
   - Expected xP vs baseline
   - Strategic reasoning
   - Total ROI summary

## Architecture

**Backend Components:**
- [transfer_planning_agent_service.py](../fpl_team_picker/domain/services/transfer_planning_agent_service.py) - Main agent orchestration
- [agent_tools.py](../fpl_team_picker/domain/services/agent_tools.py) - Tools wrapping existing services
- [transfer_plan.py](../fpl_team_picker/domain/models/transfer_plan.py) - Pydantic models

**Agent Tools:**
- `get_multi_gw_xp_predictions`: Multi-GW xP forecasting (3/5 GW horizons)
- More tools coming in Phase 2-4 (DGW detection, fixture analysis, optimizer integration)

## Configuration

Edit [settings.py](../fpl_team_picker/config/settings.py) or create `config.json`:
```python
# Available configuration options
config.transfer_planning_agent.api_key = None  # Reads from ANTHROPIC_API_KEY env var if not set
config.transfer_planning_agent.model = "claude-sonnet-4-5"  # or "claude-haiku-3-7" for speed
config.transfer_planning_agent.default_horizon = 3
config.transfer_planning_agent.default_strategy = "balanced"
config.transfer_planning_agent.default_hit_roi_threshold = 5.0
config.transfer_planning_agent.max_iterations = 10
config.transfer_planning_agent.temperature = 0.7
config.transfer_planning_agent.timeout_seconds = 120
```

Or create `config.json` in project root:
```json
{
  "transfer_planning_agent": {
    "api_key": "sk-ant-...",
    "model": "claude-sonnet-4-5",
    "default_horizon": 3,
    "default_strategy": "balanced"
  }
}
```

## Development Status

**✅ Phase 1 Complete:**
- Core backend infrastructure
- Pydantic models for transfer plans
- Agent service with strategy modes
- Basic tool: multi-GW xP predictions
- CLI test script

**🚧 Coming Next (Phase 2-4):**
- Additional agent tools (DGW detection, fixture swings, SA optimizer)
- Transfer sequence validation
- Chip timing recommendations
- Marimo UI interface

## Troubleshooting

**Error: `Anthropic API key is required`**

The agent needs an Anthropic API key to run. Set it using one of these methods:
```bash
# Option 1: Standard environment variable
export ANTHROPIC_API_KEY=your_key_here

# Option 2: Config-specific environment variable
export FPL_TRANSFER_PLANNING_AGENT_API_KEY=your_key_here

# Option 3: Add to config.json in project root
echo '{"transfer_planning_agent": {"api_key": "your_key_here"}}' > config.json
```

**Error: Data loading fails**
- Ensure you have run `fpl-gameweek-manager` at least once to populate the database

**Error: `current_squad` not found**
- The agent loads your squad from the previous gameweek
- Make sure you have historical picks data in the database

## API Costs

Estimated costs with Claude Sonnet 4.5:
- Input: ~10-20K tokens
- Output: ~2-5K tokens
- **Cost per plan: $0.10-0.30**

For lower costs, use Haiku (edit [transfer_planning_agent_service.py:40](../fpl_team_picker/domain/services/transfer_planning_agent_service.py#L40)):
```python
agent_service = TransferPlanningAgentService(model="claude-haiku-3-7")
```

## Next Steps

See the [implementation plan](../.claude/plans/mellow-swinging-pony.md) for upcoming features and roadmap.
