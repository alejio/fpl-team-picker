# FPL Manager Strategy Analysis Tool

A comprehensive toolkit for deep-diving into any FPL manager's strategy, decision-making patterns, and performance trends.

---

## Overview

This toolkit analyzes FPL managers by examining:
- ✅ Gameweek-by-gameweek performance
- ✅ Transfer patterns and timing
- ✅ Fixture-aware decision making
- ✅ Captain selection strategy
- ✅ Team value growth
- ✅ Strategic evolution over time
- ✅ Rank progression
- ✅ Comparison to template/average

---

## Quick Start

### Analyze Any Manager

```bash
# Basic analysis
uv run python experiments/manager_analysis.py <MANAGER_ID>

# Analyze specific gameweek range
uv run python experiments/manager_analysis.py <MANAGER_ID> <MAX_GW>

# Examples
uv run python experiments/manager_analysis.py 25020          # Michael Bradon
uv run python experiments/manager_analysis.py 12345 15       # Through GW15
```

### Generate Visualizations

```bash
# Edit visualize_manager_strategy.py with your manager's data
# Then run:
uv run python experiments/visualize_manager_strategy.py
```

---

## Tools Included

### 1. `manager_analysis.py`
**Purpose:** Fetch and analyze any manager's complete FPL history

**Features:**
- Fetches data directly from FPL API
- Gameweek-by-gameweek breakdown
- Transfer analysis with in/out players
- Fixture difficulty indicators
- Performance trend analysis
- Rank progression tracking

**Output:**
- Console output with detailed GW analysis
- Can redirect to file for saving

**Example Output:**
```
================================================================================
FPL MANAGER STRATEGY ANALYSIS
================================================================================

Manager: Wirtz team ever (ID: 25020)
Overall Points: 618
Overall Rank: 768,198

PERFORMANCE TREND ANALYSIS
--------------------------------------------------------------------------------
First Half Average (GW1-5): 55.00
Second Half Average (GW6+): 68.60
Improvement: +13.60 points/GW
✅ CONFIRMED: Manager improves over the season!

[... detailed GW-by-GW analysis ...]
```

### 2. `visualize_manager_strategy.py`
**Purpose:** Create comprehensive visual analysis charts

**Charts Generated:**
1. Points per gameweek with trend lines
2. Overall rank progression
3. Team value growth
4. Transfer activity timeline
5. Captain selection distribution
6. Strategic evolution phases
7. Fixture difficulty analysis
8. Bank balance management

**Output Files:**
- `michael_bradon_strategy_visualization.png` (6-chart dashboard)
- `michael_bradon_fixture_analysis.png` (4-chart fixture focus)

### 3. Strategy Analysis Documents

**`MICHAEL_BRADON_STRATEGY_ANALYSIS.md`**
- Full 47-section deep dive
- Phase-by-phase strategy breakdown
- Transfer timeline
- Strengths and weaknesses
- Lessons learned

**`EXECUTIVE_SUMMARY_BRADON.md`**
- TL;DR findings
- Key moments analysis
- Predicted rest of season
- Actionable lessons
- How to beat the manager

---

## Key Metrics Analyzed

### Performance Metrics
- Total points
- Average points per gameweek
- First half vs second half comparison
- Rank progression
- GW rank vs overall rank

### Transfer Metrics
- Total transfers made
- Points hits taken
- Transfer timing patterns
- In/out player analysis
- Fixture targeting in transfers

### Financial Metrics
- Team value growth
- Bank balance strategy
- Strategic banking before pivots
- Value captured from early transfers

### Strategic Metrics
- Captain selection patterns
- Differential vs template decisions
- Team stacking behavior
- Fixture awareness level
- Wildcard/chip timing

---

## Use Cases

### 1. Learning from Top Managers
```bash
# Analyze top 100 managers
uv run python experiments/manager_analysis.py 123456  # Get manager ID from FPL site
```

**What to look for:**
- Transfer patterns before price rises
- Fixture-based team stacking
- Captain rotation strategies
- Chip timing decisions

### 2. Head-to-Head League Analysis
```bash
# Analyze your mini-league rivals
uv run python experiments/manager_analysis.py <RIVAL_ID>
```

**What to look for:**
- Where they're gaining points
- Transfer hit patterns (exploitable weakness)
- Template vs differential balance
- Upcoming fixture targeting

### 3. Strategy Validation
```bash
# Compare your strategy to successful managers
uv run python experiments/manager_analysis.py <YOUR_ID>
uv run python experiments/manager_analysis.py <TOP_MANAGER_ID>
```

**Compare:**
- Transfer discipline (hits taken)
- Fixture awareness
- Captain differentials
- Value building

### 4. Pattern Recognition
Run analysis on multiple managers to identify:
- Common traits of top performers
- Mistakes of struggling managers
- Optimal wildcard timing
- Effective captain strategies

---

## How It Works

### Data Sources

1. **FPL API Endpoints**
   - `bootstrap-static/` - Players, teams, gameweeks
   - `entry/<id>/` - Manager information
   - `entry/<id>/history/` - Gameweek history
   - `entry/<id>/event/<gw>/picks/` - Team picks per GW
   - `fixtures/` - Fixture difficulty data

2. **FPL Dataset Builder** (via `client`)
   - Enhanced player metrics
   - Team statistics
   - Historical data

### Analysis Pipeline

```
1. Fetch Manager Data
   ├── Manager profile
   ├── Season history
   └── GW-by-GW picks

2. Enrich with Context
   ├── Fixture difficulty
   ├── Player information
   ├── Transfer identification
   └── Strategic patterns

3. Analyze Trends
   ├── Performance trajectory
   ├── Transfer patterns
   ├── Fixture awareness
   └── Strategic evolution

4. Generate Insights
   ├── Strengths/weaknesses
   ├── Key moments
   ├── Predicted strategy
   └── Actionable lessons
```

---

## Customization

### Modify Analysis Focus

Edit `manager_analysis.py` to add custom metrics:

```python
def analyze_custom_metric(self, gameweek: int) -> float:
    """Add your custom analysis here."""
    picks = self.fetch_gw_picks(gameweek)
    # Your analysis logic
    return metric_value
```

### Add New Visualizations

Edit `visualize_manager_strategy.py`:

```python
# Add new subplot
ax_new = fig.add_subplot(gs[row, col])

# Your visualization code
ax_new.plot(...)
ax_new.set_title('Your Custom Chart')
```

### Export to Different Formats

```python
# In manager_analysis.py, add export methods
def export_to_json(self, filename: str):
    """Export analysis to JSON."""
    data = {
        'manager': self.manager_history,
        'gameweeks': self.gw_analysis,
        'transfers': self.transfer_history,
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
```

---

## Advanced Usage

### Batch Analysis

Analyze multiple managers:

```python
manager_ids = [25020, 12345, 67890]

for manager_id in manager_ids:
    analyzer = ManagerAnalyzer(manager_id)
    report = analyzer.generate_comprehensive_report()

    # Save to individual files
    with open(f'experiments/manager_{manager_id}_analysis.txt', 'w') as f:
        # Redirect output to file
        pass
```

### Comparative Analysis

Compare two managers:

```python
manager1 = ManagerAnalyzer(25020)
manager2 = ManagerAnalyzer(12345)

# Compare metrics
report1 = manager1.generate_comprehensive_report()
report2 = manager2.generate_comprehensive_report()

# Generate comparison charts
# ...
```

### Time Series Analysis

Track manager over multiple seasons:

```python
# Requires historical season data
seasons = ['2022-23', '2023-24', '2024-25']

for season in seasons:
    analyzer = ManagerAnalyzer(25020, season=season)
    # Analyze season patterns
    # ...
```

---

## Interpreting Results

### Performance Trends

**Improving Manager (like Michael Bradon):**
- Second half avg > First half avg
- Rank improves over time
- Transfer strategy evolves
- Pattern: Learning → Optimization → Mastery

**Declining Manager:**
- First half avg > Second half avg
- Rank worsens over time
- Reactive transfers
- Pattern: Template → Panic → Hits

**Consistent Manager:**
- Steady points per GW
- Stable rank
- Predictable strategy
- Pattern: Plan → Execute → Maintain

### Transfer Patterns

**Aggressive (High Risk):**
- Frequent transfers (10+ in 10 GWs)
- Multiple hits taken
- Chasing points
- High variance in results

**Balanced (Optimal):**
- 1 transfer every 1-2 GWs
- 0-2 hits all season
- Fixture-driven
- Steady improvement

**Passive (Conservative):**
- Infrequent transfers (5- in 10 GWs)
- No hits
- Template heavy
- Safe but slow growth

### Fixture Awareness

**Low Awareness:**
- Transfers ignore fixtures
- Captain vs hard fixtures (🔴)
- Random team selection
- Points below average

**High Awareness (like Bradon):**
- Every transfer targets 🟢
- Captain with easiest fixture
- Team stacks for schedules
- Points above average

---

## Common Patterns

### Top Managers (Top 10k)
- Fixture-first decisions
- 0-2 hits per season
- Early wildcard (GW4-6)
- Differential captains (2-3 per season)
- Team value growth (£2m+)
- Bold template departures

### Average Managers (1M-3M)
- Mix of fixtures and template
- 3-5 hits per season
- Mid wildcard (GW8-10)
- Safe captains (Salah/Haaland)
- Team value growth (£1m)
- Some differentials

### Struggling Managers (5M+)
- Template-heavy
- 5+ hits per season
- Late/panicked wildcard
- Always template captain
- Team value stagnant
- Few differentials

---

## FAQ

**Q: How do I find a manager's ID?**
A: Go to their FPL team page, look at URL: `fantasy.premierleague.com/entry/<ID>/`

**Q: Can I analyze managers from previous seasons?**
A: Currently limited to current season (2024/25). Historical API endpoints exist but require additional implementation.

**Q: Why are some transfers not detected?**
A: Early GW1 teams don't have previous GW to compare. Wildcards show as "0 transfers" to avoid listing all 15 changes.

**Q: How accurate is fixture difficulty?**
A: Uses FPL's official difficulty rating (1-5). Green (🟢) = 1-2, Yellow (🟡) = 3, Red (🔴) = 4-5.

**Q: Can I use this for mini-league analysis?**
A: Yes! Get all manager IDs from your mini-league and batch analyze them.

---

## Future Enhancements

### Planned Features
- [ ] Chip timing analysis (BB, TC, FH)
- [ ] Differential ownership tracking
- [ ] Price change prediction
- [ ] Expected points comparison
- [ ] Mini-league batch analysis
- [ ] Season-over-season comparison
- [ ] Machine learning prediction of next transfer

### Contribution Ideas
- Add more visualization types
- Export to CSV/JSON
- Web dashboard interface
- Live tracking during gameweeks
- Slack/Discord bot integration
- Team similarity clustering

---

## Example: Michael Bradon Analysis

### Command Run
```bash
uv run python experiments/manager_analysis.py 25020 > experiments/michael_bradon_analysis.txt
```

### Key Findings
- ✅ **24.7% improvement** from first half to second half
- ✅ **Rank jump:** 6.9M → 768k (88.9% improvement)
- ✅ **0 hits taken** (perfect transfer discipline)
- ✅ **Fixture-first strategy** (every transfer targeted easy fixtures)
- ✅ **Bold moves** (sold Salah in GW7)
- ✅ **Team value:** +£2.7m growth

### Strategic Profile
**Archetype:** Fixture-Focused Optimizer
**Risk Level:** Medium-High (bold differentials)
**Learning Curve:** Fast (GW1-3 template → GW4+ mastery)
**Predicted Finish:** Top 200k

---

## Credits

**Author:** FPL Team Picker AI
**Data Source:** Official FPL API
**Tools:** Python, pandas, matplotlib, seaborn, requests
**License:** MIT

---

## Support

For questions or issues:
1. Check the FAQ section above
2. Review example output files
3. Consult the source code comments
4. Experiment with test manager IDs

---

**Remember:** FPL is a game of small edges. Use this tool to learn from the best, identify patterns, and optimize your own strategy. Good luck! 🍀⚽📊
