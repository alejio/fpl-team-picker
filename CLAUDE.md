# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Fantasy Premier League (FPL) analysis suite that provides both season-start team building and weekly gameweek management. The project implements a comprehensive mathematical framework combining Poisson distributions, team ratings, player performance metrics, fixture analysis, and live data integration for optimal FPL decision making.

## Data Loading

This project now uses the fpl-dataset-builder database client library instead of CSV files.

### Usage
```python
# Load datasets from database
from client import (
    get_current_players,
    get_current_teams,
    get_fixtures_normalized,
    get_player_xg_xa_rates,
    get_gameweek_live_data
)

players = get_current_players()
teams = get_current_teams()
fixtures = get_fixtures_normalized()
```

### Available Data Functions
- `get_current_players()` - Current season player data (prices, positions, teams)
- `get_current_teams()` - Team reference data for ID lookups
- `get_fixtures_normalized()` - Normalized fixture data with team IDs and kickoff times
- `get_player_xg_xa_rates()` - Expected goals (xG90) and expected assists (xA90) rates per 90 minutes
- `get_gameweek_live_data(gw)` - Real-time player performance data for active/completed gameweeks
- `get_player_deltas_current()` - Week-over-week performance and market movement tracking
- `get_database_summary()` - Database status and table information

### Benefits
- Always fresh data from centralized database
- No CSV file dependencies or path management
- Better performance than file I/O
- Single source of truth shared with dataset builder
- Consistent DataFrame structure maintained

### Installation
The fpl-dataset-builder is configured as a local dependency. Simply run:

```bash
uv sync
```

This will automatically install fpl-dataset-builder from the local path along with all its dependencies.

**Configuration in pyproject.toml:**
```toml
dependencies = [
    # ... other dependencies  
    "fpl-dataset-builder",
]

[tool.uv.sources]
fpl-dataset-builder = { path = "../fpl-dataset-builder", editable = true }
```

### Data Coverage
The database includes all the data previously available in CSV files:
- Current season player data with prices, positions, teams
- Historical player statistics including form, ICT index, points per game
- Expected goals (xG90) and expected assists (xA90) rates per 90 minutes
- Normalized fixture data with team IDs and kickoff times
- Historical match results for Poisson baseline calculations
- Player availability tracking and injury status
- Live gameweek performance data
- Week-over-week performance and market movement tracking

## Expected Points Models

The suite includes two complementary xP calculation approaches:

### Multi-Gameweek Model (Season Planning)
**Fast implementation with smart shortcuts for season-start team building:**

### 1. League Baselines (Hardcoded)
- μ_home = 1.43, μ_away = 1.15 (known Premier League averages)
- Skip historical calculation for speed

### 2. Team Strength Ratings (Simplified)
- Use inverse of last season final position as team strength proxy
- Simple scaling factor [0.7, 1.3] based on opponent rank
- Skip complex attack/defense separation for MVP

### 3. Enhanced Minutes Model (SBP + Availability)
- **Selected By Percentage (SBP)** based start probabilities (0.05-0.95)
- **Availability status** integration ('i'=injured, 's'=suspended, 'd'=doubtful, 'a'=available)  
- **Position-specific durability** (GKP: 90min avg, outfield: 70-85min based on price)
- **Probabilistic scenarios**: Full game, partial start, substitute appearance, no show
- **Price-based priors**: Premium players (£7m+) get higher start probability adjustments

### 4. Multi-Gameweek xP Calculation (5-week horizon)
- **Per-minute production**: xG90 and xA90 from rates dataset
- **Fixture scaling**: Multiply by opponent difficulty [0.7, 1.3] for each GW
- **Temporal weighting**: GW1 (1.0), GW2 (0.9), GW3 (0.8), GW4 (0.7), GW5 (0.6)
- **Expected contributions**: Scale by expected minutes per gameweek
- **FPL points conversion**:
  - Appearance: 2 pts if ≥60 mins, 1 pt if >0 mins
  - Goals: Position multipliers (GK/DEF: 6, MID: 5, FWD: 4)
  - Assists: 3 × xA_exp
  - Clean sheets: Simplified P(CS) by team strength
- **Cumulative xP**: Sum weighted xP across GW1-5

### Single Gameweek Model (Weekly Management)
**Form-weighted predictions with live data integration for weekly decisions:**

#### 1. Live Data Integration
- **Real-time performance**: Current gameweek player statistics from FPL API
- **Performance deltas**: Week-over-week trends and market movements
- **Availability status**: Real-time injury, suspension, and fitness updates
- **Market intelligence**: Ownership changes and price movements

#### 2. Form-Weighted Calculations
- **Recent form emphasis**: Blend recent performance (70%) with season average (30%)
- **Momentum indicators**: Visual form tracking using emoji system
- **Enhanced minutes model**: Live data improves start probability accuracy
- **Fixture-specific scaling**: Single gameweek opponent difficulty only

#### 3. Dynamic Performance Adjustments
- **Form multipliers**: Recent performance impact on xP estimates
- **Momentum tracking**: 🔥 Hot (6+ point delta), 📈 Rising (3+ delta), ➡️ Stable, 📉 Declining (<2 delta), ❄️ Cold (negative delta)
- **Live validation**: Cross-reference predictions with actual performance data
- **Transfer risk flagging**: Identify players with concerning recent trends

## Data Processing Notes

- All player IDs are consistent across datasets for joining
- Team IDs reference fpl_teams_current.csv
- UTC timestamps throughout
- Handle missing data with appropriate fallbacks
- Use pandas for data manipulation and merging
- Live data files update in real-time during gameweeks
- Performance deltas calculated automatically between gameweeks
- Graceful degradation when live data unavailable

## Implementation Architecture

**Core Components:**

1. **Data Loading & Processing** (`load_datasets()`)
   - FPL players, xG/xA rates, fixtures, teams data
   - Data validation and consistency checks
   - Missing data handling with position-based fallbacks

2. **Team Strength & Fixture Analysis** (`get_team_strength_ratings()`)
   - 2023-24 final table position mapping to strength [0.7, 1.3]
   - Multi-gameweek fixture difficulty matrix creation
   - Home/away advantage incorporation

3. **Minutes Prediction Model** (`calculate_expected_minutes_probabilistic()`)
   - SBP-based start probability calculation
   - Availability status processing (injured/suspended/doubtful)
   - Position and price-based durability modeling
   - Probabilistic scenario weighting

4. **Expected Points Engine** (`calculate_multi_gw_xp()`)
   - **Statistical xG/xA Estimation** for players missing historical data
   - 5-gameweek temporal weighting (1.0, 0.9, 0.8, 0.7, 0.6)
   - Fixture difficulty scaling per gameweek
   - FPL scoring conversion (goals, assists, clean sheets, appearances)
   - Transfer risk flagging for poor GW2-3 fixtures

   **xG/xA Estimation Model** (`estimate_missing_xg_xa_rates()`)
   - **Multi-factor estimation**: Price, position, team strength, Selected By Percentage
   - **Premium player boosts**: £8m+ players get significant xG/xA multipliers
   - **Team quality adjustment**: Better teams (higher strength) create more chances
   - **Position-specific caps**: Prevents unrealistic estimates (e.g., DEF max xG90: 0.35)
   - **Ownership weighting**: Higher SBP suggests better underlying stats

5. **Team Optimization** (`select_optimal_team()`)
   - Simulated Annealing with 5,000 iterations
   - Constraint satisfaction (budget, formation, 3-per-team rule)
   - Starting 11 formation optimization across 8 valid formations
   - Must-include/exclude player handling

6. **Interactive Interface** (Marimo cells)
   - Real-time optimization with user constraints
   - Visual squad analysis and performance metrics
   - Transfer risk assessment and stability warnings

### Gameweek Manager Components (NEW):

7. **Live Data Loading & Processing** (`load_gameweek_datasets()`)
   - Enhanced dataset loading with live gameweek data integration
   - Performance delta tracking and trend analysis
   - Availability status and market movement processing
   - Graceful fallback when live data unavailable

8. **Form-Weighted xP Engine** (`calculate_form_weighted_xp()`)
   - Blends recent performance (70%) with season baseline (30%)
   - Integrates live performance data and delta analysis
   - Dynamic form multipliers based on recent point trends
   - Enhanced minutes prediction using live start/substitute data

9. **Smart Gameweek Optimization Suite**
   - **Constraint-Based Optimization**: Pre-optimization player inclusion/exclusion controls
   - **Smart Transfer Decision Engine**: Auto-selects optimal number of transfers (0-3) based on net XP after penalties
   - **Formation-Flexible Starting 11**: Automatically finds best lineup from optimal squad
   - **Transfer Analysis Engine**: Hit calculations and opportunity cost analysis with scenario comparison
   - **Captain Selection Tool**: Risk-adjusted captaincy recommendations
   - **Squad Management**: Budget tracking and constraint validation

10. **Interactive Fixture Difficulty Visualization** (`fixture_difficulty_viz.py`)
    - Dynamic team strength calculation with GW8+ transition logic
    - Interactive plotly heatmap showing 5-gameweek fixture difficulty
    - Home/away advantage modeling with opponent strength scaling
    - Visual transfer strategy guidance with color-coded difficulty ratings

11. **Retro Analysis Framework** (`retro_analysis_section`)
    - Post-gameweek performance validation against predictions
    - Top performers vs model predictions comparison
    - Template vs differential analysis for learning
    - Model accuracy tracking and improvement identification

## Simulation Methodology

### Team Optimization Algorithm (Simulated Annealing)

The model uses **Simulated Annealing** to solve the complex combinatorial optimization problem of selecting the optimal 15-player FPL squad. This approach was chosen over greedy algorithms to escape local optima and find globally competitive solutions.

#### Core Optimization Problem
- **Objective**: Maximize weighted 5-gameweek xP for starting 11 players
- **Squad Constraints**: 
  - 15 players total: 2 GKP, 5 DEF, 5 MID, 3 FWD
  - £100m budget limit
  - Max 3 players per real team
- **Transfer Planning**: Minimize "forced transfers" penalty (players with poor GW2-3 fixtures)
- **Formation Flexibility**: Starting 11 auto-selected from best formation among 8 valid options

#### Algorithm Implementation

**1. Initialization**
- Generate random valid 15-player squad respecting all constraints
- Up to 1000 attempts to create feasible starting solution
- Filter to players with valid xP data only

**2. Neighbor Generation (Player Swapping)**
- Randomly select one player from current squad
- Find all valid replacements for same position
- Ensure replacement maintains budget and team limits
- Random selection from affordable alternatives

**3. Acceptance Criteria**
- **Better solutions**: Always accepted (xP improvement)
- **Worse solutions**: Accepted with probability exp(Δ/T)
  - Δ = change in expected points (negative for worse)
  - T = temperature (starts at 1.0, linearly decreases to 0.01)
- Temperature schedule enables exploration early, exploitation late

**4. Starting 11 Optimization**
- For each 15-player squad, find best starting formation
- Test all 8 valid formations: (1,3,5,2), (1,3,4,3), (1,4,5,1), (1,4,4,2), (1,4,3,3), (1,5,4,1), (1,5,3,2), (1,5,2,3)
- Rank players by position by xP, select highest for each formation
- Choose formation with maximum total xP

**5. Convergence Parameters**
- **Iterations**: 5,000 default (adjustable)
- **Temperature decay**: Linear from 1.0 to 0.01
- **Progress tracking**: Report improvements every 5 iterations

#### Key Features
- **Constraint satisfaction**: All solutions guaranteed valid
- **Global search**: Temperature allows escaping local optima
- **Computational efficiency**: ~5,000 evaluations typical
- **Robustness**: Fallback mechanisms for edge cases
- **Formation agnostic**: Automatically finds best starting 11 arrangement

#### Output Metrics
- Starting 11 expected points and cost
- Budget utilization and remaining funds
- xP per £1m efficiency ratio
- Rule compliance verification
- Formation breakdown and bench analysis

This methodology balances solution quality with computational speed, typically finding high-quality solutions within seconds while respecting all FPL constraints.

## Common Commands

**Primary Interfaces:**
```bash
# Season-start team building (multi-gameweek optimization)
marimo run fpl_xp_model.py

# Weekly gameweek management (single-GW optimization with live data)
marimo run fpl_gameweek_manager.py

# Development/edit modes
marimo edit fpl_xp_model.py
marimo edit fpl_gameweek_manager.py
```

**Weekly Workflow:**
```bash
# Monday: Save predictions before gameweek
marimo run fpl_xp_model.py  # Save predictions section

# Tuesday-Friday: Weekly planning
marimo run fpl_gameweek_manager.py  # Analyze upcoming GW

# Monday: Post-gameweek analysis
marimo run fpl_gameweek_manager.py  # Retro analysis section
```

**Development:**
```bash
# Install dependencies
uv sync

# Check project structure
ls -la

# Run git operations (when needed)
git status
git add .
git commit -m "Description"
```

## Technical Specifications

**Dependencies:**
- Python 3.13+
- marimo>=0.14.16 (interactive notebook interface)
- pandas>=2.3.1 (data manipulation)
- numpy>=2.3.2 (numerical computations) 
- matplotlib>=3.10.5, seaborn>=0.13.2 (visualization)
- plotly>=6.3.0 (interactive visualizations)
- pyarrow>=21.0.0 (efficient data I/O)

**Key Model Assumptions:**
- **Statistical xG/xA estimation** for new transfers using price, position, team strength, and SBP
- **Enhanced minutes model** using SBP + availability rather than simple price proxy
- **Team strength from 2023-24 final table** positions [0.7, 1.3] scaling
- **Hardcoded league baselines** (μ_home=1.43, μ_away=1.15) for speed
- **Single-point estimates** (no uncertainty propagation in v0.1)
- **Simplified clean sheet probabilities** based on team strength and fixture difficulty
- **No BPS/bonus, cards, saves, penalties** in v0.1 (core scoring only)
- **Position-based xG/xA fallbacks** for edge cases after statistical estimation

## FPL Rules Reference

**IMPORTANT**: All implementations must comply with official FPL rules documented in `fpl_rules.md`. Key constraints include squad composition (2-5-5-3), £100m budget, max 3 players per team, and valid formations.

## Performance Benchmark

**2024-25 FPL Winner**: 2,810 points (74 points per gameweek average)
- Target: Our 5-week xP projections should align with ~370 points (5 × 74) for competitive squads
- Validation: Premium players should project 8-12 xP per gameweek, budget options 4-7 xP

## Current Implementation Status

**✅ COMPLETED - Season-Start Builder (v1.0):**
- Multi-gameweek xP calculations (GW1-5 weighted horizon)
- **Statistical xG/xA estimation model** for new transfers and missing data
- Fixture difficulty adjustments across 5-week period
- Enhanced minutes model using Selected By Percentage (SBP) and availability status
- Simulated Annealing team optimization with constraint satisfaction
- Interactive Marimo notebook interface
- Formation-flexible starting 11 selection (8 valid formations)
- Transfer risk analysis and squad stability metrics
- Budget utilization optimization and efficiency tracking
- Team customization with must-include/exclude player constraints

**✅ COMPLETED - Gameweek Manager (v1.1):**
- **Live data integration** with real-time gameweek performance tracking
- **Form-weighted xP calculations** blending recent performance (70%) with season baseline (30%)
- **Performance delta analysis** for week-over-week trend identification
- **Momentum indicators** with visual form tracking (🔥📈➡️📉❄️)
- **Enhanced minutes prediction** using live start/substitute data
- **Smart transfer optimization** with automatic 0-3 transfer decision based on net XP after penalties
- **Constraint-based optimization** with pre-optimization player inclusion/exclusion controls
- **Comprehensive scenario analysis** comparing all transfer options with XP gain calculations
- **Paginated player display** showing all players ranked by expected points
- **Transfer analysis engine** with hit calculations and opportunity cost analysis
- **Captain selection tools** with risk-adjusted recommendations
- **Squad management suite** with budget tracking and constraint validation
- **Retro analysis framework** for post-gameweek performance validation

**✅ INTERACTIVE FEATURES:**
- **Dual-tool workflow**: Season planning + weekly management
- Real-time squad optimization with constraint handling
- Starting 11 auto-selection from best formation
- Transfer risk warnings and fixture analysis
- Budget and efficiency analysis with live market data
- Rule compliance validation across both tools
- Multi-gameweek vs single-gameweek breakdown per player
- **Live performance monitoring** with momentum tracking
- **Post-gameweek analysis** for continuous model improvement

**⏭️ FUTURE ENHANCEMENTS:**
- **Price change prediction** and market intelligence
- **Live performance monitoring dashboard** with real-time alerts
- Historical back-testing and validation across multiple seasons
- Complex minute modeling from detailed game logs
- Venue-specific attack/defense ratings
- Advanced injury data integration with return predictions
- Bonus points (BPS) modeling and prediction
- Sensitivity analysis and uncertainty quantification
- Auto-transfer suggestions with risk assessment
- Mini-league strategy optimization

## Future Improvements

### Dual-Tool Integration Workflow

**Weekly FPL Cycle:**
1. **Season Start**: Use `fpl_xp_model.py` for initial 15-player squad building
2. **Weekly Planning**: Use `fpl_gameweek_manager.py` for lineup and transfer decisions
3. **Post-Gameweek**: Use retro analysis section for model validation and learning
4. **Continuous Improvement**: Apply insights to enhance both models

**Data Flow Integration:**
- **Live data** enhances both season planning and weekly management
- **Performance deltas** inform transfer decisions and form analysis
- **Retro analysis** validates prediction accuracy and identifies improvements
- **Prediction storage** enables consistent tracking across tools

### Feature Roadmap
- **Heatmap Visualization**:
  - Show upcoming 5 fixtures and difficulty for each team in a heatmap
  - Integrate live data trends into visual analysis
- **Advanced Market Intelligence**:
  - Price change prediction using ownership and performance trends
  - Transfer window impact analysis on player selection
  - Market sentiment integration from live data feeds