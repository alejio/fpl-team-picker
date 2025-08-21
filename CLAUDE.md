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

**Modular Core Components (v1.2):**

### 1. **Data Loading Module** (`fpl_data_loader.py`)
   - **`fetch_fpl_data()`** - Centralized FPL data loading from database
   - **`fetch_manager_team()`** - Manager team data retrieval from previous gameweek
   - **`process_current_squad()`** - Current squad processing with captain info
   - **`load_gameweek_datasets()`** - Comprehensive gameweek data orchestration
   - Historical form data collection with configurable windows
   - Data validation and consistency checks with graceful fallbacks

### 2. **Expected Points Engine** (`xp_model.py`)
   - **`XPModel` class** - Dedicated XP calculation engine
   - **Form-weighted predictions** - 70% recent form + 30% season baseline
   - **Statistical xG/xA estimation** - Multi-factor estimation for missing data
   - **Dynamic team strength integration** - Real-time strength calculations
   - **Enhanced minutes prediction** - SBP + availability + form-based modeling
   - **Multi-gameweek capability** - 5-gameweek temporal weighting (1.0, 0.9, 0.8, 0.7, 0.6)

### 3. **Dynamic Team Strength** (`dynamic_team_strength.py`)
   - **`DynamicTeamStrength` class** - Evolving team strength ratings
   - **Historical transition logic** - Weighted 2024-25 baseline → current season focus
   - **GW8+ pure current season** - Responsive to current form once sufficient data
   - **Rolling window analysis** - 6-gameweek performance windows
   - **Venue-specific adjustments** - Home/away advantage incorporation

### 4. **Optimization Engine** (`fpl_optimization.py`)
   - **`optimize_team_with_transfers()`** - Smart 0-3 transfer optimization
   - **`calculate_total_budget_pool()`** - Advanced budget analysis
   - **`premium_acquisition_planner()`** - Multi-transfer premium targeting
   - **`get_best_starting_11()`** - Formation-flexible lineup selection
   - **`select_captain()`** - Risk-adjusted captaincy recommendations
   - Constraint-based optimization with must-include/exclude controls

### 5. **Visualization Suite** (`fpl_visualization.py`)
   - **`create_team_strength_visualization()`** - Interactive team strength displays
   - **`create_player_trends_visualization()`** - Historical performance trends
   - **`create_trends_chart()`** - Interactive plotly-based charts
   - **`create_fixture_difficulty_visualization()`** - Fixture difficulty heatmaps
   - Real-time chart updates with error handling and graceful degradation

### 6. **Prediction Storage** (`prediction_storage.py`)
   - **`PredictionStorage` class** - Retro analysis prediction tracking
   - **`save_gameweek_predictions()`** - Systematic prediction archival
   - **`load_saved_predictions()`** - Historical prediction retrieval
   - **`compare_predictions_vs_actual()`** - Model accuracy validation
   - Timestamped prediction storage for continuous improvement

### 7. **Team Optimization (Legacy)** (`select_optimal_team()`)
   - Simulated Annealing with 5,000 iterations for season-start building
   - Constraint satisfaction (budget, formation, 3-per-team rule)
   - Starting 11 formation optimization across 8 valid formations
   - Must-include/exclude player handling

### 8. **Interactive Interfaces** (Marimo notebooks)
   - **Season Planning** (`fpl_xp_model.py`) - Multi-gameweek team building
   - **Weekly Management** (`fpl_gameweek_manager.py`) - Form-weighted gameweek optimization
   - Real-time optimization with user constraints
   - Visual squad analysis and performance metrics
   - Transfer risk assessment and stability warnings

### 9. **Form Analytics Dashboard** (Enhanced)
   - **Hot/Cold player detection** - 🔥📈➡️📉❄️ momentum indicators
   - **Performance delta tracking** - Week-over-week trend analysis
   - **Current squad form analysis** - Team-specific form evaluation
   - **Transfer risk identification** - Form-based transfer recommendations

### 10. **Player Performance Trends** (New Visualization)
   - **Interactive trend charts** - Plotly-based historical performance visualization
   - **Multi-attribute analysis** - Points, xG, xA, minutes, value ratios
   - **Multi-player comparison** - Side-by-side player trend analysis
   - **Real-time data integration** - Live gameweek data incorporation

### 11. **Fixture Difficulty Matrix** (Enhanced)
   - **Dynamic team strength integration** - Real-time difficulty calculations
   - **5-gameweek fixture heatmaps** - Visual fixture analysis
   - **Home/away venue adjustments** - Contextual difficulty scaling
   - **Transfer planning integration** - Fixture-informed decision making

### 12. **Retro Analysis Framework** (`prediction_storage.py`)
    - **Systematic prediction archival** - Timestamped prediction storage
    - **Model accuracy validation** - Prediction vs actual comparison
    - **Continuous improvement tracking** - Performance metric evolution
    - **Template vs differential analysis** - Strategic learning framework

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

**Modular Architecture Specifications:**

### `fpl_data_loader.py`
- **Purpose**: Centralized data orchestration and loading
- **Key Functions**:
  - `fetch_fpl_data(target_gameweek, form_window=5)` - Database integration with configurable form windows
  - `fetch_manager_team(previous_gameweek)` - Manager team data retrieval with error handling
  - `process_current_squad(team_data, players, teams)` - Squad processing with captain information
  - `load_gameweek_datasets(target_gameweek)` - Comprehensive data loading orchestration
- **Features**: Graceful fallbacks, data validation, column standardization, historical form data collection

### `xp_model.py`
- **Purpose**: Dedicated Expected Points calculation engine
- **Key Classes**: `XPModel` - Main XP calculation engine with configurable parameters
- **Key Methods**:
  - `calculate_expected_points()` - Main XP calculation with form weighting
  - `_apply_form_weighting()` - 70% recent form + 30% season baseline blending
  - `_estimate_missing_xg_xa()` - Statistical estimation for missing xG/xA data
  - `_calculate_expected_minutes()` - Enhanced minutes prediction with SBP + availability
- **Features**: Multi-gameweek capability, form weighting, statistical estimation, caching

### `dynamic_team_strength.py`
- **Purpose**: Evolving team strength ratings throughout the season
- **Key Classes**: `DynamicTeamStrength` - Team strength calculator with transition logic
- **Key Methods**:
  - `get_team_strength()` - Main strength calculation with historical transition
  - `_calculate_current_season_strength()` - Current season performance analysis
  - `_load_historical_strength()` - 2024-25 season baseline strength ratings
- **Features**: GW8+ transition to current season, rolling 6-GW windows, venue adjustments

### `fpl_optimization.py`
- **Purpose**: Advanced transfer optimization and budget analysis
- **Key Functions**:
  - `optimize_team_with_transfers()` - Smart 0-3 transfer optimization with scenario analysis
  - `calculate_total_budget_pool()` - Budget analysis including sellable player values
  - `premium_acquisition_planner()` - Multi-transfer scenarios for expensive targets
  - `get_best_starting_11()` - Formation-flexible lineup selection
  - `select_captain()` - Risk-adjusted captaincy recommendations
- **Features**: Constraint handling, budget optimization, transfer cost analysis, scenario comparison

### `fpl_visualization.py`
- **Purpose**: Interactive visualization suite with Plotly integration
- **Key Functions**:
  - `create_team_strength_visualization()` - Dynamic team strength displays
  - `create_player_trends_visualization()` - Historical performance trend setup
  - `create_trends_chart()` - Interactive Plotly-based trend charts
  - `create_fixture_difficulty_visualization()` - Fixture difficulty heatmaps
- **Features**: Error handling, graceful degradation, interactive charts, real-time updates

### `prediction_storage.py`
- **Purpose**: Systematic prediction archival for retro analysis
- **Key Classes**: `PredictionStorage` - Prediction management system
- **Key Methods**:
  - `save_gameweek_predictions()` - Timestamped prediction storage
  - `load_saved_predictions()` - Historical prediction retrieval
  - `compare_predictions_vs_actual()` - Model accuracy validation
- **Features**: JSON metadata storage, CSV prediction data, automated timestamping

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

## Current Implementation Status (v1.2)

**🆕 VERSION 1.2 HIGHLIGHTS:**
- **Complete modular refactor** - Separated monolithic code into 6 specialized modules
- **Enhanced form analytics** - Hot/cold player detection with momentum indicators
- **Dynamic team strength** - Evolving ratings with historical transition logic
- **Advanced visualizations** - Interactive trends, fixture analysis, team strength displays
- **Prediction storage system** - Systematic archival for model validation and improvement
- **Premium acquisition planning** - Multi-transfer scenarios with budget pool analysis
- **Improved error handling** - Graceful fallbacks and validation throughout

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

**✅ COMPLETED - Gameweek Manager (v1.2) - Modular Architecture:**
- **Modular codebase restructure** - Separated concerns into dedicated modules
- **Enhanced data loading** (`fpl_data_loader.py`) - Centralized, robust data orchestration
- **Dedicated XP engine** (`xp_model.py`) - Form-weighted calculations with 70% recent + 30% baseline
- **Dynamic team strength** (`dynamic_team_strength.py`) - Evolving ratings with historical transition
- **Advanced optimization suite** (`fpl_optimization.py`) - Smart 0-3 transfer decisions with budget pool analysis
- **Comprehensive visualizations** (`fpl_visualization.py`) - Interactive charts, trends, and fixture analysis
- **Prediction storage system** (`prediction_storage.py`) - Retro analysis and model validation
- **Form analytics dashboard** - Hot/cold player detection with momentum indicators (🔥📈➡️📉❄️)
- **Player performance trends** - Interactive historical visualization with multi-attribute analysis
- **Fixture difficulty matrix** - Dynamic 5-gameweek heatmaps with venue adjustments
- **Enhanced optimization** - Constraint-based optimization with premium acquisition planning
- **Captain selection tools** - Risk-adjusted recommendations with fixture outlook
- **Transfer analysis engine** - Comprehensive scenario analysis with XP gain calculations
- **Squad management suite** - Budget tracking, constraint validation, and transfer cost analysis

**✅ INTERACTIVE FEATURES (v1.2):**
- **Modular dual-tool workflow**: Season planning + weekly management with shared components
- **Real-time optimization** with constraint handling and smart transfer decisions
- **Formation-flexible lineup selection** - Auto-selects best starting 11 from optimal squad
- **Advanced visualization suite** - Interactive trends, team strength, and fixture difficulty
- **Transfer risk assessment** - Form-based warnings and fixture analysis
- **Budget pool analysis** - Total available funds with sellable value calculations
- **Premium acquisition planning** - Multi-transfer scenarios for expensive targets
- **Live performance monitoring** - Form analytics dashboard with momentum tracking
- **Prediction archival system** - Systematic storage for model validation and improvement
- **Rule compliance validation** - Automatic constraint checking across both tools
- **Multi-gameweek vs single-gameweek** - Strategic vs tactical decision breakdowns

**⏭️ FUTURE ENHANCEMENTS (v1.3+):**
- **Advanced injury prediction** - Return date modeling with medical data integration
- **Price change prediction** - Market intelligence with ownership trend analysis
- **Bonus points (BPS) modeling** - Real-time BPS prediction and captain optimization
- **Historical back-testing framework** - Multi-season validation with performance benchmarking
- **Advanced venue modeling** - Stadium-specific attack/defense rating adjustments
- **Uncertainty quantification** - Confidence intervals and risk assessment
- **Auto-transfer suggestions** - AI-powered transfer recommendations with risk assessment
- **Mini-league strategy optimization** - Differential vs template strategy selection
- **Live performance alerts** - Real-time notifications for form changes and opportunities
- **Advanced sensitivity analysis** - Model robustness testing and parameter optimization

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