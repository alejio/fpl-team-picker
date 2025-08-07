# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Fantasy Premier League (FPL) team picker that uses advanced expected points (xP) modeling to select optimal teams for each gameweek. The project implements a comprehensive mathematical framework combining Poisson distributions, team ratings, player performance metrics, and fixture analysis.

## Dataset Location

The project relies on datasets located in `../fpl-dataset-builder/data/`. Key files include:

- **fpl_players_current.csv**: Current season player data (prices, positions, teams)
- **vaastav_full_player_history_2024_2025.csv**: Comprehensive historical player statistics including form, ICT index, points per game
- **fpl_player_xg_xa_rates.csv**: Expected goals (xG90) and expected assists (xA90) rates per 90 minutes
- **fpl_fixtures_normalized.csv**: Normalized fixture data with team IDs and kickoff times
- **match_results_previous_season.csv**: Historical match results for Poisson baseline calculations
- **fpl_teams_current.csv**: Team reference data for ID lookups
- **injury_tracking_template.csv**: Player availability tracking

See `../fpl-dataset-builder/data/DATASET.md` for complete dataset documentation.

## Expected Points Model (MVP Version)

**Fast afternoon implementation with smart shortcuts:**

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

## Data Processing Notes

- All player IDs are consistent across datasets for joining
- Team IDs reference fpl_teams_current.csv
- UTC timestamps throughout
- Handle missing data with appropriate fallbacks
- Use pandas for data manipulation and merging

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

**Primary Interface:**
```bash
# Launch the interactive Marimo notebook (main interface)
marimo run fpl_xp_model.py

# Or run in development/edit mode
marimo edit fpl_xp_model.py
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

**✅ COMPLETED (v0.1):**
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

**✅ INTERACTIVE FEATURES:**
- Real-time squad optimization with constraint handling
- Starting 11 auto-selection from best formation
- Transfer risk warnings for poor GW2-3 fixtures
- Budget and efficiency analysis
- Rule compliance validation
- Multi-gameweek breakdown per player

**🔄 WHAT WE BUILT TODAY:**
- Complete functional FPL team picker with advanced xP modeling
- Statistical player performance prediction using xG/xA rates
- Sophisticated optimization avoiding local optima
- User-friendly interface for team customization
- Comprehensive validation against FPL rules and constraints

**⏭️ FUTURE ENHANCEMENTS:**
- Historical back-testing and validation
- Complex minute modeling from game logs
- Venue-specific attack/defense ratings
- Injury data integration
- Bonus points (BPS) modeling
- Sensitivity analysis and uncertainty quantification

## Future Improvements

### Feature Roadmap
- **Heatmap Visualization**:
  - Show upcoming 5 fixtures and difficulty for each team in a heatmap
- **Transfer Window Analysis**:
  - Investigate why new transfers like Wirtz and Ekitike aren't getting picked by the optimiser
  - Need deeper understanding of transfer impact on expected points model