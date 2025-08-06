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

### 3. Minutes Expectation (Price Proxy)
- Players >£5.0m: 75 minutes (assumed starters)
- Players ≤£5.0m: 30 minutes (assumed bench/rotation)
- Skip historical minute analysis

### 4. Player xP Calculation
- **Per-minute production**: xG90 and xA90 from rates dataset
- **Fixture scaling**: Multiply by opponent difficulty [0.7, 1.3]
- **Expected contributions**: Scale by expected minutes
- **FPL points conversion**:
  - Appearance: 2 pts if ≥60 mins, 1 pt if >0 mins
  - Goals: Position multipliers (GK/DEF: 6, MID: 5, FWD: 4)
  - Assists: 3 × xA_exp
  - Clean sheets: Simplified P(CS) by team strength

## Data Processing Notes

- All player IDs are consistent across datasets for joining
- Team IDs reference fpl_teams_current.csv
- UTC timestamps throughout
- Handle missing data with appropriate fallbacks
- Use pandas for data manipulation and merging

## Development Workflow (MVP)

**Single afternoon implementation:**

1. **Load core datasets** (players, xG rates, fixtures, teams)
2. **Apply simplified team strength** using final table positions
3. **Calculate fixture difficulty scaling** [0.7, 1.3] 
4. **Assign minutes by price threshold** (>£5m = 75 mins, ≤£5m = 30 mins)
5. **Calculate xP per player** using xG90/xA90 rates + scaling + FPL scoring
6. **Team selection**: Simulated Annealing optimization for optimal squad selection

## Simulation Methodology

### Team Optimization Algorithm (Simulated Annealing)

The model uses **Simulated Annealing** to solve the complex combinatorial optimization problem of selecting the optimal 15-player FPL squad. This approach was chosen over greedy algorithms to escape local optima and find globally competitive solutions.

#### Core Optimization Problem
- **Objective**: Maximize expected points for starting 11 players
- **Squad Constraints**: 
  - 15 players total: 2 GKP, 5 DEF, 5 MID, 3 FWD
  - £100m budget limit
  - Max 3 players per real team
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

Since this is a new project, common commands will be added as the codebase develops. Likely to include:
- Python script execution for xP calculations
- Data validation and testing commands
- Team optimization algorithms

## Key Assumptions (MVP Shortcuts)

- **Single-point estimates** (no uncertainty propagation)
- **Minutes by price proxy** (>£5m starter, ≤£5m bench/rotation)
- **Hardcoded league baselines** (μ_home=1.43, μ_away=1.15)
- **Team strength from final table** positions instead of complex ratings
- **No BPS/bonus, cards, saves, penalties** in v0.1
- **Simplified clean sheet probabilities** based on team strength only
- **One fixture difficulty factor** affects both attack and defense equally

## MVP Scope

**What we're building today:**
- Core xP calculations for all players
- GW1 fixture difficulty adjustments
- Optimal team selection within budget/formation constraints
- Quick validation that premium players rank highly

**What we're skipping:**
- Historical back-testing and validation
- Complex minute modeling from game logs
- Venue-specific attack/defense ratings
- Injury data integration
- Sensitivity analysis