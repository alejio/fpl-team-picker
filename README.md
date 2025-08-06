# FPL Team Picker 🏆

**An advanced Fantasy Premier League team optimizer using Expected Points (xP) modeling and Simulated Annealing optimization.**

Build optimal FPL squads by combining statistical player analysis, fixture difficulty assessment, and sophisticated optimization algorithms to maximize expected points within budget and formation constraints.

## 🎯 What This Does

This tool solves one of Fantasy Premier League's core challenges: **selecting the optimal 15-player squad from 600+ players while respecting complex constraints**. Instead of relying on gut feeling or basic price-per-point metrics, it uses:

- **Expected Points (xP) modeling** - Predicts player performance using xG/xA rates, fixture difficulty, and team strength
- **Simulated Annealing optimization** - Finds globally competitive team combinations that escape local optima
- **Formation flexibility** - Automatically selects the best starting 11 formation from your 15-player squad

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Access to FPL datasets (see [Dataset Requirements](#dataset-requirements))

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd fpl-team-picker

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Running the Model
```bash
# Launch the interactive Marimo notebook
marimo run fpl_xp_model.py

# Or run in edit mode for development
marimo edit fpl_xp_model.py
```

The notebook will open in your browser, providing an interactive interface to:
1. Load and process FPL data
2. Calculate expected points for all players
3. Run team optimization
4. View your optimal squad and starting 11

## 📊 How It Works

### 1. Expected Points Calculation

**Player Performance Modeling:**
- **xG90/xA90 rates** - Expected goals and assists per 90 minutes from historical data
- **Minutes prediction** - Smart proxy using player prices (>£5.0m = starters, ≤£5.0m = rotation)
- **Fixture difficulty** - Team strength adjustments based on opponent quality [0.7-1.3 scaling]

**FPL Points Conversion:**
- Appearance points (1-2 based on minutes)
- Goals (position multipliers: GK/DEF×6, MID×5, FWD×4)
- Assists (×3 multiplier)
- Clean sheets (simplified probability by team strength)

### 2. Team Optimization (Simulated Annealing)

**Why Simulated Annealing?**
The FPL team selection problem has ~10^15 possible combinations. Greedy algorithms get trapped in local optima, while Simulated Annealing explores the solution space intelligently to find globally competitive teams.

**Algorithm Process:**
1. **Initialize** - Generate random valid 15-player squad
2. **Iterate** - Swap players while respecting constraints
3. **Accept/Reject** - Better solutions always accepted; worse solutions accepted with temperature-based probability
4. **Cool down** - Temperature decreases over 5,000 iterations
5. **Optimize formation** - Find best starting 11 from 8 valid formations

**Constraints Satisfied:**
- 15 players: 2 GKP, 5 DEF, 5 MID, 3 FWD
- £100m budget limit
- Maximum 3 players per real team
- Valid starting 11 formations

## 📈 Output & Results

The model provides:

**Squad Analysis:**
- Optimal 15-player squad with expected points
- Best starting 11 formation and lineup
- Budget utilization and remaining funds
- xP per £1m efficiency metrics

**Performance Insights:**
- Player-by-player xP breakdown
- Team constraint compliance
- Formation flexibility analysis
- Bench vs starter value distribution

## 📁 Dataset Requirements

This project requires FPL datasets located in `../fpl-dataset-builder/data/`:

**Core Files:**
- `fpl_players_current.csv` - Current season player data (prices, positions, teams)
- `fpl_player_xg_xa_rates.csv` - xG90/xA90 rates per player
- `fpl_fixtures_normalized.csv` - Fixture data with team IDs
- `fpl_teams_current.csv` - Team reference data

**Optional Enhancement Files:**
- `vaastav_full_player_history_2024_2025.csv` - Historical player statistics
- `match_results_previous_season.csv` - Historical match results
- `injury_tracking_template.csv` - Player availability tracking

> **Note:** The dataset builder is a separate project. See `../fpl-dataset-builder/data/DATASET.md` for complete dataset documentation.

## ⚡ MVP Implementation Notes

This is a **fast afternoon implementation** with smart shortcuts for speed:

**Simplified Assumptions:**
- Hardcoded league baselines (μ_home=1.43, μ_away=1.15)
- Team strength from final table positions
- Minutes prediction via price proxy
- Single-point estimates (no uncertainty modeling)

**What's Included:**
- ✅ Core xP calculations for all players
- ✅ Fixture difficulty adjustments
- ✅ Simulated Annealing optimization
- ✅ Formation flexibility
- ✅ Constraint satisfaction

**Future Enhancements:**
- Historical back-testing and validation
- Complex minute modeling from game logs
- Bonus points (BPS) integration
- Injury data incorporation
- Uncertainty quantification
