# FPL Team Picker 🏆

**A comprehensive Fantasy Premier League analysis suite with season-start team building and weekly gameweek management.**

This project provides two complementary tools: a multi-gameweek optimizer for season planning and a form-weighted gameweek manager for weekly decisions. Both use advanced Expected Points (xP) modeling with live data integration for optimal FPL decision making.

## 🎯 What This Does

This suite provides two specialized tools for different FPL decision points:

### 🏗️ Season-Start Team Builder (`fpl_xp_model.py`)
**Multi-gameweek optimization for initial squad building**
- **5-week xP horizon** - Weighted predictions across GW1-5 for season planning
- **Simulated Annealing optimization** - Finds globally competitive 15-player squads
- **Formation flexibility** - Automatically selects best starting 11 from 8 valid formations
- **Transfer risk analysis** - Identifies players with poor upcoming fixtures

### ⚡ Weekly Gameweek Manager (`fpl_gameweek_manager.py`)
**Form-weighted predictions with smart optimization for weekly decisions**
- **Live data integration** - Real-time performance tracking and market movements
- **Form-weighted xP** - Blends recent performance (70%) with season baseline (30%)
- **Smart transfer optimization** - Auto-selects optimal 0-3 transfers based on net XP after penalties
- **Constraint-based optimization** - Pre-optimization player inclusion/exclusion controls
- **Comprehensive scenario analysis** - All transfer options compared with XP gain calculations
- **Captain selection** - Risk-adjusted captaincy recommendations
- **Retro analysis** - Post-gameweek validation and model improvement

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Access to fpl-dataset-builder (database client for FPL data)

### Installation
```bash
# Clone both repositories
git clone <fpl-team-picker-url>
git clone <fpl-dataset-builder-url>

# Install team picker with database dependency
cd fpl-team-picker
uv sync  # Automatically installs fpl-dataset-builder from local path
```

### Basic Usage

> **Prerequisites:** Ensure fpl-dataset-builder database is populated with current FPL data before running analysis tools.

**Season Start - Initial Squad Building:**
```bash
# Launch season-start team builder
marimo run fpl_xp_model.py
```

**Weekly Management - Gameweek Decisions:**
```bash
# Launch weekly gameweek manager
marimo run fpl_gameweek_manager.py
```

**Development Mode:**
```bash
# Edit mode for development
marimo edit fpl_xp_model.py          # Season planning tool
marimo edit fpl_gameweek_manager.py  # Weekly management tool
```

### Weekly Workflow

**Monday (Pre-Gameweek):**
1. Run `fpl_gameweek_manager.py` - Retro analysis section
2. Save predictions using the prediction storage section

**Tuesday-Friday (Planning):**
1. Run `fpl_gameweek_manager.py` - Analyze upcoming gameweek
2. Use transfer analysis and captain selection tools
3. Optimize starting 11 and make transfer decisions

**Weekend (Gameweek Active):**
1. Monitor live performance data
2. Track momentum indicators and form changes

The interfaces provide interactive controls for:
- Pre-optimization constraint customization (must-include/exclude players)
- Smart transfer optimization with automatic 0-3 transfer selection
- Real-time optimization with live data and comprehensive scenario analysis
- Transfer analysis with hit calculations and XP gain comparisons
- Formation flexibility and lineup optimization

## 📊 How It Works

### 1. Multi-Gameweek vs Single-Gameweek Models

**Season-Start Model (Multi-GW):**
- **5-week horizon** - Weighted xP across GW1-5 (1.0, 0.9, 0.8, 0.7, 0.6)
- **Enhanced minutes model** - Selected By Percentage (SBP) + availability status
- **Statistical xG/xA estimation** - For new transfers using price, position, team strength
- **Transfer risk flagging** - Identifies poor GW2-3 fixtures for planning

**Weekly Model (Single-GW):**
- **Form-weighted predictions** - Recent performance (70%) + season baseline (30%)
- **Live data integration** - Real-time performance and market movements
- **Momentum tracking** - Visual form indicators (🔥📈➡️📉❄️)
- **Dynamic adjustments** - Performance deltas and availability updates

### 2. Expected Points Calculation

**Core Components:**
- **xG90/xA90 rates** - Expected goals and assists per 90 minutes
- **Minutes prediction** - SBP-based start probabilities with injury/suspension status
- **Fixture difficulty** - Team strength scaling [0.7-1.3] based on opponent quality
- **FPL scoring** - Goals (6/5/4 by position), assists (×3), clean sheets, appearances

### 3. Team Optimization (Simulated Annealing)

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

## 🗄️ Data Architecture

This project uses the **fpl-dataset-builder** database client for all data access:

### Database Integration
- **Centralized SQLite database** - Single source of truth for all FPL data
- **Real-time updates** - Always fresh data with no stale CSV files
- **Better performance** - Database queries faster than file I/O
- **Automatic management** - No manual file handling required

### Data Access
```python
# Load data from database
from client import (
    get_current_players,    # Current season player data
    get_current_teams,      # Team reference data  
    get_fixtures_normalized, # Fixture data with team IDs
    get_player_xg_xa_rates, # xG90/xA90 rates per player
    get_gameweek_live_data, # Real-time gameweek performance
    get_player_deltas_current # Week-over-week tracking
)
```

### Available Data
- **Current season players** - Prices, positions, teams, availability status
- **xG/xA rates** - Expected goals and assists per 90 minutes
- **Fixtures** - Normalized with team IDs and difficulty ratings
- **Live gameweek data** - Real-time performance tracking
- **Performance deltas** - Week-over-week trends and market movements
- **Historical statistics** - Comprehensive player performance history

> **Note:** The fpl-dataset-builder handles all data fetching, processing, and storage. This project focuses purely on analysis and optimization.

## ⚡ Implementation Status

**✅ COMPLETED Features:**

**Season-Start Builder (v1.0):**
- Multi-gameweek xP calculations with 5-week weighted horizon
- Statistical xG/xA estimation for new transfers and missing data
- Enhanced minutes model using Selected By Percentage (SBP) + availability
- Simulated Annealing optimization with constraint satisfaction
- Formation-flexible starting 11 selection (8 valid formations)
- Transfer risk analysis and fixture difficulty assessment

**Weekly Gameweek Manager (v1.1):**
- Live data integration with real-time performance tracking
- Form-weighted xP calculations blending recent performance with baseline
- Interactive fixture difficulty visualization with dynamic team strength ratings
- Performance delta analysis and momentum tracking
- Smart transfer optimization with automatic 0-3 transfer decision based on net XP
- Constraint-based optimization with pre-optimization player inclusion/exclusion controls
- Comprehensive scenario analysis comparing all transfer options with XP gain calculations
- Paginated player display showing all players ranked by expected points
- Transfer analysis engine with hit calculations
- Captain selection tools with risk assessment
- Retro analysis framework for model validation

**🔄 Future Enhancements:**
- Price change prediction and market intelligence
- Historical back-testing across multiple seasons
- Advanced injury data integration
- Bonus points (BPS) modeling
- Uncertainty quantification and sensitivity analysis
