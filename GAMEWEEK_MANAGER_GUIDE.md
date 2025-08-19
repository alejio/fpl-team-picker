# FPL Gameweek Manager Guide

## Overview

The FPL Gameweek Manager is your weekly decision-making companion, designed to complement your season-start team builder. Use this tool every gameweek to optimize your lineup, transfers, and captaincy decisions.

## Quick Start

### Launch the Tool
```bash
marimo run fpl_gameweek_manager.py
```

### Basic Workflow
1. **Input your current squad** (15 players + budget info)
2. **Analyze single gameweek xP** for upcoming fixtures  
3. **Optimize starting 11** with formation selector
4. **Evaluate transfer options** and scenarios
5. **Select captain/vice-captain** based on xP analysis
6. **Get decision summary** ready for FPL app

## Detailed Features

### 🏟️ Current Squad Setup

**Input Requirements:**
- **Gameweek Number:** Current/upcoming gameweek
- **Money ITB:** Available budget (£0.0-£50.0m)
- **Free Transfers:** Number of free transfers (0-5)
- **Squad Selection:** Your exact 15-player FPL squad

**Squad Validation:**
- ✅ **Formation Check:** 2 GKP, 5 DEF, 5 MID, 3 FWD
- ✅ **Team Limits:** Max 3 players per Premier League team
- ✅ **Budget Analysis:** Total value + money ITB tracking

### 📊 Single Gameweek xP Calculator

**Features:**
- **Fixture-Specific:** Analyzes only the upcoming gameweek
- **Real-Time Difficulty:** Team strength vs opponent analysis
- **Minutes Prediction:** Based on ownership % and availability status
- **xP Components:** Goals, assists, clean sheets, appearance points

**Key Differences from Multi-GW Model:**
- Single gameweek focus (no temporal weighting)
- Current form emphasis over long-term projections
- Real-time availability status integration

### ⚽ Starting 11 Optimizer

**Formation Options:**
- **3-4-3:** Attacking formation for high-scoring gameweeks
- **3-5-2:** Midfield-heavy for creative players
- **4-3-3:** Balanced with strong forward line
- **4-4-2:** Classic balanced formation
- **4-5-1:** Defensive with midfield focus
- **5-3-2:** Ultra-defensive for difficult fixtures
- **5-4-1:** Conservative with single striker

**Optimization Logic:**
1. Groups players by position
2. Ranks by single gameweek xP
3. Selects best players for chosen formation
4. Auto-orders bench by xP potential

### 🔄 Transfer Analysis

**Transfer Scenarios:**
- **1-Transfer Analysis:** Most common weekly decisions
- **Position-by-Position:** Find best replacement options
- **Budget Constraints:** Respects money ITB + player sale value
- **Hit Analysis:** -4 point penalty vs xP gain calculations

**Transfer Metrics:**
- **xP Gain:** Expected point improvement
- **Transfer Cost:** Free transfer or -4 hit
- **Net Gain:** Total benefit after penalties
- **Value Rating:** xP per £1m efficiency

### 👑 Captain & Vice-Captain Selection

**Analysis Features:**
- **Expected Points:** Base xP for each player
- **Captain Multiplier:** 2x points calculation
- **Ceiling/Floor:** Optimistic and pessimistic scenarios
- **Risk Assessment:** Variance in expected outcomes

**Captain Metrics:**
- **Captain xP:** Doubled expected points
- **Ceiling xP:** Best-case scenario (3x base)
- **Floor xP:** Worst-case scenario (0.5x base)

### 📋 Decision Summary

**Ready-to-Implement Output:**
- **Starting 11:** Organized by position
- **Formation:** Clear position structure
- **Captain/Vice:** Top recommendations
- **Budget Status:** Money ITB and transfers
- **Expected Performance:** Total xP projections

## Usage Tips

### Weekly Routine

**Tuesday-Wednesday (Post-Gameweek):**
1. Input your current squad after price changes
2. Check availability status updates
3. Review upcoming fixtures

**Thursday-Friday (Analysis):**
1. Calculate single gameweek xP
2. Analyze transfer scenarios
3. Plan optimal formation

**Saturday-Sunday (Decision Day):**
1. Finalize transfers before deadline
2. Set starting 11 and captaincy
3. Export decisions to FPL app

### Strategic Considerations

**Formation Selection:**
- **Easy Fixtures:** Attacking formations (3-4-3, 4-3-3)
- **Difficult Fixtures:** Defensive formations (5-4-1, 5-3-2)
- **Balanced Gameweeks:** 4-4-2 or 4-5-1

**Transfer Strategy:**
- **Free Transfers:** Use them or lose them
- **Transfer Hits:** Only take if net gain > 4 points
- **Price Changes:** Factor in overnight price movements
- **Injury Rotation:** Reactive vs proactive transfers

**Captaincy Decisions:**
- **High Floor:** Reliable performers for safe points
- **High Ceiling:** Explosive players for rank climbs
- **Fixture Quality:** Weight opponent strength heavily
- **Ownership %:** Consider differential captains for rank gains

## Integration with Other Tools

### With Season-Start Builder (`fpl_xp_model.py`)
- Use builder for initial 15-player squad creation
- Switch to manager for weekly optimization
- Cross-reference long-term vs short-term xP

### With Retro Analysis (`fpl_retro_analysis.py`)
- Save gameweek decisions for later analysis
- Track captain success rates
- Validate transfer decision quality

### Data Pipeline
1. **Season Start:** Use `fpl_xp_model.py` for initial squad
2. **Weekly:** Use `fpl_gameweek_manager.py` for decisions
3. **Post-Gameweek:** Use `fpl_retro_analysis.py` for learning

## Common Decision Scenarios

### Scenario 1: Template vs Differential Captain
**Template (High Ownership):**
- Lower risk, follows crowd
- Good for maintaining rank
- Safer floor, limited ceiling

**Differential (Low Ownership):**
- Higher risk, contrarian pick
- Good for rank climbing
- Lower floor, higher ceiling

### Scenario 2: Free Transfer vs Hit
**Free Transfer Logic:**
- Always use free transfers if beneficial
- Bank only if no clear improvements
- Consider 2-week planning horizon

**Transfer Hit Logic:**
- Only take if net gain > 4 points
- Factor in future gameweeks
- Emergency only for injuries

### Scenario 3: Formation Changes
**Attacking (3-4-3, 4-3-3):**
- High-scoring gameweeks expected
- Premium forwards in good form
- Weak defenses facing each other

**Defensive (5-4-1, 5-3-2):**
- Low-scoring gameweeks expected
- Strong defensive matchups
- Midfield value over forwards

## Troubleshooting

### Common Issues

**Issue:** "Not enough players for selected formation"
**Solution:** Check if your squad has minimum players per position for formation

**Issue:** "No beneficial transfer scenarios found"
**Solution:** Your squad may already be optimal, or try different budget assumptions

**Issue:** "Squad validation failed"
**Solution:** Ensure exactly 15 players selected with correct position distribution

### Performance Tips

**Data Refresh:**
- Restart notebook after major data updates
- Check player availability status regularly
- Verify fixture changes and postponements

**Decision Accuracy:**
- Run analysis closer to deadline for latest data
- Consider weather and late team news
- Factor in rotation risk for popular players

This tool is designed to make your weekly FPL decisions data-driven and efficient. Use it consistently to improve your rank and points total throughout the season!