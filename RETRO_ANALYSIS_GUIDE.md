# FPL Retro Analysis Guide

## Overview

The retro analysis system allows you to evaluate your FPL model's prediction accuracy and continuously improve it based on actual gameweek results.

## Quick Start

### 1. Save Predictions (Before Each Gameweek)

Run your main model and save predictions:

```bash
marimo run fpl_xp_model.py
```

1. Navigate to the "Save Predictions for Retro Analysis" section at the bottom
2. Enter the current gameweek number
3. Click "Save Current Predictions"
4. Verify the success message shows file paths

### 2. Run Retro Analysis (After Gameweek Completion)

Launch the retro analysis tool:

```bash
marimo run fpl_retro_analysis.py
```

## Available Analysis Types

### 📊 Prediction Accuracy Overview
- **Purpose**: Overall model performance evaluation
- **Metrics**: MAE, RMSE, correlation, prediction bias
- **Insights**: How well your xP predictions matched actual points
- **Use**: Identify if model is systematically over/under-predicting

### 🔧 Model Component Validation  
- **Purpose**: Validate individual model components
- **Components**: Expected goals, assists, clean sheets, minutes
- **Insights**: Which parts of your model are most/least accurate
- **Use**: Focus improvements on specific model weaknesses

### ⭐ Top Performers vs Predictions
- **Purpose**: Identify model blind spots
- **Analysis**: Compare highest scorers vs highest xP players
- **Insights**: Players your model missed or overvalued
- **Use**: Understand systematic biases by player type

### 📍 Position-Based Analysis
- **Purpose**: Accuracy breakdown by player position
- **Metrics**: Prediction error, correlation, and bias by position
- **Insights**: Which positions are hardest to predict accurately
- **Use**: Position-specific model adjustments

### 👥 Team Selection Analysis
- **Purpose**: Evaluate optimal team performance
- **Analysis**: How your 15-player squad actually performed
- **Insights**: Team vs individual player prediction accuracy
- **Use**: Validate team selection algorithm effectiveness

### ⚠️ Transfer Risk Validation
- **Purpose**: Check if "transfer risk" flags were accurate
- **Analysis**: Did players with poor fixture predictions actually perform poorly?
- **Insights**: Fixture difficulty prediction accuracy
- **Use**: Improve transfer timing and fixture analysis

## Workflow for Continuous Improvement

### Weekly Cycle

1. **Monday**: Save predictions for upcoming gameweek
2. **Tuesday-Sunday**: Gameweek plays out
3. **Monday**: Run retro analysis on completed gameweek
4. **Monday**: Identify model improvements based on analysis
5. **Repeat**: Apply learnings to next gameweek predictions

### Monthly Review

1. **Multi-gameweek trends**: Look for patterns across multiple gameweeks
2. **Systematic biases**: Identify consistent over/under-predictions
3. **Model adjustments**: Update parameters based on learnings
4. **Documentation**: Track what changes improved accuracy

## Key Files

### Generated Files
- `predictions/gw{N}_predictions_{timestamp}.csv` - Player xP predictions
- `predictions/gw{N}_optimal_team_{timestamp}.json` - Optimal 15-player squad
- `predictions/gw{N}_metadata_{timestamp}.json` - Model parameters used

### Analysis Files
- `fpl_retro_analysis.py` - Main retro analysis notebook
- `prediction_storage.py` - Prediction saving/loading system
- `fpl_xp_model.py` - Main model (now includes prediction saving)

## Tips for Effective Analysis

### 1. Save Early and Often
- Always save predictions before gameweek deadlines
- Don't wait until after results to save predictions
- Keep multiple prediction versions if you experiment with parameters

### 2. Focus on Patterns
- Single gameweek results can be noisy
- Look for patterns across 3-5 gameweeks
- Systematic biases are more actionable than random errors

### 3. Position-Specific Insights
- Goalkeeper prediction accuracy is different from striker accuracy
- Clean sheet predictions are harder than goal predictions
- Minutes played is often the biggest error source

### 4. Use for Mini-League Edge
- Share insights about differential picks vs template
- Identify undervalued players your model found
- Track your prediction accuracy vs popular picks

### 5. Model Improvement Priority
- Fix largest systematic biases first
- Focus on high-impact players (premium assets)
- Validate that changes actually improve accuracy

## Common Issues and Solutions

### Issue: "No saved predictions found"
**Solution**: Make sure you saved predictions using the main model before running retro analysis

### Issue: "Could not match players with actual results"  
**Solution**: Check that player IDs in historical data match current player IDs

### Issue: "No actual data for this gameweek"
**Solution**: Wait for gameweek to complete and data to be updated

### Issue: Low prediction accuracy
**Possible Causes**:
- Minutes model needs improvement (biggest impact)
- Fixture difficulty scaling issues
- xG/xA rates outdated or inaccurate
- Injury/rotation not properly modeled

## Integration with Mini-Leagues

### Weekly Reports
Create simple summaries to share:
- "Model predicted X points, actual was Y points"  
- "Best prediction: Player Z (predicted 8.2, scored 9)"
- "Worst prediction: Player A (predicted 6.1, scored 1)"

### Differential Analysis
- Track which differentials your model identified correctly
- Compare your optimal team vs template picks
- Share insights about undervalued players

### Prediction Contests
- Compete with league members on prediction accuracy
- Track who has the best model over time
- Share retro analysis insights for collective improvement

This system creates a data-driven approach to FPL that continuously improves through systematic analysis!