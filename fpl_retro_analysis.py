import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    mo.md(
        r"""
        # FPL Gameweek Retro Analysis

        **Comprehensive performance evaluation and model validation**
        
        Analyze prediction accuracy, validate model components, and generate actionable insights 
        for improving your FPL team picker over time.
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import warnings
    warnings.filterwarnings('ignore')
    
    # Data paths
    DATA_DIR = Path("../fpl-dataset-builder/data/")
    PREDICTIONS_DIR = Path("predictions/")  # For storing historical predictions
    
    # Create predictions directory if it doesn't exist
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    
    # Import prediction storage system
    from prediction_storage import PredictionStorage
    
    return DATA_DIR, PREDICTIONS_DIR, Path, np, pd, plt, sns, warnings, PredictionStorage


@app.cell
def __():
    mo.md("## Data Loading & Integration")
    return


@app.cell
def __(DATA_DIR, pd):
    def load_retro_datasets():
        """Load datasets needed for retrospective analysis"""
        print("Loading datasets for retro analysis...")
        
        # Historical gameweek data (actual results)
        historical_data = pd.read_csv(DATA_DIR / "fpl_historical_gameweek_data.csv")
        
        # Current players for mapping
        players_current = pd.read_csv(DATA_DIR / "fpl_players_current.csv")
        
        # Teams for reference
        teams = pd.read_csv(DATA_DIR / "fpl_teams_current.csv")
        
        # xG/xA rates for model validation
        xg_rates = pd.read_csv(DATA_DIR / "fpl_player_xg_xa_rates.csv")
        
        print(f"Loaded: {len(historical_data)} historical records, {len(players_current)} current players")
        
        return historical_data, players_current, teams, xg_rates
    
    historical_data, players_current, teams, xg_rates = load_retro_datasets()
    return historical_data, load_retro_datasets, players_current, teams, xg_rates


@app.cell
def __(mo):
    mo.md("### Available Gameweeks")
    return


@app.cell
def __(historical_data, mo):
    # Check available gameweeks in the data
    available_gws = sorted(historical_data['GW'].unique())
    print(f"Available gameweeks: {available_gws}")
    
    # Show sample of historical data structure
    sample_data = historical_data[['name', 'position', 'team', 'GW', 'total_points', 'xP', 'minutes', 'goals_scored', 'assists']].head(10)
    
    mo.vstack([
        mo.md(f"**Available Gameweeks:** {', '.join(map(str, available_gws))}"),
        mo.md("**Sample Historical Data:**"),
        mo.ui.table(sample_data, page_size=10)
    ])
    return available_gws, sample_data


@app.cell
def __(mo):
    mo.md("## Gameweek Selection & Analysis")
    return


@app.cell
def __(available_gws, mo):
    # Create gameweek selector
    gw_selector = mo.ui.dropdown(
        options=[{"label": f"GW {gw}", "value": gw} for gw in available_gws],
        value=available_gws[0] if available_gws else 1,
        label="Select Gameweek for Analysis"
    )
    
    # Analysis type selector
    analysis_type = mo.ui.dropdown(
        options=[
            {"label": "Prediction Accuracy Overview", "value": "accuracy"},
            {"label": "Model Component Validation", "value": "components"},
            {"label": "Top Performers vs Predictions", "value": "performers"},
            {"label": "Position-Based Analysis", "value": "positions"},
            {"label": "Team Selection Analysis", "value": "team_selection"},
            {"label": "Transfer Risk Validation", "value": "transfers"}
        ],
        value="accuracy",
        label="Analysis Type"
    )
    
    mo.vstack([
        mo.md("**Select Analysis Parameters:**"),
        gw_selector,
        analysis_type
    ])
    return analysis_type, gw_selector


@app.cell
def __(gw_selector, historical_data):
    # Filter data for selected gameweek
    if gw_selector.value is not None:
        gw_data = historical_data[historical_data['GW'] == gw_selector.value].copy()
        print(f"Analyzing GW {gw_selector.value}: {len(gw_data)} player records")
    else:
        gw_data = pd.DataFrame()
        print("No gameweek selected")
    
    return (gw_data,)


@app.cell
def __(mo):
    mo.md("## Prediction Accuracy Analysis")
    return


@app.cell
def __(gw_data, analysis_type, pd, np, plt, sns, mo):
    def analyze_prediction_accuracy(gw_df):
        """Analyze how well xP predictions matched actual points"""
        if len(gw_df) == 0:
            return pd.DataFrame(), {}
        
        # Clean data - ensure we have both predictions and actuals
        valid_data = gw_df.dropna(subset=['xP', 'total_points']).copy()
        
        if len(valid_data) == 0:
            return pd.DataFrame(), {"error": "No valid prediction/actual pairs found"}
        
        # Calculate prediction accuracy metrics
        valid_data['prediction_error'] = valid_data['total_points'] - valid_data['xP']
        valid_data['absolute_error'] = abs(valid_data['prediction_error'])
        valid_data['percentage_error'] = np.where(
            valid_data['total_points'] != 0,
            (valid_data['prediction_error'] / valid_data['total_points']) * 100,
            0
        )
        
        # Summary statistics
        metrics = {
            'total_players': len(valid_data),
            'mean_absolute_error': valid_data['absolute_error'].mean(),
            'rmse': np.sqrt((valid_data['prediction_error'] ** 2).mean()),
            'correlation': valid_data['xP'].corr(valid_data['total_points']),
            'mean_predicted': valid_data['xP'].mean(),
            'mean_actual': valid_data['total_points'].mean(),
            'prediction_bias': valid_data['prediction_error'].mean()  # Positive = under-predicted
        }
        
        return valid_data, metrics
    
    def create_accuracy_visualization(valid_data, metrics):
        """Create prediction accuracy visualizations"""
        if len(valid_data) == 0:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Predicted vs Actual scatter plot
        axes[0, 0].scatter(valid_data['xP'], valid_data['total_points'], alpha=0.6)
        axes[0, 0].plot([0, valid_data[['xP', 'total_points']].max().max()], 
                       [0, valid_data[['xP', 'total_points']].max().max()], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Predicted xP')
        axes[0, 0].set_ylabel('Actual Points')
        axes[0, 0].set_title(f'Predicted vs Actual (r={metrics["correlation"]:.3f})')
        
        # 2. Prediction error distribution
        axes[0, 1].hist(valid_data['prediction_error'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.8)
        axes[0, 1].set_xlabel('Prediction Error (Actual - Predicted)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'Error Distribution (Bias: {metrics["prediction_bias"]:.2f})')
        
        # 3. Error by position
        if 'position' in valid_data.columns:
            position_errors = valid_data.groupby('position')['absolute_error'].mean().sort_values()
            axes[1, 0].bar(position_errors.index, position_errors.values)
            axes[1, 0].set_xlabel('Position')
            axes[1, 0].set_ylabel('Mean Absolute Error')
            axes[1, 0].set_title('Prediction Accuracy by Position')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Top prediction errors
        worst_predictions = valid_data.nlargest(10, 'absolute_error')[['name', 'position', 'xP', 'total_points', 'absolute_error']]
        axes[1, 1].barh(range(len(worst_predictions)), worst_predictions['absolute_error'])
        axes[1, 1].set_yticks(range(len(worst_predictions)))
        axes[1, 1].set_yticklabels([f"{row['name']} ({row['position']})" for _, row in worst_predictions.iterrows()], fontsize=8)
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_title('Worst Predictions')
        
        plt.tight_layout()
        return fig
    
    if analysis_type.value == "accuracy" and len(gw_data) > 0:
        accuracy_data, accuracy_metrics = analyze_prediction_accuracy(gw_data)
        
        if 'error' not in accuracy_metrics:
            # Create visualization
            fig = create_accuracy_visualization(accuracy_data, accuracy_metrics)
            
            # Show metrics
            metrics_display = mo.md(f"""
            ### Prediction Accuracy Metrics (GW {gw_selector.value})
            
            - **Players Analyzed:** {accuracy_metrics['total_players']}
            - **Mean Absolute Error:** {accuracy_metrics['mean_absolute_error']:.2f} points
            - **RMSE:** {accuracy_metrics['rmse']:.2f} points  
            - **Correlation:** {accuracy_metrics['correlation']:.3f}
            - **Prediction Bias:** {accuracy_metrics['prediction_bias']:.2f} points {'(Over-predicting)' if accuracy_metrics['prediction_bias'] < 0 else '(Under-predicting)' if accuracy_metrics['prediction_bias'] > 0 else '(Unbiased)'}
            - **Mean Predicted:** {accuracy_metrics['mean_predicted']:.2f} xP
            - **Mean Actual:** {accuracy_metrics['mean_actual']:.2f} points
            """)
            
            # Show plot
            plt.show()
            
            accuracy_display = mo.vstack([
                metrics_display,
                mo.md("### Top Prediction Errors"),
                mo.ui.table(accuracy_data.nlargest(15, 'absolute_error')[
                    ['name', 'position', 'team', 'xP', 'total_points', 'prediction_error', 'absolute_error']
                ].round(2), page_size=15)
            ])
        else:
            accuracy_display = mo.md(f"**Error:** {accuracy_metrics['error']}")
    else:
        accuracy_display = mo.md("Select 'Prediction Accuracy Overview' to see accuracy analysis.")
    
    accuracy_display
    return accuracy_data, accuracy_metrics, analyze_prediction_accuracy


@app.cell
def __(mo):
    mo.md("## Model Component Validation")
    return


@app.cell
def __(gw_data, analysis_type, gw_selector, pd, np, plt, mo):
    def analyze_model_components(gw_df):
        """Analyze individual model components vs actual results"""
        if len(gw_df) == 0:
            return {}, {}
        
        # Filter for players who actually played
        played_data = gw_df[gw_df['minutes'] > 0].copy()
        
        if len(played_data) == 0:
            return {}, {"error": "No players with minutes found"}
        
        analysis = {}
        
        # 1. Minutes prediction analysis
        if 'expected_minutes' in played_data.columns:
            # For now, we'll estimate expected minutes from xP model logic
            # This would ideally be stored from prediction time
            pass
        
        # 2. Goals analysis (xG vs actual)
        if 'expected_goals' in played_data.columns:
            xg_data = played_data.dropna(subset=['expected_goals', 'goals_scored'])
            if len(xg_data) > 0:
                analysis['goals'] = {
                    'predicted_total': xg_data['expected_goals'].sum(),
                    'actual_total': xg_data['goals_scored'].sum(),
                    'correlation': xg_data['expected_goals'].corr(xg_data['goals_scored']),
                    'players': len(xg_data)
                }
        
        # 3. Assists analysis (xA vs actual)  
        if 'expected_assists' in played_data.columns:
            xa_data = played_data.dropna(subset=['expected_assists', 'assists'])
            if len(xa_data) > 0:
                analysis['assists'] = {
                    'predicted_total': xa_data['expected_assists'].sum(),
                    'actual_total': xa_data['assists'].sum(),
                    'correlation': xa_data['expected_assists'].corr(xa_data['assists']),
                    'players': len(xa_data)
                }
        
        # 4. Clean sheets analysis
        clean_sheet_data = played_data[played_data['position'].isin(['GKP', 'DEF'])]
        if len(clean_sheet_data) > 0:
            analysis['clean_sheets'] = {
                'players_played': len(clean_sheet_data),
                'clean_sheets': (clean_sheet_data['goals_conceded'] == 0).sum(),
                'clean_sheet_rate': (clean_sheet_data['goals_conceded'] == 0).mean()
            }
        
        # 5. High/Low performers vs predictions
        played_data['performance_category'] = pd.cut(
            played_data['total_points'], 
            bins=[-np.inf, 2, 5, 10, np.inf], 
            labels=['Poor (0-2)', 'Average (3-5)', 'Good (6-10)', 'Excellent (11+)']
        )
        
        performance_analysis = played_data.groupby('performance_category').agg({
            'xP': 'mean',
            'total_points': 'mean',
            'name': 'count'
        }).round(2)
        performance_analysis.columns = ['Mean xP', 'Mean Actual', 'Players']
        
        return analysis, {'performance_by_category': performance_analysis}
    
    if analysis_type.value == "components" and len(gw_data) > 0:
        component_analysis, component_details = analyze_model_components(gw_data)
        
        if 'error' not in component_details:
            components_display = mo.vstack([
                mo.md(f"### Model Component Analysis (GW {gw_selector.value})"),
                
                # Goals analysis
                mo.md("**Goals (xG vs Actual):**") if 'goals' in component_analysis else mo.md(""),
                mo.md(f"""
                - Predicted Goals: {component_analysis.get('goals', {}).get('predicted_total', 'N/A'):.1f}
                - Actual Goals: {component_analysis.get('goals', {}).get('actual_total', 'N/A')}
                - Correlation: {component_analysis.get('goals', {}).get('correlation', 'N/A'):.3f}
                - Players: {component_analysis.get('goals', {}).get('players', 'N/A')}
                """) if 'goals' in component_analysis else mo.md("*Goals data not available*"),
                
                # Assists analysis  
                mo.md("**Assists (xA vs Actual):**") if 'assists' in component_analysis else mo.md(""),
                mo.md(f"""
                - Predicted Assists: {component_analysis.get('assists', {}).get('predicted_total', 'N/A'):.1f}
                - Actual Assists: {component_analysis.get('assists', {}).get('actual_total', 'N/A')}
                - Correlation: {component_analysis.get('assists', {}).get('correlation', 'N/A'):.3f}
                - Players: {component_analysis.get('assists', {}).get('players', 'N/A')}
                """) if 'assists' in component_analysis else mo.md("*Assists data not available*"),
                
                # Clean sheets
                mo.md("**Clean Sheets (Defenders/Goalkeepers):**") if 'clean_sheets' in component_analysis else mo.md(""),
                mo.md(f"""
                - Players: {component_analysis.get('clean_sheets', {}).get('players_played', 'N/A')}
                - Clean Sheets: {component_analysis.get('clean_sheets', {}).get('clean_sheets', 'N/A')}
                - Clean Sheet Rate: {component_analysis.get('clean_sheets', {}).get('clean_sheet_rate', 0):.1%}
                """) if 'clean_sheets' in component_analysis else mo.md("*Clean sheets data not available*"),
                
                # Performance categories
                mo.md("**Performance by Category:**"),
                mo.ui.table(component_details['performance_by_category']) if 'performance_by_category' in component_details else mo.md("*Performance data not available*")
            ])
        else:
            components_display = mo.md(f"**Error:** {component_details['error']}")
    else:
        components_display = mo.md("Select 'Model Component Validation' to see component analysis.")
    
    components_display
    return component_analysis, analyze_model_components


@app.cell
def __(mo):
    mo.md("## Top Performers Analysis")
    return


@app.cell
def __(gw_data, analysis_type, gw_selector, mo):
    def analyze_top_performers(gw_df):
        """Analyze top performers vs model predictions"""
        if len(gw_df) == 0:
            return pd.DataFrame(), pd.DataFrame()
        
        # Get players who actually played
        played_data = gw_df[gw_df['minutes'] > 0].copy()
        
        if len(played_data) == 0:
            return pd.DataFrame(), pd.DataFrame()
        
        # Top actual performers
        top_actual = played_data.nlargest(15, 'total_points')[
            ['name', 'position', 'team', 'total_points', 'xP', 'minutes', 'goals_scored', 'assists']
        ].copy()
        top_actual['prediction_accuracy'] = top_actual['total_points'] - top_actual['xP']
        
        # Top predicted performers (high xP)
        top_predicted = played_data.nlargest(15, 'xP')[
            ['name', 'position', 'team', 'total_points', 'xP', 'minutes', 'goals_scored', 'assists']  
        ].copy()
        top_predicted['prediction_accuracy'] = top_predicted['total_points'] - top_predicted['xP']
        
        return top_actual, top_predicted
    
    if analysis_type.value == "performers" and len(gw_data) > 0:
        top_actual, top_predicted = analyze_top_performers(gw_data)
        
        if len(top_actual) > 0:
            performers_display = mo.vstack([
                mo.md(f"### Top Performers Analysis (GW {gw_selector.value})"),
                
                mo.md("**Highest Point Scorers (Actual):**"),
                mo.ui.table(top_actual.round(2), page_size=15),
                
                mo.md("**Highest xP Players (Predicted):**"),  
                mo.ui.table(top_predicted.round(2), page_size=15),
                
                mo.md("""
                **Key Insights:**
                - *Prediction Accuracy* = Actual Points - xP (positive = under-predicted, negative = over-predicted)
                - Compare top actual performers vs top predicted to identify model blind spots
                - Look for patterns in over/under-predictions by position or team
                """)
            ])
        else:
            performers_display = mo.md("No player performance data available for this gameweek.")
    else:
        performers_display = mo.md("Select 'Top Performers vs Predictions' to see performer analysis.")
    
    performers_display
    return analyze_top_performers, top_actual, top_predicted


@app.cell
def __(mo):
    mo.md("## Position-Based Analysis")
    return


@app.cell
def __(gw_data, analysis_type, gw_selector, pd, mo):
    def analyze_by_position(gw_df):
        """Analyze prediction accuracy by position"""
        if len(gw_df) == 0:
            return pd.DataFrame()
        
        # Filter for players with valid data
        valid_data = gw_df.dropna(subset=['xP', 'total_points']).copy()
        
        if len(valid_data) == 0:
            return pd.DataFrame()
        
        # Calculate prediction metrics by position
        position_analysis = valid_data.groupby('position').agg({
            'name': 'count',
            'xP': 'mean',
            'total_points': 'mean',
            'minutes': 'mean'
        }).round(2)
        
        # Calculate prediction errors by position
        valid_data['prediction_error'] = valid_data['total_points'] - valid_data['xP'] 
        valid_data['absolute_error'] = abs(valid_data['prediction_error'])
        
        error_analysis = valid_data.groupby('position').agg({
            'prediction_error': 'mean',
            'absolute_error': 'mean'
        }).round(2)
        
        # Combine analyses
        position_stats = position_analysis.join(error_analysis)
        position_stats.columns = ['Players', 'Mean xP', 'Mean Actual', 'Mean Minutes', 'Prediction Bias', 'Mean Abs Error']
        
        # Add correlation by position (if enough players)
        correlations = []
        for pos in position_stats.index:
            pos_data = valid_data[valid_data['position'] == pos]
            if len(pos_data) >= 3:  # Need at least 3 players for meaningful correlation
                corr = pos_data['xP'].corr(pos_data['total_points'])
                correlations.append(corr)
            else:
                correlations.append(np.nan)
        
        position_stats['Correlation'] = correlations
        
        return position_stats.round(3)
    
    if analysis_type.value == "positions" and len(gw_data) > 0:
        position_stats = analyze_by_position(gw_data)
        
        if len(position_stats) > 0:
            positions_display = mo.vstack([
                mo.md(f"### Position-Based Analysis (GW {gw_selector.value})"),
                mo.ui.table(position_stats),
                mo.md("""
                **Interpretation:**
                - **Prediction Bias**: Positive = under-predicted, negative = over-predicted
                - **Mean Abs Error**: Average absolute prediction error (lower is better)
                - **Correlation**: How well xP correlates with actual points (higher is better)
                - Look for positions where model consistently over/under-predicts
                """)
            ])
        else:
            positions_display = mo.md("No position data available for analysis.")
    else:
        positions_display = mo.md("Select 'Position-Based Analysis' to see position breakdown.")
    
    positions_display
    return analyze_by_position, position_stats


@app.cell
def __(mo):
    mo.md("## Team Selection Analysis")


@app.cell
def __(gw_data, analysis_type, gw_selector, mo, pd, PredictionStorage):
    def analyze_optimal_team_performance(gw_df, gameweek):
        """Analyze how the optimal team performed vs alternatives"""
        storage = PredictionStorage()
        
        # Try to load saved team selections
        prediction_data = storage.load_gameweek_predictions(gameweek)
        
        if not prediction_data or not prediction_data.get('team_selections'):
            return None, "No saved team selections found for this gameweek"
        
        optimal_team_data = prediction_data['team_selections']['optimal_15']
        
        # Get actual performance for optimal team players
        optimal_player_ids = [p['player_id'] for p in optimal_team_data]
        
        # Filter gameweek data for optimal team players
        optimal_performance = gw_df[gw_df['element'].isin(optimal_player_ids)].copy()
        
        if len(optimal_performance) == 0:
            return None, "Could not match optimal team players with actual results"
        
        # Calculate team metrics
        team_analysis = {
            'total_players_matched': len(optimal_performance),
            'total_actual_points': optimal_performance['total_points'].sum(),
            'total_predicted_xp': sum(p['xP'] for p in optimal_team_data),
            'total_cost': sum(p['price_gbp'] for p in optimal_team_data),
            'average_points_per_player': optimal_performance['total_points'].mean(),
            'players_who_played': (optimal_performance['minutes'] > 0).sum(),
            'players_who_started': (optimal_performance['starts'] == 1).sum(),
        }
        
        # Best starting 11 analysis
        optimal_performance_sorted = optimal_performance.sort_values('total_points', ascending=False)
        
        # Calculate what the best possible starting 11 would have been
        # (This is retroactive analysis - we know the results now)
        position_counts = {'GKP': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}  # Example formation
        best_11_points = 0
        best_11_players = []
        
        for position in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_players = optimal_performance[
                optimal_performance['position'] == position
            ].nlargest(position_counts.get(position, 0), 'total_points')
            
            best_11_points += pos_players['total_points'].sum()
            best_11_players.extend(pos_players['name'].tolist())
        
        team_analysis['best_11_points'] = best_11_points
        team_analysis['best_11_players'] = best_11_players
        
        return optimal_performance, team_analysis
    
    if analysis_type.value == "team_selection" and len(gw_data) > 0:
        optimal_performance, team_analysis = analyze_optimal_team_performance(gw_data, gw_selector.value)
        
        if optimal_performance is not None:
            team_display = mo.vstack([
                mo.md(f"### Optimal Team Performance Analysis (GW {gw_selector.value})"),
                
                mo.md(f"""
                **Team Summary:**
                - **Players Matched:** {team_analysis['total_players_matched']}/15
                - **Total Actual Points:** {team_analysis['total_actual_points']:.0f}
                - **Total Predicted xP:** {team_analysis['total_predicted_xp']:.2f}
                - **Prediction Error:** {team_analysis['total_actual_points'] - team_analysis['total_predicted_xp']:.2f} points
                - **Team Cost:** £{team_analysis['total_cost']:.1f}m
                - **Average Points/Player:** {team_analysis['average_points_per_player']:.2f}
                - **Players Who Played:** {team_analysis['players_who_played']}/15
                - **Players Who Started:** {team_analysis['players_who_started']}/15
                """),
                
                mo.md("**Optimal Team Player Performance:**"),
                mo.ui.table(optimal_performance[
                    ['name', 'position', 'team', 'total_points', 'xP', 'minutes', 'goals_scored', 'assists', 'starts']
                ].round(2), page_size=15),
                
                mo.md(f"""
                **Best Possible Starting 11 (Retrospective):**
                - **Best 11 Points:** {team_analysis['best_11_points']:.0f}
                - **Players:** {', '.join(team_analysis['best_11_players'])}
                """)
            ])
        else:
            team_display = mo.md(f"**Error:** {team_analysis}")
    else:
        team_display = mo.md("Select 'Team Selection Analysis' to see optimal team performance.")
    
    team_display


@app.cell
def __(mo):
    mo.md("## Transfer Risk Validation")


@app.cell
def __(gw_data, analysis_type, gw_selector, mo, pd, PredictionStorage):
    def analyze_transfer_risk_accuracy(gw_df, gameweek):
        """Analyze if transfer risk flags were accurate"""
        storage = PredictionStorage()
        
        # Load predictions to get transfer risk flags
        prediction_data = storage.load_gameweek_predictions(gameweek)
        
        if not prediction_data:
            return None, "No saved predictions found for this gameweek"
        
        predictions_df = prediction_data['predictions']
        
        # Merge with actual results
        merged = predictions_df.merge(
            gw_df,
            left_on='player_id',
            right_on='element',
            how='inner',
            suffixes=('_pred', '_actual')
        )
        
        if len(merged) == 0:
            return None, "Could not match predictions with actual results"
        
        # Analyze transfer risk accuracy
        risk_analysis = merged.groupby('transfer_risk').agg({
            'total_points': ['count', 'mean', 'std'],
            'xP': 'mean',
            'minutes': 'mean'
        }).round(2)
        
        # Players flagged as transfer risk
        high_risk_players = merged[merged['transfer_risk'] == True]
        
        # Check if high-risk players actually performed poorly
        if len(high_risk_players) > 0:
            risk_accuracy = {
                'total_high_risk': len(high_risk_players),
                'high_risk_avg_points': high_risk_players['total_points'].mean(),
                'high_risk_avg_xp': high_risk_players['xP'].mean(),
                'high_risk_underperformed': (high_risk_players['total_points'] < 3).sum(),
                'high_risk_outperformed': (high_risk_players['total_points'] >= 8).sum()
            }
        else:
            risk_accuracy = {'total_high_risk': 0}
        
        return risk_analysis, risk_accuracy
    
    if analysis_type.value == "transfers" and len(gw_data) > 0:
        risk_analysis, risk_accuracy = analyze_transfer_risk_accuracy(gw_data, gw_selector.value)
        
        if risk_analysis is not None:
            transfer_display = mo.vstack([
                mo.md(f"### Transfer Risk Validation (GW {gw_selector.value})"),
                
                mo.md("**Performance by Transfer Risk Status:**"),
                mo.ui.table(risk_analysis),
                
                mo.md(f"""
                **Transfer Risk Accuracy:**
                - **High-Risk Players:** {risk_accuracy.get('total_high_risk', 0)}
                - **Average Points (High-Risk):** {risk_accuracy.get('high_risk_avg_points', 0):.2f}
                - **Average xP (High-Risk):** {risk_accuracy.get('high_risk_avg_xp', 0):.2f}
                - **Underperformed (<3 pts):** {risk_accuracy.get('high_risk_underperformed', 0)} players
                - **Outperformed (8+ pts):** {risk_accuracy.get('high_risk_outperformed', 0)} players
                
                **Insights:**
                - High-risk flags are accurate if flagged players scored fewer points on average
                - Look for systematic patterns in fixture difficulty vs actual performance
                """) if risk_accuracy.get('total_high_risk', 0) > 0 else mo.md("No players were flagged as high transfer risk for this gameweek.")
            ])
        else:
            transfer_display = mo.md(f"**Error:** {risk_accuracy}")
    else:
        transfer_display = mo.md("Select 'Transfer Risk Validation' to see transfer analysis.")
    
    transfer_display


@app.cell
def __(mo):
    mo.md("## Prediction Storage Management")


@app.cell  
def __(PredictionStorage, mo):
    # Show available predictions
    storage = PredictionStorage()
    available_predictions = storage.list_available_predictions()
    
    if len(available_predictions) > 0:
        storage_display = mo.vstack([
            mo.md("### Stored Predictions"),
            mo.ui.table(available_predictions, page_size=10),
            mo.md("""
            **Usage:**
            - Each row represents a saved prediction set
            - Use these timestamps to load specific predictions in analysis
            - File sizes indicate the amount of data stored
            """)
        ])
    else:
        storage_display = mo.md("""
        ### No Stored Predictions Found
        
        **To get started:**
        1. Run your main FPL model (`fpl_xp_model.py`)
        2. Use the "Save Predictions" feature at the bottom
        3. Return here after gameweeks complete to analyze accuracy
        """)
    
    storage_display


@app.cell
def __(mo):
    mo.md("## Future Enhancements")
    return


@app.cell
def __(mo):
    mo.md(f"""
    ### Planned Enhancements
    
    **Prediction Storage System:**
    - Store xP predictions before each gameweek for historical comparison
    - Track model parameter changes and their impact on accuracy
    - Build database of predictions vs actuals over time
    
    **Advanced Analytics:**
    - Transfer decision validation (was "transfer risk" accurate?)
    - Captain choice analysis (did high xP players perform as captains?)
    - Template vs differential player analysis
    - Price change prediction accuracy
    
    **Interactive Features:**
    - Multi-gameweek trend analysis
    - Player-specific prediction tracking
    - Model improvement recommendations
    - Automated weekly retro reports
    
    **Mini-League Insights:**
    - Compare your picks vs league opponents
    - Identify differential opportunities
    - Track relative performance over time
    
    **Integration with Main Model:**
    - Automatically save predictions from main optimization
    - Feed retro insights back into model improvements
    - Parameter tuning based on historical accuracy
    """)
    return


if __name__ == "__main__":
    app.run()