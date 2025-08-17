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
        # FPL 2025-26 Season Retro Analysis

        **Current season performance evaluation and model validation**
        
        Analyze how your predictions performed against actual 2025-26 season results. 
        Currently analyzing gameweek 1 data with live FPL performance metrics.
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
        """Load datasets for 2025-26 season retrospective analysis"""
        print("Loading 2025-26 season datasets for retro analysis...")
        
        # Load current season live gameweek data (2025-26)
        try:
            gameweek_data = pd.read_csv(DATA_DIR / "fpl_live_gameweek_1.csv")
            # Add GW column for consistency
            gameweek_data['GW'] = gameweek_data['event']
            print(f"Loaded current season GW1 data: {len(gameweek_data)} player records")
        except Exception as e:
            print(f"Error loading live gameweek data: {e}")
            gameweek_data = pd.DataFrame()
        
        # Current players for mapping names and positions
        players_current = pd.read_csv(DATA_DIR / "fpl_players_current.csv")
        
        # Teams for reference
        teams = pd.read_csv(DATA_DIR / "fpl_teams_current.csv")
        
        # xG/xA rates for model validation
        xg_rates = pd.read_csv(DATA_DIR / "fpl_player_xg_xa_rates.csv")
        
        # Load player deltas for recent performance trends
        try:
            player_deltas = pd.read_csv(DATA_DIR / "fpl_player_deltas_current.csv")
            print(f"Loaded player deltas: {len(player_deltas)} records")
        except Exception as e:
            print(f"Player deltas not available: {e}")
            player_deltas = pd.DataFrame()
        
        print(f"Ready for retro analysis: {len(gameweek_data)} GW records, {len(players_current)} current players")
        
        return gameweek_data, players_current, teams, xg_rates, player_deltas
    
    gameweek_data, players_current, teams, xg_rates, player_deltas = load_retro_datasets()
    return gameweek_data, load_retro_datasets, players_current, teams, xg_rates, player_deltas


@app.cell
def __(mo):
    mo.md("### Available Gameweeks")
    return


@app.cell
def __(gameweek_data, players_current, mo, pd):
    # Check available gameweeks in current season data
    if 'GW' in gameweek_data.columns:
        available_gws = sorted([int(gw) for gw in gameweek_data['GW'].unique() if pd.notna(gw)])
    else:
        # Fallback if no GW column
        available_gws = [1]
    print(f"Available gameweeks in 2025-26 season: {available_gws}")
    
    # Create full player performance report with names
    if len(gameweek_data) > 0:
        # Merge with player names and details
        performance_report = gameweek_data.merge(
            players_current[['player_id', 'web_name', 'position', 'team_id', 'price_gbp']], 
            on='player_id', 
            how='left'
        )
        
        # Calculate season totals (cumulative across all gameweeks)
        season_totals = performance_report.groupby(['player_id', 'web_name', 'position', 'price_gbp']).agg({
            'total_points': 'sum',
            'minutes': 'sum', 
            'goals_scored': 'sum',
            'assists': 'sum',
            'expected_goals': 'sum',
            'expected_assists': 'sum',
            'event': 'count'  # Number of gameweeks played
        }).reset_index()
        
        # Rename event count to games_played
        season_totals = season_totals.rename(columns={'event': 'games_played'})
        
        # Select key columns for the report
        report_cols = ['web_name', 'position', 'total_points', 'games_played', 'minutes', 'goals_scored', 'assists', 
                      'expected_goals', 'expected_assists', 'price_gbp']
        available_report_cols = [col for col in report_cols if col in season_totals.columns]
        
        # Sort by total points descending
        if 'total_points' in season_totals.columns:
            full_report = season_totals[available_report_cols].sort_values('total_points', ascending=False)
        else:
            full_report = season_totals[available_report_cols]
        
        # Round numerical columns
        numeric_cols = ['total_points', 'minutes', 'goals_scored', 'assists', 'expected_goals', 'expected_assists', 'price_gbp']
        for col in numeric_cols:
            if col in full_report.columns:
                full_report[col] = full_report[col].round(2)
    else:
        full_report = pd.DataFrame()
    
    # Check for data quality issues
    data_quality_warning = ""
    if len(full_report) > 0:
        total_points_sum = full_report['total_points'].sum() if 'total_points' in full_report.columns else 0
        players_with_points = (full_report['total_points'] > 0).sum() if 'total_points' in full_report.columns else 0
        
        # If very few players have points or total is suspiciously low, show warning
        if total_points_sum < 1000 or players_with_points < 100:
            data_quality_warning = mo.md("""
            ⚠️ **Data Quality Warning**: The performance data appears incomplete or outdated. 
            Some players may show 0 points despite actual performance (e.g., known goalscorers showing 0 goals).
            This suggests the live data needs to be refreshed with post-match results.
            """)
    
    display_components = [
        mo.md(f"**Available Gameweeks:** {', '.join(map(str, available_gws))}"),
        mo.md("**2025-26 Season Running Totals Report:**"),
        mo.md(f"*All {len(full_report)} players with cumulative season performance - sorted by total points*")
    ]
    
    if data_quality_warning:
        display_components.insert(1, data_quality_warning)
    
    display_components.append(mo.ui.table(full_report, page_size=25))
    
    mo.vstack(display_components)
    return available_gws, full_report


@app.cell
def __(mo):
    mo.md("## Gameweek Selection & Analysis")
    return


@app.cell
def __(available_gws, mo):
    # Create gameweek selector (use simple format)
    gw_options = {f"GW {gw}": int(gw) for gw in available_gws}
    gw_selector = mo.ui.dropdown(
        options=gw_options,
        label="Select Gameweek for Analysis"
    )
    
    # Analysis type selector
    analysis_options = {
        "Prediction Accuracy Overview": "accuracy",
        "Model Component Validation": "components", 
        "Top Performers vs Predictions": "performers",
        "Position-Based Analysis": "positions",
        "Team Selection Analysis": "team_selection",
        "Transfer Risk Validation": "transfers"
    }
    analysis_type = mo.ui.dropdown(
        options=analysis_options,
        label="Analysis Type"
    )
    
    mo.vstack([
        mo.md("**Select Analysis Parameters:**"),
        gw_selector,
        analysis_type
    ])
    return analysis_type, gw_selector


@app.cell
def __(gw_selector, gameweek_data, pd):
    # Filter current season data for selected gameweek
    if gw_selector.value is not None and 'GW' in gameweek_data.columns:
        gw_data = gameweek_data[gameweek_data['GW'] == gw_selector.value].copy()
        print(f"Analyzing 2025-26 GW {gw_selector.value}: {len(gw_data)} player records")
    elif gw_selector.value is not None:
        # Use all data if no GW column
        gw_data = gameweek_data.copy()
        print(f"Using all available 2025-26 data: {len(gw_data)} player records")
    else:
        gw_data = pd.DataFrame()
        print("No gameweek selected")
    
    return (gw_data,)


@app.cell
def __(mo):
    mo.md("## GW1 2025-26 Performance Analysis")
    return


@app.cell
def __(gw_data, analysis_type, players_current, pd, np, plt, sns, mo):
    def analyze_current_season_performance(gw_df):
        """Analyze 2025-26 season gameweek 1 performance"""
        if len(gw_df) == 0:
            return pd.DataFrame(), {}
        
        # For current season analysis, focus on actual performance metrics
        # Include player_id for merging with names later
        performance_cols = ['player_id', 'total_points', 'minutes', 'goals_scored', 'assists', 'expected_goals', 'expected_assists']
        available_cols = [col for col in performance_cols if col in gw_df.columns]
        
        if len(available_cols) == 0:
            return pd.DataFrame(), {"error": "No performance data available"}
        
        valid_data = gw_df[available_cols].copy()
        
        # Summary statistics for current season performance
        metrics = {
            'total_players': len(valid_data),
            'points_scored': valid_data['total_points'].sum() if 'total_points' in valid_data.columns else 0,
            'average_points': valid_data['total_points'].mean() if 'total_points' in valid_data.columns else 0,
            'players_with_points': (valid_data['total_points'] > 0).sum() if 'total_points' in valid_data.columns else 0,
            'total_goals': valid_data['goals_scored'].sum() if 'goals_scored' in valid_data.columns else 0,
            'total_assists': valid_data['assists'].sum() if 'assists' in valid_data.columns else 0,
            'minutes_played': valid_data['minutes'].sum() if 'minutes' in valid_data.columns else 0
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
        performance_data, performance_metrics = analyze_current_season_performance(gw_data)
        
        if 'error' not in performance_metrics:
            # Show metrics
            metrics_display = mo.md(f"""
            ### GW1 2025-26 Performance Metrics
            
            - **Total Players:** {performance_metrics['total_players']}
            - **Points Scored:** {performance_metrics['points_scored']} total points
            - **Average Points:** {performance_metrics['average_points']:.2f} points per player
            - **Players with Points:** {performance_metrics['players_with_points']} ({performance_metrics['players_with_points']/performance_metrics['total_players']*100:.1f}%)
            - **Total Goals:** {performance_metrics['total_goals']} goals
            - **Total Assists:** {performance_metrics['total_assists']} assists
            - **Minutes Played:** {performance_metrics['minutes_played']} total minutes
            """)
            
            # Show top performers with names
            if 'total_points' in performance_data.columns:
                # Merge with player names for top performers display
                top_performers_with_names = performance_data.merge(
                    players_current[['player_id', 'web_name', 'position']], 
                    on='player_id', 
                    how='left'
                ).nlargest(15, 'total_points')
                
                display_cols = [col for col in ['web_name', 'position', 'total_points', 'minutes', 'goals_scored', 'assists'] if col in top_performers_with_names.columns]
                
                performance_display = mo.vstack([
                    metrics_display,
                    mo.md("### Top Performers GW1"),
                    mo.ui.table(top_performers_with_names[display_cols], page_size=15)
                ])
            else:
                performance_display = metrics_display
        else:
            performance_display = mo.md(f"**Error:** {performance_metrics['error']}")
    else:
        performance_display = mo.md("Select 'Prediction Accuracy Overview' to see GW1 performance analysis.")
        performance_data, performance_metrics = pd.DataFrame(), {}
    
    performance_display
    return performance_data, performance_metrics, analyze_current_season_performance


@app.cell
def __(mo):
    mo.md("## Model Component Validation")
    return


@app.cell
def __(gw_data, analysis_type, gw_selector, players_current, pd, np, plt, mo):
    def analyze_model_components(gw_df):
        """Analyze individual model components vs actual results"""
        if len(gw_df) == 0:
            return {}, {}
        
        # Merge with player data to get position information
        merged_data = gw_df.merge(
            players_current[['player_id', 'web_name', 'position']], 
            on='player_id', 
            how='left'
        )
        
        # Filter for players who actually played
        played_data = merged_data[merged_data['minutes'] > 0].copy()
        
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
            'total_points': 'mean',
            'web_name': 'count',
            'minutes': 'mean'
        }).round(2)
        performance_analysis.columns = ['Mean Points', 'Players', 'Mean Minutes']
        
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
def __(gw_data, analysis_type, gw_selector, players_current, mo):
    def analyze_top_performers(gw_df):
        """Analyze top performers vs model predictions"""
        if len(gw_df) == 0:
            return pd.DataFrame(), pd.DataFrame()
        
        # Merge with player data to get names and positions
        merged_data = gw_df.merge(
            players_current[['player_id', 'web_name', 'position']], 
            on='player_id', 
            how='left'
        )
        
        # Get players who actually played
        played_data = merged_data[merged_data['minutes'] > 0].copy()
        
        if len(played_data) == 0:
            return pd.DataFrame(), pd.DataFrame()
        
        # Top actual performers  
        top_actual = played_data.nlargest(15, 'total_points')[
            ['web_name', 'position', 'total_points', 'minutes', 'goals_scored', 'assists']
        ].copy()
        
        # For current season analysis without stored predictions, show top performers by different metrics
        top_by_goals = played_data.nlargest(10, 'goals_scored')[
            ['web_name', 'position', 'total_points', 'goals_scored', 'minutes']
        ].copy()
        
        return top_actual, top_by_goals
    
    if analysis_type.value == "performers" and len(gw_data) > 0:
        top_actual, top_by_goals = analyze_top_performers(gw_data)
        
        if len(top_actual) > 0:
            performers_display = mo.vstack([
                mo.md(f"### Top Performers Analysis (GW {gw_selector.value})"),
                
                mo.md("**Highest Point Scorers (GW1):**"),
                mo.ui.table(top_actual.round(2), page_size=15),
                
                mo.md("**Top Goal Scorers:**"),  
                mo.ui.table(top_by_goals.round(2), page_size=10),
                
                mo.md("""
                **Key Insights:**
                - Compare top point scorers vs top goal scorers to identify value picks
                - Look for players with high minutes but low points (potential future targets)
                - Analyze position distribution of top performers for formation insights
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
def __(gw_data, analysis_type, gw_selector, players_current, pd, mo):
    def analyze_by_position(gw_df):
        """Analyze prediction accuracy by position"""
        if len(gw_df) == 0:
            return pd.DataFrame()
        
        # Merge with player data to get position information
        merged_data = gw_df.merge(
            players_current[['player_id', 'web_name', 'position']], 
            on='player_id', 
            how='left'
        )
        
        # Filter for players with valid data
        valid_data = merged_data.dropna(subset=['total_points']).copy()
        
        if len(valid_data) == 0:
            return pd.DataFrame()
        
        # Calculate prediction metrics by position
        position_analysis = valid_data.groupby('position').agg({
            'web_name': 'count',
            'total_points': ['mean', 'sum'],
            'minutes': 'mean',
            'goals_scored': 'sum',
            'assists': 'sum'
        }).round(2)
        
        # Flatten multi-level column names from aggregation
        position_analysis.columns = ['Players', 'Avg Points', 'Total Points', 'Avg Minutes', 'Total Goals', 'Total Assists']
        position_stats = position_analysis
        
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
def __(gw_data, analysis_type, gw_selector, players_current, mo, pd, PredictionStorage):
    def analyze_optimal_team_performance(gw_df, gameweek):
        """Analyze how the optimal team performed vs alternatives"""
        storage = PredictionStorage()
        
        # Try to load saved team selections
        prediction_data = storage.load_gameweek_predictions(gameweek)
        
        if not prediction_data or not prediction_data.get('team_selections'):
            return None, "No saved team selections found for this gameweek"
        
        optimal_team_data = prediction_data['team_selections']['optimal_15']
        
        # Merge gameweek data with player information
        merged_data = gw_df.merge(
            players_current[['player_id', 'web_name', 'position']], 
            on='player_id', 
            how='left'
        )
        
        # Get actual performance for optimal team players
        optimal_player_ids = [p['player_id'] for p in optimal_team_data]
        
        # Filter merged data for optimal team players  
        optimal_performance = merged_data[merged_data['player_id'].isin(optimal_player_ids)].copy()
        
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
            best_11_players.extend(pos_players['web_name'].tolist())
        
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
            on='player_id',
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