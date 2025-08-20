#!/usr/bin/env python3
"""
Fixture Difficulty Visualization

Creates a heatmap showing the difficulty of the next 5 fixtures for each team.
Uses dynamic team strength ratings to calculate opponent difficulty.
"""

import sys
sys.path.append('/Users/alex/dev/FPL/fpl-team-picker')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from client import get_current_teams, get_fixtures_normalized
from dynamic_team_strength import DynamicTeamStrength, load_historical_gameweek_data

def create_fixture_difficulty_heatmap(start_gameweek=2, num_gameweeks=5):
    """
    Create fixture difficulty heatmap for teams
    
    Args:
        start_gameweek: Starting gameweek for fixture analysis
        num_gameweeks: Number of gameweeks to analyze
    
    Returns:
        matplotlib figure
    """
    print(f"🏟️ Creating fixture difficulty visualization for GW{start_gameweek}-{start_gameweek + num_gameweeks - 1}")
    
    # Load data
    teams = get_current_teams()
    fixtures = get_fixtures_normalized()
    
    # Get dynamic team strength ratings
    calculator = DynamicTeamStrength(debug=False)
    current_season_data = load_historical_gameweek_data(start_gw=1, end_gw=start_gameweek-1)
    team_strength = calculator.get_team_strength(
        target_gameweek=start_gameweek,
        teams_data=teams,
        current_season_data=current_season_data
    )
    
    print(f"✅ Loaded team strength ratings for {len(team_strength)} teams")
    
    # Create team mapping
    team_id_to_name = dict(zip(teams['team_id'], teams['name']))
    
    # Initialize difficulty matrix
    team_names = sorted(team_strength.keys())
    difficulty_matrix = pd.DataFrame(
        index=team_names,
        columns=[f'GW{gw}' for gw in range(start_gameweek, start_gameweek + num_gameweeks)]
    )
    
    # Calculate fixture difficulty for each team and gameweek
    for gw in range(start_gameweek, start_gameweek + num_gameweeks):
        gw_fixtures = fixtures[fixtures['event'] == gw].copy()
        
        if gw_fixtures.empty:
            print(f"⚠️ No fixtures found for GW{gw}")
            continue
        
        for _, fixture in gw_fixtures.iterrows():
            home_team_id = fixture['home_team_id']
            away_team_id = fixture['away_team_id']
            
            home_team_name = team_id_to_name.get(home_team_id)
            away_team_name = team_id_to_name.get(away_team_id)
            
            if home_team_name and away_team_name:
                # Home team difficulty = opponent strength (easier = green, harder = red)
                away_strength = team_strength.get(away_team_name, 1.0)
                home_difficulty = away_strength  # Higher opponent strength = harder fixture
                
                # Away team difficulty = opponent strength + away disadvantage
                home_strength = team_strength.get(home_team_name, 1.0)
                away_difficulty = home_strength * 1.1  # 10% away disadvantage
                
                # Store difficulties (1.0 = average, <1.0 = easier, >1.0 = harder)
                difficulty_matrix.loc[home_team_name, f'GW{gw}'] = home_difficulty
                difficulty_matrix.loc[away_team_name, f'GW{gw}'] = away_difficulty
    
    # Convert to float and fill missing values
    difficulty_matrix = difficulty_matrix.astype(float)
    difficulty_matrix = difficulty_matrix.fillna(1.0)  # Average difficulty for missing fixtures
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    
    # Custom colormap: Green = Easy, Yellow = Average, Red = Hard
    colors = ['#2E8B57', '#90EE90', '#FFFF99', '#FFA500', '#FF4500', '#8B0000']
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list('difficulty', colors, N=256)
    
    # Create heatmap
    sns.heatmap(
        difficulty_matrix,
        annot=True,
        fmt='.2f',
        cmap=custom_cmap,
        center=1.0,  # Center colormap on average difficulty
        vmin=0.7,
        vmax=1.4,
        cbar_kws={'label': 'Fixture Difficulty\n(0.7=Very Easy, 1.0=Average, 1.4=Very Hard)'},
        linewidths=0.5,
        square=False
    )
    
    plt.title(f'Fixture Difficulty Heatmap: GW{start_gameweek}-{start_gameweek + num_gameweeks - 1}\n'
              f'Based on Dynamic Team Strength Ratings', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Gameweek', fontsize=12, fontweight='bold')
    plt.ylabel('Team', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Add interpretation guide
    textstr = """
    🟢 Green: Easy fixtures (weak opponents)
    🟡 Yellow: Average fixtures  
    🔴 Red: Hard fixtures (strong opponents)
    
    💡 Transfer Strategy:
    • Target players from teams with green fixtures
    • Avoid players from teams with red fixtures
    """
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.figtext(0.02, 0.02, textstr, fontsize=9, verticalalignment='bottom', bbox=props)
    
    return plt.gcf(), difficulty_matrix

def analyze_fixture_runs(difficulty_matrix, team_strength):
    """
    Analyze teams with the best/worst fixture runs
    
    Args:
        difficulty_matrix: DataFrame with fixture difficulties
        team_strength: Dict of team strength ratings
    
    Returns:
        Dict with analysis results
    """
    # Calculate average difficulty for each team
    avg_difficulty = difficulty_matrix.mean(axis=1).sort_values()
    
    # Best fixture runs (lowest average difficulty)
    best_fixtures = avg_difficulty.head(5)
    
    # Worst fixture runs (highest average difficulty)
    worst_fixtures = avg_difficulty.tail(5)
    
    # Teams to target (good teams with easy fixtures)
    target_teams = []
    avoid_teams = []
    
    for team in difficulty_matrix.index:
        team_quality = team_strength.get(team, 1.0)
        fixture_difficulty = avg_difficulty[team]
        
        # Good teams with easy fixtures = prime targets
        if team_quality >= 1.1 and fixture_difficulty <= 0.95:
            target_teams.append((team, team_quality, fixture_difficulty))
        
        # Any team with very hard fixtures = avoid
        if fixture_difficulty >= 1.15:
            avoid_teams.append((team, team_quality, fixture_difficulty))
    
    return {
        'best_fixtures': best_fixtures,
        'worst_fixtures': worst_fixtures,
        'target_teams': target_teams,
        'avoid_teams': avoid_teams,
        'avg_difficulty': avg_difficulty
    }

def print_fixture_analysis(analysis):
    """Print fixture analysis results"""
    
    print("\n🏆 FIXTURE ANALYSIS RESULTS")
    print("=" * 50)
    
    print("\n🟢 BEST FIXTURE RUNS (Easiest 5 gameweeks):")
    for i, (team, difficulty) in enumerate(analysis['best_fixtures'].items(), 1):
        print(f"  {i}. {team}: {difficulty:.3f} avg difficulty")
    
    print("\n🔴 WORST FIXTURE RUNS (Hardest 5 gameweeks):")
    for i, (team, difficulty) in enumerate(analysis['worst_fixtures'].items(), 1):
        print(f"  {i}. {team}: {difficulty:.3f} avg difficulty")
    
    if analysis['target_teams']:
        print("\n🎯 PRIME TARGETS (Good teams + Easy fixtures):")
        for team, quality, difficulty in analysis['target_teams']:
            print(f"  • {team}: Quality {quality:.2f}, Fixtures {difficulty:.3f}")
    
    if analysis['avoid_teams']:
        print("\n⚠️ TEAMS TO AVOID (Very hard fixtures):")
        for team, quality, difficulty in analysis['avoid_teams']:
            print(f"  • {team}: Quality {quality:.2f}, Fixtures {difficulty:.3f}")

if __name__ == "__main__":
    # Create visualization
    fig, difficulty_df = create_fixture_difficulty_heatmap(start_gameweek=2, num_gameweeks=5)
    
    # Show the plot
    plt.show()
    
    # Load team strength for analysis
    teams = get_current_teams()
    calculator = DynamicTeamStrength(debug=False)
    current_season_data = load_historical_gameweek_data(start_gw=1, end_gw=1)
    team_strength = calculator.get_team_strength(
        target_gameweek=2,
        teams_data=teams,
        current_season_data=current_season_data
    )
    
    # Analyze fixture runs
    analysis = analyze_fixture_runs(difficulty_df, team_strength)
    print_fixture_analysis(analysis)
    
    # Save the heatmap
    fig.savefig('/Users/alex/dev/FPL/fpl-team-picker/fixture_difficulty_heatmap.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n💾 Heatmap saved as fixture_difficulty_heatmap.png")