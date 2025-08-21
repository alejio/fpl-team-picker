"""
FPL Visualization Module

Handles all visualization functions for FPL gameweek management including:
- Team strength visualizations
- Player trends and performance charts
- Fixture difficulty visualizations
- Interactive charts with plotly
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Tuple, List, Dict, Optional


def create_team_strength_visualization(target_gameweek: int, mo_ref) -> object:
    """
    Create team strength visualization showing current ratings
    
    Args:
        target_gameweek: The gameweek to analyze
        mo_ref: Marimo reference for UI components
        
    Returns:
        Marimo UI component with team strength visualization
    """
    try:
        from client import get_current_teams
        from dynamic_team_strength import DynamicTeamStrength, load_historical_gameweek_data
        
        # Load team data
        teams = get_current_teams()
        
        # Get dynamic team strength ratings for the target gameweek
        calculator = DynamicTeamStrength(debug=False)
        current_season_data = load_historical_gameweek_data(start_gw=1, end_gw=target_gameweek-1)
        team_strength = calculator.get_team_strength(
            target_gameweek=target_gameweek,
            teams_data=teams,
            current_season_data=current_season_data
        )
        
        # Create team strength dataframe
        strength_data = []
        for team_name, strength in team_strength.items():
            # Categorize strength
            if strength >= 1.15:
                category = "🔴 Very Strong"
                difficulty_as_opponent = "Very Hard"
            elif strength >= 1.05:
                category = "🟠 Strong" 
                difficulty_as_opponent = "Hard"
            elif strength >= 0.95:
                category = "🟡 Average"
                difficulty_as_opponent = "Average"
            elif strength >= 0.85:
                category = "🟢 Weak"
                difficulty_as_opponent = "Easy"
            else:
                category = "🟢 Very Weak"
                difficulty_as_opponent = "Very Easy"
            
            strength_data.append({
                'Team': team_name,
                'Strength': round(strength, 3),
                'Category': category,
                'As Opponent': difficulty_as_opponent,
                'Attack Rating': round(strength * 1.0, 3),  # Simplified for display
                'Defense Rating': round(2.0 - strength, 3)  # Inverse relationship
            })
        
        # Sort by strength (strongest first)
        strength_df = pd.DataFrame(strength_data).sort_values('Strength', ascending=False)
        strength_df['Rank'] = range(1, len(strength_df) + 1)
        
        # Reorder columns
        display_df = strength_df[['Rank', 'Team', 'Strength', 'Category', 'As Opponent', 'Attack Rating', 'Defense Rating']]
        
        # Create summary stats
        strongest_team = strength_df.iloc[0]
        weakest_team = strength_df.iloc[-1]
        avg_strength = strength_df['Strength'].mean()
        
        return mo_ref.vstack([
            mo_ref.md(f"### 🏆 Team Strength Ratings - GW{target_gameweek}"),
            mo_ref.md(f"**Strongest:** {strongest_team['Team']} ({strongest_team['Strength']:.3f}) | **Weakest:** {weakest_team['Team']} ({weakest_team['Strength']:.3f}) | **Average:** {avg_strength:.3f}"),
            mo_ref.md("*Higher strength = better team = harder opponent when playing against them*"),
            mo_ref.md("**📊 Dynamic Team Strength Table:**"),
            mo_ref.ui.table(display_df, page_size=20),
            mo_ref.md("""
            **💡 How to Use This:**
            - **🔴 Very Strong teams**: Avoid their opponents (hard fixtures) 
            - **🟢 Weak teams**: Target their opponents (easy fixtures)
            - **Attack Rating**: Expected goals scoring ability
            - **Defense Rating**: Expected goals conceded (lower = better defense)
            """)
        ])
        
    except Exception as e:
        return mo_ref.md(f"❌ **Could not create team strength analysis:** {e}")


def create_player_trends_visualization(players_data: pd.DataFrame) -> Tuple[List[Dict], List[Dict], pd.DataFrame]:
    """
    Create interactive player trends visualization data
    
    Args:
        players_data: DataFrame with player information
        
    Returns:
        Tuple of (player_options, attribute_options, historical_data)
    """
    try:
        # Get historical data for trends
        from client import get_gameweek_live_data
        
        # Load historical gameweek data (check available gameweeks)
        historical_data = []
        max_gw = 10  # Adjust based on season progress
        
        print(f"🔍 Loading historical data for trends...")
        
        for gw in range(1, max_gw + 1):
            try:
                gw_data = get_gameweek_live_data(gw)
                if not gw_data.empty:
                    gw_data['gameweek'] = gw
                    historical_data.append(gw_data)
                    print(f"✅ GW{gw}: {len(gw_data)} player records")
                else:
                    print(f"❌ GW{gw}: No data")
                    break  # Stop if we hit missing data
            except Exception as e:
                print(f"❌ GW{gw}: Error - {e}")
                break
        
        if not historical_data:
            return [], [], pd.DataFrame()
        
        print(f"📊 Total historical datasets: {len(historical_data)}")
        
        # Combine all historical data
        all_data = pd.concat(historical_data, ignore_index=True)
        print(f"📊 Combined data shape: {all_data.shape}")
        
        # CRITICAL: Merge with current player names 
        # The live data only has player_id, we need names from current players
        if not players_data.empty:
            print(f"🔗 Merging with players data to get names...")
            player_names = players_data[['player_id', 'web_name', 'position', 'team', 'name']].drop_duplicates('player_id')
            print(f"📋 Player names data shape: {player_names.shape}")
            
            # Include current price info too (handle missing columns gracefully)
            base_cols = ['player_id', 'web_name', 'position', 'team', 'name']
            optional_cols = ['now_cost', 'selected_by_percent']
            
            # Only include columns that exist
            available_cols = base_cols + [col for col in optional_cols if col in players_data.columns]
            player_info = players_data[available_cols].drop_duplicates('player_id')
            
            print(f"📊 Using player info columns: {available_cols}")
            
            all_data = all_data.merge(
                player_info,
                on='player_id',
                how='left'
            )
            print(f"📊 After merge shape: {all_data.shape}")
            print(f"📊 Players with names: {all_data['web_name'].notna().sum()}")
        else:
            print("⚠️ No players data available for name merge!")
        
        # Calculate derived metrics (handle missing columns gracefully)
        all_data['points_per_game'] = all_data['total_points'] / all_data['gameweek']
        all_data['xg_per_90'] = (all_data.get('expected_goals', 0) / all_data.get('minutes', 1)) * 90
        all_data['xa_per_90'] = (all_data.get('expected_assists', 0) / all_data.get('minutes', 1)) * 90
        
        # Value ratio - only calculate if we have price data
        if 'now_cost' in all_data.columns:
            all_data['value_ratio'] = all_data['total_points'] / (all_data['now_cost'] / 10)  # Points per £1m
        else:
            print("⚠️ No price data available, skipping value ratio calculation")
            all_data['value_ratio'] = 0
        
        # Create player options (filter to players with data)
        player_options = []
        
        # Get players with the most total points and data availability
        valid_players = all_data[all_data['web_name'].notna()]
        print(f"📊 Valid players with names: {len(valid_players)}")
        
        if valid_players.empty:
            print("❌ No players with valid names found!")
            return [], [], pd.DataFrame()
            
        player_stats = valid_players.groupby('player_id').agg({
            'total_points': 'max',
            'web_name': 'first',
            'position': 'first', 
            'name': 'first',
            'gameweek': 'count'  # Number of gameweeks with data
        }).reset_index()
        
        # Filter to players with at least some data and sort by points
        active_players = player_stats[player_stats['gameweek'] >= 1].nlargest(100, 'total_points')
        
        for _, player_row in active_players.iterrows():
            label = f"{player_row['web_name']} ({player_row['position']}, {player_row['name']}) - {player_row['total_points']} pts"
            player_options.append({"label": label, "value": int(player_row['player_id'])})
        
        print(f"📊 Created {len(player_options)} player options for trends")
        
        # Create attribute options (only include available columns)
        base_attributes = [
            {"label": "Total Points", "value": "total_points"},
            {"label": "Points Per Game", "value": "points_per_game"},
            {"label": "Minutes Played", "value": "minutes"},
            {"label": "Goals Scored", "value": "goals_scored"},
            {"label": "Assists", "value": "assists"},
            {"label": "Expected Goals (xG)", "value": "expected_goals"},
            {"label": "Expected Assists (xA)", "value": "expected_assists"},
            {"label": "xG per 90", "value": "xg_per_90"},
            {"label": "xA per 90", "value": "xa_per_90"},
            {"label": "Clean Sheets", "value": "clean_sheets"},
            {"label": "Yellow Cards", "value": "yellow_cards"},
            {"label": "Red Cards", "value": "red_cards"},
            {"label": "Bonus Points", "value": "bonus"},
            {"label": "ICT Index", "value": "ict_index"}
        ]
        
        # Add optional attributes if they exist in the data
        optional_attributes = [
            {"label": "Price (£m)", "value": "now_cost"},
            {"label": "Selected By %", "value": "selected_by_percent"},
            {"label": "Value Ratio (pts/£1m)", "value": "value_ratio"}
        ]
        
        attribute_options = base_attributes.copy()
        for attr in optional_attributes:
            if attr["value"] in all_data.columns:
                attribute_options.append(attr)
                
        print(f"📊 Available attributes: {[attr['value'] for attr in attribute_options]}")
        
        return player_options, attribute_options, all_data
        
    except Exception as e:
        print(f"Error creating trends data: {e}")
        return [], [], pd.DataFrame()


def create_trends_chart(data: pd.DataFrame, selected_player: int, selected_attr: str, multi_players: Optional[List[int]] = None, mo_ref=None) -> object:
    """
    Create interactive trends chart
    
    Args:
        data: Historical data DataFrame
        selected_player: Selected player ID
        selected_attr: Selected attribute to chart
        multi_players: Optional list of additional player IDs to compare
        mo_ref: Marimo reference for UI components
        
    Returns:
        Marimo UI component with trends chart
    """
    if data.empty:
        return mo_ref.md("⚠️ **No data available for trends**") if mo_ref else None
    
    # Validate inputs first
    if not selected_player or not selected_attr:
        return mo_ref.md("⚠️ **Select a player and attribute to see trends**") if mo_ref else None
    
    try:
        # Extract values if they're still dict objects
        if isinstance(selected_player, dict):
            player_id = selected_player.get('value')
        else:
            player_id = selected_player
            
        if isinstance(selected_attr, dict):
            attr_name = selected_attr.get('value')
        else:
            attr_name = selected_attr
            
        # Validate we have valid values
        if not player_id or not attr_name:
            return mo_ref.md("⚠️ **Select a player and attribute to see trends**") if mo_ref else None
        
        # Convert player_id to int if needed
        if isinstance(player_id, str):
            try:
                player_id = int(player_id)
            except:
                print(f"❌ Could not convert player_id to int: {player_id}")
                return mo_ref.md("❌ **Invalid player selection**") if mo_ref else None
        
        # Filter data for selected player
        player_data = data[data['player_id'] == player_id].copy()
        
        if player_data.empty:
            return mo_ref.md(f"❌ **No data found for player ID {player_id}**") if mo_ref else None
            
        # Check if attribute exists
        if attr_name not in player_data.columns:
            return mo_ref.md(f"❌ **Attribute '{attr_name}' not available**") if mo_ref else None
        
        # Create a simple chart with error handling
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Get player name
        player_name = player_data['web_name'].iloc[0] if 'web_name' in player_data.columns else f"Player {player_id}"
        
        # Sort by gameweek and clean data
        player_data = player_data.sort_values('gameweek')
        
        # Add trace with error handling
        try:
            fig.add_trace(go.Scatter(
                x=player_data['gameweek'],
                y=player_data[attr_name],
                mode='lines+markers',
                name=player_name,
                line=dict(width=3),
                marker=dict(size=8)
            ))
        except Exception as trace_error:
            print(f"❌ Error adding trace: {trace_error}")
            return mo_ref.md(f"❌ **Error plotting data for {player_name}**") if mo_ref else None
        
        # Simple layout
        attr_label = attr_name.replace('_', ' ').title()
        fig.update_layout(
            title=f"{player_name}: {attr_label} Over Time",
            xaxis_title="Gameweek",
            yaxis_title=attr_label,
            height=400,
            showlegend=True
        )
        
        # Return using marimo's plotly support
        return mo_ref.as_html(fig) if mo_ref else fig
            
    except Exception as e:
        print(f"❌ Error creating trends chart: {e}")
        import traceback
        traceback.print_exc()
        return mo_ref.md(f"❌ **Chart error:** {str(e)}") if mo_ref else None


def create_fixture_difficulty_visualization(start_gw: int, num_gws: int = 5, mo_ref=None) -> object:
    """
    Create fixture difficulty heatmap visualization
    
    Args:
        start_gw: Starting gameweek
        num_gws: Number of gameweeks to show
        mo_ref: Marimo reference for UI components
        
    Returns:
        Marimo UI component with fixture difficulty visualization
    """
    try:
        from client import get_fixtures_normalized, get_current_teams
        from dynamic_team_strength import DynamicTeamStrength, load_historical_gameweek_data
        
        # Load required data
        fixtures = get_fixtures_normalized()
        teams = get_current_teams()
        
        # Get team strengths
        calculator = DynamicTeamStrength(debug=False)
        current_season_data = load_historical_gameweek_data(start_gw=1, end_gw=start_gw-1)
        team_strength = calculator.get_team_strength(
            target_gameweek=start_gw,
            teams_data=teams,
            current_season_data=current_season_data
        )
        
        # Filter fixtures for relevant gameweeks
        relevant_fixtures = fixtures[
            (fixtures['event'] >= start_gw) & 
            (fixtures['event'] < start_gw + num_gws)
        ].copy()
        
        if relevant_fixtures.empty:
            return mo_ref.md(f"❌ **No fixtures found for GW{start_gw}-{start_gw + num_gws - 1}**") if mo_ref else None
        
        # Create team fixture matrix
        team_names = sorted(teams['name'].unique())
        gameweeks = list(range(start_gw, start_gw + num_gws))
        
        # Initialize matrix with empty values
        fixture_matrix = []
        
        for team_name in team_names:
            team_row = {'Team': team_name}
            team_id = teams[teams['name'] == team_name]['team_id'].iloc[0]
            
            for gw in gameweeks:
                # Find fixtures for this team in this gameweek
                team_fixtures = relevant_fixtures[
                    ((relevant_fixtures['team_h'] == team_id) | 
                     (relevant_fixtures['team_a'] == team_id)) & 
                    (relevant_fixtures['event'] == gw)
                ]
                
                if not team_fixtures.empty:
                    fixture = team_fixtures.iloc[0]
                    
                    # Determine opponent and home/away
                    if fixture['team_h'] == team_id:
                        # Team is at home
                        opponent_id = fixture['team_a']
                        is_home = True
                    else:
                        # Team is away
                        opponent_id = fixture['team_h']
                        is_home = False
                    
                    # Get opponent name
                    opponent_name = teams[teams['team_id'] == opponent_id]['name'].iloc[0]
                    
                    # Get opponent strength for difficulty calculation
                    opponent_strength = team_strength.get(opponent_name, 1.0)
                    
                    # Calculate difficulty (higher opponent strength = harder fixture)
                    difficulty = opponent_strength
                    
                    # Adjust for home/away
                    if is_home:
                        difficulty *= 0.9  # Easier at home
                        venue = "(H)"
                    else:
                        difficulty *= 1.1  # Harder away
                        venue = "(A)"
                    
                    # Create display text
                    short_name = opponent_name[:3].upper()
                    team_row[f'GW{gw}'] = f"{short_name} {venue}"
                    team_row[f'GW{gw}_difficulty'] = difficulty
                else:
                    team_row[f'GW{gw}'] = "BGW"  # Blank gameweek
                    team_row[f'GW{gw}_difficulty'] = 0
            
            fixture_matrix.append(team_row)
        
        # Convert to DataFrame
        fixture_df = pd.DataFrame(fixture_matrix)
        
        # Create display DataFrame with difficulty colors
        display_columns = ['Team'] + [f'GW{gw}' for gw in gameweeks]
        display_df = fixture_df[display_columns].copy()
        
        # Add difficulty summary
        difficulty_cols = [f'GW{gw}_difficulty' for gw in gameweeks]
        fixture_df['Avg_Difficulty'] = fixture_df[difficulty_cols].mean(axis=1)
        
        # Sort by average difficulty (easiest first)
        fixture_df = fixture_df.sort_values('Avg_Difficulty')
        display_df = display_df.reindex(fixture_df.index)
        
        # Add summary column
        display_df['Avg Difficulty'] = fixture_df['Avg_Difficulty'].round(3)
        
        return mo_ref.vstack([
            mo_ref.md(f"### 📅 Fixture Difficulty Matrix - GW{start_gw} to GW{start_gw + num_gws - 1}"),
            mo_ref.md("*Teams sorted by average fixture difficulty (easiest first)*"),
            mo_ref.md("**Legend:** (H) = Home, (A) = Away, BGW = Blank Gameweek"),
            mo_ref.ui.table(display_df, page_size=20),
            mo_ref.md("""
            **💡 How to Use:**
            - **Lower difficulty** = easier fixtures = better for FPL assets
            - **Higher difficulty** = harder fixtures = avoid these teams' players
            - Consider home/away advantage when evaluating fixtures
            """)
        ]) if mo_ref else display_df
        
    except Exception as e:
        return mo_ref.md(f"❌ **Could not create fixture difficulty analysis:** {e}") if mo_ref else None


def _get_attribute_options() -> List[Dict]:
    """Helper function to get attribute options for labeling"""
    return [
        {"label": "Total Points", "value": "total_points"},
        {"label": "Points Per Game", "value": "points_per_game"},
        {"label": "Minutes Played", "value": "minutes"},
        {"label": "Goals Scored", "value": "goals_scored"},
        {"label": "Assists", "value": "assists"},
        {"label": "Expected Goals (xG)", "value": "expected_goals"},
        {"label": "Expected Assists (xA)", "value": "expected_assists"},
        {"label": "xG per 90", "value": "xg_per_90"},
        {"label": "xA per 90", "value": "xa_per_90"},
        {"label": "Clean Sheets", "value": "clean_sheets"},
        {"label": "Yellow Cards", "value": "yellow_cards"},
        {"label": "Red Cards", "value": "red_cards"},
        {"label": "Bonus Points", "value": "bonus"},
        {"label": "ICT Index", "value": "ict_index"},
        {"label": "Price (£m)", "value": "now_cost"},
        {"label": "Selected By %", "value": "selected_by_percent"},
        {"label": "Value Ratio (pts/£1m)", "value": "value_ratio"}
    ]