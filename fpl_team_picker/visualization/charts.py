"""
FPL Visualization Module

Handles all visualization functions for FPL gameweek management including:
- Team strength visualizations
- Player trends and performance charts
- Fixture difficulty visualizations
- Interactive charts with plotly
"""

import pandas as pd
from typing import Tuple, List, Dict, Optional
from ..utils.helpers import get_safe_columns, create_display_dataframe
from fpl_team_picker.config import config


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
        from client import FPLDataClient

        client = FPLDataClient()
        from ..core.team_strength import (
            DynamicTeamStrength,
            load_historical_gameweek_data,
        )

        # Load team data
        teams = client.get_current_teams()

        # Get dynamic team strength ratings for the target gameweek
        calculator = DynamicTeamStrength()  # Uses config default
        current_season_data = load_historical_gameweek_data(
            start_gw=1, end_gw=target_gameweek - 1
        )
        team_strength = calculator.get_team_strength(
            target_gameweek=target_gameweek,
            teams_data=teams,
            current_season_data=current_season_data,
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

            strength_data.append(
                {
                    "Team": team_name,
                    "Strength": round(strength, 3),
                    "Category": category,
                    "As Opponent": difficulty_as_opponent,
                    "Attack Rating": round(strength * 1.0, 3),  # Simplified for display
                    "Defense Rating": round(2.0 - strength, 3),  # Inverse relationship
                }
            )

        # Sort by strength (strongest first)
        strength_df = pd.DataFrame(strength_data).sort_values(
            "Strength", ascending=False
        )
        strength_df["Rank"] = range(1, len(strength_df) + 1)

        # Reorder columns
        display_df = strength_df[
            [
                "Rank",
                "Team",
                "Strength",
                "Category",
                "As Opponent",
                "Attack Rating",
                "Defense Rating",
            ]
        ]

        # Create summary stats
        strongest_team = strength_df.iloc[0]
        weakest_team = strength_df.iloc[-1]
        avg_strength = strength_df["Strength"].mean()

        return mo_ref.vstack(
            [
                mo_ref.md(f"### 🏆 Team Strength Ratings - GW{target_gameweek}"),
                mo_ref.md(
                    f"**Strongest:** {strongest_team['Team']} ({strongest_team['Strength']:.3f}) | **Weakest:** {weakest_team['Team']} ({weakest_team['Strength']:.3f}) | **Average:** {avg_strength:.3f}"
                ),
                mo_ref.md(
                    "*Higher strength = better team = harder opponent when playing against them*"
                ),
                mo_ref.md("**📊 Dynamic Team Strength Table:**"),
                mo_ref.ui.table(
                    display_df, page_size=config.visualization.large_table_page_size
                ),
                mo_ref.md("""
            **💡 How to Use This:**
            - **🔴 Very Strong teams**: Avoid their opponents (hard fixtures)
            - **🟢 Weak teams**: Target their opponents (easy fixtures)
            - **Attack Rating**: Expected goals scoring ability
            - **Defense Rating**: Expected goals conceded (lower = better defense)
            """),
            ]
        )

    except Exception as e:
        return mo_ref.md(f"❌ **Could not create team strength analysis:** {e}")


def create_player_trends_visualization(
    players_data: pd.DataFrame,
) -> Tuple[List[Dict], List[Dict], pd.DataFrame]:
    """
    Create interactive player trends visualization data

    Args:
        players_data: DataFrame with player information

    Returns:
        Tuple of (player_options, attribute_options, historical_data)
    """
    try:
        # Get historical data for trends
        from client import FPLDataClient

        client = FPLDataClient()

        # Load historical gameweek data (dynamically detect available gameweeks)
        historical_data = []
        print("🔍 Loading historical data for trends...")

        # Start from GW1 and continue until we find no data
        gw = 1
        consecutive_failures = 0
        max_failures = 2  # Allow up to 2 consecutive failures before stopping

        while (
            consecutive_failures < max_failures and gw <= 38
        ):  # Max 38 gameweeks in a season
            try:
                gw_data = client.get_gameweek_performance(gameweek=gw)
                if not gw_data.empty:
                    historical_data.append(gw_data)
                    print(f"✅ GW{gw}: {len(gw_data)} player records")
                    consecutive_failures = 0  # Reset failure counter on success
                else:
                    print(f"❌ GW{gw}: No data available")
                    consecutive_failures += 1
            except Exception as e:
                print(f"❌ GW{gw}: Error loading data - {str(e)[:100]}")
                consecutive_failures += 1

            gw += 1

        if not historical_data:
            return [], [], pd.DataFrame()

        print(f"📊 Total historical datasets: {len(historical_data)}")

        # Combine all historical data
        all_data = pd.concat(historical_data, ignore_index=True)
        print(f"📊 Combined data shape: {all_data.shape}")

        # CRITICAL: Merge with current player names
        # The performance data only has player_id, we need names from current players
        if not players_data.empty:
            print("🔗 Merging with provided players data to get names...")
            print(f"📊 Available columns in players_data: {list(players_data.columns)}")

            # Include current price info too (handle missing columns gracefully)
            essential_cols = ["player_id", "web_name", "position"]
            team_name_cols = [
                "name",
                "team_name",
                "team",
            ]  # Different possible team name columns
            price_cols = [
                "price",
                "now_cost",
                "price_gbp",
            ]  # Different possible price columns
            optional_cols = ["selected_by_percent"]

            # Start with essential columns
            available_cols = [
                col for col in essential_cols if col in players_data.columns
            ]

            # Add team name column if available (try different names)
            for team_col in team_name_cols:
                if team_col in players_data.columns:
                    available_cols.append(team_col)
                    break

            # Add price column if available (try different names)
            for price_col in price_cols:
                if price_col in players_data.columns:
                    available_cols.append(price_col)
                    break

            # Add optional columns that exist
            available_cols.extend(
                [col for col in optional_cols if col in players_data.columns]
            )

            player_info = players_data[available_cols].drop_duplicates("player_id")

            print(f"📊 Using player info columns: {available_cols}")

            all_data = all_data.merge(player_info, on="player_id", how="left")
            print(f"📊 After merge shape: {all_data.shape}")
            print(f"📊 Players with names: {all_data['web_name'].notna().sum()}")
        else:
            # Load current players to get names
            print("⚠️ No players data provided, loading current players for names...")
            try:
                current_players = client.get_current_players()
                if not current_players.empty:
                    # Create full name from first and second name if available
                    if (
                        "first" in current_players.columns
                        and "second" in current_players.columns
                    ):
                        current_players["name"] = (
                            current_players["first"] + " " + current_players["second"]
                        )
                    elif "first" in current_players.columns:
                        current_players["name"] = current_players["first"]
                    else:
                        current_players["name"] = current_players["web_name"]

                    # Map column names to standard format
                    column_map = {
                        "price_gbp": "now_cost",
                        "selected_by_percentage": "selected_by_percent",
                        "team_id": "team",
                    }
                    current_players = current_players.rename(columns=column_map)

                    base_cols = ["player_id", "web_name", "position", "name"]
                    optional_cols = ["team", "now_cost", "selected_by_percent"]

                    # Only include columns that exist
                    available_cols = base_cols + [
                        col for col in optional_cols if col in current_players.columns
                    ]
                    player_info = current_players[available_cols].drop_duplicates(
                        "player_id"
                    )

                    all_data = all_data.merge(player_info, on="player_id", how="left")
                    print(f"📊 Loaded {len(player_info)} current players for names")
                    print(
                        f"📊 Players with names after merge: {all_data['web_name'].notna().sum()}"
                    )
                else:
                    print("❌ Could not load current players!")
                    return [], [], pd.DataFrame()
            except Exception as e:
                print(f"❌ Error loading current players: {e}")
                return [], [], pd.DataFrame()

        # Calculate derived metrics (handle missing columns gracefully)
        # Ensure numeric types for calculations
        numeric_cols = ["total_points", "gameweek", "minutes"]
        for col in numeric_cols:
            if col in all_data.columns:
                all_data[col] = pd.to_numeric(all_data[col], errors="coerce")

        # Handle expected goals/assists if they exist
        if "expected_goals" in all_data.columns:
            all_data["expected_goals"] = pd.to_numeric(
                all_data["expected_goals"], errors="coerce"
            )
        else:
            all_data["expected_goals"] = 0

        if "expected_assists" in all_data.columns:
            all_data["expected_assists"] = pd.to_numeric(
                all_data["expected_assists"], errors="coerce"
            )
        else:
            all_data["expected_assists"] = 0

        # Safe calculations
        all_data["points_per_game"] = all_data["total_points"] / all_data[
            "gameweek"
        ].replace(0, 1)
        all_data["xg_per_90"] = (
            all_data["expected_goals"] / all_data["minutes"].replace(0, 1)
        ) * 90
        all_data["xa_per_90"] = (
            all_data["expected_assists"] / all_data["minutes"].replace(0, 1)
        ) * 90

        # Value ratio - only calculate if we have price data
        price_col = None
        if "now_cost" in all_data.columns:
            price_col = "now_cost"
        elif "price_gbp" in all_data.columns:
            price_col = "price_gbp"
        elif "price" in all_data.columns:
            price_col = "price"

        if price_col:
            all_data[price_col] = pd.to_numeric(all_data[price_col], errors="coerce")
            # Handle different price formats (tenths vs pounds)
            if price_col == "now_cost":
                price_divisor = all_data[price_col] / 10  # FPL API uses tenths
            else:
                price_divisor = all_data[price_col]  # Already in pounds
            all_data["value_ratio"] = all_data["total_points"] / price_divisor.replace(
                0, 1
            )  # Points per £1m
        else:
            print("⚠️ No price data available, skipping value ratio calculation")
            all_data["value_ratio"] = 0

        # Create player options (filter to players with data)
        player_options = []

        # Get players with the most total points and data availability
        valid_players = all_data[all_data["web_name"].notna()]
        print(f"📊 Valid players with names: {len(valid_players)}")

        if valid_players.empty:
            print("❌ No players with valid names found!")
            return [], [], pd.DataFrame()

        # Handle missing team name column gracefully
        agg_dict = {
            "total_points": "max",
            "web_name": "first",
            "position": "first",
            "gameweek": "count",  # Number of gameweeks with data
        }

        # Add team name column if it exists (try different possible names)
        team_col_found = None
        for team_col in ["name", "team_name", "team"]:
            if team_col in valid_players.columns:
                agg_dict[team_col] = "first"
                team_col_found = team_col
                break

        player_stats = valid_players.groupby("player_id").agg(agg_dict).reset_index()

        # Filter to players with at least some data and sort by points (all players)
        active_players = player_stats[player_stats["gameweek"] >= 1].sort_values(
            "total_points", ascending=False
        )

        for _, player_row in active_players.iterrows():
            # Build label with available team information
            team_info = ""
            if (
                team_col_found
                and team_col_found in player_row
                and pd.notna(player_row[team_col_found])
            ):
                team_info = f", {player_row[team_col_found]}"

            label = f"{player_row['web_name']} ({player_row['position']}{team_info}) - {player_row['total_points']} pts"
            player_options.append(
                {"label": label, "value": int(player_row["player_id"])}
            )

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
            {"label": "ICT Index", "value": "ict_index"},
        ]

        # Add optional attributes if they exist in the data
        optional_attributes = [
            {"label": "Price (£m)", "value": "now_cost"},
            {"label": "Selected By %", "value": "selected_by_percent"},
            {"label": "Value Ratio (pts/£1m)", "value": "value_ratio"},
        ]

        attribute_options = base_attributes.copy()
        for attr in optional_attributes:
            if attr["value"] in all_data.columns:
                attribute_options.append(attr)

        print(
            f"📊 Available attributes: {[attr['value'] for attr in attribute_options]}"
        )

        return player_options, attribute_options, all_data

    except Exception as e:
        print(f"Error creating trends data: {e}")
        import traceback

        traceback.print_exc()
        return [], [], pd.DataFrame()


def create_trends_chart(
    data: pd.DataFrame,
    selected_player: int,
    selected_attr: str,
    multi_players: Optional[List[int]] = None,
    mo_ref=None,
) -> object:
    """
    Create interactive trends chart with multi-player comparison support

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
        return (
            mo_ref.md("⚠️ **Select a player and attribute to see trends**")
            if mo_ref
            else None
        )

    try:
        # Extract values if they're still dict objects
        if isinstance(selected_player, dict):
            player_id = selected_player.get("value")
        else:
            player_id = selected_player

        if isinstance(selected_attr, dict):
            attr_name = selected_attr.get("value")
        else:
            attr_name = selected_attr

        # Validate we have valid values
        if not player_id or not attr_name:
            return (
                mo_ref.md("⚠️ **Select a player and attribute to see trends**")
                if mo_ref
                else None
            )

        # Convert player_id to int if needed
        if isinstance(player_id, str):
            try:
                player_id = int(player_id)
            except ValueError:
                print(f"❌ Could not convert player_id to int: {player_id}")
                return mo_ref.md("❌ **Invalid player selection**") if mo_ref else None

        # Check if attribute exists in data
        if attr_name not in data.columns:
            return (
                mo_ref.md(f"❌ **Attribute '{attr_name}' not available**")
                if mo_ref
                else None
            )

        # Create a chart with error handling
        import plotly.graph_objects as go

        fig = go.Figure()

        # Collect all player IDs to plot (primary + multi-selection)
        all_player_ids = [player_id]
        if multi_players:
            # Handle different multi_players formats
            if isinstance(multi_players, list):
                for mp in multi_players:
                    mp_id = mp.get("value", mp) if isinstance(mp, dict) else mp
                    if mp_id and mp_id != player_id:  # Avoid duplicates
                        try:
                            all_player_ids.append(int(mp_id))
                        except (ValueError, TypeError):
                            continue

        # Color palette for multiple players
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        # Plot each player
        for i, pid in enumerate(all_player_ids):
            player_data = data[data["player_id"] == pid].copy()

            if player_data.empty:
                continue

            # Get player name
            player_name = (
                player_data["web_name"].iloc[0]
                if "web_name" in player_data.columns
                else f"Player {pid}"
            )

            # Sort by gameweek and clean data
            player_data = player_data.sort_values("gameweek")

            # Add trace with error handling
            try:
                fig.add_trace(
                    go.Scatter(
                        x=player_data["gameweek"],
                        y=player_data[attr_name],
                        mode="lines+markers",
                        name=player_name,
                        line=dict(width=3, color=colors[i % len(colors)]),
                        marker=dict(size=8, color=colors[i % len(colors)]),
                    )
                )
            except Exception as trace_error:
                print(f"❌ Error adding trace for {player_name}: {trace_error}")
                continue

        # Check if we successfully added any traces
        if not fig.data:
            return (
                mo_ref.md("❌ **No valid data found for selected player(s)**")
                if mo_ref
                else None
            )

        # Update layout
        attr_label = attr_name.replace("_", " ").title()
        if len(all_player_ids) > 1:
            title = f"Player Comparison: {attr_label} Over Time"
        else:
            main_player_name = fig.data[0].name if fig.data else "Player"
            title = f"{main_player_name}: {attr_label} Over Time"

        fig.update_layout(
            title=title,
            xaxis_title="Gameweek",
            yaxis_title=attr_label,
            height=500,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
        )

        # Return using marimo's plotly support
        return mo_ref.as_html(fig) if mo_ref else fig

    except Exception as e:
        print(f"❌ Error creating trends chart: {e}")
        import traceback

        traceback.print_exc()
        return mo_ref.md(f"❌ **Chart error:** {str(e)}") if mo_ref else None


def create_fixture_difficulty_visualization(
    start_gw: int, num_gws: int = 5, mo_ref=None
) -> object:
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
        from client import FPLDataClient

        client = FPLDataClient()
        from ..core.team_strength import (
            DynamicTeamStrength,
            load_historical_gameweek_data,
        )

        # Load required data
        fixtures = client.get_fixtures_normalized()
        teams = client.get_current_teams()

        # Get team strengths
        calculator = DynamicTeamStrength(debug=False)
        current_season_data = load_historical_gameweek_data(
            start_gw=1, end_gw=start_gw - 1
        )
        team_strength = calculator.get_team_strength(
            target_gameweek=start_gw,
            teams_data=teams,
            current_season_data=current_season_data,
        )

        # Filter fixtures for relevant gameweeks
        relevant_fixtures = fixtures[
            (fixtures["event"] >= start_gw) & (fixtures["event"] < start_gw + num_gws)
        ].copy()

        if relevant_fixtures.empty:
            return (
                mo_ref.md(
                    f"❌ **No fixtures found for GW{start_gw}-{start_gw + num_gws - 1}**"
                )
                if mo_ref
                else None
            )

        # Create team fixture matrix
        team_names = sorted(teams["name"].unique())
        gameweeks = list(range(start_gw, start_gw + num_gws))

        # Initialize matrix with empty values
        fixture_matrix = []

        for team_name in team_names:
            team_row = {"Team": team_name}
            team_id = teams[teams["name"] == team_name]["team_id"].iloc[0]

            for gw in gameweeks:
                # Find fixtures for this team in this gameweek
                team_fixtures = relevant_fixtures[
                    (
                        (relevant_fixtures["home_team_id"] == team_id)
                        | (relevant_fixtures["away_team_id"] == team_id)
                    )
                    & (relevant_fixtures["event"] == gw)
                ]

                if not team_fixtures.empty:
                    fixture = team_fixtures.iloc[0]

                    # Determine opponent and home/away
                    if fixture["home_team_id"] == team_id:
                        # Team is at home
                        opponent_id = fixture["away_team_id"]
                        is_home = True
                    else:
                        # Team is away
                        opponent_id = fixture["home_team_id"]
                        is_home = False

                    # Get opponent name
                    opponent_name = teams[teams["team_id"] == opponent_id]["name"].iloc[
                        0
                    ]

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
                    team_row[f"GW{gw}"] = f"{short_name} {venue}"
                    team_row[f"GW{gw}_difficulty"] = difficulty
                else:
                    team_row[f"GW{gw}"] = "BGW"  # Blank gameweek
                    team_row[f"GW{gw}_difficulty"] = 0

            fixture_matrix.append(team_row)

        # Convert to DataFrame
        fixture_df = pd.DataFrame(fixture_matrix)

        # Create display DataFrame with difficulty colors
        display_columns = ["Team"] + [f"GW{gw}" for gw in gameweeks]
        display_df = fixture_df[display_columns].copy()

        # Add difficulty summary
        difficulty_cols = [f"GW{gw}_difficulty" for gw in gameweeks]
        fixture_df["Avg_Difficulty"] = fixture_df[difficulty_cols].mean(axis=1)

        # Sort by average difficulty (easiest first)
        fixture_df = fixture_df.sort_values("Avg_Difficulty")
        display_df = display_df.reindex(fixture_df.index)

        # Add summary column
        display_df["Avg Difficulty"] = fixture_df["Avg_Difficulty"].round(3)

        # Create interactive plotly heatmap
        import plotly.graph_objects as go

        # Create difficulty matrix for heatmap (numeric values for coloring)
        heatmap_data = []
        opponent_text_data = []
        team_labels = []

        for team_name in fixture_df["Team"]:
            team_row = []
            opponent_row = []
            team_labels.append(team_name)
            for gw in gameweeks:
                difficulty_val = fixture_df[fixture_df["Team"] == team_name][
                    f"GW{gw}_difficulty"
                ].iloc[0]
                opponent_text = display_df[display_df["Team"] == team_name][
                    f"GW{gw}"
                ].iloc[0]
                team_row.append(difficulty_val)
                opponent_row.append(opponent_text)
            heatmap_data.append(team_row)
            opponent_text_data.append(opponent_row)

        # Create hover text with fixture info
        hover_text = []
        for i, team_name in enumerate(team_labels):
            team_hover_row = []
            for j, gw in enumerate(gameweeks):
                difficulty = heatmap_data[i][j]
                fixture_text = opponent_text_data[i][j]

                if difficulty == 0:
                    hover_info = f"<b>{team_name}</b><br>GW{gw}: Blank Gameweek<br>Difficulty: N/A"
                else:
                    if difficulty <= 0.8:
                        diff_desc = "Very Easy"
                        color_desc = "🟢"
                    elif difficulty <= 0.95:
                        diff_desc = "Easy"
                        color_desc = "🟢"
                    elif difficulty <= 1.05:
                        diff_desc = "Average"
                        color_desc = "🟡"
                    elif difficulty <= 1.2:
                        diff_desc = "Hard"
                        color_desc = "🟠"
                    else:
                        diff_desc = "Very Hard"
                        color_desc = "🔴"

                    hover_info = (
                        f"<b>{team_name}</b><br>"
                        f"GW{gw}: {fixture_text}<br>"
                        f"Difficulty: {difficulty:.3f}<br>"
                        f"Rating: {color_desc} {diff_desc}"
                    )
                team_hover_row.append(hover_info)
            hover_text.append(team_hover_row)

        # Create the heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data,
                x=[f"GW{gw}" for gw in gameweeks],
                y=team_labels,
                colorscale=[
                    [0.0, "#2E8B57"],  # Dark green (very easy)
                    [0.25, "#90EE90"],  # Light green (easy)
                    [0.5, "#FFFF99"],  # Yellow (average)
                    [0.75, "#FFA500"],  # Orange (hard)
                    [1.0, "#FF4500"],  # Red (very hard)
                ],
                zmid=1.0,  # Center on average difficulty
                zmin=0.0,
                zmax=1.5,
                text=opponent_text_data,
                texttemplate="%{text}",
                textfont={"size": 9, "color": "black", "family": "Arial"},
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_text,
                colorbar=dict(
                    title=dict(
                        text="Fixture Difficulty<br>(0.0=Very Easy, 1.5=Very Hard)",
                        side="right",
                    ),
                    tickvals=[0.0, 0.5, 1.0, 1.5],
                    ticktext=["Very Easy", "Easy", "Average", "Very Hard"],
                ),
            )
        )

        fig.update_layout(
            title=f"🏟️ Fixture Difficulty Heatmap: GW{start_gw}-{start_gw + num_gws - 1}<br><sub>Interactive - Hover for details | Teams sorted by avg difficulty</sub>",
            title_x=0.5,
            xaxis_title="Gameweek",
            yaxis_title="Team",
            font=dict(size=12),
            height=max(600, 25 * len(team_labels)),  # Height based on number of teams
            width=max(800, 120 * num_gws),  # Much wider for better readability
            margin=dict(l=150, r=120, t=120, b=60),  # More left margin for team names
        )

        # Update axes with better formatting
        fig.update_xaxes(side="bottom", tickfont=dict(size=12))
        fig.update_yaxes(
            autorange="reversed",  # Teams from top to bottom (easiest first)
            tickfont=dict(size=11),
            tickmode="array",
            tickvals=list(range(len(team_labels))),
            ticktext=team_labels,  # Explicitly set team names
        )

        # Summary analysis
        best_fixtures = fixture_df.nsmallest(5, "Avg_Difficulty")[
            ["Team", "Avg_Difficulty"]
        ]
        worst_fixtures = fixture_df.nlargest(5, "Avg_Difficulty")[
            ["Team", "Avg_Difficulty"]
        ]

        analysis_md = """
        ### 🎯 Fixture Difficulty Analysis Summary

        **🟢 EASIEST FIXTURE RUNS (Target for FPL assets):**
        """

        for i, (_, row) in enumerate(best_fixtures.iterrows(), 1):
            analysis_md += (
                f"\n{i}. **{row['Team']}**: {row['Avg_Difficulty']:.3f} avg difficulty"
            )

        analysis_md += "\n\n**🔴 HARDEST FIXTURE RUNS (Avoid these teams' players):**"

        for i, (_, row) in enumerate(worst_fixtures.iterrows(), 1):
            analysis_md += (
                f"\n{i}. **{row['Team']}**: {row['Avg_Difficulty']:.3f} avg difficulty"
            )

        analysis_md += """

        **💡 FPL Strategy:**
        - 🎯 **Target players** from teams with green fixtures (easy runs)
        - ⚠️ **Avoid players** from teams with red fixtures (tough runs)
        - 🏠 **Home advantage**: Teams playing at home have easier fixtures
        - 📊 **Difficulty Scale**: 0.7=Very Easy, 1.0=Average, 1.3=Very Hard
        """

        return (
            mo_ref.vstack(
                [
                    mo_ref.md(analysis_md),
                    mo_ref.md("### 📊 Interactive Fixture Difficulty Heatmap"),
                    mo_ref.md(
                        "*🟢 Green = Easy, 🟡 Yellow = Average, 🔴 Red = Hard | Hover for opponent details*"
                    ),
                    mo_ref.as_html(fig),
                    mo_ref.md("### 📋 Detailed Fixture Matrix"),
                    mo_ref.md("*Teams sorted by average difficulty (easiest first)*"),
                    mo_ref.ui.table(
                        display_df, page_size=config.visualization.large_table_page_size
                    ),
                ]
            )
            if mo_ref
            else display_df
        )

    except Exception as e:
        return (
            mo_ref.md(f"❌ **Could not create fixture difficulty analysis:** {e}")
            if mo_ref
            else None
        )


def create_xp_results_display(
    players_xp: pd.DataFrame, target_gameweek: int, mo_ref
) -> object:
    """
    Create comprehensive XP results display with form analysis and strategic insights

    Args:
        players_xp: DataFrame with calculated XP data
        target_gameweek: Target gameweek number
        mo_ref: Marimo reference for UI components

    Returns:
        Marimo UI component with XP results display
    """
    if players_xp.empty:
        return mo_ref.md("No XP data available")

    try:
        # Add comprehensive player attributes for display
        # Start with core player info
        display_columns = [
            "web_name",
            "position",
            "name",  # Team name - will be placed right after player info
            "price",
            "selected_by_percent",
        ]

        # Add XP model outputs (both 1-GW and 5-GW)
        xp_columns = [
            "xP",
            "xP_5gw",
            "xP_per_price",
            "xP_per_price_5gw",
            "expected_minutes",
            "fixture_difficulty",
            "fixture_difficulty_5gw",
            "xP_horizon_advantage",
            "fixture_outlook",
        ]
        display_columns.extend(xp_columns)

        # Add XP component breakdown if available
        xp_components = ["xP_appearance", "xP_goals", "xP_assists", "xP_clean_sheets"]
        for component in xp_components:
            if component in players_xp.columns:
                display_columns.append(component)

        # Add form indicators if they exist
        form_columns = ["momentum", "form_multiplier", "recent_points_per_game"]
        for form_col in form_columns:
            if form_col in players_xp.columns:
                display_columns.append(form_col)

        # Add statistical data if available
        stats_columns = ["xG90", "xA90", "p_60_plus_mins"]
        for stat_col in stats_columns:
            if stat_col in players_xp.columns:
                display_columns.append(stat_col)

        # Add all enriched season stats if available
        enriched_season_stats = [
            "total_points_season",
            "form_season",
            "points_per_game_season",
            "minutes",
            "starts",
            "goals_scored",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "yellow_cards",
            "red_cards",
            "saves",
            "bonus",
            "bps",
            "influence",
            "creativity",
            "threat",
            "ict_index",
            "expected_goals",
            "expected_assists",
            "expected_goals_per_90",
            "expected_assists_per_90",
            "value_form",
            "value_season",
            "transfers_in",
            "transfers_out",
            "transfers_in_event",
            "transfers_out_event",
            "chance_of_playing_this_round",
            "chance_of_playing_next_round",
            "penalties_order",
            "corners_and_indirect_freekicks_order",
            "news",
        ]
        for enriched_stat in enriched_season_stats:
            if enriched_stat in players_xp.columns:
                display_columns.append(enriched_stat)

        # Add availability status at the end
        if "status" in players_xp.columns:
            display_columns.append("status")

        # Create safe display DataFrame
        display_df = create_display_dataframe(
            players_xp,
            core_columns=display_columns,
            sort_by="xP_5gw",
            ascending=False,
            round_decimals=3,
        )

        # Generate form analysis
        form_info = format_form_analysis(players_xp)

        # Generate strategic analysis
        strategic_info = format_strategic_analysis(players_xp)

        return mo_ref.vstack(
            [
                mo_ref.md(
                    f"### ✅ Strategic XP Model - GW{target_gameweek} + 5 Horizon"
                ),
                mo_ref.md(f"**Players analyzed:** {len(players_xp)}"),
                mo_ref.md(
                    f"**Average 1-GW XP:** {players_xp['xP'].mean():.2f} | **Average 5-GW XP:** {players_xp['xP_5gw'].mean():.2f}"
                ),
                mo_ref.md(
                    f"**Top 1-GW:** {players_xp['xP'].max():.2f} | **Top 5-GW:** {players_xp['xP_5gw'].max():.2f}"
                ),
                mo_ref.md(strategic_info),
                mo_ref.md(form_info) if form_info else mo_ref.md(""),
                mo_ref.md(
                    "**All Players - Strategic Comparison (Sorted by 5-GW XP):**"
                ),
                mo_ref.md(
                    "*Showing: Season stats (total points, form, minutes, goals, assists, etc.), 1-GW vs 5-GW XP, fixture outlook, ICT components, expected stats, and more*"
                ),
                mo_ref.ui.table(display_df, page_size=25),
            ]
        )

    except Exception as e:
        return mo_ref.md(f"❌ Error creating XP results display: {e}")


def format_form_analysis(players_xp: pd.DataFrame) -> str:
    """
    Format form analysis information from XP data

    Args:
        players_xp: DataFrame with form and XP data

    Returns:
        Formatted form analysis string
    """
    try:
        # Count form-enhanced players
        form_enhanced_players = 0
        avg_form_multiplier = 1.0
        if "form_multiplier" in players_xp.columns:
            form_enhanced_players = len(
                players_xp[players_xp["form_multiplier"] != 1.0]
            )
            avg_form_multiplier = players_xp["form_multiplier"].mean()

        # Enhanced display with form information
        form_info = ""
        if form_enhanced_players > 0:
            hot_players = len(players_xp[players_xp.get("momentum", "") == "🔥"])
            cold_players = len(players_xp[players_xp.get("momentum", "") == "❄️"])
            form_info = f"**Form Analysis:** {form_enhanced_players} players with form data | {hot_players} 🔥 Hot | {cold_players} ❄️ Cold | Avg multiplier: {avg_form_multiplier:.2f}"

        return form_info
    except Exception:
        return ""


def format_strategic_analysis(players_xp: pd.DataFrame) -> str:
    """
    Format strategic 5-gameweek analysis information

    Args:
        players_xp: DataFrame with strategic XP data

    Returns:
        Formatted strategic analysis string
    """
    try:
        fixture_advantage_players = len(
            players_xp[players_xp["xP_horizon_advantage"] > 0.5]
        )
        fixture_disadvantage_players = len(
            players_xp[players_xp["xP_horizon_advantage"] < -0.5]
        )

        # Check if we have enough data for comparison
        leaders_different = False
        if not players_xp.empty and "xP" in players_xp.columns:
            display_df = players_xp.sort_values("xP_5gw", ascending=False)
            if not display_df.empty:
                top_5gw_player = display_df.iloc[0]["web_name"]
                top_1gw_players = players_xp.nlargest(1, "xP")
                if not top_1gw_players.empty:
                    top_1gw_player = top_1gw_players.iloc[0]["web_name"]
                    leaders_different = top_5gw_player != top_1gw_player

        return f"""
**🎯 Strategic 5-Gameweek Analysis:**
- Players with fixture advantage (5-GW > 1-GW): {fixture_advantage_players}
- Players with fixture disadvantage (tough upcoming): {fixture_disadvantage_players}
- 5-GW leaders different from 1-GW: {leaders_different}
"""
    except Exception as e:
        print(f"⚠️ Error in strategic analysis: {e}")
        return "**🎯 Strategic 5-Gameweek Analysis:** Analysis unavailable"


def create_form_analytics_display(players_with_xp: pd.DataFrame, mo_ref) -> object:
    """
    Create form analytics dashboard with hot/cold player detection

    Args:
        players_with_xp: DataFrame with form and XP data
        mo_ref: Marimo reference for UI components

    Returns:
        Marimo UI component with form analytics
    """
    if players_with_xp.empty or "momentum" not in players_with_xp.columns:
        return mo_ref.md("⚠️ **No form data available** - load historical data first")

    try:
        # Hot players analysis
        hot_players = players_with_xp[players_with_xp["momentum"] == "🔥"].nlargest(
            8, "xP"
        )
        cold_players = players_with_xp[players_with_xp["momentum"] == "❄️"].nsmallest(
            8, "form_multiplier"
        )

        # Value analysis with form
        value_players = players_with_xp[
            (players_with_xp["momentum"].isin(["🔥", "📈"]))
            & (players_with_xp["price"] <= 7.5)
        ].nlargest(10, "xP_per_price")

        # Expensive underperformers
        expensive_poor = players_with_xp[
            (players_with_xp["price"] >= 8.0)
            & (players_with_xp["momentum"].isin(["❄️", "📉"]))
        ].nsmallest(6, "form_multiplier")

        insights = []

        if len(hot_players) > 0:
            hot_columns = get_safe_columns(
                hot_players,
                [
                    "web_name",
                    "position",
                    "price",
                    "xP",
                    "recent_points_per_game",
                    "momentum",
                ],
            )
            insights.extend(
                [
                    mo_ref.md("### 🔥 Hot Players (Prime Transfer Targets)"),
                    mo_ref.md(
                        f"**{len(hot_players)} players in excellent recent form**"
                    ),
                    mo_ref.ui.table(hot_players[hot_columns].round(2), page_size=8),
                ]
            )

        if len(value_players) > 0:
            value_columns = get_safe_columns(
                value_players,
                ["web_name", "position", "price", "xP", "xP_per_price", "momentum"],
            )
            insights.extend(
                [
                    mo_ref.md("### 💎 Form + Value Players (Budget-Friendly Options)"),
                    mo_ref.md(
                        f"**{len(value_players)} players with good form and great value**"
                    ),
                    mo_ref.ui.table(
                        value_players[value_columns].round(3), page_size=10
                    ),
                ]
            )

        if len(cold_players) > 0:
            cold_columns = get_safe_columns(
                cold_players,
                [
                    "web_name",
                    "position",
                    "price",
                    "xP",
                    "recent_points_per_game",
                    "momentum",
                ],
            )
            insights.extend(
                [
                    mo_ref.md("### ❄️ Cold Players (Sell Candidates)"),
                    mo_ref.md(
                        f"**{len(cold_players)} players in poor recent form - consider selling**"
                    ),
                    mo_ref.ui.table(cold_players[cold_columns].round(2), page_size=8),
                ]
            )

        if len(expensive_poor) > 0:
            expensive_columns = get_safe_columns(
                expensive_poor,
                [
                    "web_name",
                    "position",
                    "price",
                    "xP",
                    "recent_points_per_game",
                    "momentum",
                ],
            )
            insights.extend(
                [
                    mo_ref.md("### 💸 Expensive Underperformers (Priority Sells)"),
                    mo_ref.md(
                        f"**{len(expensive_poor)} expensive players in poor form - sell to fund transfers**"
                    ),
                    mo_ref.ui.table(
                        expensive_poor[expensive_columns].round(2), page_size=6
                    ),
                ]
            )

        # Summary stats
        if "form_multiplier" in players_with_xp.columns:
            momentum_counts = players_with_xp["momentum"].value_counts()
            avg_multiplier = players_with_xp["form_multiplier"].mean()

            insights.append(
                mo_ref.md(f"""
### 📊 Form Summary
**Player Distribution:** 🔥 {momentum_counts.get("🔥", 0)} | 📈 {momentum_counts.get("📈", 0)} | ➡️ {momentum_counts.get("➡️", 0)} | 📉 {momentum_counts.get("📉", 0)} | ❄️ {momentum_counts.get("❄️", 0)}

**Average Form Multiplier:** {avg_multiplier:.2f}

**Transfer Strategy:** Target 🔥 hot and 📈 rising players, avoid ❄️ cold and 📉 declining players
""")
            )

        return mo_ref.vstack(insights)

    except Exception as e:
        return mo_ref.md(f"❌ Error creating form analytics: {e}")


def create_squad_form_analysis(
    current_squad: pd.DataFrame, players_with_xp: pd.DataFrame, mo_ref
) -> object:
    """
    Create current squad form analysis display

    Args:
        current_squad: Current squad DataFrame
        players_with_xp: DataFrame with form and XP data
        mo_ref: Marimo reference for UI components

    Returns:
        Marimo UI component with squad form analysis
    """
    # Check data availability
    squad_available = hasattr(current_squad, "empty") and not current_squad.empty
    players_available = hasattr(players_with_xp, "empty") and not players_with_xp.empty
    momentum_available = (
        hasattr(players_with_xp, "columns") and "momentum" in players_with_xp.columns
    )

    # Create content based on data availability
    if squad_available and players_available and momentum_available:
        try:
            # Merge squad with form data
            squad_with_form = current_squad.merge(
                players_with_xp[
                    [
                        "player_id",
                        "xP",
                        "momentum",
                        "form_multiplier",
                        "recent_points_per_game",
                    ]
                ],
                on="player_id",
                how="left",
            )

            if not squad_with_form.empty:
                # Squad form analysis
                squad_insights = []

                # Count momentum distribution in squad
                squad_momentum = squad_with_form["momentum"].value_counts()
                squad_avg_form = squad_with_form["form_multiplier"].mean()

                # Identify problem players in squad
                problem_players = squad_with_form[
                    squad_with_form["momentum"].isin(["❄️", "📉"])
                ].sort_values("form_multiplier")

                # Identify top performers in squad
                top_performers = squad_with_form[
                    squad_with_form["momentum"].isin(["🔥", "📈"])
                ].sort_values("xP", ascending=False)

                squad_insights.append(mo_ref.md("### 🔍 Current Squad Form Analysis"))

                # Squad overview
                squad_insights.append(
                    mo_ref.md(f"""
**Your Squad Form Distribution:**
- 🔥 Hot: {squad_momentum.get("🔥", 0)} players - 📈 Rising: {squad_momentum.get("📈", 0)} players
- ➡️ Stable: {squad_momentum.get("➡️", 0)} players - 📉 Declining: {squad_momentum.get("📉", 0)} players - ❄️ Cold: {squad_momentum.get("❄️", 0)} players

**Squad Average Form Multiplier:** {squad_avg_form:.2f}
""")
                )

                # Squad health assessment
                hot_count = squad_momentum.get("🔥", 0) + squad_momentum.get("📈", 0)
                cold_count = squad_momentum.get("❄️", 0) + squad_momentum.get("📉", 0)

                if hot_count >= 8:
                    health_status = "🔥 EXCELLENT - Squad is in great form, minimal transfers needed"
                    transfer_priority = "Low"
                elif hot_count >= 5:
                    health_status = (
                        "📈 GOOD - Squad form is solid, consider tactical transfers"
                    )
                    transfer_priority = "Medium"
                elif cold_count >= 8:
                    health_status = (
                        "❄️ POOR - Squad struggling, multiple transfers recommended"
                    )
                    transfer_priority = "High"
                else:
                    health_status = (
                        "➡️ AVERAGE - Squad form is stable, monitor for improvements"
                    )
                    transfer_priority = "Low"

                squad_insights.append(mo_ref.md("### 🎯 Squad Health Assessment"))
                squad_insights.append(
                    mo_ref.md(f"""
{health_status}

**Transfer Priority:** {transfer_priority}
""")
                )

                # Show top performers if any
                if not top_performers.empty:
                    top_columns = get_safe_columns(
                        top_performers,
                        [
                            "web_name",
                            "position",
                            "momentum",
                            "recent_points_per_game",
                            "xP",
                        ],
                    )
                    squad_insights.append(mo_ref.md("### ⭐ Squad Stars (Keep!)"))
                    squad_insights.append(
                        mo_ref.ui.table(
                            top_performers[top_columns].round(2), page_size=5
                        )
                    )

                # Show problem players if any
                if not problem_players.empty:
                    problem_columns = get_safe_columns(
                        problem_players,
                        [
                            "web_name",
                            "position",
                            "momentum",
                            "recent_points_per_game",
                            "xP",
                        ],
                    )
                    squad_insights.append(
                        mo_ref.md("### ⚠️ Problem Players (Consider Selling)")
                    )
                    squad_insights.append(
                        mo_ref.ui.table(
                            problem_players[problem_columns].round(2), page_size=5
                        )
                    )

                return mo_ref.vstack(squad_insights)
            else:
                return mo_ref.md("⚠️ **Could not merge squad with form data**")
        except Exception as e:
            return mo_ref.md(f"❌ Error creating squad form analysis: {e}")
    else:
        # Debug information for missing data
        debug_parts = []
        if not squad_available:
            debug_parts.append("No squad loaded")
        if not players_available:
            debug_parts.append("No player XP data")
        if not momentum_available:
            debug_parts.append("No form/momentum data")

        debug_msg = ", ".join(debug_parts)
        return mo_ref.md(f"⚠️ **Squad form analysis unavailable**\n\n_{debug_msg}_")


def create_model_accuracy_visualization(
    target_gameweek: int,
    lookback_gameweeks: int = 5,
    algorithm_versions: Optional[List[str]] = None,
    mo_ref=None,
) -> object:
    """
    Create model accuracy tracking visualization comparing predicted vs actual results.

    Args:
        target_gameweek: Current gameweek for analysis
        lookback_gameweeks: Number of completed gameweeks to analyze
        algorithm_versions: List of algorithm versions to compare (None = current only)
        mo_ref: Marimo reference for UI components

    Returns:
        Marimo UI component with accuracy visualization
    """
    from fpl_team_picker.domain.services.performance_analytics_service import (
        PerformanceAnalyticsService,
    )
    from client import FPLDataClient

    try:
        analytics_service = PerformanceAnalyticsService()
        client = FPLDataClient()

        # Calculate gameweek range for completed gameweeks only
        start_gw = max(1, target_gameweek - lookback_gameweeks)
        end_gw = target_gameweek - 1  # Only completed gameweeks

        if end_gw < start_gw:
            return mo_ref.md(
                "⚠️ **No completed gameweeks available for accuracy analysis**"
            )

        # Use current algorithm if no versions specified
        if algorithm_versions is None:
            algorithm_versions = ["current"]

        # Recompute predictions for historical gameweeks
        predictions_df = analytics_service.batch_recompute_season(
            start_gw=start_gw,
            end_gw=end_gw,
            algorithm_versions=algorithm_versions,
            include_snapshots=True,
        )

        # Collect accuracy metrics for each gameweek and algorithm
        accuracy_data = []
        for gw in range(start_gw, end_gw + 1):
            # Get actual results
            actual_results = client.get_gameweek_performance(gw)

            for algo in algorithm_versions:
                try:
                    # Get predictions for this gameweek and algorithm
                    # Use .loc with slice(None) for player_id level since we want all players
                    gw_predictions = predictions_df.loc[(gw, slice(None), algo)]

                    # Calculate accuracy metrics
                    metrics = analytics_service.calculate_accuracy_metrics(
                        gw_predictions.reset_index(),
                        actual_results,
                        by_position=True,
                    )

                    if "overall" in metrics:
                        accuracy_data.append(
                            {
                                "gameweek": gw,
                                "algorithm": algo,
                                "mae": metrics["overall"]["mae"],
                                "rmse": metrics["overall"]["rmse"],
                                "correlation": metrics["overall"]["correlation"],
                                "mean_predicted": metrics["overall"]["mean_predicted"],
                                "mean_actual": metrics["overall"]["mean_actual"],
                                "player_count": metrics["player_count"],
                            }
                        )
                except Exception as e:
                    print(f"⚠️ Failed to calculate accuracy for GW{gw} {algo}: {e}")
                    continue

        if not accuracy_data:
            return mo_ref.md("⚠️ **No accuracy data available for visualization**")

        accuracy_df = pd.DataFrame(accuracy_data)

        # Create visualization components
        components = []

        # Overall accuracy summary
        components.append(mo_ref.md("## 📊 Model Accuracy Tracking"))

        # Summary statistics
        summary_stats = (
            accuracy_df.groupby("algorithm")
            .agg(
                {
                    "mae": "mean",
                    "rmse": "mean",
                    "correlation": "mean",
                    "player_count": "mean",
                }
            )
            .round(2)
        )

        components.append(mo_ref.md("### 📈 Overall Performance (Average)"))
        components.append(mo_ref.ui.table(summary_stats))

        # Accuracy trends chart
        import plotly.graph_objects as go

        fig = go.Figure()

        for algo in algorithm_versions:
            algo_data = accuracy_df[accuracy_df["algorithm"] == algo]

            # MAE trend
            fig.add_trace(
                go.Scatter(
                    x=algo_data["gameweek"],
                    y=algo_data["mae"],
                    mode="lines+markers",
                    name=f"{algo} - MAE",
                    hovertemplate="GW%{x}<br>MAE: %{y:.2f}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Mean Absolute Error (MAE) Trends",
            xaxis_title="Gameweek",
            yaxis_title="MAE (points)",
            hovermode="x unified",
            height=400,
        )

        components.append(mo_ref.ui.plotly(fig))

        # Correlation trends
        fig_corr = go.Figure()

        for algo in algorithm_versions:
            algo_data = accuracy_df[accuracy_df["algorithm"] == algo]

            fig_corr.add_trace(
                go.Scatter(
                    x=algo_data["gameweek"],
                    y=algo_data["correlation"],
                    mode="lines+markers",
                    name=f"{algo} - Correlation",
                    hovertemplate="GW%{x}<br>Correlation: %{y:.3f}<extra></extra>",
                )
            )

        fig_corr.update_layout(
            title="Prediction Correlation with Actual Points",
            xaxis_title="Gameweek",
            yaxis_title="Correlation Coefficient",
            hovermode="x unified",
            height=400,
        )

        components.append(mo_ref.ui.plotly(fig_corr))

        # Gameweek-by-gameweek breakdown
        components.append(mo_ref.md("### 📋 Gameweek-by-Gameweek Breakdown"))
        display_cols = [
            "gameweek",
            "algorithm",
            "mae",
            "rmse",
            "correlation",
            "player_count",
        ]
        components.append(
            mo_ref.ui.table(
                accuracy_df[display_cols].sort_values(["gameweek", "algorithm"]),
                page_size=10,
            )
        )

        return mo_ref.vstack(components)

    except Exception as e:
        return mo_ref.md(f"❌ Error creating accuracy visualization: {e}")


def create_position_accuracy_visualization(
    target_gameweek: int,
    algorithm_version: str = "current",
    mo_ref=None,
) -> object:
    """
    Create position-specific accuracy analysis visualization.

    Args:
        target_gameweek: Gameweek to analyze (must be completed)
        algorithm_version: Algorithm version to use
        mo_ref: Marimo reference for UI components

    Returns:
        Marimo UI component with position-specific accuracy analysis
    """
    from fpl_team_picker.domain.services.performance_analytics_service import (
        PerformanceAnalyticsService,
    )
    from client import FPLDataClient

    try:
        analytics_service = PerformanceAnalyticsService()
        client = FPLDataClient()

        # Recompute predictions for target gameweek
        predictions = analytics_service.recompute_historical_xp(
            target_gameweek=target_gameweek,
            algorithm_version=algorithm_version,
            include_snapshots=True,
        )

        # Get actual results
        actual_results = client.get_gameweek_performance(target_gameweek)

        # Calculate position-specific metrics
        metrics = analytics_service.calculate_accuracy_metrics(
            predictions, actual_results, by_position=True
        )

        if "error" in metrics:
            return mo_ref.md(f"⚠️ **{metrics['error']}**")

        # Create visualization components
        components = []

        components.append(
            mo_ref.md(f"## 📊 Position-Specific Accuracy (GW{target_gameweek})")
        )

        # Overall metrics
        components.append(mo_ref.md("### 📈 Overall Performance"))
        overall_df = pd.DataFrame([metrics["overall"]]).T
        overall_df.columns = ["Value"]
        components.append(mo_ref.ui.table(overall_df))

        # Position-specific breakdown
        if "by_position" in metrics:
            components.append(mo_ref.md("### 🎯 Position-Specific Breakdown"))

            position_df = pd.DataFrame(metrics["by_position"]).T
            position_df = position_df.sort_values("mae")

            components.append(mo_ref.ui.table(position_df))

            # Position accuracy comparison chart
            import plotly.graph_objects as go

            fig = go.Figure()

            positions = list(metrics["by_position"].keys())
            mae_values = [metrics["by_position"][pos]["mae"] for pos in positions]
            correlations = [
                metrics["by_position"][pos]["correlation"] for pos in positions
            ]

            # MAE by position
            fig.add_trace(
                go.Bar(
                    x=positions,
                    y=mae_values,
                    name="MAE",
                    marker_color="lightblue",
                    hovertemplate="%{x}<br>MAE: %{y:.2f}<extra></extra>",
                )
            )

            fig.update_layout(
                title="Mean Absolute Error by Position",
                xaxis_title="Position",
                yaxis_title="MAE (points)",
                height=400,
            )

            components.append(mo_ref.ui.plotly(fig))

            # Correlation by position
            fig_corr = go.Figure()

            fig_corr.add_trace(
                go.Bar(
                    x=positions,
                    y=correlations,
                    name="Correlation",
                    marker_color="lightgreen",
                    hovertemplate="%{x}<br>Correlation: %{y:.3f}<extra></extra>",
                )
            )

            fig_corr.update_layout(
                title="Prediction Correlation by Position",
                xaxis_title="Position",
                yaxis_title="Correlation Coefficient",
                height=400,
            )

            components.append(mo_ref.ui.plotly(fig_corr))

        return mo_ref.vstack(components)

    except Exception as e:
        return mo_ref.md(f"❌ Error creating position accuracy visualization: {e}")


def create_algorithm_comparison_visualization(
    start_gw: int,
    end_gw: int,
    algorithm_versions: List[str],
    mo_ref=None,
) -> object:
    """
    Create algorithm comparison visualization to test different parameter sets.

    Args:
        start_gw: Starting gameweek for comparison
        end_gw: Ending gameweek for comparison
        algorithm_versions: List of algorithm versions to compare
        mo_ref: Marimo reference for UI components

    Returns:
        Marimo UI component with algorithm comparison analysis
    """
    from fpl_team_picker.domain.services.performance_analytics_service import (
        PerformanceAnalyticsService,
        ALGORITHM_VERSIONS,
    )
    from client import FPLDataClient

    try:
        analytics_service = PerformanceAnalyticsService()
        client = FPLDataClient()

        # Validate algorithm versions
        invalid_algos = [v for v in algorithm_versions if v not in ALGORITHM_VERSIONS]
        if invalid_algos:
            return mo_ref.md(
                f"⚠️ **Invalid algorithm versions: {invalid_algos}**\n\nAvailable: {list(ALGORITHM_VERSIONS.keys())}"
            )

        # Batch recompute for all algorithms
        predictions_df = analytics_service.batch_recompute_season(
            start_gw=start_gw,
            end_gw=end_gw,
            algorithm_versions=algorithm_versions,
            include_snapshots=True,
        )

        # Calculate accuracy for each algorithm across all gameweeks
        algorithm_metrics = {}

        for algo in algorithm_versions:
            algo_accuracies = []

            for gw in range(start_gw, end_gw + 1):
                try:
                    # Get predictions for this gameweek
                    # Use .loc with slice(None) for player_id level since we want all players
                    gw_predictions = predictions_df.loc[(gw, slice(None), algo)]

                    # Get actual results
                    actual_results = client.get_gameweek_performance(gw)

                    # Calculate metrics
                    metrics = analytics_service.calculate_accuracy_metrics(
                        gw_predictions.reset_index(), actual_results, by_position=False
                    )

                    if "overall" in metrics:
                        algo_accuracies.append(metrics["overall"])

                except Exception as e:
                    print(f"⚠️ Failed to calculate accuracy for GW{gw} {algo}: {e}")
                    continue

            if algo_accuracies:
                # Average metrics across gameweeks
                algorithm_metrics[algo] = {
                    "mae": sum(m["mae"] for m in algo_accuracies)
                    / len(algo_accuracies),
                    "rmse": sum(m["rmse"] for m in algo_accuracies)
                    / len(algo_accuracies),
                    "correlation": sum(m["correlation"] for m in algo_accuracies)
                    / len(algo_accuracies),
                    "gameweeks": len(algo_accuracies),
                }

        if not algorithm_metrics:
            return mo_ref.md("⚠️ **No algorithm metrics available**")

        # Create visualization components
        components = []

        components.append(
            mo_ref.md(f"## 🔬 Algorithm Comparison (GW{start_gw}-{end_gw})")
        )

        # Algorithm comparison table
        comparison_df = pd.DataFrame(algorithm_metrics).T.round(3)
        comparison_df = comparison_df.sort_values("mae")

        components.append(mo_ref.md("### 📊 Performance Comparison"))
        components.append(mo_ref.ui.table(comparison_df))

        # Algorithm configuration details
        components.append(mo_ref.md("### ⚙️ Algorithm Configurations"))

        config_details = []
        for algo in algorithm_versions:
            if algo in ALGORITHM_VERSIONS:
                config = ALGORITHM_VERSIONS[algo]
                config_details.append(
                    f"**{algo}**: form_weight={config.form_weight}, form_window={config.form_window}"
                )

        components.append(mo_ref.md("\n\n".join(config_details)))

        # Comparison chart
        import plotly.graph_objects as go

        fig = go.Figure()

        algorithms = list(algorithm_metrics.keys())
        mae_values = [algorithm_metrics[a]["mae"] for a in algorithms]
        correlation_values = [algorithm_metrics[a]["correlation"] for a in algorithms]

        # MAE comparison
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=mae_values,
                name="MAE",
                marker_color="lightcoral",
                hovertemplate="%{x}<br>MAE: %{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Mean Absolute Error by Algorithm",
            xaxis_title="Algorithm Version",
            yaxis_title="MAE (points)",
            height=400,
        )

        components.append(mo_ref.ui.plotly(fig))

        # Correlation comparison
        fig_corr = go.Figure()

        fig_corr.add_trace(
            go.Bar(
                x=algorithms,
                y=correlation_values,
                name="Correlation",
                marker_color="lightblue",
                hovertemplate="%{x}<br>Correlation: %{y:.3f}<extra></extra>",
            )
        )

        fig_corr.update_layout(
            title="Prediction Correlation by Algorithm",
            xaxis_title="Algorithm Version",
            yaxis_title="Correlation Coefficient",
            height=400,
        )

        components.append(mo_ref.ui.plotly(fig_corr))

        # Winner summary
        best_algo = comparison_df.index[0]
        components.append(
            mo_ref.md(f"""
### 🏆 Best Performing Algorithm

**{best_algo}** achieved the lowest MAE ({comparison_df.loc[best_algo, "mae"]:.2f}) across {comparison_df.loc[best_algo, "gameweeks"]:.0f} gameweeks.

**Recommendation**: Consider using this algorithm configuration for future predictions.
""")
        )

        return mo_ref.vstack(components)

    except Exception as e:
        return mo_ref.md(f"❌ Error creating algorithm comparison: {e}")
