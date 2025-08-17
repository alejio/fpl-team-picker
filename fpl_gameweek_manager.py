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
        # FPL Gameweek Manager

        **Weekly Decision Making Tool**
        
        Optimize your FPL decisions each gameweek: starting 11, transfers, captaincy, and more.
        Perfect complement to your season-start team builder.
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
    from datetime import datetime, timedelta
    import warnings
    warnings.filterwarnings('ignore')
    
    # Data paths
    DATA_DIR = Path("../fpl-dataset-builder/data/")
    
    # Import prediction storage for integration
    from prediction_storage import PredictionStorage
    
    return DATA_DIR, Path, np, pd, plt, sns, warnings, PredictionStorage, datetime, timedelta


@app.cell
def __():
    mo.md("## Current Squad Setup")
    return


@app.cell
def __(DATA_DIR, pd, Path):
    # Enhanced dataset loading with live data
    def load_gameweek_datasets(target_gw=None):
        """Load datasets needed for gameweek management including live data"""
        print("Loading datasets for gameweek management...")
        
        # Core player data
        players = pd.read_csv(DATA_DIR / "fpl_players_current.csv")
        teams = pd.read_csv(DATA_DIR / "fpl_teams_current.csv")
        fixtures = pd.read_csv(DATA_DIR / "fpl_fixtures_normalized.csv")
        xg_rates = pd.read_csv(DATA_DIR / "fpl_player_xg_xa_rates.csv")
        
        # Historical performance for form analysis
        historical = pd.read_csv(DATA_DIR / "fpl_historical_gameweek_data.csv")
        
        # Live data integration
        live_data = None
        deltas_data = None
        
        # Load live gameweek data if available
        if target_gw is not None:
            live_file = DATA_DIR / f"fpl_live_gameweek_{target_gw}.csv"
            if live_file.exists():
                live_data = pd.read_csv(live_file)
                print(f"✅ Loaded live data for GW{target_gw}: {len(live_data)} player records")
            else:
                print(f"⚠️  Live data for GW{target_gw} not found, using historical data")
        
        # Load performance deltas
        deltas_file = DATA_DIR / "fpl_player_deltas_current.csv"
        if deltas_file.exists():
            deltas_data = pd.read_csv(deltas_file)
            print(f"✅ Loaded performance deltas: {len(deltas_data)} player records")
        else:
            print("⚠️  Performance deltas not found")
        
        print(f"Core data loaded: {len(players)} players, {len(teams)} teams, {len(fixtures)} fixtures")
        print(f"Historical data: {len(historical)} records")
        
        return players, teams, fixtures, xg_rates, historical, live_data, deltas_data
    
    # Load with default GW2 (can be updated dynamically)
    players, teams, fixtures, xg_rates, historical, live_data, deltas_data = load_gameweek_datasets(2)
    return fixtures, historical, load_gameweek_datasets, players, teams, xg_rates, live_data, deltas_data


@app.cell
def __(mo):
    mo.md("### Input Your Current Squad")
    return


@app.cell
def __(players, mo, teams, load_gameweek_datasets):
    # Create current squad input interface
    
    # Gameweek selector with live data loading
    gameweek_selector = mo.ui.number(
        value=2,  # Default to GW2 since GW1 is in progress
        start=1,
        stop=38,
        step=1,
        label="Current Gameweek"
    )
    
    # Reload data when gameweek changes
    if gameweek_selector.value != 2:  # Only reload if different from default
        _, _, _, _, _, live_data_updated, deltas_data_updated = load_gameweek_datasets(gameweek_selector.value)
        print(f"📊 Data reloaded for GW{gameweek_selector.value}")
    else:
        live_data_updated = None
        deltas_data_updated = None
    
    # Budget inputs
    money_itb = mo.ui.number(
        value=0.0,
        start=0.0,
        stop=50.0,
        step=0.1,
        label="Money in the Bank (£m)"
    )
    
    free_transfers = mo.ui.number(
        value=1,
        start=0,
        stop=5,
        step=1,
        label="Free Transfers Available"
    )
    
    # Create player options for squad selection
    def create_player_options():
        """Create formatted player options for squad selection"""
        # Merge players with teams for display
        players_teams = players.merge(teams, on='team_id', how='left')
        
        # Sort by position then price descending
        players_sorted = players_teams.sort_values(['position', 'price_gbp'], ascending=[True, False])
        
        options = []
        for _, player in players_sorted.iterrows():
            # Format: "Name (POS, TEAM) - £X.Xm"
            label = f"{player['web_name']} ({player['position']}, {player['name']}) - £{player['price_gbp']}m"
            options.append({"label": label, "value": player['player_id']})
        
        return options
    
    player_options = create_player_options()
    
    # Squad selection (15 players)
    current_squad_selector = mo.ui.multiselect(
        options=player_options,
        label="Select Your Current 15-Player Squad",
        max_selections=15
    )
    
    mo.vstack([
        mo.md("**Gameweek & Budget Information:**"),
        mo.hstack([gameweek_selector, money_itb, free_transfers]),
        mo.md("**Current Squad Selection:**"),
        mo.md("Select exactly 15 players from your current FPL squad:"),
        current_squad_selector
    ])
    
    return current_squad_selector, free_transfers, gameweek_selector, money_itb, player_options


@app.cell
def __(mo):
    mo.md("## Squad Analysis")
    return


@app.cell
def __(current_squad_selector, players, teams, gameweek_selector, money_itb, free_transfers, pd, mo):
    # Analyze current squad
    def analyze_current_squad(squad_ids, current_gw, money, transfers):
        """Analyze the selected current squad"""
        if not squad_ids or len(squad_ids) != 15:
            return None, f"Please select exactly 15 players (currently selected: {len(squad_ids) if squad_ids else 0})"
        
        # Get squad data
        squad_data = players[players['player_id'].isin(squad_ids)].copy()
        squad_with_teams = squad_data.merge(teams, on='team_id', how='left')
        
        # Calculate squad metrics
        total_value = squad_with_teams['price_gbp'].sum()
        position_counts = squad_with_teams['position'].value_counts()
        team_counts = squad_with_teams['name'].value_counts()
        
        # Check formation validity
        formation_valid = (
            position_counts.get('GKP', 0) == 2 and
            position_counts.get('DEF', 0) == 5 and
            position_counts.get('MID', 0) == 5 and
            position_counts.get('FWD', 0) == 3
        )
        
        # Check team limits
        team_violations = team_counts[team_counts > 3]
        
        # Calculate total budget
        total_budget = total_value + money
        
        squad_summary = {
            'total_players': len(squad_with_teams),
            'total_value': total_value,
            'money_itb': money,
            'total_budget': total_budget,
            'free_transfers': transfers,
            'formation_valid': formation_valid,
            'position_counts': position_counts.to_dict(),
            'team_violations': len(team_violations) > 0,
            'team_violation_details': team_violations.to_dict() if len(team_violations) > 0 else {}
        }
        
        return squad_with_teams, squad_summary
    
    # Analyze current selection
    squad_data, squad_analysis = analyze_current_squad(
        current_squad_selector.value or [],
        gameweek_selector.value,
        money_itb.value,
        free_transfers.value
    )
    
    if squad_data is not None:
        # Display squad analysis
        analysis_display = mo.vstack([
            mo.md(f"""
            ### Squad Analysis (GW{gameweek_selector.value})
            
            **Budget Situation:**
            - **Squad Value:** £{squad_analysis['total_value']:.1f}m
            - **Money ITB:** £{squad_analysis['money_itb']:.1f}m  
            - **Total Budget:** £{squad_analysis['total_budget']:.1f}m
            - **Free Transfers:** {squad_analysis['free_transfers']}
            
            **Squad Composition:**
            - **GKP:** {squad_analysis['position_counts'].get('GKP', 0)}/2
            - **DEF:** {squad_analysis['position_counts'].get('DEF', 0)}/5
            - **MID:** {squad_analysis['position_counts'].get('MID', 0)}/5
            - **FWD:** {squad_analysis['position_counts'].get('FWD', 0)}/3
            - **Formation Valid:** {'✅' if squad_analysis['formation_valid'] else '❌'}
            
            **Team Limits:**
            - **Violations:** {'❌ ' + str(squad_analysis['team_violation_details']) if squad_analysis['team_violations'] else '✅ All good'}
            """),
            mo.md("**Current Squad:**"),
            mo.ui.table(squad_data[['web_name', 'position', 'name', 'price_gbp', 'selected_by_percentage']].round(1), page_size=15)
        ])
    else:
        analysis_display = mo.md(f"**{squad_analysis}**")
    
    analysis_display
    
    return squad_analysis, squad_data, analyze_current_squad


@app.cell
def __(mo):
    mo.md("## Single Gameweek xP Calculator")
    return


@app.cell
def __(squad_data, gameweek_selector, fixtures, teams, live_data, deltas_data, np, pd):
    # Enhanced form-weighted xP calculator
    def calculate_form_weighted_xp(players_df, target_gw, fixtures_df, teams_df, live_gw_data=None, deltas_df=None):
        """Calculate expected points with form weighting from live data"""
        if players_df is None or len(players_df) == 0:
            return pd.DataFrame()
        
        players_gw = players_df.copy()
        
        # Get fixtures for target gameweek
        gw_fixtures = fixtures_df[fixtures_df['event'] == target_gw].copy()
        
        if len(gw_fixtures) == 0:
            print(f"No fixtures found for GW{target_gw}")
            return players_gw
        
        # Add team names to fixtures
        gw_fixtures = gw_fixtures.merge(
            teams_df[['team_id', 'name']], 
            left_on='home_team_id', 
            right_on='team_id', 
            how='left'
        ).drop('team_id', axis=1).rename(columns={'name': 'home_team'})
        
        gw_fixtures = gw_fixtures.merge(
            teams_df[['team_id', 'name']], 
            left_on='away_team_id', 
            right_on='team_id', 
            how='left'
        ).drop('team_id', axis=1).rename(columns={'name': 'away_team'})
        
        # Enhanced team strength (can use your existing model)
        team_strength = {
            'Manchester City': 1.3, 'Arsenal': 1.25, 'Liverpool': 1.2, 'Aston Villa': 1.1,
            'Tottenham': 1.05, 'Chelsea': 1.0, 'Newcastle': 0.95, 'Manchester Utd': 0.9,
            'West Ham': 0.85, 'Crystal Palace': 0.8, 'Brighton': 0.85, 'Bournemouth': 0.75,
            'Fulham': 0.8, 'Wolves': 0.75, 'Everton': 0.7, 'Brentford': 0.75,
            'Nottingham Forest': 0.7, 'Leicester': 0.65, 'Ipswich': 0.6, 'Southampton': 0.6
        }
        
        # Calculate fixture difficulty
        team_difficulty = {}
        for _, fixture in gw_fixtures.iterrows():
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            
            home_strength = team_strength.get(home_team, 1.0)
            away_strength = team_strength.get(away_team, 1.0)
            
            team_difficulty[home_team] = 2.0 - away_strength
            team_difficulty[away_team] = 2.0 - home_strength
        
        players_gw['fixture_difficulty'] = players_gw['name'].apply(lambda x: team_difficulty.get(x, 1.0))
        
        # Enhanced minutes calculation with live data
        def calculate_form_weighted_minutes(row):
            """Form-weighted minutes calculation using live and delta data"""
            sbp = row.get('selected_by_percentage', 0)
            availability = row.get('availability_status', 'a')
            
            # Base probability from SBP
            if sbp >= 20:
                base_prob = 0.9
            elif sbp >= 10:
                base_prob = 0.75
            elif sbp >= 5:
                base_prob = 0.6
            elif sbp >= 2:
                base_prob = 0.4
            else:
                base_prob = 0.2
            
            # Form adjustment from delta data
            form_multiplier = 1.0
            if deltas_df is not None:
                player_delta = deltas_df[deltas_df['player_id'] == row['player_id']]
                if len(player_delta) > 0:
                    delta_row = player_delta.iloc[0]
                    # Adjust based on recent performance
                    minutes_delta = delta_row.get('minutes_delta', 0)
                    points_delta = delta_row.get('total_points_delta', 0)
                    
                    # Players with more minutes/points recently are more likely to play
                    if minutes_delta > 60:  # Played full game recently
                        form_multiplier *= 1.2
                    elif minutes_delta < 30:  # Limited minutes recently
                        form_multiplier *= 0.8
                    
                    if points_delta > 6:  # Good performance
                        form_multiplier *= 1.1
                    elif points_delta < 2:  # Poor performance
                        form_multiplier *= 0.9
            
            # Live data adjustment
            if live_gw_data is not None:
                player_live = live_gw_data[live_gw_data['player_id'] == row['player_id']]
                if len(player_live) > 0:
                    live_row = player_live.iloc[0]
                    # If player started recently, higher chance to start again
                    if live_row.get('starts', 0) == 1:
                        form_multiplier *= 1.15
                    # Consider recent performance
                    recent_points = live_row.get('total_points', 0)
                    if recent_points > 8:  # Excellent performance
                        form_multiplier *= 1.2
                    elif recent_points < 2:  # Poor performance
                        form_multiplier *= 0.85
            
            # Apply availability status
            if availability == 'i' or availability == 's':  # Injured/Suspended
                return 0
            elif availability == 'd':  # Doubtful
                base_prob *= 0.5
            
            # Calculate final probability with form adjustment
            final_prob = min(0.95, base_prob * form_multiplier)  # Cap at 95%
            
            # Position-based minutes
            if row['position'] == 'GKP':
                return final_prob * 90
            else:
                return final_prob * 75
        
        players_gw['expected_minutes'] = players_gw.apply(calculate_form_weighted_minutes, axis=1)
        
        # Enhanced xG/xA rates with form weighting
        def get_form_weighted_rates(row):
            """Get xG/xA rates with recent form weighting"""
            # Base rates from data or position defaults
            base_xg = row.get('xG90', {'GKP': 0.01, 'DEF': 0.05, 'MID': 0.15, 'FWD': 0.35}[row['position']])
            base_xa = row.get('xA90', {'GKP': 0.02, 'DEF': 0.08, 'MID': 0.20, 'FWD': 0.15}[row['position']])
            
            # Form adjustment from live data
            form_xg_mult = 1.0
            form_xa_mult = 1.0
            
            if live_gw_data is not None:
                player_live = live_gw_data[live_gw_data['player_id'] == row['player_id']]
                if len(player_live) > 0:
                    live_row = player_live.iloc[0]
                    # Weight recent actual performance vs season averages (70% recent, 30% season)
                    recent_xg = live_row.get('expected_goals', 0)
                    recent_xa = live_row.get('expected_assists', 0)
                    recent_mins = live_row.get('minutes', 1)  # Avoid division by zero
                    
                    if recent_mins > 0:
                        recent_xg90 = (recent_xg / recent_mins) * 90
                        recent_xa90 = (recent_xa / recent_mins) * 90
                        
                        # Blend recent form with season average
                        if base_xg > 0:
                            form_xg_mult = 0.7 * (recent_xg90 / base_xg) + 0.3
                        if base_xa > 0:
                            form_xa_mult = 0.7 * (recent_xa90 / base_xa) + 0.3
            
            return base_xg * form_xg_mult, base_xa * form_xa_mult
        
        # Apply form-weighted rates
        form_rates = players_gw.apply(get_form_weighted_rates, axis=1, result_type='expand')
        players_gw['xG90_form'] = form_rates[0]
        players_gw['xA90_form'] = form_rates[1]
        
        # Calculate form-weighted xP
        xG_gw = (players_gw['xG90_form'] / 90) * players_gw['expected_minutes'] * players_gw['fixture_difficulty']
        xA_gw = (players_gw['xA90_form'] / 90) * players_gw['expected_minutes'] * players_gw['fixture_difficulty']
        
        # Appearance points
        appearance_pts = np.where(players_gw['expected_minutes'] >= 60, 2, 
                         np.where(players_gw['expected_minutes'] > 0, 1, 0))
        
        # Goal and assist points
        goal_multipliers = {'GKP': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
        goal_pts = xG_gw * players_gw['position'].map(goal_multipliers)
        assist_pts = xA_gw * 3
        
        # Enhanced bonus point prediction using BPS data
        bonus_pts = 0
        if live_gw_data is not None:
            # Use recent BPS to predict bonus likelihood
            def predict_bonus(row):
                player_live = live_gw_data[live_gw_data['player_id'] == row['player_id']]
                if len(player_live) > 0:
                    recent_bps = player_live.iloc[0].get('bps', 0)
                    if recent_bps > 30:  # High BPS typically gets bonus
                        return 2.0  # Approximate bonus expectation
                    elif recent_bps > 20:
                        return 1.0
                return 0.5  # Small chance for anyone
            
            bonus_pts = players_gw.apply(predict_bonus, axis=1)
        
        # Clean sheet points
        cs_multipliers = {'GKP': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
        cs_prob = players_gw['fixture_difficulty'] * 0.25
        cs_pts = np.where(players_gw['expected_minutes'] >= 60, 
                         cs_prob * players_gw['position'].map(cs_multipliers), 0)
        
        # Total form-weighted xP
        players_gw['xP_gw'] = appearance_pts + goal_pts + assist_pts + bonus_pts + cs_pts
        
        # Add form indicators
        if deltas_df is not None:
            players_gw = players_gw.merge(
                deltas_df[['player_id', 'total_points_delta', 'minutes_delta']].rename(columns={
                    'total_points_delta': 'form_points_delta',
                    'minutes_delta': 'form_minutes_delta'
                }), 
                on='player_id', 
                how='left'
            )
            
            # Create momentum indicator
            def get_momentum_indicator(row):
                points_delta = row.get('form_points_delta', 0)
                if points_delta > 6:
                    return "🔥"  # Hot
                elif points_delta > 3:
                    return "📈"  # Rising
                elif points_delta < 0:
                    return "❄️"  # Cold
                elif points_delta < 2:
                    return "📉"  # Declining
                else:
                    return "➡️"  # Stable
            
            players_gw['momentum'] = players_gw.apply(get_momentum_indicator, axis=1)
        else:
            players_gw['momentum'] = "➡️"
        
        return players_gw
    
    # Calculate form-weighted xP for current squad
    if squad_data is not None:
        squad_with_xp = calculate_form_weighted_xp(
            squad_data, 
            gameweek_selector.value, 
            fixtures, 
            teams,
            live_data,
            deltas_data
        )
        print(f"✨ Calculated form-weighted xP for {len(squad_with_xp)} players")
    else:
        squad_with_xp = pd.DataFrame()
    
    return squad_with_xp, calculate_form_weighted_xp


@app.cell
def __(squad_with_xp, mo, gameweek_selector):
    # Display squad with enhanced xP including form indicators
    if len(squad_with_xp) > 0:
        # Prepare display columns with form data
        display_cols = ['web_name', 'position', 'name', 'price_gbp', 'expected_minutes', 'fixture_difficulty', 'xP_gw']
        
        # Add momentum indicator if available
        if 'momentum' in squad_with_xp.columns:
            display_cols.insert(-1, 'momentum')
        
        # Add form deltas if available
        if 'form_points_delta' in squad_with_xp.columns:
            display_cols.insert(-1, 'form_points_delta')
        
        display_data = squad_with_xp[display_cols].round(2).sort_values('xP_gw', ascending=False)
        
        # Create enhanced display with form analysis
        form_summary = ""
        if 'momentum' in squad_with_xp.columns:
            momentum_counts = squad_with_xp['momentum'].value_counts()
            form_summary = f"""
            **Form Summary:**
            - 🔥 Hot form: {momentum_counts.get('🔥', 0)} players
            - 📈 Rising: {momentum_counts.get('📈', 0)} players  
            - ➡️ Stable: {momentum_counts.get('➡️', 0)} players
            - 📉 Declining: {momentum_counts.get('📉', 0)} players
            - ❄️ Cold: {momentum_counts.get('❄️', 0)} players
            """
        
        xp_display = mo.vstack([
            mo.md(f"### Form-Weighted Expected Points (GW{gameweek_selector.value})"),
            mo.md(form_summary) if form_summary else mo.md(""),
            mo.ui.table(display_data, page_size=15)
        ])
    else:
        xp_display = mo.md("Select your squad above to see expected points analysis.")
    
    xp_display
    return (xp_display,)


@app.cell
def __(mo):
    mo.md("## Starting 11 Optimizer")
    return


@app.cell
def __(mo):
    # Formation selector
    formation_options = [
        {"label": "3-4-3", "value": (1, 3, 4, 3)},
        {"label": "3-5-2", "value": (1, 3, 5, 2)},
        {"label": "4-3-3", "value": (1, 4, 3, 3)},
        {"label": "4-4-2", "value": (1, 4, 4, 2)},
        {"label": "4-5-1", "value": (1, 4, 5, 1)},
        {"label": "5-3-2", "value": (1, 5, 3, 2)},
        {"label": "5-4-1", "value": (1, 5, 4, 1)}
    ]
    
    formation_selector = mo.ui.dropdown(
        options=formation_options,
        value=(1, 4, 4, 2),  # Default 4-4-2
        label="Select Formation"
    )
    
    mo.vstack([
        mo.md("### Choose Your Formation"),
        formation_selector
    ])
    
    return formation_options, formation_selector


@app.cell
def __(squad_with_xp, formation_selector, mo, pd):
    # Optimize starting 11 based on formation and xP
    def optimize_starting_11(squad_df, formation):
        """Select best starting 11 for given formation"""
        if len(squad_df) == 0:
            return pd.DataFrame(), "No squad data available"
        
        gkp_count, def_count, mid_count, fwd_count = formation
        
        # Group players by position and sort by xP
        by_position = {}
        for position in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_players = squad_df[squad_df['position'] == position].sort_values('xP_gw', ascending=False)
            by_position[position] = pos_players
        
        # Check if we have enough players for formation
        if (len(by_position['GKP']) < gkp_count or
            len(by_position['DEF']) < def_count or
            len(by_position['MID']) < mid_count or
            len(by_position['FWD']) < fwd_count):
            return pd.DataFrame(), "Not enough players for selected formation"
        
        # Select best players for each position
        starting_11 = []
        starting_11.extend(by_position['GKP'].head(gkp_count).to_dict('records'))
        starting_11.extend(by_position['DEF'].head(def_count).to_dict('records'))
        starting_11.extend(by_position['MID'].head(mid_count).to_dict('records'))
        starting_11.extend(by_position['FWD'].head(fwd_count).to_dict('records'))
        
        starting_11_df = pd.DataFrame(starting_11)
        
        # Calculate bench (remaining players)
        starting_ids = set(p['player_id'] for p in starting_11)
        bench = squad_df[~squad_df['player_id'].isin(starting_ids)].sort_values('xP_gw', ascending=False)
        
        return starting_11_df, bench
    
    # Optimize starting 11
    if len(squad_with_xp) > 0 and formation_selector.value:
        starting_11, bench = optimize_starting_11(squad_with_xp, formation_selector.value)
        
        if len(starting_11) > 0:
            formation_str = "-".join(map(str, formation_selector.value[1:]))  # Skip GKP count
            total_xp = starting_11['xP_gw'].sum()
            
            starting_11_display = mo.vstack([
                mo.md(f"### Optimal Starting 11 ({formation_str})"),
                mo.md(f"**Total Expected Points:** {total_xp:.2f}"),
                mo.ui.table(
                    starting_11[['web_name', 'position', 'name', 'price_gbp', 'xP_gw']].round(2),
                    page_size=11
                ),
                mo.md("### Bench (in order of preference)"),
                mo.ui.table(
                    bench[['web_name', 'position', 'name', 'price_gbp', 'xP_gw']].round(2),
                    page_size=4
                )
            ])
        else:
            starting_11_display = mo.md(f"**Error:** {bench}")
    else:
        starting_11_display = mo.md("Select your squad and formation to see optimal starting 11.")
    
    starting_11_display
    
    return starting_11, bench, optimize_starting_11


@app.cell
def __(mo):
    mo.md("## Captain & Vice-Captain Selection")
    return


@app.cell
def __(starting_11, mo):
    # Captain analysis
    if len(starting_11) > 0:
        # Sort by xP for captaincy consideration
        captain_candidates = starting_11.sort_values('xP_gw', ascending=False).head(5)
        
        # Add captaincy multiplier analysis
        captain_analysis = captain_candidates.copy()
        captain_analysis['captain_xP'] = captain_analysis['xP_gw'] * 2  # Double points
        captain_analysis['ceiling_xP'] = captain_analysis['xP_gw'] * 3  # Optimistic scenario
        captain_analysis['floor_xP'] = captain_analysis['xP_gw'] * 0.5  # Pessimistic scenario
        
        # Captain recommendation
        best_captain = captain_analysis.iloc[0]
        
        captain_display = mo.vstack([
            mo.md("### Captain & Vice-Captain Analysis"),
            mo.md(f"""
            **Recommended Captain:** {best_captain['web_name']} ({best_captain['position']}, {best_captain['name']})
            - **Expected Points:** {best_captain['xP_gw']:.2f}
            - **As Captain:** {best_captain['captain_xP']:.2f} points
            - **Ceiling:** {best_captain['ceiling_xP']:.2f} points
            - **Floor:** {best_captain['floor_xP']:.2f} points
            """),
            mo.md("**Top Captain Candidates:**"),
            mo.ui.table(
                captain_analysis[['web_name', 'position', 'name', 'xP_gw', 'captain_xP', 'ceiling_xP', 'floor_xP']].round(2),
                page_size=5
            )
        ])
    else:
        captain_display = mo.md("Select your starting 11 to see captain recommendations.")
    
    captain_display
    return captain_analysis, captain_display


@app.cell
def __(mo):
    mo.md("## Transfer Analysis")
    return


@app.cell
def __(players, teams, squad_data, squad_analysis, gameweek_selector, squad_with_xp, calculate_form_weighted_xp, fixtures, mo, pd):
    # Enhanced transfer analysis section
    def analyze_transfer_options(current_squad, available_budget, target_gw):
        """Analyze potential transfer targets with xP calculations"""
        if current_squad is None or len(current_squad) == 0:
            return pd.DataFrame(), "No current squad data"
        
        # All available players with teams
        all_players = players.merge(teams, on='team_id', how='left')
        
        # Players not in current squad
        current_ids = set(current_squad['player_id'])
        transfer_targets = all_players[~all_players['player_id'].isin(current_ids)].copy()
        
        # Calculate form-weighted xP for transfer targets
        targets_with_xp = calculate_form_weighted_xp(transfer_targets, target_gw, fixtures, teams)
        
        if len(targets_with_xp) == 0:
            return pd.DataFrame(), "No transfer targets with xP data"
        
        # Add value metrics
        targets_with_xp['xP_per_price'] = targets_with_xp['xP_gw'] / targets_with_xp['price_gbp']
        targets_with_xp['form_rating'] = targets_with_xp['selected_by_percentage'] / 10  # Simple proxy
        
        # Sort by xP per price ratio
        targets_with_xp = targets_with_xp.sort_values(['position', 'xP_per_price'], ascending=[True, False])
        
        return targets_with_xp
    
    def analyze_transfer_scenarios(current_squad_xp, transfer_targets, budget_info):
        """Analyze different transfer scenarios"""
        if len(current_squad_xp) == 0 or len(transfer_targets) == 0:
            return []
        
        scenarios = []
        money_itb = budget_info['money_itb']
        free_transfers = budget_info['free_transfers']
        
        # Analyze 1-transfer scenarios (most common)
        for _, current_player in current_squad_xp.iterrows():
            # Find players in same position who could replace them
            same_position_targets = transfer_targets[
                transfer_targets['position'] == current_player['position']
            ].copy()
            
            # Filter by affordability (current player price + money ITB)
            max_price = current_player['price_gbp'] + money_itb
            affordable_targets = same_position_targets[
                same_position_targets['price_gbp'] <= max_price
            ].head(5)  # Top 5 affordable options
            
            for _, target in affordable_targets.iterrows():
                cost = target['price_gbp'] - current_player['price_gbp']
                transfer_cost = 0 if free_transfers > 0 else -4  # Transfer hit
                
                xp_gain = target['xP_gw'] - current_player['xP_gw']
                net_gain = xp_gain + transfer_cost
                
                scenarios.append({
                    'transfer_type': '1 Transfer',
                    'out_player': current_player['web_name'],
                    'in_player': target['web_name'],
                    'position': current_player['position'],
                    'cost': cost,
                    'xp_gain': xp_gain,
                    'transfer_hit': transfer_cost,
                    'net_gain': net_gain,
                    'target_xp': target['xP_gw'],
                    'current_xp': current_player['xP_gw']
                })
        
        # Sort by net gain
        scenarios = sorted(scenarios, key=lambda x: x['net_gain'], reverse=True)
        return scenarios[:20]  # Top 20 transfer options
    
    # Create transfer analysis interface
    if squad_data is not None and squad_analysis:
        available_budget = squad_analysis['money_itb']
        transfer_targets = analyze_transfer_options(squad_data, available_budget, gameweek_selector.value)
        
        # Transfer budget calculation
        money_available = squad_analysis['money_itb']
        free_transfers_available = squad_analysis['free_transfers']
        
        # Analyze transfer scenarios
        if len(squad_with_xp) > 0 and len(transfer_targets) > 0:
            transfer_scenarios = analyze_transfer_scenarios(
                squad_with_xp, transfer_targets, squad_analysis
            )
            
            if transfer_scenarios:
                scenarios_df = pd.DataFrame(transfer_scenarios)
                
                transfer_analysis_display = mo.vstack([
                    mo.md("### Transfer Analysis"),
                    mo.md(f"""
                    **Transfer Budget:**
                    - **Money ITB:** £{money_available:.1f}m
                    - **Free Transfers:** {free_transfers_available}
                    - **Additional Transfers:** -4 points each
                    """),
                    
                    mo.md("**Best Transfer Options (1 Transfer):**"),
                    mo.ui.table(
                        scenarios_df[['out_player', 'in_player', 'position', 'cost', 'xp_gain', 'transfer_hit', 'net_gain']].round(2).head(10),
                        page_size=10
                    ),
                    
                    mo.md("**Top Transfer Targets by Position:**"),
                    mo.ui.table(
                        transfer_targets[['web_name', 'position', 'name', 'price_gbp', 'xP_gw', 'xP_per_price', 'selected_by_percentage']].round(2).head(15),
                        page_size=15
                    )
                ])
            else:
                transfer_analysis_display = mo.md("No beneficial transfer scenarios found.")
        else:
            transfer_analysis_display = mo.md("Calculate squad xP first to see transfer analysis.")
    else:
        transfer_analysis_display = mo.md("Set up your current squad to see transfer analysis.")
    
    transfer_analysis_display
    
    return analyze_transfer_options, transfer_analysis_display, transfer_targets


@app.cell
def __(mo):
    mo.md("## Decision Summary")
    return


@app.cell
def __(starting_11, captain_analysis, gameweek_selector, squad_analysis, mo):
    # Create decision summary for easy implementation
    if len(starting_11) > 0 and len(captain_analysis) > 0:
        # Format starting 11 by position
        starting_11_by_pos = starting_11.groupby('position')['web_name'].apply(list).to_dict()
        
        # Get captain and vice-captain
        captain = captain_analysis.iloc[0]['web_name']
        vice_captain = captain_analysis.iloc[1]['web_name'] if len(captain_analysis) > 1 else "TBD"
        
        decision_summary = mo.md(f"""
        ### 📋 Decision Summary for GW{gameweek_selector.value}
        
        **💰 Budget Status:**
        - Money ITB: £{squad_analysis.get('money_itb', 0):.1f}m
        - Free Transfers: {squad_analysis.get('free_transfers', 0)}
        
        **⚽ Starting 11:**
        - **GKP:** {', '.join(starting_11_by_pos.get('GKP', []))}
        - **DEF:** {', '.join(starting_11_by_pos.get('DEF', []))}
        - **MID:** {', '.join(starting_11_by_pos.get('MID', []))}
        - **FWD:** {', '.join(starting_11_by_pos.get('FWD', []))}
        
        **👑 Captaincy:**
        - **Captain:** {captain}
        - **Vice-Captain:** {vice_captain}
        
        **📊 Expected Performance:**
        - **Total Starting 11 xP:** {starting_11['xP_gw'].sum():.2f}
        - **Captain xP:** {captain_analysis.iloc[0]['captain_xP']:.2f}
        
        **✅ Ready to implement in FPL app!**
        """)
    else:
        decision_summary = mo.md("Complete the analysis above to see your decision summary.")
    
    decision_summary
    return decision_summary


@app.cell
def __(mo):
    mo.md("## 🔍 Retro Analysis")
    return


@app.cell
def __(mo, load_gameweek_datasets):
    # Retro analysis for completed gameweeks
    mo.md("### Analyze Completed Gameweek Performance")
    
    # Gameweek selector for retro analysis
    retro_gw_selector = mo.ui.number(
        value=1,
        start=1,
        stop=38,
        step=1,
        label="Select Completed Gameweek"
    )
    
    # Load live data for the selected gameweek
    def get_retro_analysis_data(gw):
        """Load live data for retro analysis"""
        try:
            _, _, _, _, _, live_data, deltas_data = load_gameweek_datasets(gw)
            return live_data, deltas_data
        except Exception:
            return None, None
    
    mo.vstack([
        retro_gw_selector,
        mo.md("Select a completed gameweek to analyze actual performance vs predictions.")
    ])
    
    return retro_gw_selector, get_retro_analysis_data


@app.cell  
def __(retro_gw_selector, get_retro_analysis_data, players, teams, mo):
    # Retro analysis display
    if retro_gw_selector.value:
        live_data, _ = get_retro_analysis_data(retro_gw_selector.value)
        
        if live_data is not None and len(live_data) > 0:
            # Merge with player names
            retro_data = live_data.merge(
                players[['player_id', 'web_name', 'position']], 
                on='player_id', 
                how='left'
            ).merge(
                teams[['team_id', 'name']], 
                left_on='team_id', 
                right_on='team_id', 
                how='left'
            )
            
            # Get top performers
            top_performers = retro_data.nlargest(10, 'total_points')[
                ['web_name', 'position', 'name', 'total_points', 'goals_scored', 'assists', 'bonus', 'minutes']
            ]
            
            # Get biggest disappointments (high ownership, low points)
            retro_data['ownership'] = retro_data.get('selected_by_percentage', 0)
            disappointments = retro_data[
                (retro_data['ownership'] > 10) & (retro_data['total_points'] < 3)
            ].nlargest(10, 'ownership')[
                ['web_name', 'position', 'name', 'total_points', 'ownership', 'minutes']
            ]
            
            # Get differential successes (low ownership, high points)
            differentials = retro_data[
                (retro_data['ownership'] < 5) & (retro_data['total_points'] > 8)
            ].nlargest(10, 'total_points')[
                ['web_name', 'position', 'name', 'total_points', 'ownership', 'goals_scored', 'assists']
            ]
            
            retro_display = mo.vstack([
                mo.md(f"### 📊 GW{retro_gw_selector.value} Performance Analysis"),
                mo.md("**🏆 Top Performers:**"),
                mo.ui.table(top_performers.round(2)),
                mo.md("**😞 Template Disappointments (High Ownership, Low Points):**"),
                mo.ui.table(disappointments.round(2)) if len(disappointments) > 0 else mo.md("No major disappointments"),
                mo.md("**💎 Differential Successes (Low Ownership, High Points):**"),
                mo.ui.table(differentials.round(2)) if len(differentials) > 0 else mo.md("No major differentials"),
            ])
        else:
            retro_display = mo.md(f"❌ No live data available for GW{retro_gw_selector.value}. Make sure the gameweek is completed and data is available.")
    else:
        retro_display = mo.md("Select a gameweek above to see retro analysis.")
    
    retro_display
    return retro_display


@app.cell
def __(mo):
    mo.md("""
    ## Features Coming Soon
    
    **Enhanced Transfer Analysis:**
    - Multi-transfer strategy optimizer (1-4 transfers)
    - Price change predictions and timing
    - Transfer hit break-even analysis
    
    **Advanced Captain Analysis:**
    - Captaincy differentials vs template
    - Triple captain timing optimization
    - Risk/reward variance analysis
    
    **Chip Strategy:**
    - Bench boost timing
    - Free hit optimization
    - Wildcard planning
    
    **Integration Features:**
    - Import team from FPL API
    - Export decisions to FPL format
    - Historical decision tracking
    """)
    return


if __name__ == "__main__":
    app.run()