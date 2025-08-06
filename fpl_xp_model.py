import marimo

__generated_with = "0.10.8"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    mo.md(
        r"""
        # FPL Expected Points (xP) Model
        
        **MVP Implementation - Afternoon Build**
        
        Building a fast xP model with smart shortcuts to get optimal GW1 team selection.
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # Data paths
    DATA_DIR = Path("../fpl-dataset-builder/data/")
    return DATA_DIR, Path, np, pd


@app.cell
def __(DATA_DIR, pd):
    # Load core FPL datasets
    def load_datasets():
        """Load core FPL datasets"""
        print("Loading datasets...")
        
        players = pd.read_csv(DATA_DIR / "fpl_players_current.csv")
        xg_rates = pd.read_csv(DATA_DIR / "fpl_player_xg_xa_rates.csv")
        fixtures = pd.read_csv(DATA_DIR / "fpl_fixtures_normalized.csv")
        teams = pd.read_csv(DATA_DIR / "fpl_teams_current.csv")
        
        print(f"Loaded: {len(players)} players, {len(fixtures)} fixtures, {len(teams)} teams")
        print(f"xG rates for {len(xg_rates)} players")
        
        return players, xg_rates, fixtures, teams

    players, xg_rates, fixtures, teams = load_datasets()
    return fixtures, load_datasets, players, teams, xg_rates


@app.cell
def __(np):
    # Team strength ratings and minutes model
    def get_team_strength_ratings():
        """
        Simplified team strength using 2023-24 final table positions
        Returns dict mapping team name to strength factor [0.7, 1.3]
        """
        team_positions = {
            'Manchester City': 1, 'Arsenal': 2, 'Liverpool': 3, 'Aston Villa': 4,
            'Tottenham': 5, 'Chelsea': 6, 'Newcastle': 7, 'Manchester Utd': 8,
            'West Ham': 9, 'Crystal Palace': 10, 'Brighton': 11, 'Bournemouth': 12,
            'Fulham': 13, 'Wolves': 14, 'Everton': 15, 'Brentford': 16,
            'Nottingham Forest': 17, 'Luton': 18, 'Burnley': 19, 'Sheffield Utd': 20
        }
        
        # Convert to strength ratings (1st place = 1.3, 20th = 0.7)
        strength_ratings = {}
        for team, position in team_positions.items():
            strength = 1.3 - (position - 1) * (1.3 - 0.7) / 19
            strength_ratings[team] = round(strength, 3)
        
        return strength_ratings

    def calculate_minutes_expectation(players_df):
        """MVP minutes model: >£5.0m = 75 mins, ≤£5.0m = 30 mins"""
        players_df = players_df.copy()
        players_df['expected_minutes'] = np.where(
            players_df['price_gbp'] > 5.0, 75, 30
        )
        return players_df
    
    team_strength = get_team_strength_ratings()
    return get_team_strength_ratings, team_strength, calculate_minutes_expectation


@app.cell
def __(mo):
    mo.md("## Step 1: Complete Player Dataset")


@app.cell
def __(players, xg_rates, teams, calculate_minutes_expectation):
    # Add minutes expectation to players
    players_with_minutes = calculate_minutes_expectation(players)
    
    # Merge with teams to get team names
    players_teams = players_with_minutes.merge(teams, on='team_id', how='left')
    
    # Join with xG rates using mapped_player_id for referential integrity
    players_full = players_teams.merge(
        xg_rates[['mapped_player_id', 'player', 'team', 'xG90', 'xA90', 'minutes']], 
        left_on='player_id',
        right_on='mapped_player_id',
        how='left',
        suffixes=('', '_xg')
    )
    
    # Check merge success
    merge_stats = {
        'Total players': len(players_full),
        'Players with xG data': players_full['xG90'].notna().sum(),
        'Players missing xG data': players_full['xG90'].isna().sum()
    }
    
    return players_full, merge_stats


@app.cell
def __(mo, merge_stats, players_full):
    mo.md(f"""
    ### Complete Dataset
    - **Total players:** {merge_stats['Total players']}
    - **Players with xG data:** {merge_stats['Players with xG data']}  
    - **Players missing xG data:** {merge_stats['Players missing xG data']}
    """)
    
    # Show complete dataset with key columns
    complete_dataset = players_full[[
        'web_name', 'position', 'name', 'price_gbp', 'expected_minutes', 'xG90', 'xA90'
    ]].fillna(0).sort_values(['position', 'price_gbp'], ascending=[True, False])
    
    complete_dataset


@app.cell
def __(mo):
    mo.md("## Step 2: GW1 Fixtures & Difficulty Scaling")


@app.cell
def __(fixtures, teams, team_strength):
    # Get GW1 fixtures (event = 1)
    gw1_fixtures = fixtures[fixtures['event'] == 1].copy()
    
    # Add team names
    gw1_fixtures = gw1_fixtures.merge(
        teams[['team_id', 'name']], 
        left_on='home_team_id', 
        right_on='team_id', 
        how='left'
    ).rename(columns={'name': 'home_team'})
    
    gw1_fixtures = gw1_fixtures.merge(
        teams[['team_id', 'name']], 
        left_on='away_team_id', 
        right_on='team_id', 
        how='left'
    ).rename(columns={'name': 'away_team'})
    
    # Add team strength ratings
    gw1_fixtures['home_strength'] = gw1_fixtures['home_team'].map(team_strength)
    gw1_fixtures['away_strength'] = gw1_fixtures['away_team'].map(team_strength)
    
    # Calculate fixture difficulty for each team
    # Higher opponent strength = harder fixture = lower scaling
    gw1_fixtures['home_difficulty'] = 2.0 - gw1_fixtures['away_strength']  # Inverted
    gw1_fixtures['away_difficulty'] = 2.0 - gw1_fixtures['home_strength']  # Inverted
    
    # Clip to [0.7, 1.3] range
    gw1_fixtures['home_difficulty'] = gw1_fixtures['home_difficulty'].clip(0.7, 1.3)
    gw1_fixtures['away_difficulty'] = gw1_fixtures['away_difficulty'].clip(0.7, 1.3)
    
    # Show GW1 fixtures
    gw1_display = gw1_fixtures[['home_team', 'away_team', 'home_strength', 'away_strength', 'home_difficulty', 'away_difficulty']].round(3)
    
    return gw1_fixtures, gw1_display


@app.cell
def __(mo, gw1_display):
    mo.md("### GW1 Fixtures with Difficulty Scaling")
    gw1_display


@app.cell
def __(mo):
    mo.md("## Step 3: Expected Points (xP) Calculation")


@app.cell
def __(players_full, gw1_fixtures, np):
    # Create difficulty lookup for each team
    difficulty_lookup = {}
    
    for _, row in gw1_fixtures.iterrows():
        difficulty_lookup[row['home_team']] = row['home_difficulty'] 
        difficulty_lookup[row['away_team']] = row['away_difficulty']
    
    # Apply difficulty scaling to players
    players_xp = players_full.copy()
    
    # Map team names to difficulty (fill missing with 1.0)
    players_xp['difficulty_scale'] = players_xp['name'].map(difficulty_lookup).fillna(1.0)
    
    # Fill missing xG/xA with 0 for players without data
    players_xp['xG90'] = players_xp['xG90'].fillna(0)
    players_xp['xA90'] = players_xp['xA90'].fillna(0)
    
    # Calculate expected contributions
    players_xp['xG_exp'] = (players_xp['xG90'] / 90) * players_xp['expected_minutes'] * players_xp['difficulty_scale']
    players_xp['xA_exp'] = (players_xp['xA90'] / 90) * players_xp['expected_minutes'] * players_xp['difficulty_scale']
    
    # FPL Points conversion
    # Appearance points
    players_xp['appearance_pts'] = np.where(players_xp['expected_minutes'] >= 60, 2, 
                                  np.where(players_xp['expected_minutes'] > 0, 1, 0))
    
    # Goal points by position
    goal_multipliers = {'GKP': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
    players_xp['goal_multiplier'] = players_xp['position'].map(goal_multipliers)
    players_xp['goal_pts'] = players_xp['xG_exp'] * players_xp['goal_multiplier']
    
    # Assist points (always 3)
    players_xp['assist_pts'] = players_xp['xA_exp'] * 3
    
    # Clean sheet points (simplified - just use difficulty)
    cs_multipliers = {'GKP': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
    players_xp['cs_multiplier'] = players_xp['position'].map(cs_multipliers)
    # Simple CS probability based on team strength vs opponent
    players_xp['cs_prob'] = np.where(players_xp['expected_minutes'] >= 60, 
                                    players_xp['difficulty_scale'] * 0.3,  # Scale base 30% CS chance
                                    0)
    players_xp['cs_pts'] = players_xp['cs_prob'] * players_xp['cs_multiplier']
    
    # Total xP
    players_xp['xP'] = players_xp['appearance_pts'] + players_xp['goal_pts'] + players_xp['assist_pts'] + players_xp['cs_pts']
    
    return players_xp, difficulty_lookup




@app.cell
def __(mo, players_xp):
    mo.md("### All Players by Expected Points (xP)")
    
    # Show ALL players by xP, sorted descending
    all_players_xp = players_xp[
        ['web_name', 'position', 'name', 'price_gbp', 'xP', 'xG_exp', 'xA_exp', 'difficulty_scale']
    ].sort_values('xP', ascending=False).round(3)
    
    all_players_xp


@app.cell
def __(mo):
    mo.md("## Step 4: Optimal Team Selection")


@app.cell
def __(players_xp, pd, np):
    # Simulated Annealing team optimization
    def select_optimal_team(players_df, budget=100.0, formation=(2, 5, 5, 3), iterations=5000):
        """
        Select optimal 15-player squad using Simulated Annealing optimization
        Formation: (2 GKP, 5 DEF, 5 MID, 3 FWD) = 15 players
        Constraints: £100m budget, max 3 players per team
        """
        import random
        import math
        
        gkp_count, def_count, mid_count, fwd_count = formation
        position_requirements = {'GKP': gkp_count, 'DEF': def_count, 'MID': mid_count, 'FWD': fwd_count}
        
        # Filter players with valid xP data
        valid_players = players_df[players_df['xP'].notna()].copy().reset_index(drop=True)
        
        def generate_random_team():
            """Generate a random valid team respecting formation and budget constraints"""
            team = []
            remaining_budget = budget
            
            for position, count in position_requirements.items():
                position_players = valid_players[valid_players['position'] == position].copy()
                
                # Fallback: pick cheapest available players to ensure valid team
                available = position_players[position_players['price_gbp'] <= remaining_budget].sort_values('price_gbp')
                needed = count
                
                for _, player in available.head(needed * 3).iterrows():  # Try more players
                    if needed > 0 and player['price_gbp'] <= remaining_budget:
                        team.append(player.to_dict())
                        remaining_budget -= player['price_gbp']
                        needed -= 1
            
            return team if len(team) == 15 else None
        
        def get_best_starting_11(squad):
            """
            Find the best starting 11 from a 15-player squad
            Must satisfy formation constraints: 1 GKP, 3-5 DEF, 3-5 MID, 1-3 FWD
            """
            if len(squad) != 15:
                return []
            
            # Group players by position
            by_position = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
            for player in squad:
                by_position[player['position']].append(player)
            
            # Sort each position by xP (descending)
            for pos in by_position:
                by_position[pos].sort(key=lambda p: p['xP'], reverse=True)
            
            # Try different valid formations and pick the best
            valid_formations = [
                (1, 3, 5, 2), (1, 3, 4, 3), (1, 4, 5, 1),
                (1, 4, 4, 2), (1, 4, 3, 3), (1, 5, 4, 1),
                (1, 5, 3, 2), (1, 5, 2, 3)
            ]
            
            best_11 = []
            best_xp = 0
            
            for gkp, def_count, mid, fwd in valid_formations:
                if (gkp <= len(by_position['GKP']) and 
                    def_count <= len(by_position['DEF']) and
                    mid <= len(by_position['MID']) and
                    fwd <= len(by_position['FWD'])):
                    
                    formation_11 = (
                        by_position['GKP'][:gkp] +
                        by_position['DEF'][:def_count] +
                        by_position['MID'][:mid] +
                        by_position['FWD'][:fwd]
                    )
                    
                    formation_xp = sum(p['xP'] for p in formation_11)
                    
                    if formation_xp > best_xp:
                        best_xp = formation_xp
                        best_11 = formation_11
            
            return best_11
        
        def calculate_team_xp(team):
            """Calculate xP for best starting 11 from the 15-player squad"""
            if len(team) == 15:
                starting_11 = get_best_starting_11(team)
                return sum(player['xP'] for player in starting_11)
            else:
                # Fallback for partial teams during generation
                return sum(player['xP'] for player in team)
        
        def calculate_team_cost(team):
            """Calculate total cost for a team"""
            return sum(player['price_gbp'] for player in team)
        
        def is_valid_team(team):
            """Check if team satisfies formation, budget, and team limit constraints"""
            if len(team) != 15:
                return False
            
            if calculate_team_cost(team) > budget:
                return False
            
            # Check formation requirements
            position_counts = {}
            for player in team:
                pos = player['position']
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            for pos, required in position_requirements.items():
                if position_counts.get(pos, 0) != required:
                    return False
            
            # Check max 3 players per team constraint
            team_counts = {}
            for player in team:
                team_name = player['name']  # Team name
                team_counts[team_name] = team_counts.get(team_name, 0) + 1
                if team_counts[team_name] > 3:
                    return False
            
            return True
        
        def swap_player(team):
            """Generate neighbor solution by swapping one player"""
            new_team = [p.copy() for p in team]
            
            # Pick random player to replace
            replace_idx = random.randint(0, 14)
            old_player = new_team[replace_idx]
            position = old_player['position']
            
            # Get available players for this position (not already in team)
            team_player_ids = {p['player_id'] for p in new_team}
            available = valid_players[
                (valid_players['position'] == position) & 
                (~valid_players['player_id'].isin(team_player_ids))
            ]
            
            if len(available) == 0:
                return None  # No valid swap
            
            # Calculate budget constraint for replacement
            current_cost = calculate_team_cost(new_team)
            max_new_cost = budget - current_cost + old_player['price_gbp']
            
            affordable = available[available['price_gbp'] <= max_new_cost]
            if len(affordable) == 0:
                return None  # No affordable replacement
            
            # Pick random replacement
            new_player = affordable.sample(n=1).iloc[0].to_dict()
            new_team[replace_idx] = new_player
            
            return new_team if is_valid_team(new_team) else None
        
        # Generate initial random team
        current_team = None
        for attempt in range(1000):  # Try up to 1000 times to generate valid initial team
            current_team = generate_random_team()
            if current_team and is_valid_team(current_team):
                break
        
        if not current_team:
            print("Could not generate valid initial team")
            return [], budget
        
        best_team = [p.copy() for p in current_team]
        current_xp = calculate_team_xp(current_team)
        best_xp = current_xp
        
        print(f"Initial team xP: {current_xp:.2f}, cost: £{calculate_team_cost(current_team):.1f}m")
        
        # Simulated Annealing
        improvements = 0
        for iteration in range(iterations):
            # Temperature decreases linearly
            temperature = max(0.01, 1.0 * (1 - iteration / iterations))
            
            # Generate neighbor
            new_team = swap_player(current_team)
            
            if new_team is None:
                continue  # Skip if no valid neighbor
            
            new_xp = calculate_team_xp(new_team)
            delta = new_xp - current_xp
            
            # Accept if better, or with probability based on temperature
            if delta > 0 or (temperature > 0 and random.random() < math.exp(delta / temperature)):
                current_team = new_team
                current_xp = new_xp
                
                # Update best if this is the best we've seen
                if current_xp > best_xp:
                    best_team = [p.copy() for p in current_team]
                    best_xp = current_xp
                    improvements += 1
                    if improvements % 5 == 0:
                        print(f"Iteration {iteration}: New best xP: {best_xp:.2f}")
        
        remaining_budget = budget - calculate_team_cost(best_team)
        print(f"Final team xP: {best_xp:.2f}, cost: £{calculate_team_cost(best_team):.1f}m")
        return best_team, remaining_budget
    
    # Select optimal team
    optimal_team, remaining_budget = select_optimal_team(players_xp)
    
    # Convert to DataFrame for display
    if optimal_team:
        team_df = pd.DataFrame(optimal_team)[
            ['web_name', 'position', 'name', 'price_gbp', 'xP']
        ].round(3)
        
        # Add budget efficiency column
        team_df['xP_per_price'] = (team_df['xP'] / team_df['price_gbp']).round(3)
        
        total_cost = team_df['price_gbp'].sum()
        total_squad_xp = team_df['xP'].sum()
        
        # Get best starting 11 and their stats
        starting_11 = select_optimal_team.__code__.co_consts  # Get function reference
        def get_best_11_from_squad(squad_list):
            by_position = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
            for player in squad_list:
                by_position[player['position']].append(player)
            
            for pos in by_position:
                by_position[pos].sort(key=lambda p: p['xP'], reverse=True)
            
            valid_formations = [
                (1, 3, 5, 2), (1, 3, 4, 3), (1, 4, 5, 1),
                (1, 4, 4, 2), (1, 4, 3, 3), (1, 5, 4, 1),
                (1, 5, 3, 2), (1, 5, 2, 3)
            ]
            
            best_11 = []
            best_xp = 0
            
            for gkp, def_count, mid, fwd in valid_formations:
                if (gkp <= len(by_position['GKP']) and 
                    def_count <= len(by_position['DEF']) and
                    mid <= len(by_position['MID']) and
                    fwd <= len(by_position['FWD'])):
                    
                    formation_11 = (
                        by_position['GKP'][:gkp] +
                        by_position['DEF'][:def_count] +
                        by_position['MID'][:mid] +
                        by_position['FWD'][:fwd]
                    )
                    
                    formation_xp = sum(p['xP'] for p in formation_11)
                    
                    if formation_xp > best_xp:
                        best_xp = formation_xp
                        best_11 = formation_11
            
            return best_11
        
        starting_11 = get_best_11_from_squad(optimal_team)
        starting_11_xp = sum(p['xP'] for p in starting_11)
        starting_11_cost = sum(p['price_gbp'] for p in starting_11)
        
        # Check team constraint compliance
        team_counts = team_df['name'].value_counts()
        max_per_team = team_counts.max()
        violating_teams = team_counts[team_counts > 3]
        
        team_summary = {
            'total_players': len(team_df),
            'total_cost': total_cost,
            'remaining_budget': remaining_budget,
            'budget_used_pct': (total_cost / 100.0) * 100,
            'total_squad_xp': total_squad_xp,
            'starting_11_xp': starting_11_xp,
            'starting_11_cost': starting_11_cost,
            'bench_cost': total_cost - starting_11_cost,
            'avg_xp_per_starter': starting_11_xp / 11,
            'xp_per_million_starters': starting_11_xp / starting_11_cost,
            'max_per_team': max_per_team,
            'team_violations': len(violating_teams)
        }
        
        # Create starting 11 DataFrame for display
        starting_11_df = pd.DataFrame(starting_11)[
            ['web_name', 'position', 'name', 'price_gbp', 'xP']
        ].round(3)
        starting_11_df['xP_per_price'] = (starting_11_df['xP'] / starting_11_df['price_gbp']).round(3)
    else:
        team_df = pd.DataFrame()
        starting_11_df = pd.DataFrame()
        team_summary = {'error': 'Could not select valid team'}
    
    return select_optimal_team, optimal_team, team_df, starting_11_df, team_summary


@app.cell
def __(mo, team_summary, team_df):
    mo.md(f"""
    ### Your Optimal GW1 Squad (15 Players)
    
    **Budget Analysis:**
    - **Total cost:** £{team_summary.get('total_cost', 0):.1f}m 
    - **Budget used:** {team_summary.get('budget_used_pct', 0):.1f}% of £100m
    - **Budget remaining:** £{team_summary.get('remaining_budget', 0):.1f}m
    
    **Starting 11 Performance (What Actually Scores Points):**
    - **Starting 11 xP:** {team_summary.get('starting_11_xp', 0):.2f}
    - **Starting 11 cost:** £{team_summary.get('starting_11_cost', 0):.1f}m
    - **Bench cost:** £{team_summary.get('bench_cost', 0):.1f}m (4 players)
    - **Average xP per starter:** {team_summary.get('avg_xp_per_starter', 0):.2f}
    - **xP per £1m (starters):** {team_summary.get('xp_per_million_starters', 0):.2f}
    
    **Rule Compliance:**
    - **Squad size:** {team_summary.get('total_players', 0)}/15 players
    - **Max per team:** {team_summary.get('max_per_team', 0)}/3 allowed
    - **Rule violations:** {team_summary.get('team_violations', 0)}
    """)


@app.cell
def __(mo):
    mo.md("### Starting 11 (Your Point-Scoring Team)")

@app.cell  
def __(starting_11_df):
    starting_11_df

@app.cell
def __(mo):
    mo.md("### Full 15-Player Squad")

@app.cell
def __(team_df):
    team_df


if __name__ == "__main__":
    app.run()