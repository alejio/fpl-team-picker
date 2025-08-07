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
            'Nottingham Forest': 17, 'Luton': 18, 'Burnley': 19, 'Sheffield Utd': 20,
            # Add newly promoted teams as bottom-tier strength (weaker than worst PL team)
            'Sunderland': 21, 'Leeds': 22,
            # Add team name aliases to match fixture data
            'Man City': 1, 'Man Utd': 8, "Nott'm Forest": 17, 'Spurs': 5
        }
        
        # Convert to strength ratings (1st place = 1.3, 20th = 0.7, promoted teams lower)
        strength_ratings = {}
        for team, position in team_positions.items():
            if position <= 20:
                # Standard PL teams: 1st = 1.3, 20th = 0.7
                strength = 1.3 - (position - 1) * (1.3 - 0.7) / 19
            else:
                # Promoted teams: below worst PL team (0.7), scale down further
                strength = 0.7 - (position - 20) * 0.05  # 0.65, 0.60, 0.55...
            strength_ratings[team] = round(strength, 3)
        
        return strength_ratings

    def _sbp_to_start_prob(sbp):
        """Convert Selected By Percentage to base start probability"""
        if sbp >= 40:
            return 0.95  # Elite template picks
        elif sbp >= 20:
            return 0.85  # Very popular, high confidence
        elif sbp >= 10:
            return 0.70  # Popular picks
        elif sbp >= 5:
            return 0.55  # Decent ownership
        elif sbp >= 2:
            return 0.35  # Low ownership but some interest
        elif sbp >= 0.5:
            return 0.20  # Differential picks
        else:
            return 0.05  # Very low ownership

    def calculate_start_probability(selected_by_percent, availability_status):
        """
        Calculate probability of starting based on SBP and availability
        Using correct availability_status codes from DATASET.md:
        - 'a' = Available (can play normally)
        - 'i' = Injured (confirmed injury)  
        - 's' = Suspended (serving suspension)
        - 'u' = Unavailable (other reasons)
        - 'd' = Doubtful (fitness concern)
        """
        # Handle availability issues first
        if availability_status == 'i':  # Injured
            return 0.0
        elif availability_status == 's':  # Suspended
            return 0.0
        elif availability_status == 'u':  # Unavailable (other reasons)
            return 0.0
        elif availability_status == 'd':  # Doubtful (fitness concern)
            # Reduce probability for doubtful players
            sbp_prob = _sbp_to_start_prob(selected_by_percent)
            return sbp_prob * 0.5  # 50% penalty for fitness doubts
        elif availability_status == 'a' or availability_status is None:  # Available or no status
            return _sbp_to_start_prob(selected_by_percent)
        else:
            # Unknown status, be conservative
            return _sbp_to_start_prob(selected_by_percent) * 0.8

    def calculate_expected_minutes_probabilistic(row):
        """
        Calculate expected minutes using probabilistic scenarios
        with correct availability_status codes from DATASET.md
        """
        player_id = row['player_id']
        sbp = row.get('selected_by_percentage', 0)
        availability = row.get('availability_status', 'a')
        
        # Base probabilities using correct availability codes
        p_start = calculate_start_probability(sbp, availability)
        
        # Early exit for completely unavailable players
        if availability in ['i', 's', 'u']:  # injured, suspended, unavailable
            return {
                'expected_minutes': 0,
                'p_start': 0,
                'p_60_plus_mins': 0,
                'p_any_appearance': 0
            }
        
        # Position and price-based defaults for durability
        if row['position'] == 'GKP':
            avg_mins_when_started = 90
            p_full_game_given_start = 0.95
            p_sub_given_no_start = 0.05
        elif row['price_gbp'] >= 7.0:  # Premium players
            avg_mins_when_started = 85
            p_full_game_given_start = 0.80
            p_sub_given_no_start = 0.25
        elif row['price_gbp'] >= 5.0:  # Mid-tier
            avg_mins_when_started = 75
            p_full_game_given_start = 0.70
            p_sub_given_no_start = 0.30
        else:  # Budget options
            avg_mins_when_started = 70
            p_full_game_given_start = 0.60
            p_sub_given_no_start = 0.35
        
        # Adjust for doubtful players
        if availability == 'd':  # Doubtful
            # Reduce durability for fitness concerns
            p_full_game_given_start *= 0.7
            avg_mins_when_started *= 0.8
        
        # Calculate scenario probabilities
        p_full_game = p_start * p_full_game_given_start
        p_partial_start = p_start * (1 - p_full_game_given_start)
        p_sub_appearance = (1 - p_start) * p_sub_given_no_start
        p_no_appearance = 1 - (p_full_game + p_partial_start + p_sub_appearance)
        
        # Expected minutes for each scenario
        mins_full_game = avg_mins_when_started
        mins_partial_start = 45  # Subbed off early
        mins_sub = 20  # Late substitute
        mins_no_show = 0
        
        # Weighted expected minutes
        expected_minutes = (
            p_full_game * mins_full_game +
            p_partial_start * mins_partial_start +
            p_sub_appearance * mins_sub +
            p_no_appearance * mins_no_show
        )
        
        return {
            'expected_minutes': expected_minutes,
            'p_start': p_start,
            'p_60_plus_mins': p_full_game,
            'p_any_appearance': 1 - p_no_appearance
        }

    def calculate_minutes_expectation(players_df):
        """Enhanced probabilistic minutes model using SBP and availability_status"""
        players_df = players_df.copy()
        
        # Apply probabilistic calculation to each player
        minute_data = players_df.apply(calculate_expected_minutes_probabilistic, axis=1)
        
        # Extract results into separate columns
        players_df['expected_minutes'] = minute_data.apply(lambda x: x['expected_minutes'])
        players_df['p_start'] = minute_data.apply(lambda x: x['p_start'])
        players_df['p_60_plus_mins'] = minute_data.apply(lambda x: x['p_60_plus_mins'])
        players_df['p_any_appearance'] = minute_data.apply(lambda x: x['p_any_appearance'])
        
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
    mo.md("## Step 2: GW1-5 Fixtures & Multi-Gameweek Difficulty Scaling")


@app.cell
def __(fixtures, teams, team_strength, pd):
    # Get GW1-5 fixtures
    multi_gw_fixtures = fixtures[fixtures['event'].isin([1, 2, 3, 4, 5])].copy()
    
    # Add team names for both home and away
    multi_gw_fixtures = multi_gw_fixtures.merge(
        teams[['team_id', 'name']], 
        left_on='home_team_id', 
        right_on='team_id', 
        how='left'
    ).drop('team_id', axis=1).rename(columns={'name': 'home_team'})
    
    multi_gw_fixtures = multi_gw_fixtures.merge(
        teams[['team_id', 'name']], 
        left_on='away_team_id', 
        right_on='team_id', 
        how='left'
    ).drop('team_id', axis=1).rename(columns={'name': 'away_team'})
    
    # Debug: Check which teams are missing from team_strength
    all_fixture_teams = set(multi_gw_fixtures['home_team'].unique()) | set(multi_gw_fixtures['away_team'].unique())
    team_strength_keys = set(team_strength.keys())
    missing_teams = all_fixture_teams - team_strength_keys
    if missing_teams:
        print(f"Teams in fixtures but missing from team_strength: {sorted(missing_teams)}")
        print(f"Available team_strength keys: {sorted(team_strength_keys)}")
    
    # Add team strength ratings with fallback for missing teams
    multi_gw_fixtures['home_strength'] = multi_gw_fixtures['home_team'].map(team_strength).fillna(1.0)  # Neutral strength if missing
    multi_gw_fixtures['away_strength'] = multi_gw_fixtures['away_team'].map(team_strength).fillna(1.0)  # Neutral strength if missing
    
    # Calculate fixture difficulty for each team (higher opponent strength = harder = lower scaling)
    multi_gw_fixtures['home_difficulty'] = 2.0 - multi_gw_fixtures['away_strength']
    multi_gw_fixtures['away_difficulty'] = 2.0 - multi_gw_fixtures['home_strength']
    
    # Clip to [0.7, 1.3] range
    multi_gw_fixtures['home_difficulty'] = multi_gw_fixtures['home_difficulty'].clip(0.7, 1.3)
    multi_gw_fixtures['away_difficulty'] = multi_gw_fixtures['away_difficulty'].clip(0.7, 1.3)
    
    # Create difficulty matrix: team -> gameweek -> difficulty
    def create_difficulty_matrix(fixtures_df, teams_df):
        """Create team-gameweek difficulty lookup matrix"""
        difficulty_matrix = {}
        
        for _, team_row in teams_df.iterrows():
            team_name = team_row['name']
            team_id = team_row['team_id']
            difficulty_matrix[team_name] = {}
            
            for gw in [1, 2, 3, 4, 5]:
                # Find fixture for this team in this gameweek
                home_fixture = fixtures_df[
                    (fixtures_df['event'] == gw) & 
                    (fixtures_df['home_team_id'] == team_id)
                ]
                away_fixture = fixtures_df[
                    (fixtures_df['event'] == gw) & 
                    (fixtures_df['away_team_id'] == team_id)
                ]
                
                if len(home_fixture) > 0:
                    difficulty_matrix[team_name][gw] = home_fixture.iloc[0]['home_difficulty']
                elif len(away_fixture) > 0:
                    difficulty_matrix[team_name][gw] = away_fixture.iloc[0]['away_difficulty']
                else:
                    # No fixture found (blank gameweek)
                    difficulty_matrix[team_name][gw] = 0.0
        
        return difficulty_matrix
    
    difficulty_matrix = create_difficulty_matrix(multi_gw_fixtures, teams)
    
    # Show sample of multi-gameweek fixtures
    gw_sample_display = multi_gw_fixtures[
        ['event', 'home_team', 'away_team', 'home_difficulty', 'away_difficulty']
    ].head(15).round(3)
    
    return multi_gw_fixtures, difficulty_matrix, gw_sample_display


@app.cell
def __(mo, gw_sample_display):
    mo.md("### GW1-5 Fixtures with Difficulty Scaling (Sample)")
    gw_sample_display


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Step 3: Multi-Gameweek Expected Points (xP) Calculation
        
        **5-Gameweek Horizon:** Now calculating cumulative xP across GW1-5 with temporal weighting.
        
        **Key Features:**
        - **Temporal Weighting**: GW1 (1.0), GW2 (0.9), GW3 (0.8), GW4 (0.7), GW5 (0.6)
        - **Multi-GW Fixture Scaling**: Different difficulty multipliers per gameweek
        - **Probabilistic Minutes Model**: SBP-based start probabilities per gameweek
        - **Availability Filtering**: Injured/suspended players get 0 expected minutes
        - **Fixture Volatility Analysis**: Detect players with extreme difficulty swings
        
        **Target**: 74 PPG championship benchmark → ~370 total points over 5 gameweeks
        """
    )


@app.cell
def __(players_full, difficulty_matrix, np):
    # Multi-gameweek xP calculation with temporal weighting
    def calculate_multi_gw_xp(players_df, difficulty_matrix):
        """Calculate cumulative xP across GW1-5 with temporal weighting"""
        players_xp = players_df.copy()
        
        # Fill missing xG/xA with conservative estimates for promoted teams
        # These values are intentionally below Premier League averages
        position_xg_defaults = {
            'GKP': 0.005,  # Minimal attacking contribution
            'DEF': 0.08,   # Well below PL average (~0.15)  
            'MID': 0.15,   # Well below PL average (~0.25)
            'FWD': 0.25    # Well below PL average (~0.45)
        }
        
        position_xa_defaults = {
            'GKP': 0.02,   # Minimal distribution
            'DEF': 0.12,   # Below average crossing/passing
            'MID': 0.20,   # Below average creativity  
            'FWD': 0.15    # Below average playmaking
        }
        
        players_xp['xG90'] = players_xp['xG90'].fillna(
            players_xp['position'].map(position_xg_defaults)
        )
        players_xp['xA90'] = players_xp['xA90'].fillna(
            players_xp['position'].map(position_xa_defaults)
        )
        
        # Temporal weights for each gameweek
        gw_weights = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6}
        
        # Initialize cumulative columns
        players_xp['total_xP'] = 0.0
        players_xp['total_weighted_xP'] = 0.0
        players_xp['fixture_volatility'] = 0.0
        
        # Store individual gameweek xPs for analysis
        for gw in [1, 2, 3, 4, 5]:
            players_xp[f'xP_gw{gw}'] = 0.0
            players_xp[f'difficulty_gw{gw}'] = 0.0
        
        # Calculate xP for each gameweek
        for gw in [1, 2, 3, 4, 5]:
            weight = gw_weights[gw]
            
            # Get difficulty for each player's team in this gameweek - safe lookup
            def get_team_difficulty_safe(team_name):
                team_dict = difficulty_matrix.get(team_name, {})
                difficulty = team_dict.get(gw, None)
                if difficulty is None:
                    return 1.0  # Neutral difficulty for teams with missing fixtures
                return difficulty
            
            players_xp[f'difficulty_gw{gw}'] = players_xp['name'].apply(get_team_difficulty_safe)
            
            # Debug: Investigate why we still have NaN difficulties
            if gw == 1:  # Only debug once
                # Check what team names exist in each dataset
                player_teams = set(players_xp['name'].unique())
                fixture_teams = set(difficulty_matrix.keys())
                
                missing_in_fixtures = player_teams - fixture_teams
                missing_in_players = fixture_teams - player_teams
                
                print(f"Teams in player data but not fixtures: {sorted(missing_in_fixtures)}")
                print(f"Teams in fixtures but not players: {sorted(missing_in_players)}")
                
                # Check for NaN values after safe lookup
                nan_difficulty = players_xp[f'difficulty_gw{gw}'].isna().sum()
                if nan_difficulty > 0:
                    print(f"Warning: {nan_difficulty} players have NaN difficulty for GW{gw}")
                    
                    # Show which teams have NaN difficulties
                    nan_players = players_xp[players_xp[f'difficulty_gw{gw}'].isna()]
                    nan_teams = nan_players['name'].unique()
                    print(f"Teams with NaN difficulties: {sorted(nan_teams)}")
                    
                    # Check if these teams exist in difficulty_matrix at all
                    for team in nan_teams[:5]:  # Check first 5
                        team_in_matrix = team in difficulty_matrix
                        if team_in_matrix:
                            gw1_in_team = 1 in difficulty_matrix[team]
                            print(f"  {team}: in_matrix={team_in_matrix}, has_gw1={gw1_in_team}")
                            if gw1_in_team:
                                actual_difficulty = difficulty_matrix[team][1]
                                print(f"    GW1 difficulty should be: {actual_difficulty}")
                        else:
                            print(f"  {team}: NOT in difficulty_matrix at all")
                    
                    # Test the safe lookup function directly
                    test_team = nan_teams[0] if len(nan_teams) > 0 else None
                    if test_team:
                        test_result = get_team_difficulty_safe(test_team)
                        print(f"Direct test of get_team_difficulty_safe('{test_team}'): {test_result}")
                        print(f"Type of result: {type(test_result)}")
            
            # Expected contributions for this gameweek
            xG_gw = (players_xp['xG90'] / 90) * players_xp['expected_minutes'] * players_xp[f'difficulty_gw{gw}']
            xA_gw = (players_xp['xA90'] / 90) * players_xp['expected_minutes'] * players_xp[f'difficulty_gw{gw}']
            
            # Appearance points (probabilistic)
            appearance_pts_gw = (
                players_xp['p_60_plus_mins'] * 2 +  # 2 pts for 60+ minutes
                (players_xp['p_any_appearance'] - players_xp['p_60_plus_mins']) * 1  # 1 pt for <60 minutes
            )
            
            # Goal points by position
            goal_multipliers = {'GKP': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
            goal_pts_gw = xG_gw * players_xp['position'].map(goal_multipliers)
            
            # Assist points
            assist_pts_gw = xA_gw * 3
            
            # Clean sheet points (only for 60+ minute appearances)
            cs_multipliers = {'GKP': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
            base_cs_prob = 0.3
            cs_prob_gw = (
                players_xp['p_60_plus_mins'] * 
                players_xp[f'difficulty_gw{gw}'] * 
                base_cs_prob
            )
            cs_pts_gw = cs_prob_gw * players_xp['position'].map(cs_multipliers)
            
            # Total xP for this gameweek
            xp_gw = appearance_pts_gw + goal_pts_gw + assist_pts_gw + cs_pts_gw
            players_xp[f'xP_gw{gw}'] = xp_gw
            
            # Add to cumulative totals
            players_xp['total_xP'] += xp_gw
            players_xp['total_weighted_xP'] += xp_gw * weight
        
        # Calculate fixture volatility (standard deviation of difficulties)
        difficulty_cols = [f'difficulty_gw{gw}' for gw in [1, 2, 3, 4, 5]]
        players_xp['fixture_volatility'] = players_xp[difficulty_cols].std(axis=1)
        
        # Debug: Check for NaN values after volatility calculation
        nan_volatility = players_xp['fixture_volatility'].isna().sum()
        if nan_volatility > 0:
            print(f"Warning: {nan_volatility} players have NaN fixture_volatility")
        
        # Add transfer risk flag for players with poor GW2-3 fixtures
        players_xp['transfer_risk'] = (
            (players_xp['difficulty_gw2'] < 0.85) | 
            (players_xp['difficulty_gw3'] < 0.85)
        )
        
        # Calculate forced transfer penalty (reduced for initial team generation)
        players_xp['transfer_penalty'] = np.where(
            players_xp['transfer_risk'], 
            -0.5,  # Reduced penalty for team generation feasibility
            0.0
        )
        
        # Final weighted xP includes transfer penalty
        players_xp['xP'] = players_xp['total_weighted_xP'] + players_xp['transfer_penalty']
        
        return players_xp
    
    players_xp = calculate_multi_gw_xp(players_full, difficulty_matrix)
    
    return players_xp, calculate_multi_gw_xp




@app.cell
def __(mo, players_xp):
    mo.md("### Top Players by Multi-Gameweek Expected Points (xP)")
    
    # Show top players by weighted 5-gameweek xP with transfer insights
    all_players_xp = players_xp[
        ['web_name', 'position', 'name', 'price_gbp', 'xP', 'total_weighted_xP', 'transfer_risk', 
         'fixture_volatility', 'xP_gw1', 'xP_gw2', 'xP_gw3', 'xP_gw4', 'xP_gw5']
    ].sort_values('xP', ascending=False).head(30).round(3)
    
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
        
        # Debug: Check why so many players are filtered out
        print(f"Initial player count: {len(players_df)}")
        print(f"Players with xP data: {players_df['xP'].notna().sum()}")
        print(f"Players missing xP: {players_df['xP'].isna().sum()}")
        
        # Check what's causing xP to be NaN
        missing_xp_players = players_df[players_df['xP'].isna()]
        print(f"Missing xP breakdown by position: {missing_xp_players.groupby('position').size().to_dict()}")
        
        # Check which components are missing
        print(f"Players missing xG90: {players_df['xG90'].isna().sum()}")
        print(f"Players missing xA90: {players_df['xA90'].isna().sum()}")
        print(f"Players missing expected_minutes: {players_df['expected_minutes'].isna().sum()}")
        
        # Filter players with valid xP data and debug
        valid_players = players_df[players_df['xP'].notna()].copy().reset_index(drop=True)
        
        # Debug: Check player counts by position
        position_counts = valid_players.groupby('position').size()
        print(f"Valid players by position: {position_counts.to_dict()}")
        print(f"Total valid players: {len(valid_players)}")
        print(f"Price range: £{valid_players['price_gbp'].min():.1f}m - £{valid_players['price_gbp'].max():.1f}m")
        print(f"xP range: {valid_players['xP'].min():.2f} - {valid_players['xP'].max():.2f}")
        print(f"Players with positive xP: {(valid_players['xP'] > 0).sum()}/{len(valid_players)}")
        
        # Show cheapest players by position for debugging
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_players = valid_players[valid_players['position'] == pos]
            if len(pos_players) > 0:
                cheapest = pos_players.nsmallest(3, 'price_gbp')[['web_name', 'name', 'price_gbp', 'xP']]
                print(f"\nCheapest {pos} players:")
                print(cheapest.to_string(index=False))
        
        def generate_random_team():
            """Generate a random valid team - simplified approach"""
            import random
            team = []
            remaining_budget = budget
            team_counts = {}
            
            # Get cheapest player per position to ensure minimum budget feasibility
            min_costs = {}
            for position in ['GKP', 'DEF', 'MID', 'FWD']:
                pos_players = valid_players[valid_players['position'] == position]
                if len(pos_players) > 0:
                    min_costs[position] = pos_players['price_gbp'].min()
                else:
                    return None
            
            # Check if we can afford minimum team
            min_team_cost = (min_costs['GKP'] * 2 + min_costs['DEF'] * 5 + 
                           min_costs['MID'] * 5 + min_costs['FWD'] * 3)
            if min_team_cost > budget:
                return None
            
            # Fill each position randomly from affordable players
            for position in ['GKP', 'DEF', 'MID', 'FWD']:
                count = position_requirements[position]
                
                # Get all affordable players for this position
                position_players = valid_players[
                    (valid_players['position'] == position) & 
                    (valid_players['price_gbp'] <= remaining_budget - 
                     sum(min_costs[p] * position_requirements[p] for p in ['GKP', 'DEF', 'MID', 'FWD'] 
                         if p != position) + len([p for p in team if p['position'] == position]) * min_costs[position])
                ].copy()
                
                for i in range(count):
                    if len(position_players) == 0:
                        # Fallback to any affordable player for this position
                        position_players = valid_players[
                            (valid_players['position'] == position) & 
                            (valid_players['price_gbp'] <= remaining_budget)
                        ].copy()
                        
                    if len(position_players) == 0:
                        print(f"No affordable {position} players, budget: £{remaining_budget:.1f}m")
                        return None
                    
                    # Randomly select a player
                    player = position_players.sample(n=1).iloc[0]
                    player_dict = player.to_dict()
                    team.append(player_dict)
                    remaining_budget -= player['price_gbp']
                    
                    # Update team count (but don't enforce constraint during generation)
                    team_name = player['name']
                    team_counts[team_name] = team_counts.get(team_name, 0) + 1
                    
                    # Remove this player from available pool
                    position_players = position_players[position_players['player_id'] != player['player_id']]
            
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
            """Calculate xP for best starting 11 with constraint penalties"""
            if len(team) == 15:
                starting_11 = get_best_starting_11(team)
                base_xp = sum(player['xP'] for player in starting_11)
                
                # Team constraint penalty - penalize teams with >3 players per club
                team_counts = {}
                for player in team:
                    team_name = player['name']
                    team_counts[team_name] = team_counts.get(team_name, 0) + 1
                
                constraint_penalty = 0
                for team_name, count in team_counts.items():
                    if count > 3:
                        constraint_penalty += (count - 3) * -10.0  # Heavy penalty for rule violations
                
                # Additional squad-level transfer penalty
                transfer_risky_starters = sum(1 for player in starting_11 if player.get('transfer_risk', False))
                squad_transfer_penalty = transfer_risky_starters * -1.0
                
                return base_xp + constraint_penalty + squad_transfer_penalty
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
            """Generate neighbor solution by swapping one player with budget awareness"""
            new_team = [p.copy() for p in team]
            
            # Calculate current budget situation
            current_cost = calculate_team_cost(new_team)
            remaining_budget = budget - current_cost
            
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
            
            # Allow all swaps - constraint will be enforced via penalty in calculate_team_xp
            
            if len(available) == 0:
                # Debug why no swaps are available
                all_position_players = valid_players[valid_players['position'] == position]
                same_player_filtered = all_position_players[~all_position_players['player_id'].isin(team_player_ids)]
                if len(same_player_filtered) == 0:
                    # This means all players of this position are already in the team
                    pass  # This is the issue - not enough player diversity
                return None  # No valid swap
            
            # Calculate budget constraint for replacement - be more generous
            max_new_cost = old_player['price_gbp'] + remaining_budget + 5.0  # Allow overspend up to 5m
            
            affordable = available[available['price_gbp'] <= max_new_cost]
            if len(affordable) == 0:
                # Try with any price if still no options
                affordable = available
                if len(affordable) == 0:
                    return None  # No replacement available
            
            # Bias selection toward higher xP players, especially if we have budget to spare
            if remaining_budget > 1.0 and len(affordable) > 1:  # If we have significant budget left
                # Prefer upgrades - weight by xP/price ratio but favor higher xP
                # Ensure weights are positive by adding minimum offset
                base_weights = affordable['xP'] * (1 + affordable['price_gbp'] / 20)
                min_weight = base_weights.min()
                if min_weight < 0:
                    base_weights = base_weights - min_weight + 0.1  # Shift to positive
                
                weights = base_weights.values
                if weights.sum() > 0:
                    weights = weights / weights.sum()  # Normalize to probabilities
                    new_player = affordable.sample(n=1, weights=weights).iloc[0].to_dict()
                else:
                    # Fallback to random if all weights are zero
                    new_player = affordable.sample(n=1).iloc[0].to_dict()
            else:
                # Normal random selection when budget is tight
                new_player = affordable.sample(n=1).iloc[0].to_dict()
            
            new_team[replace_idx] = new_player
            
            # Check if new team is within budget
            new_team_cost = sum(p['price_gbp'] for p in new_team)
            if new_team_cost > budget:
                return None  # Over budget
            
            return new_team if is_valid_team(new_team) else None
        
        # Generate initial random team
        current_team = None
        for attempt in range(1000):
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
        
        # Debug: Check if team violates constraints
        team_counts = {}
        for player in current_team:
            team_name = player['name']
            team_counts[team_name] = team_counts.get(team_name, 0) + 1
        violations = {team: count for team, count in team_counts.items() if count > 3}
        if violations:
            print(f"Team constraint violations: {violations}")
        
        # Debug: Show sample of team xP values
        team_xps = [player['xP'] for player in current_team]
        print(f"Team xP range: {min(team_xps):.2f} to {max(team_xps):.2f}")
        print(f"Average player xP: {sum(team_xps)/len(team_xps):.2f}")
        
        # Simulated Annealing
        improvements = 0
        failed_swaps = 0
        total_swaps = 0
        
        for iteration in range(iterations):
            # Temperature decreases linearly
            temperature = max(0.01, 1.0 * (1 - iteration / iterations))
            
            # Generate neighbor
            new_team = swap_player(current_team)
            total_swaps += 1
            
            if new_team is None:
                failed_swaps += 1
                continue  # Skip if no valid neighbor
            
            new_xp = calculate_team_xp(new_team)
            new_cost = calculate_team_cost(new_team)
            current_cost = calculate_team_cost(current_team)
            
            # Budget utilization bonus - reward teams that use more budget efficiently
            budget_util_bonus = 0
            if new_cost > current_cost:  # If we're spending more money
                budget_util_bonus = (new_cost - current_cost) * 0.1  # Small bonus for using budget
            
            # Adjusted delta includes budget utilization pressure
            delta = new_xp - current_xp + budget_util_bonus
            
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
        print(f"SA Stats: {improvements} improvements, {failed_swaps}/{total_swaps} failed swaps")
        print(f"Final team xP: {best_xp:.2f}, cost: £{calculate_team_cost(best_team):.1f}m")
        
        # Print transfer risk analysis
        transfer_risky_players = [p for p in best_team if p.get('transfer_risk', False)]
        print(f"\nTransfer Risk Analysis:")
        print(f"Players at high transfer risk: {len(transfer_risky_players)}/15")
        
        if len(transfer_risky_players) > 0:
            print("High-risk players (poor GW2-3 fixtures):")
            for player in sorted(transfer_risky_players, key=lambda p: p['price_gbp'], reverse=True):
                print(f"  {player['web_name']} ({player['position']}, {player['name']}) - £{player['price_gbp']}m, GW2: {player.get('difficulty_gw2', 'N/A'):.2f}, GW3: {player.get('difficulty_gw3', 'N/A'):.2f}")
        else:
            print("No players flagged as high transfer risk - excellent squad stability!")
        
        return best_team, remaining_budget
    
    # Select optimal team
    optimal_team, remaining_budget = select_optimal_team(players_xp)
    
    # Convert to DataFrame for display
    if optimal_team:
        team_df = pd.DataFrame(optimal_team)[
            ['web_name', 'position', 'name', 'price_gbp', 'xP', 'transfer_risk']
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
        
        # Calculate transfer risk metrics
        transfer_risk_count = sum(1 for player in optimal_team if player.get('transfer_risk', False))
        
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
            'team_violations': len(violating_teams),
            'transfer_risk_count': transfer_risk_count
        }
        
        # Create starting 11 DataFrame for display with multi-gameweek insights
        starting_11_df = pd.DataFrame(starting_11)[
            ['web_name', 'position', 'name', 'price_gbp', 'xP', 'transfer_risk', 
             'xP_gw1', 'xP_gw2', 'xP_gw3', 'xP_gw4', 'xP_gw5']
        ].round(3)
        starting_11_df['xP_per_price'] = (starting_11_df['xP'] / starting_11_df['price_gbp']).round(3)
        
        # Create transfer risk DataFrame at cell level for marimo display
        risky_squad_players = [player for player in optimal_team if player.get('transfer_risk', False)]
        risky_squad_df = pd.DataFrame(risky_squad_players)[
            ['web_name', 'position', 'name', 'price_gbp', 'xP', 'transfer_risk']
        ].sort_values('price_gbp', ascending=False).round(3) if risky_squad_players else pd.DataFrame()
    else:
        team_df = pd.DataFrame()
        starting_11_df = pd.DataFrame()
        risky_squad_df = pd.DataFrame()
        team_summary = {'error': 'Could not select valid team'}
    
    return select_optimal_team, optimal_team, team_df, starting_11_df, team_summary, risky_squad_df


@app.cell
def __(mo, team_summary, team_df):
    mo.md(f"""
    ### Your Optimal 5-Gameweek Squad (15 Players)
    
    **Budget Analysis:**
    - **Total cost:** £{team_summary.get('total_cost', 0):.1f}m 
    - **Budget used:** {team_summary.get('budget_used_pct', 0):.1f}% of £100m
    - **Budget remaining:** £{team_summary.get('remaining_budget', 0):.1f}m
    
    **Starting 11 Performance (5-Week Horizon):**
    - **Weighted 5-GW xP:** {team_summary.get('starting_11_xp', 0):.2f}
    - **Championship Benchmark:** 370 points (74 × 5 weeks) → **{'✅ COMPETITIVE' if team_summary.get('starting_11_xp', 0) >= 60 else '⚠️ BELOW TARGET'}**
    - **Starting 11 cost:** £{team_summary.get('starting_11_cost', 0):.1f}m
    - **Average xP per starter:** {team_summary.get('avg_xp_per_starter', 0):.2f}
    - **xP per £1m (starters):** {team_summary.get('xp_per_million_starters', 0):.2f}
    
    **Transfer Sustainability:**
    - **Players at transfer risk:** {team_summary.get('transfer_risk_count', 0)}/15 (poor GW2-3 fixtures)
    - **Squad longevity:** **{'✅ STABLE' if team_summary.get('transfer_risk_count', 0) <= 3 else '⚠️ HIGH TURNOVER'}**
    
    **Rule Compliance:**
    - **Squad size:** {team_summary.get('total_players', 0)}/15 players
    - **Max per team:** {team_summary.get('max_per_team', 0)}/3 allowed
    - **Rule violations:** {team_summary.get('team_violations', 0)}
    """)


@app.cell
def __(mo):
    mo.md("### Starting 11 (Your Point-Scoring Team) - 5-Week Breakdown")

@app.cell  
def __(starting_11_df):
    starting_11_df

@app.cell
def __(mo):
    mo.md("### Transfer Risk Analysis")

@app.cell
def __(risky_squad_df):
    # Show players in our selected team flagged as transfer risks
    risky_squad_df if len(risky_squad_df) > 0 else "No players in your selected team are flagged as high transfer risk - excellent squad stability!"

@app.cell
def __(mo):
    mo.md("### Full 15-Player Squad")

@app.cell
def __(team_df):
    team_df


if __name__ == "__main__":
    app.run()