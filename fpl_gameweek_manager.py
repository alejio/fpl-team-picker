import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # FPL Gameweek Manager (Database-Driven)

        **Weekly Decision Making Tool**
        
        1. Load your team from database
        2. Identify needed transfers
        3. Pick optimal lineup 
        4. Select captain and vice-captain
        
        *Now fully database-driven with no external API dependencies!*
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import warnings
    warnings.filterwarnings('ignore')
    
    return pd, np, datetime, warnings


@app.cell
def __(mo):
    mo.md("## 1. Load Team from Database")
    return


@app.cell
def __(pd):
    def fetch_fpl_data():
        """Fetch FPL data from database with live gameweek stats"""
        from client import get_current_players, get_current_teams, get_gameweek_live_data
        
        print("🔄 Loading FPL data from database...")
        
        # Get base data from database
        players_base = get_current_players()
        teams_df = get_current_teams()
        live_data = get_gameweek_live_data(1)  # Current gameweek
        
        # Merge players with live gameweek stats
        players = players_base.merge(
            live_data, 
            on='player_id', 
            how='left'
        )
        
        # Standardize column names for compatibility
        players = players.rename(columns={
            'price_gbp': 'price',
            'team_id': 'team', 
            'selected_by_percentage': 'selected_by_percent',
            'availability_status': 'status'
        })
        
        # Add calculated fields
        players['points_per_game'] = players['total_points'] / players['event'].fillna(1)
        players['form'] = players['total_points'].fillna(0).astype(float)  # Simplified form
        
        teams = teams_df.rename(columns={'team_id': 'id'})
        
        # Create events data
        events = pd.DataFrame([{"id": 1, "is_current": True}])
        current_gw = 1
        
        print(f"✅ Loaded {len(players)} players, {len(teams)} teams from database")
        print(f"📅 Current GW: {current_gw}")
        
        return players, teams, events, current_gw

    def fetch_manager_team(manager_id):
        """Fetch manager's current team from database"""
        print(f"🔄 Fetching team for manager {manager_id} from database...")
        
        try:
            from client import FPLDataClient
            client = FPLDataClient()
            
            # Get manager data
            manager_data = client.get_my_manager_data()
            if manager_data.empty:
                print("❌ No manager data found in database")
                return None
            
            # Get current picks
            current_picks = client.get_my_current_picks()
            if current_picks.empty:
                print("❌ No current picks found in database")
                return None
            
            # Convert picks to the expected format
            picks = []
            for _, pick in current_picks.iterrows():
                picks.append({
                    'element': pick['player_id'],
                    'is_captain': pick.get('is_captain', False),
                    'is_vice_captain': pick.get('is_vice_captain', False),
                    'multiplier': pick.get('multiplier', 1)
                })
            
            # Get manager info from first row
            manager_info = manager_data.iloc[0]
            
            team_info = {
                "manager_id": manager_info.get('manager_id', manager_id),
                "entry_name": manager_info.get('entry_name', 'Unknown Team'),
                "total_points": manager_info.get('summary_overall_points', 0),
                "bank": manager_info.get('bank', 0),
                "team_value": manager_info.get('value', 100.0),
                "picks": picks
            }
            
            print(f"✅ Loaded team from database: {team_info['entry_name']}")
            return team_info
            
        except Exception as e:
            print(f"❌ Error fetching team from database: {e}")
            print("💡 Make sure manager data is available in the database")
            return None
    
    # Load data
    players, teams, events, current_gw = fetch_fpl_data()
    
    return players, teams, events, current_gw, fetch_manager_team


@app.cell
def __(mo):
    manager_id_input = mo.ui.number(
        value=4233026,
        label="Manager ID"
    )
    
    mo.vstack([
        mo.md("**Enter your FPL Manager ID:**"),
        mo.md("_Find your manager ID in the URL when logged into FPL website_"),
        manager_id_input
    ])
    
    return (manager_id_input,)


@app.cell
def __(fetch_manager_team, manager_id_input, players, teams, mo, pd):
    # Load team when manager ID is provided
    if manager_id_input.value:
        team_data = fetch_manager_team(manager_id_input.value)
        
        if team_data:
            # Get player details
            player_ids = [pick['element'] for pick in team_data['picks']]
            current_squad = players[players['player_id'].isin(player_ids)].copy()
            current_squad = current_squad.merge(teams[['id', 'name']], left_on='team', right_on='id', how='left')
            
            # Add captain info
            captain_id = next((pick['element'] for pick in team_data['picks'] if pick['is_captain']), None)
            vice_captain_id = next((pick['element'] for pick in team_data['picks'] if pick['is_vice_captain']), None)
            
            current_squad['role'] = current_squad['player_id'].apply(
                lambda x: '(C)' if x == captain_id else '(VC)' if x == vice_captain_id else ''
            )
            
            # Add more stats to display
            display_cols = [
                'web_name', 'position', 'name', 'price', 'role',
                'total_points', 'points_per_game', 'form', 'selected_by_percent',
                'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
                'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 
                'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat',
                'ict_index', 'expected_goals', 'expected_assists', 'expected_goal_involvements',
                'expected_goals_conceded', 'minutes', 'status'
            ]
            
            # Only include columns that exist in the dataframe
            available_cols = [col for col in display_cols if col in current_squad.columns]
            
            team_display = mo.vstack([
                mo.md(f"### ✅ {team_data['entry_name']}"),
                mo.md(f"**Points:** {team_data['total_points']:,} | **Bank:** £{team_data['bank']:.1f}m | **Value:** £{team_data['team_value']:.1f}m"),
                mo.md("**Current Squad (All Available Stats):**"),
                mo.ui.table(
                    current_squad[available_cols].round(2),
                    page_size=15
                ),
                mo.md(f"**Available columns:** {', '.join(available_cols)}")
            ])
        else:
            team_display = mo.md("❌ Could not load team data")
            current_squad = pd.DataFrame()
            team_data = None
    else:
        team_display = mo.md("Enter manager ID above to load team")
        current_squad = pd.DataFrame()
        team_data = None
    
    team_display
    
    return current_squad, team_data


@app.cell
def __(mo):
    mo.md("## 2. Transfer Analysis")
    return


@app.cell
def __(current_squad, team_data, mo, pd):
    def analyze_transfers():
        """Analyze if transfers are needed"""
        if len(current_squad) == 0:
            return mo.md("Load your team first")
        
        # Simple transfer analysis based on form and fixtures
        problems = []
        
        # Check for injured/suspended players
        injured = current_squad[current_squad['status'].isin(['i', 's', 'd'])]
        for _, player in injured.iterrows():
            problems.append({
                'player': player['web_name'],
                'issue': f"Status: {player['status']}",
                'priority': 'High' if player['status'] in ['i', 's'] else 'Medium'
            })
        
        # Check for poor form (form < 2.0)
        try:
            poor_form = current_squad[current_squad['form'].astype(float) < 2.0]
            for _, player in poor_form.iterrows():
                problems.append({
                    'player': player['web_name'],
                    'issue': f"Poor form: {player['form']}",
                    'priority': 'Low'
                })
        except:
            pass
        
        # Transfer recommendation
        high_priority = [p for p in problems if p['priority'] == 'High']
        medium_priority = [p for p in problems if p['priority'] == 'Medium']
        
        if len(high_priority) > 0:
            recommendation = f"Make {min(len(high_priority), 2)} transfers"
            reason = "Injured/suspended players need immediate replacement"
        elif len(medium_priority) > 0:
            recommendation = "Consider 1 transfer"
            reason = "Some players have concerns but not urgent"
        else:
            recommendation = "No transfers needed"
            reason = "Squad looks good for this gameweek"
        
        return mo.vstack([
            mo.md(f"### {recommendation}"),
            mo.md(f"**Reason:** {reason}"),
            mo.md("**Issues found:**"),
            mo.ui.table(pd.DataFrame(problems)) if problems else mo.md("✅ No issues found"),
            mo.md(f"**Available budget:** £{team_data['bank']:.1f}m")
        ])
    
    transfer_analysis = analyze_transfers() if len(current_squad) > 0 else mo.md("Load team first")
    transfer_analysis
    
    return (transfer_analysis,)


@app.cell
def __(mo):
    mo.md("## 3. Optimal Lineup")
    return


@app.cell
def __(current_squad, mo, pd):
    def select_best_11():
        """Select best starting 11 based on simple metrics"""
        if len(current_squad) == 0:
            return mo.md("Load your team first"), pd.DataFrame()
        
        # Simple scoring: points per game + form
        squad = current_squad.copy()
        try:
            squad['score'] = (
                squad['points_per_game'].astype(float) * 2 +
                squad['form'].astype(float) +
                squad['selected_by_percent'].astype(float) / 100
            )
        except:
            # Fallback to just price if other metrics fail
            squad['score'] = squad['price']
        
        # Select by position
        gkp = squad[squad['position'] == 'GKP'].nlargest(1, 'score')
        defenders = squad[squad['position'] == 'DEF'].nlargest(5, 'score')
        midfielders = squad[squad['position'] == 'MID'].nlargest(5, 'score')
        forwards = squad[squad['position'] == 'FWD'].nlargest(3, 'score')
        
        # Formation options - pick best 11 from available players
        formations = [
            {'name': '3-4-3', 'def': 3, 'mid': 4, 'fwd': 3},
            {'name': '3-5-2', 'def': 3, 'mid': 5, 'fwd': 2},
            {'name': '4-3-3', 'def': 4, 'mid': 3, 'fwd': 3},
            {'name': '4-4-2', 'def': 4, 'mid': 4, 'fwd': 2},
            {'name': '4-5-1', 'def': 4, 'mid': 5, 'fwd': 1},
        ]
        
        best_formation = None
        best_score = 0
        best_11 = pd.DataFrame()
        
        for formation in formations:
            try:
                lineup = pd.concat([
                    gkp,
                    defenders.head(formation['def']),
                    midfielders.head(formation['mid']),
                    forwards.head(formation['fwd'])
                ])
                
                if len(lineup) == 11:
                    total_score = lineup['score'].sum()
                    if total_score > best_score:
                        best_score = total_score
                        best_formation = formation
                        best_11 = lineup
            except:
                continue
        
        if len(best_11) > 0:
            # Bench (remaining players)
            bench = squad[~squad['player_id'].isin(best_11['player_id'])].nlargest(4, 'score')
            
            # Enhanced display for starting 11
            starting_11_cols = [
                'web_name', 'position', 'name', 'price', 'score',
                'total_points', 'points_per_game', 'form', 'selected_by_percent',
                'goals_scored', 'assists', 'minutes', 'status'
            ]
            available_starting_cols = [col for col in starting_11_cols if col in best_11.columns]
            
            bench_cols = [
                'web_name', 'position', 'name', 'price',
                'total_points', 'points_per_game', 'form', 'minutes'
            ]
            available_bench_cols = [col for col in bench_cols if col in bench.columns]
            
            return mo.vstack([
                mo.md(f"### ⚽ Best Formation: {best_formation['name']}"),
                mo.md("**Starting 11:**"),
                mo.ui.table(
                    best_11[available_starting_cols].round(2),
                    page_size=11
                ),
                mo.md("**Bench:**"),
                mo.ui.table(
                    bench[available_bench_cols].round(2),
                    page_size=4
                )
            ]), best_11
        else:
            return mo.md("❌ Could not create valid formation"), pd.DataFrame()
    
    lineup_display, starting_11 = select_best_11()
    lineup_display
    
    return starting_11, lineup_display


@app.cell
def __(mo):
    mo.md("## 4. Captain Selection")
    return


@app.cell
def __(starting_11, mo):
    def select_captain():
        """Select captain and vice-captain"""
        if len(starting_11) == 0:
            return mo.md("Select your starting 11 first")
        
        # Captain selection based on expected performance
        candidates = starting_11.copy()
        try:
            candidates['captain_score'] = (
                candidates['points_per_game'].astype(float) * 3 +
                candidates['form'].astype(float) * 2 +
                candidates['price'] * 0.5
            )
        except:
            candidates['captain_score'] = candidates['price']
        
        # Top 3 captain options
        captain_options = candidates.nlargest(3, 'captain_score')
        
        captain = captain_options.iloc[0]
        vice_captain = captain_options.iloc[1] if len(captain_options) > 1 else captain
        
        # Enhanced captain options display
        captain_display_cols = [
            'web_name', 'position', 'name', 'price', 'captain_score',
            'total_points', 'points_per_game', 'form', 'selected_by_percent',
            'goals_scored', 'assists', 'expected_goals', 'expected_assists'
        ]
        available_captain_cols = [col for col in captain_display_cols if col in captain_options.columns]
        
        return mo.vstack([
            mo.md(f"### 👑 **Captain:** {captain['web_name']}"),
            mo.md(f"### ⚪ **Vice-Captain:** {vice_captain['web_name']}"),
            mo.md("**Top Captain Options:**"),
            mo.ui.table(
                captain_options[available_captain_cols].round(2),
                page_size=3
            )
        ])
    
    captain_display = select_captain()
    captain_display
    
    return (captain_display,)


@app.cell
def __(mo, current_squad, team_data):
    # Summary
    if len(current_squad) > 0 and team_data:
        mo.md(f"""
        ## ✅ Summary
        
        **Team:** {team_data['entry_name']}
        **Players:** {len(current_squad)}/15
        **Budget:** £{team_data['bank']:.1f}m remaining
        
        **Next Steps:**
        1. Review transfer recommendations above
        2. Set your starting 11 formation
        3. Choose captain and vice-captain
        4. Make changes in FPL before deadline
        """)
    else:
        mo.md("Enter your manager ID above to get started")
    
    return


if __name__ == "__main__":
    app.run()