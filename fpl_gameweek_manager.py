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
        # FPL Gameweek Manager (Simplified)

        **Weekly Decision Making Tool**
        
        1. Load your team from FPL API
        2. Identify needed transfers
        3. Pick optimal lineup 
        4. Select captain and vice-captain
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import requests
    from datetime import datetime
    import warnings
    warnings.filterwarnings('ignore')
    
    return pd, np, requests, datetime, warnings


@app.cell
def __(mo):
    mo.md("## 1. Load Team from FPL API")
    return


@app.cell
def __(requests, pd):
    def fetch_fpl_data():
        """Fetch FPL data from API"""
        print("🔄 Fetching FPL data...")
        
        try:
            # Get bootstrap data
            bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            response = requests.get(bootstrap_url)
            response.raise_for_status()
            data = response.json()
            
            # Create DataFrames
            players = pd.DataFrame(data['elements'])
            teams = pd.DataFrame(data['teams'])
            events = pd.DataFrame(data['events'])
            
            # Clean players data
            players['player_id'] = players['id']
            players['price'] = players['now_cost'] / 10
            players['position'] = players['element_type'].map({1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'})
            
            # Get current gameweek
            current_gw = None
            for event in data['events']:
                if event['is_current']:
                    current_gw = event['id']
                    break
            
            print(f"✅ Loaded {len(players)} players, {len(teams)} teams")
            print(f"📅 Current GW: {current_gw}")
            
            return players, teams, events, current_gw
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, None, None

    def fetch_manager_team(manager_id):
        """Fetch manager's current team"""
        print(f"🔄 Fetching team for manager {manager_id}...")
        
        try:
            # Get manager data
            url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/"
            response = requests.get(url)
            response.raise_for_status()
            manager_data = response.json()
            
            # Get current picks
            current_event = manager_data.get("current_event", 1)
            picks_url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/event/{current_event}/picks/"
            picks_response = requests.get(picks_url)
            picks_response.raise_for_status()
            picks_data = picks_response.json()
            
            team_info = {
                "manager_id": manager_id,
                "entry_name": manager_data.get("name", ""),
                "total_points": manager_data.get("summary_overall_points", 0),
                "bank": picks_data.get("entry_history", {}).get("bank", 0) / 10,
                "team_value": picks_data.get("entry_history", {}).get("value", 0) / 10,
                "picks": picks_data.get("picks", [])
            }
            
            print(f"✅ Loaded team: {team_info['entry_name']}")
            return team_info
            
        except Exception as e:
            print(f"❌ Error fetching team: {e}")
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
            
            team_display = mo.vstack([
                mo.md(f"### ✅ {team_data['entry_name']}"),
                mo.md(f"**Points:** {team_data['total_points']:,} | **Bank:** £{team_data['bank']:.1f}m | **Value:** £{team_data['team_value']:.1f}m"),
                mo.ui.table(
                    current_squad[['web_name', 'position', 'name', 'price', 'role']].round(1),
                    page_size=15
                )
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
            
            return mo.vstack([
                mo.md(f"### ⚽ Best Formation: {best_formation['name']}"),
                mo.md("**Starting 11:**"),
                mo.ui.table(
                    best_11[['web_name', 'position', 'name', 'price', 'score']].round(2),
                    page_size=11
                ),
                mo.md("**Bench:**"),
                mo.ui.table(
                    bench[['web_name', 'position', 'name', 'price']].round(1),
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
        
        return mo.vstack([
            mo.md(f"### 👑 **Captain:** {captain['web_name']}"),
            mo.md(f"### ⚪ **Vice-Captain:** {vice_captain['web_name']}"),
            mo.md("**Top Captain Options:**"),
            mo.ui.table(
                captain_options[['web_name', 'position', 'name', 'price', 'captain_score']].round(2),
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