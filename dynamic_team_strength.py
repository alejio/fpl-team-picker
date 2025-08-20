"""
Dynamic Team Strength Calculation Module

Calculates team strength ratings that evolve throughout the season:
- GW1-7: Weighted combination of 2024-25 historical data + current season
- GW8+: Pure current season performance (ignoring historical data)

This provides stable early season ratings while becoming fully responsive 
to current season performance once sufficient data is available.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DynamicTeamStrength:
    """
    Dynamic team strength calculator with historical transition logic
    
    Key Features:
    - Uses 2024-25 season data as historical baseline
    - Transitions to current season focus at GW8
    - Rolling 6-gameweek windows for current season metrics
    - Home/away venue-specific adjustments
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize dynamic team strength calculator
        
        Args:
            debug: Enable debug logging
        """
        self.debug = debug
        self._historical_cache = {}
        self._current_season_cache = {}
        
        # Transition parameters
        self.HISTORICAL_TRANSITION_GW = 8  # GW8+ = current season only
        self.ROLLING_WINDOW_SIZE = 6  # Games for current season rolling average
        
        if debug:
            print(f"🏟️ DynamicTeamStrength initialized - Transition at GW{self.HISTORICAL_TRANSITION_GW}")
    
    def get_team_strength(self, 
                         target_gameweek: int,
                         teams_data: pd.DataFrame,
                         current_season_data: List[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Get team strength ratings for target gameweek
        
        Args:
            target_gameweek: Target gameweek for strength calculation
            teams_data: Team reference data for name mapping
            current_season_data: List of gameweek DataFrames for current season
            
        Returns:
            Dict mapping team name to strength rating [0.7, 1.3]
        """
        if target_gameweek >= self.HISTORICAL_TRANSITION_GW:
            return self._get_current_season_strength(target_gameweek, teams_data, current_season_data)
        else:
            return self._get_weighted_strength(target_gameweek, teams_data, current_season_data)
    
    def _get_current_season_strength(self, 
                                   target_gameweek: int,
                                   teams_data: pd.DataFrame,
                                   current_season_data: List[pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate team strength using current season data only (GW8+)
        
        Uses rolling 6-gameweek window for stability while being responsive
        """
        if self.debug:
            print(f"🔄 Calculating current season strength for GW{target_gameweek}")
        
        if not current_season_data or len(current_season_data) == 0:
            if self.debug:
                print("⚠️ No current season data, falling back to historical")
            return self._get_historical_baseline(teams_data)
        
        # Get rolling window of recent gameweeks
        recent_gws = current_season_data[-self.ROLLING_WINDOW_SIZE:]
        
        if len(recent_gws) == 0:
            return self._get_historical_baseline(teams_data)
        
        # Aggregate team performance across rolling window
        team_stats = self._aggregate_current_season_stats(recent_gws, teams_data)
        
        # Calculate strength ratings
        strength_ratings = self._calculate_strength_from_stats(team_stats, is_current_season=True)
        
        if self.debug:
            print(f"✅ Current season strength calculated using {len(recent_gws)} recent gameweeks")
        
        return strength_ratings
    
    def _get_weighted_strength(self, 
                              target_gameweek: int,
                              teams_data: pd.DataFrame,
                              current_season_data: List[pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate weighted team strength (historical + current) for early season
        
        Weighting schedule:
        - GW1-3: 80% historical, 20% current
        - GW4-7: 60% historical, 40% current
        """
        if self.debug:
            print(f"🔄 Calculating weighted strength for GW{target_gameweek}")
        
        # Determine weights based on gameweek
        if target_gameweek <= 3:
            historical_weight = 0.8
            current_weight = 0.2
        elif target_gameweek <= 7:
            historical_weight = 0.6
            current_weight = 0.4
        else:
            # Shouldn't reach here, but safety fallback
            historical_weight = 0.0
            current_weight = 1.0
        
        # Get historical baseline
        historical_strength = self._get_historical_baseline(teams_data)
        
        # Get current season component
        if current_season_data and len(current_season_data) > 0:
            # Use available current season data
            available_gws = current_season_data[:target_gameweek-1]  # Up to previous GW
            current_stats = self._aggregate_current_season_stats(available_gws, teams_data)
            current_strength = self._calculate_strength_from_stats(current_stats, is_current_season=True)
        else:
            # No current data available, use historical only
            current_strength = historical_strength
            current_weight = 0.0
            historical_weight = 1.0
        
        # Weighted combination
        weighted_strength = {}
        for team_name in historical_strength.keys():
            weighted_value = (
                historical_strength[team_name] * historical_weight + 
                current_strength.get(team_name, historical_strength[team_name]) * current_weight
            )
            weighted_strength[team_name] = round(weighted_value, 3)
        
        if self.debug:
            print(f"✅ Weighted strength: {historical_weight:.0%} historical + {current_weight:.0%} current")
        
        return weighted_strength
    
    def _get_historical_baseline(self, teams_data: pd.DataFrame) -> Dict[str, float]:
        """
        Load and calculate team strength from 2024-25 season data
        """
        if 'historical_baseline' in self._historical_cache:
            return self._historical_cache['historical_baseline']
        
        try:
            # Load 2024-25 vaastav data
            historical_path = '/Users/alex/dev/FPL/fpl-dataset-builder/data/vaastav_full_player_history_2024_2025.csv'
            historical_df = pd.read_csv(historical_path)
            
            if self.debug:
                print(f"📂 Loaded 2024-25 historical data: {historical_df.shape}")
            
            # Aggregate team-level statistics
            team_stats = historical_df.groupby('team').agg({
                'goals_scored': 'sum',
                'goals_conceded': 'sum',
                'expected_goals': 'sum',
                'expected_goals_conceded': 'sum',
                'clean_sheets': 'sum',
                'total_points': 'sum'
            }).round(3)
            
            # Calculate per-game averages (38 games in season)
            team_stats['xG_per_game'] = team_stats['expected_goals'] / 38
            team_stats['xGA_per_game'] = team_stats['expected_goals_conceded'] / 38
            team_stats['goals_per_game'] = team_stats['goals_scored'] / 38
            team_stats['goals_conceded_per_game'] = team_stats['goals_conceded'] / 38
            
            # Add team names
            team_stats = team_stats.reset_index()
            
            # Handle different team_id column naming
            if 'team_id' in teams_data.columns:
                team_stats = team_stats.merge(
                    teams_data[['team_id', 'name']], 
                    left_on='team', 
                    right_on='team_id', 
                    how='left'
                )
            else:
                # Fallback: create team name mapping
                team_names = {
                    1: 'Arsenal', 2: 'Aston Villa', 3: 'Burnley', 4: 'Bournemouth', 5: 'Brentford',
                    6: 'Brighton', 7: 'Chelsea', 8: 'Crystal Palace', 9: 'Everton', 10: 'Fulham',
                    11: 'Leeds', 12: 'Liverpool', 13: 'Man City', 14: 'Man Utd', 15: 'Newcastle',
                    16: "Nott'm Forest", 17: 'Sunderland', 18: 'Spurs', 19: 'West Ham', 20: 'Wolves'
                }
                team_stats['name'] = team_stats['team'].map(team_names)
            
            # Calculate strength ratings
            strength_ratings = self._calculate_strength_from_stats(team_stats, is_current_season=False)
            
            # Cache for future use
            self._historical_cache['historical_baseline'] = strength_ratings
            
            if self.debug:
                print(f"✅ Historical baseline calculated for {len(strength_ratings)} teams")
            
            return strength_ratings
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error loading historical data: {e}")
            
            # Fallback to static ratings
            return self._get_static_fallback_ratings()
    
    def _aggregate_current_season_stats(self, 
                                       gameweek_data_list: List[pd.DataFrame],
                                       teams_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate team statistics from current season gameweek data
        """
        if not gameweek_data_list:
            return pd.DataFrame()
        
        try:
            # Combine all gameweek data
            all_gw_data = pd.concat(gameweek_data_list, ignore_index=True)
            
            # Get player-team mapping from current players
            from client import get_current_players
            players = get_current_players()
            
            # Merge gameweek data with team info
            gw_with_teams = all_gw_data.merge(
                players[['player_id', 'team_id']], 
                on='player_id', 
                how='left'
            )
            
            # Aggregate by team
            team_stats = gw_with_teams.groupby('team_id').agg({
                'goals_scored': 'sum',
                'goals_conceded': 'sum',
                'expected_goals': 'sum',
                'expected_goals_conceded': 'sum',
                'clean_sheets': 'sum',
                'total_points': 'sum',
                'event': 'nunique'  # Number of gameweeks
            }).round(3)
            
            # Calculate per-game averages
            team_stats['xG_per_game'] = team_stats['expected_goals'] / team_stats['event']
            team_stats['xGA_per_game'] = team_stats['expected_goals_conceded'] / team_stats['event']
            team_stats['goals_per_game'] = team_stats['goals_scored'] / team_stats['event']
            team_stats['goals_conceded_per_game'] = team_stats['goals_conceded'] / team_stats['event']
            
            # Add team names
            team_stats = team_stats.reset_index()
            
            # Handle different team_id column naming
            if 'team_id' in teams_data.columns:
                team_stats = team_stats.merge(
                    teams_data[['team_id', 'name']], 
                    left_on='team_id', 
                    right_on='team_id', 
                    how='left'
                )
            else:
                # Fallback: create team name mapping
                team_names = {
                    1: 'Arsenal', 2: 'Aston Villa', 3: 'Burnley', 4: 'Bournemouth', 5: 'Brentford',
                    6: 'Brighton', 7: 'Chelsea', 8: 'Crystal Palace', 9: 'Everton', 10: 'Fulham',
                    11: 'Leeds', 12: 'Liverpool', 13: 'Man City', 14: 'Man Utd', 15: 'Newcastle',
                    16: "Nott'm Forest", 17: 'Sunderland', 18: 'Spurs', 19: 'West Ham', 20: 'Wolves'
                }
                team_stats['name'] = team_stats['team_id'].map(team_names)
            
            if self.debug:
                print(f"📊 Current season stats aggregated for {len(team_stats)} teams over {team_stats['event'].max()} gameweeks")
            
            return team_stats
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error aggregating current season data: {e}")
            return pd.DataFrame()
    
    def _calculate_strength_from_stats(self, team_stats: pd.DataFrame, is_current_season: bool = True) -> Dict[str, float]:
        """
        Calculate team strength ratings from aggregated statistics
        
        Uses xG and xGA per game as primary indicators with actual goals as secondary
        """
        if team_stats.empty:
            return self._get_static_fallback_ratings()
        
        strength_ratings = {}
        
        # Use xG/xGA if available, otherwise fall back to actual goals
        if 'xG_per_game' in team_stats.columns and 'xGA_per_game' in team_stats.columns:
            attack_metric = 'xG_per_game'
            defense_metric = 'xGA_per_game'
        else:
            attack_metric = 'goals_per_game'
            defense_metric = 'goals_conceded_per_game'
        
        # Calculate league averages for normalization
        league_avg_attack = team_stats[attack_metric].mean()
        league_avg_defense = team_stats[defense_metric].mean()
        
        for _, team in team_stats.iterrows():
            team_name = team['name']
            
            # Attack strength (relative to league average)
            attack_ratio = team[attack_metric] / league_avg_attack if league_avg_attack > 0 else 1.0
            
            # Defense strength (inverse - lower xGA = stronger defense)
            defense_ratio = league_avg_defense / team[defense_metric] if team[defense_metric] > 0 else 1.0
            
            # Combined strength (average of attack and defense)
            combined_strength = (attack_ratio + defense_ratio) / 2
            
            # Normalize to [0.7, 1.3] range
            # Map [0.5, 1.5] to [0.7, 1.3] with clipping for outliers
            normalized_strength = 0.7 + (combined_strength - 0.5) * (1.3 - 0.7) / (1.5 - 0.5)
            normalized_strength = np.clip(normalized_strength, 0.7, 1.3)
            
            strength_ratings[team_name] = round(normalized_strength, 3)
        
        if self.debug:
            strongest = max(strength_ratings.items(), key=lambda x: x[1])
            weakest = min(strength_ratings.items(), key=lambda x: x[1])
            print(f"🏆 Strongest: {strongest[0]} ({strongest[1]})")
            print(f"📉 Weakest: {weakest[0]} ({weakest[1]})")
        
        return strength_ratings
    
    def _get_static_fallback_ratings(self) -> Dict[str, float]:
        """
        Fallback to static 2023-24 final table ratings if dynamic calculation fails
        """
        team_positions = {
            'Manchester City': 1, 'Arsenal': 2, 'Liverpool': 3, 'Aston Villa': 4,
            'Tottenham': 5, 'Chelsea': 6, 'Newcastle': 7, 'Manchester Utd': 8,
            'West Ham': 9, 'Crystal Palace': 10, 'Brighton': 11, 'Bournemouth': 12,
            'Fulham': 13, 'Wolves': 14, 'Everton': 15, 'Brentford': 16,
            'Nottingham Forest': 17, 'Luton': 18, 'Burnley': 19, 'Sheffield Utd': 20,
            # Handle team name variations
            'Man City': 1, 'Man Utd': 8, "Nott'm Forest": 17, 'Spurs': 5,
            # Current season promoted teams (conservative ratings)
            'Leicester': 16, 'Southampton': 18, 'Ipswich': 20
        }
        
        strength_ratings = {}
        for team, position in team_positions.items():
            if position <= 20:
                strength = 1.3 - (position - 1) * (1.3 - 0.7) / 19
            else:
                strength = 0.7 - (position - 20) * 0.05
            strength_ratings[team] = round(strength, 3)
        
        if self.debug:
            print("⚠️ Using static fallback ratings")
        
        return strength_ratings


# Utility functions for backward compatibility
def get_dynamic_team_strength(target_gameweek: int,
                             teams_data: pd.DataFrame,
                             current_season_data: List[pd.DataFrame] = None,
                             debug: bool = False) -> Dict[str, float]:
    """
    Get dynamic team strength ratings for target gameweek
    
    Args:
        target_gameweek: Target gameweek for strength calculation
        teams_data: Team reference data
        current_season_data: List of current season gameweek DataFrames
        debug: Enable debug logging
        
    Returns:
        Dict mapping team name to strength rating [0.7, 1.3]
    """
    calculator = DynamicTeamStrength(debug=debug)
    return calculator.get_team_strength(target_gameweek, teams_data, current_season_data)


def load_historical_gameweek_data(start_gw: int = 1, end_gw: int = None) -> List[pd.DataFrame]:
    """
    Load current season gameweek data for team strength calculation
    
    Args:
        start_gw: Starting gameweek to load
        end_gw: Ending gameweek to load (None = latest available)
        
    Returns:
        List of gameweek DataFrames
    """
    from client import get_gameweek_live_data
    
    gameweek_data = []
    max_gw = end_gw if end_gw else 38
    
    for gw in range(start_gw, max_gw + 1):
        try:
            gw_data = get_gameweek_live_data(gw)
            if not gw_data.empty:
                gameweek_data.append(gw_data)
        except:
            # No data available for this gameweek
            continue
    
    return gameweek_data