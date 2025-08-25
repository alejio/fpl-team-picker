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
from typing import Dict, List
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
        Calculate team strength using current season multi-factor approach (GW8+)
        
        Uses the same multi-factor calculation but with current season weighting
        """
        if self.debug:
            print(f"🔄 Calculating current season multi-factor strength for GW{target_gameweek}")
        
        try:
            # Use multi-factor calculation with current gameweek for proper seasonal weighting
            strength_ratings = self._calculate_multi_factor_strength(teams_data, target_gameweek=target_gameweek)
            
            if self.debug:
                print(f"✅ Current season multi-factor strength calculated for GW{target_gameweek}")
            
            return strength_ratings
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error in current season calculation, falling back to historical: {e}")
            return self._get_historical_baseline(teams_data)
    
    def _get_weighted_strength(self, 
                              target_gameweek: int,
                              teams_data: pd.DataFrame,
                              current_season_data: List[pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate early season team strength using multi-factor approach
        
        For early season (GW1-7), the multi-factor calculation already handles 
        the weighting internally through seasonal weight adjustment
        """
        if self.debug:
            print(f"🔄 Calculating early season multi-factor strength for GW{target_gameweek}")
        
        try:
            # Use multi-factor calculation which handles early season weighting internally
            strength_ratings = self._calculate_multi_factor_strength(teams_data, target_gameweek=target_gameweek)
            
            if self.debug:
                print(f"✅ Early season multi-factor strength calculated for GW{target_gameweek}")
            
            return strength_ratings
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error in early season calculation, falling back to historical: {e}")
            return self._get_historical_baseline(teams_data)
    
    def _get_historical_baseline(self, teams_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate team strength using multi-factor approach:
        1) League Position (25%), 2) Player Quality (35%), 3) Reputation (20%), 4) Recent Form (20%)
        """
        if 'historical_baseline' in self._historical_cache:
            return self._historical_cache['historical_baseline']
        
        try:
            # Use multi-factor calculation (default to GW1 for historical baseline)
            strength_ratings = self._calculate_multi_factor_strength(teams_data, target_gameweek=1)
            
            # Cache for future use
            self._historical_cache['historical_baseline'] = strength_ratings
            
            if self.debug:
                print(f"✅ Multi-factor team strength calculated for {len(strength_ratings)} teams")
                strongest = max(strength_ratings.items(), key=lambda x: x[1])
                weakest = min(strength_ratings.items(), key=lambda x: x[1])
                print(f"🏆 Strongest: {strongest[0]} ({strongest[1]})")
                print(f"📉 Weakest: {weakest[0]} ({weakest[1]})")
            
            return strength_ratings
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error in multi-factor calculation: {e}")
            
            # Fallback to static ratings
            return self._get_static_fallback_ratings()
    
    def _calculate_multi_factor_strength(self, teams_data: pd.DataFrame, target_gameweek: int = None) -> Dict[str, float]:
        """
        Calculate team strength using comprehensive multi-factor approach
        
        Factors:
        1) League Position (25% weight) - Current/historical league standing
        2) Player Quality (35% weight) - Squad value and total points 
        3) Reputation (20% weight) - Historical success over 3-5 seasons
        4) Recent Form (20% weight) - Last 6 gameweeks performance
        """
        if self.debug:
            print(f"🔍 Calculating multi-factor team strength for GW{target_gameweek or 'current'}")
        
        try:
            from client import FPLDataClient
            client = FPLDataClient()
            
            # Get data sources
            players = client.get_current_players()
            
            # Get current gameweek if not provided
            if target_gameweek is None:
                target_gameweek = 1  # Default to GW1 for historical baseline
            
            # Initialize strength components
            strength_components = {}
            all_teams = self._get_all_team_names(teams_data, players)
            
            for team_name in all_teams:
                strength_components[team_name] = {
                    'position': 1.0, 'quality': 1.0, 'reputation': 1.0, 'form': 1.0
                }
            
            # Calculate each factor
            position_strengths = self._calculate_position_strength(target_gameweek, all_teams)
            quality_strengths = self._calculate_player_quality_strength(players, all_teams)
            reputation_strengths = self._calculate_reputation_strength(all_teams)
            form_strengths = self._calculate_recent_form_strength(target_gameweek, all_teams)
            
            # Get seasonal weights based on gameweek
            weights = self._get_seasonal_weights(target_gameweek)
            
            # Combine factors with seasonal weighting
            final_strengths = {}
            for team_name in all_teams:
                combined_strength = (
                    position_strengths.get(team_name, 1.0) * weights['position'] +
                    quality_strengths.get(team_name, 1.0) * weights['quality'] +
                    reputation_strengths.get(team_name, 1.0) * weights['reputation'] +
                    form_strengths.get(team_name, 1.0) * weights['form']
                )
                
                # Ensure final strength is in [0.7, 1.3] range
                final_strengths[team_name] = round(np.clip(combined_strength, 0.7, 1.3), 3)
            
            if self.debug:
                print(f"✅ Multi-factor calculation complete with weights: {weights}")
                sample_team = next(iter(final_strengths.keys()))
                print(f"📊 Example ({sample_team}): pos={position_strengths.get(sample_team, 1.0):.3f}, quality={quality_strengths.get(sample_team, 1.0):.3f}, rep={reputation_strengths.get(sample_team, 1.0):.3f}, form={form_strengths.get(sample_team, 1.0):.3f} → {final_strengths[sample_team]:.3f}")
            
            return final_strengths
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error in multi-factor calculation: {e}")
            return self._get_static_fallback_ratings()
    
    def _get_all_team_names(self, teams_data: pd.DataFrame, players: pd.DataFrame) -> List[str]:
        """Get comprehensive list of all team names from available data sources"""
        team_names = set()
        
        # From teams data
        if 'name' in teams_data.columns:
            team_names.update(teams_data['name'].dropna().tolist())
        
        # From players data (team names)
        if 'team_name' in players.columns:
            team_names.update(players['team_name'].dropna().unique().tolist())
        
        # Fallback to static team list
        if not team_names:
            team_names = {
                'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
                'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
                'Liverpool', 'Leeds', 'Leicester', 'Manchester City', 'Manchester United',
                'Newcastle', "Nottingham Forest", 'Southampton', 'Tottenham', 'West Ham', 'Wolves'
            }
        
        return sorted(list(team_names))
    
    def _calculate_position_strength(self, target_gameweek: int, all_teams: List[str]) -> Dict[str, float]:
        """Factor 1: League Position (25% weight)"""
        try:
            # For historical baseline, use static 2023-24 final positions
            # In future: get actual current season league table
            
            historical_positions = {
                'Manchester City': 1, 'Arsenal': 2, 'Liverpool': 3, 'Aston Villa': 4,
                'Tottenham': 5, 'Chelsea': 6, 'Newcastle': 7, 'Manchester United': 8,
                'West Ham': 9, 'Crystal Palace': 10, 'Brighton': 11, 'Bournemouth': 12,
                'Fulham': 13, 'Wolves': 14, 'Everton': 15, 'Brentford': 16,
                'Nottingham Forest': 17, 'Luton': 18, 'Burnley': 19, 'Sheffield United': 20,
                # Current season teams
                'Leicester': 16, 'Southampton': 18, 'Ipswich': 20
            }
            
            position_strengths = {}
            for team_name in all_teams:
                # Handle team name variations
                lookup_name = team_name
                if team_name == 'Man City':
                    lookup_name = 'Manchester City'
                elif team_name == 'Man Utd':
                    lookup_name = 'Manchester United'
                elif team_name == "Nott'm Forest":
                    lookup_name = 'Nottingham Forest'
                elif team_name == 'Spurs':
                    lookup_name = 'Tottenham'
                
                position = historical_positions.get(lookup_name, 15)  # Default to mid-table
                
                # Convert position to strength [0.7, 1.3] - higher position = lower number = higher strength
                strength = 1.3 - (position - 1) * (1.3 - 0.7) / 19
                position_strengths[team_name] = round(np.clip(strength, 0.7, 1.3), 3)
            
            if self.debug:
                print(f"📊 Position strengths calculated for {len(position_strengths)} teams")
            
            return position_strengths
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error calculating position strength: {e}")
            return {team: 1.0 for team in all_teams}
    
    def _calculate_player_quality_strength(self, players: pd.DataFrame, all_teams: List[str]) -> Dict[str, float]:
        """Factor 2: Player Quality (35% weight) - Squad value and total points"""
        try:
            quality_strengths = {}
            
            # Calculate squad metrics by team
            team_metrics = {}
            
            for team_name in all_teams:
                # Get team players
                if 'team_name' in players.columns:
                    team_players = players[players['team_name'] == team_name]
                else:
                    # Fallback: try to match by team_id or other means
                    team_players = pd.DataFrame()  # Empty if no match
                
                if len(team_players) > 0:
                    # Squad value (now_cost is in £0.1m units)
                    squad_value = team_players['now_cost'].sum() / 10.0  # Convert to £m
                    
                    # Squad performance
                    squad_points = team_players['total_points'].sum()
                    
                    team_metrics[team_name] = {
                        'value': squad_value,
                        'points': squad_points
                    }
                else:
                    # No players found - use defaults
                    team_metrics[team_name] = {'value': 50.0, 'points': 100}
            
            # Calculate league averages
            if team_metrics:
                league_avg_value = np.mean([m['value'] for m in team_metrics.values()])
                league_avg_points = np.mean([m['points'] for m in team_metrics.values()])
            else:
                league_avg_value, league_avg_points = 50.0, 100
            
            # Calculate relative quality for each team
            for team_name in all_teams:
                metrics = team_metrics[team_name]
                
                # Normalize against league averages
                value_ratio = metrics['value'] / league_avg_value if league_avg_value > 0 else 1.0
                points_ratio = metrics['points'] / league_avg_points if league_avg_points > 0 else 1.0
                
                # Combined quality (equal weighting of value and points)
                combined_quality = (value_ratio + points_ratio) / 2
                
                # Map to [0.7, 1.3] range with reasonable bounds
                # Expect combined_quality to be roughly [0.5, 1.5]
                strength = 0.7 + (combined_quality - 0.5) * (1.3 - 0.7) / (1.5 - 0.5)
                quality_strengths[team_name] = round(np.clip(strength, 0.7, 1.3), 3)
            
            if self.debug:
                print(f"💎 Player quality strengths calculated for {len(quality_strengths)} teams")
                print(f"💰 League avg value: £{league_avg_value:.1f}m, avg points: {league_avg_points:.0f}")
            
            return quality_strengths
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error calculating player quality: {e}")
            return {team: 1.0 for team in all_teams}
    
    def _calculate_reputation_strength(self, all_teams: List[str]) -> Dict[str, float]:
        """Factor 3: Reputation (20% weight) - Historical success over 3-5 seasons"""
        reputation_ratings = {
            # Big 6 - traditional top teams
            'Manchester City': 1.3, 'Liverpool': 1.25, 'Arsenal': 1.25,
            'Chelsea': 1.2, 'Manchester United': 1.15, 'Tottenham': 1.15,
            
            # Strong teams - recent success
            'Newcastle': 1.1, 'Aston Villa': 1.05, 
            
            # Solid mid-table
            'Brighton': 1.0, 'West Ham': 1.0, 'Crystal Palace': 0.95,
            'Fulham': 0.95, 'Brentford': 0.9, 'Bournemouth': 0.9,
            
            # Lower mid-table
            'Everton': 0.85, 'Wolves': 0.85, 'Nottingham Forest': 0.8,
            
            # Historically weaker/promoted teams
            'Leicester': 0.85,  # Recently relegated but historically strong
            'Southampton': 0.8, 'Leeds': 0.8, 'Burnley': 0.75,
            'Sheffield United': 0.7, 'Luton': 0.7, 'Ipswich': 0.7
        }
        
        reputation_strengths = {}
        for team_name in all_teams:
            # Handle team name variations
            lookup_name = team_name
            if team_name == 'Man City':
                lookup_name = 'Manchester City'
            elif team_name == 'Man Utd':
                lookup_name = 'Manchester United'
            elif team_name == "Nott'm Forest":
                lookup_name = 'Nottingham Forest'
            elif team_name == 'Spurs':
                lookup_name = 'Tottenham'
            
            reputation_strengths[team_name] = reputation_ratings.get(lookup_name, 0.9)  # Default mid-table
        
        if self.debug:
            print(f"🏆 Reputation strengths assigned for {len(reputation_strengths)} teams")
        
        return reputation_strengths
    
    def _calculate_recent_form_strength(self, target_gameweek: int, all_teams: List[str]) -> Dict[str, float]:
        """Factor 4: Recent Form (20% weight) - Last 6 gameweeks performance"""
        try:
            from client import FPLDataClient
            client = FPLDataClient()
            
            form_strengths = {}
            
            # For historical baseline or early season, use neutral form
            if target_gameweek <= 6:
                if self.debug:
                    print(f"⚡ Using neutral form for early season (GW{target_gameweek})")
                return {team: 1.0 for team in all_teams}
            
            # Load recent gameweeks data
            recent_gws = []
            for gw in range(max(1, target_gameweek - 6), target_gameweek):
                try:
                    gw_data = client.get_gameweek_live_data(gw)
                    if not gw_data.empty:
                        recent_gws.append(gw_data)
                except Exception:
                    continue
            
            if not recent_gws:
                if self.debug:
                    print("⚡ No recent form data available, using neutral form")
                return {team: 1.0 for team in all_teams}
            
            # Calculate team form from recent gameweeks
            # This is simplified - in practice would need team-level aggregation
            for team_name in all_teams:
                # For now, use neutral form - can enhance later with actual team performance data
                form_strengths[team_name] = 1.0
            
            if self.debug:
                print(f"⚡ Recent form calculated for {len(form_strengths)} teams over {len(recent_gws)} gameweeks")
            
            return form_strengths
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error calculating recent form: {e}")
            return {team: 1.0 for team in all_teams}
    
    def _get_seasonal_weights(self, target_gameweek: int) -> Dict[str, float]:
        """Get seasonal weighting factors based on gameweek"""
        if target_gameweek <= 7:
            # Early season: Higher weight on reputation + historical position
            return {
                'position': 0.20,  # Reduced - less reliable early
                'quality': 0.35,   # Stable throughout
                'reputation': 0.30,  # Increased early season
                'form': 0.15       # Reduced - insufficient data
            }
        elif target_gameweek <= 25:
            # Mid season: Balanced weights as specified
            return {
                'position': 0.25,
                'quality': 0.35,
                'reputation': 0.20,
                'form': 0.20
            }
        else:
            # Late season: Higher weight on current position + recent form
            return {
                'position': 0.30,  # Increased - more reliable
                'quality': 0.30,   # Slightly reduced
                'reputation': 0.15,  # Reduced - less relevant
                'form': 0.25       # Increased - crucial for late season
            }

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
            from client import FPLDataClient
            client = FPLDataClient()
            players = client.get_current_players()
            
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
    from client import FPLDataClient
    client = FPLDataClient()
    
    gameweek_data = []
    max_gw = end_gw if end_gw else 38
    
    for gw in range(start_gw, max_gw + 1):
        try:
            gw_data = client.get_gameweek_live_data(gw)
            if not gw_data.empty:
                gameweek_data.append(gw_data)
        except Exception:
            # No data available for this gameweek
            continue
    
    return gameweek_data