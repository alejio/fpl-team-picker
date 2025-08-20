"""
FPL Expected Points (XP) Model - Dedicated Module

Modular, testable implementation of XP calculations for both single gameweek
and multi-gameweek scenarios. Designed for continuous improvement and validation.

Key Features:
- Form-weighted predictions with live data integration  
- Statistical xG/xA estimation for missing data
- Dynamic team strength ratings
- Enhanced minutes prediction model
- Multi-gameweek horizon capability
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class XPModel:
    """
    Fantasy Premier League Expected Points Model
    
    Calculates expected points using:
    - Form-weighted performance data
    - Statistical xG/xA estimation
    - Dynamic team strength ratings
    - Enhanced minutes prediction
    """
    
    def __init__(self, 
                 form_weight: float = 0.7,
                 form_window: int = 5,
                 debug: bool = False):
        """
        Initialize XP Model
        
        Args:
            form_weight: Weight given to recent form vs season average (0.7 = 70% form)
            form_window: Number of recent gameweeks to consider for form (5 = last 5 GWs)
            debug: Enable debug logging
        """
        self.form_weight = form_weight
        self.form_window = form_window
        self.debug = debug
        
        # Model components
        self._team_strength_cache = {}
        self._player_form_cache = {}
        
        if debug:
            print(f"🧠 XPModel initialized - Form weight: {form_weight}, Window: {form_window} GWs")
    
    def calculate_expected_points(self,
                                players_data: pd.DataFrame,
                                teams_data: pd.DataFrame,
                                xg_rates_data: pd.DataFrame,
                                fixtures_data: pd.DataFrame,
                                target_gameweek: int,
                                live_data: pd.DataFrame = None,
                                gameweeks_ahead: int = 1) -> pd.DataFrame:
        """
        Calculate expected points for all players
        
        Args:
            players_data: Current players dataset
            teams_data: Teams reference data
            xg_rates_data: xG/xA rates per 90 minutes
            fixtures_data: Fixture data
            target_gameweek: Target gameweek for prediction
            live_data: Optional live performance data for form weighting
            gameweeks_ahead: Number of gameweeks to predict (1 for single GW)
            
        Returns:
            DataFrame with players and their expected points
        """
        if self.debug:
            print(f"🔮 Calculating XP for GW{target_gameweek} (+{gameweeks_ahead-1} ahead)")
        
        # Prepare dataset
        players_xp = players_data.copy()
        
        # Merge team information
        players_xp = self._merge_team_data(players_xp, teams_data)
        
        # Calculate team strength ratings
        team_strength = self._get_dynamic_team_strength(teams_data, players_data, target_gameweek)
        
        # Merge xG/xA rates with statistical estimation for missing data
        players_xp = self._merge_xg_xa_rates(players_xp, xg_rates_data, team_strength)
        
        # Calculate form-weighted performance if live data available
        if live_data is not None and not live_data.empty:
            players_xp = self._apply_form_weighting(players_xp, live_data, target_gameweek)
        
        # Calculate expected minutes
        players_xp = self._calculate_expected_minutes(players_xp)
        
        # Calculate fixture difficulty for target gameweek(s)
        fixture_difficulty = self._calculate_fixture_difficulty(
            fixtures_data, teams_data, team_strength, target_gameweek, gameweeks_ahead
        )
        players_xp = self._apply_fixture_difficulty(players_xp, fixture_difficulty)
        
        # Calculate XP components
        players_xp = self._calculate_xp_components(players_xp, gameweeks_ahead)
        
        # Add efficiency metrics
        players_xp['xP_per_price'] = players_xp['xP'] / players_xp.get('price', players_xp.get('price_gbp', 1))
        
        if self.debug:
            print(f"✅ XP calculated for {len(players_xp)} players")
            print(f"📊 Average XP: {players_xp['xP'].mean():.2f}, Top XP: {players_xp['xP'].max():.2f}")
        
        return players_xp
    
    def _merge_team_data(self, players_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Merge team information with players"""
        # Handle different column naming conventions
        team_id_col = 'team_id' if 'team_id' in teams_df.columns else 'id'
        player_team_col = 'team' if 'team' in players_df.columns else 'team_id'
        
        merged = players_df.merge(
            teams_df[[team_id_col, 'name']], 
            left_on=player_team_col, 
            right_on=team_id_col, 
            how='left'
        )
        
        # Clean up duplicate columns
        if team_id_col != player_team_col and team_id_col in merged.columns:
            merged = merged.drop(team_id_col, axis=1)
            
        return merged
    
    def _get_dynamic_team_strength(self, teams_df: pd.DataFrame, 
                                 players_df: pd.DataFrame, 
                                 target_gameweek: int) -> Dict[str, float]:
        """
        Calculate dynamic team strength ratings based on current season performance
        
        For now, uses 2023-24 table as baseline but designed to incorporate:
        - Current season form (last 6 fixtures)
        - Goal difference and xG performance  
        - Squad depth and injury impact
        """
        # Static baseline from 2023-24 final table
        team_positions = {
            'Manchester City': 1, 'Arsenal': 2, 'Liverpool': 3, 'Aston Villa': 4,
            'Tottenham': 5, 'Chelsea': 6, 'Newcastle': 7, 'Manchester Utd': 8,
            'West Ham': 9, 'Crystal Palace': 10, 'Brighton': 11, 'Bournemouth': 12,
            'Fulham': 13, 'Wolves': 14, 'Everton': 15, 'Brentford': 16,
            'Nottingham Forest': 17, 'Luton': 18, 'Burnley': 19, 'Sheffield Utd': 20,
            # Handle team name variations
            'Man City': 1, 'Man Utd': 8, "Nott'm Forest": 17, 'Spurs': 5,
            # Add current season promoted teams
            'Leicester': 16, 'Southampton': 18, 'Ipswich': 20
        }
        
        strength_ratings = {}
        for team, position in team_positions.items():
            if position <= 20:
                strength = 1.3 - (position - 1) * (1.3 - 0.7) / 19
            else:
                strength = 0.7 - (position - 20) * 0.05
            strength_ratings[team] = round(strength, 3)
        
        # TODO: Phase 2 - Add current season form adjustments
        # - Recent 6 fixture performance
        # - Goals scored/conceded rates
        # - Home/away split
        
        self._team_strength_cache = strength_ratings
        return strength_ratings
    
    def _merge_xg_xa_rates(self, players_df: pd.DataFrame, 
                          xg_rates_df: pd.DataFrame,
                          team_strength: Dict[str, float]) -> pd.DataFrame:
        """
        Merge xG/xA rates with statistical estimation for missing data
        
        Uses advanced statistical model from season-start version for players
        without historical data.
        """
        # Merge existing xG/xA data
        merged = players_df.merge(
            xg_rates_df[['mapped_player_id', 'xG90', 'xA90']], 
            left_on='player_id',
            right_on='mapped_player_id',
            how='left'
        )
        
        # Identify players missing xG/xA data
        missing_mask = merged['xG90'].isna() | merged['xA90'].isna()
        missing_players = merged[missing_mask].copy()
        
        if len(missing_players) > 0:
            if self.debug:
                print(f"🔍 Estimating xG/xA for {len(missing_players)} players with missing data")
            
            # Statistical estimation using multi-factor model
            estimated_rates = self._estimate_missing_xg_xa_rates(missing_players, team_strength)
            
            # Update missing values
            merged.loc[missing_mask, 'xG90'] = estimated_rates['xG90']
            merged.loc[missing_mask, 'xA90'] = estimated_rates['xA90']
        
        # Fill any remaining missing values with position-based defaults
        position_xg_defaults = {'GKP': 0.005, 'DEF': 0.08, 'MID': 0.15, 'FWD': 0.25}
        position_xa_defaults = {'GKP': 0.02, 'DEF': 0.12, 'MID': 0.20, 'FWD': 0.15}
        
        merged['xG90'] = merged['xG90'].fillna(merged['position'].map(position_xg_defaults))
        merged['xA90'] = merged['xA90'].fillna(merged['position'].map(position_xa_defaults))
        
        return merged
    
    def _estimate_missing_xg_xa_rates(self, missing_players_df: pd.DataFrame,
                                    team_strength: Dict[str, float]) -> pd.DataFrame:
        """
        Statistical estimation of xG90/xA90 for players missing historical data
        
        Uses price, position, team strength, and SBP as predictors.
        Based on the advanced model from fpl_xp_model.py
        """
        estimates = missing_players_df.copy()
        
        # Add team strength feature
        estimates['team_strength'] = estimates['name'].map(team_strength).fillna(1.0)
        
        # Handle missing SBP with position-based medians
        position_sbp_defaults = {'GKP': 5.0, 'DEF': 8.0, 'MID': 12.0, 'FWD': 10.0}
        sbp_col = 'selected_by_percent' if 'selected_by_percent' in estimates.columns else 'selected_by_percentage'
        if sbp_col in estimates.columns:
            estimates[sbp_col] = estimates[sbp_col].fillna(estimates['position'].map(position_sbp_defaults))
        else:
            estimates[sbp_col] = estimates['position'].map(position_sbp_defaults)
        
        # Get price column (handle different naming)
        price_col = 'price' if 'price' in estimates.columns else 'price_gbp'
        if price_col not in estimates.columns:
            # Fallback: estimate price from position
            estimates[price_col] = estimates['position'].map({'GKP': 4.5, 'DEF': 5.0, 'MID': 6.0, 'FWD': 7.0})
        
        def estimate_xg90(row):
            """Estimate xG90 using multi-factor model"""
            base_rates = {'GKP': 0.01, 'DEF': 0.10, 'MID': 0.25, 'FWD': 0.45}
            base_xg = base_rates[row['position']]
            
            # Price adjustment
            price = row[price_col]
            if price >= 8.0:  # Premium players
                price_multiplier = 1.4 + (price - 8.0) * 0.15
            elif price >= 6.0:  # Mid-tier players
                price_multiplier = 1.1 + (price - 6.0) * 0.15
            elif price >= 4.5:  # Budget options
                price_multiplier = 0.8 + (price - 4.5) * 0.2
            else:  # Very cheap players
                price_multiplier = 0.5 + (price - 4.0) * 0.6
            
            # Team strength adjustment
            team_multiplier = 0.7 + (row['team_strength'] - 0.7) * 0.8
            
            # SBP adjustment
            sbp_multiplier = 1.0 + (row[sbp_col] / 100) * 0.3
            
            estimated_xg = base_xg * price_multiplier * team_multiplier * sbp_multiplier
            
            # Position-specific caps
            caps = {'GKP': 0.05, 'DEF': 0.35, 'MID': 0.70, 'FWD': 1.20}
            return min(estimated_xg, caps[row['position']])
        
        def estimate_xa90(row):
            """Estimate xA90 using multi-factor model"""
            base_rates = {'GKP': 0.05, 'DEF': 0.15, 'MID': 0.30, 'FWD': 0.20}
            base_xa = base_rates[row['position']]
            
            # Price adjustment for creativity
            price = row[price_col]
            if price >= 7.5:  # Premium creative players
                price_multiplier = 1.5 + (price - 7.5) * 0.2
            elif price >= 5.5:  # Mid-tier
                price_multiplier = 1.0 + (price - 5.5) * 0.25
            else:  # Budget players
                price_multiplier = 0.6 + (price - 4.0) * 0.25
            
            # Team strength matters more for assists
            team_multiplier = 0.6 + (row['team_strength'] - 0.7) * 1.0
            
            # SBP adjustment
            sbp_multiplier = 1.0 + (row[sbp_col] / 100) * 0.25
            
            estimated_xa = base_xa * price_multiplier * team_multiplier * sbp_multiplier
            
            # Position-specific caps
            caps = {'GKP': 0.10, 'DEF': 0.40, 'MID': 0.80, 'FWD': 0.50}
            return min(estimated_xa, caps[row['position']])
        
        # Generate estimates
        estimates['xG90'] = estimates.apply(estimate_xg90, axis=1)
        estimates['xA90'] = estimates.apply(estimate_xa90, axis=1)
        
        return estimates[['xG90', 'xA90']]
    
    def _apply_form_weighting(self, players_df: pd.DataFrame,
                            live_data: pd.DataFrame,
                            target_gameweek: int) -> pd.DataFrame:
        """
        Apply form weighting to blend recent performance with season averages
        
        Phase 1 implementation:
        - Calculate recent form metrics (last 3-5 GWs)
        - Apply form_weight blend (70% form + 30% season)
        - Add momentum indicators
        """
        try:
            # Get recent performance data for form calculation
            recent_performance = self._calculate_recent_form(live_data, target_gameweek)
            
            if recent_performance.empty:
                if self.debug:
                    print("📈 No recent form data available - using season averages")
                return players_df
            
            # Merge form data with players
            players_with_form = players_df.merge(
                recent_performance[['player_id', 'form_multiplier', 'momentum', 'recent_points_per_game', 'form_trend']],
                on='player_id',
                how='left'
            )
            
            # Fill missing form data with neutral values
            players_with_form['form_multiplier'] = players_with_form['form_multiplier'].fillna(1.0)
            players_with_form['momentum'] = players_with_form['momentum'].fillna('➡️')
            players_with_form['recent_points_per_game'] = players_with_form['recent_points_per_game'].fillna(0.0)
            players_with_form['form_trend'] = players_with_form['form_trend'].fillna(0.0)
            
            # Apply form weighting to xG/xA rates
            # Blend: 70% form-adjusted + 30% season baseline
            season_weight = 1 - self.form_weight
            
            # Adjust xG/xA based on recent form performance
            players_with_form['xG90_form_adjusted'] = (
                players_with_form['xG90'] * 
                (self.form_weight * players_with_form['form_multiplier'] + season_weight)
            )
            
            players_with_form['xA90_form_adjusted'] = (
                players_with_form['xA90'] * 
                (self.form_weight * players_with_form['form_multiplier'] + season_weight)
            )
            
            # Use form-adjusted rates for XP calculation
            players_with_form['xG90'] = players_with_form['xG90_form_adjusted']
            players_with_form['xA90'] = players_with_form['xA90_form_adjusted']
            
            if self.debug:
                form_players = len(players_with_form[players_with_form['form_multiplier'] != 1.0])
                avg_multiplier = players_with_form['form_multiplier'].mean()
                print(f"📈 Applied form weighting to {form_players} players (avg multiplier: {avg_multiplier:.2f})")
            
            return players_with_form
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Form weighting failed: {e} - using season averages")
            return players_df
    
    def _calculate_recent_form(self, live_data: pd.DataFrame, target_gameweek: int) -> pd.DataFrame:
        """
        Calculate recent form metrics for players
        
        Returns DataFrame with:
        - player_id
        - form_multiplier (0.5-2.0 range)
        - momentum (emoji indicator)
        - recent_points_per_game
        - form_trend (points delta)
        """
        if live_data.empty:
            return pd.DataFrame()
        
        try:
            # Get data for form window (last 3-5 gameweeks before target)
            form_start_gw = max(1, target_gameweek - self.form_window)
            form_end_gw = target_gameweek - 1
            
            # Filter live data for form window
            form_data = live_data[
                (live_data['event'] >= form_start_gw) & 
                (live_data['event'] <= form_end_gw)
            ].copy()
            
            if form_data.empty:
                return pd.DataFrame()
            
            # Calculate form metrics by player
            player_form = form_data.groupby('player_id').agg({
                'total_points': ['sum', 'count', 'mean'],
                'event': 'count'
            }).round(2)
            
            # Flatten column names
            player_form.columns = ['total_points_sum', 'points_count', 'recent_points_per_game', 'gameweeks_played']
            player_form = player_form.reset_index()
            
            # Calculate form multiplier based on recent performance
            # Use points per game relative to position expectations
            def calculate_form_multiplier(recent_ppg, gameweeks_played):
                """
                Convert recent points per game to form multiplier
                
                Logic:
                - Excellent form (8+ PPG): 1.5-2.0x multiplier
                - Good form (5-8 PPG): 1.1-1.5x multiplier  
                - Average form (3-5 PPG): 0.9-1.1x multiplier
                - Poor form (1-3 PPG): 0.6-0.9x multiplier
                - Very poor form (<1 PPG): 0.5-0.6x multiplier
                """
                if gameweeks_played < 2:  # Need at least 2 games for reliable form
                    return 1.0
                
                if recent_ppg >= 8.0:
                    return min(2.0, 1.5 + (recent_ppg - 8) * 0.1)  # Cap at 2.0x
                elif recent_ppg >= 5.0:
                    return 1.1 + (recent_ppg - 5) * 0.133  # 1.1 to 1.5
                elif recent_ppg >= 3.0:
                    return 0.9 + (recent_ppg - 3) * 0.1  # 0.9 to 1.1
                elif recent_ppg >= 1.0:
                    return 0.6 + (recent_ppg - 1) * 0.15  # 0.6 to 0.9
                else:
                    return max(0.5, 0.5 + recent_ppg * 0.1)  # 0.5 to 0.6
            
            player_form['form_multiplier'] = player_form.apply(
                lambda row: calculate_form_multiplier(row['recent_points_per_game'], row['gameweeks_played']), 
                axis=1
            )
            
            # Calculate momentum indicators
            def get_momentum_indicator(recent_ppg, form_multiplier):
                """Get emoji momentum indicator"""
                if form_multiplier >= 1.4:
                    return '🔥'  # Hot
                elif form_multiplier >= 1.15:
                    return '📈'  # Rising
                elif form_multiplier >= 0.85:
                    return '➡️'  # Stable
                elif form_multiplier >= 0.7:
                    return '📉'  # Declining
                else:
                    return '❄️'  # Cold
            
            player_form['momentum'] = player_form.apply(
                lambda row: get_momentum_indicator(row['recent_points_per_game'], row['form_multiplier']),
                axis=1
            )
            
            # Calculate form trend (recent vs season average if available)
            # For now, use recent PPG as trend indicator
            player_form['form_trend'] = player_form['recent_points_per_game'] - 4.0  # 4.0 as baseline
            
            if self.debug:
                hot_players = len(player_form[player_form['momentum'] == '🔥'])
                cold_players = len(player_form[player_form['momentum'] == '❄️'])
                print(f"🔥 Hot players: {hot_players}, ❄️ Cold players: {cold_players}")
            
            return player_form
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Form calculation failed: {e}")
            return pd.DataFrame()
    
    def _calculate_expected_minutes(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate expected minutes using enhanced model
        
        Incorporates:
        - SBP-based start probability
        - Availability status (injury/suspension/doubt)
        - Position and price-based durability
        - Manager rotation patterns (TODO: Phase 2)
        """
        def get_start_probability(sbp, availability):
            """Convert SBP to start probability with availability adjustments"""
            def sbp_to_prob(sbp_val):
                if sbp_val >= 40: return 0.95
                elif sbp_val >= 20: return 0.85
                elif sbp_val >= 10: return 0.70
                elif sbp_val >= 5: return 0.55
                elif sbp_val >= 2: return 0.35
                elif sbp_val >= 0.5: return 0.20
                else: return 0.05
            
            # Handle availability status
            if availability in ['i', 's', 'u']:  # injured, suspended, unavailable
                return 0.0
            elif availability == 'd':  # doubtful
                return sbp_to_prob(sbp) * 0.5
            else:  # available
                return sbp_to_prob(sbp)
        
        def calculate_expected_minutes_enhanced(row):
            """Enhanced minutes calculation with probabilistic scenarios"""
            # Get start probability
            sbp_col = 'selected_by_percent' if 'selected_by_percent' in row.index else 'selected_by_percentage'
            sbp = row.get(sbp_col, 5.0)
            availability = row.get('status', 'a')
            
            p_start = get_start_probability(sbp, availability)
            
            if availability in ['i', 's', 'u']:
                return 0  # No minutes if unavailable
            
            # Position and price-based durability
            price = row.get('price', row.get('price_gbp', 5.0))
            
            if row['position'] == 'GKP':
                avg_mins_when_started = 90
                p_full_game_given_start = 0.95
                p_sub_given_no_start = 0.05
            elif price >= 7.0:  # Premium outfield players
                avg_mins_when_started = 85
                p_full_game_given_start = 0.80
                p_sub_given_no_start = 0.25
            elif price >= 5.0:  # Mid-tier players
                avg_mins_when_started = 75
                p_full_game_given_start = 0.70
                p_sub_given_no_start = 0.30
            else:  # Budget players
                avg_mins_when_started = 70
                p_full_game_given_start = 0.60
                p_sub_given_no_start = 0.35
            
            # Availability adjustments
            if availability == 'd':  # doubtful
                p_full_game_given_start *= 0.7
                avg_mins_when_started *= 0.8
            
            # Calculate scenario probabilities
            p_full_game = p_start * p_full_game_given_start
            p_partial_start = p_start * (1 - p_full_game_given_start)
            p_sub_appearance = (1 - p_start) * p_sub_given_no_start
            
            # Expected minutes calculation
            expected_minutes = (
                p_full_game * avg_mins_when_started +
                p_partial_start * 45 +
                p_sub_appearance * 20
            )
            
            return expected_minutes
        
        # Calculate for all players
        players_df['expected_minutes'] = players_df.apply(calculate_expected_minutes_enhanced, axis=1)
        
        # Also calculate useful probability metrics
        players_df['p_60_plus_mins'] = players_df.apply(
            lambda row: get_start_probability(
                row.get('selected_by_percent', row.get('selected_by_percentage', 5.0)), 
                row.get('status', 'a')
            ) * (0.80 if row.get('price', row.get('price_gbp', 5.0)) >= 7.0 else 0.70), 
            axis=1
        )
        
        return players_df
    
    def _calculate_fixture_difficulty(self, fixtures_df: pd.DataFrame,
                                    teams_df: pd.DataFrame,
                                    team_strength: Dict[str, float],
                                    target_gameweek: int,
                                    gameweeks_ahead: int) -> Dict[str, float]:
        """
        Calculate fixture difficulty for target gameweek(s)
        
        Returns dict mapping team name to difficulty multiplier [0.7, 1.3]
        """
        team_difficulty = {}
        
        for gw_offset in range(gameweeks_ahead):
            current_gw = target_gameweek + gw_offset
            gw_fixtures = fixtures_df[fixtures_df['event'] == current_gw].copy()
            
            if gw_fixtures.empty:
                continue
            
            # Add team names to fixtures
            team_id_col = 'team_id' if 'team_id' in teams_df.columns else 'id'
            
            gw_fixtures = gw_fixtures.merge(
                teams_df[[team_id_col, 'name']], 
                left_on='home_team_id', 
                right_on=team_id_col, 
                how='left'
            ).rename(columns={'name': 'home_team'}).drop(team_id_col, axis=1)
            
            gw_fixtures = gw_fixtures.merge(
                teams_df[[team_id_col, 'name']], 
                left_on='away_team_id', 
                right_on=team_id_col, 
                how='left'
            ).rename(columns={'name': 'away_team'}).drop(team_id_col, axis=1)
            
            # Calculate difficulty based on opponent strength
            for _, fixture in gw_fixtures.iterrows():
                home_team = fixture['home_team']
                away_team = fixture['away_team']
                
                # Home team difficulty = inverse of away team strength (with home advantage)
                away_strength = team_strength.get(away_team, 1.0)
                home_difficulty = 2.0 - away_strength  # Easier opponent = higher difficulty multiplier
                
                # Away team difficulty = inverse of home team strength (without home advantage)
                home_strength = team_strength.get(home_team, 1.0)
                away_difficulty = 2.0 - home_strength
                
                # Clip to valid range [0.7, 1.3]
                home_difficulty = np.clip(home_difficulty, 0.7, 1.3)
                away_difficulty = np.clip(away_difficulty, 0.7, 1.3)
                
                # For multi-gameweek, average the difficulties
                if home_team in team_difficulty:
                    team_difficulty[home_team] = (team_difficulty[home_team] + home_difficulty) / 2
                else:
                    team_difficulty[home_team] = home_difficulty
                
                if away_team in team_difficulty:
                    team_difficulty[away_team] = (team_difficulty[away_team] + away_difficulty) / 2
                else:
                    team_difficulty[away_team] = away_difficulty
        
        return team_difficulty
    
    def _apply_fixture_difficulty(self, players_df: pd.DataFrame,
                                fixture_difficulty: Dict[str, float]) -> pd.DataFrame:
        """Apply fixture difficulty multipliers to players"""
        players_df['fixture_difficulty'] = players_df['name'].map(fixture_difficulty).fillna(1.0)
        return players_df
    
    def _calculate_xp_components(self, players_df: pd.DataFrame, 
                               gameweeks_ahead: int) -> pd.DataFrame:
        """
        Calculate expected points components
        
        Components:
        - Appearance points (1 for >0 mins, 2 for >=60 mins)
        - Goals (position multipliers: GK/DEF=6, MID=5, FWD=4)
        - Assists (3 points each)
        - Clean sheets (GK/DEF=4, MID=1, FWD=0)
        """
        # Expected contributions per gameweek
        xG_per_gw = (players_df['xG90'] / 90) * players_df['expected_minutes'] * players_df['fixture_difficulty']
        xA_per_gw = (players_df['xA90'] / 90) * players_df['expected_minutes'] * players_df['fixture_difficulty']
        
        # Appearance points
        appearance_pts_per_gw = (
            players_df['p_60_plus_mins'] * 2 +
            (players_df.get('p_any_appearance', players_df['expected_minutes'] / 90).clip(0, 1) - players_df['p_60_plus_mins']) * 1
        )
        
        # Goal points by position
        goal_multipliers = {'GKP': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
        goal_pts_per_gw = xG_per_gw * players_df['position'].map(goal_multipliers)
        
        # Assist points
        assist_pts_per_gw = xA_per_gw * 3
        
        # Clean sheet points (simplified model)
        cs_multipliers = {'GKP': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
        base_cs_prob = 0.3  # 30% base clean sheet probability
        cs_prob_per_gw = (
            players_df['p_60_plus_mins'] * 
            players_df['fixture_difficulty'] * 
            base_cs_prob
        )
        cs_pts_per_gw = cs_prob_per_gw * players_df['position'].map(cs_multipliers)
        
        # Total XP per gameweek
        xp_per_gw = appearance_pts_per_gw + goal_pts_per_gw + assist_pts_per_gw + cs_pts_per_gw
        
        # Apply multi-gameweek scaling if needed
        if gameweeks_ahead > 1:
            # Simple scaling for now - could add temporal weighting in future
            players_df['xP'] = xp_per_gw * gameweeks_ahead
        else:
            players_df['xP'] = xp_per_gw
        
        # Store component breakdown for analysis
        players_df['xP_appearance'] = appearance_pts_per_gw * gameweeks_ahead
        players_df['xP_goals'] = goal_pts_per_gw * gameweeks_ahead
        players_df['xP_assists'] = assist_pts_per_gw * gameweeks_ahead
        players_df['xP_clean_sheets'] = cs_pts_per_gw * gameweeks_ahead
        
        return players_df


# Utility functions for backward compatibility
def calculate_expected_points_single_gw(players_data: pd.DataFrame,
                                      teams_data: pd.DataFrame,
                                      xg_rates_data: pd.DataFrame,
                                      fixtures_data: pd.DataFrame,
                                      target_gameweek: int,
                                      live_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate expected points for single gameweek (backward compatibility)
    """
    model = XPModel(debug=False)
    return model.calculate_expected_points(
        players_data=players_data,
        teams_data=teams_data,
        xg_rates_data=xg_rates_data,
        fixtures_data=fixtures_data,
        target_gameweek=target_gameweek,
        live_data=live_data,
        gameweeks_ahead=1
    )


def calculate_expected_points_multi_gw(players_data: pd.DataFrame,
                                     teams_data: pd.DataFrame,
                                     xg_rates_data: pd.DataFrame,
                                     fixtures_data: pd.DataFrame,
                                     target_gameweek: int,
                                     gameweeks_ahead: int = 5) -> pd.DataFrame:
    """
    Calculate expected points for multiple gameweeks (backward compatibility)
    """
    model = XPModel(debug=False)
    return model.calculate_expected_points(
        players_data=players_data,
        teams_data=teams_data,
        xg_rates_data=xg_rates_data,
        fixtures_data=fixtures_data,
        target_gameweek=target_gameweek,
        live_data=None,
        gameweeks_ahead=gameweeks_ahead
    )