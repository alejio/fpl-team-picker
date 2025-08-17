"""
Prediction Storage System for FPL Retro Analysis

This module handles saving and loading predictions from the main xP model
for future comparison with actual results in retrospective analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


class PredictionStorage:
    """Handle storage and retrieval of FPL predictions for retro analysis"""
    
    def __init__(self, storage_dir="predictions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
    def save_gameweek_predictions(self, gameweek, players_xp_df, team_selections=None, metadata=None):
        """
        Save predictions for a specific gameweek
        
        Args:
            gameweek (int): Gameweek number
            players_xp_df (pd.DataFrame): Players with xP predictions
            team_selections (dict, optional): Optimal team selections
            metadata (dict, optional): Model parameters and settings used
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main predictions
        predictions_file = self.storage_dir / f"gw{gameweek}_predictions_{timestamp}.csv"
        
        # Select key columns for storage
        prediction_columns = [
            'player_id', 'web_name', 'position', 'name', 'price_gbp',
            'xP', 'total_weighted_xP', 'total_xP', 
            'xP_gw1', 'xP_gw2', 'xP_gw3', 'xP_gw4', 'xP_gw5',
            'expected_minutes', 'p_start', 'p_60_plus_mins', 'p_any_appearance',
            'transfer_risk', 'fixture_volatility', 'selected_by_percentage'
        ]
        
        # Filter to available columns
        available_columns = [col for col in prediction_columns if col in players_xp_df.columns]
        predictions_data = players_xp_df[available_columns].copy()
        
        # Add metadata
        predictions_data['gameweek'] = gameweek
        predictions_data['prediction_timestamp'] = timestamp
        
        # Save predictions
        predictions_data.to_csv(predictions_file, index=False)
        print(f"Saved predictions for GW{gameweek}: {predictions_file}")
        
        # Save team selections if provided
        if team_selections:
            team_file = self.storage_dir / f"gw{gameweek}_optimal_team_{timestamp}.json"
            with open(team_file, 'w') as f:
                json.dump(team_selections, f, indent=2, default=str)
            print(f"Saved optimal team for GW{gameweek}: {team_file}")
        
        # Save metadata if provided
        if metadata:
            metadata_file = self.storage_dir / f"gw{gameweek}_metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"Saved metadata for GW{gameweek}: {metadata_file}")
        
        return {
            'predictions_file': str(predictions_file),
            'team_file': str(team_file) if team_selections else None,
            'metadata_file': str(metadata_file) if metadata else None,
            'timestamp': timestamp
        }
    
    def load_gameweek_predictions(self, gameweek, timestamp=None):
        """
        Load predictions for a specific gameweek
        
        Args:
            gameweek (int): Gameweek number
            timestamp (str, optional): Specific timestamp, uses latest if None
            
        Returns:
            dict: Contains predictions, team, and metadata DataFrames/dicts
        """
        # Find prediction files for the gameweek
        pattern = f"gw{gameweek}_predictions_*.csv"
        prediction_files = list(self.storage_dir.glob(pattern))
        
        if not prediction_files:
            print(f"No predictions found for GW{gameweek}")
            return None
        
        # Use specific timestamp or latest
        if timestamp:
            target_file = self.storage_dir / f"gw{gameweek}_predictions_{timestamp}.csv"
            if not target_file.exists():
                print(f"Prediction file with timestamp {timestamp} not found")
                return None
        else:
            # Use the latest file
            target_file = sorted(prediction_files)[-1]
            timestamp = target_file.stem.split('_')[-1]
        
        # Load predictions
        predictions_df = pd.read_csv(target_file)
        
        # Load team selections if available
        team_file = self.storage_dir / f"gw{gameweek}_optimal_team_{timestamp}.json"
        team_data = None
        if team_file.exists():
            with open(team_file, 'r') as f:
                team_data = json.load(f)
        
        # Load metadata if available
        metadata_file = self.storage_dir / f"gw{gameweek}_metadata_{timestamp}.json"
        metadata = None
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        print(f"Loaded predictions for GW{gameweek} from {target_file}")
        
        return {
            'predictions': predictions_df,
            'team_selections': team_data,
            'metadata': metadata,
            'timestamp': timestamp,
            'file_path': str(target_file)
        }
    
    def list_available_predictions(self):
        """List all available prediction files"""
        prediction_files = list(self.storage_dir.glob("gw*_predictions_*.csv"))
        
        if not prediction_files:
            return pd.DataFrame(columns=['gameweek', 'timestamp', 'file_path'])
        
        # Parse filenames to extract gameweek and timestamp
        file_info = []
        for file_path in prediction_files:
            filename = file_path.stem
            parts = filename.split('_')
            gameweek = int(parts[0][2:])  # Remove 'gw' prefix
            timestamp = parts[-1]
            
            file_info.append({
                'gameweek': gameweek,
                'timestamp': timestamp,
                'file_path': str(file_path),
                'file_size_kb': file_path.stat().st_size / 1024
            })
        
        return pd.DataFrame(file_info).sort_values(['gameweek', 'timestamp'])
    
    def compare_predictions_with_actuals(self, gameweek, historical_data_df, timestamp=None):
        """
        Compare stored predictions with actual results
        
        Args:
            gameweek (int): Gameweek to analyze
            historical_data_df (pd.DataFrame): Actual results from historical data
            timestamp (str, optional): Specific prediction timestamp
            
        Returns:
            pd.DataFrame: Merged prediction vs actual data
        """
        # Load predictions
        prediction_data = self.load_gameweek_predictions(gameweek, timestamp)
        if not prediction_data:
            return None
        
        predictions_df = prediction_data['predictions']
        
        # Filter historical data for the gameweek
        actual_data = historical_data_df[historical_data_df['GW'] == gameweek].copy()
        
        if len(actual_data) == 0:
            print(f"No actual data found for GW{gameweek}")
            return None
        
        # Merge predictions with actuals
        # Try multiple join strategies for robustness
        
        # First, try exact player_id match
        merged = predictions_df.merge(
            actual_data,
            left_on='player_id',
            right_on='element',  # FPL API uses 'element' for player_id
            how='inner',
            suffixes=('_pred', '_actual')
        )
        
        if len(merged) == 0:
            # Try name-based matching as fallback
            merged = predictions_df.merge(
                actual_data,
                left_on='web_name',
                right_on='name',
                how='inner',
                suffixes=('_pred', '_actual')
            )
        
        if len(merged) == 0:
            print(f"Could not match predictions with actual data for GW{gameweek}")
            return None
        
        # Calculate comparison metrics
        merged['prediction_error'] = merged['total_points'] - merged['xP']
        merged['absolute_error'] = abs(merged['prediction_error'])
        merged['percentage_error'] = np.where(
            merged['total_points'] != 0,
            (merged['prediction_error'] / merged['total_points']) * 100,
            0
        )
        
        print(f"Successfully merged {len(merged)} players for GW{gameweek} comparison")
        
        return merged
    
    def generate_accuracy_report(self, gameweek, historical_data_df, timestamp=None):
        """
        Generate a comprehensive accuracy report for a gameweek
        
        Args:
            gameweek (int): Gameweek to analyze
            historical_data_df (pd.DataFrame): Actual results
            timestamp (str, optional): Specific prediction timestamp
            
        Returns:
            dict: Comprehensive accuracy metrics and insights
        """
        comparison_data = self.compare_predictions_with_actuals(
            gameweek, historical_data_df, timestamp
        )
        
        if comparison_data is None:
            return None
        
        # Calculate comprehensive metrics
        report = {
            'gameweek': gameweek,
            'timestamp': timestamp,
            'total_players': len(comparison_data),
            'mean_absolute_error': comparison_data['absolute_error'].mean(),
            'rmse': np.sqrt((comparison_data['prediction_error'] ** 2).mean()),
            'correlation': comparison_data['xP'].corr(comparison_data['total_points']),
            'prediction_bias': comparison_data['prediction_error'].mean(),
            'mean_predicted_xp': comparison_data['xP'].mean(),
            'mean_actual_points': comparison_data['total_points'].mean(),
        }
        
        # Position-based analysis
        position_analysis = comparison_data.groupby('position').agg({
            'prediction_error': ['mean', 'std'],
            'absolute_error': 'mean',
            'xP': 'mean',
            'total_points': 'mean',
            'web_name': 'count'
        }).round(3)
        
        report['position_analysis'] = position_analysis.to_dict()
        
        # Top prediction errors
        worst_predictions = comparison_data.nlargest(10, 'absolute_error')[
            ['web_name', 'position', 'name_pred', 'xP', 'total_points', 'prediction_error', 'absolute_error']
        ]
        report['worst_predictions'] = worst_predictions.to_dict('records')
        
        # Best predictions
        best_predictions = comparison_data.nsmallest(10, 'absolute_error')[
            ['web_name', 'position', 'name_pred', 'xP', 'total_points', 'prediction_error', 'absolute_error']
        ]
        report['best_predictions'] = best_predictions.to_dict('records')
        
        return report


def save_predictions_from_model(players_xp_df, optimal_team, gameweek, model_params=None):
    """
    Convenience function to save predictions from the main model
    
    Args:
        players_xp_df (pd.DataFrame): Players with xP predictions
        optimal_team (list): Optimal team selection
        gameweek (int): Current gameweek
        model_params (dict, optional): Model parameters used
    """
    storage = PredictionStorage()
    
    # Prepare team selections data
    team_data = {
        'optimal_15': optimal_team,
        'gameweek': gameweek,
        'total_cost': sum(p['price_gbp'] for p in optimal_team),
        'total_xp': sum(p['xP'] for p in optimal_team)
    }
    
    # Prepare metadata
    metadata = {
        'gameweek': gameweek,
        'model_version': 'v0.1_mvp',
        'prediction_date': datetime.now().isoformat(),
        'model_params': model_params or {},
        'player_count': len(players_xp_df),
        'valid_xp_count': players_xp_df['xP'].notna().sum()
    }
    
    # Save all data
    result = storage.save_gameweek_predictions(
        gameweek=gameweek,
        players_xp_df=players_xp_df,
        team_selections=team_data,
        metadata=metadata
    )
    
    return result


if __name__ == "__main__":
    # Example usage and testing
    storage = PredictionStorage()
    
    # List available predictions
    available = storage.list_available_predictions()
    print("Available predictions:")
    print(available)