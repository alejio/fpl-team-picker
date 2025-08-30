"""
FPL Utilities Module

Common utility functions used across FPL gameweek management modules including:
- DataFrame column validation and safe access
- Data formatting and display helpers
- Common filtering and validation operations
"""

import pandas as pd
from typing import List


def get_safe_columns(df: pd.DataFrame, preferred_columns: List[str]) -> List[str]:
    """
    Get columns that exist in the DataFrame, with fallback to first few columns
    
    Args:
        df: DataFrame to check columns for
        preferred_columns: List of preferred column names
        
    Returns:
        List of safe column names that exist in the DataFrame
    """
    if df.empty:
        return preferred_columns[:3]  # Return first 3 as fallback
    
    available_columns = list(df.columns)
    safe_columns = []
    
    for col in preferred_columns:
        if col in available_columns:
            safe_columns.append(col)
    
    # Fallback to first 3 columns if none of preferred are found
    return safe_columns if safe_columns else available_columns[:3]


def create_display_dataframe(df: pd.DataFrame, 
                           core_columns: List[str],
                           optional_columns: List[str] = None,
                           sort_by: str = None,
                           ascending: bool = False,
                           round_decimals: int = 2) -> pd.DataFrame:
    """
    Create a cleaned DataFrame for display with safe column handling
    
    Args:
        df: Source DataFrame
        core_columns: Essential columns that must be included
        optional_columns: Optional columns to include if available
        sort_by: Column to sort by (if available)
        ascending: Sort direction
        round_decimals: Number of decimal places for rounding
        
    Returns:
        Cleaned DataFrame ready for display
    """
    if df.empty:
        return pd.DataFrame()
    
    # Get safe core columns
    display_columns = get_safe_columns(df, core_columns)
    
    # Add optional columns that exist
    if optional_columns:
        for col in optional_columns:
            if col in df.columns and col not in display_columns:
                display_columns.append(col)
    
    # Create display DataFrame
    display_df = df[display_columns].copy()
    
    # Round numeric columns
    numeric_columns = display_df.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 0:
        display_df[numeric_columns] = display_df[numeric_columns].round(round_decimals)
    
    # Sort if column is available
    if sort_by and sort_by in display_df.columns:
        display_df = display_df.sort_values(sort_by, ascending=ascending)
    
    return display_df
