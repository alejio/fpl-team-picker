"""
FPL Utilities Module

Common utility functions used across FPL gameweek management modules including:
- DataFrame column validation and safe access
- Data formatting and display helpers
- Common filtering and validation operations
"""

import pandas as pd
from typing import List, Optional, Dict, Any


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


def get_safe_columns_with_fallback(df: pd.DataFrame, preferred_columns: List[str], min_columns: int = 3) -> List[str]:
    """
    Get columns that exist in the DataFrame with guaranteed minimum number
    
    Args:
        df: DataFrame to check columns for  
        preferred_columns: List of preferred column names
        min_columns: Minimum number of columns to return
        
    Returns:
        List of safe column names that exist in the DataFrame
    """
    if df.empty:
        return preferred_columns[:min_columns]
    
    available_columns = list(df.columns)
    safe_columns = []
    
    # Add preferred columns that exist
    for col in preferred_columns:
        if col in available_columns:
            safe_columns.append(col)
    
    # Add additional columns to meet minimum if needed
    while len(safe_columns) < min_columns and len(safe_columns) < len(available_columns):
        for col in available_columns:
            if col not in safe_columns:
                safe_columns.append(col)
                break
    
    return safe_columns[:min_columns] if len(safe_columns) > min_columns else safe_columns


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, bool]:
    """
    Validate that DataFrame contains required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Dictionary mapping column names to presence (True/False)
    """
    if df.empty:
        return {col: False for col in required_columns}
    
    available_columns = set(df.columns)
    return {col: col in available_columns for col in required_columns}


def safe_column_access(df: pd.DataFrame, column: str, default_value: Any = None) -> pd.Series:
    """
    Safely access DataFrame column with default fallback
    
    Args:
        df: DataFrame to access column from
        column: Column name to access
        default_value: Default value if column doesn't exist
        
    Returns:
        Column Series or Series filled with default_value
    """
    if df.empty or column not in df.columns:
        return pd.Series([default_value] * len(df), index=df.index if not df.empty else [])
    
    return df[column]


def filter_columns_by_availability(df: pd.DataFrame, columns_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Filter column groups by availability in DataFrame
    
    Args:
        df: DataFrame to check columns against
        columns_dict: Dictionary mapping category names to column lists
        
    Returns:
        Dictionary with only available columns for each category
    """
    if df.empty:
        return {category: [] for category in columns_dict.keys()}
    
    available_columns = set(df.columns)
    filtered_dict = {}
    
    for category, column_list in columns_dict.items():
        filtered_dict[category] = [col for col in column_list if col in available_columns]
    
    return filtered_dict


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


def format_currency(value: float, decimals: int = 1) -> str:
    """Format value as currency string"""
    return f"£{value:.{decimals}f}m"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage string"""
    return f"{value:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers with default fallback for division by zero"""
    return numerator / denominator if denominator != 0 else default