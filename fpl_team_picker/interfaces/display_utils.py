"""
Display utilities for the presentation layer.

Helper functions for formatting and resolving data for UI display in Marimo notebooks.
"""

import pandas as pd
from typing import List, Any, Optional
from pydantic import BaseModel, Field, ValidationError, model_validator


class DataContractError(Exception):
    """Raised when interface data contracts are violated."""

    pass


class InterfaceTeamData(BaseModel):
    """Pydantic model for interface team data validation."""

    # Flexible team ID field
    id: Optional[int] = Field(None, ge=1, le=20)
    team_id: Optional[int] = Field(None, ge=1, le=20)
    name: str = Field(..., min_length=1)

    # Additional team fields from FPL data
    short_name: Optional[str] = Field(
        None, description="Team short name (e.g., ARS, LIV)"
    )
    as_of_utc: Optional[Any] = Field(None, description="Data timestamp")

    @model_validator(mode="after")
    def validate_at_least_one_id(self):
        """Ensure at least one ID field is provided."""
        if self.id is None and self.team_id is None:
            raise ValueError("Either 'id' or 'team_id' must be provided")
        return self

    @property
    def effective_team_id(self) -> int:
        """Get the effective team ID regardless of field name."""
        return self.id if self.id is not None else self.team_id

    class Config:
        """Validation configuration."""

        extra = "forbid"


def validate_teams_dataframe(
    teams_df: pd.DataFrame, context: str = "team data"
) -> List[InterfaceTeamData]:
    """
    Validate teams DataFrame using Pydantic automatic validation.

    Args:
        teams_df: Teams DataFrame to validate
        context: Context for error messages

    Returns:
        List of validated InterfaceTeamData objects

    Raises:
        DataContractError: If validation fails
    """
    if teams_df.empty:
        raise DataContractError(f"Empty teams DataFrame in {context}")

    validated_teams = []
    validation_errors = []

    for _, row in teams_df.iterrows():
        try:
            # Let Pydantic handle validation automatically
            team = InterfaceTeamData.model_validate(row.to_dict())
            validated_teams.append(team)
        except ValidationError as e:
            validation_errors.append(f"Team {row.get('name', 'unknown')}: {str(e)}")

    if validation_errors:
        raise DataContractError(
            f"Team validation failed in {context}: {validation_errors[:3]}"
        )

    print(f"✅ Validated {len(validated_teams)} teams using Pydantic")
    return validated_teams


def resolve_team_names_pydantic(
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    context: str = "team name resolution",
) -> pd.DataFrame:
    """
    Resolve team names using Pydantic automatic validation.

    Maps team IDs in players DataFrame to team names for display purposes.

    Args:
        players_df: Players DataFrame with team_id or team column
        teams_df: Teams DataFrame with id/team_id and name columns
        context: Context for error messages

    Returns:
        DataFrame with validated team names added in 'name' column

    Raises:
        DataContractError: If resolution fails
    """
    # Validate teams first using Pydantic
    validated_teams = validate_teams_dataframe(teams_df, context)

    # Create team mapping from validated team data
    team_mapping = {team.effective_team_id: team.name for team in validated_teams}

    # Apply team names to player data
    result_df = players_df.copy()

    # Determine team column
    team_col = None
    if "team_id" in result_df.columns:
        team_col = "team_id"
    elif "team" in result_df.columns:
        team_col = "team"
    else:
        raise DataContractError(
            f"No team column found in {context}. Available: {list(result_df.columns)[:10]}"
        )

    # Apply mapping with validation
    result_df["name"] = result_df[team_col].map(team_mapping)

    # Check for unmapped teams
    unmapped = result_df["name"].isna().sum()
    if unmapped > 0:
        missing_team_ids = result_df[result_df["name"].isna()][team_col].unique()
        raise DataContractError(
            f"Team mapping failed in {context}: {unmapped} players unmapped. "
            f"Missing team IDs: {missing_team_ids}"
        )

    print(
        f"✅ Resolved team names for {len(result_df)} players using Pydantic validation"
    )
    return result_df
