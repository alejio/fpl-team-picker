"""
Interface-layer data contract validation using Pydantic models.

Leverages existing Pydantic domain models and validation patterns to enforce
data contracts at the presentation layer using fail-fast principles.
"""

import pandas as pd
from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field, ValidationError, model_validator
from fpl_team_picker.domain.models.player import (
    Position,
)


class DataContractError(Exception):
    """Raised when interface data contracts are violated."""

    pass


class InterfacePlayerData(BaseModel):
    """Pydantic model for interface player data validation."""

    player_id: int = Field(..., gt=0)
    web_name: str = Field(..., min_length=1)
    position: Position = Field(...)
    price: float = Field(..., ge=3.9, le=15.0)

    # Optional XP fields
    xP: Optional[float] = Field(None, ge=0.0)
    xP_5gw: Optional[float] = Field(None, ge=0.0)

    # Team identification (flexible)
    team_id: Optional[int] = Field(None, ge=1, le=20)
    team: Optional[Union[int, str]] = Field(None)
    name: Optional[str] = Field(None)  # Team name

    class Config:
        """Allow extra fields for forward compatibility."""

        extra = "allow"


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


class TransferConstraintPlayer(BaseModel):
    """Pydantic model for transfer constraint player validation."""

    player_id: int = Field(..., gt=0)
    web_name: str = Field(..., min_length=1)
    position: Position = Field(...)
    price: float = Field(..., ge=3.9, le=15.0)
    xP: float = Field(..., ge=0.0)

    # Require team identification
    team_name: str = Field(..., min_length=1, description="Team name for display")

    @classmethod
    def from_player_row(
        cls, player_row: pd.Series, team_name: str
    ) -> "TransferConstraintPlayer":
        """Create from pandas Series with validated team name."""
        return cls(
            player_id=int(player_row["player_id"]),
            web_name=str(player_row["web_name"]),
            position=Position(player_row["position"]),
            price=float(player_row["price"]),
            xP=float(player_row.get("xP", player_row.get("xP_5gw", 0))),
            team_name=team_name,
        )

    def to_display_label(self, horizon_label: str = "XP") -> str:
        """Generate display label for UI components."""
        return f"{self.web_name} ({self.position.value}, {self.team_name}) - £{self.price:.1f}m, {self.xP:.2f} {horizon_label}"


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


def create_transfer_constraint_players(
    players_df: pd.DataFrame,
    teams: List[InterfaceTeamData],
    sort_column: str = "xP",
    context: str = "transfer constraints",
) -> List[TransferConstraintPlayer]:
    """
    Create validated transfer constraint players using Pydantic automatic validation.

    Args:
        players_df: Players DataFrame
        teams: Validated team data
        sort_column: Column to use for XP value
        context: Context for error messages

    Returns:
        List of validated TransferConstraintPlayer objects

    Raises:
        DataContractError: If validation fails
    """
    if players_df.empty:
        raise DataContractError(f"Empty players DataFrame in {context}")

    # Create team mapping using validated team data
    team_mapping = {team.effective_team_id: team.name for team in teams}

    validated_players = []
    validation_errors = []

    for _, row in players_df.iterrows():
        try:
            # Get team ID from player data
            team_id = row.get("team_id") or row.get("team")
            if team_id is None:
                raise ValueError("No team identification found")

            # Get team name from mapping
            team_name = team_mapping.get(int(team_id))
            if team_name is None:
                # Check if 'name' column already exists (already resolved)
                team_name = row.get("name")
                if team_name is None:
                    raise ValueError(f"Team name not found for team_id {team_id}")

            # Use the specified sort column for XP value
            xp_value = row.get(sort_column, row.get("xP", 0))

            # Let Pydantic handle validation automatically
            player = TransferConstraintPlayer.from_player_row(row, team_name)
            # Override XP with specified column value
            player.xP = float(xp_value)
            validated_players.append(player)

        except (ValidationError, ValueError) as e:
            validation_errors.append(
                f"Player {row.get('web_name', 'unknown')}: {str(e)}"
            )

    if (
        validation_errors and len(validation_errors) > len(players_df) * 0.1
    ):  # >10% failure
        raise DataContractError(
            f"Too many validation errors in {context}: {validation_errors[:5]}"
        )

    if validation_errors:
        print(
            f"⚠️ {len(validation_errors)} players failed validation: {validation_errors[:3]}"
        )

    print(f"✅ Created {len(validated_players)} validated transfer constraint players")
    return validated_players


def resolve_team_names_pydantic(
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    context: str = "team name resolution",
) -> pd.DataFrame:
    """
    Resolve team names using Pydantic automatic validation.

    Args:
        players_df: Players DataFrame
        teams_df: Teams DataFrame
        context: Context for error messages

    Returns:
        DataFrame with validated team names

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
