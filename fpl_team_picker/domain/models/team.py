"""Team domain model."""

from datetime import datetime

from pydantic import BaseModel, Field


class TeamDomain(BaseModel):
    """Domain model for FPL teams."""

    team_id: int = Field(
        ..., ge=1, le=20, description="Team ID (1-20 for Premier League)"
    )
    name: str = Field(..., min_length=1, max_length=100, description="Full team name")
    short_name: str = Field(
        ..., min_length=2, max_length=3, description="3-letter team code"
    )
    as_of_utc: datetime = Field(..., description="Data timestamp")

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}
