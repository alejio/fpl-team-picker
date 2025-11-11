"""Fixture domain model."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class FixtureDomain(BaseModel):
    """Domain model for FPL fixtures."""

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    fixture_id: int = Field(..., gt=0, description="Unique fixture ID")
    event: int = Field(..., ge=1, le=38, description="Gameweek number")
    kickoff_utc: datetime = Field(..., description="Kickoff time in UTC")
    home_team_id: int = Field(..., ge=1, le=20, description="Home team ID")
    away_team_id: int = Field(..., ge=1, le=20, description="Away team ID")
    as_of_utc: datetime = Field(..., description="Data timestamp")

    @field_serializer("kickoff_utc", "as_of_utc", when_used="json")
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format."""
        return value.isoformat()

    @property
    def involves_team(self) -> set[int]:
        """Get set of team IDs involved in this fixture."""
        return {self.home_team_id, self.away_team_id}

    def is_home_fixture(self, team_id: int) -> bool:
        """Check if the given team is playing at home."""
        return self.home_team_id == team_id

    def is_away_fixture(self, team_id: int) -> bool:
        """Check if the given team is playing away."""
        return self.away_team_id == team_id

    def get_opponent(self, team_id: int) -> int:
        """Get the opponent team ID for the given team."""
        if team_id == self.home_team_id:
            return self.away_team_id
        elif team_id == self.away_team_id:
            return self.home_team_id
        else:
            raise ValueError(f"Team {team_id} is not involved in this fixture")
