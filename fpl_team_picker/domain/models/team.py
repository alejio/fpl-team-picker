"""Team domain model."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class TeamDomain(BaseModel):
    """Domain model for FPL teams."""

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    team_id: int = Field(
        ..., ge=1, le=20, description="Team ID (1-20 for Premier League)"
    )
    name: str = Field(..., min_length=1, max_length=100, description="Full team name")
    short_name: str = Field(
        ..., min_length=2, max_length=3, description="3-letter team code"
    )
    as_of_utc: datetime = Field(..., description="Data timestamp")

    @field_serializer("as_of_utc", when_used="json")
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format."""
        return value.isoformat()
