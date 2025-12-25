"""Transfer plan domain models."""

from enum import Enum

from pydantic import BaseModel, Field


class StrategyMode(str, Enum):
    """Transfer strategy modes."""

    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


class Transfer(BaseModel):
    """Single player transfer."""

    player_out_id: int = Field(..., description="Player ID being transferred out")
    player_in_id: int = Field(..., description="Player ID being transferred in")
    player_out_name: str = Field(..., description="Name of player out")
    player_in_name: str = Field(..., description="Name of player in")
    position: str = Field(..., description="Position (GKP/DEF/MID/FWD)")
    cost: float = Field(..., description="Price difference (positive = more expensive)")
