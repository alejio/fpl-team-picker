"""
Prediction Storage Service - Saves and retrieves gameweek predictions.

Allows users to save their committed predictions (ML xP, calibrated xP, captain choices)
for accurate performance tracking and historical analysis.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field


class SavedPrediction(BaseModel):
    """Individual player prediction snapshot."""

    player_id: int
    web_name: str
    position: str
    team: str
    now_cost: float
    ml_xp: float
    calibrated_xp: float
    xp_uncertainty: Optional[float] = None
    fixture_difficulty: Optional[int] = None
    opponent_team: Optional[str] = None
    is_home: Optional[bool] = None
    is_captain: bool = False
    is_vice_captain: bool = False
    in_starting_xi: bool = True


class SquadSnapshot(BaseModel):
    """Squad composition at time of save."""

    captain_id: int
    vice_captain_id: int
    player_ids: list[int]
    total_squad_value: float
    in_the_bank: float = 0.0
    free_transfers: int = 1


class GameweekPredictions(BaseModel):
    """Complete gameweek prediction snapshot."""

    gameweek: int
    saved_at: datetime = Field(default_factory=datetime.now)
    squad: SquadSnapshot
    predictions: list[SavedPrediction]
    model_info: dict = Field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert predictions to DataFrame for analysis."""
        return pd.DataFrame([p.model_dump() for p in self.predictions])

    def get_total_xp(self) -> float:
        """Calculate total squad calibrated xP."""
        return sum(p.calibrated_xp for p in self.predictions)

    def get_captain_name(self) -> str:
        """Get captain web name."""
        for p in self.predictions:
            if p.is_captain:
                return p.web_name
        return "Unknown"

    def get_captain_xp(self) -> float:
        """Get captain calibrated xP."""
        for p in self.predictions:
            if p.is_captain:
                return p.calibrated_xp
        return 0.0


class PredictionStorageService:
    """Service for saving and loading gameweek predictions."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize prediction storage service.

        Args:
            storage_dir: Directory to store predictions. Defaults to data/predictions/
        """
        if storage_dir is None:
            # Default to project root / data / predictions
            project_root = Path(__file__).parent.parent.parent.parent
            storage_dir = project_root / "data" / "predictions"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, gameweek: int) -> Path:
        """Get filepath for gameweek predictions."""
        return self.storage_dir / f"gw{gameweek}_predictions.json"

    def save_predictions(
        self,
        gameweek: int,
        predictions_df: pd.DataFrame,
        team_data: dict,
        model_info: Optional[dict] = None,
    ) -> Path:
        """
        Save predictions for a gameweek.

        Args:
            gameweek: Gameweek number
            predictions_df: DataFrame with columns: player_id, web_name, position, team,
                           now_cost, ml_xp, calibrated_xp, xp_uncertainty (optional),
                           fixture_difficulty (optional), opponent_team (optional),
                           is_home (optional)
            team_data: Team data dict with 'picks' key containing squad info
            model_info: Optional dict with model metadata (model_path, calibration_config, etc.)

        Returns:
            Path to saved file

        Raises:
            ValueError: If required columns are missing
        """
        # Validate required columns
        required_cols = [
            "player_id",
            "web_name",
            "position",
            "team",
            "now_cost",
            "ml_xp",
            "calibrated_xp",
        ]
        missing = set(required_cols) - set(predictions_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Extract squad info
        picks = team_data.get("picks", [])
        captain_id = next((p["element"] for p in picks if p.get("is_captain")), 0)
        vice_captain_id = next(
            (p["element"] for p in picks if p.get("is_vice_captain")), 0
        )
        player_ids = [p["element"] for p in picks]

        # Get squad value info
        entry_info = team_data.get("entry", {})
        total_value = entry_info.get("value", 0) / 10.0  # Convert from FPL units
        in_the_bank = entry_info.get("bank", 0) / 10.0
        free_transfers = team_data.get("transfers", {}).get("limit", 1)

        squad = SquadSnapshot(
            captain_id=captain_id,
            vice_captain_id=vice_captain_id,
            player_ids=player_ids,
            total_squad_value=total_value,
            in_the_bank=in_the_bank,
            free_transfers=free_transfers,
        )

        # Build predictions list (only for players in squad)
        predictions = []
        for _, row in predictions_df.iterrows():
            player_id = int(row["player_id"])
            if player_id not in player_ids:
                continue

            # Determine if in starting XI (first 11 picks)
            pick = next((p for p in picks if p["element"] == player_id), None)
            in_starting_xi = pick["position"] <= 11 if pick else False

            prediction = SavedPrediction(
                player_id=player_id,
                web_name=row["web_name"],
                position=row["position"],
                team=row["team"],
                now_cost=float(row["now_cost"]),
                ml_xp=float(row["ml_xp"]),
                calibrated_xp=float(row["calibrated_xp"]),
                xp_uncertainty=(
                    float(row["xp_uncertainty"])
                    if "xp_uncertainty" in row and pd.notna(row["xp_uncertainty"])
                    else None
                ),
                fixture_difficulty=(
                    int(row["fixture_difficulty"])
                    if "fixture_difficulty" in row
                    and pd.notna(row["fixture_difficulty"])
                    else None
                ),
                opponent_team=(
                    str(row["opponent_team"])
                    if "opponent_team" in row and pd.notna(row["opponent_team"])
                    else None
                ),
                is_home=(
                    bool(row["is_home"])
                    if "is_home" in row and pd.notna(row["is_home"])
                    else None
                ),
                is_captain=player_id == captain_id,
                is_vice_captain=player_id == vice_captain_id,
                in_starting_xi=in_starting_xi,
            )
            predictions.append(prediction)

        # Create gameweek predictions
        gw_predictions = GameweekPredictions(
            gameweek=gameweek,
            saved_at=datetime.now(),
            squad=squad,
            predictions=predictions,
            model_info=model_info or {},
        )

        # Save to file
        filepath = self._get_filepath(gameweek)
        with open(filepath, "w") as f:
            f.write(gw_predictions.model_dump_json(indent=2))

        return filepath

    def load_predictions(self, gameweek: int) -> Optional[GameweekPredictions]:
        """
        Load saved predictions for a gameweek.

        Args:
            gameweek: Gameweek number

        Returns:
            GameweekPredictions object or None if not found
        """
        filepath = self._get_filepath(gameweek)
        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            data = f.read()

        return GameweekPredictions.model_validate_json(data)

    def list_saved_gameweeks(self) -> list[int]:
        """
        Get list of gameweeks with saved predictions.

        Returns:
            Sorted list of gameweek numbers
        """
        gameweeks = []
        for filepath in self.storage_dir.glob("gw*_predictions.json"):
            try:
                gw = int(filepath.stem.split("_")[0][2:])  # Extract from "gw14"
                gameweeks.append(gw)
            except (ValueError, IndexError):
                continue

        return sorted(gameweeks)

    def prediction_exists(self, gameweek: int) -> bool:
        """Check if predictions exist for a gameweek."""
        return self._get_filepath(gameweek).exists()

    def get_prediction_summary(self, gameweek: int) -> Optional[dict]:
        """
        Get summary of saved predictions without loading full data.

        Args:
            gameweek: Gameweek number

        Returns:
            Summary dict with key stats or None if not found
        """
        predictions = self.load_predictions(gameweek)
        if not predictions:
            return None

        return {
            "gameweek": gameweek,
            "saved_at": predictions.saved_at,
            "total_xp": predictions.get_total_xp(),
            "captain": predictions.get_captain_name(),
            "captain_xp": predictions.get_captain_xp(),
            "squad_value": predictions.squad.total_squad_value,
            "num_players": len(predictions.predictions),
        }
