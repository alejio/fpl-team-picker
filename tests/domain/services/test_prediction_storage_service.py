"""Tests for PredictionStorageService."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from fpl_team_picker.domain.services.prediction_storage_service import (
    PredictionStorageService,
    GameweekPredictions,
    SavedPrediction,
    SquadSnapshot,
)


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory."""
    storage_dir = tmp_path / "predictions"
    storage_dir.mkdir()
    return storage_dir


@pytest.fixture
def storage_service(temp_storage_dir):
    """Create a PredictionStorageService with temp storage."""
    return PredictionStorageService(storage_dir=temp_storage_dir)


@pytest.fixture
def sample_predictions_df():
    """Create sample predictions DataFrame."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "web_name": ["Haaland", "Salah", "Saka", "Watkins", "Alexander-Arnold"],
            "position": ["FWD", "MID", "MID", "FWD", "DEF"],
            "team": ["MCI", "LIV", "ARS", "AVL", "LIV"],
            "now_cost": [15.0, 13.5, 10.0, 9.0, 7.5],
            "ml_xp": [8.5, 7.2, 6.5, 5.8, 5.0],
            "calibrated_xp": [9.1, 7.8, 6.9, 6.2, 5.3],
            "xp_uncertainty": [2.3, 1.8, 1.5, 1.9, 1.2],
            "fixture_difficulty": [2, 3, 2, 4, 3],
            "opponent_team": ["BRE", "NEW", "BOU", "MUN", "NEW"],
            "is_home": [True, False, True, False, False],
        }
    )


@pytest.fixture
def sample_team_data():
    """Create sample team data."""
    return {
        "picks": [
            {"element": 1, "position": 1, "is_captain": True, "is_vice_captain": False},
            {"element": 2, "position": 2, "is_captain": False, "is_vice_captain": True},
            {"element": 3, "position": 3, "is_captain": False, "is_vice_captain": False},
            {"element": 4, "position": 4, "is_captain": False, "is_vice_captain": False},
            {"element": 5, "position": 5, "is_captain": False, "is_vice_captain": False},
        ],
        "entry": {"value": 1005, "bank": 5},  # £100.5m squad, £0.5m in bank
        "transfers": {"limit": 1},
    }


class TestPredictionStorageService:
    """Test suite for PredictionStorageService."""

    def test_initialization_creates_directory(self, tmp_path):
        """Test that initialization creates storage directory if it doesn't exist."""
        storage_dir = tmp_path / "new_predictions"
        assert not storage_dir.exists()

        service = PredictionStorageService(storage_dir=storage_dir)
        assert storage_dir.exists()
        assert service.storage_dir == storage_dir

    def test_save_predictions_basic(
        self, storage_service, sample_predictions_df, sample_team_data
    ):
        """Test basic save predictions functionality."""
        gameweek = 14
        result_path = storage_service.save_predictions(
            gameweek=gameweek,
            predictions_df=sample_predictions_df,
            team_data=sample_team_data,
        )

        assert result_path.exists()
        assert result_path.name == "gw14_predictions.json"

        # Verify file contents
        with open(result_path, "r") as f:
            data = json.load(f)

        assert data["gameweek"] == 14
        assert len(data["predictions"]) == 5
        assert data["squad"]["captain_id"] == 1
        assert data["squad"]["vice_captain_id"] == 2

    def test_save_predictions_with_model_info(
        self, storage_service, sample_predictions_df, sample_team_data
    ):
        """Test saving predictions with model metadata."""
        model_info = {
            "model_path": "models/custom/random-forest_gw1-14.joblib",
            "calibration_enabled": True,
            "empirical_blend_weight": 0.5,
        }

        result_path = storage_service.save_predictions(
            gameweek=15,
            predictions_df=sample_predictions_df,
            team_data=sample_team_data,
            model_info=model_info,
        )

        # Load and verify model info
        with open(result_path, "r") as f:
            data = json.load(f)

        assert data["model_info"]["model_path"] == model_info["model_path"]
        assert data["model_info"]["calibration_enabled"] is True

    def test_save_predictions_missing_columns_raises_error(
        self, storage_service, sample_team_data
    ):
        """Test that missing required columns raises ValueError."""
        incomplete_df = pd.DataFrame(
            {
                "player_id": [1, 2],
                "web_name": ["Haaland", "Salah"],
                # Missing position, team, now_cost, ml_xp, calibrated_xp
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            storage_service.save_predictions(
                gameweek=14, predictions_df=incomplete_df, team_data=sample_team_data
            )

    def test_save_predictions_filters_squad_only(
        self, storage_service, sample_predictions_df, sample_team_data
    ):
        """Test that only squad players are saved."""
        # Add extra player not in squad
        extended_df = pd.concat(
            [
                sample_predictions_df,
                pd.DataFrame(
                    {
                        "player_id": [99],
                        "web_name": ["Non-Squad Player"],
                        "position": ["GKP"],
                        "team": ["CHE"],
                        "now_cost": [5.0],
                        "ml_xp": [2.0],
                        "calibrated_xp": [2.1],
                    }
                ),
            ],
            ignore_index=True,
        )

        result_path = storage_service.save_predictions(
            gameweek=14, predictions_df=extended_df, team_data=sample_team_data
        )

        # Verify only squad players saved
        with open(result_path, "r") as f:
            data = json.load(f)

        player_ids = [p["player_id"] for p in data["predictions"]]
        assert 99 not in player_ids
        assert len(data["predictions"]) == 5  # Only the 5 squad players

    def test_save_predictions_handles_optional_columns(
        self, storage_service, sample_team_data
    ):
        """Test that optional columns are handled gracefully."""
        minimal_df = pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4, 5],
                "web_name": ["Haaland", "Salah", "Saka", "Watkins", "TAA"],
                "position": ["FWD", "MID", "MID", "FWD", "DEF"],
                "team": ["MCI", "LIV", "ARS", "AVL", "LIV"],
                "now_cost": [15.0, 13.5, 10.0, 9.0, 7.5],
                "ml_xp": [8.5, 7.2, 6.5, 5.8, 5.0],
                "calibrated_xp": [9.1, 7.8, 6.9, 6.2, 5.3],
                # No xp_uncertainty, fixture_difficulty, opponent_team, is_home
            }
        )

        result_path = storage_service.save_predictions(
            gameweek=14, predictions_df=minimal_df, team_data=sample_team_data
        )

        # Verify it saves successfully with None for optional fields
        with open(result_path, "r") as f:
            data = json.load(f)

        first_pred = data["predictions"][0]
        assert first_pred["xp_uncertainty"] is None
        assert first_pred["fixture_difficulty"] is None
        assert first_pred["opponent_team"] is None
        assert first_pred["is_home"] is None

    def test_load_predictions_success(
        self, storage_service, sample_predictions_df, sample_team_data
    ):
        """Test loading saved predictions."""
        gameweek = 14
        storage_service.save_predictions(
            gameweek=gameweek,
            predictions_df=sample_predictions_df,
            team_data=sample_team_data,
        )

        loaded = storage_service.load_predictions(gameweek)
        assert loaded is not None
        assert isinstance(loaded, GameweekPredictions)
        assert loaded.gameweek == 14
        assert len(loaded.predictions) == 5
        assert loaded.squad.captain_id == 1

    def test_load_predictions_not_found(self, storage_service):
        """Test loading non-existent predictions returns None."""
        loaded = storage_service.load_predictions(999)
        assert loaded is None

    def test_prediction_exists(
        self, storage_service, sample_predictions_df, sample_team_data
    ):
        """Test checking if predictions exist."""
        assert not storage_service.prediction_exists(14)

        storage_service.save_predictions(
            gameweek=14, predictions_df=sample_predictions_df, team_data=sample_team_data
        )

        assert storage_service.prediction_exists(14)

    def test_list_saved_gameweeks(
        self, storage_service, sample_predictions_df, sample_team_data
    ):
        """Test listing all saved gameweeks."""
        assert storage_service.list_saved_gameweeks() == []

        # Save multiple gameweeks
        for gw in [12, 14, 13]:  # Out of order
            storage_service.save_predictions(
                gameweek=gw,
                predictions_df=sample_predictions_df,
                team_data=sample_team_data,
            )

        saved_gws = storage_service.list_saved_gameweeks()
        assert saved_gws == [12, 13, 14]  # Should be sorted

    def test_get_prediction_summary(
        self, storage_service, sample_predictions_df, sample_team_data
    ):
        """Test getting prediction summary."""
        storage_service.save_predictions(
            gameweek=14, predictions_df=sample_predictions_df, team_data=sample_team_data
        )

        summary = storage_service.get_prediction_summary(14)
        assert summary is not None
        assert summary["gameweek"] == 14
        assert summary["captain"] == "Haaland"
        assert summary["captain_xp"] == 9.1  # calibrated_xp
        assert summary["squad_value"] == 100.5
        assert summary["num_players"] == 5
        assert isinstance(summary["saved_at"], datetime)

    def test_get_prediction_summary_not_found(self, storage_service):
        """Test getting summary for non-existent predictions returns None."""
        summary = storage_service.get_prediction_summary(999)
        assert summary is None

    def test_to_dataframe(
        self, storage_service, sample_predictions_df, sample_team_data
    ):
        """Test converting GameweekPredictions to DataFrame."""
        storage_service.save_predictions(
            gameweek=14, predictions_df=sample_predictions_df, team_data=sample_team_data
        )

        loaded = storage_service.load_predictions(14)
        df = loaded.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "player_id" in df.columns
        assert "web_name" in df.columns
        assert "calibrated_xp" in df.columns
        assert df[df["is_captain"]]["player_id"].iloc[0] == 1

    def test_squad_snapshot_totals(
        self, storage_service, sample_predictions_df, sample_team_data
    ):
        """Test squad snapshot captures correct totals."""
        storage_service.save_predictions(
            gameweek=14, predictions_df=sample_predictions_df, team_data=sample_team_data
        )

        loaded = storage_service.load_predictions(14)
        assert loaded.squad.total_squad_value == 100.5
        assert loaded.squad.in_the_bank == 0.5
        assert loaded.squad.free_transfers == 1

    def test_starting_xi_flag(
        self, storage_service, sample_predictions_df, sample_team_data
    ):
        """Test in_starting_xi flag is set correctly."""
        # Add more players to test bench
        extended_team_data = {
            **sample_team_data,
            "picks": [
                *sample_team_data["picks"],
                {
                    "element": 6,
                    "position": 12,
                    "is_captain": False,
                    "is_vice_captain": False,
                },  # Bench
            ],
        }

        extended_df = pd.concat(
            [
                sample_predictions_df,
                pd.DataFrame(
                    {
                        "player_id": [6],
                        "web_name": ["Bench Player"],
                        "position": ["DEF"],
                        "team": ["CHE"],
                        "now_cost": [4.5],
                        "ml_xp": [2.0],
                        "calibrated_xp": [2.1],
                    }
                ),
            ],
            ignore_index=True,
        )

        storage_service.save_predictions(
            gameweek=14, predictions_df=extended_df, team_data=extended_team_data
        )

        loaded = storage_service.load_predictions(14)
        df = loaded.to_dataframe()

        # First 5 should be in starting XI
        starting_players = df[df["in_starting_xi"]]
        bench_players = df[~df["in_starting_xi"]]

        assert len(starting_players) == 5
        assert len(bench_players) == 1
        assert bench_players.iloc[0]["web_name"] == "Bench Player"

    def test_invalid_json_file_ignored_in_list(self, storage_service, temp_storage_dir):
        """Test that invalid JSON files are ignored when listing gameweeks."""
        # Create invalid file
        invalid_file = temp_storage_dir / "gw_invalid_predictions.json"
        invalid_file.write_text("invalid")

        # Should not raise error
        saved_gws = storage_service.list_saved_gameweeks()
        assert saved_gws == []

    def test_get_total_xp(
        self, storage_service, sample_predictions_df, sample_team_data
    ):
        """Test calculating total squad xP."""
        storage_service.save_predictions(
            gameweek=14, predictions_df=sample_predictions_df, team_data=sample_team_data
        )

        loaded = storage_service.load_predictions(14)
        total_xp = loaded.get_total_xp()

        # Sum of calibrated_xp for all 5 players
        expected = 9.1 + 7.8 + 6.9 + 6.2 + 5.3
        assert abs(total_xp - expected) < 0.01

    def test_captain_methods(
        self, storage_service, sample_predictions_df, sample_team_data
    ):
        """Test captain-related methods."""
        storage_service.save_predictions(
            gameweek=14, predictions_df=sample_predictions_df, team_data=sample_team_data
        )

        loaded = storage_service.load_predictions(14)
        assert loaded.get_captain_name() == "Haaland"
        assert loaded.get_captain_xp() == 9.1
