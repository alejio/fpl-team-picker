"""Tests for PlayerDomain model."""

import pytest
from datetime import datetime

from fpl_team_picker.domain.models.player import (
    AvailabilityStatus,
    LiveDataDomain,
    PlayerDomain,
    Position,
)


class TestPlayerDomain:
    """Test PlayerDomain model validation."""

    def test_valid_player_creation(self):
        """Test creating a valid player."""
        player = PlayerDomain(
            player_id=1,
            web_name="Haaland",
            first_name="Erling",
            last_name="Haaland",
            team_id=11,
            position=Position.FWD,
            price=15.0,
            selected_by_percent=55.2,
            availability_status=AvailabilityStatus.AVAILABLE,
            as_of_utc=datetime.utcnow(),
        )

        assert player.player_id == 1
        assert player.web_name == "Haaland"
        assert player.position == Position.FWD
        assert player.is_available is True
        assert player.full_name == "Erling Haaland"

    def test_player_price_validation(self):
        """Test price validation rules."""
        # Valid price
        PlayerDomain(
            player_id=1,
            web_name="Test",
            team_id=1,
            position=Position.GKP,
            price=4.5,  # Valid 0.1m increment
            selected_by_percent=10.0,
            as_of_utc=datetime.utcnow(),
        )

        # Invalid price increment
        with pytest.raises(ValueError, match="Price must be in 0.1m increments"):
            PlayerDomain(
                player_id=1,
                web_name="Test",
                team_id=1,
                position=Position.GKP,
                price=4.55,  # Invalid increment
                selected_by_percent=10.0,
                as_of_utc=datetime.utcnow(),
            )

    def test_player_validation_errors(self):
        """Test various validation errors."""
        base_data = {
            "player_id": 1,
            "web_name": "Test",
            "team_id": 1,
            "position": Position.GKP,
            "price": 4.5,
            "selected_by_percent": 10.0,
            "as_of_utc": datetime.utcnow(),
        }

        # Invalid team_id
        with pytest.raises(ValueError):
            PlayerDomain(**{**base_data, "team_id": 25})  # > 20

        # Invalid position
        with pytest.raises(ValueError):
            PlayerDomain(**{**base_data, "position": "INVALID"})

        # Price too low
        with pytest.raises(ValueError):
            PlayerDomain(**{**base_data, "price": 3.8})

        # Price too high
        with pytest.raises(ValueError):
            PlayerDomain(**{**base_data, "price": 15.1})

    def test_name_field_validation(self):
        """Test name field trimming and validation."""
        player = PlayerDomain(
            player_id=1,
            web_name="  Test  ",
            first_name="  John  ",
            last_name="  Doe  ",
            team_id=1,
            position=Position.GKP,
            price=4.5,
            selected_by_percent=10.0,
            as_of_utc=datetime.utcnow(),
        )

        assert player.web_name == "Test"  # web_name gets trimmed by validator
        assert player.first_name == "John"  # first_name trimmed
        assert player.last_name == "Doe"  # last_name trimmed
        assert player.full_name == "John Doe"

    def test_availability_status_enum(self):
        """Test availability status enum values."""
        player = PlayerDomain(
            player_id=1,
            web_name="Test",
            team_id=1,
            position=Position.GKP,
            price=4.5,
            selected_by_percent=10.0,
            availability_status=AvailabilityStatus.INJURED,
            as_of_utc=datetime.utcnow(),
        )

        assert player.availability_status == AvailabilityStatus.INJURED
        assert player.is_available is False


class TestLiveDataDomain:
    """Test LiveDataDomain model validation."""

    def test_valid_live_data_creation(self):
        """Test creating valid live data."""
        live_data = LiveDataDomain(
            player_id=1,
            gameweek=1,
            minutes=90,
            total_points=12,
            goals_scored=2,
            assists=1,
            clean_sheets=0,
            goals_conceded=1,
            yellow_cards=0,
            red_cards=0,
            saves=0,
            bonus=3,
            bps=45,
            influence=55.2,
            creativity=33.1,
            threat=88.5,
            ict_index=176.8,
            expected_goals=1.2,
            expected_assists=0.8,
            value=15.0,
            was_home=True,
            opponent_team=2,
        )

        assert live_data.player_id == 1
        assert live_data.gameweek == 1
        assert live_data.total_points == 12

    def test_live_data_validation_errors(self):
        """Test live data validation errors."""
        base_data = {
            "player_id": 1,
            "gameweek": 1,
            "minutes": 90,
            "total_points": 12,
            "goals_scored": 2,
            "assists": 1,
            "clean_sheets": 0,
            "goals_conceded": 1,
            "yellow_cards": 0,
            "red_cards": 0,
            "saves": 0,
            "bonus": 3,
            "bps": 45,
            "influence": 55.2,
            "creativity": 33.1,
            "threat": 88.5,
            "ict_index": 176.8,
            "value": 15.0,
            "was_home": True,
            "opponent_team": 2,
        }

        # Invalid gameweek
        with pytest.raises(ValueError):
            LiveDataDomain(**{**base_data, "gameweek": 39})

        # Invalid minutes
        with pytest.raises(ValueError):
            LiveDataDomain(**{**base_data, "minutes": 125})

        # Invalid value (too high) - first constraint that's checked
        with pytest.raises(ValueError):
            LiveDataDomain(**{**base_data, "value": 15.1})
