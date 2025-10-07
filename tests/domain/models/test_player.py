"""Tests for PlayerDomain model."""

import pytest
from datetime import datetime

from fpl_team_picker.domain.models.player import (
    AvailabilityStatus,
    EnrichedPlayerDomain,
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


class TestEnrichedPlayerDomain:
    """Test EnrichedPlayerDomain model with derived metrics."""

    def test_enriched_player_creation_with_derived_metrics(self):
        """Test creating enriched player with all derived metrics."""
        player = EnrichedPlayerDomain(
            # Core PlayerDomain fields
            player_id=1,
            web_name="Salah",
            first_name="Mohamed",
            last_name="Salah",
            team_id=11,
            position=Position.MID,
            price=13.0,
            selected_by_percent=45.2,
            availability_status=AvailabilityStatus.AVAILABLE,
            as_of_utc=datetime.utcnow(),
            # Enhanced fields
            total_points_season=150,
            form_season=8.5,
            points_per_game_season=6.8,
            minutes=900,
            starts=10,
            goals_scored=12,
            assists=8,
            clean_sheets=2,
            goals_conceded=5,
            yellow_cards=1,
            red_cards=0,
            saves=0,
            bonus=15,
            bps=450,
            influence=500.0,
            creativity=400.0,
            threat=600.0,
            ict_index=1500.0,
            expected_goals=10.5,
            expected_assists=7.2,
            expected_goals_per_90=1.05,
            expected_assists_per_90=0.72,
            value_form=100.0,
            value_season=95.0,
            transfers_in=50000,
            transfers_out=10000,
            transfers_in_event=5000,
            transfers_out_event=1000,
            chance_of_playing_this_round=100.0,
            chance_of_playing_next_round=100.0,
            penalties_order=1,
            corners_and_indirect_freekicks_order=1,
            news="",
            # Derived metrics
            points_per_million=11.5,
            form_per_million=0.65,
            value_score=95.0,
            value_confidence=0.95,
            form_trend="improving",
            form_momentum=0.8,
            recent_form_5gw=8.2,
            season_consistency=0.85,
            expected_points_per_game=7.0,
            points_above_expected=10.0,
            overperformance_risk=0.3,
            ownership_trend="rising",
            transfer_momentum=40000.0,
            ownership_risk=0.2,
            set_piece_priority=0.9,
            penalty_taker=True,
            corner_taker=True,
            freekick_taker=True,
            injury_risk=0.1,
            rotation_risk=0.05,
            data_quality_score=0.95,
        )

        assert player.player_id == 1
        assert player.web_name == "Salah"
        assert player.value_score == 95.0
        assert player.injury_risk == 0.1

    def test_enriched_player_computed_properties(self):
        """Test computed properties on enriched player."""
        player = EnrichedPlayerDomain(
            player_id=1,
            web_name="Test",
            team_id=1,
            position=Position.MID,
            price=10.0,
            selected_by_percent=20.0,
            as_of_utc=datetime.utcnow(),
            # Enhanced fields
            total_points_season=100,
            form_season=6.0,
            points_per_game_season=5.0,
            minutes=900,
            starts=10,
            goals_scored=8,
            assists=5,
            clean_sheets=2,
            goals_conceded=10,
            yellow_cards=2,
            red_cards=0,
            saves=0,
            bonus=10,
            bps=300,
            influence=300.0,
            creativity=250.0,
            threat=400.0,
            ict_index=950.0,
            expected_goals=7.0,
            expected_assists=4.5,
            expected_goals_per_90=0.7,
            expected_assists_per_90=0.45,
            value_form=60.0,
            value_season=55.0,
            transfers_in=20000,
            transfers_out=5000,
            transfers_in_event=2000,
            transfers_out_event=500,
            penalties_order=2,
            news="",
            # Derived metrics
            points_per_million=10.0,
            form_per_million=0.6,
            value_score=85.0,
            value_confidence=0.8,
            form_trend="improving",
            form_momentum=0.5,
            recent_form_5gw=6.2,
            season_consistency=0.75,
            expected_points_per_game=5.5,
            points_above_expected=5.0,
            overperformance_risk=0.4,
            ownership_trend="rising",
            transfer_momentum=15000.0,
            ownership_risk=0.3,
            set_piece_priority=0.7,
            penalty_taker=True,
            corner_taker=False,
            freekick_taker=False,
            injury_risk=0.2,
            rotation_risk=0.15,
            data_quality_score=0.9,
        )

        # Test derived metric computed properties
        assert player.is_high_value is True  # value_score >= 80
        assert player.has_injury_concern is False  # injury_risk <= 0.5
        assert player.has_rotation_concern is False  # rotation_risk <= 0.5
        assert player.is_form_improving is True  # form_trend == "improving"
        assert player.is_form_declining is False
        assert player.is_ownership_rising is True  # ownership_trend == "rising"
        assert player.has_overperformance_risk is False  # overperformance_risk < 0.7
        assert player.is_reliable_data is True  # data_quality_score >= 0.7

        # Test inherited properties
        assert player.is_penalty_taker is True
        assert player.goals_per_90 == 0.8  # (8 * 90) / 900
        assert player.assists_per_90 == 0.5  # (5 * 90) / 900

    def test_enriched_player_validation_errors(self):
        """Test validation of derived metrics fields."""
        base_data = {
            "player_id": 1,
            "web_name": "Test",
            "team_id": 1,
            "position": Position.MID,
            "price": 10.0,
            "selected_by_percent": 20.0,
            "as_of_utc": datetime.utcnow(),
            "total_points_season": 100,
            "form_season": 6.0,
            "points_per_game_season": 5.0,
            "minutes": 900,
            "starts": 10,
            "goals_scored": 8,
            "assists": 5,
            "clean_sheets": 2,
            "goals_conceded": 10,
            "yellow_cards": 2,
            "red_cards": 0,
            "saves": 0,
            "bonus": 10,
            "bps": 300,
            "influence": 300.0,
            "creativity": 250.0,
            "threat": 400.0,
            "ict_index": 950.0,
            "expected_goals": 7.0,
            "expected_assists": 4.5,
            "expected_goals_per_90": 0.7,
            "expected_assists_per_90": 0.45,
            "value_form": 60.0,
            "value_season": 55.0,
            "transfers_in": 20000,
            "transfers_out": 5000,
            "transfers_in_event": 2000,
            "transfers_out_event": 500,
            "news": "",
            "points_per_million": 10.0,
            "form_per_million": 0.6,
            "value_score": 85.0,
            "value_confidence": 0.8,
            "form_trend": "stable",
            "form_momentum": 0.0,
            "recent_form_5gw": 6.0,
            "season_consistency": 0.75,
            "expected_points_per_game": 5.5,
            "points_above_expected": 0.0,
            "overperformance_risk": 0.4,
            "ownership_trend": "stable",
            "transfer_momentum": 0.0,
            "ownership_risk": 0.3,
            "set_piece_priority": 0.5,
            "penalty_taker": False,
            "corner_taker": False,
            "freekick_taker": False,
            "injury_risk": 0.2,
            "rotation_risk": 0.15,
            "data_quality_score": 0.9,
        }

        # Invalid value_score (> 100)
        with pytest.raises(ValueError):
            EnrichedPlayerDomain(**{**base_data, "value_score": 105.0})

        # Invalid value_confidence (> 1.0)
        with pytest.raises(ValueError):
            EnrichedPlayerDomain(**{**base_data, "value_confidence": 1.5})

        # Invalid form_momentum (< -1.0)
        with pytest.raises(ValueError):
            EnrichedPlayerDomain(**{**base_data, "form_momentum": -1.5})

        # Invalid injury_risk (> 1.0)
        with pytest.raises(ValueError):
            EnrichedPlayerDomain(**{**base_data, "injury_risk": 1.2})

        # Invalid rotation_risk (< 0.0)
        with pytest.raises(ValueError):
            EnrichedPlayerDomain(**{**base_data, "rotation_risk": -0.1})

    def test_enriched_player_high_injury_and_rotation_risk(self):
        """Test players with high injury and rotation concerns."""
        player = EnrichedPlayerDomain(
            player_id=1,
            web_name="Fragile",
            team_id=1,
            position=Position.DEF,
            price=5.0,
            selected_by_percent=10.0,
            as_of_utc=datetime.utcnow(),
            total_points_season=50,
            form_season=3.0,
            points_per_game_season=2.5,
            minutes=450,
            starts=5,
            goals_scored=1,
            assists=2,
            clean_sheets=3,
            goals_conceded=8,
            yellow_cards=3,
            red_cards=0,
            saves=0,
            bonus=2,
            bps=100,
            influence=100.0,
            creativity=50.0,
            threat=50.0,
            ict_index=200.0,
            expected_goals=0.5,
            expected_assists=1.0,
            expected_goals_per_90=0.1,
            expected_assists_per_90=0.2,
            value_form=30.0,
            value_season=25.0,
            transfers_in=5000,
            transfers_out=15000,
            transfers_in_event=500,
            transfers_out_event=1500,
            news="",
            points_per_million=10.0,
            form_per_million=0.6,
            value_score=40.0,
            value_confidence=0.5,
            form_trend="declining",
            form_momentum=-0.6,
            recent_form_5gw=2.5,
            season_consistency=0.4,
            expected_points_per_game=2.0,
            points_above_expected=-5.0,
            overperformance_risk=0.8,
            ownership_trend="falling",
            transfer_momentum=-10000.0,
            ownership_risk=0.7,
            set_piece_priority=0.1,
            penalty_taker=False,
            corner_taker=False,
            freekick_taker=False,
            injury_risk=0.75,
            rotation_risk=0.65,
            data_quality_score=0.6,
        )

        # Test high risk properties
        assert player.is_high_value is False  # value_score < 80
        assert player.has_injury_concern is True  # injury_risk > 0.5
        assert player.has_rotation_concern is True  # rotation_risk > 0.5
        assert player.is_form_improving is False
        assert player.is_form_declining is True
        assert player.is_ownership_rising is False
        assert player.has_overperformance_risk is True  # overperformance_risk >= 0.7
        assert player.is_reliable_data is False  # data_quality_score < 0.7
