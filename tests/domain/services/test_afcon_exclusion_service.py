"""Unit tests for AFCON Exclusion Service.

Tests for:
- Loading AFCON data from JSON file
- Getting player IDs with filters
- Gameweek affected checks
- Tournament info retrieval
- Exclusion summary generation
- Error handling (missing file, invalid JSON)
"""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from fpl_team_picker.domain.services.afcon_exclusion_service import (
    AFCONExclusionService,
)


class TestAFCONExclusionServiceInit:
    """Test service initialization."""

    def test_init_with_default_path(self):
        """Test initialization with default data path."""
        service = AFCONExclusionService()
        assert service.data_path is not None
        assert service.data_path.name == "afcon_2025_players.json"
        assert service._afcon_data is None  # Not loaded yet

    def test_init_with_custom_path(self):
        """Test initialization with custom data path."""
        with TemporaryDirectory() as tmpdir:
            custom_path = Path(tmpdir) / "custom_afcon.json"
            service = AFCONExclusionService(data_path=custom_path)
            assert service.data_path == custom_path


class TestAFCONDataLoading:
    """Test loading AFCON data from JSON file."""

    def test_load_valid_data(self):
        """Test loading valid AFCON data file."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {
                "tournament_name": "AFCON 2025",
                "players": [
                    {
                        "player_id": 381,
                        "web_name": "Salah",
                        "country": "Egypt",
                        "impact": "high",
                    },
                    {
                        "player_id": 413,
                        "web_name": "Marmoush",
                        "country": "Egypt",
                        "impact": "high",
                    },
                ],
                "affected_gameweeks": [17, 18, 19],
            }
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            data = service.load_afcon_data()

            assert data == test_data
            assert len(data["players"]) == 2
            assert data["affected_gameweeks"] == [17, 18, 19]

    def test_load_data_caching(self):
        """Test that data is cached after first load."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {"players": [], "affected_gameweeks": []}
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            data1 = service.load_afcon_data()
            data2 = service.load_afcon_data()

            # Should return same object (cached)
            assert data1 is data2
            assert data1 is service._afcon_data

    def test_load_missing_file(self):
        """Test handling of missing data file."""
        with TemporaryDirectory() as tmpdir:
            missing_file = Path(tmpdir) / "nonexistent.json"
            service = AFCONExclusionService(data_path=missing_file)
            data = service.load_afcon_data()

            assert data == {"players": [], "affected_gameweeks": []}

    def test_load_invalid_json(self):
        """Test handling of invalid JSON file."""
        with TemporaryDirectory() as tmpdir:
            invalid_file = Path(tmpdir) / "invalid.json"
            with open(invalid_file, "w") as f:
                f.write("not valid json {")

            service = AFCONExclusionService(data_path=invalid_file)
            data = service.load_afcon_data()

            assert data == {"players": [], "affected_gameweeks": []}


class TestGetAFCONPlayerIds:
    """Test getting AFCON player IDs with filters."""

    @pytest.fixture
    def sample_service(self, tmp_path):
        """Create service with sample data."""
        data_file = tmp_path / "afcon_2025_players.json"
        test_data = {
            "players": [
                {
                    "player_id": 381,
                    "web_name": "Salah",
                    "country": "Egypt",
                    "impact": "high",
                },
                {
                    "player_id": 413,
                    "web_name": "Marmoush",
                    "country": "Egypt",
                    "impact": "high",
                },
                {
                    "player_id": 587,
                    "web_name": "Bissouma",
                    "country": "Mali",
                    "impact": "medium",
                },
                {
                    "player_id": 128,
                    "web_name": "Onyeka",
                    "country": "Nigeria",
                    "impact": "low",
                },
            ],
            "affected_gameweeks": [17, 18, 19],
        }
        with open(data_file, "w") as f:
            json.dump(test_data, f)

        service = AFCONExclusionService(data_path=data_file)
        # Pre-load data to avoid file path issues
        service.load_afcon_data()
        return service

    def test_get_all_player_ids(self, sample_service):
        """Test getting all player IDs without filters."""
        player_ids = sample_service.get_afcon_player_ids()
        assert player_ids == {381, 413, 587, 128}

    def test_filter_by_impact_high(self, sample_service):
        """Test filtering by high impact."""
        player_ids = sample_service.get_afcon_player_ids(impact_filter=["high"])
        assert player_ids == {381, 413}

    def test_filter_by_impact_medium(self, sample_service):
        """Test filtering by medium impact."""
        player_ids = sample_service.get_afcon_player_ids(impact_filter=["medium"])
        assert player_ids == {587}

    def test_filter_by_impact_multiple(self, sample_service):
        """Test filtering by multiple impact levels."""
        player_ids = sample_service.get_afcon_player_ids(
            impact_filter=["high", "medium"]
        )
        assert player_ids == {381, 413, 587}

    def test_filter_by_country(self, sample_service):
        """Test filtering by country."""
        player_ids = sample_service.get_afcon_player_ids(country_filter=["Egypt"])
        assert player_ids == {381, 413}

    def test_filter_by_country_multiple(self, sample_service):
        """Test filtering by multiple countries."""
        player_ids = sample_service.get_afcon_player_ids(
            country_filter=["Egypt", "Mali"]
        )
        assert player_ids == {381, 413, 587}

    def test_filter_by_impact_and_country(self, sample_service):
        """Test filtering by both impact and country."""
        player_ids = sample_service.get_afcon_player_ids(
            impact_filter=["high"], country_filter=["Egypt"]
        )
        assert player_ids == {381, 413}

    def test_filter_no_matches(self, sample_service):
        """Test filtering with no matches."""
        player_ids = sample_service.get_afcon_player_ids(country_filter=["Brazil"])
        assert player_ids == set()

    def test_player_without_id_skipped(self, sample_service):
        """Test that players without player_id are skipped."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {
                "players": [
                    {"player_id": 381, "web_name": "Salah"},
                    {"web_name": "NoID"},  # Missing player_id
                ],
                "affected_gameweeks": [],
            }
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            player_ids = service.get_afcon_player_ids()
            assert player_ids == {381}


class TestGetAFCONPlayersInfo:
    """Test getting full player information."""

    def test_get_players_info(self):
        """Test getting full player information list."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {
                "players": [
                    {
                        "player_id": 381,
                        "web_name": "Salah",
                        "full_name": "Mohamed Salah",
                        "country": "Egypt",
                        "impact": "high",
                    },
                ],
                "affected_gameweeks": [],
            }
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            players = service.get_afcon_players_info()

            assert len(players) == 1
            assert players[0]["player_id"] == 381
            assert players[0]["web_name"] == "Salah"

    def test_get_players_info_empty(self):
        """Test getting players info when no players."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {"players": [], "affected_gameweeks": []}
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            players = service.get_afcon_players_info()

            assert players == []


class TestIsGameweekAffected:
    """Test checking if gameweek is affected by AFCON."""

    def test_gameweek_affected(self):
        """Test that affected gameweek returns True."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {"players": [], "affected_gameweeks": [17, 18, 19, 20, 21, 22]}
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            assert service.is_gameweek_affected(17) is True
            assert service.is_gameweek_affected(20) is True
            assert service.is_gameweek_affected(22) is True

    def test_gameweek_not_affected(self):
        """Test that unaffected gameweek returns False."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {"players": [], "affected_gameweeks": [17, 18, 19]}
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            assert service.is_gameweek_affected(16) is False
            assert service.is_gameweek_affected(20) is False


class TestGetTournamentInfo:
    """Test getting tournament information."""

    def test_get_tournament_info(self):
        """Test getting full tournament information."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {
                "tournament_name": "Africa Cup of Nations 2025",
                "tournament_dates": {"start": "2025-12-21", "end": "2026-01-18"},
                "affected_gameweeks": [17, 18, 19],
                "players": [],
            }
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            info = service.get_tournament_info()

            assert info["name"] == "Africa Cup of Nations 2025"
            assert info["dates"]["start"] == "2025-12-21"
            assert info["affected_gameweeks"] == [17, 18, 19]

    def test_get_tournament_info_missing_fields(self):
        """Test getting tournament info with missing fields."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {"players": []}
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            info = service.get_tournament_info()

            assert info["name"] == ""
            assert info["dates"] == {}
            assert info["affected_gameweeks"] == []


class TestGetExclusionSummary:
    """Test generating exclusion summary."""

    def test_exclusion_summary_empty(self):
        """Test summary with no exclusions."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {"players": [], "affected_gameweeks": []}
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            summary = service.get_exclusion_summary(set())

            assert summary == "No AFCON exclusions"

    def test_exclusion_summary_by_impact(self):
        """Test summary grouped by impact level."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {
                "players": [
                    {
                        "player_id": 381,
                        "web_name": "Salah",
                        "country": "Egypt",
                        "impact": "high",
                    },
                    {
                        "player_id": 413,
                        "web_name": "Marmoush",
                        "country": "Egypt",
                        "impact": "high",
                    },
                    {
                        "player_id": 587,
                        "web_name": "Bissouma",
                        "country": "Mali",
                        "impact": "medium",
                    },
                    {
                        "player_id": 128,
                        "web_name": "Onyeka",
                        "country": "Nigeria",
                        "impact": "low",
                    },
                ],
                "affected_gameweeks": [],
            }
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            summary = service.get_exclusion_summary({381, 413, 587, 128})

            assert "HIGH:" in summary
            assert "Salah" in summary
            assert "Marmoush" in summary
            assert "MEDIUM:" in summary
            assert "Bissouma" in summary
            assert "LOW:" in summary
            assert "Onyeka" in summary

    def test_exclusion_summary_unknown_players(self):
        """Test summary with player IDs not in data."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {"players": [], "affected_gameweeks": []}
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            summary = service.get_exclusion_summary({999, 1000})

            assert "2 players excluded" in summary
            assert "AFCON" in summary

    def test_exclusion_summary_fallback_to_full_name(self):
        """Test summary uses full_name when web_name missing."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {
                "players": [
                    {
                        "player_id": 381,
                        "full_name": "Mohamed Salah",
                        "country": "Egypt",
                        "impact": "high",
                    },
                ],
                "affected_gameweeks": [],
            }
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            summary = service.get_exclusion_summary({381})

            assert "Mohamed Salah" in summary

    def test_exclusion_summary_unknown_name(self):
        """Test summary with player having no name."""
        with TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "afcon_2025_players.json"
            test_data = {
                "players": [
                    {"player_id": 381, "country": "Egypt", "impact": "high"},
                ],
                "affected_gameweeks": [],
            }
            with open(data_file, "w") as f:
                json.dump(test_data, f)

            service = AFCONExclusionService(data_path=data_file)
            summary = service.get_exclusion_summary({381})

            assert "Unknown" in summary
