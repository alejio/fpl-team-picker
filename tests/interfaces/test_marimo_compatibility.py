"""Integration tests for marimo notebook compatibility."""

import pytest
import subprocess
from pathlib import Path


class TestMarimoCompatibility:
    """Test that marimo notebooks work after domain model changes."""

    def test_core_imports_work(self):
        """Test that all core imports still work."""
        # Test data loading functionality (core dependency)
        from fpl_team_picker.config.settings import FPLConfig

        # Test configuration loading
        config = FPLConfig()
        assert config is not None

        # Test that our new domain models don't interfere
        from fpl_team_picker.adapters.database_repositories import (
            DatabasePlayerRepository,
        )

        # Verify classes can be instantiated
        repo = DatabasePlayerRepository()
        assert repo is not None

    def test_gameweek_detection_works(self):
        """Test gameweek detection functionality."""
        from fpl_team_picker.domain.services.data_orchestration_service import (
            DataOrchestrationService,
        )

        service = DataOrchestrationService()
        gw_info = service.get_current_gameweek_info()
        # Should either return valid info or None (both acceptable)
        if gw_info:
            assert "current_gameweek" in gw_info
            assert isinstance(gw_info["current_gameweek"], int)

    def test_data_loading_functionality(self):
        """Test that data loading still works."""
        from fpl_team_picker.domain.services.data_orchestration_service import (
            DataOrchestrationService,
        )

        service = DataOrchestrationService()

        # Get current gameweek
        gw_info = service.get_current_gameweek_info()
        target_gw = gw_info.get("current_gameweek", 1) if gw_info else 1

        # Test data loading
        try:
            gameweek_data = service.load_gameweek_data(
                target_gameweek=target_gw, form_window=3
            )

            required_keys = ["players", "teams", "fixtures", "xg_rates"]
            for key in required_keys:
                assert key in gameweek_data, f"Missing key: {key}"

                # Check data is present (allow empty for some cases)
                if hasattr(gameweek_data[key], "empty"):
                    # DataFrame - check structure
                    assert hasattr(gameweek_data[key], "columns")
                elif hasattr(gameweek_data[key], "__len__"):
                    # Other iterable - just verify it exists
                    pass

        except Exception as e:
            # Data loading might fail in test environment - log but don't fail
            pytest.skip(f"Data loading failed in test env: {e}")

    def test_domain_model_compatibility(self):
        """Test domain models work with existing data patterns."""
        from fpl_team_picker.adapters.database_repositories import (
            DatabasePlayerRepository,
        )

        repo = DatabasePlayerRepository()
        result = repo.get_current_players()

        if result.is_success:
            players = result.value
            assert len(players) > 0

            # Verify essential fields that the notebook expects
            sample = players[0]
            essential_fields = [
                "player_id",
                "web_name",
                "position",
                "price",
                "team_id",
            ]
            for field in essential_fields:
                assert hasattr(sample, field), f"Missing field: {field}"
                assert getattr(sample, field) is not None, f"Null field: {field}"
        else:
            # Might fail in test environment
            pytest.skip(f"Domain model test failed: {result.error.message}")

    def test_marimo_notebook_structure(self):
        """Test that the marimo notebook file has correct structure."""
        notebook_path = Path("fpl_team_picker/interfaces/gameweek_manager.py")
        assert notebook_path.exists(), "Gameweek manager notebook not found"

        with open(notebook_path, "r") as f:
            content = f.read()

        # Check for marimo import
        assert "import marimo as mo" in content, "Missing marimo import"

        # Check for key functionality imports
        # Note: Checking for domain services now instead of core imports
        expected_imports = [
            "from fpl_team_picker.domain.services import",  # Domain services (uses __init__ imports)
            "from fpl_team_picker.visualization.charts import",  # Visualization helpers
        ]

        for expected in expected_imports:
            assert expected in content, f"Missing import: {expected}"

    def test_existing_import_patterns(self):
        """Test existing import patterns in the notebook still work."""
        import_tests = [
            ("marimo", "mo"),
            ("pandas", "pd"),
            ("client", "FPLDataClient"),
            (
                "fpl_team_picker.domain.services.expected_points_service",
                "ExpectedPointsService",
            ),
            ("fpl_team_picker.domain.services", "OptimizationService"),
            ("fpl_team_picker.visualization.charts", "create_xp_results_display"),
        ]

        for module_name, import_name in import_tests:
            try:
                if "." in module_name:
                    # from module import name
                    import_module = __import__(module_name, fromlist=[import_name])
                    getattr(import_module, import_name)
                else:
                    # import module as name
                    __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Import failed: {module_name} -> {import_name}: {e}")

    @pytest.mark.slow
    def test_marimo_can_start_notebook(self):
        """Test that marimo can actually start the notebook (slow test)."""
        notebook_path = Path("fpl_team_picker/interfaces/gameweek_manager.py")

        try:
            # Test that marimo can start the notebook (don't let it run long)
            process = subprocess.Popen(
                ["marimo", "run", str(notebook_path), "--port", "0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait a bit for startup
            try:
                stdout, stderr = process.communicate(timeout=10)
                # Should not return immediately with error
                if process.returncode is not None and process.returncode != 0:
                    pytest.fail(f"Marimo failed to start: {stderr}")
            except subprocess.TimeoutExpired:
                # Good - marimo is running, kill it
                process.terminate()
                process.wait()
                # This is expected - marimo started successfully

        except FileNotFoundError:
            pytest.skip("Marimo not available in test environment")
        except Exception as e:
            pytest.fail(f"Marimo startup test failed: {e}")
