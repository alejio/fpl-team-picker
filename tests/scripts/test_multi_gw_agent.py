"""Tests for multi_gw_agent.py CLI script."""

import sys
from pathlib import Path
import importlib.util
from unittest.mock import Mock, patch

import pytest
import pandas as pd
import typer.testing

# Add project and scripts to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

# Import the script module
spec = importlib.util.spec_from_file_location(
    "multi_gw_agent",
    project_root / "scripts" / "multi_gw_agent.py",
)
multi_gw_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_gw_agent)

app = multi_gw_agent.app


class TestMultiGWAgentCLI:
    """Test multi_gw_agent.py CLI functionality."""

    @pytest.fixture
    def sample_squad(self):
        """Create a sample 15-player squad."""
        return pd.DataFrame(
            {
                "player_id": list(range(1, 16)),
                "web_name": [f"Player{i}" for i in range(1, 16)],
                "name": [f"Player {i}" for i in range(1, 16)],
                "position": ["GKP", "GKP"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
                "price": [5.0] * 2 + [6.0] * 5 + [8.0] * 5 + [10.0] * 3,
            }
        )

    @pytest.fixture
    def sample_gameweek_data(self, sample_squad):
        """Create sample gameweek data."""
        return {
            "players": pd.DataFrame(
                {
                    "player_id": [1, 2, 3],
                    "web_name": ["Salah", "Son", "Haaland"],
                    "position": ["MID", "MID", "FWD"],
                }
            ),
            "teams": pd.DataFrame({"team_id": [1, 2, 3], "team_name": ["A", "B", "C"]}),
            "fixtures": pd.DataFrame({"fixture_id": [1, 2, 3]}),
            "current_squad": sample_squad,
            "manager_team": {"bank": 50.0, "transfers": {"limit": 1}},
        }

    @pytest.fixture
    def mock_multi_gw_plan(self):
        """Create a mock MultiGWPlan."""
        from fpl_team_picker.domain.models.transfer_plan import (
            MultiGWPlan,
            WeeklyTransferPlan,
            StrategyMode,
        )

        return MultiGWPlan(
            start_gameweek=18,
            end_gameweek=20,
            strategy_mode=StrategyMode.BALANCED,
            weekly_plans=[
                WeeklyTransferPlan(
                    gameweek=18,
                    transfers=[],
                    expected_xp=60.0,
                    baseline_xp=60.0,
                    xp_gain=0.0,
                    hit_cost=0,
                    net_gain=0.0,
                    reasoning="Hold FT for better opportunity",
                    confidence="medium",
                ),
                WeeklyTransferPlan(
                    gameweek=19,
                    transfers=[],
                    expected_xp=62.0,
                    baseline_xp=62.0,
                    xp_gain=0.0,
                    hit_cost=0,
                    net_gain=0.0,
                    reasoning="Continue holding",
                    confidence="medium",
                ),
                WeeklyTransferPlan(
                    gameweek=20,
                    transfers=[],
                    expected_xp=65.0,
                    baseline_xp=60.0,
                    xp_gain=5.0,
                    hit_cost=0,
                    net_gain=5.0,
                    reasoning="Use saved transfers",
                    confidence="high",
                ),
            ],
            total_xp_gain=5.0,
            total_hit_cost=0,
            net_roi=5.0,
            best_case_roi=8.0,
            worst_case_roi=2.0,
            final_reasoning="Strategic plan to bank transfers",
        )

    def test_main_command_data_loading_error(self):
        """Test handling of data loading errors."""
        # Setup mock to raise error
        mock_data_service = Mock()
        mock_data_service.load_gameweek_data.side_effect = ValueError(
            "Data loading failed"
        )

        # Patch the classes in the module's namespace
        with patch.object(
            multi_gw_agent, "DataOrchestrationService", return_value=mock_data_service
        ):
            runner = typer.testing.CliRunner()
            result = runner.invoke(app, ["main", "--gameweek", "18"])

            # Should handle error gracefully
            assert (
                result.exit_code != 0
                or "error" in result.stdout.lower()
                or "Error" in result.stdout
            )

    def test_main_command_agent_error(self, sample_gameweek_data):
        """Test handling of agent generation errors."""
        # Setup mocks
        mock_data_service = Mock()
        mock_data_service.load_gameweek_data.return_value = sample_gameweek_data

        mock_agent_service = Mock()
        mock_agent_service.generate_multi_gw_plan.side_effect = Exception("Agent error")

        # Patch the classes in the module's namespace
        with patch.object(
            multi_gw_agent, "DataOrchestrationService", return_value=mock_data_service
        ):
            with patch.object(
                multi_gw_agent,
                "TransferPlanningAgentService",
                return_value=mock_agent_service,
            ):
                runner = typer.testing.CliRunner()
                result = runner.invoke(app, ["main", "--gameweek", "18"])

                # Should handle error gracefully
                assert (
                    result.exit_code != 0
                    or "error" in result.stdout.lower()
                    or "Error" in result.stdout
                )

    def test_help_command(self):
        """Test that help command works."""
        runner = typer.testing.CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert (
            "Multi-GW Transfer Planning Agent" in result.stdout
            or "help" in result.stdout.lower()
        )

    def test_main_command_help(self):
        """Test that main command help works."""
        runner = typer.testing.CliRunner()
        result = runner.invoke(app, ["main", "--help"])

        assert result.exit_code == 0
        # Should show parameter descriptions
        assert "gameweek" in result.stdout.lower() or "horizon" in result.stdout.lower()
