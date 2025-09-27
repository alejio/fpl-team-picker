"""Clean gameweek manager interface using domain services."""

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell
def _(mo):
    mo.md(r"""# 🎯 FPL Gameweek Manager (Clean Architecture)

    This is a clean version demonstrating the domain services pattern for frontend-agnostic business logic.
    """)
    return


@app.cell
def _(mo):
    from fpl_team_picker.core.data_loader import get_current_gameweek_info

    # Automatically detect current gameweek
    gw_info = get_current_gameweek_info()
    current_gw = gw_info["current_gameweek"]
    status_message = gw_info["message"]

    # Manual override option
    gameweek_input = mo.ui.number(
        value=current_gw,
        start=1,
        stop=38,
        label="Target Gameweek",
    )

    mo.vstack(
        [
            mo.md("### 📅 Current Gameweek Status"),
            mo.md(status_message),
            mo.md("**Target Gameweek:**"),
            gameweek_input,
        ]
    )
    return (gameweek_input,)


@app.cell
def _(gameweek_input, mo, pd):
    # Initialize defaults
    target_gw = gameweek_input.value if gameweek_input.value else None
    gameweek_data = {}
    data_status = "No gameweek selected"

    if target_gw:
        # Use domain service for data orchestration
        from fpl_team_picker.domain.services import DataOrchestrationService
        from fpl_team_picker.adapters.database_repositories import (
            DatabasePlayerRepository,
            DatabaseTeamRepository,
            DatabaseFixtureRepository,
        )

        try:
            # Initialize repositories
            player_repo = DatabasePlayerRepository()
            team_repo = DatabaseTeamRepository()
            fixture_repo = DatabaseFixtureRepository()

            # Create data orchestration service
            data_service = DataOrchestrationService(
                player_repo, team_repo, fixture_repo
            )

            # Load gameweek data
            data_result = data_service.load_gameweek_data(target_gw, form_window=5)

            if data_result.is_success:
                gameweek_data = data_result.value
                players_count = len(gameweek_data["players"])
                teams_count = len(gameweek_data["teams"])
                fixtures_count = len(gameweek_data["fixtures"])
                data_status = f"✅ Loaded: {players_count} players, {teams_count} teams, {fixtures_count} fixtures"
            else:
                data_status = f"❌ Error: {data_result.error.message}"

        except Exception as e:
            data_status = f"❌ Service Error: {str(e)}"

    mo.md(f"**Data Status:** {data_status}")
    return gameweek_data, data_status


@app.cell
def _(gameweek_data, gameweek_input, mo, pd):
    # Initialize defaults
    players_with_xp = pd.DataFrame()
    xp_status = "Load gameweek data first"

    if gameweek_data and gameweek_input.value:
        # Use domain service for expected points calculation
        from fpl_team_picker.domain.services import ExpectedPointsService
        from fpl_team_picker.config import config as xp_config

        try:
            # Create expected points service
            xp_service = ExpectedPointsService()

            # Calculate combined 1GW and 5GW results
            xp_result = xp_service.calculate_combined_results(
                gameweek_data, use_ml_model=xp_config.xp_model.use_ml_model
            )

            if xp_result.is_success:
                players_with_xp = xp_result.value
                model_info = xp_service.get_model_info(xp_config.xp_model.use_ml_model)
                xp_status = f"✅ {model_info['type']} Model: {len(players_with_xp)} players calculated"
            else:
                xp_status = f"❌ XP Error: {xp_result.error.message}"

        except Exception as e:
            xp_status = f"❌ XP Service Error: {str(e)}"

    mo.vstack(
        [
            mo.md("### 🎯 Expected Points Calculation"),
            mo.md(f"**Status:** {xp_status}"),
            mo.ui.table(players_with_xp.head(10))
            if not players_with_xp.empty
            else mo.md("No data"),
        ]
    )
    return players_with_xp, xp_status


@app.cell
def _(gameweek_data, mo, players_with_xp):
    # Transfer optimization example
    optimization_status = "Load expected points data first"

    if not players_with_xp.empty and gameweek_data:
        from fpl_team_picker.domain.services import TransferOptimizationService

        try:
            # Create transfer optimization service
            transfer_service = TransferOptimizationService()

            # Get current squad if available
            current_squad = gameweek_data.get("current_squad")
            team_data = gameweek_data.get("manager_team")

            if current_squad is not None and not current_squad.empty:
                # Try optimization
                optimization_result = transfer_service.optimize_transfers(
                    players_with_xp=players_with_xp,
                    current_squad=current_squad,
                    team_data=team_data,
                    teams=gameweek_data["teams"],
                    optimization_horizon="5gw",
                )

                if optimization_result.is_success:
                    optimization_status = "✅ Transfer optimization completed"
                else:
                    optimization_status = (
                        f"❌ Optimization Error: {optimization_result.error.message}"
                    )
            else:
                optimization_status = (
                    "⚠️ No current squad data available for optimization"
                )

        except Exception as e:
            optimization_status = f"❌ Transfer Service Error: {str(e)}"

    mo.vstack(
        [
            mo.md("### 🔄 Transfer Optimization"),
            mo.md(f"**Status:** {optimization_status}"),
        ]
    )
    return (optimization_status,)


@app.cell
def _(mo):
    mo.md(r"""## 🏗️ Architecture Summary

    This clean version demonstrates:

    1. **Domain Services**: Business logic abstracted from UI
    2. **Repository Pattern**: Data access abstraction
    3. **Result Types**: Structured error handling
    4. **Frontend Agnostic**: Same services work for any frontend

    Key benefits:
    - ✅ Business logic testable in isolation
    - ✅ Frontend can be swapped (React, CLI, etc.)
    - ✅ Structured error handling
    - ✅ Clean separation of concerns
    """)
    return


if __name__ == "__main__":
    app.run()
