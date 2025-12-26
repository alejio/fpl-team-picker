#!/usr/bin/env python3
"""Transfer Recommendation Agent - Single-GW recommendations with multi-GW context.

This CLI uses an LLM-based agent to generate 3-5 ranked transfer scenarios for a single
gameweek, considering the next 3 gameweeks for strategic context (DGWs, fixture runs, chip timing).

Usage:
    # Default: 5 scenarios, balanced strategy
    uv run python scripts/transfer_recommendation_agent.py --gameweek 18

    # Conservative strategy with 3 options
    uv run python scripts/transfer_recommendation_agent.py --gameweek 18 --strategy conservative --num-options 3

    # Aggressive with lower hit threshold
    uv run python scripts/transfer_recommendation_agent.py --gameweek 18 --strategy aggressive --roi-threshold 4.0
"""

import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fpl_team_picker.domain.models.transfer_plan import StrategyMode  # noqa: E402
from fpl_team_picker.domain.models.transfer_recommendation import (  # noqa: E402
    SingleGWRecommendation,
)
from fpl_team_picker.domain.services.data_orchestration_service import (  # noqa: E402
    DataOrchestrationService,
)
from fpl_team_picker.domain.services.transfer_planning_agent_service import (  # noqa: E402
    TransferPlanningAgentService,
)

app = typer.Typer(
    help="FPL Transfer Recommendation Agent - Single-GW recommendations with multi-GW context"
)
console = Console()


@app.command()
def main(
    gameweek: int = typer.Option(
        ..., "--gameweek", "-g", help="Target gameweek (1-38)"
    ),
    strategy: str = typer.Option(
        "balanced",
        "--strategy",
        "-s",
        help="Strategy mode: balanced, conservative, aggressive",
    ),
    num_options: int = typer.Option(
        5, "--num-options", "-n", help="Number of scenarios to generate (3-5)"
    ),
    roi_threshold: float = typer.Option(
        5.0, "--roi-threshold", "-r", help="Minimum 3GW xP gain required for -4 hit"
    ),
    free_transfers_override: int | None = typer.Option(
        None,
        "--free-transfers",
        "-ft",
        help="Override calculated free transfers (use if calculation is wrong)",
    ),
    model: str = typer.Option(
        "claude-sonnet-4-5",
        "--model",
        "-m",
        help="Anthropic model to use (e.g., claude-haiku-3-7 for faster/cheaper)",
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """Generate single-GW transfer recommendations using LLM agent.

    The agent will:
    1. Analyze current squad weaknesses
    2. Get multi-GW xP predictions
    3. Analyze fixture context (DGWs, fixture runs)
    4. Check template player landscape
    5. Validate top scenarios with SA optimizer
    6. Return 3-5 ranked transfer options + hold baseline
    """
    try:
        # Validate inputs
        if not (1 <= gameweek <= 38):
            logger.error(
                f"[red]Error: gameweek must be 1-38, got {gameweek}[/red]", style="bold"
            )
            raise typer.Exit(1)

        if not (3 <= num_options <= 5):
            logger.error(
                f"[red]Error: num-options must be 3-5, got {num_options}[/red]",
                style="bold",
            )
            raise typer.Exit(1)

        try:
            strategy_mode = StrategyMode(strategy.lower())
        except ValueError:
            valid_strategies = [s.value for s in StrategyMode]
            logger.error(
                f"[red]Error: strategy must be one of {valid_strategies}, got '{strategy}'[/red]",
                style="bold",
            )
            raise typer.Exit(1)

        # Configure logging
        if debug:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.remove()
            logger.add(sys.stderr, level="INFO")

        console.print("\n[bold cyan]🤖 Transfer Recommendation Agent[/bold cyan]")
        console.print(
            f"[cyan]Target: GW{gameweek} | Strategy: {strategy_mode.value} | Options: {num_options} | ROI Threshold: +{roi_threshold} xP[/cyan]\n"
        )

        # Load gameweek data
        console.print("[yellow]📊 Loading gameweek data...[/yellow]")
        data_service = DataOrchestrationService()
        gw_data = data_service.load_gameweek_data(gameweek)

        # Get current squad (15 players)
        current_squad = gw_data["current_squad"]
        if len(current_squad) != 15:
            console.print(
                f"[red]Error: Expected 15-player squad, got {len(current_squad)}[/red]"
            )
            raise typer.Exit(1)

        # Get free transfers using the new DataOrchestrationService method
        team_data = gw_data.get("manager_team")
        calculated_fts = data_service.get_free_transfers(team_data)

        # Allow manual override if calculation is wrong (e.g., AFCON special grants)
        if free_transfers_override is not None:
            free_transfers = free_transfers_override
            console.print(
                f"[yellow]⚠️  Free Transfers (Overridden): {free_transfers} (calculated: {calculated_fts})[/yellow]"
            )
            # Update the gameweek_data with overridden value
            if team_data:
                team_data["transfers"]["limit"] = free_transfers
        else:
            free_transfers = calculated_fts
            console.print(f"[cyan]Free Transfers Available: {free_transfers}[/cyan]")

        # Initialize agent service
        console.print("[yellow]🧠 Initializing agent service...[/yellow]")
        agent_service = TransferPlanningAgentService(model=model, debug=debug)

        # Generate recommendations
        console.print(
            "[yellow]🔍 Agent analyzing transfer options (this may take 30-60 seconds)...[/yellow]\n"
        )
        recommendations = agent_service.generate_single_gw_recommendations(
            target_gameweek=gameweek,
            current_squad=current_squad,
            gameweek_data=gw_data,
            strategy_mode=strategy_mode,
            hit_roi_threshold=roi_threshold,
            num_recommendations=num_options,
        )

        # Print formatted output
        print_recommendations(recommendations)

        logger.info("\n[green]✅ Recommendations generated successfully![/green]\n")

    except KeyboardInterrupt:
        logger.info("\n[yellow]⚠️  Interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        logger.error(f"\n[red]❌ Error: {e}[/red]", style="bold")
        if debug:
            logger.exception("Full traceback:")
        raise typer.Exit(1)


def print_recommendations(rec: SingleGWRecommendation):
    """Print recommendations in rich formatted output."""

    # Header
    console.print(
        f"\n[bold cyan]📊 Transfer Recommendations for GW{rec.target_gameweek}[/bold cyan]\n"
    )
    console.print(
        f"[cyan]Budget: £{rec.budget_available:.1f}m | Free Transfers: {rec.current_free_transfers}[/cyan]\n"
    )

    # Hold option
    console.print("[bold yellow]🛑 HOLD OPTION (Baseline)[/bold yellow]")
    hold_table = Table(show_header=False, box=None, padding=(0, 2))
    hold_table.add_row("GW1 xP:", f"[green]{rec.hold_option.xp_gw1:.1f}[/green] points")
    hold_table.add_row("3GW xP:", f"[green]{rec.hold_option.xp_3gw:.1f}[/green] points")
    hold_table.add_row(
        "Next Week FTs:", f"[cyan]{rec.hold_option.free_transfers_next_week}[/cyan]"
    )
    hold_table.add_row("Reasoning:", f"[white]{rec.hold_option.reasoning}[/white]")
    console.print(hold_table)
    console.print()

    # Transfer scenarios
    for i, scenario in enumerate(rec.recommended_scenarios, 1):
        is_top = scenario.scenario_id == rec.top_recommendation_id
        star = "⭐" if is_top else "📍"
        style = "bold green" if is_top else "bold white"

        console.print(
            f"[{style}]{star} OPTION {i}: {scenario.scenario_id.upper()}[/{style}]"
        )

        # Transfers table
        if scenario.transfers:
            transfer_table = Table(show_header=True, box=None, padding=(0, 1))
            transfer_table.add_column("Out", style="red")
            transfer_table.add_column("→", justify="center")
            transfer_table.add_column("In", style="green")
            transfer_table.add_column("Cost", justify="right")

            for t in scenario.transfers:
                cost_str = f"£{t.cost:+.1f}m" if t.cost != 0 else "£0.0m"
                cost_style = "red" if t.cost > 0 else "green" if t.cost < 0 else "white"
                transfer_table.add_row(
                    t.player_out_name,
                    "→",
                    t.player_in_name,
                    f"[{cost_style}]{cost_str}[/{cost_style}]",
                )

            console.print(transfer_table)

        # Metrics table
        metrics_table = Table(show_header=False, box=None, padding=(0, 2))

        # GW1 metrics
        gw1_gain_color = "green" if scenario.xp_gain_gw1 > 0 else "red"
        net_gw1_color = "green" if scenario.net_gain_gw1 > 0 else "red"
        metrics_table.add_row(
            "GW1:",
            f"[cyan]{scenario.xp_gw1:.1f}[/cyan] xP ([{gw1_gain_color}]{scenario.xp_gain_gw1:+.1f}[/{gw1_gain_color}]) | Net: [{net_gw1_color}]{scenario.net_gain_gw1:+.1f}[/{net_gw1_color}]",
        )

        # 3GW metrics
        roi_color = "green" if scenario.net_roi_3gw > 0 else "red"
        metrics_table.add_row(
            "3GW:",
            f"[cyan]{scenario.xp_3gw:.1f}[/cyan] xP ([green]{scenario.xp_gain_3gw:+.1f}[/green]) | ROI: [{roi_color}]{scenario.net_roi_3gw:+.1f}[/{roi_color}]",
        )

        # Hit cost and confidence
        hit_str = f"[red]{scenario.hit_cost}[/red]" if scenario.hit_cost < 0 else "0"
        confidence_color = {
            "high": "green",
            "medium": "yellow",
            "low": "red",
        }.get(scenario.confidence, "white")
        metrics_table.add_row(
            "Hit Cost:",
            f"{hit_str} | Confidence: [{confidence_color}]{scenario.confidence}[/{confidence_color}]",
        )

        # SA validation
        if scenario.sa_validated and scenario.sa_deviation is not None:
            deviation_color = "green" if abs(scenario.sa_deviation) < 1.0 else "yellow"
            metrics_table.add_row(
                "SA Deviation:",
                f"[{deviation_color}]{scenario.sa_deviation:+.1f}[/{deviation_color}] xP vs optimizer",
            )

        console.print(metrics_table)

        # Context flags
        flags = []
        if scenario.leverages_dgw:
            flags.append("[cyan]DGW[/cyan]")
        if scenario.leverages_fixture_swing:
            flags.append("[yellow]Fixture Swing[/yellow]")
        if scenario.prepares_for_chip:
            flags.append("[magenta]Chip Prep[/magenta]")

        if flags:
            console.print(f"  Flags: {' | '.join(flags)}")

        # Reasoning
        console.print(f"  [white italic]{scenario.reasoning}[/white italic]")
        console.print()

    # Final recommendation
    console.print("[bold green]🏆 TOP RECOMMENDATION[/bold green]")
    top_rec_table = Table(show_header=False, box=None, padding=(0, 2))
    top_rec_table.add_row(
        "Choice:", f"[bold cyan]{rec.top_recommendation_id}[/bold cyan]"
    )
    top_rec_table.add_row("Reasoning:", f"[white]{rec.final_reasoning}[/white]")
    console.print(top_rec_table)

    # Context analysis summary
    if rec.context_analysis:
        console.print("\n[bold]📋 Context Analysis[/bold]")
        context_table = Table(show_header=False, box=None, padding=(0, 2))

        if "dgw_opportunities" in rec.context_analysis:
            dgw_str = (
                ", ".join(rec.context_analysis["dgw_opportunities"])
                if rec.context_analysis["dgw_opportunities"]
                else "None detected"
            )
            context_table.add_row("DGW Opportunities:", f"[cyan]{dgw_str}[/cyan]")

        if "fixture_swings" in rec.context_analysis:
            swing_str = (
                ", ".join(rec.context_analysis["fixture_swings"])
                if rec.context_analysis["fixture_swings"]
                else "None detected"
            )
            context_table.add_row("Fixture Swings:", f"[yellow]{swing_str}[/yellow]")

        if "chip_timing" in rec.context_analysis:
            context_table.add_row(
                "Chip Timing:",
                f"[magenta]{rec.context_analysis['chip_timing']}[/magenta]",
            )

        console.print(context_table)


if __name__ == "__main__":
    app()
