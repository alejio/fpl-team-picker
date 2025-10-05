import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # 📊 xP Model Accuracy Tracking & Experimentation

        This notebook enables **algorithm development and validation** through:
        - Historical accuracy analysis
        - Algorithm variant comparison (A/B testing)
        - Position-specific performance evaluation
        - Parameter optimization experiments

        **Use this for:** Model development, validation, and research

        **For predictions:** Use the gameweek manager interface
        """
    )
    return


@app.cell
def _():
    # Core imports
    import pandas as pd
    from fpl_team_picker.domain.services.performance_analytics_service import (
        PerformanceAnalyticsService,
        ALGORITHM_VERSIONS,
    )
    from fpl_team_picker.visualization.charts import (
        create_model_accuracy_visualization,
        create_position_accuracy_visualization,
        create_algorithm_comparison_visualization,
    )
    from client import FPLDataClient

    # Initialize services
    analytics_service = PerformanceAnalyticsService()
    client = FPLDataClient()

    return (
        ALGORITHM_VERSIONS,
        PerformanceAnalyticsService,
        analytics_service,
        client,
        create_algorithm_comparison_visualization,
        create_model_accuracy_visualization,
        create_position_accuracy_visualization,
        pd,
    )


@app.cell
def _(mo):
    mo.md("## ⚙️ Configuration")
    return


@app.cell
def _(mo):
    # Gameweek selection for analysis
    current_gw_input = mo.ui.number(
        start=1,
        stop=38,
        step=1,
        value=8,
        label="Current Gameweek:",
    )

    lookback_input = mo.ui.number(
        start=1,
        stop=10,
        step=1,
        value=5,
        label="Lookback Gameweeks:",
    )

    mo.hstack([current_gw_input, lookback_input], justify="start")
    return current_gw_input, lookback_input


@app.cell
def _(ALGORITHM_VERSIONS, mo):
    # Algorithm selection for comparison
    available_algorithms = list(ALGORITHM_VERSIONS.keys())

    algorithm_selector = mo.ui.multiselect(
        options=available_algorithms,
        value=["current"],
        label="Select Algorithms to Compare:",
    )

    mo.vstack(
        [
            algorithm_selector,
            mo.md(f"**Available Algorithms:** {', '.join(available_algorithms)}"),
        ]
    )
    return algorithm_selector, available_algorithms


@app.cell
def _(ALGORITHM_VERSIONS, mo):
    mo.md("### 📋 Algorithm Configurations")
    return


@app.cell
def _(ALGORITHM_VERSIONS, mo, pd):
    # Display algorithm configurations
    algo_configs = []
    for name, config in ALGORITHM_VERSIONS.items():
        algo_configs.append(
            {
                "Algorithm": name,
                "Form Weight": config.form_weight,
                "Form Window": config.form_window,
                "Team Strength": config.use_team_strength,
            }
        )

    config_df = pd.DataFrame(algo_configs)
    mo.ui.table(config_df)
    return algo_configs, config_df


@app.cell
def _(mo):
    mo.md("## 📈 Accuracy Tracking Over Time")
    return


@app.cell
def _(
    algorithm_selector,
    create_model_accuracy_visualization,
    current_gw_input,
    lookback_input,
    mo,
):
    # Model accuracy visualization
    if current_gw_input.value > 1:
        accuracy_viz = create_model_accuracy_visualization(
            target_gameweek=current_gw_input.value,
            lookback_gameweeks=lookback_input.value,
            algorithm_versions=algorithm_selector.value,
            mo_ref=mo,
        )
    else:
        accuracy_viz = mo.md("⚠️ **Select a gameweek > 1 to view accuracy tracking**")

    accuracy_viz
    return (accuracy_viz,)


@app.cell
def _(mo):
    mo.md("## 🎯 Position-Specific Accuracy Analysis")
    return


@app.cell
def _(mo):
    # Position analysis gameweek selector
    position_gw_input = mo.ui.number(
        start=1,
        stop=38,
        step=1,
        value=7,
        label="Gameweek for Position Analysis:",
    )

    position_algo_selector = mo.ui.dropdown(
        options=["current", "v1.0", "experimental_high_form", "experimental_low_form"],
        value="current",
        label="Algorithm Version:",
    )

    mo.hstack([position_gw_input, position_algo_selector], justify="start")
    return position_algo_selector, position_gw_input


@app.cell
def _(
    create_position_accuracy_visualization,
    mo,
    position_algo_selector,
    position_gw_input,
):
    # Position-specific accuracy visualization
    position_accuracy_viz = create_position_accuracy_visualization(
        target_gameweek=position_gw_input.value,
        algorithm_version=position_algo_selector.value,
        mo_ref=mo,
    )

    position_accuracy_viz
    return (position_accuracy_viz,)


@app.cell
def _(mo):
    mo.md("## 🔬 Algorithm Comparison (A/B Testing)")
    return


@app.cell
def _(mo):
    # Algorithm comparison range
    comparison_start_gw = mo.ui.number(
        start=1,
        stop=38,
        step=1,
        value=3,
        label="Start Gameweek:",
    )

    comparison_end_gw = mo.ui.number(
        start=1,
        stop=38,
        step=1,
        value=7,
        label="End Gameweek:",
    )

    mo.hstack([comparison_start_gw, comparison_end_gw], justify="start")
    return comparison_end_gw, comparison_start_gw


@app.cell
def _(ALGORITHM_VERSIONS, mo):
    # Algorithm selection for comparison
    comparison_algos = mo.ui.multiselect(
        options=list(ALGORITHM_VERSIONS.keys()),
        value=["current", "experimental_high_form", "experimental_low_form"],
        label="Algorithms to Compare:",
    )

    comparison_algos
    return (comparison_algos,)


@app.cell
def _(
    comparison_algos,
    comparison_end_gw,
    comparison_start_gw,
    create_algorithm_comparison_visualization,
    mo,
):
    # Algorithm comparison visualization
    if len(comparison_algos.value) > 0:
        algo_comparison_viz = create_algorithm_comparison_visualization(
            start_gw=comparison_start_gw.value,
            end_gw=comparison_end_gw.value,
            algorithm_versions=comparison_algos.value,
            mo_ref=mo,
        )
    else:
        algo_comparison_viz = mo.md("⚠️ **Select at least one algorithm to compare**")

    algo_comparison_viz
    return (algo_comparison_viz,)


@app.cell
def _(mo):
    mo.md(
        """
        ## 💡 Usage Guide

        ### Accuracy Tracking
        - Monitor model performance over time
        - Identify accuracy trends and anomalies
        - Track MAE (Mean Absolute Error) and correlation metrics

        ### Position Analysis
        - Identify which positions need model improvements
        - Compare GKP, DEF, MID, FWD prediction quality
        - Target position-specific refinements

        ### Algorithm Comparison
        - Test form_weight variations (0.5, 0.7, 0.9)
        - Compare form_window sizes (3GW, 5GW, 8GW)
        - A/B test new algorithm parameters
        - Select best performer for gameweek manager

        ### Next Steps
        1. Analyze accuracy trends to identify improvements
        2. Test algorithm variants on historical data
        3. Validate improvements before deployment
        4. Update gameweek manager with best algorithm
        """
    )
    return


def main():
    """CLI entry point for xP accuracy tracking notebook."""
    app.run()


if __name__ == "__main__":
    app.run()
