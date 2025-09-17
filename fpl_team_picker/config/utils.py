"""
Configuration Utilities

Helper functions for managing FPL configuration including validation,
export, and debugging utilities.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from .settings import FPLConfig, load_config


def export_config_to_json(config: FPLConfig, output_path: Path) -> None:
    """
    Export configuration to JSON file

    Args:
        config: FPLConfig instance to export
        output_path: Path where to save the JSON file
    """
    config_dict = config.model_dump()

    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    print(f"✅ Configuration exported to {output_path}")


def validate_config_file(config_path: Path) -> List[str]:
    """
    Validate a configuration file and return any issues

    Args:
        config_path: Path to configuration file

    Returns:
        List of validation messages (empty if valid)
    """
    try:
        load_config(config_path=config_path)
        print("✅ Configuration file is valid!")
        return []
    except Exception as e:
        return [f"Configuration validation failed: {str(e)}"]


def compare_configs(config1: FPLConfig, config2: FPLConfig) -> Dict[str, Any]:
    """
    Compare two configurations and return differences

    Args:
        config1: First configuration
        config2: Second configuration

    Returns:
        Dictionary of differences
    """
    dict1 = config1.model_dump()
    dict2 = config2.model_dump()

    differences = {}

    def compare_dicts(d1, d2, path=""):
        for key in set(d1.keys()) | set(d2.keys()):
            current_path = f"{path}.{key}" if path else key

            if key not in d1:
                differences[current_path] = {"config1": "<missing>", "config2": d2[key]}
            elif key not in d2:
                differences[current_path] = {"config1": d1[key], "config2": "<missing>"}
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                compare_dicts(d1[key], d2[key], current_path)
            elif d1[key] != d2[key]:
                differences[current_path] = {"config1": d1[key], "config2": d2[key]}

    compare_dicts(dict1, dict2)
    return differences


def print_config_summary(config: FPLConfig) -> None:
    """
    Print a human-readable summary of the configuration

    Args:
        config: Configuration to summarize
    """
    print("🔧 FPL Configuration Summary")
    print("=" * 50)

    print("📊 XP Model:")
    print(f"  • Form Weight: {config.xp_model.form_weight:.1%} recent form")
    print(f"  • Form Window: {config.xp_model.form_window} gameweeks")
    print(f"  • Debug Mode: {'On' if config.xp_model.debug else 'Off'}")

    print("\n💪 Team Strength:")
    print(
        f"  • Historical Transition: GW{config.team_strength.historical_transition_gw}+"
    )
    print(f"  • Rolling Window: {config.team_strength.rolling_window_size} games")
    print(
        f"  • Strength Range: {config.team_strength.min_strength:.1f} - {config.team_strength.max_strength:.1f}"
    )

    print("\n🔄 Optimization:")
    print(f"  • Transfer Cost: {config.optimization.transfer_cost} points")
    print(f"  • Max Transfers: {config.optimization.max_transfers}")
    print(
        f"  • Premium Threshold: £{config.optimization.premium_player_threshold:.1f}m"
    )

    print("\n📈 Visualization:")
    print(f"  • Default Page Size: {config.visualization.default_page_size}")
    print(
        f"  • Chart Size: {config.visualization.chart_width}x{config.visualization.chart_height}"
    )

    print("\n⚽ Minutes Model:")
    print(f"  • GKP Full Game Prob: {config.minutes_model.gkp_full_game_prob:.1%}")
    print(
        f"  • Premium Threshold: £{config.minutes_model.premium_price_threshold:.1f}m"
    )

    print("=" * 50)


def create_config_template() -> str:
    """
    Create a configuration template with all available options

    Returns:
        JSON string template
    """
    template_config = FPLConfig()
    return template_config.model_dump_json(indent=2)


def get_env_var_examples() -> Dict[str, str]:
    """
    Get examples of environment variables that can override config

    Returns:
        Dictionary of environment variable examples
    """
    return {
        "FPL_XP_MODEL_FORM_WEIGHT": "0.8",
        "FPL_XP_MODEL_DEBUG": "false",
        "FPL_TEAM_STRENGTH_HISTORICAL_TRANSITION_GW": "10",
        "FPL_OPTIMIZATION_TRANSFER_COST": "3.5",
        "FPL_VISUALIZATION_DEFAULT_PAGE_SIZE": "20",
    }


if __name__ == "__main__":
    # Example usage
    config = load_config()
    print_config_summary(config)

    # Export template
    template_path = Path("config_template.json")
    with open(template_path, "w") as f:
        f.write(create_config_template())
    print(f"📄 Configuration template saved to {template_path}")
