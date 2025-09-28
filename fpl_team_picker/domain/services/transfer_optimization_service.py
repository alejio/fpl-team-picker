"""Transfer optimization service."""

from typing import Dict, Any, List, Set, Optional
import pandas as pd


class TransferOptimizationService:
    """Service for transfer optimization and squad management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the service with configuration.

        Args:
            config: Configuration dictionary for optimization
        """
        self.config = config or {}

    def optimize_transfers(
        self,
        players_with_xp: pd.DataFrame,
        current_squad: pd.DataFrame,
        team_data: Dict[str, Any],
        teams: pd.DataFrame,
        optimization_horizon: str = "5gw",
        must_include_ids: Optional[Set[int]] = None,
        must_exclude_ids: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:
        """Optimize transfers for the current squad.

        Args:
            players_with_xp: DataFrame with expected points calculations - guaranteed clean
            current_squad: Current squad data - guaranteed clean
            team_data: Manager team data - guaranteed clean
            teams: Teams reference data - guaranteed clean
            optimization_horizon: "1gw" or "5gw" for optimization horizon
            must_include_ids: Set of player IDs that must be included
            must_exclude_ids: Set of player IDs that must be excluded

        Returns:
            Optimization results
        """
        # Set defaults for constraints
        must_include_ids = must_include_ids or set()
        must_exclude_ids = must_exclude_ids or set()

        # Load optimization function
        from fpl_team_picker.optimization.optimizer import (
            optimize_team_with_transfers,
        )

        # Perform optimization with correct parameter names
        optimization_result = optimize_team_with_transfers(
            current_squad=current_squad,
            team_data=team_data,
            players_with_xp=players_with_xp,
            must_include_ids=must_include_ids,
            must_exclude_ids=must_exclude_ids,
        )

        # Unpack the tuple returned by optimize_team_with_transfers
        # Returns: (marimo_component, optimal_squad_df, optimization_results)
        display_component, optimal_squad, best_scenario = optimization_result

        return {
            "display_component": display_component,
            "optimal_squad": optimal_squad,
            "best_scenario": best_scenario,
            "horizon": optimization_horizon,
            "constraints": {
                "must_include": list(must_include_ids),
                "must_exclude": list(must_exclude_ids),
            },
        }

    def get_starting_eleven_from_squad(
        self,
        current_squad: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Get the optimal starting eleven from a 15-player squad.

        Args:
            current_squad: DataFrame containing exactly 15 players - guaranteed clean

        Returns:
            Starting eleven player list
        """
        from fpl_team_picker.optimization.optimizer import get_best_starting_11

        starting_11_data = get_best_starting_11(current_squad)
        return starting_11_data[0]  # Extract just the player list from tuple

    def get_optimal_team_from_database(
        self,
        players_with_xp: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Build an optimal 11-player team from the full player database.

        This method is designed for analysis and testing purposes where you want
        to build the theoretically best team without squad constraints.

        Args:
            players_with_xp: DataFrame containing all available players - guaranteed clean

        Returns:
            Optimal starting eleven
        """
        # Filter out unavailable players
        available_players = players_with_xp[
            ~players_with_xp["status"].isin(["i", "s", "u"])
        ].copy()

        # Get the best players by position for a balanced team
        return self._select_optimal_team_from_all_players(available_players)

    def get_captain_recommendation_from_squad(
        self,
        current_squad: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Get captain recommendation from a 15-player squad.

        Args:
            current_squad: DataFrame containing exactly 15 players - guaranteed clean

        Returns:
            Captain recommendation
        """
        # Convert DataFrame to list for captain selection
        candidate_list = current_squad.to_dict("records")

        # Get the best captain based on xP
        best_captain = max(candidate_list, key=lambda p: p.get("xP", 0))
        return {
            "player_id": best_captain["player_id"],
            "web_name": best_captain["web_name"],
            "position": best_captain["position"],
            "xP": best_captain.get("xP", 0),
            "reason": "Highest expected points in squad",
        }

    def get_captain_recommendation_from_database(
        self,
        players_with_xp: pd.DataFrame,
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """Get captain recommendation from the full player database.

        This method is designed for analysis purposes where you want to find
        the best captain from all available players.

        Args:
            players_with_xp: DataFrame containing all available players - guaranteed clean
            top_n: Number of top players to consider for captain selection

        Returns:
            Captain recommendation
        """
        # Filter out unavailable players and get top candidates
        available_players = players_with_xp[
            ~players_with_xp["status"].isin(["i", "s", "u"])
        ].copy()

        # Get top players by xP
        top_candidates = available_players.nlargest(top_n, "xP")
        candidate_list = top_candidates.to_dict("records")

        # Get the best captain based on xP
        best_captain = candidate_list[0]  # Already sorted by xP descending
        return {
            "player_id": best_captain["player_id"],
            "web_name": best_captain["web_name"],
            "position": best_captain["position"],
            "xP": best_captain.get("xP", 0),
            "reason": f"Highest expected points among all {len(available_players)} available players",
        }

    def analyze_budget_situation(
        self, current_squad: pd.DataFrame, team_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze budget situation including sellable player values.

        Args:
            current_squad: Current squad data - guaranteed clean
            team_data: Manager team data - guaranteed clean

        Returns:
            Budget analysis
        """
        from fpl_team_picker.optimization.optimizer import (
            calculate_total_budget_pool,
        )

        return calculate_total_budget_pool(current_squad, team_data)

    def get_premium_acquisition_plan(
        self,
        target_player_price: float,
        current_squad: pd.DataFrame,
        team_data: Dict[str, Any],
        players_with_xp: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Create a plan for acquiring premium players through multiple transfers.

        Args:
            target_player_price: Price of the target premium player
            current_squad: Current squad data - guaranteed clean
            team_data: Manager team data - guaranteed clean
            players_with_xp: Available players - guaranteed clean

        Returns:
            Acquisition plan
        """
        from fpl_team_picker.optimization.optimizer import (
            premium_acquisition_planner,
        )

        return premium_acquisition_planner(
            target_price=target_player_price,
            current_squad_df=current_squad,
            team_data=team_data,
            available_players_df=players_with_xp,
        )

    def _select_optimal_team_from_all_players(
        self, players: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Select optimal starting 11 from the full player database.

        Args:
            players: DataFrame of all available players - guaranteed clean

        Returns:
            List of 11 player dictionaries forming optimal starting team
        """
        # Sort players by xP within each position
        players_sorted = players.sort_values("xP", ascending=False)

        # Group by position
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for _, player in players_sorted.iterrows():
            position = player["position"]
            if position in by_position:
                by_position[position].append(player.to_dict())

        # Formation: 1 GKP, 4 DEF, 4 MID, 2 FWD (4-4-2 formation)
        formation = {"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2}

        starting_11 = []
        for position, needed in formation.items():
            available = by_position[position][:needed]  # Take top players
            starting_11.extend(available)

        return starting_11

    def optimize_transfers_simple(
        self,
        players_with_xp: pd.DataFrame,
        current_squad: pd.DataFrame,
        gameweek_data: Dict[str, Any],
        use_5gw_horizon: bool = True,
    ) -> Dict[str, Any]:
        """Simplified transfer optimization method for clean notebook interface.

        Args:
            players_with_xp: DataFrame with calculated expected points - guaranteed clean
            current_squad: Current squad DataFrame - guaranteed clean
            gameweek_data: Complete gameweek data - guaranteed clean
            use_5gw_horizon: Whether to use 5GW horizon (True) or 1GW (False)

        Returns:
            Simplified transfer recommendations
        """
        # Extract required data from gameweek_data
        teams = gameweek_data.get("teams", pd.DataFrame())
        team_data = gameweek_data.get("manager_team", {})

        # Set default team data if not available
        if not team_data:
            team_data = {
                "bank_balance": 0.0,
                "transfers_made": 0,
            }

        # Determine optimization horizon
        horizon = "5gw" if use_5gw_horizon else "1gw"

        # Call the main optimization method
        opt_data = self.optimize_transfers(
            players_with_xp=players_with_xp,
            current_squad=current_squad,
            team_data=team_data,
            teams=teams,
            optimization_horizon=horizon,
        )

        # Create simplified transfer recommendations
        transfer_recommendations = []
        if opt_data.get("best_scenario"):
            best_scenario = opt_data["best_scenario"]
            transfer_recommendations.append(
                {
                    "transfer_count": best_scenario.get("transfers", 0),
                    "net_xp_gain": best_scenario.get("xp_gain", 0.0),
                    "total_cost": best_scenario.get("cost", 0.0),
                    "summary": best_scenario.get("description", "Transfer scenario"),
                }
            )

        return {
            "transfer_recommendations": transfer_recommendations,
            "optimization_horizon": horizon,
            "analysis_complete": True,
        }

    def validate_optimization_constraints(
        self,
        must_include_ids: Optional[Set[int]] = None,
        must_exclude_ids: Optional[Set[int]] = None,
        budget_limit: float = 100.0,
    ) -> Dict[str, Any]:
        """Validate optimization constraints.

        Args:
            must_include_ids: Set of player IDs that must be included
            must_exclude_ids: Set of player IDs that must be excluded
            budget_limit: Budget limit in millions

        Returns:
            Validation result with constraints
        """
        must_include_ids = must_include_ids or set()
        must_exclude_ids = must_exclude_ids or set()

        # Check for conflicts
        conflicts = must_include_ids.intersection(must_exclude_ids)

        return {
            "valid": len(conflicts) == 0,
            "conflicts": list(conflicts),
            "must_include_count": len(must_include_ids),
            "must_exclude_count": len(must_exclude_ids),
            "budget_limit": budget_limit,
        }
