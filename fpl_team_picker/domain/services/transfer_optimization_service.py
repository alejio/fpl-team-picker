"""Transfer optimization service."""

from typing import Dict, Any, List, Set, Optional
import pandas as pd

from fpl_team_picker.domain.common.result import Result, DomainError, ErrorType


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
    ) -> Result[Dict[str, Any]]:
        """Optimize transfers for the current squad.

        Args:
            players_with_xp: DataFrame with expected points calculations (full database)
            current_squad: Current squad data (exactly 15 players)
            team_data: Manager team data with required fields
            teams: Teams reference data (exactly 20 teams)
            optimization_horizon: "1gw" or "5gw" for optimization horizon
            must_include_ids: Set of player IDs that must be included
            must_exclude_ids: Set of player IDs that must be excluded

        Returns:
            Result containing optimization results or error
        """
        try:
            # Validate players_with_xp (should be full database)
            if len(players_with_xp) < 100:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Expected full player database (>100 players), got {len(players_with_xp)}",
                    )
                )

            required_player_columns = {"player_id", "position", "web_name", "xP"}
            missing_player_columns = required_player_columns - set(
                players_with_xp.columns
            )
            if missing_player_columns:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Players database missing required columns: {missing_player_columns}",
                    )
                )

            # Validate current_squad (should be exactly 15 players)
            if len(current_squad) != 15:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Current squad must contain exactly 15 players, got {len(current_squad)}",
                    )
                )

            required_squad_columns = {"player_id", "position", "web_name"}
            missing_squad_columns = required_squad_columns - set(current_squad.columns)
            if missing_squad_columns:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Current squad missing required columns: {missing_squad_columns}",
                    )
                )

            # Validate team_data structure
            required_team_fields = {"bank_balance", "transfers_made"}
            missing_team_fields = required_team_fields - set(team_data.keys())
            if missing_team_fields:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Team data missing required fields: {missing_team_fields}",
                    )
                )

            # Validate teams (should be exactly 20 teams)
            if len(teams) != 20:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Teams data must contain exactly 20 teams, got {len(teams)}",
                    )
                )

            # Validate optimization horizon
            if optimization_horizon not in ["1gw", "5gw"]:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Invalid optimization horizon: {optimization_horizon}. Must be '1gw' or '5gw'",
                    )
                )

            # Set defaults for constraints
            must_include_ids = must_include_ids or set()
            must_exclude_ids = must_exclude_ids or set()

            # Validate constraints
            constraint_validation = self.validate_optimization_constraints(
                must_include_ids, must_exclude_ids, players_with_xp
            )
            if constraint_validation.is_failure:
                return constraint_validation

            # Load optimization function
            from fpl_team_picker.optimization.optimizer import (
                optimize_team_with_transfers,
            )

            # Perform optimization WITHOUT modifying global config
            # Instead, we'll pass the horizon as a parameter or handle it differently
            optimization_result = optimize_team_with_transfers(
                expected_points_df=players_with_xp,
                current_squad_df=current_squad,
                teams_df=teams,
                team_data=team_data,
                must_include_player_ids=must_include_ids,
                must_exclude_player_ids=must_exclude_ids,
                # Note: If the function requires horizon, it should be a parameter
                # not a global config mutation
            )

            # Validate optimization result
            if optimization_result is None:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.CALCULATION_ERROR,
                        message="Optimization returned no result",
                    )
                )

            return Result(
                value={
                    "optimization_result": optimization_result,
                    "horizon": optimization_horizon,
                    "constraints": {
                        "must_include": list(must_include_ids),
                        "must_exclude": list(must_exclude_ids),
                    },
                }
            )

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Transfer optimization failed: {str(e)}",
                )
            )

    def get_starting_eleven_from_squad(
        self,
        current_squad: pd.DataFrame,
    ) -> Result[List[Dict[str, Any]]]:
        """Get the optimal starting eleven from a 15-player squad.

        Args:
            current_squad: DataFrame containing exactly 15 players from manager's squad

        Returns:
            Result containing starting eleven or error
        """
        try:
            from fpl_team_picker.optimization.optimizer import get_best_starting_11

            # Validate input is a proper squad
            if len(current_squad) != 15:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Squad must contain exactly 15 players, got {len(current_squad)}",
                    )
                )

            # Validate required columns
            required_columns = {"player_id", "position", "web_name"}
            missing_columns = required_columns - set(current_squad.columns)
            if missing_columns:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Squad missing required columns: {missing_columns}",
                    )
                )

            starting_11_data = get_best_starting_11(current_squad)
            starting_11 = starting_11_data[0]  # Extract just the player list from tuple

            # Validate we got exactly 11 players
            if len(starting_11) != 11:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.CALCULATION_ERROR,
                        message=f"Starting eleven selection returned {len(starting_11)} players instead of 11",
                    )
                )

            return Result(value=starting_11)

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Starting eleven selection failed: {str(e)}",
                )
            )

    def get_optimal_team_from_database(
        self,
        players_with_xp: pd.DataFrame,
    ) -> Result[List[Dict[str, Any]]]:
        """Build an optimal 11-player team from the full player database.

        This method is designed for analysis and testing purposes where you want
        to build the theoretically best team without squad constraints.

        Args:
            players_with_xp: DataFrame containing all available players with xP calculations

        Returns:
            Result containing optimal starting eleven or error
        """
        try:
            # Validate this is the full database (not a squad)
            if len(players_with_xp) < 100:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Expected full player database (>100 players), got {len(players_with_xp)} players. Use get_starting_eleven_from_squad() for squad data.",
                    )
                )

            # Validate required columns
            required_columns = {"player_id", "position", "web_name", "xP", "status"}
            missing_columns = required_columns - set(players_with_xp.columns)
            if missing_columns:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Player database missing required columns: {missing_columns}",
                    )
                )

            # Filter out unavailable players
            available_players = players_with_xp[
                ~players_with_xp["status"].isin(["i", "s", "u"])
            ].copy()

            if len(available_players) < 15:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Insufficient available players: {len(available_players)} (need at least 15)",
                    )
                )

            # Get the best players by position for a balanced team
            starting_11 = self._select_optimal_team_from_all_players(available_players)

            # Validate we got exactly 11 players
            if len(starting_11) != 11:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.CALCULATION_ERROR,
                        message=f"Optimal team selection returned {len(starting_11)} players instead of 11",
                    )
                )

            return Result(value=starting_11)

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Optimal team selection failed: {str(e)}",
                )
            )

    def get_captain_recommendation_from_squad(
        self,
        current_squad: pd.DataFrame,
    ) -> Result[Dict[str, Any]]:
        """Get captain recommendation from a 15-player squad.

        Args:
            current_squad: DataFrame containing exactly 15 players from manager's squad

        Returns:
            Result containing captain recommendation or error
        """
        try:
            # Validate input is a proper squad
            if len(current_squad) != 15:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Squad must contain exactly 15 players, got {len(current_squad)}",
                    )
                )

            # Validate required columns
            required_columns = {"player_id", "position", "web_name", "xP"}
            missing_columns = required_columns - set(current_squad.columns)
            if missing_columns:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Squad missing required columns: {missing_columns}",
                    )
                )

            # Convert DataFrame to list for captain selection
            candidate_list = current_squad.to_dict("records")

            # Get the best captain based on xP
            best_captain = max(candidate_list, key=lambda p: p.get("xP", 0))
            captain_recommendation = {
                "player_id": best_captain["player_id"],
                "web_name": best_captain["web_name"],
                "position": best_captain["position"],
                "xP": best_captain.get("xP", 0),
                "reason": "Highest expected points in squad",
            }

            return Result(value=captain_recommendation)

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Captain selection failed: {str(e)}",
                )
            )

    def get_captain_recommendation_from_database(
        self,
        players_with_xp: pd.DataFrame,
        top_n: int = 20,
    ) -> Result[Dict[str, Any]]:
        """Get captain recommendation from the full player database.

        This method is designed for analysis purposes where you want to find
        the best captain from all available players.

        Args:
            players_with_xp: DataFrame containing all available players with xP calculations
            top_n: Number of top players to consider for captain selection

        Returns:
            Result containing captain recommendation or error
        """
        try:
            # Validate this is the full database (not a squad)
            if len(players_with_xp) < 100:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Expected full player database (>100 players), got {len(players_with_xp)} players. Use get_captain_recommendation_from_squad() for squad data.",
                    )
                )

            # Validate required columns
            required_columns = {"player_id", "position", "web_name", "xP", "status"}
            missing_columns = required_columns - set(players_with_xp.columns)
            if missing_columns:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Player database missing required columns: {missing_columns}",
                    )
                )

            # Filter out unavailable players and get top candidates
            available_players = players_with_xp[
                ~players_with_xp["status"].isin(["i", "s", "u"])
            ].copy()

            if len(available_players) == 0:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="No available players for captain selection",
                    )
                )

            # Get top players by xP
            top_candidates = available_players.nlargest(top_n, "xP")
            candidate_list = top_candidates.to_dict("records")

            # Get the best captain based on xP
            best_captain = candidate_list[0]  # Already sorted by xP descending
            captain_recommendation = {
                "player_id": best_captain["player_id"],
                "web_name": best_captain["web_name"],
                "position": best_captain["position"],
                "xP": best_captain.get("xP", 0),
                "reason": f"Highest expected points among all {len(available_players)} available players",
            }

            return Result(value=captain_recommendation)

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Captain selection failed: {str(e)}",
                )
            )

    def analyze_budget_situation(
        self, current_squad: pd.DataFrame, team_data: Dict[str, Any]
    ) -> Result[Dict[str, Any]]:
        """Analyze budget situation including sellable player values.

        Args:
            current_squad: Current squad data (exactly 15 players)
            team_data: Manager team data with sell prices and bank balance

        Returns:
            Result containing budget analysis or error
        """
        try:
            # Validate current_squad
            if len(current_squad) != 15:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Current squad must contain exactly 15 players, got {len(current_squad)}",
                    )
                )

            required_squad_columns = {"player_id", "web_name", "price"}
            missing_squad_columns = required_squad_columns - set(current_squad.columns)
            if missing_squad_columns:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Current squad missing required columns: {missing_squad_columns}",
                    )
                )

            # Validate team_data structure
            required_team_fields = {"bank_balance"}
            missing_team_fields = required_team_fields - set(team_data.keys())
            if missing_team_fields:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Team data missing required fields: {missing_team_fields}",
                    )
                )

            from fpl_team_picker.optimization.optimizer import (
                calculate_total_budget_pool,
            )

            budget_analysis = calculate_total_budget_pool(current_squad, team_data)

            # Validate result
            if budget_analysis is None:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.CALCULATION_ERROR,
                        message="Budget analysis returned no result",
                    )
                )

            return Result(value=budget_analysis)

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Budget analysis failed: {str(e)}",
                )
            )

    def get_premium_acquisition_plan(
        self,
        target_player_price: float,
        current_squad: pd.DataFrame,
        team_data: Dict[str, Any],
        players_with_xp: pd.DataFrame,
    ) -> Result[Dict[str, Any]]:
        """Create a plan for acquiring premium players through multiple transfers.

        Args:
            target_player_price: Price of the target premium player (4.0-15.0)
            current_squad: Current squad data (exactly 15 players)
            team_data: Manager team data with bank balance
            players_with_xp: Available players with expected points (full database)

        Returns:
            Result containing acquisition plan or error
        """
        try:
            # Validate target_player_price
            if not (4.0 <= target_player_price <= 15.0):
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Invalid target player price: £{target_player_price}m. Must be between £4.0m and £15.0m",
                    )
                )

            # Validate current_squad
            if len(current_squad) != 15:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Current squad must contain exactly 15 players, got {len(current_squad)}",
                    )
                )

            required_squad_columns = {"player_id", "web_name", "price", "position"}
            missing_squad_columns = required_squad_columns - set(current_squad.columns)
            if missing_squad_columns:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Current squad missing required columns: {missing_squad_columns}",
                    )
                )

            # Validate team_data structure
            required_team_fields = {"bank_balance"}
            missing_team_fields = required_team_fields - set(team_data.keys())
            if missing_team_fields:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Team data missing required fields: {missing_team_fields}",
                    )
                )

            # Validate players_with_xp (should be full database)
            if len(players_with_xp) < 100:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Expected full player database (>100 players), got {len(players_with_xp)}",
                    )
                )

            required_player_columns = {
                "player_id",
                "web_name",
                "price",
                "position",
                "xP",
            }
            missing_player_columns = required_player_columns - set(
                players_with_xp.columns
            )
            if missing_player_columns:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Players database missing required columns: {missing_player_columns}",
                    )
                )

            from fpl_team_picker.optimization.optimizer import (
                premium_acquisition_planner,
            )

            acquisition_plan = premium_acquisition_planner(
                target_price=target_player_price,
                current_squad_df=current_squad,
                team_data=team_data,
                available_players_df=players_with_xp,
            )

            # Validate result
            if acquisition_plan is None:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.CALCULATION_ERROR,
                        message="Premium acquisition planning returned no result",
                    )
                )

            return Result(value=acquisition_plan)

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Premium acquisition planning failed: {str(e)}",
                )
            )

    def validate_optimization_constraints(
        self,
        must_include_ids: Set[int],
        must_exclude_ids: Set[int],
        players_with_xp: pd.DataFrame,
    ) -> Result[bool]:
        """Validate optimization constraints.

        Args:
            must_include_ids: Player IDs that must be included
            must_exclude_ids: Player IDs that must be excluded
            players_with_xp: Available players data

        Returns:
            Result indicating constraint validity
        """
        try:
            # Check for conflicts
            conflicts = must_include_ids.intersection(must_exclude_ids)
            if conflicts:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Conflicting constraints - players both included and excluded: {conflicts}",
                    )
                )

            # Check if players exist
            available_ids = set(players_with_xp["player_id"].tolist())

            missing_include = must_include_ids - available_ids
            if missing_include:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Must-include players not found: {missing_include}",
                    )
                )

            missing_exclude = must_exclude_ids - available_ids
            if missing_exclude:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message=f"Must-exclude players not found: {missing_exclude}",
                    )
                )

            return Result(value=True)

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.VALIDATION_ERROR,
                    message=f"Constraint validation failed: {str(e)}",
                )
            )

    def _select_optimal_team_from_all_players(
        self, players: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Select optimal starting 11 from the full player database.

        Args:
            players: DataFrame of all available players with xP calculations

        Returns:
            List of 11 player dictionaries forming optimal starting team

        Raises:
            ValueError: If insufficient players available in any position
        """
        # Validate required columns
        required_columns = {"player_id", "position", "web_name", "xP"}
        missing_columns = required_columns - set(players.columns)
        if missing_columns:
            raise ValueError(
                f"Players DataFrame missing required columns: {missing_columns}"
            )

        # Sort players by xP within each position
        players_sorted = players.sort_values("xP", ascending=False)

        # Group by position
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for _, player in players_sorted.iterrows():
            position = player["position"]
            if position in by_position:
                by_position[position].append(player.to_dict())

        # Formation: 1 GKP, 4 DEF, 4 MID, 2 FWD (4-4-2 formation)
        # This ensures we can always form a valid team
        formation = {"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2}

        # Validate we have enough players in each position
        for position, needed in formation.items():
            available_count = len(by_position[position])
            if available_count < needed:
                raise ValueError(
                    f"Insufficient {position} players: need {needed}, have {available_count}"
                )

        starting_11 = []
        for position, needed in formation.items():
            available = by_position[position][:needed]  # Take top players
            starting_11.extend(available)

        # Final validation
        if len(starting_11) != 11:
            raise ValueError(
                f"Failed to select exactly 11 players, got {len(starting_11)}"
            )

        return starting_11
