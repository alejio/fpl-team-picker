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
        current_squad: Optional[pd.DataFrame],
        team_data: Optional[Dict[str, Any]],
        teams: pd.DataFrame,
        optimization_horizon: str = "5gw",
        must_include_ids: Optional[Set[int]] = None,
        must_exclude_ids: Optional[Set[int]] = None,
    ) -> Result[Dict[str, Any]]:
        """Optimize transfers for the current squad.

        Args:
            players_with_xp: DataFrame with expected points calculations
            current_squad: Current squad data (optional)
            team_data: Manager team data (optional)
            teams: Teams reference data
            optimization_horizon: "1gw" or "5gw" for optimization horizon
            must_include_ids: Set of player IDs that must be included
            must_exclude_ids: Set of player IDs that must be excluded

        Returns:
            Result containing optimization results or error
        """
        try:
            # Validate inputs
            if players_with_xp.empty:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="Players with expected points data is empty",
                    )
                )

            # Check if optimization is available
            if current_squad is None or current_squad.empty:
                return Result(
                    error=DomainError(
                        error_type=ErrorType.VALIDATION_ERROR,
                        message="Current squad data is required for optimization",
                    )
                )

            # Load optimization configuration
            from fpl_team_picker.config import config as opt_config
            from fpl_team_picker.optimization.optimizer import (
                optimize_team_with_transfers,
            )

            # Set defaults for constraints
            must_include_ids = must_include_ids or set()
            must_exclude_ids = must_exclude_ids or set()

            # Temporarily override optimization horizon in config
            original_horizon = opt_config.optimization.horizon
            opt_config.optimization.horizon = optimization_horizon

            try:
                # Perform optimization
                optimization_result = optimize_team_with_transfers(
                    expected_points_df=players_with_xp,
                    current_squad_df=current_squad,
                    teams_df=teams,
                    team_data=team_data,
                    must_include_player_ids=must_include_ids,
                    must_exclude_player_ids=must_exclude_ids,
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

            finally:
                # Restore original horizon
                opt_config.optimization.horizon = original_horizon

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Transfer optimization failed: {str(e)}",
                )
            )

    def get_starting_eleven(
        self,
        players_with_xp: pd.DataFrame,
        current_squad: Optional[pd.DataFrame] = None,
    ) -> Result[List[Dict[str, Any]]]:
        """Get the optimal starting eleven from available players.

        Args:
            players_with_xp: DataFrame with expected points calculations
            current_squad: Optional current squad to prioritize

        Returns:
            Result containing starting eleven or error
        """
        try:
            from fpl_team_picker.optimization.optimizer import get_best_starting_11

            # Use current squad if available, otherwise use top players
            source_data = (
                current_squad
                if current_squad is not None and not current_squad.empty
                else players_with_xp
            )

            starting_11 = get_best_starting_11(source_data)
            return Result(value=starting_11)

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Starting eleven selection failed: {str(e)}",
                )
            )

    def get_captain_recommendation(
        self,
        players_with_xp: pd.DataFrame,
        current_squad: Optional[pd.DataFrame] = None,
    ) -> Result[Dict[str, Any]]:
        """Get captain recommendation based on expected points and risk analysis.

        Args:
            players_with_xp: DataFrame with expected points calculations
            current_squad: Optional current squad to limit choices

        Returns:
            Result containing captain recommendation or error
        """
        try:
            from fpl_team_picker.optimization.optimizer import select_captain

            # Use current squad if available for captain selection
            captain_data = (
                current_squad
                if current_squad is not None and not current_squad.empty
                else players_with_xp
            )

            captain_recommendation = select_captain(captain_data)
            return Result(value=captain_recommendation)

        except Exception as e:
            return Result(
                error=DomainError(
                    error_type=ErrorType.CALCULATION_ERROR,
                    message=f"Captain selection failed: {str(e)}",
                )
            )

    def analyze_budget_situation(
        self, current_squad: pd.DataFrame, team_data: Optional[Dict[str, Any]]
    ) -> Result[Dict[str, Any]]:
        """Analyze budget situation including sellable player values.

        Args:
            current_squad: Current squad data
            team_data: Manager team data with sell prices

        Returns:
            Result containing budget analysis or error
        """
        try:
            from fpl_team_picker.optimization.optimizer import (
                calculate_total_budget_pool,
            )

            budget_analysis = calculate_total_budget_pool(current_squad, team_data)
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
        team_data: Optional[Dict[str, Any]],
        players_with_xp: pd.DataFrame,
    ) -> Result[Dict[str, Any]]:
        """Create a plan for acquiring premium players through multiple transfers.

        Args:
            target_player_price: Price of the target premium player
            current_squad: Current squad data
            team_data: Manager team data
            players_with_xp: Available players with expected points

        Returns:
            Result containing acquisition plan or error
        """
        try:
            from fpl_team_picker.optimization.optimizer import (
                premium_acquisition_planner,
            )

            acquisition_plan = premium_acquisition_planner(
                target_price=target_player_price,
                current_squad_df=current_squad,
                team_data=team_data,
                available_players_df=players_with_xp,
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
