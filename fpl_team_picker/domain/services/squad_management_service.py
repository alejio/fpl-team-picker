"""Squad management service for starting XI, captain selection, and budget analysis."""

from typing import Dict, Any, List, Optional
import pandas as pd
from fpl_team_picker.domain.services.optimization_service import OptimizationService


class SquadManagementService:
    """Service for squad management operations including starting XI and captain selection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the service with configuration.

        Args:
            config: Configuration dictionary for squad management
        """
        self.config = config or {}
        self.optimization_service = OptimizationService(config)

    def get_starting_eleven(
        self,
        squad: pd.DataFrame,
        xp_column: str = "xP",
    ) -> Dict[str, Any]:
        """Get optimal starting eleven from squad with formation analysis.

        Args:
            squad: DataFrame containing squad players - guaranteed clean
            xp_column: Column to use for XP sorting ('xP' for current GW, 'xP_5gw' for strategic)

        Returns:
            Starting eleven data with formation and total XP - guaranteed valid
        """
        # Delegate to optimization service
        starting_11, formation_name, total_xp = (
            self.optimization_service.find_optimal_starting_11(squad, xp_column)
        )

        return {
            "starting_11": starting_11,
            "formation": formation_name,
            "total_xp": total_xp,
            "xp_column_used": xp_column,
        }

    def get_captain_recommendation(
        self, players: pd.DataFrame, top_n: int = 5
    ) -> Dict[str, Any]:
        """Get captain recommendation with risk-adjusted analysis.

        Delegates to OptimizationService for captain recommendation logic.

        Args:
            players: DataFrame of players to consider (squad or all players)
            top_n: Number of top candidates to analyze (default 5)

        Returns:
            Captain recommendation data structure
        """
        return self.optimization_service.get_captain_recommendation(players, top_n)

    def get_bench_players(
        self,
        squad: pd.DataFrame,
        starting_11: List[Dict[str, Any]],
        xp_column: str = "xP",
    ) -> List[Dict[str, Any]]:
        """Get bench players from squad excluding starting XI.

        Args:
            squad: Full squad DataFrame
            starting_11: List of starting XI players
            xp_column: Column to use for bench ordering

        Returns:
            Result containing ordered bench players
        """
        # Delegate to optimization service
        return self.optimization_service.find_bench_players(
            squad, starting_11, xp_column
        )

    def analyze_budget_situation(
        self, squad: pd.DataFrame, team_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze budget situation including sellable player values.

        Args:
            squad: Current squad data
            team_data: Manager team data with bank balance

        Returns:
            Result containing budget analysis
        """
        # Delegate to optimization service
        bank_balance = team_data.get("bank", 0.0)
        return self.optimization_service.calculate_budget_pool(squad, bank_balance)
