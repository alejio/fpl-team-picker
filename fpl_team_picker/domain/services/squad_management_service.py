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
        self,
        players: List[Dict[str, Any]],
        include_risk_analysis: bool = True,
    ) -> Dict[str, Any]:
        """Get captain recommendation from a list of players.

        Args:
            players: List of player dictionaries - guaranteed clean
            include_risk_analysis: Whether to include detailed risk analysis

        Returns:
            Captain recommendation with analysis - guaranteed valid
        """
        # Sort by expected points (prefer xP, fallback to other XP columns)
        xp_key = self._get_player_xp_key(players[0])
        captain_candidates = sorted(
            players, key=lambda p: p.get(xp_key, 0), reverse=True
        )

        best_captain = captain_candidates[0]
        vice_captain = (
            captain_candidates[1] if len(captain_candidates) > 1 else best_captain
        )

        captain_xp = best_captain.get(xp_key, 0)
        vice_xp = vice_captain.get(xp_key, 0)

        recommendation = {
            "captain": {
                "player_id": best_captain["player_id"],
                "web_name": best_captain["web_name"],
                "position": best_captain["position"],
                "xp": captain_xp,
                "captain_points": captain_xp * 2,
            },
            "vice_captain": {
                "player_id": vice_captain["player_id"],
                "web_name": vice_captain["web_name"],
                "position": vice_captain["position"],
                "xp": vice_xp,
                "captain_points": vice_xp * 2,
            },
            "advantage": (captain_xp - vice_xp) * 2,
            "xp_column_used": xp_key,
        }

        # Add risk analysis if requested
        if include_risk_analysis:
            recommendation["risk_analysis"] = self._analyze_captain_risk(best_captain)

        return recommendation

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

    def _get_player_xp_key(self, player: Dict[str, Any]) -> str:
        """Determine which XP key to use for a player dictionary."""
        for key in ["xP", "xP_5gw", "total_points", "points", "expected_points"]:
            if key in player:
                return key
        return "player_id"  # Fallback

    def _analyze_captain_risk(self, captain: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk factors for captain selection."""
        risk_factors = []
        risk_level = "Low"

        # Availability risk
        status = captain.get("status", "a")
        if status in ["i", "d"]:
            risk_factors.append("injury_risk")
            risk_level = "High"
        elif status == "s":
            risk_factors.append("suspended")
            risk_level = "High"

        # Minutes risk
        expected_mins = captain.get("expected_minutes", 90)
        if expected_mins < 60:
            risk_factors.append("rotation_risk")
            if risk_level == "Low":
                risk_level = "Medium"

        # Fixture difficulty
        fixture_outlook = captain.get("fixture_outlook", "")
        if "Hard" in fixture_outlook or "🔴" in fixture_outlook:
            risk_factors.append("difficult_fixture")
            if risk_level == "Low":
                risk_level = "Medium"

        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "expected_minutes": expected_mins,
            "availability_status": status,
            "fixture_outlook": fixture_outlook,
        }
