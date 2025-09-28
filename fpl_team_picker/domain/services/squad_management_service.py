"""Squad management service for starting XI, captain selection, and budget analysis."""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


class SquadManagementService:
    """Service for squad management operations including starting XI and captain selection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the service with configuration.

        Args:
            config: Configuration dictionary for squad management
        """
        self.config = config or {}

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
        # Filter out unavailable players
        available_squad = squad.copy()
        if "status" in available_squad.columns:
            unavailable_mask = available_squad["status"].isin(["i", "s", "u"])
            if unavailable_mask.any():
                available_squad = available_squad[~unavailable_mask]

        # Determine XP column to use
        sort_col = self._get_xp_column(available_squad, xp_column)

        # Get optimal formation and players
        starting_11, formation_name, total_xp = self._select_optimal_formation(
            available_squad, sort_col
        )

        return {
            "starting_11": starting_11,
            "formation": formation_name,
            "total_xp": total_xp,
            "xp_column_used": sort_col,
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

        # Get starting XI player IDs
        starting_ids = {player["player_id"] for player in starting_11}

        # Filter bench players
        bench_squad = squad[~squad["player_id"].isin(starting_ids)].copy()

        # Determine XP column and sort
        sort_col = self._get_xp_column(bench_squad, xp_column)
        bench_squad = bench_squad.sort_values(sort_col, ascending=False)

        # Convert to list and limit to 4 players
        bench_players = bench_squad.head(4).to_dict("records")

        return bench_players

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

        # Use existing budget calculation from optimizer
        from fpl_team_picker.optimization.optimizer import (
            calculate_total_budget_pool,
        )

        budget_analysis = calculate_total_budget_pool(squad, team_data)

        return budget_analysis

    def _get_xp_column(self, squad: pd.DataFrame, preferred_column: str) -> str:
        """Determine which XP column to use based on availability."""
        if preferred_column in squad.columns:
            return preferred_column
        elif "xP_5gw" in squad.columns:
            return "xP_5gw"
        elif "xP" in squad.columns:
            return "xP"
        else:
            # Fallback to any numeric column that might represent points
            numeric_cols = squad.select_dtypes(include=["number"]).columns
            for col in ["total_points", "points", "expected_points"]:
                if col in numeric_cols:
                    return col
            # Last resort - use first numeric column
            return numeric_cols[0] if len(numeric_cols) > 0 else "player_id"

    def _get_player_xp_key(self, player: Dict[str, Any]) -> str:
        """Determine which XP key to use for a player dictionary."""
        for key in ["xP", "xP_5gw", "total_points", "points", "expected_points"]:
            if key in player:
                return key
        return "player_id"  # Fallback

    def _select_optimal_formation(
        self, squad: pd.DataFrame, xp_column: str
    ) -> Tuple[List[Dict], str, float]:
        """Select optimal formation and players from squad."""
        # Group by position and sort by XP
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for _, player in squad.iterrows():
            player_dict = player.to_dict()
            by_position[player["position"]].append(player_dict)

        # Sort each position by XP
        for pos in by_position:
            by_position[pos].sort(key=lambda p: p.get(xp_column, 0), reverse=True)

        # Valid FPL formations
        formations = [
            (1, 3, 5, 2),  # 3-5-2
            (1, 3, 4, 3),  # 3-4-3
            (1, 4, 5, 1),  # 4-5-1
            (1, 4, 4, 2),  # 4-4-2
            (1, 4, 3, 3),  # 4-3-3
            (1, 5, 4, 1),  # 5-4-1
            (1, 5, 3, 2),  # 5-3-2
            (1, 5, 2, 3),  # 5-2-3
        ]

        formation_names = {
            (1, 3, 5, 2): "3-5-2",
            (1, 3, 4, 3): "3-4-3",
            (1, 4, 5, 1): "4-5-1",
            (1, 4, 4, 2): "4-4-2",
            (1, 4, 3, 3): "4-3-3",
            (1, 5, 4, 1): "5-4-1",
            (1, 5, 3, 2): "5-3-2",
            (1, 5, 2, 3): "5-2-3",
        }

        best_11, best_xp, best_formation = [], 0, ""

        for gkp, def_count, mid, fwd in formations:
            # Check if we have enough players in each position
            if (
                gkp <= len(by_position["GKP"])
                and def_count <= len(by_position["DEF"])
                and mid <= len(by_position["MID"])
                and fwd <= len(by_position["FWD"])
            ):
                formation_11 = (
                    by_position["GKP"][:gkp]
                    + by_position["DEF"][:def_count]
                    + by_position["MID"][:mid]
                    + by_position["FWD"][:fwd]
                )
                formation_xp = sum(p.get(xp_column, 0) for p in formation_11)

                if formation_xp > best_xp:
                    best_xp = formation_xp
                    best_11 = formation_11
                    best_formation = formation_names.get(
                        (gkp, def_count, mid, fwd), f"{gkp}-{def_count}-{mid}-{fwd}"
                    )

        return best_11, best_formation, best_xp

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
