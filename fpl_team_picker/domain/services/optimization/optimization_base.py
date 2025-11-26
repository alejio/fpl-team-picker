"""Base utilities and data contracts for FPL optimization.

This module contains shared functionality used across all optimization modules:
- Data validation contracts (Pydantic models)
- XP calculation utilities
- Formation enumeration
- Team constraint helpers
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from fpl_team_picker.config import config


class InitialSquadOptimizationInput(BaseModel):
    """Data contract for initial squad optimization inputs."""

    budget: float = Field(ge=0, le=200, description="Budget in millions")
    formation: Tuple[int, int, int, int] = Field(
        description="Formation as (GKP, DEF, MID, FWD)"
    )
    iterations: int = Field(ge=100, le=50000, description="SA iterations")
    must_include_ids: List[int] = Field(default_factory=list)
    must_exclude_ids: List[int] = Field(default_factory=list)
    xp_column: str = Field(default="xP")

    @field_validator("formation")
    @classmethod
    def validate_formation(
        cls, v: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Validate formation sums to 15 players."""
        if len(v) != 4:
            raise ValueError("Formation must have 4 elements (GKP, DEF, MID, FWD)")
        if sum(v) != 15:
            raise ValueError(f"Formation must sum to 15 players, got {sum(v)}")
        if v[0] < 1:  # GKP
            raise ValueError("Formation must include at least 1 GKP")
        if v[1] < 3:  # DEF
            raise ValueError("Formation must include at least 3 DEF")
        if v[2] < 2:  # MID
            raise ValueError("Formation must include at least 2 MID")
        if v[3] < 1:  # FWD
            raise ValueError("Formation must include at least 1 FWD")
        return v

    @field_validator("must_include_ids", "must_exclude_ids")
    @classmethod
    def validate_no_overlap(cls, v: List[int], info) -> List[int]:
        """Ensure no duplicate IDs."""
        if len(v) != len(set(v)):
            raise ValueError("Duplicate player IDs found in constraints")
        return v


class OptimizationBaseMixin:
    """Mixin providing shared optimization utilities.

    This mixin provides core functionality used by all optimization modules:
    - XP column selection based on optimization horizon
    - Availability-adjusted XP calculations
    - Formation enumeration
    - Team constraint validation
    """

    def get_optimization_xp_column(self) -> str:
        """Get the XP column to use for optimization based on configuration.

        Returns:
            'xP' for 1-gameweek optimization, 'xP_3gw' for 3-gameweek optimization, 'xP_5gw' for 5-gameweek optimization
        """
        if config.optimization.optimization_horizon == "1gw":
            return "xP"
        elif config.optimization.optimization_horizon == "3gw":
            return "xP_3gw"
        else:  # "5gw"
            return "xP_5gw"

    def get_adjusted_xp(self, player: Dict, xp_col: str) -> float:
        """Calculate availability-adjusted xP for a player.

        Accounts for injury/availability risk by scaling xP based on:
        - chance_of_playing_this_round (preferred) or chance_of_playing_next_round
        - expected_minutes if available
        - status (injured/doubtful players get penalty)

        Args:
            player: Player dictionary with xP and availability info
            xp_col: Column name for xP values

        Returns:
            Adjusted xP accounting for availability risk
        """
        base_xp = player.get(xp_col, 0.0)
        if base_xp <= 0:
            return 0.0

        # Get availability multiplier (0.0 to 1.0)
        availability_multiplier = 1.0

        # Prefer chance_of_playing_this_round for current gameweek
        chance_this = player.get("chance_of_playing_this_round")
        chance_next = player.get("chance_of_playing_next_round")

        if chance_this is not None and not pd.isna(chance_this):
            availability_multiplier = float(chance_this) / 100.0
        elif chance_next is not None and not pd.isna(chance_next):
            availability_multiplier = float(chance_next) / 100.0
        else:
            # Fallback to expected_minutes ratio if available
            expected_mins = player.get("expected_minutes")
            if expected_mins is not None and not pd.isna(expected_mins):
                # Position-based default minutes
                position = player.get("position", "MID")
                position_defaults = {"GKP": 90, "DEF": 80, "MID": 75, "FWD": 70}
                base_minutes = position_defaults.get(position, 75)
                availability_multiplier = min(float(expected_mins) / base_minutes, 1.0)
            else:
                # Check status for injured/doubtful players
                status = player.get("status", "a")
                if status in ["i", "d"]:  # injured or doubtful
                    # Apply penalty based on config
                    if status == "i":
                        availability_multiplier = (
                            config.minutes_model.injury_full_game_multiplier
                        )
                    else:  # doubtful
                        availability_multiplier = (
                            config.minutes_model.injury_avg_minutes_multiplier
                        )

        # Clamp multiplier to valid range
        availability_multiplier = max(0.0, min(1.0, availability_multiplier))

        return base_xp * availability_multiplier

    def _count_players_per_team(self, team: List[Dict]) -> Dict[str, int]:
        """Count players per team.

        Args:
            team: List of player dictionaries with 'team' field

        Returns:
            Dictionary mapping team names to player counts
        """
        team_counts = {}
        for player in team:
            team_name = player["team"]
            team_counts[team_name] = team_counts.get(team_name, 0) + 1
        return team_counts

    def _enumerate_formations_for_players(
        self, by_position: Dict[str, List[Dict]], xp_column: str = "xP"
    ) -> Tuple[List[Dict], str, float]:
        """Core formation enumeration logic.

        Evaluates all valid FPL formations and returns the one with highest XP.

        Args:
            by_position: Players grouped by position (GKP, DEF, MID, FWD)
            xp_column: Column name for XP values

        Returns:
            Tuple of (best_11_players, formation_name, total_xp)
        """
        formations = [
            (1, 3, 5, 2),
            (1, 3, 4, 3),
            (1, 4, 5, 1),
            (1, 4, 4, 2),
            (1, 4, 3, 3),
            (1, 5, 4, 1),
            (1, 5, 3, 2),
            (1, 5, 2, 3),
        ]
        formation_names = {
            "(1, 3, 5, 2)": "3-5-2",
            "(1, 3, 4, 3)": "3-4-3",
            "(1, 4, 5, 1)": "4-5-1",
            "(1, 4, 4, 2)": "4-4-2",
            "(1, 4, 3, 3)": "4-3-3",
            "(1, 5, 4, 1)": "5-4-1",
            "(1, 5, 3, 2)": "5-3-2",
            "(1, 5, 2, 3)": "5-2-3",
        }

        best_11, best_xp, best_formation = [], 0, ""

        for gkp, def_count, mid, fwd in formations:
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
                        str((gkp, def_count, mid, fwd)),
                        f"{gkp}-{def_count}-{mid}-{fwd}",
                    )

        return best_11, best_formation, best_xp

    def _group_by_position(self, players: List[Dict]) -> Dict[str, List[Dict]]:
        """Group players by position for formation enumeration."""
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for player in players:
            by_position[player["position"]].append(player)
        return by_position

    def _get_best_starting_11_from_squad(
        self, squad: List[Dict], xp_column: str = "xP"
    ) -> List[Dict]:
        """Get best starting 11 from 15-player squad.

        Args:
            squad: List of 15 player dictionaries
            xp_column: Column name for XP values

        Returns:
            List of 11 player dicts forming best starting team
        """
        if len(squad) != 15:
            return []

        # Group by position
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for player in squad:
            by_position[player["position"]].append(player)

        # Sort by xP
        for pos in by_position:
            by_position[pos].sort(key=lambda p: p[xp_column], reverse=True)

        # Use shared formation enumeration logic (return just the players, ignore formation name)
        best_11, _, _ = self._enumerate_formations_for_players(by_position, xp_column)
        return best_11

    def validate_optimization_constraints(
        self,
        must_include_ids: Optional[set] = None,
        must_exclude_ids: Optional[set] = None,
        budget_limit: float = 100.0,
    ) -> Dict[str, Any]:
        """Validate optimization constraints for conflicts.

        Args:
            must_include_ids: Player IDs that must be included
            must_exclude_ids: Player IDs that must be excluded
            budget_limit: Budget limit in millions

        Returns:
            Validation result with conflicts detected
        """
        must_include_ids = must_include_ids or set()
        must_exclude_ids = must_exclude_ids or set()

        conflicts = must_include_ids.intersection(must_exclude_ids)

        return {
            "valid": len(conflicts) == 0,
            "conflicts": list(conflicts),
            "must_include_count": len(must_include_ids),
            "must_exclude_count": len(must_exclude_ids),
            "budget_limit": budget_limit,
        }
