"""
FPL Chip Assessment Module

Provides heuristic-based analysis for optimal chip usage timing:
- Wildcard: Analyze transfer opportunities and fixture runs
- Free Hit: Detect double gameweeks and squad improvement potential
- Bench Boost: Calculate bench strength for target gameweek
- Triple Captain: Identify optimal captaincy candidates

Uses simple heuristics rather than complex optimization for fast analysis.
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChipRecommendation:
    """Structured chip recommendation with reasoning"""

    chip_name: str
    status: str  # "🟢 RECOMMENDED", "🟡 CONSIDER", "🔴 HOLD"
    reasoning: str
    key_metrics: Dict[str, float]
    optimal_gameweek: Optional[int] = None


class ChipAssessmentEngine:
    """Heuristic-based chip assessment for FPL decision making"""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize with configuration thresholds"""
        self.config = config or {
            "wildcard_min_transfers": 4,
            "bench_boost_min_points": 8.0,
            "triple_captain_min_xp": 8.0,
            "free_hit_min_improvement": 15.0,
            "double_gameweek_bonus": 10.0,
        }

    def assess_all_chips(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
        team_data: Dict,
        available_chips: List[str],
    ) -> Dict[str, ChipRecommendation]:
        """Assess all available chips and return recommendations"""
        recommendations = {}

        if "wildcard" in available_chips:
            recommendations["wildcard"] = self.assess_wildcard(
                current_squad, all_players, fixtures, target_gameweek, team_data
            )

        if "free_hit" in available_chips:
            recommendations["free_hit"] = self.assess_free_hit(
                current_squad, all_players, fixtures, target_gameweek, team_data
            )

        if "bench_boost" in available_chips:
            recommendations["bench_boost"] = self.assess_bench_boost(
                current_squad, fixtures, target_gameweek
            )

        if "triple_captain" in available_chips:
            recommendations["triple_captain"] = self.assess_triple_captain(
                current_squad, all_players, fixtures, target_gameweek
            )

        return recommendations

    def assess_wildcard(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
        team_data: Dict,
    ) -> ChipRecommendation:
        """Assess wildcard chip using transfer opportunity analysis"""

        # Count beneficial transfer opportunities
        transfer_opportunities = self._count_transfer_opportunities(
            current_squad, all_players
        )

        # Analyze fixture run quality over next 5 gameweeks
        fixture_outlook = self._analyze_fixture_run(
            current_squad, fixtures, target_gameweek, horizon=5
        )

        # Calculate squad freshness (how many players need replacing)
        squad_freshness = self._calculate_squad_freshness(current_squad)

        # Budget utilization analysis
        budget_available = team_data.get("bank", 0)
        budget_efficiency = self._analyze_budget_efficiency(
            current_squad, budget_available
        )

        key_metrics = {
            "transfer_opportunities": transfer_opportunities,
            "fixture_outlook": fixture_outlook,
            "squad_freshness": squad_freshness,
            "budget_available": budget_available,
            "budget_efficiency": budget_efficiency,
        }

        # Recommendation logic
        if (
            transfer_opportunities >= self.config["wildcard_min_transfers"]
            and fixture_outlook < 0.9
        ):
            status = "🟢 RECOMMENDED"
            reasoning = f"{transfer_opportunities} strong transfers available with poor fixture run ahead"
        elif transfer_opportunities >= self.config["wildcard_min_transfers"]:
            status = "🟡 CONSIDER"
            reasoning = f"{transfer_opportunities} transfers available, but fixtures look decent"
        elif fixture_outlook < 0.8:
            status = "🟡 CONSIDER"
            reasoning = "Very poor fixture run ahead, consider wildcard for fresh start"
        else:
            status = "🔴 HOLD"
            reasoning = "Few beneficial transfers available, squad looks strong"

        return ChipRecommendation(
            chip_name="Wildcard",
            status=status,
            reasoning=reasoning,
            key_metrics=key_metrics,
        )

    def assess_free_hit(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
        team_data: Dict,
    ) -> ChipRecommendation:
        """Assess free hit chip focusing on double gameweeks and squad gaps"""

        # Detect double gameweeks
        double_gw_teams, double_gw_strength = self._detect_double_gameweek(
            fixtures, target_gameweek
        )

        # Count missing/unavailable key players
        missing_players = self._count_unavailable_players(current_squad)

        # Estimate temporary squad improvement potential
        temp_squad_improvement = self._estimate_temp_squad_improvement(
            current_squad, all_players, target_gameweek, team_data.get("bank", 0)
        )

        # Analyze current squad's gameweek outlook
        current_outlook = self._analyze_single_gw_outlook(
            current_squad, fixtures, target_gameweek
        )

        key_metrics = {
            "double_gw_teams": len(double_gw_teams),
            "double_gw_strength": double_gw_strength,
            "missing_players": missing_players,
            "temp_squad_improvement": temp_squad_improvement,
            "current_outlook": current_outlook,
        }

        # Recommendation logic
        if double_gw_teams and double_gw_strength > 1.2:
            status = "🟢 RECOMMENDED"
            reasoning = f"Strong double gameweek opportunity - {len(double_gw_teams)} quality teams playing twice"
        elif temp_squad_improvement > self.config["free_hit_min_improvement"]:
            status = "🟢 RECOMMENDED"
            reasoning = (
                f"Major squad improvement possible (+{temp_squad_improvement:.1f} xP)"
            )
        elif missing_players >= 3 or current_outlook < 0.7:
            status = "🟡 CONSIDER"
            reasoning = f"Squad issues present ({missing_players} missing players, poor outlook)"
        else:
            status = "🔴 HOLD"
            reasoning = "Current squad looks strong for this gameweek"

        return ChipRecommendation(
            chip_name="Free Hit",
            status=status,
            reasoning=reasoning,
            key_metrics=key_metrics,
        )

    def assess_bench_boost(
        self, current_squad: pd.DataFrame, fixtures: pd.DataFrame, target_gameweek: int
    ) -> ChipRecommendation:
        """Assess bench boost chip based on bench strength"""

        # Check if xP column exists
        if current_squad.empty or "xP" not in current_squad.columns:
            return ChipRecommendation(
                chip_name="Bench Boost",
                status="🔴 HOLD",
                reasoning="Expected points data not available",
                key_metrics={"bench_xp": 0, "data_available": False},
            )

        # Identify bench players (lowest 4 xP players typically)
        if len(current_squad) >= 15:
            bench_players = current_squad.nsmallest(4, "xP")
        else:
            bench_players = pd.DataFrame()

        # Calculate bench expected points
        bench_xp = bench_players["xP"].sum() if not bench_players.empty else 0

        # Analyze bench player fixtures
        bench_fixture_quality = self._analyze_bench_fixtures(
            bench_players, fixtures, target_gameweek
        )

        # Check for bench player rotation risks
        bench_rotation_risk = self._assess_bench_rotation_risk(bench_players)

        key_metrics = {
            "bench_xp": bench_xp,
            "bench_fixture_quality": bench_fixture_quality,
            "bench_rotation_risk": bench_rotation_risk,
            "bench_size": len(bench_players),
        }

        # Recommendation logic
        if (
            bench_xp >= self.config["bench_boost_min_points"]
            and bench_fixture_quality > 1.0
        ):
            status = "🟢 RECOMMENDED"
            reasoning = f"Strong bench ({bench_xp:.1f} xP) with good fixtures"
        elif bench_xp >= self.config["bench_boost_min_points"]:
            status = "🟡 CONSIDER"
            reasoning = f"Decent bench strength ({bench_xp:.1f} xP) but mixed fixtures"
        else:
            status = "🔴 HOLD"
            reasoning = f"Weak bench ({bench_xp:.1f} xP) - not worth boosting"

        return ChipRecommendation(
            chip_name="Bench Boost",
            status=status,
            reasoning=reasoning,
            key_metrics=key_metrics,
        )

    def assess_triple_captain(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        fixtures: pd.DataFrame,
        target_gameweek: int,
    ) -> ChipRecommendation:
        """Assess triple captain chip based on high xP candidates"""

        # Check if xP column exists
        if current_squad.empty or "xP" not in current_squad.columns:
            return ChipRecommendation(
                chip_name="Triple Captain",
                status="🔴 HOLD",
                reasoning="Expected points data not available",
                key_metrics={"top_candidate_xp": 0, "data_available": False},
            )

        # Find best captain candidates in current squad
        captain_candidates = current_squad.nlargest(5, "xP")

        # Get the top candidate
        if not captain_candidates.empty:
            top_candidate = captain_candidates.iloc[0]
            top_xp = top_candidate["xP"]
            top_name = top_candidate["web_name"]
        else:
            top_xp = 0
            top_name = "None"

        # Check for fixture quality of top candidates
        captain_fixture_quality = self._analyze_captain_fixtures(
            captain_candidates, fixtures, target_gameweek
        )

        # Assess rotation risk for top candidates
        rotation_risk = self._assess_captain_rotation_risk(captain_candidates)

        # Compare with premium alternatives not in squad
        premium_alternatives = self._find_premium_captain_alternatives(
            current_squad, all_players
        )

        key_metrics = {
            "top_candidate_xp": top_xp,
            "top_candidate_name": top_name,
            "captain_fixture_quality": captain_fixture_quality,
            "rotation_risk": rotation_risk,
            "premium_alternatives": len(premium_alternatives),
        }

        # Recommendation logic
        if (
            top_xp >= self.config["triple_captain_min_xp"]
            and captain_fixture_quality > 1.1
            and rotation_risk < 0.3
        ):
            status = "🟢 RECOMMENDED"
            reasoning = (
                f"{top_name} excellent candidate ({top_xp:.1f} xP, great fixture)"
            )
        elif top_xp >= self.config["triple_captain_min_xp"]:
            status = "🟡 CONSIDER"
            reasoning = f"{top_name} good candidate ({top_xp:.1f} xP) but check fixtures/rotation"
        else:
            status = "🔴 HOLD"
            reasoning = f"No premium candidates in squad (best: {top_xp:.1f} xP)"

        return ChipRecommendation(
            chip_name="Triple Captain",
            status=status,
            reasoning=reasoning,
            key_metrics=key_metrics,
        )

    # Helper methods for analysis

    def _count_transfer_opportunities(
        self, current_squad: pd.DataFrame, all_players: pd.DataFrame
    ) -> int:
        """Count how many beneficial transfers are available"""
        if current_squad.empty or all_players.empty:
            return 0

        # Check if required columns exist
        if "xP" not in current_squad.columns or "xP" not in all_players.columns:
            return 0

        current_player_ids = set(current_squad["player_id"].tolist())
        available_players = all_players[
            ~all_players["player_id"].isin(current_player_ids)
        ]

        transfer_count = 0

        for position in ["GKP", "DEF", "MID", "FWD"]:
            current_pos = current_squad[current_squad["position"] == position]
            available_pos = available_players[available_players["position"] == position]

            if current_pos.empty or available_pos.empty:
                continue

            # Find upgrades (better xP for similar or higher price)
            for _, current_player in current_pos.iterrows():
                current_xp = current_player.get("xP", 0)
                current_price = current_player.get("price", 0)

                # Look for upgrades within reasonable price range
                upgrades = available_pos[
                    (available_pos["xP"] > current_xp + 2)  # Significant improvement
                    & (
                        available_pos["price"] <= current_price + 2
                    )  # Affordable upgrade
                ]

                if not upgrades.empty:
                    transfer_count += 1

        return min(transfer_count, 11)  # Cap at reasonable number

    def _analyze_fixture_run(
        self,
        current_squad: pd.DataFrame,
        fixtures: pd.DataFrame,
        start_gw: int,
        horizon: int = 5,
    ) -> float:
        """Analyze fixture difficulty over next N gameweeks"""
        if current_squad.empty:
            return 1.0

        total_difficulty = 0
        team_count = 0

        for _, player in current_squad.iterrows():
            team_name = player.get("name", "")
            if not team_name:
                continue

            team_difficulties = []
            for gw in range(start_gw, start_gw + horizon):
                # This is simplified - in practice, you'd match team fixtures
                # For now, use a default moderate difficulty
                team_difficulties.append(1.0)

            if team_difficulties:
                avg_difficulty = sum(team_difficulties) / len(team_difficulties)
                total_difficulty += avg_difficulty
                team_count += 1

        return total_difficulty / team_count if team_count > 0 else 1.0

    def _calculate_squad_freshness(self, current_squad: pd.DataFrame) -> float:
        """Calculate what percentage of squad might need refreshing"""
        if current_squad.empty:
            return 0.0

        # Simple heuristic: players with low xP per price ratio
        if "price" in current_squad.columns and "xP" in current_squad.columns:
            current_squad = current_squad.copy()
            current_squad["xp_per_price"] = current_squad["xP"] / current_squad[
                "price"
            ].clip(lower=4.0)

            # Count players in bottom quartile of efficiency
            threshold = current_squad["xp_per_price"].quantile(0.25)
            stale_players = (current_squad["xp_per_price"] <= threshold).sum()

            return stale_players / len(current_squad)

        return 0.0

    def _analyze_budget_efficiency(
        self, current_squad: pd.DataFrame, budget_available: float
    ) -> float:
        """Analyze how efficiently current budget could be used"""
        if budget_available < 1.0:
            return 1.0  # No budget to improve

        # Simple heuristic: more budget = more improvement potential
        return min(budget_available / 10.0, 1.0)

    def _detect_double_gameweek(
        self, fixtures: pd.DataFrame, target_gw: int
    ) -> Tuple[List[str], float]:
        """Detect if any teams have double gameweeks"""
        if fixtures.empty:
            return [], 0.0

        # Count fixtures per team for target gameweek
        gw_fixtures = fixtures[fixtures["event"] == target_gw]

        if gw_fixtures.empty:
            return [], 0.0

        # Count home and away appearances
        team_fixture_counts = {}

        for _, fixture in gw_fixtures.iterrows():
            home_team = fixture.get("home_team_id", "")
            away_team = fixture.get("away_team_id", "")

            team_fixture_counts[home_team] = team_fixture_counts.get(home_team, 0) + 1
            team_fixture_counts[away_team] = team_fixture_counts.get(away_team, 0) + 1

        # Find teams with 2+ fixtures
        double_gw_teams = [
            team for team, count in team_fixture_counts.items() if count >= 2
        ]

        # Estimate strength of double gameweek (simplified)
        avg_strength = 1.0 + (len(double_gw_teams) * 0.1)

        return double_gw_teams, avg_strength

    def _count_unavailable_players(self, current_squad: pd.DataFrame) -> int:
        """Count players who are injured, suspended, or doubtful"""
        if current_squad.empty:
            return 0

        unavailable_count = 0

        for _, player in current_squad.iterrows():
            availability = player.get("availability_status", "a")
            if availability in [
                "i",
                "s",
                "u",
                "d",
            ]:  # injured, suspended, unavailable, doubtful
                unavailable_count += 1

        return unavailable_count

    def _estimate_temp_squad_improvement(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        target_gw: int,
        budget: float,
    ) -> float:
        """Estimate xP improvement possible with temporary squad"""
        if current_squad.empty or all_players.empty:
            return 0.0

        # Check if required columns exist
        if "xP" not in current_squad.columns or "xP" not in all_players.columns:
            return 0.0

        current_best_11_xp = current_squad.nlargest(11, "xP")["xP"].sum()

        # Simple estimate: could we build a much better team for one week?
        # This is a simplified calculation
        total_budget = budget + current_squad["price"].sum()

        # Estimate what top 11 players we could afford
        affordable_players = all_players[all_players["price"] <= total_budget / 11]

        if not affordable_players.empty:
            temp_best_11_xp = affordable_players.nlargest(11, "xP")["xP"].sum()
            improvement = temp_best_11_xp - current_best_11_xp
            return max(0, improvement)

        return 0.0

    def _analyze_single_gw_outlook(
        self, current_squad: pd.DataFrame, fixtures: pd.DataFrame, target_gw: int
    ) -> float:
        """Analyze how good current squad looks for single gameweek"""
        if current_squad.empty:
            return 0.5

        # Check if xP column exists
        if "xP" not in current_squad.columns:
            return 0.5

        # Simplified: average xP of starting 11
        starting_11_xp = current_squad.nlargest(11, "xP")["xP"].mean()

        # Normalize to 0-1 scale (assuming 4-8 xP range per player)
        return min(max((starting_11_xp - 4) / 4, 0), 1)

    def _analyze_bench_fixtures(
        self, bench_players: pd.DataFrame, fixtures: pd.DataFrame, target_gw: int
    ) -> float:
        """Analyze fixture quality for bench players"""
        if bench_players.empty:
            return 1.0

        # Simplified: assume average fixture quality
        return 1.0

    def _assess_bench_rotation_risk(self, bench_players: pd.DataFrame) -> float:
        """Assess rotation risk for bench players"""
        if bench_players.empty:
            return 0.0

        # Simplified: lower priced players have higher rotation risk
        if "price" in bench_players.columns:
            avg_price = bench_players["price"].mean()
            # Lower price = higher rotation risk
            return max(0, min(1, (7 - avg_price) / 3))

        return 0.3  # Default moderate risk

    def _analyze_captain_fixtures(
        self, captain_candidates: pd.DataFrame, fixtures: pd.DataFrame, target_gw: int
    ) -> float:
        """Analyze fixture quality for captain candidates"""
        if captain_candidates.empty:
            return 1.0

        # Simplified: assume good fixtures for top players
        return 1.1

    def _assess_captain_rotation_risk(self, captain_candidates: pd.DataFrame) -> float:
        """Assess rotation risk for captain candidates"""
        if captain_candidates.empty:
            return 1.0

        # Simplified: premium players have low rotation risk
        if "price" in captain_candidates.columns:
            avg_price = captain_candidates["price"].mean()
            # Higher price = lower rotation risk
            return max(0, min(1, (10 - avg_price) / 5))

        return 0.2  # Default low risk for top candidates

    def _find_premium_captain_alternatives(
        self, current_squad: pd.DataFrame, all_players: pd.DataFrame
    ) -> pd.DataFrame:
        """Find premium captain alternatives not in current squad"""
        if current_squad.empty or all_players.empty:
            return pd.DataFrame()

        # Check if required columns exist
        if "xP" not in all_players.columns:
            return pd.DataFrame()

        current_player_ids = set(current_squad["player_id"].tolist())
        premium_players = all_players[
            (~all_players["player_id"].isin(current_player_ids))
            & (all_players["price"] >= 9.0)
            & (all_players["xP"] >= 8.0)
        ]

        return premium_players.nlargest(5, "xP")
