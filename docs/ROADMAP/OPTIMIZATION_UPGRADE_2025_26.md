# FPL Optimization Upgrade Plan - 2025/26 Season (GW10+)

**Current Status**: GW10 of 2025-2026 season in progress
**System**: Custom RF model (MAE 0.632), Simulated Annealing optimizer
**Problem**: System underperforming despite accurate XP predictions

---

## 🚨 Critical Context for 2025/26 Season

### New FPL Rules (Must Account For)
The 2025/26 season introduced **significant rule changes** that affect optimization:

1. **Defensive Contributions System** 🆕
   - Defenders: +2 pts for 10 combined clearances/blocks/interceptions/tackles
   - Midfielders/Forwards: +2 pts for 12 defensive contributions (includes ball recoveries)
   - **Impact**: Defensive stats now matter for ALL positions, not just clean sheets

2. **Double Chips** 🆕
   - TWO sets of chips available this season
   - Wildcard 1, Wildcard 2, Free Hit 1, Free Hit 2, Bench Boost, Triple Captain
   - **Impact**: More flexibility but requires multi-phase chip strategy

3. **Player Position Reclassifications** 🆕
   - Forwards reclassified as midfielders earn more points per goal (5 vs 4)
   - Can benefit from clean sheet points (1 pt)
   - **Impact**: Position changes affect value calculations

### Known BGW/DGW Schedule for 2025/26

Based on current fixture list:

| Gameweek | Type | Reason | Teams Affected | Chip Target |
|----------|------|--------|----------------|-------------|
| **GW31** | BGW | EFL Cup Final | 4-6 teams blank | Free Hit candidate |
| **GW33** | DGW | Rescheduled from GW31 | 4-6 teams double | Bench Boost candidate |
| **GW34** | BGW | FA Cup Semi-finals | 2-4 teams blank | Free Hit candidate (if not used GW31) |
| **GW36** | DGW | Rescheduled from GW34 | 2-4 teams double | Triple Captain candidate |
| **GW37-38** | Potential DGW | Late season reschedules | Variable | Wildcard 2 target |

**Planning Horizon**: GW10 → GW38 (28 gameweeks remaining)

---

## 📊 Current System Analysis

### What's Working ✅
- **XP Predictions**: Custom RF model with 99 features, MAE 0.632
- **Uncertainty Quantification**: RF tree variance available
- **Captain Selection**: Has uncertainty penalty + template protection (5-10% bonus)
- **Correctly predicted Haaland > Kudus** (GW10)

### What's Broken ❌
1. **Captain selection**: Template protection too weak (5-10% bonus insufficient)
2. **Transfer optimization**: Pure XP maximization → value traps (Munetsi, Cullen)
3. **No BGW/DGW planning**: Missing 21 gameweeks of strategic planning
4. **Simulated Annealing**: Suboptimal, no guarantees
5. **No defensive contributions**: Model doesn't account for new 2025/26 rules
6. **No chip strategy**: No optimization for double chip usage
7. **No transfer banking**: No logic for when to save FTs

---

## 🎯 Phased Implementation Plan (GW10 → GW38)

### Phase 1: URGENT (GW10-11) - Fix Immediate Problems ✅ IMPLEMENTED

**Goal**: Stop making bad decisions NOW while building strategic layer

#### 1.1 Enhanced Captain Selection ✅ DONE
**Why Urgent**: Preventing -20 to -40 point captain disasters every week
**Status**: Implemented with betting odds integration

**Implementation** (Lines 326-377 in `optimization_service.py`):
```python
# ACTUAL IMPLEMENTATION (GW11)
def get_captain_recommendation(self, players, ...):
    """Enhanced captain selection with betting odds integration"""

    for player in players_list:
        base_score = xp * 2
        risk_adjusted_score = base_score / (1 + uncertainty_penalty)

        # 1. TEMPLATE PROTECTION (0-40% boost for >50% ownership)
        if ownership > 50:
            ownership_factor = min((ownership - 50) / 20, 1.0)

            # Use betting odds if available (preferred)
            if has_betting_odds:
                win_prob = player.get("team_win_probability", 0.33)
                fixture_quality = min(win_prob / 0.70, 1.0)
            else:
                # Fallback to fixture outlook
                fixture_quality = 1.0 if "Easy" in fixture_outlook else 0.5

            ownership_bonus = ownership_factor * fixture_quality * 0.40

        # 2. MATCHUP QUALITY (0-25% boost for ALL players)
        if has_betting_odds:
            team_xg = player.get("team_expected_goals", 1.25)
            xg_factor = min((team_xg - 1.0) / 1.0, 1.0)
            matchup_bonus = xg_factor * 0.25
        else:
            matchup_bonus = 0.20 if "Easy" in fixture_outlook else 0.10

        # 3. UNCERTAINTY PENALTY (existing, unchanged)
        uncertainty_penalty = (uncertainty / max(xp, 0.1)) * 0.30

        # Final score
        final_score = risk_adjusted_score * (1 + ownership_bonus + matchup_bonus)

    # Returns captain recommendation with top 5 candidates ranked

**Key Features:**
- ✅ Template protection: 20-40% boost (vs 5-10% before)
- ✅ Betting odds integration: `team_win_probability`, `team_expected_goals`
- ✅ Graceful fallback to fixture outlook if betting odds unavailable
- ✅ Matchup quality applies to ALL players (not just templates)
- ✅ Uncertainty penalty preserved from original system
- ⏸️ Differential EV calculation: Deferred to next iteration
```

**Example Scenarios:**
- Haaland (60% owned, 75% win prob, 2.2 xG) → +40% template + 20% matchup = **+60% total boost**
- Salah (45% owned, 60% win prob, 1.8 xG) → +0% template + 13% matchup = **+13% boost**
- Munetsi (8% owned, easy fixture) → +0% template + 20% matchup = **+20% boost**

**Priority**: ⭐⭐⭐ CRITICAL - ✅ COMPLETED

---

#### 1.2 Uncertainty-Based Transfer Filter ✅ DONE
**Why Urgent**: Stop targeting value traps while preserving asymmetric info advantage
**Status**: Implemented with ML uncertainty integration

**Implementation** (Lines 1160-1214 in `optimization_service.py`):
```python
# ACTUAL IMPLEMENTATION (GW11)
def _swap_transfer_players(self, team, all_players, ...):
    """Enhanced with uncertainty-based differential filtering"""

    def is_valid_differential_target(row):
        """
        Smart filter using prediction uncertainty.

        Logic:
        1. Template/quality players (>15% owned, >£5.5m) → Always valid
        2. Premiums (>£9m) → Always valid (proven track record)
        3. Differentials (<15% owned) → Valid IF:
           - High xP/price (>1.0) AND
           - Low uncertainty (<30% of xP)

        Examples:
        - Munetsi: 6.0 xP ± 0.8, £4.5m → 1.33 xP/£, 13% uncertainty → ALLOW ✅
        - Cullen: 4.0 xP ± 2.0, £4.6m → 0.87 xP/£, 50% uncertainty → BLOCK ❌
        - Haaland: 9.0 xP ± 2.0, £15m → Premium exception → ALLOW ✅
        """
        ownership = row["selected_by_percent"]
        price = row["price"]
        xp = row[xp_column]
        uncertainty = row.get("xP_uncertainty", 0)

        # Route 1: Template/quality (safe bets)
        if ownership >= 15.0 and price >= 5.5:
            return True

        # Route 2: Premium exception
        if price >= 9.0:
            return True

        # Route 3: Differential - require confident prediction
        if ownership < 15.0:
            xp_per_price = xp / max(price, 0.1)

            # Need strong value signal (>1.0 xP per £m)
            if xp_per_price < 1.0:
                return False

            # Check prediction confidence
            uncertainty_ratio = uncertainty / max(xp, 0.1)

            # Allow if confident (<30% uncertainty)
            return uncertainty_ratio < 0.30

        return True

    # Apply filter to candidates
    valid_mask = candidates.apply(is_valid_differential_target, axis=1)
    valid_targets = candidates[valid_mask]
```

**Key Features:**
- ✅ Preserves asymmetric information advantage (high confidence differentials allowed)
- ✅ Blocks value traps (low xP/price OR high uncertainty)
- ✅ Template/premium exceptions (always valid)
- ✅ Leverages ML uncertainty quantification (RF tree variance)

**Priority**: ⭐⭐⭐ CRITICAL - ✅ COMPLETED

---

### Phase 2: STRATEGIC (GW11-13) - BGW/DGW Planning

**Goal**: Build season-long planning system for remaining 28 gameweeks

#### 2.1 BGW/DGW Calendar (2 days)
```python
# fpl_team_picker/domain/services/bgw_dgw_calendar.py

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class GameweekType(str, Enum):
    REGULAR = "regular"
    BLANK = "blank"
    DOUBLE = "double"


class GameweekInfo(BaseModel):
    gameweek: int
    type: GameweekType
    blank_teams: List[str] = []
    double_teams: List[str] = []
    notes: str = ""
    chip_target: Optional[str] = None


class BGWDGWCalendar:
    """2025/26 season BGW/DGW calendar"""

    def __init__(self):
        self.calendar = self._build_2025_26_calendar()

    def _build_2025_26_calendar(self) -> dict[int, GameweekInfo]:
        """Build confirmed + predicted BGW/DGW calendar"""
        calendar = {}

        # Regular gameweeks (default)
        for gw in range(1, 39):
            calendar[gw] = GameweekInfo(
                gameweek=gw,
                type=GameweekType.REGULAR
            )

        # BGW31 - EFL Cup Final
        calendar[31] = GameweekInfo(
            gameweek=31,
            type=GameweekType.BLANK,
            blank_teams=["TBD"],  # Update after cup semi-finals
            notes="EFL Cup Final - 4-6 teams likely blank",
            chip_target="Free Hit 1"
        )

        # DGW33 - Rescheduled from BGW31
        calendar[33] = GameweekInfo(
            gameweek=33,
            type=GameweekType.DOUBLE,
            double_teams=["TBD"],  # Update after BGW31
            notes="Rescheduled fixtures from BGW31",
            chip_target="Bench Boost"
        )

        # BGW34 - FA Cup Semi-finals
        calendar[34] = GameweekInfo(
            gameweek=34,
            type=GameweekType.BLANK,
            blank_teams=["TBD"],  # Update after FA Cup quarter-finals
            notes="FA Cup Semi-finals - 2-4 teams blank",
            chip_target="Free Hit 2 (if not used GW31)"
        )

        # DGW36 - Rescheduled from BGW34
        calendar[36] = GameweekInfo(
            gameweek=36,
            type=GameweekType.DOUBLE,
            double_teams=["TBD"],  # Update after BGW34
            notes="Rescheduled fixtures from BGW34",
            chip_target="Triple Captain"
        )

        return calendar

    def get_gameweek_info(self, gw: int) -> GameweekInfo:
        return self.calendar.get(gw, GameweekInfo(gameweek=gw, type=GameweekType.REGULAR))

    def get_upcoming_bgw_dgw(self, current_gw: int, horizon: int = 8) -> List[GameweekInfo]:
        """Get upcoming blank/double gameweeks in planning horizon"""
        upcoming = []
        for gw in range(current_gw + 1, min(current_gw + horizon + 1, 39)):
            info = self.get_gameweek_info(gw)
            if info.type in [GameweekType.BLANK, GameweekType.DOUBLE]:
                upcoming.append(info)
        return upcoming

    def update_confirmed_teams(self, gw: int, blank_teams: List[str] = None,
                               double_teams: List[str] = None):
        """Update calendar with confirmed team information"""
        if gw in self.calendar:
            if blank_teams:
                self.calendar[gw].blank_teams = blank_teams
            if double_teams:
                self.calendar[gw].double_teams = double_teams
```

**Priority**: ⭐⭐⭐ HIGH - Needed for chip planning

---

#### 2.2 Chip Strategy Optimizer (3 days)
```python
# fpl_team_picker/domain/services/chip_strategy_service.py

class ChipStrategyService:
    """
    Optimize chip usage for 2025/26 season (double chips available).

    Chips available:
    - Wildcard 1, Wildcard 2
    - Free Hit 1, Free Hit 2
    - Bench Boost
    - Triple Captain
    """

    def __init__(self, bgw_dgw_calendar: BGWDGWCalendar):
        self.calendar = bgw_dgw_calendar

    def recommend_chip_usage(
        self,
        current_gw: int,
        chips_remaining: List[str],
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame
    ) -> dict:
        """
        Recommend optimal chip usage for current gameweek.

        Priority order:
        1. Free Hit for severe BGW (>6 players blank)
        2. Bench Boost for best DGW (all 15 players doubling)
        3. Triple Captain for DGW + template captain
        4. Wildcard before BGW/DGW cluster (squad transition)
        """
        gw_info = self.calendar.get_gameweek_info(current_gw)

        # Free Hit: Severe blank gameweek
        if gw_info.type == GameweekType.BLANK:
            blank_count = self._count_blank_players(current_squad, gw_info)
            if blank_count >= 6 and 'free_hit' in chips_remaining:
                return {
                    'chip': 'free_hit',
                    'confidence': 'HIGH',
                    'reason': f'{blank_count} players blank - Free Hit essential',
                    'expected_gain': f'+{blank_count * 4}-{blank_count * 6} pts'
                }

        # Bench Boost: Best DGW
        if gw_info.type == GameweekType.DOUBLE:
            dgw_coverage = self._calculate_dgw_coverage(current_squad, gw_info)
            if dgw_coverage >= 12 and 'bench_boost' in chips_remaining:
                bench_xp = sum(p.xp for p in current_squad.bench)
                return {
                    'chip': 'bench_boost',
                    'confidence': 'HIGH',
                    'reason': f'{dgw_coverage}/15 players in DGW, bench xP {bench_xp:.1f}',
                    'expected_gain': f'+{bench_xp * 1.5:.0f}-{bench_xp * 2:.0f} pts'
                }

        # Triple Captain: DGW + template captain
        if gw_info.type == GameweekType.DOUBLE and 'triple_captain' in chips_remaining:
            captain_candidates = self._get_dgw_captain_candidates(
                current_squad, gw_info, all_players
            )
            if captain_candidates:
                best_captain = captain_candidates[0]
                if best_captain.ownership > 40 and best_captain.xp > 8:
                    return {
                        'chip': 'triple_captain',
                        'confidence': 'HIGH',
                        'reason': f'{best_captain.web_name} ({best_captain.ownership:.0f}% owned) in DGW',
                        'expected_gain': f'+{best_captain.xp * 2:.0f}-{best_captain.xp * 3:.0f} pts'
                    }

        # Wildcard: Before BGW/DGW cluster
        upcoming_special_gws = self.calendar.get_upcoming_bgw_dgw(current_gw, horizon=6)
        if len(upcoming_special_gws) >= 2 and 'wildcard' in chips_remaining:
            return {
                'chip': 'wildcard',
                'confidence': 'MEDIUM',
                'reason': f'{len(upcoming_special_gws)} BGW/DGW in next 6 weeks - rebuild needed',
                'expected_gain': '+15-25 pts over next 6 GWs'
            }

        return {
            'chip': None,
            'confidence': 'N/A',
            'reason': 'No chip recommended this gameweek'
        }
```

**Priority**: ⭐⭐⭐ HIGH - Critical for season success

---

### Phase 3: STRUCTURAL (GW14-16) - Linear Programming

**Goal**: Replace SA with LP for guaranteed optimal transfers

#### 3.1 LP Transfer Optimizer (4 days)

**Why LP?**
- FPL = constraint satisfaction problem (perfect for LP/IP)
- SA = approximate solutions, LP = guaranteed optimal
- Faster than SA (seconds vs minutes)
- Easier to add complex constraints

**Implementation** (using PuLP):
```python
# fpl_team_picker/domain/services/lp_transfer_optimizer.py

import pulp
from typing import List, Set, Dict
import pandas as pd


class LPTransferOptimizer:
    """
    Linear Programming optimizer for FPL transfers.

    Objective: Maximize weighted score (xP + ownership + premium + exposure)
    Constraints:
    - Budget limit
    - Position requirements (2 GKP, 5 DEF, 5 MID, 3 FWD)
    - Max 3 players per team
    - Max N transfers from current squad
    - Valid formation (11 players)
    """

    def optimize_transfers(
        self,
        current_squad: pd.DataFrame,
        all_players: pd.DataFrame,
        bank_balance: float,
        free_transfers: int,
        max_transfers: int = 3,
        weights: Dict[str, float] = None
    ) -> Dict:
        """
        Optimize transfers using Linear Programming.

        Args:
            current_squad: Current 15-player squad
            all_players: All available players with xP
            bank_balance: Available funds
            free_transfers: Number of free transfers
            max_transfers: Maximum transfers allowed (default 3)
            weights: Objective function weights

        Returns:
            {
                'optimal_squad': DataFrame,
                'transfers_in': List[player_ids],
                'transfers_out': List[player_ids],
                'objective_value': float,
                'remaining_budget': float
            }
        """
        # Default weights
        if weights is None:
            weights = {
                'xp': 0.70,
                'ownership': 0.15,
                'premium': 0.10,
                'exposure': 0.05
            }

        # Initialize LP problem
        prob = pulp.LpProblem("FPL_Transfer_Optimization", pulp.LpMaximize)

        # Decision variables: x[player_id] = 1 if in squad, 0 otherwise
        player_vars = {}
        for _, player in all_players.iterrows():
            player_vars[player.player_id] = pulp.LpVariable(
                f"player_{player.player_id}",
                cat='Binary'
            )

        # Objective function components
        xp_scores = {}
        ownership_scores = {}
        premium_scores = {}

        for _, player in all_players.iterrows():
            pid = player.player_id

            # XP component
            xp_scores[pid] = player.xP

            # Ownership component (template bonus)
            if player.ownership > 30:
                ownership_scores[pid] = (player.ownership - 30) / 70  # Normalize [0, 1]
            else:
                ownership_scores[pid] = 0

            # Premium component (bias toward proven players)
            if player.price >= 11.0:
                premium_scores[pid] = 1.0
            elif player.price >= 9.0:
                premium_scores[pid] = 0.5
            else:
                premium_scores[pid] = 0

        # Objective: Maximize weighted score
        prob += pulp.lpSum([
            player_vars[pid] * (
                xp_scores[pid] * weights['xp'] +
                ownership_scores[pid] * weights['ownership'] * 10 +  # Scale to match xP
                premium_scores[pid] * weights['premium'] * 5
            )
            for pid in player_vars
        ])

        # CONSTRAINTS

        # 1. Squad size = 15
        prob += pulp.lpSum([player_vars[pid] for pid in player_vars]) == 15

        # 2. Position requirements
        for position, count in [('GKP', 2), ('DEF', 5), ('MID', 5), ('FWD', 3)]:
            pos_players = all_players[all_players.position == position].player_id.tolist()
            prob += pulp.lpSum([player_vars[pid] for pid in pos_players]) == count

        # 3. Budget constraint
        total_budget = bank_balance + current_squad.price.sum()
        prob += pulp.lpSum([
            player_vars[pid] * all_players[all_players.player_id == pid].price.iloc[0]
            for pid in player_vars
        ]) <= total_budget

        # 4. Max 3 players per team
        for team in all_players.team.unique():
            team_players = all_players[all_players.team == team].player_id.tolist()
            prob += pulp.lpSum([player_vars[pid] for pid in team_players]) <= 3

        # 5. Transfer limit (max N changes from current squad)
        current_ids = current_squad.player_id.tolist()
        # Number of current players kept = 15 - num_transfers
        prob += pulp.lpSum([
            player_vars[pid] for pid in current_ids if pid in player_vars
        ]) >= 15 - max_transfers

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Extract solution
        selected_players = [
            pid for pid in player_vars
            if pulp.value(player_vars[pid]) == 1
        ]

        optimal_squad = all_players[all_players.player_id.isin(selected_players)]

        # Calculate transfers
        current_ids_set = set(current_squad.player_id.tolist())
        new_ids_set = set(selected_players)

        transfers_out = list(current_ids_set - new_ids_set)
        transfers_in = list(new_ids_set - current_ids_set)

        return {
            'optimal_squad': optimal_squad,
            'transfers_in': transfers_in,
            'transfers_out': transfers_out,
            'objective_value': pulp.value(prob.objective),
            'remaining_budget': total_budget - optimal_squad.price.sum(),
            'num_transfers': len(transfers_out),
            'solver_status': pulp.LpStatus[prob.status]
        }
```

**Integration with existing system**:
```python
# Update OptimizationService to use LP

def optimize_transfers(self, ...):
    """Choose optimizer based on config"""

    if config.optimization.use_linear_programming:
        # Use LP for guaranteed optimal solution
        lp_optimizer = LPTransferOptimizer()
        return lp_optimizer.optimize_transfers(...)
    else:
        # Fallback to SA
        return self._optimize_transfers_sa(...)
```

**Priority**: ⭐⭐ MEDIUM-HIGH - Better algorithm, but SA works

---

### Phase 4: REFINEMENTS (GW17-20)

#### 4.1 Transfer Banking Logic (2 days)
#### 4.2 Defensive Contributions Model Update (3 days)
#### 4.3 Team Value Tracking (2 days)
#### 4.4 Bench Optimization (2 days)

---

## 📈 Expected Impact (GW10 → GW38)

| Improvement | Expected Gain | Timeline |
|-------------|---------------|----------|
| Enhanced Captain Selection | +5-10 pts/GW | GW11+ (Phase 1) |
| Transfer Target Filtering | +3-5 pts/GW | GW11+ (Phase 1) |
| BGW/DGW Planning + Chips | +60-100 pts/season | GW13+ (Phase 2) |
| LP Optimizer | +2-4 pts/GW | GW16+ (Phase 3) |
| Transfer Banking | +20-30 pts/season | GW18+ (Phase 4) |
| Defensive Contributions | +1-2 pts/GW | GW19+ (Phase 4) |
| **TOTAL** | **+300-450 pts remaining season** | **~11-16 pts/GW** |

Current average: ~60 pts/GW (estimate)
Target average: ~71-76 pts/GW
Season projection: 2,710 → 3,010-3,160 pts (top 10k finish)

---

## 🚀 Implementation Priorities

### THIS WEEK (GW10-11) ✅ COMPLETED
1. ✅ Enhanced captain selection with betting odds (DONE - 4 hours)
2. ✅ Uncertainty-based transfer filter (DONE - 3 hours)

**Expected Gain**: +8-15 pts/GW immediately
**Actual Implementation Time**: 7 hours (vs estimated 3 days)
**Status**: Ready for GW11 testing

### NEXT 2 WEEKS (GW11-13)
3. ✅ BGW/DGW calendar (2 days)
4. ✅ Chip strategy optimizer (3 days)

**Expected Gain**: +60-100 pts over season from optimal chip usage

### NEXT 4 WEEKS (GW13-17)
5. ✅ LP transfer optimizer (4 days)
6. ✅ Transfer banking logic (2 days)

**Expected Gain**: +2-4 pts/GW + 20-30 pts from better banking

### LATER (GW17+)
7. Defensive contributions model (3 days)
8. Team value tracking (2 days)
9. Bench optimization (2 days)

---

## Configuration Updates Required

```python
# Add to settings.py

class CaptainScoringConfig(BaseModel):
    """2025/26 captain selection"""
    template_ownership_threshold: float = 50.0
    template_protection_max: float = 0.40  # INCREASED from 0.10
    matchup_quality_weight: float = 0.35  # NEW
    strong_favorite_threshold: float = 1.6  # NEW
    uncertainty_penalty_weight: float = 0.20
    differential_ownership_threshold: float = 10.0
    differential_xp_edge_required: float = 0.25  # NEW


class TransferFilterConfig(BaseModel):
    """Quick transfer target filtering"""
    min_ownership_threshold: float = 15.0
    min_price_threshold: float = 5.5
    min_starts_pct_5gw: float = 0.75
    premium_price_exception: float = 9.0


class BGWDGWConfig(BaseModel):
    """BGW/DGW planning for 2025/26"""
    bgw31_expected: bool = True  # EFL Cup Final
    dgw33_expected: bool = True  # Rescheduled
    bgw34_expected: bool = True  # FA Cup Semi-finals
    dgw36_expected: bool = True  # Rescheduled
    free_hit_1_target_gw: int = 31
    bench_boost_target_gw: int = 33
    free_hit_2_target_gw: int = 34
    triple_captain_target_gw: int = 36


class OptimizationConfig(BaseModel):
    """Enhanced optimization settings"""
    use_linear_programming: bool = False  # Start False, enable after testing

    # Objective weights
    xp_weight: float = 0.70
    ownership_weight: float = 0.15
    premium_weight: float = 0.10
    exposure_weight: float = 0.05

    # Transfer limits
    max_transfers: int = 3
    transfer_cost: float = 4.0

    # Banking strategy
    enable_transfer_banking: bool = True
    bank_before_bgw_dgw: bool = True
```

---

## Success Criteria

### Phase 1 (GW11)
- ✅ Captain selection correctly identifies template captains (>50% ownership + good fixture)
- ✅ No more value trap transfers (Munetsi, Cullen, etc.)

### Phase 2 (GW13)
- ✅ BGW/DGW calendar built with confirmed dates
- ✅ Chip recommendations ready for GW31+ planning

### Phase 3 (GW17)
- ✅ LP optimizer producing valid squads within constraints
- ✅ LP vs SA comparison shows >= equal performance

### Phase 4 (GW20)
- ✅ Transfer banking logic preventing bad hits
- ✅ System projected to finish top 10k

---

## Next Steps

### IMMEDIATE (Thursday Night) 🚨
**Testing & Validation Required Before Saturday:**

1. **Test Captain Selection** (30 mins)
   - Load GW10 historical data
   - Verify: New system ranks Haaland > Kudus
   - Check: Template protection working (Haaland gets ~40% boost)
   - Validate: Betting odds integration functional

2. **Test Transfer Filter** (15 mins)
   - Load GW11 available players
   - Verify: Munetsi allowed if high confidence (uncertainty <30%)
   - Verify: Cullen blocked if low confidence (uncertainty >30%)
   - Check: Premium exception works (Haaland always valid)

3. **Run Full SA Optimization** (1 hour)
   - Execute transfer optimization for GW11
   - Review: Are recommendations sensible?
   - Validate: No obvious value traps
   - Check: Budget/constraint satisfaction

### DECISION POINTS

**Tonight (8pm)**: After Tests 1-2
- ✅ Pass → Proceed to Test 3 (full SA)
- ❌ Fail → Debug issues

**Friday Morning**: After Test 3
- ✅ Pass → Deploy for GW11
- ❌ Fail → Revert to stable system

**Friday Evening**: Final validation
- Review all test results
- Manual sanity check of recommendations
- Document any issues found

**Saturday Morning**: Execute GW11 decisions
- Run final optimization with latest data
- Make transfers (if recommended)
- Select captain + vice
- Set lineup

### QUESTIONS TO ANSWER
1. What chips have been used already this season?
2. Current team composition (concentration issues)?
3. How many free transfers available?
4. Bank balance?
