"""Transfer planning evaluation datasets with realistic FPL scenarios.

This module defines test cases covering common transfer planning scenarios:
- Premium upgrades with hit analysis
- Differential punts
- Template safety plays
- DGW exploitation
- Fixture swing opportunities
- Budget-constrained decisions
- Injury emergencies
"""

from typing import Any, Dict

from pydantic_evals import Case, Dataset


def create_premium_upgrade_case() -> Case[Dict[str, Any], Dict[str, Any]]:
    """Test case: Should recommend premium upgrade despite -4 hit if ROI justifies."""
    return Case(
        name="premium_upgrade_with_hit",
        inputs={
            "scenario": "GW18, 1 FT, £2.0m ITB",
            "context": "Watkins has tough fixtures (avg difficulty 4.0), Haaland has DGW in GW20",
            "expected_behavior": "Recommend Watkins → Haaland despite -4 hit if 3GW ROI > threshold",
        },
        expected_output={
            "should_include_hold": True,
            "should_have_hit_scenario": True,
            "top_pick_reasoning_mentions": ["DGW", "ROI", "fixture"],
            "min_scenarios": 3,
            "max_scenarios": 5,
        },
        metadata={
            "difficulty": "medium",
            "category": "premium_upgrade",
            "key_decision": "hit_analysis",
        },
    )


def create_differential_punt_case() -> Case[Dict[str, Any], Dict[str, Any]]:
    """Test case: Should identify differential opportunities with good fixtures."""
    return Case(
        name="differential_punt_opportunity",
        inputs={
            "scenario": "GW20, 1 FT, £1.5m ITB",
            "context": "Palmer (18% owned) has 3 easy fixtures, Saka (45% owned) has tough run",
            "expected_behavior": "Recommend Saka → Palmer as differential with medium confidence",
        },
        expected_output={
            "should_include_hold": True,
            "should_have_differential_scenario": True,
            "differential_ownership_threshold": 25.0,  # Below 25% is differential
            "top_pick_reasoning_mentions": ["differential", "fixture", "ownership"],
            "min_scenarios": 3,
        },
        metadata={
            "difficulty": "medium",
            "category": "differential",
            "key_decision": "risk_vs_reward",
        },
    )


def create_template_safety_case() -> Case[Dict[str, Any], Dict[str, Any]]:
    """Test case: Conservative strategy should prefer template players."""
    return Case(
        name="template_safety_conservative",
        inputs={
            "scenario": "GW22, 1 FT, £0.5m ITB, CONSERVATIVE strategy",
            "context": "Salah (55% owned) available, alternative differentials exist",
            "expected_behavior": "Prioritize template players (>40% owned) for rank protection",
        },
        expected_output={
            "should_include_hold": True,
            "should_prioritize_template": True,
            "template_ownership_threshold": 40.0,  # Above 40% is template
            "top_pick_confidence": "high",  # Conservative should be confident in template
            "top_pick_reasoning_mentions": ["template", "safety", "ownership"],
        },
        metadata={
            "difficulty": "easy",
            "category": "template_safety",
            "strategy_mode": "conservative",
        },
    )


def create_dgw_opportunity_case() -> Case[Dict[str, Any], Dict[str, Any]]:
    """Test case: Should identify and leverage DGW opportunities."""
    return Case(
        name="dgw_exploitation",
        inputs={
            "scenario": "GW19, 2 FTs, £3.0m ITB",
            "context": "Liverpool and Man City have DGW in GW20, time to prepare",
            "expected_behavior": "Recommend transfers targeting DGW players, flag leverages_dgw",
        },
        expected_output={
            "should_include_hold": True,
            "should_have_dgw_scenario": True,
            "dgw_scenario_should_be_flagged": True,  # leverages_dgw = True
            "top_pick_reasoning_mentions": ["DGW", "gameweek 20", "double"],
            "context_analysis_should_mention_dgw": True,
        },
        metadata={
            "difficulty": "medium",
            "category": "dgw_exploitation",
            "key_decision": "timing",
        },
    )


def create_fixture_swing_case() -> Case[Dict[str, Any], Dict[str, Any]]:
    """Test case: Should identify fixture difficulty swings."""
    return Case(
        name="fixture_swing_exploitation",
        inputs={
            "scenario": "GW16, 1 FT, £1.0m ITB",
            "context": "Arsenal go from hard fixtures (4.5 avg) to easy (2.0 avg) in GW17-20",
            "expected_behavior": "Recommend bringing in Arsenal assets, flag fixture swing",
        },
        expected_output={
            "should_include_hold": True,
            "should_have_fixture_swing_scenario": True,
            "fixture_swing_should_be_flagged": True,  # leverages_fixture_swing = True
            "top_pick_reasoning_mentions": ["fixture", "run", "easy"],
            "context_analysis_should_mention_fixtures": True,
        },
        metadata={
            "difficulty": "medium",
            "category": "fixture_swing",
            "key_decision": "timing",
        },
    )


def create_budget_constraint_case() -> Case[Dict[str, Any], Dict[str, Any]]:
    """Test case: Should respect budget constraints in recommendations."""
    return Case(
        name="budget_constrained_decision",
        inputs={
            "scenario": "GW25, 1 FT, £0.1m ITB (tight budget)",
            "context": "Want premium forward but can only afford £9.0m max",
            "expected_behavior": "Only recommend affordable transfers, respect budget limit",
        },
        expected_output={
            "should_include_hold": True,
            "all_scenarios_should_be_affordable": True,
            "top_pick_reasoning_mentions": ["budget", "afford"],
            "should_not_recommend_unaffordable": True,
        },
        metadata={
            "difficulty": "easy",
            "category": "budget_constraint",
            "key_decision": "affordability",
        },
    )


def create_injury_emergency_case() -> Case[Dict[str, Any], Dict[str, Any]]:
    """Test case: Should prioritize replacing injured players."""
    return Case(
        name="injury_emergency_response",
        inputs={
            "scenario": "GW24, 1 FT, £2.0m ITB",
            "context": "Key midfielder injured (0% chance of playing), must replace urgently",
            "expected_behavior": "Recommend immediate replacement, don't bank transfer",
        },
        expected_output={
            "should_include_hold": False,  # Don't hold when injury emergency
            "should_recommend_immediate_action": True,
            "top_pick_confidence": "high",  # Urgency = high confidence
            "top_pick_reasoning_mentions": ["injury", "replace", "urgent"],
        },
        metadata={
            "difficulty": "easy",
            "category": "injury_emergency",
            "key_decision": "urgency",
        },
    )


def create_aggressive_ceiling_case() -> Case[Dict[str, Any], Dict[str, Any]]:
    """Test case: Aggressive strategy should target high-ceiling players."""
    return Case(
        name="aggressive_ceiling_seeking",
        inputs={
            "scenario": "GW15, 2 FTs, £5.0m ITB, AGGRESSIVE strategy",
            "context": "Multiple premium options with high ceilings available",
            "expected_behavior": "Recommend high-ceiling picks, willing to take -4 for upside",
        },
        expected_output={
            "should_include_hold": True,
            "should_have_hit_scenario": True,  # Aggressive willing to take hits
            "top_pick_reasoning_mentions": ["ceiling", "upside", "haul"],
            "should_target_differentials": True,  # Aggressive seeks differentials
        },
        metadata={
            "difficulty": "medium",
            "category": "aggressive_strategy",
            "strategy_mode": "aggressive",
        },
    )


def create_bank_transfer_case() -> Case[Dict[str, Any], Dict[str, Any]]:
    """Test case: Should recommend banking transfer when squad is strong."""
    return Case(
        name="bank_transfer_decision",
        inputs={
            "scenario": "GW12, 1 FT, £0.5m ITB",
            "context": "Squad has strong fixtures across the board, no obvious upgrades",
            "expected_behavior": "Recommend hold option as top choice, bank for 2 FTs next week",
        },
        expected_output={
            "should_include_hold": True,
            "top_recommendation_should_be_hold": True,  # Hold is top pick
            "hold_reasoning_mentions": ["bank", "2 FTs", "strong"],
            "min_scenarios": 3,  # Still provide alternatives
        },
        metadata={
            "difficulty": "easy",
            "category": "hold_decision",
            "key_decision": "patience",
        },
    )


def create_chip_preparation_case() -> Case[Dict[str, Any], Dict[str, Any]]:
    """Test case: Should position squad for upcoming chip usage."""
    return Case(
        name="wildcard_preparation",
        inputs={
            "scenario": "GW19, 2 FTs, £4.0m ITB",
            "context": "Planning Wildcard in GW20, need short-term coverage for GW19",
            "expected_behavior": "Recommend cheap short-term fixes, flag chip preparation",
        },
        expected_output={
            "should_include_hold": True,
            "should_have_chip_prep_scenario": True,
            "chip_prep_should_be_flagged": True,  # prepares_for_chip = True
            "top_pick_reasoning_mentions": ["chip", "wildcard", "short-term"],
            "context_analysis_should_mention_chip": True,
        },
        metadata={
            "difficulty": "hard",
            "category": "chip_preparation",
            "key_decision": "timing",
        },
    )


# Create dataset with all scenarios
# Note: Comprehensive evaluation dataset for FPL transfer planning agent.
# Covers premium upgrades, differentials, template safety, DGWs, fixture swings,
# budget constraints, injuries, aggressive/conservative strategies, and chip prep.
transfer_planning_dataset = Dataset(
    name="fpl_transfer_planning",
    cases=[
        create_premium_upgrade_case(),
        create_differential_punt_case(),
        create_template_safety_case(),
        create_dgw_opportunity_case(),
        create_fixture_swing_case(),
        create_budget_constraint_case(),
        create_injury_emergency_case(),
        create_aggressive_ceiling_case(),
        create_bank_transfer_case(),
        create_chip_preparation_case(),
    ],
)


# Quick access functions for subset datasets
def get_basic_scenarios() -> Dataset:
    """Get basic transfer scenarios (upgrades, differentials, template)."""
    return Dataset(
        cases=[
            create_premium_upgrade_case(),
            create_differential_punt_case(),
            create_template_safety_case(),
        ],
    )


def get_strategic_scenarios() -> Dataset:
    """Get strategic scenarios (DGW, fixture swings, chip prep)."""
    return Dataset(
        cases=[
            create_dgw_opportunity_case(),
            create_fixture_swing_case(),
            create_chip_preparation_case(),
        ],
    )


def get_constraint_scenarios() -> Dataset:
    """Get constraint-based scenarios (budget, injury, hold)."""
    return Dataset(
        cases=[
            create_budget_constraint_case(),
            create_injury_emergency_case(),
            create_bank_transfer_case(),
        ],
    )
