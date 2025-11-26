"""Optimization service for FPL squad and transfer optimization.

This service contains core FPL optimization algorithms including:
- Starting XI selection with formation optimization
- Bench player selection
- Budget pool calculations
- Transfer scenario analysis (0-3 transfers) using LP or SA
- Premium player acquisition planning
- Captain selection
- Initial squad generation using simulated annealing

Optimization Methods:
- Linear Programming (LP): Optimal, fast, deterministic (default for transfers)
- Simulated Annealing (SA): Exploratory, non-linear objectives (squad generation)

This is a thin facade that composes all optimization mixins.
"""

from typing import Dict, Optional, Any

from .optimization import (
    SquadSelectionMixin,
    CaptainServiceMixin,
    SquadGenerationMixin,
)


class OptimizationService(
    SquadSelectionMixin,
    CaptainServiceMixin,
    SquadGenerationMixin,
):
    """Service for FPL optimization algorithms and constraint satisfaction.

    This class composes all optimization functionality through mixins:
    - OptimizationBaseMixin: Shared utilities (inherited via other mixins)
    - SquadSelectionMixin: Starting XI, bench, budget calculations
    - CaptainServiceMixin: Captain recommendations
    - TransferOptimizationMixin: LP/SA transfer optimization (inherited via SquadGenerationMixin)
    - SquadGenerationMixin: Initial squad/wildcard optimization
    """

    def __init__(self, optimization_config: Optional[Dict[str, Any]] = None):
        """Initialize optimization service.

        Args:
            optimization_config: Optional optimization configuration override
        """
        self.config = optimization_config or {}
