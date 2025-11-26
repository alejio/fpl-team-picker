"""Optimization module for FPL squad and transfer optimization.

This module provides:
- Starting XI and bench selection
- Captain recommendations
- Transfer optimization (LP and SA)
- Squad generation and wildcard optimization

Usage:
    from fpl_team_picker.domain.services.optimization import OptimizationService

    service = OptimizationService()
    result = service.optimize_transfers(...)
"""

from .optimization_base import OptimizationBaseMixin, InitialSquadOptimizationInput
from .squad_selection import SquadSelectionMixin
from .captain_service import CaptainServiceMixin
from .transfer_core import TransferOptimizationMixin
from .squad_generation import SquadGenerationMixin

__all__ = [
    "OptimizationBaseMixin",
    "InitialSquadOptimizationInput",
    "SquadSelectionMixin",
    "CaptainServiceMixin",
    "TransferOptimizationMixin",
    "SquadGenerationMixin",
]
