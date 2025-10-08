"""
Core FPL Analysis Components (Legacy - being migrated to domain/services)

This package contains legacy business logic being migrated:
- Expected Points (xP) models (will move to domain/services)

Note: data_loader → domain/services/data_orchestration_service (migrated)
Note: team_strength → domain/services/team_analytics_service (migrated)
"""

from .xp_model import XPModel

__all__ = [
    "XPModel",
]
