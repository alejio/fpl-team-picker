"""
ML Training module for FPL Expected Points prediction.

Provides a unified API for training, evaluating, and deploying ML models
across different configurations (unified, position-specific, hybrid).
"""

from .config import (
    TrainingConfig,
    PositionTrainingConfig,
    HybridPipelineConfig,
    EvaluationConfig,
)
from .trainer import MLTrainer
from .evaluator import ModelEvaluator
from .pipelines import build_pipeline, get_param_space, REGRESSOR_MAP

__all__ = [
    # Config
    "TrainingConfig",
    "PositionTrainingConfig",
    "HybridPipelineConfig",
    "EvaluationConfig",
    # Core classes
    "MLTrainer",
    "ModelEvaluator",
    # Pipeline utilities
    "build_pipeline",
    "get_param_space",
    "REGRESSOR_MAP",
]
