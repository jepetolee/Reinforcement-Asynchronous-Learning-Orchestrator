"""
Service layer for RALO components.

This module provides service abstractions for:
- ModelService: Model loading, weight synchronization, inference
- SamplingService: vLLM initialization/management, generation parameters, sample formatting
- TrainingService: Trainer wrapping, gradient collection, backward, batch processing
- OrchestratorService: SamplerClient/TrainerClient wrapping, problem fetch, sample upload, gradient upload
"""

from .model_service import ModelService
from .sampling_service import SamplingService
from .training_service import TrainingService
from .orchestrator_service import OrchestratorService

__all__ = [
    "ModelService",
    "SamplingService",
    "TrainingService",
    "OrchestratorService",
]

