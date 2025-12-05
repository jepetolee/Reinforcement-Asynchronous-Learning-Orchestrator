from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Type

from .sample_schema import SampleSchema


class SamplerAlgorithm:
    def __init__(self, ralo, config):
        self.ralo = ralo
        self.config = config

    def run(self) -> None:
        """Run the sampler algorithm (legacy method, may call process_problem in a loop)."""
        raise NotImplementedError

    def create_sample_schema(self) -> SampleSchema:
        """
        Create the sample schema used by this algorithm.

        Returns:
            SampleSchema instance
        """
        raise NotImplementedError

    def process_problem(self, problem: Dict[str, Any], services: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a problem and return a list of samples.

        Args:
            problem: Problem dictionary
            services: Dictionary of services {'model': ModelService, 'sampling': SamplingService, 'orchestrator': OrchestratorService}

        Returns:
            List of sample dictionaries
        """
        raise NotImplementedError

    def format_samples_for_upload(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format samples for upload to orchestrator.

        Args:
            samples: List of sample dictionaries

        Returns:
            Formatted dictionary for upload
        """
        raise NotImplementedError


class TrainerAlgorithm:
    def __init__(self, ralo, config):
        self.ralo = ralo
        self.config = config

    def run(self) -> None:
        """Run the trainer algorithm (legacy method, may call training loop)."""
        raise NotImplementedError

    def compute_loss(self, model, batch: Dict[str, Any], services: Dict[str, Any]) -> Any:
        """
        Compute loss from a batch.

        Args:
            model: Model instance
            batch: Batch dictionary
            services: Dictionary of services {'model': ModelService, 'training': TrainingService, 'orchestrator': OrchestratorService}

        Returns:
            Loss tensor
        """
        raise NotImplementedError

    def prepare_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare a batch from a list of samples.

        Args:
            samples: List of sample dictionaries

        Returns:
            Batch dictionary
        """
        raise NotImplementedError


_SAMPLER_ALGOS: Dict[str, Type[SamplerAlgorithm]] = {}
_TRAINER_ALGOS: Dict[str, Type[TrainerAlgorithm]] = {}


def register_sampler_algorithm(name: str):
    def decorator(cls: Type[SamplerAlgorithm]):
        _SAMPLER_ALGOS[name.lower()] = cls
        return cls

    return decorator


def register_trainer_algorithm(name: str):
    def decorator(cls: Type[TrainerAlgorithm]):
        _TRAINER_ALGOS[name.lower()] = cls
        return cls

    return decorator


def get_sampler_algorithm(name: str) -> Type[SamplerAlgorithm]:
    key = (name or "treepo").lower()
    if key not in _SAMPLER_ALGOS:
        raise KeyError(f"Unknown sampler algorithm: {name}")
    return _SAMPLER_ALGOS[key]


def get_trainer_algorithm(name: str) -> Type[TrainerAlgorithm]:
    key = (name or "treepo").lower()
    if key not in _TRAINER_ALGOS:
        raise KeyError(f"Unknown trainer algorithm: {name}")
    return _TRAINER_ALGOS[key]

