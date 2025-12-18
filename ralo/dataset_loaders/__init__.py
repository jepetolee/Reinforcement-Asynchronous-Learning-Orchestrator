"""Dataset loaders for various HuggingFace datasets.

This module provides a flexible system for loading and transforming different
HuggingFace datasets into a unified format for training.
"""

from typing import List, Dict, Any, Optional
import random

from .base import BaseDatasetLoader
from .combined import CombinedDatasetLoader

# Registry for dataset loaders
_LOADER_REGISTRY: Dict[str, type] = {}


def register_loader(name: str, loader_class: type):
    """Register a dataset loader class."""
    _LOADER_REGISTRY[name] = loader_class


def get_loader(name: str) -> Optional[type]:
    """Get a loader class by name."""
    return _LOADER_REGISTRY.get(name)


def load_datasets(dataset_configs: List[Dict[str, Any]], shuffle_seed: Optional[int] = None, max_items: Optional[int] = None) -> List[Dict[str, str]]:
    """Load and combine multiple datasets.
    
    Args:
        dataset_configs: List of dataset configuration dictionaries
        shuffle_seed: Optional random seed for shuffling combined dataset
        max_items: Optional maximum number of items to return (None = no limit)
    
    Returns:
        List of QA dictionaries with 'Q' (question) and 'A' (answer) keys
    """
    if len(dataset_configs) == 1:
        # Single dataset - use direct loader
        cfg = dataset_configs[0]
        loader_name = cfg.get("loader", "auto")
        loader = _create_loader(loader_name, cfg)
        qas = loader.load()
        
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(qas)
        
        # Apply max_items limit if specified
        if max_items is not None and max_items > 0:
            qas = qas[:max_items]
        
        return qas
    else:
        # Multiple datasets - use combined loader
        loader = CombinedDatasetLoader(dataset_configs)
        qas = loader.load()
        
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            random.shuffle(qas)
        
        # Apply max_items limit if specified
        if max_items is not None and max_items > 0:
            qas = qas[:max_items]
        
        return qas


def _create_loader(loader_name: str, config: Dict[str, Any]) -> BaseDatasetLoader:
    """Create a dataset loader instance."""
    # Auto-detect loader from dataset name
    if loader_name == "auto":
        loader_name = _detect_loader_name(config.get("name", ""))
    
    loader_class = get_loader(loader_name)
    if loader_class is None:
        # Fallback to generic HuggingFace loader
        from .generic import GenericHFDatasetLoader
        loader_class = GenericHFDatasetLoader
    
    return loader_class(config)


def _detect_loader_name(dataset_name: str) -> str:
    """Auto-detect loader name from dataset name."""
    name_lower = dataset_name.lower()
    
    if "iiv" in name_lower:
        return "iiv"
    elif "cos_e" in name_lower or "cose" in name_lower:
        return "cose_iiv"
    elif "competition_math" in name_lower:
        return "competition_math"
    elif "math" in name_lower:
        return "math_dataset"
    elif "gsm8k" in name_lower:
        return "gsm8k"
    elif "hendrycks" in name_lower:
        return "hendrycks_math"
    else:
        return "generic"


# Import loaders to register them
from .competition_math import CompetitionMathLoader
register_loader("competition_math", CompetitionMathLoader)

from .generic import GenericHFDatasetLoader
register_loader("generic", GenericHFDatasetLoader)

from .iiv import IIVDatasetLoader
register_loader("iiv", IIVDatasetLoader)

from .cose_iiv import CosEIIVLoader
register_loader("cose_iiv", CosEIIVLoader)

