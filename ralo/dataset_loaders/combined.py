"""Combined dataset loader for multiple datasets."""

from typing import List, Dict, Any, Optional

from .base import BaseDatasetLoader


class CombinedDatasetLoader(BaseDatasetLoader):
    """Loader that combines multiple datasets.
    
    This loader loads multiple datasets and combines them into a single list.
    """
    
    def __init__(self, dataset_configs: List[Dict[str, Any]]):
        """Initialize with list of dataset configurations.
        
        Args:
            dataset_configs: List of dataset configuration dictionaries
        """
        self.dataset_configs = dataset_configs
        self.loaders = []
        
        # Create loader for each dataset
        for cfg in dataset_configs:
            loader_name = cfg.get("loader", "auto")
            loader = self._create_loader(loader_name, cfg)
            self.loaders.append(loader)
    
    def _create_loader(self, loader_name: str, config: Dict[str, Any]) -> BaseDatasetLoader:
        """Create a dataset loader instance."""
        # Import here to avoid circular imports
        from .generic import GenericHFDatasetLoader
        
        # Auto-detect loader from dataset name
        if loader_name == "auto":
            dataset_name = config.get("name", "").lower()
            if "competition_math" in dataset_name:
                from .competition_math import CompetitionMathLoader
                return CompetitionMathLoader(config)
            else:
                # Use generic loader for unknown datasets
                return GenericHFDatasetLoader(config)
        else:
            # Try to get specific loader
            from . import get_loader
            loader_class = get_loader(loader_name)
            if loader_class is None:
                # Fallback to generic HuggingFace loader
                return GenericHFDatasetLoader(config)
            return loader_class(config)
    
    def load(self) -> List[Dict[str, str]]:
        """Load and combine all datasets.
        
        Returns:
            Combined list of all QA pairs from all datasets
        """
        all_qas = []
        
        for i, loader in enumerate(self.loaders):
            cfg = self.dataset_configs[i]
            dataset_name = cfg.get("name", f"dataset_{i}")
            
            try:
                qas = loader.load()
                all_qas.extend(qas)
                print(f"[DATASET] Loaded {len(qas)} examples from {dataset_name}")
            except Exception as e:
                print(f"[DATASET] Warning: Failed to load {dataset_name}: {e}")
                continue
        
        print(f"[DATASET] Combined total: {len(all_qas)} examples from {len(self.loaders)} datasets")
        
        # Shuffle combined dataset if any config specifies shuffle_seed
        # Use the first non-None shuffle_seed found
        shuffle_seed = None
        for cfg in self.dataset_configs:
            seed = cfg.get("shuffle_seed")
            if seed is not None:
                shuffle_seed = seed
                break
        
        if shuffle_seed is not None:
            import random
            random.seed(shuffle_seed)
            random.shuffle(all_qas)
        
        return all_qas
    
    def extract_qa(self, example: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Not used in combined loader."""
        raise NotImplementedError("CombinedDatasetLoader does not use extract_qa directly")

