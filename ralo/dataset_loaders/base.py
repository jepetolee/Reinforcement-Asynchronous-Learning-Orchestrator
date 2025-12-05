"""Base dataset loader class."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders.
    
    Each dataset loader is responsible for:
    1. Loading data from HuggingFace datasets
    2. Transforming it into a unified format (Q: question, A: answer)
    3. Applying user-provided filter functions (if specified)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dataset loader.
        
        Args:
            config: Dataset configuration dictionary containing:
                - name: HuggingFace dataset identifier
                - split: Dataset split to use (e.g., "train", "test")
                - filter_fn: Optional callable or module path to filter function (e.g., "module.path:function_name")
                - shuffle_seed: Optional random seed for shuffling
                - Other dataset-specific parameters
        """
        self.config = config
        self.name = config.get("name", "")
        self.split = config.get("split", "train")
        self.shuffle_seed = config.get("shuffle_seed")
        self._dataset = None
        self._filter_fn = None
        
        # Load user-provided filter function if specified
        filter_fn_spec = config.get("filter_fn")
        if filter_fn_spec:
            self._filter_fn = self._load_filter_function(filter_fn_spec)
    
    @abstractmethod
    def load(self) -> List[Dict[str, str]]:
        """Load and transform the dataset.
        
        Returns:
            List of dictionaries with 'Q' (question) and 'A' (answer) keys.
            Each dictionary represents one problem-solution pair.
        """
        pass
    
    @abstractmethod
    def extract_qa(self, example: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract question and answer from a dataset example.
        
        Args:
            example: A single example from the dataset
            
        Returns:
            Dictionary with 'Q' and 'A' keys, or None if extraction fails
        """
        pass
    
    def _load_filter_function(self, filter_fn_spec: str):
        """Load a filter function from a module path.
        
        Args:
            filter_fn_spec: Module path to filter function (e.g., "my_module:filter_fn")
                           or a callable object
        
        Returns:
            Callable filter function, or None if loading fails
        """
        if callable(filter_fn_spec):
            return filter_fn_spec
        
        if not isinstance(filter_fn_spec, str):
            return None
        
        try:
            if ":" in filter_fn_spec:
                # Format: "module.path:function_name"
                module_path, function_name = filter_fn_spec.rsplit(":", 1)
                module = __import__(module_path, fromlist=[function_name])
                return getattr(module, function_name)
            else:
                # Assume it's a module path to a function named "filter_fn"
                module = __import__(filter_fn_spec, fromlist=["filter_fn"])
                return getattr(module, "filter_fn", None)
        except (ImportError, AttributeError) as e:
            print(f"[DATASET] Warning: Failed to load filter function '{filter_fn_spec}': {e}")
            return None
    
    def should_include_example(self, example: Dict[str, Any]) -> bool:
        """Check if an example should be included based on user-provided filter.
        
        Args:
            example: A single example from the dataset
            
        Returns:
            True if example should be included, False otherwise
        """
        if self._filter_fn is None:
            return True  # No filter specified, include all
        
        try:
            return bool(self._filter_fn(example))
        except Exception as e:
            print(f"[DATASET] Warning: Filter function raised exception: {e}")
            return True  # Include on error to avoid data loss

