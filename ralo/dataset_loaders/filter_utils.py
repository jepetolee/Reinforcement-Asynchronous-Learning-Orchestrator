"""Utility filter functions for datasets.

Users can create custom filter functions and reference them in their config.
"""

from typing import Dict, Any, List, Optional, Callable


def create_level_filter(target_levels: List[str]) -> Callable[[Dict[str, Any]], bool]:
    """Create a filter function for difficulty levels.
    
    Example usage in YAML:
        filter_fn: "ralo.dataset_loaders.filter_utils:create_level_filter(['Level 3', 'Level 4'])"
    
    Note: For backward compatibility only. Users should create their own filter functions.
    """
    target_set = set(target_levels)
    
    def filter_fn(example: Dict[str, Any]) -> bool:
        level = example.get("level")
        if level is None:
            return True  # Include if no level field
        return level in target_set
    
    return filter_fn


# Example filter functions users can reference

def filter_level_1(example: Dict[str, Any]) -> bool:
    """Filter examples to include only Level 1."""
    level = example.get("level")
    if level is None:
        return False  # Exclude if no level field
    return level == "Level 1"


def filter_by_length(example: Dict[str, Any], min_length: int = 0, max_length: Optional[int] = None) -> bool:
    """Filter examples by problem length."""
    problem = example.get("problem") or example.get("question") or ""
    if not isinstance(problem, str):
        return False
    
    length = len(problem.split())
    if length < min_length:
        return False
    if max_length is not None and length > max_length:
        return False
    return True

