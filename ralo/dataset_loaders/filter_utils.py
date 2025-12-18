"""Utility filter functions for datasets.

Users can create custom filter functions and reference them in their config.
"""

from typing import Dict, Any, List, Optional, Callable


def create_level_filter(target_levels: List[str]) -> Callable[[Dict[str, Any]], bool]:
    """Create a filter function for difficulty levels.

    NOTE: Config에서는 이 함수를 직접 호출하는 형태
    (예: ``create_level_filter([...])``)를 사용할 수 없습니다.
    `filter_fn` 로더는 `"module.path:function_name"` 형태만 지원하므로,
    실제로는 아래처럼 **미리 바인딩된 래퍼 함수**를 만들어 써야 합니다.

    예시:

        # my_filters.py
        from ralo.dataset_loaders.filter_utils import create_level_filter
        filter_levels_3_4 = create_level_filter(["Level 3", "Level 4"])

        # YAML
        dataset:
          filter_fn: "my_filters:filter_levels_3_4"

    이 헬퍼는 그런 래퍼 함수를 만들기 위한 유틸입니다.
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


def filter_levels_3_4_5(example: Dict[str, Any]) -> bool:
    """Example filter: include only Level 3, 4, 5 problems."""
    level = example.get("level")
    if level is None:
        return False
    return level in {"Level 3", "Level 4", "Level 5"}


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


# Counter for limiting dataset size (used by filter_first_n)
_filter_counters: Dict[str, int] = {}


def filter_first_n(example: Dict[str, Any], n: int = 100, counter_key: str = "default") -> bool:
    """Filter to include only the first n examples.
    
    This function uses a module-level counter to track how many examples
    have been included. It should be used with a single dataset pass.
    
    Args:
        example: Dataset example (unused, but required by filter interface)
        n: Maximum number of examples to include
        counter_key: Key to use for the counter (allows multiple counters)
    
    Returns:
        True if this example should be included (count < n), False otherwise
    """
    if counter_key not in _filter_counters:
        _filter_counters[counter_key] = 0
    
    if _filter_counters[counter_key] < n:
        _filter_counters[counter_key] += 1
        return True
    return False


def filter_openbookqa_100(example: Dict[str, Any]) -> bool:
    """Filter to include only first 100 examples from OpenBookQA dataset."""
    return filter_first_n(example, n=100, counter_key="openbookqa_100")

