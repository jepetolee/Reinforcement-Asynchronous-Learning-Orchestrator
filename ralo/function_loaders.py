"""Function loader utilities for dynamically loading user-defined functions.

Similar to dataset loaders, this module provides utilities for loading functions
from module paths (e.g., "my_module:function_name").
"""

from typing import Any, Callable, List, Optional


def load_function(function_spec: Any) -> Optional[Callable]:
    """Load a function from a module path or return a callable.
    
    Args:
        function_spec: Module path to function (e.g., "my_module:function_name")
                      or a callable object
    
    Returns:
        Callable function, or None if loading fails
    """
    if callable(function_spec):
        return function_spec
    
    if not isinstance(function_spec, str):
        return None
    
    try:
        if ":" in function_spec:
            # Format: "module.path:function_name"
            module_path, function_name = function_spec.rsplit(":", 1)
            module = __import__(module_path, fromlist=[function_name])
            return getattr(module, function_name)
        else:
            # Assume it's a module path to a function named after the module
            module = __import__(function_spec, fromlist=["*"])
            # Try common function names
            for name in ["function", "fn", function_spec.split(".")[-1]]:
                if hasattr(module, name):
                    return getattr(module, name)
            return None
    except (ImportError, AttributeError) as e:
        print(f"[FUNCTION_LOADER] Warning: Failed to load function '{function_spec}': {e}")
        return None


def load_reward_functions(reward_fn_specs: Optional[List[Any]]) -> List[Callable]:
    """Load multiple reward functions from module paths or callables.
    
    Args:
        reward_fn_specs: List of module paths or callables (e.g., ["my_module:correct_fn", "my_module:format_fn"])
    
    Returns:
        List of callable reward functions
    """
    if not reward_fn_specs:
        return []
    
    reward_fns = []
    for spec in reward_fn_specs:
        fn = load_function(spec)
        if fn is not None:
            reward_fns.append(fn)
        else:
            print(f"[FUNCTION_LOADER] Warning: Failed to load reward function '{spec}', skipping")
    
    return reward_fns

