from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional


def _resolve_callable(path: str) -> Callable[..., Any]:
    """
    Resolve a dotted import path like "pkg.module:function".
    """
    if ":" not in path:
        raise ValueError(f"Invalid callable path: {path}")
    mod_name, func_name = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    if not callable(fn):
        raise ValueError(f"Resolved object is not callable: {path}")
    return fn


def get_benchmark(loader: str):
    """
    Resolve benchmark loader by name.
    builtin:aime_2024 | builtin:aime_2025 | hf:<dataset_id> | module:function
    Returns a Benchmark class or factory function(config)->Benchmark.
    """
    if loader.startswith("builtin:"):
        name = loader.split(":", 1)[1]
        if name == "aime_2024" or name == "aime_2025":
            from .benchmarks.aime import AIMEBenchmark
            # Create partially configured factory
            def factory(cfg: Dict[str, Any]):
                return AIMEBenchmark(cfg, dataset=name)
            return factory
        elif name.startswith("gpqa"):
            from .benchmarks.gpqa import GPQABenchmark
            def factory(cfg: Dict[str, Any]):
                return GPQABenchmark(cfg)
            return factory
        elif name == "hle":
            from .benchmarks.hle import HLEBenchmark
            def factory(cfg: Dict[str, Any]):
                return HLEBenchmark(cfg)
            return factory
        else:
            raise ValueError(f"Unknown builtin benchmark: {name}")
    elif loader.startswith("hf:"):
        # Generic HF loader via HLE adapter
        from .benchmarks.hle import HLEBenchmark
        def factory(cfg: Dict[str, Any]):
            cfg = dict(cfg)
            cfg.setdefault("hf_dataset", loader[3:])
            return HLEBenchmark(cfg)
        return factory
    else:
        # module:function
        return _resolve_callable(loader)


def get_metric(spec: str) -> Callable[..., Dict[str, float]]:
    """
    Resolve metric function by spec string.
    Supports:
      - builtin:exact_match
      - builtin:mc_accuracy@1
      - builtin:accuracy@K (e.g., builtin:accuracy@5)
      - module:function
    """
    if spec.startswith("builtin:"):
        name = spec.split(":", 1)[1]
        from .metrics import builtin as _builtin
        if name == "exact_match":
            return _builtin.exact_match
        if name.startswith("mc_accuracy@"):
            k = int(name.split("@", 1)[1])
            def fn(preds, refs, candidates=None, **kw):
                return _builtin.mc_accuracy_at_k(preds=preds, refs=refs, candidates=candidates, k=k, **kw)
            return fn
        if name.startswith("accuracy@"):
            k = int(name.split("@", 1)[1])
            def fn(preds, refs, candidates=None, **kw):
                return _builtin.accuracy_at_k(preds=preds, refs=refs, candidates=candidates, k=k, **kw)
            return fn
        raise ValueError(f"Unknown builtin metric: {name}")
    else:
        return _resolve_callable(spec)


