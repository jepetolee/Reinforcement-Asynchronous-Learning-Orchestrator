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
    builtin:aime_2024 | builtin:aime_2025 | builtin:olympiad_bench | hf:<dataset_id> | module:function
    Returns a Benchmark class or factory function(config)->Benchmark.
    """
    if loader.startswith("builtin:"):
        name = loader.split(":", 1)[1]

        # AIME family (future-proof: any builtin:aime_XXXX)
        if name.startswith("aime_"):
            from .benchmarks.aime import AIMEBenchmark

            def factory(cfg: Dict[str, Any]):
                # AIMEBenchmark internally handles special routing for aime_2025
                return AIMEBenchmark(cfg, dataset=name)

            return factory

        # GPQA variants, e.g. builtin:gpqa, builtin:gpqa_diamond, ...
        if name.startswith("gpqa"):
            from .benchmarks.gpqa import GPQABenchmark

            def factory(cfg: Dict[str, Any]):
                return GPQABenchmark(cfg)

            return factory

        # Named adapters (small finite set, not per-dataset)
        if name == "olympiad_bench":
            from .benchmarks.olympiad import OlympiadBenchBenchmark

            def factory(cfg: Dict[str, Any]):
                return OlympiadBenchBenchmark(cfg)

            return factory

        if name == "hle":
            # Backwards-compatible alias: "hle" → BaseBenchmark (generic HF QA loader)
            from .benchmarks.hle import BaseBenchmark

            def factory(cfg: Dict[str, Any]):
                return BaseBenchmark(cfg)

            return factory

        if name == "iiv":
            # IIV (Information Integration and Verification) benchmark
            from .benchmarks.iiv import IIVBenchmark

            def factory(cfg: Dict[str, Any]):
                return IIVBenchmark(cfg)

            return factory

        raise ValueError(f"Unknown builtin benchmark: {name}")
    elif loader.startswith("hf:"):
        # Generic HF loader via BaseBenchmark adapter
        from .benchmarks.hle import BaseBenchmark
        def factory(cfg: Dict[str, Any]):
            cfg = dict(cfg)
            cfg.setdefault("hf_dataset", loader[3:])
            return BaseBenchmark(cfg)
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
        if name.startswith("pass@"):
            k = int(name.split("@", 1)[1])
            def fn(preds, refs, candidates=None, **kw):
                return _builtin.pass_at_k(preds=preds, refs=refs, candidates=candidates, k=k, **kw)
            return fn
        if name.startswith("accuracy@"):
            k = int(name.split("@", 1)[1])
            def fn(preds, refs, candidates=None, **kw):
                return _builtin.accuracy_at_k(preds=preds, refs=refs, candidates=candidates, k=k, **kw)
            return fn
        if name.startswith("idk@"):
            k = int(name.split("@", 1)[1])
            def fn(preds, refs, candidates=None, **kw):
                return _builtin.idk_at_k(preds=preds, refs=refs, candidates=candidates, k=k, **kw)
            return fn
        if name.startswith("idk_accuracy@"):
            k = int(name.split("@", 1)[1])
            def fn(preds, refs, candidates=None, **kw):
                return _builtin.idk_accuracy_at_k(preds=preds, refs=refs, candidates=candidates, k=k, **kw)
            return fn
        raise ValueError(f"Unknown builtin metric: {name}")
    else:
        return _resolve_callable(spec)


