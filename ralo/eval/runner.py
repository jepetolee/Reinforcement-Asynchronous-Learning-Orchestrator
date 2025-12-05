from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .registry import get_benchmark, get_metric


def run_benchmark(benchmark_cfg: Dict[str, Any], sampling_service) -> Tuple[Dict[str, float], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Execute a benchmark using the provided SamplingService.

    Args:
        benchmark_cfg: dict with fields: name, loader, split, max_items, num_candidates, prompt_template, answer_extractor, metrics
        sampling_service: initialized SamplingService (vLLM) ready to generate

    Returns:
        (metrics_dict, sample_rows)
    """
    name = benchmark_cfg.get("name", "benchmark")
    loader = benchmark_cfg.get("loader", f"builtin:{name}")
    num_candidates = int(benchmark_cfg.get("num_candidates", 1) or 1)
    metrics_specs = benchmark_cfg.get("metrics") or ["builtin:accuracy@1"]

    # Instantiate benchmark
    bench_factory = get_benchmark(loader)
    bench = bench_factory(benchmark_cfg)

    # Collect prompts and references
    items = list(bench.load_items())
    prompts = [bench.make_prompt(it) for it in items]
    refs = [bench.reference(it) for it in items]

    # Generate
    SamplingParams = sampling_service.create_sampling_params(
        n=num_candidates,
        max_tokens=int(benchmark_cfg.get("max_tokens", 512)),
        temperature=None,  # use service default
        top_p=1.0,
        include_stop_str_in_output=False,
    )
    outputs = sampling_service.generate(prompts, sampling_params=SamplingParams, use_tqdm=False)

    # Extract predictions per item (list of strings per prompt)
    all_candidates: List[List[str]] = []
    flat_preds: List[str] = []
    sample_rows: List[Dict[str, Any]] = []
    generated_tokens = 0
    for prompt, out, ref in zip(prompts, outputs, refs):
        # vLLM output format: out.outputs[i].text
        cands: List[str] = []
        try:
            for gen in getattr(out, "outputs", []) or []:
                text = getattr(gen, "text", None)
                if text is None and isinstance(gen, dict):
                    text = gen.get("text")
                text = text or ""
                pred = bench.extract_prediction(text)
                cands.append(str(pred))
                tokens = getattr(gen, "token_ids", None)
                if tokens is None and isinstance(gen, dict):
                    tokens = gen.get("token_ids")
                if isinstance(tokens, list):
                    generated_tokens += len(tokens)
        except Exception:
            cands = []
        if not cands:
            cands = [""]
        all_candidates.append(cands)
        flat_preds.append(cands[0])
        # Collect a small row (first candidate only)
        sample_rows.append({
            "prompt": prompt,
            "pred": cands[0],
            "ref": ref,
        })

    # Compute metrics
    metrics: Dict[str, float] = {}
    for spec in metrics_specs:
        metric_fn = get_metric(spec)
        try:
            res = metric_fn(preds=flat_preds, refs=refs, candidates=all_candidates, benchmark=name)
            # merge
            for k, v in (res or {}).items():
                metrics[k] = float(v)
        except Exception as e:
            print(f"[EVAL] Error computing metric {spec}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Debug: Print first few predictions vs references for debugging
    if len(flat_preds) > 0 and len(refs) > 0:
        print(f"[EVAL DEBUG] First 3 predictions vs references:")
        for i in range(min(3, len(flat_preds))):
            print(f"  [{i}] pred: {repr(flat_preds[i][:100])} | ref: {repr(refs[i])}")

    # Optionally truncate sample rows to avoid huge payloads
    max_rows = int(benchmark_cfg.get("samples_preview", 16))
    if max_rows >= 0:
        sample_rows = sample_rows[:max_rows]
    else:
        sample_rows = None

    stats = {
        "generated_tokens": int(generated_tokens),
        "num_prompts": len(prompts),
        "num_candidates": num_candidates,
    }
    return metrics, sample_rows, stats


