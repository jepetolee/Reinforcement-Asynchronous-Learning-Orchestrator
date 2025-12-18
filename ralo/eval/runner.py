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
    
    # Get tokenizer from sampling service for thinking mode support
    # Try to get tokenizer from vLLM engine if available
    tokenizer = None
    try:
        vllm_engine = sampling_service.get_vllm_engine()
        if hasattr(vllm_engine, 'llm_engine') and hasattr(vllm_engine.llm_engine, 'tokenizer'):
            tokenizer = vllm_engine.llm_engine.tokenizer.tokenizer
        elif hasattr(vllm_engine, 'tokenizer'):
            tokenizer = vllm_engine.tokenizer
    except Exception:
        pass
    
    # Check if thinking mode should be enabled (for Qwen3 models)
    enable_thinking = benchmark_cfg.get("enable_thinking", False)
    
    # Apply chat template with thinking mode if tokenizer is available and enabled
    if tokenizer is not None and enable_thinking and hasattr(tokenizer, 'apply_chat_template'):
        # For thinking mode, use apply_chat_template with enable_thinking=True
        prompts = []
        for it in items:
            raw_prompt = bench.make_prompt(it)
            # Try to extract question from the prompt if it's in AIME format
            question = it.get("problem", "") or it.get("question", "")
            if not question and "Problem:" in raw_prompt:
                # Extract question from prompt
                parts = raw_prompt.split("Problem:", 1)
                if len(parts) > 1:
                    question = parts[1].split("Solution:", 1)[0].strip()
            
            if question:
                # Use chat template with thinking mode
                system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,  # Enable thinking mode for Qwen3
                    )
                    prompts.append(formatted_prompt)
                except Exception:
                    # Fallback to raw prompt if chat template fails
                    prompts.append(raw_prompt)
            else:
                prompts.append(raw_prompt)
    else:
        # Use make_prompt directly (no thinking mode)
    prompts = [bench.make_prompt(it) for it in items]
    
    refs = [bench.reference(it) for it in items]

    # Generate
    # Allow temperature and top_p to be configured in benchmark config
    # For thinking mode: Temperature=0.6, TopP=0.95, TopK=20, MinP=0
    temperature = benchmark_cfg.get("temperature")
    if temperature is not None:
        temperature = float(temperature)
    top_p = benchmark_cfg.get("top_p")
    if top_p is not None:
        top_p = float(top_p)
    else:
        top_p = 1.0  # default
    
    # For thinking mode, also configure top_k and min_p
    top_k = benchmark_cfg.get("top_k")
    if top_k is not None:
        top_k = int(top_k)
    elif enable_thinking:
        top_k = 20  # Default for thinking mode
    
    min_p = benchmark_cfg.get("min_p")
    if min_p is not None:
        min_p = float(min_p)
    elif enable_thinking:
        min_p = 0.0  # Default for thinking mode
    
    # Create sampling params
    sampling_kwargs = {
        "n": num_candidates,
        "max_tokens": int(benchmark_cfg.get("max_tokens", 512)),
        "temperature": temperature,  # None = use service default, otherwise use configured value
        "top_p": top_p,
        "include_stop_str_in_output": False,
    }
    
    # Add top_k and min_p if specified (vLLM supports these)
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k
    if min_p is not None:
        sampling_kwargs["min_p"] = min_p
    
    SamplingParams = sampling_service.create_sampling_params(**sampling_kwargs)
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


