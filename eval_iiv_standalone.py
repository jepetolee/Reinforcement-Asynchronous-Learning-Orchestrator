#!/usr/bin/env python3
"""
Standalone script to evaluate IIV metrics (IDK@16 and IDK_Accuracy@16) using trained weights.

Usage:
    python eval_iiv_standalone.py --weights weights_v7.pt --model Qwen/Qwen3-4B
"""

import os
# Enable insecure serialization for vLLM (needed for function serialization in apply_model)
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Module-level variable for weights loading (to avoid closure issues)
_weights_path_for_loading = None


def _is_idk(text: str) -> bool:
    """Check if text contains IDK pattern."""
    text_lower = str(text).lower().strip()
    idk_patterns = [
        "i don't know", "i don't know.", "i don't know,",
        "idk", "idk.", "idk,",
        "i do not know", "i do not know.", "i do not know,",
    ]
    for pattern in idk_patterns:
        if pattern in text_lower:
            return True
    return False


def _math_equal(pred: str, ref: str) -> bool:
    """Check if two mathematical expressions are equal using math_verify if available."""
    try:
        from math_verify import ExprExtractionConfig, parse, verify
        try:
            pred_parsed = parse(pred, extraction_config=[ExprExtractionConfig()])
            ref_parsed = parse(ref, extraction_config=[ExprExtractionConfig()])
            return verify(ref_parsed, pred_parsed)
        except Exception:
            pass
    except ImportError:
        pass
    # Fallback to normalized string comparison
    pred_norm = pred.strip().lower()
    ref_norm = ref.strip().lower()
    return pred_norm == ref_norm


def idk_at_k(predictions: List[List[str]], references: List[str], k: int = 16) -> float:
    """
    IDK@K: for each item where reference is "I don't know", check if any of the first K candidates
    contains IDK pattern.
    """
    idk_ref_count = 0
    idk_correct_count = 0
    
    for preds, ref in zip(predictions, references):
        ref_str = str(ref).strip().lower()
        # Only evaluate items where reference is "I don't know" (no hint case)
        if ref_str in ["i don't know", "idk", "i do not know"]:
            idk_ref_count += 1
            preds_k = preds[:k]
            # Check if any candidate contains IDK pattern
            for p in preds_k:
                if _is_idk(str(p)):
                    idk_correct_count += 1
                    break
    
    if idk_ref_count == 0:
        return 0.0
    
    return idk_correct_count / idk_ref_count


def idk_accuracy_at_k(predictions: List[List[str]], references: List[str], k: int = 16) -> float:
    """
    IDK_Accuracy@K: Accuracy@K but excluding items where the prediction is IDK.
    """
    total_items = len(references)
    total_fraction = 0.0
    excluded_count = 0
    
    for preds, ref in zip(predictions, references):
        preds_k = preds[:k]
        
        # Check if any candidate is IDK
        has_idk = False
        for p in preds_k:
            if _is_idk(str(p)):
                has_idk = True
                break
        
        if has_idk:
            # Exclude this item from accuracy calculation
            excluded_count += 1
            continue
        
        # Calculate accuracy for this item (no IDK in candidates)
        correct_count = 0
        for p in preds_k:
            if _math_equal(str(p), str(ref)):
                correct_count += 1
        # Divide by K
        per_item_acc = correct_count / float(k) if k > 0 else 0.0
        total_fraction += per_item_acc
    
    # Average over items that were NOT excluded (attempted to answer)
    valid_items = total_items - excluded_count
    if valid_items == 0:
        return 0.0
    
    return total_fraction / valid_items


def make_iiv_prompt(tokenizer: AutoTokenizer, question: str, hint: Optional[str] = None) -> str:
    """Create IIV prompt based on hint presence (matching training prompt function)."""
    if hint is None or str(hint).strip() == "":
        # No hint: instruct to say "I don't know"
        system_prompt = (
            "You are given a question, but you do not have enough information to answer it. "
            "If you cannot answer the question with certainty, you must respond with \"I don't know\" "
            "instead of guessing."
        )
        user_content = question
    else:
        # Has hint: include hint and ask for answer
        system_prompt = (
            "You are given a question and a hint. Use the hint to answer the question accurately. "
            "Please reason step by step and provide your final answer."
        )
        user_content = f"Question: {question}\n\nHint: {hint}"
    
    # Apply chat template (matching training setup)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    return formatted_prompt


def load_weights_vllm(llm: LLM, weights_path: str):
    """Load weights into vLLM model using apply_model."""
    global _weights_path_for_loading
    print(f"Loading weights from {weights_path}...")
    try:
        # Use vLLM's apply_model method (recommended way for vLLM v1)
        if hasattr(llm, 'apply_model'):
            # Store weights_path as a module-level variable to avoid closure issues
            _weights_path_for_loading = weights_path
            
            def apply_state_dict(model):
                """Function to apply state_dict to model.
                Loads weights inside function to avoid closure serialization issues.
                """
                try:
                    # Import at function level to avoid closure capture
                    import torch as _torch
                    # Use module-level variable (set before calling apply_model)
                    weights_file = _weights_path_for_loading
                    
                    # Load weights inside function
                    checkpoint = _torch.load(weights_file, map_location="cpu")
                    if isinstance(checkpoint, dict):
                        if "model_state_dict" in checkpoint:
                            state_dict_to_load = checkpoint["model_state_dict"]
                        elif "state_dict" in checkpoint:
                            state_dict_to_load = checkpoint["state_dict"]
                        else:
                            state_dict_to_load = checkpoint
                    else:
                        state_dict_to_load = checkpoint
                    
                    # Remove "module." prefix
                    state_dict_to_load = {k.replace("module.", ""): v for k, v in state_dict_to_load.items()}
                    
                    # Convert to bfloat16
                    state_dict_bfloat16 = {}
                    for k, v in state_dict_to_load.items():
                        if isinstance(v, _torch.Tensor):
                            state_dict_bfloat16[k] = v.to(_torch.bfloat16)
                        else:
                            state_dict_bfloat16[k] = v
                    
                    # Apply weights
                    if hasattr(model, 'load_weights'):
                        model.load_weights(state_dict_bfloat16.items())
                    elif hasattr(model, 'load_state_dict'):
                        model.load_state_dict(state_dict_bfloat16, strict=False)
                    else:
                        raise AttributeError("Model has neither load_weights nor load_state_dict")
                    return True
                except Exception as e:
                    print(f"Error applying weights in worker: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            llm.apply_model(apply_state_dict)
            print("Weights loaded successfully using apply_model")
        else:
            # Fallback: try to access model directly (may not work with vLLM v1)
            print("Warning: apply_model not available, trying direct access...")
            try:
                # Try different vLLM structures
                model_runner = None
                if hasattr(llm, 'llm_engine'):
                    llm_engine = llm.llm_engine
                    if hasattr(llm_engine, 'engine_core'):
                        engine_core = llm_engine.engine_core
                        if hasattr(engine_core, 'model_executor'):
                            model_executor = engine_core.model_executor
                            if hasattr(model_executor, 'driver_worker'):
                                model_runner = model_executor.driver_worker.model_runner
                            elif hasattr(model_executor, 'model_runner'):
                                model_runner = model_executor.model_runner
                    
                    if model_runner is None:
                        # Try older structure
                        if hasattr(llm_engine, 'model_executor'):
                            model_executor = llm_engine.model_executor
                            if hasattr(model_executor, 'driver_worker'):
                                model_runner = model_executor.driver_worker.model_runner
                
                if model_runner is not None:
                    model = model_runner.model
                    if hasattr(model, 'load_state_dict'):
                        model.load_state_dict(state_dict, strict=False)
                        print("Weights loaded successfully using load_state_dict")
                    else:
                        raise AttributeError("Model runner found but no load_state_dict method")
                else:
                    raise AttributeError("Cannot find model_runner in vLLM structure")
            except Exception as e:
                print(f"Warning: Failed to load weights into vLLM model: {e}")
                print("Continuing with base model...")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"Warning: Failed to load weights file: {e}")
        print("Continuing with base model...")
        import traceback
        traceback.print_exc()


def generate_samples_vllm(
    llm: LLM,
    tokenizer: AutoTokenizer,
    formatted_prompt: str,
    num_samples: int = 16,
    max_tokens: int = 1500,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: Optional[int] = None,
    min_p: float = 0.0,
) -> List[str]:
    """Generate multiple samples for a prompt using vLLM.
    
    Args:
        llm: vLLM LLM instance
        tokenizer: Tokenizer instance
        formatted_prompt: Already formatted prompt (with chat template applied)
        num_samples: Number of samples to generate
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling (optional)
        min_p: Min-p sampling
    
    Returns:
        List of generated text samples
    """
    
    # Create sampling params
    sampling_params_kwargs = {
        "n": num_samples,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if top_k is not None:
        sampling_params_kwargs["top_k"] = top_k
    if min_p > 0:
        sampling_params_kwargs["min_p"] = min_p
    
    sampling_params = SamplingParams(**sampling_params_kwargs)
    
    # Generate using vLLM (batch generation)
    outputs = llm.generate([formatted_prompt], sampling_params)
    
    # Extract generated texts
    samples = []
    if outputs and len(outputs) > 0:
        for output in outputs[0].outputs:
            samples.append(output.text.strip())
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate IIV metrics on trained model")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights file (e.g., weights_v7.pt). If not provided, uses base model.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B", help="Model name or path")
    parser.add_argument("--dataset", type=str, default="jepetolee/IIV_testing", help="IIV dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples per prompt")
    parser.add_argument("--max_tokens", type=int, default=1500, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling (optional)")
    parser.add_argument("--min_p", type=float, default=0.0, help="Min-p sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items to evaluate")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset)[args.split]
    
    # Load tokenizer
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # Initialize vLLM
    print(f"Initializing vLLM for {args.model}...")
    vllm_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.8,
        "trust_remote_code": True,
    }
    llm = LLM(**vllm_kwargs)
    
    # Load trained weights if provided
    if args.weights:
        load_weights_vllm(llm, args.weights)
        print("Using trained weights from checkpoint")
    else:
        print("Using base model (no weights loaded)")
    
    # Prepare data
    questions = []
    references = []
    hints = []
    
    for item in dataset:
        question = item.get("Problem") or item.get("problem") or item.get("question")
        answer = item.get("Answer") or item.get("answer")
        hint = item.get("Hint") or item.get("hint") or ""
        
        if question is None or answer is None:
            continue
        
        questions.append(question)
        hints.append(hint)
        
        # Reference: "I don't know" if no hint, otherwise the answer
        if hint is None or str(hint).strip() == "":
            references.append("I don't know")
        else:
            references.append(str(answer).strip())
        
        if args.max_items and len(questions) >= args.max_items:
            break
    
    print(f"Evaluating {len(questions)} items...")
    
    # Generate predictions
    print("Preparing prompts...")
    
    # 1. Prompts for IIV evaluation (ALWAYS without hint)
    all_prompts_no_hint = []
    
    # 2. Prompts for Accuracy evaluation (With hint if available)
    # Only for items that actually have hints. For items without hints, we don't need to generate again.
    all_prompts_with_hint = []
    indices_with_hint = []
    
    for i, (question, hint) in enumerate(zip(questions, hints)):
        # 1. No Hint Prompt (for IIV)
        prompt_no_hint = make_iiv_prompt(tokenizer, question, hint=None)
        all_prompts_no_hint.append(prompt_no_hint)
        
        # 2. With Hint Prompt (for Accuracy)
        has_hint = hint and str(hint).strip() != ""
        if has_hint:
            prompt_with_hint = make_iiv_prompt(tokenizer, question, hint=hint)
            all_prompts_with_hint.append(prompt_with_hint)
            indices_with_hint.append(i)
            
    # Combine all prompts for batch generation
    combined_prompts = all_prompts_no_hint + all_prompts_with_hint
    
    print(f"Generating samples: {len(all_prompts_no_hint)} (No Hint) + {len(all_prompts_with_hint)} (With Hint) = {len(combined_prompts)} total requests")
    
    # Create sampling params
    sampling_params_kwargs = {
        "n": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.top_k is not None:
        sampling_params_kwargs["top_k"] = args.top_k
    if args.min_p > 0:
        sampling_params_kwargs["min_p"] = args.min_p
    
    sampling_params = SamplingParams(**sampling_params_kwargs)
    
    # Batch generation
    outputs = llm.generate(combined_prompts, sampling_params)
    
    # Separate outputs
    # First N outputs are No Hint results
    # Next M outputs are With Hint results
    num_no_hint = len(all_prompts_no_hint)
    outputs_no_hint = outputs[:num_no_hint]
    outputs_with_hint = outputs[num_no_hint:]
    
    # Process results
    all_predictions_no_hint = []  # List of samples for each item (generated without hint)
    all_predictions_with_hint_map = {} # Map index -> samples (generated with hint)
    
    generation_details = []
    
    # Process No Hint results
    for i, output in enumerate(outputs_no_hint):
        samples = [o.text.strip() for o in output.outputs]
        all_predictions_no_hint.append(samples)
        
        # Store generation details (No Hint version)
        generation_details.append({
            "index": i,
            "type": "no_hint",
            "question": questions[i],
            "hint_used": False,
            "original_hint": hints[i] if hints[i] else "",
            "reference": "I don't know", # Reference for IIV is always IDK
            "prompt": all_prompts_no_hint[i],
            "samples": samples,
        })
        
    # Process With Hint results
    for j, output in enumerate(outputs_with_hint):
        original_idx = indices_with_hint[j]
        samples = [o.text.strip() for o in output.outputs]
        all_predictions_with_hint_map[original_idx] = samples
        
        # Store generation details (With Hint version)
        generation_details.append({
            "index": original_idx,
            "type": "with_hint",
            "question": questions[original_idx],
            "hint_used": True,
            "original_hint": hints[original_idx],
            "reference": references[original_idx], # Reference is the actual answer
            "prompt": all_prompts_with_hint[j],
            "samples": samples,
        })
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # 1. IIV Metrics (Using No Hint Generations)
    # Evaluate IDK behavior on items that originally had hints vs items that didn't
    
    iiv_preds_orig_no_hint = []
    iiv_refs_orig_no_hint = []
    
    iiv_preds_orig_with_hint = []
    iiv_refs_orig_with_hint = []
    
    for i, (preds, hint) in enumerate(zip(all_predictions_no_hint, hints)):
        reference_idk = "I don't know"
        
        if hint is None or str(hint).strip() == "":
            # Originally No Hint
            iiv_preds_orig_no_hint.append(preds)
            iiv_refs_orig_no_hint.append(reference_idk)
        else:
            # Originally With Hint (but generated without)
            iiv_preds_orig_with_hint.append(preds)
            iiv_refs_orig_with_hint.append(reference_idk)
            
    # Calculate IIV metrics
    idk_16_orig_no_hint = idk_at_k(iiv_preds_orig_no_hint, iiv_refs_orig_no_hint, k=16)
    idk_acc_16_orig_no_hint = idk_accuracy_at_k(iiv_preds_orig_no_hint, iiv_refs_orig_no_hint, k=16)
    
    idk_16_orig_with_hint = idk_at_k(iiv_preds_orig_with_hint, iiv_refs_orig_with_hint, k=16)
    idk_acc_16_orig_with_hint = idk_accuracy_at_k(iiv_preds_orig_with_hint, iiv_refs_orig_with_hint, k=16)
    
    # Overall IIV metrics (all generated without hint)
    all_refs_idk = ["I don't know"] * len(all_predictions_no_hint)
    idk_16_overall = idk_at_k(all_predictions_no_hint, all_refs_idk, k=16)
    idk_acc_16_overall = idk_accuracy_at_k(all_predictions_no_hint, all_refs_idk, k=16)
    
    # 2. Accuracy Metrics (Using With Hint Generations where available)
    # Evaluate how well model answers when hint is provided
    
    acc_preds = []
    acc_refs = []
    
    for i in indices_with_hint:
        preds = all_predictions_with_hint_map[i]
        # Reference is the actual answer (stored in references list, except we might have modified it earlier)
        # We need to get the actual answer from dataset, or references array (which had logic applied)
        # Let's re-extract the answer from references array. Note: references array logic in lines 380-384
        # put "I don't know" if no hint. But for indices_with_hint, we know there was a hint.
        # So references[i] should be the answer.
        acc_preds.append(preds)
        acc_refs.append(references[i])
        
    # Calculate Accuracy@16 (we can use idk_accuracy_at_k but it filters IDK. Regular accuracy might be better?)
    # Using idk_accuracy_at_k will give accuracy on non-IDK answers.
    if acc_preds:
        acc_16_with_hint_provided = idk_accuracy_at_k(acc_preds, acc_refs, k=16)
        # Also calculate regular coverage/accuracy?
        # Let's stick to the requested metrics.
    else:
        acc_16_with_hint_provided = 0.0

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Total Items: {len(questions)}")
    print(f"  - Original No Hint: {len(iiv_preds_orig_no_hint)}")
    print(f"  - Original With Hint: {len(iiv_preds_orig_with_hint)}")
    print(f"Samples per item: {args.num_samples}")
    
    print(f"\n[IIV Metrics] (Generated WITHOUT Hint)")
    print(f"  Subset: Original No Hint")
    print(f"    IDK@16: {idk_16_orig_no_hint:.4f} ({idk_16_orig_no_hint*100:.2f}%)")
    print(f"    IDK_Accuracy@16: {idk_acc_16_orig_no_hint:.4f}")
    
    print(f"  Subset: Original With Hint (Hint Removed)")
    print(f"    IDK@16: {idk_16_orig_with_hint:.4f} ({idk_16_orig_with_hint*100:.2f}%)")
    print(f"    IDK_Accuracy@16: {idk_acc_16_orig_with_hint:.4f}")
    
    print(f"  Overall (All Items No Hint)")
    print(f"    IDK@16: {idk_16_overall:.4f} ({idk_16_overall*100:.2f}%)")
    print(f"    IDK_Accuracy@16: {idk_acc_16_overall:.4f}")
    
    print(f"\n[Accuracy Metrics] (Generated WITH Hint)")
    print(f"  Subset: Original With Hint (Hint Provided)")
    print(f"    IDK_Accuracy@16: {acc_16_with_hint_provided:.4f} ({acc_16_with_hint_provided*100:.2f}%)")
    print("="*60)
    
    # Save results to JSON
    results = {
        "dataset": args.dataset,
        "num_items": len(questions),
        "metrics": {
            "iiv_no_hint": {
                "idk@16": idk_16_orig_no_hint,
                "idk_accuracy@16": idk_acc_16_orig_no_hint,
            },
            "iiv_with_hint_removed": {
                "idk@16": idk_16_orig_with_hint,
                "idk_accuracy@16": idk_acc_16_orig_with_hint,
            },
            "iiv_overall": {
                "idk@16": idk_16_overall,
                "idk_accuracy@16": idk_acc_16_overall,
            },
            "accuracy_hint_provided": {
                "idk_accuracy@16": acc_16_with_hint_provided,
            }
        },
        "config": {
            "weights": args.weights if args.weights else "base_model",
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
        },
        "generations": generation_details,
    }
    
    if args.weights:
        output_file = f"eval_iiv_results_{Path(args.weights).stem}.json"
    else:
        output_file = "eval_iiv_results_base_model.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_file}")
    
    # Also save a summary-only version (without generations for smaller file)
    results_summary = {
        "dataset": results["dataset"],
        "num_items": results["num_items"],
        "num_items_no_hint": results["num_items_no_hint"],
        "num_items_with_hint": results["num_items_with_hint"],
        "num_samples": results["num_samples"],
        "metrics": results["metrics"],
        "config": results["config"],
    }
    summary_file = output_file.replace(".json", "_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

