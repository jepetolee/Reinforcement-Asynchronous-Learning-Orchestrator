import json
import re
import argparse
from typing import List, Dict, Any
from pathlib import Path

def _is_idk(text: str) -> bool:
    """Check if text contains IDK pattern (enhanced)."""
    text_lower = str(text).lower().strip()
    
    # IDK patterns - expanded list
    idk_patterns = [
        "i don't know", "i do not know", 
        "idk", 
        "i'm not sure", "i am not sure",
        "i don't recall", "i do not recall",
        "i don't think i know", "i don't think i have",
        "i cannot answer", "i can't answer",
        "no information", "not enough information",
        "unsure", "uncertain"
    ]
    
    # Check for direct matches
    for pattern in idk_patterns:
        # Check if pattern exists as a phrase (simple check)
        if pattern in text_lower:
            return True
            
    return False

def _math_equal(pred: str, ref: str) -> bool:
    """Simple string equality check for evaluation."""
    # Normalize strings
    pred_norm = pred.strip().lower()
    ref_norm = ref.strip().lower()
    
    # Basic containment check for long generation vs short reference
    if len(ref_norm) > 0 and ref_norm in pred_norm:
        return True
        
    return pred_norm == ref_norm

def analyze_results(json_path: str):
    print(f"Analyzing {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    generations = data.get("generations", [])
    print(f"Loaded {len(generations)} items.")
    
    # Storage for metrics
    stats = {
        "no_hint": {"total_items": 0, "total_samples": 0, "idk_count": 0, "correct_count": 0, "attempted_count": 0, "any_idk_count": 0},
        "with_hint": {"total_items": 0, "total_samples": 0, "idk_count": 0, "correct_count": 0, "attempted_count": 0, "any_idk_count": 0},
        "overall": {"total_items": 0, "total_samples": 0, "idk_count": 0, "correct_count": 0, "attempted_count": 0, "any_idk_count": 0}
    }
    
    # Process each item
    for item in generations:
        hint = item.get("hint", "")
        samples = item.get("samples", [])
        
        # Determine category
        category = "no_hint" if not hint or str(hint).strip() == "" else "with_hint"
        
        # Update item counts
        stats[category]["total_items"] += 1
        stats["overall"]["total_items"] += 1
        
        # Process samples for this item
        item_idk_count = 0
        
        # Determine the correct reference
        # NOTE: For "with_hint" items, the reference in JSON might be the actual answer
        # But for "no_hint" items, the reference is usually "I don't know" or the answer depending on dataset
        # We need to know what the 'correct' answer is if they didn't say IDK.
        reference = item.get("reference", "")
        
        # Check each sample
        has_any_idk = False
        
        for sample in samples:
            is_idk = _is_idk(sample)
            is_correct = False
            
            if is_idk:
                item_idk_count += 1
                has_any_idk = True
                # Update stats
                stats[category]["idk_count"] += 1
                stats["overall"]["idk_count"] += 1
            else:
                # Not IDK - check if correct
                # We only check correctness if it's NOT IDK
                stats[category]["attempted_count"] += 1
                stats["overall"]["attempted_count"] += 1
                
                # Check correctness against reference
                # Warning: if reference is "I don't know", then correctness is weird here
                # But usually reference contains the ground truth answer
                if reference and reference.lower() != "i don't know":
                    if _math_equal(sample, reference):
                        is_correct = True
                        stats[category]["correct_count"] += 1
                        stats["overall"]["correct_count"] += 1
            
            stats[category]["total_samples"] += 1
            stats["overall"]["total_samples"] += 1
            
        if has_any_idk:
            stats[category]["any_idk_count"] += 1
            stats["overall"]["any_idk_count"] += 1

    # Print Report
    print("\n" + "="*60)
    print(f"ANALYSIS REPORT: {Path(json_path).name}")
    print("="*60)
    
    for cat in ["no_hint", "with_hint", "overall"]:
        s = stats[cat]
        if s["total_items"] == 0:
            continue
            
        print(f"\nCategory: {cat.upper()} ({s['total_items']} items)")
        
        # IDK Rate (Sample level)
        idk_rate = s["idk_count"] / s["total_samples"] if s["total_samples"] > 0 else 0
        print(f"  IDK Rate (per sample): {idk_rate:.4f} ({s['idk_count']}/{s['total_samples']})")
        
        # IDK@Any (Item level - at least one IDK in K samples)
        idk_any_rate = s["any_idk_count"] / s["total_items"]
        print(f"  IDK Coverage (Any in K): {idk_any_rate:.4f} ({s['any_idk_count']}/{s['total_items']})")
        
        # Accuracy on Attempted (Filtered)
        # This is accuracy ONLY on samples where the model did NOT say IDK
        acc_filtered = s["correct_count"] / s["attempted_count"] if s["attempted_count"] > 0 else 0
        print(f"  Accuracy (Filtered/Attempted): {acc_filtered:.4f} ({s['correct_count']}/{s['attempted_count']})")
        
        # Effective Accuracy (Correct / Total Samples)
        # This treats IDK as incorrect for the purpose of answering questions
        acc_effective = s["correct_count"] / s["total_samples"] if s["total_samples"] > 0 else 0
        print(f"  Effective Accuracy (All samples): {acc_effective:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str, help="Path to evaluation results JSON")
    args = parser.parse_args()
    
    analyze_results(args.json_file)

