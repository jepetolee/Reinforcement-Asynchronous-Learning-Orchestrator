from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


def _normalize_text(s: str) -> str:
    s = s.strip()
    # remove surrounding quotes and spaces
    s = re.sub(r"\s+", " ", s)
    return s


def _math_equal(pred: str, ref: str) -> bool:
    """
    Check if two mathematical expressions are equal using math_verify if available,
    otherwise fall back to normalized string comparison.
    """
    try:
        from math_verify import ExprExtractionConfig, parse, verify
        try:
            pred_parsed = parse(pred, extraction_config=[ExprExtractionConfig()])
            ref_parsed = parse(ref, extraction_config=[ExprExtractionConfig()])
            return verify(ref_parsed, pred_parsed)
        except Exception:
            # If parsing fails, fall back to string comparison
            pass
    except ImportError:
        # math_verify not available, use string comparison
        pass
    # Fallback to normalized string comparison
    return _normalize_text(pred) == _normalize_text(ref)


def exact_match(preds: List[Any], refs: List[Any], **kw) -> Dict[str, float]:
    """
    Exact match over stringified preds/refs.
    For mathematical problems, uses math_verify if available for proper comparison.
    """
    total = max(1, len(refs))
    correct = 0
    for p, r in zip(preds, refs):
        if _math_equal(str(p), str(r)):
            correct += 1
    return {"accuracy@1": correct / total}


def pass_at_k(preds: List[Any], refs: List[Any], candidates: Optional[List[List[Any]]] = None, k: int = 1, **kw) -> Dict[str, float]:
    """
    Pass@K: for each item, 1 if any of the first K candidates matches the reference, else 0.
    This corresponds to the usual "at least one success in K attempts" notion.
    For mathematical problems, uses math_verify if available for proper comparison.
    """
    if candidates is None:
        # fallback: treat preds as single candidate per item
        candidates = [[p] for p in preds]
    total = max(1, len(refs))
    passed = 0
    for i, r in enumerate(refs):
        cand = candidates[i] if i < len(candidates) else []
        cand_k = cand[:k]
        matched = False
        for p in cand_k:
            if _math_equal(str(p), str(r)):
                matched = True
                break
        if matched:
            passed += 1
    return {f"pass@{k}": passed / total}


def accuracy_at_k(preds: List[Any], refs: List[Any], candidates: Optional[List[List[Any]]] = None, k: int = 1, **kw) -> Dict[str, float]:
    """
    Accuracy@K: per-item average correctness over K candidates.
    For each item i, we compute (# correct among first K candidates) / K,
    then average this value over all items.
    For mathematical problems, uses math_verify if available for proper comparison.
    """
    if candidates is None:
        candidates = [[p] for p in preds]
    total_items = max(1, len(refs))
    total_fraction = 0.0
    for i, r in enumerate(refs):
        cand = candidates[i] if i < len(candidates) else []
        cand_k = cand[:k]
        correct_count = 0
        for p in cand_k:
            if _math_equal(str(p), str(r)):
                correct_count += 1
        # Divide by K (not len(cand_k)) so that missing candidates count as incorrect
        per_item_acc = correct_count / float(k) if k > 0 else 0.0
        total_fraction += per_item_acc
    return {f"accuracy@{k}": total_fraction / total_items}


def mc_accuracy_at_k(preds: List[Any], refs: List[Any], candidates: Optional[List[List[Any]]] = None, k: int = 1, **kw) -> Dict[str, float]:
    """
    Multiple-choice accuracy@K. We treat refs as letters A-Z and predictions likewise.
    """
    if candidates is None:
        candidates = [[p] for p in preds]
    total = max(1, len(refs))
    correct = 0
    for i, r in enumerate(refs):
        cand = candidates[i] if i < len(candidates) else []
        cand_k = cand[:k]
        r_letter = _extract_option_letter(str(r))
        matched = False
        for p in cand_k:
            p_letter = _extract_option_letter(str(p))
            if p_letter and r_letter and p_letter == r_letter:
                matched = True
                break
        if matched:
            correct += 1
    return {f"mc_accuracy@{k}": correct / total}


def _extract_option_letter(s: str) -> Optional[str]:
    m = re.search(r"\b([A-Z])\b", s.strip())
    return m.group(1) if m else None


