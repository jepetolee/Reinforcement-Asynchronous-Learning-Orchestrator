from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


def _normalize_text(s: str) -> str:
    s = s.strip()
    # remove surrounding quotes and spaces
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(preds: List[Any], refs: List[Any], **kw) -> Dict[str, float]:
    """
    Simple exact match over stringified preds/refs.
    """
    total = max(1, len(refs))
    correct = 0
    for p, r in zip(preds, refs):
        ps = _normalize_text(str(p))
        rs = _normalize_text(str(r))
        if ps == rs:
            correct += 1
    return {"accuracy@1": correct / total}


def accuracy_at_k(preds: List[Any], refs: List[Any], candidates: Optional[List[List[Any]]] = None, k: int = 1, **kw) -> Dict[str, float]:
    """
    Accuracy@K where candidates[i] is a list of predictions for item i.
    """
    if candidates is None:
        # fallback: treat preds as single candidate per item
        candidates = [[p] for p in preds]
    total = max(1, len(refs))
    correct = 0
    for i, r in enumerate(refs):
        cand = candidates[i] if i < len(candidates) else []
        cand_k = cand[:k]
        rs = _normalize_text(str(r))
        matched = False
        for p in cand_k:
            ps = _normalize_text(str(p))
            if ps == rs:
                matched = True
                break
        if matched:
            correct += 1
    return {f"accuracy@{k}": correct / total}


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


