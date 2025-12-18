"""Reward functions for reinforcement learning.

This module contains reward functions used in RL training. Each reward function
takes (answer, item) as arguments and returns a scalar reward value.
"""

import re
from typing import Any, Dict

try:
    from math_verify import ExprExtractionConfig, parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False


def extract_boxed_answer(solution_text):
    """Extract answer from \\boxed{} pattern in solution text."""
    match = re.search(r"\\boxed\{(.*?)\}", solution_text)
    if match:
        return match.group(1)
    return None


def correct_fn(answer, item):
    """Standard correctness reward function.
    
    Uses math_verify if available for mathematical equivalence checking,
    otherwise falls back to string comparison.
    
    Args:
        answer: Generated answer string
        item: Problem item dictionary with 'A' key containing reference answer
    
    Returns:
        Reward: +1 if answer matches reference, -1 otherwise
    """
    if not MATH_VERIFY_AVAILABLE:
        # Fallback to simple string comparison
        gold_str = str(item.get("A", "")).strip()
        answer_str = str(answer).strip()
        return 1.0 if gold_str.lower() == answer_str.lower() else -1.0
    
    try:
        gold_parsed = parse(item["A"], extraction_config=[ExprExtractionConfig()])
        answer_parsed = parse(answer, extraction_config=[ExprExtractionConfig()])
        return 1.0 if verify(gold_parsed, answer_parsed) else -1.0
    except Exception:
        # Fallback to string comparison if parsing fails
        gold_str = str(item.get("A", "")).strip()
        answer_str = str(answer).strip()
        return 1.0 if gold_str.lower() == answer_str.lower() else -1.0


def format_fn(text, _, **kwargs):
    """Format checking reward function.
    
    Checks if text contains specific format markers (e.g., </think>).
    
    Args:
        text: Generated text
        _: Unused item argument
        **kwargs: Additional arguments
    
    Returns:
        Reward: Positive value if format is correct, 0.0 otherwise
    """
    count = 0.0
    if text.count("\n</think>\n") == 1:
        count += 1.0
    return count


def iiv_reward_fn(answer, item):
    """IIV (Is It Valid) reward function.
    
    IIV is a task where the model must determine if the given information (hint) is valid
    enough to answer the question. Special reward policy:
    
    If Hint is missing or empty (not valid):
        - If answer contains "I don't know" or "IDK": Reward 0.0 (no penalty, correct IDK)
        - If answer does NOT contain IDK: Reward -1.0 (penalty for answering without hint)
    
    If Hint is present and valid:
        - Reward +1.0 if answer matches the correct answer (using math_verify)
        - Reward -1.0 if answer does not match (penalty for wrong answer)
    
    This policy encourages models to say "I don't know" when no hint is available,
    and to answer correctly only when hint is provided.
    
    Args:
        answer: Generated answer string
        item: Problem item dictionary with 'A' (answer) and 'Hint' keys
    
    Returns:
        Reward: +1.0 (correct with hint), 0.0 (correct IDK), or -1.0 (incorrect/no IDK)
    """
    hint = item.get("Hint", "")
    if hint is None or str(hint).strip() == "":
        # No hint: should output "I don't know"
        # If IDK is expressed, no penalty (0.0 reward)
        # If IDK is NOT expressed, penalty (-1.0 reward)
        answer_lower = str(answer).lower().strip()
        idk_patterns = [
            "i don't know", "i don't know.", "i don't know,", 
            "idk", "idk.", "idk,", 
            "i do not know", "i do not know.", "i do not know,"
        ]
        for pattern in idk_patterns:
            if pattern in answer_lower:
                return 0.0  # No penalty for correct IDK
        return -1.0  # Penalty for answering without hint
    else:
        # Has hint: should output correct answer
        # Standard reward: +1 for correct, -1 for incorrect
        if not MATH_VERIFY_AVAILABLE:
            # Fallback to string comparison
            gold_str = str(item.get("A", "")).strip()
            answer_str = str(answer).strip()
            return 1.0 if gold_str.lower() == answer_str.lower() else -1.0
        
        try:
            gold_parsed = parse(item["A"], extraction_config=[ExprExtractionConfig()])
            answer_parsed = parse(answer, extraction_config=[ExprExtractionConfig()])
            return 1.0 if verify(gold_parsed, answer_parsed) else -1.0
        except Exception:
            # Fallback to string comparison if parsing fails
            gold_str = str(item.get("A", "")).strip()
            answer_str = str(answer).strip()
            return 1.0 if gold_str.lower() == answer_str.lower() else -1.0


# Default reward functions (used if not specified in config)
default_reward_fns = [correct_fn]

