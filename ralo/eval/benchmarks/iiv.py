"""IIV (Is It Valid) benchmark for evaluation.

This benchmark handles datasets with Problem, Answer, and Hint fields.
IIV is a task where the model must determine if the given information (hint) is valid
enough to answer the question. During evaluation, if Hint is missing, the model should
output "I don't know". If Hint is present and valid, the model should output the correct answer.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from datasets import load_dataset


class IIVBenchmark:
    """
    IIV benchmark loader for datasets with Problem, Answer, Hint fields.
    
    Config keys:
      - hf_dataset: string (required)
      - split: string (default: "test")
      - question_key: string (default: "Problem")
      - answer_key: string (default: "Answer")
      - hint_key: string (default: "Hint")
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)
        self.hf_dataset = self.cfg.get("hf_dataset")
        if not self.hf_dataset:
            raise ValueError("IIVBenchmark requires 'hf_dataset' in config")
        self.split = self.cfg.get("split", "test")
        self.max_items: Optional[int] = self.cfg.get("max_items")
        
        self.question_key = self.cfg.get("question_key", "Problem")
        self.answer_key = self.cfg.get("answer_key", "Answer")
        self.hint_key = self.cfg.get("hint_key", "Hint")
        self.prompt_template = self.cfg.get("prompt_template")  # optional
        self.answer_extractor = self.cfg.get("answer_extractor")  # optional
        
        self._ds = load_dataset(self.hf_dataset)[self.split]
    
    def load_items(self) -> Iterable[Dict[str, Any]]:
        """Load items with Problem, Answer, and Hint fields."""
        for i, row in enumerate(self._ds):
            if self.max_items is not None and i >= int(self.max_items):
                break
            q = row.get(self.question_key)
            a = row.get(self.answer_key)
            hint = row.get(self.hint_key, "")
            if hint is None:
                hint = ""
            yield {"question": q, "answer": a, "hint": str(hint).strip(), "row": row}
    
    def make_prompt(self, item: Dict[str, Any]) -> str:
        """Create prompt based on hint presence.
        
        If hint is missing/empty:
            - Returns question with instruction to say "I don't know"
        If hint is present:
            - Returns question with hint included
        """
        question = item.get("question", "")
        hint = item.get("hint", "")
        
        if not hint or hint.strip() == "":
            # No hint: instruct to say "I don't know"
            return f"{question}\n\n(Note: You do not have enough information to answer this question. If you cannot answer with certainty, respond with 'I don't know'.)"
        else:
            # Has hint: include hint
            return f"Question: {question}\n\nHint: {hint}"
    
    def reference(self, item: Dict[str, Any]) -> Any:
        """Return reference answer.
        
        For IIV, if hint is missing, reference is "I don't know".
        Otherwise, reference is the actual answer.
        """
        hint = item.get("hint", "")
        if not hint or hint.strip() == "":
            return "I don't know"
        else:
            return str(item.get("answer", ""))
    
    def extract_prediction(self, text: str) -> Any:
        """Extract prediction from generated text.
        
        For IIV, we look for "I don't know" patterns or extract the answer directly.
        """
        text = (text or "").strip().lower()
        idk_patterns = ["i don't know", "i don't know.", "i don't know,", "idk", "idk.", "idk,",
                        "i do not know", "i do not know.", "i do not know,"]
        for pattern in idk_patterns:
            if pattern in text:
                return "I don't know"
        # Otherwise, return the text as-is (answer extraction can be done by metrics)
        return (text or "").strip()

