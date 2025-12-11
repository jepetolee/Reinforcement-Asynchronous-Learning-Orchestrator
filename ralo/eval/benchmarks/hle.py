from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from datasets import load_dataset


class BaseBenchmark:
    """
    Generic HF loader for QA-like tasks (user-configurable).
    Expects dataset with fields for question and answer; can be configured via cfg keys:
      - hf_dataset: string (required)
      - split: string (default: "test")
      - question_key: string (default: "question")
      - answer_key: string (default: "answer")
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)
        self.hf_dataset = self.cfg.get("hf_dataset")
        if not self.hf_dataset:
            raise ValueError("BaseBenchmark requires 'hf_dataset' in config")
        self.split = self.cfg.get("split", "test")
        self.max_items: Optional[int] = self.cfg.get("max_items")

        self.question_key = self.cfg.get("question_key", "question")
        self.answer_key = self.cfg.get("answer_key", "answer")
        self.prompt_template = self.cfg.get("prompt_template")  # optional
        self.answer_extractor = self.cfg.get("answer_extractor")  # optional

        self._ds = load_dataset(self.hf_dataset)[self.split]

    def load_items(self) -> Iterable[Dict[str, Any]]:
        for i, row in enumerate(self._ds):
            if self.max_items is not None and i >= int(self.max_items):
                break
            q = row.get(self.question_key)
            a = row.get(self.answer_key)
            yield {"question": q, "answer": a}

    def make_prompt(self, item: Dict[str, Any]) -> str:
        if self.prompt_template:
            # allow custom prompt via callable path later (registry can resolve)
            return str(item.get("question", ""))
        return str(item.get("question", ""))

    def reference(self, item: Dict[str, Any]) -> Any:
        return str(item.get("answer", ""))

    def extract_prediction(self, text: str) -> Any:
        return (text or "").strip()


