from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


class GPQABenchmark:
    """
    GPQA adapter (Idavidrein/gpqa), supports 'lite'/'diamond' via config.
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)
        self.split = self.cfg.get("split", "validation")
        self.hf_dataset = self.cfg.get("hf_dataset") or "Idavidrein/gpqa"
        self.hf_config = self.cfg.get("config")  # "diamond" or "lite" etc.
        self.max_items: Optional[int] = self.cfg.get("max_items")
        self.prompt_template = self.cfg.get("prompt_template")  # optional
        self.answer_extractor = self.cfg.get("answer_extractor")  # optional

        if self.hf_config:
            self._ds = load_dataset(self.hf_dataset, self.hf_config)[self.split]
        else:
            self._ds = load_dataset(self.hf_dataset)[self.split]

    def load_items(self) -> Iterable[Dict[str, Any]]:
        for i, row in enumerate(self._ds):
            if self.max_items is not None and i >= int(self.max_items):
                break
            # GPQA schema: question, answer, choices/options (varies)
            question = row.get("question") or row.get("prompt") or row.get("problem")
            options = row.get("options") or row.get("choices")
            # 'answer' might be text or index; also consider 'correct' key
            gold = row.get("answer", row.get("correct"))
            yield {"question": question, "options": options, "answer": gold}

    def make_prompt(self, item: Dict[str, Any]) -> str:
        q = str(item.get("question", ""))
        opts = item.get("options") or []
        if isinstance(opts, list) and opts:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            formatted = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(opts)])
            return f"{q}\n\nOptions:\n{formatted}\n\nAnswer with the correct option letter only."
        return q

    def reference(self, item: Dict[str, Any]) -> Any:
        gold = item.get("answer")
        # If gold is index, convert to letter
        if isinstance(gold, int):
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            try:
                return letters[gold]
            except Exception:
                return str(gold)
        return str(gold) if gold is not None else ""

    def extract_prediction(self, text: str) -> Any:
        # Extract the first capital letter A-Z
        import re
        m = re.search(r"\b([A-Z])\b", text or "")
        if m:
            return m.group(1)
        # fallback
        return (text or "").strip()


