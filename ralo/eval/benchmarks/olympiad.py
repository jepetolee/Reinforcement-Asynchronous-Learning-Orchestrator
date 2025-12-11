from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from datasets import load_dataset


class OlympiadBenchBenchmark:
    """
    OlympiadBench adapter for lmms-lab/OlympiadBench.

    We use:
      - question: combination of `context` (if present) and `question`
      - answer: first element of `final_answer` list (or stringified value)

    Split defaults to English test split: `test_en`.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg)
        # Allow overriding hf_dataset/split via config if needed
        self.hf_dataset = self.cfg.get("hf_dataset", "lmms-lab/OlympiadBench")
        self.split = self.cfg.get("split", "test_en")
        self.max_items: Optional[int] = self.cfg.get("max_items")
        self.prompt_template = self.cfg.get("prompt_template")  # optional, currently unused
        self.answer_extractor = self.cfg.get("answer_extractor")  # optional, currently unused

        self._ds = load_dataset(self.hf_dataset)[self.split]

    def load_items(self) -> Iterable[Dict[str, Any]]:
        for i, row in enumerate(self._ds):
            if self.max_items is not None and i >= int(self.max_items):
                break

            context = row.get("context") or ""
            question = row.get("question") or ""

            final_answer = row.get("final_answer")
            answer_str = ""
            if isinstance(final_answer, list):
                if final_answer:
                    answer_str = str(final_answer[0])
            elif final_answer is not None:
                answer_str = str(final_answer)

            yield {
                "context": context,
                "question": question,
                "answer": answer_str,
            }

    def make_prompt(self, item: Dict[str, Any]) -> str:
        """
        Build a pure-text prompt by concatenating context and question.
        """
        context = item.get("context") or ""
        question = item.get("question", "") or ""

        if context:
            prompt = f"{context}\n\n{question}"
        else:
            prompt = str(question)

        return prompt

    def reference(self, item: Dict[str, Any]) -> Any:
        ans = item.get("answer", "")
        return str(ans) if ans is not None else ""

    def extract_prediction(self, text: str) -> Any:
        # For now, treat full generated text as the answer (no boxed parsing here)
        return (text or "").strip()


