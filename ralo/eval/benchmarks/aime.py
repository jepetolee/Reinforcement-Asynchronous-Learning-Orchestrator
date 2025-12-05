from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


class AIMEBenchmark:
    """
    AIME 2024/2025 adapter (HuggingFaceH4/aime_20xx).
    """
    def __init__(self, cfg: Dict[str, Any], dataset: str = "aime_2024"):
        self.cfg = dict(cfg)
        self.dataset = dataset  # "aime_2024" or "aime_2025"
        # AIME datasets typically only have 'train' split, use it as default
        self.split = self.cfg.get("split", "train")
        self.max_items: Optional[int] = self.cfg.get("max_items")
        self.prompt_template = self.cfg.get("prompt_template") or "builtin:aime_cot"
        self.answer_extractor = self.cfg.get("answer_extractor") or "builtin:boxed"

        # Load dataset and check available splits
        dataset_dict = load_dataset("HuggingFaceH4/" + self.dataset)
        available_splits = list(dataset_dict.keys())
        
        # If requested split doesn't exist, try common alternatives
        # For AIME, prioritize 'train' since that's typically the only available split
        if self.split not in available_splits:
            # Try common split names in order of preference (train first for AIME)
            for fallback_split in ["train", "validation", "val", "dev", "test"]:
                if fallback_split in available_splits:
                    print(f"[AIMEBenchmark] Split '{self.split}' not found, using '{fallback_split}' instead. Available splits: {available_splits}")
                    self.split = fallback_split
                    break
            else:
                # If no fallback found, use the first available split
                if available_splits:
                    print(f"[AIMEBenchmark] Split '{self.split}' not found, using '{available_splits[0]}' instead. Available splits: {available_splits}")
                    self.split = available_splits[0]
                else:
                    raise ValueError(f"No splits available in dataset HuggingFaceH4/{self.dataset}")
        
        self._ds = dataset_dict[self.split]

    def load_items(self) -> Iterable[Dict[str, Any]]:
        for i, row in enumerate(self._ds):
            if self.max_items is not None and i >= int(self.max_items):
                break
            # Try multiple field names to be robust
            problem = row.get("problem") or row.get("question") or row.get("prompt")
            # Some datasets use 'answer' or 'solution'
            answer = row.get("answer") or row.get("solution")
            yield {"problem": problem, "answer": answer}

    def make_prompt(self, item: Dict[str, Any]) -> str:
        question = item.get("problem", "")
        if self.prompt_template == "builtin:aime_cot":
            return (
                "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
                f"Problem:\n{question}\n\nSolution:"
            )
        # fallback: no special formatting
        return str(question)

    def reference(self, item: Dict[str, Any]) -> Any:
        ref = item.get("answer", "")
        if not ref:
            return ""
        # If answer contains boxed form, extract content
        boxed = _extract_boxed_answer(str(ref))
        return boxed if boxed is not None else str(ref).strip()

    def extract_prediction(self, text: str) -> Any:
        boxed = _extract_boxed_answer(text or "")
        return boxed if boxed is not None else (text or "").strip()


def _extract_boxed_answer(solution_text: str) -> Optional[str]:
    import re
    match = re.search(r"\\boxed\{(.*?)\}", solution_text)
    if match:
        return match.group(1)
    return None


