"""Generic HuggingFace dataset loader.

This loader attempts to auto-detect common field names in HuggingFace datasets.
"""

import re
from typing import List, Dict, Any, Optional

from datasets import load_dataset

from .base import BaseDatasetLoader


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} if present."""
    if not text:
        return None
    match = re.search(r"\\boxed\{(.*?)\}", text)
    if match:
        return match.group(1)
    return None


class GenericHFDatasetLoader(BaseDatasetLoader):
    """Generic loader that auto-detects common field names."""
    
    # Common field name patterns to try
    QUESTION_FIELDS = ["problem", "question", "prompt", "input", "instruction", "task"]
    ANSWER_FIELDS = ["solution", "answer", "output", "response", "target"]
    LEVEL_FIELDS = ["level", "difficulty", "grade", "complexity"]
    
    def load(self) -> List[Dict[str, str]]:
        """Load dataset and auto-detect field names."""
        from datasets import load_dataset
        
        dataset = load_dataset(self.name)[self.split]
        
        # Detect field names from first example
        if len(dataset) == 0:
            return []
        
        first_example = dataset[0]
        question_field = self._detect_field(first_example, self.QUESTION_FIELDS)
        answer_field = self._detect_field(first_example, self.ANSWER_FIELDS)
        level_field = self._detect_field(first_example, self.LEVEL_FIELDS)
        
        if question_field is None or answer_field is None:
            raise ValueError(
                f"Could not auto-detect question/answer fields in dataset '{self.name}'. "
                f"Available fields: {list(first_example.keys())}. "
                f"Please use a specific loader or specify field mappings in config."
            )
        
        self._question_field = question_field
        self._answer_field = answer_field
        self._level_field = level_field
        
        QAs = []
        for example in dataset:
            # Apply user-provided filter if specified
            if not self.should_include_example(example):
                continue
            qa = self.extract_qa(example)
            if qa is not None:
                QAs.append(qa)
        
        # Shuffle if seed is provided
        if self.shuffle_seed is not None:
            import random
            random.seed(self.shuffle_seed)
            random.shuffle(QAs)
        
        return QAs
    
    def extract_qa(self, example: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract Q and A using detected field names."""
        question = example.get(self._question_field)
        answer_raw = example.get(self._answer_field)
        
        if question is None or answer_raw is None:
            return None
        
        # Try to extract boxed answer, otherwise use raw answer
        if isinstance(answer_raw, str):
            answer = extract_boxed_answer(answer_raw) or answer_raw.strip()
        else:
            answer = str(answer_raw).strip()
        
        if not answer:
            return None
        
        return {"Q": question, "A": answer}
    
    def _detect_field(self, example: Dict[str, Any], candidates: List[str]) -> Optional[str]:
        """Detect which field name exists in the example."""
        for candidate in candidates:
            if candidate in example:
                return candidate
        return None

