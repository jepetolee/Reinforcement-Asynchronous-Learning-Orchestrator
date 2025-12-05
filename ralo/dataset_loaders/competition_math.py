"""Loader for competition_math dataset."""

import re
from typing import List, Dict, Any, Optional

from datasets import load_dataset

from .base import BaseDatasetLoader


def extract_boxed_answer(solution_text: str) -> Optional[str]:
    """Extract answer from \\boxed{} in solution text."""
    if not solution_text:
        return None
    match = re.search(r"\\boxed\{(.*?)\}", solution_text)
    if match:
        return match.group(1)
    return None


class CompetitionMathLoader(BaseDatasetLoader):
    """Loader for qwedsacf/competition_math dataset."""
    
    def load(self) -> List[Dict[str, str]]:
        """Load competition_math dataset and transform to QA format."""
        from datasets import load_dataset
        
        dataset = load_dataset(self.name)[self.split]
        
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
        """Extract Q and A from competition_math example.
        
        Format:
        - problem: The problem statement
        - solution: The solution text (contains \\boxed{answer})
        - level: Difficulty level (optional, used for filtering)
        """
        problem = example.get("problem")
        solution = example.get("solution")
        
        if problem is None:
            return None
        
        # Extract answer from boxed solution
        answer = extract_boxed_answer(solution)
        if answer is None:
            return None
        
        return {"Q": problem, "A": answer}

