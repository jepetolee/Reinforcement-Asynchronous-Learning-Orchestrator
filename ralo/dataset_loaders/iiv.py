"""IIV (Is It Valid) dataset loader.

This loader handles datasets with Problem, Answer, and Hint fields.
IIV is a task where the model must determine if the given information (hint) is valid
enough to answer the question. If Hint is missing or empty, the model should output
"I don't know". If Hint is present and valid, the model should output the correct answer.
"""

from typing import List, Dict, Any, Optional

from datasets import load_dataset

from .base import BaseDatasetLoader


class IIVDatasetLoader(BaseDatasetLoader):
    """Loader for IIV testing datasets with Problem, Answer, Hint fields."""
    
    def load(self) -> List[Dict[str, str]]:
        """Load dataset with Problem, Answer, Hint fields."""
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
        """Extract Q, A, and Hint from example.
        
        Returns:
            Dictionary with 'Q', 'A', and 'Hint' keys.
            'Hint' will be None or empty string if not present.
        """
        problem = example.get("Problem") or example.get("problem") or example.get("question")
        answer = example.get("Answer") or example.get("answer")
        hint = example.get("Hint") or example.get("hint") or ""
        
        if problem is None or answer is None:
            return None
        
        # Normalize hint: convert None to empty string
        if hint is None:
            hint = ""
        else:
            hint = str(hint).strip()
        
        return {
            "Q": str(problem).strip(),
            "A": str(answer).strip(),
            "Hint": hint,  # Empty string if no hint
        }

