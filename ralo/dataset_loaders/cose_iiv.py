"""cos_e dataset loader for IIV (Is It Valid) task.

This loader handles Salesforce/cos_e dataset with:
- id, question, choices, answer, abstractive_explanation, extractive_explanation

For IIV task, it converts to Q, A, Hint format where:
- Q: question with choices formatted
- A: answer
- Hint: abstractive_explanation or extractive_explanation (can be empty for no-hint cases)
"""

from typing import List, Dict, Any, Optional
import random

from datasets import load_dataset

from .base import BaseDatasetLoader


class CosEIIVLoader(BaseDatasetLoader):
    """Loader for cos_e dataset in IIV format."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the loader.
        
        Config options:
            - name: dataset name (default: "Salesforce/cos_e")
            - split: dataset split (default: "train")
            - hint_field: which explanation to use as hint ("abstractive_explanation", "extractive_explanation", or "both")
            - no_hint_ratio: ratio of examples without hints (0.0-1.0, default: 0.5)
            - use_extractive_if_available: if True, prefer extractive_explanation when both are available
        """
        super().__init__(config)
        self.hint_field = config.get("hint_field", "abstractive_explanation")
        self.no_hint_ratio = float(config.get("no_hint_ratio", 0.5))
        self.use_extractive_if_available = config.get("use_extractive_if_available", False)
        self.random_seed = config.get("shuffle_seed")  # Use shuffle_seed for hint randomization too
    
    def load(self) -> List[Dict[str, str]]:
        """Load dataset and convert to IIV format."""
        # cos_e requires a config name (v1.0 or v1.11)
        dataset_config = self.config.get("dataset_config", "v1.11")
        dataset = load_dataset(self.name, dataset_config)[self.split]
        
        # Get max_items limit
        max_items = self.config.get("max_items")
        
        # Set random seed for hint randomization
        if self.random_seed is not None:
            random.seed(self.random_seed)
        
        QAs = []
        for example in dataset:
            # Apply user-provided filter if specified
            if not self.should_include_example(example):
                continue
            qa = self.extract_qa(example)
            if qa is not None:
                QAs.append(qa)
                # Stop if max_items limit reached
                if max_items is not None and len(QAs) >= max_items:
                    break
        
        # Shuffle if seed is provided
        if self.shuffle_seed is not None:
            random.seed(self.shuffle_seed)
            random.shuffle(QAs)
        
        return QAs
    
    def extract_qa(self, example: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract Q, A, and Hint from cos_e example.
        
        Returns:
            Dictionary with 'Q', 'A', and 'Hint' keys.
            'Hint' will be empty string if no hint should be provided (based on no_hint_ratio).
        """
        question = example.get("question", "")
        choices = example.get("choices", [])
        answer = example.get("answer", "")
        abstractive = example.get("abstractive_explanation", "")
        extractive = example.get("extractive_explanation", "")
        
        if not question or not answer:
            return None
        
        # Format question with choices
        if choices and isinstance(choices, list):
            choices_text = "\n".join([f"- {choice}" for choice in choices])
            formatted_question = f"{question}\n\nChoices:\n{choices_text}"
        else:
            formatted_question = question
        
        # Determine hint based on configuration
        hint = ""
        
        # Randomly decide if this example should have a hint (based on no_hint_ratio)
        should_have_hint = random.random() >= self.no_hint_ratio
        
        if should_have_hint:
            # Choose which explanation to use as hint
            if self.hint_field == "both":
                # Prefer extractive if available and configured, otherwise use abstractive
                if self.use_extractive_if_available and extractive:
                    hint = str(extractive).strip()
                elif abstractive:
                    hint = str(abstractive).strip()
                elif extractive:
                    hint = str(extractive).strip()
            elif self.hint_field == "extractive_explanation" and extractive:
                hint = str(extractive).strip()
            elif self.hint_field == "abstractive_explanation" and abstractive:
                hint = str(abstractive).strip()
            else:
                # Fallback: use whichever is available
                if abstractive:
                    hint = str(abstractive).strip()
                elif extractive:
                    hint = str(extractive).strip()
        
        # Normalize hint
        if hint is None:
            hint = ""
        
        return {
            "Q": formatted_question.strip(),
            "A": str(answer).strip(),
            "Hint": hint,  # Empty string if no hint
        }

