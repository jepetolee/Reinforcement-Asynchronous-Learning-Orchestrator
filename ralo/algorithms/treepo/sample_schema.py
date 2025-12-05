"""
TreePOSampleSchema: Sample schema for TreePO algorithm.
"""

from typing import Any, Dict, List

from ..sample_schema import SampleSchema


class TreePOSampleSchema(SampleSchema):
    """Sample schema for TreePO algorithm."""

    REQUIRED_FIELDS = [
        "prompt",
        "token_ids",
        "gen_logps",
        "rewards",
        "advantage",
        "entropy",
        "mean_length",
        "text",
        "current_answer",
        "depth",
    ]

    OPTIONAL_FIELDS = [
        "main_prompt",
        "prompt_in_this_step",
        "finished_reason",
        "entropy_in_this_step",
        "mean_entropy",
    ]

    def __init__(self, sample: Dict[str, Any] = None):
        """
        Initialize TreePOSampleSchema.

        Args:
            sample: Optional sample dictionary to validate
        """
        if sample is not None:
            self.validate(sample)

    def validate(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that a sample matches TreePO schema.

        Args:
            sample: Sample dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(sample, dict):
            return False

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in sample:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert schema to dictionary representation.

        Returns:
            Dictionary with schema metadata
        """
        return {
            "type": "TreePOSampleSchema",
            "required_fields": self.REQUIRED_FIELDS,
            "optional_fields": self.OPTIONAL_FIELDS,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreePOSampleSchema":
        """
        Create TreePOSampleSchema from dictionary.

        Args:
            data: Dictionary representation (not used, but kept for interface compatibility)

        Returns:
            TreePOSampleSchema instance
        """
        return cls()

    def get_required_fields(self) -> List[str]:
        """
        Get list of required field names.

        Returns:
            List of required field names
        """
        return self.REQUIRED_FIELDS.copy()

    def get_optional_fields(self) -> List[str]:
        """
        Get list of optional field names.

        Returns:
            List of optional field names
        """
        return self.OPTIONAL_FIELDS.copy()

    @staticmethod
    def create_sample(
        prompt: str,
        token_ids: List[int],
        gen_logps: List[float],
        rewards: Dict[str, float],
        advantage: float,
        entropy: float,
        mean_length: int,
        text: str,
        current_answer: str,
        depth: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a TreePO sample dictionary.

        Args:
            prompt: Main prompt
            token_ids: List of token IDs
            gen_logps: List of generation log probabilities
            rewards: Dictionary of rewards
            advantage: Advantage value
            entropy: Entropy value
            mean_length: Mean length
            text: Generated text
            current_answer: Current answer
            depth: Tree depth
            **kwargs: Additional optional fields

        Returns:
            Sample dictionary
        """
        sample = {
            "prompt": prompt,
            "token_ids": token_ids,
            "gen_logps": gen_logps,
            "rewards": rewards,
            "advantage": advantage,
            "entropy": entropy,
            "mean_length": mean_length,
            "text": text,
            "current_answer": current_answer,
            "depth": depth,
        }
        sample.update(kwargs)
        return sample

