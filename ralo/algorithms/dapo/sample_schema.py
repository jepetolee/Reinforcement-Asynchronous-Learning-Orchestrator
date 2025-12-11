"""
DAPOSampleSchema: Sample schema for DAPO algorithm.
"""

from typing import Any, Dict, List

from ..sample_schema import SampleSchema


class DAPOSampleSchema(SampleSchema):
    """Sample schema for DAPO algorithm."""

    REQUIRED_FIELDS = [
        "prompt",
        "token_ids",
        "gen_logps",
        "rewards",
        "advantage",
        "text",
        "current_answer",
    ]

    OPTIONAL_FIELDS = [
        "entropy",
        "mean_length",
        "finished_reason",
        "prompt_in_this_step", # Kept for compatibility if needed
    ]

    def __init__(self, sample: Dict[str, Any] = None):
        """
        Initialize DAPOSampleSchema.

        Args:
            sample: Optional sample dictionary to validate
        """
        if sample is not None:
            self.validate(sample)

    def validate(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that a sample matches DAPO schema.

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
            "type": "DAPOSampleSchema",
            "required_fields": self.REQUIRED_FIELDS,
            "optional_fields": self.OPTIONAL_FIELDS,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAPOSampleSchema":
        """
        Create DAPOSampleSchema from dictionary.

        Args:
            data: Dictionary representation (not used, but kept for interface compatibility)

        Returns:
            DAPOSampleSchema instance
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

