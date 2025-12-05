"""
SampleSchema: Base class for defining sample structures.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class SampleSchema(ABC):
    """Base class for sample schema definitions."""

    @abstractmethod
    def validate(self, sample: Dict[str, Any]) -> bool:
        """
        Validate that a sample matches the schema.

        Args:
            sample: Sample dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert schema to dictionary representation.

        Returns:
            Dictionary representation of schema
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleSchema":
        """
        Create schema instance from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            SampleSchema instance
        """
        pass

    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """
        Get list of required field names.

        Returns:
            List of required field names
        """
        pass

    @abstractmethod
    def get_optional_fields(self) -> List[str]:
        """
        Get list of optional field names.

        Returns:
            List of optional field names
        """
        pass

