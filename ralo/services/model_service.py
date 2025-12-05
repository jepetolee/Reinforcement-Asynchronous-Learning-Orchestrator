"""
ModelService: Handles model loading, weight synchronization, and tokenizer management.
"""

import io
import os
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils import convert_lm_head_to_fp32


class ModelService:
    """Service for model loading, weight synchronization, and tokenizer management."""

    def __init__(
        self,
        model_path: str,
        model_id: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize ModelService.

        Args:
            model_path: Path to the model
            model_id: Model identifier (defaults to basename of model_path)
            device: Device to load model on (defaults to 'cuda' if available)
        """
        self.model_path = model_path
        if model_id is None:
            base_id = os.path.basename(str(model_path)).strip() or "default"
            self.model_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", base_id)
        else:
            self.model_id = model_id

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._last_synced_version: Optional[int] = None

    def load_model(self, dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
        """
        Load model from model_path.

        Args:
            dtype: Data type for model weights

        Returns:
            Loaded model
        """
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model = self.model.to(dtype=dtype)
            convert_lm_head_to_fp32(self.model)
            self.model.train()
            self.model.requires_grad_(True)
            self.model.to(self.device)
        return self.model

    def load_tokenizer(self) -> AutoTokenizer:
        """
        Load tokenizer from model_path.

        Returns:
            Loaded tokenizer
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self.tokenizer

    def get_model(self) -> torch.nn.Module:
        """Get the loaded model, loading it if necessary."""
        if self.model is None:
            self.load_model()
        return self.model

    def get_tokenizer(self) -> AutoTokenizer:
        """Get the loaded tokenizer, loading it if necessary."""
        if self.tokenizer is None:
            self.load_tokenizer()
        return self.tokenizer

    def load_weights_from_bytes(self, state_bytes: bytes, strict: bool = False) -> bool:
        """
        Load model weights from bytes.

        Args:
            state_bytes: Serialized state dict bytes
            strict: Whether to strictly enforce that the keys match

        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            return False

        buffer = io.BytesIO(state_bytes)
        try:
            state_dict = torch.load(buffer, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=strict)
            self.model.to(self.device)
            return True
        except Exception as exc:
            print(f"[ModelService] Failed to load weights: {exc}")
            return False

    def get_state_dict(self) -> dict:
        """
        Get current model state dict.

        Returns:
            State dict with CPU tensors
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        state_dict = self.model.state_dict()
        return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}

    def set_last_synced_version(self, version: int):
        """Set the last synced weight version."""
        self._last_synced_version = version

    def get_last_synced_version(self) -> Optional[int]:
        """Get the last synced weight version."""
        return self._last_synced_version

