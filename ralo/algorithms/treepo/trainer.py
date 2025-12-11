"""
TreePO Trainer Algorithm: Tree-based Policy Optimization training.
"""

from typing import Any, Dict, List

import torch

from ...utils import pad_lists
from ..registry import TrainerAlgorithm, register_trainer_algorithm


@register_trainer_algorithm("treepo")
class TreePOTrainerAlgorithm(TrainerAlgorithm):
    """TreePO trainer algorithm implementation."""

    def run(self) -> None:
        """Run the trainer (legacy method)."""
        self.ralo._rlvr_run_trainer()

    def compute_loss(self, model, batch: Dict[str, Any], services: Dict[str, Any]) -> torch.Tensor:
        """
        Compute TreePO loss from a batch.

        Args:
            model: Model instance
            batch: Batch dictionary
            services: Dictionary with 'model', 'training', 'orchestrator' services

        Returns:
            Loss tensor
        """
        training_service = services["training"]
        device = training_service.get_device()
        tokenizer = self.ralo.tokenizer

        prompt_length = batch["plen"]
        inputs = batch["inputs"].to(device)
        advantages = batch["rewards"].to(device).unsqueeze(1)

        # Forward pass
        if "#computed_logits" not in batch:
            logits = model(inputs, use_cache=False).logits
            logits = logits[:, :-1, :]
        else:
            logits = batch["#computed_logits"].to(device)

        input_ids = inputs[:, 1:]
        per_token_logps = training_service.get_per_token_logps(logits, input_ids)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

        # TreePO loss computation
        if "gen_logps" in batch:
            clip_param = self.ralo.clip_param
            clip_param_high = self.ralo.TreePO_kwargs.get("clip_param_high", 0.28)
            ratio = torch.exp(per_token_logps - pad_lists(batch["gen_logps"], pad_value=0.0).to(device))
            clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param_high)
            per_token_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        else:
            raise Exception("TreePO requires gen_logps in batch")

        valid_tokens = completion_mask.sum()
        loss = (per_token_loss * completion_mask).sum() / torch.clamp(valid_tokens, min=1)
        return loss

    def prepare_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare a batch from a list of samples.

        Args:
            samples: List of sample dictionaries

        Returns:
            Batch dictionary
        """
        # This is a placeholder - full implementation will be added in Phase 4
        raise NotImplementedError("TreePO trainer prepare_batch not yet fully implemented.")

