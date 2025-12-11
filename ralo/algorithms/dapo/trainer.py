"""
DAPO Trainer Algorithm: Decoupled Clip and Dynamic sAmpling Policy Optimization.
"""

from typing import Any, Dict, List
import torch

from ...utils import pad_lists
from ..registry import TrainerAlgorithm, register_trainer_algorithm


@register_trainer_algorithm("dapo")
class DAPOTrainerAlgorithm(TrainerAlgorithm):
    """DAPO trainer algorithm implementation."""

    def run(self) -> None:
        """Run the trainer loop.
        
        Reuse the standard training loop implemented on the RALO class,
        which handles fetching batches from the orchestrator and applying
        optimizer steps. DAPO-specific loss is computed via compute_loss().
        """
        # Reuse RLVR training loop infrastructure
        self.ralo._rlvr_run_trainer()

    def compute_loss(self, model, batch: Dict[str, Any], services: Dict[str, Any]) -> torch.Tensor:
        """
        Compute DAPO loss from a batch.
        Implements Clip-Higher and Token-Level Policy Gradient.

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
        
        # Get DAPO config
        dapo_config = getattr(self.ralo, "DAPO_kwargs", {})
        clip_param = self.ralo.clip_param
        # Clip-Higher parameter: default higher than standard clip_param (0.2)
        # Paper suggests raising the ceiling.
        clip_param_high = dapo_config.get("clip_param_high", 0.3) 

        prompt_length = batch["plen"]
        inputs = batch["inputs"].to(device)
        # In DAPO sampler, we stored advantages in "rewards" key
        advantages = batch["rewards"].to(device).unsqueeze(1)

        # Forward pass
        if "#computed_logits" not in batch:
            # If logits not pre-computed (e.g. by PPO rollup)
            logits = model(inputs, use_cache=False).logits
            logits = logits[:, :-1, :]
        else:
            logits = batch["#computed_logits"].to(device)

        input_ids = inputs[:, 1:]
        
        # Get per-token log probabilities
        per_token_logps = training_service.get_per_token_logps(logits, input_ids)
        
        # Slice to get only response part
        # Align with generated logprobs
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        
        # Mask padding
        # inputs original shape includes prompt. 
        # completion part starts at prompt_length
        completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
        
        # Ensure shapes match (handle potential off-by-one or padding issues)
        min_len = min(per_token_logps.shape[1], completion_mask.shape[1])
        per_token_logps = per_token_logps[:, :min_len]
        completion_mask = completion_mask[:, :min_len]

        # Old logprobs from sampling phase
        if "gen_logps" in batch:
            old_logps = pad_lists(batch["gen_logps"], pad_value=0.0).to(device)
            # Ensure old_logps matches current slice
            if old_logps.shape[1] > min_len:
                old_logps = old_logps[:, :min_len]
            elif old_logps.shape[1] < min_len:
                # This case is tricky, means generated length < input length (maybe due to padding mismatch?)
                # We slice current logps to match old logps
                min_len = old_logps.shape[1]
                per_token_logps = per_token_logps[:, :min_len]
                completion_mask = completion_mask[:, :min_len]
            
            # Policy Ratio
            ratio = torch.exp(per_token_logps - old_logps)
            
            # Clip-Higher
            # Lower bound: 1 - epsilon
            # Upper bound: 1 + epsilon_high
            clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param_high)
            
            # PPO Loss
            per_token_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        else:
            raise Exception("DAPO requires gen_logps in batch")

        valid_tokens = completion_mask.sum()
        loss = (per_token_loss * completion_mask).sum() / torch.clamp(valid_tokens, min=1)
        
        return loss

    def prepare_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare a batch from a list of samples.
        """
        # Not implemented as batch preparation is handled by format_samples_for_upload
        # or standard collator in this framework version.
        raise NotImplementedError("DAPO trainer prepare_batch not yet fully implemented.")

