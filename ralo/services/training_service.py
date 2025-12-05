"""
TrainingService: Handles trainer initialization, gradient collection, and training loop.
"""

import io
import os
import socket
import time
import uuid
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from typing import TYPE_CHECKING

from ..utils import encode_gradients

if TYPE_CHECKING:
    from ..ralo import CPUOffloadTrainer


class TrainingService:
    """Service for trainer management, gradient collection, and training operations."""

    def __init__(
        self,
        model_path: str,
        lr: float = 1e-6,
        accum_steps: int = 16,
        grad_offload: bool = False,
        gradient_checkpointing_ratio: float = 1.0,
        init_optimizer: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize TrainingService.

        Args:
            model_path: Path to the model
            lr: Learning rate
            accum_steps: Gradient accumulation steps
            grad_offload: Whether to offload gradients to CPU
            gradient_checkpointing_ratio: Ratio of layers to enable gradient checkpointing
            init_optimizer: Whether to initialize optimizer (orchestrator handles optimizer)
            device: Device to use (defaults to CUDA if available)
        """
        self.model_path = model_path
        self.lr = lr
        self.accum_steps = accum_steps
        self.grad_offload = grad_offload
        self.gradient_checkpointing_ratio = gradient_checkpointing_ratio
        self.init_optimizer = init_optimizer

        if device is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if dist.is_initialized():
                torch.cuda.set_device(local_rank)
                self.device = torch.device("cuda", local_rank)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.trainer: Optional["CPUOffloadTrainer"] = None
        self._worker_micro_step: int = 0
        self._last_synced_version: Optional[int] = None
        self._last_version_poll: float = 0.0
        self._version_poll_interval: float = 5.0  # Default, can be overridden via set_version_poll_interval
        self._lock_owner_id: Optional[str] = None
        self._unique_worker_id: Optional[str] = None  # Unique ID for this trainer instance
    
    def set_version_poll_interval(self, interval: float):
        """Set the version poll interval for weight updates."""
        self._version_poll_interval = interval

    def get_unique_worker_id(self) -> str:
        """Get or generate a unique worker ID that includes random component for multi-node safety."""
        if self._unique_worker_id is None:
            # Generate unique ID: hostname-rank-uuid
            rank = dist.get_rank() if dist.is_initialized() else 0
            hostname = socket.gethostname()
            unique_suffix = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
            self._unique_worker_id = f"{hostname}-rank{rank}-{unique_suffix}"
        return self._unique_worker_id

    def initialize(self):
        """Initialize the trainer."""
        if self.trainer is None:
            # Lazy import to avoid circular dependency
            from ..ralo import CPUOffloadTrainer
            
            self.trainer = CPUOffloadTrainer(
                self.model_path,
                lr=self.lr,
                accum_steps=self.accum_steps,
                grad_offload=self.grad_offload,
                gradient_checkpointing_ratio=self.gradient_checkpointing_ratio,
                init_optimizer=self.init_optimizer,
            )
            self.accum_steps = self.trainer.accum_steps
        return self.trainer

    def get_trainer(self) -> "CPUOffloadTrainer":
        """Get the trainer, initializing if necessary."""
        if self.trainer is None:
            self.initialize()
        return self.trainer

    def get_model(self) -> torch.nn.Module:
        """Get the model from trainer."""
        return self.get_trainer().get_model()

    def get_device(self) -> torch.device:
        """Get the device."""
        return self.device

    def backward(self, loss: torch.Tensor):
        """
        Perform backward pass.

        Args:
            loss: Loss tensor
        """
        self.get_trainer().backward(loss)

    def collect_gradients(self, to_cpu: bool = True, clear: bool = True) -> Dict[str, torch.Tensor]:
        """
        Collect gradients from model.

        Args:
            to_cpu: Whether to move gradients to CPU
            clear: Whether to clear gradients after collection

        Returns:
            Dictionary of gradient tensors
        """
        return self.get_trainer().collect_gradients(to_cpu=to_cpu, clear=clear)

    def get_per_token_logps(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token log probabilities.

        Args:
            logits: Logits tensor [batch, seq_len, vocab_size]
            input_ids: Input token IDs [batch, seq_len]

        Returns:
            Per-token log probabilities [batch, seq_len]
        """
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def get_lock_owner_id(self) -> str:
        """Get or create lock owner ID. Uses unique worker ID for multi-node safety."""
        if self._lock_owner_id is None:
            # Use unique worker ID to avoid conflicts across nodes
            self._lock_owner_id = self.get_unique_worker_id()
        return self._lock_owner_id

    def set_lock_owner_id(self, owner_id: str):
        """Set lock owner ID. Note: This will override the unique worker ID."""
        self._lock_owner_id = owner_id

    def get_worker_micro_step(self) -> int:
        """Get current worker micro step."""
        return self._worker_micro_step

    def increment_micro_step(self):
        """Increment worker micro step."""
        self._worker_micro_step += 1

    def reset_micro_step(self):
        """Reset worker micro step."""
        self._worker_micro_step = 0

    def should_accumulate(self) -> bool:
        """Check if we should accumulate gradients (not ready to send)."""
        return self._worker_micro_step % self.accum_steps != 0

    def maybe_pull_weights(
        self,
        orchestrator_service,
        force: bool = False,
    ) -> bool:
        """
        Pull weights from orchestrator if a new version is available.

        Args:
            orchestrator_service: OrchestratorService instance
            force: Force update even if version hasn't changed

        Returns:
            True if updated, False otherwise
        """
        now = time.time()
        if not force and (now - self._last_version_poll) < self._version_poll_interval:
            return False

        self._last_version_poll = now

        target_version = orchestrator_service.latest_version()
        if target_version is None or target_version <= 0:
            return False

        if self._last_synced_version is not None and target_version <= self._last_synced_version and not force:
            return False

        # Download and load weights
        state_bytes = orchestrator_service.download_weights(version=target_version)
        if state_bytes is None:
            print(f"[TrainingService] Failed to download weights version {target_version}")
            return False

        buffer = io.BytesIO(state_bytes)
        try:
            state_dict = torch.load(buffer, map_location="cpu")
            self.get_model().load_state_dict(state_dict, strict=False)
            self._last_synced_version = target_version
            print(f"[TrainingService] Updated local weights to version {target_version}")
            return True
        except Exception as exc:
            print(f"[TrainingService] Failed to load weights version {target_version}: {exc}")
            return False

    def get_last_synced_version(self) -> Optional[int]:
        """Get last synced weight version."""
        return self._last_synced_version

    def set_last_synced_version(self, version: int):
        """Set last synced weight version."""
        self._last_synced_version = version

    def send_gradients_with_retry(
        self,
        orchestrator_service,
        step_id: int,
        grad_state: Dict[str, torch.Tensor],
        batch_meta: Dict[str, Any],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Send gradients to orchestrator with retry logic.

        Args:
            orchestrator_service: OrchestratorService instance
            step_id: Training step ID
            grad_state: Dictionary of gradient tensors
            batch_meta: Batch metadata
            max_retries: Maximum number of retry attempts

        Returns:
            Response dictionary

        Raises:
            RuntimeError: If all retries fail
        """
        attempts = 0
        last_error = None
        rank = dist.get_rank() if dist.is_initialized() else 0

        while attempts < max_retries:
            try:
                result = orchestrator_service.send_gradients(
                    step_id=step_id,
                    grad_state=grad_state,
                    batch_meta=batch_meta,
                    worker_id=self.get_lock_owner_id(),
                    model_version=self._last_synced_version,
                )
                if rank == 0:
                    print(f"[TrainingService rank={rank}] Successfully sent gradients for step {step_id}")
                return result
            except Exception as exc:
                last_error = exc
                attempts += 1
                if rank == 0:
                    print(f"[TrainingService rank={rank}] Failed to send gradients (attempt {attempts}/{max_retries}): {exc}")
                if attempts < max_retries:
                    time.sleep(min(2 ** attempts, 10))

        if rank == 0:
            print(f"[TrainingService rank={rank}] ERROR: Failed to upload gradients after {attempts} attempts: {last_error}")
        raise RuntimeError(f"Failed to upload gradients after {attempts} attempts: {last_error}") from last_error

