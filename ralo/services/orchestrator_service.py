"""
OrchestratorService: Wraps SamplerClient and TrainerClient for orchestrator communication.
"""

import socket
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import torch

from ..sampler.client import SamplerClient
from ..trainer.client import TrainerClient


class OrchestratorService:
    """Service for orchestrator communication via HTTP clients."""

    def __init__(
        self, 
        orchestrator_url: str, 
        model_id: str = "default", 
        retry_interval: float = 1.0,
        timeout_config: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize OrchestratorService.

        Args:
            orchestrator_url: URL of the orchestrator server
            model_id: Model identifier
            retry_interval: Interval between retries for sampler operations
            timeout_config: Dictionary of timeout values for different operations
        """
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.model_id = model_id
        self.retry_interval = retry_interval
        self.timeout_config = timeout_config or {}

        self._sampler_client: Optional[SamplerClient] = None
        self._trainer_client: Optional[TrainerClient] = None
        self._lock_owner_id: Optional[str] = None

    @property
    def sampler_client(self) -> SamplerClient:
        """Get or create SamplerClient."""
        if self._sampler_client is None:
            self._sampler_client = SamplerClient(self.orchestrator_url, retry_interval=self.retry_interval)
        return self._sampler_client

    @property
    def trainer_client(self) -> TrainerClient:
        """Get or create TrainerClient."""
        if self._trainer_client is None:
            self._trainer_client = TrainerClient(self.orchestrator_url)
        return self._trainer_client

    def get_lock_owner_id(self) -> str:
        """Get or create lock owner ID."""
        if self._lock_owner_id is None:
            self._lock_owner_id = f"{socket.gethostname()}-rank0"
        return self._lock_owner_id

    def set_lock_owner_id(self, owner_id: str):
        """Set lock owner ID."""
        self._lock_owner_id = owner_id

    # Sampler methods
    def fetch_problem(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Fetch a problem from orchestrator.

        Args:
            timeout: Request timeout in seconds (defaults to config value)

        Returns:
            Problem data dictionary
        """
        if timeout is None:
            timeout = self.timeout_config.get("fetch_problem_timeout", 10.0)
        return self.sampler_client.fetch_problem(timeout=timeout)

    def wait_for_problem(self) -> Dict[str, Any]:
        """
        Wait for a problem to be available from orchestrator.

        Returns:
            Problem data dictionary
        """
        return self.sampler_client.wait_for_problem()

    def upload_samples(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Upload samples to orchestrator.

        Args:
            payload: Sample data dictionary
            timeout: Request timeout in seconds (defaults to config value)

        Returns:
            Response dictionary
        """
        if timeout is None:
            timeout = self.timeout_config.get("upload_samples_timeout", 300.0)
        return self.sampler_client.upload_samples(payload, timeout=timeout)

    # Evaluation methods (via SamplerClient)
    def get_eval_job(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get next evaluation job (preview).
        """
        if timeout is None:
            timeout = self.timeout_config.get("stats_timeout", 5.0)
        try:
            return self.sampler_client.get_eval_job(timeout=timeout)
        except Exception:
            return {"empty": True}

    def claim_eval_job(self, job_id: str, owner: Optional[str] = None, ttl: Optional[float] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Claim an evaluation job and receive full payload.
        """
        if owner is None:
            owner = self.get_lock_owner_id()
        if ttl is None:
            ttl = self.timeout_config.get("lock_ttl", 30.0)
        if timeout is None:
            timeout = self.timeout_config.get("lock_timeout", 5.0)
        return self.sampler_client.claim_eval_job(job_id, owner=owner, ttl=float(ttl), timeout=timeout)

    def report_eval_job(self, job_id: str, metrics: Dict[str, Any], samples=None, duration_sec: Optional[float] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Report evaluation results back to orchestrator.
        """
        if timeout is None:
            timeout = self.timeout_config.get("upload_samples_timeout", 300.0)
        return self.sampler_client.report_eval_job(job_id, metrics=metrics, samples=samples, duration_sec=duration_sec, timeout=timeout)

    def get_eval_stats(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if timeout is None:
            timeout = self.timeout_config.get("stats_timeout", 5.0)
        try:
            return self.sampler_client.eval_stats(timeout=timeout)
        except Exception:
            return {"ok": False}

    def report_compute_usage(
        self,
        role: str,
        gpu_seconds: float,
        tokens: int = 0,
        worker_id: Optional[str] = None,
        device_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Report compute usage (GPU seconds / tokens) back to the orchestrator.
        """
        if timeout is None:
            timeout = self.timeout_config.get("stats_timeout", 5.0)
        payload: Dict[str, Any] = {
            "role": role,
            "gpu_seconds": float(max(gpu_seconds, 0.0)),
            "tokens": int(max(tokens, 0)),
            "worker_id": worker_id or self.get_lock_owner_id(),
        }
        if device_count is not None:
            payload["device_count"] = int(max(device_count, 1))
        if metadata:
            payload["metadata"] = metadata
        try:
            return self.trainer_client.report_compute_usage(payload, timeout=timeout)
        except Exception:
            return {"ok": False}

    # Trainer methods
    def register_trainer(
        self,
        worker_id: Optional[str] = None,
        accum_steps: int = 1,
        max_steps: Optional[int] = None,
        allow_failure: bool = False,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Register trainer with orchestrator.

        Args:
            worker_id: Worker identifier
            accum_steps: Accumulation steps
            max_steps: Maximum training steps
            allow_failure: If True, return empty dict on failure instead of raising
            timeout: Request timeout in seconds (defaults to config value)

        Returns:
            Registration response dictionary
        """
        if timeout is None:
            timeout = self.timeout_config.get("register_timeout", 10.0)
        if worker_id is None:
            worker_id = self.get_lock_owner_id()

        payload = {
            "model_id": self.model_id,
            "worker_id": worker_id,
            "accum_steps": accum_steps,
        }
        if max_steps is not None:
            payload["max_steps"] = max_steps

        try:
            return self.trainer_client.register(payload, timeout=timeout)
        except Exception as exc:
            if allow_failure:
                print(f"[OrchestratorService] Registration failed (allowed): {exc}")
                return {}
            raise

    def get_batch(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get a training batch from orchestrator.

        Args:
            timeout: Request timeout in seconds (defaults to config value)

        Returns:
            Batch data dictionary or None if empty
        """
        if timeout is None:
            timeout = self.timeout_config.get("get_batch_timeout", 60.0)
        return self.trainer_client.get_batch(timeout=timeout)

    def send_gradients(
        self,
        step_id: int,
        grad_state: Dict[str, "torch.Tensor"],
        batch_meta: Dict[str, Any],
        worker_id: Optional[str] = None,
        model_version: Optional[int] = None,
        timeout: Optional[float] = None,
        chunk_size_mb: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send gradients to orchestrator.

        Args:
            step_id: Training step ID
            grad_state: Dictionary of gradient tensors
            batch_meta: Batch metadata
            worker_id: Worker identifier
            model_version: Model version used for this step
            timeout: Request timeout in seconds (defaults to config value)
            chunk_size_mb: Chunk size in MB for large uploads

        Returns:
            Response dictionary
        """
        from ..utils import encode_gradients

        if timeout is None:
            timeout = self.timeout_config.get("send_gradients_timeout", 300.0)
        if chunk_size_mb is None:  # Use config value if not explicitly provided
            chunk_size_mb = self.timeout_config.get("chunk_size_mb", 50)
        if worker_id is None:
            worker_id = self.get_lock_owner_id()

        payload = {
            "model_id": self.model_id,
            "worker_id": worker_id,
            "step_id": int(step_id),
            "microsteps": int(batch_meta.get("microsteps", 1)),
            "token_count": int(batch_meta.get("token_count", 0)),
            "sample_count": int(batch_meta.get("sample_count", 0)),
            "_batch_id": batch_meta.get("_batch_id"),
            "_batch_ids": batch_meta.get("_batch_ids"),
        }
        if model_version is not None:
            payload["model_version"] = int(model_version)

        payload.update(encode_gradients(grad_state))
        return self.trainer_client.send_gradients(payload, timeout=timeout, chunk_size_mb=chunk_size_mb)

    def download_weights(self, version: int = -1, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Download weights from orchestrator.

        Args:
            version: Weight version (-1 for latest)
            timeout: Request timeout in seconds (defaults to config value)

        Returns:
            Weight bytes or None on failure
        """
        if timeout is None:
            timeout = self.timeout_config.get("download_weights_timeout", 600.0)
        download_chunk_size_mb = self.timeout_config.get("download_chunk_size_mb", 32)
        if version == -1:
            version = self.latest_version()
            if version < 0:
                return None

        return self.trainer_client.download_weights(self.model_id, version, timeout=timeout, chunk_size_mb=download_chunk_size_mb)

    def latest_version(self, timeout: Optional[float] = None) -> int:
        """
        Get latest weight version from orchestrator.

        Args:
            timeout: Request timeout in seconds (defaults to config value)

        Returns:
            Latest version number or -1 on failure
        """
        if timeout is None:
            timeout = self.timeout_config.get("version_check_timeout", 5.0)
        try:
            return self.trainer_client.latest_version(self.model_id, timeout=timeout)
        except Exception:
            return -1

    def next_step(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get next global step from orchestrator.

        Args:
            timeout: Request timeout in seconds (defaults to config value)

        Returns:
            Step response dictionary
        """
        if timeout is None:
            timeout = self.timeout_config.get("next_step_timeout", 5.0)
        return self.trainer_client.next_step(timeout=timeout)

    def send_heartbeat(
        self,
        step_id: int,
        microstep: int,
        total_microsteps: int,
        worker_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Send heartbeat to orchestrator.

        Args:
            step_id: Current step ID
            microstep: Current microstep
            total_microsteps: Total microsteps
            worker_id: Worker identifier
            timeout: Request timeout in seconds (defaults to config value)

        Returns:
            True if successful, False otherwise
        """
        if timeout is None:
            timeout = self.timeout_config.get("heartbeat_timeout", 2.0)
        if worker_id is None:
            worker_id = self.get_lock_owner_id()

        return self.trainer_client.send_heartbeat(
            worker_id=worker_id,
            step_id=step_id,
            microstep=microstep,
            total_microsteps=total_microsteps,
            timeout=timeout,
        )

    def stats(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get orchestrator stats.

        Args:
            timeout: Request timeout in seconds (defaults to config value)

        Returns:
            Stats dictionary
        """
        if timeout is None:
            timeout = self.timeout_config.get("stats_timeout", 5.0)
        return self.trainer_client.stats(timeout=timeout)

    def acquire_lock(self, key: str, owner: Optional[str] = None, ttl: Optional[float] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Acquire a lock from orchestrator.

        Args:
            key: Lock key
            owner: Lock owner (defaults to lock_owner_id)
            ttl: Time to live in seconds (defaults to config value)
            timeout: Request timeout in seconds (defaults to config value)

        Returns:
            Lock response dictionary
        """
        if timeout is None:
            timeout = self.timeout_config.get("lock_timeout", 5.0)
        if ttl is None:
            ttl = self.timeout_config.get("lock_ttl", 30.0)
        if owner is None:
            owner = self.get_lock_owner_id()
        return self.trainer_client.acquire_lock(key, owner, ttl=ttl, timeout=timeout)

    def release_lock(self, key: str, owner: Optional[str] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Release a lock from orchestrator.

        Args:
            key: Lock key
            owner: Lock owner (defaults to lock_owner_id)
            timeout: Request timeout in seconds (defaults to config value)

        Returns:
            Lock response dictionary
        """
        if timeout is None:
            timeout = self.timeout_config.get("lock_timeout", 5.0)
        if owner is None:
            owner = self.get_lock_owner_id()
        return self.trainer_client.release_lock(key, owner, timeout=timeout)

