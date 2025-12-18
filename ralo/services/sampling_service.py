"""
SamplingService: Handles vLLM initialization, generation, and weight synchronization.
"""

import os
import time
from typing import Any, Dict, List, Optional

import torch

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None


class SamplingService:
    """Service for vLLM initialization, generation, and weight synchronization."""

    def __init__(
        self,
        model_path: str,
        model_id: str,
        orchestrator_url: str,
        vllm_kwargs: Optional[Dict[str, Any]] = None,
        gen_temperature: float = 0.9,
        version_poll_interval: float = 5.0,
    ):
        """
        Initialize SamplingService.

        Args:
            model_path: Path to the model
            model_id: Model identifier
            orchestrator_url: URL of the orchestrator server
            vllm_kwargs: Additional kwargs for vLLM LLM initialization
            gen_temperature: Default generation temperature
            version_poll_interval: Interval in seconds to check for new weight versions
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not available. Please install vllm package.")

        self.model_path = model_path
        self.model_id = model_id
        self.orchestrator_url = orchestrator_url
        self.gen_temperature = gen_temperature

        default_kwargs = {"enable_chunked_prefill": True, "gpu_memory_utilization": 0.8}
        self.vllm_kwargs = {**default_kwargs, **(vllm_kwargs or {})}

        self.vllm_gen: Optional[LLM] = None
        self._current_version: int = -1
        self._last_version_poll: float = 0.0
        self._version_poll_interval: float = version_poll_interval

    def initialize(self, gen_device: int, gen_rank: int = 0):
        """
        Initialize vLLM engine.

        Args:
            gen_device: GPU device ID
            gen_rank: Process rank
        """
        # Clean up distributed training environment variables
        cleanup_keys = [
            "RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_NAME",
            "GROUP_WORLD_SIZE",
            "ROLE_WORLD_SIZE",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
            "TORCHELASTIC_RUN_ID",
            "TORCHELASTIC_USE_AGENT_STORE",
            "TORCHELASTIC_ERROR_FILE",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "NCCL_COMM_ID",
            "NCCL_DEBUG",
            "NCCL_SOCKET_IFNAME",
        ]
        for key in cleanup_keys:
            os.environ.pop(key, None)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gen_device)
        torch.cuda.set_device(0)
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        print(f"[SamplingService {gen_rank}] Initializing vLLM on GPU {gen_device}")
        self.vllm_gen = LLM(model=self.model_path, **self.vllm_kwargs)
        print(f"[SamplingService {gen_rank}] vLLM initialized")

    def get_vllm_engine(self) -> LLM:
        """Get the vLLM engine, initializing if necessary."""
        if self.vllm_gen is None:
            raise RuntimeError("vLLM engine not initialized. Call initialize() first.")
        return self.vllm_gen

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = False,
    ) -> List[Any]:
        """
        Generate text using vLLM.

        Args:
            prompts: List of prompt strings
            sampling_params: Sampling parameters (defaults to temperature-based)
            use_tqdm: Whether to show progress bar

        Returns:
            List of generation outputs
        """
        if self.vllm_gen is None:
            raise RuntimeError("vLLM engine not initialized. Call initialize() first.")

        if sampling_params is None:
            sampling_params = SamplingParams(temperature=self.gen_temperature, top_p=1.0)

        return self.vllm_gen.generate(prompts, sampling_params, use_tqdm=use_tqdm)

    def create_sampling_params(
        self,
        n: int = 1,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        logprobs: Optional[int] = None,
        include_stop_str_in_output: bool = False,
    ) -> SamplingParams:
        """
        Create SamplingParams with defaults.

        Args:
            n: Number of samples per prompt
            temperature: Sampling temperature (defaults to gen_temperature)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter (optional)
            min_p: Min-p sampling parameter (optional)
            logprobs: Number of logprobs to return
            include_stop_str_in_output: Whether to include stop string in output

        Returns:
            SamplingParams instance
        """
        if temperature is None:
            temperature = self.gen_temperature

        kwargs = {
            "n": n,
            "temperature": float(temperature) if temperature is not None else 1.0,
            "max_tokens": int(max_tokens),
            "top_p": float(top_p) if top_p is not None else 1.0,
            "include_stop_str_in_output": include_stop_str_in_output,
        }
        # Handle top_k (may be None, string 'None', or actual value)
        if top_k is not None and str(top_k).strip().lower() not in ('none', ''):
            try:
                kwargs["top_k"] = int(top_k)
            except (ValueError, TypeError):
                pass  # Skip if conversion fails
        # Handle min_p (may be None, string 'None', or actual value)
        if min_p is not None and str(min_p).strip().lower() not in ('none', ''):
            try:
                kwargs["min_p"] = float(min_p)
            except (ValueError, TypeError):
                pass  # Skip if conversion fails
        if logprobs is not None:
            kwargs["logprobs"] = int(logprobs)

        return SamplingParams(**kwargs)

    def get_latest_version(self, orchestrator_service) -> int:
        """
        Get latest weight version from orchestrator.

        Args:
            orchestrator_service: OrchestratorService instance

        Returns:
            Latest version or -1 on failure
        """
        try:
            return orchestrator_service.latest_version()
        except Exception:
            return -1

    def maybe_update_weights(
        self,
        orchestrator_service,
        gen_rank: int = 0,
        force: bool = False,
    ) -> bool:
        """
        Update vLLM weights from orchestrator if a new version is available.

        Args:
            orchestrator_service: OrchestratorService instance
            gen_rank: Process rank for logging
            force: Force update even if version hasn't changed

        Returns:
            True if updated, False otherwise
        """
        if self.vllm_gen is None:
            print(f"[SamplingService {gen_rank}] ✗ Cannot update weights: vLLM engine not initialized")
            return False

        now = time.time()
        elapsed_since_last_poll = now - self._last_version_poll
        if not force and elapsed_since_last_poll < self._version_poll_interval:
            # Polling interval not reached yet, skip silently
            return False

        self._last_version_poll = now

        # Log current version before checking
        print(f"[SamplingService {gen_rank}] [VERSION CHECK] Current local version: v{self._current_version}")

        try:
            latest = orchestrator_service.latest_version()
            print(f"[SamplingService {gen_rank}] [VERSION CHECK] Latest server version: v{latest}")
        except Exception as e:
            print(f"[SamplingService {gen_rank}] ✗ [VERSION CHECK FAILED] Failed to check latest version from orchestrator: {e}")
            latest = -1

        if latest is None or latest < 0:
            print(f"[SamplingService {gen_rank}] ✗ [VERSION CHECK FAILED] Latest version check returned invalid value: {latest} (current: v{self._current_version})")
            return False

        # Log version comparison
        if self._current_version != latest:
            print(f"[SamplingService {gen_rank}] [VERSION CHECK] Version mismatch detected: local=v{self._current_version}, server=v{latest}")
        else:
            # Log when versions match (to confirm checking is happening)
            print(f"[SamplingService {gen_rank}] [VERSION CHECK] Already at latest version: v{self._current_version}")

        # Only update if we have a new version
        if latest <= self._current_version and not force:
            # Already at latest version
            print(f"[SamplingService {gen_rank}] [VERSION CHECK] No update needed (latest={latest} <= current={self._current_version})")
            return False

        # Skip initial version 0 if we haven't loaded it yet
        if self._current_version == -1 and latest == 0:
            self._current_version = 0
            print(f"[SamplingService {gen_rank}] [VERSION INIT] Initializing to version v{self._current_version} (server has v{latest})")
            return False

        # We have a new version to update to
        print(f"[SamplingService {gen_rank}] [WEIGHT UPDATE] Starting update: v{self._current_version} → v{latest}")

        # Load weights from server using vLLM's collective RPC
        # Note: load_weights_from_server is defined in ralo.py as a module-level function
        # that will be called by vLLM's collective_rpc. We import it here.
        try:
            from ..ralo import load_weights_from_server
        except ImportError:
            # Fallback: define it here if import fails
            def load_weights_from_server(model_runner, orchestrator_url, model_id, version):
                """Load weights from orchestrator server in vLLM worker."""
                try:
                    import requests
                    import io as _io
                    import os
                    target_dtype = torch.bfloat16
                    url = f"{orchestrator_url}/weights/download"
                    params = {"model_id": model_id, "version": int(version)}
                    resp = requests.get(url, params=params, stream=True, timeout=600)
                    worker_id = os.environ.get('RANK', '0')
                    if resp.status_code != 200:
                        return (False, f"Worker {worker_id} HTTP {resp.status_code} while downloading weights")
                    buf = _io.BytesIO()
                    for chunk in resp.iter_content(chunk_size=32 * 1024 * 1024):
                        if chunk:
                            buf.write(chunk)
                    buf.seek(0)
                    loaded_sd = torch.load(buf, map_location="cpu")
                    state_dict_to_load = {}
                    for k, v in loaded_sd.items():
                        if isinstance(v, torch.Tensor):
                                state_dict_to_load[k] = v.to(target_dtype)
                        else:
                            state_dict_to_load[k] = v
                    model_runner.model.load_weights(state_dict_to_load.items())
                    return (True, f"Worker {worker_id}: Weights loaded successfully from server version {version}.")
                except Exception as e:
                    import traceback
                    import os
                    worker_id = os.environ.get('RANK', '0')
                    return (
                        False,
                        f"Worker {worker_id} FAILED while downloading/applying: {str(e)}\n{traceback.format_exc()}",
                    )

        attempt = 0
        max_attempts = 3
        last_results = None

        while attempt < max_attempts:
            attempt += 1
            try:
                # Check if weights exist
                import requests

                try:
                    r = requests.get(
                        f"{self.orchestrator_url}/weights/download",
                        params={"model_id": self.model_id, "version": int(latest)},
                        timeout=5,
                        stream=True,
                    )
                    if r.status_code == 404:
                        print(f"[SamplingService {gen_rank}] Server reported 404 for weights v{latest}; retry later")
                        return False
                except Exception:
                    pass

                # Update weights via collective RPC or apply_model
                # Try multiple paths to find model_runner (vLLM version compatibility)
                model_runner = None
                use_apply_model = False
                
                try:
                    # Try vLLM 1.0+ structure: llm_engine.model_executor.driver_worker.model_runner
                    if hasattr(self.vllm_gen, 'llm_engine'):
                        llm_engine = self.vllm_gen.llm_engine
                        # Try engine_core path (vLLM latest versions)
                        if hasattr(llm_engine, 'engine_core'):
                            engine_core = llm_engine.engine_core
                            # Try core_engines path (vLLM v1)
                            if hasattr(engine_core, 'core_engines'):
                                core_engines = engine_core.core_engines
                                if core_engines and len(core_engines) > 0:
                                    # Get first core engine
                                    first_engine = core_engines[0]
                                    if hasattr(first_engine, 'model_runner'):
                                        model_runner = first_engine.model_runner
                                    elif hasattr(first_engine, 'model'):
                                        # vLLM v1 might use 'model' instead of 'model_runner'
                                        model_runner = first_engine.model
                            # Try model_executor path within engine_core
                            if model_runner is None and hasattr(engine_core, 'model_executor'):
                                model_executor = engine_core.model_executor
                                if hasattr(model_executor, 'driver_worker'):
                                    if hasattr(model_executor.driver_worker, 'model_runner'):
                                        model_runner = model_executor.driver_worker.model_runner
                                elif hasattr(model_executor, 'model_runner'):
                                    model_runner = model_executor.model_runner
                            # Try driver_worker path within engine_core
                            if model_runner is None and hasattr(engine_core, 'driver_worker'):
                                if hasattr(engine_core.driver_worker, 'model_runner'):
                                    model_runner = engine_core.driver_worker.model_runner
                            # Try direct model_runner in engine_core
                            if model_runner is None and hasattr(engine_core, 'model_runner'):
                                model_runner = engine_core.model_runner
                        # Try model_executor path (older vLLM versions)
                        elif hasattr(llm_engine, 'model_executor'):
                            model_executor = llm_engine.model_executor
                            if hasattr(model_executor, 'driver_worker'):
                                if hasattr(model_executor.driver_worker, 'model_runner'):
                                    model_runner = model_executor.driver_worker.model_runner
                            elif hasattr(model_executor, 'model_runner'):
                                model_runner = model_executor.model_runner
                        elif hasattr(llm_engine, 'driver_worker'):
                            if hasattr(llm_engine.driver_worker, 'model_runner'):
                                model_runner = llm_engine.driver_worker.model_runner
                        elif hasattr(llm_engine, 'model_runner'):
                            model_runner = llm_engine.model_runner
                    
                    # Fallback to direct model_runner
                    if model_runner is None and hasattr(self.vllm_gen, 'model_runner'):
                        model_runner = self.vllm_gen.model_runner
                    
                    # If model_runner not found, try using apply_model (vLLM v1)
                    if model_runner is None and hasattr(self.vllm_gen, 'apply_model'):
                        use_apply_model = True
                        print(f"[SamplingService {gen_rank}] Using apply_model for weight update (vLLM v1)")
                    
                    if model_runner is None and not use_apply_model:
                        raise AttributeError("Cannot find model_runner in vLLM LLM object")
                        
                except AttributeError as e:
                    print(f"[SamplingService {gen_rank}] ERROR finding model_runner: {e}")
                    # Try apply_model as fallback
                    if hasattr(self.vllm_gen, 'apply_model'):
                        use_apply_model = True
                        print(f"[SamplingService {gen_rank}] Falling back to apply_model for weight update")
                    else:
                        print(f"[SamplingService {gen_rank}] vLLM object attributes: {dir(self.vllm_gen)}")
                        if hasattr(self.vllm_gen, 'llm_engine'):
                            print(f"[SamplingService {gen_rank}] llm_engine attributes: {dir(self.vllm_gen.llm_engine)}")
                            if hasattr(self.vllm_gen.llm_engine, 'engine_core'):
                                print(f"[SamplingService {gen_rank}] engine_core attributes: {dir(self.vllm_gen.llm_engine.engine_core)}")
                        raise
                
                if use_apply_model:
                    # Use apply_model for vLLM v1
                    # IMPORTANT: The function passed to apply_model must not capture large objects in closure
                    # because vLLM serializes the function to send to worker processes
                    # Instead, we pass orchestrator URL and version, and download inside the function
                    orchestrator_url = self.orchestrator_url
                    model_id = self.model_id
                    target_version = int(latest)
                    
                    def apply_state_dict(model):
                        """Function to download and apply state_dict to model.
                        Downloads weights inside the function to avoid closure serialization issues.
                        """
                        try:
                            import requests
                            import io as _io
                            url = f"{orchestrator_url}/weights/download"
                            params = {"model_id": model_id, "version": target_version}
                            resp = requests.get(url, params=params, stream=True, timeout=600)
                            if resp.status_code != 200:
                                raise RuntimeError(f"Failed to download weights: HTTP {resp.status_code}")
                            
                            buf = _io.BytesIO()
                            for chunk in resp.iter_content(chunk_size=32 * 1024 * 1024):
                                if chunk:
                                    buf.write(chunk)
                            buf.seek(0)
                            loaded_sd = torch.load(buf, map_location="cpu")
                            
                            # Prepare state dict (all tensors to bfloat16 for consistency)
                            state_dict_to_load = {}
                            for k, v in loaded_sd.items():
                                if isinstance(v, torch.Tensor):
                                        state_dict_to_load[k] = v.to(torch.bfloat16)
                                else:
                                    state_dict_to_load[k] = v
                            
                            # Apply weights
                            try:
                                model.load_weights(state_dict_to_load.items())
                                return True
                            except Exception as e:
                                # Fallback: try load_state_dict if load_weights doesn't work
                                if hasattr(model, 'load_state_dict'):
                                    model.load_state_dict(state_dict_to_load, strict=False)
                                    return True
                                raise
                        except Exception as e:
                            print(f"[SamplingService worker] Error in apply_state_dict: {e}")
                            raise
                    
                    try:
                        self.vllm_gen.apply_model(apply_state_dict)
                        old_version = self._current_version
                        self._current_version = latest
                        print(f"[SamplingService {gen_rank}] ✓ [WEIGHT UPDATE SUCCESS] Updated from v{old_version} to v{self._current_version} using apply_model")
                        return True
                    except Exception as e:
                        print(f"[SamplingService {gen_rank}] ✗ [WEIGHT UPDATE FAILED] Attempt {attempt}/{max_attempts} failed: {e}")
                        import traceback
                        traceback.print_exc()
                        last_results = None
                        continue  # Retry
                else:
                    # Use collective_rpc with model_runner
                    results = self.vllm_gen.collective_rpc(
                        load_weights_from_server,
                        model_runner,
                        [self.orchestrator_url, self.model_id, latest],
                    )
                    last_results = results

                # Check if all workers succeeded
                def _ok(res):
                    try:
                        return bool(res[0])
                    except Exception:
                        return bool(res)

                all_ok = isinstance(results, (list, tuple)) and len(results) > 0 and all(_ok(r) for r in results)

                if all_ok:
                    old_version = self._current_version
                    self._current_version = latest
                    print(f"[SamplingService {gen_rank}] ✓ [WEIGHT UPDATE SUCCESS] Updated from v{old_version} to v{self._current_version} using collective_rpc")
                    return True
                else:
                    print(f"[SamplingService {gen_rank}] ✗ [WEIGHT UPDATE FAILED] Attempt {attempt}/{max_attempts} failed: some workers failed")
                    if last_results:
                        for i, res in enumerate(last_results):
                            if not _ok(res):
                                print(f"[SamplingService {gen_rank}]   Worker {i} result: {res}")
                    time.sleep(1.0)
            except Exception as e:
                print(f"[SamplingService {gen_rank}] ✗ [WEIGHT UPDATE EXCEPTION] Attempt {attempt}/{max_attempts} failed with exception: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)

        # All attempts failed
        print(f"[SamplingService {gen_rank}] ✗ [WEIGHT UPDATE FINAL FAILURE] Failed to update to v{latest} after {max_attempts} attempts (current version remains v{self._current_version})")
        return False

    def get_current_version(self) -> int:
        """Get current weight version."""
        return self._current_version

    def set_current_version(self, version: int):
        """Set current weight version."""
        self._current_version = version

