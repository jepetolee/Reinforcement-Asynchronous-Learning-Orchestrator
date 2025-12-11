import gc
import os
import re
import threading
import time
from typing import Any, Dict, Optional

import torch

from ralo.utils import extract_gradient_tensors, bytes_list_to_json


class GradientAggregator:
    """Handles gradient ingestion, optimizer steps, and weight versioning."""

    def __init__(
        self,
        model,
        optimizer,
        sample_manager,
        weights_dir: str,
        keep_last_versions: int = 2,
        update_steps: int = 50,
    ):
        self.model = model
        self.optimizer = optimizer
        self.sample_manager = sample_manager
        self.weights_dir = weights_dir
        self.keep_last_versions = keep_last_versions
        self.update_steps = max(1, int(update_steps))
        self._pending_batches = 0  # Use _pending_batches internally, access via property
        self.total_gradients = 0
        self.latest_versions: Dict[str, int] = {}
        self.current_version = -1
        self.param_map = dict(model.named_parameters()) if model is not None else {}
        # CPU optimizer state for CPU-based updates (saves GPU memory)
        self.cpu_optimizer_state: Optional[Dict[str, Any]] = None
        # Track pending batches by batch_id to handle timeouts
        self.pending_batch_tracking: Dict[str, Dict[str, Any]] = {}
        # Track gradients by worker_id to handle multi-node conflicts
        self.worker_gradient_tracking: Dict[str, Dict[str, Any]] = {}
        # Thread safety for async optimizer steps
        self._optimizer_lock = threading.Lock()
        self._optimizer_step_in_progress = False
        self._pending_optimizer_step = False
        # File-based gradient storage queue (memory efficient)
        self._gradient_file_queue = []
    
    @property
    def pending_batches(self) -> int:
        """Thread-safe access to pending_batches."""
        with self._optimizer_lock:
            return self._pending_batches
    
    @pending_batches.setter
    def pending_batches(self, value: int):
        """Thread-safe setter for pending_batches."""
        with self._optimizer_lock:
            self._pending_batches = value

    def _weight_dir(self, model_id: str) -> str:
        return os.path.join(self.weights_dir, model_id)

    def _save_weights(self, model_id: str, new_version: int) -> None:
        model_dir = self._weight_dir(model_id)
        os.makedirs(model_dir, exist_ok=True)
        out_path = os.path.join(model_dir, f"weights_v{new_version}.pt")
        # Model is already on CPU, so just detach (no need to move to CPU)
        state_dict_cpu = {
            k: (v.detach() if isinstance(v, torch.Tensor) else v)
            for k, v in self.model.state_dict().items()
        }
        with open(out_path, "wb") as f:
            torch.save(state_dict_cpu, f)

    def _maybe_prune(self, model_id: str) -> None:
        model_dir = self._weight_dir(model_id)
        if not os.path.isdir(model_dir):
            return
        files = []
        for name in os.listdir(model_dir):
            if name.startswith("weights_v") and name.endswith(".pt"):
                match = re.match(r"weights_v(\d+)\.pt", name)
                if match:
                    files.append((int(match.group(1)), os.path.join(model_dir, name)))
        if len(files) <= self.keep_last_versions:
            return
        files.sort(key=lambda x: x[0])
        for _, path in files[:-self.keep_last_versions]:
            try:
                os.remove(path)
            except OSError:
                pass

    def ingest_file(self, gradient_file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest gradient from file path (memory efficient - only stores file path, not data)."""
        if self.model is None or self.optimizer is None:
            return {"ok": False, "error": "model not loaded"}

        worker_id = metadata.get("worker_id", "unknown")
        step_id = metadata.get("step_id", -1)
        microsteps = int(metadata.get("microsteps", 1))
        batch_id = metadata.get("batch_id")
        batch_ids = metadata.get("batch_ids")
        should_step = False
        new_version: Optional[int] = None

        # Create unique gradient key to prevent conflicts
        gradient_key = f"{worker_id}-step{step_id}-{time.time()}"

        # Check for duplicate gradient uploads (same worker_id + step_id)
        for key, info in list(self.worker_gradient_tracking.items()):
            if info.get("worker_id") == worker_id and info.get("step_id") == step_id:
                elapsed = time.time() - info.get("timestamp", 0)
                if elapsed < 1.0:  # Within 1 second, likely duplicate
                    print(f"[ORCH] Skipping duplicate gradient from {worker_id} step {step_id} (elapsed: {elapsed:.2f}s)")
                    return {
                        "ok": True,
                        "pending_batches": int(self.pending_batches),
                        "version": None,
                        "duplicate": True,
                    }

        # Add file path to queue (memory efficient - no data loaded)
        with self._optimizer_lock:
            self._gradient_file_queue.append({
                "file_path": gradient_file_path,
                "metadata": metadata,
                "timestamp": time.time()
            })
            self._pending_batches += 1
            should_step = (self._pending_batches >= self.update_steps)

        self.total_gradients += 1

        # Track gradient by worker_id and step_id for duplicate detection
        self.worker_gradient_tracking[gradient_key] = {
            "worker_id": worker_id,
            "step_id": step_id,
            "microsteps": microsteps,
            "timestamp": time.time(),
            "batch_id": batch_id,
            "batch_ids": batch_ids,
        }

        # Remove old tracking entries (older than 5 minutes)
        now = time.time()
        for key in list(self.worker_gradient_tracking.keys()):
            if now - self.worker_gradient_tracking[key].get("timestamp", 0) > 300:
                self.worker_gradient_tracking.pop(key, None)

        # Track pending batches for timeout handling
        if batch_ids:
            if isinstance(batch_ids, (list, tuple)):
                for bid in batch_ids:
                    if bid:
                        self.pending_batch_tracking[bid] = {
                            "microsteps": microsteps,
                            "timestamp": time.time(),
                            "worker_id": worker_id,
                            "gradient_key": gradient_key,
                        }
        elif batch_id:
            self.pending_batch_tracking[batch_id] = {
                "microsteps": microsteps,
                "timestamp": time.time(),
                "worker_id": worker_id,
                "gradient_key": gradient_key,
            }

        if should_step:
            # Run optimizer step asynchronously to avoid blocking HTTP requests
            new_version = self._apply_optimizer_step_with_files_async()

        if batch_ids:
            if isinstance(batch_ids, (list, tuple)):
                for bid in batch_ids:
                    self.sample_manager.mark_processed(bid)
                    batch_info = self.pending_batch_tracking.pop(bid, None)
                    if batch_info:
                        gradient_key = batch_info.get("gradient_key")
                        if gradient_key:
                            self.worker_gradient_tracking.pop(gradient_key, None)
        elif batch_id:
            self.sample_manager.mark_processed(batch_id)
            batch_info = self.pending_batch_tracking.pop(batch_id, None)
            if batch_info:
                gradient_key = batch_info.get("gradient_key")
                if gradient_key:
                    self.worker_gradient_tracking.pop(gradient_key, None)

        return {
            "ok": True,
            "pending_batches": int(self.pending_batches),
            "version": int(new_version) if new_version is not None else None,
            "stepped": bool(new_version is not None),
        }

    def finalize(self) -> Optional[int]:
        if self.pending_batches <= 0 or self.model is None or self.optimizer is None:
            return None
        # For finalize, wait for async step to complete
        # Always use file-based path (disk-based storage only)
        return self._apply_optimizer_step_with_files()

    def _apply_optimizer_step_with_files_async(self) -> Optional[int]:
        """Run optimizer step with files asynchronously to avoid blocking HTTP requests."""
        with self._optimizer_lock:
            if self._optimizer_step_in_progress:
                # Optimizer step already in progress, mark pending
                self._pending_optimizer_step = True
                return None
            
            # Start optimizer step in background thread
            self._optimizer_step_in_progress = True
            self._pending_optimizer_step = False
        
        def run_step():
            try:
                version = self._apply_optimizer_step_with_files()
                # Check if another step is needed
                with self._optimizer_lock:
                    self._optimizer_step_in_progress = False
                    if self._pending_optimizer_step and self._pending_batches >= self.update_steps:
                        self._pending_optimizer_step = False
                        self._optimizer_step_in_progress = True
                        # Schedule next step
                        threading.Thread(target=run_step, daemon=True).start()
            except MemoryError as e:
                # Memory error - critical failure, stop processing
                print(f"[ORCH ERROR] Memory error in async optimizer step: {e}")
                print(f"[ORCH ERROR] Optimizer step aborted due to insufficient memory")
                import traceback
                traceback.print_exc()
                # Force garbage collection
                gc.collect()
                with self._optimizer_lock:
                    self._optimizer_step_in_progress = False
                    # Don't mark pending step - memory issue needs resolution
                    self._pending_optimizer_step = False
            except Exception as e:
                # Other errors - log and continue
                print(f"[ORCH] Error in async optimizer step (files): {e}")
                import traceback
                traceback.print_exc()
                # Force garbage collection to prevent memory leaks
                gc.collect()
                with self._optimizer_lock:
                    self._optimizer_step_in_progress = False
        
        # Start optimizer step in background thread
        threading.Thread(target=run_step, daemon=True).start()
        return None  # Return immediately, version will be updated asynchronously

    def _apply_optimizer_step_with_files(self) -> int:
        # Get pending batches and files atomically
        gradient_files = []
        with self._optimizer_lock:
            batches = max(1, int(self._pending_batches))
            self._pending_batches = 0
            
            # Get files for this step
            # Note: We take all available files up to batches count, or all if batches > len(queue)
            # But logic says batches = pending_batches, so we should take that many
            # However, pending_batches tracks number of uploads, and queue has uploads
            # So we take 'batches' number of items from queue
            count = min(batches, len(self._gradient_file_queue))
            gradient_files = self._gradient_file_queue[:count]
            self._gradient_file_queue = self._gradient_file_queue[count:]
            
            # If we took fewer files than batches (shouldn't happen if logic is correct), adjust batches
            if count < batches:
                batches = count
        
        if batches == 0:
            return self.current_version

        # Model is already on CPU, so we can use it directly
        # Get current model weights as Parameter objects for optimizer
        cpu_params = []
        cpu_param_map = {}  # Map param name to Parameter object
        for name, param in self.model.named_parameters():
            # Model is on CPU, just ensure it's a Parameter
            cpu_param = torch.nn.Parameter(param.data.clone())
            cpu_params.append(cpu_param)
            cpu_param_map[name] = cpu_param
        
        # Create CPU optimizer for CPU-based parameter updates
        cpu_optimizer = torch.optim.AdamW(cpu_params, lr=self.optimizer.param_groups[0]['lr'], 
                                          weight_decay=self.optimizer.param_groups[0].get('weight_decay', 0.0))
        
        # Load optimizer state if available
        if self.cpu_optimizer_state is not None:
            try:
                cpu_optimizer.load_state_dict(self.cpu_optimizer_state)
            except Exception:
                self.cpu_optimizer_state = None
        
        # Process files one by one and accumulate gradients directly into parameters
        # This is the key memory optimization: we don't load all gradients into memory at once
        processed_count = 0
        failed_files = []  # Track failed files for re-queuing
        
        for file_info in gradient_files:
            file_path = file_info["file_path"]
            if not os.path.exists(file_path):
                continue
                
            try:
                # Read gradient from file
                with open(file_path, 'rb') as f:
                    blob = f.read()
                
                payload = bytes_list_to_json(blob)
                grads = extract_gradient_tensors(payload)
                
                # Accumulate directly into parameter gradients
                for name, tensor in grads.items():
                    if name in cpu_param_map:
                        param = cpu_param_map[name]
                        # Ensure tensor is on CPU
                        grad_tensor = tensor.cpu() if tensor.device.type != 'cpu' else tensor
                        
                        if param.grad is None:
                            param.grad = grad_tensor.clone()
                        else:
                            param.grad.add_(grad_tensor)
                
                processed_count += 1
                
                # Free memory immediately
                del grads, payload, blob
                
                # Delete file immediately after successful processing
                try:
                    os.remove(file_path)
                    meta_file = file_path.replace('.bin', '.meta.json')
                    if os.path.exists(meta_file):
                        os.remove(meta_file)
                except OSError:
                    pass
                    
            except MemoryError as e:
                # Memory error - stop processing and preserve failed files
                print(f"[ORCH ERROR] Memory error processing gradient file {file_path}: {e}")
                print(f"[ORCH ERROR] Stopping optimizer step due to memory error. {len(gradient_files) - processed_count} files remaining.")
                # Keep failed file and remaining files for later processing
                failed_files.append(file_info)
                failed_files.extend(gradient_files[gradient_files.index(file_info) + 1:])
                # Free memory and exit
                if processed_count > 0:
                    # Clear accumulated gradients to free memory
                    for param in cpu_params:
                        if param.grad is not None:
                            param.grad = None
                    del cpu_params, cpu_param_map
                    gc.collect()
                
                # Re-queue failed files
                with self._optimizer_lock:
                    # Add failed files back to front of queue for retry
                    self._gradient_file_queue = failed_files + self._gradient_file_queue
                    # Restore pending_batches count
                    self._pending_batches += len(failed_files)
                
                raise  # Re-raise to let async step handle it
                
            except Exception as e:
                # Other errors - log and preserve file for retry
                print(f"[ORCH] Error processing gradient file {file_path}: {e}")
                import traceback
                traceback.print_exc()
                # Keep file for retry (don't delete)
                failed_files.append(file_info)
        
        # Re-queue failed files for retry
        if failed_files:
            print(f"[ORCH] Re-queuing {len(failed_files)} failed gradient files for retry")
            with self._optimizer_lock:
                # Add failed files back to front of queue
                self._gradient_file_queue = failed_files + self._gradient_file_queue
                # Restore pending_batches count for failed files
                self._pending_batches += len(failed_files)
        
        if processed_count == 0:
            return self.current_version
            
        # Normalize gradients
        for param in cpu_params:
            if param.grad is not None:
                param.grad.div_(processed_count)
        
        # Perform optimizer step on CPU
        cpu_optimizer.step()
        cpu_optimizer.zero_grad(set_to_none=True)
        
        # Save CPU optimizer state
        self.cpu_optimizer_state = cpu_optimizer.state_dict()
        
        # Update model with CPU-updated weights
        for name, param in self.model.named_parameters():
            if name in cpu_param_map:
                param.data.copy_(cpu_param_map[name].data)
        
        # Force garbage collection
        del cpu_params, cpu_param_map, cpu_optimizer
        gc.collect()

        base_id = (
            os.path.basename(str(self.model.name_or_path)).strip()
            if hasattr(self.model, "name_or_path")
            else "default"
        )
        safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", base_id or "default")
        prev = int(self.latest_versions.get(safe_model_id, -1))
        new_version = prev + 1
        self._save_weights(safe_model_id, new_version)
        self.latest_versions[safe_model_id] = new_version
        self.current_version = new_version
        self._maybe_prune(safe_model_id)
        print(f"[ORCH] Auto-triggered optimizer step (disk-based) -> version {new_version}")
        return new_version

    def handle_timeout_batches(self, timeout_batch_ids: list, batch_timeout: float = 600.0) -> int:
        """
        Handle timeout batches by adjusting pending_batches.
        
        Args:
            timeout_batch_ids: List of batch IDs that have timed out
            batch_timeout: Timeout threshold in seconds
            
        Returns:
            Number of batches handled
        """
        handled = 0
        now = time.time()
        
        # Check all tracked batches for timeout
        timeout_batches = []
        for batch_id, batch_info in list(self.pending_batch_tracking.items()):
            timestamp = batch_info.get("timestamp", 0)
            elapsed = now - timestamp
            if elapsed > batch_timeout or batch_id in timeout_batch_ids:
                timeout_batches.append((batch_id, batch_info))
        
        # Adjust pending_batches for timeout batches (thread-safe)
        # Count each gradient upload as 1, so reduce by 1 for timeout
        # (microsteps are handled by trainer, orchestrator only counts uploads)
        for batch_id, batch_info in timeout_batches:
            with self._optimizer_lock:
                self._pending_batches = max(0, self._pending_batches - 1)
            handled += 1
            microsteps = batch_info.get("microsteps", 0)  # Keep for logging only
            print(f"[ORCH] Adjusted pending_batches for timeout batch {batch_id} (-1 gradient upload, was {microsteps} microsteps)")
            # Remove from tracking
            self.pending_batch_tracking.pop(batch_id, None)
        
        return handled
