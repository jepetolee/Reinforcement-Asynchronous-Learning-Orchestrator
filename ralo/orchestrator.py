import copy
import json
import os
import shutil
import re
import random
import io
import time
import threading
import sys
import logging
import math

import torch
import torch.nn as nn
import bottle
from bottle import Bottle, request, response
from transformers import AutoModelForCausalLM
import transformers
# Suppress transformers CUDA detection message for orchestrator (runs on CPU)
transformers.logging.set_verbosity_error()
# Disable HTTP access logs from wsgiref (will be controlled per-instance via log_http_access)
logging.getLogger('wsgiref').setLevel(logging.ERROR)
# Also disable the BaseHTTPRequestHandler logger if it exists
logging.getLogger('http.server').setLevel(logging.ERROR)

from .utils import json_to_bytes_list, bytes_list_to_json
from .orchestrator_components.problem_provider import ProblemProvider
from .orchestrator_components.sample_manager import SampleQueueManager
from .orchestrator_components.gradient_manager import GradientAggregator
from .logging import ExperimentLogger, NoOpLogger
from .eval.scaling import fit_scale_rl

# Try to import psutil for memory monitoring (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


class OrchestratorServer:
    """
    Central orchestrator that exposes the training queue, tracks incoming gradients,
    applies optimizer steps, and serves the latest weights to trainers/samplers.
    """

    def __init__(
        self,
        model_path,
        host="0.0.0.0",
        port=59888,
        update_steps=50,
        keep_last_versions=2,
        queue_size=1600,
        batch_timeout=3600.0,
        timeout_check_interval=60.0,
        problem_timeout=600.0,
        status_report_interval=30.0,
        lock_ttl=30.0,
        server_threads=10,
        lr=1e-6,
        train_data=None,
        epochs=1,
        logger: ExperimentLogger | None = None,
        # Evaluation
        evaluation=None,
        log_sample_upload=True,
        log_batch_dispatch=True,
        log_gradient_received=True,
        log_gradient_reassembled=True,
        log_gradient_chunks=True,
        log_optimizer_step=True,
        log_processing_gradient=True,
        log_status_report=True,
        log_http_access=True,
        chunk_timeout=600.0,
        max_concurrent_uploads=100,
        chunk_cleanup_interval=60.0,
        gradient_chunks_dir=None,
        gradient_storage_dir=None,
        max_gradient_disk_mb=1024000.0,  # 1TB default
        max_gradient_file_size_mb=10000.0,  # 10GB default - prevents OOM
    ):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
        self.logger = logger or NoOpLogger()

        if model_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.model = self.model.to(dtype=torch.bfloat16)
            # Keep model on CPU - we don't need GPU for optimizer steps anymore
            # All gradient accumulation and optimizer updates happen on CPU
            self.model = self.model.to("cpu")
            self.model.train()
            self.model.requires_grad_(True)
        else:
            self.model = None

        self.app = Bottle()
        self.problem_provider = ProblemProvider(train_data=train_data, epochs=epochs, problem_timeout=problem_timeout)
        self.sample_manager = SampleQueueManager(maxsize=queue_size, batch_timeout=batch_timeout)
        self.timeout_check_interval = timeout_check_interval
        self.status_report_interval = status_report_interval
        self.lock_ttl = lock_ttl
        self.optimizer = (
            torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.0)
            if self.model is not None
            else None
        )
        self.weights_dir = os.path.abspath(f"./orchestrator_weights_{self.port}")
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Evaluation job manager (simple in-memory queue + JSON results on disk)
        self.eval_enabled = bool(getattr(evaluation, "enabled", False)) if evaluation is not None else False
        self.eval_config = evaluation
        self.eval_namespace = getattr(evaluation, "wandb_namespace", "eval") if evaluation is not None else "eval"
        self.eval_shutdown_timeout = float(getattr(evaluation, "shutdown_timeout_sec", 300.0)) if evaluation is not None else 0.0
        self.eval_results_dir = os.path.abspath(f"./eval_results_{self.port}")
        os.makedirs(self.eval_results_dir, exist_ok=True)
        self._eval_lock = threading.Lock()
        self._eval_jobs: dict[str, dict] = {}  # job_id -> meta
        self._eval_queue: list[str] = []
        self._eval_running: dict[str, dict] = {}  # job_id -> {owner, expires_at}
        self._eval_fit_history: dict[str, dict[str, list[dict]]] = {}
        self._eval_fit_results: dict[str, dict[str, dict]] = {}
        self._compute_lock = threading.Lock()
        self._compute_totals: dict[str, dict[str, float | int | str | None]] = {}
        self._compute_version_totals: dict[int, dict] = {}
        self._init_compute_totals()

        # Disk-based gradient storage directories
        if gradient_chunks_dir is None:
            self.gradient_chunks_dir = os.path.abspath(f"./orchestrator_gradient_chunks_{self.port}")
        else:
            self.gradient_chunks_dir = os.path.abspath(gradient_chunks_dir)
        os.makedirs(self.gradient_chunks_dir, exist_ok=True)
        
        if gradient_storage_dir is None:
            self.gradient_storage_dir = os.path.abspath(f"./orchestrator_gradients_{self.port}")
        else:
            self.gradient_storage_dir = os.path.abspath(gradient_storage_dir)
        os.makedirs(self.gradient_storage_dir, exist_ok=True)
        
        self.max_gradient_disk_mb = max_gradient_disk_mb
        
        # Chunk storage limit: 1TB (1024000MB) for chunk files
        self.max_chunk_disk_mb = 1024000.0  # 1TB limit for chunk storage
        
        self.gradient_aggregator = GradientAggregator(
            self.model,
            self.optimizer,
            self.sample_manager,
            self.weights_dir,
            keep_last_versions=keep_last_versions,
            update_steps=update_steps,
        )
        self.current_version = -1
        self._should_stop = False
        self._end_received = False  # Track if sampler has sent end signal
        self._state_lock = threading.Lock()
        self.global_step = 0
        self.locks = {}
        self.counter = 0
        self.batch_size = update_steps
        # Gradient chunks storage: {upload_id: {"chunk_files": {idx: file_path}, "timestamp": time, "total_chunks": int}}
        self._gradient_chunks = {}  # Temporary storage for chunked gradient uploads (file paths only)
        self._chunk_lock = threading.Lock()  # Lock for gradient chunks access
        self._trainer_progress = {}  # Track trainer accumulation progress: {worker_id: {step_id, microstep, total, timestamp}}
        self.chunk_timeout = chunk_timeout
        self.max_concurrent_uploads = max_concurrent_uploads
        self.chunk_cleanup_interval = chunk_cleanup_interval
        self.mean_entropy = 0
        self.mean_length = 0
        self.sum_accuracy = 0.0
        self.latest_versions = self.gradient_aggregator.latest_versions
        self.keep_last_versions = keep_last_versions
        self.logger.init({
            "model_path": model_path,
            "port": port,
            "update_steps": update_steps,
        })

    def _weights_file_path(self, model_id, version):
        safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id)
        model_dir = os.path.join(self.weights_dir, safe_model_id)
        return os.path.join(model_dir, f"weights_v{int(version)}.pt")

    def _eval_version_dir(self, version: int) -> str:
        p = os.path.join(self.eval_results_dir, f"v{int(version)}")
        os.makedirs(p, exist_ok=True)
        return p

    def _schedule_eval_jobs_for_version(self, version: int):
        if not self.eval_enabled or self.eval_config is None:
            return
        try:
            benches = list(getattr(self.eval_config, "benchmarks", []) or [])
        except Exception:
            benches = []
        if not benches:
            return
        with self._eval_lock:
            for bench in benches:
                # bench can be dataclass or dict-like
                name = getattr(bench, "name", None) or (bench.get("name") if isinstance(bench, dict) else None)
                if not name:
                    continue
                job_id = f"eval_v{int(version)}_{name}"
                if job_id in self._eval_jobs:
                    # Already scheduled
                    continue
                # Convert to simple dict cfg for transport
                cfg = {
                    "name": name,
                    "loader": getattr(bench, "loader", None) or (bench.get("loader") if isinstance(bench, dict) else None),
                    "split": getattr(bench, "split", None) or (bench.get("split") if isinstance(bench, dict) else None),
                    "config": getattr(bench, "config", None) if not isinstance(bench, dict) else bench.get("config"),
                    "max_items": getattr(bench, "max_items", None) if not isinstance(bench, dict) else bench.get("max_items"),
                    "num_candidates": int(getattr(bench, "num_candidates", 1) if not isinstance(bench, dict) else bench.get("num_candidates", 1)),
                    "prompt_template": getattr(bench, "prompt_template", None) if not isinstance(bench, dict) else bench.get("prompt_template"),
                    "answer_extractor": getattr(bench, "answer_extractor", None) if not isinstance(bench, dict) else bench.get("answer_extractor"),
                    "metrics": list(getattr(bench, "metrics", []) if not isinstance(bench, dict) else bench.get("metrics", [])),
                }
                job = {
                    "job_id": job_id,
                    "version": int(version),
                    "benchmark_cfg": cfg,
                    "devices": list(getattr(self.eval_config, "devices", []) or []),
                    "created_at": time.time(),
                    "state": "pending",
                    "owner": None,
                    "expires_at": None,
                }
                self._eval_jobs[job_id] = job
                self._eval_queue.append(job_id)
                # Ensure version dir exists early
                self._eval_version_dir(version)
        self._capture_compute_version(int(version))

    def _init_compute_totals(self):
        with self._compute_lock:
            base_roles = ("actor", "learner", "evaluator")
            self._compute_totals = {}
            for role in base_roles:
                self._compute_totals[role] = {
                    "gpu_seconds": 0.0,
                    "gpu_hours": 0.0,
                    "tokens": 0,
                    "reports": 0,
                    "device_seconds": 0.0,
                    "last_report_at": None,
                    "last_worker": None,
                }

    def _update_compute_usage(
        self,
        role: str,
        gpu_seconds: float,
        tokens: int,
        worker_id: str,
        device_count: int,
        metadata: dict | None = None,
    ):
        role_key = (role or "unknown").strip().lower() or "unknown"
        payload = {
            "gpu_seconds": 0.0,
            "gpu_hours": 0.0,
            "tokens": 0,
            "reports": 0,
            "device_seconds": 0.0,
            "last_report_at": None,
            "last_worker": None,
        }
        with self._compute_lock:
            entry = self._compute_totals.setdefault(role_key, copy.deepcopy(payload))
            entry["gpu_seconds"] = float(entry.get("gpu_seconds", 0.0)) + gpu_seconds
            entry["tokens"] = int(entry.get("tokens", 0)) + tokens
            entry["reports"] = int(entry.get("reports", 0)) + 1
            entry["device_seconds"] = float(entry.get("device_seconds", 0.0)) + max(1, device_count) * gpu_seconds
            entry["last_report_at"] = time.time()
            entry["last_worker"] = worker_id
            if metadata:
                entry["last_metadata"] = metadata
            entry["gpu_hours"] = float(entry.get("gpu_seconds", 0.0)) / 3600.0

    def _snapshot_compute_totals(self) -> dict:
        with self._compute_lock:
            snapshot = copy.deepcopy(self._compute_totals)
        total_gpu_seconds = 0.0
        total_tokens = 0
        for role_data in snapshot.values():
            if isinstance(role_data, dict):
                seconds = float(role_data.get("gpu_seconds", 0.0))
                role_data["gpu_hours"] = seconds / 3600.0
                total_gpu_seconds += seconds
                total_tokens += int(role_data.get("tokens", 0))
        snapshot["total"] = {
            "gpu_seconds": total_gpu_seconds,
            "gpu_hours": total_gpu_seconds / 3600.0 if total_gpu_seconds else 0.0,
            "tokens": total_tokens,
        }
        return snapshot

    def _capture_compute_version(self, version: int):
        if version is None:
            return
        snapshot = self._snapshot_compute_totals()
        with self._compute_lock:
            self._compute_version_totals[int(version)] = snapshot

    def _record_eval_measurement(self, bench_name: str, metrics: dict, compute_snapshot: dict):
        if not self.eval_enabled or not isinstance(metrics, dict):
            return
        gpu_hours = 0.0
        try:
            gpu_hours = float(compute_snapshot.get("gpu_hours_total", 0.0) or 0.0)
        except Exception:
            gpu_hours = 0.0
        if gpu_hours > 0:
            compute_scalar = gpu_hours
        else:
            try:
                compute_scalar = float(
                    compute_snapshot.get("optimizer_step")
                    or compute_snapshot.get("version")
                    or 1
                )
            except Exception:
                compute_scalar = float(compute_snapshot.get("version", 1))
        compute_scalar = max(1.0, compute_scalar)
        timestamp = float(compute_snapshot.get("timestamp", time.time()))
        pending_fit: list[tuple[str, list[dict]]] = []
        with self._eval_lock:
            bench_hist = self._eval_fit_history.setdefault(bench_name, {})
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                history = bench_hist.setdefault(metric_name, [])
                history.append(
                    {
                        "compute": compute_scalar,
                        "reward": float(value),
                        "version": compute_snapshot.get("version"),
                        "timestamp": timestamp,
                        "gpu_hours": gpu_hours,
                        "tokens": compute_snapshot.get("tokens_total"),
                    }
                )
                if len(history) > 256:
                    del history[0 : len(history) - 256]
                if getattr(self.eval_config, "enable_scale_fit", False):
                    pending_fit.append((metric_name, list(history)))
        if getattr(self.eval_config, "enable_scale_fit", False):
            for metric_name, entries in pending_fit:
                self._maybe_fit_eval_curve(bench_name, metric_name, entries)

    def _maybe_fit_eval_curve(self, bench_name: str, metric_name: str, entries: list | None = None):
        if entries is None:
            with self._eval_lock:
                entries = list(self._eval_fit_history.get(bench_name, {}).get(metric_name, []))
        if not entries or len(entries) < 4:
            return
        data = [
            (item["compute"], item["reward"])
            for item in entries
            if isinstance(item.get("compute"), (int, float))
            and isinstance(item.get("reward"), (int, float))
            and item["compute"] > 0
        ]
        if len(data) < 4:
            return
        fit = fit_scale_rl(data)
        if not fit:
            return
        latest_version = None
        for item in reversed(entries):
            if item.get("version") is not None:
                latest_version = int(item["version"])
                break
        fit.update(
            {
                "benchmark": bench_name,
                "metric": metric_name,
                "updated_at": time.time(),
                "num_points": len(data),
                "latest_version": latest_version,
            }
        )
        with self._eval_lock:
            bench_results = self._eval_fit_results.setdefault(bench_name, {})
            bench_results[metric_name] = fit
        self._log_eval_fit_to_wandb(bench_name, metric_name, fit)
        self._persist_eval_fit_results()

    def _log_eval_fit_to_wandb(self, bench_name: str, metric_name: str, fit: dict):
        try:
            payload = {}
            for key in ("R0", "A", "C_mid", "B", "loss"):
                val = fit.get(key)
                if isinstance(val, (int, float)) and math.isfinite(val):
                    payload[f"{self.eval_namespace}/{bench_name}/{metric_name}/fit_{key.lower()}"] = float(val)
            latest_version = fit.get("latest_version")
            if latest_version is not None:
                payload["version"] = int(latest_version)
            if payload:
                self.logger.log(payload)
        except Exception:
            pass

    def _persist_eval_fit_results(self):
        try:
            summary_path = os.path.join(self.eval_results_dir, "eval_fit_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                summary = {
                    "results": self._eval_fit_results,
                    "compute_totals": self._snapshot_compute_totals(),
                }
                with self._compute_lock:
                    summary["per_version"] = copy.deepcopy(self._compute_version_totals)
                json.dump(summary, f)
        except Exception as e:
            print(f"[ORCH] Failed to write eval fit summary: {e}")

    def _wait_for_eval_completion(self, timeout: float | None = None):
        """
        Block until all evaluation jobs are reported or timeout reached.
        """
        if not self.eval_enabled:
            return
        deadline = None if timeout is None or timeout <= 0 else time.time() + timeout
        while True:
            with self._eval_lock:
                remaining = [
                    jid for jid, job in self._eval_jobs.items()
                    if job.get("state") in {"pending", "running"}
                ]
                queue_len = len(self._eval_queue)
            if not remaining and queue_len == 0:
                return
            if deadline is not None and time.time() >= deadline:
                print(f"[ORCH] Timeout while waiting for evaluation jobs to finish ({len(remaining)} running/pending, queue={queue_len}).")
                return
            time.sleep(2.0)

    def _scan_latest_version(self, model_id):
        try:
            safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id)
            model_dir = os.path.join(self.weights_dir, safe_model_id)
            if not os.path.isdir(model_dir):
                return -1
            best = -1
            for name in os.listdir(model_dir):
                if name.startswith("weights_v") and name.endswith(".pt"):
                    m = re.match(r"^weights_v(\d+)\.pt$", name)
                    if m:
                        ver = int(m.group(1))
                        if ver > best:
                            best = ver
            return best
        except Exception:
            return -1

    def _finalize_pending_gradients(self):
        version = self.gradient_aggregator.finalize()
        if version is not None:
            self.current_version = version
            self.latest_versions = self.gradient_aggregator.latest_versions
            # Schedule evaluation for the new version
            try:
                self._schedule_eval_jobs_for_version(int(version))
            except Exception as _:
                pass

    def _lock_cleanup(self):
        now = time.time()
        expired = []
        for key, meta in list(self.locks.items()):
            if meta["expires_at"] <= now:
                expired.append(key)
        for key in expired:
            self.locks.pop(key, None)
    
    def _cleanup_stale_chunks(self):
        """Remove stale gradient chunks that haven't been finalized within timeout."""
        current_time = time.time()
        stale_uploads = []
        chunk_total_mb = 0.0
        
        with self._chunk_lock:
            # Calculate total chunk disk usage
            for upload_id, chunk_data in list(self._gradient_chunks.items()):
                timestamp = chunk_data.get("timestamp", 0)
                chunk_files = chunk_data.get("chunk_files", {})
                
                # Calculate disk usage for this upload
                chunk_mb = 0.0
                for chunk_file in chunk_files.values():
                    if os.path.exists(chunk_file):
                        chunk_mb += os.path.getsize(chunk_file) / (1024 * 1024)
                chunk_total_mb += chunk_mb
                
                # Check if stale (only mark as stale if timeout AND incomplete)
                expected_total = chunk_data.get("total_chunks", len(chunk_files))
                received_count = len(chunk_files)
                is_complete = received_count >= expected_total
                is_stale = timestamp + self.chunk_timeout < current_time
                
                # Remove stale chunks (complete or incomplete after timeout)
                if is_stale:
                    stale_uploads.append(upload_id)
                    if not is_complete:
                        # Log warning for stale incomplete uploads
                        parts = upload_id.split('_')
                        worker_info = '_'.join(parts[:-2]) if len(parts) >= 3 else upload_id
                        print(f"[ORCH] WARNING: Stale incomplete upload {upload_id} from {worker_info} ({received_count}/{expected_total} chunks, {current_time - timestamp:.1f}s old)")
            
            # Remove stale uploads and their files
            for upload_id in stale_uploads:
                chunk_data = self._gradient_chunks.pop(upload_id, None)
                if chunk_data:
                    chunk_files = chunk_data.get("chunk_files", {})
                    chunk_mb = 0.0
                    for chunk_file in chunk_files.values():
                        try:
                            if os.path.exists(chunk_file):
                                file_size = os.path.getsize(chunk_file) / (1024 * 1024)
                                chunk_mb += file_size
                                os.remove(chunk_file)
                        except (OSError, FileNotFoundError) as e:
                            # File may have been already deleted by another process
                            pass
                    chunk_total_mb -= chunk_mb
                    parts = upload_id.split('_')
                    worker_info = '_'.join(parts[:-2]) if len(parts) >= 3 else upload_id
                    print(f"[ORCH] Cleaned up stale gradient chunks: {upload_id} ({chunk_mb:.1f}MB) from {worker_info}")
            
            # Check chunk storage disk limit (1TB)
            if chunk_total_mb > self.max_chunk_disk_mb:
                # Remove oldest chunks if over limit
                all_chunks = []
                for upload_id, chunk_data in list(self._gradient_chunks.items()):
                    chunk_files = chunk_data.get("chunk_files", {})
                    timestamp = chunk_data.get("timestamp", 0)
                    for chunk_file in chunk_files.values():
                        if os.path.exists(chunk_file):
                            file_size_mb = os.path.getsize(chunk_file) / (1024 * 1024)
                            all_chunks.append((chunk_file, timestamp, file_size_mb, upload_id))
                
                # Sort by timestamp (oldest first)
                all_chunks.sort(key=lambda x: x[1])
                removed_mb = 0.0
                while chunk_total_mb > self.max_chunk_disk_mb and all_chunks:
                    chunk_file, _, file_size_mb, upload_id = all_chunks.pop(0)
                    try:
                        if os.path.exists(chunk_file):
                            os.remove(chunk_file)
                            chunk_total_mb -= file_size_mb
                            removed_mb += file_size_mb
                            # Remove from chunk_data
                            if upload_id in self._gradient_chunks:
                                chunk_data = self._gradient_chunks[upload_id]
                                chunk_files = chunk_data.get("chunk_files", {})
                                # Remove the chunk file from dict
                                for idx, path in list(chunk_files.items()):
                                    if path == chunk_file:
                                        chunk_files.pop(idx)
                                        break
                    except (OSError, FileNotFoundError):
                        pass
                if removed_mb > 0:
                    print(f"[ORCH] Removed {removed_mb:.1f}MB of old chunk files to stay under 1TB limit")
        
        # Check disk limit (for gradient storage directory)
        gradient_files = []
        gradient_total_mb = 0.0
        if os.path.exists(self.gradient_storage_dir):
            for filename in os.listdir(self.gradient_storage_dir):
                if filename.endswith('.bin'):
                    filepath = os.path.join(self.gradient_storage_dir, filename)
                    try:
                        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                        gradient_total_mb += file_size_mb
                        gradient_files.append((filepath, os.path.getmtime(filepath), file_size_mb))
                    except (OSError, FileNotFoundError):
                        pass
        
        # Check gradient storage disk limit
        if gradient_total_mb > self.max_gradient_disk_mb:
            # Remove oldest gradient files if over limit
            gradient_files.sort(key=lambda x: x[1])  # Sort by modification time
            removed_mb = 0.0
            while gradient_total_mb > self.max_gradient_disk_mb and gradient_files:
                filepath, _, file_size_mb = gradient_files.pop(0)
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        # Also remove metadata file if exists
                        meta_file = filepath.replace('.bin', '.meta.json')
                        if os.path.exists(meta_file):
                            os.remove(meta_file)
                        gradient_total_mb -= file_size_mb
                        removed_mb += file_size_mb
                except OSError:
                    pass
            if removed_mb > 0:
                print(f"[ORCH] Removed {removed_mb:.1f}MB of old gradient files to stay under disk limit")
        
        return len(stale_uploads), chunk_total_mb + gradient_total_mb
    
    def _check_memory_usage(self):
        """
        Check current memory usage and return memory stats.
        Returns: (process_memory_mb, available_memory_mb, memory_percent, is_low_memory)
        """
        if not PSUTIL_AVAILABLE:
            return None, None, None, False
        
        try:
            process = psutil.Process()
            process_mem = process.memory_info()
            process_mb = process_mem.rss / (1024 * 1024)
            
            system_mem = psutil.virtual_memory()
            available_mb = system_mem.available / (1024 * 1024)
            memory_percent = system_mem.percent
            
            # Consider low memory if available < 20GB
            is_low_memory = available_mb < 20480
            
            return process_mb, available_mb, memory_percent, is_low_memory
        except Exception as e:
            # Fail silently if memory check fails
            return None, None, None, False
    
    def _log_chunk_status(self):
        """Log current gradient chunk status for monitoring."""
        with self._chunk_lock:
            num_uploads = len(self._gradient_chunks)
            total_chunks = sum(len(data.get("chunk_files", {})) for data in self._gradient_chunks.values())
            total_mb = 0.0
            for data in self._gradient_chunks.values():
                chunk_files = data.get("chunk_files", {})
                for chunk_file in chunk_files.values():
                    if os.path.exists(chunk_file):
                        total_mb += os.path.getsize(chunk_file) / (1024 * 1024)
        
        # Warning if too many concurrent uploads
        if num_uploads > self.max_concurrent_uploads * 0.8:
            print(f"[ORCH WARNING] High gradient chunk count: {num_uploads} uploads, {total_chunks} chunks, {total_mb:.1f}MB")
        
        return num_uploads, total_chunks, total_mb

    def run_server(self):
        @self.app.route("/upload", method="POST")
        def do_upload():
            try:
                blob = request.body.read()
            except (ConnectionResetError, OSError) as e:
                response.status = 500
                response.content_type = 'application/json'
                return json.dumps({"ok": False, "error": f"connection error: {e}"})
            if not blob:
                return json.dumps({"ok": False, "error": "empty"})
            try:
                payload = bytes_list_to_json(blob)
            except Exception:
                response.status = 400
                return json.dumps({"ok": False, "error": "invalid payload"})

            result = self.sample_manager.enqueue(payload)
            # Persist uploaded samples on orchestrator side (append-only JSONL, tensors stripped/converted)
            try:
                if "end" not in payload:
                    from pathlib import Path
                    samples_dir = Path(getattr(self, "samples_dir", f"./orchestrator_samples_{self.port}"))
                    samples_dir.mkdir(parents=True, exist_ok=True)
                    safe_payload = dict(payload)
                    # Drop heavy/tensor fields we never want to persist
                    safe_payload.pop("inputs", None)
                    safe_payload.pop("#computed_logits", None)
                    # Recursively convert tensors to lists for JSON
                    def _json_safe(obj):
                        try:
                            import torch
                            if isinstance(obj, torch.Tensor):
                                if obj.numel() == 1:
                                    return obj.item()
                                return obj.detach().cpu().tolist()
                        except Exception:
                            pass
                        if isinstance(obj, dict):
                            return {k: _json_safe(v) for k, v in obj.items()}
                        if isinstance(obj, (list, tuple)):
                            return [_json_safe(x) for x in obj]
                        if isinstance(obj, (str, int, float, bool)) or obj is None:
                            return obj
                        # Fallback to string representation
                        return str(obj)
                    safe_payload = _json_safe(safe_payload)
                    # Append JSONL
                    import json as _json, time as _time
                    fname = samples_dir / f"samples_{int(_time.time())}.jsonl"
                    with fname.open("a", encoding="utf-8") as f:
                        f.write(_json.dumps(safe_payload, ensure_ascii=False) + "\n")
            except Exception as save_err:
                if self.log_sample_upload:
                    print(f"[ORCH] Warning: failed to persist samples: {save_err}")
            # Mark problem as completed if problem_id is present
            problem_id = payload.get("_problem_id")
            if problem_id:
                self.problem_provider.mark_problem_completed(problem_id)
            with self._state_lock:
                if "end" in payload:
                    self._end_received = True
                    print("[ORCH] Received end signal from sampler")
            if "end" not in payload:
                try:
                    mean_entropy = sum(payload.get("entropy", [])) / max(
                        1, len(payload.get("entropy", []))
                    )
                except Exception:
                    mean_entropy = 0.0
                try:
                    mean_length = sum(payload.get("mean_length", [])) / max(
                        1, len(payload.get("mean_length", []))
                    )
                except Exception:
                    mean_length = 0.0
                accuracy_rate = payload.get("Accuracy", None)
                self.counter += 1
                self.mean_entropy += float(mean_entropy)
                self.mean_length += float(mean_length)
                try:
                    self.sum_accuracy += (
                        float(accuracy_rate)
                        if isinstance(accuracy_rate, (int, float))
                        else 0.0
                    )
                except Exception:
                    pass
                # Defer logging until optimizer step; just accumulate here.
            remain = result.get("remain_cnt", 0)
            # Only print every 10th sample to reduce noise
            if self.log_sample_upload and self.sample_manager.total_enqueued % 10 == 0:
                print(f"[ORCH] Sample uploaded | Queue: {remain} samples | Total enqueued: {self.sample_manager.total_enqueued}")
            return json.dumps(result)

        @self.app.route("/get", method="GET")
        def do_get():
            if self._should_stop:
                return json_to_bytes_list({"end": 1})
            batch_info = self.sample_manager.begin_batch()
            if batch_info is None:
                with self._state_lock:
                    if self._end_received:
                        if self.gradient_aggregator.pending_batches > 0:
                            print(
                                f"[ORCH] Queue empty with {self.gradient_aggregator.pending_batches} pending batches; finalizing optimizer step"
                            )
                            self._finalize_pending_gradients()
                        self._should_stop = True
                        return json_to_bytes_list({"end": 1})
                return b"empty"

            item = batch_info["batch"]
            remain = batch_info["queue_size"]
            processing = batch_info["processing"]
            # Only print every 5th batch to reduce noise
            if self.log_batch_dispatch and self.sample_manager.total_dequeued % 5 == 0:
                print(f"[ORCH] Batch dispatched to trainer | Queue: {remain} | Processing: {processing} | Total dequeued: {self.sample_manager.total_dequeued}")
            response.content_type = "application/octet-stream"
            return json_to_bytes_list(item)

        @self.app.route("/stop", method="POST")
        def do_stop():
            self._should_stop = True
            self.sample_manager.enqueue({"end": 1})
            if self.eval_enabled:
                print("[ORCH] Waiting for evaluation jobs to finish before stopping...")
                self._wait_for_eval_completion(timeout=self.eval_shutdown_timeout or None)
            self.logger.close()
            print("[ORCH] Received stop signal")
            return json.dumps({"ok": True, "message": "Stop signal received"})

        @self.app.route("/problem/get", method="GET")
        def problem_get():
            """Sampler가 문제 데이터를 요청하는 엔드포인트"""
            payload = self.problem_provider.get_problem()
            return json.dumps(payload)

        @self.app.route("/trainer/register", method="POST")
        def trainer_register():
            payload = request.json or {}
            model_id = payload.get("model_id", "default")
            latest = int(self.latest_versions.get(model_id, -1))
            weights_path = (
                self._weights_file_path(model_id, latest) if latest >= 0 else None
            )
            if latest < 0 or not (weights_path and os.path.exists(weights_path)):
                latest = self._scan_latest_version(model_id)
                self.latest_versions[model_id] = latest
            info = {
                "ok": True,
                "model_id": model_id,
                "latest_version": int(latest),
                "update_steps": int(self.batch_size),
                "orchestrator_url": f"http://{self.host}:{self.port}",
            }
            return json.dumps(info)

        @self.app.route("/test/post", method="POST")
        def test_post():
            print(f"[ORCH] /test/post called")
            response.content_type = 'application/json'
            return json.dumps({"ok": True, "message": "POST works"})

        @self.app.route("/gradient/upload_chunk", method="POST")
        def gradient_upload_chunk():
            try:
                upload_id = request.headers.get("X-Upload-ID")
                chunk_idx = int(request.headers.get("X-Chunk-Index", -1))
                total_chunks = int(request.headers.get("X-Total-Chunks", -1))
                
                if not upload_id or chunk_idx < 0 or total_chunks <= 0:
                    response.status = 400
                    response.content_type = 'application/json'
                    return json.dumps({"ok": False, "error": "missing required headers"})
                
                # Read request body with enhanced error handling
                try:
                    blob = request.body.read()
                except (ConnectionResetError, OSError) as e:
                    response.status = 500
                    response.content_type = 'application/json'
                    err_msg = f"connection error reading body: {e}"
                    print(f"[ORCH] Chunk upload error: {err_msg}")
                    return json.dumps({"ok": False, "error": err_msg})
                except Exception as e:
                    response.status = 500
                    response.content_type = 'application/json'
                    err_msg = f"error reading body: {e}"
                    print(f"[ORCH] Chunk upload error: {err_msg}")
                    return json.dumps({"ok": False, "error": err_msg})
                
                if not blob:
                    response.status = 400
                    response.content_type = 'application/json'
                    return json.dumps({"ok": False, "error": "empty chunk data"})
                
                current_time = time.time()
                
                with self._chunk_lock:
                    # Check concurrent upload limit
                    if len(self._gradient_chunks) >= self.max_concurrent_uploads:
                        response.status = 503
                        response.content_type = 'application/json'
                        return json.dumps({"ok": False, "error": "too many concurrent uploads"})
                    
                    if upload_id not in self._gradient_chunks:
                        self._gradient_chunks[upload_id] = {
                            "chunk_files": {},
                            "timestamp": current_time,
                            "total_chunks": total_chunks
                        }
                        # Extract worker info from upload_id (format: host-rank_step_timestamp)
                        parts = upload_id.split('_')
                        worker_info = '_'.join(parts[:-2]) if len(parts) >= 3 else upload_id
                        if self.log_gradient_chunks:
                            print(f"[ORCH] Receiving gradient chunks from {worker_info} ({total_chunks} chunks, ~{len(blob) * total_chunks / (1024*1024):.1f}MB)")
                    
                    # Save chunk to disk instead of RAM
                    chunk_file_path = os.path.join(self.gradient_chunks_dir, f"{upload_id}_chunk_{chunk_idx}.bin")
                    with open(chunk_file_path, 'wb') as f:
                        f.write(blob)
                    
                    # Update timestamp on each chunk receive
                    self._gradient_chunks[upload_id]["timestamp"] = current_time
                    self._gradient_chunks[upload_id]["chunk_files"][chunk_idx] = chunk_file_path
                
                # Log progress every 10% or at completion
                with self._chunk_lock:
                    received = len(self._gradient_chunks[upload_id]["chunk_files"])
                if self.log_gradient_chunks and (received == total_chunks or received % max(1, total_chunks // 10) == 0):
                    progress_pct = (received / total_chunks * 100) if total_chunks > 0 else 0
                    print(f"[ORCH] Gradient chunks: {received}/{total_chunks} ({progress_pct:.0f}%)")
                
                return json.dumps({"ok": True, "chunk_received": chunk_idx})
            except (ConnectionResetError, OSError) as err:
                response.status = 500
                response.content_type = 'application/json'
                err_msg = f"chunk upload failed: connection error: {err}"
                print(f"[ORCH] Chunk upload error: {err_msg}")
                return json.dumps({"ok": False, "error": err_msg})
            except Exception as err:
                response.status = 400
                response.content_type = 'application/json'
                err_msg = f"chunk upload failed: {err}"
                print(f"[ORCH] Chunk upload error: {err_msg}")
                import traceback
                traceback.print_exc()
                return json.dumps({"ok": False, "error": err_msg})

        @self.app.route("/gradient/upload_finalize", method="POST")
        def gradient_upload_finalize():
            print(f"[ORCH] /gradient/upload_finalize called!")
            import sys
            sys.stdout.flush()
            try:
                payload_data = request.json
                upload_id = payload_data.get("upload_id")
                
                if not upload_id:
                    response.status = 400
                    response.content_type = 'application/json'
                    return json.dumps({"ok": False, "error": "missing upload_id"})
                
                with self._chunk_lock:
                    if upload_id not in self._gradient_chunks:
                        response.status = 400
                        response.content_type = 'application/json'
                        return json.dumps({"ok": False, "error": "invalid upload_id"})
                    
                    chunk_data = self._gradient_chunks[upload_id]
                    chunk_files = chunk_data.get("chunk_files", {})
                    chunk_indices = sorted(chunk_files.keys())
                    
                    # Check if all chunks received
                    expected_total = chunk_data.get("total_chunks", len(chunk_indices))
                    if len(chunk_indices) < expected_total:
                        response.status = 400
                        response.content_type = 'application/json'
                        return json.dumps({"ok": False, "error": f"incomplete chunks: {len(chunk_indices)}/{expected_total}"})
                
                # Reassemble chunks by streaming directly to disk (avoid loading all into memory)
                gradient_file = os.path.join(self.gradient_storage_dir, f"gradient_{upload_id}.bin")
                total_size = 0
                with open(gradient_file, 'wb') as out_f:
                    for idx in chunk_indices:
                        chunk_file = chunk_files.get(idx)
                        if chunk_file and os.path.exists(chunk_file):
                            # Get chunk size before streaming
                            chunk_size = os.path.getsize(chunk_file)
                            with open(chunk_file, 'rb') as in_f:
                                # Stream chunk directly to output file
                                shutil.copyfileobj(in_f, out_f)
                            total_size += chunk_size
                
                size_mb = total_size / (1024 * 1024)
                
                # Check file size limit to prevent OOM
                if size_mb > self.max_gradient_file_size_mb:
                    # Delete the assembled file and chunks
                    try:
                        if os.path.exists(gradient_file):
                            os.remove(gradient_file)
                    except (OSError, FileNotFoundError):
                        pass
                    for chunk_file in chunk_files.values():
                        try:
                            if os.path.exists(chunk_file):
                                os.remove(chunk_file)
                        except (OSError, FileNotFoundError):
                            pass
                    # Clean up from memory
                    with self._chunk_lock:
                        self._gradient_chunks.pop(upload_id, None)
                    
                    error_msg = f"Gradient file too large: {size_mb:.1f}MB (max: {self.max_gradient_file_size_mb:.1f}MB). Rejected to prevent OOM."
                    print(f"[ORCH ERROR] {error_msg}")
                    sys.stdout.flush()
                    response.status = 400
                    response.content_type = 'application/json'
                    return json.dumps({"ok": False, "error": error_msg})
                
                if self.log_gradient_reassembled:
                    print(f"[ORCH] Reassembled gradient: {len(chunk_files)} chunks → {size_mb:.1f}MB (streamed)")
                    sys.stdout.flush()
                
                # Delete chunk files (no longer needed after reassembly)
                for chunk_file in chunk_files.values():
                    try:
                        if os.path.exists(chunk_file):
                            os.remove(chunk_file)
                    except (OSError, FileNotFoundError):
                        pass
                
                # Save metadata (temporary, will be updated after parsing)
                metadata = {
                    "worker_id": None,  # Will be filled after parsing
                    "step_id": None,
                    "upload_id": upload_id,
                    "timestamp": time.time(),
                }
                metadata_file = os.path.join(self.gradient_storage_dir, f"gradient_{upload_id}.meta.json")
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f)
                
                # Clean up chunks from memory (do this before reading blob to reduce memory pressure)
                with self._chunk_lock:
                    self._gradient_chunks.pop(upload_id, None)
                
                # Warn for large files that might cause memory issues
                if size_mb > 5000:  # Warn for files > 5GB
                    print(f"[ORCH WARNING] Large gradient file ({size_mb:.1f}MB) - memory usage may be high during parsing")
                    sys.stdout.flush()
                
                # Check available memory before reading large file
                process_mb, available_mb, memory_percent, is_low_memory = self._check_memory_usage()
                if available_mb is not None:
                    # Estimate memory needed: file size + 2x for parsing/deserialization overhead
                    estimated_need_mb = size_mb * 3
                    if available_mb < estimated_need_mb:
                        # Clean up and reject
                        try:
                            if os.path.exists(gradient_file):
                                os.remove(gradient_file)
                        except (OSError, FileNotFoundError):
                            pass
                        error_msg = f"Insufficient memory to process gradient: {size_mb:.1f}MB file needs ~{estimated_need_mb:.1f}MB, but only {available_mb:.1f}MB available"
                        print(f"[ORCH ERROR] {error_msg}")
                        sys.stdout.flush()
                        response.status = 500
                        response.content_type = 'application/json'
                        return json.dumps({"ok": False, "error": error_msg})
                    elif is_low_memory:
                        print(f"[ORCH WARNING] Low available memory ({available_mb:.1f}MB) - processing large gradient file ({size_mb:.1f}MB) may cause issues")
                        sys.stdout.flush()
                
                try:
                    with open(gradient_file, 'rb') as f:
                        blob = f.read()
                except MemoryError:
                    response.status = 500
                    response.content_type = 'application/json'
                    error_msg = json.dumps({"ok": False, "error": f"out of memory reading gradient file ({size_mb:.1f}MB)"})
                    print(f"[ORCH ERROR] Failed to read gradient file due to insufficient memory: {size_mb:.1f}MB")
                    return error_msg
                
                # Parse to get metadata
                try:
                    payload = bytes_list_to_json(blob)
                    # Clear blob from memory as soon as possible after parsing
                    del blob
                except MemoryError:
                    response.status = 500
                    response.content_type = 'application/json'
                    error_msg = json.dumps({"ok": False, "error": f"out of memory parsing gradient ({size_mb:.1f}MB)"})
                    print(f"[ORCH ERROR] Failed to parse gradient due to insufficient memory: {size_mb:.1f}MB")
                    return error_msg
                except Exception as err:
                    response.status = 400
                    response.content_type = 'application/json'
                    error_msg = json.dumps({"ok": False, "error": f"invalid payload: {err}"})
                    print(f"[ORCH] /gradient/upload_finalize: invalid payload: {err}")
                    import traceback
                    traceback.print_exc()
                    return error_msg

                worker_id = payload.get("worker_id", "unknown")
                step_id = payload.get("step_id", -1)
                if self.log_processing_gradient:
                    print(f"[ORCH] Processing reassembled gradient from {worker_id} (step {step_id})")

                # Update metadata with parsed information
                metadata["worker_id"] = worker_id
                metadata["step_id"] = step_id
                metadata["microsteps"] = payload.get("microsteps", 1)
                # Include batch_id and batch_ids for completion tracking
                metadata["batch_id"] = payload.get("_batch_id")
                metadata["batch_ids"] = payload.get("_batch_ids")
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f)

                try:
                    # Use file-based ingestion for memory efficiency
                    result = self.gradient_aggregator.ingest_file(gradient_file, metadata)
                except Exception as err:
                    response.status = 400
                    response.content_type = 'application/json'
                    err_msg = f"gradient ingest failed: {err}"
                    print(f"[ORCH] Gradient ingest error from {worker_id} (step {step_id}): {err}")
                    import traceback
                    traceback.print_exc()
                    error_response = json.dumps({"ok": False, "error": err_msg})
                    return error_response
                
                pending = result.get("pending_batches", 0)
                total_grads = self.gradient_aggregator.total_gradients
                progress_pct = (pending / self.batch_size * 100) if self.batch_size > 0 else 0
                new_version = result.get("version")
                
                # Update orchestrator's current_version if optimizer step completed
                if new_version is not None:
                    self.current_version = new_version
                    self.latest_versions = self.gradient_aggregator.latest_versions
                    if self.log_optimizer_step:
                        print(f"[ORCH] ✓ Optimizer step completed -> version {new_version} | "
                              f"Total gradients: {total_grads} | Global step: {self.global_step}")
                # Schedule eval for new version
                try:
                    ver_to_schedule = new_version if new_version is not None else self.current_version
                    if ver_to_schedule is not None:
                        self._schedule_eval_jobs_for_version(int(ver_to_schedule))
                except Exception:
                    pass
                else:
                    # Also check if gradient_aggregator.current_version was updated asynchronously
                    if self.gradient_aggregator.current_version > self.current_version:
                        self.current_version = self.gradient_aggregator.current_version
                        self.latest_versions = self.gradient_aggregator.latest_versions
                    if self.log_gradient_received:
                        print(f"[ORCH] Gradient received from {worker_id} (step {step_id}) | "
                              f"Pending: {pending}/{self.batch_size} ({progress_pct:.1f}%) | "
                              f"Total: {total_grads} gradients | Queue: {self.sample_manager.train_queue.qsize()} samples")
                # If optimizer step occurred, flush aggregated metrics to logger
                stepped = result.get("stepped", False)
                if stepped and new_version is not None:
                    # Always log metrics when optimizer step completes (even if counter is 0)
                    if self.counter > 0:
                        metrics = {
                            "accuracy_avg": self.sum_accuracy / self.counter,
                            "mean_entropy": self.mean_entropy / self.counter,
                            "mean_length": self.mean_length / self.counter,
                            "version": new_version,
                        }
                    else:
                        # Use zeros if no metrics collected
                        metrics = {
                            "accuracy_avg": 0.0,
                            "mean_entropy": 0.0,
                            "mean_length": 0.0,
                            "version": new_version,
                        }
                    if self.log_optimizer_step:
                        print(f"[ORCH] Logging metrics to wandb (v{new_version}): "
                              f"accuracy_avg={metrics['accuracy_avg']:.4f}, "
                              f"mean_entropy={metrics['mean_entropy']:.4f}, "
                              f"mean_length={metrics['mean_length']:.4f}, "
                              f"counter={self.counter}")
                    try:
                        self.logger.log(metrics)
                    except Exception as e:
                        print(f"[ORCH] Failed to log metrics: {e}")
                    # Reset accumulators after logging
                    self.mean_entropy = 0
                    self.mean_length = 0
                    self.counter = 0
                    self.sum_accuracy = 0.0
                elif stepped and new_version is None:
                    if self.log_optimizer_step:
                        print(f"[ORCH] WARNING: Optimizer step completed but new_version is None (counter={self.counter})")

                return json.dumps(result)
            except Exception as outer_err:
                response.status = 500
                response.content_type = 'application/json'
                err_msg = f"unexpected error in gradient_upload_finalize: {outer_err}"
                print(f"[ORCH] Unexpected error in /gradient/upload_finalize: {outer_err}")
                import traceback
                traceback.print_exc()
                return json.dumps({"ok": False, "error": err_msg})

        @self.app.route("/optimizer/step", method="POST")
        def optimizer_step():
            version = self._finalize_pending_gradients()
            return json.dumps({"ok": True, "version": int(version) if version is not None else None})

        @self.app.route("/trainer/heartbeat", method="POST")
        def trainer_heartbeat():
            try:
                payload = request.json
                worker_id = payload.get("worker_id", "unknown")
                step_id = payload.get("step_id", -1)
                microstep = payload.get("microstep", 0)
                total_microsteps = payload.get("total_microsteps", 1)
                
                import time
                self._trainer_progress[worker_id] = {
                    "step_id": step_id,
                    "microstep": microstep,
                    "total": total_microsteps,
                    "timestamp": time.time(),
                    "progress_pct": (microstep / total_microsteps * 100) if total_microsteps > 0 else 0
                }
                
                return json.dumps({"ok": True})
            except Exception as err:
                response.status = 400
                response.content_type = 'application/json'
                return json.dumps({"ok": False, "error": str(err)})

        @self.app.route("/step/next", method="POST")
        def step_next():
            with self._state_lock:
                self.global_step += 1
                step_val = self.global_step
            return json.dumps({"ok": True, "step": int(step_val)})

        @self.app.route("/health", method="GET")
        def health_check():
            """Health check endpoint with memory and system status."""
            process_mb, available_mb, memory_percent, is_low_memory = self._check_memory_usage()
            
            health_status = {
                "ok": True,
                "status": "healthy" if not is_low_memory else "low_memory",
                "memory": {
                    "process_mb": process_mb,
                    "available_mb": available_mb,
                    "memory_percent": memory_percent,
                    "is_low_memory": is_low_memory
                } if process_mb is not None else None
            }
            
            # Add basic server info
            with self._state_lock:
                health_status["server"] = {
                    "global_step": self.global_step,
                    "current_version": self.current_version,
                    "queue_size": self.sample_manager.train_queue.qsize(),
                    "pending_gradients": int(self.gradient_aggregator.pending_batches),
                    "total_gradients": int(self.gradient_aggregator.total_gradients)
                }
            
            return json.dumps(health_status)

        @self.app.route("/stats", method="GET")
        def stats():
            with self._state_lock:
                self._lock_cleanup()
                locks_snapshot = {
                    key: {"owner": meta["owner"], "expires_at": meta["expires_at"]}
                    for key, meta in self.locks.items()
                }
                info = {
                    "ok": True,
                    "global_step": int(self.global_step),
                    "total_enqueued": int(self.sample_manager.total_enqueued),
                    "total_dequeued": int(self.sample_manager.total_dequeued),
                    "queue_size": self.sample_manager.train_queue.qsize(),
                    "locks": locks_snapshot,
                    "pending_batches": int(self.gradient_aggregator.pending_batches),
                    "total_gradients": int(self.gradient_aggregator.total_gradients),
                    "latest_versions": self.latest_versions,
                    "current_version": int(self.current_version),
                    "update_steps": self.batch_size,
                    "should_stop": bool(self._should_stop),
                    "end_received": bool(self._end_received),
                    "problems_distributed": int(self.problem_provider.problems_distributed),
                    "total_problems": int(self.problem_provider.total_problems),
                    "stage_summary": self.sample_manager.stage_snapshot(),
                }
            return json.dumps(info)

        @self.app.route("/lock/acquire", method="POST")
        def lock_acquire():
            payload = request.json if request.json is not None else {}
            if not payload and request.forms:
                payload = dict(request.forms)
            key = payload.get("key")
            owner = payload.get("owner")
            ttl = float(payload.get("ttl", self.lock_ttl))
            if not key or not owner:
                response.status = 400
                return json.dumps({"ok": False, "error": "missing key or owner"})
            ttl = max(ttl, 1.0)
            now = time.time()
            with self._state_lock:
                self._lock_cleanup()
                current = self.locks.get(key)
                if current is None or current["expires_at"] <= now:
                    expires_at = now + ttl
                    self.locks[key] = {"owner": owner, "expires_at": expires_at}
                    return json.dumps(
                        {"ok": True, "acquired": True, "owner": owner, "expires_at": expires_at}
                    )
                if current["owner"] == owner:
                    current["expires_at"] = now + ttl
                    return json.dumps(
                        {
                            "ok": True,
                            "acquired": True,
                            "owner": owner,
                            "expires_at": current["expires_at"],
                            "renewed": True,
                        }
                    )
                return json.dumps(
                    {
                        "ok": True,
                        "acquired": False,
                        "owner": current["owner"],
                        "expires_at": current["expires_at"],
                    }
                )

        @self.app.route("/lock/release", method="POST")
        def lock_release():
            payload = request.json if request.json is not None else {}
            if not payload and request.forms:
                payload = dict(request.forms)
            key = payload.get("key")
            owner = payload.get("owner")
            if not key or not owner:
                response.status = 400
                return json.dumps({"ok": False, "error": "missing key or owner"})
            with self._state_lock:
                current = self.locks.get(key)
                if current and current["owner"] == owner:
                    self.locks.pop(key, None)
                    return json.dumps({"ok": True, "released": True})
                return json.dumps({"ok": True, "released": False})

        @self.app.route("/weights/version", method="GET")
        def weights_version():
            model_id = request.query.get("model_id") or "default"
            client_addr = getattr(request, 'remote_addr', 'unknown') or 'unknown'
            latest = int(self.latest_versions.get(model_id, -1))
            if latest < 0 or not os.path.exists(self._weights_file_path(model_id, latest)):
                latest = self._scan_latest_version(model_id)
                self.latest_versions[model_id] = latest
            print(f"[ORCH] Version check requested from {client_addr} -> latest_version: v{latest} (model_id: {model_id})")
            return json.dumps({"latest_version": latest})

        # ===== Evaluation Endpoints =====
        @self.app.route("/eval/job/get", method="GET")
        def eval_job_get():
            if not self.eval_enabled:
                response.status = 404
                return json.dumps({"ok": False, "error": "evaluation disabled"})
            with self._eval_lock:
                # Requeue expired running jobs
                now = time.time()
                expired = []
                for jid, meta in list(self._eval_running.items()):
                    if meta.get("expires_at", 0) <= now:
                        expired.append(jid)
                for jid in expired:
                    try:
                        job = self._eval_jobs.get(jid)
                        if job and job.get("state") == "running":
                            job["state"] = "pending"
                            job["owner"] = None
                            job["expires_at"] = None
                            self._eval_queue.append(jid)
                    finally:
                        self._eval_running.pop(jid, None)
                # Get next pending job
                job_id = None
                while self._eval_queue:
                    cand = self._eval_queue[0]
                    job = self._eval_jobs.get(cand)
                    if not job or job.get("state") != "pending":
                        self._eval_queue.pop(0)
                        continue
                    job_id = cand
                    break
                if not job_id:
                    return json.dumps({"empty": True})
                job = self._eval_jobs[job_id]
                # Return a lightweight preview; claim to get full payload
                return json.dumps({"ok": True, "job_id": job_id, "version": job["version"], "benchmark": job["benchmark_cfg"].get("name")})

        @self.app.route("/eval/job/claim", method="POST")
        def eval_job_claim():
            if not self.eval_enabled:
                response.status = 404
                return json.dumps({"ok": False, "error": "evaluation disabled"})
            payload = request.json or {}
            job_id = payload.get("job_id")
            owner = payload.get("owner") or f"{request.remote_addr or 'unknown'}"
            ttl = float(payload.get("ttl", self.lock_ttl))
            ttl = max(5.0, ttl)
            with self._eval_lock:
                job = self._eval_jobs.get(job_id)
                if not job or job.get("state") != "pending":
                    response.status = 404
                    return json.dumps({"ok": False, "error": "job not found or not pending"})
                # Remove from queue if head
                try:
                    if self._eval_queue and self._eval_queue[0] == job_id:
                        self._eval_queue.pop(0)
                    else:
                        # Remove wherever it is
                        if job_id in self._eval_queue:
                            self._eval_queue.remove(job_id)
                except Exception:
                    pass
                job["state"] = "running"
                job["owner"] = owner
                job["expires_at"] = time.time() + ttl
                self._eval_running[job_id] = {"owner": owner, "expires_at": job["expires_at"]}
                # Provide full payload
                return json.dumps({
                    "ok": True,
                    "job_id": job_id,
                    "version": job["version"],
                    "benchmark_cfg": job["benchmark_cfg"],
                    "devices": job["devices"],
                })

        @self.app.route("/eval/job/report", method="POST")
        def eval_job_report():
            if not self.eval_enabled:
                response.status = 404
                return json.dumps({"ok": False, "error": "evaluation disabled"})
            payload = request.json or {}
            job_id = payload.get("job_id")
            metrics = payload.get("metrics", {})
            samples = payload.get("samples")  # optional sample table rows
            duration = payload.get("duration_sec")
            with self._eval_lock:
                job = self._eval_jobs.get(job_id)
                if not job:
                    response.status = 404
                    return json.dumps({"ok": False, "error": "job not found"})
                job["state"] = "reported"
                job["reported_at"] = time.time()
                self._eval_running.pop(job_id, None)
            # Persist results
            version = int(job["version"])
            bench_name = job["benchmark_cfg"].get("name", "benchmark")
            out_dir = self._eval_version_dir(version)
            out_path = os.path.join(out_dir, f"{bench_name}.json")
            compute_snapshot = {
                "version": version,
                "global_step": int(self.global_step),
                "optimizer_step": int(self.current_version),
                "total_gradients": int(self.gradient_aggregator.total_gradients),
                "timestamp": time.time(),
            }
            compute_summary = self._snapshot_compute_totals()
            compute_snapshot["compute_totals"] = compute_summary
            total_gpu_seconds = float(compute_summary.get("total", {}).get("gpu_seconds", 0.0))
            compute_snapshot["gpu_seconds_total"] = total_gpu_seconds
            compute_snapshot["gpu_hours_total"] = total_gpu_seconds / 3600.0 if total_gpu_seconds else 0.0
            compute_snapshot["tokens_total"] = int(compute_summary.get("total", {}).get("tokens", 0))
            with self._compute_lock:
                version_totals = copy.deepcopy(self._compute_version_totals.get(version))
            if version_totals:
                compute_snapshot["version_totals"] = version_totals
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "job_id": job_id,
                        "version": version,
                        "benchmark": job["benchmark_cfg"],
                        "metrics": metrics,
                        "samples": samples if isinstance(samples, list) else None,
                        "duration_sec": duration,
                        "reported_at": time.time(),
                        "compute": compute_snapshot,
                    }, f)
            except Exception as e:
                print(f"[ORCH] Failed to save eval results {bench_name} v{version}: {e}")
            # Log to wandb
            try:
                if isinstance(metrics, dict):
                    to_log = {}
                    for k, v in metrics.items():
                        # Flatten numeric metrics
                        if isinstance(v, (int, float)):
                            key = f"{self.eval_namespace}/{bench_name}/{k}"
                            to_log[key] = float(v)
                    # Log compute usage metrics only if enabled
                    if getattr(self.eval_config, "log_compute_usage", False):
                        gpu_hours_total = compute_snapshot.get("gpu_hours_total")
                        if gpu_hours_total:
                            to_log[f"{self.eval_namespace}/{bench_name}/compute_gpu_hours"] = float(gpu_hours_total)
                            totals_meta = compute_snapshot.get("compute_totals", {})
                            if isinstance(totals_meta, dict):
                                for role_name, meta in totals_meta.items():
                                    if role_name == "total" or not isinstance(meta, dict):
                                        continue
                                    role_hours = float(meta.get("gpu_seconds", 0.0)) / 3600.0
                                    to_log[f"{self.eval_namespace}/{bench_name}/compute_{role_name}_hours"] = role_hours
                    # Don't include version in metrics dict - eval metrics should use eval step, not model version as wandb step
                    # The version is already included in compute_snapshot for internal tracking
                    # This prevents eval metrics from using model version as wandb step (which causes step mismatch)
                    if to_log:
                        # Log eval metrics without version field to use wandb's auto-increment step
                        # The metrics themselves contain the version context in their keys (eval/*/metric)
                        self.logger.log(to_log)
            except Exception:
                pass
            self._record_eval_measurement(bench_name, metrics, compute_snapshot)
            return json.dumps({"ok": True})

        @self.app.route("/eval/results", method="GET")
        def eval_results():
            version = request.query.get("version")
            if version is None:
                response.status = 400
                return json.dumps({"ok": False, "error": "missing version"})
            try:
                ver = int(version)
            except Exception:
                response.status = 400
                return json.dumps({"ok": False, "error": "invalid version"})
            out_dir = self._eval_version_dir(ver)
            result = {"ok": True, "version": ver, "benchmarks": {}}
            try:
                for name in os.listdir(out_dir):
                    if not name.endswith(".json"):
                        continue
                    bench_name = name[:-5]
                    try:
                        with open(os.path.join(out_dir, name), "r", encoding="utf-8") as f:
                            data = json.load(f)
                            result["benchmarks"][bench_name] = data.get("metrics", {})
                    except Exception:
                        pass
            except Exception:
                pass
            return json.dumps(result)

        @self.app.route("/eval/stats", method="GET")
        def eval_stats():
            with self._eval_lock:
                pending = len([j for j in self._eval_jobs.values() if j.get("state") == "pending"])
                running = len([j for j in self._eval_jobs.values() if j.get("state") == "running"])
                reported = len([j for j in self._eval_jobs.values() if j.get("state") == "reported"])
                queue_len = len(self._eval_queue)
            return json.dumps({"ok": True, "pending": pending, "running": running, "reported": reported, "queue": queue_len})

        @self.app.route("/eval/fit", method="GET")
        def eval_fit_summary():
            if not self.eval_enabled:
                response.status = 404
                return json.dumps({"ok": False, "error": "evaluation disabled"})
            with self._eval_lock:
                results = copy.deepcopy(self._eval_fit_results)
            payload = {
                "ok": True,
                "results": results,
                "compute_totals": self._snapshot_compute_totals(),
            }
            with self._compute_lock:
                payload["per_version"] = copy.deepcopy(self._compute_version_totals)
            return json.dumps(payload)

        @self.app.route("/compute/report", method="POST")
        def compute_report():
            payload = request.json or {}
            role = str(payload.get("role") or "unknown").strip().lower()
            try:
                gpu_seconds = float(payload.get("gpu_seconds") or 0.0)
            except Exception:
                gpu_seconds = 0.0
            try:
                tokens = int(payload.get("tokens") or 0)
            except Exception:
                tokens = 0
            worker_id = payload.get("worker_id") or (request.remote_addr or "unknown")
            try:
                device_count = int(payload.get("device_count") or 1)
            except Exception:
                device_count = 1
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None
            if gpu_seconds <= 0 and tokens <= 0:
                response.status = 400
                return json.dumps({"ok": False, "error": "invalid compute payload"})
            self._update_compute_usage(role, gpu_seconds, tokens, worker_id, device_count, metadata)
            snapshot = self._snapshot_compute_totals()
            return json.dumps({"ok": True, "totals": snapshot})

        @self.app.route("/compute/stats", method="GET")
        def compute_stats():
            snapshot = self._snapshot_compute_totals()
            with self._compute_lock:
                version_totals = copy.deepcopy(self._compute_version_totals)
            return json.dumps({"ok": True, "totals": snapshot, "per_version": version_totals})

        @self.app.route("/weights/upload", method="POST")
        def weights_upload():
            q = request.query
            model_id = q.get("model_id") or "default"
            ver_str = q.get("version")
            if ver_str is None:
                response.status = 400
                return json.dumps({"ok": False, "error": "missing version"})
            try:
                version = int(ver_str)
            except Exception:
                response.status = 400
                return json.dumps({"ok": False, "error": "invalid version"})
            try:
                body = request.body.read()
            except (ConnectionResetError, OSError) as e:
                response.status = 500
                response.content_type = 'application/json'
                return json.dumps({"ok": False, "error": f"connection error: {e}"})
            if not body:
                response.status = 400
                return json.dumps({"ok": False, "error": "empty body"})
            safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id)
            model_dir = os.path.join(self.weights_dir, safe_model_id)
            os.makedirs(model_dir, exist_ok=True)
            out_path = os.path.join(model_dir, f"weights_v{version}.pt")
            with open(out_path, "wb") as f:
                f.write(body)
            try:
                os.fsync(os.open(out_path, os.O_RDONLY))
            except Exception:
                pass
            prev = int(self.latest_versions.get(model_id, -1))
            if version > prev and os.path.exists(out_path):
                self.latest_versions[model_id] = version
            self._maybe_prune_weights(model_id, keep_last=2)
            return json.dumps(
                {"ok": True, "saved": out_path, "latest_version": int(self.latest_versions.get(model_id, version))}
            )

        @self.app.route("/weights/upload_chunk", method="POST")
        def weights_upload_chunk():
            q = request.query
            model_id = q.get("model_id") or "default"
            ver_str = q.get("version")
            part_str = q.get("part")
            total_str = q.get("total")
            if ver_str is None or part_str is None or total_str is None:
                response.status = 400
                return json.dumps({"ok": False, "error": "missing version/part/total"})
            try:
                version = int(ver_str)
                part = int(part_str)
                total = int(total_str)
                assert 0 <= part < total
            except Exception:
                response.status = 400
                return json.dumps({"ok": False, "error": "invalid version/part/total"})
            try:
                body = request.body.read()
            except (ConnectionResetError, OSError) as e:
                response.status = 500
                response.content_type = 'application/json'
                return json.dumps({"ok": False, "error": f"connection error: {e}"})
            if body is None:
                response.status = 400
                return json.dumps({"ok": False, "error": "empty body"})
            safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id)
            model_dir = os.path.join(self.weights_dir, safe_model_id)
            os.makedirs(model_dir, exist_ok=True)
            out_path = os.path.join(model_dir, f"weights_v{version}.pt")
            if os.path.exists(out_path):
                prev = int(self.latest_versions.get(model_id, -1))
                if version > prev:
                    self.latest_versions[model_id] = version
                return json.dumps(
                    {
                        "ok": True,
                        "assembled": True,
                        "latest_version": int(self.latest_versions.get(model_id, version)),
                    }
                )
            tmp_dir = os.path.join(model_dir, f"tmp_v{version}")
            os.makedirs(tmp_dir, exist_ok=True)
            chunk_path = os.path.join(tmp_dir, f"part_{part:06d}.bin")
            with open(chunk_path, "wb") as f:
                f.write(body)
            manifest_path = os.path.join(tmp_dir, "manifest.json")
            if not os.path.exists(manifest_path):
                with open(manifest_path, "w") as mf:
                    mf.write(json.dumps({"total": total}))
            existing = [
                name
                for name in os.listdir(tmp_dir)
                if name.startswith("part_") and name.endswith(".bin")
            ]
            received = len(existing)
            if received >= total:
                with open(manifest_path, "r") as mf:
                    meta = json.loads(mf.read() or "{}")
                expected_total = int(meta.get("total", total))
                if expected_total != total:
                    expected_total = total
                with open(out_path, "wb") as out_f:
                    for i in range(expected_total):
                        part_file = os.path.join(tmp_dir, f"part_{i:06d}.bin")
                        if not os.path.exists(part_file):
                            response.status = 500
                            return json.dumps(
                                {"ok": False, "error": f"missing part {i} at assemble time"}
                            )
                        with open(part_file, "rb") as pf:
                            shutil.copyfileobj(pf, out_f)
                try:
                    for name in os.listdir(tmp_dir):
                        os.remove(os.path.join(tmp_dir, name))
                    os.rmdir(tmp_dir)
                except Exception:
                    pass
                prev = int(self.latest_versions.get(model_id, -1))
                if version > prev and os.path.exists(out_path):
                    self.latest_versions[model_id] = version
                self._maybe_prune_weights(model_id, keep_last=2)
                return json.dumps(
                    {
                        "ok": True,
                        "assembled": True,
                        "latest_version": int(self.latest_versions.get(model_id, version)),
                    }
                )
            return json.dumps({"ok": True, "received_parts": received, "total": total})

        @self.app.route("/weights/download", method="GET")
        def weights_download():
            q = request.query
            model_id = q.get("model_id") or "default"
            ver_str = q.get("version") or "latest"
            client_addr = getattr(request, 'remote_addr', 'unknown') or 'unknown'
            latest = int(self.latest_versions.get(model_id, -1))
            if ver_str == "latest":
                version = latest
            else:
                try:
                    version = int(ver_str)
                except Exception:
                    response.status = 400
                    print(f"[ORCH] Weight download request from {client_addr} FAILED: invalid version string '{ver_str}' (model_id: {model_id})")
                    return json.dumps({"ok": False, "error": "invalid version"})
            if version < 0:
                response.status = 404
                print(f"[ORCH] Weight download request from {client_addr} FAILED: no weights available (model_id: {model_id})")
                return json.dumps({"ok": False, "error": "no weights available"})
            safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id)
            model_dir = os.path.join(self.weights_dir, safe_model_id)
            file_path = os.path.join(model_dir, f"weights_v{version}.pt")
            if not os.path.exists(file_path):
                response.status = 404
                print(f"[ORCH] Weight download request from {client_addr} FAILED: version v{version} not found (model_id: {model_id})")
                return json.dumps({"ok": False, "error": "requested version not found"})
            # Get file size for logging
            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            except Exception:
                file_size_mb = 0.0
            print(f"[ORCH] Weight download requested from {client_addr} -> sending version v{version} ({file_size_mb:.1f}MB, model_id: {model_id})")
            response.content_type = "application/octet-stream"
            response.set_header(
                "Content-Disposition", f'attachment; filename="{os.path.basename(file_path)}"'
            )
            with open(file_path, "rb") as f:
                data = f.read()
            return data

        # Server will be started in start() method
        print(f"[ORCH] Routes registered. Server will be started via start() method.")

    def _maybe_prune_weights(self, model_id, keep_last=None):
        try:
            keep = self.keep_last_versions if keep_last is None else int(keep_last)
            safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id)
            model_dir = os.path.join(self.weights_dir, safe_model_id)
            if not os.path.isdir(model_dir):
                return
            files = []
            for name in os.listdir(model_dir):
                if name.startswith("weights_v") and name.endswith(".pt"):
                    m = re.match(r"^weights_v(\d+)\.pt$", name)
                    if m:
                        ver = int(m.group(1))
                        files.append((ver, os.path.join(model_dir, name)))
            if len(files) <= keep:
                return
            files.sort(key=lambda x: x[0])
            to_remove = files[:-keep]
            for _, fp in to_remove:
                try:
                    os.remove(fp)
                except Exception:
                    pass
        except Exception:
            pass

    def start(self):
        """
        Start the multi-threaded Bottle server with thread pool.
        This method should be called after run_server() has registered all routes.
        The server will use a thread pool limited to server_threads concurrent requests.
        """
        # Initialize problem queue with train_data
        self.problem_provider.initialize()
        if self.problem_provider.train_data:
            print(f"[ORCH] Problem queue initialized with {self.problem_provider.problem_queue.qsize()} items")
        else:
            print("[ORCH] Warning: No train_data provided, samplers will not receive problem data")

        if self.model is not None:
            try:
                base_id = (
                    os.path.basename(str(self.model.name_or_path)).strip()
                    if hasattr(self.model, "name_or_path")
                    else "default"
                )
            except Exception:
                base_id = "default"
            safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", base_id or "default")
            latest = self._scan_latest_version(safe_model_id)
            if latest < 0:
                # Model is already on CPU, save directly
                model_dir = os.path.join(self.weights_dir, safe_model_id)
                os.makedirs(model_dir, exist_ok=True)
                out_path = os.path.join(model_dir, "weights_v0.pt")
                state_dict_cpu = {
                    k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in self.model.state_dict().items()
                }
                with open(out_path, "wb") as f:
                    torch.save(state_dict_cpu, f)
                self.latest_versions[safe_model_id] = 0
                self.current_version = 0
                print("[ORCH] Saved initial weights as version 0")
            else:
                self.latest_versions[safe_model_id] = latest
                self.current_version = latest
                file_path = self._weights_file_path(safe_model_id, latest)
                model_dir = os.path.join(self.weights_dir, safe_model_id)
                if os.path.exists(file_path):
                    try:
                        state_dict = torch.load(file_path, map_location="cpu")
                        self.model.load_state_dict(state_dict, strict=False)
                        # Model stays on CPU
                        print(f"[ORCH] Loaded weights version {latest} from disk (CPU)")
                    except (RuntimeError, EOFError, OSError) as e:
                        print(f"[ORCH] WARNING: Failed to load weights version {latest} from {file_path}: {e}")
                        print(f"[ORCH] Removing corrupted file and creating new initial weights...")
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
                        # Create new initial weights
                        os.makedirs(model_dir, exist_ok=True)
                        out_path = os.path.join(model_dir, "weights_v0.pt")
                        state_dict_cpu = {
                            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                            for k, v in self.model.state_dict().items()
                        }
                        with open(out_path, "wb") as f:
                            torch.save(state_dict_cpu, f)
                        self.latest_versions[safe_model_id] = 0
                        self.current_version = 0
                        print(f"[ORCH] Saved new initial weights as version 0")
            print(f"[ORCH] Training will stop when queue is empty and end signal is received")
            self.gradient_aggregator.current_version = self.current_version

        # Model stays on CPU - no GPU needed for orchestrator
        if self.model is not None:
            print(f"[ORCH] Model is on CPU (no GPU required for orchestrator)")

        # Start periodic chunk cleanup thread
        self._chunk_cleanup_stop = threading.Event()
        def chunk_cleanup_loop():
            while not self._chunk_cleanup_stop.wait(timeout=self.chunk_cleanup_interval):
                try:
                    stale_count, total_mb = self._cleanup_stale_chunks()
                    if stale_count > 0:
                        print(f"[ORCH] Cleaned up {stale_count} stale gradient chunk upload(s), {total_mb:.1f}MB remaining")
                    # Log chunk status periodically
                    num_uploads, total_chunks, chunk_mb = self._log_chunk_status()
                except Exception as e:
                    print(f"[ORCH] Error in chunk cleanup: {e}")
        
        self._chunk_cleanup_thread = threading.Thread(target=chunk_cleanup_loop, daemon=True)
        self._chunk_cleanup_thread.start()
        
        # Start periodic status reporting thread
        self._status_report_stop = threading.Event()
        def status_report_loop():
            report_interval = self.status_report_interval  # Use configured status report interval
            timeout_check_interval = self.timeout_check_interval  # Use configured timeout check interval
            last_timeout_check = time.time()
            
            while not self._status_report_stop.wait(timeout=report_interval):
                try:
                    # Check for timeout batches and problems periodically
                    now = time.time()
                    if now - last_timeout_check >= timeout_check_interval:
                        requeued_batches, timeout_batch_ids = self.sample_manager.requeue_timeout_batches()
                        requeued_problems = self.problem_provider.requeue_timeout_problems()
                        if requeued_batches > 0:
                            print(f"[ORCH] Requeued {requeued_batches} timeout batch(es) (trainer may have died)")
                            # Adjust pending_batches for timeout batches
                            handled = self.gradient_aggregator.handle_timeout_batches(
                                timeout_batch_ids, 
                                batch_timeout=self.sample_manager.batch_timeout
                            )
                            if handled > 0:
                                print(f"[ORCH] Adjusted pending_batches for {handled} timeout batch(es)")
                        if requeued_problems > 0:
                            print(f"[ORCH] Requeued {requeued_problems} timeout problem(s) (sampler may have died)")
                        last_timeout_check = now
                    
                    with self._state_lock:
                        queue_size = self.sample_manager.train_queue.qsize()
                        pending = int(self.gradient_aggregator.pending_batches)
                        total_grads = int(self.gradient_aggregator.total_gradients)
                        total_enqueued = int(self.sample_manager.total_enqueued)
                        total_dequeued = int(self.sample_manager.total_dequeued)
                        problems_dist = int(self.problem_provider.problems_distributed)
                        total_problems = int(self.problem_provider.total_problems)
                        progress_pct = (pending / self.batch_size * 100) if self.batch_size > 0 else 0
                        stage = self.sample_manager.stage_snapshot()
                    
                    # Format trainer progress
                    now = time.time()
                    trainer_status = []
                    for worker_id, progress in list(self._trainer_progress.items()):
                        # Remove stale entries (older than 60 seconds)
                        if now - progress["timestamp"] > 60:
                            self._trainer_progress.pop(worker_id, None)
                            continue
                        short_id = worker_id.split('-')[-1] if '-' in worker_id else worker_id[-8:]
                        trainer_status.append(f"{short_id}:{progress['microstep']}/{progress['total']}({progress['progress_pct']:.0f}%)")
                    
                    trainer_str = f" | Trainers: [{', '.join(trainer_status)}]" if trainer_status else ""
                    
                    # Get chunk status
                    num_uploads, total_chunks, chunk_mb = self._log_chunk_status()
                    chunk_str = f" | Chunks: {num_uploads} uploads ({chunk_mb:.1f}MB)" if num_uploads > 0 else ""
                    
                    # Check memory usage
                    process_mb, available_mb, memory_percent, is_low_memory = self._check_memory_usage()
                    if process_mb is not None:
                        mem_str = f" | Memory: {process_mb:.0f}MB / {available_mb:.0f}MB avail ({memory_percent:.1f}%)"
                        if is_low_memory:
                            mem_str += " ⚠️LOW"
                    else:
                        mem_str = ""
                    
                    # Sync current_version from gradient_aggregator (handles async updates)
                    if self.gradient_aggregator.current_version > self.current_version:
                        self.current_version = self.gradient_aggregator.current_version
                        self.latest_versions = self.gradient_aggregator.latest_versions
                    
                    # Get evaluation job status
                    eval_str = ""
                    if self.eval_enabled:
                        with self._eval_lock:
                            eval_pending = len([j for j in self._eval_jobs.values() if j.get("state") == "pending"])
                            eval_running = len([j for j in self._eval_jobs.values() if j.get("state") == "running"])
                            eval_reported = len([j for j in self._eval_jobs.values() if j.get("state") == "reported"])
                            if eval_pending > 0 or eval_running > 0:
                                eval_str = f" | Eval: {eval_running} running, {eval_pending} pending, {eval_reported} done"
                    
                    if self.log_status_report:
                        # Get current time for status report
                        current_time_str = time.strftime("%H:%M:%S")
                        print(f"[ORCH STATUS] [{current_time_str}] Step: {self.global_step} | "
                              f"Gradients: {total_grads} (pending: {pending}/{self.batch_size}, {progress_pct:.1f}%) | "
                              f"Queue: {queue_size} samples | "
                              f"Processing: {total_dequeued}/{total_enqueued} completed | "
                              f"Stages: generated={stage['generated_total']} waiting={stage['awaiting_gradient']} completed={stage['completed_total']} | "
                              f"Problems: {problems_dist}/{total_problems} | "
                              f"Version: {self.current_version}{chunk_str}{mem_str}{trainer_str}{eval_str}")
                except Exception as e:
                    # Don't let status report errors crash the server
                    print(f"[ORCH] Error in status report loop: {e}")
                    import traceback
                    traceback.print_exc()
        
        self._status_report_thread = threading.Thread(target=status_report_loop, daemon=True)
        self._status_report_thread.start()
        
        from wsgiref.simple_server import make_server, WSGIServer, WSGIRequestHandler
        from concurrent.futures import ThreadPoolExecutor
        
        class ThreadPoolWSGIServer(WSGIServer):
            """WSGI server with thread pool for limiting concurrent requests."""
            def __init__(self, *args, max_workers=10, log_http_access=True, **kwargs):
                super().__init__(*args, **kwargs)
                self.executor = ThreadPoolExecutor(max_workers=max_workers)
                self.max_workers = max_workers
                self.log_http_access = log_http_access
            
            def process_request(self, request, client_address):
                """Submit request to thread pool instead of creating new thread."""
                def handle_request():
                    try:
                        self.finish_request(request, client_address)
                        self.shutdown_request(request)
                    except Exception:
                        self.handle_error(request, client_address)
                        self.shutdown_request(request)
                self.executor.submit(handle_request)
            
            def server_close(self):
                """Shutdown thread pool when server closes."""
                super().server_close()
                self.executor.shutdown(wait=True)
        
        # Create a handler class factory that captures log_http_access
        def make_handler_class(log_http_access):
            class CustomHandler(WSGIRequestHandler):
                def log_message(self, format, *args):
                    if log_http_access:
                        super().log_message(format, *args)
            return CustomHandler
        
        print(f"[ORCH] Starting multi-threaded server on http://{self.host}:{self.port} with {self.server_threads} thread pool workers")
        
        try:
            server = make_server(
                self.host,
                self.port,
                self.app,
                server_class=lambda *args, **kwargs: ThreadPoolWSGIServer(*args, max_workers=self.server_threads, log_http_access=self.log_http_access, **kwargs),
                handler_class=make_handler_class(self.log_http_access)
            )
            print(f"[ORCH] Orchestrator server started on http://{self.host}:{self.port}")
            print(f"[ORCH] Thread pool mode enabled: {self.server_threads} concurrent request handlers (requests beyond this limit will queue)")
            print(f"[ORCH] Status reports every {self.status_report_interval} seconds. Use /stats endpoint for detailed info.")
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[ORCH] Shutting down server...")
            self._status_report_stop.set()
            self._chunk_cleanup_stop.set()
             # Wait for evaluation jobs before exiting
            if self.eval_enabled:
                print("[ORCH] Waiting for evaluation jobs to finish before shutdown...")
                self._wait_for_eval_completion(timeout=self.eval_shutdown_timeout or None)
            # Wait for threads to finish
            if self._status_report_thread.is_alive():
                self._status_report_thread.join(timeout=2.0)
            if self._chunk_cleanup_thread.is_alive():
                self._chunk_cleanup_thread.join(timeout=2.0)
            print("[ORCH] Server stopped.")
        except OSError as e:
            # Handle socket errors (port already in use, connection refused, etc.)
            print(f"[ORCH ERROR] Server socket error: {e}")
            print(f"[ORCH] Shutting down server due to socket error...")
            self._status_report_stop.set()
            self._chunk_cleanup_stop.set()
            self._wait_for_eval_completion(timeout=self.eval_shutdown_timeout or None)
            if self._status_report_thread.is_alive():
                self._status_report_thread.join(timeout=2.0)
            if self._chunk_cleanup_thread.is_alive():
                self._chunk_cleanup_thread.join(timeout=2.0)
            raise
        except MemoryError as e:
            # Handle out of memory errors
            print(f"[ORCH ERROR] Out of memory: {e}")
            print(f"[ORCH] Shutting down server due to memory error...")
            self._status_report_stop.set()
            self._chunk_cleanup_stop.set()
            self._wait_for_eval_completion(timeout=self.eval_shutdown_timeout or None)
            if self._status_report_thread.is_alive():
                self._status_report_thread.join(timeout=2.0)
            if self._chunk_cleanup_thread.is_alive():
                self._chunk_cleanup_thread.join(timeout=2.0)
            raise
        except Exception as e:
            # Catch any other unexpected errors
            print(f"[ORCH ERROR] Unexpected server error: {e}")
            import traceback
            traceback.print_exc()
            print(f"[ORCH] Shutting down server due to unexpected error...")
            self._status_report_stop.set()
            self._chunk_cleanup_stop.set()
            self._wait_for_eval_completion(timeout=self.eval_shutdown_timeout or None)
            if self._status_report_thread.is_alive():
                self._status_report_thread.join(timeout=2.0)
            if self._chunk_cleanup_thread.is_alive():
                self._chunk_cleanup_thread.join(timeout=2.0)
            raise
