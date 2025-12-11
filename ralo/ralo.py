from __future__ import annotations

import json, os, shutil, re, random, io, requests, sys, time, socket, threading, uuid
import torch
import torch.nn as nn
import math
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import deque
from .utils import json_to_bytes_list, bytes_list_to_json, enable_gradient_checkpointing, encode_gradients
from .sampler.client import SamplerClient
from .trainer.client import TrainerClient
from .config import SamplerConfig, TrainerConfig
from .algorithms import get_sampler_algorithm, get_trainer_algorithm
from .services import ModelService, SamplingService, TrainingService, OrchestratorService
from collections import defaultdict
def get_world_size(): return int(os.environ.get('WORLD_SIZE', 1))
#torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False) #update

def distbarrier():
    if dist.is_initialized(): dist.barrier()


class _RaloWrapper:
    """Wrapper class to pass to sampler algorithm instead of full RALO instance."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._hooks = {}


def _process_eval_job(orchestrator_service, sampling_service, owner_id, gen_rank, processed_versions):
    """
    Process a single evaluation job using the provided SamplingService.
    Ensures each version is evaluated only once per worker.
    
    Args:
        orchestrator_service: OrchestratorService instance
        sampling_service: SamplingService instance (already initialized)
        owner_id: Worker ID for claiming jobs
        gen_rank: Generation worker rank for logging
        processed_versions: Set of version numbers already processed by this worker
    
    Returns:
        True if job was processed, False if no job available or already processed
    """
    import traceback
    try:
        preview = orchestrator_service.get_eval_job()
    except Exception:
        preview = {"empty": True}
    
    if not isinstance(preview, dict) or preview.get("empty"):
        return False
    
    job_id = preview.get("job_id")
    job_version = preview.get("version")
    
    if not job_id:
        return False
    
    # Check if this version has already been processed by this worker
    if job_version is not None and job_version in processed_versions:
        return False
    
    try:
        claim = orchestrator_service.claim_eval_job(job_id=job_id, owner=owner_id, ttl=90.0)
    except Exception:
        return False
    
    if not isinstance(claim, dict) or not claim.get("ok"):
        return False
    
    # Get version from claim (more reliable than preview)
    claim_version = claim.get("version")
    if claim_version is not None and claim_version in processed_versions:
        # Another worker might have processed it, but we already checked
        # This is a race condition protection
        return False
    
    benchmark_cfg = claim.get("benchmark_cfg", {})
    benchmark_name = benchmark_cfg.get("name", "unknown")
    start_t = time.time()
    
    # Log evaluation start
    print(f"[EVAL {gen_rank}] Starting evaluation: {benchmark_name} for version {claim_version} (job_id: {job_id})")
    
    try:
        # Update weights if needed before evaluation
        print(f"[EVAL {gen_rank}] Updating weights to version {claim_version} before evaluation...")
        try:
            updated = sampling_service.maybe_update_weights(orchestrator_service, gen_rank=gen_rank, force=True)
            if updated:
                print(f"[EVAL {gen_rank}] Weights updated to version {claim_version}")
            else:
                print(f"[EVAL {gen_rank}] Weights already at version {claim_version}")
        except Exception as e:
            print(f"[EVAL {gen_rank}] Warning: Weight update failed (continuing anyway): {e}")
        
        print(f"[EVAL {gen_rank}] Running benchmark {benchmark_name}...")
        from .eval.runner import run_benchmark
        metrics, sample_rows, bench_stats = run_benchmark(
            benchmark_cfg=benchmark_cfg,
            sampling_service=sampling_service,
        )
        print(f"[EVAL {gen_rank}] Benchmark {benchmark_name} completed: {metrics}")
    except Exception as e:
        print(f"[EVAL {gen_rank}] Failed running benchmark {benchmark_cfg.get('name')}: {e}")
        traceback.print_exc()
        metrics, sample_rows, bench_stats = {"error": 1.0}, None, {"generated_tokens": 0}
    
    duration = time.time() - start_t
    generated_tokens = int((bench_stats or {}).get("generated_tokens", 0))
    
    try:
        orchestrator_service.report_compute_usage(
            role="evaluator",
            gpu_seconds=float(max(duration, 0.0)),
            tokens=generated_tokens,
            worker_id=owner_id,
        )
    except Exception:
        pass
    
    try:
        orchestrator_service.report_eval_job(job_id=job_id, metrics=metrics, samples=sample_rows, duration_sec=duration)
        # Mark this version as processed after successful report
        if claim_version is not None:
            processed_versions.add(claim_version)
        print(f"[EVAL {gen_rank}] Completed benchmark {benchmark_cfg.get('name')} for version {claim_version}: {metrics}")
    except Exception as e:
        print(f"[EVAL {gen_rank}] Failed to report results for {job_id}: {e}")
    
    return True


def _tree_node_to_dict(node, visited_nodes=None):
    """
    Convert TreeNode to dictionary representation (recursive).
    
    Args:
        node: TreeNode instance
        visited_nodes: Set of visited node IDs to prevent cycles
    
    Returns:
        Dictionary representation of the tree node
    """
    if visited_nodes is None:
        visited_nodes = set()
    
    # Use id() as unique identifier to prevent infinite recursion
    node_id = id(node)
    if node_id in visited_nodes:
        return {"_cycle_detected": True, "_node_id": node_id}
    visited_nodes.add(node_id)
    
    # Convert node to dict
    node_dict = {
        'item': node.item.copy() if hasattr(node.item, 'copy') else dict(node.item) if isinstance(node.item, dict) else node.item,
        'budget': node.budget,
        'endurance': node.endurance,
        'depth': node.depth,
        'birth_order': node.birth_order,
        'child_count': node.child_count,
        'children_rewards': list(node.children_rewards) if hasattr(node, 'children_rewards') else [],
        'children': []
    }
    
    # Recursively convert children
    for child in node.children:
        child_dict = _tree_node_to_dict(child, visited_nodes)
        node_dict['children'].append(child_dict)
    
    visited_nodes.remove(node_id)
    return node_dict


def _extract_tree_roots(finished_nodes):
    """
    Extract root nodes from finished nodes by traversing up to parents.
    
    Args:
        finished_nodes: List of finished TreeNode instances
    
    Returns:
        Set of root nodes (nodes with no parent)
    """
    roots = set()
    for node in finished_nodes:
        current = node
        # Traverse up to root
        while hasattr(current, 'parent') and current.parent is not None:
            current = current.parent
        roots.add(current)
    return roots


def _save_samples(finished_samples, finished_nodes, log_dir, gen_rank, problem_id=None, node_id=None, node_version=None, save_tree=True, save_individual=False, head_node=None):
    """
    Save generated samples to file.
    For TreePO: saves tree structure (if save_tree=True).
    For other algorithms or if save_individual=True: saves individual samples.
    
    Args:
        finished_samples: List of sample dictionaries
        finished_nodes: Optional list of TreeNode instances (for TreePO tree structure)
        log_dir: Log directory path (str or Path)
        gen_rank: Generation worker rank
        problem_id: Optional problem ID for filename
        node_id: Optional node ID for metadata
        node_version: Optional node version for metadata
        save_tree: Whether to save tree structure (for TreePO, default: True)
        save_individual: Whether to save individual samples even for TreePO (default: False)
        head_node: Optional root TreeNode (for TreePO, if provided, saves the complete tree from root)
    """
    try:
        import json
        from pathlib import Path
        
        # Determine samples directory
        if log_dir:
            samples_dir = Path(log_dir) / "samples"
        else:
            # Fallback: use RUN_ID from environment or default
            run_id = os.environ.get("RUN_ID", "default")
            samples_dir = Path("logs") / run_id / "samples"
        
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # If TreePO with tree structure, save tree (if enabled)
        if (finished_nodes or head_node) and save_tree:
            # Create tree filename
            if problem_id:
                tree_filename = f"tree_rank{gen_rank}_{problem_id}_{timestamp}.json"
            else:
                tree_filename = f"tree_rank{gen_rank}_{timestamp}.json"
            
            tree_filepath = samples_dir / tree_filename
            
            # If head_node is provided, use it directly (complete tree from root)
            # Otherwise, extract root nodes from finished_nodes
            if head_node is not None:
                # Save complete tree from root node
                tree_dict = _tree_node_to_dict(head_node)
                tree_record = {
                    'tree': tree_dict,  # Single complete tree
                    'num_finished_nodes': len(finished_nodes) if finished_nodes else 0,
                    'metadata': {
                        'timestamp': time.time(),
                        'gen_rank': gen_rank,
                        'problem_id': problem_id,
                        'node_id': node_id,
                        'node_version': node_version,
                    }
                }
            else:
                # Fallback: extract root nodes from finished_nodes (for backward compatibility)
                root_nodes = _extract_tree_roots(finished_nodes) if finished_nodes else []
                
                # Convert trees to dictionaries
                trees_data = []
                for root in root_nodes:
                    tree_dict = _tree_node_to_dict(root)
                    trees_data.append(tree_dict)
                
                # Save tree structure
                tree_record = {
                    'trees': trees_data,
                    'num_trees': len(trees_data),
                    'num_finished_nodes': len(finished_nodes) if finished_nodes else 0,
                    'metadata': {
                        'timestamp': time.time(),
                        'gen_rank': gen_rank,
                        'problem_id': problem_id,
                        'node_id': node_id,
                        'node_version': node_version,
                    }
                }
            
            with open(tree_filepath, 'w', encoding='utf-8') as f:
                json.dump(tree_record, f, ensure_ascii=False, indent=2, default=str)
            
            if head_node is not None:
                print(f"[GEN {gen_rank}] Saved complete tree structure ({len(finished_nodes) if finished_nodes else 0} finished nodes) to {tree_filepath}")
            else:
                print(f"[GEN {gen_rank}] Saved tree structure ({len(trees_data) if 'trees' in tree_record else 0} trees, {len(finished_nodes) if finished_nodes else 0} finished nodes) to {tree_filepath}")
        
        # For non-tree algorithms or if explicitly requested, save individual samples
        if finished_samples and (save_individual or not finished_nodes):
            # Create samples filename
            if problem_id:
                samples_filename = f"samples_rank{gen_rank}_{problem_id}_{timestamp}.jsonl"
            else:
                samples_filename = f"samples_rank{gen_rank}_{timestamp}.jsonl"
            
            samples_filepath = samples_dir / samples_filename
            
            # Save each sample as a JSON line
            with open(samples_filepath, 'a', encoding='utf-8') as f:
                for idx, sample in enumerate(finished_samples):
                    # Convert torch.Tensor to Python types for JSON serialization
                    def convert_value(value):
                        if isinstance(value, torch.Tensor):
                            return value.item() if value.numel() == 1 else value.tolist()
                        return value
                    
                    # Create sample record with metadata
                    sample_record = {
                        'sample': {k: convert_value(v) for k, v in sample.items()},
                        'metadata': {
                            'timestamp': time.time(),
                            'gen_rank': gen_rank,
                            'problem_id': problem_id,
                            'node_id': node_id,
                            'node_version': node_version,
                            'sample_index': idx,
                            'total_samples': len(finished_samples),
                        }
                    }
                    f.write(json.dumps(sample_record, ensure_ascii=False, default=str) + '\n')
            
            print(f"[GEN {gen_rank}] Saved {len(finished_samples)} samples to {samples_filepath}")
    except Exception as e:
        # Don't crash if saving fails
        print(f"[GEN {gen_rank}] Warning: Failed to save samples: {e}")


def _gen_worker_static(Q_data, pending_tasks, gen_device, gen_rank, worker_args, rollout_prompt_fn=None, reward_fns=None):
    """Static version of gen_worker that doesn't require self (for multiprocessing)."""
    # Create prompt and reward functions in worker process
    # These will be recreated from the model_path and tokenizer
    model_path = worker_args['model_path']
    
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{gen_device}'
    cleanup_keys = [  
        'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'LOCAL_RANK',  
        'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_NAME',   
        'GROUP_WORLD_SIZE', 'ROLE_WORLD_SIZE',  
        'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_MAX_RESTARTS',  
        'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_USE_AGENT_STORE',  
        'TORCHELASTIC_ERROR_FILE',  
        'TORCH_NCCL_ASYNC_ERROR_HANDLING',  
        'NCCL_COMM_ID', 'NCCL_DEBUG', 'NCCL_SOCKET_IFNAME',  
    ]  
    for key in cleanup_keys: os.environ.pop(key, None)
    
    torch.cuda.set_device(0)
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    print(f"[GEN {gen_rank}] Generation worker process uses GPU {gen_device}")
    print(f"[GEN {gen_rank}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[GEN {gen_rank}] PID: {os.getpid()}")

    model_id = worker_args['model_id']
    orchestrator_url = worker_args['orchestrator_url']
    
    # Initialize tokenizer in worker process
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Recreate rollout_prompt_fn if not provided (default implementation)
    # This matches the implementation in ralo_cli.py
    # Check if thinking mode is enabled
    enable_thinking = worker_args.get("enable_thinking", False)
    
    if rollout_prompt_fn is None:
        # Get system_prompt from worker_args (set from YAML config or default)
        system_prompt = worker_args.get("system_prompt", "Please reason step by step, and put your final answer within \\boxed{}.")
        def default_rollout_prompt_fn(item):
            # Default implementation using tokenizer
            # This should match the implementation in ralo_cli.py
            # For thinking mode, use enable_thinking=True
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if enable_thinking and hasattr(tokenizer, 'apply_chat_template'):
                # Enable thinking mode for Qwen3
                template_kwargs["enable_thinking"] = True
            
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item.get("Q", item) if isinstance(item, dict) else str(item)},
                ],
                **template_kwargs
            )
        rollout_prompt_fn = default_rollout_prompt_fn
    
    # Recreate reward_fns if not provided (default implementation)
    # This matches the implementation in ralo_cli.py
    if reward_fns is None:
        try:
            from math_verify import ExprExtractionConfig, parse, verify
            def correct_fn(answer, item):
                gold_parsed = parse(item["A"], extraction_config=[ExprExtractionConfig()])
                answer_parsed = parse(answer, extraction_config=[ExprExtractionConfig()])
                return 1 if verify(gold_parsed, answer_parsed) else -1
            reward_fns = [correct_fn]
        except ImportError:
            # If math_verify is not available, use a simple reward function
            def simple_reward_fn(answer, item):
                # Simple reward: 1 if answer matches, -1 otherwise
                return 1 if str(answer).strip() == str(item.get("A", "")).strip() else -1
            reward_fns = [simple_reward_fn]
    
    # Create a minimal wrapper for sampler algorithm
    # Include all attributes that sampler_algo might need
    ralo_wrapper = _RaloWrapper(
        rollout_prompt_fn=rollout_prompt_fn,
        reward_fns=reward_fns,
        tokenizer=tokenizer,  # Add tokenizer for format_samples_for_upload
        TreePO_kwargs=worker_args.get('TreePO_kwargs', {}),
        rollout_num=worker_args.get('rollout_num', 8),
        gen_temperature=worker_args.get('gen_temperature', 0.9),
        gen_top_p=worker_args.get('gen_top_p'),
        gen_top_k=worker_args.get('gen_top_k'),
        gen_min_p=worker_args.get('gen_min_p'),
        enable_thinking=worker_args.get('enable_thinking', False),
        gen_max_tokens=worker_args.get('gen_max_tokens', 4096),
        train_batch_size=worker_args.get('train_batch_size', 1),
        max_pending_samples=worker_args.get('max_pending_samples', 1600),
        gen_pending_time=worker_args.get('gen_pending_time', 10.0),
        max_batch_retry=worker_args.get('max_batch_retry', 3),
        epochs=worker_args.get('epochs', 1),
        clip_param=worker_args.get('clip_param', 0.2),
        call_hook=lambda name, *args, **kwargs: None,  # No-op hook
    )
    
    # Initialize SamplingService in worker process
    sampling_service = SamplingService(
        model_path=model_path,
        model_id=model_id,
        orchestrator_url=orchestrator_url,
        vllm_kwargs=worker_args.get('vllm_kwargs', {}),
        gen_temperature=worker_args.get('gen_temperature', 0.9),
        version_poll_interval=worker_args.get('version_poll_interval', 5.0),
    )
    sampling_service.initialize(gen_device=gen_device, gen_rank=gen_rank)
    
    # Initialize OrchestratorService in worker process
    orchestrator_service = OrchestratorService(
        orchestrator_url=orchestrator_url,
        model_id=model_id,
        retry_interval=worker_args.get('gen_pending_time', 10.0),
    )
    
    # Initialize model service
    model_service = ModelService(model_path=model_path, model_id=model_id)
    
    # Get sampler algorithm
    sampler_algorithm_name = worker_args.get('sampler_algorithm_name', 'treepo')
    sampler_config_dict = worker_args.get('sampler_config_dict', {})
    # Reconstruct SamplerConfig from dict
    from .config import SamplerConfig
    sampler_config = SamplerConfig(**sampler_config_dict) if sampler_config_dict else SamplerConfig()
    algo_cls = get_sampler_algorithm(sampler_algorithm_name)
    sampler_algo = algo_cls(ralo_wrapper, sampler_config)
    
    # Services dict for algorithm
    services = {
        'model': model_service,
        'sampling': sampling_service,
        'orchestrator': orchestrator_service,
    }

    def QueueGetNowait(Q):
        try: return Q.get_nowait()
        except: return None

    # Initialize weight version tracking
    curr_ver = -1
    try:
        initial_version = orchestrator_service.latest_version()
        if initial_version is not None and initial_version >= 0:
            curr_ver = initial_version
            sampling_service.set_current_version(curr_ver)
            print(f'[VLLM PROC {gen_rank}] Detected weights version {curr_ver} on server (will update when new version is available)')
    except Exception as e:
        print(f'[VLLM PROC {gen_rank}] Could not fetch initial version from server: {e}, starting with version -1')
    
    def try_update_model():
        """Update vLLM weights from orchestrator if new version available."""
        nonlocal curr_ver
        old_version = curr_ver
        try:
            updated = sampling_service.maybe_update_weights(orchestrator_service, gen_rank=gen_rank)
            if updated:
                curr_ver = sampling_service.get_current_version()
                print(f"[GEN {gen_rank}] ✓ Model weights successfully updated: v{old_version} → v{curr_ver}")
            # Note: maybe_update_weights already logs when updates occur, so we don't need to log when update is False
        except Exception as e:
            print(f"[GEN {gen_rank}] ✗ Exception in try_update_model: {e}")
            import traceback
            traceback.print_exc()

    # Main generation loop using algorithm
    rollout_number = worker_args.get('rollout_num', 8)
    training_batch_size = worker_args.get('train_batch_size', 1)
    problem_count = 0
    sampler_compute_interval = max(5.0, float(worker_args.get('sampler_compute_report_interval', 60.0)))
    sampler_token_threshold = max(0, int(worker_args.get('sampler_compute_token_threshold', 32768)))
    sampler_compute_last = time.time()
    sampler_gpu_seconds_acc = 0.0
    sampler_tokens_acc = 0
    sampler_worker_id = f"{socket.gethostname()}-sampler-{gen_rank}-{os.getpid()}"
    max_pending_samples = worker_args.get('max_pending_samples', 1600)
    gen_pending_time = worker_args.get('gen_pending_time', 10.0)
    max_batch_retry = worker_args.get('max_batch_retry', 3)
    max_upload_retries = worker_args.get('max_upload_retries', 10)
    max_fetch_retries = worker_args.get('max_fetch_retries', 10)
    retry_backoff_factor = worker_args.get('retry_backoff_factor', 2)
    retry_max_wait = worker_args.get('retry_max_wait', 60)
    log_error_throttle_interval = worker_args.get('log_error_throttle_interval', 60)
    
    # Error log throttling to prevent log spam
    error_counts = defaultdict(int)
    last_error_log_time = {}
    
    def log_error_with_throttle(error_msg, min_interval=None):
        """Log error with throttling to prevent spam"""
        if min_interval is None:
            min_interval = log_error_throttle_interval
        now = time.time()
        error_key = str(error_msg)[:200]  # First 200 chars of error message
        
        if error_key not in last_error_log_time:
            last_error_log_time[error_key] = 0
        
        if now - last_error_log_time[error_key] >= min_interval:
            count = error_counts[error_key]
            if count > 0:
                print(f"[GEN {gen_rank}] {error_msg} (occurred {count + 1} times in last {min_interval}s)")
            else:
                print(f"[GEN {gen_rank}] {error_msg}")
            last_error_log_time[error_key] = now
            error_counts[error_key] = 0
        else:
            error_counts[error_key] += 1

    def flush_sampler_compute(force: bool = False):
        nonlocal sampler_compute_last, sampler_gpu_seconds_acc, sampler_tokens_acc
        if orchestrator_service is None:
            sampler_compute_last = time.time()
            sampler_gpu_seconds_acc = 0.0
            sampler_tokens_acc = 0
            return
        if not force:
            if sampler_tokens_acc <= 0 and sampler_gpu_seconds_acc <= 0:
                return
            elapsed = time.time() - sampler_compute_last
            if elapsed < sampler_compute_interval and sampler_tokens_acc < sampler_token_threshold:
                return
        gpu_seconds = sampler_gpu_seconds_acc
        if gpu_seconds <= 0 and force:
            gpu_seconds = max(0.0, time.time() - sampler_compute_last)
        if gpu_seconds <= 0 and sampler_tokens_acc <= 0:
            sampler_compute_last = time.time()
            return
        try:
            orchestrator_service.report_compute_usage(
                role="actor",
                gpu_seconds=float(max(gpu_seconds, 0.0)),
                tokens=int(max(sampler_tokens_acc, 0)),
                worker_id=sampler_worker_id,
            )
        except Exception:
            pass
        sampler_compute_last = time.time()
        sampler_gpu_seconds_acc = 0.0
        sampler_tokens_acc = 0
    
    orchestrator_failure_count = 0
    eval_owner_id = f"{socket.gethostname()}-sampler-{gen_rank}-{os.getpid()}"
    # Track versions that this worker has already evaluated
    # This ensures each version is evaluated only once per worker
    # (orchestrator's claim mechanism ensures only one worker can claim a job)
    processed_eval_versions = set()
    
    for it in range(99999999):
        # Check for evaluation jobs before fetching problems
        # Process evaluation job if available (uses same SamplingService to avoid GPU memory issues)
        # Each version is evaluated only once - processed_versions tracks completed evaluations
        if orchestrator_service is not None:
            try:
                eval_processed = _process_eval_job(orchestrator_service, sampling_service, eval_owner_id, gen_rank, processed_eval_versions)
                if eval_processed:
                    # Evaluation job was processed, continue to next iteration to check for more
                    continue
            except Exception as e:
                # Log evaluation errors but don't crash
                print(f"[EVAL {gen_rank}] Error checking/processing evaluation job: {e}")
                pass
        
        # Each worker fetches problems directly from orchestrator
        raw_item = None
        try:
            data = orchestrator_service.fetch_problem()
            orchestrator_failure_count = 0  # Reset counter on success
            if data.get("end"):
                print(f'\n[GEN {gen_rank}] Generation worker finished, sending end signal to orchestrator ...')
                time.sleep(5)
                try:
                    orchestrator_service.upload_samples({'end': 1})
                except Exception as e:
                    print(f"[GEN {gen_rank}] Failed to send end signal: {e}")
                break
            if data.get("ok") and "problem" in data:
                problem = data["problem"]
                problem_id = data.get("_problem_id")
                task_id = uuid.uuid4().hex
                raw_item = {'task_id': task_id, 'batch': problem, 'retry': 0, '_problem_id': problem_id}
                problem_count += 1
                if problem_count % 50 == 0:
                    print(f"[GEN {gen_rank}] Fetched {problem_count} problems directly")
            elif data.get("empty"):
                time.sleep(0.5)
                continue
            else:
                print(f"[GEN {gen_rank}] Unexpected response from /problem/get: {data}")
                time.sleep(1)
                continue
        except Exception as e:
            orchestrator_failure_count += 1
            error_msg = str(e)
            
            # Check if we should gracefully shutdown
            if orchestrator_failure_count >= max_fetch_retries:
                log_error_with_throttle(
                    f"Orchestrator unavailable for {max_fetch_retries} consecutive attempts. Exiting gracefully.",
                    min_interval=0
                )
                flush_sampler_compute(force=True)
                break  # Exit gracefully
            
            # Distinguish timeout errors for better logging
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                log_error_with_throttle(
                    f"Timeout fetching problem (orchestrator may be busy, failure {orchestrator_failure_count}/{max_fetch_retries}): {e}"
                )
                time.sleep(min(2 * orchestrator_failure_count, 30))  # Exponential backoff, max 30s
            elif "connection" in error_msg.lower() or "refused" in error_msg.lower():
                wait_time = min(retry_backoff_factor ** orchestrator_failure_count, retry_max_wait)
                log_error_with_throttle(
                    f"Connection error fetching problem (failure {orchestrator_failure_count}/{max_fetch_retries}, waiting {wait_time}s): {e}"
                )
                time.sleep(wait_time)
            else:
                log_error_with_throttle(
                    f"Error fetching problem (failure {orchestrator_failure_count}/{max_fetch_retries}): {e}"
                )
                time.sleep(min(orchestrator_failure_count, 10))
            continue
        
        if raw_item is None:
            continue

        task_id = raw_item.get('task_id')
        prompt_item = raw_item.get('batch')
        retry_count = int(raw_item.get('retry', 0))
        problem_id = raw_item.get('_problem_id')

        if task_id:
            try:
                pending_tasks[task_id] = {'task': raw_item, 'timestamp': time.time()}
            except Exception as reg_error:
                print(f"[GEN {gen_rank}] Failed to register pending batch {task_id}: {reg_error}")

        try:
            # Update model weights if available
            try_update_model()
            
            gen_start_time = time.time()
            if prompt_item is None:
                if task_id:
                    pending_tasks.pop(task_id, None)
                continue
            
            # Use algorithm to process problem
            finished_samples = sampler_algo.process_problem(prompt_item, services)
            # Note: hooks are not called in worker process (would require passing hook functions)

            # Get tree structure if available (TreePO algorithm stores it)
            finished_nodes = None
            head_node = None
            if hasattr(sampler_algo, '_last_finished_nodes'):
                finished_nodes = sampler_algo._last_finished_nodes
            if hasattr(sampler_algo, '_last_head_node'):
                head_node = sampler_algo._last_head_node

            # Format and upload samples in batches
            for sending_batch in range(0, len(finished_samples), training_batch_size):
                batch_samples = finished_samples[sending_batch:sending_batch+training_batch_size]
                data = sampler_algo.format_samples_for_upload(batch_samples, problem=prompt_item)
                
                # Add metadata
                try:
                    hostname = os.uname().nodename
                except Exception:
                    hostname = 'unknown'
                data['node_id'] = f"{hostname}-gpu{gen_rank}"
                data['node_version'] = curr_ver
                if problem_id:
                    data['_problem_id'] = problem_id

                # Save samples/tree structure (only once per problem, for first batch)
                if sending_batch == 0:
                    log_dir = worker_args.get('log_dir')
                    # Do not save on sampler node; orchestrator will persist samples.
                    # Keep flags for potential future use, but skip local save.
                    # save_samples_enabled = worker_args.get('save_samples', True)
                    # save_tree_enabled = worker_args.get('save_tree_structure', True)
                    # save_individual_enabled = worker_args.get('save_individual_samples', False)
                    pass

                # Backpressure: if server queue is full, wait and retry until enqueued
                # With max retry limit and exponential backoff to prevent infinite loops
                resp_json = None
                upload_retry_count = 0
                max_queue_full_retries = 100  # Allow many retries for queue full (normal backpressure)
                
                while upload_retry_count < max_upload_retries + max_queue_full_retries:
                    try:
                        resp_json = orchestrator_service.upload_samples(data)
                        if isinstance(resp_json, dict) and resp_json.get('queued') is False:
                            # Queue full - this is expected backpressure, allow more retries
                            if upload_retry_count < max_queue_full_retries:
                                time.sleep(gen_pending_time)
                                upload_retry_count += 1
                                continue
                            else:
                                log_error_with_throttle(
                                    f"Queue full, max backpressure retries ({max_queue_full_retries}) exceeded. Dropping samples."
                                )
                                break
                        # Successfully uploaded
                        break
                    except Exception as upload_err:
                        upload_retry_count += 1
                        error_msg = str(upload_err)
                        
                        if upload_retry_count >= max_upload_retries:
                            log_error_with_throttle(
                                f"Max upload retries ({max_upload_retries}) exceeded. Dropping samples: {upload_err}",
                                min_interval=0
                            )
                            resp_json = None
                            break  # Give up after max retries
                        
                        # Calculate wait time with exponential backoff
                        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                            wait_time = min(retry_backoff_factor ** upload_retry_count, retry_max_wait)
                            log_error_with_throttle(
                                f"Connection error uploading samples (retry {upload_retry_count}/{max_upload_retries}, waiting {wait_time}s): {upload_err}"
                            )
                            time.sleep(wait_time)
                        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                            wait_time = min(2 * upload_retry_count, retry_max_wait)
                            log_error_with_throttle(
                                f"Timeout uploading samples (retry {upload_retry_count}/{max_upload_retries}, waiting {wait_time}s): {upload_err}"
                            )
                            time.sleep(wait_time)
                        else:
                            wait_time = min(upload_retry_count, 10)
                            log_error_with_throttle(
                                f"Error uploading samples (retry {upload_retry_count}/{max_upload_retries}, waiting {wait_time}s): {upload_err}"
                            )
                            time.sleep(wait_time)
                        continue

                # Check if queue is too full
                try:
                    remain_cnt = (resp_json or {}).get('remain_cnt', 0)
                    if remain_cnt > max_pending_samples:
                        print(f'[GEN {gen_rank}] pending samples too many ({remain_cnt}), wait for training process ...')
                        time.sleep(gen_pending_time)
                except Exception:
                    pass
                inputs_tensor = data.get("inputs")
                if isinstance(inputs_tensor, torch.Tensor):
                    sampler_tokens_acc += int(inputs_tensor.numel())

            if task_id:
                pending_tasks.pop(task_id, None)
            sampler_gpu_seconds_acc += max(time.time() - gen_start_time, 0.0)
            flush_sampler_compute()

        except Exception as e:
            print(f'[GEN {gen_rank}] Error in generation worker: {e}')
            import traceback
            traceback.print_exc()
            if prompt_item is not None:
                if retry_count >= max_batch_retry:
                    print(f"[GEN {gen_rank}] Dropping batch after {retry_count} retries.")
                else:
                    try:
                        raw_item['retry'] = retry_count + 1
                        Q_data.put(raw_item)
                    except Exception as requeue_error:
                        print(f"[GEN {gen_rank}] Failed to requeue batch: {requeue_error}")
            if task_id:
                pending_tasks.pop(task_id, None)
            continue
    flush_sampler_compute(force=True)
    
def load_weights_from_server(model_runner, orchestrator_url, model_id, version):
    """
    Runs inside each vLLM worker process to download and load weights directly
    from the orchestrator server, avoiding serialization of large tensors via RPC.
    
    Note: This function is called by vLLM's collective_rpc, so it must be a module-level function.
    The first argument is the model_runner object from vLLM.
    """
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
        loaded_sd = torch.load(buf, map_location='cpu')
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
        return (False, f"Worker {worker_id} FAILED while downloading/applying: {str(e)}\n{traceback.format_exc()}")

# Note: TreeNode, get_ancestors, backpropagation, extract_boxed_answer have been moved to
# ralo/algorithms/treepo/__init__.py. Import them from there if needed.
# pad_lists has been moved to ralo/utils.py

class CPUOffloadTrainer:
    def __init__(self, model_patch, lr=1e-6, accum_steps=16, grad_offload=False, gradient_checkpointing_ratio=1, init_optimizer=True):
        # Initialize distributed first to avoid race conditions during model loading
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if get_world_size() > 1:
            if not dist.is_initialized():
                torch.distributed.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
        else: 
            device = 'cuda'
        
        # Synchronize before loading model to avoid file system conflicts
        if dist.is_initialized():
            dist.barrier()
        
        # Load model with optimizations for distributed loading
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"[CPUOffloadTrainer] Rank 0: Loading model from {model_patch}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_patch,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None  # We'll move manually to avoid conflicts
        )
        
        # Ensure all ranks finish loading before proceeding
        if dist.is_initialized():
            dist.barrier()
        
        # Convert dtype if not already set (some models may not respect torch_dtype)
        if self.model.dtype != torch.bfloat16:
            self.model = self.model.to(dtype=torch.bfloat16)
        
        self.model.train()
        
        # Move to device after loading
        self.model.to(device)
        self.device = self.model.device
        
        # Synchronize after moving to device
        if dist.is_initialized():
            dist.barrier()
        
        self.accum_steps = accum_steps
        self.model.gradient_checkpointing_enable()
        enable_gradient_checkpointing(self.model, gradient_checkpointing_ratio)
        self.opt = None
        self.cpu_optimizer = None
        self.original_device_map = {}
        self.init_optimizer = init_optimizer
        if init_optimizer:
            from .cpuadamw import CPUAdamW, DistributedCPUAdamW
            if dist.is_initialized():
                CPUAdamW = DistributedCPUAdamW
            self.opt = CPUAdamW(self.model.parameters(), lr=lr, accum_steps=accum_steps, grad_offload=grad_offload)
            self.cpu_optimizer = self.opt.cpu_optimizer if hasattr(self.opt, "cpu_optimizer") else None
        self.engine = self.model
    
    def backward(self, loss): 
        # Pre-backward memory cleanup to maximize available memory
        # This is critical for preventing OOM during backward pass
        self._clear_activations()
        
        # Perform backward pass
        # Note: Gradient checkpointing will recompute activations during backward,
        # so we need maximum available memory before this call
        loss.backward()
        
        # Immediately offload gradients to CPU if grad_offload is enabled
        # This prevents OOM during backward pass for long sequences
        # This is the key optimization inspired by LSRL that allows training 14K+ sequences
        if self.opt is not None and hasattr(self.opt, 'grad_offload') and self.opt.grad_offload:
            self._offload_gradients_to_cpu()
        
        # Clip gradients (this should be done before offloading for accuracy)
        # But we do it after offload check to minimize memory usage
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Aggressively free intermediate activations after backward
        # This is critical for long sequences (14K+) during gradient accumulation
        self._clear_activations()

    def _offload_gradients_to_cpu(self):
        """
        Immediately offload gradients to CPU to free GPU memory.
        This is called after backward() when grad_offload=True to prevent OOM.
        This is the key optimization that allows training long sequences (14K+) on limited GPU memory.
        """
        if self.opt is None:
            return
        
        import torch.distributed as dist
        
        # For DistributedCPUAdamW
        if dist.is_initialized() and hasattr(self.opt, 'gpu_params'):
            # All-reduce gradients first (distributed training)
            for p in self.opt.gpu_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.AVG)
            
            # Only rank 0 offloads to CPU
            if dist.get_rank() == 0 and hasattr(self.opt, 'original_device_map'):
                async_grad_transfers = []
                for cpu_p, original_p in self.opt.original_device_map.items():
                    if original_p.grad is not None:
                        scaled_grad = (original_p.grad / self.opt.accum_steps).to('cpu', non_blocking=True)
                        async_grad_transfers.append((cpu_p, scaled_grad))
                        original_p.grad = None
                
                for cpu_p, scaled_grad in async_grad_transfers:
                    if cpu_p.grad is None:
                        cpu_p.grad = scaled_grad
                    else:
                        cpu_p.grad = cpu_p.grad + scaled_grad
            else:
                # Other ranks just clear gradients
                for p in self.opt.gpu_params:
                    if p.grad is not None:
                        p.grad = None
        
        # For SoloCPUAdamW
        elif hasattr(self.opt, 'original_device_map'):
            async_grad_transfers = []
            for cpu_p, original_p in self.opt.original_device_map.items():
                if original_p.grad is not None:
                    # Move gradient to CPU asynchronously and scale by accum_steps
                    scaled_grad = (original_p.grad / self.opt.accum_steps).to('cpu', non_blocking=True)
                    async_grad_transfers.append((cpu_p, scaled_grad))
                    # Clear GPU gradient immediately to free memory
                    original_p.grad = None
            
            # Accumulate gradients on CPU
            for cpu_p, scaled_grad in async_grad_transfers:
                if cpu_p.grad is None:
                    cpu_p.grad = scaled_grad
                else:
                    cpu_p.grad = cpu_p.grad + scaled_grad
        
        # Clear any remaining GPU gradients and free memory
        # This is critical to prevent memory fragmentation
        torch.cuda.empty_cache()
        
        # Synchronize to ensure transfer completes
        # This helps reduce memory fragmentation by ensuring all operations finish
        torch.cuda.synchronize()
        
        # Additional cleanup after gradient offload
        # This helps free any remaining intermediate tensors
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def _clear_activations(self):
        """
        Aggressively clear intermediate activations to free GPU memory.
        This is especially important during gradient accumulation for long sequences.
        Enhanced version to prevent OOM errors.
        """
        # For models with attention, explicitly clear KV cache if exists
        # Do this first to free model-specific caches
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for layer in self.model.model.layers:
                # Clear any cached activations in attention layers
                if hasattr(layer, 'self_attn'):
                    if hasattr(layer.self_attn, '_attn_cache'):
                        layer.self_attn._attn_cache = None
                    # Clear past_key_values if exists (for generation)
                    if hasattr(layer.self_attn, 'past_key_value'):
                        layer.self_attn.past_key_value = None
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, '_cache'):
                    layer.mlp._cache = None
        
        # Clear Python references to intermediate tensors
        # This helps Python GC to free memory faster
        import gc
        gc.collect()
        
        # Clear CUDA cache to reduce memory fragmentation
        # This is especially important after many accumulation steps
        torch.cuda.empty_cache()
        
        # Force synchronization to ensure all operations complete
        # This helps reduce memory fragmentation
        torch.cuda.synchronize()

    def step(self):
        if self.opt is None:
            return False
        return self.opt.step()

    def get_model(self): 
        return self.model

    def collect_gradients(self, to_cpu=True, clear=True):
        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            tensor = param.grad.detach()
            if to_cpu:
                tensor = tensor.cpu()
            grads[name] = tensor.clone()
            if clear:
                param.grad = None
        return grads

class RALO:
    def __init__(self, model_path, epochs=1, rollout_num=8, train_data=None, gen_device=4, train_batch_size=2, 
                clip_param=0.2,  orchestrator_url="http://localhost:59888",
                 gen_max_tokens=4096, gen_temperature=0.9, gen_top_p=None, gen_top_k=None, gen_min_p=None, enable_thinking=False,
                 max_pending_samples=1600, gen_pending_time=10, skip_zero_groups=False, vllm_kwargs=None, 
                  TreePO_kwargs=None, init_trainer=True, max_batch_retry=3, pending_retry_timeout=360,
                  sampler_config: SamplerConfig | None = None, trainer_config: TrainerConfig | None = None,
                  orchestrator_config=None, system_prompt=None, **kwargs):

        self.model_path = model_path
        self.gen_device = [gen_device] if isinstance(gen_device, int) else list(gen_device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_id = os.path.basename(str(model_path)).strip() or 'default'
        self.model_id = re.sub(r'[^A-Za-z0-9_.-]+', '-', base_id)
        TreePO_kwargs = TreePO_kwargs or {}
        self.TreePO_kwargs = dict(TreePO_kwargs)
        self.sampler_config = sampler_config or SamplerConfig()
        self.trainer_config = trainer_config or TrainerConfig()
        self.sampler_algorithm_name = self.sampler_config.algorithm.lower()
        self.trainer_algorithm_name = self.trainer_config.algorithm.lower()
        sampler_params = self.sampler_config.params or {}
        trainer_params = self.trainer_config.params or {}
        self.sampler_compute_report_interval = float(sampler_params.get("compute_report_interval", 60.0))
        self.sampler_compute_token_threshold = int(sampler_params.get("compute_report_token_threshold", 32768))
        self.max_upload_retries = int(sampler_params.get("max_upload_retries", 10))
        self.max_fetch_retries = int(sampler_params.get("max_fetch_retries", 10))
        self.retry_backoff_factor = float(sampler_params.get("retry_backoff_factor", 2))
        self.retry_max_wait = int(sampler_params.get("retry_max_wait", 60))
        self.log_error_throttle_interval = int(sampler_params.get("log_error_throttle_interval", 60))
        self.trainer_compute_report_interval = float(trainer_params.get("compute_report_interval", 60.0))
        self.trainer_compute_token_threshold = int(trainer_params.get("compute_report_token_threshold", 32768))
        cfg_treepo = sampler_params.get('treepo_kwargs') or sampler_params.get('TreePO_kwargs')
        if cfg_treepo:
            self.TreePO_kwargs.update(cfg_treepo)
        # Extract DAPO_kwargs from sampler and trainer params
        cfg_dapo_sampler = sampler_params.get('dapo_kwargs') or sampler_params.get('DAPO_kwargs')
        cfg_dapo_trainer = trainer_params.get('dapo_kwargs') or trainer_params.get('DAPO_kwargs')
        # Merge DAPO kwargs (trainer params take precedence if both exist)
        dapo_kwargs = {}
        if cfg_dapo_sampler:
            dapo_kwargs.update(cfg_dapo_sampler)
        if cfg_dapo_trainer:
            dapo_kwargs.update(cfg_dapo_trainer)
        self.DAPO_kwargs = dapo_kwargs
        self.rollout_num = rollout_num
        self.train_data = train_data
        self.reward_fns = []
        self.epochs = epochs
        # Track max steps target for the central controller (orchestrator)
        self._max_steps_for_orch = epochs * len(train_data) * rollout_num // train_batch_size // get_world_size() if train_data else None

        self._hooks = {}

        if train_batch_size > rollout_num:
            assert train_batch_size % rollout_num == 0, "train_batch_size must be divisible by rollout_num"
            self.num_mix_forward_batches = train_batch_size // rollout_num
            self.train_batch_size = rollout_num
            raise Exception("mix_forward_batches does not faster, use train_batch_size == rollnum instead")
        else:
            assert rollout_num % train_batch_size == 0, "rollout_num must be divisible by train_batch_size"
            self.train_batch_size = train_batch_size
            self.num_mix_forward_batches = 1

        self.skip_zero_groups = skip_zero_groups
        self.orchestrator_url = orchestrator_url.rstrip('/') if orchestrator_url else None
        self._sampler_client = None
        self._trainer_client = None
        self.gen_max_tokens = gen_max_tokens
        self.gen_temperature = gen_temperature
        self.gen_top_p = gen_top_p
        self.gen_top_k = gen_top_k
        self.gen_min_p = gen_min_p
        self.enable_thinking = enable_thinking
        # Store system_prompt for worker processes (default if not provided)
        self.system_prompt = system_prompt if system_prompt is not None else "Please reason step by step, and put your final answer within \\boxed{}."
        # Read clip_param from trainer_config if available, otherwise use parameter
        self.clip_param = self.trainer_config.params.get("clip_param", clip_param)

        self.max_pending_samples = max_pending_samples
        self.vllm_kwargs = vllm_kwargs or {}
        self.gen_pending_time = gen_pending_time
        self._lock_owner_id = None
        # Read max_batch_retry and pending_retry_timeout from trainer_config if available
        self.max_batch_retry = max(0, int(self.trainer_config.params.get("max_batch_retry", max_batch_retry)))
        self.pending_retry_timeout = max(5, int(self.trainer_config.params.get("pending_retry_timeout", pending_retry_timeout)))
        # Connection retry settings for get_batch
        self.max_get_batch_retries = int(trainer_params.get("max_get_batch_retries", 10))
        self.get_batch_retry_backoff_factor = float(trainer_params.get("get_batch_retry_backoff_factor", 2.0))
        self.get_batch_retry_max_wait = int(trainer_params.get("get_batch_retry_max_wait", 60))
        self._worker_micro_step = 0
        self._last_synced_version = None
        self._last_version_poll = 0.0
        # Get version_poll_interval from sampler config if available
        self._version_poll_interval = self.sampler_config.params.get("version_poll_interval", 5.0)
        self._init_trainer = init_trainer  # Store for later use in train()
        self._orchestrator_config = orchestrator_config  # Store orchestrator config for timeout settings

        # Initialize service layer
        if self.orchestrator_url:
            # Extract timeout config from orchestrator config if available
            timeout_config = None
            # Try to get orchestrator config - check if passed directly or via config
            orch_cfg = self._orchestrator_config
            
            if orch_cfg is not None:
                timeout_config = {
                    "get_batch_timeout": getattr(orch_cfg, 'get_batch_timeout', 60.0),
                    "send_gradients_timeout": getattr(orch_cfg, 'send_gradients_timeout', 300.0),
                    "download_weights_timeout": getattr(orch_cfg, 'download_weights_timeout', 600.0),
                    "upload_samples_timeout": getattr(orch_cfg, 'upload_samples_timeout', 300.0),
                    "fetch_problem_timeout": getattr(orch_cfg, 'fetch_problem_timeout', 10.0),
                    "register_timeout": getattr(orch_cfg, 'register_timeout', 10.0),
                    "stats_timeout": getattr(orch_cfg, 'stats_timeout', 5.0),
                    "heartbeat_timeout": getattr(orch_cfg, 'heartbeat_timeout', 2.0),
                    "version_check_timeout": getattr(orch_cfg, 'version_check_timeout', 5.0),
                    "next_step_timeout": getattr(orch_cfg, 'next_step_timeout', 5.0),
                    "lock_timeout": getattr(orch_cfg, 'lock_timeout', 5.0),
                    "lock_ttl": getattr(orch_cfg, 'lock_ttl', 30.0),
                    "chunk_size_mb": getattr(orch_cfg, 'chunk_size_mb', 50),
                    "download_chunk_size_mb": getattr(orch_cfg, 'download_chunk_size_mb', 32),
                }
            self.orchestrator_service = OrchestratorService(
                orchestrator_url=self.orchestrator_url,
                model_id=self.model_id,
                retry_interval=self.gen_pending_time,
                timeout_config=timeout_config,
            )
        else:
            self.orchestrator_service = None

        self.model_service = ModelService(model_path=self.model_path, model_id=self.model_id)
        self.sampling_service = None  # Will be initialized in gen_worker
        self.training_service = None  # Will be initialized in train()
        self._eval_stop_event = threading.Event()
        self._eval_thread = None

    def _eval_worker_loop(self):
        """
        Background evaluation worker that claims eval jobs from orchestrator
        and runs them using a temporary SamplingService instance.
        Stops when stop event is set and no pending/running jobs remain.
        """
        if self.orchestrator_service is None or not self.orchestrator_url:
            return
        owner_id = f"{socket.gethostname()}-eval-{os.getpid()}"
        orch = self.orchestrator_service
        stop_event = self._eval_stop_event
        idle_sleep = 5.0
        import traceback
        while True:
            try:
                preview = orch.get_eval_job()
            except Exception:
                preview = {"empty": True}
            if not isinstance(preview, dict):
                preview = {"empty": True}
            if preview.get("empty"):
                if stop_event.is_set():
                    stats = orch.get_eval_stats()
                    pending = stats.get("pending", 0) if isinstance(stats, dict) else 0
                    running = stats.get("running", 0) if isinstance(stats, dict) else 0
                    if pending == 0 and running == 0:
                        break
                time.sleep(idle_sleep)
                continue
            job_id = preview.get("job_id")
            if not job_id:
                time.sleep(1.0)
                continue
            try:
                claim = orch.claim_eval_job(job_id=job_id, owner=owner_id, ttl=90.0)
            except Exception:
                time.sleep(1.0)
                continue
            if not isinstance(claim, dict) or not claim.get("ok"):
                time.sleep(1.0)
                continue
            benchmark_cfg = claim.get("benchmark_cfg", {})
            devices = claim.get("devices") or []
            target_devices = devices if (isinstance(devices, list) and len(devices) > 0) else self.gen_device
            target_gpu = int(target_devices[0]) if (isinstance(target_devices, list) and len(target_devices) > 0) else 0
            from .services import SamplingService as _SamplingService
            eval_sampling = None
            start_t = time.time()
            try:
                eval_sampling = _SamplingService(
                    model_path=self.model_path,
                    model_id=self.model_id,
                    orchestrator_url=self.orchestrator_url,
                    vllm_kwargs=self.vllm_kwargs,
                    gen_temperature=self.gen_temperature,
                    version_poll_interval=self._version_poll_interval,
                )
                eval_sampling.initialize(gen_device=target_gpu, gen_rank=0)
                try:
                    eval_sampling.maybe_update_weights(orch, gen_rank=0, force=True)
                except Exception:
                    pass
                from .eval.runner import run_benchmark
                metrics, sample_rows, bench_stats = run_benchmark(
                    benchmark_cfg=benchmark_cfg,
                    sampling_service=eval_sampling,
                )
            except Exception as e:
                print(f"[EVAL] Failed running benchmark {benchmark_cfg.get('name')}: {e}")
                traceback.print_exc()
                metrics, sample_rows, bench_stats = {"error": 1.0}, None, {"generated_tokens": 0}
            finally:
                try:
                    del eval_sampling
                except Exception:
                    pass
            duration = time.time() - start_t
            generated_tokens = int((bench_stats or {}).get("generated_tokens", 0))
            try:
                orch.report_compute_usage(
                    role="evaluator",
                    gpu_seconds=float(max(duration, 0.0)),
                    tokens=generated_tokens,
                    worker_id=owner_id,
                )
            except Exception:
                pass
            try:
                orch.report_eval_job(job_id=job_id, metrics=metrics, samples=sample_rows, duration_sec=duration)
            except Exception as e:
                print(f"[EVAL] Failed to report results for {job_id}: {e}")

        print("[EVAL] Evaluation worker exiting (no pending jobs).")

    def _stop_eval_worker(self):
        """
        Signal eval worker to stop after outstanding jobs complete and wait briefly.
        """
        if getattr(self, "_eval_stop_event", None) is None:
            return
        self._eval_stop_event.set()
        thread = getattr(self, "_eval_thread", None)
        if thread and thread.is_alive():
            thread.join(timeout=600.0)

    @property
    def sampler_client(self):
        """Deprecated: Use orchestrator_service instead."""
        if self._sampler_client is None and self.orchestrator_url:
            self._sampler_client = SamplerClient(self.orchestrator_url)
        return self._sampler_client

    @property
    def trainer_client(self):
        """Deprecated: Use orchestrator_service instead."""
        if self._trainer_client is None and self.orchestrator_url:
            self._trainer_client = TrainerClient(self.orchestrator_url)
        return self._trainer_client

    def run_generation_only(self):
        algo_cls = get_sampler_algorithm(self.sampler_algorithm_name)
        algo = algo_cls(self, self.sampler_config)
        algo.run()

    def train(self):
        # Initialize TrainingService before running algorithm
        if self.training_service is None:
            trainer_params = self.trainer_config.params
            self.training_service = TrainingService(
                model_path=self.model_path,
                lr=trainer_params.get('lr', 1e-6),
                accum_steps=trainer_params.get('accum_steps', 8192),
                grad_offload=trainer_params.get('grad_offload', False),
                gradient_checkpointing_ratio=trainer_params.get('gradient_checkpointing_ratio', 1.0),
                init_optimizer=False,  # Orchestrator handles optimizer
            )
            # Set version poll interval from config if available
            version_poll_interval = trainer_params.get('version_poll_interval', 5.0)
            self.training_service.set_version_poll_interval(version_poll_interval)
            self.training_service.initialize()
            self.trainer = self.training_service.get_trainer()
            self.accum_steps = self.training_service.accum_steps
        
        if self.trainer_algorithm_name == 'treepo':
            print('\nUsing RLVR training loop (TreePO algorithm selected)...\n')
            print('Available RLVR kwargs: cache_max_length, soft_max_length, hard_max_length, clip_param_high\n')
        
        algo_cls = get_trainer_algorithm(self.trainer_algorithm_name)
        algo = algo_cls(self, self.trainer_config)
        algo.run()
    
    def set_hook(self, name, func): self._hooks[name] = func
    def set_hooks(self, **hooks): self._hooks.update(hooks)
    def call_hook(self, name, *args, **kwargs):
        if name in self._hooks: return self._hooks[name](*args, **kwargs)
        return None

    def add_reward(self, reward_fn):
        self.reward_fns.append(reward_fn)

    def set_rollout_prompt_fn(self, user_fn): self._rollout_prompt_fn = user_fn
    def set_policy_prompt_fn(self, user_fn): self._policy_prompt_fn = user_fn
    def rollout_prompt_fn(self, item): return self._rollout_prompt_fn(self, item)
    def policy_prompt_fn(self, item): return self._policy_prompt_fn(self, item)

    def get_batch(self):
        """Get batch from orchestrator. Use orchestrator_service.get_batch() directly instead."""
        if not self.orchestrator_service:
            return None
        try:
            return self.orchestrator_service.get_batch()
        except Exception:
            return None

       
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove non-picklable/heavy runtime attributes before spawn
        state.pop('trainer', None)
        state.pop('tokenizer', None)
        state.pop('Q_data', None)
        state.pop('gen_procs', None)
        state.pop('manager', None)
        state.pop('pending_monitor_stop', None)
        state.pop('pending_monitor_thread', None)
        # Services are recreated in worker processes
        state.pop('model_service', None)
        state.pop('sampling_service', None)
        state.pop('training_service', None)
        state.pop('orchestrator_service', None)
        return state

    def gen_worker(self, Q_data, gen_device, gen_rank=0):

        os.environ["CUDA_VISIBLE_DEVICES"] = f'{gen_device}'
        cleanup_keys = [  
            'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'LOCAL_RANK',  
            'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_NAME',   
            'GROUP_WORLD_SIZE', 'ROLE_WORLD_SIZE',  
            'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_MAX_RESTARTS',  
            'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_USE_AGENT_STORE',  
            'TORCHELASTIC_ERROR_FILE',  
            'TORCH_NCCL_ASYNC_ERROR_HANDLING',  
            'NCCL_COMM_ID', 'NCCL_DEBUG', 'NCCL_SOCKET_IFNAME',  
        ]  
        for key in cleanup_keys: os.environ.pop(key, None)
        
        torch.cuda.set_device(0)
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        print(f"[GEN {gen_rank}] Generation worker process uses GPU {gen_device}")
        print(f"[GEN {gen_rank}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"[GEN {gen_rank}] PID: {os.getpid()}")

        # Initialize services in worker process
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Initialize SamplingService in worker process
        sampling_service = SamplingService(
            model_path=self.model_path,
            model_id=self.model_id,
            orchestrator_url=self.orchestrator_url,
            vllm_kwargs=self.vllm_kwargs,
            gen_temperature=self.gen_temperature,
            version_poll_interval=self._version_poll_interval,
        )
        sampling_service.initialize(gen_device=gen_device, gen_rank=gen_rank)
        
        # Initialize OrchestratorService in worker process
        orchestrator_service = OrchestratorService(
            orchestrator_url=self.orchestrator_url,
            model_id=self.model_id,
            retry_interval=self.gen_pending_time,
        )
        
        # Initialize model service
        model_service = ModelService(model_path=self.model_path, model_id=self.model_id)
        
        # Get sampler algorithm
        algo_cls = get_sampler_algorithm(self.sampler_algorithm_name)
        sampler_algo = algo_cls(self, self.sampler_config)
        
        # Services dict for algorithm
        services = {
            'model': model_service,
            'sampling': sampling_service,
            'orchestrator': orchestrator_service,
        }

        def QueueGetNowait(Q):
            try: return Q.get_nowait()
            except: return None

        # Initialize weight version tracking
        curr_ver = -1
        try:
            initial_version = orchestrator_service.latest_version()
            if initial_version is not None and initial_version >= 0:
                curr_ver = initial_version
                sampling_service.set_current_version(curr_ver)
                print(f'[VLLM PROC {gen_rank}] Detected weights version {curr_ver} on server (will update when new version is available)')
        except Exception as e:
            print(f'[VLLM PROC {gen_rank}] Could not fetch initial version from server: {e}, starting with version -1')
        
        def try_update_model():
            """Update vLLM weights from orchestrator if new version available."""
            nonlocal curr_ver
            old_version = curr_ver
            updated = sampling_service.maybe_update_weights(orchestrator_service, gen_rank=gen_rank)
            if updated:
                curr_ver = sampling_service.get_current_version()
                print(f"[GEN {gen_rank}] Successfully updated model weights from version {old_version} to version {curr_ver}")


        # Main generation loop using algorithm
        rollout_number = self.rollout_num 
        training_batch_size = self.train_batch_size
        problem_count = 0
        sampler_compute_interval = max(5.0, float(self.sampler_compute_report_interval))
        sampler_token_threshold = max(0, int(self.sampler_compute_token_threshold))
        sampler_compute_last = time.time()
        sampler_gpu_seconds_acc = 0.0
        sampler_tokens_acc = 0
        sampler_worker_id = f"{socket.gethostname()}-sampler-{gen_rank}-{os.getpid()}"

        def flush_sampler_compute(force: bool = False):
            nonlocal sampler_compute_last, sampler_gpu_seconds_acc, sampler_tokens_acc
            if orchestrator_service is None:
                sampler_compute_last = time.time()
                sampler_gpu_seconds_acc = 0.0
                sampler_tokens_acc = 0
                return
            if not force:
                if sampler_tokens_acc <= 0 and sampler_gpu_seconds_acc <= 0:
                    return
                elapsed = time.time() - sampler_compute_last
                if elapsed < sampler_compute_interval and sampler_tokens_acc < sampler_token_threshold:
                    return
            gpu_seconds = sampler_gpu_seconds_acc
            if gpu_seconds <= 0 and force:
                gpu_seconds = max(0.0, time.time() - sampler_compute_last)
            if gpu_seconds <= 0 and sampler_tokens_acc <= 0:
                sampler_compute_last = time.time()
                return
            try:
                orchestrator_service.report_compute_usage(
                    role="actor",
                    gpu_seconds=float(max(gpu_seconds, 0.0)),
                    tokens=int(max(sampler_tokens_acc, 0)),
                    worker_id=sampler_worker_id,
                )
            except Exception:
                pass
            sampler_compute_last = time.time()
            sampler_gpu_seconds_acc = 0.0
            sampler_tokens_acc = 0
        
        for it in range(99999999):
            # Each worker fetches problems directly from orchestrator
            raw_item = None
            try:
                data = orchestrator_service.fetch_problem()
                if data.get("end"):
                    print(f'\n[GEN {gen_rank}] Generation worker finished, sending end signal to orchestrator ...')
                    time.sleep(5)
                    try:
                        orchestrator_service.upload_samples({'end': 1})
                    except Exception as e:
                        print(f"[GEN {gen_rank}] Failed to send end signal: {e}")
                    break
                if data.get("ok") and "problem" in data:
                    problem = data["problem"]
                    problem_id = data.get("_problem_id")
                    task_id = uuid.uuid4().hex
                    raw_item = {'task_id': task_id, 'batch': problem, 'retry': 0, '_problem_id': problem_id}
                    problem_count += 1
                    if problem_count % 50 == 0:
                        print(f"[GEN {gen_rank}] Fetched {problem_count} problems directly")
                elif data.get("empty"):
                    time.sleep(0.5)
                    continue
                else:
                    print(f"[GEN {gen_rank}] Unexpected response from /problem/get: {data}")
                    time.sleep(1)
                    continue
            except Exception as e:
                error_msg = str(e)
                # Distinguish timeout errors for better logging
                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    print(f"[GEN {gen_rank}] Timeout fetching problem (orchestrator may be busy): {e}")
                    time.sleep(2)  # Longer wait for timeout
                elif "connection" in error_msg.lower() or "refused" in error_msg.lower():
                    print(f"[GEN {gen_rank}] Connection error fetching problem: {e}")
                    time.sleep(5)  # Longer wait for connection errors
                else:
                    print(f"[GEN {gen_rank}] Error fetching problem: {e}")
                    time.sleep(1)
                continue
            
            if raw_item is None:
                continue

            task_id = raw_item.get('task_id')
            prompt_item = raw_item.get('batch')
            retry_count = int(raw_item.get('retry', 0))
            problem_id = raw_item.get('_problem_id')

            if task_id:
                try:
                    pending_tasks[task_id] = {'task': raw_item, 'timestamp': time.time()}
                except Exception as reg_error:
                    print(f"[GEN {gen_rank}] Failed to register pending batch {task_id}: {reg_error}")

            try:
                # Update model weights if available
                try_update_model()
                
                gen_start_time = time.time()
                if prompt_item is None:
                    if task_id:
                        pending_tasks.pop(task_id, None)
                    continue
                
                # Use algorithm to process problem
                finished_samples = sampler_algo.process_problem(prompt_item, services)
                # Note: hooks are not called in worker process (would require passing hook functions)

                # Format and upload samples in batches
                for sending_batch in range(0, len(finished_samples), training_batch_size):
                    batch_samples = finished_samples[sending_batch:sending_batch+training_batch_size]
                    data = sampler_algo.format_samples_for_upload(batch_samples, problem=prompt_item)
                    
                    # Add metadata
                    try:
                        hostname = os.uname().nodename
                    except Exception:
                        hostname = 'unknown'
                    data['node_id'] = f"{hostname}-gpu{gen_rank}"
                    data['node_version'] = curr_ver
                    if problem_id:
                        data['_problem_id'] = problem_id

                    # Backpressure: if server queue is full, wait and retry until enqueued
                    resp_json = None
                    while True:
                        try:
                            resp_json = orchestrator_service.upload_samples(data)
                            if isinstance(resp_json, dict) and resp_json.get('queued') is False:
                                time.sleep(gen_pending_time)
                                continue
                            break
                        except Exception as upload_err:
                            error_msg = str(upload_err)
                            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                                print(f"[GEN {gen_rank}] Timeout uploading samples (retrying): {upload_err}")
                                time.sleep(2)  # Longer wait for timeout
                            else:
                                print(f"[GEN {gen_rank}] Error uploading samples (retrying): {upload_err}")
                                time.sleep(1.0)
                            continue

                    # Check if queue is too full
                    try:
                        remain_cnt = (resp_json or {}).get('remain_cnt', 0)
                        if remain_cnt > max_pending_samples:
                            print(f'[GEN {gen_rank}] pending samples too many ({remain_cnt}), wait for training process ...')
                            time.sleep(gen_pending_time)
                    except Exception:
                        pass
                    inputs_tensor = data.get("inputs")
                    if isinstance(inputs_tensor, torch.Tensor):
                        sampler_tokens_acc += int(inputs_tensor.numel())

                if task_id:
                    pending_tasks.pop(task_id, None)
                sampler_gpu_seconds_acc += max(time.time() - gen_start_time, 0.0)
                flush_sampler_compute()

            except Exception as e:
                print(f'[GEN {gen_rank}] Error in generation worker: {e}')
                import traceback
                traceback.print_exc()
                if prompt_item is not None:
                    if retry_count >= max_batch_retry:
                        print(f"[GEN {gen_rank}] Dropping batch after {retry_count} retries.")
                    else:
                        try:
                            raw_item['retry'] = retry_count + 1
                            Q_data.put(raw_item)
                        except Exception as requeue_error:
                            print(f"[GEN {gen_rank}] Failed to requeue batch: {requeue_error}")
                if task_id:
                    pending_tasks.pop(task_id, None)
                continue
        flush_sampler_compute(force=True)

    def start_gen_worker(self):
        print('\nSTART vLLM generation...\n')
        ctx = mp.get_context('spawn')
        # Limit queue size to avoid preloading all problems
        max_queue_size = 20  # Keep only a small buffer of problems
        self.Q_data = ctx.Queue(maxsize=max_queue_size)
        self.manager = ctx.Manager()
        self.pending_tasks = self.manager.dict()

        # Prepare arguments for gen_worker (avoid passing self to prevent pickle issues)
        # Note: rollout_prompt_fn and reward_fns are functions and cannot be pickled,
        # so they will be recreated in the worker process
        # Convert sampler_config to dict to ensure pickle compatibility
        from dataclasses import asdict
        sampler_config_dict = asdict(self.sampler_config) if hasattr(self.sampler_config, '__dict__') else {
            'algorithm': getattr(self.sampler_config, 'algorithm', 'treepo'),
            'params': getattr(self.sampler_config, 'params', {}),
        }
        
        # Get log_dir from environment or use default
        log_dir = os.environ.get("LOG_DIR") or os.environ.get("RUN_ID")
        if log_dir and not os.path.isabs(log_dir):
            # If relative path, assume it's under logs/
            log_dir = f"logs/{log_dir}"
        
        # Get sample saving configuration from sampler config
        sampler_params = sampler_config_dict.get('params', {})
        # Force sample saving from orchestrator side so logs stay with orchestrator
        # regardless of sampler node FS.
        save_samples_enabled = True
        save_tree_enabled = sampler_params.get('save_tree_structure', True)
        save_individual_enabled = sampler_params.get('save_individual_samples', False)
        
        worker_args = {
            'model_path': self.model_path,
            'model_id': self.model_id,
            'orchestrator_url': self.orchestrator_url,
            'vllm_kwargs': self.vllm_kwargs,
            'gen_temperature': self.gen_temperature,
            'gen_top_p': getattr(self, 'gen_top_p', None),
            'gen_top_k': getattr(self, 'gen_top_k', None),
            'gen_min_p': getattr(self, 'gen_min_p', None),
            'enable_thinking': getattr(self, 'enable_thinking', False),
            'system_prompt': getattr(self, 'system_prompt', "Please reason step by step, and put your final answer within \\boxed{}."),
            'gen_max_tokens': self.gen_max_tokens,
            'version_poll_interval': self._version_poll_interval,
            'gen_pending_time': self.gen_pending_time,
            'sampler_algorithm_name': self.sampler_algorithm_name,
            'sampler_config_dict': sampler_config_dict,  # Pass as dict instead of object
            'rollout_num': self.rollout_num,
            'train_batch_size': self.train_batch_size,
            'sampler_compute_report_interval': self.sampler_compute_report_interval,
            'sampler_compute_token_threshold': self.sampler_compute_token_threshold,
            'max_pending_samples': self.max_pending_samples,
            'max_batch_retry': self.max_batch_retry,
            'max_upload_retries': self.max_upload_retries,
            'max_fetch_retries': self.max_fetch_retries,
            'retry_backoff_factor': self.retry_backoff_factor,
            'retry_max_wait': self.retry_max_wait,
            'log_error_throttle_interval': self.log_error_throttle_interval,
            'TreePO_kwargs': self.TreePO_kwargs,
            'epochs': self.epochs,
            'clip_param': self.clip_param,
            'log_dir': log_dir,  # Add log_dir for sample saving
            'save_samples': save_samples_enabled,  # Enable/disable sample saving
            'save_tree_structure': save_tree_enabled,  # Enable/disable tree structure saving (TreePO)
            'save_individual_samples': save_individual_enabled,  # Enable/disable individual sample saving
        }

        # Start worker processes first
        # Note: We cannot pass functions directly, so we pass None and let worker recreate them
        # The functions will be recreated in the worker process using the tokenizer
        self.gen_procs = []
        for it, gendevice in enumerate(self.gen_device):
            # Pass None for functions - they will be recreated in worker process
            p = ctx.Process(target=_gen_worker_static, args=(
                self.Q_data, 
                self.pending_tasks, 
                gendevice, 
                it, 
                worker_args,
                None,  # rollout_prompt_fn - will be recreated
                None,  # reward_fns - will be recreated
            ))
            p.start()
            self.gen_procs.append(p)

        # Workers now fetch problems directly - no central fetcher needed
        print(f"[SAMPLER] Workers will fetch problems directly from orchestrator at {self.orchestrator_url}")

        self.pending_monitor_stop = threading.Event()
        self.pending_monitor_thread = threading.Thread(target=self._pending_monitor_loop, daemon=True)
        self.pending_monitor_thread.start()

    def wait_gen_workers(self):
        # Wait for worker processes (they fetch problems directly now)
        for p in getattr(self, 'gen_procs', []):
            try:
                p.join()
            except Exception:
                pass
        
        # Stop pending monitor
        if hasattr(self, 'pending_monitor_stop'):
            self.pending_monitor_stop.set()
        if hasattr(self, 'pending_monitor_thread'):
            try:
                self.pending_monitor_thread.join(timeout=5)
            except Exception:
                pass
        if hasattr(self, 'manager'):
            try:
                self.manager.shutdown()
            except Exception:
                pass

    def _pending_monitor_loop(self):
        check_interval = max(5, self.pending_retry_timeout // 2)
        while True:
            if getattr(self, 'pending_monitor_stop', None) is None:
                return
            if self.pending_monitor_stop.wait(timeout=check_interval):
                break
            try:
                pending_snapshot = list(getattr(self, 'pending_tasks', {}).items())
            except Exception as e:
                print(f"[GEN MONITOR] Failed to snapshot pending tasks: {e}")
                continue
            now = time.time()
            for task_id, meta in pending_snapshot:
                try:
                    task = meta.get('task')
                    timestamp = float(meta.get('timestamp', 0))
                except Exception:
                    try:
                        self.pending_tasks.pop(task_id, None)
                    except Exception:
                        pass
                    continue
                if task is None:
                    try:
                        self.pending_tasks.pop(task_id, None)
                    except Exception:
                        pass
                    continue
                if now - timestamp < self.pending_retry_timeout:
                    continue
                retry = int(task.get('retry', 0))
                if retry >= self.max_batch_retry:
                    print(f"[GEN MONITOR] Dropping batch {task_id} after {retry} retries (timeout).")
                    try:
                        self.pending_tasks.pop(task_id, None)
                    except Exception:
                        pass
                    continue
                task['retry'] = retry + 1
                try:
                    self.pending_tasks.pop(task_id, None)
                except Exception:
                    pass
                try:
                    self.Q_data.put(task)
                    print(f"[GEN MONITOR] Requeued timed-out batch {task_id} (retry {task['retry']}).")
                except Exception as e:
                    print(f"[GEN MONITOR] Failed to requeue batch {task_id}: {e}")

    def _rlvr_run_sampler(self):
        self.start_gen_worker()
        # Evaluation jobs are now processed by generation workers directly
        # No separate evaluation worker thread needed - avoids GPU memory conflicts
        self.wait_gen_workers()
        
    def _rlvr_run_trainer(self):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        training_service = self.training_service
        orchestrator_service = self.orchestrator_service
        
        # TrainingService will automatically generate unique worker ID with UUID
        # No need to manually set - it uses get_unique_worker_id() internally
        
        if self.trainer is not None:
            # Register with orchestrator
            orchestrator_service.register_trainer(
                worker_id=training_service.get_lock_owner_id(),
                accum_steps=self.accum_steps,
                max_steps=getattr(self, '_max_steps_for_orch', None),
                allow_failure=True,
            )
        
        should_start_gen = os.environ.get('START_GEN_WITH_TRAIN', '0') == '1'
        if should_start_gen and self.rank == 0:
            self.start_gen_worker()

        def get_batch_with_waiting():
            """Get batch from orchestrator with connection error handling and retry logic."""
            import requests
            from collections import defaultdict
            
            # Error log throttling
            error_counts = defaultdict(int)
            last_error_log_time = {}
            log_error_throttle_interval = 60  # Log same error max once per minute
            
            def log_error_with_throttle(error_msg, min_interval=None):
                """Log error with throttling to prevent spam"""
                if min_interval is None:
                    min_interval = log_error_throttle_interval
                now = time.time()
                error_key = str(error_msg)[:200]  # First 200 chars
                
                if error_key not in last_error_log_time:
                    last_error_log_time[error_key] = 0
                
                if now - last_error_log_time[error_key] >= min_interval:
                    count = error_counts[error_key]
                    if count > 0:
                        print(f"[TRAINER rank={self.rank}] {error_msg} (occurred {count + 1} times in last {min_interval}s)")
                    else:
                        print(f"[TRAINER rank={self.rank}] {error_msg}")
                    last_error_log_time[error_key] = now
                    error_counts[error_key] = 0
                else:
                    error_counts[error_key] += 1
            
            connection_retry_count = 0
            max_retries = self.max_get_batch_retries
            backoff_factor = self.get_batch_retry_backoff_factor
            max_wait = self.get_batch_retry_max_wait
            
            while True:
                try:
                    batch = orchestrator_service.get_batch()
                    if batch is not None:
                        # Success - reset retry counter
                        if connection_retry_count > 0:
                            print(f"[TRAINER rank={self.rank}] Successfully reconnected to orchestrator after {connection_retry_count} retries")
                        connection_retry_count = 0
                        return batch
                    
                    # Batch is None (queue empty) - wait and retry
                    time.sleep(5)
                    continue
                    
                except (requests.exceptions.ConnectionError, 
                        requests.exceptions.ConnectTimeout,
                        ConnectionRefusedError,
                        ConnectionError,
                        OSError) as e:
                    connection_retry_count += 1
                    error_msg = str(e)
                    
                    if connection_retry_count >= max_retries:
                        log_error_with_throttle(
                            f"Max connection retries ({max_retries}) exceeded. Orchestrator appears to be down: {error_msg}",
                            min_interval=0
                        )
                        raise RuntimeError(
                            f"Failed to connect to orchestrator after {max_retries} retries. "
                            f"Orchestrator may be down or unreachable."
                        )
                    
                    # Calculate wait time with exponential backoff
                    if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                        wait_time = min(backoff_factor ** connection_retry_count, max_wait)
                    elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                        wait_time = min(2 * connection_retry_count, max_wait)
                    else:
                        wait_time = min(connection_retry_count * 2, max_wait)
                    
                    log_error_with_throttle(
                        f"Connection error getting batch (retry {connection_retry_count}/{max_retries}, waiting {wait_time}s): {error_msg}"
                    )
                    time.sleep(wait_time)

        def request_global_step():
            try:
                data = orchestrator_service.next_step()
                return int(data.get('step', 0))
            except Exception as e:
                if self.rank == 0:
                    print(f"[TRAINING PROC] failed to fetch global step: {e}")
                return None

        self.device = training_service.get_device()
        _pending_batch_ids: list[str] = []
        progress_bar = None
        if self.rank == 0:
            progress_bar = tqdm(total=None)
            print(f"[TRAINER rank={self.rank}] Starting training loop with accum_steps={self.accum_steps}")

        assert self.num_mix_forward_batches == 1, "mix_forward_batches does not faster"
        
        # Get trainer algorithm
        algo_cls = get_trainer_algorithm(self.trainer_algorithm_name)
        trainer_algo = algo_cls(self, self.trainer_config)
        
        # Services dict for algorithm
        services = {
            'model': self.model_service,
            'training': training_service,
            'orchestrator': orchestrator_service,
        }
        
        last_global_step = 0
        compute_worker_id = f"{socket.gethostname()}-learner-rank{self.rank}"
        trainer_compute_interval = max(5.0, float(self.trainer_compute_report_interval))
        trainer_token_threshold = max(0, int(self.trainer_compute_token_threshold))
        compute_last_report = time.time()
        compute_tokens_acc = 0
        compute_gpu_seconds_acc = 0.0

        def flush_trainer_compute(force: bool = False):
            nonlocal compute_last_report, compute_tokens_acc, compute_gpu_seconds_acc
            if orchestrator_service is None:
                compute_tokens_acc = 0
                compute_gpu_seconds_acc = 0.0
                compute_last_report = time.time()
                return
            if not force:
                if compute_tokens_acc <= 0 and compute_gpu_seconds_acc <= 0:
                    return
                since_last = time.time() - compute_last_report
                if since_last < trainer_compute_interval and compute_tokens_acc < trainer_token_threshold:
                    return
            payload_seconds = compute_gpu_seconds_acc
            if payload_seconds <= 0 and force:
                payload_seconds = max(0.0, time.time() - compute_last_report)
            if payload_seconds <= 0 and compute_tokens_acc <= 0:
                compute_last_report = time.time()
                return
            try:
                orchestrator_service.report_compute_usage(
                    role="learner",
                    gpu_seconds=float(max(payload_seconds, 0.0)),
                    tokens=int(max(compute_tokens_acc, 0)),
                    worker_id=compute_worker_id,
                )
            except Exception:
                pass
            compute_last_report = time.time()
            compute_tokens_acc = 0
            compute_gpu_seconds_acc = 0.0

        while True:
            batch = get_batch_with_waiting()

            if 'end' in batch:
                break
            batch_id = batch.get('_batch_id')
            if batch_id:
                _pending_batch_ids.append(batch_id)
            iter_start_time = time.time()

            # Pull latest weights from orchestrator
            training_service.maybe_pull_weights(orchestrator_service)
            
            # Check if orchestrator wants us to stop
            try:
                stats = orchestrator_service.stats()
                if stats.get('should_stop', False):
                    print(f"[TRAINER] Orchestrator signaled stop (global_step: {stats.get('global_step')})")
                    if hasattr(self, 'Q_data') and self.Q_data is not None:
                        try:
                            for _ in range(get_world_size()):
                                self.Q_data.put({'end': 1})
                            print(f"[TRAINER] Sent stop signal to sampler workers")
                        except Exception as e:
                            print(f"[TRAINER] Failed to signal sampler: {e}")
                    break
            except Exception:
                pass  # Continue training if check fails

            step_id = request_global_step()
            if step_id is None or step_id <= last_global_step:
                step_id = last_global_step + 1
            last_global_step = step_id

            # Use algorithm to compute loss
            model = training_service.get_model()
            loss = trainer_algo.compute_loss(model, batch, services)
            
            # Clear activations after loss computation to free memory before backward
            # This is especially important for long sequences during gradient accumulation
            # Do this BEFORE backward to maximize available memory
            if hasattr(training_service.get_trainer(), '_clear_activations'):
                training_service.get_trainer()._clear_activations()

            # Perform backward pass (this will trigger gradient checkpointing recomputation)
            training_service.backward(loss)
            training_service.increment_micro_step()
            
            # More frequent memory cleanup during gradient accumulation
            # This prevents OOM when accumulating gradients over many micro-steps
            # Changed from 25% to 10% intervals for more aggressive cleanup
            micro_step = training_service.get_worker_micro_step()
            if micro_step % max(1, self.accum_steps // 10) == 0:  # Every 10% of accumulation
                if hasattr(training_service.get_trainer(), '_clear_activations'):
                    training_service.get_trainer()._clear_activations()
                
                # Monitor memory usage and warn if approaching limit
                if self.rank == 0 and torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    memory_free = memory_total - memory_reserved
                    memory_usage_pct = (memory_reserved / memory_total) * 100
                    
                    # Warn if memory usage is high (>85%)
                    if memory_usage_pct > 85:
                        print(f"[TRAINER rank={self.rank}] WARNING: High GPU memory usage: {memory_usage_pct:.1f}% "
                              f"({memory_reserved:.2f}GB/{memory_total:.2f}GB reserved, {memory_free:.2f}GB free)")
            
            # Additional cleanup at critical points (25%, 50%, 75%)
            if micro_step % max(1, self.accum_steps // 4) == 0:  # Every 25% of accumulation
                if hasattr(training_service.get_trainer(), '_clear_activations'):
                    training_service.get_trainer()._clear_activations()
            
            if self.rank == 0:
                if micro_step % max(1, self.accum_steps // 10) == 0 or micro_step == 1:
                    progress_pct = (micro_step / self.accum_steps * 100) if self.accum_steps > 0 else 0
                    print(f"[TRAINER rank={self.rank}] Accumulating gradients: {micro_step}/{self.accum_steps} ({progress_pct:.1f}%)")
            
            # Send heartbeat to orchestrator periodically
            micro_step = training_service.get_worker_micro_step()
            if micro_step % max(1, self.accum_steps // 10) == 0 or micro_step == 1:
                orchestrator_service.send_heartbeat(
                        step_id=step_id,
                    microstep=micro_step,
                        total_microsteps=self.accum_steps,
                    worker_id=training_service.get_lock_owner_id(),
                )
            
            if not training_service.should_accumulate():
                grad_state = training_service.collect_gradients()
                if grad_state:
                    if self.rank == 0:
                        print(f"[TRAINER rank={self.rank}] Collected {len(grad_state)} gradient tensors for step {step_id}")
                    batch_inputs = batch.get('inputs')
                    token_count = int(batch_inputs.numel()) if isinstance(batch_inputs, torch.Tensor) else 0
                    batch_meta = {
                        'microsteps': self.accum_steps,
                        'token_count': token_count,
                        'sample_count': int(batch.get('sample_count', 0)),
                        'loss': float(loss.item()) if hasattr(loss, 'item') else float(loss),
                        '_batch_id': batch.get('_batch_id'),
                        '_batch_ids': list(_pending_batch_ids),
                    }
                    training_service.send_gradients_with_retry(
                        orchestrator_service,
                        step_id=step_id,
                        grad_state=grad_state,
                        batch_meta=batch_meta,
                    )
                    compute_tokens_acc += token_count
                    compute_gpu_seconds_acc += max(time.time() - iter_start_time, 0.0)
                    flush_trainer_compute()
                    training_service.reset_micro_step()
                    _pending_batch_ids.clear()
                    training_service.maybe_pull_weights(orchestrator_service, force=True)
                else:
                    if self.rank == 0:
                        print(f"[TRAINER rank={self.rank}] WARNING: No gradients collected for step {step_id}")

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_description(f"Gradient Step: {step_id}")

        if self.skip_zero_groups:
            print('\n\nSome groups had same rewards and skipped, so the training steps may be less than expected.\n')

        distbarrier()
        flush_trainer_compute(force=True)
