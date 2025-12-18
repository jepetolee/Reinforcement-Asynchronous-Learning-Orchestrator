import argparse
import os
import random
import re
import socket
import sys
import time
import string
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "32"

from ralo import OrchestratorServer, RALO
from ralo.config import apply_cli_overrides, apply_env_overrides, load_experiment_config
from ralo.logging import build_logger
from ralo.logging_utils import setup_logging
from ralo.function_loaders import load_function, load_reward_functions
from ralo.rewards import (
    correct_fn,
    format_fn,
    iiv_reward_fn,
    extract_boxed_answer,
    default_reward_fns,
)


def make_prompt_fn(self, item):
    # Default system prompt for backward compatibility
    default_system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    return self.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": default_system_prompt},
            {"role": "user", "content": item["Q"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def make_iiv_prompt_fn(self, item):
    """IIV (Is It Valid) prompt function.
    
    IIV is a task where the model must determine if the given information (hint) is valid
    enough to answer the question. If hint is missing/empty (not valid):
        - Instructs model to output "I don't know"
    If hint is present and valid:
        - Includes hint in the prompt and asks for the answer
    """
    hint = item.get("Hint", "")
    if hint is None or str(hint).strip() == "":
        # No hint: instruct to say "I don't know"
        system_prompt = (
            "You are given a question, but you do not have enough information to answer it. "
            "If you cannot answer the question with certainty, you must respond with \"I don't know\" "
            "instead of guessing."
        )
        user_content = item["Q"]
    else:
        # Has hint: include hint and ask for answer
        system_prompt = (
            "You are given a question and a hint. Use the hint to answer the question accurately. "
            "Please reason step by step and provide your final answer."
        )
        user_content = f"Question: {item['Q']}\n\nHint: {hint}"
    
    return self.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_run_id() -> str:
    """Generate a unique run ID: timestamp + 6 random characters."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{timestamp}_{random_suffix}"


def parse_args():
    parser = argparse.ArgumentParser(description="RALO runner")
    parser.add_argument("command", choices=["orch", "gen", "train"], nargs="?", default="train")
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    parser.add_argument(
        "--orchestrator",
        type=str,
        help="Override orchestrator URL for trainer/sampler commands",
    )
    parser.add_argument("--orch-port", type=int, help="Override orchestrator port when running `orch`")
    parser.add_argument("--wandb-project", type=str, help="Override wandb project")
    parser.add_argument("--wandb-run-name", type=str, help="Override wandb run name")
    parser.add_argument("--sampler-algo", type=str, help="Override sampler algorithm")
    parser.add_argument("--trainer-algo", type=str, help="Override trainer algorithm")
    parser.add_argument("--log-dir", type=str, help="Directory for log files (default: logs/{run_id})")
    parser.add_argument("--log-file", type=str, help="Specific log file name (overrides default process_name.log)")
    parser.add_argument("--run-id", type=str, help="Run ID for this execution (default: auto-generated)")
    return parser.parse_args()


def load_dataset_for_orch(dataset_cfg):
    """Load dataset(s) for orchestrator using the new dataset loader system.
    
    Supports both single dataset (legacy) and multiple datasets (new format).
    """
    from ralo.dataset_loaders import load_datasets
    
    # Convert DatasetConfig to list of dataset config dicts
    if dataset_cfg.datasets is not None:
        # New multi-dataset format
        dataset_configs = []
        for ds_cfg in dataset_cfg.datasets:
            # Handle both dict and SingleDatasetConfig objects
            if isinstance(ds_cfg, dict):
                config_dict = {
                    "name": ds_cfg.get("name"),
                    "split": ds_cfg.get("split", "train"),
                    "loader": ds_cfg.get("loader", "auto"),
                    "filter_fn": ds_cfg.get("filter_fn"),  # User-provided filter function
                    "shuffle_seed": ds_cfg.get("shuffle_seed"),  # Individual shuffle, optional
                }
            else:
                # SingleDatasetConfig object
                from dataclasses import asdict
                config_dict = asdict(ds_cfg)
            dataset_configs.append(config_dict)
        
        shuffle_seed = dataset_cfg.shuffle_seed  # Combined shuffle seed
        max_items = dataset_cfg.max_items  # Get max_items from config
        QAs = load_datasets(dataset_configs, shuffle_seed=shuffle_seed, max_items=max_items)
    else:
        # Legacy single dataset format (backward compatible)
        config_dict = {
            "name": dataset_cfg.name,
            "split": dataset_cfg.split,
            "loader": dataset_cfg.loader or "auto",
            "filter_fn": dataset_cfg.filter_fn,  # User-provided filter function
            "shuffle_seed": dataset_cfg.shuffle_seed,
        }
        # Include extra fields (loader-specific configuration like hint_field, no_hint_ratio, etc.)
        if hasattr(dataset_cfg, '_extra_fields') and dataset_cfg._extra_fields:
            config_dict.update(dataset_cfg._extra_fields)
        max_items = dataset_cfg.max_items  # Get max_items from config
        QAs = load_datasets([config_dict], shuffle_seed=None, max_items=max_items)  # shuffle handled in loader
    
    return QAs


def build_sampler_instance(config, orch_url):
    sampler_cfg = config.sampler
    params = sampler_cfg.params
    gen_devices = params.get("gen_devices")
    if gen_devices is None:
        gen_devices_env = os.environ.get("GEN_DEVICES")
        if gen_devices_env:
            gen_devices = [int(x) for x in gen_devices_env.split(",") if x.strip()]
        else:
            gen_devices = [0]
    return RALO(
        model_path=config.model_path,
        epochs=params.get("epochs", config.epochs),
        train_data=None,
        rollout_num=params.get("rollout_num", 16),
        train_batch_size=params.get("train_batch_size", 1),
        gen_device=gen_devices,
        gen_max_tokens=params.get("gen_max_tokens", 1024),
        gen_temperature=params.get("gen_temperature", 0.8),
        gen_top_p=params.get("gen_top_p"),
        gen_top_k=params.get("gen_top_k"),
        gen_min_p=params.get("gen_min_p"),
        enable_thinking=params.get("enable_thinking", False),
        genlog_filename=params.get("genlog_filename"),
        max_pending_samples=params.get("max_pending_samples", 12800),
        gen_pending_time=params.get("gen_pending_time", 10),
        skip_zero_groups=params.get("skip_zero_groups", False),
        vllm_kwargs=params.get("vllm_kwargs"),
        TreePO_kwargs=params.get("treepo_kwargs"),
        orchestrator_url=orch_url,
        init_trainer=False,
        max_batch_retry=params.get("max_batch_retry", 3),
        sampler_config=sampler_cfg,
        trainer_config=config.trainer,
        orchestrator_config=config.orchestrator,
    )


def build_trainer_instance(config, orch_url):
    trainer_cfg = config.trainer
    params = trainer_cfg.params
    trainer_kwargs = {
        "lr": params.get("lr", config.lr),
        "accum_steps": params.get("accum_steps", config.update_steps),
    }
    return RALO(
        model_path=config.model_path,
        epochs=params.get("epochs", config.epochs),
        train_data=None,
        rollout_num=params.get("rollout_num", 16),
        train_batch_size=params.get("train_batch_size", 1),
        gen_max_tokens=params.get("gen_max_tokens", 1024),
        gen_temperature=params.get("gen_temperature", 0.8),
        gen_device=params.get("gen_device", []),
        orchestrator_url=orch_url,
        TreePO_kwargs=params.get("treepo_kwargs"),
        clip_param=params.get("clip_param", 0.2),
        max_batch_retry=params.get("max_batch_retry", 3),
        pending_retry_timeout=params.get("pending_retry_timeout", 360),
        sampler_config=config.sampler,
        trainer_config=trainer_cfg,
        orchestrator_config=config.orchestrator,
        **trainer_kwargs,
    )


def load_user_functions(config):
    """Load user-defined functions from config or use defaults.
    
    Args:
        config: ExperimentConfig instance
        
    Returns:
        Dictionary with loaded functions: reward_fns, extract_boxed_answer_fn, 
        format_fn, system_prompt, prompt_fn
    """
    func_cfg = config.functions
    
    # Load reward functions
    if func_cfg.reward_fns:
        loaded_reward_fns = load_reward_functions(func_cfg.reward_fns)
    else:
        # Default: use correct_fn
        loaded_reward_fns = [correct_fn]
    
    # Load extract_boxed_answer function
    if func_cfg.extract_boxed_answer_fn:
        extract_boxed_answer_fn = load_function(func_cfg.extract_boxed_answer_fn)
    else:
        # Default: use extract_boxed_answer from rewards module (already imported)
        extract_boxed_answer_fn = extract_boxed_answer
    
    # Load format function
    if func_cfg.format_fn:
        format_fn_loaded = load_function(func_cfg.format_fn)
    else:
        # Default: use format_fn from rewards module (already imported)
        format_fn_loaded = format_fn
    
    # Get system prompt (use default if not specified in config)
    default_system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    system_prompt_loaded = func_cfg.system_prompt or default_system_prompt
    
    # Load prompt function
    if func_cfg.prompt_fn:
        prompt_fn_loaded = load_function(func_cfg.prompt_fn)
    else:
        # Default: use local function with loaded system prompt
        def make_prompt_fn_with_system(self, item):
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt_loaded},
                    {"role": "user", "content": item["Q"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_fn_loaded = make_prompt_fn_with_system
    
    return {
        "reward_fns": loaded_reward_fns,
        "extract_boxed_answer_fn": extract_boxed_answer_fn,
        "format_fn": format_fn_loaded,
        "system_prompt": system_prompt_loaded,
        "prompt_fn": prompt_fn_loaded,
    }


def attach_common_hooks(ralo_instance, user_functions=None):
    """Attach common hooks to RALO instance.
    
    Args:
        ralo_instance: RALO instance
        user_functions: Dictionary of loaded user functions (from load_user_functions)
                       If None, uses default functions
    """
    if user_functions is None:
        # Fallback to default functions
        user_functions = {
            "reward_fns": default_reward_fns,
            "prompt_fn": make_prompt_fn,
        }
    
    # Add reward functions
    for reward_fn in user_functions.get("reward_fns", []):
        ralo_instance.add_reward(reward_fn)
    
    # Set system_prompt on RALO instance for worker processes
    if "system_prompt" in user_functions:
        ralo_instance.system_prompt = user_functions["system_prompt"]
    
    # Set prompt functions
    prompt_fn = user_functions.get("prompt_fn", make_prompt_fn)
    ralo_instance.set_policy_prompt_fn(prompt_fn)
    ralo_instance.set_rollout_prompt_fn(prompt_fn)


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    apply_env_overrides(config, os.environ)
    apply_cli_overrides(
        config,
        {
            "orchestrator_port": args.orch_port,
            "wandb_run_name": args.wandb_run_name,
            "wandb_project": args.wandb_project,
            "sampler_algo": args.sampler_algo,
            "trainer_algo": args.trainer_algo,
        },
    )

    # Determine run_id and log directory
    run_id = args.run_id or os.environ.get("RUN_ID") or generate_run_id()
    if args.log_dir:
        log_dir = Path(args.log_dir)
    elif config.orchestrator.log_dir:
        log_dir = Path(config.orchestrator.log_dir)
    else:
        log_dir = Path("logs") / run_id
    
    # Determine process name and log file name based on command
    process_name = {
        "orch": "orchestrator",
        "gen": "sampler",
        "train": "trainer"
    }.get(args.command, "process")
    
    if args.log_file:
        # Use explicitly specified log file name
        log_file = log_dir / args.log_file
    else:
        # Add node identifier to log filename if in SLURM environment
        slurm_nodeid = os.environ.get("SLURM_NODEID")
        slurm_jobid = os.environ.get("SLURM_JOB_ID")
        
        if slurm_nodeid is not None:
            # Include node ID in filename for easy identification
            log_file = log_dir / f"{process_name}_node{slurm_nodeid}.log"
        elif slurm_jobid:
            # Fallback to job ID if node ID not available
            log_file = log_dir / f"{process_name}_job{slurm_jobid}.log"
        else:
            log_file = log_dir / f"{process_name}.log"
    
    # Set up logging to file (while also printing to console)
    stdout_tee, stderr_tee = setup_logging(str(log_file), also_log_stderr=True)
    
    # Save run_id and PID to files
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "run_id.txt", "w") as f:
        f.write(run_id)
    with open(log_dir / f"{process_name}.pid", "w") as f:
        f.write(str(os.getpid()))
    
    # Get hostname (needed for both log filename and SLURM info)
    hostname = socket.gethostname()
    
    # Log SLURM environment information if available
    slurm_info = {}
    slurm_vars = [
        "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_NODELIST",
        "SLURM_NODEID", "SLURM_STEP_ID", "SLURM_PROCID", "SLURM_LOCALID",
        "SLURM_CPUS_ON_NODE", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"
    ]
    for var in slurm_vars:
        value = os.environ.get(var)
        if value:
            slurm_info[var] = value
    
    print(f"[CLI] Run ID: {run_id}")
    print(f"[CLI] Log file: {log_file}")
    print(f"[CLI] PID: {os.getpid()}")
    print(f"[CLI] Hostname: {hostname}")
    
    if slurm_info:
        print(f"[CLI] SLURM Job ID: {slurm_info.get('SLURM_JOB_ID', 'N/A')}")
        print(f"[CLI] SLURM Job Name: {slurm_info.get('SLURM_JOB_NAME', 'N/A')}")
        print(f"[CLI] SLURM Node List: {slurm_info.get('SLURM_NODELIST', slurm_info.get('SLURM_JOB_NODELIST', 'N/A'))}")
        print(f"[CLI] SLURM Node ID: {slurm_info.get('SLURM_NODEID', 'N/A')}")
        if 'SLURM_GPUS_ON_NODE' in slurm_info:
            print(f"[CLI] SLURM GPUs on Node: {slurm_info['SLURM_GPUS_ON_NODE']}")
        if 'SLURM_GPUS_PER_NODE' in slurm_info:
            print(f"[CLI] SLURM GPUs per Node: {slurm_info['SLURM_GPUS_PER_NODE']}")
    else:
        print(f"[CLI] Not running in SLURM environment (or SLURM vars not set)")
    
    # Save SLURM info to a separate file for easy reference
    if slurm_info:
        slurm_info_file = log_dir / f"{process_name}_slurm_info.txt"
        with open(slurm_info_file, "w") as f:
            f.write(f"Hostname: {hostname}\n")
            f.write(f"Run ID: {run_id}\n")
            f.write(f"PID: {os.getpid()}\n")
            f.write(f"Process: {process_name}\n")
            f.write("\nSLURM Environment Variables:\n")
            for key, value in sorted(slurm_info.items()):
                f.write(f"{key}={value}\n")

    if args.command == "orch":
        dataset = load_dataset_for_orch(config.dataset)
        logger = build_logger(config.wandb)
        orch_cfg = config.orchestrator
        log_ctrl = orch_cfg.log_control
        server = OrchestratorServer(
            config.model_path,
            port=args.orch_port or config.orchestrator_port,
            update_steps=config.update_steps,
            lr=config.lr,
            train_data=dataset,
            epochs=config.epochs,
            queue_size=orch_cfg.queue_size,
            batch_timeout=orch_cfg.batch_timeout,
            timeout_check_interval=orch_cfg.timeout_check_interval,
            keep_last_versions=orch_cfg.keep_last_versions,
            problem_timeout=orch_cfg.problem_timeout,
            status_report_interval=orch_cfg.status_report_interval,
            lock_ttl=orch_cfg.lock_ttl,
            server_threads=orch_cfg.server_threads,
            evaluation=config.evaluation,
            chunk_timeout=orch_cfg.chunk_timeout,
            max_concurrent_uploads=orch_cfg.max_concurrent_uploads,
            chunk_cleanup_interval=orch_cfg.chunk_cleanup_interval,
            gradient_chunks_dir=orch_cfg.gradient_chunks_dir,
            gradient_storage_dir=orch_cfg.gradient_storage_dir,
            max_gradient_disk_mb=orch_cfg.max_gradient_disk_mb,
            max_gradient_file_size_mb=orch_cfg.max_gradient_file_size_mb,
            logger=logger,
            log_sample_upload=log_ctrl.log_sample_upload,
            log_batch_dispatch=log_ctrl.log_batch_dispatch,
            log_gradient_received=log_ctrl.log_gradient_received,
            log_gradient_reassembled=log_ctrl.log_gradient_reassembled,
            log_gradient_chunks=log_ctrl.log_gradient_chunks,
            log_optimizer_step=log_ctrl.log_optimizer_step,
            log_processing_gradient=log_ctrl.log_processing_gradient,
            log_status_report=log_ctrl.log_status_report,
            log_http_access=log_ctrl.log_http_access,
        )
        server.run_server()  # Register all routes
        server.start()  # Start multi-threaded server
        return

    orch_url = (
        args.orchestrator
        or os.environ.get("ORCH_SERVER")
        or "http://127.0.0.1:59888"
    )
    
    # Load user-defined functions from config
    user_functions = load_user_functions(config)
    
    if args.command == "gen":
        ralo_instance = build_sampler_instance(config, orch_url)
        attach_common_hooks(ralo_instance, user_functions)
        ralo_instance.run_generation_only()
        return

    print(f"[CLI] Trainer will stream gradients to orchestrator at {orch_url}")
    ralo_instance = build_trainer_instance(config, orch_url)
    attach_common_hooks(ralo_instance, user_functions)
    ralo_instance.train()


if __name__ == "__main__":
    main()
