from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml optional
    yaml = None


@dataclass
class SamplerConfig:
    algorithm: str = "treepo"
    params: Dict[str, Any] = field(
        default_factory=lambda: {
            "epochs": 1,
            "rollout_num": 16,
            "train_batch_size": 1,
            "gen_max_tokens": 1024,
            "gen_temperature": 0.8,
            "max_pending_samples": 12800,  # Maximum pending samples before sampler pauses (default: 12800)
            "gen_pending_time": 10.0,  # Wait time in seconds when queue is full (default: 10 seconds)
            "version_poll_interval": 5.0,  # Interval in seconds to check for new weight versions (default: 5 seconds)
            "max_batch_retry": 3,  # Maximum retry count for batch processing failures (default: 3)
            "max_upload_retries": 10,  # Maximum retry count for sample upload failures (default: 10)
            "max_fetch_retries": 10,  # Maximum consecutive failures when fetching problems before graceful shutdown (default: 10)
            "retry_backoff_factor": 2,  # Exponential backoff multiplier for retries (default: 2)
            "retry_max_wait": 60,  # Maximum wait time in seconds between retries (default: 60)
            "log_error_throttle_interval": 60,  # Minimum interval in seconds between logging the same error (default: 60)
            "compute_report_interval": 60.0,  # Seconds between GPU telemetry reports
            "compute_report_token_threshold": 32768,  # Flush telemetry after this many tokens
            "save_samples": True,  # Save generated samples to file (default: True)
            "save_tree_structure": True,  # Save tree structure for TreePO (default: True, only applies to TreePO)
            "save_individual_samples": False,  # Save individual samples even for TreePO (default: False, tree structure is saved instead)
            "treepo_kwargs": {
                "generation_length": 1024,
                "depth": 7,
                "budget_coefficient": 2,
                "sampling_batch_size": 16,
            },
        }
    )


@dataclass
class TrainerConfig:
    algorithm: str = "treepo"
    params: Dict[str, Any] = field(
        default_factory=lambda: {
            "epochs": 1,
            "rollout_num": 16,
            "train_batch_size": 1,
            "accum_steps": 8192,
            "lr": 1e-6,
            "max_batch_retry": 3,  # Maximum retry count for batch processing failures (default: 3)
            "clip_param": 0.2,  # PPO clipping parameter (default: 0.2)
            "pending_retry_timeout": 360.0,  # Timeout in seconds for pending batch retry (default: 360 seconds = 6 minutes)
            "max_get_batch_retries": 10,  # Maximum retry count for get_batch connection errors (default: 10)
            "get_batch_retry_backoff_factor": 2.0,  # Exponential backoff multiplier for get_batch retries (default: 2.0)
            "get_batch_retry_max_wait": 60,  # Maximum wait time in seconds between get_batch retries (default: 60)
            "compute_report_interval": 60.0,  # Seconds between learner GPU telemetry reports
            "compute_report_token_threshold": 32768,  # Flush telemetry after this many tokens
            "treepo_kwargs": {},
        }
    )


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "entropy seesaw"
    run_name: str = "TreePO_experiment"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """
    Declarative benchmark specification for auto-eval.
    """
    name: str = "aime_2024"
    loader: str = "builtin:aime_2024"  # builtin:aime_2024 | builtin:aime_2025 | hf:<dataset_id>
    split: str = "test"
    config: Optional[str] = None  # HF config name if needed (e.g., gpqa diamond)
    max_items: Optional[int] = None  # limit items for faster smoke tests
    num_candidates: int = 1  # for accuracy@K generation
    prompt_template: Optional[str] = None  # builtin:aime_cot | module:function
    answer_extractor: Optional[str] = None  # builtin:boxed | module:function
    metrics: List[str] = field(default_factory=lambda: ["builtin:accuracy@1"])


@dataclass
class EvaluationConfig:
    """
    Evaluation subsystem configuration.
    """
    enabled: bool = False
    schedule: str = "on_version_change"  # on_version_change | manual
    max_parallel_jobs: int = 1
    devices: List[int] = field(default_factory=list)  # evaluator GPU ids; empty -> share sampler GPUs
    wandb_namespace: str = "eval"
    shutdown_timeout_sec: float = 3000.0  # wait for eval completion before shutdown
    benchmarks: List[BenchmarkConfig] = field(default_factory=lambda: [
        BenchmarkConfig(name="aime_2024", loader="builtin:aime_2024", split="test",
                        prompt_template="builtin:aime_cot", answer_extractor="builtin:boxed",
                        metrics=["builtin:accuracy@1", "builtin:accuracy@5"]),
        BenchmarkConfig(name="aime_2025", loader="builtin:aime_2025", split="test",
                        prompt_template="builtin:aime_cot", answer_extractor="builtin:boxed",
                        metrics=["builtin:accuracy@1"]),
    ])


@dataclass
class DatasetConfig:
    """Dataset configuration supporting single or multiple datasets.
    
    Supports two formats:
    1. Single dataset (backward compatible):
       name: "qwedsacf/competition_math"
       split: "train"
       filter_fn: "my_module:my_filter_fn"  # Optional: custom filter function
       shuffle_seed: 42
       max_items: 40  # Optional: limit number of items (for testing)
    
    2. Multiple datasets:
       datasets:
         - name: "qwedsacf/competition_math"
           split: "train"
           filter_fn: "my_module:my_filter_fn"  # Optional
         - name: "gsm8k"
           split: "train"
       shuffle_seed: 42  # Applied to combined dataset
       max_items: 40  # Optional: limit total number of items (for testing)
    
    Note: filter_levels is deprecated. Use filter_fn to provide custom filtering logic.
    """
    # Legacy single dataset format (for backward compatibility)
    name: str = "qwedsacf/competition_math"
    split: str = "train"
    filter_fn: Optional[str] = None  # Custom filter function (e.g., "module.path:function_name")
    shuffle_seed: int = 42
    loader: Optional[str] = None  # Loader name for single dataset (None = auto-detect)
    max_items: Optional[int] = None  # Optional: limit number of items (for testing, None = no limit)
    
    # New multi-dataset format
    datasets: Optional[List[Dict[str, Any]]] = None


@dataclass
class LogControlConfig:
    """Log control configuration for orchestrator output."""
    log_sample_upload: bool = True  # Enable sample upload logs
    log_batch_dispatch: bool = True  # Enable batch dispatch logs
    log_gradient_received: bool = True  # Enable gradient received logs
    log_gradient_reassembled: bool = True  # Enable reassembled gradient logs
    log_gradient_chunks: bool = True  # Enable gradient chunks progress logs
    log_optimizer_step: bool = True  # Enable optimizer step completion logs
    log_processing_gradient: bool = True  # Enable processing reassembled gradient logs
    log_status_report: bool = True  # Enable periodic status report logs
    log_http_access: bool = True  # Enable HTTP access logs (WSGI server logs)


@dataclass
class OrchestratorConfig:
    """Orchestrator server configuration."""
    batch_timeout: float = 3600.0  # Timeout for batch processing in seconds (default: 1 hour)
    queue_size: int = 1600  # Maximum size of training queue
    timeout_check_interval: float = 60.0  # Interval to check for timeout batches in seconds (default: 1 minute)
    keep_last_versions: int = 2  # Number of weight versions to keep
    problem_timeout: float = 600.0  # Timeout for problem processing in seconds (default: 10 minutes)
    chunk_size_mb: int = 50  # Chunk size in MB for gradient uploads (default: 50MB)
    download_chunk_size_mb: int = 32  # Chunk size in MB for weight downloads (default: 32MB)
    status_report_interval: float = 30.0  # Interval in seconds for status report output (default: 30 seconds)
    lock_ttl: float = 30.0  # Lock Time-To-Live in seconds (default: 30 seconds)
    server_threads: int = 10  # Number of threads for handling concurrent requests (default: 10)
    # Gradient chunk management
    chunk_timeout: float = 1200.0  # Timeout in seconds for stale gradient chunks (default: 20 minutes)
    max_concurrent_uploads: int = 200  # Maximum concurrent gradient chunk uploads (default: 200)
    chunk_cleanup_interval: float = 60.0  # Interval in seconds for chunk cleanup (default: 60 seconds)
    # Disk-based gradient storage
    gradient_chunks_dir: Optional[str] = None  # Directory for gradient chunk files (default: auto-generated)
    gradient_storage_dir: Optional[str] = None  # Directory for restored gradient files (default: auto-generated)
    max_gradient_disk_mb: float = 1024000.0  # Maximum disk usage in MB for gradient files (default: 1TB)
    max_gradient_file_size_mb: float = 100000.0  # Maximum size of a single gradient file in MB (default: 100GB) - prevents OOM
    # Logging
    log_dir: Optional[str] = None  # Directory for log files (default: logs/{run_id})
    # HTTP request timeouts (in seconds)
    get_batch_timeout: float = 60.0  # Timeout for trainer get_batch requests (default: 60 seconds)
    send_gradients_timeout: float = 300.0  # Timeout for gradient upload requests (default: 5 minutes)
    download_weights_timeout: float = 600.0  # Timeout for weight download requests (default: 10 minutes)
    upload_samples_timeout: float = 300.0  # Timeout for sample upload requests (default: 5 minutes)
    fetch_problem_timeout: float = 10.0  # Timeout for problem fetch requests (default: 10 seconds)
    register_timeout: float = 10.0  # Timeout for trainer registration (default: 10 seconds)
    stats_timeout: float = 5.0  # Timeout for stats requests (default: 5 seconds)
    heartbeat_timeout: float = 2.0  # Timeout for heartbeat requests (default: 2 seconds)
    version_check_timeout: float = 5.0  # Timeout for version check requests (default: 5 seconds)
    next_step_timeout: float = 5.0  # Timeout for next_step requests (default: 5 seconds)
    lock_timeout: float = 5.0  # Timeout for lock acquire/release requests (default: 5 seconds)
    log_control: LogControlConfig = field(default_factory=LogControlConfig)  # Log control configuration


@dataclass
class FunctionConfig:
    """Configuration for user-defined functions.
    
    Supports module path format (e.g., "my_module:function_name") or callable objects.
    Functions are loaded dynamically at runtime.
    """
    # Reward functions (list of module paths or callables)
    reward_fns: Optional[List[str]] = None  # e.g., ["my_module:correct_fn", "my_module:format_fn"]
    # Answer extraction function (for extracting boxed answers)
    extract_boxed_answer_fn: Optional[str] = None  # e.g., "my_module:extract_boxed_answer"
    # Format function (for format checking)
    format_fn: Optional[str] = None  # e.g., "my_module:format_fn"
    # System prompt
    system_prompt: Optional[str] = None  # e.g., "Please reason step by step..."
    # Prompt function (for generating prompts)
    prompt_fn: Optional[str] = None  # e.g., "my_module:make_prompt_fn"


@dataclass
class ExperimentConfig:
    model_path: str = "Qwen/Qwen2.5-7B"
    orchestrator_port: int = 59888
    update_steps: int = 512 * 16
    lr: float = 1e-6
    epochs: int = 10
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    functions: FunctionConfig = field(default_factory=FunctionConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        def build(sub_cls, key):
            value = data.get(key, {})
            # Handle None case (when YAML section exists but is empty/commented)
            if value is None:
                value = {}
            return sub_cls(**value)
        
        def build_dataset_config(dataset_data: Dict[str, Any]) -> DatasetConfig:
            """Build DatasetConfig supporting both single and multi-dataset formats."""
            # Check if it's the new multi-dataset format
            if "datasets" in dataset_data and isinstance(dataset_data["datasets"], list):
                return DatasetConfig(
                    datasets=dataset_data["datasets"],
                    shuffle_seed=dataset_data.get("shuffle_seed", 42),
                    max_items=dataset_data.get("max_items", None)
                )
            else:
                # Legacy single dataset format
                return DatasetConfig(**dataset_data)

        # Handle nested log_control in orchestrator config
        orch_data = data.get("orchestrator", {})
        log_control_data = orch_data.pop("log_control", {})
        if log_control_data:
            orch_data["log_control"] = LogControlConfig(**log_control_data)
        
        return cls(
            model_path=data.get("model_path", cls.model_path),
            orchestrator_port=data.get("orchestrator_port", cls.orchestrator_port),
            update_steps=data.get("update_steps", cls.update_steps),
            lr=data.get("lr", cls.lr),
            epochs=data.get("epochs", cls.epochs),
            sampler=build(SamplerConfig, "sampler"),
            trainer=build(TrainerConfig, "trainer"),
            wandb=build(WandbConfig, "wandb"),
            dataset=build_dataset_config(data.get("dataset", {})),
            orchestrator=OrchestratorConfig(**orch_data) if orch_data else build(OrchestratorConfig, "orchestrator"),
            evaluation=build(EvaluationConfig, "evaluation"),
            functions=build(FunctionConfig, "functions"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load YAML configs. Install with `pip install pyyaml`."
        )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_experiment_config(path: Optional[str] = None) -> ExperimentConfig:
    if not path:
        return ExperimentConfig()
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        data = _load_yaml(cfg_path)
    else:
        with cfg_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    return ExperimentConfig.from_dict(data or {})


def apply_env_overrides(config: ExperimentConfig, env: Dict[str, str]) -> None:
    project = env.get("WANDB_PROJECT")
    if project:
        config.wandb.project = project
    run_name = env.get("WANDB_RUN_NAME")
    if run_name:
        config.wandb.run_name = run_name
    if env.get("WANDB_DISABLED") == "true":
        config.wandb.enabled = False
    sampler_algo = env.get("SAMPLER_ALGO")
    if sampler_algo:
        config.sampler.algorithm = sampler_algo
    trainer_algo = env.get("TRAINER_ALGO")
    if trainer_algo:
        config.trainer.algorithm = trainer_algo


def apply_cli_overrides(config: ExperimentConfig, overrides: Dict[str, Any]) -> None:
    if "orchestrator_port" in overrides and overrides["orchestrator_port"]:
        config.orchestrator_port = overrides["orchestrator_port"]
    wandb_run = overrides.get("wandb_run_name")
    if wandb_run:
        config.wandb.run_name = wandb_run
    wandb_project = overrides.get("wandb_project")
    if wandb_project:
        config.wandb.project = wandb_project
    sampler_algo = overrides.get("sampler_algo")
    if sampler_algo:
        config.sampler.algorithm = sampler_algo
    trainer_algo = overrides.get("trainer_algo")
    if trainer_algo:
        config.trainer.algorithm = trainer_algo

