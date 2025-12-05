from ralo.config import ExperimentConfig, apply_env_overrides, load_experiment_config


def test_default_config_loads():
    cfg = load_experiment_config()
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.sampler.algorithm == "treepo"
    assert cfg.trainer.algorithm == "treepo"


def test_env_override_changes_wandb(monkeypatch):
    cfg = load_experiment_config()
    env = {"WANDB_PROJECT": "demo", "WANDB_RUN_NAME": "run", "WANDB_DISABLED": "true"}
    apply_env_overrides(cfg, env)
    assert cfg.wandb.project == "demo"
    assert cfg.wandb.run_name == "run"
    assert cfg.wandb.enabled is False
