from __future__ import annotations

from typing import Any, Dict, Optional

from ralo.config import WandbConfig


class ExperimentLogger:
    def init(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize logging session."""

    def log(self, metrics: Dict[str, Any]) -> None:
        """Log aggregated metrics."""

    def close(self) -> None:
        """Tear down logging session."""


class NoOpLogger(ExperimentLogger):
    pass


class WandbLogger(ExperimentLogger):
    def __init__(self, cfg: WandbConfig):
        import wandb  # type: ignore

        self._wandb = wandb
        self._cfg = cfg
        self._run = None

    def init(self, context: Optional[Dict[str, Any]] = None) -> None:
        config = dict(context or {})
        config.update(self._cfg.extras)
        # settings를 사용하여 터미널 감지 문제 해결 (로그 파일로 리다이렉트 시 발생하는 문제)
        try:
            self._run = self._wandb.init(
                project=self._cfg.project,
                name=self._cfg.run_name,
                entity=self._cfg.entity,
                tags=self._cfg.tags,
                config=config,
                settings=self._wandb.Settings(
                    console="off",  # 콘솔 출력 비활성화 (터미널 감지 문제 해결)
                    _disable_stats=True,  # 통계 비활성화
                ),
            )
        except Exception as e:
            # settings가 실패하면 기본 설정으로 재시도
            try:
                self._run = self._wandb.init(
                    project=self._cfg.project,
                    name=self._cfg.run_name,
                    entity=self._cfg.entity,
                    tags=self._cfg.tags,
                    config=config,
                )
            except Exception:
                # wandb 초기화 실패 시 예외를 다시 발생시켜서 NoOpLogger로 fallback
                raise e

    def log(self, metrics: Dict[str, Any]) -> None:
        if self._run is None:
            return
        self._wandb.log(metrics)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def build_logger(cfg: WandbConfig) -> ExperimentLogger:
    if not cfg.enabled:
        return NoOpLogger()
    try:
        return WandbLogger(cfg)
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"[LOGGER] Failed to initialize wandb logger ({exc}), using NoOpLogger.")
        return NoOpLogger()

