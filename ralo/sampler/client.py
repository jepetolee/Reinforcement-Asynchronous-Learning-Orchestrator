import time
from typing import Any, Dict

import requests

from ralo.utils import json_to_bytes_list


class SamplerClient:
    """HTTP helper for sampler workers."""

    def __init__(self, orchestrator_url: str, retry_interval: float = 1.0):
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.session = requests.Session()
        self.retry_interval = retry_interval

    def fetch_problem(self, timeout: float = 10.0) -> Dict[str, Any]:
        resp = self.session.get(f"{self.orchestrator_url}/problem/get", timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def upload_samples(self, payload: Dict[str, Any], timeout: float = 300.0) -> Dict[str, Any]:
        resp = self.session.post(
            f"{self.orchestrator_url}/upload",
            data=json_to_bytes_list(payload),
            timeout=timeout,
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"ok": True}

    def wait_for_problem(self) -> Dict[str, Any]:
        while True:
            data = self.fetch_problem()
            if data.get("ok") or data.get("end"):
                return data
            if data.get("empty"):
                time.sleep(self.retry_interval)
                continue
            return data

    # === Evaluation APIs ===
    def get_eval_job(self, timeout: float = 5.0) -> Dict[str, Any]:
        resp = self.session.get(f"{self.orchestrator_url}/eval/job/get", timeout=timeout)
        if resp.status_code == 404:
            return {"empty": True}
        resp.raise_for_status()
        return resp.json()

    def claim_eval_job(self, job_id: str, owner: str, ttl: float = 30.0, timeout: float = 5.0) -> Dict[str, Any]:
        resp = self.session.post(
            f"{self.orchestrator_url}/eval/job/claim",
            json={"job_id": job_id, "owner": owner, "ttl": ttl},
            timeout=timeout,
        )
        if resp.status_code == 404:
            return {"ok": False, "error": "not found"}
        resp.raise_for_status()
        return resp.json()

    def report_eval_job(self, job_id: str, metrics: Dict[str, Any], samples=None, duration_sec: float | None = None, timeout: float = 30.0) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"job_id": job_id, "metrics": metrics}
        if samples is not None:
            payload["samples"] = samples
        if duration_sec is not None:
            payload["duration_sec"] = duration_sec
        resp = self.session.post(
            f"{self.orchestrator_url}/eval/job/report",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code == 404:
            return {"ok": False, "error": "not found"}
        resp.raise_for_status()
        return resp.json()

    def eval_stats(self, timeout: float = 5.0) -> Dict[str, Any]:
        resp = self.session.get(f"{self.orchestrator_url}/eval/stats", timeout=timeout)
        if resp.status_code == 404:
            return {"ok": False}
        resp.raise_for_status()
        return resp.json()
