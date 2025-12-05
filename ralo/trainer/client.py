import io
from typing import Any, Dict

import requests

from ralo.utils import bytes_list_to_json, json_to_bytes_list


class TrainerClient:
    """HTTP helper for trainer processes."""

    def __init__(self, orchestrator_url: str):
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.session = requests.Session()

    def register(self, payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
        resp = self.session.post(
            f"{self.orchestrator_url}/trainer/register", json=payload, timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()

    def stats(self, timeout: float = 5.0) -> Dict[str, Any]:
        resp = self.session.get(f"{self.orchestrator_url}/stats", timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def get_batch(self, timeout: float = 60.0) -> Dict[str, Any] | None:
        """
        Get a batch from orchestrator. Increased timeout to handle orchestrator busy periods.
        
        Returns:
            Dict with batch data if successful, None if queue is empty.
            
        Raises:
            ConnectionError, requests.exceptions.ConnectionError: When orchestrator is unreachable.
                These exceptions should be caught by caller (get_batch_with_waiting) for retry logic.
            Other exceptions: Propagated to caller for handling.
        """
        resp = self.session.get(f"{self.orchestrator_url}/get", timeout=timeout)
        if resp.status_code != 200 or resp.content == b"empty":
            return None
        return bytes_list_to_json(resp.content)

    def send_gradients(self, payload: Dict[str, Any], timeout: float = 300.0, chunk_size_mb: int = 50) -> Dict[str, Any]:
        """Send gradients using chunked upload (disk-based storage only)."""
        data = json_to_bytes_list(payload)
        size_mb = len(data) / (1024 * 1024)
        
        # Always use chunked upload (disk-based storage)
        chunk_size = chunk_size_mb * 1024 * 1024
        total_chunks = (len(data) + chunk_size - 1) // chunk_size
        import time
        upload_id = f"{payload.get('worker_id', 'unknown')}_{payload.get('step_id', 0)}_{int(time.time() * 1000000)}"
        
        # Log chunked upload start
        worker_id = payload.get('worker_id', 'unknown')
        step_id = payload.get('step_id', 0)
        print(f"[TRAINER] Uploading gradient in {total_chunks} chunks (~{size_mb:.1f}MB total, {chunk_size_mb}MB/chunk)")
        
        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_size
            end = min(len(data), start + chunk_size)
            chunk_data = data[start:end]
            
            resp = self.session.post(
                f"{self.orchestrator_url}/gradient/upload_chunk",
                data=chunk_data,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Upload-ID": upload_id,
                    "X-Chunk-Index": str(chunk_idx),
                    "X-Total-Chunks": str(total_chunks),
                },
                timeout=timeout,
            )
            resp.raise_for_status()
        
        # Finalize upload
        print(f"[TRAINER] Finalizing gradient upload (all {total_chunks} chunks sent)")
        resp = self.session.post(
            f"{self.orchestrator_url}/gradient/upload_finalize",
            json={"upload_id": upload_id},
            timeout=timeout,
        )
        resp.raise_for_status()
        try:
            result = resp.json()
            print(f"[TRAINER] Gradient upload completed successfully")
            return result
        except Exception:
            print(f"[TRAINER] Gradient upload completed (response parsing skipped)")
            return {"ok": True}

    def latest_version(self, model_id: str = "default", timeout: float = 5.0) -> int:
        resp = self.session.get(
            f"{self.orchestrator_url}/weights/version",
            params={"model_id": model_id},
            timeout=timeout,
        )
        resp.raise_for_status()
        return int(resp.json().get("latest_version", -1))

    def download_weights(self, model_id: str, version: int, timeout: float = 600.0, chunk_size_mb: int = 32) -> bytes:
        resp = self.session.get(
            f"{self.orchestrator_url}/weights/download",
            params={"model_id": model_id, "version": int(version)},
            timeout=timeout,
            stream=True,
        )
        resp.raise_for_status()
        buf = io.BytesIO()
        chunk_size = chunk_size_mb * 1024 * 1024
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                buf.write(chunk)
        return buf.getvalue()

    def upload_weights(self, model_id: str, version: int, data: bytes, timeout: float = 120.0) -> bool:
        resp = self.session.post(
            f"{self.orchestrator_url}/weights/upload",
            params={"model_id": model_id, "version": int(version)},
            data=data,
            timeout=timeout,
        )
        return resp.ok

    def next_step(self, timeout: float = 5.0) -> Dict[str, Any]:
        resp = self.session.post(f"{self.orchestrator_url}/step/next", timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def acquire_lock(self, key: str, owner: str, ttl: float = 30.0, timeout: float = 5.0) -> Dict[str, Any]:
        payload = {"key": key, "owner": owner, "ttl": ttl}
        resp = self.session.post(f"{self.orchestrator_url}/lock/acquire", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def release_lock(self, key: str, owner: str, timeout: float = 5.0) -> Dict[str, Any]:
        payload = {"key": key, "owner": owner}
        resp = self.session.post(f"{self.orchestrator_url}/lock/release", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def send_heartbeat(self, worker_id: str, step_id: int, microstep: int, total_microsteps: int, timeout: float = 2.0) -> bool:
        """Send heartbeat to orchestrator with accumulation progress."""
        try:
            payload = {
                "worker_id": worker_id,
                "step_id": step_id,
                "microstep": microstep,
                "total_microsteps": total_microsteps,
            }
            resp = self.session.post(
                f"{self.orchestrator_url}/trainer/heartbeat",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            return True
        except Exception:
            # Silently fail - heartbeat is not critical
            return False

    def report_compute_usage(self, payload: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
        resp = self.session.post(
            f"{self.orchestrator_url}/compute/report",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
