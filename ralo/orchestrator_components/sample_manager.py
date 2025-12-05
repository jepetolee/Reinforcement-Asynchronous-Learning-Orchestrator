import queue
import time
from typing import Any, Dict, Optional


class SampleQueueManager:
    """Wraps the orchestrator's training queue and tracks in-flight batches."""

    def __init__(self, maxsize: int = 1600, batch_timeout: float = 600.0):
        self.train_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=maxsize)
        self.processing_batches: Dict[str, Dict[str, Any]] = {}
        self.total_enqueued = 0
        self.total_dequeued = 0
        self._batch_id_counter = 0
        self.batch_timeout = batch_timeout  # Timeout in seconds (default 5 minutes)
        self.stage_counts = {
            "generated_total": 0,
            "completed_total": 0,
        }

    def enqueue(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.train_queue.put(payload, block=False)
            self.total_enqueued += 1
            self.stage_counts["generated_total"] += 1
            remain = self.train_queue.qsize()
            return {"ok": True, "queued": True, "remain_cnt": remain}
        except queue.Full:
            remain = self.train_queue.qsize()
            return {"ok": True, "queued": False, "remain_cnt": remain}

    def begin_batch(self) -> Optional[Dict[str, Any]]:
        try:
            item = self.train_queue.get(block=False)
        except queue.Empty:
            return None

        batch_id = f"batch_{self._batch_id_counter}"
        self._batch_id_counter += 1
        if isinstance(item, dict):
            item = dict(item)
            item["_batch_id"] = batch_id

        self.processing_batches[batch_id] = {
            "payload": item,
            "status": "awaiting_gradient",
            "timestamp": time.time(),
        }

        return {
            "batch": item,
            "batch_id": batch_id,
            "queue_size": self.train_queue.qsize(),
            "processing": len(self.processing_batches),
        }

    def mark_processed(self, batch_id: Optional[str]) -> None:
        if not batch_id:
            return
        if batch_id in self.processing_batches:
            del self.processing_batches[batch_id]
            self.total_dequeued += 1
            self.stage_counts["completed_total"] += 1

    def stage_snapshot(self) -> Dict[str, Any]:
        return {
            "generated_total": int(self.stage_counts["generated_total"]),
            "awaiting_gradient": len(self.processing_batches),
            "completed_total": int(self.stage_counts["completed_total"]),
        }

    def requeue_timeout_batches(self) -> tuple[int, list[str]]:
        """Requeue batches that have been processing for too long (trainer died).
        Returns (number of batches requeued, list of timeout batch IDs)."""
        now = time.time()
        requeued = 0
        timeout_batch_ids = []
        timeout_batches = []
        
        for batch_id, batch_info in list(self.processing_batches.items()):
            timestamp = batch_info.get("timestamp", 0)
            elapsed = now - timestamp
            if elapsed > self.batch_timeout:
                timeout_batches.append((batch_id, batch_info))
                timeout_batch_ids.append(batch_id)
        
        for batch_id, batch_info in timeout_batches:
            # Check if batch still exists (may have been removed by another thread)
            if batch_id not in self.processing_batches:
                continue
            payload = batch_info.get("payload")
            if payload:
                # Remove _batch_id to avoid confusion
                requeue_payload = dict(payload)
                requeue_payload.pop("_batch_id", None)
                try:
                    self.train_queue.put(requeue_payload, block=False)
                    requeued += 1
                    print(f"[ORCH] Requeued timeout batch {batch_id} (elapsed: {elapsed:.0f}s)")
                except queue.Full:
                    print(f"[ORCH] WARNING: Could not requeue timeout batch {batch_id} (queue full)")
                # Only delete if still exists (thread-safe)
                if batch_id in self.processing_batches:
                    del self.processing_batches[batch_id]
        
        return requeued, timeout_batch_ids
