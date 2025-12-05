import queue
import time
from typing import Any, Dict, Iterable, Optional


class ProblemProvider:
    """Manages orchestrator problem data and exposes a queue-based API."""

    def __init__(self, train_data: Optional[Iterable[Dict[str, Any]]] = None, epochs: int = 1, problem_timeout: float = 600.0):
        self.train_data = list(train_data) if train_data else []
        self.epochs = max(1, int(epochs))
        self.problem_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.total_problems = len(self.train_data) * self.epochs
        self.problems_distributed = 0
        self.problem_timeout = problem_timeout  # Timeout in seconds (default 10 minutes)
        self._outstanding_problems: Dict[str, Dict[str, Any]] = {}  # Track problems given to samplers

    def initialize(self) -> None:
        if not self.train_data:
            return
        for _ in range(self.epochs):
            for item in self.train_data:
                try:
                    self.problem_queue.put(item, block=False)
                except queue.Full:
                    break

    def get_problem(self) -> Dict[str, Any]:
        if not self.train_data:
            return {"end": 1, "message": "no problem data available"}
        try:
            problem_item = self.problem_queue.get(block=False)
            self.problems_distributed += 1
            # Track outstanding problem (use a simple hash as ID)
            problem_id = f"prob_{hash(str(problem_item))}_{self.problems_distributed}"
            self._outstanding_problems[problem_id] = {
                "problem": problem_item,
                "timestamp": time.time(),
            }
            return {"ok": True, "problem": problem_item, "_problem_id": problem_id}
        except queue.Empty:
            if self.problems_distributed >= self.total_problems:
                return {"end": 1, "message": "all problems distributed"}
            return {"ok": False, "empty": True, "message": "queue temporarily empty"}

    def mark_problem_completed(self, problem_id: Optional[str]) -> None:
        """Mark a problem as completed (sampler finished processing)."""
        if problem_id and problem_id in self._outstanding_problems:
            del self._outstanding_problems[problem_id]

    def requeue_timeout_problems(self) -> int:
        """Requeue problems that have been outstanding for too long (sampler died).
        Returns the number of problems requeued."""
        now = time.time()
        requeued = 0
        timeout_problems = []
        
        for problem_id, problem_info in list(self._outstanding_problems.items()):
            timestamp = problem_info.get("timestamp", 0)
            elapsed = now - timestamp
            if elapsed > self.problem_timeout:
                timeout_problems.append((problem_id, problem_info))
        
        for problem_id, problem_info in timeout_problems:
            problem = problem_info.get("problem")
            if problem:
                try:
                    self.problem_queue.put(problem, block=False)
                    requeued += 1
                    self.problems_distributed = max(0, self.problems_distributed - 1)
                    print(f"[ORCH] Requeued timeout problem {problem_id} (elapsed: {elapsed:.0f}s)")
                except queue.Full:
                    print(f"[ORCH] WARNING: Could not requeue timeout problem {problem_id} (queue full)")
                del self._outstanding_problems[problem_id]
        
        return requeued
