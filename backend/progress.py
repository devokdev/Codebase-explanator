from __future__ import annotations

import threading
import time
import uuid
from typing import Dict


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict] = {}

    def create(self, kind: str) -> str:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "kind": kind,
                "status": "pending",
                "message": "Queued...",
                "progress": 0,
                "started_at": now,
                "updated_at": now,
                "result": None,
                "error": None,
            }
        return job_id

    def update(self, job_id: str, **fields: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(fields)
            job["updated_at"] = time.time()

    def finish(self, job_id: str, result: Dict) -> None:
        self.update(job_id, status="completed", progress=100, message="Completed.", result=result, error=None)

    def fail(self, job_id: str, error: str) -> None:
        self.update(job_id, status="failed", message=error, error=error)

    def get(self, job_id: str) -> Dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            copied = dict(job)

        copied["elapsed_seconds"] = max(0, int(time.time() - copied["started_at"]))
        return copied
