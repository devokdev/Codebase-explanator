from __future__ import annotations

from pathlib import Path
import threading
import time
import uuid
from typing import Dict

from .utils import ensure_directory, read_json, write_json


class JobStore:
    def __init__(self, storage_dir: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict] = {}
        self.storage_dir = ensure_directory(storage_dir) if storage_dir is not None else None
        self._load_jobs()

    def _job_path(self, job_id: str) -> Path | None:
        if self.storage_dir is None:
            return None
        return self.storage_dir / f"{job_id}.json"

    def _save_job(self, job_id: str) -> None:
        path = self._job_path(job_id)
        if path is not None:
            write_json(path, self._jobs[job_id])

    def _load_jobs(self) -> None:
        if self.storage_dir is None:
            return

        for path in self.storage_dir.glob("*.json"):
            job = read_json(path, default=None)
            if not isinstance(job, dict) or "job_id" not in job:
                continue
            if job.get("status") in {"pending", "running"}:
                job["status"] = "failed"
                job["message"] = "Backend restarted during ingestion. Please ingest again."
                job["error"] = job["message"]
                job["updated_at"] = time.time()
                write_json(path, job)
            self._jobs[job["job_id"]] = job

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
            self._save_job(job_id)
        return job_id

    def update(self, job_id: str, **fields: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(fields)
            job["updated_at"] = time.time()
            self._save_job(job_id)

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
