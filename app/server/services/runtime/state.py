from __future__ import annotations

from dataclasses import dataclass, field
import threading
from time import monotonic
from typing import Any
from collections.abc import Mapping

from domain.jobs import JobStatusResponse


@dataclass
class JobState:
    job_id: str
    job_type: str
    status: str
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=monotonic)
    completed_at: float | None = None
    version: int = 0
    stop_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def update(self, **kwargs: Any) -> None:
        with self.lock:
            changed = False
            for key, value in kwargs.items():
                if hasattr(self, key):
                    changed = True
                    setattr(self, key, value)
            if changed:
                self.version += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "job_type": self.job_type,
                "status": self.status,
                "progress": self.progress,
                "result": self.result,
                "error": self.error,
                "created_at": self.created_at,
                "completed_at": self.completed_at,
                "version": self.version,
            }

    def merge_result(self, result_delta: Mapping[str, Any]) -> JobStatusResponse:
        with self.lock:
            existing = dict(self.result or {})
            existing.update(dict(result_delta))
            self.result = existing
            self.version += 1
            return JobStatusResponse(
                job_id=self.job_id,
                job_type=self.job_type,
                status=self.status,
                progress=self.progress,
                result=self.result,
                error=self.error,
                created_at=self.created_at,
                completed_at=self.completed_at,
                version=self.version,
            )
