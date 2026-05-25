from __future__ import annotations

import threading
import time

from services.runtime.jobs import JobManager
from services.session.session_service import ClinicalSessionService


class _SnapshotCancelManager:
    def get_job_status(self, job_id: str) -> dict[str, object]:
        return {
            "job_id": job_id,
            "job_type": "clinical",
            "status": "running",
            "progress": 0.5,
        }

    def cancel_job(self, job_id: str) -> dict[str, object]:
        return {
            "job_id": job_id,
            "job_type": "clinical",
            "status": "running",
            "progress": 0.5,
        }


def test_clinical_cancel_response_converts_job_snapshot_to_success_bool() -> None:
    service = ClinicalSessionService.__new__(ClinicalSessionService)
    service.job_manager = _SnapshotCancelManager()

    response = service.cancel_clinical_job("job-123")

    assert response.success is True
    assert response.job_id == "job-123"


def test_running_cancel_preserves_running_status_until_worker_exits() -> None:
    manager = JobManager()
    started = threading.Event()
    release = threading.Event()

    def runner() -> dict[str, int]:
        started.set()
        release.wait(timeout=2)
        return {"ok": 1}

    job_id = manager.start_job("runtime_test", runner)
    assert started.wait(timeout=1)
    snapshot = manager.cancel_job(job_id)
    assert snapshot is not None
    assert snapshot["status"] == "running"
    assert manager.is_job_running("runtime_test") is True
    release.set()
    for _ in range(20):
        terminal = manager.get_job_status(job_id)
        if terminal and terminal["status"] in {"cancelled", "completed", "failed"}:
            break
        time.sleep(0.05)
    assert terminal is not None
    assert terminal["status"] == "cancelled"


def test_running_cancel_blocks_duplicate_job_until_terminal() -> None:
    manager = JobManager()
    started = threading.Event()
    release = threading.Event()

    def runner() -> dict[str, int]:
        started.set()
        release.wait(timeout=2)
        return {"ok": 1}

    job_id = manager.start_job("runtime_test", runner)
    assert started.wait(timeout=1)
    manager.cancel_job(job_id)
    assert manager.is_job_running("runtime_test") is True
    release.set()


def test_job_result_merge_is_single_source_of_truth() -> None:
    manager = JobManager()
    release = threading.Event()

    def runner() -> dict[str, int]:
        release.wait(timeout=1)
        return {}

    job_id = manager.start_job("runtime_test", runner)
    manager.update_result(job_id, {"a": 1})
    snapshot = manager.update_result(job_id, {"b": 2})
    release.set()
    assert snapshot is not None
    assert snapshot["result"] == {"a": 1, "b": 2}
