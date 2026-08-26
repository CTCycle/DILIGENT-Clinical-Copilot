from __future__ import annotations

import threading
import time

import pytest
from services.runtime.jobs import JobManager
from services.session.session_service import ClinicalSessionService


###############################################################################
def accepts_named_job_id(job_id: str) -> dict[str, object]:
    return {"job_id": job_id}

###############################################################################
def accepts_kwargs(**kwargs: object) -> dict[str, object]:
    return dict(kwargs)

###############################################################################
def accepts_no_job_id() -> dict[str, object]:
    return {}

###############################################################################
class _SnapshotCancelManager:

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> dict[str, object]:
        return {
            "job_id": job_id,
            "job_type": "clinical",
            "status": "running",
            "progress": 0.5,
        }

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> dict[str, object]:
        return {
            "job_id": job_id,
            "job_type": "clinical",
            "status": "running",
            "progress": 0.5,
        }

###############################################################################
def test_clinical_cancel_response_converts_job_snapshot_to_success_bool() -> None:
    service = ClinicalSessionService.__new__(ClinicalSessionService)
    service.job_manager = _SnapshotCancelManager()

    response = service.cancel_clinical_job("job-123")

    assert response.success is True
    assert response.job_id == "job-123"

###############################################################################
@pytest.mark.parametrize(
    ("runner", "expected"),
    [
        (accepts_named_job_id, True),
        (accepts_kwargs, True),
        (accepts_no_job_id, False),
        (len, False),
    ],
)
def test_runner_job_id_detection_covers_supported_and_rejected_callables(
    runner, expected: bool
) -> None:  # type: ignore[no-untyped-def]
    assert JobManager().runner_accepts_job_id(runner) is expected

###############################################################################
def test_running_cancel_remains_active_until_worker_exits() -> None:
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
    assert snapshot["stop_requested"] is True
    assert manager.is_job_running("runtime_test") is True
    running = manager.get_running_job("runtime_test")
    assert running is not None
    assert running["job_id"] == job_id
    release.set()
    for _ in range(20):
        terminal = manager.get_job_status(job_id)
        if terminal and terminal["status"] in {"cancelled", "completed", "failed"}:
            break
        time.sleep(0.05)
    assert terminal is not None
    assert terminal["status"] == "cancelled"
    assert manager.is_job_running("runtime_test") is False

###############################################################################
def test_running_cancel_blocks_duplicate_scope_until_worker_exits() -> None:
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
    for _ in range(20):
        if not manager.is_job_running("runtime_test"):
            break
        time.sleep(0.05)
    assert manager.is_job_running("runtime_test") is False

###############################################################################
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

###############################################################################
def test_job_running_checks_can_be_scoped() -> None:
    manager = JobManager()
    started = threading.Event()
    release = threading.Event()

    def runner() -> dict[str, int]:
        started.set()
        release.wait(timeout=5)
        return {"ok": 1}

    job_id = manager.start_job("runtime_test", runner, scope_key="scope:a")
    try:
        assert started.wait(timeout=1)
        assert manager.is_job_running("runtime_test") is True
        assert manager.is_job_running("runtime_test", scope_key="scope:a") is True
        assert manager.is_job_running("runtime_test", scope_key="scope:b") is False
        running = manager.get_running_job("runtime_test", scope_key="scope:a")
        assert running is not None
        assert running["job_id"] == job_id
        assert running["scope_key"] == "scope:a"
    finally:
        manager.cancel_job(job_id)
        release.set()

###############################################################################
def test_shutdown_stops_new_work_and_waits_for_cooperative_worker() -> None:
    manager = JobManager()
    started = threading.Event()
    release = threading.Event()

    def runner() -> dict[str, int]:
        started.set()
        release.wait(timeout=2)
        return {"ok": 1}

    job_id = manager.start_job("shutdown_test", runner)
    assert started.wait(timeout=1)
    assert manager.shutdown(timeout=0.01) is False
    with pytest.raises(RuntimeError, match="shutting down"):
        manager.start_job("shutdown_test", dict)
    release.set()
    for _ in range(20):
        if not manager.threads:
            break
        time.sleep(0.05)
    assert manager.get_job_status(job_id)["status"] == "cancelled"
    assert manager.shutdown(timeout=1) is True


###############################################################################
def test_terminal_job_records_are_bounded() -> None:
    manager = JobManager()
    manager.max_terminal_jobs = 2
    job_ids = [manager.start_job("bounded_test", dict) for _ in range(3)]
    for job_id in job_ids:
        for _ in range(20):
            if manager.get_job_status(job_id) is None or manager.get_job_status(job_id)["status"] in {
                "completed",
                "failed",
                "cancelled",
            }:
                break
            time.sleep(0.05)
    assert len(manager.jobs) <= 2
