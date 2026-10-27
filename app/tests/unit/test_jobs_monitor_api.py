# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import threading

import pytest
from fastapi import HTTPException, Response

from api import jobs as jobs_api
from services.runtime.jobs import JobManager


###############################################################################
@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> JobManager:
    job_manager = JobManager()
    monkeypatch.setattr(jobs_api, "get_job_manager", lambda: job_manager)
    return job_manager


###############################################################################
def _start_blocking_job(manager: JobManager, job_type: str) -> tuple[str, threading.Event]:
    release = threading.Event()
    reported = threading.Event()

    def runner(job_id: str) -> dict[str, object]:
        manager.update_result(job_id, {"progress_message": "Working", "phase": "embed"})
        reported.set()
        release.wait(5)
        return {}

    job_id = manager.start_job(job_type, runner)
    assert reported.wait(5)
    return job_id, release


###############################################################################
def test_running_job_without_measured_progress_is_indeterminate(manager: JobManager) -> None:
    job_id, release = _start_blocking_job(manager, "rag_update")
    try:
        response = jobs_api.list_jobs(Response())
        (job,) = response.jobs
        assert job.job_id == job_id
        assert job.status == "running"
        assert job.progress is None
        assert (job.phase, job.message) == ("embed", "Working")
        assert job.cancellable is True

        manager.update_progress(job_id, 40.0)
        assert jobs_api.list_jobs(Response()).jobs[0].progress == 40.0
    finally:
        release.set()


###############################################################################
def test_list_puts_active_jobs_first_and_reports_wall_clock_times(manager: JobManager) -> None:
    done_id = manager.start_job("clinical", lambda: {})
    manager.threads[done_id].join(5)
    active_id, release = _start_blocking_job(manager, "ollama_pull")
    try:
        response = Response()
        jobs = jobs_api.list_jobs(response).jobs
        assert response.headers["Cache-Control"] == "no-store"
        assert [job.job_id for job in jobs] == [active_id, done_id]
        finished = jobs[1]
        assert finished.status == "completed"
        assert finished.progress == 100.0
        assert finished.finished_at is not None
        assert finished.finished_at >= finished.started_at
    finally:
        release.set()


###############################################################################
def test_cancel_is_limited_to_safe_job_types(manager: JobManager) -> None:
    revision_id, release_revision = _start_blocking_job(manager, "session_revision")
    rag_id, release_rag = _start_blocking_job(manager, "rag_update")
    try:
        with pytest.raises(HTTPException) as unsafe:
            jobs_api.cancel_job(revision_id)
        assert unsafe.value.status_code == 409

        assert jobs_api.cancel_job(rag_id).success is True
        assert manager.should_stop(rag_id) is True
        stopping = next(job for job in jobs_api.list_jobs(Response()).jobs if job.job_id == rag_id)
        assert stopping.cancellable is False

        with pytest.raises(HTTPException) as missing:
            jobs_api.cancel_job("missing")
        assert missing.value.status_code == 404
    finally:
        release_revision.set()
        release_rag.set()
