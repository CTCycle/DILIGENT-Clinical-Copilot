from __future__ import annotations

import threading
from datetime import datetime

from repositories.schemas.models import Base
from repositories.serialization.data import DataSerializer
from services.inspection.revision_runner import build_revision_job_scope_key
from services.inspection.service import DataInspectionService
from services.runtime.jobs import JobManager
from sqlalchemy import create_engine

###############################################################################
def build_serializer() -> DataSerializer:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return DataSerializer(engine=engine)

###############################################################################
def build_service(serializer: DataSerializer, jobs: JobManager | None = None) -> DataInspectionService:
    return DataInspectionService(serializer=serializer, jobs=jobs or JobManager())

###############################################################################
def seed_session(serializer: DataSerializer, patient_name: str = "Revision Patient") -> int:
    session_id = serializer.save_clinical_session(
        {
            "patient_name": patient_name,
            "session_timestamp": datetime(2025, 1, 4, 9, 0),
            "version": 1,
            "anamnesis": "Source text for revision",
            "drugs": "drug-c",
            "session_result_payload": {
                "original_session_text": "Source text for revision",
                "report": "Initial report",
            },
        }
    )
    if session_id is None:
        raise AssertionError("Session seed failed")
    return session_id

###############################################################################
def test_revision_job_conflict_is_scoped_to_root_session(monkeypatch) -> None:
    serializer = build_serializer()
    jobs = JobManager()
    service = build_service(serializer, jobs)
    first_session_id = seed_session(serializer, "First")
    second_session_id = seed_session(serializer, "Second")
    first_started = threading.Event()
    release_first = threading.Event()

    def runner() -> dict[str, bool]:
        first_started.set()
        release_first.wait(timeout=10)
        return {"released": True}

    first_job_id = jobs.start_job(
        service.REVISION_JOB_TYPE,
        runner,
        scope_key=build_revision_job_scope_key(first_session_id),
    )
    started_background_jobs: list[str] = []

    def fake_start_revision_background_job(**kwargs):
        root_session_id = int(kwargs["root_session_id"])
        job_id = jobs.start_job(
            service.REVISION_JOB_TYPE,
            lambda: {"ok": True},
            scope_key=build_revision_job_scope_key(root_session_id),
        )
        started_background_jobs.append(job_id)
        status = jobs.get_job_status(job_id)
        if status is None:
            raise AssertionError("Expected fake revision job status")
        return status

    monkeypatch.setattr(
        service,
        "_start_revision_background_job",
        fake_start_revision_background_job,
    )
    assert first_started.wait(timeout=1)
    try:
        try:
            service.start_revision_job(
                first_session_id,
                selected_text=None,
                revision_instruction=None,
                model_overrides={},
                metadata={},
            )
        except ValueError as exc:
            assert "already running" in str(exc)
        else:
            raise AssertionError("Expected same-root revision conflict")

        second_started = service.start_revision_job(
            second_session_id,
            selected_text=None,
            revision_instruction=None,
            model_overrides={},
            metadata={},
        )
        assert second_started["job_type"] == service.REVISION_JOB_TYPE
        assert second_started["job_id"] != first_job_id
        jobs.cancel_job(str(second_started["job_id"]))
    finally:
        jobs.cancel_job(first_job_id)
        for job_id in started_background_jobs:
            jobs.cancel_job(job_id)
        release_first.set()

###############################################################################
def test_missing_revision_job_status_returns_recoverable_process_local_failure() -> None:
    service = build_service(build_serializer())

    status = service.get_job_status("missing-job", expected_type=service.REVISION_JOB_TYPE)

    assert status is not None
    assert status["status"] == "failed"
    assert status["result"] == {
        "recoverable": True,
        "recovery_action": "reload_revision_run_and_retry",
    }
    assert "process-local" in str(status["error"])

###############################################################################
def test_startup_reconciliation_marks_running_revision_runs_failed() -> None:
    serializer = build_serializer()
    session_id = seed_session(serializer)
    source_version = serializer.get_version_record_for_session(session_id)
    if source_version is None:
        raise AssertionError("Missing source version")
    target_shell = serializer.create_revision_version_shell(
        session_id,
        reviewer_note=None,
        configuration={},
        pipeline_run_id="pipe-lost-001",
        initiated_by=None,
    )
    if target_shell is None:
        raise AssertionError("Missing target shell")
    serializer.create_or_update_revision_run(
        pipeline_run_id="pipe-lost-001",
        session_id=session_id,
        root_session_id=session_id,
        source_version_id=int(source_version["version_id"]),
        target_revision_version_id=int(target_shell["version_id"]),
        revision_mode="default",
        revision_kind="llm_assisted_revision",
        configuration={},
        reviewer_note=None,
        status="running",
    )

    build_service(serializer)

    run = serializer.get_revision_run("pipe-lost-001")
    assert run is not None
    assert run["status"] == "failed"
    assert run["error"]["code"] == "revision_job_process_lost"
