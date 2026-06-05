from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from repositories.schemas.models import Base
from repositories.serialization.data import DataSerializer
from services.inspection.service import DataInspectionService
from services.runtime.jobs import JobManager
from sqlalchemy import create_engine


def build_service() -> tuple[DataInspectionService, DataSerializer]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    serializer = DataSerializer(engine=engine)
    return DataInspectionService(serializer=serializer, jobs=JobManager()), serializer


def seed_session(serializer: DataSerializer) -> int:
    session_id = serializer.save_clinical_session(
        {
            "patient_name": "Revision Run Patient",
            "session_timestamp": datetime(2025, 1, 3, 9, 0),
            "version": 1,
            "anamnesis": "Source text",
            "drugs": "drug-b",
            "session_result_payload": {
                "original_session_text": "Source text",
                "report": "Initial report",
            },
        }
    )
    if session_id is None:
        raise AssertionError("Session seed failed")
    return session_id


def test_start_revision_job_persists_run_and_single_draft_shell() -> None:
    service, serializer = build_service()
    session_id = seed_session(serializer)

    def fake_revision_runner(**kwargs: Any) -> dict[str, Any]:
        time.sleep(0.01)
        return {
            "session_id": None,
            "pipeline_run_id": kwargs["pipeline_run_id"],
            "version": kwargs["version"],
            "result_payload": {"report": "draft only"},
        }

    service.run_revision_job = fake_revision_runner  # type: ignore[method-assign]

    started = service.start_revision_job(
        session_id,
        selected_text=None,
        revision_instruction="Focus on chronology.",
        model_overrides={"clinical_model": "model-x"},
        metadata={"reviewer": "Reviewer B", "revision_note": "check chronology"},
    )

    job_id = str(started["job_id"])
    for _ in range(40):
        payload = service.get_job_status(job_id, expected_type=service.REVISION_JOB_TYPE)
        if payload and payload["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)

    status_payload = service.get_job_status(job_id, expected_type=service.REVISION_JOB_TYPE)
    assert status_payload is not None
    pipeline_run_id = str(status_payload["result"]["pipeline_run_id"])

    run = service.get_revision_run(pipeline_run_id)
    assert run is not None
    assert run["revision_mode"] == "instruction_guided"
    assert run["revision_kind"] == "llm_assisted_revision"
    assert run["status"] in {"running", "failed", "completed"}

    versions = service.list_session_versions(session_id)
    draft_versions = [item for item in versions if item["pipeline_run_id"] == pipeline_run_id]
    assert len(draft_versions) == 1
    assert draft_versions[0]["version_status"] == "draft_revision"


def test_retry_revision_job_reuses_existing_draft_shell() -> None:
    service, serializer = build_service()
    session_id = seed_session(serializer)
    captured_calls: list[dict[str, Any]] = []

    def fake_revision_runner(**kwargs: Any) -> dict[str, Any]:
        captured_calls.append(dict(kwargs))
        time.sleep(0.01)
        return {
            "session_id": None,
            "pipeline_run_id": kwargs["pipeline_run_id"],
            "version": kwargs["version"],
            "result_payload": {"report": "draft only"},
        }

    service.run_revision_job = fake_revision_runner  # type: ignore[method-assign]

    started = service.start_revision_job(
        session_id,
        selected_text="ALT trend",
        revision_instruction="Focus on chronology.",
        model_overrides={"clinical_model": "model-x"},
        metadata={"reviewer": "Reviewer B", "revision_note": "check chronology"},
    )

    initial_job_id = str(started["job_id"])
    for _ in range(40):
        payload = service.get_job_status(initial_job_id, expected_type=service.REVISION_JOB_TYPE)
        if payload and payload["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)

    initial_status = service.get_job_status(
        initial_job_id, expected_type=service.REVISION_JOB_TYPE
    )
    assert initial_status is not None
    pipeline_run_id = str(initial_status["result"]["pipeline_run_id"])
    target_revision_version_id = int(
        initial_status["result"]["target_revision_version_id"]
    )

    run = service.get_revision_run(pipeline_run_id)
    assert run is not None
    serializer.create_or_update_revision_run(
        pipeline_run_id=pipeline_run_id,
        session_id=int(run["session_id"]),
        root_session_id=int(run["root_session_id"]),
        source_version_id=int(run["source_version_id"]),
        target_revision_version_id=int(run["target_revision_version_id"]),
        revision_mode=str(run["revision_mode"]),
        revision_kind=str(run["revision_kind"]),
        configuration=dict(run["configuration"] or {}),
        reviewer_note=run["reviewer_note"],
        status="failed",
        initiated_by=run["initiated_by"],
        actor_source=str(run["actor_source"]),
        actor_confidence=str(run["actor_confidence"]),
        error={"message": "Simulated failed run"},
        trace_id=run["trace_id"],
    )

    retried = service.retry_revision_job(pipeline_run_id)
    retry_job_id = str(retried["job_id"])
    for _ in range(40):
        payload = service.get_job_status(retry_job_id, expected_type=service.REVISION_JOB_TYPE)
        if payload and payload["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)

    retry_status = service.get_job_status(
        retry_job_id, expected_type=service.REVISION_JOB_TYPE
    )
    assert retry_status is not None
    assert str(retry_status["result"]["pipeline_run_id"]) == pipeline_run_id
    assert int(retry_status["result"]["target_revision_version_id"]) == target_revision_version_id

    versions = service.list_session_versions(session_id)
    draft_versions = [item for item in versions if item["pipeline_run_id"] == pipeline_run_id]
    assert len(draft_versions) == 1
    assert int(draft_versions[0]["version_id"]) == target_revision_version_id

    assert len(captured_calls) == 2
    assert captured_calls[1]["pipeline_run_id"] == pipeline_run_id
    assert captured_calls[1]["target_revision_version_id"] == target_revision_version_id
    assert captured_calls[1]["selected_text"] == "ALT trend"
    assert captured_calls[1]["revision_instruction"] == "Focus on chronology."
