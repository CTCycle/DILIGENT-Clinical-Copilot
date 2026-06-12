from __future__ import annotations

from datetime import datetime

from repositories.schemas.models import Base
from repositories.serialization.data import DataSerializer
from sqlalchemy import create_engine


###############################################################################
def build_serializer() -> DataSerializer:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return DataSerializer(engine=engine)


###############################################################################
def test_list_session_versions_includes_official_versions_and_draft_shells() -> None:
    serializer = build_serializer()
    root_session_id = serializer.save_clinical_session(
        {
            "patient_name": "Version History Patient",
            "session_timestamp": datetime(2025, 1, 1, 8, 0),
            "version": 1,
            "anamnesis": "Root source",
            "drugs": "drug-a",
            "session_result_payload": {
                "original_session_text": "Root source",
                "report": "Root report",
            },
        }
    )
    if root_session_id is None:
        raise AssertionError("Root session was not saved")

    revision_session_id = serializer.save_clinical_session(
        {
            "patient_name": "Version History Patient",
            "session_timestamp": datetime(2025, 1, 2, 8, 0),
            "version": 2,
            "original_session_id": root_session_id,
            "anamnesis": "Revised source",
            "drugs": "drug-a",
            "session_result_payload": {
                "original_session_text": "Revised source",
                "report": "Revised report",
            },
        }
    )
    if revision_session_id is None:
        raise AssertionError("Revision session was not saved")

    shell = serializer.create_revision_version_shell(
        revision_session_id,
        reviewer_note="Investigate lab mismatch",
        configuration={"mode": "default"},
        pipeline_run_id="pipe-001",
        initiated_by="Reviewer A",
    )
    assert shell is not None
    assert shell["version_status"] == "draft_revision"
    assert shell["version_number"] == 3

    versions = serializer.list_session_versions(root_session_id)
    assert [item["version_number"] for item in versions] == [1, 2, 3]
    assert versions[0]["revision_kind"] == "original"
    assert versions[1]["revision_kind"] == "llm_assisted_revision"
    assert versions[1]["version_status"] == "current"
    assert versions[2]["version_status"] == "draft_revision"
    assert versions[2]["session_id"] is None

    detail = serializer.get_session_version_detail(
        root_session_id,
        version_id=int(versions[1]["version_id"]),
    )
    assert detail is not None
    assert detail["version"]["version_number"] == 2
    assert detail["session"]["session_id"] == revision_session_id


###############################################################################
def test_failed_revision_run_preserves_failed_version_shell() -> None:
    serializer = build_serializer()
    root_session_id = serializer.save_clinical_session(
        {
            "patient_name": "Failed Revision Patient",
            "session_timestamp": datetime(2025, 1, 1, 8, 0),
            "version": 1,
            "anamnesis": "Root source",
            "drugs": "drug-a",
            "session_result_payload": {
                "original_session_text": "Root source",
                "report": "Root report",
            },
        }
    )
    if root_session_id is None:
        raise AssertionError("Root session was not saved")

    source_version = serializer.get_version_record_for_session(root_session_id)
    if source_version is None:
        raise AssertionError("Source version was not created")

    pipeline_run_id = "pipe-failed-001"
    shell = serializer.create_revision_version_shell(
        root_session_id,
        reviewer_note="Failure regression",
        configuration={"mode": "default"},
        pipeline_run_id=pipeline_run_id,
        initiated_by="Reviewer A",
    )
    assert shell is not None

    serializer.create_or_update_revision_run(
        pipeline_run_id=pipeline_run_id,
        session_id=root_session_id,
        root_session_id=root_session_id,
        source_version_id=int(source_version["version_id"]),
        target_revision_version_id=int(shell["version_id"]),
        revision_mode="default",
        revision_kind="llm_assisted_revision",
        configuration={"mode": "default"},
        reviewer_note="Failure regression",
        status="running",
    )

    serializer.fail_revision_run(
        pipeline_run_id=pipeline_run_id,
        error={"message": "generation failed"},
    )

    failed_run = serializer.get_revision_run(pipeline_run_id)
    assert failed_run is not None
    assert failed_run["status"] == "failed"
    assert failed_run["error"] == {"message": "generation failed"}

    versions = serializer.list_session_versions(root_session_id)
    failed_shell = next(
        item for item in versions if item["pipeline_run_id"] == pipeline_run_id
    )
    assert failed_shell["session_id"] is None
    assert failed_shell["version_status"] == "qa_failed"
    assert failed_shell["llm_qa_status"] == "failed"
