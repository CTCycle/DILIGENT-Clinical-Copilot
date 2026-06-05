from __future__ import annotations

from datetime import datetime

from repositories.schemas.models import Base
from repositories.serialization.data import DataSerializer
from services.inspection.service import DataInspectionService
from services.runtime.jobs import JobManager
from sqlalchemy import create_engine


def build_service() -> tuple[DataInspectionService, DataSerializer]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    serializer = DataSerializer(engine=engine)
    service = DataInspectionService(serializer=serializer, jobs=JobManager())
    return service, serializer


def seed_session(serializer: DataSerializer) -> int:
    session_id = serializer.save_clinical_session(
        {
            "patient_name": "Repository Session",
            "session_timestamp": datetime(2025, 1, 2, 10, 15),
            "version": 1,
            "anamnesis": "Stable source clinical narrative",
            "drugs": "amoxicillin",
            "session_result_payload": {
                "original_session_text": "Stable source clinical narrative",
                "report": "Initial report draft",
            },
        }
    )
    if session_id is None:
        raise AssertionError("Session seed failed")
    return session_id


def test_legacy_update_session_route_now_performs_safe_manual_report_edit() -> None:
    service, serializer = build_service()
    session_id = seed_session(serializer)

    updated = service.update_session(
        session_id,
        session_text="Revised report content via legacy route",
        metadata={"reviewer": "Legacy Reviewer"},
    )

    assert updated is not None
    assert updated["official_report_text"] == "Revised report content via legacy route"
    assert updated["source_clinical_text"] == "Stable source clinical narrative"
    assert updated["version"] == 1
    assert updated["metadata"]["reviewer"] == "Legacy Reviewer"
    assert len(updated["manual_edit_history"]) == 1
    assert updated["manual_edit_history"][0]["edited_fields"] == ["report_text"]


def test_metadata_only_update_does_not_create_manual_edit_audit() -> None:
    service, serializer = build_service()
    session_id = seed_session(serializer)

    updated = service.update_session(
        session_id,
        session_text=None,
        metadata={"reviewer": "Metadata Only"},
    )

    assert updated is not None
    assert updated["metadata"]["reviewer"] == "Metadata Only"
    assert updated["official_report_text"] == "Initial report draft"
    assert updated["source_clinical_text"] == "Stable source clinical narrative"
    assert updated["manual_edit_history"] == []
