from __future__ import annotations

from datetime import datetime

from repositories.schemas.models import Base
from repositories.serialization.data import DataSerializer
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


def build_serializer() -> tuple[DataSerializer, sessionmaker]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return DataSerializer(engine=engine), sessionmaker(bind=engine, future=True)


def seed_session(serializer: DataSerializer) -> int:
    session_id = serializer.save_clinical_session(
        {
            "patient_name": "Manual Edit Patient",
            "session_timestamp": datetime(2025, 1, 1, 8, 30),
            "version": 1,
            "anamnesis": "Original anamnesis",
            "drugs": "acetaminophen",
            "final_report": "Legacy report section",
            "session_result_payload": {
                "original_session_text": "Original clinical source text",
                "report": "Original official report",
            },
        }
    )
    if session_id is None:
        raise AssertionError("Session seed failed")
    return session_id


def test_manual_edit_updates_report_without_mutating_source_text_or_version() -> None:
    serializer, session_factory = build_serializer()
    session_id = seed_session(serializer)

    response = serializer.update_current_report_text_with_manual_audit(
        session_id,
        report_text="Corrected official report",
        edited_fields=["report_text"],
        reviewer_note="Fixed incorrect chronology wording.",
        edited_by="Reviewer One",
        metadata={"origin": "unit-test"},
    )

    assert response is not None
    session = response["session"]
    audit = response["audit"]
    assert session["session_id"] == session_id
    assert session["version"] == 1
    assert session["official_report_text"] == "Corrected official report"
    assert session["report"] == "Corrected official report"
    assert session["source_clinical_text"] == "Original clinical source text"
    assert session["session_text"] == "Original clinical source text"
    assert len(session["manual_edit_history"]) == 1

    assert audit["edited_by"] == "Reviewer One"
    assert audit["actor_source"] == "manual_entry"
    assert audit["actor_confidence"] == "unverified"
    assert audit["reviewer_note"] == "Fixed incorrect chronology wording."
    assert audit["edited_fields"] == ["report_text"]
    assert audit["previous_text_hash"] != audit["new_text_hash"]

    detail = serializer.get_session_detail(session_id)
    assert detail is not None
    assert detail["official_report_text"] == "Corrected official report"
    assert detail["source_clinical_text"] == "Original clinical source text"
    assert len(detail["manual_edit_history"]) == 1

    from repositories.schemas.models import ClinicalSession

    with session_factory() as db_session:
        sessions = db_session.execute(select(ClinicalSession)).scalars().all()
        assert len(sessions) == 1
        assert int(sessions[0].version or 0) == 1
