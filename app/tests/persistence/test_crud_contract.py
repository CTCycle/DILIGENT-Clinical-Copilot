from __future__ import annotations

from repositories.serialization.data import DataSerializer


###############################################################################
def test_session_crud_round_trip(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    serializer = DataSerializer(engine=persistence_engine)
    session_id = serializer.save_clinical_session(
        {
            "patient_name": "CRUD Patient",
            "patient_visit_date": "2026-01-15",
            "anamnesis": "Persistent fatigue.",
            "drugs": "Drug Alpha 10 mg",
            "laboratory_analysis": "ALT 120 U/L",
            "session_status": "successful",
        }
    )
    assert session_id is not None

    items, total = serializer.list_sessions(
        search="CRUD Patient",
        status_filter=None,
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    assert total == 1
    assert items[0]["session_id"] == session_id
    detail = serializer.get_session_detail(session_id)
    assert detail is not None
    assert detail["patient_name"] == "CRUD Patient"
    assert detail["visit_date"].isoformat() == "2026-01-15"
    assert serializer.delete_session(session_id) is True
    assert serializer.get_session_detail(session_id) is None
    assert serializer.delete_session(session_id) is False
