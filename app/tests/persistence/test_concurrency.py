from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from repositories.serialization.data import DataSerializer

###############################################################################
def test_concurrent_session_writes_remain_isolated(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    def save(index: int) -> int:
        serializer = DataSerializer(engine=persistence_engine)
        session_id = serializer.save_clinical_session(
            {"patient_name": f"Concurrent Patient {index}"}
        )
        assert session_id is not None
        return int(session_id)

    with ThreadPoolExecutor(max_workers=4) as executor:
        session_ids = list(executor.map(save, range(4)))

    serializer = DataSerializer(engine=persistence_engine)
    items, total = serializer.list_sessions(
        search="Concurrent Patient",
        status_filter=None,
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    assert total == 4
    assert {item["session_id"] for item in items} == set(session_ids)
