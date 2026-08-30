from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from repository_fixtures import build_repository_graph


###############################################################################
def test_concurrent_session_writes_remain_isolated(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    def save(index: int) -> int:
        repository = build_repository_graph(
            engine=persistence_engine
        ).clinical_session_repository
        session_id = repository.save_clinical_session(
            {"patient_name": f"Concurrent Patient {index}"}
        )
        assert session_id is not None
        return int(session_id)

    with ThreadPoolExecutor(max_workers=4) as executor:
        session_ids = list(executor.map(save, range(4)))

    repository = build_repository_graph(
        engine=persistence_engine
    ).clinical_session_repository
    items, total = repository.list_sessions(
        search="Concurrent Patient",
        status_filter=None,
        date_mode=None,
        filter_date=None,
        offset=0,
        limit=10,
    )
    assert total == 4
    assert {item["session_id"] for item in items} == set(session_ids)
