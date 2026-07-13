from __future__ import annotations

from collections import Counter

from sqlalchemy import event

from repositories.serialization.data import DataSerializer


def test_session_listing_uses_bounded_query_shape(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    serializer = DataSerializer(engine=persistence_engine)
    for index in range(3):
        serializer.save_clinical_session({"patient_name": f"Count {index}"})

    statements: Counter[str] = Counter()

    def before_cursor_execute(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            statements[statement.split()[0].upper()] += 1

    event.listen(persistence_engine, "before_cursor_execute", before_cursor_execute)
    try:
        serializer.list_sessions(
            search=None,
            status_filter=None,
            date_mode=None,
            filter_date=None,
            offset=0,
            limit=10,
        )
    finally:
        event.remove(persistence_engine, "before_cursor_execute", before_cursor_execute)

    assert statements["SELECT"] == 2
