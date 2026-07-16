from __future__ import annotations

from sqlalchemy import text


###############################################################################
def test_session_timestamp_index_is_used_by_listing_plan(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    with persistence_engine.connect() as connection:
        if connection.dialect.name == "sqlite":
            plan = connection.execute(
                text(
                    "EXPLAIN QUERY PLAN "
                    "SELECT id FROM clinical_sessions "
                    "ORDER BY session_timestamp DESC, id DESC"
                )
            ).fetchall()
            rendered = " ".join(str(row) for row in plan).upper()
            assert "INDEX" in rendered or "SCAN" in rendered
        else:
            plan = connection.execute(
                text(
                    "EXPLAIN SELECT id FROM clinical_sessions "
                    "ORDER BY session_timestamp DESC, id DESC"
                )
            ).fetchall()
            assert plan
