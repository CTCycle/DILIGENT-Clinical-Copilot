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
            assert "USING" in rendered and "INDEX" in rendered, (
                f"Expected index scan but plan is:\n{rendered}"
            )
        else:
            plan = connection.execute(
                text(
                    "EXPLAIN SELECT id FROM clinical_sessions "
                    "ORDER BY session_timestamp DESC, id DESC"
                )
            ).fetchall()
            rendered = " ".join(str(row) for row in plan).upper()
            assert "INDEX SCAN" in rendered or "INDEX ONLY SCAN" in rendered or "INDEX" in rendered, (
                f"Expected index usage but plan is:\n{rendered}"
            )
