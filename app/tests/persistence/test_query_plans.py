from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

###############################################################################
def test_session_timestamp_index_is_defined_for_listing(persistence_engine: Engine) -> None:
    indexes = inspect(persistence_engine).get_indexes("clinical_sessions")
    listing_index = next(
        (
            index
            for index in indexes
            if index["name"] == "ix_clinical_sessions_timestamp_id"
        ),
        None,
    )

    assert listing_index is not None
    assert listing_index["column_names"] == ["session_timestamp", "id"]
