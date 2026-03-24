from __future__ import annotations

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine

from DILIGENT.server.entities.inspection import (
    CatalogListFilters,
    MAX_SEARCH_LENGTH,
    SessionListFilters,
)
from DILIGENT.server.repositories.serialization.data import _RepositorySerializationService
from DILIGENT.server.repositories.schemas.models import Base


# -----------------------------------------------------------------------------
def test_session_search_filter_strips_control_characters() -> None:
    filters = SessionListFilters(search=" \x00  metformin\t\n ")

    assert filters.search == "metformin"


# -----------------------------------------------------------------------------
def test_catalog_search_filter_rejects_oversized_values() -> None:
    oversized = "a" * (MAX_SEARCH_LENGTH + 1)

    with pytest.raises(ValidationError):
        CatalogListFilters(search=oversized)


# -----------------------------------------------------------------------------
def test_search_pattern_escapes_like_wildcards() -> None:
    service = object.__new__(_RepositorySerializationService)

    pattern = service.build_search_pattern(r"  100%_match\check  ")

    assert pattern == r"%100\%\_match\\check%"


# -----------------------------------------------------------------------------
def test_schema_guard_rejects_missing_required_columns() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE drugs")
        connection.exec_driver_sql(
            "CREATE TABLE drugs ("
            "id INTEGER NOT NULL PRIMARY KEY, "
            "canonical_name TEXT NOT NULL, "
            "canonical_name_norm VARCHAR NOT NULL, "
            "rxnorm_rxcui VARCHAR, "
            "livertox_nbk_id VARCHAR"
            ")"
        )

    service = object.__new__(_RepositorySerializationService)
    service.engine = engine

    with pytest.raises(RuntimeError, match="missing required column"):
        service.ensure_session_result_table()
