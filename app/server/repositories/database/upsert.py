from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from repositories.schemas.models import ApplicationConfiguration


def upsert_application_configuration(
    db_session: Session,
    *,
    payload: dict[str, Any],
    schema_version: int = 1,
) -> ApplicationConfiguration:
    """Atomically replace the fixed-id application configuration row."""
    dialect = db_session.get_bind().dialect.name
    values = {
        "id": 1,
        "schema_version": int(schema_version),
        "payload": payload,
    }
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        raise ValueError(f"Unsupported upsert dialect: {dialect}")
    statement = insert(ApplicationConfiguration).values(**values)
    statement = statement.on_conflict_do_update(
        index_elements=[ApplicationConfiguration.id],
        set_={
            "schema_version": statement.excluded.schema_version,
            "payload": statement.excluded.payload,
            "revision": ApplicationConfiguration.revision + 1,
        },
    )
    db_session.execute(statement)
    db_session.flush()
    row = db_session.get(ApplicationConfiguration, 1)
    if row is None:
        raise RuntimeError("Application configuration upsert did not return a row")
    return row
