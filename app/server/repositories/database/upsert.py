from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from repositories.schemas.models import (
    ApplicationConfiguration,
    DrugAlias,
    DrugRxnormCode,
)


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

###############################################################################
def _dialect_insert(db_session: Session, model):
    dialect = db_session.get_bind().dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        raise ValueError(f"Unsupported upsert dialect: {dialect}")
    return insert(model)

###############################################################################
def upsert_drug_alias(
    db_session: Session,
    *,
    drug_id: int,
    alias: str,
    alias_norm: str,
    alias_kind: str,
    source: str,
    term_type: str | None,
) -> None:
    statement = _dialect_insert(db_session, DrugAlias).values(
        drug_id=drug_id,
        alias=alias,
        alias_norm=alias_norm,
        alias_kind=alias_kind,
        source=source,
        term_type=term_type,
    )
    statement = statement.on_conflict_do_update(
        index_elements=[
            DrugAlias.drug_id,
            DrugAlias.alias_norm,
            DrugAlias.alias_kind,
            DrugAlias.source,
        ],
        set_={
            "alias": statement.excluded.alias,
            "term_type": statement.excluded.term_type,
        },
    )
    db_session.execute(statement)

###############################################################################
def upsert_drug_aliases(db_session: Session, values: list[dict[str, Any]]) -> None:
    """Atomically upsert a deduplicated batch of drug aliases."""
    if not values:
        return
    statement = _dialect_insert(db_session, DrugAlias).values(values)
    statement = statement.on_conflict_do_update(
        index_elements=[
            DrugAlias.drug_id,
            DrugAlias.alias_norm,
            DrugAlias.alias_kind,
            DrugAlias.source,
        ],
        set_={
            "alias": statement.excluded.alias,
            "term_type": statement.excluded.term_type,
        },
    )
    db_session.execute(statement)

###############################################################################
def upsert_drug_rxnorm_code(
    db_session: Session,
    *,
    drug_id: int,
    rxcui: str,
) -> None:
    statement = _dialect_insert(db_session, DrugRxnormCode).values(
        drug_id=drug_id,
        rxcui=rxcui,
    )
    statement = statement.on_conflict_do_nothing(
        index_elements=[DrugRxnormCode.rxcui]
    )
    db_session.execute(statement)

###############################################################################
def upsert_drug_rxnorm_codes(
    db_session: Session, values: list[dict[str, Any]]
) -> None:
    """Atomically insert a deduplicated batch of RxCUI mappings."""
    if not values:
        return
    statement = _dialect_insert(db_session, DrugRxnormCode).values(values)
    statement = statement.on_conflict_do_nothing(
        index_elements=[DrugRxnormCode.rxcui]
    )
    db_session.execute(statement)
