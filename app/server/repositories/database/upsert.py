from __future__ import annotations

from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from repositories.schemas.configuration import ApplicationConfiguration
from repositories.schemas.knowledge import (
    DrugAlias,
    DrugRxnormCode,
    LiverToxMonograph,
)


###############################################################################
def upsert_application_configuration(
    db_session: Session,
    *,
    payload: dict[str, Any],
) -> ApplicationConfiguration:
    """Atomically replace the fixed-id application configuration row."""
    values = {
        "id": 1,
        "payload": payload,
    }
    statement = dialect_insert(db_session, ApplicationConfiguration).values(**values)
    statement = statement.on_conflict_do_update(
        index_elements=[ApplicationConfiguration.id],
        set_={
            "payload": statement.excluded.payload,
            "revision": ApplicationConfiguration.revision + 1,
            "updated_at": func.now(),
        },
    )
    result = db_session.execute(statement.returning(ApplicationConfiguration))
    row = result.scalar_one_or_none()
    if row is None:
        raise RuntimeError("Application configuration upsert did not return a row")
    return row


###############################################################################
def insert_application_configuration_if_missing(
    db_session: Session,
    *,
    payload: dict[str, Any],
) -> bool:
    """Insert the fixed-id configuration row without overwriting user state."""
    statement = dialect_insert(
        db_session,
        ApplicationConfiguration,
    ).values(id=1, payload=payload)
    statement = statement.on_conflict_do_nothing(
        index_elements=[ApplicationConfiguration.id]
    )
    result = db_session.execute(statement)
    return bool(result.rowcount)


###############################################################################
def dialect_insert(db_session: Session, model: Any) -> Any:
    dialect = db_session.get_bind().dialect.name
    if dialect == "sqlite":
        return sqlite_insert(model)
    elif dialect == "postgresql":
        return postgres_insert(model)
    raise ValueError(f"Unsupported upsert dialect: {dialect}")


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
    statement = dialect_insert(db_session, DrugAlias).values(
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
    statement = dialect_insert(db_session, DrugAlias).values(values)
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
    statement = dialect_insert(db_session, DrugRxnormCode).values(
        drug_id=drug_id,
        rxcui=rxcui,
    )
    statement = statement.on_conflict_do_nothing(index_elements=[DrugRxnormCode.rxcui])
    db_session.execute(statement)


###############################################################################
def upsert_drug_rxnorm_codes(db_session: Session, values: list[dict[str, Any]]) -> None:
    """Atomically insert a deduplicated batch of RxCUI mappings."""
    if not values:
        return
    statement = dialect_insert(db_session, DrugRxnormCode).values(values)
    statement = statement.on_conflict_do_nothing(index_elements=[DrugRxnormCode.rxcui])
    db_session.execute(statement)


###############################################################################
def upsert_livertox_monographs(
    db_session: Session, values: list[dict[str, Any]]
) -> None:
    """Atomically replace a batch of LiverTox monograph records by identity."""
    if not values:
        return
    statement = dialect_insert(db_session, LiverToxMonograph).values(values)
    statement = statement.on_conflict_do_update(
        index_elements=[LiverToxMonograph.monograph_key],
        set_={
            key: getattr(statement.excluded, key)
            for key in (
                "drug_id",
                "drug_name_norm",
                "nbk_id",
                "excerpt",
                "likelihood_score",
                "last_update",
                "reference_count",
                "year_approved",
                "agent_classification",
                "primary_classification",
                "secondary_classification",
                "include_in_livertox",
                "source_url",
                "source_last_modified",
            )
        },
    )
    db_session.execute(statement)
