from __future__ import annotations

import re

SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
POSTGRES_DATABASE_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,62}$")


# ----------------------------------------------------------------------------- 
def normalize_postgres_engine(engine: str | None) -> str:
    if not engine:
        return "postgresql+psycopg"
    lowered = engine.lower()
    if lowered in {"postgres", "postgresql"}:
        return "postgresql+psycopg"
    return engine


###############################################################################
def validate_sql_identifier(
    identifier: str,
    *,
    label: str = "identifier",
) -> str:
    normalized = str(identifier or "").strip()
    if not normalized:
        raise ValueError(f"Invalid SQL {label}: empty value")
    if not SQL_IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(f"Invalid SQL {label}: {normalized!r}")
    return normalized


###############################################################################
def validate_postgres_database_name(database_name: str) -> str:
    normalized = str(database_name or "").strip()
    if not normalized:
        raise ValueError("Invalid PostgreSQL database name: empty value")
    if not POSTGRES_DATABASE_NAME_RE.fullmatch(normalized):
        raise ValueError("Invalid PostgreSQL database name")
    return normalized


MISSING_TABLE_MESSAGE = "Table %s does not exist"
