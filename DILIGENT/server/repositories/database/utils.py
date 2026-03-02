from __future__ import annotations

import re

SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


MISSING_TABLE_MESSAGE = "Table %s does not exist"
