from __future__ import annotations

from collections import Counter

from sqlalchemy import event, select

from repositories.schemas.models import Drug, DrugAlias, DrugRxnormCode
from repositories.serialization.data import DataSerializer


def test_rxnav_ingestion_uses_set_based_writes(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    statements: Counter[str] = Counter()
    commits = 0

    def before_cursor_execute(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        normalized = statement.upper()
        for table in ("DRUGS", "DRUG_RXNORM_CODES", "DRUG_ALIASES"):
            if f"INSERT INTO {table}" in normalized:
                statements[table] += 1

    def after_commit(_connection) -> None:
        nonlocal commits
        commits += 1

    event.listen(persistence_engine, "before_cursor_execute", before_cursor_execute)
    event.listen(persistence_engine, "commit", after_commit)
    try:
        serializer = DataSerializer(engine=persistence_engine)
        serializer.upsert_drugs_catalog_records(
            [
                {
                    "rxcui": "1001",
                    "raw_name": "Drug Alpha 10 MG Tablet",
                    "term_type": "SCD",
                    "name": "Drug Alpha",
                    "brand_names": ["Alpha Brand"],
                    "synonyms": ["Alpha Synonym"],
                },
                {
                    "rxcui": "1002",
                    "raw_name": "Drug Beta 20 MG Tablet",
                    "term_type": "SCD",
                    "name": "Drug Beta",
                    "brand_names": ["Beta Brand"],
                    "synonyms": ["Beta Synonym"],
                },
            ]
        )
    finally:
        event.remove(
            persistence_engine, "before_cursor_execute", before_cursor_execute
        )
        event.remove(persistence_engine, "commit", after_commit)

    with persistence_engine.connect() as connection:
        assert connection.execute(select(Drug)).fetchall()
        assert connection.execute(select(DrugRxnormCode)).fetchall()
        assert connection.execute(select(DrugAlias)).fetchall()

    assert statements == Counter(
        {"DRUGS": 1, "DRUG_RXNORM_CODES": 1, "DRUG_ALIASES": 1}
    )
    assert commits == 1
