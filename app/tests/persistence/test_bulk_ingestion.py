from __future__ import annotations

from collections import Counter

import pandas as pd
from sqlalchemy import event, select

from repositories.schemas.knowledge import Drug, DrugAlias, DrugRxnormCode, LiverToxMonograph
from repository_fixtures import build_repository_graph

###############################################################################
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
        repository = build_repository_graph(engine=persistence_engine).drug_catalog_repository
        repository.upsert_drugs_catalog_records(
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

###############################################################################
def test_livertox_ingestion_uses_set_based_writes(persistence_engine) -> None:  # type: ignore[no-untyped-def]
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
        for table in ("DRUGS", "DRUG_ALIASES", "LIVERTOX_MONOGRAPHS"):
            if f"INSERT INTO {table}" in normalized:
                statements[table] += 1

    def after_commit(_connection) -> None:
        nonlocal commits
        commits += 1

    event.listen(persistence_engine, "before_cursor_execute", before_cursor_execute)
    event.listen(persistence_engine, "commit", after_commit)
    try:
        repository = build_repository_graph(engine=persistence_engine).knowledge_repository
        repository.save_livertox_records(
            pd.DataFrame(
                [
                    {
                        "drug_name": "Drug Alpha",
                        "nbk_id": "NBK001",
                        "ingredient": "Alpha ingredient",
                        "brand_name": "Alpha brand",
                        "synonyms": "Alpha synonym",
                        "excerpt": "Alpha excerpt",
                        "source_url": "https://example.test/alpha",
                    },
                    {
                        "drug_name": "Drug Beta",
                        "nbk_id": "NBK002",
                        "ingredient": "Beta ingredient",
                        "brand_name": "Beta brand",
                        "synonyms": "Beta synonym",
                        "excerpt": "Beta excerpt",
                        "source_url": "https://example.test/beta",
                    },
                ]
            )
        )
    finally:
        event.remove(
            persistence_engine, "before_cursor_execute", before_cursor_execute
        )
        event.remove(persistence_engine, "commit", after_commit)

    with persistence_engine.connect() as connection:
        assert connection.execute(select(Drug)).fetchall()
        assert connection.execute(select(DrugAlias)).fetchall()
        assert connection.execute(select(LiverToxMonograph)).fetchall()

    assert statements == Counter(
        {"DRUGS": 1, "DRUG_ALIASES": 1, "LIVERTOX_MONOGRAPHS": 1}
    )
    assert commits == 1
