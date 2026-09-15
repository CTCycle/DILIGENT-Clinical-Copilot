from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest
from repositories.schemas.clinical import ClinicalDrugMention
from repositories.schemas.knowledge import (
    Drug,
    DrugAlias,
    DrugRxnormCode,
    LiverToxMonograph,
)
from repository_fixtures import build_repository_graph
from sqlalchemy import event, select


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
        repository = build_repository_graph(
            engine=persistence_engine
        ).drug_catalog_repository
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
        event.remove(persistence_engine, "before_cursor_execute", before_cursor_execute)
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
        repository = build_repository_graph(
            engine=persistence_engine
        ).knowledge_repository
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
        event.remove(persistence_engine, "before_cursor_execute", before_cursor_execute)
        event.remove(persistence_engine, "commit", after_commit)

    with persistence_engine.connect() as connection:
        assert connection.execute(select(Drug)).fetchall()
        assert connection.execute(select(DrugAlias)).fetchall()
        assert connection.execute(select(LiverToxMonograph)).fetchall()

    assert statements == Counter(
        {"DRUGS": 1, "DRUG_ALIASES": 1, "LIVERTOX_MONOGRAPHS": 1}
    )
    assert commits == 1

###############################################################################
def test_rxnav_snapshot_replacement_reconciles_only_rxnav_owned_rows(
    persistence_engine,
) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    repository = graph.drug_catalog_repository
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
    with graph.context.session_factory() as db_session:
        alpha = db_session.scalar(
            select(Drug).where(Drug.canonical_name_norm == "drug alpha")
        )
        assert alpha is not None
        db_session.add(
            DrugAlias(
                drug_id=alpha.id,
                alias="Alpha clinical alias",
                alias_norm="alpha clinical alias",
                alias_kind="clinical",
                source="clinical",
                term_type=None,
            )
        )
        db_session.commit()

    repository.replace_rxnav_catalog_records(
        [
            {
                "rxcui": "2001",
                "raw_name": "Drug Gamma 30 MG Tablet",
                "term_type": "SCD",
                "name": "Drug Gamma",
                "brand_names": ["Gamma Brand"],
                "synonyms": ["Gamma Synonym"],
            }
        ]
    )

    with graph.context.session_factory() as db_session:
        alpha = db_session.scalar(
            select(Drug).where(Drug.canonical_name_norm == "drug alpha")
        )
        assert alpha is not None
        assert alpha.rxnav_last_update is None
        assert db_session.scalar(
            select(DrugRxnormCode).where(DrugRxnormCode.rxcui == "1001")
        ) is None
        assert db_session.scalar(
            select(DrugAlias).where(
                DrugAlias.drug_id == alpha.id,
                DrugAlias.source == "clinical",
            )
        ) is not None
        assert db_session.scalar(
            select(Drug).where(Drug.canonical_name_norm == "drug gamma")
        ) is not None

###############################################################################
def _large_rxnav_snapshot(size: int = 350) -> list[dict[str, object]]:
    return [
        {
            "rxcui": str(50000 + index),
            "raw_name": f"RepairDrug {index} 10 MG Tablet",
            "term_type": "SCD",
            "name": f"RepairDrug {index}",
            "brand_names": [f"RepairBrand {index}"],
            "synonyms": [f"Repair Synonym {index}"],
        }
        for index in range(size)
    ]

###############################################################################
def test_rxnav_snapshot_replacement_batches_sql_parameters(
    persistence_engine,
) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    repository = graph.drug_catalog_repository

    repository.replace_rxnav_catalog_records(_large_rxnav_snapshot())

    with graph.context.session_factory() as db_session:
        assert len(db_session.execute(select(Drug)).scalars().all()) == 350
        assert len(db_session.execute(select(DrugRxnormCode)).scalars().all()) == 350
        assert len(db_session.execute(select(DrugAlias)).scalars().all()) >= 350 * 3

###############################################################################
def test_rxnav_snapshot_replacement_rolls_back_after_later_batch_failure(
    persistence_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    repository = graph.drug_catalog_repository
    repository.upsert_drugs_catalog_records(
        [
            {
                "rxcui": "49001",
                "raw_name": "Stable Drug 10 MG Tablet",
                "term_type": "SCD",
                "name": "Stable Drug",
                "brand_names": ["Stable Brand"],
                "synonyms": ["Stable Synonym"],
            }
        ]
    )

    import repositories.drug_catalog_repository as repository_module

    original_upsert_aliases = repository_module.upsert_drug_aliases
    call_count = 0

    def fail_on_second_alias_batch(db_session, values):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("synthetic later RxNav alias batch failure")
        original_upsert_aliases(db_session, values)

    monkeypatch.setattr(repository_module, "upsert_drug_aliases", fail_on_second_alias_batch)

    with pytest.raises(RuntimeError, match="later RxNav alias batch failure"):
        repository.replace_rxnav_catalog_records(_large_rxnav_snapshot())

    with graph.context.session_factory() as db_session:
        assert db_session.scalar(
            select(DrugRxnormCode).where(DrugRxnormCode.rxcui == "49001")
        ) is not None
        assert db_session.scalar(
            select(DrugRxnormCode).where(DrugRxnormCode.rxcui == "50000")
        ) is None
        assert db_session.scalar(
            select(Drug).where(Drug.canonical_name_norm == "stable drug")
        ) is not None

###############################################################################
def test_session_drug_persistence_retains_rxnav_and_source_provenance(
    persistence_engine,
) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    session_id = graph.clinical_session_repository.save_clinical_session(
        {
            "patient_name": "Provenance Patient",
            "drugs": "Amoxicillin/clavulanate",
            "matched_drugs": [
                {
                    "raw_drug_name": "Amoxicillin/clavulanate",
                    "rxcui": "12345",
                    "accepted_rxnav_rxcui": "12345",
                    "rxnav_candidates": [
                        {"rxcui": "12345", "accepted": True}
                    ],
                    "rxnav_validation_status": "exact_rxcui",
                    "source_provenance": {
                        "rxnav": {"validated": True, "rxcui": "12345"},
                        "dilirank": {
                            "knowledge_source": "fda_dilirank_2",
                            "evidence_scope": "regimen_component",
                        },
                    },
                }
            ],
        }
    )
    assert session_id is not None

    with graph.context.session_factory() as db_session:
        mention = db_session.query(ClinicalDrugMention).one()
        assert mention.evidence_json["rxcui"] == "12345"
        assert mention.evidence_json["rxnav_candidates"][0]["accepted"] is True
        assert mention.evidence_json["source_provenance"]["dilirank"][
            "evidence_scope"
        ] == "regimen_component"

###############################################################################
def test_livertox_snapshot_replacement_reconciles_source_owned_rows(
    persistence_engine,
) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    repository = graph.knowledge_repository
    repository.save_livertox_records(
        pd.DataFrame(
            [
                {
                    "drug_name": "Drug Alpha",
                    "nbk_id": "NBK001",
                    "brand_name": "Alpha Brand",
                    "excerpt": "Alpha excerpt",
                },
                {
                    "drug_name": "Drug Beta",
                    "nbk_id": "NBK002",
                    "brand_name": "Beta Brand",
                    "excerpt": "Beta excerpt",
                },
            ]
        )
    )
    with graph.context.session_factory() as db_session:
        alpha = db_session.scalar(
            select(Drug).where(Drug.canonical_name_norm == "drug alpha")
        )
        assert alpha is not None
        db_session.add(
            DrugAlias(
                drug_id=alpha.id,
                alias="Alpha RxNav alias",
                alias_norm="alpha rxnav alias",
                alias_kind="synonym",
                source="rxnav",
                term_type=None,
            )
        )
        db_session.commit()

    repository.replace_livertox_records(
        pd.DataFrame(
            [
                {
                    "drug_name": "Drug Alpha",
                    "nbk_id": "NBK003",
                    "brand_name": "Alpha Updated Brand",
                    "excerpt": "Updated alpha excerpt",
                }
            ]
        )
    )

    with graph.context.session_factory() as db_session:
        alpha = db_session.scalar(
            select(Drug).where(Drug.canonical_name_norm == "drug alpha")
        )
        beta = db_session.scalar(
            select(Drug).where(Drug.canonical_name_norm == "drug beta")
        )
        assert alpha is not None
        assert beta is not None
        assert alpha.livertox_nbk_id == "NBK003"
        assert beta.livertox_nbk_id is None
        assert db_session.scalar(
            select(DrugAlias).where(
                DrugAlias.drug_id == alpha.id,
                DrugAlias.alias == "Alpha RxNav alias",
                DrugAlias.source == "rxnav",
            )
        ) is not None
        assert db_session.scalar(
            select(DrugAlias).where(
                DrugAlias.alias == "Beta Brand",
                DrugAlias.source == "livertox",
            )
        ) is None
        assert db_session.scalar(
            select(LiverToxMonograph).where(LiverToxMonograph.nbk_id == "NBK002")
        ) is None
        assert db_session.scalar(
            select(LiverToxMonograph).where(LiverToxMonograph.nbk_id == "NBK003")
        ) is not None
