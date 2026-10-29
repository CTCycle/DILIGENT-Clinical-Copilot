# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from collections import Counter

import pandas as pd
from repositories.schemas.clinical import ClinicalDrugMention
from repositories.schemas.knowledge import (
    Drug,
    DrugAlias,
    DrugRxnormCode,
    LiverToxMonograph,
)
from repositories.terminology_lookup_repository import TerminologyLookupRepository
from repositories.terminology_repository import TerminologyRepository
from repository_fixtures import build_repository_graph
from services.updater.terminology.snapshot import StagedConcept, TerminologySnapshot
from sqlalchemy import event, select

###############################################################################
def test_terminology_snapshot_uses_batched_set_based_writes(persistence_engine) -> None:  # type: ignore[no-untyped-def]
    statements: Counter[str] = Counter()

    def before_cursor_execute(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        normalized = statement.upper()
        for table in ("DRUG_RXNORM_CODES", "DRUG_ALIASES"):
            if f"INSERT INTO {table}" in normalized:
                statements[table] += 1

    graph = build_repository_graph(engine=persistence_engine)
    repository = TerminologyRepository(graph.context)
    event.listen(persistence_engine, "before_cursor_execute", before_cursor_execute)
    try:
        _persist(repository, _large_rxnorm_snapshot())
    finally:
        event.remove(persistence_engine, "before_cursor_execute", before_cursor_execute)

    with graph.context.session_factory() as db_session:
        assert len(db_session.execute(select(Drug)).scalars().all()) == 350
        assert len(db_session.execute(select(DrugRxnormCode)).scalars().all()) == 350
        assert len(db_session.execute(select(DrugAlias)).scalars().all()) == 350 * 3
    # Rows are written in parameter-bounded batches, never one statement per row.
    assert 1 <= statements["DRUG_RXNORM_CODES"] <= 10
    assert 1 <= statements["DRUG_ALIASES"] <= 20

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
def test_rxnorm_refresh_replaces_only_rxnorm_owned_rows(
    persistence_engine,
) -> None:  # type: ignore[no-untyped-def]
    graph = build_repository_graph(engine=persistence_engine)
    repository = TerminologyRepository(graph.context)
    _persist(repository, _large_rxnorm_snapshot(size=2))
    with graph.context.session_factory() as db_session:
        first = db_session.scalar(select(Drug).where(Drug.canonical_name_norm == "repairdrug 0"))
        assert first is not None
        first_id = int(first.id)
    lookup = TerminologyLookupRepository(graph.context)
    lookup.record_alias(
        drug_id=first_id, alias="Reviewer alias", alias_norm="reviewer alias",
        alias_kind="synonym", source="curated", validation_status="confirmed",
    )
    lookup.record_alias(
        drug_id=first_id, alias="Runtime alias", alias_norm="runtime alias",
        alias_kind="synonym", source="rxnav_runtime", validation_status="verified",
    )

    replacement = _large_rxnorm_snapshot(size=1)
    replacement.concepts["60000"] = StagedConcept(key="60000", name="Drug Gamma", rxcui="60000")
    replacement.add_alias("60000", "Drug Gamma", "canonical")
    _persist(repository, replacement)

    with graph.context.session_factory() as db_session:
        dropped = db_session.scalar(select(Drug).where(Drug.canonical_name_norm == "repairdrug 1"))
        # Concepts are never deleted (clinical rows reference them); only
        # the source-owned codes and aliases are replaced.
        assert dropped is not None
        assert db_session.scalar(select(DrugRxnormCode).where(DrugRxnormCode.rxcui == "50001")) is None
        kept = {
            (row.alias_norm, row.source)
            for row in db_session.execute(select(DrugAlias).where(DrugAlias.drug_id == first_id)).scalars()
        }
        assert {("reviewer alias", "curated"), ("runtime alias", "rxnav_runtime")} <= kept
        assert db_session.scalar(select(Drug).where(Drug.canonical_name_norm == "drug gamma")) is not None

###############################################################################
def _large_rxnorm_snapshot(size: int = 350) -> TerminologySnapshot:
    snapshot = TerminologySnapshot(source="rxnorm", version=f"bulk-{size}", license="Public domain")
    for index in range(size):
        key = str(50000 + index)
        snapshot.concepts[key] = StagedConcept(key=key, name=f"RepairDrug {index}", rxcui=key)
        snapshot.add_alias(key, f"RepairDrug {index}", "canonical")
        snapshot.add_alias(key, f"RepairBrand {index}", "brand")
        snapshot.add_alias(key, f"Repair Synonym {index}", "synonym")
    return snapshot

###############################################################################
def _persist(repository: TerminologyRepository, snapshot: TerminologySnapshot) -> None:
    run_id = repository.start_run(source=snapshot.source, mode="full", license_text=snapshot.license)
    stats = repository.persist_snapshot(snapshot, run_id)
    repository.finish_run(run_id, status="completed", version=snapshot.version, stats=stats.to_dict())

###############################################################################
def test_session_drug_persistence_retains_terminology_and_source_provenance(
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
                    "terminology": {"status": "verified", "method": "exact_alias", "accepted": True},
                    "source_provenance": {
                        "terminology": {"status": "verified", "rxcui": "12345"},
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
        assert mention.match_status == "verified"
        assert mention.evidence_json["terminology"]["method"] == "exact_alias"
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
