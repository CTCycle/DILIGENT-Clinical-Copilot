# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any

import pandas as pd
from repositories.schemas.base import Base
from repositories.schemas.knowledge import (
    Drug,
    DrugAlias,
    DrugRxnormCode,
    LiverToxMonograph,
)
from repositories.schemas.configuration import ReferenceCatalogEntry
from repository_fixtures import build_repository_graph
from services.text.vocabulary import record_text_normalization_observation
from services.updater import livertox_index
from services.updater.livertox_core import LiverToxUpdater
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

###############################################################################
def build_serializer() -> tuple[Any, Any]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return build_repository_graph(engine=engine), engine

###############################################################################
def fetch_counts(engine: Any) -> tuple[int, int, int, int]:
    factory = sessionmaker(bind=engine, future=True)
    with factory() as db_session:
        drugs = len(db_session.execute(select(Drug)).scalars().all())
        rxcui_codes = len(db_session.execute(select(DrugRxnormCode)).scalars().all())
        aliases = len(db_session.execute(select(DrugAlias)).scalars().all())
        monographs = len(db_session.execute(select(LiverToxMonograph)).scalars().all())
    return drugs, rxcui_codes, aliases, monographs

###############################################################################
def test_livertox_upsert_idempotent_twice() -> None:
    serializer, engine = build_serializer()
    frame = pd.DataFrame(
        [
            {
                "drug_name": "Acetaminophen",
                "nbk_id": pd.NA,
                "ingredient": "Acetaminophen",
                "brand_name": "Tylenol",
                "synonyms": "Paracetamol",
                "excerpt": "LiverTox excerpt",
                "likelihood_score": "A",
                "last_update": "2025-01-01",
                "reference_count": 10,
                "year_approved": 1955,
                "agent_classification": "Drug",
                "primary_classification": "Analgesic",
                "secondary_classification": "Acetanilide",
                "include_in_livertox": "yes",
                "source_url": "https://example.test/livertox",
                "source_last_modified": "2025-01-01",
            }
        ]
    )

    serializer.knowledge_repository.save_livertox_records(frame)
    first_counts = fetch_counts(engine)
    serializer.knowledge_repository.save_livertox_records(frame)
    second_counts = fetch_counts(engine)

    assert first_counts == second_counts

###############################################################################
def test_livertox_duplicate_nbk_is_nulled() -> None:
    serializer, _ = build_serializer()
    updater = LiverToxUpdater(
        sources_path=".",
        redownload=False,
        knowledge_repository=serializer.knowledge_repository,
    )
    frame = pd.DataFrame(
        [
            {
                "drug_name": "Drug Alpha",
                "nbk_id": "NBK547852",
                "ingredient": pd.NA,
                "brand_name": pd.NA,
                "synonyms": pd.NA,
                "excerpt": "Excerpt A",
                "source_url": "https://example.test/a",
                "source_last_modified": "2025-01-01",
                "last_update": "2025-01-01",
            },
            {
                "drug_name": "Drug Beta",
                "nbk_id": "NBK547852",
                "ingredient": pd.NA,
                "brand_name": pd.NA,
                "synonyms": pd.NA,
                "excerpt": "Excerpt B",
                "source_url": "https://example.test/b",
                "source_last_modified": "2025-01-02",
                "last_update": "2025-01-02",
            },
        ]
    )

    finalized = livertox_index.finalize_dataset(updater, frame)

    assert finalized["nbk_id"].isna().all()

###############################################################################
def test_text_normalization_runtime_observation_writes_only_runtime_manifest() -> None:
    _, engine = build_serializer()
    factory = sessionmaker(bind=engine, future=True)
    from services.text import vocabulary as vocabulary_module

    vocabulary_module.get_default_repository = lambda: type(  # type: ignore[method-assign]
        "Repo",
        (),
        {"engine": engine, "session_factory": factory},
    )()
    record_text_normalization_observation(
        "Bactrim",
        category="observed_unresolved_query",
    )

    with factory() as db_session:
        entry_count = db_session.execute(
            select(func.count()).select_from(ReferenceCatalogEntry)
        ).scalar_one()
        runtime_observations = (
            db_session.execute(
                select(ReferenceCatalogEntry).where(
                    ReferenceCatalogEntry.manifest == "runtime_observations",
                    ReferenceCatalogEntry.domain == "text_normalization",
                )
            )
            .scalars()
            .all()
        )

    assert int(entry_count) == 1
    assert len(runtime_observations) == 1
