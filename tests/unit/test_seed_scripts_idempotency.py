from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

try:
    from DILIGENT.server.repositories.serialization.data import DataSerializer
    from DILIGENT.server.repositories.schemas.models import (
        Base,
        Drug,
        DrugAlias,
        LiverToxMonograph,
    )
except ModuleNotFoundError:
    DataSerializer = None  # type: ignore[assignment]
    Base = None  # type: ignore[assignment]
    Drug = None  # type: ignore[assignment]
    DrugAlias = None  # type: ignore[assignment]
    LiverToxMonograph = None  # type: ignore[assignment]

try:
    from DILIGENT.server.services.updater.livertox import LiverToxUpdater
except ModuleNotFoundError:
    LiverToxUpdater = None  # type: ignore[assignment]


###############################################################################
class QueryStub:
    def __init__(self, engine: Any) -> None:
        self.database = SimpleNamespace(backend=SimpleNamespace(engine=engine))


# -----------------------------------------------------------------------------
def build_serializer() -> tuple[Any, Any]:
    if DataSerializer is None or Base is None:
        raise RuntimeError("Serialization dependencies are not available")
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    serializer = DataSerializer(queries=QueryStub(engine))
    return serializer, engine


# -----------------------------------------------------------------------------
def fetch_counts(engine: Any) -> tuple[int, int, int]:
    factory = sessionmaker(bind=engine, future=True)
    with factory() as db_session:
        drugs = len(db_session.execute(select(Drug)).scalars().all())
        aliases = len(db_session.execute(select(DrugAlias)).scalars().all())
        monographs = len(db_session.execute(select(LiverToxMonograph)).scalars().all())
    return drugs, aliases, monographs


# -----------------------------------------------------------------------------
@pytest.mark.skipif(DataSerializer is None, reason="Serialization optional dependencies missing")
def test_rxnav_upsert_idempotent_twice() -> None:
    serializer, engine = build_serializer()
    payload = [
        {
            "rxcui": "100",
            "raw_name": "Acetaminophen [Tylenol] 500 MG Tablet",
            "term_type": "SCD",
            "name": "Acetaminophen",
            "brand_names": "Tylenol",
            "synonyms": '["Paracetamol"]',
        },
        {
            "rxcui": "200",
            "raw_name": "Ibuprofen 200 MG Tablet",
            "term_type": "SCD",
            "name": "Ibuprofen",
            "brand_names": "Advil",
            "synonyms": '["Advil"]',
        },
    ]

    serializer.upsert_drugs_catalog_records(payload)
    first_counts = fetch_counts(engine)
    serializer.upsert_drugs_catalog_records(payload)
    second_counts = fetch_counts(engine)

    assert first_counts == second_counts


# -----------------------------------------------------------------------------
@pytest.mark.skipif(DataSerializer is None, reason="Serialization optional dependencies missing")
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

    serializer.save_livertox_records(frame)
    first_counts = fetch_counts(engine)
    serializer.save_livertox_records(frame)
    second_counts = fetch_counts(engine)

    assert first_counts == second_counts


# -----------------------------------------------------------------------------
@pytest.mark.skipif(
    DataSerializer is None or LiverToxUpdater is None,
    reason="Serialization/updater optional dependencies missing",
)
def test_livertox_duplicate_nbk_is_nulled() -> None:
    serializer, _ = build_serializer()
    updater = LiverToxUpdater(sources_path=".", redownload=False, serializer=serializer)
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

    finalized = updater.finalize_dataset(frame)

    assert finalized["nbk_id"].isna().all()


# -----------------------------------------------------------------------------
@pytest.mark.skipif(DataSerializer is None, reason="Serialization optional dependencies missing")
def test_livertox_does_not_match_by_nbk() -> None:
    serializer, engine = build_serializer()
    factory = sessionmaker(bind=engine, future=True)
    with factory() as db_session:
        existing = serializer.ensure_drug(
            db_session,
            canonical_name="Drug Alpha",
            canonical_name_norm="drug alpha",
            rxnorm_rxcui=None,
            livertox_nbk_id="NBK999999",
        )
        existing_id = int(existing.id)
        db_session.commit()

    with factory() as db_session:
        created = serializer.ensure_drug(
            db_session,
            canonical_name="Drug Beta",
            canonical_name_norm="drug beta",
            rxnorm_rxcui=None,
            livertox_nbk_id="NBK999999",
            use_livertox_nbk_lookup=False,
        )
        db_session.commit()
        assert int(created.id) != existing_id

    with factory() as db_session:
        drugs = db_session.execute(select(Drug).order_by(Drug.id)).scalars().all()
        assert len(drugs) == 2
        assert drugs[0].livertox_nbk_id == "NBK999999"
        assert drugs[1].livertox_nbk_id is None


# -----------------------------------------------------------------------------
@pytest.mark.skipif(DataSerializer is None, reason="Serialization optional dependencies missing")
def test_ensure_drug_conflict_raises() -> None:
    serializer, engine = build_serializer()
    factory = sessionmaker(bind=engine, future=True)

    with factory() as db_session:
        serializer.ensure_drug(
            db_session,
            canonical_name="Drug One",
            canonical_name_norm="drug one",
            rxnorm_rxcui="111",
            livertox_nbk_id=None,
        )
        serializer.ensure_drug(
            db_session,
            canonical_name="Drug Two",
            canonical_name_norm="drug two",
            rxnorm_rxcui=None,
            livertox_nbk_id=None,
        )
        db_session.commit()

    with factory() as db_session:
        with pytest.raises(RuntimeError):
            serializer.ensure_drug(
                db_session,
                canonical_name="Drug Two",
                canonical_name_norm="drug two",
                rxnorm_rxcui="111",
                livertox_nbk_id=None,
            )
        db_session.rollback()

    with factory() as db_session:
        rows = db_session.execute(select(Drug)).scalars().all()
        assert len(rows) == 2
