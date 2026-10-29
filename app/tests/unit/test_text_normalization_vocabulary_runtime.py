# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from types import SimpleNamespace

from common.utils.seed_terms import detect_seed_matches, load_seed_term_catalog
from repositories.schemas.base import Base
from repositories.schemas.configuration import ReferenceCatalogEntry
from repositories.schemas.knowledge import DrugAlias
from repository_fixtures import build_repository_graph
from services.clinical.preparation import ClinicalKnowledgePreparation
from services.text import vocabulary as vocabulary_module
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

###############################################################################
def test_runtime_upsert_list_and_deactivate_term() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    vocabulary_module.get_text_normalization_snapshot.cache_clear()
    vocabulary_module.get_default_repository = lambda: SimpleNamespace(  # type: ignore[method-assign]
        engine=engine, session_factory=factory
    )
    payload = vocabulary_module.upsert_text_normalization_term_payload(
        category="brand_combo_preference",
        term="Bactrim",
        replacement="trimethoprim sulfamethoxazole",
        source="runtime",
        is_active=True,
    )
    assert payload["replacement"] == "trimethoprim sulfamethoxazole"
    rows = vocabulary_module.list_text_normalization_term_payloads(
        category="brand_combo_preference"
    )
    assert len(rows) == 1
    assert rows[0]["is_active"] is True
    updated = vocabulary_module.deactivate_text_normalization_term_payload(
        category="brand_combo_preference",
        term="Bactrim",
    )
    assert updated is True
    rows = vocabulary_module.list_text_normalization_term_payloads(
        category="brand_combo_preference"
    )
    assert rows[0]["is_active"] is False

###############################################################################
def test_seed_mapping_categories_are_loaded() -> None:
    # Smoke check: default snapshot exposes the new mapping fields.
    snapshot = vocabulary_module.get_text_normalization_snapshot()
    assert isinstance(snapshot.lab_marker_aliases, dict)
    assert isinstance(snapshot.knowledge_source_references, dict)

###############################################################################
def test_detects_keywords_and_stopwords_from_catalog() -> None:
    catalog = load_seed_term_catalog()
    matches = detect_seed_matches(
        "Patient therapy with tablets and mg dosage.", catalog
    )
    assert "therapy" in matches["matched_keywords"]
    assert (
        "tablets" in matches["matched_stopwords"]
        or "mg" in matches["matched_stopwords"]
    )
    assert isinstance(matches["matched_term_counts"], dict)

###############################################################################
def test_session_mentions_record_observations_but_never_aliases() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    graph = build_repository_graph(engine=engine, session_factory=factory)
    preparation = ClinicalKnowledgePreparation(
        knowledge_repository=graph.knowledge_repository,
    )

    matched_drugs = [
        {"raw_drug_name": "Atorvastatina", "terminology": {"accepted": True}},
        {"raw_drug_name": "Unknown Herb", "terminology": {"accepted": False}},
        {"raw_drug_name": "Mystery Tonic"},
    ]
    assert preparation.record_drug_mention_observations(matched_drugs) is True

    with factory() as db_session:
        aliases = db_session.execute(select(DrugAlias)).scalars().all()
        terms = db_session.execute(
            select(ReferenceCatalogEntry.category, ReferenceCatalogEntry.value).where(
                ReferenceCatalogEntry.manifest == "runtime_observations"
            )
        ).all()

    assert aliases == []
    assert sorted(terms) == [
        ("observed_unresolved_query", "Mystery Tonic"),
        ("observed_unresolved_query", "Unknown Herb"),
    ]
