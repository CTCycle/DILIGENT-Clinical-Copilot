from __future__ import annotations

from types import SimpleNamespace

from common.utils.seed_terms import detect_seed_matches, load_seed_term_catalog
from repositories.schemas.base import Base
from repositories.schemas.clinical import ClinicalSession
from repositories.schemas.configuration import ReferenceCatalogEntry
from repositories.schemas.knowledge import (
    Drug,
    DrugAlias,
)
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
    assert isinstance(snapshot.brand_combo_preferences, dict)
    assert isinstance(snapshot.knowledge_source_references, dict)

###############################################################################
def test_detects_keywords_and_stopwords_from_catalog() -> None:
    catalog = load_seed_term_catalog()
    matches = detect_seed_matches(
        "Patient uses Bactrim tablets and mg dosage.", catalog
    )
    assert "bactrim" in matches["matched_keywords"]
    assert (
        "tablets" in matches["matched_stopwords"]
        or "mg" in matches["matched_stopwords"]
    )
    assert isinstance(matches["matched_term_counts"], dict)

###############################################################################
def test_session_learning_promotes_only_direct_high_confidence_aliases() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    graph = build_repository_graph(engine=engine, session_factory=factory)
    preparation = ClinicalKnowledgePreparation(
        knowledge_repository=graph.knowledge_repository,
        drug_catalog_repository=graph.drug_catalog_repository,
    )

    db_session = factory()
    try:
        atorvastatin = Drug(
            canonical_name="atorvastatin",
            canonical_name_norm="atorvastatin",
            livertox_nbk_id=None,
        )
        loop_diuretics = Drug(
            canonical_name="Loop Diuretics",
            canonical_name_norm="loop diuretics",
            livertox_nbk_id=None,
        )
        db_session.add_all([atorvastatin, loop_diuretics])
        db_session.flush()
        session = ClinicalSession(
            patient_name="Test", session_status="successful"
        )
        db_session.add(session)
        db_session.commit()
        session_id = int(session.id)
    finally:
        db_session.close()

    matched_drugs = [
        {
            "raw_drug_name": "Atorvastatina",
            "matched_drug_name": "atorvastatin",
            "match_reason": "exact_canonical",
            "match_confidence": 1.0,
        },
        {
            "raw_drug_name": "Furosemide",
            "matched_drug_name": "Loop Diuretics",
            "match_reason": "exact_alias_ranked",
            "match_confidence": 0.92,
        },
        {
            "raw_drug_name": "Unknown Herb",
            "matched_drug_name": None,
            "match_reason": "no_match",
            "match_confidence": None,
        },
    ]
    resolved_drugs = preparation.resolve_session_drug_ids(matched_drugs)
    assert preparation.learn_session_drug_mentions(session_id, resolved_drugs) is True

    db_session = factory()
    try:
        aliases = db_session.execute(
            select(DrugAlias.alias, DrugAlias.drug_id).where(
                DrugAlias.source == "session"
            )
        ).all()
        terms = db_session.execute(
            select(ReferenceCatalogEntry.category, ReferenceCatalogEntry.value).where(
                ReferenceCatalogEntry.manifest == "runtime_observations"
            )
        ).all()
    finally:
        db_session.close()

    assert aliases == [("Atorvastatina", int(atorvastatin.id))]
    assert sorted(terms) == [
        ("observed_unpromoted_query", "Furosemide"),
        ("observed_unresolved_query", "Unknown Herb"),
    ]
