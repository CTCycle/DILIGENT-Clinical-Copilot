from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from domain.clinical import DrugEntry, PatientDrugs
from repositories.schemas.models import (
    Base,
    Drug,
    DrugRxnormCode,
    KbMatchCache,
    LiverToxMonograph,
)
from repositories.serialization.data import DataSerializer
from services.clinical.drug_resolution import DrugResolutionService
from services.clinical.matches_core import LiverToxMatcher

###############################################################################
def _livertox_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nbk_id": "NBK001",
                "drug_name": "Acetaminophen",
                "excerpt": "Acetaminophen can cause dose-related liver injury.",
                "synonyms": "Tylenol; Paracetamol",
                "ingredient": "Acetaminophen",
                "brand_name": "Tylenol",
                "monograph_key": "acetaminophen",
            },
            {
                "nbk_id": "NBK002",
                "drug_name": "Amoxicillin-Clavulanate",
                "excerpt": "Amoxicillin-clavulanate is a common cause of cholestatic DILI.",
                "synonyms": "Co-amoxi; Amoxicillin clavulanate",
                "ingredient": "Amoxicillin Clavulanate",
                "brand_name": "Co-amoxi",
                "monograph_key": "amoxicillin-clavulanate",
            },
            {
                "nbk_id": "NBK003",
                "drug_name": "Amoxicillin",
                "excerpt": "Amoxicillin monograph.",
                "synonyms": "",
                "ingredient": "Amoxicillin",
                "brand_name": "",
                "monograph_key": "amoxicillin",
            },
            {
                "nbk_id": "NBK004",
                "drug_name": "Clavulanate",
                "excerpt": "Clavulanate monograph.",
                "synonyms": "",
                "ingredient": "Clavulanate",
                "brand_name": "",
                "monograph_key": "clavulanate",
            },
            {
                "nbk_id": "NBK005",
                "drug_name": "Metformin",
                "excerpt": "Metformin monograph.",
                "synonyms": "SharedAlias",
                "ingredient": "Metformin",
                "brand_name": "",
                "monograph_key": "metformin",
            },
            {
                "nbk_id": "NBK006",
                "drug_name": "Metformin XR",
                "excerpt": "Metformin XR monograph.",
                "synonyms": "SharedAlias",
                "ingredient": "Metformin",
                "brand_name": "",
                "monograph_key": "metformin-xr",
            },
            {
                "nbk_id": "NBK007",
                "drug_name": "Abiraterone",
                "excerpt": "Abiraterone monograph.",
                "synonyms": "",
                "ingredient": "Abiraterone",
                "brand_name": "Zytiga",
                "monograph_key": "abiraterone",
            },
            {
                "nbk_id": "NBK008",
                "drug_name": "Corticosteroids",
                "excerpt": "Corticosteroids monograph.",
                "synonyms": "",
                "ingredient": "Prednisone; Prednisolone",
                "brand_name": "Deltasone",
                "monograph_key": "corticosteroids",
            },
            {
                "nbk_id": "NBK009",
                "drug_name": "Tamsulosin",
                "excerpt": "Tamsulosin monograph.",
                "synonyms": "",
                "ingredient": "Tamsulosin",
                "brand_name": "Flomax",
                "monograph_key": "tamsulosin",
            },
            {
                "nbk_id": "NBK010",
                "drug_name": "Vitamin D",
                "excerpt": "Vitamin D monograph.",
                "synonyms": "",
                "ingredient": "Vitamin D; Calcitriol",
                "brand_name": "Citrical plus D",
                "monograph_key": "vitamin-d",
            },
            {
                "nbk_id": "NBK011",
                "drug_name": "Leuprolide",
                "excerpt": "Leuprolide monograph.",
                "synonyms": "",
                "ingredient": "Leuprolide",
                "brand_name": "Eligard",
                "monograph_key": "leuprolide",
            },
        ]
    )

###############################################################################
def _catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rxcui": "161",
                "term_type": "IN",
                "raw_name": "Acetaminophen",
                "name": "Acetaminophen",
                "brand_names": "Tylenol",
                "synonyms": '["Tylenol", "Paracetamol"]',
            },
            {
                "rxcui": "999",
                "term_type": "SBD",
                "raw_name": "amoxicillin clavulanate Oral Tablet",
                "name": "amoxicillin clavulanate",
                "brand_names": "Co-amoxi",
                "synonyms": '["Co-amoxi", "Amoxicillin Clavulanate"]',
            },
            {
                "rxcui": "1007",
                "term_type": "SBD",
                "raw_name": "Abirateron Sandoz",
                "name": "Abiraterone",
                "brand_names": "Abirateron Sandoz",
                "synonyms": '["Abirateron"]',
            },
            {
                "rxcui": "1008",
                "term_type": "SBD",
                "raw_name": "Prednison Spirig HC",
                "name": "Prednisone",
                "brand_names": "Prednison Spirig HC",
                "synonyms": '["Prednison"]',
            },
            {
                "rxcui": "1009",
                "term_type": "SBD",
                "raw_name": "Pradif T",
                "name": "Tamsulosin",
                "brand_names": "Pradif T",
                "synonyms": "[]",
            },
            {
                "rxcui": "1010",
                "term_type": "SBD",
                "raw_name": "Vi-De3 4500 IU/ml",
                "name": "Vitamin D",
                "brand_names": "Vi-De3",
                "synonyms": "[]",
            },
            {
                "rxcui": "1011",
                "term_type": "SBD",
                "raw_name": "Calcimagon D3",
                "name": "Vitamin D",
                "brand_names": "Calcimagon D3; Calcimagon",
                "synonyms": "[]",
            },
            {
                "rxcui": "1012",
                "term_type": "IN",
                "raw_name": "Leuprorelina",
                "name": "Leuprolide",
                "brand_names": "",
                "synonyms": '["Leuprorelina", "Leuprorelin"]',
            },
        ]
    )

###############################################################################
def _resolver() -> DrugResolutionService:
    return DrugResolutionService(
        LiverToxMatcher(_livertox_frame(), drugs_catalog_df=_catalog_frame())
    )

###############################################################################
def _resolve_one(name: str) -> dict:
    resolved = _resolver().resolve(PatientDrugs(entries=[DrugEntry(name=name)]))
    assert resolved
    return next(iter(resolved.values()))

###############################################################################
def test_exact_generic_drug_gets_resolution_decision() -> None:
    payload = _resolve_one("Acetaminophen")

    assert payload["decision_status"] == "accepted_rxnav_validated"
    assert payload["accepted_rxnav_rxcui"] == "161"
    assert payload["accepted_livertox_nbk_id"] == "NBK001"
    assert payload["resolution_decision"]["accepted_livertox_name"] == "Acetaminophen"

###############################################################################
def test_brand_alias_maps_to_rxcui_and_livertox_monograph() -> None:
    payload = _resolve_one("Tylenol")

    assert payload["decision_status"] == "accepted_rxnav_validated"
    assert payload["rxnav_validation_status"] in {"brand_to_rxcui", "alias_to_rxcui"}
    assert payload["accepted_livertox_name"] == "Acetaminophen"

###############################################################################
def test_catalog_backed_product_labels_resolve_to_livertox_monographs_with_excerpts() -> (
    None
):
    expected = {
        "Abirateron Sandoz": "Abiraterone",
        "Prednison Spirig HC": "Corticosteroids",
        "Pradif T": "Tamsulosin",
        "Vi-De3 4'500 IU/ml": "Vitamin D",
        "Calcimagon D3": "Vitamin D",
        "leuprorelina": "Leuprolide",
    }

    for label, accepted_name in expected.items():
        payload = _resolve_one(label)

        assert payload["accepted_livertox_name"] == accepted_name
        assert payload["missing_livertox"] is False
        assert payload["extracted_excerpts"]

###############################################################################
def test_opaque_product_label_is_not_guessed_without_source_backing() -> None:
    payload = _resolve_one("Unmapped Pharma Brand")

    assert payload["accepted_livertox_name"] is None
    assert payload["missing_livertox"] is True

###############################################################################
def test_combination_product_preserves_parent_and_components() -> None:
    resolved = _resolver().resolve(
        PatientDrugs(entries=[DrugEntry(name="Amoxicillin-Clavulanate")])
    )

    parent = resolved["amoxicillin clavulanate"]
    assert parent["decision_status"] == "accepted_rxnav_validated"
    assert parent["accepted_livertox_name"] == "Amoxicillin-Clavulanate"
    assert set(parent["regimen_components"]) == {"amoxicillin", "clavulanate"}
    assert any(
        item["accepted_livertox_name"] == "Amoxicillin" for item in resolved.values()
    )

###############################################################################
def test_ambiguous_alias_requires_review() -> None:
    payload = _resolve_one("SharedAlias")

    assert payload["decision_status"] == "ambiguous_requires_review"
    assert payload["requires_human_review"] is True
    assert payload["accepted_livertox_name"] is None

###############################################################################
def test_broad_category_is_not_concrete_drug_match() -> None:
    payload = _resolve_one("vitamins")

    assert payload["decision_status"] in {
        "ambiguous_requires_review",
        "missing_livertox",
    }
    assert payload["accepted_rxnav_rxcui"] is None
    assert payload["accepted_livertox_name"] is None

###############################################################################
def test_false_positive_lab_text_is_rejected_before_matching() -> None:
    payload = _resolve_one("ALT")

    assert payload["decision_status"] == "rejected_false_positive"
    assert payload["missing_livertox"] is True

###############################################################################
def test_candidates_from_another_extracted_drug_do_not_contaminate_mention() -> None:
    resolved = _resolver().resolve(
        PatientDrugs(
            entries=[
                DrugEntry(name="Levothyroxine sodium"),
                DrugEntry(name="Diazepam"),
            ]
        )
    )

    levothyroxine = resolved["levothyroxine sodium"]
    candidate_names = {
        candidate["normalized_name"]
        for candidate in levothyroxine["livertox_candidates"]
    }

    assert "diazepam" not in candidate_names
    assert "diazepam oral" not in candidate_names


###############################################################################
# DB match-cache integration tests

###############################################################################
def _build_cache_db() -> tuple[DataSerializer, Drug, LiverToxMonograph]:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    serializer = DataSerializer(engine=engine)
    factory = sessionmaker(bind=engine, future=True)
    with factory() as session:
        drug = Drug(
            canonical_name="Acetaminophen",
            canonical_name_norm="acetaminophen",
            livertox_nbk_id="NBK0001",
        )
        session.add(drug)
        session.flush()

        monograph = LiverToxMonograph(
            drug_id=int(drug.id),
            monograph_key="test_monograph_001",
            drug_name_norm="acetaminophen",
            nbk_id="NBK0001",
            excerpt="Cached Acetaminophen excerpt.",
        )
        session.add(monograph)
        session.flush()

        cache = KbMatchCache(
            raw_drug_name="Acetaminophen",
            raw_drug_name_norm="acetaminophen",
            normalized_drug_key="acetaminophen",
            drug_id=int(drug.id),
            livertox_monograph_key="test_monograph_001",
            livertox_nbk_id="NBK0001",
            source="livertox",
            confidence=0.96,
            evidence_json="{}",
        )
        session.add(cache)
        session.commit()
    return serializer, drug, monograph

###############################################################################
def test_db_cache_hit_returns_cached_result() -> None:
    serializer, drug, monograph = _build_cache_db()
    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0001",
                "drug_name": "Acetaminophen",
                "excerpt": "Fresh Acetaminophen excerpt.",
                "synonyms": "Tylenol; Paracetamol",
                "ingredient": "Acetaminophen",
                "brand_name": "Tylenol",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)
    cache_lookup = lambda key: serializer.load_livertox_match_from_db_cache(
        normalized_drug_key=key,
    )
    resolver = DrugResolutionService(
        matcher,
        cache_lookup=cache_lookup,
    )
    resolved = resolver.resolve(PatientDrugs(entries=[DrugEntry(name="Acetaminophen")]))
    assert resolved
    payload = next(iter(resolved.values()))

    assert "cache_hit_previous_match" in payload.get("match_reason", "")
    assert payload["accepted_livertox_name"] == "Acetaminophen"
    assert payload["accepted_livertox_nbk_id"] == "NBK0001"
    assert payload["missing_livertox"] is False
    assert payload["extracted_excerpts"] == ["Cached Acetaminophen excerpt."]
    assert payload.get("decision_status") in {
        "accepted_exact_livertox",
        "accepted_livertox_without_rxnav",
    }

###############################################################################
def test_db_cache_miss_falls_through_to_pipeline() -> None:
    serializer, drug, monograph = _build_cache_db()
    factory = sessionmaker(
        bind=serializer.session_factory.kw["bind"],
        future=True,
    )
    with factory() as session:
        session.query(KbMatchCache).delete()
        session.commit()

    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0001",
                "drug_name": "Acetaminophen",
                "excerpt": "Fresh Acetaminophen excerpt.",
                "synonyms": "Tylenol; Paracetamol",
                "ingredient": "Acetaminophen",
                "brand_name": "Tylenol",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)
    cache_lookup = lambda key: serializer.load_livertox_match_from_db_cache(
        normalized_drug_key=key,
    )
    resolver = DrugResolutionService(
        matcher,
        cache_lookup=cache_lookup,
    )
    resolved = resolver.resolve(PatientDrugs(entries=[DrugEntry(name="Acetaminophen")]))
    assert resolved
    payload = next(iter(resolved.values()))

    assert "cache_hit_previous_match" not in payload.get("match_reason", "")
    assert payload["accepted_livertox_name"] == "Acetaminophen"
    assert payload["missing_livertox"] is False

###############################################################################
def test_db_cache_low_confidence_not_used() -> None:
    serializer, drug, monograph = _build_cache_db()
    factory = sessionmaker(
        bind=serializer.session_factory.kw["bind"],
        future=True,
    )
    with factory() as session:
        existing = session.query(KbMatchCache).first()
        existing.confidence = 0.70
        session.commit()

    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0001",
                "drug_name": "Acetaminophen",
                "excerpt": "Fresh Acetaminophen excerpt.",
                "synonyms": "Tylenol; Paracetamol",
                "ingredient": "Acetaminophen",
                "brand_name": "Tylenol",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)
    cache_lookup = lambda key: serializer.load_livertox_match_from_db_cache(
        normalized_drug_key=key,
    )
    resolver = DrugResolutionService(
        matcher,
        cache_lookup=cache_lookup,
    )
    resolved = resolver.resolve(PatientDrugs(entries=[DrugEntry(name="Acetaminophen")]))
    assert resolved
    payload = next(iter(resolved.values()))

    assert "cache_hit_previous_match" not in payload.get("match_reason", "")
    assert payload["accepted_livertox_name"] == "Acetaminophen"
    assert payload["missing_livertox"] is False
    assert payload["extracted_excerpts"] == ["Fresh Acetaminophen excerpt."]

###############################################################################
def test_db_cache_uses_previously_resolved_rxcui() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    serializer = DataSerializer(engine=engine)
    factory = sessionmaker(bind=engine, future=True)
    with factory() as session:
        drug = Drug(
            canonical_name="Omeprazole",
            canonical_name_norm="omeprazole",
            livertox_nbk_id="NBK0002",
        )
        session.add(drug)
        session.flush()

        monograph = LiverToxMonograph(
            drug_id=int(drug.id),
            monograph_key="test_monograph_002",
            drug_name_norm="omeprazole",
            nbk_id="NBK0002",
            excerpt="Cached Omeprazole excerpt.",
        )
        session.add(monograph)
        session.flush()

        cache = KbMatchCache(
            raw_drug_name="Omeprazole",
            raw_drug_name_norm="omeprazole",
            normalized_drug_key="omeprazole",
            drug_id=int(drug.id),
            rxnorm_rxcui="7646",
            livertox_monograph_key="test_monograph_002",
            livertox_nbk_id="NBK0002",
            source="rxnav",
            confidence=0.98,
            evidence_json='{"match_reason": "rxnorm_direct"}',
        )
        session.add(
            DrugRxnormCode(
                drug_id=int(drug.id),
                rxcui="7646",
            )
        )
        session.add(cache)
        session.commit()

    frame = pd.DataFrame(
        [
            {
                "nbk_id": "NBK0002",
                "drug_name": "Omeprazole",
                "excerpt": "Fresh Omeprazole excerpt.",
                "synonyms": "Losec",
                "ingredient": "Omeprazole",
                "brand_name": "Losec",
            },
        ]
    )
    matcher = LiverToxMatcher(frame)
    cache_lookup = lambda key: serializer.load_livertox_match_from_db_cache(
        normalized_drug_key=key,
    )
    resolver = DrugResolutionService(
        matcher,
        cache_lookup=cache_lookup,
    )
    resolved = resolver.resolve(PatientDrugs(entries=[DrugEntry(name="Omeprazole")]))
    assert resolved
    payload = next(iter(resolved.values()))

    assert "cache_hit_previous_match" in payload.get("match_reason", "")
    assert payload["decision_status"] == "accepted_rxnav_validated"
    assert payload["accepted_rxnav_rxcui"] == "7646"
    assert payload["accepted_livertox_nbk_id"] == "NBK0002"
