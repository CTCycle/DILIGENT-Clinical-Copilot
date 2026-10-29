# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from sqlalchemy import delete, select

import services.clinical.preparation as preparation_module
from domain.clinical import DrugEntry, PatientDrugs
from repositories.context import RepositoryContext
from repositories.knowledge_repository import KnowledgeRepository
from repositories.schemas.knowledge import Drug
from repositories.schemas.terminology import DrugLiverToxLink
from repositories.terminology_repository import TerminologyRepository
from services.clinical.preparation import ClinicalKnowledgePreparation
from services.terminology.index import TerminologyIndexCache
from services.terminology.resolver import ResolverSettings, TerminologyResolver
from terminology_fixtures import build_context, build_sqlite_engine, seed_reference_terminology


###############################################################################
@pytest.fixture(scope="module")
def seeded_context(tmp_path_factory) -> RepositoryContext:  # type: ignore[no-untyped-def]
    context = build_context(build_sqlite_engine(tmp_path_factory.mktemp("clinical") / "t.db"))
    seed_reference_terminology(context)
    return context


###############################################################################
@pytest.fixture
def preparation(seeded_context: RepositoryContext, monkeypatch: pytest.MonkeyPatch) -> ClinicalKnowledgePreparation:
    # Local-only resolution with an index private to this database.
    monkeypatch.setattr(
        preparation_module,
        "build_terminology_resolver",
        lambda lookup: TerminologyResolver(lookup, TerminologyIndexCache(), settings=ResolverSettings()),
    )
    return ClinicalKnowledgePreparation(knowledge_repository=KnowledgeRepository(seeded_context))


# -----------------------------------------------------------------------------
def prepare(
    preparation: ClinicalKnowledgePreparation,
    names: list[str],
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    prepared = asyncio.run(
        preparation.prepare_inputs(
            PatientDrugs(entries=[DrugEntry(name=name, source="therapy") for name in names]),
            clinical_context="",
            pattern_score=None,
            **kwargs,
        )
    )
    assert prepared is not None
    return prepared.resolved_drugs


# -----------------------------------------------------------------------------
def by_mention(resolved: dict[str, dict[str, Any]], mention: str) -> dict[str, Any]:
    matches = [payload for payload in resolved.values() if mention in payload["raw_mentions"]]
    assert matches, f"no payload for {mention}"
    return min(matches, key=lambda payload: len(payload.get("regimen_components") or []) == 0)


###############################################################################
def test_prepare_inputs_links_verified_identities_to_their_monographs(
    preparation: ClinicalKnowledgePreparation,
) -> None:
    resolved = prepare(preparation, ["Lipitor", "Dafalgan", "HCTZ", "metformin 500mg", "Bactrim"])

    lipitor = by_mention(resolved, "Lipitor")
    assert (lipitor["decision_status"], lipitor["accepted_livertox_name"]) == ("accepted_verified", "atorvastatin")
    assert lipitor["terminology"]["matched_alias"] == "Lipitor"
    assert "Atorvastatin LiverTox excerpt." in lipitor["knowledge_prompt"]
    dafalgan = by_mention(resolved, "Dafalgan")
    assert (dafalgan["drug_name"], dafalgan["terminology"]["source"]) == ("acetaminophen", "swissmedic")
    # A verified identity without a monograph stays explicit instead of guessing.
    for name in ("HCTZ", "metformin 500mg"):
        assert by_mention(resolved, name)["decision_status"] == "missing_livertox"
    # The combination monograph is reached through its component set.
    bactrim = by_mention(resolved, "Bactrim")
    assert bactrim["accepted_livertox_name"] == "Trimethoprim-Sulfamethoxazole"
    assert bactrim["livertox_candidates"][0]["link_type"] == "component_set"


###############################################################################
def test_combination_without_combination_monograph_keeps_components(
    preparation: ClinicalKnowledgePreparation,
    seeded_context: RepositoryContext,
) -> None:
    # Remove the combination monograph link so only components could apply.
    with seeded_context.session_factory() as db_session:
        combo = db_session.execute(
            select(Drug.id).where(Drug.canonical_name == "amoxicillin / clavulanate")
        ).scalar_one()
        db_session.execute(delete(DrugLiverToxLink).where(DrugLiverToxLink.drug_id == combo))
        db_session.commit()
    try:
        resolved = prepare(preparation, ["Augmentin"])
    finally:
        TerminologyRepository(seeded_context).rebuild_livertox_links()

    payloads = [payload for payload in resolved.values() if "Augmentin" in payload["raw_mentions"]]
    components = {payload["drug_name"] for payload in payloads if not payload["is_regimen_parent"]}
    assert components == {"amoxicillin", "clavulanate"}
    assert all(payload["regimen_components"] == ["amoxicillin", "clavulanate"] for payload in payloads)


###############################################################################
def test_spelling_and_lookalike_inputs_are_flagged_not_silently_accepted(
    preparation: ClinicalKnowledgePreparation,
) -> None:
    resolved = prepare(preparation, ["amoxcillin", "hydroxazine", "Zurcal", "Augmentin"])

    typo = by_mention(resolved, "amoxcillin")
    assert typo["terminology"]["status"] == "probable"
    lookalike = by_mention(resolved, "hydroxazine")
    assert lookalike["decision_status"] == "ambiguous_requires_review"
    assert lookalike["matched_livertox_row"] is None
    assert {item["name"] for item in lookalike["terminology"]["candidates"]} == {"hydroxyzine", "hydralazine"}

    codes = {issue.code for issue in preparation.build_match_audit_issues(resolved)}
    assert {"drug_identity_ambiguous", "drug_composition_unverified"} <= codes


###############################################################################
class FakeIdentityClient:

    # -------------------------------------------------------------------------
    def __init__(self, proposals: list[dict[str, Any]]) -> None:
        self.proposals = proposals
        self.calls = 0

    # -------------------------------------------------------------------------
    async def llm_structured_call(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {"proposals": self.proposals}


###############################################################################
def test_llm_identity_is_used_only_after_verified_local_resolution(
    preparation: ClinicalKnowledgePreparation,
) -> None:
    client = FakeIdentityClient(
        [
            {
                "original_mention": "Sortis",
                "proposed_canonical_name": "atorvastatin",
                "confidence": 0.9,
                "rationale": "Swiss atorvastatin brand.",
            },
            {
                "original_mention": "Mysterix",
                "proposed_canonical_name": "mysterixamide",
                "confidence": 0.4,
                "rationale": "Unknown.",
            },
        ]
    )

    resolved = prepare(
        preparation,
        ["Sortis", "Mysterix"],
        identity_resolution_client=client,
        identity_resolution_model="identity-model",
    )

    assert client.calls == 1
    sortis = by_mention(resolved, "Sortis")
    assert (sortis["decision_status"], sortis["accepted_livertox_name"]) == ("accepted_verified", "atorvastatin")
    assert sortis["identity_candidates"][0]["source"] == "llm_candidate_local_validation"
    unknown = by_mention(resolved, "Mysterix")
    assert unknown["decision_status"] == "unresolved"
    assert unknown["matched_livertox_row"] is None
