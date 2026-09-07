from __future__ import annotations

import asyncio
from datetime import date

from domain.clinical.entities import DrugEntry, PatientDrugs
from services.clinical.deterministic_extraction import DeterministicDrugExtractionResult
from services.clinical.drug_deduplication import (
    build_deduplication_audit,
    deduplicate_detected_drugs,
)
from services.clinical.drug_resolution.normalizer import DrugMentionNormalizer
from services.clinical.parser import DrugsParser
from services.session.preflight import LocalModelBatchPreflightResult

###############################################################################
def test_anamnesis_and_therapy_source_fields_are_preserved() -> None:
    parser = DrugsParser(client=None)
    anamnesis_entry = parser.normalize_entry(
        DrugEntry(name="Amoxicillin", therapy_start_date="2026-01-01"),
        source="anamnesis",
        historical_flag=True,
    )
    therapy_entry = parser.normalize_entry(
        DrugEntry(name="Amoxicillin", therapy_start_date="2026-02-01"),
        source="therapy",
        historical_flag=False,
    )
    assert anamnesis_entry is not None
    assert therapy_entry is not None
    assert anamnesis_entry.source == "anamnesis"
    assert anamnesis_entry.historical_flag is True
    assert therapy_entry.source == "therapy"
    assert therapy_entry.historical_flag is False

###############################################################################
def test_conservative_preparation_keeps_bullets_and_multiline_entries() -> None:
    parser = DrugsParser(client=None)
    prepared = parser.conservative_prepare_drug_section_text(
        "- Ursodeoxycholic acid 300 mg BID\n  oral\n\n* Prednisone 25 mg/day"
    )
    assert "Ursodeoxycholic acid" in prepared
    assert "Prednisone 25 mg/day" in prepared
    assert "\n" in prepared

###############################################################################
def test_drug_without_temporal_information_is_filtered() -> None:
    parser = DrugsParser(client=None)
    no_temporal = DrugEntry(name="Drug A")
    with_temporal = DrugEntry(name="Drug B", therapy_start_date="2026-03-10")
    assert parser.drug_entry_has_temporal_information(no_temporal) is False
    assert parser.drug_entry_has_temporal_information(with_temporal) is True

###############################################################################
def test_batch_preflight_flags_cover_concurrent_and_sequential_paths() -> None:
    allow = LocalModelBatchPreflightResult(
        concurrency_allowed=True,
        provider="openai",
        model="gpt-4.1-mini",
    )
    deny = LocalModelBatchPreflightResult(
        concurrency_allowed=False,
        provider="ollama",
        model="qwen3:14b",
        reason="runtime status unavailable",
    )
    assert allow.concurrency_allowed is True
    assert deny.concurrency_allowed is False
    assert deny.reason

###############################################################################
def test_source_differences_prevent_cross_section_collapse() -> None:
    parser = DrugsParser(client=None)
    entries = [
        DrugEntry(
            name="Acetaminophen",
            source="anamnesis",
            historical_flag=True,
            therapy_start_date="2025-12-01",
        ),
        DrugEntry(
            name="Acetaminophen",
            source="therapy",
            historical_flag=False,
            therapy_start_date="2026-01-15",
        ),
    ]
    deduped = parser.deduplicate_drug_entries(entries)
    # Current pipeline keeps section-specific origin and must not collapse these two.
    assert len(deduped) == 2

###############################################################################
def test_structural_gate_does_not_require_catalog_recognition() -> None:
    normalizer = DrugMentionNormalizer()
    for value in (
        "Ecografia Addome",
        "Adiuvante",
        "Effettuata",
        "Inizio",
        "Introdotto",
    ):
        assert normalizer._normalize_entry(DrugEntry(name=value)) is not None
    assert (
        normalizer._normalize_entry(
            DrugEntry(name="Paziente nota per abuso di etile circa ogni giorno")
        )
        is None
    )

###############################################################################
def test_demographic_case_opening_is_not_normalized_as_drug() -> None:
    normalizer = DrugMentionNormalizer()
    for value in (
        "58-year-old woman evaluated",
        "46-year-old man evaluated",
        "72 year old patient admitted",
    ):
        assert normalizer._normalize_entry(DrugEntry(name=value)) is None

###############################################################################
def test_novel_inn_suffix_candidate_remains_for_missing_livertox_resolution() -> None:
    normalizer = DrugMentionNormalizer()
    mention = normalizer._normalize_entry(DrugEntry(name="Trialzumab"))
    assert mention is not None
    assert mention.normalized_name == "trialzumab"

###############################################################################
def test_therapy_hybrid_fallback_uses_complete_block_context(monkeypatch) -> None:

    ###############################################################################
    class StructuredClientStub:

        # -------------------------------------------------------------------------
        async def llm_structured_call(self, **kwargs: object) -> object:
            raise AssertionError("Patched section extractor should be used")

    parser = DrugsParser(client=StructuredClientStub())
    source = "Amoxicillin/clavulanate 875/125 mg\ncontinued twice daily."
    monkeypatch.setattr(
        parser,
        "extract_drugs_from_therapy_deterministic",
        lambda cleaned: DeterministicDrugExtractionResult(
            entries=[],
            unresolved_lines=["Amoxicillin/clavulanate 875/125 mg"],
            regimen_lines=[],
        ),
    )
    captured: dict[str, str] = {}

    async def fake_section_extract(
        source_text: str,
        **kwargs: object,
    ) -> PatientDrugs:
        captured["source_text"] = source_text
        return PatientDrugs(
            entries=[
                DrugEntry(
                    name="Amoxicillin/clavulanate",
                    source="therapy",
                    evidence="Amoxicillin/clavulanate",
                    source_span=[0, 23],
                )
            ]
        )

    monkeypatch.setattr(parser, "llm_extract_drugs_from_section", fake_section_extract)
    entries = asyncio.run(parser.extract_drugs_from_therapy_hybrid(source))

    assert captured["source_text"] == source
    assert [entry.name for entry in entries] == ["Amoxicillin/clavulanate"]

###############################################################################
def test_cross_source_duplicates_keep_best_entry_and_merge_provenance() -> None:
    therapy = PatientDrugs(
        entries=[
            DrugEntry(
                name="Acetaminophen",
                source="therapy",
                therapy_start_date="2026-01-10",
                temporal_classification="temporal_known",
                evidence="Acetaminophen 500 mg twice daily",
            ),
            DrugEntry(
                name="Acetaminophen",
                source="therapy",
                evidence="Acetaminophen listed without timing",
            ),
        ]
    )
    anamnesis = PatientDrugs(
        entries=[
            DrugEntry(
                name="acetaminophen",
                source="anamnesis",
                historical_flag=True,
                evidence="Previously used acetaminophen",
            ),
            DrugEntry(
                name="Metformin",
                source="anamnesis",
                historical_flag=True,
                evidence="Historical metformin",
            ),
        ]
    )

    result = deduplicate_detected_drugs(therapy, anamnesis, date(2026, 2, 1))

    assert [entry.name for entry in result.entries] == ["Acetaminophen", "Metformin"]
    assert result.entries[0].source == "therapy"
    assert result.entries[0].therapy_start_date == "2026-01-10"
    assert "Previously used acetaminophen" in (result.entries[0].evidence or "")

    audit = build_deduplication_audit(therapy, anamnesis, result)
    assert audit[0]["origins"] == ["therapy", "anamnesis"]
    assert audit[0]["merged_entry_count"] == 3
    assert len(audit[0]["evidence_snippets"]) == 3
