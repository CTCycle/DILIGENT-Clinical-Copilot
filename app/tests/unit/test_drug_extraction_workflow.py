from __future__ import annotations

from domain.clinical.entities import DrugEntry
from services.clinical.parser import DrugsParser
from services.session.preflight import LocalModelBatchPreflightResult


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


def test_conservative_preparation_keeps_bullets_and_multiline_entries() -> None:
    parser = DrugsParser(client=None)
    prepared = parser.conservative_prepare_drug_section_text(
        "- Ursodeoxycholic acid 300 mg BID\n"
        "  oral\n\n"
        "* Prednisone 25 mg/day"
    )
    assert "Ursodeoxycholic acid" in prepared
    assert "Prednisone 25 mg/day" in prepared
    assert "\n" in prepared


def test_drug_without_temporal_information_is_filtered() -> None:
    parser = DrugsParser(client=None)
    no_temporal = DrugEntry(name="Drug A")
    with_temporal = DrugEntry(name="Drug B", therapy_start_date="2026-03-10")
    assert parser.drug_entry_has_temporal_information(no_temporal) is False
    assert parser.drug_entry_has_temporal_information(with_temporal) is True


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
