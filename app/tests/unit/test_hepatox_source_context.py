from __future__ import annotations

from domain.clinical.entities import DrugEntry
from services.clinical.hepatox_assessment import (
    assess_pattern_compatibility,
    assess_temporal_plausibility,
    summarize_drug_source_context,
)


def test_summarize_drug_source_context_uses_entry_source() -> None:
    therapy = DrugEntry(name="Drug A", source="therapy")
    anamnesis = DrugEntry(name="Drug B", source="anamnesis")
    assert "therapy" in summarize_drug_source_context(therapy).lower()
    assert "anamnesis" in summarize_drug_source_context(anamnesis).lower()


def test_temporal_plausibility_reflects_available_timing_fields() -> None:
    rich = DrugEntry(
        name="Drug A",
        therapy_start_date="2026-01-01",
        suspension_status=True,
    )
    poor = DrugEntry(name="Drug B")
    assert "sequence" in assess_temporal_plausibility(rich, None).lower()
    assert "limited" in assess_temporal_plausibility(poor, None).lower()


def test_pattern_compatibility_handles_missing_excerpt() -> None:
    entry = DrugEntry(name="Drug A")
    message = assess_pattern_compatibility(entry, "mixed", None)
    assert "unavailable" in message.lower()
