from __future__ import annotations

from datetime import date

from domain.clinical.entities import DrugEntry, PatientDrugs
from services.clinical.drug_deduplication import (
    build_deduplication_audit,
    deduplicate_detected_drugs,
)


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
