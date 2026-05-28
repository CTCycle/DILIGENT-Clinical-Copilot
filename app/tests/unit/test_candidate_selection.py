from __future__ import annotations

from datetime import date

from domain.clinical.entities import DrugEntry, PatientDrugs
from services.clinical.candidate_selection import select_relevant_candidates


def _therapy_entry(name: str, therapy_start_date: str) -> DrugEntry:
    return DrugEntry(
        name=name,
        source="therapy",
        temporal_classification="temporal_known",
        therapy_start_date=therapy_start_date,
    )


def test_candidate_selection_penalizes_future_localized_therapy_dates() -> None:
    selected = select_relevant_candidates(
        PatientDrugs(entries=[_therapy_entry("Futuremab", "20.12.2026")]),
        PatientDrugs(entries=[]),
        visit_date=date(2025, 1, 20),
    )

    assert selected.relevant == []
    assert selected.excluded == [
        {
            "drug": "Futuremab",
            "reason": "Historical or temporally conflicting exposure.",
        }
    ]
    assert selected.ordered_analysis_drugs.entries == []


def test_candidate_selection_accepts_past_localized_therapy_dates() -> None:
    selected = select_relevant_candidates(
        PatientDrugs(entries=[_therapy_entry("Pastimab", "20.12.2024")]),
        PatientDrugs(entries=[]),
        visit_date=date(2025, 1, 20),
    )

    assert selected.relevant == [
        {
            "drug": "Pastimab",
            "reason": "Active or plausibly timed exposure with aligned relevance.",
        }
    ]
    assert [entry.name for entry in selected.ordered_analysis_drugs.entries] == [
        "Pastimab"
    ]
