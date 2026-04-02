from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from DILIGENT.server.domain.clinical import DrugEntry, PatientDrugs


@dataclass(frozen=True)
class CandidateSelectionResult:
    relevant: list[dict[str, str]]
    excluded: list[dict[str, str]]
    unresolved: list[dict[str, str]]
    ordered_analysis_drugs: PatientDrugs


def _score_drug(entry: DrugEntry, visit_date: date | None) -> int:
    score = 0
    if entry.source == "therapy":
        score += 3
    if entry.temporal_classification == "temporal_known":
        score += 2
    if entry.therapy_start_date:
        score += 2
    if entry.suspension_date:
        score += 1
    if entry.historical_flag:
        score -= 3
    if visit_date is not None and entry.therapy_start_date and entry.therapy_start_date > visit_date.isoformat():
        score -= 4
    return score


def select_relevant_candidates(
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
    *,
    visit_date: date | None,
) -> CandidateSelectionResult:
    candidates = [*therapy_drugs.entries, *anamnesis_drugs.entries]
    scored: list[tuple[DrugEntry, int]] = [(entry, _score_drug(entry, visit_date)) for entry in candidates if (entry.name or "").strip()]
    scored.sort(key=lambda item: item[1], reverse=True)

    relevant: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    selected_entries: list[DrugEntry] = []

    for entry, score in scored:
        name = (entry.name or "").strip()
        if not name:
            continue
        if score >= 3:
            rationale = "Active or plausibly timed exposure with compatible relevance."
            relevant.append({"drug": name, "reason": rationale})
            selected_entries.append(entry)
            continue
        if score <= -1:
            reason = "Historical or temporally incompatible exposure."
            excluded.append({"drug": name, "reason": reason})
            continue
        unresolved.append(
            {
                "drug": name,
                "reason": "Insufficient temporal detail to confirm relevance.",
            }
        )
        selected_entries.append(entry)

    return CandidateSelectionResult(
        relevant=relevant,
        excluded=excluded,
        unresolved=unresolved,
        ordered_analysis_drugs=PatientDrugs(entries=selected_entries),
    )

