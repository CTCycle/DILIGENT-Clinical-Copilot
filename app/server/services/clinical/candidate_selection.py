from __future__ import annotations

from datetime import date, datetime

from domain.clinical.entities import DrugEntry, PatientDrugs
from domain.clinical.extras import CandidateSelectionResult
from services.text.normalization import normalize_drug_query_name


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
    therapy_start_date = _parse_therapy_date(entry.therapy_start_date)
    if (
        visit_date is not None
        and therapy_start_date is not None
        and therapy_start_date > visit_date
    ):
        return min(score - 8, -1)
    return score


def _parse_therapy_date(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    for date_format in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue
    return None


def select_relevant_candidates(
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
    *,
    visit_date: date | None,
) -> CandidateSelectionResult:
    candidates = _deduplicate_candidates(
        [*therapy_drugs.entries, *anamnesis_drugs.entries], visit_date
    )
    scored: list[tuple[DrugEntry, int]] = [
        (entry, _score_drug(entry, visit_date))
        for entry in candidates
        if (entry.name or "").strip()
    ]
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
            rationale = "Active or plausibly timed exposure with aligned relevance."
            relevant.append({"drug": name, "reason": rationale})
            selected_entries.append(entry)
            continue
        if score <= -1:
            reason = "Historical or temporally conflicting exposure."
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


def _deduplicate_candidates(
    entries: list[DrugEntry],
    visit_date: date | None,
) -> list[DrugEntry]:
    selected: dict[str, DrugEntry] = {}
    order: list[str] = []
    for entry in entries:
        key = normalize_drug_query_name(entry.name)
        if not key:
            continue
        existing = selected.get(key)
        if existing is None:
            selected[key] = entry
            order.append(key)
            continue
        if _score_drug(entry, visit_date) > _score_drug(existing, visit_date):
            selected[key] = entry
    return [selected[key] for key in order]
