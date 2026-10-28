# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import date

from common.utils.clinical_dates import parse_clinical_date
from domain.clinical.entities import DrugEntry, PatientDrugs
from domain.clinical.extras import CandidateSelectionResult
from services.text.normalization import normalize_drug_query_name

# RUCAM 2016: onset up to 30 days after cessation remains compatible
# (cholestatic/mixed); hepatocellular injury is narrower, so this is the
# outer bound for keeping a stopped drug in the consultation set.
MAX_DAYS_FROM_STOP_TO_ONSET = 30

###############################################################################
def _timing_fits_injury(entry: DrugEntry, reference_date: date | None) -> bool:
    """A stopped exposure whose start and stop dates fit the injury onset."""
    stop = _parse_therapy_date(entry.suspension_date)
    if reference_date is None or stop is None:
        return False
    start = _parse_therapy_date(entry.therapy_start_date)
    if start is not None and start > reference_date:
        return False
    return (reference_date - stop).days <= MAX_DAYS_FROM_STOP_TO_ONSET

###############################################################################
def _score_drug(
    entry: DrugEntry, visit_date: date | None, onset_date: date | None = None
) -> int:
    score = 0
    if entry.source == "therapy":
        score += 3
    if entry.temporal_classification == "temporal_known":
        score += 2
    if entry.therapy_start_date:
        score += 2
    if entry.suspension_date:
        score += 1
    therapy_start_date = _parse_therapy_date(entry.therapy_start_date)
    if (
        visit_date is not None
        and therapy_start_date is not None
        and therapy_start_date > visit_date
    ):
        return min(score - 8, -1)
    if _timing_fits_injury(entry, onset_date or visit_date):
        return max(score, 3)
    if entry.historical_flag:
        score -= 3
    return score

###############################################################################
def _parse_therapy_date(value: str | None) -> date | None:
    return parse_clinical_date(value)

###############################################################################
def select_relevant_candidates(
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
    *,
    visit_date: date | None,
    onset_date: date | None = None,
) -> CandidateSelectionResult:
    candidates = _deduplicate_candidates(
        [*therapy_drugs.entries, *anamnesis_drugs.entries], visit_date, onset_date
    )
    scored: list[tuple[DrugEntry, int]] = [
        (entry, _score_drug(entry, visit_date, onset_date))
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
            rationale = (
                "Stopped exposure documented in the anamnesis whose timing fits the injury onset."
                if entry.source == "anamnesis"
                and _timing_fits_injury(entry, onset_date or visit_date)
                else "Active or plausibly timed exposure with aligned relevance."
            )
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

###############################################################################
def _deduplicate_candidates(
    entries: list[DrugEntry],
    visit_date: date | None,
    onset_date: date | None,
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
        if _score_drug(entry, visit_date, onset_date) > _score_drug(
            existing, visit_date, onset_date
        ):
            selected[key] = entry
    return [selected[key] for key in order]
