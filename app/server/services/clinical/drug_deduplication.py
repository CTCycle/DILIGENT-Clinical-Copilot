from __future__ import annotations

from datetime import date, datetime
from typing import Any

from domain.clinical.entities import DrugEntry, PatientDrugs
from services.text.normalization import normalize_drug_query_name


###############################################################################
def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    for date_format in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value.strip(), date_format).date()
        except ValueError:
            continue
    return None


###############################################################################
def _entry_score(entry: DrugEntry, visit_date: date | None) -> int:
    score = 1
    if entry.source == "therapy":
        score += 4
    if entry.temporal_classification == "temporal_known":
        score += 3
    if entry.therapy_start_date:
        score += 2
    if entry.suspension_date:
        score += 1
    if entry.evidence:
        score += 1
    if entry.historical_flag:
        score -= 3
    start_date = _parse_date(entry.therapy_start_date)
    if visit_date is not None and start_date is not None and start_date > visit_date:
        score -= 8
    return score


###############################################################################
def _merge_evidence(entries: list[DrugEntry]) -> str | None:
    snippets = list(
        dict.fromkeys(
            evidence.strip()
            for entry in entries
            if (evidence := entry.evidence) and evidence.strip()
        )
    )
    if not snippets:
        return None
    return " | ".join(snippets)[:500]


###############################################################################
def deduplicate_detected_drugs(
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
    visit_date: date | None,
) -> PatientDrugs:
    grouped: dict[str, list[DrugEntry]] = {}
    order: list[str] = []
    for entry in [*therapy_drugs.entries, *anamnesis_drugs.entries]:
        key = normalize_drug_query_name(entry.name)
        if not key:
            continue
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(entry)

    selected: list[DrugEntry] = []
    for key in order:
        entries = grouped[key]
        primary = max(entries, key=lambda entry: _entry_score(entry, visit_date))
        merged_evidence = _merge_evidence(entries)
        selected.append(
            primary.model_copy(update={"evidence": merged_evidence or primary.evidence})
        )
    return PatientDrugs(entries=selected)


###############################################################################
def build_deduplication_audit(
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
    deduplicated_drugs: PatientDrugs,
) -> list[dict[str, Any]]:
    source_entries = [*therapy_drugs.entries, *anamnesis_drugs.entries]
    audit: list[dict[str, Any]] = []
    for selected in deduplicated_drugs.entries:
        key = normalize_drug_query_name(selected.name)
        merged = [
            entry
            for entry in source_entries
            if normalize_drug_query_name(entry.name) == key
        ]
        audit.append(
            {
                "normalized_name": key,
                "selected_entry": selected.model_dump(),
                "origins": list(
                    dict.fromkeys(
                        entry.source for entry in merged if entry.source is not None
                    )
                ),
                "raw_mentions": list(
                    dict.fromkeys(entry.name for entry in merged if entry.name)
                ),
                "evidence_snippets": list(
                    dict.fromkeys(
                        entry.evidence for entry in merged if entry.evidence is not None
                    )
                ),
                "merged_entry_count": len(merged),
            }
        )
    return audit
