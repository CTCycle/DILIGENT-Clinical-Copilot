# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from datetime import date
from typing import Any

from common.utils.clinical_dates import parse_clinical_date
from domain.clinical.entities import DrugEntry, PatientDrugs
from services.text.normalization import normalize_drug_query_name

###############################################################################
def _parse_date(value: str | None) -> date | None:
    return parse_clinical_date(value)

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
def find_drug_date_conflicts(
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
) -> list[dict[str, Any]]:
    """Blocking issues for drugs whose therapy and anamnesis dates disagree.

    Deduplication keeps one entry per drug, so a contradicting start or stop
    date in the other section would otherwise be dropped silently.
    """
    excluded_attributions = {"negated", "allergy", "family_history"}
    grouped: dict[str, dict[str, list[DrugEntry]]] = {}
    for section, entries in (
        ("therapy", therapy_drugs.entries),
        ("anamnesis", anamnesis_drugs.entries),
    ):
        for entry in entries:
            key = normalize_drug_query_name(entry.name)
            if key and entry.attribution not in excluded_attributions:
                grouped.setdefault(key, {}).setdefault(section, []).append(entry)

    issues: list[dict[str, Any]] = []
    for sections in grouped.values():
        if len(sections) < 2:
            continue
        for field, label in (
            ("therapy_start_date", "start"),
            ("suspension_date", "stop"),
        ):
            dates = {
                section: {
                    parsed
                    for entry in entries
                    if (parsed := _parse_date(getattr(entry, field))) is not None
                }
                for section, entries in sections.items()
            }
            if not dates["therapy"] or not dates["anamnesis"]:
                continue
            if dates["therapy"] == dates["anamnesis"]:
                continue
            drug_name = sections["therapy"][0].name
            therapy_text = ", ".join(sorted(d.isoformat() for d in dates["therapy"]))
            anamnesis_text = ", ".join(
                sorted(d.isoformat() for d in dates["anamnesis"])
            )
            issues.append(
                {
                    "severity": "error",
                    "code": "drug_date_conflict",
                    "message": (
                        f"{drug_name}: the therapy section gives {label} date "
                        f"{therapy_text} but the anamnesis gives {anamnesis_text}; "
                        "the exposure chronology requires review."
                    ),
                    "field": "drugs",
                }
            )
    return issues

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
