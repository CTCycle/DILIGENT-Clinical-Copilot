from __future__ import annotations

from typing import Any

from common.exceptions import ServiceDependencyError
from domain.clinical.entities import (
    DrugRucamAssessment,
    PatientRucamAssessmentBundle,
    PipelineIssue,
)
from services.clinical.match_quality import classify_match_evidence
from services.text.normalization import normalize_drug_query_name

PROGRESS_SEQUENCE: list[tuple[str, float]] = [
    ("preflight.validated", 2.0),
    ("sections.loaded", 6.0),
    ("assessment.bundle", 10.0),
    ("drugs.extracting", 16.0),
    ("drugs.resolving", 30.0),
    ("diseases.extracting", 38.0),
    ("labs.extracting", 46.0),
    ("pattern.assessing", 54.0),
    ("candidates.selecting", 61.0),
    ("rucam.initial", 68.0),
    ("retrieval.query", 75.0),
    ("retrieval.evidence", 82.0),
    ("rucam.refined", 88.0),
    ("report.generating", 94.0),
    ("session.saving", 99.0),
    ("completed", 100.0),
]

###############################################################################
class ClinicalPersistenceError(ServiceDependencyError):
    default_detail = (
        "Clinical analysis completed, but the result could not be saved. "
        "No clinical report was finalized."
    )

###############################################################################
def emit_progress(
    progress_callback: Any, stage: str, progress: float, detail: str | None = None
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(stage, progress, detail)
    except TypeError:
        progress_callback(stage, progress)

###############################################################################
def extract_deterministic_drugs(
    service: Any,
    *,
    text: str,
    source: str,
) -> Any:
    parser = getattr(service, "drugs_parser", None)
    if parser is None:
        return type(
            "_Fallback",
            (),
            {"entries": [], "unresolved_lines": [], "regimen_lines": []},
        )()
    method = getattr(parser, f"extract_drugs_from_{source}_deterministic", None)
    if callable(method):
        return method(text)
    return type(
        "_Fallback", (), {"entries": [], "unresolved_lines": [], "regimen_lines": []}
    )()

###############################################################################
def append_warning_issue(
    service: Any,
    issues: list[PipelineIssue],
    *,
    code: str,
    message: str,
    field: str | None = None,
) -> None:
    if hasattr(service, "append_warning_issue"):
        service.append_warning_issue(
            issues,
            code=code,
            message=message,
            field=field,
        )
        return
    issues.append(
        PipelineIssue(
            severity="warning",
            code=code,
            message=message,
            field=field,
        )
    )

###############################################################################
def has_temporal_information(service: Any, entry: Any) -> bool:
    parser = getattr(service, "drugs_parser", None)
    checker = getattr(parser, "drug_entry_has_temporal_information", None)
    if callable(checker):
        return bool(checker(entry))
    return True

###############################################################################
def resolve_rucam_source(entries: list[DrugRucamAssessment]) -> str:
    if not entries:
        return "not_calculated_insufficient_data"
    if any(
        entry.calculation_method == "source_reported"
        and (entry.score_source or "") == "laboratory_history"
        for entry in entries
    ):
        return "provided"
    if any(entry.total_score is not None for entry in entries):
        return "calculated"
    return "not_calculated_insufficient_data"

###############################################################################
def build_single_matched_drug_row(
    *,
    detected_name: str,
    resolved: dict[str, Any],
    rucam_entry: DrugRucamAssessment | None,
) -> dict[str, Any]:
    matched_row = resolved.get("row") or {}
    match_notes = resolved.get("match_notes") or []
    if not isinstance(match_notes, list):
        match_notes = []
    match_confidence = resolved.get("match_confidence")
    if match_confidence is not None:
        try:
            match_confidence = float(match_confidence)
        except (TypeError, ValueError):
            match_confidence = None
    match_quality = classify_match_evidence(
        match_status=resolved.get("match_status"),
        match_reason=resolved.get("match_reason"),
        match_confidence=match_confidence,
        match_notes=match_notes,
        missing_livertox=bool(resolved.get("missing_livertox", True)),
        ambiguous_match=bool(resolved.get("ambiguous_match", False)),
    )
    return {
        "raw_drug_name": detected_name,
        "matched_drug_name": matched_row.get("drug_name"),
        "nbk_id": matched_row.get("nbk_id"),
        "rxnorm_rxcui": matched_row.get("rxnorm_rxcui"),
        "match_confidence": match_confidence,
        "match_reason": resolved.get("match_reason"),
        "match_notes": match_notes,
        "match_status": resolved.get("match_status"),
        "evidence_quality": match_quality["evidence_quality"],
        "evidence_warnings": match_quality["evidence_warnings"],
        "match_candidates": resolved.get("match_candidates", []),
        "chosen_candidate": resolved.get("chosen_candidate"),
        "rejected_candidates": resolved.get("rejected_candidates", []),
        "resolution_decision": resolved.get("resolution_decision"),
        "rxnav_candidates": resolved.get("rxnav_candidates", []),
        "livertox_candidates": resolved.get("livertox_candidates", []),
        "accepted_rxnav_rxcui": resolved.get("accepted_rxnav_rxcui"),
        "accepted_livertox_nbk_id": resolved.get("accepted_livertox_nbk_id"),
        "accepted_livertox_name": resolved.get("accepted_livertox_name"),
        "requires_human_review": resolved.get("requires_human_review", False),
        "decision_status": resolved.get("decision_status"),
        "rxnav_validation_status": resolved.get("rxnav_validation_status"),
        "missing_livertox": resolved.get("missing_livertox", True),
        "ambiguous_match": resolved.get("ambiguous_match", False),
        "regimen_group_ids": resolved.get("regimen_group_ids", []),
        "regimen_components": resolved.get("regimen_components", []),
        "origins": resolved.get("origins", []),
        "raw_mentions": resolved.get("raw_mentions", []),
        "rucam": rucam_entry.model_dump() if rucam_entry is not None else None,
    }

###############################################################################
def _normalized_resolved_drug_map(prepared_inputs: Any) -> dict[str, dict[str, Any]]:
    if prepared_inputs is None:
        return {}
    resolved_drug_map: dict[str, dict[str, Any]] = {}
    for key, value in prepared_inputs.resolved_drugs.items():
        normalized_key = normalize_drug_query_name(key)
        if normalized_key and isinstance(value, dict):
            resolved_drug_map[normalized_key] = value
    return resolved_drug_map

###############################################################################
def _normalized_rucam_map(
    rucam_bundle: PatientRucamAssessmentBundle,
) -> dict[str, DrugRucamAssessment]:
    rucam_by_name: dict[str, DrugRucamAssessment] = {}
    for item in rucam_bundle.entries:
        normalized_key = normalize_drug_query_name(item.drug_name)
        if normalized_key:
            rucam_by_name[normalized_key] = item
    return rucam_by_name

###############################################################################
def build_matched_drugs_payload(
    *,
    detected_drugs: list[str],
    prepared_inputs: Any,
    rucam_bundle: PatientRucamAssessmentBundle,
) -> list[dict[str, Any]]:
    resolved_drug_map = _normalized_resolved_drug_map(prepared_inputs)
    rucam_by_name = _normalized_rucam_map(rucam_bundle)
    matched_drugs_payload: list[dict[str, Any]] = []
    for detected_name in detected_drugs:
        normalized_name = normalize_drug_query_name(detected_name)
        resolved = resolved_drug_map.get(normalized_name, {})
        if prepared_inputs is None:
            resolved = {
                "match_status": "missing_match",
                "match_reason": "knowledge_base_unavailable",
                "missing_livertox": True,
            }
        rucam_entry = rucam_by_name.get(normalized_name)
        matched_drugs_payload.append(
            build_single_matched_drug_row(
                detected_name=detected_name,
                resolved=resolved,
                rucam_entry=rucam_entry,
            )
        )
    return matched_drugs_payload
