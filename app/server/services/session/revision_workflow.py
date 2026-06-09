from __future__ import annotations

from typing import Any

from domain.clinical.entities import (
    ClinicalLabEntry,
    DeterministicDiseaseExtractionResult,
    DeterministicDrugExtractionResult,
    DiseaseContextEntry,
    DrugEntry,
    DrugRucamAssessment,
    LiverInjuryOnsetContext,
    PatientDiseaseContext,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
)
from domain.clinical.extras import CandidateSelectionResult
from domain.clinical.revision import (
    RevisionCandidateSelectionResolution,
    RevisionConsultationExecution,
    RevisionConsultationInputs,
    RevisionExtractionResolution,
    RevisionFinalizationOutputs,
)
from services.clinical.report_language import phrase
from services.clinical.candidate_selection import select_relevant_candidates
from services.session.session_shared import run_clinical_job


def _unique_non_empty_names(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        folded = cleaned.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        normalized.append(cleaned)
    return normalized


def _normalize_candidate_selection_names(values: list[Any]) -> list[str]:
    normalized_values: list[str | None] = []
    for value in values:
        if isinstance(value, dict):
            normalized_values.append(str(value.get("drug") or "").strip() or None)
            continue
        normalized_values.append(str(value or "").strip() or None)
    return _unique_non_empty_names(normalized_values)


def _candidate_selection_index(
    values: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for item in values:
        if not isinstance(item, dict):
            continue
        drug_name = str(item.get("drug") or "").strip()
        if not drug_name:
            continue
        indexed[drug_name.casefold()] = {
            "drug": drug_name,
            "reason": str(item.get("reason") or "").strip(),
        }
    return indexed


def _rebuild_drug_entries(payload: Any) -> list[DrugEntry]:
    if not isinstance(payload, list):
        return []
    entries: list[DrugEntry] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            entries.append(DrugEntry.model_validate(item))
        except Exception:
            continue
    return entries


def _rebuild_disease_entries(payload: Any) -> list[DiseaseContextEntry]:
    if not isinstance(payload, list):
        return []
    entries: list[DiseaseContextEntry] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            entries.append(DiseaseContextEntry.model_validate(item))
        except Exception:
            continue
    return entries


def _load_persisted_deterministic_drug_extraction(
    payload: Any,
) -> DeterministicDrugExtractionResult | None:
    if not isinstance(payload, dict):
        return None
    entries = _rebuild_drug_entries(payload.get("entries"))
    unresolved_lines = [
        str(line).strip()
        for line in (payload.get("unresolved_lines") or [])
        if str(line).strip()
    ]
    regimen_lines = [
        str(line).strip()
        for line in (payload.get("regimen_lines") or [])
        if str(line).strip()
    ]
    if not entries and not unresolved_lines and not regimen_lines:
        return None
    return DeterministicDrugExtractionResult(
        entries=entries,
        unresolved_lines=unresolved_lines,
        regimen_lines=regimen_lines,
    )


def _load_persisted_deterministic_disease_extraction(
    payload: Any,
) -> DeterministicDiseaseExtractionResult | None:
    if not isinstance(payload, dict):
        return None
    entries = _rebuild_disease_entries(payload.get("entries"))
    matched_lines = [
        str(line).strip()
        for line in (payload.get("matched_lines") or [])
        if str(line).strip()
    ]
    unresolved_lines = [
        str(line).strip()
        for line in (payload.get("unresolved_lines") or [])
        if str(line).strip()
    ]
    if not entries and not matched_lines and not unresolved_lines:
        return None
    return DeterministicDiseaseExtractionResult(
        context=PatientDiseaseContext(entries=entries),
        matched_lines=matched_lines,
        unresolved_lines=unresolved_lines,
    )


def _load_revision_source_deterministic_extraction(
    session_metadata: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    source_payload = (
        session_metadata.get("source_deterministic_extraction")
        if isinstance(session_metadata, dict)
        else None
    )
    if not isinstance(source_payload, dict):
        return {}, {
            "therapy": "recomputed_current_revision",
            "anamnesis": "recomputed_current_revision",
            "diseases": "recomputed_current_revision",
        }
    loaded: dict[str, Any] = {}
    therapy_payload = _load_persisted_deterministic_drug_extraction(
        source_payload.get("therapy")
    )
    anamnesis_payload = _load_persisted_deterministic_drug_extraction(
        source_payload.get("anamnesis")
    )
    disease_payload = _load_persisted_deterministic_disease_extraction(
        source_payload.get("diseases")
    )
    if therapy_payload is not None:
        loaded["therapy"] = therapy_payload
    if anamnesis_payload is not None:
        loaded["anamnesis"] = anamnesis_payload
    if disease_payload is not None:
        loaded["diseases"] = disease_payload
    return loaded, {
        "therapy": (
            "persisted_source_version"
            if therapy_payload is not None
            else "recomputed_current_revision"
        ),
        "anamnesis": (
            "persisted_source_version"
            if anamnesis_payload is not None
            else "recomputed_current_revision"
        ),
        "diseases": (
            "persisted_source_version"
            if disease_payload is not None
            else "recomputed_current_revision"
        ),
    }


def _load_revision_source_disease_context(
    session_metadata: dict[str, Any] | None,
) -> tuple[PatientDiseaseContext | None, str]:
    source_structured_case = (
        session_metadata.get("source_structured_case")
        if isinstance(session_metadata, dict)
        else None
    )
    if not isinstance(source_structured_case, dict):
        return None, "recomputed_current_revision"
    entries = _rebuild_disease_entries(
        source_structured_case.get("anamnesis_diseases")
    )
    if not entries:
        return None, "recomputed_current_revision"
    return PatientDiseaseContext(entries=entries), "persisted_source_version"


def _load_revision_source_lab_timeline(
    session_metadata: dict[str, Any] | None,
) -> tuple[PatientLabTimeline | None, LiverInjuryOnsetContext | None, dict[str, str]]:
    source_lab_timeline = (
        session_metadata.get("source_lab_timeline")
        if isinstance(session_metadata, dict)
        else None
    )
    source_onset_context = (
        session_metadata.get("source_onset_context")
        if isinstance(session_metadata, dict)
        else None
    )
    lab_entries: list[ClinicalLabEntry] = []
    if isinstance(source_lab_timeline, list):
        for item in source_lab_timeline:
            if not isinstance(item, dict):
                continue
            try:
                lab_entries.append(ClinicalLabEntry.model_validate(item))
            except Exception:
                continue
    onset_context: LiverInjuryOnsetContext | None = None
    if isinstance(source_onset_context, dict):
        try:
            onset_context = LiverInjuryOnsetContext.model_validate(source_onset_context)
        except Exception:
            onset_context = None
    return (
        PatientLabTimeline(entries=lab_entries) if lab_entries else None,
        onset_context,
        {
            "lab_timeline": (
                "persisted_source_version"
                if lab_entries
                else "recomputed_current_revision"
            ),
            "onset_context": (
                "persisted_source_version"
                if onset_context is not None
                else "recomputed_current_revision"
            ),
        },
    )


def _build_revision_anamnesis_validation_stage(
    *,
    anamnesis_deterministic: Any,
    anamnesis_drugs: PatientDrugs,
) -> dict[str, Any]:
    deterministic_names = _unique_non_empty_names(
        [getattr(entry, "name", None) for entry in anamnesis_deterministic.entries]
    )
    revised_names = _unique_non_empty_names(
        [getattr(entry, "name", None) for entry in anamnesis_drugs.entries]
    )
    deterministic_name_keys = {name.casefold() for name in deterministic_names}
    revised_name_keys = {name.casefold() for name in revised_names}
    overlapping_names = [
        name for name in revised_names if name.casefold() in deterministic_name_keys
    ]
    deterministic_only_names = [
        name
        for name in deterministic_names
        if name.casefold() not in revised_name_keys
    ]
    revised_only_names = [
        name for name in revised_names if name.casefold() not in deterministic_name_keys
    ]
    unresolved_lines = list(getattr(anamnesis_deterministic, "unresolved_lines", []))
    status = "verified"
    if revised_only_names:
        status = "supplemented"
    elif not revised_names and (deterministic_names or unresolved_lines):
        status = "requires_human_review"
    elif not revised_names and not deterministic_names:
        status = "no_structured_revision_changes"
    return {
        "status": status,
        "deterministic_detected_names": deterministic_names,
        "revised_detected_names": revised_names,
        "overlapping_names": overlapping_names,
        "deterministic_only_names": deterministic_only_names,
        "revised_only_names": revised_only_names,
        "unresolved_lines": unresolved_lines,
    }


def _build_revision_extraction_bundle(
    *,
    therapy_deterministic: Any,
    anamnesis_deterministic: Any,
    disease_deterministic: Any,
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
    source_modes: dict[str, str] | None = None,
) -> dict[str, Any]:
    source_modes = source_modes or {}
    return {
        "status": "resolved",
        "therapy_source": str(
            source_modes.get("therapy") or "recomputed_current_revision"
        ),
        "anamnesis_source": str(
            source_modes.get("anamnesis") or "recomputed_current_revision"
        ),
        "disease_source": str(
            source_modes.get("diseases") or "recomputed_current_revision"
        ),
        "therapy_deterministic_names": _unique_non_empty_names(
            [getattr(entry, "name", None) for entry in therapy_deterministic.entries]
        ),
        "anamnesis_deterministic_names": _unique_non_empty_names(
            [getattr(entry, "name", None) for entry in anamnesis_deterministic.entries]
        ),
        "disease_deterministic_names": _unique_non_empty_names(
            [
                getattr(entry, "name", None)
                for entry in disease_deterministic.context.entries
            ]
        ),
        "therapy_structured_names": _unique_non_empty_names(
            [getattr(entry, "name", None) for entry in therapy_drugs.entries]
        ),
        "anamnesis_structured_names": _unique_non_empty_names(
            [getattr(entry, "name", None) for entry in anamnesis_drugs.entries]
        ),
        "therapy_unresolved_lines": list(
            getattr(therapy_deterministic, "unresolved_lines", [])
        ),
        "anamnesis_unresolved_lines": list(
            getattr(anamnesis_deterministic, "unresolved_lines", [])
        ),
        "anamnesis_regimen_lines": list(
            getattr(anamnesis_deterministic, "regimen_lines", [])
        ),
        "disease_unresolved_lines": list(
            getattr(disease_deterministic, "unresolved_lines", [])
        ),
    }


def _resolve_revision_extraction(
    *,
    therapy_deterministic: Any,
    anamnesis_deterministic: Any,
    disease_deterministic: Any,
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
    source_modes: dict[str, str] | None = None,
) -> RevisionExtractionResolution:
    extraction_bundle = _build_revision_extraction_bundle(
        therapy_deterministic=therapy_deterministic,
        anamnesis_deterministic=anamnesis_deterministic,
        disease_deterministic=disease_deterministic,
        therapy_drugs=therapy_drugs,
        anamnesis_drugs=anamnesis_drugs,
        source_modes=source_modes,
    )
    return RevisionExtractionResolution(
        therapy_deterministic=therapy_deterministic,
        anamnesis_deterministic=anamnesis_deterministic,
        disease_deterministic=disease_deterministic,
        therapy_drugs=therapy_drugs,
        anamnesis_drugs=anamnesis_drugs,
        extraction_bundle=extraction_bundle,
    )


def _build_revision_missing_anamnesis_drugs_stage(
    *,
    anamnesis_drugs: PatientDrugs,
    validation_stage: dict[str, Any],
) -> dict[str, Any]:
    supplemental_drug_names = list(validation_stage.get("revised_only_names") or [])
    supplemental_entries = [
        entry.model_dump()
        for entry in anamnesis_drugs.entries
        if str(entry.name or "").strip()
        and str(entry.name).strip().casefold()
        in {name.casefold() for name in supplemental_drug_names}
    ]
    return {
        "status": (
            "supplemented" if supplemental_entries else "no_missing_drugs_detected"
        ),
        "supplemental_drug_names": supplemental_drug_names,
        "supplemental_entries": supplemental_entries,
    }


def _build_revision_analysis_drugs(
    *,
    base_analysis_drugs: PatientDrugs,
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
    missing_stage: dict[str, Any] | None,
) -> PatientDrugs:
    if not isinstance(missing_stage, dict):
        return base_analysis_drugs
    supplemental_names = _unique_non_empty_names(
        list(missing_stage.get("supplemental_drug_names") or [])
    )
    if not supplemental_names:
        return base_analysis_drugs
    supplemental_keys = {name.casefold() for name in supplemental_names}
    collected_entries = []
    seen_entry_names: set[str] = set()
    for entry in [
        *anamnesis_drugs.entries,
        *therapy_drugs.entries,
        *base_analysis_drugs.entries,
    ]:
        entry_name = str(getattr(entry, "name", "") or "").strip()
        if not entry_name:
            continue
        entry_key = entry_name.casefold()
        if entry_key in seen_entry_names:
            continue
        if entry_key in supplemental_keys:
            seen_entry_names.add(entry_key)
            collected_entries.append(entry)
    return PatientDrugs(entries=collected_entries or base_analysis_drugs.entries)


def _reconcile_revision_candidate_selection(
    *,
    candidate_selection: CandidateSelectionResult,
    analysis_drugs: PatientDrugs,
    missing_stage: dict[str, Any] | None,
) -> CandidateSelectionResult:
    if not isinstance(missing_stage, dict):
        return candidate_selection
    supplemental_names = _unique_non_empty_names(
        list(missing_stage.get("supplemental_drug_names") or [])
    )
    if not supplemental_names:
        return candidate_selection
    supplemental_keys = {name.casefold() for name in supplemental_names}
    relevant = _candidate_selection_index(list(candidate_selection.relevant))
    excluded = _candidate_selection_index(list(candidate_selection.excluded))
    unresolved = _candidate_selection_index(list(candidate_selection.unresolved))
    for supplemental_name in supplemental_names:
        key = supplemental_name.casefold()
        unresolved.pop(key, None)
        excluded.pop(key, None)
        relevant[key] = {
            "drug": supplemental_name,
            "reason": (
                "Revision pipeline promoted this drug from staged anamnesis additions "
                "for targeted reassessment."
            ),
        }
    return CandidateSelectionResult(
        relevant=list(relevant.values()),
        excluded=list(excluded.values()),
        unresolved=list(unresolved.values()),
        ordered_analysis_drugs=analysis_drugs,
    )


def _build_revision_candidate_selection_stage(
    *,
    candidate_selection: CandidateSelectionResult,
) -> dict[str, Any]:
    return {
        "status": "reconciled",
        "analysis_drug_names": _unique_non_empty_names(
            [
                getattr(entry, "name", None)
                for entry in candidate_selection.ordered_analysis_drugs.entries
            ]
        ),
        "relevant_drug_names": _normalize_candidate_selection_names(
            list(candidate_selection.relevant or [])
        ),
        "excluded_drug_names": _normalize_candidate_selection_names(
            list(candidate_selection.excluded or [])
        ),
        "unresolved_drug_names": _normalize_candidate_selection_names(
            list(candidate_selection.unresolved or [])
        ),
    }


def _select_revision_candidates(
    *,
    extraction_bundle: dict[str, Any],
    anamnesis_deterministic: Any,
    anamnesis_drugs: PatientDrugs,
    therapy_drugs: PatientDrugs,
    lab_timeline: Any,
    onset_context: Any,
    pattern_score: Any,
    visit_date: Any | None,
) -> RevisionCandidateSelectionResolution:
    validate_anamnesis_drugs = _build_revision_anamnesis_validation_stage(
        anamnesis_deterministic=anamnesis_deterministic,
        anamnesis_drugs=anamnesis_drugs,
    )
    extract_missing_anamnesis_drugs = _build_revision_missing_anamnesis_drugs_stage(
        anamnesis_drugs=anamnesis_drugs,
        validation_stage=validate_anamnesis_drugs,
    )
    revise_labs_timeline = _build_revision_lab_revision_stage(
        lab_timeline=lab_timeline,
        onset_context=onset_context,
        pattern_score=pattern_score,
    )
    base_candidate_selection = select_relevant_candidates(
        therapy_drugs=therapy_drugs,
        anamnesis_drugs=anamnesis_drugs,
        visit_date=visit_date,
    )
    analysis_drugs = _build_revision_analysis_drugs(
        base_analysis_drugs=base_candidate_selection.ordered_analysis_drugs,
        therapy_drugs=therapy_drugs,
        anamnesis_drugs=anamnesis_drugs,
        missing_stage=extract_missing_anamnesis_drugs,
    )
    candidate_selection = _reconcile_revision_candidate_selection(
        candidate_selection=base_candidate_selection,
        analysis_drugs=analysis_drugs,
        missing_stage=extract_missing_anamnesis_drugs,
    )
    return RevisionCandidateSelectionResolution(
        analysis_drugs=analysis_drugs,
        candidate_selection=candidate_selection,
        entity_pipeline={
            "resolve_revision_extraction": extraction_bundle,
            "validate_anamnesis_drugs": validate_anamnesis_drugs,
            "extract_missing_anamnesis_drugs": extract_missing_anamnesis_drugs,
            "revise_labs_timeline": revise_labs_timeline,
            "reconcile_revision_candidates": _build_revision_candidate_selection_stage(
                candidate_selection=candidate_selection
            ),
        },
    )


def _build_revision_lab_revision_stage(
    *,
    lab_timeline: Any,
    onset_context: Any,
    pattern_score: Any,
) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    marker_names: list[str] = []
    for entry in lab_timeline.entries:
        source = str(getattr(entry, "source", "") or "unknown").strip() or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1
        marker_name = str(getattr(entry, "marker_name", "") or "").strip()
        if marker_name:
            marker_names.append(marker_name)
    return {
        "status": "revised",
        "lab_entry_count": len(lab_timeline.entries),
        "source_counts": source_counts,
        "marker_names": _unique_non_empty_names(marker_names),
        "onset_context_present": onset_context is not None,
        "pattern_classification": getattr(pattern_score, "classification", None),
    }


def _build_revision_snapshot_merge_stage(
    *,
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
    disease_context: Any,
    lab_timeline: Any,
    analysis_drugs: PatientDrugs,
    candidate_selection: Any,
    rucam_bundle: PatientRucamAssessmentBundle,
) -> dict[str, Any]:
    return {
        "status": "merged",
        "therapy_drug_names": _unique_non_empty_names(
            [getattr(entry, "name", None) for entry in therapy_drugs.entries]
        ),
        "anamnesis_drug_names": _unique_non_empty_names(
            [getattr(entry, "name", None) for entry in anamnesis_drugs.entries]
        ),
        "disease_names": _unique_non_empty_names(
            [getattr(entry, "name", None) for entry in disease_context.entries]
        ),
        "lab_marker_names": _unique_non_empty_names(
            [getattr(entry, "marker_name", None) for entry in lab_timeline.entries]
        ),
        "analysis_drug_names": _unique_non_empty_names(
            [getattr(entry, "name", None) for entry in analysis_drugs.entries]
        ),
        "relevant_drug_names": _normalize_candidate_selection_names(
            list(getattr(candidate_selection, "relevant", []) or [])
        ),
        "excluded_drug_names": _normalize_candidate_selection_names(
            list(getattr(candidate_selection, "excluded", []) or [])
        ),
        "unresolved_drug_names": _normalize_candidate_selection_names(
            list(getattr(candidate_selection, "unresolved", []) or [])
        ),
        "rucam_assessment_count": len(rucam_bundle.entries),
    }


def _build_revision_snapshot_context(
    entity_pipeline: dict[str, Any] | None,
) -> str | None:
    if not isinstance(entity_pipeline, dict) or not entity_pipeline:
        return None
    chunks: list[str] = []
    validate_stage = entity_pipeline.get("validate_anamnesis_drugs")
    if isinstance(validate_stage, dict):
        revised_only_names = _unique_non_empty_names(
            list(validate_stage.get("revised_only_names") or [])
        )
        if revised_only_names:
            chunks.append(
                "Revision anamnesis additions:\n" + ", ".join(revised_only_names)
            )
    missing_stage = entity_pipeline.get("extract_missing_anamnesis_drugs")
    if isinstance(missing_stage, dict):
        supplemental_names = _unique_non_empty_names(
            list(missing_stage.get("supplemental_drug_names") or [])
        )
        if supplemental_names:
            chunks.append(
                "Revision supplemental anamnesis drugs:\n"
                + ", ".join(supplemental_names)
            )
    labs_stage = entity_pipeline.get("revise_labs_timeline")
    if isinstance(labs_stage, dict):
        marker_names = _unique_non_empty_names(
            list(labs_stage.get("marker_names") or [])
        )
        if marker_names:
            chunks.append("Revision lab markers:\n" + ", ".join(marker_names))
    merge_stage = entity_pipeline.get("merge_revision_snapshot")
    if isinstance(merge_stage, dict):
        analysis_names = _unique_non_empty_names(
            list(merge_stage.get("analysis_drug_names") or [])
        )
        disease_names = _unique_non_empty_names(
            list(merge_stage.get("disease_names") or [])
        )
        if analysis_names:
            chunks.append("Revision analysis drugs:\n" + ", ".join(analysis_names))
        if disease_names:
            chunks.append(
                "Revision disease context:\n" + ", ".join(disease_names)
            )
    if not chunks:
        return None
    return "\n\n".join(chunks)


def _build_revision_consultation_drugs(
    *,
    entity_pipeline: dict[str, Any] | None,
    analysis_drugs: PatientDrugs,
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
) -> PatientDrugs:
    if not isinstance(entity_pipeline, dict):
        return analysis_drugs
    merge_stage = entity_pipeline.get("merge_revision_snapshot")
    if not isinstance(merge_stage, dict):
        return analysis_drugs
    target_names = _unique_non_empty_names(
        list(merge_stage.get("analysis_drug_names") or [])
    )
    if not target_names:
        return analysis_drugs
    target_keys = {name.casefold() for name in target_names}
    collected_entries = []
    seen_entry_names: set[str] = set()
    for entry in [
        *analysis_drugs.entries,
        *therapy_drugs.entries,
        *anamnesis_drugs.entries,
    ]:
        entry_name = str(getattr(entry, "name", "") or "").strip()
        if not entry_name or entry_name.casefold() not in target_keys:
            continue
        if entry_name.casefold() in seen_entry_names:
            continue
        seen_entry_names.add(entry_name.casefold())
        collected_entries.append(entry)
    return PatientDrugs(entries=collected_entries or analysis_drugs.entries)


def _build_revision_consultation_context(
    *,
    structured_context: str,
    revision_snapshot_context: str | None,
    revision_focus_context: str | None,
    session_metadata: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    previous_report = str(
        (session_metadata or {}).get("source_official_report_text") or ""
    ).strip()
    source_version_id = (session_metadata or {}).get("source_version_id")
    revision_version_id = (session_metadata or {}).get("target_revision_version_id")
    pipeline_run_id = str(
        (session_metadata or {}).get("pipeline_run_id") or ""
    ).strip()
    previous_assessments = (
        (session_metadata or {}).get("source_rucam_assessments")
        if isinstance((session_metadata or {}).get("source_rucam_assessments"), list)
        else []
    )
    assessment_summaries: list[str] = []
    for item in previous_assessments[:5]:
        if not isinstance(item, dict):
            continue
        drug_name = str(item.get("drug_name") or "").strip()
        causality = str(
            item.get("causality_category")
            or item.get("causality_assessment")
            or ""
        ).strip()
        if drug_name and causality:
            assessment_summaries.append(f"{drug_name}: {causality}")
    context_metadata: dict[str, Any] = {
        "source_version_id": source_version_id,
        "revision_version_id": revision_version_id,
        "pipeline_run_id": pipeline_run_id or None,
        "has_previous_report": bool(previous_report),
        "previous_assessment_count": len(previous_assessments),
        "previous_assessment_summaries": assessment_summaries,
    }
    chunks = [structured_context]
    snapshot_context = str(revision_snapshot_context or "").strip()
    if snapshot_context:
        chunks.append("Revision entity snapshot:\n" + snapshot_context)
    focus_context = str(revision_focus_context or "").strip()
    if focus_context:
        chunks.append("Revision focus context:\n" + focus_context)
    revision_metadata_lines: list[str] = []
    if source_version_id is not None:
        revision_metadata_lines.append(f"Source version: {source_version_id}")
    if revision_version_id is not None:
        revision_metadata_lines.append(f"Revision version: {revision_version_id}")
    if pipeline_run_id:
        revision_metadata_lines.append(f"Pipeline run: {pipeline_run_id}")
    if revision_metadata_lines:
        chunks.append(
            "Revision metadata:\n" + "\n".join(revision_metadata_lines)
        )
    if previous_report:
        chunks.append(
            "Previous report for comparison only:\n" + previous_report
        )
    if assessment_summaries:
        chunks.append(
            "Previous per-drug assessments for comparison only:\n"
            + "\n".join(assessment_summaries)
        )
    chunks.append(
        "Revision evidence handling:\n"
        "Use source evidence and revised structured artifacts as primary support.\n"
        "Treat previous report content as comparison context only."
    )
    return "\n\n".join(chunks), context_metadata


def _build_revision_consultation_inputs(
    *,
    structured_context: str,
    entity_pipeline: dict[str, Any] | None,
    revision_focus_context: str | None,
    session_metadata: dict[str, Any] | None,
    analysis_drugs: PatientDrugs,
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
) -> RevisionConsultationInputs:
    snapshot_context = _build_revision_snapshot_context(entity_pipeline)
    consultation_context, context_metadata = _build_revision_consultation_context(
        structured_context=structured_context,
        revision_snapshot_context=snapshot_context,
        revision_focus_context=revision_focus_context,
        session_metadata=session_metadata,
    )
    consultation_drugs = _build_revision_consultation_drugs(
        entity_pipeline=entity_pipeline,
        analysis_drugs=analysis_drugs,
        therapy_drugs=therapy_drugs,
        anamnesis_drugs=anamnesis_drugs,
    )
    return RevisionConsultationInputs(
        analysis_drugs=consultation_drugs,
        snapshot_context=snapshot_context,
        consultation_context=consultation_context,
        context_metadata=context_metadata,
    )


def _build_revision_consultation_execution_payload(
    consultation_inputs: RevisionConsultationInputs,
    service_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "analysis_drug_names": [
            entry.name
            for entry in consultation_inputs.analysis_drugs.entries
            if entry.name
        ],
        "snapshot_context_present": bool(
            str(consultation_inputs.snapshot_context or "").strip()
        ),
        "consultation_context_length": len(
            str(consultation_inputs.consultation_context or "").strip()
        ),
        "source_version_id": consultation_inputs.context_metadata.get(
            "source_version_id"
        ),
        "revision_version_id": consultation_inputs.context_metadata.get(
            "revision_version_id"
        ),
        "pipeline_run_id": consultation_inputs.context_metadata.get("pipeline_run_id"),
        "has_previous_report_context": bool(
            consultation_inputs.context_metadata.get("has_previous_report")
        ),
        "previous_assessment_count": int(
            consultation_inputs.context_metadata.get("previous_assessment_count") or 0
        ),
    }
    if isinstance(service_payload, dict):
        payload.update(service_payload)
    return payload


async def _run_revision_consultation(
    service: Any,
    *,
    payload: Any,
    structured_context: str,
    entity_pipeline: dict[str, Any] | None,
    revision_focus_context: str | None,
    session_metadata: dict[str, Any] | None,
    analysis_drugs: PatientDrugs,
    therapy_drugs: PatientDrugs,
    anamnesis_drugs: PatientDrugs,
    prepared_inputs: Any,
    report_language: str,
    rag_query: dict[str, str],
    rucam_bundle: PatientRucamAssessmentBundle,
    issues: list[Any],
    stop_check: Any,
) -> RevisionConsultationExecution:
    consultation_inputs = _build_revision_consultation_inputs(
        structured_context=structured_context,
        entity_pipeline=entity_pipeline,
        revision_focus_context=revision_focus_context,
        session_metadata=session_metadata,
        analysis_drugs=analysis_drugs,
        therapy_drugs=therapy_drugs,
        anamnesis_drugs=anamnesis_drugs,
    )
    revision_consultation_runner = getattr(
        service, "run_revision_consultation", None
    )
    service_payload: dict[str, Any] | None = None
    if callable(revision_consultation_runner):
        clinical_session, final_report, service_payload = (
            await revision_consultation_runner(
                payload=payload,
                analysis_drugs=consultation_inputs.analysis_drugs,
                prepared_inputs=prepared_inputs,
                consultation_context=consultation_inputs.consultation_context,
                consultation_context_metadata=consultation_inputs.context_metadata,
                report_language=report_language,
                rag_query=rag_query,
                rucam_bundle=rucam_bundle,
                issues=issues,
                progress_callback=None,
                stop_check=stop_check,
            )
        )
    else:
        clinical_session, final_report = await service.run_consultation(
            payload=payload,
            analysis_drugs=consultation_inputs.analysis_drugs,
            prepared_inputs=prepared_inputs,
            consultation_context=consultation_inputs.consultation_context,
            report_language=report_language,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            issues=issues,
            progress_callback=None,
            stop_check=stop_check,
        )
    return RevisionConsultationExecution(
        inputs=consultation_inputs,
        clinical_session=clinical_session,
        final_report=final_report,
        payload=_build_revision_consultation_execution_payload(
            consultation_inputs,
            service_payload=service_payload,
        ),
    )


def _build_revision_finalization_payload(
    *,
    final_report: str,
    generated_report: str,
    report_comparison_payload: dict[str, Any],
    faithfulness_audit: Any,
    clinical_session: Any,
) -> dict[str, Any]:
    return {
        "final_report_present": bool(str(final_report or "").strip()),
        "generated_report_present": bool(str(generated_report or "").strip()),
        "manual_review_required": bool(
            getattr(faithfulness_audit, "manual_review_required", False)
        ),
        "blocking_issue_count": len(
            list(getattr(faithfulness_audit, "blocking_issues", []) or [])
        ),
        "comparison_outcome": str(
            report_comparison_payload.get("outcome") or ""
        ).strip()
        or None,
        "consultation_model": getattr(clinical_session, "llm_model", None),
    }


def _finalize_revision_report_outputs(
    *,
    report_metadata: Any,
    faithfulness_audit: Any,
    report_comparison_payload: dict[str, Any],
    generated_report: str,
    final_report: str,
    report_language: str,
    clinical_session: Any,
) -> RevisionFinalizationOutputs:
    resolved_final_report = final_report or phrase(
        "narrative_fallback", report_language
    )
    return RevisionFinalizationOutputs(
        final_report=resolved_final_report,
        generated_report=generated_report,
        report_metadata=report_metadata,
        faithfulness_audit=faithfulness_audit,
        report_comparison_payload=report_comparison_payload,
        payload=_build_revision_finalization_payload(
            final_report=resolved_final_report,
            generated_report=generated_report,
            report_comparison_payload=report_comparison_payload,
            faithfulness_audit=faithfulness_audit,
            clinical_session=clinical_session,
        ),
    )
