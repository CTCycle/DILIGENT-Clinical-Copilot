from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from common.exceptions import (
    ServiceConflictError,
    ServiceError,
    ServiceValidationError,
)
from common.utils.languages import (
    MISSING_VISIT_LABEL_BY_LANGUAGE,
    resolve_supported_language_code,
)
from common.utils.logger import logger
from configurations.llm_configs import LLMRuntimeConfig
from configurations.startup import server_settings
from domain.clinical.entities import (
    ClinicalPipelineValidationError,
    ClinicalSectionExtractionResult,
    ClinicalSessionRequest,
    DrugRucamAssessment,
    PatientData,
    PatientRucamAssessmentBundle,
)
from domain.jobs import JobStartResponse
from services.clinical.candidate_selection import select_relevant_candidates
from services.clinical.language import ClinicalLanguageDetector
from services.clinical.match_quality import classify_match_evidence
from services.session.clinical_input_extractor import ClinicalInputExtractionError
from services.session.session_shared import NarrativeBuilder, run_clinical_job
from services.security.access_keys import AccessKeyService
from services.text.normalization import normalize_drug_query_name


def build_single_matched_drug_row_workflow(
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
        "match_confidence": match_confidence,
        "match_reason": resolved.get("match_reason"),
        "match_notes": match_notes,
        "match_status": resolved.get("match_status"),
        "evidence_quality": match_quality["evidence_quality"],
        "evidence_warnings": match_quality["evidence_warnings"],
        "match_candidates": resolved.get("match_candidates", []),
        "chosen_candidate": resolved.get("chosen_candidate"),
        "rejected_candidates": resolved.get("rejected_candidates", []),
        "missing_livertox": resolved.get("missing_livertox", True),
        "ambiguous_match": resolved.get("ambiguous_match", False),
        "regimen_group_ids": resolved.get("regimen_group_ids", []),
        "regimen_components": resolved.get("regimen_components", []),
        "origins": resolved.get("origins", []),
        "raw_mentions": resolved.get("raw_mentions", []),
        "rucam": rucam_entry.model_dump() if rucam_entry is not None else None,
    }


def build_matched_drugs_payload_workflow(
    service: Any,
    *,
    detected_drugs: list[str],
    prepared_inputs,
    rucam_bundle: PatientRucamAssessmentBundle,
) -> list[dict[str, Any]]:
    resolved_drug_map = service._normalized_resolved_drug_map(prepared_inputs)
    rucam_by_name = service._normalized_rucam_map(rucam_bundle)
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
            build_single_matched_drug_row_workflow(
                detected_name=detected_name,
                resolved=resolved,
                rucam_entry=rucam_entry,
            )
        )
    return matched_drugs_payload


async def process_single_patient_workflow(
    service: Any,
    payload: PatientData,
    *,
    patient_image_base64: str | None = None,
    section_extraction: ClinicalSectionExtractionResult | None = None,
    session_version: int = 1,
    original_session_id: int | None = None,
    session_metadata: dict[str, Any] | None = None,
    original_session_text: str | None = None,
    revision_focus_context: str | None = None,
    progress_callback=None,
    stop_check=None,
) -> dict[str, Any]:
    service.run_stop_check(stop_check)
    logger.info(
        "Starting Drug-Induced Liver Injury (DILI) analysis for patient: %s",
        payload.name,
    )

    global_start_time = time.perf_counter()
    service.emit_progress(progress_callback, stage="session_initialization", value=5.0)
    language_result = ClinicalLanguageDetector.detect(payload)
    report_language = language_result.report_language
    validation_bundle = service.build_validation_bundle_for_payload(payload)
    service.ensure_submission_requirements(payload)
    service.run_stop_check(stop_check)

    issues = []
    cleaned_therapy_text = service.drugs_parser.clean_text(payload.drugs or "")
    therapy_drugs = await service.extract_therapy_drugs(
        cleaned_therapy_text=cleaned_therapy_text,
        issues=issues,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )
    anamnesis_text = payload.anamnesis or ""
    anamnesis_drugs = await service.extract_anamnesis_drugs(
        anamnesis_text=anamnesis_text,
        issues=issues,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )
    disease_context = await service.extract_disease_context(
        anamnesis_text=anamnesis_text,
        issues=issues,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )
    lab_timeline, onset_context = await service.extract_lab_timeline(
        payload=payload,
        issues=issues,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )
    pattern_assessment = service.assess_pattern(
        lab_timeline=lab_timeline,
        validation_bundle=validation_bundle,
        issues=issues,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )
    pattern_score = pattern_assessment.score
    all_detected_drugs = type(therapy_drugs)(
        entries=[*therapy_drugs.entries, *anamnesis_drugs.entries]
    )
    candidate_selection = select_relevant_candidates(
        therapy_drugs=therapy_drugs,
        anamnesis_drugs=anamnesis_drugs,
        visit_date=payload.visit_date,
    )
    analysis_drugs = candidate_selection.ordered_analysis_drugs
    rucam_bundle = service.estimate_rucam(
        payload=payload,
        analysis_drugs=analysis_drugs,
        anamnesis_drugs=anamnesis_drugs,
        disease_context=disease_context,
        lab_timeline=lab_timeline,
        onset_context=onset_context,
        pattern_score=pattern_score,
        report_language=report_language,
        issues=issues,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )
    structured_context = service.build_structured_clinical_context(
        payload,
        therapy_drugs=therapy_drugs,
        anamnesis_drugs=anamnesis_drugs,
        disease_context=disease_context,
        lab_timeline=lab_timeline,
        onset_context=onset_context,
        pattern_score=pattern_score,
    )
    if revision_focus_context:
        structured_context = (
            f"{structured_context}\n\n"
            "Revision focus context:\n"
            f"{revision_focus_context.strip()}"
        )
    rag_query = service.build_rag_query(
        payload=payload,
        analysis_drugs=analysis_drugs,
        structured_context=structured_context,
        pattern_score=pattern_score,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )
    prepared_inputs = await service.run_livertox_lookup(
        all_detected_drugs=all_detected_drugs,
        structured_context=structured_context,
        pattern_score=pattern_score,
        issues=issues,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )
    rucam_bundle = service.reestimate_rucam_with_livertox(
        payload=payload,
        analysis_drugs=analysis_drugs,
        anamnesis_drugs=anamnesis_drugs,
        disease_context=disease_context,
        lab_timeline=lab_timeline,
        onset_context=onset_context,
        pattern_score=pattern_score,
        report_language=report_language,
        prepared_inputs=prepared_inputs,
        rucam_bundle=rucam_bundle,
        issues=issues,
    )
    clinical_session, final_report = await service.run_consultation(
        payload=payload,
        analysis_drugs=analysis_drugs,
        prepared_inputs=prepared_inputs,
        report_language=report_language,
        rag_query=rag_query,
        rucam_bundle=rucam_bundle,
        issues=issues,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )

    patient_label = payload.name or "Unknown patient"
    report_language_key = resolve_supported_language_code(report_language)
    visit_label = (
        payload.visit_date.strftime("%d %B %Y")
        if payload.visit_date
        else MISSING_VISIT_LABEL_BY_LANGUAGE.get(report_language_key, "Not provided")
    )
    global_elapsed = time.perf_counter() - global_start_time
    detected_drugs = [entry.name for entry in analysis_drugs.entries if entry.name]
    anamnesis_detected_drugs = [entry.name for entry in anamnesis_drugs.entries if entry.name]
    anamnesis_detected_diseases = [entry.name for entry in disease_context.entries if entry.name]
    matched_drugs_payload = build_matched_drugs_payload_workflow(
        service,
        detected_drugs=detected_drugs,
        prepared_inputs=prepared_inputs,
        rucam_bundle=rucam_bundle,
    )
    serialized_issues = service.serialize_pipeline_issues(issues)
    pattern_strings = service.pattern_analyzer.stringify_scores(pattern_score)
    narrative = NarrativeBuilder.build_patient_narrative(
        patient_label=patient_label,
        visit_label=visit_label,
        anamnesis=payload.anamnesis,
        drugs_text=payload.drugs,
        pattern_score=pattern_score,
        pattern_strings=pattern_strings,
        detected_drugs=detected_drugs,
        anamnesis_detected_drugs=anamnesis_detected_drugs,
        rucam_assessments=rucam_bundle.entries,
        report_language=report_language,
        issues=issues,
        final_report=final_report,
    )
    result_payload = {
        "report": narrative,
        "issues": serialized_issues,
        "pattern_status": pattern_assessment.status,
        "detected_drugs": detected_drugs,
        "anamnesis_drugs": anamnesis_detected_drugs,
        "anamnesis_diseases": anamnesis_detected_diseases,
        "matched_drugs": matched_drugs_payload,
        "rucam_assessments": [item.model_dump() for item in rucam_bundle.entries],
        "lab_timeline": [entry.model_dump() for entry in lab_timeline.entries],
        "onset_context": onset_context.model_dump() if onset_context else None,
        "detected_input_language": language_result.detected_input_language,
        "report_language": language_result.report_language,
        "relevant_drugs": candidate_selection.relevant,
        "excluded_drugs": candidate_selection.excluded,
        "unresolved_drugs": candidate_selection.unresolved,
        "structured_case": {
            "therapy_drugs": [entry.model_dump() for entry in therapy_drugs.entries],
            "anamnesis_drugs": [entry.model_dump() for entry in anamnesis_drugs.entries],
            "anamnesis_diseases": [entry.model_dump() for entry in disease_context.entries],
        },
        "section_extraction": (
            section_extraction.model_dump() if section_extraction is not None else None
        ),
        "runtime_settings": {
            "use_cloud_services": LLMRuntimeConfig.is_cloud_enabled(),
            "llm_provider": LLMRuntimeConfig.get_llm_provider(),
            "cloud_model": LLMRuntimeConfig.get_cloud_model(),
            "text_extraction_model": LLMRuntimeConfig.get_text_extraction_model(),
            "clinical_model": LLMRuntimeConfig.get_clinical_model(),
            "ollama_temperature": LLMRuntimeConfig.get_ollama_temperature(),
            "cloud_temperature": LLMRuntimeConfig.get_cloud_temperature(),
            "ollama_reasoning": LLMRuntimeConfig.is_ollama_reasoning_enabled(),
        },
        "revision": {
            "version": session_version,
            "original_session_id": original_session_id,
            "metadata": session_metadata or {},
            "focus_context": revision_focus_context,
        },
    }
    if original_session_text is not None:
        result_payload["original_session_text"] = original_session_text
    persisted_session_id = await asyncio.to_thread(
        service.serializer.save_clinical_session,
        {
            "patient_name": payload.name,
            "patient_visit_date": payload.visit_date,
            "patient_image_base64": patient_image_base64,
            "session_timestamp": datetime.now(),
            "version": session_version,
            "original_session_id": original_session_id,
            "metadata": session_metadata or {},
            "hepatic_pattern": pattern_score.classification,
            "anamnesis": payload.anamnesis,
            "drugs": payload.drugs,
            "laboratory_analysis": payload.laboratory_analysis,
            "section_extraction": (
                section_extraction.model_dump() if section_extraction is not None else None
            ),
            "text_extraction_model": getattr(service.drugs_parser, "model", None),
            "clinical_model": getattr(clinical_session, "llm_model", None),
            "total_duration": global_elapsed,
            "final_report": final_report,
            "detected_drugs": detected_drugs,
            "matched_drugs": matched_drugs_payload,
            "issues": serialized_issues,
            "session_status": "successful",
            "session_result_payload": result_payload,
        },
    )
    if persisted_session_id is not None:
        result_payload["session_id"] = persisted_session_id
    return result_payload


def start_clinical_job_workflow(
    service: Any,
    request_payload: ClinicalSessionRequest,
) -> JobStartResponse:
    if service.job_manager.is_job_running(service.JOB_TYPE):
        raise ServiceConflictError("Clinical analysis is already in progress")

    service.apply_persisted_runtime_configuration()
    if LLMRuntimeConfig.is_cloud_enabled():
        provider = LLMRuntimeConfig.get_llm_provider()
        active_keys = [
            item
            for item in AccessKeyService().list_access_keys(provider)
            if item.is_active
        ]
        if not active_keys:
            raise ServiceValidationError(
                f"Configure an active {provider.title()} access key before running cloud analysis."
            )

    try:
        preprocessed_request, section_extraction = asyncio.run(
            service.preprocess_unified_input(request_payload)
        )
    except ClinicalInputExtractionError as exc:
        raise ServiceValidationError(str(exc)) from exc
    patient_payload = service.build_patient_payload(preprocessed_request)
    try:
        service.ensure_submission_requirements(patient_payload)
    except ClinicalPipelineValidationError as exc:
        raise ServiceValidationError(service.serialize_pipeline_issues(exc.issues)) from exc
    job_id = service.job_manager.start_job(
        job_type=service.JOB_TYPE,
        runner=run_clinical_job,
        kwargs={
            "service": service,
            "payload": patient_payload,
            "patient_image_base64": request_payload.patient_image_base64,
            "section_extraction": section_extraction,
        },
    )
    job_status = service.job_manager.get_job_status(job_id)
    if job_status is None:
        raise ServiceError("Failed to initialize clinical analysis job")
    return JobStartResponse(
        job_id=job_id,
        job_type=job_status["job_type"],
        status=job_status["status"],
        message="Clinical analysis job started",
        poll_interval=server_settings.jobs.polling_interval,
    )
