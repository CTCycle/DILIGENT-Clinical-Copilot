from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any

from typing import cast
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
from services.llm.runtime_config import LLMRuntimeConfig
from configurations.startup import get_server_settings
from domain.clinical.entities import (
    ClinicalPipelineValidationError,
    ClinicalSectionExtractionResult,
    ClinicalSessionRequest,
    PatientData,
    PatientDrugs,
    PipelineIssue,
)
from domain.clinical.robustness import NormalizedDocument
from domain.jobs import JobStartResponse
from services.clinical.candidate_selection import select_relevant_candidates
from services.clinical.deterministic_extraction import extract_deterministic_diseases
from services.clinical.dili_evidence import DiliEvidenceBuilder
from services.clinical.drug_deduplication import (
    build_deduplication_audit,
    deduplicate_detected_drugs,
)
from services.clinical.language import ClinicalLanguageDetector
from services.clinical.pattern_resolution import (
    HepaticPatternResolutionInput,
    copy_pattern_score_with_resolution,
    resolve_hepatic_pattern,
)
from services.clinical.report_language import phrase
from services.security.access_keys import AccessKeyService
from services.session.clinical_input_extractor import ClinicalInputExtractionError
from services.session.document_normalizer import DocumentNormalizer
from services.session.preflight import check_parser_batch_capacity
from services.retrieval.readiness import check_rag_readiness
from services.session.robust_pipeline import (
    audit_report,
    build_extraction_artifact,
    build_fact_graph,
    build_run_bundle_index,
    render_fact_graph_report,
    validate_fact_graph,
)
from services.session.session_shared import NarrativeBuilder, run_clinical_job
from services.session.workflow_shared import (
    ClinicalPersistenceError,
    PROGRESS_SEQUENCE as _PROGRESS_SEQUENCE,
    append_warning_issue as _append_warning_issue,
    build_matched_drugs_payload as build_matched_drugs_payload_workflow,
    emit_progress as _emit_progress,
    extract_deterministic_drugs as _extract_deterministic_drugs,
    has_temporal_information as _has_temporal_information,
    resolve_rucam_source as _resolve_rucam_source,
)

_CLOUD_PROVIDERS = {"openai", "gemini"}

###############################################################################
async def process_single_patient_workflow(
    service: Any,
    payload: PatientData,
    *,
    patient_image_base64: str | None = None,
    section_extraction: ClinicalSectionExtractionResult | None = None,
    normalized_document: NormalizedDocument | None = None,
    report_mode: str = "faithful_only",
    session_version: int = 1,
    original_session_id: int | None = None,
    progress_callback=None,
    stop_check=None,
) -> dict[str, Any]:
    service.run_stop_check(stop_check)
    logger.info(
        "Starting Drug-Induced Liver Injury (DILI) analysis for patient: %s",
        payload.name,
    )

    global_start_time = time.perf_counter()
    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[1][1],
        _PROGRESS_SEQUENCE[1][0],
    )
    language_result = ClinicalLanguageDetector.detect(payload)
    report_language = language_result.report_language
    validation_bundle = service.build_validation_bundle_for_payload(payload)
    service.ensure_submission_requirements(payload)
    service.run_stop_check(stop_check)
    if normalized_document is None:
        normalized_document = DocumentNormalizer().normalize(
            section_extraction.source_text if section_extraction is not None else ""
        )
    extraction_artifact = build_extraction_artifact(
        normalized_document=normalized_document,
        section_extraction=section_extraction,
        payload=payload,
    )

    issues = []
    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[2][1],
        _PROGRESS_SEQUENCE[2][0],
    )
    cleaned_therapy_text = service.drugs_parser.clean_text(payload.drugs or "")
    cleaned_anamnesis_text = service.drugs_parser.clean_text(payload.anamnesis or "")

    therapy_deterministic = _extract_deterministic_drugs(
        service, text=cleaned_therapy_text, source="therapy"
    )
    anamnesis_deterministic = _extract_deterministic_drugs(
        service, text=cleaned_anamnesis_text, source="anamnesis"
    )
    disease_deterministic = extract_deterministic_diseases(cleaned_anamnesis_text)

    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[3][1],
        _PROGRESS_SEQUENCE[3][0],
    )

    preflight = await check_parser_batch_capacity(task_count=2)
    if preflight.concurrency_allowed:
        anamnesis_drugs, therapy_drugs = await asyncio.gather(
            service.extract_anamnesis_drugs(
                anamnesis_text=cleaned_anamnesis_text,
                issues=issues,
                progress_callback=progress_callback,
                stop_check=stop_check,
            ),
            service.extract_therapy_drugs(
                cleaned_therapy_text=cleaned_therapy_text,
                issues=issues,
                progress_callback=progress_callback,
                stop_check=stop_check,
            ),
        )
    else:
        logger.info(
            "Parser batch preflight denied concurrency for provider=%s model=%s: %s",
            preflight.provider,
            preflight.model,
            preflight.reason,
        )
        _append_warning_issue(
            service,
            issues,
            code="parser_batch_preflight_sequential_fallback",
            message="Parser batch preflight denied concurrent extraction; using sequential extraction for local runtime safety.",
            field="clinical_input",
        )
        therapy_drugs = await service.extract_therapy_drugs(
            cleaned_therapy_text=cleaned_therapy_text,
            issues=issues,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )
        anamnesis_drugs = await service.extract_anamnesis_drugs(
            anamnesis_text=cleaned_anamnesis_text,
            issues=issues,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )

    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[4][1],
        _PROGRESS_SEQUENCE[4][0],
    )

    anamnesis_text = payload.anamnesis or ""
    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[5][1],
        _PROGRESS_SEQUENCE[5][0],
    )
    disease_context = await service.extract_disease_context(
        anamnesis_text=anamnesis_text,
        issues=issues,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )

    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[6][1],
        _PROGRESS_SEQUENCE[6][0],
    )
    lab_timeline, onset_context = await service.extract_lab_timeline(
        payload=payload,
        issues=issues,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )

    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[7][1],
        _PROGRESS_SEQUENCE[7][0],
    )
    pattern_assessment = service.assess_pattern(
        lab_timeline=lab_timeline,
        validation_bundle=validation_bundle,
        issues=issues,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )
    calculated_pattern_score = pattern_assessment.score
    pattern_score = calculated_pattern_score
    explicit_hepatic_pattern = None
    lab_extractor = getattr(service, "lab_extractor", None)
    if (
        lab_extractor is not None
        and hasattr(lab_extractor, "extract_explicit_hepatic_pattern")
        and payload.laboratory_analysis
    ):
        try:
            explicit_hepatic_pattern = lab_extractor.extract_explicit_hepatic_pattern(
                payload.laboratory_analysis
            )
        except Exception:
            explicit_hepatic_pattern = None
    hepatic_pattern_resolution = resolve_hepatic_pattern(
        HepaticPatternResolutionInput(
            explicit_pattern=explicit_hepatic_pattern,
            calculated_pattern=calculated_pattern_score.classification,
            r_score=calculated_pattern_score.r_score,
        )
    )
    issues.extend(hepatic_pattern_resolution.warnings)
    pattern_score = copy_pattern_score_with_resolution(
        calculated_pattern_score,
        hepatic_pattern_resolution,
    )
    temporal_uncertain_count = sum(
        1
        for entry in [*anamnesis_drugs.entries, *therapy_drugs.entries]
        if not _has_temporal_information(service, entry)
    )
    filtered_out_count = 0
    if temporal_uncertain_count > 0:
        _append_warning_issue(
            service,
            issues,
            code="drugs_missing_temporal_information_present",
            message=f"{temporal_uncertain_count} extracted drug entries have uncertain temporal information and are reported with reduced causal confidence.",
            field="drugs",
        )

    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[8][1],
        _PROGRESS_SEQUENCE[8][0],
    )
    deduplicated_drugs = deduplicate_detected_drugs(
        therapy_drugs,
        anamnesis_drugs,
        payload.visit_date,
    )
    candidate_selection = select_relevant_candidates(
        therapy_drugs=deduplicated_drugs,
        anamnesis_drugs=PatientDrugs(entries=[]),
        visit_date=payload.visit_date,
    )

    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[9][1],
        _PROGRESS_SEQUENCE[9][0],
    )
    analysis_drugs = candidate_selection.ordered_analysis_drugs
    effective_candidate_selection = candidate_selection

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
    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[10][1],
        _PROGRESS_SEQUENCE[10][0],
    )
    rag_query = service.build_rag_query(
        payload=payload,
        analysis_drugs=analysis_drugs,
        structured_context=structured_context,
        pattern_score=pattern_score,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )
    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[11][1],
        _PROGRESS_SEQUENCE[11][0],
    )
    prepared_inputs = await service.run_livertox_lookup(
        all_detected_drugs=analysis_drugs,
        structured_context=structured_context,
        pattern_score=pattern_score,
        issues=issues,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )
    match_audit_issues: list[PipelineIssue] = []
    if prepared_inputs is not None and hasattr(service, "build_match_audit_issues"):
        match_audit_issues = service.build_match_audit_issues(
            prepared_inputs.resolved_drugs
        )
        issues.extend(match_audit_issues)
    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[12][1],
        _PROGRESS_SEQUENCE[12][0],
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
    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[13][1],
        _PROGRESS_SEQUENCE[13][0],
    )

    clinical_session, llm_clinical_summary = await service.run_consultation(
        payload=payload,
        analysis_drugs=analysis_drugs,
        prepared_inputs=prepared_inputs,
        consultation_context=structured_context,
        report_language=report_language,
        rag_query=rag_query,
        rucam_bundle=rucam_bundle,
        issues=issues,
        progress_callback=progress_callback,
        stop_check=stop_check,
    )
    dili_evidence_bundle = DiliEvidenceBuilder().build(
        payload=payload,
        drugs=analysis_drugs,
        labs=lab_timeline,
        resolved_drugs=(
            prepared_inputs.resolved_drugs if prepared_inputs is not None else None
        ),
        rucam_bundle=rucam_bundle,
    )
    structured_dili_report = DiliEvidenceBuilder.render(dili_evidence_bundle)
    final_report = structured_dili_report

    fact_graph = build_fact_graph(
        extraction_artifact=extraction_artifact,
        payload=payload,
        therapy_drugs=therapy_drugs,
        anamnesis_drugs=anamnesis_drugs,
        lab_timeline=lab_timeline,
        pattern_score=pattern_score,
        rucam_bundle=rucam_bundle,
    )
    fact_graph_validation = validate_fact_graph(fact_graph)
    generated_report, report_metadata = render_fact_graph_report(
        fact_graph=fact_graph,
        patient_name=payload.name,
        visit_date=payload.visit_date,
        report_mode=report_mode,
        report_language=report_language,
    )
    faithfulness_audit = audit_report(
        extraction_artifact=extraction_artifact,
        fact_graph_validation=fact_graph_validation,
        report_metadata=report_metadata,
    )
    try:
        report_comparison_payload = json.loads(faithfulness_audit.discrepancy_report)
    except Exception:
        report_comparison_payload = {
            "outcome": "comparison_not_possible",
            "agreements": ["Unable to parse structured comparison payload."],
            "omissions": ["Comparison payload is not structured JSON."],
            "differences": ["Falling back to raw discrepancy report text."],
            "unsupported": [
                faithfulness_audit.discrepancy_report or "No details available."
            ],
            "manual_review": "yes"
            if faithfulness_audit.manual_review_required
            else "no",
        }
    if faithfulness_audit.blocking_issues:
        issues.extend(
            PipelineIssue(
                severity="error",
                code=str(issue.get("code", "faithfulness_gate_blocked")),
                message=str(
                    issue.get("message", "Faithfulness gate blocked finalization.")
                )[:500],
            )
            for issue in faithfulness_audit.blocking_issues
        )
    if not final_report:
        final_report = phrase("narrative_fallback", report_language)

    patient_label = payload.name or "Unknown patient"
    report_language_key = resolve_supported_language_code(report_language)
    visit_label = (
        payload.visit_date.strftime("%d %B %Y")
        if payload.visit_date
        else MISSING_VISIT_LABEL_BY_LANGUAGE.get(report_language_key, "Not provided")
    )
    global_elapsed = time.perf_counter() - global_start_time
    detected_drugs = [entry.name for entry in analysis_drugs.entries if entry.name]
    anamnesis_detected_drugs = [
        entry.name for entry in anamnesis_drugs.entries if entry.name
    ]
    anamnesis_detected_diseases = [
        entry.name for entry in disease_context.entries if entry.name
    ]

    matched_drugs_payload = build_matched_drugs_payload_workflow(
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
        "final_report": final_report,
        "issues": serialized_issues,
        "pattern_status": pattern_assessment.status,
        "detected_drugs": detected_drugs,
        "anamnesis_drugs": anamnesis_detected_drugs,
        "anamnesis_diseases": anamnesis_detected_diseases,
        "matched_drugs": matched_drugs_payload,
        "rucam_assessments": [item.model_dump() for item in rucam_bundle.entries],
        "lab_timeline": [entry.model_dump() for entry in lab_timeline.entries],
        "onset_context": onset_context.model_dump() if onset_context else None,
        "dili_evidence_bundle": dili_evidence_bundle.model_dump(),
        "llm_clinical_summary": llm_clinical_summary,
        "detected_input_language": language_result.detected_input_language,
        "report_language": language_result.report_language,
        "relevant_drugs": effective_candidate_selection.relevant,
        "excluded_drugs": effective_candidate_selection.excluded,
        "unresolved_drugs": effective_candidate_selection.unresolved,
        "extraction_metadata": {
            "drug_filtering": {
                "filtered_out_count": filtered_out_count,
                "temporal_uncertain_count": temporal_uncertain_count,
                "reason": "temporal_uncertainty_retained_with_low_confidence",
            },
            "hepatic_pattern": {
                "value": hepatic_pattern_resolution.final_value,
                "explicit_value": hepatic_pattern_resolution.explicit_value,
                "calculated_value": hepatic_pattern_resolution.calculated_value,
                "final_value": hepatic_pattern_resolution.final_value,
                "source": hepatic_pattern_resolution.source,
                "conflict": hepatic_pattern_resolution.conflict,
                "r_score": hepatic_pattern_resolution.r_score,
            },
            "rucam": {"source": _resolve_rucam_source(rucam_bundle.entries)},
        },
        "structured_case": {
            "therapy_drugs": [entry.model_dump() for entry in therapy_drugs.entries],
            "anamnesis_drugs": [
                entry.model_dump() for entry in anamnesis_drugs.entries
            ],
            "anamnesis_diseases": [
                entry.model_dump() for entry in disease_context.entries
            ],
            "deduplicated_analysis_drugs": build_deduplication_audit(
                therapy_drugs,
                anamnesis_drugs,
                deduplicated_drugs,
            ),
        },
        "deterministic_extraction": {
            "therapy": {
                "entries": [
                    entry.model_dump() for entry in therapy_deterministic.entries
                ],
                "unresolved_lines": therapy_deterministic.unresolved_lines,
            },
            "anamnesis": {
                "entries": [
                    entry.model_dump() for entry in anamnesis_deterministic.entries
                ],
                "regimen_lines": anamnesis_deterministic.regimen_lines,
                "unresolved_lines": anamnesis_deterministic.unresolved_lines,
            },
            "diseases": {
                "entries": [
                    entry.model_dump()
                    for entry in disease_deterministic.context.entries
                ],
                "matched_lines": disease_deterministic.matched_lines,
                "unresolved_lines": disease_deterministic.unresolved_lines,
            },
        },
        "section_extraction": section_extraction.model_dump()
        if section_extraction is not None
        else None,
        "resolved_runtime": dict(getattr(service, "resolved_runtime", {})),
        "stage_elapsed_ms": dict(getattr(service, "stage_elapsed_ms", {})),
        "fallback_reasons": dict(getattr(service, "fallback_reasons", {})),
        "structured_failure_kind": dict(
            getattr(service, "structured_failure_kind", {})
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
            "use_rag": bool(payload.use_rag),
            "resolved_runtime": dict(getattr(service, "resolved_runtime", {})),
            "stage_elapsed_ms": dict(getattr(service, "stage_elapsed_ms", {})),
            "fallback_reasons": dict(getattr(service, "fallback_reasons", {})),
            "structured_failure_kind": dict(
                getattr(service, "structured_failure_kind", {})
            ),
        },
        "manual_review_required": faithfulness_audit.manual_review_required,
        "blocking_issues": faithfulness_audit.blocking_issues,
        "report_comparison": report_comparison_payload,
        "pipeline_artifacts": {
            "normalized_document": normalized_document.model_dump(),
            "extraction_artifact": extraction_artifact.model_dump(),
            "section_extraction_audit": (
                section_extraction.metadata if section_extraction is not None else None
            ),
            "extraction_strategy_decisions": {
                "laboratory_history": getattr(
                    service,
                    "latest_lab_extraction_audit",
                    None,
                ),
            },
            "hepatic_pattern_resolution": hepatic_pattern_resolution.model_dump(),
            "match_audit": {
                "issues": [issue.model_dump() for issue in match_audit_issues],
                "resolved_drug_count": (
                    len(prepared_inputs.resolved_drugs)
                    if prepared_inputs is not None
                    else 0
                ),
            },
            "dili_evidence_bundle": dili_evidence_bundle.model_dump(),
            "llm_clinical_summary": llm_clinical_summary,
            "deterministic_extraction": {
                "therapy": {
                    "entries": [
                        entry.model_dump() for entry in therapy_deterministic.entries
                    ],
                    "unresolved_lines": therapy_deterministic.unresolved_lines,
                },
                "anamnesis": {
                    "entries": [
                        entry.model_dump() for entry in anamnesis_deterministic.entries
                    ],
                    "regimen_lines": anamnesis_deterministic.regimen_lines,
                    "unresolved_lines": anamnesis_deterministic.unresolved_lines,
                },
                "diseases": {
                    "entries": [
                        entry.model_dump()
                        for entry in disease_deterministic.context.entries
                    ],
                    "matched_lines": disease_deterministic.matched_lines,
                    "unresolved_lines": disease_deterministic.unresolved_lines,
                },
            },
            "fact_graph": fact_graph.model_dump(),
            "fact_graph_validation": fact_graph_validation.model_dump(),
            "generated_report": generated_report,
            "llm_clinical_summary": llm_clinical_summary,
            "report_metadata": report_metadata.model_dump(),
            "faithfulness_audit": faithfulness_audit.model_dump(),
            "discrepancy_report": faithfulness_audit.discrepancy_report,
        },
    }
    result_payload["run_bundle_index"] = build_run_bundle_index(
        run_id="pending", session_id=None
    ).model_dump()
    _emit_progress(
        progress_callback,
        "clinical",
        _PROGRESS_SEQUENCE[14][1],
        _PROGRESS_SEQUENCE[14][0],
    )

    persisted_session_id = None
    try:
        persisted_session_id = await asyncio.to_thread(
            service.serializer.save_clinical_session,
            {
                "patient_name": payload.name,
                "patient_visit_date": payload.visit_date,
                "patient_image_base64": patient_image_base64,
                "session_timestamp": datetime.now(),
                "version": session_version,
                "original_session_id": original_session_id,
                "metadata": {"use_rag": bool(payload.use_rag)},
                "session_kind": "standard_assessment",
                "hepatic_pattern": pattern_score.classification,
                "anamnesis": payload.anamnesis,
                "drugs": payload.drugs,
                "laboratory_analysis": payload.laboratory_analysis,
                "section_extraction": section_extraction.model_dump()
                if section_extraction is not None
                else None,
                "text_extraction_model": getattr(service.drugs_parser, "model", None),
                "clinical_model": getattr(clinical_session, "llm_model", None),
                "total_duration": global_elapsed,
                "final_report": final_report,
                "llm_clinical_summary": llm_clinical_summary,
                "detected_drugs": detected_drugs,
                "matched_drugs": matched_drugs_payload,
                "issues": serialized_issues,
                "session_status": "successful",
                "session_result_payload": result_payload,
            },
        )
        if persisted_session_id is not None:
            result_payload["session_id"] = persisted_session_id
            result_payload["run_bundle_index"] = build_run_bundle_index(
                run_id=str(persisted_session_id),
                session_id=persisted_session_id,
            ).model_dump()
            await asyncio.to_thread(
                service.serializer.upsert_session_result_payload,
                persisted_session_id,
                result_payload,
            )
    except Exception as exc:
        logger.warning(
            "Clinical assessment persistence failed: error_type=%s",
            type(exc).__name__,
        )
        raise ClinicalPersistenceError() from exc
    if persisted_session_id is None:
        logger.warning("Clinical assessment persistence returned no session id.")
        raise ClinicalPersistenceError()
    return result_payload

###############################################################################
def start_clinical_job_workflow(
    service: Any,
    request_payload: ClinicalSessionRequest,
) -> JobStartResponse:
    if service.job_manager.is_job_running(
        service.JOB_TYPE, scope_key="clinical:global"
    ):
        raise ServiceConflictError("Clinical analysis is already in progress")

    service.apply_persisted_runtime_configuration()
    preflight = service.validate_clinical_input(request_payload)
    if not preflight.ready:
        raise ServiceValidationError(
            [issue.model_dump() for issue in preflight.blocking_issues]
        )
    rag_readiness = check_rag_readiness(requested=request_payload.use_rag)
    if request_payload.use_rag and not rag_readiness.available:
        raise ServiceValidationError(
            [
                {
                    "severity": "blocking",
                    "code": rag_readiness.reason_code or "rag_unavailable",
                    "message": rag_readiness.message or "RAG is unavailable.",
                    "field": "use_rag",
                }
            ]
        )
    if LLMRuntimeConfig.is_cloud_enabled():
        provider = LLMRuntimeConfig.get_llm_provider().strip().lower()
        if provider not in _CLOUD_PROVIDERS:
            raise ServiceValidationError(
                f"Unsupported cloud provider '{provider}' for access-key validation."
            )
        active_keys = [
            item
            for item in AccessKeyService().list_access_keys(cast(Any, provider))
            if item.is_active
        ]
        if not active_keys:
            raise ServiceValidationError(
                f"Configure an active {provider.title()} access key before running cloud analysis."
            )

    try:
        prepared = service.prepare_structured_clinical_input(request_payload)
        normalized_document = prepared["normalized_document"]
        section_extraction = prepared["section_extraction"]
        patient_payload = prepared["patient_payload"]
    except ClinicalInputExtractionError as exc:
        raise ServiceValidationError(str(exc)) from exc
    try:
        service.ensure_submission_requirements(patient_payload)
    except ClinicalPipelineValidationError as exc:
        raise ServiceValidationError(
            service.serialize_pipeline_issues(exc.issues)
        ) from exc
    job_id = service.job_manager.start_job(
        job_type=service.JOB_TYPE,
        runner=run_clinical_job,
        scope_key="clinical:global",
        kwargs={
            "service": service,
            "payload": patient_payload,
            "patient_image_base64": request_payload.patient_image_base64,
            "section_extraction": section_extraction,
            "normalized_document": normalized_document,
            "report_mode": request_payload.report_mode,
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
        poll_interval=get_server_settings().jobs.polling_interval,
    )
