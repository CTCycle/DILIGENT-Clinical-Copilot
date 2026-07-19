from __future__ import annotations

from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, cast

from common.exceptions import ServiceValidationError
from common.utils.logger import logger
from services.llm.runtime_config import LLMRuntimeConfig
from domain.clinical.entities import (
    ClinicalPipelineValidationError,
    ClinicalSessionRequest,
)
from domain.clinical.robustness import (
    ClinicalInputPreflightIssue,
    ClinicalInputPreflightResult,
)
from services.clinical.deterministic_extraction import extract_deterministic_diseases
from services.llm.provider_factory import select_llm_provider
from services.retrieval.readiness import check_rag_readiness
from services.security.access_keys import AccessKeyService
from services.session.robust_pipeline import build_extraction_artifact
from services.session.text_section_parser import parse_initial_text_sections

###############################################################################
@dataclass(frozen=True)
class LocalModelBatchPreflightResult:
    concurrency_allowed: bool
    provider: str
    model: str | None
    reason: str | None = None

###############################################################################
async def check_parser_batch_capacity(
    task_count: int,
    model: str | None = None,
) -> LocalModelBatchPreflightResult:
    provider, resolved_model = LLMRuntimeConfig.resolve_provider_and_model("parser")
    normalized_provider = (provider or "").strip().lower()
    selected_model = (model or resolved_model or "").strip() or None

    if task_count <= 1:
        return LocalModelBatchPreflightResult(
            concurrency_allowed=True,
            provider=normalized_provider,
            model=selected_model,
            reason=None,
        )

    if normalized_provider != "ollama":
        return LocalModelBatchPreflightResult(
            concurrency_allowed=True,
            provider=normalized_provider,
            model=selected_model,
            reason=None,
        )

    if not selected_model:
        return LocalModelBatchPreflightResult(
            concurrency_allowed=False,
            provider=normalized_provider,
            model=selected_model,
            reason="Parser model is not configured for local runtime.",
        )

    client: Any = select_llm_provider(
        provider=normalized_provider,
        default_model=selected_model,
        max_retries=0,
    )
    try:
        is_server_online = getattr(client, "is_server_online", None)
        if not callable(is_server_online):
            return LocalModelBatchPreflightResult(
                concurrency_allowed=False,
                provider=normalized_provider,
                model=selected_model,
                reason="Local runtime status endpoint is unavailable.",
            )
        is_online = await cast(Any, is_server_online())
        if not is_online:
            return LocalModelBatchPreflightResult(
                concurrency_allowed=False,
                provider=normalized_provider,
                model=selected_model,
                reason="Local runtime is unreachable.",
            )
        list_models = getattr(client, "list_models", None)
        if not callable(list_models):
            return LocalModelBatchPreflightResult(
                concurrency_allowed=False,
                provider=normalized_provider,
                model=selected_model,
                reason="Local runtime model listing is unavailable.",
            )
        available_models = await cast(Any, list_models())
        normalized_models = {(item or "").strip() for item in available_models}
        if selected_model not in normalized_models:
            return LocalModelBatchPreflightResult(
                concurrency_allowed=False,
                provider=normalized_provider,
                model=selected_model,
                reason="Configured parser model is not available locally.",
            )
        get_cached_residency_plan = getattr(client, "get_cached_residency_plan", None)
        if callable(get_cached_residency_plan):
            try:
                await cast(Any, get_cached_residency_plan(force_refresh=True))
            except Exception:
                return LocalModelBatchPreflightResult(
                    concurrency_allowed=False,
                    provider=normalized_provider,
                    model=selected_model,
                    reason="Local runtime status cannot be inspected safely.",
                )
        else:
            return LocalModelBatchPreflightResult(
                concurrency_allowed=False,
                provider=normalized_provider,
                model=selected_model,
                reason="Local runtime status API is unavailable.",
            )
        return LocalModelBatchPreflightResult(
            concurrency_allowed=True,
            provider=normalized_provider,
            model=selected_model,
            reason=None,
        )
    except Exception as exc:  # noqa: BLE001
        return LocalModelBatchPreflightResult(
            concurrency_allowed=False,
            provider=normalized_provider,
            model=selected_model,
            reason=str(exc),
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close_result = close()
                if isawaitable(close_result):
                    await cast(Any, close_result)
            except Exception:
                pass

###############################################################################
def validate_clinical_input_preflight(
    service: Any,
    request_payload: ClinicalSessionRequest,
) -> ClinicalInputPreflightResult:
    blocking: list[ClinicalInputPreflightIssue] = []
    non_blocking: list[ClinicalInputPreflightIssue] = []
    service.apply_persisted_runtime_configuration()
    runtime_settings = _runtime_settings()
    deterministic_diagnostics: dict[str, Any] = {}
    rag_readiness = check_rag_readiness(requested=request_payload.use_rag)
    if request_payload.use_rag and not rag_readiness.available:
        non_blocking.append(
            ClinicalInputPreflightIssue(
                severity="non_blocking",
                code=rag_readiness.reason_code or "rag_unavailable",
                message=rag_readiness.message
                or "RAG evidence retrieval is unavailable for this assessment.",
                field="use_rag",
            )
        )
    _validate_ui_metadata(request_payload, blocking)
    _validate_provider_key(blocking)
    _validate_requested_provider(request_payload, blocking, runtime_settings)
    _validate_persistence(service, blocking)
    extraction_quality: dict[str, Any] = {}
    clinical_input = (request_payload.clinical_input or "").strip()
    if not clinical_input:
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="clinical_input_missing",
                message="Clinical input is required.",
                field="clinical_input",
            )
        )
    _validate_knowledge_bases(service, blocking)
    if not clinical_input:
        return _result(
            blocking,
            non_blocking,
            runtime_settings,
            extraction_quality,
            deterministic_diagnostics,
            rag_readiness,
        )
    if len(clinical_input.split()) < 60:
        non_blocking.append(
            ClinicalInputPreflightIssue(
                severity="non_blocking",
                code="clinical_input_too_short",
                message=(
                    "Clinical input contains fewer than 60 words and may not provide "
                    "enough context for a reliable assessment."
                ),
                field="clinical_input",
            )
        )
    parse_result = parse_initial_text_sections(clinical_input)
    if parse_result.missing_required_sections:
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="required_sections_missing",
                message="Missing required sections: "
                + ", ".join(parse_result.missing_required_sections),
                field="clinical_input",
            )
        )
    if parse_result.malformed_sections:
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="required_sections_malformed",
                message="Malformed required sections: "
                + ", ".join(parse_result.malformed_sections),
                field="clinical_input",
            )
        )
    try:
        prepared = service.prepare_structured_clinical_input(request_payload)
        section_extraction = prepared["section_extraction"]
        patient_payload = prepared["patient_payload"]
        normalized_document = prepared["normalized_document"]
        therapy_result = service.drugs_parser.extract_drugs_from_therapy_deterministic(
            service.drugs_parser.clean_text(patient_payload.drugs or "")
        )
        anamnesis_result = (
            service.drugs_parser.extract_drugs_from_anamnesis_deterministic(
                service.drugs_parser.clean_text(patient_payload.anamnesis or "")
            )
        )
        extraction_artifact = build_extraction_artifact(
            normalized_document=normalized_document,
            section_extraction=section_extraction,
            payload=patient_payload,
        )
        disease_context = extract_deterministic_diseases(
            service.disease_extractor.clean_text(patient_payload.anamnesis or "")
        )
        deterministic_diagnostics = {
            "parser": section_extraction.metadata,
            "section_coverage": {
                "anamnesis_chars": len(section_extraction.anamnesis),
                "therapy_chars": len(section_extraction.drugs),
                "laboratory_analysis_chars": len(
                    section_extraction.laboratory_analysis
                ),
            },
            "therapy": {
                "drug_count": len(therapy_result.entries),
                "unresolved_line_count": len(therapy_result.unresolved_lines),
            },
            "anamnesis": {
                "drug_count": len(anamnesis_result.entries),
                "regimen_line_count": len(anamnesis_result.regimen_lines),
                "unresolved_line_count": len(anamnesis_result.unresolved_lines),
            },
            "diseases": {
                "disease_count": len(disease_context.context.entries),
                "matched_line_count": len(disease_context.matched_lines),
                "unresolved_line_count": len(disease_context.unresolved_lines),
            },
        }
        extraction_quality = {
            "confidence": extraction_artifact.confidence,
            "section_confidence": section_extraction.confidence,
            "requires_review": bool(
                section_extraction.metadata.get("requires_review")
                if isinstance(section_extraction.metadata, dict)
                else False
            ),
            "requires_review_sections": (
                section_extraction.metadata.get("requires_review_sections", [])
                if isinstance(section_extraction.metadata, dict)
                else []
            ),
            "timed_drug_count": len(extraction_artifact.timed_drugs),
            "contamination_flags": extraction_artifact.contamination_flags.model_dump(),
        }
        section_confidence = float(section_extraction.confidence)
        requires_review_sections = extraction_quality["requires_review_sections"]
        if section_confidence < 0.65:
            blocking.append(
                ClinicalInputPreflightIssue(
                    severity="blocking",
                    code="section_extraction_confidence_too_low",
                    message=(
                        "Clinical input section extraction confidence is too low "
                        f"for safe processing ({section_confidence:.2f})."
                    ),
                    field="clinical_input",
                )
            )
        elif section_confidence < 0.85:
            non_blocking.append(
                ClinicalInputPreflightIssue(
                    severity="non_blocking",
                    code="section_extraction_confidence_needs_review",
                    message=(
                        "Clinical input section extraction is usable but should be "
                        f"reviewed before processing ({section_confidence:.2f})."
                    ),
                    field="clinical_input",
                )
            )
        if isinstance(requires_review_sections, list) and requires_review_sections:
            non_blocking.append(
                ClinicalInputPreflightIssue(
                    severity="non_blocking",
                    code="section_extraction_requires_review",
                    message=(
                        "Review inferred or low-confidence sections before running: "
                        + ", ".join(str(item) for item in requires_review_sections)
                    ),
                    field="clinical_input",
                )
            )
        if anamnesis_result.unresolved_lines:
            non_blocking.append(
                ClinicalInputPreflightIssue(
                    severity="non_blocking",
                    code="anamnesis_regimen_lines_need_review",
                    message=(
                        f"{len(anamnesis_result.unresolved_lines)} anamnesis regimen/history lines "
                        "could not be fully resolved deterministically."
                    ),
                    field="anamnesis",
                )
            )
        if not disease_context.context.entries:
            non_blocking.append(
                ClinicalInputPreflightIssue(
                    severity="non_blocking",
                    code="anamnesis_disease_context_sparse",
                    message="No deterministic disease/context entries were detected from anamnesis.",
                    field="anamnesis",
                )
            )
        if extraction_artifact.confidence < 0.55:
            non_blocking.append(
                ClinicalInputPreflightIssue(
                    severity="non_blocking",
                    code="minimum_extraction_quality_not_met",
                    message=(
                        "Clinical input extraction confidence is below the minimum threshold; "
                        "manual review is recommended."
                    ),
                    field="clinical_input",
                )
            )
        if not extraction_artifact.timed_drugs:
            non_blocking.append(
                ClinicalInputPreflightIssue(
                    severity="non_blocking",
                    code="timed_drug_feasibility_failed",
                    message=(
                        "No drug with explicit source-reported timing was detected; "
                        "manual review is recommended."
                    ),
                    field="drugs",
                )
            )
        if any(extraction_artifact.contamination_flags.model_dump().values()):
            non_blocking.append(
                ClinicalInputPreflightIssue(
                    severity="non_blocking",
                    code="manual_review_required",
                    message="Possible non-clinical contamination was detected.",
                    field="clinical_input",
                )
            )
        service.ensure_submission_requirements(patient_payload)
    except ClinicalPipelineValidationError as exc:
        for issue in exc.issues:
            blocking.append(
                ClinicalInputPreflightIssue(
                    severity="blocking",
                    code=issue.code,
                    message=issue.message,
                    field=issue.field,
                )
            )
    except ServiceValidationError as exc:
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="preflight_validation_failed",
                message=str(exc),
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Clinical input pre-flight failed: error_type=%s",
            type(exc).__name__,
        )
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="preflight_failed",
                message=(
                    "The application could not complete the safety checks. "
                    "Review the input and runtime configuration, then retry."
                ),
            )
        )
    return _result(
        blocking,
        non_blocking,
        runtime_settings,
        extraction_quality,
        deterministic_diagnostics,
        rag_readiness,
    )

###############################################################################
def _validate_knowledge_bases(
    service: Any,
    blocking: list[ClinicalInputPreflightIssue],
) -> None:
    try:
        livertox_rows, _ = service.serializer.list_livertox_catalog(
            search=None,
            offset=0,
            limit=1,
        )
        rxnav_rows, _ = service.serializer.list_rxnav_catalog(
            search=None,
            offset=0,
            limit=1,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Clinical pre-flight knowledge-base check failed: error_type=%s",
            type(exc).__name__,
        )
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="knowledge_base_unavailable",
                message="The local clinical knowledge base could not be inspected.",
                field="knowledge_base",
            )
        )
        return
    if not livertox_rows:
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="livertox_catalog_empty",
                message=(
                    "LiverTox catalog is empty. Run the LiverTox update job from "
                    "Data Inspection before clinical analysis."
                ),
                field="knowledge_base",
            )
        )
    if not rxnav_rows:
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="rxnav_catalog_empty",
                message=(
                    "RxNav catalog is empty. Run the RxNav update job from Data "
                    "Inspection before clinical analysis."
                ),
                field="knowledge_base",
            )
        )

###############################################################################
def _validate_ui_metadata(
    request_payload: ClinicalSessionRequest,
    blocking: list[ClinicalInputPreflightIssue],
) -> None:
    if not request_payload.visit_date:
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="visit_date_missing",
                message="Visit date is required.",
                field="visit_date",
            )
        )

###############################################################################
def _validate_provider_key(blocking: list[ClinicalInputPreflightIssue]) -> None:
    if not LLMRuntimeConfig.is_cloud_enabled():
        return
    provider = LLMRuntimeConfig.get_llm_provider().strip().lower()
    if provider not in _CLOUD_PROVIDERS:
        return
    active_keys = [
        item
        for item in AccessKeyService().list_access_keys(cast(Any, provider))
        if item.is_active
    ]
    if not active_keys:
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="active_provider_key_missing",
                message=f"Configure an active {provider.title()} access key before running cloud analysis.",
                field="selected_model_providers",
            )
        )

###############################################################################
def _validate_requested_provider(
    request_payload: ClinicalSessionRequest,
    blocking: list[ClinicalInputPreflightIssue],
    runtime_settings: dict[str, Any],
) -> None:
    selected = {
        item.strip().lower()
        for item in request_payload.selected_model_providers
        if item and item.strip()
    }
    provider = str(
        runtime_settings.get("clinical_provider")
        or runtime_settings.get("llm_provider")
        or ""
    ).lower()
    if not selected:
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="provider_selection_missing",
                message="At least one model provider must be selected.",
                field="selected_model_providers",
            )
        )
        return
    if provider and provider not in selected:
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="requested_provider_mismatch",
                message="The active runtime provider must match the requested provider exactly.",
                field="selected_model_providers",
            )
        )

###############################################################################
def _validate_persistence(
    service: Any,
    blocking: list[ClinicalInputPreflightIssue],
) -> None:
    if not hasattr(service.serializer, "session_factory"):
        return
    try:
        with service.serializer.session_factory() as db_session:
            db_session.connection()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Clinical pre-flight persistence check failed: error_type=%s",
            type(exc).__name__,
        )
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="persistence_unavailable",
                message="Session storage is not writable or reachable.",
                field="session_storage",
            )
        )

###############################################################################
def _runtime_settings() -> dict[str, Any]:
    parser_provider, parser_model = LLMRuntimeConfig.resolve_provider_and_model(
        "parser"
    )
    clinical_provider, clinical_model = LLMRuntimeConfig.resolve_provider_and_model(
        "clinical"
    )
    return {
        "use_cloud_services": LLMRuntimeConfig.is_cloud_enabled(),
        "llm_provider": LLMRuntimeConfig.get_llm_provider(),
        "cloud_model": LLMRuntimeConfig.get_cloud_model(),
        "text_extraction_provider": parser_provider,
        "text_extraction_model": parser_model,
        "clinical_provider": clinical_provider,
        "clinical_model": clinical_model,
    }

###############################################################################
def _result(
    blocking: list[ClinicalInputPreflightIssue],
    non_blocking: list[ClinicalInputPreflightIssue],
    runtime_settings: dict[str, Any],
    extraction_quality: dict[str, Any],
    deterministic_diagnostics: dict[str, Any],
    rag_readiness: Any,
) -> ClinicalInputPreflightResult:
    return ClinicalInputPreflightResult(
        ready=not blocking,
        blocking_issues=[_present_preflight_issue(issue) for issue in blocking],
        non_blocking_issues=[
            _present_preflight_issue(issue) for issue in non_blocking
        ],
        runtime_settings=runtime_settings,
        extraction_quality=extraction_quality,
        deterministic_diagnostics=deterministic_diagnostics,
        rag_readiness=rag_readiness,
    )


###############################################################################
def _present_preflight_issue(
    issue: ClinicalInputPreflightIssue,
) -> ClinicalInputPreflightIssue:
    title, consequence = _ISSUE_PRESENTATIONS.get(
        issue.code,
        (
            "Review required"
            if issue.severity == "non_blocking"
            else "Analysis cannot start",
            (
                "Continuing may reduce the completeness or reliability of the assessment."
                if issue.severity == "non_blocking"
                else "The analysis cannot be started safely until this issue is corrected."
            ),
        ),
    )
    affected_section = _FIELD_LABELS.get(
        issue.field or "",
        (issue.field or "Analysis configuration").replace("_", " ").title(),
    )
    return issue.model_copy(
        update={
            "title": title,
            "description": issue.message,
            "affected_section": affected_section,
            "consequence": consequence,
            "continuation_allowed": issue.severity == "non_blocking",
        }
    )


_FIELD_LABELS = {
    "anamnesis": "Anamnesis",
    "clinical_input": "Clinical input",
    "drugs": "Pharmacological therapy",
    "knowledge_base": "Clinical knowledge base",
    "laboratory_analysis": "Laboratory analysis",
    "selected_model_providers": "Model configuration",
    "session_storage": "Session storage",
    "use_rag": "RAG evidence",
    "visit_date": "Visit date",
}

_ISSUE_PRESENTATIONS: dict[str, tuple[str, str]] = {
    "active_provider_key_missing": (
        "Provider access key missing",
        "The configured cloud model cannot be contacted without an active access key.",
    ),
    "anamnesis_disease_context_sparse": (
        "Clinical context may be incomplete",
        "Relevant comorbidities or competing causes may be underrepresented in the assessment.",
    ),
    "anamnesis_regimen_lines_need_review": (
        "Medication history needs review",
        "Some historical medication details may be omitted or interpreted with uncertainty.",
    ),
    "clinical_input_missing": (
        "Clinical input is empty",
        "There is no clinical information available to analyse.",
    ),
    "clinical_input_too_short": (
        "Clinical input may be too brief",
        "The assessment may omit important chronology, competing causes, or clinical context.",
    ),
    "livertox_catalog_empty": (
        "LiverTox data is unavailable",
        "Drug-specific hepatotoxicity evidence cannot be evaluated.",
    ),
    "knowledge_base_unavailable": (
        "Clinical knowledge base is unavailable",
        "Medication matching and evidence-backed assessment cannot be completed reliably.",
    ),
    "manual_review_required": (
        "Possible non-clinical content detected",
        "Administrative or bibliography text may reduce extraction quality.",
    ),
    "missing_anamnesis": (
        "Anamnesis is missing",
        "The assessment cannot evaluate clinical context or competing causes.",
    ),
    "missing_drugs": (
        "Pharmacological therapy is missing",
        "There is no medication exposure available for DILI causality assessment.",
    ),
    "missing_laboratory_analysis": (
        "Laboratory analysis is missing",
        "The liver injury pattern and severity cannot be evaluated.",
    ),
    "missing_timed_drug": (
        "Medication timing is missing",
        "Latency and exposure chronology cannot be evaluated reliably.",
    ),
    "missing_visit_date": (
        "Visit date is missing",
        "The workflow cannot anchor the clinical chronology or persist a valid assessment date.",
    ),
    "minimum_extraction_quality_not_met": (
        "Clinical extraction confidence is low",
        "Important facts may be incomplete or assigned to the wrong section.",
    ),
    "persistence_unavailable": (
        "Session storage is unavailable",
        "The analysis result cannot be stored reliably.",
    ),
    "preflight_failed": (
        "Safety checks could not finish",
        "The analysis cannot start because processing readiness is unknown.",
    ),
    "preflight_validation_failed": (
        "Clinical input is invalid",
        "The backend cannot process the current request safely.",
    ),
    "provider_selection_missing": (
        "No model provider selected",
        "No configured model is available to perform the analysis.",
    ),
    "requested_provider_mismatch": (
        "Selected provider does not match runtime",
        "The request would run with a different provider than the one selected in the interface.",
    ),
    "rag_embedding_model_missing": (
        "RAG embedding model is not configured",
        "The assessment can continue, but indexed evidence will not be retrieved.",
    ),
    "rag_ollama_model_unavailable": (
        "RAG embedding model is unavailable",
        "The assessment can continue, but indexed evidence will not be retrieved.",
    ),
    "rag_ollama_unavailable": (
        "RAG evidence service is unavailable",
        "The assessment can continue, but indexed evidence will not be retrieved.",
    ),
    "rag_unavailable": (
        "RAG evidence is unavailable",
        "The assessment can continue, but indexed evidence will not be retrieved.",
    ),
    "required_sections_malformed": (
        "Required sections are malformed",
        "The application cannot reliably separate the clinical sections.",
    ),
    "required_sections_missing": (
        "Required clinical sections are missing",
        "The analysis would lack mandatory clinical, therapy, or laboratory information.",
    ),
    "rxnav_catalog_empty": (
        "RxNav data is unavailable",
        "Medication normalization and matching cannot be completed reliably.",
    ),
    "section_extraction_confidence_needs_review": (
        "Section extraction needs review",
        "The analysis can continue, but some information may have been assigned to the wrong section.",
    ),
    "section_extraction_confidence_too_low": (
        "Section extraction is unreliable",
        "The clinical sections cannot be separated safely enough for analysis.",
    ),
    "section_extraction_requires_review": (
        "Inferred sections need review",
        "One or more sections were inferred and may not accurately represent the source text.",
    ),
    "timed_drug_feasibility_failed": (
        "Medication chronology is incomplete",
        "Causality and latency assessment will be less reliable without exposure timing.",
    ),
    "visit_date_missing": (
        "Visit date is missing",
        "The workflow cannot anchor the clinical chronology or persist a valid assessment date.",
    ),
}

_CLOUD_PROVIDERS = {"openai", "gemini"}
