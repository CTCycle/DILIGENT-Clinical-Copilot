from __future__ import annotations

from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, cast

from common.exceptions import ServiceValidationError
from configurations.llm_configs import LLMRuntimeConfig
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
from services.security.access_keys import AccessKeyService
from services.session.robust_pipeline import build_extraction_artifact
from services.session.text_section_parser import parse_initial_text_sections


@dataclass(frozen=True)
class LocalModelBatchPreflightResult:
    concurrency_allowed: bool
    provider: str
    model: str | None
    reason: str | None = None


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


def validate_clinical_input_preflight(
    service: Any,
    request_payload: ClinicalSessionRequest,
) -> ClinicalInputPreflightResult:
    blocking: list[ClinicalInputPreflightIssue] = []
    non_blocking: list[ClinicalInputPreflightIssue] = []
    runtime_settings = _runtime_settings()
    deterministic_diagnostics: dict[str, Any] = {}
    service.apply_persisted_runtime_configuration()
    _validate_ui_metadata(request_payload, blocking)
    _validate_provider_key(blocking)
    _validate_requested_provider(request_payload, blocking, runtime_settings)
    _validate_persistence(service, non_blocking)
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
        return _result(
            blocking,
            non_blocking,
            runtime_settings,
            extraction_quality,
            deterministic_diagnostics,
        )
    livertox_rows, _ = service.serializer.list_livertox_catalog(
        search=None,
        offset=0,
        limit=1,
    )
    if not livertox_rows:
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="livertox_catalog_empty",
                message="LiverTox catalog is empty. Rebuild LiverTox data.",
                field="knowledge_base",
            )
        )
    rxnav_rows, _ = service.serializer.list_rxnav_catalog(
        search=None,
        offset=0,
        limit=1,
    )
    if not rxnav_rows:
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="rxnav_catalog_empty",
                message="RxNav catalog is empty. Rebuild RxNav data.",
                field="knowledge_base",
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
            "timed_drug_count": len(extraction_artifact.timed_drugs),
            "contamination_flags": extraction_artifact.contamination_flags.model_dump(),
        }
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
        blocking.append(
            ClinicalInputPreflightIssue(
                severity="blocking",
                code="preflight_failed",
                message=str(exc),
            )
        )
    return _result(
        blocking,
        non_blocking,
        runtime_settings,
        extraction_quality,
        deterministic_diagnostics,
    )


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
    provider = str(runtime_settings.get("llm_provider") or "").lower()
    if not selected:
        # Backward-compatible behavior for clients that do not send
        # selected_model_providers: assume the active runtime provider.
        if provider:
            selected = {provider}
        else:
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


def _validate_persistence(
    service: Any,
    non_blocking: list[ClinicalInputPreflightIssue],
) -> None:
    if not hasattr(service.serializer, "session_factory"):
        return
    try:
        with service.serializer.session_factory() as db_session:
            db_session.connection()
    except Exception as exc:  # noqa: BLE001
        non_blocking.append(
            ClinicalInputPreflightIssue(
                severity="non_blocking",
                code="persistence_unavailable",
                message=f"Session persistence is not writable or reachable: {exc}",
            )
        )


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


def _result(
    blocking: list[ClinicalInputPreflightIssue],
    non_blocking: list[ClinicalInputPreflightIssue],
    runtime_settings: dict[str, Any],
    extraction_quality: dict[str, Any],
    deterministic_diagnostics: dict[str, Any],
) -> ClinicalInputPreflightResult:
    return ClinicalInputPreflightResult(
        ready=not blocking,
        blocking_issues=blocking,
        non_blocking_issues=non_blocking,
        runtime_settings=runtime_settings,
        extraction_quality=extraction_quality,
        deterministic_diagnostics=deterministic_diagnostics,
    )


_CLOUD_PROVIDERS = {"openai", "gemini"}
