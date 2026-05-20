from __future__ import annotations

import asyncio
from typing import Any

from common.exceptions import ServiceValidationError
from configurations.llm_configs import LLMRuntimeConfig
from domain.clinical.entities import ClinicalPipelineValidationError, ClinicalSessionRequest
from domain.clinical.robustness import (
    ClinicalInputPreflightIssue,
    ClinicalInputPreflightResult,
)
from services.security.access_keys import AccessKeyService
from services.session.document_normalizer import DocumentNormalizer
from services.session.robust_pipeline import build_extraction_artifact


def validate_clinical_input_preflight(
    service: Any,
    request_payload: ClinicalSessionRequest,
) -> ClinicalInputPreflightResult:
    blocking: list[ClinicalInputPreflightIssue] = []
    non_blocking: list[ClinicalInputPreflightIssue] = []
    runtime_settings = _runtime_settings()
    service.apply_persisted_runtime_configuration()
    _validate_ui_metadata(request_payload, non_blocking)
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
        return _result(blocking, non_blocking, runtime_settings, extraction_quality)
    normalized_document = DocumentNormalizer().normalize(clinical_input)
    try:
        preprocessed_request, section_extraction = asyncio.run(
            service.preprocess_unified_input(request_payload)
        )
        patient_payload = service.build_patient_payload(preprocessed_request)
        extraction_artifact = build_extraction_artifact(
            normalized_document=normalized_document,
            section_extraction=section_extraction,
            payload=patient_payload,
        )
        extraction_quality = {
            "confidence": extraction_artifact.confidence,
            "timed_drug_count": len(extraction_artifact.timed_drugs),
            "contamination_flags": extraction_artifact.contamination_flags.model_dump(),
        }
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
    return _result(blocking, non_blocking, runtime_settings, extraction_quality)


def _validate_ui_metadata(
    request_payload: ClinicalSessionRequest,
    non_blocking: list[ClinicalInputPreflightIssue],
) -> None:
    if not request_payload.visit_date:
        non_blocking.append(
            ClinicalInputPreflightIssue(
                severity="non_blocking",
                code="visit_date_missing",
                message="Report date is missing; manual confirmation is recommended.",
                field="visit_date",
            )
        )


def _validate_provider_key(blocking: list[ClinicalInputPreflightIssue]) -> None:
    if not LLMRuntimeConfig.is_cloud_enabled():
        return
    provider = LLMRuntimeConfig.get_llm_provider()
    active_keys = [
        item
        for item in AccessKeyService().list_access_keys(provider)
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
    parser_provider, parser_model = LLMRuntimeConfig.resolve_provider_and_model("parser")
    clinical_provider, clinical_model = LLMRuntimeConfig.resolve_provider_and_model("clinical")
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
) -> ClinicalInputPreflightResult:
    return ClinicalInputPreflightResult(
        ready=not blocking,
        blocking_issues=blocking,
        non_blocking_issues=non_blocking,
        runtime_settings=runtime_settings,
        extraction_quality=extraction_quality,
    )
