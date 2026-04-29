from __future__ import annotations

from DILIGENT.server.common.utils.languages import (
    VALIDATION_MESSAGE_BUNDLES,
    resolve_supported_language_code,
)
from DILIGENT.server.domain.clinical.entities import (
    ClinicalPipelineValidationError,
    DrugEntry,
    PatientData,
    PatientDrugs,
    PipelineIssue,
)
from DILIGENT.server.domain.clinical.validation import ValidationMessageBundle


###############################################################################
def build_validation_bundle(report_language: str) -> ValidationMessageBundle:
    language_code = resolve_supported_language_code(report_language)
    message_bundle = VALIDATION_MESSAGE_BUNDLES.get(
        language_code,
        VALIDATION_MESSAGE_BUNDLES["en"],
    )
    return ValidationMessageBundle(**message_bundle)


def ensure_required_sections(
    payload: PatientData,
    *,
    bundle: ValidationMessageBundle,
) -> None:
    issues: list[PipelineIssue] = []
    if not (payload.anamnesis or "").strip():
        issues.append(
            PipelineIssue(
                severity="error",
                code="missing_anamnesis",
                message=bundle.missing_anamnesis,
                field="anamnesis",
            )
        )
    if payload.visit_date is None:
        issues.append(
            PipelineIssue(
                severity="error",
                code="missing_visit_date",
                message=bundle.missing_visit_date,
                field="visit_date",
            )
        )
    if issues:
        raise ClinicalPipelineValidationError(issues=issues, message=issues[0].message)


def has_timing_information(entry: DrugEntry) -> bool:
    return bool(
        entry.therapy_start_date
        or entry.suspension_date
        or entry.temporal_classification == "temporal_known"
        or entry.therapy_start_status is True
        or entry.suspension_status is True
    )


def ensure_timed_therapy_drug(
    therapy_drugs: PatientDrugs,
    *,
    bundle: ValidationMessageBundle,
) -> None:
    if any(has_timing_information(entry) for entry in therapy_drugs.entries):
        return
    issue = PipelineIssue(
        severity="error",
        code="missing_timed_drug",
        message=bundle.missing_timed_drug,
        field="drugs",
    )
    raise ClinicalPipelineValidationError(issues=[issue], message=issue.message)
