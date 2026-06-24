from __future__ import annotations

from typing import Literal, Protocol

from common.utils.text_utils import unique_preserve_order
from domain.clinical.revision import (
    RevisionFinalReportPayload,
    RevisionQaValidationPayload,
)

###############################################################################
class ReviewerInstructionProfileLike(Protocol):
    target_sections: list[
        Literal[
            "anamnesis",
            "therapy",
            "labs",
            "livertox_matching",
            "dili_assessment",
            "final_report",
            "qa",
            "unknown",
        ]
    ]
    target_entities: list[
        Literal[
            "drugs",
            "diseases",
            "labs",
            "report_wording",
            "source_evidence",
            "matching_errors",
            "causality_reasoning",
            "missing_data",
            "ambiguity_resolution",
            "other",
        ]
    ]

###############################################################################
def build_revision_qa_validation_payload(
    *,
    result_payload: dict[str, object],
    instruction_profile: ReviewerInstructionProfileLike | None,
    final_report_payload: RevisionFinalReportPayload,
) -> RevisionQaValidationPayload:
    revision_payload = result_payload.get("revision")
    if not isinstance(revision_payload, dict):
        revision_payload = {}
    pipeline_artifacts = result_payload.get("pipeline_artifacts")
    if not isinstance(pipeline_artifacts, dict):
        pipeline_artifacts = {}

    blocking_issues = [
        str(item).strip()
        for item in (result_payload.get("blocking_issues") or [])
        if str(item).strip()
    ]
    manual_review_required = bool(result_payload.get("manual_review_required"))
    warnings = list(final_report_payload.warnings)
    addressed_items: list[str] = []
    unaddressed_items: list[str] = []

    structured_case = result_payload.get("structured_case")
    has_structured_case = isinstance(structured_case, dict) and bool(structured_case)
    has_report_comparison = isinstance(result_payload.get("report_comparison"), dict)
    has_faithfulness_audit = isinstance(
        pipeline_artifacts.get("faithfulness_audit"),
        dict,
    )
    has_livertox_decisions = bool(revision_payload.get("livertox_revision_decisions"))
    has_revised_assessments = bool(revision_payload.get("revised_dili_assessments"))

    if instruction_profile is not None:
        for section in instruction_profile.target_sections:
            if section == "unknown":
                continue
            if section in {"anamnesis", "therapy"}:
                if has_structured_case:
                    addressed_items.append(f"section:{section}")
                else:
                    unaddressed_items.append(f"section:{section}")
            elif section == "labs":
                if has_revised_assessments:
                    addressed_items.append("section:labs")
                else:
                    unaddressed_items.append("section:labs")
            elif section == "livertox_matching":
                if has_livertox_decisions:
                    addressed_items.append("section:livertox_matching")
                else:
                    unaddressed_items.append("section:livertox_matching")
            elif section == "dili_assessment":
                if has_revised_assessments:
                    addressed_items.append("section:dili_assessment")
                else:
                    unaddressed_items.append("section:dili_assessment")
            elif section == "final_report":
                if final_report_payload.report_present:
                    addressed_items.append("section:final_report")
                else:
                    unaddressed_items.append("section:final_report")
            elif section == "qa":
                if has_faithfulness_audit or has_report_comparison:
                    addressed_items.append("section:qa")
                else:
                    unaddressed_items.append("section:qa")

        for entity in instruction_profile.target_entities:
            if entity == "other":
                continue
            if entity in {"drugs", "diseases"}:
                if has_structured_case:
                    addressed_items.append(f"entity:{entity}")
                else:
                    unaddressed_items.append(f"entity:{entity}")
            elif entity == "labs":
                if has_revised_assessments:
                    addressed_items.append("entity:labs")
                else:
                    unaddressed_items.append("entity:labs")
            elif entity == "report_wording":
                if final_report_payload.report_present:
                    addressed_items.append("entity:report_wording")
                else:
                    unaddressed_items.append("entity:report_wording")
            elif entity in {"source_evidence", "ambiguity_resolution"}:
                if has_report_comparison or has_faithfulness_audit:
                    addressed_items.append(f"entity:{entity}")
                else:
                    unaddressed_items.append(f"entity:{entity}")
            elif entity == "matching_errors":
                if has_livertox_decisions:
                    addressed_items.append("entity:matching_errors")
                else:
                    unaddressed_items.append("entity:matching_errors")
            elif entity == "causality_reasoning":
                if has_revised_assessments:
                    addressed_items.append("entity:causality_reasoning")
                else:
                    unaddressed_items.append("entity:causality_reasoning")
            elif entity == "missing_data":
                if final_report_payload.report_present:
                    addressed_items.append("entity:missing_data")
                else:
                    unaddressed_items.append("entity:missing_data")

    addressed_items = unique_preserve_order(addressed_items)
    unaddressed_items = unique_preserve_order(unaddressed_items)
    if unaddressed_items:
        warnings.append(
            "Some reviewer-requested sections or entities could not be verified as addressed."
        )

    if blocking_issues:
        status = "failed"
        version_status = "qa_failed"
    elif manual_review_required:
        status = "requires_human_review"
        version_status = "requires_human_review"
    elif warnings:
        status = "passed_with_warnings"
        version_status = "llm_qa_passed"
    elif has_faithfulness_audit or final_report_payload.report_present:
        status = "passed"
        version_status = "llm_qa_passed"
    else:
        status = "requires_human_review"
        version_status = "requires_human_review"
        warnings.append(
            "Revision QA could not confirm a persisted final report or QA audit."
        )

    deduped_warnings = unique_preserve_order(warnings)
    return RevisionQaValidationPayload(
        status=status,
        version_status=version_status,
        addressed_items=addressed_items,
        unaddressed_items=unaddressed_items,
        warnings=deduped_warnings,
        blocking_issues=blocking_issues,
        manual_review_required=manual_review_required,
        finding_count=len(blocking_issues) + len(deduped_warnings),
    )
