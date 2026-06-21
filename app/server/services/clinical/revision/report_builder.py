from __future__ import annotations

from typing import Literal, Protocol

from domain.clinical.revision import RevisionFinalReportPayload


###############################################################################
class ReviewerInstructionProfileLike(Protocol):
    instruction_summary: str
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


###############################################################################
def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


###############################################################################
def build_revision_final_report_payload(
    *,
    result_payload: dict[str, object],
    selected_text: str | None,
    instruction_profile: ReviewerInstructionProfileLike | None,
) -> RevisionFinalReportPayload:
    report_text = str(result_payload.get("report") or "").strip()
    report_comparison = result_payload.get("report_comparison")
    comparison_outcome: str | None = None
    warnings: list[str] = []
    if isinstance(report_comparison, dict):
        comparison_outcome = str(report_comparison.get("outcome") or "").strip() or None
        manual_review = (
            str(report_comparison.get("manual_review") or "").strip().casefold()
        )
        if manual_review in {"yes", "true", "required"}:
            warnings.append("Report comparison still requests manual review.")
    if not report_text:
        warnings.append("Revision output did not produce a final report body.")
    changed_focus_areas = (
        instruction_profile.target_sections.copy()
        if instruction_profile is not None
        else ["unknown"]
    )
    return RevisionFinalReportPayload(
        report_text=report_text,
        report_present=bool(report_text),
        report_character_count=len(report_text),
        source_excerpt_present=bool(str(selected_text or "").strip()),
        reviewer_instruction_summary=(
            instruction_profile.instruction_summary
            if instruction_profile is not None
            else None
        ),
        comparison_outcome=comparison_outcome,
        changed_focus_areas=changed_focus_areas,
        warnings=_unique_preserve_order(warnings),
    )
