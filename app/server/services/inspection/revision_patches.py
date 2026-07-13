from __future__ import annotations

from domain.inspection import RevisionReportPatch


###############################################################################
def apply_report_patches(report: str, patches: list[RevisionReportPatch]) -> str:
    ordered = sorted(patches, key=lambda item: item.start, reverse=True)
    previous_start = len(report) + 1
    updated = report
    for patch in ordered:
        if patch.start > patch.end or patch.end > len(report):
            raise ValueError("Revision patch range is outside the source report.")
        if patch.end > previous_start:
            raise ValueError("Revision patches overlap.")
        if report[patch.start : patch.end] != patch.expected_text:
            raise ValueError("Revision patch source text is stale.")
        if not patch.evidence_references:
            raise ValueError("Revision patches require evidence references.")
        updated = updated[: patch.start] + patch.replacement + updated[patch.end :]
        previous_start = patch.start
    return updated


###############################################################################
def validate_draft_report(report: str, patches: list[RevisionReportPatch]) -> str:
    patched = apply_report_patches(report, patches)
    if not patched.strip():
        raise ValueError("Revision produced an empty report.")
    return patched
