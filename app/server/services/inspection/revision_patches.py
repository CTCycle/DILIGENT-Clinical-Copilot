from __future__ import annotations

from domain.inspection import RevisionReportPatch

###############################################################################
def apply_report_patches(report: str, patches: list[RevisionReportPatch]) -> str:
    resolved: list[tuple[int, int, RevisionReportPatch]] = []
    for patch in patches:
        if patch.start > patch.end:
            raise ValueError("Revision patch range is outside the source report.")
        if patch.end > len(report):
            raise ValueError("Revision patch range is outside the source report.")

        start = patch.start
        end = patch.end
        if report[start:end] != patch.expected_text:
            matches: list[int] = []
            search_from = 0
            while True:
                match_start = report.find(patch.expected_text, search_from)
                if match_start < 0:
                    break
                matches.append(match_start)
                search_from = match_start + 1
            if len(matches) != 1:
                raise ValueError("Revision patch source text is stale.")
            start = matches[0]
            end = start + len(patch.expected_text)

        resolved.append((start, end, patch))

    ordered = sorted(resolved, key=lambda item: item[0], reverse=True)
    previous_start = len(report) + 1
    updated = report
    for start, end, patch in ordered:
        if end > previous_start:
            raise ValueError("Revision patches overlap.")
        if not patch.evidence_references:
            raise ValueError("Revision patches require evidence references.")
        updated = updated[:start] + patch.replacement + updated[end:]
        previous_start = start
    return updated

###############################################################################
def validate_draft_report(report: str, patches: list[RevisionReportPatch]) -> str:
    patched = apply_report_patches(report, patches)
    if not patched.strip():
        raise ValueError("Revision produced an empty report.")
    return patched
