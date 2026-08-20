from __future__ import annotations

import pytest

from domain.inspection import RevisionReportPatch
from services.inspection.revision_patches import apply_report_patches

###############################################################################
def test_apply_report_patches_relocates_a_unique_source_substring() -> None:
    report = "Header\n\nCanonical report text\n\nFooter"
    patch = RevisionReportPatch(
        start=0,
        end=len("Canonical report text"),
        expected_text="Canonical report text",
        replacement="Updated report text",
        evidence_references=["clinical_record"],
    )

    assert apply_report_patches(report, [patch]) == (
        "Header\n\nUpdated report text\n\nFooter"
    )

###############################################################################
def test_apply_report_patches_rejects_ambiguous_relocation() -> None:
    report = "Repeated\nRepeated"
    patch = RevisionReportPatch(
        start=0,
        end=3,
        expected_text="Repeated",
        replacement="Updated",
        evidence_references=["clinical_record"],
    )

    with pytest.raises(ValueError, match="source text is stale"):
        apply_report_patches(report, [patch])
