from __future__ import annotations

from domain.clinical.entities import RagDocumentReference
from services.clinical.report_finalizer import ReportFinalizer

###############################################################################
def test_page_and_line_locations_render_without_synthetic_pages() -> None:
    lines = ReportFinalizer.render_canonical_references(
        [
            RagDocumentReference(
                file_name="alpha.pdf",
                page_start=2,
                page_end=3,
                line_start=18,
                line_end=54,
            ),
            RagDocumentReference(file_name="notes.txt", line_start=40, line_end=58),
            RagDocumentReference(file_name="source.xml"),
        ]
    )
    assert "- alpha.pdf, pp. 2-3, lines 18-54" in lines
    assert "- notes.txt, lines 40-58" in lines
    assert "- source.xml, location not available" in lines

###############################################################################
def test_line_ranges_merge_only_when_contiguous_or_overlapping() -> None:
    lines = ReportFinalizer.render_canonical_references(
        [
            RagDocumentReference(file_name="notes.txt", line_start=10, line_end=20),
            RagDocumentReference(file_name="notes.txt", line_start=19, line_end=30),
            RagDocumentReference(file_name="notes.txt", line_start=40, line_end=50),
        ]
    )
    assert lines == ["- notes.txt, lines 10-30", "- notes.txt, lines 40-50"]

###############################################################################
def test_sanitizer_removes_known_sources_and_preserves_clinical_study_mentions() -> (
    None
):
    references = [
        RagDocumentReference(
            file_name="alpha.pdf", page_start=2, line_start=18, line_end=31
        )
    ]
    text = "A cohort study reported delayed onset. [alpha.pdf, p. 2, lines 18-31]\n\n## References\n- alpha.pdf, p. 2"
    sanitized = ReportFinalizer.sanitize_generated_text(text, references)
    assert "cohort study reported" in sanitized
    assert "alpha.pdf" not in sanitized
    assert "## References" not in sanitized
