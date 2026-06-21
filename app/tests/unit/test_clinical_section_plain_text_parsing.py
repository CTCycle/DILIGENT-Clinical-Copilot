from __future__ import annotations

from services.session.clinical_section_parsers import find_dili_section_headings

###############################################################################
def test_blank_lines_do_not_create_markers() -> None:
    text = "line one\n\nline two\n\nline three"
    assert find_dili_section_headings(text) == []
