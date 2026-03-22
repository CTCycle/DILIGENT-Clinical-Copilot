from __future__ import annotations

from DILIGENT.server.services.updater.livertox_sanitizer import (
    LiverToxExcerptSanitizer,
)


# -----------------------------------------------------------------------------
def test_sanitizer_removes_boilerplate_and_preserves_core_content() -> None:
    sanitizer = LiverToxExcerptSanitizer()
    raw_excerpt = (
        "livertox LiverTox Clinical and Research Information on Drug-Induced Liver Injury "
        "2012 National Institute of Diabetes and Digestive and Kidney Diseases "
        "books-source-type Database Acetaminophen chapter navigation text "
        "OVERVIEW Acetaminophen is a common analgesic with dose-dependent hepatotoxicity. "
        "In severe overdose, aminotransferase elevations can exceed 1000 U/L. "
        "OTHER REFERENCE LINKS PubMed.gov ClinicalTrials.gov NCBI Bookshelf footer."
    )

    cleaned = sanitizer.sanitize(raw_excerpt)

    assert cleaned.startswith("OVERVIEW")
    assert "LiverTox Clinical and Research Information" not in cleaned
    assert "books-source-type Database" not in cleaned
    assert "OTHER REFERENCE LINKS" not in cleaned
    assert "dose-dependent hepatotoxicity" in cleaned
    assert "aminotransferase elevations" in cleaned

