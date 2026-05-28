from __future__ import annotations

from services.retrieval.seed_terms import detect_seed_matches, load_seed_term_catalog


def test_detects_keywords_and_stopwords_from_catalog() -> None:
    catalog = load_seed_term_catalog()
    matches = detect_seed_matches(
        "Patient uses Bactrim tablets and mg dosage.", catalog
    )
    assert "bactrim" in matches["matched_keywords"]
    assert (
        "tablets" in matches["matched_stopwords"]
        or "mg" in matches["matched_stopwords"]
    )
    assert isinstance(matches["matched_term_counts"], dict)
