from __future__ import annotations

from DILIGENT.server.services.research.brave import BraveResearchService


# -----------------------------------------------------------------------------
def test_query_rewrite_generates_concise_queries() -> None:
    service = BraveResearchService()
    queries = [service.normalize_question(
        "Please help me find supporting evidence for acetaminophen and possible DILI risk in adults."
    )]

    assert 1 <= len(queries) <= 3
    assert all(query.strip() for query in queries)
    assert all(len(query) <= 180 for query in queries)


# -----------------------------------------------------------------------------
def test_source_selection_dedupes_domains_and_respects_allow_block_lists() -> None:
    service = BraveResearchService()
    results = [
        {
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            "title": "PubMed Case Series",
            "description": "Case details",
            "score": 0.91,
        },
        {
            "url": "https://pubmed.ncbi.nlm.nih.gov/23456789/",
            "title": "PubMed Cohort",
            "description": "Another result from same domain",
            "score": 0.88,
        },
        {
            "url": "https://www.nejm.org/doi/full/10.1056/example",
            "title": "NEJM Study",
            "description": "Clinical details",
            "score": 0.95,
        },
        {
            "url": "https://spam.example.org/article",
            "title": "Spam Site",
            "description": "Low quality",
            "score": 0.99,
        },
    ]

    selected = service.select_sources(
        results,
        allowed_domains=["pubmed.ncbi.nlm.nih.gov", "nejm.org", "spam.example.org"],
        blocked_domains=["spam.example.org"],
    )

    selected_domains = {service.get_domain(item.url) for item in selected}
    assert "spam.example.org" not in selected_domains
    assert "pubmed.ncbi.nlm.nih.gov" in selected_domains
    assert "nejm.org" in selected_domains
    assert len({item.url for item in selected}) == len(selected)

