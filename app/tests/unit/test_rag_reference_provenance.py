from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from domain.clinical.entities import PipelineIssue, RagDocumentReference
from services.retrieval.embeddings import EmbeddingModelMismatchError
from services.clinical.rag_support import RagSupportService


###############################################################################
def test_line_metadata_is_propagated() -> None:
    reference = RagSupportService.build_document_reference(
        {
            "file_name": "alpha.pdf",
            "metadata": {
                "page_start": 2,
                "page_end": 3,
                "line_start": 18,
                "line_end": 54,
            },
        }
    )
    assert reference == RagDocumentReference(
        file_name="alpha.pdf", page_start=2, page_end=3, line_start=18, line_end=54
    )


###############################################################################
@pytest.mark.parametrize(
    "kwargs",
    [
        {"page_start": 3, "page_end": 2},
        {"line_start": 20, "line_end": 10},
        {"line_start": 0},
    ],
)
def test_invalid_location_ranges_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        RagDocumentReference(file_name="alpha.pdf", **kwargs)


###############################################################################
def test_embedding_mismatch_degrades_to_a_recorded_rag_warning() -> None:
    issues: list[PipelineIssue] = []
    service = RagSupportService(
        similarity_search=object(),
        max_excerpt_length=1000,
        rag_candidate_k=5,
        rag_top_n=2,
        rag_use_reranking=False,
        pipeline_issues=issues,
    )
    service.search_supporting_documents = lambda _query: (_ for _ in ()).throw(  # type: ignore[method-assign]
        EmbeddingModelMismatchError("stale vector index")
    )

    result = asyncio.run(
        service.fetch_rag_documents(
            {"amoxicillin-clavulanate": "supporting evidence"},
            "amoxicillin-clavulanate",
        )
    )

    assert result is None
    assert [issue.code for issue in issues] == ["rag_retrieval_unavailable"]
