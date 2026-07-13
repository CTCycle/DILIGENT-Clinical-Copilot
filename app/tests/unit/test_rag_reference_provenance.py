from __future__ import annotations

import pytest
from pydantic import ValidationError

from domain.clinical.entities import RagDocumentReference
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
