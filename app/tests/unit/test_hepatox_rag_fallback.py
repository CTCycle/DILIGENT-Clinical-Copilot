from __future__ import annotations

import asyncio

from domain.clinical import PatientDrugs
from services.clinical.hepatox_core import HepatoxConsultation


class FailingRagConsultation(HepatoxConsultation):
    def __init__(self) -> None:
        self.drugs = PatientDrugs(entries=[])

    def search_supporting_documents(self, query_text: str):
        raise RuntimeError("embedding backend unavailable")


def test_fetch_rag_documents_degrades_when_embedding_backend_fails() -> None:
    consultation = FailingRagConsultation()

    result = asyncio.run(
        consultation.fetch_rag_documents({"Nivolumab": "nivolumab dili"}, "Nivolumab")
    )

    assert result is not None
    assert "RAG retrieval unavailable" in result
    assert "embedding backend unavailable" in result
    assert len(consultation.pipeline_issues) == 1
    issue = consultation.pipeline_issues[0]
    assert issue.severity == "warning"
    assert issue.code == "rag_retrieval_unavailable"
    assert issue.field == "rag"
    assert "Nivolumab" in issue.raw_line
