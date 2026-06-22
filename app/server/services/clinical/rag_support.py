from __future__ import annotations

import asyncio
import re
from typing import Any

from common.utils.logger import logger
from domain.clinical.entities import PipelineIssue
from services.retrieval.embeddings import (
    EmbeddingModelMismatchError,
    SimilaritySearch,
)
from services.text.normalization import normalize_drug_query_name

RATE_LIMIT_WAIT_HINT_RE = re.compile(
    r"please\s+try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)

###############################################################################
class RagSupportService:
    """RAG document retrieval, similarity search, and language repair utilities."""

    # -------------------------------------------------------------------------
    def __init__(self, consultation: Any) -> None:
        self.consultation = consultation

    # -------------------------------------------------------------------------
    async def fetch_rag_documents(
        self, rag_query: dict[str, str] | None, drug_name: str
    ) -> str | None:
        if not rag_query:
            return None
        normalized_key = normalize_drug_query_name(drug_name)
        drug_rag_query = rag_query.get(drug_name) or rag_query.get(normalized_key)
        if drug_rag_query is None:
            for key, value in rag_query.items():
                if normalize_drug_query_name(key) == normalized_key:
                    drug_rag_query = value
                    break
        if not drug_rag_query:
            return None
        try:
            return await asyncio.to_thread(
                self.search_supporting_documents,
                drug_rag_query,
            )
        except EmbeddingModelMismatchError:
            raise
        except Exception as exc:
            logger.warning(
                "RAG retrieval unavailable for drug '%s'; continuing without supporting documents: %s",
                drug_name,
                exc,
            )
            self.record_rag_retrieval_issue(drug_name=drug_name, error=exc)
            return f"No additional documents provided (reason: RAG retrieval unavailable: {exc})."

    # -------------------------------------------------------------------------
    def record_rag_retrieval_issue(self, *, drug_name: str, error: Exception) -> None:
        issue = PipelineIssue(
            severity="warning",
            code="rag_retrieval_unavailable",
            message=(
                "Internal RAG retrieval was unavailable for "
                f"{drug_name}; analysis continued without supporting documents."
            ),
            field="rag",
            raw_line=f"{drug_name}: {error}",
        )
        if not hasattr(self.consultation, "pipeline_issues"):
            self.consultation.pipeline_issues = []
        self.consultation.pipeline_issues.append(issue)

    # -------------------------------------------------------------------------
    def ensure_similarity_search(self) -> bool:
        if self.consultation.similarity_search is not None:
            return True
        try:
            self.consultation.similarity_search = SimilaritySearch()
        except Exception as exc:
            logger.error("Failed to initialize similarity search: %s", exc)
            self.consultation.similarity_search = None
            return False
        return True

    # -------------------------------------------------------------------------
    def select_excerpt(self, excerpts: list[str]) -> str | None:
        excerpts = [chunk.strip() for chunk in excerpts if chunk.strip()]
        if not excerpts:
            return None
        combined = "\n\n".join(excerpts)
        if len(combined) <= self.consultation.MAX_EXCERPT_LENGTH:
            return combined
        truncated = combined[: self.consultation.MAX_EXCERPT_LENGTH]
        cutoff = truncated.rfind("\n")
        if cutoff > 2000:
            truncated = truncated[:cutoff]
        return truncated.strip()

    # -------------------------------------------------------------------------
    def search_supporting_documents(self, query_text: str | Any) -> str | None:
        if not isinstance(query_text, str):
            return None
        normalized = query_text.strip()
        if not normalized or not self.ensure_similarity_search():
            return None

        consultation = self.consultation
        results = (
            consultation.similarity_search.search_with_reranking(
                normalized,
                candidate_k=consultation.rag_candidate_k,
                final_top_n=consultation.rag_top_n,
                use_reranking=consultation.rag_use_reranking,
            )
            if consultation.similarity_search
            else None
        )
        if not results:
            return None
        fragments: list[str] = []
        for index, record in enumerate(results, start=1):
            fragment = self.format_similarity_fragment(index, record)
            if fragment:
                fragments.append(fragment)
        if not fragments:
            return None
        return "\n".join(fragments)

    # -------------------------------------------------------------------------
    async def repair_language_once(
        self,
        *,
        source_text: str,
        report_language: str,
    ) -> str:
        consultation = self.consultation
        language_map = "en=English, it=Italian, de=German, fr=French, es=Spanish"
        repair_system = (
            "You rewrite clinical text into the requested language only. "
            "Do not add new clinical facts."
        )
        repair_user = (
            f"Target language code: {report_language}\n"
            f"Language map: {language_map}\n"
            "Rewrite the text entirely in the target language. "
            "Do not produce bilingual output. Keep drug names and direct quotes unchanged.\n\n"
            f"Text:\n{source_text}"
        )
        chat_kwargs: dict[str, Any] = {
            "model": consultation.llm_model,
            "messages": [
                {"role": "system", "content": repair_system},
                {"role": "user", "content": repair_user},
            ],
        }
        if consultation.chat_supports_temperature:
            chat_kwargs["temperature"] = 0.0
        else:
            chat_kwargs["options"] = {"temperature": 0.0}
        repaired = await consultation.llm_client.chat(**chat_kwargs)
        return consultation.drug_analysis.coerce_chat_text(repaired).strip()

    # -------------------------------------------------------------------------
    @staticmethod
    def extract_rate_limit_wait_hint_seconds(exc: Exception) -> float | None:
        message = str(exc)
        match = RATE_LIMIT_WAIT_HINT_RE.search(message)
        if match is None:
            return None
        try:
            parsed = float(match.group(1))
        except TypeError, ValueError:
            return None
        if parsed <= 0:
            return None
        return min(parsed + 0.25, 30.0)

    # -------------------------------------------------------------------------
    def format_similarity_fragment(
        self, index: int, record: dict[str, Any]
    ) -> str | None:
        text = str(record.get("text", "")).strip()
        if not text:
            return None
        header = self.format_similarity_header(
            index,
            distance=record.get("distance"),
            rerank_score=record.get("rerank_score"),
        )
        return f"{header}\n{text}"

    # -------------------------------------------------------------------------
    @staticmethod
    def format_similarity_header(
        index: int,
        *,
        distance: Any,
        rerank_score: Any = None,
    ) -> str:
        segments = [f"Document {index}"]
        if isinstance(rerank_score, (int, float)):
            segments.append(f"Rerank: {float(rerank_score):.4f}")
        if isinstance(distance, (int, float)):
            segments.append(f"Distance: {float(distance):.4f}")
        return f"[{' | '.join(segments)}]"
