from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass
from typing import Any

from common.utils.logger import logger
from domain.clinical.entities import PipelineIssue, RagDocumentReference
from services.retrieval.embeddings import (
    EmbeddingModelMismatchError,
    SimilaritySearch,
)
from services.text.normalization import normalize_drug_query_name
from services.llm.generation_policy import GenerationPurpose

RATE_LIMIT_WAIT_HINT_RE = re.compile(
    r"please\s+try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)

###############################################################################
@dataclass(frozen=True)
class RagRetrievalBundle:
    context_text: str | None
    references: tuple[RagDocumentReference, ...]

###############################################################################
class RagSupportService:
    """RAG document retrieval, similarity search, and language repair utilities."""

    # -------------------------------------------------------------------------
    def __init__(self, consultation: Any) -> None:
        self.consultation = consultation
        self._retrieval_issue_lock = threading.Lock()
        self._retrieval_failed_drugs: list[str] = []

    # -------------------------------------------------------------------------
    async def fetch_rag_documents(
        self, rag_query: dict[str, str] | None, drug_name: str
    ) -> RagRetrievalBundle | None:
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
            return None

    # -------------------------------------------------------------------------
    def record_rag_retrieval_issue(self, *, drug_name: str, error: Exception) -> None:
        with self._retrieval_issue_lock:
            if drug_name not in self._retrieval_failed_drugs:
                self._retrieval_failed_drugs.append(drug_name)
            failed_names = ", ".join(self._retrieval_failed_drugs)
            count = len(self._retrieval_failed_drugs)
            message = (
                "Internal RAG retrieval became unavailable; analysis continued "
                f"without supporting documents for {count} drug"
                f"{'s' if count != 1 else ''}: {failed_names}."
            )
            raw_line = f"{failed_names}: {error}"
            if not hasattr(self.consultation, "pipeline_issues"):
                self.consultation.pipeline_issues = []
            existing = next(
                (
                    issue
                    for issue in self.consultation.pipeline_issues
                    if issue.code == "rag_retrieval_unavailable"
                ),
                None,
            )
            if existing is not None:
                existing.message = message
                existing.raw_line = raw_line
                return
            self.consultation.pipeline_issues.append(
                PipelineIssue(
                    severity="warning",
                    code="rag_retrieval_unavailable",
                    message=message,
                    field="rag",
                    raw_line=raw_line,
                )
            )

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
    def search_supporting_documents(
        self, query_text: str | Any
    ) -> RagRetrievalBundle | None:
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
        references: list[RagDocumentReference] = []
        excluded_count = 0
        seen_references: set[
            tuple[str, int | None, int | None, int | None, int | None]
        ] = set()
        for index, record in enumerate(results, start=1):
            if not self.is_context_eligible(record):
                excluded_count += 1
                continue
            fragment = self.format_similarity_fragment(index, record)
            if fragment:
                fragments.append(fragment)
            reference = self.build_document_reference(record)
            if reference is None:
                continue
            dedupe_key = (
                reference.file_name.casefold(),
                reference.page_start,
                reference.page_end,
                reference.line_start,
                reference.line_end,
            )
            if dedupe_key in seen_references:
                continue
            seen_references.add(dedupe_key)
            references.append(reference)
        if excluded_count:
            self.record_low_relevance_issue(excluded_count)
        if not fragments:
            return None
        return RagRetrievalBundle(
            context_text="\n".join(fragments),
            references=tuple(references),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def is_context_eligible(record: dict[str, Any]) -> bool:
        """Never inject an explicitly non-relevant reranked chunk into clinical prompts."""
        text = str(record.get("text") or "").strip()
        if not text:
            return False
        score = record.get("rerank_score")
        return not isinstance(score, (int, float)) or float(score) > 0.0

    # -------------------------------------------------------------------------
    def record_low_relevance_issue(self, excluded_count: int) -> None:
        if not hasattr(self.consultation, "pipeline_issues"):
            self.consultation.pipeline_issues = []
        if any(
            issue.code == "rag_low_relevance_excluded"
            for issue in self.consultation.pipeline_issues
        ):
            return
        self.consultation.pipeline_issues.append(
            PipelineIssue(
                severity="warning",
                code="rag_low_relevance_excluded",
                message=(
                    f"Excluded {excluded_count} low-relevance retrieval chunk"
                    f"{'s' if excluded_count != 1 else ''} from clinical context."
                ),
                field="rag",
            )
        )

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
            "purpose": GenerationPurpose.JSON_REPAIR,
        }
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
        except (TypeError, ValueError):
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
    def build_document_reference(record: dict[str, Any]) -> RagDocumentReference | None:
        metadata = record.get("metadata")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        file_name = str(
            record.get("file_name")
            or metadata_dict.get("source_file_name")
            or metadata_dict.get("file_name")
            or metadata_dict.get("source_relative_path")
            or record.get("source")
            or ""
        ).strip()
        if not file_name:
            return None

        page_number = RagSupportService._coerce_page_number(record.get("page_number"))
        page_start = RagSupportService._coerce_page_number(
            metadata_dict.get("page_start")
        )
        page_end = RagSupportService._coerce_page_number(metadata_dict.get("page_end"))
        line_start = RagSupportService._coerce_page_number(
            record.get("line_start") or metadata_dict.get("line_start")
        )
        line_end = RagSupportService._coerce_page_number(
            record.get("line_end") or metadata_dict.get("line_end")
        )
        if page_number is not None:
            page_start = page_start or page_number
            page_end = page_end or page_number

        return RagDocumentReference(
            file_name=file_name,
            page_start=page_start,
            page_end=page_end,
            line_start=line_start,
            line_end=line_end,
            document_title=RagSupportService._coerce_optional_text(
                record.get("document_title") or metadata_dict.get("document_title")
            ),
            section_title=RagSupportService._coerce_optional_text(
                record.get("section_title") or metadata_dict.get("section_title")
            ),
            chunk_id=RagSupportService._coerce_optional_text(
                record.get("chunk_id") or metadata_dict.get("chunk_id")
            ),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_page_number(value: Any) -> int | None:
        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 1 else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_optional_text(value: Any) -> str | None:
        stripped = str(value).strip() if value is not None else ""
        return stripped or None

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
