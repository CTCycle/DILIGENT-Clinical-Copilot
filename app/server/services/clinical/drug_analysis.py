from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any

from common.prompts.clinical_assessment import (
    LIVERTOX_CLINICAL_SYSTEM_PROMPT,
    LIVERTOX_CLINICAL_USER_PROMPT,
    LIVERTOX_REVISION_CLINICAL_SYSTEM_PROMPT,
    LIVERTOX_REVISION_CLINICAL_USER_PROMPT,
)
from common.utils.logger import logger
from domain.clinical.entities import DrugRucamAssessment, DrugSuspensionContext

###############################################################################
class DrugAnalysisService:
    """Handles per-drug LLM consultation — building prompts, calling the LLM, parsing responses."""

    # -------------------------------------------------------------------------
    def __init__(self, consultation: Any) -> None:
        self.consultation = consultation

    # -------------------------------------------------------------------------
    async def _build_and_run_drug_analysis(
        self,
        *,
        drug_name: str,
        canonical_name: str,
        origins: list[str],
        extraction_metadata: list[dict[str, Any]],
        livertox_status: str,
        excerpt: str,
        rag_context: str | None,
        clinical_context: str,
        suspension: DrugSuspensionContext,
        visit_date: date | None,
        pattern_summary: str,
        metadata: dict[str, Any] | None,
        rucam: DrugRucamAssessment | None,
        knowledge_prompt: str,
        report_language: str,
        system_template: str,
        user_template: str,
    ) -> str:
        consultation = self.consultation
        start_details = consultation.format_start_prompt(suspension)
        suspension_details = consultation.format_suspension_prompt(suspension)
        timeline_note = (
            suspension.note
            or "No explicit timeline notes were available from extraction metadata."
        )
        visit_date_anchor = consultation.format_visit_date_anchor(visit_date)
        score, metadata_block = consultation.prepare_metadata_prompt(metadata)
        retrieved_documents_block = (
            f"Retrieved documents:\n{rag_context.strip()}"
            if rag_context and rag_context.strip()
            else ""
        )
        origin_block = ", ".join(origins) if origins else "unknown"
        metadata_items = [
            f"- {json.dumps(item, ensure_ascii=False)}"
            for item in extraction_metadata
            if isinstance(item, dict) and item
        ]
        extraction_block = (
            "\n".join(metadata_items) if metadata_items else "- Not available"
        )
        rucam_block = consultation.format_rucam_prompt_block(rucam)
        user_prompt = user_template.format(
            drug_name=consultation.escape_braces(drug_name.strip() or drug_name),
            report_language=consultation.escape_braces(report_language),
            canonical_name=consultation.escape_braces(
                canonical_name.strip() or canonical_name
            ),
            origins=consultation.escape_braces(origin_block),
            extraction_metadata=consultation.escape_braces(extraction_block),
            livertox_status=consultation.escape_braces(livertox_status),
            excerpt=consultation.escape_braces(excerpt),
            retrieved_documents_block=consultation.escape_braces(
                retrieved_documents_block
            ),
            clinical_context=consultation.escape_braces(clinical_context),
            visit_date_anchor=consultation.escape_braces(visit_date_anchor),
            therapy_start_details=consultation.escape_braces(start_details),
            suspension_details=consultation.escape_braces(suspension_details),
            timeline_note=consultation.escape_braces(timeline_note),
            pattern_summary=consultation.escape_braces(pattern_summary),
            rucam_block=consultation.escape_braces(rucam_block),
            knowledge_prompt=consultation.escape_braces(knowledge_prompt),
            metadata_block=consultation.escape_braces(metadata_block),
            livertox_score=consultation.escape_braces(score),
        )
        messages = [
            {"role": "system", "content": system_template.strip()},
            {"role": "user", "content": user_prompt},
        ]
        chat_kwargs: dict[str, Any] = {
            "model": consultation.llm_model,
            "messages": messages,
        }
        if consultation.chat_supports_temperature:
            chat_kwargs["temperature"] = consultation.temperature
        else:
            chat_kwargs["options"] = {"temperature": consultation.temperature}
        raw_response: Any = None
        for attempt in range(1, consultation.analysis_retry_attempts + 1):
            try:
                raw_response = await consultation.llm_client.chat(**chat_kwargs)
                break
            except Exception as exc:
                if attempt >= consultation.analysis_retry_attempts:
                    raise RuntimeError(
                        f"LLM analysis failed for {drug_name}: {exc}"
                    ) from exc
                delay = consultation.analysis_runner.retry_backoff_seconds(
                    attempt, exc=exc
                )
                logger.warning(
                    "Retrying LLM analysis for '%s' after error (attempt %d/%d, delay %.2fs): %s",
                    drug_name,
                    attempt,
                    consultation.analysis_retry_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        response_text = self.coerce_chat_text(raw_response).strip()
        if not consultation.is_materially_in_report_language(
            response_text, report_language
        ):
            logger.warning(
                "Language mismatch detected for drug analysis '%s' (target=%s); applying one repair pass",
                drug_name,
                report_language,
            )
            repaired_text = await consultation.rag_support.repair_language_once(
                source_text=response_text,
                report_language=report_language,
            )
            if repaired_text:
                response_text = repaired_text
        return response_text

    # -------------------------------------------------------------------------
    async def request_drug_analysis(
        self,
        *,
        drug_name: str,
        canonical_name: str,
        origins: list[str],
        extraction_metadata: list[dict[str, Any]],
        livertox_status: str,
        excerpt: str,
        rag_context: str | None,
        clinical_context: str,
        suspension: DrugSuspensionContext,
        visit_date: date | None,
        pattern_summary: str,
        metadata: dict[str, Any] | None,
        rucam: DrugRucamAssessment | None,
        knowledge_prompt: str = "No supplemental knowledge prompt available.",
        report_language: str = "en",
    ) -> str:
        return await self._build_and_run_drug_analysis(
            drug_name=drug_name,
            canonical_name=canonical_name,
            origins=origins,
            extraction_metadata=extraction_metadata,
            livertox_status=livertox_status,
            excerpt=excerpt,
            rag_context=rag_context,
            clinical_context=clinical_context,
            suspension=suspension,
            visit_date=visit_date,
            pattern_summary=pattern_summary,
            metadata=metadata,
            rucam=rucam,
            knowledge_prompt=knowledge_prompt,
            report_language=report_language,
            system_template=LIVERTOX_CLINICAL_SYSTEM_PROMPT,
            user_template=LIVERTOX_CLINICAL_USER_PROMPT,
        )

    # -------------------------------------------------------------------------
    async def request_revision_drug_analysis(
        self,
        *,
        drug_name: str,
        canonical_name: str,
        origins: list[str],
        extraction_metadata: list[dict[str, Any]],
        livertox_status: str,
        excerpt: str,
        rag_context: str | None,
        clinical_context: str,
        suspension: DrugSuspensionContext,
        visit_date: date | None,
        pattern_summary: str,
        metadata: dict[str, Any] | None,
        rucam: DrugRucamAssessment | None,
        knowledge_prompt: str = "No supplemental knowledge prompt available.",
        report_language: str = "en",
    ) -> str:
        return await self._build_and_run_drug_analysis(
            drug_name=drug_name,
            canonical_name=canonical_name,
            origins=origins,
            extraction_metadata=extraction_metadata,
            livertox_status=livertox_status,
            excerpt=excerpt,
            rag_context=rag_context,
            clinical_context=clinical_context,
            suspension=suspension,
            visit_date=visit_date,
            pattern_summary=pattern_summary,
            metadata=metadata,
            rucam=rucam,
            knowledge_prompt=knowledge_prompt,
            report_language=report_language,
            system_template=LIVERTOX_REVISION_CLINICAL_SYSTEM_PROMPT,
            user_template=LIVERTOX_REVISION_CLINICAL_USER_PROMPT,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def coerce_chat_text(raw_response: Any) -> str:
        if isinstance(raw_response, str):
            return raw_response.strip()
        if isinstance(raw_response, dict):
            for key in ("content", "text", "response"):
                value = raw_response.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return json.dumps(raw_response, ensure_ascii=False)
        return str(raw_response).strip()
