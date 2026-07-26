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
    LIVERTOX_CONCLUSION_SYSTEM_PROMPT,
    LIVERTOX_CONCLUSION_USER_PROMPT,
    LIVERTOX_REVISION_CONCLUSION_SYSTEM_PROMPT,
    LIVERTOX_REVISION_CONCLUSION_USER_PROMPT,
)
from common.utils.logger import logger
from domain.clinical.entities import DrugRucamAssessment, DrugSuspensionContext
from services.llm.generation_policy import GenerationPurpose
from services.clinical import hepatox_scoring

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
        start_details = consultation.exposure_timeline.format_start_prompt(suspension)
        suspension_details = consultation.exposure_timeline.format_suspension_prompt(suspension)
        timeline_note = (
            suspension.note
            or "No explicit timeline notes were available from extraction metadata."
        )
        visit_date_anchor = consultation.exposure_timeline.format_visit_date_anchor(visit_date)
        score, metadata_block = self.prepare_metadata_prompt(metadata)
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
        rucam_block = self.format_rucam_prompt_block(rucam)
        user_prompt = user_template.format(
            drug_name=self.escape_braces(drug_name.strip() or drug_name),
            report_language=self.escape_braces(report_language),
            canonical_name=self.escape_braces(
                canonical_name.strip() or canonical_name
            ),
            origins=self.escape_braces(origin_block),
            extraction_metadata=self.escape_braces(extraction_block),
            livertox_status=self.escape_braces(livertox_status),
            excerpt=self.escape_braces(excerpt),
            retrieved_documents_block=self.escape_braces(
                retrieved_documents_block
            ),
            clinical_context=self.escape_braces(clinical_context),
            visit_date_anchor=self.escape_braces(visit_date_anchor),
            therapy_start_details=self.escape_braces(start_details),
            suspension_details=self.escape_braces(suspension_details),
            timeline_note=self.escape_braces(timeline_note),
            pattern_summary=self.escape_braces(pattern_summary),
            rucam_block=self.escape_braces(rucam_block),
            knowledge_prompt=self.escape_braces(knowledge_prompt),
            metadata_block=self.escape_braces(metadata_block),
            livertox_score=self.escape_braces(score),
        )
        messages = [
            {"role": "system", "content": system_template.strip()},
            {"role": "user", "content": user_prompt},
        ]
        chat_kwargs: dict[str, Any] = {
            "model": consultation.llm_model,
            "messages": messages,
            "purpose": GenerationPurpose.CLINICAL_SYNTHESIS,
        }
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
        if not self.is_materially_in_report_language(
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

    @staticmethod
    def escape_braces(value: str) -> str:
        return value.replace("{", "{{").replace("}", "}}")

    @staticmethod
    def is_materially_in_report_language(text: str, report_language: str) -> bool:
        return hepatox_scoring.is_materially_in_report_language(text, report_language)

    @staticmethod
    def resolve_livertox_score(metadata: dict[str, Any] | None) -> str:
        if not metadata:
            return "Not available"
        score = str(metadata.get("likelihood_score") or "").strip()
        return score.upper() if score and score.isalpha() else score or "Not available"

    def prepare_metadata_prompt(self, metadata: dict[str, Any] | None) -> tuple[str, str]:
        score = self.resolve_livertox_score(metadata)
        details = [f"- Likelihood score: {score}"]
        if metadata:
            for label, raw in (
                ("Agent classification", metadata.get("agent_classification")),
                ("Primary classification", metadata.get("primary_classification")),
                ("Secondary classification", metadata.get("secondary_classification")),
                ("Reference count", metadata.get("reference_count")),
                ("Year approved", metadata.get("year_approved")),
            ):
                value = str(raw).strip() if raw is not None else ""
                if value and value.lower() != "nan":
                    details.append(f"- {label}: {value}")
        if len(details) == 1:
            details.append("- No additional LiverTox metadata was available.")
        return score, "\n".join(details)

    @staticmethod
    def format_rucam_prompt_block(rucam: DrugRucamAssessment | None) -> str:
        if rucam is None:
            return "Estimated RUCAM not available."
        limitations = ", ".join((rucam.limitations or [])[:3]) or "not specified"
        return (
            f"- Score: {rucam.total_score}\n- Category: {rucam.causality_category}\n"
            f"- Confidence: {rucam.confidence}\n- Estimated due to incomplete clinical data: yes\n"
            f"- Key limitations: {limitations}"
        )

    async def _run_conclusion(self, *, clinical_context: str, multi_drug_report: str, report_language: str, system_template: str, user_template: str) -> str | None:
        report_body = multi_drug_report.strip()
        if not report_body:
            return None
        context_body = clinical_context.strip() or "No clinical context was provided."
        messages = [
            {"role": "system", "content": system_template.strip()},
            {"role": "user", "content": user_template.format(
                report_language=self.escape_braces(report_language),
                clinical_context=self.escape_braces(context_body),
                multi_drug_report=self.escape_braces(report_body),
            )},
        ]
        raw_response: Any = None
        for attempt in range(1, self.consultation.analysis_retry_attempts + 1):
            try:
                raw_response = await self.consultation.llm_client.chat(
                    model=self.consultation.llm_model,
                    messages=messages,
                    purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
                )
                break
            except Exception as exc:
                if attempt >= self.consultation.analysis_retry_attempts:
                    logger.error("Failed to generate clinical conclusion: %s", exc)
                    return None
                await asyncio.sleep(self.consultation.analysis_runner.retry_backoff_seconds(attempt, exc=exc))
        conclusion = self.coerce_chat_text(raw_response).strip()
        if conclusion and not self.is_materially_in_report_language(conclusion, report_language):
            repaired = await self.consultation.rag_support.repair_language_once(
                source_text=conclusion, report_language=report_language
            )
            if repaired:
                conclusion = repaired
        return conclusion or None

    async def generate_conclusion(self, **kwargs: Any) -> str | None:
        return await self._run_conclusion(
            **kwargs,
            system_template=LIVERTOX_CONCLUSION_SYSTEM_PROMPT,
            user_template=LIVERTOX_CONCLUSION_USER_PROMPT,
        )

    async def generate_revision_conclusion(self, **kwargs: Any) -> str | None:
        return await self._run_conclusion(
            **kwargs,
            system_template=LIVERTOX_REVISION_CONCLUSION_SYSTEM_PROMPT,
            user_template=LIVERTOX_REVISION_CONCLUSION_USER_PROMPT,
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
