from __future__ import annotations

import asyncio
import json
import re
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
from common.prompts.clinical_assessment_builders import (
    build_livertox_conclusion_user_prompt,
    build_livertox_drug_assessment_user_prompt,
)
from common.prompts.language import (
    CLINICAL_LANGUAGE_REWRITE_SYSTEM_PROMPT,
    build_clinical_language_rewrite_user_prompt,
)
from common.utils.logger import logger
from domain.clinical.entities import DrugRucamAssessment, DrugSuspensionContext
from services.clinical import hepatox_scoring
from services.clinical.exposure_timeline import ExposureTimelineService
from services.llm.cloud import CloudLLMClient
from services.llm.generation_policy import GenerationPurpose
from services.llm.ollama_client import OllamaClient

RATE_LIMIT_WAIT_HINT_RE = re.compile(
    r"please\s+try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)


###############################################################################
class DrugAnalysisService:
    """Handles per-drug LLM orchestration, invocation, and response processing."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        llm_client: OllamaClient | CloudLLMClient,
        llm_model: str,
        exposure_timeline: ExposureTimelineService,
        retry_attempts: int,
    ) -> None:
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.exposure_timeline = exposure_timeline
        self.retry_attempts = max(int(retry_attempts), 1)

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
        start_details = self.exposure_timeline.format_start_prompt(suspension)
        suspension_details = self.exposure_timeline.format_suspension_prompt(suspension)
        timeline_note = (
            suspension.note
            or "No explicit timeline notes were available from extraction metadata."
        )
        visit_date_anchor = self.exposure_timeline.format_visit_date_anchor(visit_date)
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
        user_prompt = build_livertox_drug_assessment_user_prompt(
            revision=user_template == LIVERTOX_REVISION_CLINICAL_USER_PROMPT,
            drug_name=self.escape_braces(drug_name.strip() or drug_name),
            report_language=self.escape_braces(report_language),
            canonical_name=self.escape_braces(canonical_name.strip() or canonical_name),
            origins=self.escape_braces(origin_block),
            extraction_metadata=self.escape_braces(extraction_block),
            livertox_status=self.escape_braces(livertox_status),
            excerpt=self.escape_braces(excerpt),
            retrieved_documents_block=self.escape_braces(retrieved_documents_block),
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
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_prompt},
        ]
        raw_response: Any = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                raw_response = await self._chat(
                    model=self.llm_model,
                    messages=messages,
                    purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
                )
                break
            except Exception as exc:
                if attempt >= self.retry_attempts:
                    raise RuntimeError(
                        f"LLM analysis failed for {drug_name}: {exc}"
                    ) from exc
                delay = self.retry_backoff_seconds(attempt, exc=exc)
                logger.warning(
                    "Retrying LLM analysis for '%s' after error (attempt %d/%d, delay %.2fs): %s",
                    drug_name,
                    attempt,
                    self.retry_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        response_text = self.coerce_chat_text(raw_response).strip()
        if not self.is_materially_in_report_language(response_text, report_language):
            logger.warning(
                "Language mismatch detected for drug analysis '%s' (target=%s); applying one repair pass",
                drug_name,
                report_language,
            )
            repaired_text = await self.repair_language_once(
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
    def escape_braces(value: str) -> str:
        return value.replace("{", "{{").replace("}", "}}")

    # -------------------------------------------------------------------------
    @staticmethod
    def is_materially_in_report_language(text: str, report_language: str) -> bool:
        return hepatox_scoring.is_materially_in_report_language(text, report_language)

    # -------------------------------------------------------------------------
    @staticmethod
    def resolve_livertox_score(metadata: dict[str, Any] | None) -> str:
        if not metadata:
            return "Not available"
        score = str(metadata.get("likelihood_score") or "").strip()
        return score.upper() if score and score.isalpha() else score or "Not available"

    # -------------------------------------------------------------------------
    def prepare_metadata_prompt(
        self, metadata: dict[str, Any] | None
    ) -> tuple[str, str]:
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

    # -------------------------------------------------------------------------
    @staticmethod
    def format_rucam_prompt_block(rucam: DrugRucamAssessment | None) -> str:
        if rucam is None:
            return "RUCAM evidence is not available."
        limitations = ", ".join((rucam.limitations or [])[:3]) or "not specified"
        if rucam.total_score is None:
            component_labels = ", ".join(
                component.label for component in (rucam.components or [])[:5]
            ) or "none captured"
            return (
                "- Numerical updated-RUCAM total: not calculated\n"
                f"- Evidence checklist components: {component_labels}\n"
                f"- Key limitations: {limitations}\n"
                "- Interpretation: do not infer a RUCAM probability category from this checklist."
            )
        return (
            f"- Patient-record RUCAM score: {rucam.total_score}\n"
            f"- Patient-record category: {rucam.causality_category}\n"
            f"- Source: {rucam.score_source or 'current patient record'}\n"
            "- Interpretation: this score was preserved from the patient record and was not independently recalculated by DILIGENT.\n"
            f"- Key limitations: {limitations}"
        )

    # -------------------------------------------------------------------------
    async def _run_conclusion(
        self,
        *,
        clinical_context: str,
        multi_drug_report: str,
        report_language: str,
        system_template: str,
        user_template: str,
    ) -> str | None:
        report_body = multi_drug_report.strip()
        if not report_body:
            return None
        context_body = clinical_context.strip() or "No clinical context was provided."
        user_prompt = build_livertox_conclusion_user_prompt(
            revision=user_template == LIVERTOX_REVISION_CONCLUSION_USER_PROMPT,
            report_language=self.escape_braces(report_language),
            clinical_context=self.escape_braces(context_body),
            multi_drug_report=self.escape_braces(report_body),
        )
        messages = [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_prompt},
        ]
        raw_response: Any = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                raw_response = await self._chat(
                    model=self.llm_model,
                    messages=messages,
                    purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
                )
                break
            except Exception as exc:
                if attempt >= self.retry_attempts:
                    logger.error("Failed to generate clinical conclusion: %s", exc)
                    return None
                await asyncio.sleep(self.retry_backoff_seconds(attempt, exc=exc))
        conclusion = self.coerce_chat_text(raw_response).strip()
        if conclusion and not self.is_materially_in_report_language(
            conclusion, report_language
        ):
            repaired = await self.repair_language_once(
                source_text=conclusion, report_language=report_language
            )
            if repaired:
                conclusion = repaired
        return conclusion or None

    # -------------------------------------------------------------------------
    async def generate_conclusion(self, **kwargs: Any) -> str | None:
        return await self._run_conclusion(
            **kwargs,
            system_template=LIVERTOX_CONCLUSION_SYSTEM_PROMPT,
            user_template=LIVERTOX_CONCLUSION_USER_PROMPT,
        )

    # -------------------------------------------------------------------------
    async def generate_revision_conclusion(self, **kwargs: Any) -> str | None:
        return await self._run_conclusion(
            **kwargs,
            system_template=LIVERTOX_REVISION_CONCLUSION_SYSTEM_PROMPT,
            user_template=LIVERTOX_REVISION_CONCLUSION_USER_PROMPT,
        )

    # -------------------------------------------------------------------------
    async def repair_language_once(
        self,
        *,
        source_text: str,
        report_language: str,
    ) -> str:
        repaired = await self._chat(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": CLINICAL_LANGUAGE_REWRITE_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_clinical_language_rewrite_user_prompt(
                        source_text=source_text,
                        report_language=report_language,
                    ),
                },
            ],
            purpose=GenerationPurpose.FAITHFUL_REWRITE,
        )
        return self.coerce_chat_text(repaired).strip()

    # -------------------------------------------------------------------------
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        purpose: GenerationPurpose,
    ) -> dict[str, Any] | str:
        if isinstance(self.llm_client, CloudLLMClient):
            return await self.llm_client.chat(
                model=model,
                messages=messages,
                purpose=purpose,
            )
        return await self.llm_client.chat(
            model=model,
            messages=messages,
            purpose=purpose,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def extract_rate_limit_wait_hint_seconds(exc: Exception) -> float | None:
        match = RATE_LIMIT_WAIT_HINT_RE.search(str(exc))
        if match is None:
            return None
        try:
            parsed = float(match.group(1))
        except (TypeError, ValueError):
            return None
        return min(parsed + 0.25, 30.0) if parsed > 0 else None

    # -------------------------------------------------------------------------
    def retry_backoff_seconds(
        self, attempt: int, *, exc: Exception | None = None
    ) -> float:
        if exc is not None:
            hinted_wait = self.extract_rate_limit_wait_hint_seconds(exc)
            if hinted_wait is not None:
                return hinted_wait
        normalized_attempt = max(int(attempt), 1)
        return min(8.0, 0.75 * (2 ** (normalized_attempt - 1)))

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
