from __future__ import annotations

import asyncio
from typing import Any

from common.prompts.clinical_assessment import (
    LIVERTOX_CONCLUSION_SYSTEM_PROMPT,
    LIVERTOX_CONCLUSION_USER_PROMPT,
    LIVERTOX_REVISION_CONCLUSION_SYSTEM_PROMPT,
    LIVERTOX_REVISION_CONCLUSION_USER_PROMPT,
)
from common.utils.logger import logger
from domain.clinical.entities import DrugClinicalAssessment
from services.clinical.report_language import (
    report_heading,
)
from services.text.vocabulary import get_text_normalization_snapshot

###############################################################################
class ReportFinalizer:
    """Builds the final patient report and conclusion from per-drug assessments."""

    # -------------------------------------------------------------------------
    def __init__(self, consultation: Any) -> None:
        self.consultation = consultation

    # -------------------------------------------------------------------------
    async def _build_and_finalize_report(
        self,
        entries: list[DrugClinicalAssessment],
        *,
        clinical_context: str | None,
        report_language: str,
        generate_conclusion_fn,
    ) -> str | None:
        consultation = self.consultation
        matched_entries: list[DrugClinicalAssessment] = []
        unresolved_entries: list[DrugClinicalAssessment] = []
        for entry in entries:
            if self.should_render_as_matched_drug(entry):
                matched_entries.append(entry)
                continue
            unresolved_entries.append(entry)

        matched_sections = [
            consultation.render_matched_drug_section(
                entry, report_language=report_language
            )
            for entry in matched_entries
        ]
        matched_sections = [section for section in matched_sections if section]
        unresolved_section = consultation.render_unresolved_mentions_section(
            unresolved_entries,
            report_language=report_language,
        )
        sections: list[str] = []
        if matched_sections:
            sections.append(
                consultation.render_drug_assessment_section(
                    matched_entries,
                    report_language,
                )
            )
        if unresolved_section:
            sections.append(unresolved_section)
        if not sections:
            return None

        combined_report = "\n\n---\n\n".join(sections)
        if matched_sections:
            conclusion = await generate_conclusion_fn(
                clinical_context=clinical_context or "",
                multi_drug_report="\n\n---\n\n".join(matched_sections),
                report_language=report_language,
            )
            if conclusion:
                heading = report_heading("report_section_summary", report_language)
                combined_report = f"{combined_report}\n\n## {heading}\n\n{conclusion}"
        return combined_report

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
        consultation = self.consultation
        report_body = multi_drug_report.strip()
        if not report_body:
            return None
        context_body = clinical_context.strip()
        if not context_body:
            context_body = "No clinical context was provided."
        user_prompt = user_template.format(
            report_language=consultation.escape_braces(report_language),
            clinical_context=consultation.escape_braces(context_body),
            multi_drug_report=consultation.escape_braces(report_body),
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
                    logger.error("Failed to generate clinical conclusion: %s", exc)
                    return None
                delay = consultation.analysis_runner.retry_backoff_seconds(
                    attempt, exc=exc
                )
                logger.warning(
                    "Retrying clinical conclusion generation after error (attempt %d/%d, delay %.2fs): %s",
                    attempt,
                    consultation.analysis_retry_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        conclusion = consultation.drug_analysis.coerce_chat_text(raw_response).strip()
        if conclusion and not consultation.is_materially_in_report_language(
            conclusion, report_language
        ):
            logger.warning(
                "Language mismatch detected for global conclusion (target=%s); applying one repair pass",
                report_language,
            )
            repaired = await consultation.rag_support.repair_language_once(
                source_text=conclusion,
                report_language=report_language,
            )
            if repaired:
                conclusion = repaired
        return conclusion or None

    # -------------------------------------------------------------------------
    async def finalize_patient_report(
        self,
        entries: list[DrugClinicalAssessment],
        *,
        clinical_context: str | None,
        report_language: str,
    ) -> str | None:
        return await self._build_and_finalize_report(
            entries,
            clinical_context=clinical_context,
            report_language=report_language,
            generate_conclusion_fn=self.generate_conclusion,
        )

    # -------------------------------------------------------------------------
    async def finalize_revision_patient_report(
        self,
        entries: list[DrugClinicalAssessment],
        *,
        clinical_context: str | None,
        report_language: str,
    ) -> str | None:
        return await self._build_and_finalize_report(
            entries,
            clinical_context=clinical_context,
            report_language=report_language,
            generate_conclusion_fn=self.generate_revision_conclusion,
        )

    # -------------------------------------------------------------------------
    async def generate_conclusion(
        self,
        *,
        clinical_context: str,
        multi_drug_report: str,
        report_language: str,
    ) -> str | None:
        return await self._run_conclusion(
            clinical_context=clinical_context,
            multi_drug_report=multi_drug_report,
            report_language=report_language,
            system_template=LIVERTOX_CONCLUSION_SYSTEM_PROMPT,
            user_template=LIVERTOX_CONCLUSION_USER_PROMPT,
        )

    # -------------------------------------------------------------------------
    async def generate_revision_conclusion(
        self,
        *,
        clinical_context: str,
        multi_drug_report: str,
        report_language: str,
    ) -> str | None:
        return await self._run_conclusion(
            clinical_context=clinical_context,
            multi_drug_report=multi_drug_report,
            report_language=report_language,
            system_template=LIVERTOX_REVISION_CONCLUSION_SYSTEM_PROMPT,
            user_template=LIVERTOX_REVISION_CONCLUSION_USER_PROMPT,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def should_render_as_matched_drug(entry: DrugClinicalAssessment) -> bool:
        status = (entry.match_status or "").strip().lower()
        return status in {"matched", "matched_with_excerpt", "matched_no_excerpt"}

    # -------------------------------------------------------------------------
    @staticmethod
    def bibliography_source_label() -> str:
        return get_text_normalization_snapshot().knowledge_source_references.get(
            "livertox", "LiverTox"
        )
