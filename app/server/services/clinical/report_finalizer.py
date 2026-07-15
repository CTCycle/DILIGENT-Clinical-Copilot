from __future__ import annotations

import asyncio
import re
from typing import Any

from common.prompts.clinical_assessment import (
    LIVERTOX_CONCLUSION_SYSTEM_PROMPT,
    LIVERTOX_CONCLUSION_USER_PROMPT,
    LIVERTOX_REVISION_CONCLUSION_SYSTEM_PROMPT,
    LIVERTOX_REVISION_CONCLUSION_USER_PROMPT,
)
from common.utils.logger import logger
from domain.clinical.entities import (
    DrugClinicalAssessment,
    PatientLabTimeline,
    RagDocumentReference,
)
from services.clinical.report_language import (
    phrase,
)
from services.clinical.report_language import report_heading
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
        references = [
            reference for entry in entries for reference in entry.rag_references
        ]
        matched_sections = [
            self.sanitize_generated_text(section, references)
            for section in matched_sections
        ]
        unresolved_section = consultation.render_unresolved_mentions_section(
            unresolved_entries,
            report_language=report_language,
        )
        sections: list[str] = []
        if matched_sections:
            sections.append(
                self.render_drug_assessment_section(
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
                conclusion = self.sanitize_generated_text(conclusion, references)
                heading = report_heading("report_section_summary", report_language)
                combined_report = f"{combined_report}\n\n## {heading}\n\n{conclusion}"
        bibliography = self.build_rag_bibliography_section(
            entries,
            report_language=report_language,
        )
        if bibliography:
            combined_report = f"{combined_report}\n\n{bibliography}"
        return combined_report

    # -------------------------------------------------------------------------
    def render_drug_assessment_section(
        self,
        assessments: list[DrugClinicalAssessment],
        language: str,
    ) -> str:
        lines = [f"## {report_heading('drug_assessments', language)}", ""]
        for assessment in assessments:
            lines.extend(
                [
                    self.consultation.render_matched_drug_section(
                        assessment, report_language=language
                    ),
                    "",
                ]
            )
        return "\n".join(lines).strip()

    # -------------------------------------------------------------------------
    def render_laboratory_section(
        self, lab_timeline: PatientLabTimeline | None, language: str
    ) -> str:
        lines = [f"## {report_heading('laboratory_history', language)}", ""]
        if lab_timeline is None or not lab_timeline.entries:
            lines.append(phrase("not_available", language))
            return "\n".join(lines).strip()
        for entry in lab_timeline.entries:
            value = entry.value if entry.value is not None else (entry.value_text or "?")
            lines.append(f"- {entry.marker_name}: {value} {entry.unit or ''}".strip())
        return "\n".join(lines).strip()

    # -------------------------------------------------------------------------
    def render_bibliography_section(
        self, matches: list[dict[str, Any]], language: str
    ) -> str:
        lines = [f"## {report_heading('bibliography', language)}", ""]
        for match in matches:
            name = str(match.get("matched_livertox_name") or match.get("extracted_name") or "").strip()
            if name:
                strategy = str(match.get("match_strategy") or "unknown").strip()
                status = "rxnav_validated" if match.get("rxnav_validated") else "rxnav_unvalidated"
                lines.append(f"- {name} ({strategy}, {status})")
        if len(lines) == 2:
            lines.append(phrase("not_available", language))
        return "\n".join(lines).strip()

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
        return status in {
            "accepted_exact_livertox",
            "accepted_livertox_without_rxnav",
            "accepted_rxnav_validated",
            "matched",
            "matched_with_excerpt",
            "matched_no_excerpt",
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def bibliography_source_label() -> str:
        return get_text_normalization_snapshot().knowledge_source_references.get(
            "livertox", "LiverTox"
        )

    # -------------------------------------------------------------------------
    def build_rag_bibliography_section(
        self,
        entries: list[DrugClinicalAssessment],
        *,
        report_language: str,
    ) -> str | None:
        references = [
            reference for entry in entries for reference in entry.rag_references
        ]
        rendered_lines = self.render_canonical_references(references)
        if not rendered_lines:
            return None

        heading = report_heading("bibliography", report_language)
        return "\n".join([f"## {heading}", "", *rendered_lines]).strip()

    # -------------------------------------------------------------------------
    @classmethod
    def render_canonical_references(
        cls, references: list[RagDocumentReference]
    ) -> list[str]:
        grouped: dict[
            tuple[str, int | None, int | None], list[tuple[int, int] | None]
        ] = {}
        page_only: dict[str, list[tuple[int, int]]] = {}
        for reference in references:
            file_name = reference.file_name.strip()
            if not file_name:
                continue
            key = (
                file_name,
                reference.page_start,
                reference.page_end or reference.page_start,
            )
            line_range = (
                (reference.line_start, reference.line_end or reference.line_start)
                if reference.line_start is not None
                else None
            )
            if line_range is None and reference.page_start is not None:
                page_only.setdefault(file_name, []).append(
                    (reference.page_start, reference.page_end or reference.page_start)
                )
                continue
            grouped.setdefault(key, []).append(line_range)
        lines: list[str] = []
        for file_name in sorted(page_only, key=str.casefold):
            ranges = cls.merge_ranges(page_only[file_name])
            segments = [
                str(start) if start == end else f"{start}-{end}"
                for start, end in ranges
            ]
            label = "p." if len(segments) == 1 and "-" not in segments[0] else "pp."
            lines.append(f"- {file_name}, {label} {', '.join(segments)}")
        for (file_name, page_start, page_end), ranges in sorted(
            grouped.items(),
            key=lambda item: (item[0][0].casefold(), item[0][1] or 0, item[0][2] or 0),
        ):
            merged = cls.merge_ranges([item for item in ranges if item is not None])
            if merged:
                for line_start, line_end in merged:
                    lines.append(
                        cls.format_location(
                            file_name, page_start, page_end, line_start, line_end
                        )
                    )
            else:
                lines.append(
                    cls.format_location(file_name, page_start, page_end, None, None)
                )
        return list(dict.fromkeys(lines))

    # -------------------------------------------------------------------------
    @staticmethod
    def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in sorted(set(ranges)):
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    # -------------------------------------------------------------------------
    @staticmethod
    def format_location(
        file_name: str,
        page_start: int | None,
        page_end: int | None,
        line_start: int | None,
        line_end: int | None,
    ) -> str:
        locations: list[str] = []
        if page_start is not None:
            if page_end is not None and page_end != page_start:
                locations.append(f"pp. {page_start}-{page_end}")
            else:
                locations.append(f"p. {page_start}")
        if line_start is not None:
            locations.append(
                f"lines {line_start}-{line_end}"
                if line_end != line_start
                else f"line {line_start}"
            )
        return f"- {file_name}, {', '.join(locations) if locations else 'location not available'}"

    # -------------------------------------------------------------------------
    @staticmethod
    def sanitize_generated_text(
        text: str, references: list[RagDocumentReference]
    ) -> str:
        forbidden = {
            "bibliography",
            "references",
            "sources",
            "works cited",
            "bibliografia",
            "riferimenti",
            "fonti",
            "literaturverzeichnis",
            "quellen",
            "bibliographie",
            "références",
            "fuentes",
            "referencias",
        }
        kept: list[str] = []
        dropping = False
        for line in str(text or "").splitlines():
            heading = re.match(r"^##\s+(.+?)\s*$", line)
            if heading:
                dropping = heading.group(1).strip().casefold() in forbidden
                if dropping:
                    continue
            if dropping:
                continue
            candidate = line
            for reference in references:
                escaped = re.escape(reference.file_name)
                candidate = re.sub(
                    rf"\[?{escaped}\s*,?\s*(?:pp?\.\s*\d+(?:-\d+)?(?:\s*,\s*lines?\s*\d+(?:-\d+)?)?|lines?\s*\d+(?:-\d+)?)\]?",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                )
            kept.append(candidate.rstrip())
        return "\n".join(kept).strip()

    # -------------------------------------------------------------------------
    @staticmethod
    def expand_reference_pages(reference: RagDocumentReference) -> list[int]:
        if reference.page_start is None:
            return []
        page_end = reference.page_end or reference.page_start
        if page_end < reference.page_start:
            page_end = reference.page_start
        return list(range(reference.page_start, page_end + 1))

    # -------------------------------------------------------------------------
    @staticmethod
    def format_bibliography_reference(*, file_name: str, pages: list[int]) -> str:
        page_segments: list[str] = []
        start = end = pages[0]
        for page in pages[1:]:
            if page == end + 1:
                end = page
                continue
            page_segments.append(str(start) if start == end else f"{start}-{end}")
            start = end = page
        page_segments.append(str(start) if start == end else f"{start}-{end}")
        page_label = (
            "p." if len(page_segments) == 1 and "-" not in page_segments[0] else "pp."
        )
        return f"- {file_name}, {page_label} {', '.join(page_segments)}"
