from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

from domain.clinical.entities import (
    DrugClinicalAssessment,
    PatientLabTimeline,
    RagDocumentReference,
)
from services.clinical.report_language import (
    evidence_quality_label,
    limitation_label,
    phrase,
    rucam_summary_text,
)
from services.clinical.report_language import report_heading
from services.text.vocabulary import get_text_normalization_snapshot
from services.clinical.hepatox_constants import (
    BIBLIOGRAPHY_LINE_RE,
    DRIFT_SECTION_LINE_RE,
    LIVERTOX_TITLE_LINE_RE,
    NOT_AVAILABLE_TEXT,
    REDUNDANT_REPORT_LINE_RE,
    REPORT_LABEL_LINE_RE,
    STRUCTURED_DILI_SECTION_LINE_RE,
)

###############################################################################
class ReportFinalizer:
    """Builds the final patient report and conclusion from per-drug assessments."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        pass

    # -------------------------------------------------------------------------
    async def _build_and_finalize_report(
        self,
        entries: list[DrugClinicalAssessment],
        *,
        clinical_context: str | None,
        report_language: str,
        generate_conclusion_fn,
    ) -> str | None:
        matched_entries: list[DrugClinicalAssessment] = []
        unresolved_entries: list[DrugClinicalAssessment] = []
        for entry in entries:
            if self.should_render_as_matched_drug(entry):
                matched_entries.append(entry)
                continue
            unresolved_entries.append(entry)

        matched_sections = [
            self.render_matched_drug_section(
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
        unresolved_section = self.render_unresolved_mentions_section(
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
                    self.render_matched_drug_section(
                        assessment, report_language=language
                    ),
                    "",
                ]
            )
        return "\n".join(lines).strip()

    @staticmethod
    def remove_redundant_report_sentence(text: str) -> str:
        if not text:
            return ""
        cleaned_lines: list[str] = []
        for raw_line in text.splitlines():
            if STRUCTURED_DILI_SECTION_LINE_RE.match(raw_line.strip()):
                break
            compact = re.sub(r"[\s*_`#:\-]+", " ", raw_line).strip()
            if compact and REDUNDANT_REPORT_LINE_RE.search(compact):
                continue
            cleaned_lines.append(raw_line)
        cleaned = "\n".join(cleaned_lines).strip()
        return re.sub(r"\n{3,}", "\n\n", cleaned)

    def render_matched_drug_section(
        self, entry: DrugClinicalAssessment, *, report_language: str = "en"
    ) -> str:
        score = self.resolve_livertox_score(entry.matched_livertox_row)
        title = self.format_drug_heading(entry.drug_name, score)
        body = self.sanitize_renderable_body(entry)
        if not body:
            body = self.build_fallback_technical_note(entry, report_language=report_language)
        localized_rucam = (
            rucam_summary_text(entry.rucam, report_language)
            if entry.rucam is not None
            else phrase("rucam_not_calculated", report_language)
        )
        clinical_commentary = self.render_clinical_commentary(
            entry, report_language=report_language
        )
        return (
            f"**{title}**\n\n{clinical_commentary}\n\n"
            f"**RUCAM**: {localized_rucam}\n\n"
            f"**{phrase('report_label', report_language)}**\n\n{body}\n\n"
            f"**{phrase('bibliography_source', report_language)}**: {self.bibliography_source_label()}"
        ).strip()

    @staticmethod
    def render_clinical_commentary(
        entry: DrugClinicalAssessment, *, report_language: str = "en"
    ) -> str:
        quality = evidence_quality_label(
            entry.evidence_quality or phrase("unknown", report_language), report_language
        )
        matched_name = (
            str(entry.matched_livertox_row.get("drug_name") or "").strip()
            if isinstance(entry.matched_livertox_row, dict)
            else ""
        )
        target = matched_name or entry.canonical_name or phrase("not_available", report_language)
        segments = [
            phrase("commentary_evidence_match", report_language, quality=quality, target=target)
        ]
        if entry.evidence_warnings:
            segments.append(
                phrase(
                    "commentary_evidence_warnings",
                    report_language,
                    warnings="; ".join(entry.evidence_warnings[:3]),
                )
            )
        else:
            segments.append(phrase("commentary_no_evidence_warnings", report_language))
        review_claims = [claim for claim in entry.claims if claim.requires_review]
        limitations = list(entry.narrative.limitations if entry.narrative else [])
        if entry.rucam is not None and entry.rucam.total_score is None:
            segments.append(phrase("commentary_rucam_not_assessable", report_language))
        if limitations:
            segments.append(
                phrase(
                    "commentary_limitations",
                    report_language,
                    limitations="; ".join(
                        limitation_label(item, report_language) for item in limitations[:3]
                    ),
                )
            )
        segments.append(
            phrase(
                "commentary_review_required" if review_claims or limitations else "commentary_no_review_required",
                report_language,
            )
        )
        return f"**{phrase('clinical_commentary', report_language)}**: " + " ".join(segments)

    def sanitize_renderable_body(self, entry: DrugClinicalAssessment) -> str:
        text = entry.paragraph.strip() if entry.paragraph else ""
        if not text:
            return ""
        expected_name = (entry.drug_name or "").strip().lower()
        lines: list[str] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                if lines and lines[-1]:
                    lines.append("")
                continue
            compact = re.sub(r"[\s*_`#:\-]+", " ", stripped).strip()
            if REDUNDANT_REPORT_LINE_RE.search(compact) or REPORT_LABEL_LINE_RE.match(stripped):
                continue
            if BIBLIOGRAPHY_LINE_RE.match(stripped) or stripped == "---":
                continue
            if stripped.lower().startswith("## global synthesis") or DRIFT_SECTION_LINE_RE.match(stripped) or STRUCTURED_DILI_SECTION_LINE_RE.match(stripped):
                break
            title_match = LIVERTOX_TITLE_LINE_RE.match(stripped)
            if title_match:
                if expected_name and expected_name not in stripped.lower():
                    continue
                continue
            lines.append(raw_line.rstrip())
        sanitized = re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip()).strip()
        if "local livertox excerpt not available" in re.sub(r"\s+", " ", sanitized).lower():
            return ""
        return sanitized

    def build_fallback_technical_note(self, entry: DrugClinicalAssessment, *, report_language: str = "en") -> str:
        if entry.suspension.excluded:
            return self.build_excluded_paragraph(entry, report_language)
        if entry.ambiguous_match:
            return self.build_ambiguous_match_paragraph(entry, report_language)
        if entry.missing_livertox:
            return phrase("matched_no_excerpt", report_language) if entry.matched_livertox_row else phrase("livertox_missing", report_language)
        return self.build_error_paragraph(entry, report_language)

    def render_unresolved_mentions_section(self, entries: list[DrugClinicalAssessment], *, report_language: str = "en") -> str | None:
        if not entries:
            return None
        lines = [f"## {report_heading('unresolved_mentions', report_language)}", ""]
        for entry in entries:
            label = (entry.drug_name or "").strip() or phrase("unnamed_drug", report_language)
            reason = self.describe_unresolved_entry(entry, report_language)
            rucam = rucam_summary_text(entry.rucam, report_language) if entry.rucam is not None else phrase("rucam_not_calculated", report_language)
            technical = self.build_fallback_technical_note(entry, report_language=report_language)
            context = reason if technical.strip() == reason.strip() else f"{technical} {reason}"
            recommendation = (
                "Il farmaco deve rimanere nella diagnosi differenziale solo in base alla cronologia clinica disponibile; prima di attribuire causalità sono necessari verifica dell'esposizione, andamento dopo sospensione e revisione delle cause alternative."
                if report_language.lower().startswith("it")
                else "The drug should remain in the differential diagnosis only to the extent supported by the available timeline; exposure verification, the course after withdrawal, and competing-cause review are required before causality is assigned."
            )
            lines.extend([f"### {label}", "", f"{context} {rucam}. {recommendation}", ""])
        return "\n".join(lines).strip()

    def describe_unresolved_entry(self, entry: DrugClinicalAssessment, report_language: str = "en") -> str:
        status = (entry.match_status or "").strip().lower()
        if status in {"ambiguous", "ambiguous_match"} or entry.ambiguous_match:
            candidates = ", ".join(entry.match_candidates) if entry.match_candidates else phrase("rucam_insufficient_data", report_language)
            return f"{phrase('livertox_ambiguous', report_language)} {phrase('candidate_matches', report_language, candidates=candidates)} {phrase('manual_curation', report_language)}"
        if status in {"missing", "missing_match"}:
            return phrase("no_matching_record", report_language)
        if status == "matched_no_excerpt":
            return phrase("matched_no_excerpt", report_language)
        if entry.missing_livertox:
            return phrase("matched_no_excerpt", report_language) if entry.matched_livertox_row else phrase("livertox_missing", report_language)
        return phrase("deterministic_section_unavailable", report_language)

    def build_excluded_paragraph(self, entry: DrugClinicalAssessment, report_language: str = "en") -> str:
        suspension = entry.suspension
        if report_language.startswith("it"):
            detail = f"La terapia è stata sospesa il {suspension.suspension_date.isoformat()} molto prima della visita; questa esposizione è stata quindi esclusa dalla valutazione attiva di causalità DILI." if suspension.suspension_date is not None else "La terapia risulta sospesa molto prima della visita ed è stata esclusa dalla valutazione attiva di causalità DILI."
            return f"{detail} È consigliata una verifica manuale della latenza se l'esposizione torna clinicamente rilevante."
        detail = f"The therapy was suspended on {suspension.suspension_date.isoformat()} well before the visit, so this exposure was excluded from active DILI causality assessment." if suspension.suspension_date is not None else "The therapy was reported as suspended well before the visit and was excluded from active DILI causality assessment."
        return f"{detail} Manual latency verification is suggested if the exposure history becomes clinically relevant again."

    @staticmethod
    def build_missing_excerpt_paragraph(entry: DrugClinicalAssessment, report_language: str = "en") -> str:
        _ = entry
        return phrase("livertox_missing", report_language)

    def build_ambiguous_match_paragraph(self, entry: DrugClinicalAssessment, report_language: str = "en") -> str:
        candidates = ", ".join(entry.match_candidates) if entry.match_candidates else phrase("rucam_insufficient_data", report_language)
        return f"{phrase('livertox_ambiguous', report_language)} {phrase('candidate_matches', report_language, candidates=candidates)} {phrase('manual_curation', report_language)}"

    @staticmethod
    def build_error_paragraph(entry: DrugClinicalAssessment, report_language: str = "en") -> str:
        _ = entry
        return phrase("rucam_insufficient_data", report_language)

    @staticmethod
    def resolve_livertox_score(metadata: dict[str, Any] | None) -> str:
        if not metadata:
            return NOT_AVAILABLE_TEXT
        score = metadata.get("likelihood_score")
        if score is None:
            return NOT_AVAILABLE_TEXT
        text = str(score).strip()
        if not text or text.lower() == "nan":
            return NOT_AVAILABLE_TEXT
        return text.upper() if text.isalpha() else text

    @staticmethod
    def format_drug_heading(drug_name: str, score: str) -> str:
        normalized_name = drug_name.strip() if drug_name else ""
        normalized_score = score.strip() if score else ""
        return f"{normalized_name or 'Unnamed drug'} - LiverTox score {normalized_score or NOT_AVAILABLE_TEXT}"

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
    async def finalize_patient_report(
        self,
        entries: list[DrugClinicalAssessment],
        *,
        clinical_context: str | None,
        report_language: str,
        generate_conclusion: Callable[..., Awaitable[str | None]],
    ) -> str | None:
        return await self._build_and_finalize_report(
            entries,
            clinical_context=clinical_context,
            report_language=report_language,
            generate_conclusion_fn=generate_conclusion,
        )

    # -------------------------------------------------------------------------
    async def finalize_revision_patient_report(
        self,
        entries: list[DrugClinicalAssessment],
        *,
        clinical_context: str | None,
        report_language: str,
        generate_conclusion: Callable[..., Awaitable[str | None]],
    ) -> str | None:
        return await self._build_and_finalize_report(
            entries,
            clinical_context=clinical_context,
            report_language=report_language,
            generate_conclusion_fn=generate_conclusion,
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
