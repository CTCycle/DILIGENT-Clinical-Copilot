from __future__ import annotations

import asyncio
import inspect
import json
import re
from datetime import date, datetime
from typing import Any

from DILIGENT.app.api.models.prompts import (
    CLINICAL_REPORT_REWRITE_SYSTEM_PROMPT,
    CLINICAL_REPORT_REWRITE_USER_PROMPT,
    LIVERTOX_CLINICAL_SYSTEM_PROMPT,
    LIVERTOX_CLINICAL_USER_PROMPT,
)
from DILIGENT.app.api.models.providers import initialize_llm_client
from DILIGENT.app.api.schemas.clinical import (
    DrugEntry,
    DrugClinicalAssessment,
    DrugSuspensionContext,
    HepatotoxicityPatternScore,
    PatientData,
    PatientDrugClinicalReport,
    PatientDrugs,
)
from DILIGENT.app.configurations import ClientRuntimeConfig
from DILIGENT.app.constants import DEFAULT_LLM_TIMEOUT_SECONDS, MAX_EXCERPT_LENGTH
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.serializer import DataSerializer
from DILIGENT.app.utils.services.essay import LiverToxMatch, LiverToxMatcher


###############################################################################
class HepatotoxicityPatternAnalyzer:

    def __init__(self) -> None:
        pass
       
    # -------------------------------------------------------------------------
    def calculate_hepatotoxicity_pattern(self, payload: PatientData) -> HepatotoxicityPatternScore:
        alt_value = self.parse_marker_value(payload.alt)
        alt_max_value = self.parse_marker_value(payload.alt_max)
        alp_value = self.parse_marker_value(payload.alp)
        alp_max_value = self.parse_marker_value(payload.alp_max)

        alt_multiple = self.safe_ratio(alt_value, alt_max_value)
        alp_multiple = self.safe_ratio(alp_value, alp_max_value)

        r_score: float | None = None
        if alt_multiple is not None and alp_multiple not in (None, 0.0):
            r_score = alt_multiple / alp_multiple

        classification = "indeterminate"
        if r_score is not None:
            if r_score > 5:
                classification = "hepatocellular"
            elif r_score < 2:
                classification = "cholestatic"
            else:
                classification = "mixed"

        return HepatotoxicityPatternScore(
            alt_multiple=alt_multiple,
            alp_multiple=alp_multiple,
            r_score=r_score,
            classification=classification,
        )

    # -------------------------------------------------------------------------
    def parse_marker_value(self, raw: str | None) -> float | None:
        if raw is None:
            return None
        normalized = raw.replace(",", ".")
        match = re.search(r"[-+]?\d*\.?\d+", normalized)
        if not match:
            return None
        try:
            return float(match.group())
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    def safe_ratio(self, value: float | None, reference: float | None) -> float | None:
        if value is None or reference is None:
            return None
        if reference == 0:
            return None
        return value / reference

    # -------------------------------------------------------------------------
    def stringify_scores(
        self, pattern_score: HepatotoxicityPatternScore | None
        ) -> dict[str, str]:
            if not pattern_score:
                return {}

            mapping = {
                "alt_multiple": (pattern_score.alt_multiple, "{:.2f}x ULN"),
                "alp_multiple": (pattern_score.alp_multiple, "{:.2f}x ULN"),
                "r_score": (pattern_score.r_score, "{:.2f}"),
            }

            return {
                key: fmt.format(val) if val is not None else "Not available"
                for key, (val, fmt) in mapping.items()
            }


###############################################################################
class HepatoxConsultation:    
    
    def __init__(
        self,
        drugs: PatientDrugs,
        *,
        patient_name: str | None = None,
        timeout_s: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.drugs = drugs
        self.timeout_s = timeout_s
        self.serializer = DataSerializer()
        self.livertox_df = None
        self.master_list_df = None
        self.matcher: LiverToxMatcher | None = None
        self.llm_client = initialize_llm_client(purpose="clinical", timeout_s=timeout_s)
        self.MAX_EXCERPT_LENGTH = MAX_EXCERPT_LENGTH
        self.patient_name = (patient_name or "").strip() or None
        provider, model_candidate = ClientRuntimeConfig.resolve_provider_and_model(
            "clinical"
        )
        self.llm_model = model_candidate or ClientRuntimeConfig.get_clinical_model()
        try:
            chat_signature = inspect.signature(self.llm_client.chat)
        except (TypeError, ValueError):
            chat_signature = None
        self.chat_supports_temperature = (
            chat_signature is not None and "temperature" in chat_signature.parameters
        )
        self.temperature = 0.2
        self.report_temperature = 0.1

    # -------------------------------------------------------------------------
    async def run_analysis(
        self,
        *,
        anamnesis: str | None = None,
        exams: str | None = None,
        visit_date: date | None = None,
        pattern_score: HepatotoxicityPatternScore | None = None,
    ) -> dict[str, Any] | None:
        logger.info("Toxicity analysis stage 1/3: validating inputs")
        patient_drugs = self.collect_patient_drugs()
        if not patient_drugs:
            logger.info("No drugs detected for toxicity analysis")
            return None
        if not self.ensure_livertox_loaded():
            return None

        if self.matcher is None:
            return PatientDrugClinicalReport(entries=[], final_report=None).model_dump()        

        logger.info("Toxicity analysis: performing clinical assessment for matched drugs")
        matches = await self.matcher.match_drug_names(patient_drugs)
        resolved = self.resolve_matches(patient_drugs, matches)
        report = await self.compile_clinical_assessment(
            resolved,
            anamnesis=anamnesis,
            exams=exams,
            visit_date=visit_date,
            pattern_score=pattern_score,
        )
        return report.model_dump()

    # -------------------------------------------------------------------------
    def ensure_livertox_loaded(self) -> bool:
        if self.matcher is not None:
            return True
        try:
            dataset = self.serializer.get_livertox_records()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed loading LiverTox monographs from database: %s", exc)
            self.matcher = None
            return False
        if dataset is None or dataset.empty:
            logger.warning(
                "LiverTox monograph table is empty; toxicity essay cannot run"
            )
            self.matcher = None
            return False
        self.livertox_df = dataset
        self.master_list_df = None
        self.matcher = LiverToxMatcher(dataset)
        return True

    # -------------------------------------------------------------------------
    def collect_patient_drugs(self) -> list[str]:
        return [entry.name for entry in self.drugs.entries if entry.name]

    # -------------------------------------------------------------------------
    def resolve_matches(
        self,
        patient_drugs: list[str],
        matches: list[LiverToxMatch | None],
    ) -> list[dict[str, Any]]:
        if self.matcher is None:
            return []
        return self.matcher.build_patient_mapping(patient_drugs, matches)

    # -------------------------------------------------------------------------
    async def compile_clinical_assessment(
        self,
        resolved_entries: list[dict[str, Any]],
        *,
        anamnesis: str | None,
        exams: str | None,
        visit_date: date | None,
        pattern_score: HepatotoxicityPatternScore | None,
    ) -> PatientDrugClinicalReport:
        normalized_anamnesis = (anamnesis or "").strip()
        if not normalized_anamnesis:
            normalized_anamnesis = "No anamnesis information was provided."
        normalized_exams = (exams or "").strip()
        if not normalized_exams:
            normalized_exams = "No exam results were provided."
        pattern_prompt = self.format_pattern_prompt(pattern_score)

        entries: list[DrugClinicalAssessment] = []
        llm_jobs: list[tuple[int, Any]] = []

        for idx, drug_entry in enumerate(self.drugs.entries):
            resolved = resolved_entries[idx] if idx < len(resolved_entries) else {}
            if not isinstance(resolved, dict):
                resolved = {}

            matched_row = resolved.get("matched_livertox_row")
            if not isinstance(matched_row, dict):
                matched_row = None
            raw_excerpts = resolved.get("extracted_excerpts")
            if isinstance(raw_excerpts, str):
                excerpts_list = [raw_excerpts]
            elif isinstance(raw_excerpts, list):
                excerpts_list = [item for item in raw_excerpts if isinstance(item, str)]
            else:
                excerpts_list = []
            
            suspension = self.evaluate_suspension(drug_entry, visit_date)
            entry = DrugClinicalAssessment(
                drug_name=drug_entry.name,
                matched_livertox_row=matched_row if isinstance(matched_row, dict) else None,
                extracted_excerpts=excerpts_list,
                suspension=suspension,
            )
            entries.append(entry)

            if suspension.excluded:
                entry.paragraph = self.build_excluded_paragraph(drug_entry.name, suspension)
                continue

            excerpt = self.select_excerpt(excerpts_list)
            if excerpt is None:
                entry.paragraph = self.build_missing_excerpt_paragraph(drug_entry.name)
                continue

            llm_jobs.append(
                (
                    idx,
                    self.request_drug_analysis(
                        drug_name=drug_entry.name,
                        excerpt=excerpt,
                        anamnesis=normalized_anamnesis,
                        exams=normalized_exams,
                        suspension=suspension,
                        pattern_summary=pattern_prompt,
                    ),
                )
            )

        if llm_jobs:
            indices, tasks = zip(*llm_jobs)
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, outcome in zip(indices, responses):
                entry = entries[idx]
                if isinstance(outcome, Exception):
                    logger.error(
                        "Clinical analysis for drug '%s' failed: %s",
                        entry.drug_name,
                        outcome,
                    )
                    entry.paragraph = self.build_error_paragraph(entry.drug_name)
                else:
                    entry.paragraph = outcome

        logger.info("Composing final clinical report for current patient")
        final_report = self.compose_final_report(entries)
        refined_report = await self.rewrite_patient_report(
            entries,
            final_report=final_report,
            anamnesis=normalized_anamnesis,
            visit_date=visit_date,
            pattern_score=pattern_score,
        )
        if refined_report:
            final_report = refined_report
            
        return PatientDrugClinicalReport(entries=entries, final_report=final_report)

    # -------------------------------------------------------------------------
    def select_excerpt(self, excerpts: list[str]) -> str | None:
        excerpts = [chunk.strip() for chunk in excerpts if chunk.strip()]
        if not excerpts:
            return None
        combined = "\n\n".join(excerpts)
        if len(combined) <= self.MAX_EXCERPT_LENGTH:
            return combined
        truncated = combined[: self.MAX_EXCERPT_LENGTH]
        cutoff = truncated.rfind("\n")
        if cutoff > 2000:
            truncated = truncated[:cutoff]
        return truncated.strip()

    # -------------------------------------------------------------------------
    def evaluate_suspension(
        self, entry: DrugEntry, visit_date: date | None
    ) -> DrugSuspensionContext:
        start_reported = bool(entry.therapy_start_status) or bool(
            entry.therapy_start_date
        )
        start_date = self.parse_start_date(entry.therapy_start_date, visit_date)
        start_interval_days: int | None = None
        if start_reported and start_date is not None and visit_date is not None:
            start_interval_days = (visit_date - start_date).days
        start_note = self.format_start_note(
            start_reported=start_reported,
            start_date=start_date,
            start_interval_days=start_interval_days,
            visit_date=visit_date,
        )

        suspended = bool(entry.suspension_status)
        parsed_date = self.parse_suspension_date(entry.suspension_date, visit_date)
        interval_days: int | None = None
        if not suspended:
            combined_note = " ".join(
                part
                for part in (
                    start_note,
                    "Active therapy; no suspension reported.",
                )
                if part
            )
            return DrugSuspensionContext(
                suspended=False,
                suspension_date=None,
                excluded=False,
                note=combined_note or None,
                interval_days=None,
                start_reported=start_reported,
                start_date=start_date,
                start_interval_days=start_interval_days,
                start_note=start_note,
            )

        if parsed_date is None:
            suspension_note = (
                "Suspension reported without a reliable date; drug kept in analysis."
            )
        elif visit_date is None:
            suspension_note = (
                f"Suspended on {parsed_date.isoformat()}, but visit date missing; drug kept in analysis."
            )
        else:
            interval_days = (visit_date - parsed_date).days
            if interval_days < 0:
                suspension_note = (
                    f"Suspended on {parsed_date.isoformat()} ({abs(interval_days)} days after the visit); treat as ongoing exposure."
                )
            elif interval_days == 0:
                suspension_note = (
                    f"Suspended on {parsed_date.isoformat()} (same day as the visit); residual exposure is expected."
                )
            else:
                suspension_note = (
                    f"Suspended on {parsed_date.isoformat()} ({interval_days} days before the visit); compare this latency with LiverTox guidance."
                )

        combined_note = " ".join(part for part in (start_note, suspension_note) if part)
        return DrugSuspensionContext(
            suspended=suspended,
            suspension_date=parsed_date,
            excluded=False,
            note=combined_note or None,
            interval_days=interval_days,
            start_reported=start_reported,
            start_date=start_date,
            start_interval_days=start_interval_days,
            start_note=start_note,
        )

    # -------------------------------------------------------------------------
    def parse_timeline_date(
        self, raw_date: str | None, visit_date: date | None
    ) -> date | None:
        if raw_date is None:
            return None
        text = str(raw_date).strip()
        if not text:
            return None
        normalized = text.replace("/", "-").replace(".", "-").replace(",", "-")
        tokens = [token for token in normalized.split("-") if token]
        candidates: list[str] = []
        if visit_date is not None and len(tokens) == 2:
            day, month = tokens
            candidates.extend(
                [
                    f"{day.zfill(2)}-{month.zfill(2)}-{visit_date.year}",
                    f"{month.zfill(2)}-{day.zfill(2)}-{visit_date.year}",
                    f"{visit_date.year}-{month.zfill(2)}-{day.zfill(2)}",
                ]
            )
        candidates.append("-".join(tokens))
        candidates.append(text)
        candidates.append(normalized)
        checked: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in checked:
                continue
            checked.add(candidate)
            parsed = self.try_parse_date(candidate)
            if parsed is not None:
                return parsed
        return None

    # -------------------------------------------------------------------------
    def parse_suspension_date(
        self, raw_date: str | None, visit_date: date | None
    ) -> date | None:
        return self.parse_timeline_date(raw_date, visit_date)

    # -------------------------------------------------------------------------
    def parse_start_date(
        self, raw_date: str | None, visit_date: date | None
    ) -> date | None:
        return self.parse_timeline_date(raw_date, visit_date)

    # -------------------------------------------------------------------------
    def format_start_note(
        self,
        *,
        start_reported: bool,
        start_date: date | None,
        start_interval_days: int | None,
        visit_date: date | None,
    ) -> str:
        if not start_reported:
            return (
                "Therapy start was not documented; assume chronic exposure unless another source clarifies the onset."
            )
        if start_date is None:
            return (
                "Therapy start was reported but no reliable date could be parsed from the notes."
            )
        if visit_date is None or start_interval_days is None:
            return (
                f"Therapy started on {start_date.isoformat()}, but the visit date was unavailable for latency comparisons."
            )
        if start_interval_days < 0:
            humanized = self.humanize_interval(abs(start_interval_days))
            return (
                f"Therapy was documented to start on {start_date.isoformat()}, {humanized} after the visit; verify this discrepancy manually."
            )
        if start_interval_days == 0:
            return (
                f"Therapy started on {start_date.isoformat()}, coinciding with the clinical visit."
            )
        humanized = self.humanize_interval(start_interval_days)
        return (
            f"Therapy started on {start_date.isoformat()}, roughly {humanized} before the visit."
        )

    # -------------------------------------------------------------------------
    def humanize_interval(self, days: int) -> str:
        if days <= 1:
            return "1 day"
        if days < 14:
            return f"{days} days"
        weeks = days / 7
        if days < 60:
            rounded_weeks = round(weeks, 1)
            return f"{rounded_weeks:g} weeks"
        months = days / 30.4375
        if days < 365:
            rounded_months = round(months, 1)
            return f"{rounded_months:g} months"
        years = days / 365.25
        rounded_years = round(years, 1)
        return f"{rounded_years:g} years"

    # -------------------------------------------------------------------------
    @staticmethod
    def try_parse_date(value: str) -> date | None:
        cleaned = value.strip()
        if not cleaned:
            return None
        iso_candidate = cleaned.replace(".", "-").replace("/", "-")
        try:
            return date.fromisoformat(iso_candidate)
        except ValueError:
            pass
        for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d", "%d.%m.%Y", "%Y.%m.%d"):
            try:
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
        return None

    # -------------------------------------------------------------------------
    def format_suspension_prompt(self, suspension: DrugSuspensionContext) -> str:
        segments: list[str] = []
        if not suspension.suspended:
            segments.append("Active therapy; no suspension reported.")
        elif suspension.suspension_date is None:
            segments.append(
                "Reported as suspended without a reliable date; evaluate latency with the LiverTox excerpt."
            )
        elif suspension.interval_days is None:
            segments.append(
                f"Suspended on {suspension.suspension_date.isoformat()}, but the interval relative to the visit is unclear; rely on LiverTox latency guidance."
            )
        elif suspension.interval_days < 0:
            days = abs(suspension.interval_days)
            segments.append(
                f"Suspended on {suspension.suspension_date.isoformat()} ({days} days after the visit); treat as ongoing exposure."
            )
        elif suspension.interval_days == 0:
            segments.append(
                f"Suspended on {suspension.suspension_date.isoformat()} (same day as the visit); residual exposure is expected."
            )
        else:
            segments.append(
                f"Suspended on {suspension.suspension_date.isoformat()} ({suspension.interval_days} days before the visit); compare with LiverTox latency guidance."
            )

        return " ".join(segment for segment in segments if segment)

    # -------------------------------------------------------------------------
    def format_start_prompt(self, suspension: DrugSuspensionContext) -> str:
        if suspension.start_note:
            return suspension.start_note
        if suspension.start_reported:
            return "Therapy start was reported, but no reliable date was available."
        return (
            "No therapy start information was detected; treat the exposure window as chronic unless contradicted."
        )

    # -------------------------------------------------------------------------
    def format_pattern_prompt(
        self, pattern_score: HepatotoxicityPatternScore | None
    ) -> str:
        if pattern_score is None:
            return (
                "Hepatotoxicity pattern classification was unavailable; weigh pattern matches qualitatively."
            )
        classification = pattern_score.classification.replace("_", " ")
        segments: list[str] = [
            f"Observed liver injury pattern: {classification.capitalize()}.",
        ]
        if pattern_score.r_score is not None:
            segments.append(f"R ratio ≈ {pattern_score.r_score:.2f}.")
        if pattern_score.alt_multiple is not None:
            segments.append(
                f"ALT is about {pattern_score.alt_multiple:.2f} × the upper reference limit."
            )
        if pattern_score.alp_multiple is not None:
            segments.append(
                f"ALP is about {pattern_score.alp_multiple:.2f} × the upper reference limit."
            )
        segments.append(
            "Treat drugs whose known hepatotoxicity pattern matches this classification as stronger causal candidates, and downgrade mismatches."
        )
        return " ".join(segments)

    # -------------------------------------------------------------------------
    async def request_drug_analysis(
        self,
        *,
        drug_name: str,
        excerpt: str,
        anamnesis: str,
        exams: str,
        suspension: DrugSuspensionContext,
        pattern_summary: str,
    ) -> str:
        start_details = self.format_start_prompt(suspension)
        suspension_details = self.format_suspension_prompt(suspension)
        user_prompt = LIVERTOX_CLINICAL_USER_PROMPT.format(
            drug_name=self.escape_braces(drug_name.strip() or drug_name),
            excerpt=self.escape_braces(excerpt),
            anamnesis=self.escape_braces(anamnesis),
            exams=self.escape_braces(exams),
            therapy_start_details=self.escape_braces(start_details),
            suspension_details=self.escape_braces(suspension_details),
            pattern_summary=self.escape_braces(pattern_summary),
        )
        messages = [
            {"role": "system", "content": LIVERTOX_CLINICAL_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt},
        ]
        chat_kwargs: dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
        }
        if self.chat_supports_temperature:
            chat_kwargs["temperature"] = self.temperature
        else:
            chat_kwargs["options"] = {"temperature": self.temperature}
        try:
            raw_response = await self.llm_client.chat(**chat_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LLM analysis failed for {drug_name}: {exc}") from exc
        return self.coerce_chat_text(raw_response)

    # -------------------------------------------------------------------------
    @staticmethod
    def escape_braces(value: str) -> str:
        return value.replace("{", "{{").replace("}", "}}")

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

    # -------------------------------------------------------------------------
    def compose_final_report(
        self, entries: list[DrugClinicalAssessment]
    ) -> str | None:
        if not entries:
            return None
        blocks: list[str] = []
        for entry in entries:
            name = entry.drug_name.strip() or entry.drug_name
            summary = (entry.paragraph or "No analysis available.").strip()
            blocks.append(f"{name}\nConclusions: {summary}")
        report = "\n\n".join(blocks).strip()
        return report or None

    # -------------------------------------------------------------------------
    def classify_entry_status(self, entry: DrugClinicalAssessment) -> str:
        if entry.suspension.excluded:
            return "excluded_from_analysis"
        if not entry.extracted_excerpts:
            return "not_evaluated_missing_livertox"
        if entry.paragraph is None:
            return "analysis_unavailable"
        normalized = entry.paragraph.lower()
        if "manual review is recommended" in normalized:
            return "analysis_failed"
        return "assessed"

    # -------------------------------------------------------------------------
    def summarize_entries_for_prompt(
        self, entries: list[DrugClinicalAssessment]
    ) -> str:
        blocks: list[str] = []
        for entry in entries:
            status = self.classify_entry_status(entry)
            lines: list[str] = [f"- Drug: {entry.drug_name}", f"  status: {status}"]
            suspension_note = entry.suspension.note or "No suspension details available."
            lines.append(f"  therapy_timing: {suspension_note}")
            if entry.suspension.interval_days is not None:
                lines.append(
                    f"  suspension_interval_days: {entry.suspension.interval_days}"
                )
            if entry.suspension.start_interval_days is not None:
                lines.append(
                    f"  start_interval_days: {entry.suspension.start_interval_days}"
                )
            summary = entry.paragraph or "No analysis was produced for this drug."
            lines.append(f"  assessment: {summary}")
            blocks.append("\n".join(lines))
        return "\n".join(blocks)

    # -------------------------------------------------------------------------
    async def rewrite_patient_report(
        self,
        entries: list[DrugClinicalAssessment],
        *,
        final_report: str | None,
        anamnesis: str,
        visit_date: date | None,
        pattern_score: HepatotoxicityPatternScore | None,
    ) -> str | None:
        if not entries or not final_report:
            return None
        summary_payload = self.summarize_entries_for_prompt(entries)
        if not summary_payload.strip():
            return None
        patient_label = self.patient_name or "Unnamed patient"
        visit_label = visit_date.isoformat() if visit_date else "Unknown visit date"
        anamnesis_payload = anamnesis.strip() or "No anamnesis information provided."
        pattern_summary_raw = self.format_pattern_prompt(pattern_score)
        instruction_clause = (
            "Treat drugs whose known hepatotoxicity pattern matches this classification as stronger causal candidates, and downgrade mismatches."
        )
        pattern_summary = pattern_summary_raw.replace(instruction_clause, "").strip()
        if not pattern_summary:
            pattern_summary = pattern_summary_raw
        user_prompt = CLINICAL_REPORT_REWRITE_USER_PROMPT.format(
            patient_name=self.escape_braces(patient_label),
            visit_date=self.escape_braces(visit_label),
            pattern_summary=self.escape_braces(pattern_summary),
            anamnesis=self.escape_braces(anamnesis_payload),
            drug_summaries=self.escape_braces(summary_payload),
            initial_report=self.escape_braces(final_report),
        )
        messages = [
            {
                "role": "system",
                "content": CLINICAL_REPORT_REWRITE_SYSTEM_PROMPT.strip(),
            },
            {"role": "user", "content": user_prompt},
        ]
        chat_kwargs: dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
        }
        if self.chat_supports_temperature:
            chat_kwargs["temperature"] = self.report_temperature
        else:
            chat_kwargs["options"] = {"temperature": self.report_temperature}
        try:
            raw_response = await self.llm_client.chat(**chat_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error("Holistic report rewrite failed: %s", exc)
            return None
        refined = self.coerce_chat_text(raw_response)
        return refined or None

    # -------------------------------------------------------------------------
    def build_excluded_paragraph(
        self, drug_name: str, suspension: DrugSuspensionContext
    ) -> str:
        if suspension.suspension_date is not None:
            return (
                f"The therapy was suspended on {suspension.suspension_date.isoformat()} well before the visit; the drug was excluded from this DILI assessment."
            )
        return (
            "The therapy was reported as suspended well before the visit and was excluded from the current DILI assessment."
        )

    # -------------------------------------------------------------------------
    def build_missing_excerpt_paragraph(self, drug_name: str) -> str:
        return (
            "No LiverTox excerpt was available for this drug, so hepatotoxic involvement could not be evaluated."
        )

    # -------------------------------------------------------------------------
    def build_error_paragraph(self, drug_name: str) -> str:
        return (
            "Automated analysis was unavailable due to a technical issue; manual review is recommended."
        )

