from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from DILIGENT.server.models.prompts import (
    LIVERTOX_CONCLUSION_SYSTEM_PROMPT,
    LIVERTOX_CONCLUSION_USER_PROMPT,
    LIVERTOX_CLINICAL_SYSTEM_PROMPT,
    LIVERTOX_CLINICAL_USER_PROMPT,
    LIVERTOX_REPORT_EXAMPLE,
)
from DILIGENT.server.models.providers import initialize_llm_client
from DILIGENT.server.entities.clinical import (
    ClinicalPipelineValidationError,
    DrugEntry,
    DrugClinicalAssessment,
    DrugSuspensionContext,
    HepatotoxicityPatternAssessment,
    HepatotoxicityPatternScore,
    PatientData,
    PatientDrugClinicalReport,
    PatientDrugs,
    PipelineIssue,
)
from DILIGENT.server.configurations import LLMRuntimeConfig, server_settings
from DILIGENT.common.constants import (
    DEFAULT_DILI_CLASSIFICATION,
    R_SCORE_CHOLESTATIC_THRESHOLD,
    R_SCORE_HEPATOCELLULAR_THRESHOLD,
)
from DILIGENT.common.utils.logger import logger
from DILIGENT.server.services.retrieval.embeddings import SimilaritySearch
from DILIGENT.server.services.clinical.preparation import HepatoxPreparedInputs
from DILIGENT.server.services.text.normalization import normalize_drug_query_name
from DILIGENT.server.services.research.tavily import tavily_research_service


###############################################################################
NOT_AVAILABLE_TEXT = "Not available"


###############################################################################
class HepatotoxicityPatternCalculator:
    # -------------------------------------------------------------------------
    def calculate(
        self,
        *,
        alt_value: float | None,
        alt_uln: float | None,
        alp_value: float | None,
        alp_uln: float | None,
    ) -> HepatotoxicityPatternScore:
        alt_multiple = self.safe_ratio(alt_value, alt_uln)
        alp_multiple = self.safe_ratio(alp_value, alp_uln)

        r_score = None
        if alt_multiple is not None and alp_multiple not in (None, 0.0):
            r_score = alt_multiple / alp_multiple

        classification = DEFAULT_DILI_CLASSIFICATION
        if r_score is not None:
            if r_score > R_SCORE_HEPATOCELLULAR_THRESHOLD:
                classification = "hepatocellular"
            elif r_score < R_SCORE_CHOLESTATIC_THRESHOLD:
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
    @staticmethod
    def safe_ratio(value: float | None, reference: float | None) -> float | None:
        if value is None or reference is None:
            return None
        if reference == 0:
            return None
        return value / reference


###############################################################################
class HepatotoxicityPatternAnalyzer:

    def __init__(self) -> None:
        self.r_score: float | None = None
        self.calculator = HepatotoxicityPatternCalculator()

    # -------------------------------------------------------------------------
    def calculate_hepatotoxicity_pattern(
        self, payload: PatientData
    ) -> HepatotoxicityPatternScore:
        alt_value = self.parse_marker_value(payload.alt)
        alt_max_value = self.parse_marker_value(payload.alt_max)
        alp_value = self.parse_marker_value(payload.alp)
        alp_max_value = self.parse_marker_value(payload.alp_max)

        score = self.calculator.calculate(
            alt_value=alt_value,
            alt_uln=alt_max_value,
            alp_value=alp_value,
            alp_uln=alp_max_value,
        )
        self.r_score = score.r_score
        return score

    # -------------------------------------------------------------------------
    def assess_payload(
        self,
        payload: PatientData,
        *,
        allow_missing_labs: bool = False,
    ) -> HepatotoxicityPatternAssessment:
        field_mapping = [
            ("alt", "ALT"),
            ("alt_max", "ALT upper limit"),
            ("alp", "ALP"),
            ("alp_max", "ALP upper limit"),
        ]
        parsed_values: dict[str, float] = {}
        issues: list[PipelineIssue] = []
        for field_name, label in field_mapping:
            raw_value = getattr(payload, field_name, None)
            parsed = self.parse_marker_value(raw_value)
            if parsed is None:
                issues.append(
                    PipelineIssue(
                        severity="warning",
                        code="missing_labs",
                        message=(
                            f"{label} is missing or invalid. ALT, ALT max, ALP, and ALP max "
                            "are required for a determined hepatotoxicity pattern."
                        ),
                        field=field_name,
                    )
                )
                continue
            parsed_values[field_name] = parsed

        if issues:
            if not allow_missing_labs:
                raise ClinicalPipelineValidationError(
                    issues=issues,
                    message="Missing laboratory values required for hepatotoxicity pattern assessment.",
                )
            indeterminate = HepatotoxicityPatternScore(
                alt_multiple=None,
                alp_multiple=None,
                r_score=None,
                classification=DEFAULT_DILI_CLASSIFICATION,
            )
            self.r_score = indeterminate.r_score
            return HepatotoxicityPatternAssessment(
                score=indeterminate,
                status="undetermined_due_to_missing_labs",
                issues=issues,
            )

        score = self.calculator.calculate(
            alt_value=parsed_values["alt"],
            alt_uln=parsed_values["alt_max"],
            alp_value=parsed_values["alp"],
            alp_uln=parsed_values["alp_max"],
        )
        self.r_score = score.r_score
        return HepatotoxicityPatternAssessment(
            score=score,
            status="ok",
            issues=[],
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
        return self.calculator.safe_ratio(value, reference)

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
            key: fmt.format(val) if val is not None else NOT_AVAILABLE_TEXT
            for key, (val, fmt) in mapping.items()
        }


###############################################################################
class HepatoxConsultation:
    def __init__(
        self,
        drugs: PatientDrugs,
        *,
        patient_name: str | None = None,
        timeout_s: float = server_settings.external_data.clinical_llm_timeout,
    ) -> None:
        self.drugs = drugs
        self.timeout_s = timeout_s
        self.llm_client = initialize_llm_client(purpose="clinical", timeout_s=timeout_s)
        self.MAX_EXCERPT_LENGTH = server_settings.external_data.max_excerpt_length
        self.patient_name = (patient_name or "").strip() or None
        _, model_candidate = LLMRuntimeConfig.resolve_provider_and_model("clinical")
        self.llm_model = model_candidate or LLMRuntimeConfig.get_clinical_model()
        try:
            chat_signature = inspect.signature(self.llm_client.chat)
        except (TypeError, ValueError):
            chat_signature = None
        self.chat_supports_temperature = (
            chat_signature is not None and "temperature" in chat_signature.parameters
        )
        self.temperature = LLMRuntimeConfig.get_ollama_temperature()
        self.similarity_search: SimilaritySearch | None = None
        self.rag_top_k = server_settings.rag.top_k_documents
        self.max_parallel_analyses = max(
            1,
            int(
                getattr(
                    server_settings.external_data,
                    "clinical_llm_max_concurrency",
                    3,
                )
            ),
        )
        self.analysis_retry_attempts = max(
            1,
            int(
                getattr(
                    server_settings.external_data,
                    "clinical_llm_retry_attempts",
                    2,
                )
            ),
        )

    # -------------------------------------------------------------------------
    async def run_analysis(
        self,
        *,
        prepared_inputs: HepatoxPreparedInputs | None,
        visit_date: date | None = None,
        rag_query: dict[str, str] | None = None,
        use_web_search: bool = False,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any] | None:
        if prepared_inputs is None:
            logger.info("No prepared inputs provided; skipping hepatotoxicity consultation")
            return None

        resolved_mapping = prepared_inputs.resolved_drugs
        if not resolved_mapping:
            logger.info("No matched drugs available for hepatotoxicity consultation")
            return None

        logger.info("Running clinical hepatotoxicity assessment for matched drugs")
        report = await self.compile_clinical_assessment(
            resolved_mapping,
            clinical_context=prepared_inputs.clinical_context,
            visit_date=visit_date,
            pattern_prompt=prepared_inputs.pattern_prompt,
            rag_query=rag_query,
            use_web_search=use_web_search,
            progress_callback=progress_callback,
        )
        return report.model_dump()

    # -------------------------------------------------------------------------
    async def compile_clinical_assessment(
        self,
        resolved_drugs: dict[str, dict[str, Any]],
        *,
        clinical_context: str | None,
        visit_date: date | None,
        pattern_prompt: str,
        rag_query: dict[str, str] | None = None,
        use_web_search: bool = False,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> PatientDrugClinicalReport:
        normalized_context = clinical_context.strip() if clinical_context else ""
        pattern_summary = (
            pattern_prompt.strip()
            or "Hepatotoxicity pattern classification was unavailable; weigh pattern matches qualitatively."
        )
        entries: list[DrugClinicalAssessment] = []
        llm_jobs: list[tuple[int, Any]] = []

        # iterate over all drugs to identify those with LiverTox excerpts and those without
        for idx, drug_entry in enumerate(self.drugs.entries):
            entry, job = await self.prepare_drug_assessment(
                idx=idx,
                drug_entry=drug_entry,
                resolved_drugs=resolved_drugs,
                visit_date=visit_date,
                normalized_context=normalized_context,
                pattern_summary=pattern_summary,
                rag_query=rag_query,
                use_web_search=use_web_search,
            )
            entries.append(entry)
            if job:
                llm_jobs.append(job)

        self.emit_progress(progress_callback, stage="llm_analysis", fraction=0.0)
        if llm_jobs:
            semaphore = asyncio.Semaphore(self.max_parallel_analyses)
            pending_tasks = [
                asyncio.create_task(self.execute_bounded_job(idx, task, semaphore))
                for idx, task in llm_jobs
            ]
            completed = 0
            total = len(pending_tasks)
            for task in asyncio.as_completed(pending_tasks):
                idx, outcome = await task
                entry = entries[idx]
                if isinstance(outcome, Exception):
                    logger.error(
                        "Clinical analysis for drug '%s' failed: %s",
                        entry.drug_name,
                        outcome,
                    )
                    entry.paragraph = self.build_error_paragraph(entry)
                else:
                    normalized_outcome = (
                        outcome.strip() if isinstance(outcome, str) else str(outcome).strip()
                    )
                    entry.paragraph = (
                        normalized_outcome
                        if normalized_outcome
                        else self.build_error_paragraph(entry)
                    )
                completed += 1
                self.emit_progress(
                    progress_callback,
                    stage="llm_analysis",
                    fraction=completed / total if total else 1.0,
                )
        else:
            self.emit_progress(progress_callback, stage="llm_analysis", fraction=1.0)

        logger.info("Composing final clinical report for current patient")
        self.emit_progress(progress_callback, stage="report_composition", fraction=0.0)
        final_report = await self.finalize_patient_report(
            entries,
            clinical_context=normalized_context,
        )
        self.emit_progress(progress_callback, stage="report_composition", fraction=1.0)

        return PatientDrugClinicalReport(entries=entries, final_report=final_report)

    # -------------------------------------------------------------------------
    @staticmethod
    def emit_progress(
        progress_callback: Callable[[str, float], None] | None,
        *,
        stage: str,
        fraction: float,
    ) -> None:
        if progress_callback is None:
            return
        bounded_fraction = min(1.0, max(0.0, float(fraction)))
        progress_callback(stage, bounded_fraction)

    # -------------------------------------------------------------------------
    @staticmethod
    async def execute_indexed_job(index: int, coroutine: Any) -> tuple[int, Any]:
        try:
            return index, await coroutine
        except Exception as exc:  # noqa: BLE001
            return index, exc

    # -------------------------------------------------------------------------
    async def execute_bounded_job(
        self,
        index: int,
        coroutine: Any,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, Any]:
        async with semaphore:
            return await self.execute_indexed_job(index, coroutine)

    # -------------------------------------------------------------------------
    async def prepare_drug_assessment(
        self,
        *,
        idx: int,
        drug_entry: DrugEntry,
        resolved_drugs: dict[str, dict[str, Any]],
        visit_date: date | None,
        normalized_context: str,
        pattern_summary: str,
        rag_query: dict[str, str] | None,
        use_web_search: bool,
    ) -> tuple[DrugClinicalAssessment, tuple[int, Any] | None]:
        raw_name = drug_entry.name or ""
        normalized_drug_key = normalize_drug_query_name(raw_name)

        livertox_data = resolved_drugs.get(normalized_drug_key, {})
        matched_row = livertox_data.get("matched_livertox_row", None)
        excerpts_list = livertox_data.get("extracted_excerpts", [])
        canonical_name = str(livertox_data.get("canonical_name") or raw_name).strip() or raw_name
        origins = [
            origin
            for origin in livertox_data.get("origins", [])
            if isinstance(origin, str) and origin.strip()
        ]
        if not origins and drug_entry.source in {"therapy", "anamnesis"}:
            origins = [drug_entry.source]
        extraction_metadata = livertox_data.get("extraction_metadata", [])
        missing_livertox = bool(livertox_data.get("missing_livertox"))
        ambiguous_match = bool(livertox_data.get("ambiguous_match"))
        match_candidates = [
            str(candidate).strip()
            for candidate in livertox_data.get("match_candidates", [])
            if str(candidate).strip()
        ]

        suspension = self.evaluate_suspension(drug_entry, visit_date)
        matched_lvt_row = matched_row if isinstance(matched_row, dict) else None
        entry = DrugClinicalAssessment(
            drug_name=drug_entry.name,
            canonical_name=canonical_name,
            origins=origins,
            extraction_metadata=extraction_metadata,
            matched_livertox_row=matched_lvt_row,
            extracted_excerpts=excerpts_list,
            missing_livertox=missing_livertox,
            ambiguous_match=ambiguous_match,
            match_candidates=match_candidates,
            suspension=suspension,
        )

        if suspension.excluded:
            entry.paragraph = self.build_excluded_paragraph(entry)
            return entry, None

        if entry.ambiguous_match:
            entry.paragraph = self.build_ambiguous_match_paragraph(entry)
            return entry, None

        excerpt = self.select_excerpt(excerpts_list)
        if excerpt is None or entry.missing_livertox:
            entry.missing_livertox = True
            entry.paragraph = self.build_missing_excerpt_paragraph(entry)
            return entry, None

        rag_documents = await self.fetch_rag_documents(rag_query, raw_name)
        web_evidence = self.web_evidence_disabled_text()
        if use_web_search:
            web_evidence = await self.fetch_web_evidence_for_drug(
                drug_name=drug_entry.name,
            )
        job = self.request_drug_analysis(
            drug_name=drug_entry.name,
            canonical_name=entry.canonical_name or drug_entry.name,
            origins=entry.origins,
            extraction_metadata=entry.extraction_metadata,
            livertox_status=(
                "missing_livertox"
                if entry.missing_livertox
                else "ambiguous_match"
                if entry.ambiguous_match
                else "matched"
            ),
            excerpt=excerpt,
            rag_documents=rag_documents or None,
            clinical_context=normalized_context,
            suspension=suspension,
            visit_date=visit_date,
            pattern_summary=pattern_summary,
            metadata=entry.matched_livertox_row,
            web_evidence=web_evidence,
        )
        return entry, (idx, job)

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
        return await asyncio.to_thread(
            self.search_supporting_documents,
            drug_rag_query,
            self.rag_top_k,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def web_evidence_disabled_text() -> str:
        return "No web evidence available (reason: web search disabled for this session)."

    # -------------------------------------------------------------------------
    async def fetch_web_evidence_for_drug(self, *, drug_name: str) -> str:
        normalized_name = (drug_name or "").strip()
        if not normalized_name:
            return "No web evidence available (reason: missing drug name)."
        try:
            outcome = await tavily_research_service.search_sources(
                question=f"{normalized_name} drug induced liver injury evidence",
                mode="fast",
                allowed_domains=None,
                blocked_domains=None,
            )
        except Exception as exc:  # noqa: BLE001
            return f"No web evidence available (reason: {exc})."
        return tavily_research_service.format_clinical_evidence_block(
            sources=outcome.sources,
            message=outcome.message,
        )

    # -------------------------------------------------------------------------
    def ensure_similarity_search(self) -> bool:
        if self.similarity_search is not None:
            return True
        try:
            self.similarity_search = SimilaritySearch()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize similarity search: %s", exc)
            self.similarity_search = None
            return False
        return True

    # -------------------------------------------------------------------------
    def select_excerpt(self, excerpts: list[str]) -> str | None:
        excerpts = [chunk.strip() for chunk in excerpts if chunk.strip()]
        if not excerpts:
            return None
        combined = "\n\n".join(excerpts)
        if len(combined) <= self.MAX_EXCERPT_LENGTH:
            return combined
        # Keep the most informative text while respecting the token budget
        truncated = combined[: self.MAX_EXCERPT_LENGTH]
        cutoff = truncated.rfind("\n")
        if cutoff > 2000:
            truncated = truncated[:cutoff]
        return truncated.strip()

    # -------------------------------------------------------------------------
    def search_supporting_documents(
        self, query_text: str | Any, top_k: int | None = None
    ) -> str | None:
        if not isinstance(query_text, str):
            return None
        normalized = query_text.strip()
        if not normalized or not self.ensure_similarity_search():
            return None

        results = (
            self.similarity_search.search(normalized, top_k=top_k)
            if self.similarity_search
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
    def format_similarity_fragment(self, index: int, record: dict[str, Any]) -> str | None:
        text = str(record.get("text", "")).strip()
        if not text:
            return None
        header = self.format_similarity_header(index, record.get("distance"))
        return f"{header}\n{text}"

    # -------------------------------------------------------------------------
    @staticmethod
    def format_similarity_header(index: int, distance: Any) -> str:
        if isinstance(distance, (int, float)):
            return f"[Document {index} | Distance: {distance:.4f}]"
        return f"[Document {index}]"

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
            # No suspension means we track exposure but keep contextual notes
            if entry.source == "anamnesis" or bool(entry.historical_flag):
                exposure_note = (
                    "Historical mention from anamnesis without explicit active regimen; "
                    "treat current exposure as uncertain unless confirmed in therapy."
                )
            else:
                exposure_note = "Active therapy; no suspension reported."
            combined_note = " ".join(
                part
                for part in (
                    start_note,
                    exposure_note,
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
            suspension_note = f"Suspended on {parsed_date.isoformat()}, but visit date missing; drug kept in analysis."
        else:
            interval_days = (visit_date - parsed_date).days
            if interval_days < 0:
                suspension_note = f"Suspended on {parsed_date.isoformat()} ({abs(interval_days)} days after the visit); treat as ongoing exposure."
            elif interval_days == 0:
                suspension_note = f"Suspended on {parsed_date.isoformat()} (same day as the visit); residual exposure is expected."
            else:
                suspension_note = f"Suspended on {parsed_date.isoformat()} ({interval_days} days before the visit); compare this latency with LiverTox guidance."

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
            return "Therapy start was not documented; assume chronic exposure unless another source clarifies the onset."
        if start_date is None:
            return "Therapy start was reported but no reliable date could be parsed from the notes."
        if visit_date is None or start_interval_days is None:
            return f"Therapy started on {start_date.isoformat()}, but the visit date was unavailable for latency comparisons."
        if start_interval_days < 0:
            humanized = self.humanize_interval(abs(start_interval_days))
            return f"Therapy was documented to start on {start_date.isoformat()}, {humanized} after the visit; verify this discrepancy manually."
        if start_interval_days == 0:
            return f"Therapy started on {start_date.isoformat()}, coinciding with the clinical visit."
        humanized = self.humanize_interval(start_interval_days)
        return f"Therapy started on {start_date.isoformat()}, roughly {humanized} before the visit."

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
        return "No therapy start information was detected; treat the exposure window as chronic unless contradicted."

    # -------------------------------------------------------------------------
    @staticmethod
    def format_visit_date_anchor(visit_date: date | None) -> str:
        if visit_date is None:
            return "Not provided."
        return visit_date.isoformat()

    # -------------------------------------------------------------------------
    def resolve_livertox_score(self, metadata: dict[str, Any] | None) -> str:
        if not metadata:
            return NOT_AVAILABLE_TEXT
        score = metadata.get("likelihood_score")
        if score is None:
            return NOT_AVAILABLE_TEXT
        text = str(score).strip()
        if not text or text.lower() == "nan":
            return NOT_AVAILABLE_TEXT
        return text.upper() if text.isalpha() else text

    # -------------------------------------------------------------------------
    def prepare_metadata_prompt(
        self, metadata: dict[str, Any] | None
    ) -> tuple[str, str]:
        score = self.resolve_livertox_score(metadata)
        details: list[str] = [f"- Likelihood score: {score}"]
        if metadata:
            mapping = [
                ("Agent classification", metadata.get("agent_classification")),
                ("Primary classification", metadata.get("primary_classification")),
                ("Secondary classification", metadata.get("secondary_classification")),
                ("Reference count", metadata.get("reference_count")),
                ("Year approved", metadata.get("year_approved")),
            ]
            seen: set[str] = set()
            for label, raw in mapping:
                if raw is None:
                    continue
                value = str(raw).strip()
                if not value or value.lower() == "nan":
                    continue
                key = f"{label}:{value}"
                if key in seen:
                    continue
                seen.add(key)
                details.append(f"- {label}: {value}")
        if len(details) == 1:
            details.append("- No additional LiverTox metadata was available.")
        return score, "\n".join(details)

    # -------------------------------------------------------------------------
    def format_drug_heading(self, drug_name: str, score: str) -> str:
        normalized_name = drug_name.strip() if drug_name else ""
        if not normalized_name:
            normalized_name = "Unnamed drug"
        normalized_score = score.strip() if score else ""
        if not normalized_score:
            normalized_score = NOT_AVAILABLE_TEXT
        return f"{normalized_name} - LiverTox score {normalized_score}"

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
        rag_documents: str | None,
        clinical_context: str,
        suspension: DrugSuspensionContext,
        visit_date: date | None,
        pattern_summary: str,
        metadata: dict[str, Any] | None,
        web_evidence: str,
    ) -> str:
        start_details = self.format_start_prompt(suspension)
        suspension_details = self.format_suspension_prompt(suspension)
        timeline_note = (
            suspension.note
            or "No explicit timeline notes were available from extraction metadata."
        )
        visit_date_anchor = self.format_visit_date_anchor(visit_date)
        score, metadata_block = self.prepare_metadata_prompt(metadata)
        rag_documents = rag_documents or "No additional documents provided."
        origin_block = ", ".join(origins) if origins else "unknown"
        metadata_items = [
            f"- {json.dumps(item, ensure_ascii=False)}"
            for item in extraction_metadata
            if isinstance(item, dict) and item
        ]
        extraction_block = "\n".join(metadata_items) if metadata_items else "- Not available"
        user_prompt = LIVERTOX_CLINICAL_USER_PROMPT.format(
            drug_name=self.escape_braces(drug_name.strip() or drug_name),
            canonical_name=self.escape_braces(canonical_name.strip() or canonical_name),
            origins=self.escape_braces(origin_block),
            extraction_metadata=self.escape_braces(extraction_block),
            livertox_status=self.escape_braces(livertox_status),
            excerpt=self.escape_braces(excerpt),
            documents=self.escape_braces(rag_documents),
            web_evidence=self.escape_braces(web_evidence),
            clinical_context=self.escape_braces(clinical_context),
            visit_date_anchor=self.escape_braces(visit_date_anchor),
            therapy_start_details=self.escape_braces(start_details),
            suspension_details=self.escape_braces(suspension_details),
            timeline_note=self.escape_braces(timeline_note),
            pattern_summary=self.escape_braces(pattern_summary),
            metadata_block=self.escape_braces(metadata_block),
            livertox_score=self.escape_braces(score),
            example_block=self.escape_braces(LIVERTOX_REPORT_EXAMPLE),
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
        for attempt in range(1, self.analysis_retry_attempts + 1):
            try:
                # Ask the clinical model to synthesise findings for this drug
                raw_response = await self.llm_client.chat(**chat_kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.analysis_retry_attempts:
                    raise RuntimeError(
                        f"LLM analysis failed for {drug_name}: {exc}"
                    ) from exc
                delay = self.retry_backoff_seconds(attempt)
                logger.warning(
                    "Retrying LLM analysis for '%s' after error (attempt %d/%d): %s",
                    drug_name,
                    attempt,
                    self.analysis_retry_attempts,
                    exc,
                )
                await asyncio.sleep(delay)
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
    def retry_backoff_seconds(self, attempt: int) -> float:
        normalized_attempt = max(int(attempt), 1)
        return min(2.0, 0.35 * normalized_attempt)

    # -------------------------------------------------------------------------
    async def finalize_patient_report(
        self,
        entries: list[DrugClinicalAssessment],
        *,
        clinical_context: str | None,
    ) -> str | None:
        paragraphs = [
            entry.paragraph.strip()
            for entry in entries
            if entry.paragraph and entry.paragraph.strip()
        ]
        if not paragraphs:
            return None
        separator = "\n\n---\n\n" if len(paragraphs) > 1 else "\n\n"
        combined_report = separator.join(paragraphs)
        conclusion = await self.generate_conclusion(
            clinical_context=clinical_context or "",
            multi_drug_report=combined_report,
        )
        if conclusion:
            combined_report = (
                f"{combined_report}\n\n## Global Synthesis and Clinical Recommendations\n\n{conclusion}"
            )
        return combined_report

    # -------------------------------------------------------------------------
    async def generate_conclusion(
        self,
        *,
        clinical_context: str,
        multi_drug_report: str,
    ) -> str | None:
        report_body = multi_drug_report.strip()
        if not report_body:
            return None
        context_body = clinical_context.strip()
        if not context_body:
            context_body = "No clinical context was provided."
        user_prompt = LIVERTOX_CONCLUSION_USER_PROMPT.format(
            clinical_context=self.escape_braces(context_body),
            multi_drug_report=self.escape_braces(report_body),
        )
        messages = [
            {"role": "system", "content": LIVERTOX_CONCLUSION_SYSTEM_PROMPT.strip()},
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
            logger.error("Failed to generate clinical conclusion: %s", exc)
            return None
        conclusion = self.coerce_chat_text(raw_response).strip()
        return conclusion or None

    # -------------------------------------------------------------------------
    def build_excluded_paragraph(self, entry: DrugClinicalAssessment) -> str:
        score = self.resolve_livertox_score(entry.matched_livertox_row)
        heading = self.format_drug_heading(entry.drug_name, score)
        suspension = entry.suspension
        if suspension.suspension_date is not None:
            detail = f"The therapy was suspended on {suspension.suspension_date.isoformat()} well before the visit, so the drug was excluded from this DILI assessment."
        else:
            detail = "The therapy was reported as suspended well before the visit and was excluded from the current DILI assessment."
        recommendation = "Manual verification of latency is suggested if the exposure history becomes relevant again."
        return (
            f"{heading}\n{detail} {recommendation}\nBibliography source: LiverTox"
        )

    # -------------------------------------------------------------------------
    def build_missing_excerpt_paragraph(self, entry: DrugClinicalAssessment) -> str:
        score = self.resolve_livertox_score(entry.matched_livertox_row)
        heading = self.format_drug_heading(entry.drug_name, score)
        if entry.matched_livertox_row:
            note = (
                "Automated assessment unavailable: a LiverTox monograph was matched, "
                "but no local excerpt text is available in the current knowledge-base snapshot."
            )
        else:
            note = (
                "Automated assessment unavailable: no local LiverTox excerpt is available for this drug."
            )
        guidance = "Review the LiverTox monograph manually before attributing causality."
        return f"{heading}\n{note} {guidance}\nBibliography source: LiverTox"

    # -------------------------------------------------------------------------
    def build_ambiguous_match_paragraph(self, entry: DrugClinicalAssessment) -> str:
        heading = self.format_drug_heading(entry.drug_name, NOT_AVAILABLE_TEXT)
        candidates = ", ".join(entry.match_candidates) if entry.match_candidates else "Not available"
        note = (
            "Drug matching was ambiguous; no LiverTox excerpt was injected to avoid an "
            "incorrect attribution."
        )
        details = f"Candidate matches: {candidates}."
        guidance = "Manual curation is required before causality assessment."
        return f"{heading}\n{note} {details} {guidance}\nBibliography source: LiverTox"

    # -------------------------------------------------------------------------
    def build_error_paragraph(self, entry: DrugClinicalAssessment) -> str:
        score = self.resolve_livertox_score(entry.matched_livertox_row)
        heading = self.format_drug_heading(entry.drug_name, score)
        message = "Automated analysis was unavailable due to a technical issue; a clinician should review the LiverTox documentation manually."
        return f"{heading}\n{message}\nBibliography source: LiverTox"

