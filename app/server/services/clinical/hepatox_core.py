from __future__ import annotations

import asyncio
from datetime import datetime
import re
from collections.abc import Callable
from datetime import date
from typing import Any

from services.llm.runtime_config import LLMRuntimeConfig
from configurations.startup import get_server_settings
from domain.clinical.entities import (
    DrugClinicalAssessment,
    DrugEntry,
    DrugRucamAssessment,
    DrugSuspensionContext,
    PatientDrugClinicalReport,
    PatientDrugs,
    PatientRucamAssessmentBundle,
    PipelineIssue,
)
from services.clinical.preparation import HepatoxPreparedInputs
from services.clinical.report_language import (
    evidence_quality_label,
    limitation_label,
    phrase,
    report_heading,
    rucam_summary_text,
)
from services.llm.provider_factory import initialize_llm_client
from services.retrieval.embeddings import SimilaritySearch
from services.retrieval.settings import build_effective_rag_settings
from services.text.normalization import normalize_drug_query_name
from services.clinical.analysis_runner import AnalysisRunner
from services.clinical.drug_analysis import DrugAnalysisService
from services.clinical import hepatox_scoring
from services.clinical.rag_support import RagRetrievalBundle, RagSupportService
from services.clinical.report_finalizer import ReportFinalizer
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
class HepatoxConsultation:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        drugs: PatientDrugs,
        *,
        patient_name: str | None = None,
        timeout_s: float = get_server_settings().runtime.clinical_llm_timeout,
    ) -> None:
        self.drugs = drugs
        self.timeout_s = timeout_s
        self.llm_client = initialize_llm_client(purpose="clinical", timeout_s=timeout_s)
        self.MAX_EXCERPT_LENGTH = get_server_settings().runtime.max_excerpt_length
        self.patient_name = (patient_name or "").strip() or None
        provider, model_candidate = LLMRuntimeConfig.resolve_provider_and_model(
            "clinical"
        )
        self.llm_model = model_candidate or LLMRuntimeConfig.get_clinical_model()
        self.chat_supports_temperature = False
        self.similarity_search: SimilaritySearch | None = None
        rag_settings = build_effective_rag_settings()
        self.rag_use_reranking = bool(rag_settings.use_reranking)
        self.rag_top_n = max(int(rag_settings.retrieval_selected_count), 1)
        self.rag_candidate_k = max(
            int(rag_settings.retrieval_candidate_count), self.rag_top_n
        )
        self.pipeline_issues: list[PipelineIssue] = []
        default_parallel_analyses = 3 if provider == "ollama" else 1
        self.max_parallel_analyses = max(
            1,
            int(
                getattr(
                    get_server_settings().runtime,
                    "clinical_llm_max_concurrency",
                    default_parallel_analyses,
                )
            ),
        )
        default_retry_attempts = 1
        configured_retry_attempts = int(
            getattr(
                get_server_settings().runtime,
                "clinical_llm_retry_attempts",
                default_retry_attempts,
            )
        )
        # Keep consultation responsive when cloud providers are timing out.
        # One attempt is enough before falling back to deterministic outputs.
        self.analysis_retry_attempts = max(1, min(configured_retry_attempts, 1))

        # Focused sub-services
        self.analysis_runner = AnalysisRunner(self)
        self.drug_analysis = DrugAnalysisService(self)
        self.report_finalizer = ReportFinalizer(self)
        self.rag_support = RagSupportService(self)

    # -------------------------------------------------------------------------
    async def run_analysis(
        self,
        *,
        prepared_inputs: HepatoxPreparedInputs | None,
        visit_date: date | None = None,
        report_language: str = "en",
        rag_query: dict[str, str] | None = None,
        rucam_bundle: PatientRucamAssessmentBundle | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any] | None:
        return await self.analysis_runner.run_analysis(
            prepared_inputs=prepared_inputs,
            visit_date=visit_date,
            report_language=report_language,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            progress_callback=progress_callback,
        )

    # -------------------------------------------------------------------------
    async def run_revision_analysis(
        self,
        *,
        prepared_inputs: HepatoxPreparedInputs | None,
        visit_date: date | None = None,
        report_language: str = "en",
        rag_query: dict[str, str] | None = None,
        rucam_bundle: PatientRucamAssessmentBundle | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any] | None:
        return await self.analysis_runner.run_revision_analysis(
            prepared_inputs=prepared_inputs,
            visit_date=visit_date,
            report_language=report_language,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            progress_callback=progress_callback,
        )

    # -------------------------------------------------------------------------
    async def compile_clinical_assessment(
        self,
        resolved_drugs: dict[str, dict[str, Any]],
        *,
        clinical_context: str | None,
        visit_date: date | None,
        report_language: str,
        pattern_prompt: str,
        rag_query: dict[str, str] | None = None,
        rucam_bundle: PatientRucamAssessmentBundle | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> PatientDrugClinicalReport:
        return await self.analysis_runner.compile_clinical_assessment(
            resolved_drugs,
            clinical_context=clinical_context,
            visit_date=visit_date,
            report_language=report_language,
            pattern_prompt=pattern_prompt,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            progress_callback=progress_callback,
        )

    # -------------------------------------------------------------------------
    async def compile_revision_clinical_assessment(
        self,
        resolved_drugs: dict[str, dict[str, Any]],
        *,
        clinical_context: str | None,
        visit_date: date | None,
        report_language: str,
        pattern_prompt: str,
        rag_query: dict[str, str] | None = None,
        rucam_bundle: PatientRucamAssessmentBundle | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> PatientDrugClinicalReport:
        return await self.analysis_runner.compile_revision_clinical_assessment(
            resolved_drugs,
            clinical_context=clinical_context,
            visit_date=visit_date,
            report_language=report_language,
            pattern_prompt=pattern_prompt,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            progress_callback=progress_callback,
        )

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
        return await AnalysisRunner.execute_indexed_job(index, coroutine)

    # -------------------------------------------------------------------------
    async def execute_bounded_job(
        self,
        index: int,
        coroutine: Any,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, Any]:
        return await self.analysis_runner.execute_bounded_job(
            index, coroutine, semaphore
        )

    # -------------------------------------------------------------------------
    async def prepare_drug_assessment(
        self,
        *,
        idx: int,
        drug_entry: DrugEntry,
        resolved_drugs: dict[str, dict[str, Any]],
        visit_date: date | None,
        report_language: str,
        normalized_context: str,
        pattern_summary: str,
        rag_query: dict[str, str] | None,
        rucam_by_key: dict[str, DrugRucamAssessment],
    ) -> tuple[DrugClinicalAssessment, tuple[int, Any] | None]:
        return await self.analysis_runner.prepare_drug_assessment(
            idx=idx,
            drug_entry=drug_entry,
            resolved_drugs=resolved_drugs,
            visit_date=visit_date,
            report_language=report_language,
            normalized_context=normalized_context,
            pattern_summary=pattern_summary,
            rag_query=rag_query,
            rucam_by_key=rucam_by_key,
        )

    # -------------------------------------------------------------------------
    async def prepare_revision_drug_assessment(
        self,
        *,
        idx: int,
        drug_entry: DrugEntry,
        resolved_drugs: dict[str, dict[str, Any]],
        visit_date: date | None,
        report_language: str,
        normalized_context: str,
        pattern_summary: str,
        rag_query: dict[str, str] | None,
        rucam_by_key: dict[str, DrugRucamAssessment],
    ) -> tuple[DrugClinicalAssessment, tuple[int, Any] | None]:
        return await self.analysis_runner.prepare_revision_drug_assessment(
            idx=idx,
            drug_entry=drug_entry,
            resolved_drugs=resolved_drugs,
            visit_date=visit_date,
            report_language=report_language,
            normalized_context=normalized_context,
            pattern_summary=pattern_summary,
            rag_query=rag_query,
            rucam_by_key=rucam_by_key,
        )

    # -------------------------------------------------------------------------
    def resolve_livertox_data_for_entry(
        self,
        *,
        raw_name: str,
        normalized_key: str,
        resolved_drugs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return self.analysis_runner.resolve_livertox_data_for_entry(
            raw_name=raw_name,
            normalized_key=normalized_key,
            resolved_drugs=resolved_drugs,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def livertox_payload_rank(payload: dict[str, Any]) -> int:
        return AnalysisRunner.livertox_payload_rank(payload)

    # -------------------------------------------------------------------------
    async def fetch_rag_documents(
        self, rag_query: dict[str, str] | None, drug_name: str
    ) -> RagRetrievalBundle | None:
        if (
            type(self).search_supporting_documents
            is not HepatoxConsultation.search_supporting_documents
        ):
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
            except Exception as exc:
                self.record_rag_retrieval_issue(drug_name=drug_name, error=exc)
                return None
        return await self.rag_support.fetch_rag_documents(rag_query, drug_name)

    # -------------------------------------------------------------------------
    def record_rag_retrieval_issue(self, *, drug_name: str, error: Exception) -> None:
        self.rag_support.record_rag_retrieval_issue(drug_name=drug_name, error=error)

    # -------------------------------------------------------------------------
    def ensure_similarity_search(self) -> bool:
        return self.rag_support.ensure_similarity_search()

    # -------------------------------------------------------------------------
    def select_excerpt(self, excerpts: list[str]) -> str | None:
        return self.rag_support.select_excerpt(excerpts)

    # -------------------------------------------------------------------------
    def search_supporting_documents(
        self, query_text: str | Any
    ) -> RagRetrievalBundle | None:
        return self.rag_support.search_supporting_documents(query_text)

    # -------------------------------------------------------------------------
    def format_similarity_fragment(
        self, index: int, record: dict[str, Any]
    ) -> str | None:
        return self.rag_support.format_similarity_fragment(index, record)

    # -------------------------------------------------------------------------
    def format_similarity_header(
        self,
        index: int,
        *,
        distance: Any,
        rerank_score: Any = None,
    ) -> str:
        return self.rag_support.format_similarity_header(
            index, distance=distance, rerank_score=rerank_score
        )

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
            if entry.source == "anamnesis" or bool(entry.historical_flag):
                exposure_note = (
                    "Historical mention from anamnesis without explicit active regimen; "
                    "treat current exposure as uncertain unless confirmed in therapy."
                )
            else:
                exposure_note = "Active therapy; no suspension reported."
            combined_note = " ".join(
                part for part in (start_note, exposure_note) if part
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
                f"Suspended on {parsed_date.isoformat()}, but visit date missing; "
                "drug kept in analysis."
            )
        else:
            interval_days = (visit_date - parsed_date).days
            if interval_days < 0:
                suspension_note = (
                    f"Suspended on {parsed_date.isoformat()} "
                    f"({abs(interval_days)} days after the visit); "
                    "treat as ongoing exposure."
                )
            elif interval_days == 0:
                suspension_note = (
                    f"Suspended on {parsed_date.isoformat()} "
                    "(same day as the visit); residual exposure is expected."
                )
            else:
                suspension_note = (
                    f"Suspended on {parsed_date.isoformat()} "
                    f"({interval_days} days before the visit); compare this latency "
                    "with LiverTox guidance."
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
        candidates.extend(["-".join(tokens), text, normalized])
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
            return f"{round(weeks, 1):g} weeks"
        months = days / 30.4375
        if days < 365:
            return f"{round(months, 1):g} months"
        years = days / 365.25
        return f"{round(years, 1):g} years"

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
    def format_rucam_prompt_block(self, rucam: DrugRucamAssessment | None) -> str:
        if rucam is None:
            return "Estimated RUCAM not available."
        limitations = ", ".join((rucam.limitations or [])[:3]) or "not specified"
        return (
            f"- Score: {rucam.total_score}\n"
            f"- Category: {rucam.causality_category}\n"
            f"- Confidence: {rucam.confidence}\n"
            f"- Estimated due to incomplete clinical data: yes\n"
            f"- Key limitations: {limitations}"
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def is_materially_in_report_language(text: str, report_language: str) -> bool:
        return hepatox_scoring.is_materially_in_report_language(text, report_language)

    # -------------------------------------------------------------------------
    async def repair_language_once(
        self,
        *,
        source_text: str,
        report_language: str,
    ) -> str:
        return await self.rag_support.repair_language_once(
            source_text=source_text, report_language=report_language
        )

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
        return await self.drug_analysis.request_drug_analysis(
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
        return await self.drug_analysis.request_revision_drug_analysis(
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
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def escape_braces(value: str) -> str:
        return value.replace("{", "{{").replace("}", "}}")

    # -------------------------------------------------------------------------
    @staticmethod
    def coerce_chat_text(raw_response: Any) -> str:
        return DrugAnalysisService.coerce_chat_text(raw_response)

    # -------------------------------------------------------------------------
    def extract_rate_limit_wait_hint_seconds(self, exc: Exception) -> float | None:
        return self.rag_support.extract_rate_limit_wait_hint_seconds(exc)

    # -------------------------------------------------------------------------
    def retry_backoff_seconds(
        self, attempt: int, *, exc: Exception | None = None
    ) -> float:
        return self.analysis_runner.retry_backoff_seconds(attempt, exc=exc)

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def should_render_as_matched_drug(self, entry: DrugClinicalAssessment) -> bool:
        return self.report_finalizer.should_render_as_matched_drug(entry)

    # -------------------------------------------------------------------------
    def render_matched_drug_section(
        self,
        entry: DrugClinicalAssessment,
        *,
        report_language: str = "en",
    ) -> str:
        score = self.resolve_livertox_score(entry.matched_livertox_row)
        title = self.format_drug_heading(entry.drug_name, score)
        body = self.sanitize_renderable_body(entry)
        if not body:
            body = self.build_fallback_technical_note(
                entry, report_language=report_language
            )
        rucam = entry.rucam
        localized_rucam = (
            rucam_summary_text(rucam, report_language)
            if rucam is not None
            else phrase("rucam_not_calculated", report_language)
        )
        clinical_commentary = self.render_clinical_commentary(
            entry,
            report_language=report_language,
        )
        report_label = phrase("report_label", report_language)
        bibliography_label = phrase("bibliography_source", report_language)
        return (
            f"**{title}**\n\n"
            f"{clinical_commentary}\n\n"
            f"**RUCAM**: {localized_rucam}\n\n"
            f"**{report_label}**\n\n"
            f"{body}\n\n"
            f"**{bibliography_label}**: {ReportFinalizer.bibliography_source_label()}"
        ).strip()

    # -------------------------------------------------------------------------
    @staticmethod
    def render_clinical_commentary(
        entry: DrugClinicalAssessment,
        *,
        report_language: str = "en",
    ) -> str:
        quality = evidence_quality_label(
            entry.evidence_quality or phrase("unknown", report_language),
            report_language,
        )
        matched_name = ""
        if isinstance(entry.matched_livertox_row, dict):
            matched_name = str(
                entry.matched_livertox_row.get("drug_name") or ""
            ).strip()
        target = (
            matched_name
            or entry.canonical_name
            or phrase("not_available", report_language)
        )
        segments = [
            phrase(
                "commentary_evidence_match",
                report_language,
                quality=quality,
                target=target,
            )
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
        rucam = entry.rucam
        if rucam is not None and rucam.total_score is None:
            segments.append(phrase("commentary_rucam_not_assessable", report_language))
        if limitations:
            segments.append(
                phrase(
                    "commentary_limitations",
                    report_language,
                    limitations="; ".join(
                        limitation_label(item, report_language)
                        for item in limitations[:3]
                    ),
                )
            )
        if review_claims or limitations:
            segments.append(phrase("commentary_review_required", report_language))
        else:
            segments.append(phrase("commentary_no_review_required", report_language))
        return f"**{phrase('clinical_commentary', report_language)}**: " + " ".join(
            segments
        )

    # -------------------------------------------------------------------------
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
            if REDUNDANT_REPORT_LINE_RE.search(
                re.sub(r"[\s*_`#:\-]+", " ", stripped).strip()
            ):
                continue
            if REPORT_LABEL_LINE_RE.match(stripped):
                continue
            if BIBLIOGRAPHY_LINE_RE.match(stripped):
                continue
            if stripped == "---":
                continue
            if stripped.lower().startswith("## global synthesis"):
                break
            if DRIFT_SECTION_LINE_RE.match(stripped):
                break
            if STRUCTURED_DILI_SECTION_LINE_RE.match(stripped):
                break
            title_match = LIVERTOX_TITLE_LINE_RE.match(stripped)
            if title_match:
                if expected_name and expected_name not in stripped.lower():
                    continue
                continue
            lines.append(raw_line.rstrip())
        sanitized = "\n".join(lines).strip()
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()
        normalized = re.sub(r"\s+", " ", sanitized).strip().lower()
        if "local livertox excerpt not available" in normalized:
            return ""
        return sanitized

    # -------------------------------------------------------------------------
    def build_fallback_technical_note(
        self,
        entry: DrugClinicalAssessment,
        *,
        report_language: str = "en",
    ) -> str:
        if entry.suspension.excluded:
            return self.build_excluded_paragraph(entry, report_language=report_language)
        if entry.ambiguous_match:
            return self.build_ambiguous_match_paragraph(
                entry,
                report_language=report_language,
            )
        if entry.missing_livertox:
            if entry.matched_livertox_row:
                return phrase("matched_no_excerpt", report_language)
            return phrase("livertox_missing", report_language)
        return self.build_error_paragraph(entry, report_language=report_language)

    # -------------------------------------------------------------------------
    def render_unresolved_mentions_section(
        self,
        entries: list[DrugClinicalAssessment],
        *,
        report_language: str = "en",
    ) -> str | None:
        if not entries:
            return None
        lines: list[str] = [
            f"## {report_heading('unresolved_mentions', report_language)}",
            "",
        ]
        for entry in entries:
            label = (entry.drug_name or "").strip() or phrase(
                "unnamed_drug", report_language
            )
            reason = self.describe_unresolved_entry(
                entry, report_language=report_language
            )
            rucam_summary = (
                rucam_summary_text(entry.rucam, report_language)
                if entry.rucam is not None
                else phrase("rucam_not_calculated", report_language)
            )
            technical_context = self.build_fallback_technical_note(
                entry,
                report_language=report_language,
            )
            clinical_context = (
                reason
                if technical_context.strip() == reason.strip()
                else f"{technical_context} {reason}"
            )
            if report_language.lower().startswith("it"):
                recommendation = (
                    "Il farmaco deve rimanere nella diagnosi differenziale solo in base "
                    "alla cronologia clinica disponibile; prima di attribuire causalità "
                    "sono necessari verifica dell'esposizione, andamento dopo sospensione "
                    "e revisione delle cause alternative."
                )
            else:
                recommendation = (
                    "The drug should remain in the differential diagnosis only to the "
                    "extent supported by the available timeline; exposure verification, "
                    "the course after withdrawal, and competing-cause review are required "
                    "before causality is assigned."
                )
            lines.extend(
                [
                    f"### {label}",
                    "",
                    f"{clinical_context} {rucam_summary}. {recommendation}",
                    "",
                ]
            )
        return "\n".join(lines).strip()

    # -------------------------------------------------------------------------
    def describe_unresolved_entry(
        self,
        entry: DrugClinicalAssessment,
        report_language: str = "en",
    ) -> str:
        status = (entry.match_status or "").strip().lower()
        if status in {"ambiguous", "ambiguous_match"} or entry.ambiguous_match:
            candidates = (
                ", ".join(entry.match_candidates)
                if entry.match_candidates
                else phrase("rucam_insufficient_data", report_language)
            )
            return (
                f"{phrase('livertox_ambiguous', report_language)} "
                f"{phrase('candidate_matches', report_language, candidates=candidates)} "
                f"{phrase('manual_curation', report_language)}"
            )
        if status in {"missing", "missing_match"}:
            return phrase("no_matching_record", report_language)
        if status == "matched_no_excerpt":
            return phrase("matched_no_excerpt", report_language)
        if entry.missing_livertox:
            if entry.matched_livertox_row:
                return phrase("matched_no_excerpt", report_language)
            return phrase("livertox_missing", report_language)
        return phrase("deterministic_section_unavailable", report_language)

    # -------------------------------------------------------------------------
    def build_excluded_paragraph(
        self,
        entry: DrugClinicalAssessment,
        report_language: str = "en",
    ) -> str:
        suspension = entry.suspension
        if report_language.startswith("it"):
            if suspension.suspension_date is not None:
                detail = (
                    f"La terapia è stata sospesa il {suspension.suspension_date.isoformat()} "
                    "molto prima della visita; questa esposizione è stata quindi esclusa "
                    "dalla valutazione attiva di causalità DILI."
                )
            else:
                detail = (
                    "La terapia risulta sospesa molto prima della visita ed è stata "
                    "esclusa dalla valutazione attiva di causalità DILI."
                )
            recommendation = (
                "È consigliata una verifica manuale della latenza se l'esposizione "
                "torna clinicamente rilevante."
            )
            return f"{detail} {recommendation}"
        if suspension.suspension_date is not None:
            detail = (
                f"The therapy was suspended on {suspension.suspension_date.isoformat()} "
                "well before the visit, so this exposure was excluded from active DILI "
                "causality assessment."
            )
        else:
            detail = (
                "The therapy was reported as suspended well before the visit and was "
                "excluded from active DILI causality assessment."
            )
        recommendation = (
            "Manual latency verification is suggested if the exposure history becomes "
            "clinically relevant again."
        )
        return f"{detail} {recommendation}"

    # -------------------------------------------------------------------------
    def build_missing_excerpt_paragraph(
        self,
        entry: DrugClinicalAssessment,
        report_language: str = "en",
    ) -> str:
        _ = entry
        return phrase("livertox_missing", report_language)

    # -------------------------------------------------------------------------
    def build_ambiguous_match_paragraph(
        self,
        entry: DrugClinicalAssessment,
        report_language: str = "en",
    ) -> str:
        candidates = (
            ", ".join(entry.match_candidates)
            if entry.match_candidates
            else phrase("rucam_insufficient_data", report_language)
        )
        note = phrase("livertox_ambiguous", report_language)
        details = phrase("candidate_matches", report_language, candidates=candidates)
        guidance = phrase("manual_curation", report_language)
        return f"{note} {details} {guidance}"

    # -------------------------------------------------------------------------
    def build_error_paragraph(
        self,
        entry: DrugClinicalAssessment,
        report_language: str = "en",
    ) -> str:
        _ = entry
        message = phrase("rucam_insufficient_data", report_language)
        return message
