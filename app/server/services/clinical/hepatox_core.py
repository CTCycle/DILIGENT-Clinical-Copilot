from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from services.llm.prompts import (
    LIVERTOX_CONCLUSION_SYSTEM_PROMPT,
    LIVERTOX_CONCLUSION_USER_PROMPT,
    LIVERTOX_CLINICAL_SYSTEM_PROMPT,
    LIVERTOX_CLINICAL_USER_PROMPT,
    LIVERTOX_REPORT_EXAMPLE_TEMPLATE,
)
from services.llm.provider_factory import initialize_llm_client
from domain.clinical.entities import (
    ClinicalLabEntry,
    ClinicalPipelineValidationError,
    DrugEntry,
    DrugClinicalAssessment,
    DrugRucamAssessment,
    DrugSuspensionContext,
    HepatotoxicityPatternAssessment,
    HepatotoxicityPatternScore,
    PatientDrugClinicalReport,
    PatientDrugs,
    PatientLabTimeline,
    PatientRucamAssessmentBundle,
    PipelineIssue,
)
from configurations.startup import server_settings
from configurations.llm_configs import LLMRuntimeConfig
from common.constants import (
    DEFAULT_DILI_CLASSIFICATION,
    R_SCORE_CHOLESTATIC_THRESHOLD,
    R_SCORE_HEPATOCELLULAR_THRESHOLD,
)
from common.utils.logger import logger
from services.clinical.match_quality import classify_match_evidence
from services.retrieval.embeddings import SimilaritySearch
from services.clinical.preparation import HepatoxPreparedInputs
from services.text.normalization import normalize_drug_query_name
from services.research.brave import BraveResearchService
from services.text.vocabulary import get_text_normalization_snapshot
from services.clinical.report_language import (
    phrase,
    report_heading,
    rucam_summary_text,
)


###############################################################################
NOT_AVAILABLE_TEXT = "Not available"
REDUNDANT_REPORT_LINE_RE = re.compile(
    r"generated\s+report.*?(drug[- ]induced\s+liver\s+injury|\bdili\b)",
    re.IGNORECASE,
)
LIVERTOX_TITLE_LINE_RE = re.compile(
    r"^\s*\*{0,2}[^*\n]+?\s*-\s*LiverTox score\b.*\*{0,2}\s*$",
    re.IGNORECASE,
)
REPORT_LABEL_LINE_RE = re.compile(r"^\s*\*{0,2}\s*Report\s*\*{0,2}\s*$", re.IGNORECASE)
BIBLIOGRAPHY_LINE_RE = re.compile(
    r"^\s*\*{0,2}\s*Bibliography source\s*\*{0,2}\s*:\s*LiverTox\s*$",
    re.IGNORECASE,
)
DRIFT_SECTION_LINE_RE = re.compile(
    r"^\s*(medication|assessment|plan)\s*$", re.IGNORECASE
)
STRUCTURED_DILI_SECTION_LINE_RE = re.compile(
    r"^\s*#{0,6}\s*\*{0,2}\s*Structured\s+DILI\s+Assessment\s+Report\s*\*{0,2}\s*$",
    re.IGNORECASE,
)
RATE_LIMIT_WAIT_HINT_RE = re.compile(
    r"please\s+try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)


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
        self, lab_timeline: PatientLabTimeline
    ) -> HepatotoxicityPatternScore:
        anchor = self.select_anchor_pair(lab_timeline)
        if anchor is None:
            score = HepatotoxicityPatternScore(
                alt_multiple=None,
                alp_multiple=None,
                r_score=None,
                classification=DEFAULT_DILI_CLASSIFICATION,
            )
            self.r_score = None
            return score
        score = self.calculator.calculate(
            alt_value=anchor["alt_value"],
            alt_uln=anchor["alt_uln"],
            alp_value=anchor["alp_value"],
            alp_uln=anchor["alp_uln"],
        )
        self.r_score = score.r_score
        return score

    # -------------------------------------------------------------------------
    def assess_payload(
        self,
        lab_timeline: PatientLabTimeline,
    ) -> HepatotoxicityPatternAssessment:
        score = self.calculate_hepatotoxicity_pattern(lab_timeline)
        if score.r_score is None:
            issue = PipelineIssue(
                severity="error",
                code="missing_hepatotoxicity_inputs",
                message=(
                    "Provide laboratory data sufficient to determine hepatotoxicity pattern, "
                    "ideally dated ALT or AST, ALP, and bilirubin."
                ),
                field="laboratory_analysis",
            )
            raise ClinicalPipelineValidationError(issues=[issue], message=issue.message)
        self.r_score = score.r_score
        return HepatotoxicityPatternAssessment(
            score=score,
            status="ok",
            issues=[],
        )

    # -------------------------------------------------------------------------
    def select_anchor_pair(
        self, lab_timeline: PatientLabTimeline
    ) -> dict[str, float] | None:
        dated_candidates = self.group_entries_by_date(lab_timeline.entries)
        for sample_date in sorted(dated_candidates):
            bucket = dated_candidates[sample_date]
            pair = self.build_anchor_from_bucket(bucket)
            if pair is not None:
                return pair
        undated = self.build_anchor_from_bucket(lab_timeline.entries)
        return undated

    # -------------------------------------------------------------------------
    def group_entries_by_date(
        self,
        entries: list[ClinicalLabEntry],
    ) -> dict[str, list[ClinicalLabEntry]]:
        grouped: dict[str, list[ClinicalLabEntry]] = {}
        for entry in entries:
            if not entry.sample_date:
                continue
            grouped.setdefault(entry.sample_date, []).append(entry)
        return grouped

    # -------------------------------------------------------------------------
    def build_anchor_from_bucket(
        self,
        entries: list[ClinicalLabEntry],
    ) -> dict[str, float] | None:
        alt_like = self.pick_best_entry(entries, {"ALT", "AST"})
        alp = self.pick_best_entry(entries, {"ALP"})
        if alt_like is None or alp is None:
            return None
        alt_value = self.parse_entry_value(alt_like)
        alp_value = self.parse_entry_value(alp)
        if alt_value is None or alp_value is None:
            return None
        alt_uln = self.resolve_uln(alt_like, fallback=40.0)
        alp_uln = self.resolve_uln(alp, fallback=120.0)
        if alt_uln <= 0 or alp_uln <= 0:
            return None
        return {
            "alt_value": alt_value,
            "alt_uln": alt_uln,
            "alp_value": alp_value,
            "alp_uln": alp_uln,
        }

    # -------------------------------------------------------------------------
    def pick_best_entry(
        self,
        entries: list[ClinicalLabEntry],
        marker_names: set[str],
    ) -> ClinicalLabEntry | None:
        selected: ClinicalLabEntry | None = None
        for entry in entries:
            if entry.marker_name.upper() not in marker_names:
                continue
            if selected is None:
                selected = entry
                continue
            selected_value = self.parse_entry_value(selected)
            current_value = self.parse_entry_value(entry)
            if selected_value is None and current_value is not None:
                selected = entry
            elif (
                current_value is not None
                and selected_value is not None
                and current_value > selected_value
            ):
                selected = entry
        return selected

    # -------------------------------------------------------------------------
    def parse_entry_value(self, entry: ClinicalLabEntry) -> float | None:
        if entry.value is not None:
            return float(entry.value)
        return self.parse_marker_value(entry.value_text)

    # -------------------------------------------------------------------------
    def resolve_uln(self, entry: ClinicalLabEntry, *, fallback: float) -> float:
        if entry.upper_limit_normal is not None and entry.upper_limit_normal > 0:
            return float(entry.upper_limit_normal)
        parsed = self.parse_marker_value(entry.upper_limit_text)
        if parsed is not None and parsed > 0:
            return parsed
        return fallback

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
from services.clinical import hepatox_assessment, hepatox_prompts, hepatox_scoring
from services.clinical import hepatox_timeline

class HepatoxConsultation:
    def __init__(
        self,
        drugs: PatientDrugs,
        *,
        patient_name: str | None = None,
        timeout_s: float = server_settings.external_data.default_llm_timeout,
    ) -> None:
        self.drugs = drugs
        self.timeout_s = timeout_s
        self.llm_client = initialize_llm_client(purpose="clinical", timeout_s=timeout_s)
        self.MAX_EXCERPT_LENGTH = server_settings.external_data.max_excerpt_length
        self.research_service = BraveResearchService()
        self.patient_name = (patient_name or "").strip() or None
        provider, model_candidate = LLMRuntimeConfig.resolve_provider_and_model(
            "clinical"
        )
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
        self.rag_use_reranking = bool(server_settings.rag.use_reranking)
        self.rag_top_n = max(int(server_settings.rag.rerank_top_n), 1)
        self.rag_candidate_k = max(
            int(server_settings.rag.rerank_candidate_k), self.rag_top_n
        )
        self.pipeline_issues: list[PipelineIssue] = []
        default_parallel_analyses = 3 if provider == "ollama" else 1
        self.max_parallel_analyses = max(
            1,
            int(
                getattr(
                    server_settings.external_data,
                    "clinical_llm_max_concurrency",
                    default_parallel_analyses,
                )
            ),
        )
        default_retry_attempts = 1
        configured_retry_attempts = int(
            getattr(
                server_settings.external_data,
                "clinical_llm_retry_attempts",
                default_retry_attempts,
            )
        )
        # Keep consultation responsive when cloud providers are timing out.
        # One attempt is enough before falling back to deterministic outputs.
        self.analysis_retry_attempts = max(1, min(configured_retry_attempts, 1))

    # -------------------------------------------------------------------------
    async def run_analysis(
        self,
        *,
        prepared_inputs: HepatoxPreparedInputs | None,
        visit_date: date | None = None,
        report_language: str = "en",
        rag_query: dict[str, str] | None = None,
        use_web_search: bool = False,
        rucam_bundle: PatientRucamAssessmentBundle | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any] | None:
        return await hepatox_assessment.run_analysis(self, prepared_inputs=prepared_inputs, visit_date=visit_date, report_language=report_language, rag_query=rag_query, use_web_search=use_web_search, rucam_bundle=rucam_bundle, progress_callback=progress_callback)

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
        use_web_search: bool = False,
        rucam_bundle: PatientRucamAssessmentBundle | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> PatientDrugClinicalReport:
        return await hepatox_assessment.compile_clinical_assessment(self, resolved_drugs, clinical_context=clinical_context, visit_date=visit_date, report_language=report_language, pattern_prompt=pattern_prompt, rag_query=rag_query, use_web_search=use_web_search, rucam_bundle=rucam_bundle, progress_callback=progress_callback)

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
        return await hepatox_assessment.execute_indexed_job(index, coroutine)

    # -------------------------------------------------------------------------
    async def execute_bounded_job(
        self,
        index: int,
        coroutine: Any,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, Any]:
        return await hepatox_assessment.execute_bounded_job(self, index, coroutine, semaphore)

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
        use_web_search: bool,
        rucam_by_key: dict[str, DrugRucamAssessment],
    ) -> tuple[DrugClinicalAssessment, tuple[int, Any] | None]:
        return await hepatox_assessment.prepare_drug_assessment(self, idx=idx, drug_entry=drug_entry, resolved_drugs=resolved_drugs, visit_date=visit_date, report_language=report_language, normalized_context=normalized_context, pattern_summary=pattern_summary, rag_query=rag_query, use_web_search=use_web_search, rucam_by_key=rucam_by_key)

    # -------------------------------------------------------------------------
    def resolve_livertox_data_for_entry(
        self,
        *,
        raw_name: str,
        normalized_key: str,
        resolved_drugs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return hepatox_assessment.resolve_livertox_data_for_entry(self, raw_name=raw_name, normalized_key=normalized_key, resolved_drugs=resolved_drugs)

    # -------------------------------------------------------------------------
    @staticmethod
    def livertox_payload_rank(payload: dict[str, Any]) -> int:
        return hepatox_assessment.livertox_payload_rank(payload)

    # -------------------------------------------------------------------------
    async def fetch_rag_documents(
        self, rag_query: dict[str, str] | None, drug_name: str
    ) -> str | None:
        return await hepatox_assessment.fetch_rag_documents(self, rag_query, drug_name)

    # -------------------------------------------------------------------------
    def record_rag_retrieval_issue(self, *, drug_name: str, error: Exception) -> None:
        return hepatox_assessment.record_rag_retrieval_issue(self, drug_name=drug_name, error=error)

    # -------------------------------------------------------------------------
    @staticmethod
    def web_evidence_disabled_text() -> str:
        return hepatox_prompts.web_evidence_disabled_text()

    # -------------------------------------------------------------------------
    async def fetch_web_evidence_for_drug(self, *, drug_name: str) -> str:
        return await hepatox_assessment.fetch_web_evidence_for_drug(self, drug_name=drug_name)

    # -------------------------------------------------------------------------
    def ensure_similarity_search(self) -> bool:
        return hepatox_assessment.ensure_similarity_search(self)

    # -------------------------------------------------------------------------
    def select_excerpt(self, excerpts: list[str]) -> str | None:
        return hepatox_assessment.select_excerpt(self, excerpts)

    # -------------------------------------------------------------------------
    def search_supporting_documents(self, query_text: str | Any) -> str | None:
        return hepatox_assessment.search_supporting_documents(self, query_text)

    # -------------------------------------------------------------------------
    def format_similarity_fragment(
        self, index: int, record: dict[str, Any]
    ) -> str | None:
        return hepatox_prompts.format_similarity_fragment(self, index, record)

    # -------------------------------------------------------------------------
    @staticmethod
    def format_similarity_header(
        index: int,
        *,
        distance: Any,
        rerank_score: Any = None,
    ) -> str:
        return hepatox_prompts.format_similarity_header(index, distance=distance, rerank_score=rerank_score)

    # -------------------------------------------------------------------------
    def evaluate_suspension(
        self, entry: DrugEntry, visit_date: date | None
    ) -> DrugSuspensionContext:
        return hepatox_timeline.evaluate_suspension(self, entry, visit_date)

    # -------------------------------------------------------------------------
    def parse_timeline_date(
        self, raw_date: str | None, visit_date: date | None
    ) -> date | None:
        return hepatox_timeline.parse_timeline_date(self, raw_date, visit_date)

    # -------------------------------------------------------------------------
    def parse_suspension_date(
        self, raw_date: str | None, visit_date: date | None
    ) -> date | None:
        return hepatox_timeline.parse_suspension_date(self, raw_date, visit_date)

    # -------------------------------------------------------------------------
    def parse_start_date(
        self, raw_date: str | None, visit_date: date | None
    ) -> date | None:
        return hepatox_timeline.parse_start_date(self, raw_date, visit_date)

    # -------------------------------------------------------------------------
    def format_start_note(
        self,
        *,
        start_reported: bool,
        start_date: date | None,
        start_interval_days: int | None,
        visit_date: date | None,
    ) -> str:
        return hepatox_prompts.format_start_note(self, start_reported=start_reported, start_date=start_date, start_interval_days=start_interval_days, visit_date=visit_date)

    # -------------------------------------------------------------------------
    def humanize_interval(self, days: int) -> str:
        return hepatox_timeline.humanize_interval(self, days)

    # -------------------------------------------------------------------------
    @staticmethod
    def try_parse_date(value: str) -> date | None:
        return hepatox_timeline.try_parse_date(value)

    # -------------------------------------------------------------------------
    def format_suspension_prompt(self, suspension: DrugSuspensionContext) -> str:
        return hepatox_prompts.format_suspension_prompt(self, suspension)

    # -------------------------------------------------------------------------
    def format_start_prompt(self, suspension: DrugSuspensionContext) -> str:
        return hepatox_prompts.format_start_prompt(self, suspension)

    # -------------------------------------------------------------------------
    @staticmethod
    def format_visit_date_anchor(visit_date: date | None) -> str:
        return hepatox_prompts.format_visit_date_anchor(visit_date)

    # -------------------------------------------------------------------------
    def resolve_livertox_score(self, metadata: dict[str, Any] | None) -> str:
        return hepatox_timeline.resolve_livertox_score(self, metadata)

    # -------------------------------------------------------------------------
    def prepare_metadata_prompt(
        self, metadata: dict[str, Any] | None
    ) -> tuple[str, str]:
        return hepatox_prompts.prepare_metadata_prompt(self, metadata)

    # -------------------------------------------------------------------------
    def format_drug_heading(self, drug_name: str, score: str) -> str:
        return hepatox_prompts.format_drug_heading(self, drug_name, score)

    # -------------------------------------------------------------------------
    def summarize_rucam_components(
        self,
        rucam: DrugRucamAssessment | None,
    ) -> str:
        return hepatox_scoring.summarize_rucam_components(self, rucam)

    # -------------------------------------------------------------------------
    def format_rucam_limitations(self, rucam: DrugRucamAssessment | None) -> str:
        return hepatox_scoring.format_rucam_limitations(self, rucam)

    # -------------------------------------------------------------------------
    def format_rucam_prompt_block(self, rucam: DrugRucamAssessment | None) -> str:
        return hepatox_prompts.format_rucam_prompt_block(self, rucam)

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
        return await hepatox_assessment.repair_language_once(self, source_text=source_text, report_language=report_language)

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
        rucam: DrugRucamAssessment | None,
        knowledge_prompt: str = "No supplemental knowledge prompt available.",
        report_language: str = "en",
    ) -> str:
        return await hepatox_assessment.request_drug_analysis(self, drug_name=drug_name, canonical_name=canonical_name, origins=origins, extraction_metadata=extraction_metadata, livertox_status=livertox_status, excerpt=excerpt, rag_documents=rag_documents, clinical_context=clinical_context, suspension=suspension, visit_date=visit_date, pattern_summary=pattern_summary, metadata=metadata, web_evidence=web_evidence, rucam=rucam, knowledge_prompt=knowledge_prompt, report_language=report_language)

    # -------------------------------------------------------------------------
    @staticmethod
    def escape_braces(value: str) -> str:
        return hepatox_prompts.escape_braces(value)

    # -------------------------------------------------------------------------
    @staticmethod
    def coerce_chat_text(raw_response: Any) -> str:
        return hepatox_assessment.coerce_chat_text(raw_response)

    # -------------------------------------------------------------------------
    @staticmethod
    def extract_rate_limit_wait_hint_seconds(exc: Exception) -> float | None:
        return hepatox_assessment.extract_rate_limit_wait_hint_seconds(exc)

    # -------------------------------------------------------------------------
    def retry_backoff_seconds(
        self, attempt: int, *, exc: Exception | None = None
    ) -> float:
        return hepatox_assessment.retry_backoff_seconds(self, attempt, exc=exc)

    # -------------------------------------------------------------------------
    @staticmethod
    def remove_redundant_report_sentence(text: str) -> str:
        return hepatox_prompts.remove_redundant_report_sentence(text)

    # -------------------------------------------------------------------------
    async def finalize_patient_report(
        self,
        entries: list[DrugClinicalAssessment],
        *,
        clinical_context: str | None,
        report_language: str,
    ) -> str | None:
        return await hepatox_assessment.finalize_patient_report(self, entries, clinical_context=clinical_context, report_language=report_language)

    # -------------------------------------------------------------------------
    @staticmethod
    def should_render_as_matched_drug(entry: DrugClinicalAssessment) -> bool:
        return hepatox_assessment.should_render_as_matched_drug(entry)

    # -------------------------------------------------------------------------
    def render_matched_drug_section(
        self,
        entry: DrugClinicalAssessment,
        *,
        report_language: str = "en",
    ) -> str:
        return hepatox_prompts.render_matched_drug_section(self, entry, report_language=report_language)

    # -------------------------------------------------------------------------
    @staticmethod
    def render_evidence_quality_lines(entry: DrugClinicalAssessment) -> str:
        return hepatox_prompts.render_evidence_quality_lines(entry)

    # -------------------------------------------------------------------------
    def sanitize_renderable_body(self, entry: DrugClinicalAssessment) -> str:
        return hepatox_prompts.sanitize_renderable_body(self, entry)

    # -------------------------------------------------------------------------
    def build_fallback_technical_note(self, entry: DrugClinicalAssessment) -> str:
        return hepatox_prompts.build_fallback_technical_note(self, entry)

    # -------------------------------------------------------------------------
    def render_unresolved_mentions_section(
        self,
        entries: list[DrugClinicalAssessment],
        *,
        report_language: str = "en",
    ) -> str | None:
        return hepatox_prompts.render_unresolved_mentions_section(self, entries, report_language=report_language)

    # -------------------------------------------------------------------------
    def describe_unresolved_entry(
        self,
        entry: DrugClinicalAssessment,
        report_language: str = "en",
    ) -> str:
        return hepatox_prompts.describe_unresolved_entry(self, entry, report_language)

    # -------------------------------------------------------------------------
    async def generate_conclusion(
        self,
        *,
        clinical_context: str,
        multi_drug_report: str,
        report_language: str,
    ) -> str | None:
        return await hepatox_assessment.generate_conclusion(self, clinical_context=clinical_context, multi_drug_report=multi_drug_report, report_language=report_language)

    # -------------------------------------------------------------------------
    def build_excluded_paragraph(self, entry: DrugClinicalAssessment) -> str:
        return hepatox_prompts.build_excluded_paragraph(self, entry)

    # -------------------------------------------------------------------------
    def build_missing_excerpt_paragraph(
        self,
        entry: DrugClinicalAssessment,
        report_language: str = "en",
    ) -> str:
        return hepatox_prompts.build_missing_excerpt_paragraph(self, entry, report_language)

    # -------------------------------------------------------------------------
    def build_ambiguous_match_paragraph(
        self,
        entry: DrugClinicalAssessment,
        report_language: str = "en",
    ) -> str:
        return hepatox_prompts.build_ambiguous_match_paragraph(self, entry, report_language)

    # -------------------------------------------------------------------------
    def build_error_paragraph(
        self,
        entry: DrugClinicalAssessment,
        report_language: str = "en",
    ) -> str:
        return hepatox_prompts.build_error_paragraph(self, entry, report_language)

    def bibliography_source_label(self) -> str:
        return hepatox_assessment.bibliography_source_label(self)

