from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date
from typing import Any

from services.llm.runtime_config import LLMRuntimeConfig
from configurations.startup import get_server_settings
from domain.clinical.entities import (
    DrugClinicalAssessment,
    DrugEntry,
    DrugRucamAssessment,
    PatientDrugClinicalReport,
    PatientDrugs,
    PatientRucamAssessmentBundle,
    PipelineIssue,
)
from services.clinical.preparation import HepatoxPreparedInputs
from services.llm.provider_factory import initialize_llm_client
from services.retrieval.embeddings import SimilaritySearch
from services.retrieval.settings import build_effective_rag_settings
from services.text.normalization import normalize_drug_query_name
from services.clinical.analysis_runner import AnalysisRunner
from services.clinical.drug_analysis import DrugAnalysisService
from services.clinical.rag_support import RagRetrievalBundle, RagSupportService
from services.clinical.report_finalizer import ReportFinalizer
from services.clinical.exposure_timeline import ExposureTimelineService

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
        self.exposure_timeline = ExposureTimelineService()
        self.analysis_runner = AnalysisRunner(self)
        self.drug_analysis = DrugAnalysisService(self)
        self.report_finalizer = ReportFinalizer()
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
