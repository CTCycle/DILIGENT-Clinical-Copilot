from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from common.utils.logger import logger
from services.llm.runtime_config import LLMRuntimeConfig
from configurations.startup import get_server_settings
from domain.clinical.entities import (
    PatientDrugs,
    PatientRucamAssessmentBundle,
    PipelineIssue,
)
from services.clinical.preparation import HepatoxPreparedInputs
from services.llm.provider_factory import initialize_llm_client
from services.retrieval.embeddings import SimilaritySearch
from services.retrieval.settings import build_effective_rag_settings
from services.clinical.analysis_runner import (
    AnalysisRunner,
    emit_progress,
    resolve_livertox_data_for_entry,
)
from services.clinical.drug_analysis import DrugAnalysisService
from services.clinical.rag_support import RagSupportService
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
        timeout_s: float | None = None,
    ) -> None:
        self.drugs = drugs
        if timeout_s is None:
            timeout_s = get_server_settings().runtime.clinical_llm_timeout
        self.timeout_s = timeout_s
        self.llm_client = initialize_llm_client(purpose="clinical", timeout_s=timeout_s)
        runtime_settings = get_server_settings().runtime
        self.MAX_EXCERPT_LENGTH = runtime_settings.max_excerpt_length
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
                    runtime_settings,
                    "clinical_llm_max_concurrency",
                    default_parallel_analyses,
                )
            ),
        )
        default_retry_attempts = 1
        configured_retry_attempts = int(
            getattr(
                runtime_settings,
                "clinical_llm_retry_attempts",
                default_retry_attempts,
            )
        )
        # Keep consultation responsive when cloud providers are timing out.
        # One attempt is enough before falling back to deterministic outputs.
        self.analysis_retry_attempts = max(1, min(configured_retry_attempts, 1))

        # Focused sub-services
        self.exposure_timeline = ExposureTimelineService()
        self.report_finalizer = ReportFinalizer()
        self.rag_support = RagSupportService(
            similarity_search=self.similarity_search,
            max_excerpt_length=self.MAX_EXCERPT_LENGTH,
            rag_candidate_k=self.rag_candidate_k,
            rag_top_n=self.rag_top_n,
            rag_use_reranking=self.rag_use_reranking,
            pipeline_issues=self.pipeline_issues,
        )
        self.drug_analysis = DrugAnalysisService(
            llm_client=self.llm_client,
            llm_model=self.llm_model,
            exposure_timeline=self.exposure_timeline,
            retry_attempts=self.analysis_retry_attempts,
        )
        self.analysis_runner = AnalysisRunner(
            drugs=self.drugs,
            exposure_timeline=self.exposure_timeline,
            drug_analysis=self.drug_analysis,
            rag_support=self.rag_support,
            report_finalizer=self.report_finalizer,
            max_parallel_analyses=self.max_parallel_analyses,
            pipeline_issues=self.pipeline_issues,
            resolve_livertox_data_for_entry=resolve_livertox_data_for_entry,
            emit_progress=emit_progress,
        )

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
        if prepared_inputs is None:
            logger.info(
                "No prepared inputs provided; skipping hepatotoxicity consultation"
            )
            return None
        if not prepared_inputs.resolved_drugs:
            logger.info("No matched drugs available for hepatotoxicity consultation")
            return None
        report = await self.analysis_runner.compile_clinical_assessment(
            prepared_inputs.resolved_drugs,
            clinical_context=prepared_inputs.clinical_context,
            visit_date=visit_date,
            report_language=report_language,
            pattern_prompt=prepared_inputs.pattern_prompt,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            progress_callback=progress_callback,
        )
        return report.model_dump()

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
        if prepared_inputs is None:
            logger.info(
                "No prepared inputs provided; skipping revision hepatotoxicity consultation"
            )
            return None
        if not prepared_inputs.resolved_drugs:
            logger.info(
                "No matched drugs available for revision hepatotoxicity consultation"
            )
            return None
        report = await self.analysis_runner.compile_revision_clinical_assessment(
            prepared_inputs.resolved_drugs,
            clinical_context=prepared_inputs.clinical_context,
            visit_date=visit_date,
            report_language=report_language,
            pattern_prompt=prepared_inputs.pattern_prompt,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            progress_callback=progress_callback,
        )
        payload = report.model_dump()
        payload["revision_consultation_metadata"] = {
            "drug_analysis_entrypoint": "request_revision_drug_analysis",
            "report_finalization_entrypoint": "finalize_report",
            "conclusion_entrypoint": "generate_revision_conclusion",
            "synthesis_mode": "revision_comparison_aware",
        }
        return payload
