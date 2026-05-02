from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any
from collections.abc import Callable

from pydantic import ValidationError

from common.exceptions import (
    ServiceConflictError,
    ServiceError,
    ServiceNotFoundError,
    ServiceValidationError,
)
from domain.clinical.extras import HepatoxPreparedInputs
from services.llm.cloud import LLMError

from domain.clinical.entities import (
    ClinicalSectionExtractionResult,
    ClinicalPipelineValidationError,
    ClinicalSessionRequest,
    DrugRucamAssessment,
    HepatotoxicityPatternAssessment,
    LiverInjuryOnsetContext,
    PatientLabTimeline,
    PatientData,
    PatientDiseaseContext,
    PatientDrugs,
    PatientRucamAssessmentBundle,
    PipelineIssue,
)
from domain.clinical.validation import ValidationMessageBundle
from domain.jobs import (
    JobCancelResponse,
    JobStartResponse,
    JobStatusResponse,
)
from configurations.startup import server_settings
from configurations.llm_configs import LLMRuntimeConfig
from common.utils.languages import (
    MISSING_VISIT_LABEL_BY_LANGUAGE,
    resolve_supported_language_code,
)
from repositories.serialization.data import DataSerializer
from repositories.serialization.model_configs import (
    ModelConfigSerializer,
)
from services.runtime.jobs import (
    JobManager,
    job_manager as default_job_manager,
)
from common.utils.logger import logger
from services.clinical.hepatox_core import (
    HepatotoxicityPatternAnalyzer,
    HepatoxConsultation,
)
from services.clinical.job_progress import (
    ClinicalConsultationProgressCallback,
    StageProgressFractionCallback,
)
from services.clinical.language import ClinicalLanguageDetector
from services.clinical.match_quality import classify_match_evidence
from services.clinical.preparation import ClinicalKnowledgePreparation
from services.clinical.candidate_selection import (
    select_relevant_candidates,
)
from services.clinical.disease import DiseaseExtractor
from services.clinical.labs import ClinicalLabExtractor
from services.clinical.parser import DrugsParser
from services.clinical.rucam import RucamScoreEstimator
from services.clinical.validation import (
    build_validation_bundle,
    ensure_required_sections,
    ensure_timed_therapy_drug,
    has_timing_information,
)
from services.session.payload import PayloadSanitizationService
from services.session.clinical_input_extractor import (
    ClinicalInputExtractionError,
    ClinicalInputExtractor,
)
from services.llm.model_config import ModelConfigService
from services.retrieval.query import DILIQueryBuilder
from services.session.session_shared import (
    NarrativeBuilder,
    run_clinical_job,
)
from services.session.formatting_mixin import (
    ClinicalSessionFormattingMixin,
)
from services.text.normalization import normalize_drug_query_name

drugs_parser = DrugsParser(timeout_s=server_settings.external_data.default_llm_timeout)
disease_extractor = DiseaseExtractor(
    timeout_s=server_settings.external_data.default_llm_timeout
)
lab_extractor = ClinicalLabExtractor(
    timeout_s=server_settings.external_data.default_llm_timeout
)
pattern_analyzer = HepatotoxicityPatternAnalyzer()
rucam_estimator = RucamScoreEstimator()
serializer = DataSerializer()
payload_sanitization_service = PayloadSanitizationService()


###############################################################################
class ClinicalSessionService(ClinicalSessionFormattingMixin):
    JOB_TYPE = "clinical"

    def __init__(
        self,
        *,
        drugs_parser: DrugsParser,
        disease_extractor: DiseaseExtractor,
        lab_extractor: ClinicalLabExtractor,
        pattern_analyzer: HepatotoxicityPatternAnalyzer,
        rucam_estimator: RucamScoreEstimator,
        serializer: DataSerializer,
        payload_sanitizer: PayloadSanitizationService,
        input_preparator: ClinicalKnowledgePreparation | None = None,
        clinical_input_extractor: ClinicalInputExtractor | None = None,
        hepatox_consultation_cls: type[HepatoxConsultation] | None = None,
        job_manager: JobManager | None = None,
    ) -> None:
        self.drugs_parser = drugs_parser
        self.disease_extractor = disease_extractor
        self.lab_extractor = lab_extractor
        self.pattern_analyzer = pattern_analyzer
        self.rucam_estimator = rucam_estimator
        self.serializer = serializer
        self.payload_sanitizer = payload_sanitizer
        self.input_preparator = input_preparator or ClinicalKnowledgePreparation()
        self.clinical_input_extractor = clinical_input_extractor or ClinicalInputExtractor()
        self.hepatox_consultation_cls = hepatox_consultation_cls or HepatoxConsultation
        self.job_manager = job_manager or default_job_manager
        self.model_config_service = ModelConfigService(
            serializer=ModelConfigSerializer()
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def emit_progress(
        progress_callback: Callable[[str, float], None] | None,
        *,
        stage: str,
        value: float,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(stage, value)

    # -------------------------------------------------------------------------
    @staticmethod
    def build_stage_progress_callback(
        progress_callback: Callable[[str, float], None] | None,
        *,
        stage: str,
        start_value: float,
        end_value: float,
    ) -> Callable[[float], None] | None:
        if progress_callback is None:
            return None
        return StageProgressFractionCallback(
            progress_callback=progress_callback,
            stage=stage,
            start_value=start_value,
            end_value=end_value,
        )

    def apply_persisted_runtime_configuration(self) -> None:
        self.model_config_service.ensure_defaults()
        parser_provider, parser_model = LLMRuntimeConfig.resolve_provider_and_model(
            "parser"
        )
        clinical_provider, clinical_model_resolved = (
            LLMRuntimeConfig.resolve_provider_and_model("clinical")
        )
        logger.info(
            "Resolved LLM runtime from persisted model config: cloud=%s provider=%s cloud_model=%s text_extraction_provider=%s text_extraction_model=%s clinical_provider=%s clinical_model=%s ollama_temperature=%.2f cloud_temperature=%.2f reasoning=%s",
            LLMRuntimeConfig.is_cloud_enabled(),
            LLMRuntimeConfig.get_llm_provider(),
            LLMRuntimeConfig.get_cloud_model(),
            parser_provider,
            parser_model,
            clinical_provider,
            clinical_model_resolved,
            LLMRuntimeConfig.get_ollama_temperature(),
            LLMRuntimeConfig.get_cloud_temperature(),
            LLMRuntimeConfig.is_ollama_reasoning_enabled(),
        )

    # -------------------------------------------------------------------------
    async def preprocess_unified_input(
        self, request_payload: ClinicalSessionRequest
    ) -> tuple[ClinicalSessionRequest, ClinicalSectionExtractionResult | None]:
        clinical_input = (request_payload.clinical_input or "").strip()
        if not clinical_input:
            raise ClinicalInputExtractionError(
                "Clinical input is required."
            )
        try:
            extraction = await self.clinical_input_extractor.extract(
                clinical_input=clinical_input
            )
        except ClinicalInputExtractionError as exc:
            raise ServiceValidationError(str(exc)) from exc

        if (
            not extraction.anamnesis
            or not extraction.drugs
            or not extraction.laboratory_analysis
        ):
            raise ServiceValidationError(
                "Clinical input must contain anamnesis, current therapy, and laboratory analysis sections."
            )
        return (
            request_payload.model_copy(
                update={
                    "anamnesis": extraction.anamnesis,
                    "drugs": extraction.drugs,
                    "laboratory_analysis": extraction.laboratory_analysis,
                }
            ),
            extraction,
        )

    # -------------------------------------------------------------------------
    def build_patient_payload(
        self,
        request_payload: ClinicalSessionRequest,
    ) -> PatientData:
        try:
            payload_data = self.payload_sanitizer.sanitize_dili_payload(
                patient_name=request_payload.name,
                visit_date=request_payload.visit_date,
                anamnesis=request_payload.anamnesis,
                drugs=request_payload.drugs,
                laboratory_analysis=request_payload.laboratory_analysis,
                use_rag=request_payload.use_rag,
                use_web_search=request_payload.use_web_search,
            )
            return PatientData.model_validate(payload_data)
        except ValidationError as exc:
            raise ServiceValidationError(
                self.serialize_validation_errors(exc.errors()),
            ) from exc

    # -------------------------------------------------------------------------
    def build_validation_bundle_for_payload(
        self, payload: PatientData
    ) -> ValidationMessageBundle:
        language_result = ClinicalLanguageDetector.detect(payload)
        return build_validation_bundle(language_result.report_language)

    # -------------------------------------------------------------------------
    def ensure_submission_requirements(self, payload: PatientData) -> None:
        validation_bundle = self.build_validation_bundle_for_payload(payload)
        ensure_required_sections(payload, bundle=validation_bundle)

        cleaned_therapy_text = self.drugs_parser.clean_text(payload.drugs or "")
        if not cleaned_therapy_text:
            ensure_timed_therapy_drug(
                self.build_fallback_therapy_drugs(payload.drugs),
                bundle=validation_bundle,
            )
            return

        lines = [
            segment.strip()
            for segment in cleaned_therapy_text.split("\n")
            if segment.strip()
        ]
        parsed_entries = [
            parsed
            for parsed in (self.drugs_parser.parse_line(line) for line in lines)
            if parsed is not None
        ]
        if any(has_timing_information(entry) for entry in parsed_entries):
            return
        ensure_timed_therapy_drug(
            self.build_fallback_therapy_drugs(payload.drugs),
            bundle=validation_bundle,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def run_stop_check(stop_check: Callable[[], None] | None) -> None:
        if stop_check is not None:
            stop_check()

    # -------------------------------------------------------------------------
    @staticmethod
    def append_warning_issue(
        issues: list[PipelineIssue],
        *,
        code: str,
        message: str,
        field: str | None = None,
    ) -> None:
        issues.append(
            PipelineIssue(
                severity="warning",
                code=code,
                message=message,
                field=field,
            )
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def append_knowledge_base_unavailable_issue(
        issues: list[PipelineIssue],
    ) -> None:
        ClinicalSessionService.append_warning_issue(
            issues,
            code="knowledge_base_unavailable",
            message=(
                "Local RxNav/LiverTox knowledge base is unavailable or empty; "
                "drug matching and evidence-backed consultation were skipped. "
                "Rebuild or fetch the RxNav and LiverTox datasets before relying "
                "on this report."
            ),
            field="knowledge_base",
        )

    # -------------------------------------------------------------------------
    async def extract_therapy_drugs(
        self,
        *,
        cleaned_therapy_text: str,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> PatientDrugs:
        self.emit_progress(progress_callback, stage="therapy_extraction", value=22.0)
        therapy_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="therapy_extraction",
            start_value=22.0,
            end_value=30.0,
        )
        start_time = time.perf_counter()
        try:
            therapy_drugs = await self.drugs_parser.extract_drugs_from_therapy(
                cleaned_therapy_text,
                progress_callback=therapy_progress_callback,
            )
            self.run_stop_check(stop_check)
            elapsed = time.perf_counter() - start_time
            logger.info("Therapy drugs extraction required %.4f seconds", elapsed)
            logger.info(
                "Detected %s drugs from therapy list", len(therapy_drugs.entries)
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.warning(
                (
                    "Therapy drugs extraction failed after %.4f seconds; "
                    "falling back to line-based parsing: %s"
                ),
                elapsed,
                exc,
            )
            self.append_warning_issue(
                issues,
                code="therapy_extraction_fallback",
                message=(
                    "Therapy extraction via LLM was unavailable; "
                    "the analysis continued using the raw therapy list."
                ),
                field="drugs",
            )
            therapy_drugs = self.build_fallback_therapy_drugs(cleaned_therapy_text)
        self.emit_progress(progress_callback, stage="therapy_extraction", value=30.0)
        self.run_stop_check(stop_check)
        return therapy_drugs

    # -------------------------------------------------------------------------
    async def extract_anamnesis_drugs(
        self,
        *,
        anamnesis_text: str,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> PatientDrugs:
        self.emit_progress(progress_callback, stage="anamnesis_extraction", value=30.0)
        anamnesis_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="anamnesis_extraction",
            start_value=30.0,
            end_value=42.0,
        )
        start_time = time.perf_counter()
        try:
            anamnesis_drugs = await self.drugs_parser.extract_drugs_from_anamnesis(
                anamnesis_text,
                progress_callback=anamnesis_progress_callback,
            )
            self.run_stop_check(stop_check)
            elapsed = time.perf_counter() - start_time
            logger.info("Anamnesis drugs extraction required %.4f seconds", elapsed)
            logger.info(
                "Detected %s drugs from anamnesis", len(anamnesis_drugs.entries)
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.warning(
                (
                    "Anamnesis drugs extraction failed after %.4f seconds; "
                    "continuing without historical drug mentions: %s"
                ),
                elapsed,
                exc,
            )
            self.append_warning_issue(
                issues,
                code="anamnesis_extraction_failed",
                message=(
                    "Drug extraction from anamnesis was unavailable; "
                    "the analysis continued without historical drug mentions."
                ),
                field="anamnesis",
            )
            anamnesis_drugs = PatientDrugs(entries=[])
        self.emit_progress(progress_callback, stage="anamnesis_extraction", value=42.0)
        self.run_stop_check(stop_check)
        return anamnesis_drugs

    # -------------------------------------------------------------------------
    async def extract_disease_context(
        self,
        *,
        anamnesis_text: str,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> PatientDiseaseContext:
        self.emit_progress(
            progress_callback, stage="anamnesis_disease_extraction", value=42.0
        )
        disease_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="anamnesis_disease_extraction",
            start_value=42.0,
            end_value=48.0,
        )
        start_time = time.perf_counter()
        max_attempts = 2
        backoff_seconds = 1.5
        timeout_s = max(float(self.disease_extractor.timeout_s), 1.0)
        for attempt in range(1, max_attempts + 1):
            try:
                disease_context = await asyncio.wait_for(
                    self.disease_extractor.extract_diseases_from_anamnesis(
                        anamnesis_text,
                        progress_callback=disease_progress_callback,
                    ),
                    timeout=timeout_s,
                )
                self.run_stop_check(stop_check)
                elapsed = time.perf_counter() - start_time
                logger.info(
                    "Anamnesis disease extraction required %.4f seconds", elapsed
                )
                logger.info(
                    "Detected %s diseases from anamnesis", len(disease_context.entries)
                )
                self.emit_progress(
                    progress_callback, stage="anamnesis_disease_extraction", value=48.0
                )
                self.run_stop_check(stop_check)
                return disease_context
            except TimeoutError as exc:
                elapsed = time.perf_counter() - start_time
                if attempt < max_attempts:
                    logger.warning(
                        (
                            "Anamnesis disease extraction timed out after %.4fs "
                            "(attempt %d/%d). Retrying in %.1fs."
                        ),
                        elapsed,
                        attempt,
                        max_attempts,
                        backoff_seconds,
                    )
                    await asyncio.sleep(backoff_seconds)
                    self.run_stop_check(stop_check)
                    continue
                raise RuntimeError(
                    "Disease extraction timed out while waiting for Ollama. "
                    "Please verify Ollama responsiveness and model readiness."
                ) from exc
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                logger.warning(
                    (
                        "Anamnesis disease extraction failed after %.4f seconds; "
                        "continuing without structured disease timeline: %s"
                    ),
                    elapsed,
                    exc,
                )
                self.append_warning_issue(
                    issues,
                    code="anamnesis_disease_extraction_failed",
                    message=(
                        "Disease extraction from anamnesis was unavailable; "
                        "the analysis continued without structured disease timeline."
                    ),
                    field="anamnesis",
                )
                disease_context = PatientDiseaseContext(entries=[])
                self.emit_progress(
                    progress_callback, stage="anamnesis_disease_extraction", value=48.0
                )
                self.run_stop_check(stop_check)
                return disease_context
        return PatientDiseaseContext(entries=[])

    # -------------------------------------------------------------------------
    async def extract_lab_timeline(
        self,
        *,
        payload: PatientData,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> tuple[PatientLabTimeline, LiverInjuryOnsetContext | None]:
        self.emit_progress(
            progress_callback, stage="anamnesis_lab_extraction", value=48.0
        )
        lab_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="anamnesis_lab_extraction",
            start_value=48.0,
            end_value=52.0,
        )
        start_time = time.perf_counter()
        try:
            lab_timeline, onset_context = await self.lab_extractor.extract_from_payload(
                payload,
                progress_callback=lab_progress_callback,
            )
            self.run_stop_check(stop_check)
            elapsed = time.perf_counter() - start_time
            logger.info("Anamnesis lab extraction required %.4f seconds", elapsed)
            logger.info("Detected %s timeline lab entries", len(lab_timeline.entries))
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.warning(
                (
                    "Anamnesis lab extraction failed after %.4f seconds; "
                    "continuing without structured lab timeline: %s"
                ),
                elapsed,
                exc,
            )
            self.append_warning_issue(
                issues,
                code="anamnesis_lab_extraction_failed",
                message=(
                    "Longitudinal lab extraction from anamnesis was unavailable; "
                    "the analysis continued without timeline enrichment."
                ),
                field="anamnesis",
            )
            lab_timeline = PatientLabTimeline(entries=[])
            onset_context = None
        self.emit_progress(
            progress_callback, stage="anamnesis_lab_extraction", value=52.0
        )
        self.run_stop_check(stop_check)
        return lab_timeline, onset_context

    # -------------------------------------------------------------------------
    def assess_pattern(
        self,
        *,
        lab_timeline: PatientLabTimeline,
        validation_bundle: Any,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> HepatotoxicityPatternAssessment:
        try:
            pattern_assessment = self.pattern_analyzer.assess_payload(lab_timeline)
        except ClinicalPipelineValidationError as exc:
            localized = [
                PipelineIssue(
                    severity=item.severity,
                    code=item.code,
                    message=(
                        validation_bundle.insufficient_labs
                        if item.code == "missing_hepatotoxicity_inputs"
                        else item.message
                    ),
                    field=item.field,
                    line_index=item.line_index,
                    raw_line=item.raw_line,
                )
                for item in exc.issues
            ]
            raise ClinicalPipelineValidationError(
                issues=localized,
                message=localized[0].message if localized else exc.args[0],
            ) from exc
        issues.extend(pattern_assessment.issues)
        pattern_score = pattern_assessment.score
        logger.info(
            "Patient hepatotoxicity pattern classified as %s (R=%.3f, status=%s)",
            pattern_score.classification,
            pattern_score.r_score
            if pattern_score.r_score is not None
            else float("nan"),
            pattern_assessment.status,
        )
        self.emit_progress(
            progress_callback, stage="hepatotoxicity_pattern", value=54.0
        )
        self.run_stop_check(stop_check)
        return pattern_assessment

    # -------------------------------------------------------------------------
    def estimate_rucam(
        self,
        *,
        payload: PatientData,
        analysis_drugs: PatientDrugs,
        anamnesis_drugs: PatientDrugs,
        disease_context: PatientDiseaseContext,
        lab_timeline: PatientLabTimeline,
        onset_context: LiverInjuryOnsetContext | None,
        pattern_score,
        report_language: str,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> PatientRucamAssessmentBundle:
        self.emit_progress(progress_callback, stage="rucam_estimation", value=52.0)
        start_time = time.perf_counter()
        try:
            rucam_bundle = self.rucam_estimator.estimate(
                payload=payload,
                analysis_drugs=analysis_drugs,
                anamnesis_drugs=anamnesis_drugs,
                disease_context=disease_context,
                lab_timeline=lab_timeline,
                onset_context=onset_context,
                pattern_score=pattern_score,
                resolved_drugs=None,
                report_language=report_language,
            )
            elapsed = time.perf_counter() - start_time
            logger.info("RUCAM estimation required %.4f seconds", elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.warning(
                "RUCAM estimation failed after %.4f seconds; continuing without RUCAM: %s",
                elapsed,
                exc,
            )
            self.append_warning_issue(
                issues,
                code="rucam_estimation_failed",
                message=(
                    "RUCAM estimation was unavailable; the analysis continued without "
                    "per-drug estimated RUCAM."
                ),
            )
            rucam_bundle = PatientRucamAssessmentBundle(entries=[])
        self.emit_progress(progress_callback, stage="rucam_estimation", value=54.0)
        self.run_stop_check(stop_check)
        return rucam_bundle

    # -------------------------------------------------------------------------
    @staticmethod
    def build_rag_query(
        *,
        payload: PatientData,
        analysis_drugs: PatientDrugs,
        structured_context: str,
        pattern_score,
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> dict[str, str] | None:
        ClinicalSessionService.emit_progress(
            progress_callback, stage="rag_query_building", value=54.0
        )
        rag_query: dict[str, str] | None = None
        if payload.use_rag:
            query_builder = DILIQueryBuilder(analysis_drugs)
            logger.info("RAG retrieval enabled for clinical consultation")
            rag_query = query_builder.build_dili_queries(
                clinical_context=structured_context,
                pattern_classification=pattern_score.classification,
                r_score=pattern_score.r_score,
            )
        ClinicalSessionService.emit_progress(
            progress_callback, stage="rag_query_building", value=56.0
        )
        ClinicalSessionService.run_stop_check(stop_check)
        return rag_query

    # -------------------------------------------------------------------------
    async def run_livertox_lookup(
        self,
        *,
        all_detected_drugs: PatientDrugs,
        structured_context: str,
        pattern_score,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> HepatoxPreparedInputs | None:
        self.emit_progress(progress_callback, stage="livertox_lookup", value=56.0)
        livertox_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="livertox_lookup",
            start_value=56.0,
            end_value=62.0,
        )
        prepared_inputs = await self.input_preparator.prepare_inputs(
            all_detected_drugs,
            clinical_context=structured_context,
            pattern_score=pattern_score,
            progress_callback=livertox_progress_callback,
        )
        self.run_stop_check(stop_check)
        if prepared_inputs is None and all_detected_drugs.entries:
            self.append_knowledge_base_unavailable_issue(issues)
        self.emit_progress(progress_callback, stage="livertox_lookup", value=62.0)
        return prepared_inputs

    # -------------------------------------------------------------------------
    def reestimate_rucam_with_livertox(
        self,
        *,
        payload: PatientData,
        analysis_drugs: PatientDrugs,
        anamnesis_drugs: PatientDrugs,
        disease_context: PatientDiseaseContext,
        lab_timeline: PatientLabTimeline,
        onset_context: LiverInjuryOnsetContext | None,
        pattern_score,
        report_language: str,
        prepared_inputs,
        rucam_bundle: PatientRucamAssessmentBundle,
        issues: list[PipelineIssue],
    ) -> PatientRucamAssessmentBundle:
        try:
            return self.rucam_estimator.estimate(
                payload=payload,
                analysis_drugs=analysis_drugs,
                anamnesis_drugs=anamnesis_drugs,
                disease_context=disease_context,
                lab_timeline=lab_timeline,
                onset_context=onset_context,
                pattern_score=pattern_score,
                resolved_drugs=prepared_inputs.resolved_drugs
                if prepared_inputs
                else None,
                report_language=report_language,
            )
        except Exception as exc:
            logger.warning("RUCAM re-estimation with LiverTox metadata failed: %s", exc)
            self.append_warning_issue(
                issues,
                code="rucam_reestimate_failed",
                message=(
                    "RUCAM refinement with matched LiverTox metadata failed; "
                    "using preliminary estimates."
                ),
            )
            return rucam_bundle

    # -------------------------------------------------------------------------
    async def run_consultation(
        self,
        *,
        payload: PatientData,
        analysis_drugs: PatientDrugs,
        prepared_inputs,
        report_language: str,
        rag_query: dict[str, str] | None,
        rucam_bundle: PatientRucamAssessmentBundle,
        issues: list[PipelineIssue],
        progress_callback: Callable[[str, float], None] | None,
        stop_check: Callable[[], None] | None,
    ) -> tuple[HepatoxConsultation, str | None]:
        clinical_session = self.hepatox_consultation_cls(
            analysis_drugs,
            patient_name=payload.name,
        )
        final_report: str | None = None
        start_time = time.perf_counter()
        try:
            consultation_progress_callback = ClinicalConsultationProgressCallback(
                progress_callback=progress_callback,
            )
            drug_assessment = await clinical_session.run_analysis(
                prepared_inputs=prepared_inputs,
                visit_date=payload.visit_date,
                report_language=report_language,
                rag_query=rag_query,
                use_web_search=payload.use_web_search,
                rucam_bundle=rucam_bundle,
                progress_callback=consultation_progress_callback,
            )
            self.run_stop_check(stop_check)
            elapsed = time.perf_counter() - start_time
            logger.info("Hepato-toxicity consultation required %.4f seconds", elapsed)
            if isinstance(drug_assessment, dict):
                raw_final_report = drug_assessment.get("final_report")
                if isinstance(raw_final_report, str):
                    final_report = raw_final_report.strip()
                elif raw_final_report is None:
                    final_report = None
                else:
                    final_report = str(raw_final_report).strip()
            issues.extend(getattr(clinical_session, "pipeline_issues", []))
        except LLMError as exc:
            self.append_warning_issue(
                issues,
                code="clinical_llm_unavailable",
                message=(
                    "Clinical LLM analysis is unavailable; report generated without "
                    "per-drug synthesis."
                ),
            )
            logger.warning(
                "Clinical LLM unavailable for patient '%s': %s",
                payload.name or "unknown",
                exc,
            )
        return clinical_session, final_report

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalized_resolved_drug_map(prepared_inputs) -> dict[str, dict[str, Any]]:
        if prepared_inputs is None:
            return {}
        resolved_drug_map: dict[str, dict[str, Any]] = {}
        for key, value in prepared_inputs.resolved_drugs.items():
            normalized_key = normalize_drug_query_name(key)
            if normalized_key and isinstance(value, dict):
                resolved_drug_map[normalized_key] = value
        return resolved_drug_map

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalized_rucam_map(
        rucam_bundle: PatientRucamAssessmentBundle,
    ) -> dict[str, DrugRucamAssessment]:
        rucam_by_name: dict[str, DrugRucamAssessment] = {}
        for item in rucam_bundle.entries:
            normalized_key = normalize_drug_query_name(item.drug_name)
            if normalized_key:
                rucam_by_name[normalized_key] = item
        return rucam_by_name

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_single_matched_drug_row(
        *,
        detected_name: str,
        resolved: dict[str, Any],
        rucam_entry: DrugRucamAssessment | None,
    ) -> dict[str, Any]:
        if not resolved:
            resolved = {
                "match_status": "missing_match",
                "match_reason": "knowledge_base_unavailable",
                "match_notes": [
                    "No local RxNav/LiverTox evidence record was available."
                ],
                "missing_livertox": True,
                "ambiguous_match": False,
            }
        matched_row = resolved.get("matched_livertox_row")
        if not isinstance(matched_row, dict):
            matched_row = {}
        match_notes = resolved.get("match_notes", [])
        if not isinstance(match_notes, list):
            match_notes = []
        match_confidence = resolved.get("match_confidence")
        if match_confidence is not None:
            try:
                match_confidence = float(match_confidence)
            except (TypeError, ValueError):
                match_confidence = None
        match_quality = classify_match_evidence(
            match_status=resolved.get("match_status"),
            match_reason=resolved.get("match_reason"),
            match_confidence=match_confidence,
            match_notes=match_notes,
            missing_livertox=bool(resolved.get("missing_livertox", True)),
            ambiguous_match=bool(resolved.get("ambiguous_match", False)),
        )
        return {
            "raw_drug_name": detected_name,
            "matched_drug_name": matched_row.get("drug_name"),
            "nbk_id": matched_row.get("nbk_id"),
            "match_confidence": match_confidence,
            "match_reason": resolved.get("match_reason"),
            "match_notes": match_notes,
            "match_status": resolved.get("match_status"),
            "evidence_quality": match_quality["evidence_quality"],
            "evidence_warnings": match_quality["evidence_warnings"],
            "match_candidates": resolved.get("match_candidates", []),
            "chosen_candidate": resolved.get("chosen_candidate"),
            "rejected_candidates": resolved.get("rejected_candidates", []),
            "missing_livertox": resolved.get("missing_livertox", True),
            "ambiguous_match": resolved.get("ambiguous_match", False),
            "regimen_group_ids": resolved.get("regimen_group_ids", []),
            "regimen_components": resolved.get("regimen_components", []),
            "origins": resolved.get("origins", []),
            "raw_mentions": resolved.get("raw_mentions", []),
            "rucam": rucam_entry.model_dump() if rucam_entry is not None else None,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def build_matched_drugs_payload(
        *,
        detected_drugs: list[str],
        prepared_inputs,
        rucam_bundle: PatientRucamAssessmentBundle,
    ) -> list[dict[str, Any]]:
        resolved_drug_map = ClinicalSessionService._normalized_resolved_drug_map(
            prepared_inputs
        )
        rucam_by_name = ClinicalSessionService._normalized_rucam_map(rucam_bundle)

        matched_drugs_payload: list[dict[str, Any]] = []
        for detected_name in detected_drugs:
            normalized_name = normalize_drug_query_name(detected_name)
            resolved = resolved_drug_map.get(normalized_name, {})
            rucam_entry = rucam_by_name.get(normalized_name)
            matched_drugs_payload.append(
                ClinicalSessionService._build_single_matched_drug_row(
                    detected_name=detected_name,
                    resolved=resolved,
                    rucam_entry=rucam_entry,
                )
            )
        return matched_drugs_payload

    # -------------------------------------------------------------------------
    async def process_single_patient(
        self,
        payload: PatientData,
        *,
        patient_image_base64: str | None = None,
        section_extraction: ClinicalSectionExtractionResult | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
        stop_check: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        self.run_stop_check(stop_check)
        logger.info(
            "Starting Drug-Induced Liver Injury (DILI) analysis for patient: %s",
            payload.name,
        )

        global_start_time = time.perf_counter()
        self.emit_progress(progress_callback, stage="session_initialization", value=5.0)
        language_result = ClinicalLanguageDetector.detect(payload)
        report_language = language_result.report_language
        validation_bundle = build_validation_bundle(report_language)
        ensure_required_sections(payload, bundle=validation_bundle)
        self.run_stop_check(stop_check)

        issues: list[PipelineIssue] = []
        cleaned_therapy_text = self.drugs_parser.clean_text(payload.drugs or "")
        therapy_drugs = await self.extract_therapy_drugs(
            cleaned_therapy_text=cleaned_therapy_text,
            issues=issues,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )
        ensure_timed_therapy_drug(therapy_drugs, bundle=validation_bundle)
        anamnesis_drugs = await self.extract_anamnesis_drugs(
            anamnesis_text=payload.anamnesis,
            issues=issues,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )
        disease_context = await self.extract_disease_context(
            anamnesis_text=payload.anamnesis,
            issues=issues,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )
        lab_timeline, onset_context = await self.extract_lab_timeline(
            payload=payload,
            issues=issues,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )
        pattern_assessment = self.assess_pattern(
            lab_timeline=lab_timeline,
            validation_bundle=validation_bundle,
            issues=issues,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )
        pattern_score = pattern_assessment.score
        all_detected_drugs = PatientDrugs(
            entries=[*therapy_drugs.entries, *anamnesis_drugs.entries]
        )
        candidate_selection = select_relevant_candidates(
            therapy_drugs=therapy_drugs,
            anamnesis_drugs=anamnesis_drugs,
            visit_date=payload.visit_date,
        )
        analysis_drugs = candidate_selection.ordered_analysis_drugs
        logger.info(
            "Using %s deduplicated drugs for matching/consultation",
            len(analysis_drugs.entries),
        )
        rucam_bundle = self.estimate_rucam(
            payload=payload,
            analysis_drugs=analysis_drugs,
            anamnesis_drugs=anamnesis_drugs,
            disease_context=disease_context,
            lab_timeline=lab_timeline,
            onset_context=onset_context,
            pattern_score=pattern_score,
            report_language=report_language,
            issues=issues,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )
        structured_context = self.build_structured_clinical_context(
            payload,
            therapy_drugs=therapy_drugs,
            anamnesis_drugs=anamnesis_drugs,
            disease_context=disease_context,
            lab_timeline=lab_timeline,
            onset_context=onset_context,
            pattern_score=pattern_score,
        )
        rag_query = self.build_rag_query(
            payload=payload,
            analysis_drugs=analysis_drugs,
            structured_context=structured_context,
            pattern_score=pattern_score,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )
        prepared_inputs = await self.run_livertox_lookup(
            all_detected_drugs=all_detected_drugs,
            structured_context=structured_context,
            pattern_score=pattern_score,
            issues=issues,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )
        rucam_bundle = self.reestimate_rucam_with_livertox(
            payload=payload,
            analysis_drugs=analysis_drugs,
            anamnesis_drugs=anamnesis_drugs,
            disease_context=disease_context,
            lab_timeline=lab_timeline,
            onset_context=onset_context,
            pattern_score=pattern_score,
            report_language=report_language,
            prepared_inputs=prepared_inputs,
            rucam_bundle=rucam_bundle,
            issues=issues,
        )
        clinical_session, final_report = await self.run_consultation(
            payload=payload,
            analysis_drugs=analysis_drugs,
            prepared_inputs=prepared_inputs,
            report_language=report_language,
            rag_query=rag_query,
            rucam_bundle=rucam_bundle,
            issues=issues,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )

        patient_label = payload.name or "Unknown patient"
        report_language_key = resolve_supported_language_code(report_language)
        visit_label = (
            payload.visit_date.strftime("%d %B %Y")
            if payload.visit_date
            else MISSING_VISIT_LABEL_BY_LANGUAGE.get(
                report_language_key, "Not provided"
            )
        )
        global_elapsed = time.perf_counter() - global_start_time
        logger.info(
            "Total time for Drug Induced Liver Injury (DILI) assessment is %.4f seconds",
            global_elapsed,
        )
        detected_drugs = [entry.name for entry in analysis_drugs.entries if entry.name]
        anamnesis_detected_drugs = [
            entry.name for entry in anamnesis_drugs.entries if entry.name
        ]
        anamnesis_detected_diseases = [
            entry.name for entry in disease_context.entries if entry.name
        ]
        matched_drugs_payload = self.build_matched_drugs_payload(
            detected_drugs=detected_drugs,
            prepared_inputs=prepared_inputs,
            rucam_bundle=rucam_bundle,
        )
        serialized_issues = self.serialize_pipeline_issues(issues)
        pattern_strings = self.pattern_analyzer.stringify_scores(pattern_score)
        narrative = NarrativeBuilder.build_patient_narrative(
            patient_label=patient_label,
            visit_label=visit_label,
            anamnesis=payload.anamnesis,
            drugs_text=payload.drugs,
            pattern_score=pattern_score,
            pattern_strings=pattern_strings,
            detected_drugs=detected_drugs,
            anamnesis_detected_drugs=anamnesis_detected_drugs,
            rucam_assessments=rucam_bundle.entries,
            report_language=report_language,
            issues=issues,
            final_report=final_report,
        )
        result_payload = {
            "report": narrative,
            "issues": serialized_issues,
            "pattern_status": pattern_assessment.status,
            "detected_drugs": detected_drugs,
            "anamnesis_drugs": anamnesis_detected_drugs,
            "anamnesis_diseases": anamnesis_detected_diseases,
            "matched_drugs": matched_drugs_payload,
            "rucam_assessments": [item.model_dump() for item in rucam_bundle.entries],
            "lab_timeline": [entry.model_dump() for entry in lab_timeline.entries],
            "onset_context": onset_context.model_dump() if onset_context else None,
            "detected_input_language": language_result.detected_input_language,
            "report_language": language_result.report_language,
            "relevant_drugs": candidate_selection.relevant,
            "excluded_drugs": candidate_selection.excluded,
            "unresolved_drugs": candidate_selection.unresolved,
            "structured_case": {
                "therapy_drugs": [
                    entry.model_dump() for entry in therapy_drugs.entries
                ],
                "anamnesis_drugs": [
                    entry.model_dump() for entry in anamnesis_drugs.entries
                ],
                "anamnesis_diseases": [
                    entry.model_dump() for entry in disease_context.entries
                ],
            },
            "section_extraction": (
                section_extraction.model_dump() if section_extraction is not None else None
            ),
            "runtime_settings": {
                "use_cloud_services": LLMRuntimeConfig.is_cloud_enabled(),
                "llm_provider": LLMRuntimeConfig.get_llm_provider(),
                "cloud_model": LLMRuntimeConfig.get_cloud_model(),
                "text_extraction_model": LLMRuntimeConfig.get_text_extraction_model(),
                "clinical_model": LLMRuntimeConfig.get_clinical_model(),
                "ollama_temperature": LLMRuntimeConfig.get_ollama_temperature(),
                "cloud_temperature": LLMRuntimeConfig.get_cloud_temperature(),
                "ollama_reasoning": LLMRuntimeConfig.is_ollama_reasoning_enabled(),
            },
        }
        self.emit_progress(progress_callback, stage="finalization", value=96.0)
        self.run_stop_check(stop_check)
        await asyncio.to_thread(
            self.serializer.save_clinical_session,
            {
                "patient_name": payload.name,
                "patient_visit_date": payload.visit_date,
                "patient_image_base64": patient_image_base64,
                "session_timestamp": datetime.now(),
                "hepatic_pattern": pattern_score.classification,
                "anamnesis": payload.anamnesis,
                "drugs": payload.drugs,
                "laboratory_analysis": payload.laboratory_analysis,
                "section_extraction": (
                    section_extraction.model_dump()
                    if section_extraction is not None
                    else None
                ),
                "text_extraction_model": getattr(self.drugs_parser, "model", None),
                "clinical_model": getattr(clinical_session, "llm_model", None),
                "total_duration": global_elapsed,
                "final_report": final_report,
                "detected_drugs": detected_drugs,
                "matched_drugs": matched_drugs_payload,
                "issues": serialized_issues,
                "session_status": "successful",
                "session_result_payload": result_payload,
            },
        )
        self.emit_progress(progress_callback, stage="finalization", value=99.0)
        self.run_stop_check(stop_check)
        return result_payload

    # -------------------------------------------------------------------------
    async def start_clinical_session(
        self,
        request_payload: ClinicalSessionRequest,
    ) -> str:
        try:
            preprocessed_request, section_extraction = (
                await self.preprocess_unified_input(request_payload)
            )
        except ClinicalInputExtractionError as exc:
            raise ServiceValidationError(str(exc)) from exc
        patient_payload = self.build_patient_payload(preprocessed_request)
        try:
            self.ensure_submission_requirements(patient_payload)
        except ClinicalPipelineValidationError as exc:
            raise ServiceValidationError(
                self.serialize_pipeline_issues(exc.issues),
            ) from exc
        self.apply_persisted_runtime_configuration()
        try:
            single_result = await self.process_single_patient(
                patient_payload,
                patient_image_base64=request_payload.patient_image_base64,
                section_extraction=section_extraction,
            )
        except ClinicalPipelineValidationError as exc:
            raise ServiceValidationError(
                self.serialize_pipeline_issues(exc.issues),
            ) from exc
        report = str(single_result.get("report", "")).strip()
        return report

    # -------------------------------------------------------------------------
    def start_clinical_job(
        self,
        request_payload: ClinicalSessionRequest,
    ) -> JobStartResponse:
        if self.job_manager.is_job_running(self.JOB_TYPE):
            raise ServiceConflictError(
                "Clinical analysis is already in progress",
            )

        try:
            preprocessed_request, section_extraction = asyncio.run(
                self.preprocess_unified_input(request_payload)
            )
        except ClinicalInputExtractionError as exc:
            raise ServiceValidationError(str(exc)) from exc
        patient_payload = self.build_patient_payload(preprocessed_request)
        try:
            self.ensure_submission_requirements(patient_payload)
        except ClinicalPipelineValidationError as exc:
            raise ServiceValidationError(
                self.serialize_pipeline_issues(exc.issues),
            ) from exc
        self.apply_persisted_runtime_configuration()

        job_id = self.job_manager.start_job(
            job_type=self.JOB_TYPE,
            runner=run_clinical_job,
            kwargs={
                "service": self,
                "payload": patient_payload,
                "patient_image_base64": request_payload.patient_image_base64,
                "section_extraction": section_extraction,
            },
        )

        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise ServiceError(
                "Failed to initialize clinical analysis job",
            )

        return JobStartResponse(
            job_id=job_id,
            job_type=job_status["job_type"],
            status=job_status["status"],
            message="Clinical analysis job started",
            poll_interval=server_settings.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def get_clinical_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise ServiceNotFoundError(
                "Job not found.",
            )
        return JobStatusResponse(**job_status)

    # -------------------------------------------------------------------------
    def cancel_clinical_job(self, job_id: str) -> JobCancelResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise ServiceNotFoundError(
                "Job not found.",
            )
        success = self.job_manager.cancel_job(job_id)
        if success:
            logger.info("Clinical analysis stop requested for job %s", job_id)
        return JobCancelResponse(
            job_id=job_id,
            success=success,
            message="Cancellation requested" if success else "Job cannot be cancelled",
        )

