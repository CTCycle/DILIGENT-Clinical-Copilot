from __future__ import annotations

import asyncio
import time
from typing import Any
from collections.abc import Callable

from pydantic import ValidationError

from common.exceptions import (
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
from domain.clinical.robustness import ClinicalInputPreflightResult
from domain.clinical.validation import ValidationMessageBundle
from domain.jobs import (
    JobCancelResponse,
    JobStartResponse,
    JobStatusResponse,
)
from configurations.llm_configs import LLMRuntimeConfig
from repositories.serialization.data import DataSerializer
from repositories.serialization.model_configs import (
    ModelConfigSerializer,
)
from services.runtime.jobs import (
    JobManager,
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
from services.clinical.preparation import ClinicalKnowledgePreparation
from services.clinical.disease import DiseaseExtractor
from services.clinical.labs import ClinicalLabExtractor
from services.clinical.parser import DrugsParser
from services.clinical.drug_blocks import isolate_drug_blocks
from services.clinical.rucam import RucamScoreEstimator
from services.clinical.validation import (
    build_validation_bundle,
    ensure_required_sections,
    has_timing_information,
)
from services.session.payload import PayloadSanitizationService
from services.session.clinical_input_extractor import (
    ClinicalInputExtractionError,
    ClinicalInputExtractor,
)
from services.llm.model_config import ModelConfigService
from services.retrieval.query import DILIQueryBuilder
from services.session.formatting_mixin import (
    ClinicalSessionFormattingMixin,
)
from services.session.session_workflow import (
    build_matched_drugs_payload_workflow,
    build_single_matched_drug_row_workflow,
    process_single_patient_workflow,
    start_clinical_job_workflow,
)
from services.session.preflight import validate_clinical_input_preflight
from services.text.normalization import normalize_drug_query_name

###############################################################################
class ClinicalSessionService(ClinicalSessionFormattingMixin):
    JOB_TYPE = "clinical"
    CLOUD_STEP_TIMEOUT_CAP_S = 180.0
    LOCAL_STEP_TIMEOUT_CAP_S = 1800.0
    CLOUD_CONSULTATION_TIMEOUT_S = 600.0
    LOCAL_CONSULTATION_TIMEOUT_S = 3600.0

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
        job_manager: JobManager,
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
        self.job_manager = job_manager
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

    # -------------------------------------------------------------------------
    @staticmethod
    def _resolve_runtime_timeout(
        *,
        base_timeout_s: float,
        cloud_cap_s: float | None = None,
        local_cap_s: float | None = None,
    ) -> float:
        base = max(float(base_timeout_s), 1.0)
        if LLMRuntimeConfig.is_cloud_enabled():
            cap = cloud_cap_s if cloud_cap_s is not None else ClinicalSessionService.CLOUD_STEP_TIMEOUT_CAP_S
        else:
            cap = local_cap_s if local_cap_s is not None else ClinicalSessionService.LOCAL_STEP_TIMEOUT_CAP_S
        return min(base, max(float(cap), 1.0))

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
            # Keep submission permissive when therapy content cannot provide timing.
            return

        lines = [block.text.strip() for block in isolate_drug_blocks(cleaned_therapy_text) if block.text.strip()]
        parsed_entries = [
            parsed
            for parsed in (self.drugs_parser.parse_line(line) for line in lines)
            if parsed is not None
        ]
        if any(has_timing_information(entry) for entry in parsed_entries):
            return
        # Do not block session start when therapy timing is not explicitly available.
        # Downstream stages can still assess DILI with uncertainty notes.
        return

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
        timeout_s = self._resolve_runtime_timeout(
            base_timeout_s=float(getattr(self.drugs_parser, "timeout_s", 1.0))
        )
        try:
            therapy_drugs = await asyncio.wait_for(
                self.drugs_parser.extract_drugs_from_therapy(
                    cleaned_therapy_text,
                    progress_callback=therapy_progress_callback,
                ),
                timeout=timeout_s,
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
        timeout_s = self._resolve_runtime_timeout(
            base_timeout_s=float(getattr(self.drugs_parser, "timeout_s", 1.0))
        )
        try:
            anamnesis_drugs = await asyncio.wait_for(
                self.drugs_parser.extract_drugs_from_anamnesis(
                    anamnesis_text,
                    progress_callback=anamnesis_progress_callback,
                ),
                timeout=timeout_s,
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
                    "Anamnesis drugs extraction failed after %.4f seconds "
                    "(timeout budget %.1fs); continuing without historical "
                    "drug mentions: %s"
                ),
                elapsed,
                timeout_s,
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
        max_attempts = 1
        backoff_seconds = 0.0
        timeout_s = self._resolve_runtime_timeout(
            base_timeout_s=float(self.disease_extractor.timeout_s)
        )
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
                            "(timeout budget %.1fs, attempt %d/%d). Retrying in %.1fs."
                        ),
                        elapsed,
                        timeout_s,
                        attempt,
                        max_attempts,
                        backoff_seconds,
                    )
                    await asyncio.sleep(backoff_seconds)
                    self.run_stop_check(stop_check)
                    continue
                logger.warning(
                    (
                        "Anamnesis disease extraction timed out after %.4f seconds "
                        "(timeout budget %.1fs); continuing without structured "
                        "disease timeline."
                    ),
                    elapsed,
                    timeout_s,
                )
                self.append_warning_issue(
                    issues,
                    code="anamnesis_disease_extraction_timeout",
                    message=(
                        "Disease extraction from anamnesis timed out; "
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
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                logger.warning(
                    (
                        "Anamnesis disease extraction failed after %.4f seconds "
                        "(timeout budget %.1fs); continuing without structured "
                        "disease timeline: %s"
                    ),
                    elapsed,
                    timeout_s,
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
        timeout_s = self._resolve_runtime_timeout(
            base_timeout_s=float(getattr(self.lab_extractor, "timeout_s", 1.0))
        )
        try:
            lab_timeline, onset_context = await asyncio.wait_for(
                self.lab_extractor.extract_from_payload(
                    payload,
                    progress_callback=lab_progress_callback,
                ),
                timeout=timeout_s,
            )
            self.run_stop_check(stop_check)
            elapsed = time.perf_counter() - start_time
            logger.info("Anamnesis lab extraction required %.4f seconds", elapsed)
            logger.info("Detected %s timeline lab entries", len(lab_timeline.entries))
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.warning(
                (
                    "Anamnesis lab extraction failed after %.4f seconds "
                    "(timeout budget %.1fs); continuing with deterministic "
                    "lab timeline fallback: %s"
                ),
                elapsed,
                timeout_s,
                exc,
            )
            self.append_warning_issue(
                issues,
                code="anamnesis_lab_extraction_failed",
                message=(
                    "Longitudinal lab extraction from anamnesis was unavailable; "
                    "the analysis continued with deterministic lab parsing fallback."
                ),
                field="anamnesis",
            )
            # Deterministic fallback: recover lab markers directly from text
            # so pattern estimation can still proceed when LLM extraction fails.
            fallback_entries = []
            primary_labs_text = self.lab_extractor.clean_text(payload.laboratory_analysis)
            supplemental_anamnesis_text = self.lab_extractor.clean_text(payload.anamnesis)
            fallback_entries.extend(
                self.lab_extractor.extract_entries_from_text(
                    text=primary_labs_text,
                    source="laboratory_analysis",
                    visit_date=payload.visit_date,
                )
            )
            fallback_entries.extend(
                self.lab_extractor.extract_entries_from_text(
                    text=supplemental_anamnesis_text,
                    source="anamnesis",
                    visit_date=payload.visit_date,
                )
            )
            normalized_entries = []
            seen_keys: set[tuple[str, str, str, str]] = set()
            for entry in fallback_entries:
                prepared = self.lab_extractor.normalize_entry(
                    entry, visit_date=payload.visit_date
                )
                if prepared is None:
                    continue
                key = self.lab_extractor.dedupe_key(prepared)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                normalized_entries.append(prepared)
            normalized_entries.sort(key=self.lab_extractor.lab_entry_sort_key)
            lab_timeline = PatientLabTimeline(entries=normalized_entries)
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
        consultation_timeout_s = (
            self.CLOUD_CONSULTATION_TIMEOUT_S
            if LLMRuntimeConfig.is_cloud_enabled()
            else self.LOCAL_CONSULTATION_TIMEOUT_S
        )
        try:
            consultation_progress_callback = ClinicalConsultationProgressCallback(
                progress_callback=progress_callback,
            )
            drug_assessment = await asyncio.wait_for(
                clinical_session.run_analysis(
                    prepared_inputs=prepared_inputs,
                    visit_date=payload.visit_date,
                    report_language=report_language,
                    rag_query=rag_query,
                    rucam_bundle=rucam_bundle,
                    progress_callback=consultation_progress_callback,
                ),
                timeout=consultation_timeout_s,
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
        except TimeoutError as exc:
            self.append_warning_issue(
                issues,
                code="clinical_llm_timeout",
                message=(
                    "Clinical LLM analysis timed out; report generated without "
                    "per-drug synthesis."
                ),
            )
            logger.warning(
                "Clinical LLM timeout for patient '%s' after %.1fs: %s",
                payload.name or "unknown",
                consultation_timeout_s,
                exc,
            )
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
        if not final_report:
            drug_names = [
                (entry.name or "").strip()
                for entry in analysis_drugs.entries
                if (entry.name or "").strip()
            ]
            unique_drugs: list[str] = []
            seen_drugs: set[str] = set()
            for name in drug_names:
                key = name.casefold()
                if key in seen_drugs:
                    continue
                seen_drugs.add(key)
                unique_drugs.append(name)
                if len(unique_drugs) >= 8:
                    break
            if report_language.lower().startswith("it"):
                if unique_drugs:
                    final_report = (
                        "Report finale generato in modalità di fallback per indisponibilità del motore clinico. "
                        f"Farmaci sospetti identificati nel testo: {', '.join(unique_drugs)}. "
                        "Rivedere manualmente la valutazione clinica e la conclusione specialistica originale."
                    )
                else:
                    final_report = (
                        "Report finale generato in modalità di fallback per indisponibilità del motore clinico. "
                        "Non sono stati identificati farmaci sospetti affidabili; è necessaria revisione manuale."
                    )
            else:
                if unique_drugs:
                    final_report = (
                        "Final report generated in fallback mode because clinical synthesis was unavailable. "
                        f"Suspected drugs detected from source text: {', '.join(unique_drugs)}. "
                        "Manual review against the original specialist assessment is required."
                    )
                else:
                    final_report = (
                        "Final report generated in fallback mode because clinical synthesis was unavailable. "
                        "No reliable suspected drugs were detected; manual specialist review is required."
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
    @staticmethod
    def _build_single_matched_drug_row(
        *,
        detected_name: str,
        resolved: dict[str, Any],
        rucam_entry: DrugRucamAssessment | None,
    ) -> dict[str, Any]:
        return build_single_matched_drug_row_workflow(
            detected_name=detected_name,
            resolved=resolved,
            rucam_entry=rucam_entry,
        )
    @staticmethod
    def build_matched_drugs_payload(
        *,
        detected_drugs: list[str],
        prepared_inputs,
        rucam_bundle: PatientRucamAssessmentBundle,
    ) -> list[dict[str, Any]]:
        return build_matched_drugs_payload_workflow(
            service=ClinicalSessionService,
            detected_drugs=detected_drugs,
            prepared_inputs=prepared_inputs,
            rucam_bundle=rucam_bundle,
        )
    async def process_single_patient(
        self,
        payload: PatientData,
        *,
        patient_image_base64: str | None = None,
        section_extraction: ClinicalSectionExtractionResult | None = None,
        normalized_document: Any | None = None,
        report_mode: str = "faithful_only",
        session_version: int = 1,
        original_session_id: int | None = None,
        session_metadata: dict[str, Any] | None = None,
        original_session_text: str | None = None,
        revision_focus_context: str | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
        stop_check: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        return await process_single_patient_workflow(
            self,
            payload,
            patient_image_base64=patient_image_base64,
            section_extraction=section_extraction,
            normalized_document=normalized_document,
            report_mode=report_mode,
            session_version=session_version,
            original_session_id=original_session_id,
            session_metadata=session_metadata,
            original_session_text=original_session_text,
            revision_focus_context=revision_focus_context,
            progress_callback=progress_callback,
            stop_check=stop_check,
        )
    def start_clinical_job(
        self,
        request_payload: ClinicalSessionRequest,
    ) -> JobStartResponse:
        return start_clinical_job_workflow(self, request_payload)

    def validate_clinical_input(
        self,
        request_payload: ClinicalSessionRequest,
    ) -> ClinicalInputPreflightResult:
        return validate_clinical_input_preflight(self, request_payload)

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
        cancelled_snapshot = self.job_manager.cancel_job(job_id)
        success = cancelled_snapshot is not None
        if success:
            logger.info("Clinical analysis stop requested for job %s", job_id)
        return JobCancelResponse(
            job_id=job_id,
            success=success,
            message="Cancellation requested" if success else "Job cannot be cancelled",
        )

