from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from common.exceptions import (
    ServiceNotFoundError,
    ServiceValidationError,
)
from common.utils.logger import logger
from configurations.llm_configs import LLMRuntimeConfig
from configurations.startup import get_server_settings
from domain.clinical.entities import (
    ClinicalSectionExtractionResult,
    ClinicalSessionRequest,
    PatientData,
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
from repositories.serialization.data import DataSerializer
from repositories.serialization.model_configs import (
    ModelConfigSerializer,
)
from services.clinical.disease import DiseaseExtractor
from services.clinical.drug_blocks import isolate_drug_blocks
from services.clinical.hepatox_core import (
    HepatotoxicityPatternAnalyzer,
    HepatoxConsultation,
)
from services.clinical.job_progress import (
    StageProgressFractionCallback,
)
from services.clinical.labs import ClinicalLabExtractor
from services.clinical.language import ClinicalLanguageDetector
from services.clinical.parser import DrugsParser
from services.clinical.preparation import ClinicalKnowledgePreparation
from services.clinical.rucam import RucamScoreEstimator
from services.clinical.validation import (
    build_validation_bundle,
    ensure_required_sections,
    has_timing_information,
)
from services.llm.model_config import ModelConfigService
from services.runtime.jobs import (
    JobManager,
)
from services.session.clinical_input_extractor import (
    ClinicalInputExtractor,
)
from services.session.document_normalizer import DocumentNormalizer
from services.session.formatting_mixin import (
    ClinicalSessionFormattingMixin,
)
from services.session.payload import PayloadSanitizationService
from services.session.preflight import validate_clinical_input_preflight
from services.session.text_section_parser import (
    InitialTextSectionParseResult,
    build_section_extraction_from_initial_text,
    parse_initial_text_sections,
)
from services.session.revision_workflow import process_revision_patient_workflow
from services.session.session_workflow import (
    build_matched_drugs_payload_workflow,
    process_single_patient_workflow,
    start_clinical_job_workflow,
)


###############################################################################
from services.session.consultation import ClinicalSessionConsultationMixin
from services.session.extraction_pipeline import ClinicalSessionExtractionPipelineMixin

###############################################################################
class ClinicalSessionService(
    ClinicalSessionFormattingMixin,
    ClinicalSessionConsultationMixin,
    ClinicalSessionExtractionPipelineMixin,
):
    JOB_TYPE = "clinical"

    # -------------------------------------------------------------------------
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
        self.clinical_input_extractor = (
            clinical_input_extractor or ClinicalInputExtractor()
        )
        self.hepatox_consultation_cls = hepatox_consultation_cls or HepatoxConsultation
        self.job_manager = job_manager
        self.model_config_service = ModelConfigService(
            serializer=ModelConfigSerializer()
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def emit_progress(
        progress_callback: Callable[..., None] | None,
        *,
        stage: str,
        value: float,
        detail: str | None = None,
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(stage, value, detail)
        except TypeError:
            progress_callback(stage, value)

    # -------------------------------------------------------------------------
    @staticmethod
    def build_stage_progress_callback(
        progress_callback: Callable[..., None] | None,
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
        runtime_settings = get_server_settings().runtime
        minimum_timeout_s = float(getattr(runtime_settings, "minimum_llm_timeout", 1.0))
        base = max(float(base_timeout_s), minimum_timeout_s)
        if LLMRuntimeConfig.is_cloud_enabled():
            requested = cloud_cap_s
        else:
            requested = local_cap_s
        if requested is None:
            return base
        return max(base, max(float(requested), minimum_timeout_s))

    # -------------------------------------------------------------------------
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
            raise ServiceValidationError("Clinical input is required.")
        parse_result = parse_initial_text_sections(clinical_input)
        if parse_result.missing_required_sections or parse_result.malformed_sections:
            details: list[str] = []
            if parse_result.missing_required_sections:
                details.append(
                    "missing sections: "
                    + ", ".join(parse_result.missing_required_sections)
                )
            if parse_result.malformed_sections:
                details.append(
                    "malformed sections: " + ", ".join(parse_result.malformed_sections)
                )
            raise ServiceValidationError(
                "Clinical input sections are invalid (" + "; ".join(details) + ")."
            )
        extraction = build_section_extraction_from_initial_text(
            parse_result,
            clinical_input,
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
    async def prepare_revision_source_request(
        self,
        *,
        session_detail: dict[str, Any],
        use_rag: bool,
    ) -> tuple[
        ClinicalSessionRequest,
        ClinicalSectionExtractionResult | None,
        str,
    ]:
        source_text = str(session_detail.get("session_text") or "").strip()
        if not source_text:
            raise ServiceValidationError("Clinical source text is required.")

        payload = session_detail.get("result_payload")
        section_extraction_payload = (
            payload.get("section_extraction") if isinstance(payload, dict) else None
        )
        section_extraction: ClinicalSectionExtractionResult | None = None
        if isinstance(section_extraction_payload, ClinicalSectionExtractionResult):
            section_extraction = section_extraction_payload
        elif isinstance(section_extraction_payload, dict):
            try:
                section_extraction = ClinicalSectionExtractionResult.model_validate(
                    section_extraction_payload
                )
            except ValidationError:
                section_extraction = None
        if (
            section_extraction is not None
            and section_extraction.anamnesis
            and section_extraction.drugs
            and section_extraction.laboratory_analysis
        ):
            return (
                ClinicalSessionRequest(
                    name=session_detail.get("patient_name"),
                    visit_date=session_detail.get("visit_date"),
                    clinical_input=source_text,
                    anamnesis=section_extraction.anamnesis,
                    drugs=section_extraction.drugs,
                    laboratory_analysis=section_extraction.laboratory_analysis,
                    use_rag=use_rag,
                ),
                section_extraction,
                "persisted_section_extraction",
            )

        sections = session_detail.get("sections")
        if isinstance(sections, dict):
            anamnesis = str(sections.get("anamnesis") or "").strip()
            drugs = str(sections.get("drugs") or sections.get("therapy") or "").strip()
            laboratory_analysis = str(
                sections.get("laboratory_analysis")
                or sections.get("laboratory_history")
                or ""
            ).strip()
            if anamnesis and drugs and laboratory_analysis:
                section_extraction = ClinicalSectionExtractionResult(
                    source_text=source_text,
                    anamnesis=anamnesis,
                    drugs=drugs,
                    laboratory_analysis=laboratory_analysis,
                    confidence=1.0,
                    metadata={"parser": "persisted_session_sections_v1"},
                )
                return (
                    ClinicalSessionRequest(
                        name=session_detail.get("patient_name"),
                        visit_date=session_detail.get("visit_date"),
                        clinical_input=source_text,
                        anamnesis=anamnesis,
                        drugs=drugs,
                        laboratory_analysis=laboratory_analysis,
                        use_rag=use_rag,
                    ),
                    section_extraction,
                    "persisted_session_sections",
                )

        request = ClinicalSessionRequest(
            name=session_detail.get("patient_name"),
            visit_date=session_detail.get("visit_date"),
            clinical_input=source_text,
            use_rag=use_rag,
        )
        preprocessed_request, section_extraction = await self.preprocess_unified_input(
            request
        )
        return preprocessed_request, section_extraction, "reparsed_source_text"

    # -------------------------------------------------------------------------
    def validate_assessment_prerequisites_without_llm(
        self, request_payload: ClinicalSessionRequest
    ) -> InitialTextSectionParseResult:
        clinical_input = (request_payload.clinical_input or "").strip()
        if not clinical_input:
            raise ServiceValidationError("Clinical input is required.")
        if not request_payload.visit_date:
            raise ServiceValidationError("Visit date is required.")

        livertox_rows, _ = self.serializer.list_livertox_catalog(
            search=None, offset=0, limit=1
        )
        if not livertox_rows:
            raise ServiceValidationError(
                "LiverTox catalog is empty. Rebuild LiverTox data before clinical analysis."
            )
        rxnav_rows, _ = self.serializer.list_rxnav_catalog(
            search=None, offset=0, limit=1
        )
        if not rxnav_rows:
            raise ServiceValidationError(
                "RxNav catalog is empty. Rebuild RxNav data before clinical analysis."
            )

        parse_result = parse_initial_text_sections(clinical_input)
        if parse_result.missing_required_sections or parse_result.malformed_sections:
            details: list[str] = []
            if parse_result.missing_required_sections:
                details.append(
                    "missing sections: "
                    + ", ".join(parse_result.missing_required_sections)
                )
            if parse_result.malformed_sections:
                details.append(
                    "malformed sections: " + ", ".join(parse_result.malformed_sections)
                )
            raise ServiceValidationError(
                "Clinical input sections are invalid (" + "; ".join(details) + ")."
            )
        return parse_result

    # -------------------------------------------------------------------------
    def prepare_structured_clinical_input(
        self,
        request_payload: ClinicalSessionRequest,
    ) -> dict[str, Any]:
        clinical_input = (request_payload.clinical_input or "").strip()
        if not clinical_input:
            raise ServiceValidationError("Clinical input is required.")

        parse_result = self.validate_assessment_prerequisites_without_llm(
            request_payload
        )
        section_extraction = build_section_extraction_from_initial_text(
            parse_result,
            clinical_input,
        )
        if (
            not section_extraction.anamnesis
            or not section_extraction.drugs
            or not section_extraction.laboratory_analysis
        ):
            raise ServiceValidationError(
                "Clinical input must contain anamnesis, current therapy, and laboratory analysis sections."
            )

        normalized_document = DocumentNormalizer().normalize(clinical_input)
        request_with_sections = request_payload.model_copy(
            update={
                "anamnesis": section_extraction.anamnesis,
                "drugs": section_extraction.drugs,
                "laboratory_analysis": section_extraction.laboratory_analysis,
            }
        )
        patient_payload = self.build_patient_payload(request_with_sections)
        return {
            "parse_result": parse_result,
            "section_extraction": section_extraction,
            "normalized_document": normalized_document,
            "request_payload": request_with_sections,
            "patient_payload": patient_payload,
        }

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

        lines = [
            block.text.strip()
            for block in isolate_drug_blocks(cleaned_therapy_text)
            if block.text.strip()
        ]
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
    def build_matched_drugs_payload(
        *,
        detected_drugs: list[str],
        prepared_inputs,
        rucam_bundle: PatientRucamAssessmentBundle,
    ) -> list[dict[str, Any]]:
        return build_matched_drugs_payload_workflow(
            detected_drugs=detected_drugs,
            prepared_inputs=prepared_inputs,
            rucam_bundle=rucam_bundle,
        )

    # -------------------------------------------------------------------------
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
            progress_callback=progress_callback,
            stop_check=stop_check,
        )

    # -------------------------------------------------------------------------
    async def process_revision_patient(
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
        return await process_revision_patient_workflow(
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

    # -------------------------------------------------------------------------
    def start_clinical_job(
        self,
        request_payload: ClinicalSessionRequest,
    ) -> JobStartResponse:
        return start_clinical_job_workflow(self, request_payload)

    # -------------------------------------------------------------------------
    def validate_clinical_input(
        self,
        request_payload: ClinicalSessionRequest,
    ) -> ClinicalInputPreflightResult:
        return validate_clinical_input_preflight(self, request_payload)

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
        cancelled_snapshot = self.job_manager.cancel_job(job_id)
        success = cancelled_snapshot is not None
        if success:
            logger.info("Clinical analysis stop requested for job %s", job_id)
        return JobCancelResponse(
            job_id=job_id,
            success=success,
            message="Cancellation requested" if success else "Job cannot be cancelled",
        )
