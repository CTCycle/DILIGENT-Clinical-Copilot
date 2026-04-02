from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from datetime import datetime
from collections.abc import Callable, Sequence
from typing import Any

from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from DILIGENT.server.models.cloud import LLMError

from DILIGENT.server.domain.clinical import (
    ClinicalLabEntry,
    ClinicalPipelineValidationError,
    ClinicalSessionRequest,
    DiseaseContextEntry,
    DrugEntry,
    DrugRucamAssessment,
    LiverInjuryOnsetContext,
    PatientLabTimeline,
    PatientData,
    PatientDiseaseContext,
    PatientDrugs,
    PatientRucamAssessmentBundle,
    PipelineIssue,
)
from DILIGENT.server.domain.jobs import (
    JobCancelResponse,
    JobStartResponse,
    JobStatusResponse,
)
from DILIGENT.server.configurations import LLMRuntimeConfig, server_settings
from DILIGENT.server.repositories.serialization.data import DataSerializer
from DILIGENT.server.services.jobs import job_manager
from DILIGENT.server.common.utils.logger import logger
from DILIGENT.server.common.utils.types import coerce_bool_or_unknown
from DILIGENT.server.services.clinical.hepatox import (
    HepatotoxicityPatternAnalyzer,
    HepatoxConsultation,
)
from DILIGENT.server.services.clinical.language import detect_clinical_language
from DILIGENT.server.services.clinical.preparation import ClinicalKnowledgePreparation
from DILIGENT.server.services.clinical.candidate_selection import select_relevant_candidates
from DILIGENT.server.services.clinical.disease import DiseaseExtractor
from DILIGENT.server.services.clinical.labs import ClinicalLabExtractor
from DILIGENT.server.services.clinical.parser import DrugsParser
from DILIGENT.server.services.clinical.rucam import RucamScoreEstimator
from DILIGENT.server.services.clinical.validation import (
    build_validation_bundle,
    ensure_required_sections,
    ensure_timed_therapy_drug,
)
from DILIGENT.server.services.payload import PayloadSanitizationService
from DILIGENT.server.services.retrieval.query import DILIQueryBuilder
from DILIGENT.server.services.text.normalization import normalize_drug_query_name

drugs_parser = DrugsParser(timeout_s=server_settings.external_data.parser_llm_timeout)
disease_extractor = DiseaseExtractor(timeout_s=server_settings.external_data.disease_llm_timeout)
lab_extractor = ClinicalLabExtractor(timeout_s=server_settings.external_data.disease_llm_timeout)
pattern_analyzer = HepatotoxicityPatternAnalyzer()
rucam_estimator = RucamScoreEstimator()
input_preparator = ClinicalKnowledgePreparation()
router = APIRouter(tags=["session"])
serializer = DataSerializer()
payload_sanitization_service = PayloadSanitizationService()
NOT_AVAILABLE = "Not available"
CLINICAL_PROGRESS_MESSAGES: dict[str, str] = {
    "session_initialization": "Initializing clinical session",
    "hepatotoxicity_pattern": "Calculating hepatotoxicity pattern",
    "therapy_extraction": "Extracting drugs from therapy",
    "anamnesis_extraction": "Extracting drugs from anamnesis",
    "anamnesis_disease_extraction": "Extracting diseases from anamnesis",
    "anamnesis_lab_extraction": "Extracting longitudinal labs from clinical text",
    "rag_query_building": "Building RAG queries",
    "livertox_lookup": "Consulting LiverTox knowledge base",
    "rucam_estimation": "Estimating per-drug RUCAM",
    "llm_analysis": "Running LLM drug-by-drug assessment",
    "report_composition": "Composing final clinical report",
    "finalization": "Finalizing and persisting session",
}


###############################################################################
class ClinicalJobCancelled(Exception):
    pass


###############################################################################
class ClinicalJobProgressCallback:
    def __init__(self, *, job_id: str) -> None:
        self.job_id = job_id

    # -------------------------------------------------------------------------
    def __call__(self, stage: str, progress: float) -> None:
        report_clinical_job_progress(self.job_id, stage=stage, progress=progress)


###############################################################################
class ClinicalConsultationProgressCallback:
    def __init__(
        self,
        *,
        progress_callback: Callable[[str, float], None] | None,
    ) -> None:
        self.progress_callback = progress_callback

    # -------------------------------------------------------------------------
    def __call__(self, stage: str, fraction: float) -> None:
        if self.progress_callback is None:
            return
        bounded_fraction = min(1.0, max(0.0, float(fraction)))
        if stage == "llm_analysis":
            self.progress_callback("llm_analysis", 62.0 + (bounded_fraction * 24.0))
        elif stage == "report_composition":
            self.progress_callback("report_composition", 86.0 + (bounded_fraction * 8.0))


###############################################################################
class StageProgressFractionCallback:
    def __init__(
        self,
        *,
        progress_callback: Callable[[str, float], None],
        stage: str,
        start_value: float,
        end_value: float,
    ) -> None:
        self.progress_callback = progress_callback
        self.stage = stage
        self.lower = min(start_value, end_value)
        self.span = max(0.0, end_value - self.lower)

    # -------------------------------------------------------------------------
    def __call__(self, fraction: float) -> None:
        bounded_fraction = min(1.0, max(0.0, float(fraction)))
        self.progress_callback(self.stage, self.lower + (self.span * bounded_fraction))

###############################################################################
def report_clinical_job_progress(job_id: str, *, stage: str, progress: float) -> None:
    if job_manager.should_stop(job_id):
        raise ClinicalJobCancelled("Clinical job stop requested.")
    bounded = min(100.0, max(0.0, float(progress)))
    message = CLINICAL_PROGRESS_MESSAGES.get(stage, stage.replace("_", " ").strip())
    job_manager.update_progress(job_id, bounded)
    job_manager.update_result(
        job_id,
        {
            "progress_stage": stage,
            "progress_message": message,
        },
    )


###############################################################################
def build_failed_session_payload(
    *,
    payload: PatientData,
    runtime_overrides: dict[str, Any],
    issues: list[dict[str, Any]],
    error_message: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "patient_name": payload.name,
        "session_timestamp": datetime.now(),
        "hepatic_pattern": "indeterminate",
        "anamnesis": payload.anamnesis,
        "drugs": payload.drugs,
        "laboratory_analysis": payload.laboratory_analysis,
        "parsing_model": runtime_overrides.get("parsing_model")
        or LLMRuntimeConfig.get_parsing_model(),
        "clinical_model": runtime_overrides.get("clinical_model")
        or LLMRuntimeConfig.get_clinical_model(),
        "total_duration": elapsed_seconds,
        "final_report": None,
        "detected_drugs": [],
        "matched_drugs": [],
        "issues": issues,
        "session_status": "failed",
        "session_result_payload": {
            "report": "",
            "issues": issues,
            "error": error_message,
            "pattern_status": "failed",
            "detected_drugs": [],
            "anamnesis_drugs": [],
            "anamnesis_diseases": [],
            "matched_drugs": [],
            "rucam_assessments": [],
            "lab_timeline": [],
            "onset_context": None,
            "detected_input_language": "en",
            "report_language": "en",
            "relevant_drugs": [],
            "excluded_drugs": [],
            "unresolved_drugs": [],
            "structured_case": {},
        },
    }


###############################################################################
class NarrativeBuilder:
    BUNDLES: dict[str, dict[str, str]] = {
        "en": {
            "no_data": "- No data provided.",
            "summary_title": "# Clinical Visit Summary",
            "patient": "- **Patient:** {value}",
            "visit_date": "- **Visit date:** {value}",
            "anamnesis_title": "## Anamnesis",
            "no_anamnesis": "_No anamnesis provided._",
            "pattern_title": "## Hepatotoxicity Pattern",
            "classification": "- **Classification:** {value}",
            "r_score": "- **R-score:** {value}",
            "therapy_title": "## Current Drugs",
            "detected_drugs": "**Detected drugs ({count}):** {value}",
            "historical_title": "## Historical Drug Mentions",
            "historical_mentions": "- **Historical mentions ({count}):** {value}",
            "warnings_title": "## Warnings",
            "report_title": "## Clinical Report",
            "no_report": "No clinical report generated.",
            "none_detected": "None detected",
        },
        "it": {
            "no_data": "- Nessun dato fornito.",
            "summary_title": "# Sintesi Visita Clinica",
            "patient": "- **Paziente:** {value}",
            "visit_date": "- **Data visita:** {value}",
            "anamnesis_title": "## Anamnesi",
            "no_anamnesis": "_Anamnesi non fornita._",
            "pattern_title": "## Pattern di Epatotossicità",
            "classification": "- **Classificazione:** {value}",
            "r_score": "- **R-score:** {value}",
            "therapy_title": "## Terapia Corrente",
            "detected_drugs": "**Farmaci rilevati ({count}):** {value}",
            "historical_title": "## Menzioni Farmaci Anamnestiche",
            "historical_mentions": "- **Menzioni storiche ({count}):** {value}",
            "warnings_title": "## Avvisi",
            "report_title": "## Report Clinico",
            "no_report": "Nessun report clinico generato.",
            "none_detected": "Nessuno rilevato",
        },
    }

    # -------------------------------------------------------------------------
    @staticmethod
    def bundle(report_language: str) -> dict[str, str]:
        if report_language.startswith("it"):
            return NarrativeBuilder.BUNDLES["it"]
        return NarrativeBuilder.BUNDLES["en"]

    # -------------------------------------------------------------------------
    @staticmethod
    def build_bullet_list(content: str | None, *, no_data_label: str) -> list[str]:
        lines: list[str] = []
        if content:
            for entry in content.splitlines():
                stripped = entry.strip()
                if stripped:
                    lines.append(f"- {stripped}")
        if not lines:
            lines.append(no_data_label)
        return lines

    # -------------------------------------------------------------------------
    @staticmethod
    def compact_spacing(content: str) -> str:
        cleaned: list[str] = []
        previous_blank = False
        for raw_line in content.splitlines():
            line = raw_line.rstrip()
            if not line:
                if previous_blank:
                    continue
                previous_blank = True
                cleaned.append("")
                continue
            previous_blank = False
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    # -------------------------------------------------------------------------
    @staticmethod
    def build_patient_narrative(
        *,
        patient_label: str,
        visit_label: str,
        anamnesis: str | None,
        drugs_text: str | None,
        pattern_score,
        pattern_strings: dict[str, str],
        detected_drugs: list[str],
        anamnesis_detected_drugs: list[str],
        report_language: str,
        issues: list[PipelineIssue],
        final_report: str | None,
    ) -> str:
        bundle = NarrativeBuilder.bundle(report_language)
        classification = getattr(pattern_score, "classification", NOT_AVAILABLE)
        r_score = pattern_strings.get("r_score", NOT_AVAILABLE)
        drug_summary = ", ".join(detected_drugs) if detected_drugs else bundle["none_detected"]

        sections: list[str] = []

        header_section = [
            bundle["summary_title"],
            "",
            bundle["patient"].format(value=patient_label),
            bundle["visit_date"].format(value=visit_label),
        ]
        sections.append("\n".join(header_section))

        anamnesis_content = anamnesis if anamnesis else bundle["no_anamnesis"]
        sections.append("\n".join([bundle["anamnesis_title"], "", anamnesis_content]))

        pattern_section = [
            bundle["pattern_title"],
            "",
            bundle["classification"].format(value=classification),
            bundle["r_score"].format(value=r_score),
        ]
        sections.append("\n".join(pattern_section))

        therapy_section = [bundle["therapy_title"], ""]
        therapy_section.extend(
            NarrativeBuilder.build_bullet_list(
                drugs_text,
                no_data_label=bundle["no_data"],
            )
        )
        therapy_section.extend(
            [
                "",
                bundle["detected_drugs"].format(count=len(detected_drugs), value=drug_summary),
            ]
        )
        sections.append("\n".join(therapy_section))

        anamnesis_drug_summary = (
            ", ".join(anamnesis_detected_drugs) if anamnesis_detected_drugs else bundle["none_detected"]
        )
        sections.append(
            "\n".join(
                [
                    bundle["historical_title"],
                    "",
                    bundle["historical_mentions"].format(
                        count=len(anamnesis_detected_drugs),
                        value=anamnesis_drug_summary,
                    ),
                ]
            )
        )

        if issues:
            warnings_section = [bundle["warnings_title"], ""]
            for issue in issues:
                warnings_section.append(f"- {issue.message}")
            sections.append("\n".join(warnings_section))

        clinical_report_section = [bundle["report_title"], ""]
        clinical_report_section.append(
            final_report.strip() if final_report else bundle["no_report"]
        )
        sections.append("\n".join(clinical_report_section))

        return NarrativeBuilder.compact_spacing("\n\n".join(sections))


###############################################################################
async def execute_clinical_job(
    payload: PatientData,
    runtime_overrides: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    def ensure_not_cancelled() -> None:
        if job_manager.should_stop(job_id):
            raise ClinicalJobCancelled("Clinical job stop requested.")

    with endpoint.runtime_override_context(
        use_cloud_services=runtime_overrides.get("use_cloud_services"),
        llm_provider=runtime_overrides.get("llm_provider"),
        cloud_model=runtime_overrides.get("cloud_model"),
        parsing_model=runtime_overrides.get("parsing_model"),
        clinical_model=runtime_overrides.get("clinical_model"),
        ollama_temperature=runtime_overrides.get("ollama_temperature"),
        ollama_reasoning=runtime_overrides.get("ollama_reasoning"),
    ):
        ensure_not_cancelled()

        report_clinical_job_progress(
            job_id,
            stage="session_initialization",
            progress=5.0,
        )
        progress_callback = ClinicalJobProgressCallback(job_id=job_id)
        job_started_at = time.perf_counter()

        try:
            result = await endpoint.process_single_patient(
                payload,
                progress_callback=progress_callback,
                stop_check=ensure_not_cancelled,
            )
        except ClinicalJobCancelled:
            job_manager.update_result(
                job_id,
                {
                    "progress_status": "cancelled",
                    "progress_message": "Clinical analysis cancelled.",
                },
            )
            return {}
        except ClinicalPipelineValidationError as exc:
            serialized_issues = [issue.model_dump() for issue in exc.issues]
            await asyncio.to_thread(
                serializer.save_clinical_session,
                build_failed_session_payload(
                    payload=payload,
                    runtime_overrides=runtime_overrides,
                    issues=serialized_issues,
                    error_message=str(exc),
                    elapsed_seconds=(time.perf_counter() - job_started_at),
                ),
            )
            job_manager.update_result(
                job_id,
                {
                    "validation_error": str(exc),
                    "issues": serialized_issues,
                },
            )
            raise
        except Exception as exc:
            if job_manager.should_stop(job_id):
                return {}
            failure_issue = PipelineIssue(
                severity="error",
                code="clinical_job_failed",
                message=str(exc).strip() or "Clinical analysis failed unexpectedly.",
            ).model_dump()
            await asyncio.to_thread(
                serializer.save_clinical_session,
                build_failed_session_payload(
                    payload=payload,
                    runtime_overrides=runtime_overrides,
                    issues=[failure_issue],
                    error_message=str(exc),
                    elapsed_seconds=(time.perf_counter() - job_started_at),
                ),
            )
            raise
    return result


###############################################################################
def run_clinical_job(
    payload: PatientData,
    runtime_overrides: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    result = asyncio.run(
        execute_clinical_job(payload=payload, runtime_overrides=runtime_overrides, job_id=job_id)
    )
    if not result:
        return {}
    return result


###############################################################################
class ClinicalSessionEndpoint:
    JOB_TYPE = "clinical"

    def __init__(
        self,
        *,
        router: APIRouter,
        drugs_parser: DrugsParser,
        disease_extractor: DiseaseExtractor,
        lab_extractor: ClinicalLabExtractor,
        pattern_analyzer: HepatotoxicityPatternAnalyzer,
        rucam_estimator: RucamScoreEstimator,
        serializer: DataSerializer,
        payload_sanitizer: PayloadSanitizationService,
    ) -> None:
        self.router = router
        self.drugs_parser = drugs_parser
        self.disease_extractor = disease_extractor
        self.lab_extractor = lab_extractor
        self.pattern_analyzer = pattern_analyzer
        self.rucam_estimator = rucam_estimator
        self.serializer = serializer
        self.payload_sanitizer = payload_sanitizer

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
    def serialize_validation_errors(
        errors: Sequence[ErrorDetails],
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for error in errors:
            error_dict: dict[str, Any] = dict(error)
            ctx = error_dict.get("ctx")
            if isinstance(ctx, dict) and "error" in ctx:
                serialized.append({**error_dict, "ctx": {**ctx, "error": str(ctx["error"])}})
                continue
            serialized.append(error_dict)
        return serialized

    # -------------------------------------------------------------------------
    @staticmethod
    def serialize_pipeline_issues(issues: Sequence[PipelineIssue]) -> list[dict[str, Any]]:
        return [issue.model_dump() for issue in issues]

    # -------------------------------------------------------------------------
    @staticmethod
    def merge_drugs_for_analysis(
        therapy_drugs: PatientDrugs,
        anamnesis_drugs: PatientDrugs,
    ) -> PatientDrugs:
        merged_entries: list[DrugEntry] = []
        seen_keys: set[str] = set()
        ordered = [*therapy_drugs.entries, *anamnesis_drugs.entries]
        for entry in ordered:
            raw_name = (entry.name or "").strip()
            if not raw_name:
                continue
            lookup_key = normalize_drug_query_name(raw_name)
            if not lookup_key or lookup_key in seen_keys:
                continue
            seen_keys.add(lookup_key)
            merged_entries.append(entry)
        return PatientDrugs(entries=merged_entries)

    # -------------------------------------------------------------------------
    @staticmethod
    def build_fallback_therapy_drugs(raw_text: str | None) -> PatientDrugs:
        if not raw_text:
            return PatientDrugs(entries=[])

        candidates = raw_text.replace(";", "\n").splitlines()
        entries: list[DrugEntry] = []
        seen_keys: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip().lstrip("-*• ").strip()
            if not normalized:
                continue
            lookup_key = normalize_drug_query_name(normalized)
            if not lookup_key or lookup_key in seen_keys:
                continue
            seen_keys.add(lookup_key)
            entries.append(
                DrugEntry(
                    name=normalized,
                    source="therapy",
                    historical_flag=False,
                )
            )
        return PatientDrugs(entries=entries)

    # -------------------------------------------------------------------------
    @staticmethod
    def format_structured_diseases(disease_context: PatientDiseaseContext) -> list[str]:
        if not disease_context.entries:
            return ["- None detected."]
        lines: list[str] = []
        for entry in disease_context.entries:
            if not isinstance(entry, DiseaseContextEntry):
                continue
            occurrence = entry.occurrence_time or "unknown"
            chronic = coerce_bool_or_unknown(entry.chronic)
            hepatic_related = coerce_bool_or_unknown(
                entry.hepatic_related
            )
            evidence = entry.evidence or "Not reported."
            lines.append(
                f"- {entry.name} | occurrence: {occurrence} | chronic: {chronic} | hepatic-related: {hepatic_related} | evidence: {evidence}"
            )
        return lines or ["- None detected."]

    # -------------------------------------------------------------------------
    @staticmethod
    def format_lab_timeline(lab_timeline: PatientLabTimeline) -> list[str]:
        if not lab_timeline.entries:
            return ["- None extracted."]
        lines: list[str] = []
        for entry in lab_timeline.entries:
            if not isinstance(entry, ClinicalLabEntry):
                continue
            date_token = entry.sample_date or entry.relative_time or "unknown_time"
            value_token = entry.value if entry.value is not None else (entry.value_text or "n/a")
            uln_token = (
                entry.upper_limit_normal
                if entry.upper_limit_normal is not None
                else (entry.upper_limit_text or "n/a")
            )
            lines.append(
                f"- {date_token} | {entry.marker_name}: {value_token} (ULN: {uln_token}) | source: {entry.source}"
            )
        return lines or ["- None extracted."]

    # -------------------------------------------------------------------------
    @staticmethod
    def format_onset_context(onset_context: LiverInjuryOnsetContext | None) -> list[str]:
        if onset_context is None:
            return ["- Onset anchor unavailable."]
        return [
            f"- Onset date: {onset_context.onset_date or 'Not available'}",
            f"- Onset basis: {onset_context.onset_basis}",
            f"- Evidence: {onset_context.evidence or 'Not reported.'}",
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def build_structured_clinical_context(
        payload: PatientData,
        *,
        therapy_drugs: PatientDrugs,
        anamnesis_drugs: PatientDrugs,
        disease_context: PatientDiseaseContext,
        lab_timeline: PatientLabTimeline,
        onset_context: LiverInjuryOnsetContext | None,
        pattern_score: Any,
    ) -> str:
        therapy_mentions = [
            entry.name.strip()
            for entry in therapy_drugs.entries
            if isinstance(entry.name, str) and entry.name.strip()
        ]
        anamnesis_mentions = [
            entry.name.strip()
            for entry in anamnesis_drugs.entries
            if isinstance(entry.name, str) and entry.name.strip()
        ]
        lines: list[str] = [
            "# Clinical Context",
            f"Anamnesis: {(payload.anamnesis or '').strip() or 'Not provided.'}",
            "",
            "# Laboratory Analysis (Raw)",
            (payload.laboratory_analysis or "").strip() or "Not provided.",
            "",
            "# Therapy List (Raw)",
            (payload.drugs or "").strip() or "Not provided.",
            "",
            "# Detected Drugs",
            f"- Therapy: {', '.join(therapy_mentions) if therapy_mentions else 'None'}",
            f"- Anamnesis: {', '.join(anamnesis_mentions) if anamnesis_mentions else 'None'}",
            "",
            "# Structured Disease Timeline (from Anamnesis)",
            *ClinicalSessionEndpoint.format_structured_diseases(disease_context),
            "",
            "# Longitudinal Laboratory Timeline",
            *ClinicalSessionEndpoint.format_lab_timeline(lab_timeline),
            "",
            "# Estimated Liver Injury Onset Anchor",
            *ClinicalSessionEndpoint.format_onset_context(onset_context),
            "",
            "# Visit Date Anchor",
            (
                f"- Visit date: {payload.visit_date.isoformat()}"
                if payload.visit_date
                else "- Visit date: Not provided."
            ),
            "",
            "# Hepatotoxicity Pattern",
            f"- Classification: {getattr(pattern_score, 'classification', 'indeterminate')}",
            f"- R score: {getattr(pattern_score, 'r_score', None)}",
        ]
        return "\n".join(lines).strip()

    # -------------------------------------------------------------------------
    def apply_runtime_overrides(
        self,
        *,
        use_cloud_services: bool | None,
        llm_provider: str | None,
        cloud_model: str | None,
        parsing_model: str | None,
        clinical_model: str | None,
        ollama_temperature: float | None,
        ollama_reasoning: bool | None,
    ) -> None:
        if use_cloud_services is not None:
            LLMRuntimeConfig.set_use_cloud_services(use_cloud_services)
        if llm_provider is not None:
            LLMRuntimeConfig.set_llm_provider(llm_provider)
        if cloud_model is not None:
            LLMRuntimeConfig.set_cloud_model(cloud_model)
        if parsing_model is not None:
            LLMRuntimeConfig.set_parsing_model(parsing_model)
        if clinical_model is not None:
            LLMRuntimeConfig.set_clinical_model(clinical_model)
        if ollama_temperature is not None:
            LLMRuntimeConfig.set_ollama_temperature(ollama_temperature)
        if ollama_reasoning is not None:
            LLMRuntimeConfig.set_ollama_reasoning(ollama_reasoning)

        parser_provider, parser_model = LLMRuntimeConfig.resolve_provider_and_model("parser")
        clinical_provider, clinical_model_resolved = LLMRuntimeConfig.resolve_provider_and_model("clinical")
        logger.info(
            "Resolved LLM runtime for request: cloud=%s provider=%s cloud_model=%s parsing_provider=%s parsing_model=%s clinical_provider=%s clinical_model=%s temperature=%.2f reasoning=%s",
            LLMRuntimeConfig.is_cloud_enabled(),
            LLMRuntimeConfig.get_llm_provider(),
            LLMRuntimeConfig.get_cloud_model(),
            parser_provider,
            parser_model,
            clinical_provider,
            clinical_model_resolved,
            LLMRuntimeConfig.get_ollama_temperature(),
            LLMRuntimeConfig.is_ollama_reasoning_enabled(),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def capture_runtime_snapshot() -> dict[str, Any]:
        return {
            "use_cloud_services": LLMRuntimeConfig.is_cloud_enabled(),
            "llm_provider": LLMRuntimeConfig.get_llm_provider(),
            "cloud_model": LLMRuntimeConfig.get_cloud_model(),
            "parsing_model": LLMRuntimeConfig.get_parsing_model(),
            "clinical_model": LLMRuntimeConfig.get_clinical_model(),
            "ollama_temperature": LLMRuntimeConfig.get_ollama_temperature(),
            "ollama_reasoning": LLMRuntimeConfig.is_ollama_reasoning_enabled(),
        }

    # -------------------------------------------------------------------------
    @contextmanager
    def runtime_override_context(
        self,
        *,
        use_cloud_services: bool | None,
        llm_provider: str | None,
        cloud_model: str | None,
        parsing_model: str | None,
        clinical_model: str | None,
        ollama_temperature: float | None,
        ollama_reasoning: bool | None,
    ):
        snapshot = self.capture_runtime_snapshot()
        self.apply_runtime_overrides(
            use_cloud_services=use_cloud_services,
            llm_provider=llm_provider,
            cloud_model=cloud_model,
            parsing_model=parsing_model,
            clinical_model=clinical_model,
            ollama_temperature=ollama_temperature,
            ollama_reasoning=ollama_reasoning,
        )
        try:
            yield
        finally:
            self.apply_runtime_overrides(
                use_cloud_services=bool(snapshot["use_cloud_services"]),
                llm_provider=str(snapshot["llm_provider"]),
                cloud_model=str(snapshot["cloud_model"]),
                parsing_model=str(snapshot["parsing_model"]),
                clinical_model=str(snapshot["clinical_model"]),
                ollama_temperature=float(snapshot["ollama_temperature"]),
                ollama_reasoning=bool(snapshot["ollama_reasoning"]),
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
            payload_data.update(
                {
                    "has_hepatic_diseases": request_payload.has_hepatic_diseases,
                }
            )
            return PatientData.model_validate(payload_data)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=self.serialize_validation_errors(exc.errors()),
            ) from exc

    # -------------------------------------------------------------------------
    async def process_single_patient(
        self,
        payload: PatientData,
        *,
        progress_callback: Callable[[str, float], None] | None = None,
        stop_check: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        if stop_check is not None:
            stop_check()
        logger.info(
            "Starting Drug-Induced Liver Injury (DILI) analysis for patient: %s",
            payload.name,
        )

        global_start_time = time.perf_counter()
        self.emit_progress(
            progress_callback,
            stage="session_initialization",
            value=5.0,
        )
        language_result = detect_clinical_language(payload)
        report_language = language_result.report_language
        validation_bundle = build_validation_bundle(report_language)
        ensure_required_sections(payload, bundle=validation_bundle)
        if stop_check is not None:
            stop_check()
        issues: list[PipelineIssue] = []
        cleaned_therapy_text = self.drugs_parser.clean_text(payload.drugs or "")

        self.emit_progress(
            progress_callback,
            stage="therapy_extraction",
            value=22.0,
        )
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
            if stop_check is not None:
                stop_check()
            elapsed = time.perf_counter() - start_time
            logger.info("Therapy drugs extraction required %.4f seconds", elapsed)
            logger.info("Detected %s drugs from therapy list", len(therapy_drugs.entries))
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
            issues.append(
                PipelineIssue(
                    severity="warning",
                    code="therapy_extraction_fallback",
                    message=(
                        "Therapy extraction via LLM was unavailable; "
                        "the analysis continued using the raw therapy list."
                    ),
                    field="drugs",
                )
            )
            therapy_drugs = self.build_fallback_therapy_drugs(cleaned_therapy_text)
        self.emit_progress(
            progress_callback,
            stage="therapy_extraction",
            value=30.0,
        )
        if stop_check is not None:
            stop_check()
        ensure_timed_therapy_drug(therapy_drugs, bundle=validation_bundle)

        self.emit_progress(
            progress_callback,
            stage="anamnesis_extraction",
            value=30.0,
        )
        anamnesis_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="anamnesis_extraction",
            start_value=30.0,
            end_value=42.0,
        )
        start_time = time.perf_counter()
        try:
            anamnesis_drugs = await self.drugs_parser.extract_drugs_from_anamnesis(
                payload.anamnesis,
                progress_callback=anamnesis_progress_callback,
            )
            if stop_check is not None:
                stop_check()
            elapsed = time.perf_counter() - start_time
            logger.info("Anamnesis drugs extraction required %.4f seconds", elapsed)
            logger.info("Detected %s drugs from anamnesis", len(anamnesis_drugs.entries))
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
            issues.append(
                PipelineIssue(
                    severity="warning",
                    code="anamnesis_extraction_failed",
                    message=(
                        "Drug extraction from anamnesis was unavailable; "
                        "the analysis continued without historical drug mentions."
                    ),
                    field="anamnesis",
                )
            )
            anamnesis_drugs = PatientDrugs(entries=[])
        self.emit_progress(
            progress_callback,
            stage="anamnesis_extraction",
            value=42.0,
        )
        if stop_check is not None:
            stop_check()

        self.emit_progress(
            progress_callback,
            stage="anamnesis_disease_extraction",
            value=42.0,
        )
        disease_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="anamnesis_disease_extraction",
            start_value=42.0,
            end_value=48.0,
        )
        start_time = time.perf_counter()
        try:
            disease_context = await self.disease_extractor.extract_diseases_from_anamnesis(
                payload.anamnesis,
                progress_callback=disease_progress_callback,
            )
            if stop_check is not None:
                stop_check()
            elapsed = time.perf_counter() - start_time
            logger.info("Anamnesis disease extraction required %.4f seconds", elapsed)
            logger.info("Detected %s diseases from anamnesis", len(disease_context.entries))
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
            issues.append(
                PipelineIssue(
                    severity="warning",
                    code="anamnesis_disease_extraction_failed",
                    message=(
                        "Disease extraction from anamnesis was unavailable; "
                        "the analysis continued without structured disease timeline."
                    ),
                    field="anamnesis",
                )
            )
            disease_context = PatientDiseaseContext(entries=[])
        self.emit_progress(
            progress_callback,
            stage="anamnesis_disease_extraction",
            value=48.0,
        )
        if stop_check is not None:
            stop_check()

        self.emit_progress(
            progress_callback,
            stage="anamnesis_lab_extraction",
            value=48.0,
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
            if stop_check is not None:
                stop_check()
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
            issues.append(
                PipelineIssue(
                    severity="warning",
                    code="anamnesis_lab_extraction_failed",
                    message=(
                        "Longitudinal lab extraction from anamnesis was unavailable; "
                        "the analysis continued without timeline enrichment."
                    ),
                    field="anamnesis",
                )
            )
            lab_timeline = PatientLabTimeline(entries=[])
            onset_context = None
        self.emit_progress(
            progress_callback,
            stage="anamnesis_lab_extraction",
            value=52.0,
        )
        if stop_check is not None:
            stop_check()

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
        pattern_score = pattern_assessment.score
        issues.extend(pattern_assessment.issues)
        logger.info(
            "Patient hepatotoxicity pattern classified as %s (R=%.3f, status=%s)",
            pattern_score.classification,
            pattern_score.r_score if pattern_score.r_score is not None else float("nan"),
            pattern_assessment.status,
        )
        self.emit_progress(
            progress_callback,
            stage="hepatotoxicity_pattern",
            value=54.0,
        )
        if stop_check is not None:
            stop_check()

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

        self.emit_progress(
            progress_callback,
            stage="rucam_estimation",
            value=52.0,
        )
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
            issues.append(
                PipelineIssue(
                    severity="warning",
                    code="rucam_estimation_failed",
                    message=(
                        "RUCAM estimation was unavailable; the analysis continued without "
                        "per-drug estimated RUCAM."
                    ),
                )
            )
            rucam_bundle = PatientRucamAssessmentBundle(entries=[])
        self.emit_progress(
            progress_callback,
            stage="rucam_estimation",
            value=54.0,
        )
        if stop_check is not None:
            stop_check()

        structured_context = self.build_structured_clinical_context(
            payload,
            therapy_drugs=therapy_drugs,
            anamnesis_drugs=anamnesis_drugs,
            disease_context=disease_context,
            lab_timeline=lab_timeline,
            onset_context=onset_context,
            pattern_score=pattern_score,
        )

        self.emit_progress(
            progress_callback,
            stage="rag_query_building",
            value=54.0,
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
        self.emit_progress(
            progress_callback,
            stage="rag_query_building",
            value=56.0,
        )
        if stop_check is not None:
            stop_check()

        self.emit_progress(
            progress_callback,
            stage="livertox_lookup",
            value=56.0,
        )
        livertox_progress_callback = self.build_stage_progress_callback(
            progress_callback,
            stage="livertox_lookup",
            start_value=56.0,
            end_value=62.0,
        )
        prepared_inputs = await input_preparator.prepare_inputs(
            all_detected_drugs,
            clinical_context=structured_context,
            pattern_score=pattern_score,
            progress_callback=livertox_progress_callback,
        )
        if stop_check is not None:
            stop_check()
        self.emit_progress(
            progress_callback,
            stage="livertox_lookup",
            value=62.0,
        )

        try:
            rucam_bundle = self.rucam_estimator.estimate(
                payload=payload,
                analysis_drugs=analysis_drugs,
                anamnesis_drugs=anamnesis_drugs,
                disease_context=disease_context,
                lab_timeline=lab_timeline,
                onset_context=onset_context,
                pattern_score=pattern_score,
                resolved_drugs=prepared_inputs.resolved_drugs if prepared_inputs else None,
            )
        except Exception as exc:
            logger.warning("RUCAM re-estimation with LiverTox metadata failed: %s", exc)
            issues.append(
                PipelineIssue(
                    severity="warning",
                    code="rucam_reestimate_failed",
                    message=(
                        "RUCAM refinement with matched LiverTox metadata failed; "
                        "using preliminary estimates."
                    ),
                )
            )

        start_time = time.perf_counter()
        clinical_session: HepatoxConsultation | None = None
        final_report: str | None = None
        try:
            clinical_session = HepatoxConsultation(analysis_drugs, patient_name=payload.name)
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
            if stop_check is not None:
                stop_check()
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
        except LLMError as exc:
            issues.append(
                PipelineIssue(
                    severity="warning",
                    code="clinical_llm_unavailable",
                    message=(
                        "Clinical LLM analysis is unavailable; report generated without "
                        "per-drug synthesis."
                    ),
                )
            )
            logger.warning(
                "Clinical LLM unavailable for patient '%s': %s",
                payload.name or "unknown",
                exc,
            )

        patient_label = payload.name or "Unknown patient"
        visit_label = (
            payload.visit_date.strftime("%d %B %Y")
            if payload.visit_date
            else ("Non disponibile" if report_language.startswith("it") else "Not provided")
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
        resolved_drug_map: dict[str, dict[str, Any]] = {}
        if prepared_inputs is not None:
            for key, value in prepared_inputs.resolved_drugs.items():
                normalized_key = normalize_drug_query_name(key)
                if normalized_key:
                    resolved_drug_map[normalized_key] = value
        rucam_by_name: dict[str, DrugRucamAssessment] = {}
        for item in rucam_bundle.entries:
            normalized_key = normalize_drug_query_name(item.drug_name)
            if normalized_key:
                rucam_by_name[normalized_key] = item
        matched_drugs_payload: list[dict[str, Any]] = []
        for detected_name in detected_drugs:
            normalized_name = normalize_drug_query_name(detected_name)
            resolved = resolved_drug_map.get(normalized_name, {})
            matched_row = (
                resolved.get("matched_livertox_row")
                if isinstance(resolved, dict)
                else None
            )
            matched_name = (
                matched_row.get("drug_name")
                if isinstance(matched_row, dict)
                else None
            )
            matched_drugs_payload.append(
                {
                    "raw_drug_name": detected_name,
                    "matched_drug_name": matched_name,
                    "nbk_id": matched_row.get("nbk_id")
                    if isinstance(matched_row, dict)
                    else None,
                    "match_confidence": resolved.get("match_confidence")
                    if isinstance(resolved, dict)
                    else None,
                    "match_reason": resolved.get("match_reason")
                    if isinstance(resolved, dict)
                    else None,
                    "match_notes": resolved.get("match_notes")
                    if isinstance(resolved, dict)
                    else [],
                    "match_status": resolved.get("match_status")
                    if isinstance(resolved, dict)
                    else None,
                    "match_candidates": resolved.get("match_candidates")
                    if isinstance(resolved, dict)
                    else [],
                    "chosen_candidate": resolved.get("chosen_candidate")
                    if isinstance(resolved, dict)
                    else None,
                    "rejected_candidates": resolved.get("rejected_candidates")
                    if isinstance(resolved, dict)
                    else [],
                    "missing_livertox": resolved.get("missing_livertox")
                    if isinstance(resolved, dict)
                    else True,
                    "ambiguous_match": resolved.get("ambiguous_match")
                    if isinstance(resolved, dict)
                    else False,
                    "regimen_group_ids": resolved.get("regimen_group_ids")
                    if isinstance(resolved, dict)
                    else [],
                    "regimen_components": resolved.get("regimen_components")
                    if isinstance(resolved, dict)
                    else [],
                    "origins": resolved.get("origins") if isinstance(resolved, dict) else [],
                    "raw_mentions": resolved.get("raw_mentions")
                    if isinstance(resolved, dict)
                    else [],
                    "rucam": rucam_by_name.get(normalized_name).model_dump()
                    if normalized_name in rucam_by_name
                    else None,
                }
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
                "therapy_drugs": [entry.model_dump() for entry in therapy_drugs.entries],
                "anamnesis_drugs": [entry.model_dump() for entry in anamnesis_drugs.entries],
                "anamnesis_diseases": [entry.model_dump() for entry in disease_context.entries],
            },
        }
        self.emit_progress(
            progress_callback,
            stage="finalization",
            value=96.0,
        )
        if stop_check is not None:
            stop_check()
        await asyncio.to_thread(
            self.serializer.save_clinical_session,
            {
                "patient_name": payload.name,
                "session_timestamp": datetime.now(),
                "hepatic_pattern": pattern_score.classification,
                "anamnesis": payload.anamnesis,
                "drugs": payload.drugs,
                "laboratory_analysis": payload.laboratory_analysis,
                "parsing_model": getattr(self.drugs_parser, "model", None),
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
        self.emit_progress(
            progress_callback,
            stage="finalization",
            value=99.0,
        )
        if stop_check is not None:
            stop_check()

        return result_payload

    # -------------------------------------------------------------------------
    async def start_clinical_session(
        self,
        request_payload: ClinicalSessionRequest = Body(...),
    ) -> PlainTextResponse:
        patient_payload = self.build_patient_payload(request_payload)
        try:
            with self.runtime_override_context(
                use_cloud_services=request_payload.use_cloud_services,
                llm_provider=request_payload.llm_provider,
                cloud_model=request_payload.cloud_model,
                parsing_model=request_payload.parsing_model,
                clinical_model=request_payload.clinical_model,
                ollama_temperature=request_payload.ollama_temperature,
                ollama_reasoning=request_payload.ollama_reasoning,
            ):
                single_result = await self.process_single_patient(patient_payload)
        except ClinicalPipelineValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=self.serialize_pipeline_issues(exc.issues),
            ) from exc
        report = str(single_result.get("report", "")).strip()
        return PlainTextResponse(content=report, status_code=status.HTTP_202_ACCEPTED)

    # -------------------------------------------------------------------------
    def start_clinical_job(
        self,
        request_payload: ClinicalSessionRequest = Body(...),
    ) -> JobStartResponse:
        if job_manager.is_job_running(self.JOB_TYPE):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Clinical analysis is already in progress",
            )

        patient_payload = self.build_patient_payload(request_payload)
        runtime_overrides = {
            "use_cloud_services": request_payload.use_cloud_services,
            "llm_provider": request_payload.llm_provider,
            "cloud_model": request_payload.cloud_model,
            "parsing_model": request_payload.parsing_model,
            "clinical_model": request_payload.clinical_model,
            "ollama_temperature": request_payload.ollama_temperature,
            "ollama_reasoning": request_payload.ollama_reasoning,
        }

        job_id = job_manager.start_job(
            job_type=self.JOB_TYPE,
            runner=run_clinical_job,
            kwargs={
                "payload": patient_payload,
                "runtime_overrides": runtime_overrides,
            },
        )

        job_status = job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize clinical analysis job",
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
        job_status = job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found.",
            )
        return JobStatusResponse(**job_status)

    # -------------------------------------------------------------------------
    def cancel_clinical_job(self, job_id: str) -> JobCancelResponse:
        job_status = job_manager.get_job_status(job_id)
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found.",
            )
        success = job_manager.cancel_job(job_id)
        if success:
            logger.info("Clinical analysis stop requested for job %s", job_id)
        return JobCancelResponse(
            job_id=job_id,
            success=success,
            message="Cancellation requested" if success else "Job cannot be cancelled",
        )
    
    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/clinical",
            self.start_clinical_session,
            methods=["POST"],
            response_model=None,
            status_code=status.HTTP_202_ACCEPTED,
            response_class=PlainTextResponse,
        )
        self.router.add_api_route(
            "/clinical/jobs",
            self.start_clinical_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_202_ACCEPTED,
        )
        self.router.add_api_route(
            "/clinical/jobs/{job_id}",
            self.get_clinical_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/clinical/jobs/{job_id}",
            self.cancel_clinical_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )


endpoint = ClinicalSessionEndpoint(
    router=router,
    drugs_parser=drugs_parser,
    disease_extractor=disease_extractor,
    lab_extractor=lab_extractor,
    pattern_analyzer=pattern_analyzer,
    rucam_estimator=rucam_estimator,
    serializer=serializer,
    payload_sanitizer=payload_sanitization_service,
)
endpoint.add_routes()

