from __future__ import annotations

import asyncio
import time
from datetime import datetime
from functools import partial
from typing import Any

from domain.clinical.entities import (
    ClinicalSectionExtractionResult,
    ClinicalPipelineValidationError,
    DrugRucamAssessment,
    PatientData,
    PipelineIssue,
)
from configurations.llm_configs import LLMRuntimeConfig
from services.runtime.jobs import JobManager
from services.clinical.job_progress import (
    CLINICAL_PROGRESS_MESSAGES,
    ClinicalJobCancelled,
)
from services.clinical.language import ClinicalLanguageDetector

NOT_AVAILABLE = "Not available"
PATIENT_LINE_TEMPLATE = "- **Patient:** {value}"


###############################################################################
def build_failed_session_payload(
    *,
    payload: PatientData,
    patient_image_base64: str | None,
    issues: list[dict[str, Any]],
    error_message: str,
    elapsed_seconds: float,
    section_extraction: ClinicalSectionExtractionResult | None = None,
) -> dict[str, Any]:
    language_result = ClinicalLanguageDetector.detect(payload)
    runtime_settings = {
        "use_cloud_services": bool(LLMRuntimeConfig.is_cloud_enabled()),
        "llm_provider": LLMRuntimeConfig.get_llm_provider(),
        "cloud_model": LLMRuntimeConfig.get_cloud_model(),
        "text_extraction_model": LLMRuntimeConfig.get_text_extraction_model(),
        "clinical_model": LLMRuntimeConfig.get_clinical_model(),
        "ollama_temperature": (LLMRuntimeConfig.get_ollama_temperature()),
        "cloud_temperature": (LLMRuntimeConfig.get_cloud_temperature()),
        "ollama_reasoning": bool(LLMRuntimeConfig.is_ollama_reasoning_enabled()),
    }
    return {
        "patient_name": payload.name,
        "patient_visit_date": payload.visit_date,
        "patient_image_base64": patient_image_base64,
        "session_timestamp": datetime.now(),
        "hepatic_pattern": "indeterminate",
        "anamnesis": payload.anamnesis,
        "drugs": payload.drugs,
        "laboratory_analysis": payload.laboratory_analysis,
        "text_extraction_model": LLMRuntimeConfig.get_text_extraction_model(),
        "clinical_model": LLMRuntimeConfig.get_clinical_model(),
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
            "detected_input_language": language_result.detected_input_language,
            "report_language": language_result.report_language,
            "relevant_drugs": [],
            "excluded_drugs": [],
            "unresolved_drugs": [],
            "structured_case": {},
            "section_extraction": (
                section_extraction.model_dump() if section_extraction is not None else None
            ),
            "runtime_settings": runtime_settings,
        },
    }


###############################################################################
class NarrativeBuilder:
    BUNDLES: dict[str, dict[str, str]] = {
        "en": {
            "no_data": "- No data provided.",
            "summary_title": "# Clinical Visit Summary",
            "patient": PATIENT_LINE_TEMPLATE,
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
            "rucam_title": "## Estimated RUCAM",
            "rucam_item": "- **{drug}**: score {score} ({category}, confidence {confidence})",
            "warnings_title": "## Warnings",
            "consistency_warning": "- Structured RUCAM classifies {drugs} as excluded despite their inclusion in the active analysis set. Review before clinical use.",
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
            "rucam_title": "## RUCAM Stimato",
            "rucam_item": "- **{drug}**: punteggio {score} ({category}, confidenza {confidence})",
            "warnings_title": "## Avvisi",
            "consistency_warning": "- Il RUCAM strutturato classifica {drugs} come esclusi nonostante siano inclusi nell'analisi attiva. Verificare prima dell'uso clinico.",
            "report_title": "## Report Clinico",
            "no_report": "Nessun report clinico generato.",
            "none_detected": "Nessuno rilevato",
        },
        "de": {
            "no_data": "- Keine Daten angegeben.",
            "summary_title": "# Zusammenfassung des Klinischen Besuchs",
            "patient": PATIENT_LINE_TEMPLATE,
            "visit_date": "- **Besuchsdatum:** {value}",
            "anamnesis_title": "## Anamnese",
            "no_anamnesis": "_Keine Anamnese angegeben._",
            "pattern_title": "## Hepatotoxizitätsmuster",
            "classification": "- **Klassifikation:** {value}",
            "r_score": "- **R-Score:** {value}",
            "therapy_title": "## Aktuelle Medikamente",
            "detected_drugs": "**Erkannte Medikamente ({count}):** {value}",
            "historical_title": "## Historische Medikamentenerwähnungen",
            "historical_mentions": "- **Historische Erwähnungen ({count}):** {value}",
            "rucam_title": "## Geschätzter RUCAM",
            "rucam_item": "- **{drug}**: Score {score} ({category}, Vertrauen {confidence})",
            "warnings_title": "## Warnhinweise",
            "consistency_warning": "- Der strukturierte RUCAM stuft {drugs} als ausgeschlossen ein, obwohl sie in der aktiven Analyse enthalten sind. Vor klinischer Verwendung prüfen.",
            "report_title": "## Klinischer Bericht",
            "no_report": "Kein klinischer Bericht erstellt.",
            "none_detected": "Keine erkannt",
        },
        "fr": {
            "no_data": "- Aucune donnée fournie.",
            "summary_title": "# Résumé de la Visite Clinique",
            "patient": PATIENT_LINE_TEMPLATE,
            "visit_date": "- **Date de visite:** {value}",
            "anamnesis_title": "## Anamnèse",
            "no_anamnesis": "_Aucune anamnèse fournie._",
            "pattern_title": "## Profil d’Hépatotoxicité",
            "classification": "- **Classification:** {value}",
            "r_score": "- **Score R:** {value}",
            "therapy_title": "## Médicaments Actuels",
            "detected_drugs": "**Médicaments détectés ({count}) :** {value}",
            "historical_title": "## Mentions Médicamenteuses Antérieures",
            "historical_mentions": "- **Mentions historiques ({count}) :** {value}",
            "rucam_title": "## RUCAM Estimé",
            "rucam_item": "- **{drug}** : score {score} ({category}, confiance {confidence})",
            "warnings_title": "## Avertissements",
            "consistency_warning": "- Le RUCAM structuré classe {drugs} comme exclus alors qu'ils figurent dans l'analyse active. Vérifier avant usage clinique.",
            "report_title": "## Rapport Clinique",
            "no_report": "Aucun rapport clinique généré.",
            "none_detected": "Aucun détecté",
        },
        "es": {
            "no_data": "- No se proporcionaron datos.",
            "summary_title": "# Resumen de la Visita Clínica",
            "patient": "- **Paciente:** {value}",
            "visit_date": "- **Fecha de visita:** {value}",
            "anamnesis_title": "## Anamnesis",
            "no_anamnesis": "_No se proporcionó anamnesis._",
            "pattern_title": "## Patrón de Hepatotoxicidad",
            "classification": "- **Clasificación:** {value}",
            "r_score": "- **Puntuación R:** {value}",
            "therapy_title": "## Fármacos Actuales",
            "detected_drugs": "**Fármacos detectados ({count}):** {value}",
            "historical_title": "## Menciones Históricas de Fármacos",
            "historical_mentions": "- **Menciones históricas ({count}):** {value}",
            "rucam_title": "## RUCAM Estimado",
            "rucam_item": "- **{drug}**: puntuación {score} ({category}, confianza {confidence})",
            "warnings_title": "## Advertencias",
            "consistency_warning": "- El RUCAM estructurado clasifica {drugs} como excluidos aunque están incluidos en el análisis activo. Revisar antes del uso clínico.",
            "report_title": "## Informe Clínico",
            "no_report": "No se generó un informe clínico.",
            "none_detected": "Ninguno detectado",
        },
    }

    # -------------------------------------------------------------------------
    @staticmethod
    def bundle(report_language: str) -> dict[str, str]:
        if report_language.startswith("de"):
            return NarrativeBuilder.BUNDLES["de"]
        if report_language.startswith("fr"):
            return NarrativeBuilder.BUNDLES["fr"]
        if report_language.startswith("es"):
            return NarrativeBuilder.BUNDLES["es"]
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
        rucam_assessments: list[DrugRucamAssessment] | None = None,
        report_language: str,
        issues: list[PipelineIssue],
        final_report: str | None,
    ) -> str:
        bundle = NarrativeBuilder.bundle(report_language)
        classification = getattr(pattern_score, "classification", NOT_AVAILABLE)
        r_score = pattern_strings.get("r_score", NOT_AVAILABLE)
        drug_summary = (
            ", ".join(detected_drugs) if detected_drugs else bundle["none_detected"]
        )

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
                bundle["detected_drugs"].format(
                    count=len(detected_drugs), value=drug_summary
                ),
            ]
        )
        sections.append("\n".join(therapy_section))

        anamnesis_drug_summary = (
            ", ".join(anamnesis_detected_drugs)
            if anamnesis_detected_drugs
            else bundle["none_detected"]
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
        if rucam_assessments:
            rucam_section = [bundle["rucam_title"], ""]
            for assessment in rucam_assessments:
                rucam_section.append(
                    bundle["rucam_item"].format(
                        drug=assessment.drug_name,
                        score=assessment.total_score,
                        category=assessment.causality_category,
                        confidence=assessment.confidence,
                    )
                )
            sections.append("\n".join(rucam_section))

        if issues:
            warnings_section = [bundle["warnings_title"], ""]
            for issue in issues:
                warnings_section.append(f"- {issue.message}")
            sections.append("\n".join(warnings_section))

        excluded_active_drugs = [
            assessment.drug_name
            for assessment in (rucam_assessments or [])
            if assessment.causality_category == "excluded"
            and assessment.drug_name in detected_drugs
        ]
        if excluded_active_drugs:
            sections.append(
                "\n".join(
                    [
                        bundle["warnings_title"],
                        "",
                        bundle["consistency_warning"].format(
                            drugs=", ".join(excluded_active_drugs)
                        ),
                    ]
                )
            )

        clinical_report_section = [bundle["report_title"], ""]
        clinical_report_section.append(
            final_report.strip() if final_report else bundle["no_report"]
        )
        sections.append("\n".join(clinical_report_section))

        return NarrativeBuilder.compact_spacing("\n\n".join(sections))


###############################################################################
async def execute_clinical_job(
    service: Any,
    payload: PatientData,
    patient_image_base64: str | None,
    job_id: str,
    section_extraction: ClinicalSectionExtractionResult | None = None,
) -> dict[str, Any]:
    service.apply_persisted_runtime_configuration()
    ensure_not_cancelled = partial(
        ensure_clinical_job_not_cancelled,
        job_manager=service.job_manager,
        job_id=job_id,
    )
    report_progress = partial(
        report_clinical_job_progress,
        job_manager=service.job_manager,
        job_id=job_id,
    )

    ensure_not_cancelled()

    report_progress(stage="session_initialization", progress=5.0)
    progress_callback = report_progress
    job_started_at = time.perf_counter()

    try:
        result = await service.process_single_patient(
            payload,
            patient_image_base64=patient_image_base64,
            section_extraction=section_extraction,
            progress_callback=progress_callback,
            stop_check=ensure_not_cancelled,
        )
    except ClinicalJobCancelled:
        service.job_manager.update_result(
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
            service.serializer.save_clinical_session,
            build_failed_session_payload(
                payload=payload,
                patient_image_base64=patient_image_base64,
                issues=serialized_issues,
                error_message=str(exc),
                elapsed_seconds=(time.perf_counter() - job_started_at),
                section_extraction=section_extraction,
            ),
        )
        service.job_manager.update_result(
            job_id,
            {
                "validation_error": str(exc),
                "issues": serialized_issues,
            },
        )
        raise
    except Exception as exc:
        if service.job_manager.should_stop(job_id):
            return {}
        failure_issue = PipelineIssue(
            severity="error",
            code="clinical_job_failed",
            message=str(exc).strip() or "Clinical analysis failed unexpectedly.",
        ).model_dump()
        await asyncio.to_thread(
            service.serializer.save_clinical_session,
            build_failed_session_payload(
                payload=payload,
                patient_image_base64=patient_image_base64,
                issues=[failure_issue],
                error_message=str(exc),
                elapsed_seconds=(time.perf_counter() - job_started_at),
                section_extraction=section_extraction,
            ),
        )
        raise
    return result


###############################################################################
def ensure_clinical_job_not_cancelled(*, job_manager: JobManager, job_id: str) -> None:
    if job_manager.should_stop(job_id):
        raise ClinicalJobCancelled("Clinical job stop requested.")


###############################################################################
def report_clinical_job_progress(
    stage: str,
    progress: float,
    *,
    job_manager: JobManager,
    job_id: str,
) -> None:
    ensure_clinical_job_not_cancelled(job_manager=job_manager, job_id=job_id)
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
def run_clinical_job(
    service: Any,
    payload: PatientData,
    patient_image_base64: str | None,
    job_id: str,
    section_extraction: ClinicalSectionExtractionResult | None = None,
) -> dict[str, Any]:
    result = asyncio.run(
        execute_clinical_job(
            service=service,
            payload=payload,
            patient_image_base64=patient_image_base64,
            job_id=job_id,
            section_extraction=section_extraction,
        )
    )
    if not result:
        return {}
    return result

