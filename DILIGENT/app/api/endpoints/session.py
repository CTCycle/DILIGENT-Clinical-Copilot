from __future__ import annotations

import asyncio
import time
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError

from DILIGENT.app.api.schemas.clinical import (
    PatientData,
)
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.serializer import DataSerializer
from DILIGENT.app.utils.services.clinical import (
    HepatotoxicityPatternAnalyzer,
    HepatoxConsultation,
)
from DILIGENT.app.utils.services.parser import (
    DrugsParser,
)

drugs_parser = DrugsParser()
pattern_analyzer = HepatotoxicityPatternAnalyzer()
router = APIRouter(tags=["session"])
serializer = DataSerializer()


# [HELPERS]
###############################################################################
def build_bullet_list(content: str | None) -> list[str]:
    lines: list[str] = []
    if content:
        for entry in content.splitlines():
            stripped = entry.strip()
            if stripped:
                lines.append(f"- {stripped}")
    if not lines:
        lines.append("- No data provided.")
    return lines


def build_patient_narrative(
    *,
    patient_label: str,
    visit_label: str,
    anamnesis: str | None,
    drugs_text: str | None,
    pattern_score,
    pattern_strings: dict[str, str],
    detected_drugs: list[str],
    final_report: str | None,
) -> str:
    classification = getattr(pattern_score, "classification", "Not available")
    alt_multiple = pattern_strings.get("alt_multiple", "Not available")
    alp_multiple = pattern_strings.get("alp_multiple", "Not available")
    r_score = pattern_strings.get("r_score", "Not available")
    drug_summary = ", ".join(detected_drugs) if detected_drugs else "None detected"

    sections: list[str] = []

    header_section = [
        "# Clinical Visit Summary",
        "",
        f"- **Patient:** {patient_label}",
        f"- **Visit date:** {visit_label}",
    ]
    sections.append("\n".join(header_section))

    anamnesis_content = anamnesis if anamnesis else "_No anamnesis provided._"
    sections.append("\n".join(["## Anamnesis and Exams", "", anamnesis_content]))

    pattern_section = [
        "## Hepato-toxicity Pattern",
        "",
        f"- **Classification:** {classification}",
        f"- **ALT multiple:** {alt_multiple}",
        f"- **ALP multiple:** {alp_multiple}",
        f"- **R-score:** {r_score}",
    ]
    sections.append("\n".join(pattern_section))

    therapy_section = ["## Pharmacological Therapy", ""]
    therapy_section.extend(build_bullet_list(drugs_text))
    therapy_section.extend(
        [
            "",
            f"**Detected drugs ({len(detected_drugs)}):** {drug_summary}",
        ]
    )
    sections.append("\n".join(therapy_section))

    clinical_report_section = ["## Clinical Report (LLM Output)", ""]
    clinical_report_section.append(
        final_report.strip() if final_report else "_No clinical report generated._"
    )
    sections.append("\n".join(clinical_report_section))

    return "\n\n".join(sections)


# [ENPOINTS]
###############################################################################
async def process_single_patient(payload: PatientData) -> str:
    if payload.anamnesis is None or payload.drugs is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Both anamnesis and drugs fields are required.",
        )

    logger.info(
        "Starting Drug-Induced Liver Injury (DILI) analysis for patient: %s",
        payload.name,
    )

    global_start_time = time.perf_counter()

    # Step 1: Calculate hepatic pattern score using ALT/ALP values
    pattern_score = pattern_analyzer.calculate_hepatotoxicity_pattern(payload)
    logger.info(
        "Patient hepatotoxicity pattern classified as %s (R=%.3f)",
        pattern_score.classification,
        pattern_score.r_score if pattern_score.r_score is not None else float("nan"),
    )

    # Step 2: Parse drug names and metadata from the raw text list
    start_time = time.perf_counter()
    drug_data = await drugs_parser.extract_drug_list(payload.drugs or "")
    elapsed = time.perf_counter() - start_time
    logger.info("Drugs extraction required %.4f seconds", elapsed)
    logger.info("Detected %s drugs", len(drug_data.entries))

    clinical_session = HepatoxConsultation(drug_data, patient_name=payload.name)
    drug_assessment = await clinical_session.run_analysis(
        clinical_context=payload.anamnesis,
        visit_date=payload.visit_date,
        pattern_score=pattern_score,
    )
    elapsed = time.perf_counter() - start_time
    logger.info("Hepato-toxicity consultation required %.4f seconds", elapsed)

    final_report: str | None = None
    if isinstance(drug_assessment, dict):
        final_report: str | None = drug_assessment.get("final_report", "").strip()

    patient_label = payload.name or "Unknown patient"
    visit_label = (
        payload.visit_date.strftime("%d %B %Y")
        if payload.visit_date
        else "Not provided"
    )

    global_elapsed = time.perf_counter() - global_start_time
    logger.info(
        "Total time for Drug Induced Liver Injury (DILI) assessment is %.4f seconds",
        global_elapsed,
    )

    # Step 3: Persist a structured representation of the session to SQLite
    detected_drugs = [entry.name for entry in drug_data.entries if entry.name]
    pattern_strings = pattern_analyzer.stringify_scores(pattern_score)
    await asyncio.to_thread(
        serializer.record_clinical_session,
        {
            "patient_name": payload.name,
            "session_timestamp": datetime.now(),
            "alt_value": payload.alt,
            "alt_upper_limit": payload.alt_max,
            "alp_value": payload.alp,
            "alp_upper_limit": payload.alp_max,
            "hepatic_pattern": pattern_score.classification,
            "anamnesis": payload.anamnesis,
            "drugs": payload.drugs,
            "parsing_model": getattr(drugs_parser, "model", None),
            "clinical_model": getattr(clinical_session, "llm_model", None),
            "total_duration": global_elapsed,
            "final_report": final_report,
        },
    )

    narrative = build_patient_narrative(
        patient_label=patient_label,
        visit_label=visit_label,
        anamnesis=payload.anamnesis,
        drugs_text=payload.drugs,
        pattern_score=pattern_score,
        pattern_strings=pattern_strings,
        detected_drugs=detected_drugs,
        final_report=final_report,
    )

    return narrative


# -----------------------------------------------------------------------------
@router.post(
    "/clinical",
    response_model=None,
    status_code=status.HTTP_202_ACCEPTED,
    response_class=PlainTextResponse,
)
async def start_clinical_session(
    name: str | None = Body(default=None),
    visit_date: date | dict[str, int] | str | None = Body(default=None),
    anamnesis: str | None = Body(default=None),
    has_hepatic_diseases: bool = Body(default=False),
    use_rag: bool = Body(default=False),
    drugs: str | None = Body(default=None),
    alt: str | None = Body(default=None),
    alt_max: str | None = Body(default=None),
    alp: str | None = Body(default=None),
    alp_max: str | None = Body(default=None),
    symptoms: list[str] | None = Body(default=None),
) -> PlainTextResponse:
    try:
        payload_data: dict[str, Any] = {
            "name": name,
            "visit_date": visit_date,
            "anamnesis": anamnesis,
            "has_hepatic_diseases": has_hepatic_diseases,
            "use_rag": use_rag,
            "drugs": drugs,
            "alt": alt,
            "alt_max": alt_max,
            "alp": alp,
            "alp_max": alp_max,
            "symptoms": symptoms or [],
        }
        payload = PatientData.model_validate(payload_data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()
        ) from exc

    single_result = await process_single_patient(payload)
    return PlainTextResponse(content=single_result)
