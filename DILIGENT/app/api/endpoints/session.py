from __future__ import annotations

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
router = APIRouter(tags=["agent"])
serializer = DataSerializer()



# [HELPERS]
###############################################################################
def build_patient_narrative(
    *,
    patient_label: str,
    visit_label: str,
    pattern_score,
    pattern_strings: dict[str, str],
    detected_drugs: list[str],
    final_report: str | None,
) -> str:
    """Render a plain-text narrative for the patient summary."""
    drug_summary = ", ".join(detected_drugs) if detected_drugs else "None detected"

    lines: list[str] = [
        "Patient Summary",
        "---------------",
        f"Name: {patient_label}",
        f"Visit date: {visit_label}",
        "",
        "Hepatotoxicity Pattern",
        "-----------------------",
        f"Classification: {pattern_score.classification}",
        f"ALT multiple: {pattern_strings.get('alt_multiple', 'N/A')}",
        f"ALP multiple: {pattern_strings.get('amp_multiple', 'N/A')}",
        f"R-score: {pattern_strings.get('r_score', 'N/A')}",
        "",
        "Medications",
        "-----------",
        f"Detected drugs ({len(detected_drugs)}): {drug_summary}",
    ]

    if final_report:
        lines.extend(
            [
                "",
                "Clinical Assessment",
                "--------------------",
                final_report,
            ]
        )

    return "\n".join(lines)


# [ENPOINTS]
###############################################################################
async def process_single_patient(payload: PatientData) -> str:
    logger.info(
        "Starting Drug-Induced Liver Injury (DILI) analysis for patient: %s",
        payload.name,
    )
    
    global_start_time = time.perf_counter()

    # 1. Calculate hepatic pattern score using ALT/ALP values
    pattern_score = pattern_analyzer.calculate_hepatotoxicity_pattern(payload)
    logger.info(
        "Patient hepatotoxicity pattern classified as %s (R=%.3f)",
        pattern_score.classification,
        pattern_score.r_score if pattern_score.r_score is not None else float("nan"),
    )
    

    # 2. Parse drugs names and info from raw text
    start_time = time.perf_counter()
    drug_data = await drugs_parser.extract_drug_list(payload.drugs or "")
    elapsed = time.perf_counter() - start_time
    logger.info("Drugs extraction required %.4f seconds", elapsed)
    logger.info("Detected %s drugs", len(drug_data.entries))

    # 3. Consult LiverTox database for hepatotoxicity info
    start_time = time.perf_counter()
    doctor = HepatoxConsultation(drug_data, patient_name=payload.name)
    drug_assessment = await doctor.run_analysis(
        anamnesis=payload.anamnesis,
        exams=payload.exams,
        visit_date=payload.visit_date,
        pattern_score=pattern_score,
    )
    elapsed = time.perf_counter() - start_time
    logger.info("Hepato-toxicity consultation required %.4f seconds", elapsed)

    final_report: str | None = None
    
    if isinstance(drug_assessment, dict):
        candidate : str = drug_assessment.get("final_report", "")
        final_report = candidate.strip()
   

    patient_label = payload.name or "Unknown patient"
    visit_label = (
        payload.visit_date.strftime("%d %B %Y")
        if payload.visit_date
        else "Not provided"
    )   
    
    global_elapsed = time.perf_counter() - global_start_time
    logger.info("Total time for Drug Induced Liver Injury (DILI) assessment is %.4f seconds", global_elapsed)

    # 4. Serialize session data to the database
    detected_drugs = [entry.name for entry in drug_data.entries if entry.name]
    drug_summary = ", ".join(detected_drugs) if detected_drugs else "None detected"
    pattern_strings = pattern_analyzer.stringify_scores(pattern_score)
    serializer.record_clinical_session(
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
            "exams": payload.exams,
            "parsing_model": getattr(drugs_parser, "model", None),
            "clinical_model": getattr(doctor, "llm_model", None),
            "total_duration": global_elapsed,
            "final_report": final_report,
        }
    )

    narrative = build_patient_narrative(
        patient_label=patient_label,
        visit_label=visit_label,
        pattern_score=pattern_score,
        pattern_strings=pattern_strings,
        detected_drugs=detected_drugs,
        final_report=final_report,
    )

    return narrative

# -----------------------------------------------------------------------------
@router.post(
    "/agent",
    response_model=None,
    status_code=status.HTTP_202_ACCEPTED,
    response_class=PlainTextResponse,
)
async def start_single_clinical_agent(
    name: str | None = Body(default=None),
    visit_date: date | dict[str, int] | str | None = Body(default=None),
    anamnesis: str | None = Body(default=None),
    has_hepatic_diseases: bool = Body(default=False),
    drugs: str | None = Body(default=None),
    exams: str | None = Body(default=None),
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
            "drugs": drugs,
            "exams": exams,
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

