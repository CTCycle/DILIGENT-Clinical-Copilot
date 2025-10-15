from __future__ import annotations

import time
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Body, HTTPException, status
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

# [ENPOINTS]
###############################################################################
async def process_single_patient(payload: PatientData) -> str:
    logger.info(
        "Starting Drug-Induced Liver Injury (DILI) analysis for patient: %s",
        payload.name,
    )
    if payload.visit_date:
        logger.info(
            "Clinical visit date: %s",
            payload.visit_date.strftime("%d-%m-%Y"),
        )

    global_start_time = time.perf_counter()

    # 1. Calculate hepatic pattern score using ALT/ALP values
    pattern_score = pattern_analyzer.analyze(payload)
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
        candidate = drug_assessment.get("final_report")
        if isinstance(candidate, str) and candidate.strip():
            final_report = candidate.strip()
    elif isinstance(drug_assessment, str) and drug_assessment.strip():
        final_report = drug_assessment.strip()

    patient_label = payload.name or "Unknown patient"
    visit_label = (
        payload.visit_date.strftime("%d %B %Y")
        if payload.visit_date
        else "Not provided"
    )

    if pattern_score.alt_multiple is not None:
        alt_multiple = f"{pattern_score.alt_multiple:.2f}x ULN"
    else:
        alt_multiple = "Not available"
    if pattern_score.alp_multiple is not None:
        alp_multiple = f"{pattern_score.alp_multiple:.2f}x ULN"
    else:
        alp_multiple = "Not available"
    if pattern_score.r_score is not None:
        r_score_line = f"{pattern_score.r_score:.2f}"
    else:
        r_score_line = "Not available"

    detected_drugs = [entry.name for entry in drug_data.entries if entry.name]
    drug_summary = ", ".join(detected_drugs) if detected_drugs else "None detected"
    
    global_elapsed = time.perf_counter() - global_start_time
    logger.info("Total time for Drug Induced Liver Injury (DILI) assessment is %.4f seconds", global_elapsed)

    try:
        serializer.record_clinical_session(
            {
                "patient_name": payload.name,
                "session_timestamp": datetime.utcnow(),
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
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to record clinical session: %s", exc)

    narrative: list[str] = [
        "Patient Summary",
        "---------------",
        f"Name: {patient_label}",
        f"Visit date: {visit_label}",
        "",
        "Hepatotoxicity Pattern",
        "-----------------------",
        f"Classification: {pattern_score.classification}",
        f"ALT multiple: {alt_multiple}",
        f"ALP multiple: {alp_multiple}",
        f"R-score: {r_score_line}",
        "",
        "Medications",
        "-----------",
        f"Detected drugs ({len(detected_drugs)}): {drug_summary}",
    ]

    if final_report:
        narrative.extend([
            "",
            "Clinical Assessment",
            "--------------------",
            final_report,
        ])

    return "\n".join(narrative)

# -----------------------------------------------------------------------------
@router.post("/agent", response_model=None, status_code=status.HTTP_202_ACCEPTED)
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
) -> str:
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
    return single_result


# -----------------------------------------------------------------------------
