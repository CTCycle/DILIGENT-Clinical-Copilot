from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import ValidationError

from DILIGENT.app.api.schemas.clinical import (
    PatientData,
)
from DILIGENT.app.constants import TASKS_PATH
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.services.clinical import (
    HepatotoxicityPatternAnalyzer,
    HepatoxConsultation,
)
from DILIGENT.app.utils.services.parser import (
    BloodTestParser,
    DiseasesParser,
    DrugsParser,
    PatientCase,
)
 
drugs_parser = DrugsParser()
diseases_parser = DiseasesParser()
pattern_analyzer = HepatotoxicityPatternAnalyzer()
router = APIRouter(tags=["agent"])

# [ENPOINTS]
###############################################################################
async def process_single_patient(payload: PatientData) -> dict[str, Any]:
    logger.info(
        "Starting Drug-Induced Liver Injury (DILI) analysis for patient: %s",
        payload.name,
    )
    if payload.visit_date:
        logger.info(
            "Clinical visit date: %s",
            payload.visit_date.strftime("%d-%m-%Y"),
        )

    # 1. Calculate hepatic pattern score using ALT/ALP values
    pattern_score = pattern_analyzer.analyze(payload)
    logger.info(
        "Patient hepatotoxicity pattern classified as %s (R=%.3f)",
        pattern_score.classification,
        pattern_score.r_score if pattern_score.r_score is not None else float("nan"),
    )

    # 2. Parse drugs names and info from raw text
    start_time = time.perf_counter()
    drug_data = drugs_parser.parse_drug_list(payload.drugs or "")
    elapsed = time.perf_counter() - start_time
    logger.info("Drugs extraction required %.4f seconds", elapsed)
    logger.info("Detected %s drugs", len(drug_data.entries))

    # 3. Optionally extract diseases from anamnesis for contextual analysis
    should_extract_diseases = bool(payload.pre_extract_diseases)
    if should_extract_diseases:
        start_time = time.perf_counter()
        diseases = await diseases_parser.extract_diseases(payload.anamnesis or "")
        elapsed = time.perf_counter() - start_time
        logger.info("Disease extraction required %.4f seconds", elapsed)
        logger.info("Detected %s diseases for this patient", len(diseases["diseases"]))
        logger.info(
            "Subset of hepatic diseases includes %s entries",
            len(diseases["hepatic_diseases"]),
        )
    else:
        diseases = {"diseases": [], "hepatic_diseases": []}
        logger.info("Disease extraction skipped based on request")

    # 4. Consult LiverTox database for hepatotoxicity info
    start_time = time.perf_counter()
    doctor = HepatoxConsultation(drug_data)
    drug_assessment = await doctor.run_analysis(
        anamnesis=payload.anamnesis,
        visit_date=payload.visit_date,
        diseases=diseases.get("diseases", []),
        hepatic_diseases=diseases.get("hepatic_diseases", []),
        diseases_pre_extracted=should_extract_diseases,
        pattern_score=pattern_score,
    )
    elapsed = time.perf_counter() - start_time
    logger.info("Drugs toxicity essay required %.4f seconds", elapsed)

    final_report: str | None = None
    if isinstance(drug_assessment, dict):
        candidate = drug_assessment.get("final_report")
        if isinstance(candidate, str) and candidate.strip():
            final_report = candidate.strip()
    elif isinstance(drug_assessment, str) and drug_assessment.strip():
        final_report = drug_assessment.strip()

    result: dict[str, Any] = {
        "status": "success",
        "code": "DILI_FINAL_REPORT",
        "final_report": final_report,
    }

    return result


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
    pre_extract_diseases: bool = Body(default=True),
) -> dict[str, Any]:
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
            "pre_extract_diseases": pre_extract_diseases,
        }
        payload = PatientData.model_validate(payload_data)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()
        ) from exc

    single_result = await process_single_patient(payload)
    return {"status": "success", "processed": 1, "patients": [single_result]}


# -----------------------------------------------------------------------------
@router.post("/batch-agent", response_model=None, status_code=status.HTTP_202_ACCEPTED)
async def start_batch_clinical_agent() -> dict[str, Any]:
    patient_case = PatientCase()
    lab_parser = BloodTestParser()

    txt_files = [path for path in Path(TASKS_PATH).glob("*.txt") if path.is_file()]
    if not txt_files:
        logger.info(
            "No .txt files found in default path. Add new files and rerun the batch agent."
        )
        return {"status": "success", "processed": 0, "patients": []}

    results: list[dict[str, Any]] = []
    for path in txt_files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error(f"Failed reading {path}: {exc}")
            continue

        cleaned_text = patient_case.clean_patient_info(text)
        sections = patient_case.split_text_by_tags(cleaned_text, path.stem)

        anamnesis_section = sections.get("anamnesis")
        drugs_section = sections.get("drugs")
        exams_section = sections.get("additional_tests")

        hepatic_markers = lab_parser.parse_hepatic_markers(sections.get("blood_tests"))

        try:
            patient_payload = PatientData(
                name=path.stem,
                anamnesis=anamnesis_section,
                drugs=drugs_section,
                exams=exams_section,
                alt=hepatic_markers["alt"],
                alt_max=hepatic_markers["alt_max"],
                alp=hepatic_markers["alp"],
                alp_max=hepatic_markers["alp_max"],
                symptoms=[],
            )
        except ValidationError as exc:
            logger.error(f"Invalid data for patient {path.stem}: {exc}")
            continue

        case = await process_single_patient(patient_payload)
        results.append(case)

    return {"status": "success", "processed": len(results), "patients": results}
