from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import ValidationError

from Pharmagent.app.utils.services.clinical import DrugToxicityEssay
from Pharmagent.app.utils.services.parser import (
    PatientCase,
    DiseasesParser,
    BloodTestParser,
    DrugsParser,
)
from Pharmagent.app.api.schemas.clinical import PatientData
from Pharmagent.app.constants import TASKS_PATH
from Pharmagent.app.logger import logger

router = APIRouter(tags=["agent"])

ALT_LABELS = {"ALT", "ALAT"}
ALP_LABELS = {"ALP"}


# [ENPOINTS]
###############################################################################
async def process_single_patient(payload: PatientData) -> dict[str, Any]:
    logger.info(
        f"Starting Drug-Induced Liver Injury (DILI) analysis for patient: {payload.name}"
        or "Unknown"
    )
    
    disease_parser = DiseasesParser()   
    drugs_parser = DrugsParser()    

    start_time = time.perf_counter()
    drug_data = drugs_parser.parse_drug_list(payload.drugs)
    elapsed = time.perf_counter() - start_time
    logger.info(f"Drugs extraction required {elapsed:.4f} seconds")
    logger.info(f"Detected {len(drug_data.entries)} drugs")

    start_time = time.perf_counter()
    diseases = await disease_parser.extract_diseases(payload.anamnesis)
    elapsed = time.perf_counter() - start_time
    logger.info(f"Disease extraction required {elapsed:.4f} seconds")  
    logger.info(f"Detected {len(diseases["diseases"])} diseases for this patient")
    logger.info(f"Subset of hepatic diseases includes {len(diseases["hepatic_disease"])} entries")

    pharmacology = DrugToxicityEssay(drug_data)  

    return {
        "name": payload.name or "Unknown",
        "anamnesis": payload.anamnesis,
        "diseases": diseases,
        "drugs": drug_data.model_dump(),       
        "exams": payload.exams,
        "alt": payload.alt,
        "alt_max": payload.alt_max,
        "alp": payload.alp,
        "alp_max": payload.alp_max,
        "symptoms": payload.symptoms,
    }


# -----------------------------------------------------------------------------
@router.post("/agent", response_model=None, status_code=status.HTTP_202_ACCEPTED)
async def start_single_clinical_agent(
    name: str | None = Body(default=None),
    anamnesis: str | None = Body(default=None),
    drugs: str | None = Body(default=None),
    exams: str | None = Body(default=None),
    alt: str | None = Body(default=None),
    alt_max: str | None = Body(default=None),
    alp: str | None = Body(default=None),
    alp_max: str | None = Body(default=None),
    symptoms: list[str] | None = Body(default=None),
) -> dict[str, Any]:
    try:
        payload = PatientData(
            name=name,
            anamnesis=anamnesis,
            drugs=drugs,
            exams=exams,
            alt=alt,
            alt_max=alt_max,
            alp=alp,
            alp_max=alp_max,
            symptoms=symptoms or [],
        )
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
