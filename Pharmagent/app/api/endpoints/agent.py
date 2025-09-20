from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from Pharmagent.app.api.schemas.clinical import PatientData
from Pharmagent.app.logger import logger
from Pharmagent.app.utils.serializer import DataSerializer
from Pharmagent.app.utils.services.parser import (
    BloodTestParser,
    DiseasesParser,
    PatientCase,
)
from Pharmagent.app.constants import TASKS_PATH

router = APIRouter(tags=["agent"])

patient = PatientCase()
serializer = DataSerializer()
disease_parser = DiseasesParser()
test_parser = BloodTestParser()


# ----------------------------------------------------------------------------
async def process_single_patient(single_payload: PatientData) -> dict[str, Any]:
    structured_text = single_payload.compose_structured_text()
    if structured_text is None:
        raise ValueError("No clinical data provided to process.")

    working_payload = (
        single_payload
        if structured_text == single_payload.info
        else single_payload.model_copy(update={"info": structured_text})
    )

    logger.info("Processing data save new patient to database")
    sections, patient_table = await run_in_threadpool(
        patient.extract_sections_from_text, working_payload
    )
    await run_in_threadpool(serializer.save_patients_info, patient_table)

    logger.info(
        f"Extracting blood tests analysis from lab results using {test_parser.model}"
    )
    start_time = time.time()
    blood_test_results = await test_parser.extract_blood_test_results(
        sections.get("blood_tests") or ""
    )
    bt_elapsed = time.time() - start_time
    logger.info(f"Time elapsed for blood tests extraction: {bt_elapsed:.2f} seconds.")

    hepatic_inputs: dict[str, Any] = test_parser.extract_hepatic_markers(
        blood_test_results
    )
    manual_markers = single_payload.manual_hepatic_markers()
    for marker, data in manual_markers.items():
        existing = hepatic_inputs.get(marker, {})
        merged: dict[str, Any] = {**existing}
        for key, value in data.items():
            if value is not None:
                merged[key] = value
        hepatic_inputs[marker] = merged

    is_valid_patient = ("ALAT" in hepatic_inputs) and ("ALP" in hepatic_inputs)
    if is_valid_patient:
        logger.info(
            f"Extracting diseases from patient anamnesis using {disease_parser.model}"
        )
        start_time = time.time()
        diseases = await disease_parser.extract_diseases(
            sections.get("anamnesis", None)
        )
        dis_elapsed = time.time() - start_time
        logger.info(
            f"Time elapsed for diseases extraction: {dis_elapsed:.2f} seconds."
        )
    else:
        diseases = None
        dis_elapsed = 0.0
        logger.info(
            "Hepatic inputs not available (ALAT and/or ALP missing). Skipping disease extraction."
        )

    return {
        "name": single_payload.name or "Unknown",
        "symptoms": single_payload.symptoms,
        "is_valid": is_valid_patient,
        "diseases": diseases or {},
        "blood_tests": blood_test_results.model_dump()
        if hasattr(blood_test_results, "model_dump")
        else blood_test_results.__dict__,
        "hepatic_inputs": hepatic_inputs,
        "timings": {"diseases_s": dis_elapsed, "blood_tests_s": bt_elapsed},
    }


###############################################################################
@router.post("/agent", response_model=None, status_code=status.HTTP_202_ACCEPTED)
async def start_single_clinical_agent(payload: PatientData) -> dict[str, Any]:
    logger.info(
        f"Starting clinical agent processing for patient: {payload.name or 'Unknown'}"
    )

    try:
        single_result = await process_single_patient(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    if not single_result.get("is_valid", False):
        return {
            "status": "unsuccess",
            "processed": 0,
            "reason": "Required hepatic inputs (ALAT and ALP) not found",
            "patients": [single_result],
        }
    return {"status": "success", "processed": 1, "patients": [single_result]}


###############################################################################
@router.post("/batch-agent", response_model=None, status_code=status.HTTP_202_ACCEPTED)
async def start_batch_clinical_agent() -> dict[str, Any]:
    txt_files = [
        path
        for path in Path(TASKS_PATH).glob("*.txt")
        if path.is_file()
    ]

    if not txt_files:
        logger.info(
            "No .txt files found in default path. Add new files and rerun the batch agent."
        )
        return {"status": "success", "processed": 0, "patients": []}

    results: list[dict[str, Any]] = []
    for path in txt_files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed reading {path}: {exc}")
            continue

        try:
            patient_payload = PatientData(name=path.stem, info=text)
        except ValidationError as exc:
            logger.error(f"Invalid data for patient {path.stem}: {exc}")
            continue

        try:
            case = await process_single_patient(patient_payload)
        except ValueError as exc:
            logger.error(f"Failed processing patient {path.stem}: {exc}")
            continue

        results.append(case)

    return {"status": "success", "processed": len(results), "patients": results}
