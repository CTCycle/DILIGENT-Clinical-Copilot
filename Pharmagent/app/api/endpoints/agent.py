from __future__ import annotations

import time
from typing import Any
import os
from os.path import isfile, join, splitext, basename
from pathlib import Path

from fastapi import APIRouter, status
from fastapi.concurrency import run_in_threadpool

from Pharmagent.app.api.schemas.clinical import PatientData, PatientOutputReport
from Pharmagent.app.logger import logger
from Pharmagent.app.utils.serializer import DataSerializer
from Pharmagent.app.utils.services.parser import (
    BloodTestParser,
    DiseasesParser,
    PatientCase,
)
from Pharmagent.app.constants import TASKS_PATH

# Single router: exposes one POST endpoint at /agent
router = APIRouter(prefix="/agent", tags=["agent"])

patient = PatientCase()
serializer = DataSerializer()
disease_parser = DiseasesParser()
test_parser = BloodTestParser()


# ----------------------------------------------------------------------------
async def process_single_patient(single_payload: PatientData) -> dict[str, Any]:    
    # 1) Extract each section and save patient info
    logger.info("Processing data save new patient to database")
    sections, patient_table = await run_in_threadpool(
        patient.extract_sections_from_text, single_payload
    )
    await run_in_threadpool(serializer.save_patients_info, patient_table)

    # 2) Extract blood tests and hepatic inputs
    logger.info(
        f"Extracting blood tests analysis from lab results using {test_parser.model}"
    )
    start_time = time.time()
    blood_test_results = await test_parser.extract_blood_test_results(
        sections.get("blood_tests") or ""
    )
    bt_elapsed = time.time() - start_time
    logger.info(f"Time elapsed for blood tests extraction: {bt_elapsed:.2f} seconds.")

    hepatic_inputs: dict[str, Any] = test_parser.extract_hepatic_markers(blood_test_results)
    is_valid_patient = ("ALAT" in hepatic_inputs) and ("ANA" in hepatic_inputs)
    logger.info('Current patient data is valid, proceeding with disease extraction')

    # 3) Only then, extract diseases if hepatic inputs are present
    diseases: dict[str, Any] | None = None 
    dis_elapsed = 0.0   
    if is_valid_patient:
        logger.info(
            f"Extracting diseases from patient anamnesis using {disease_parser.model}"
        )
        start_time = time.time()
        diseases = await disease_parser.extract_diseases(sections.get("anamnesis", None))
        dis_elapsed = time.time() - start_time
        logger.info(f"Time elapsed for diseases extraction: {dis_elapsed:.2f} seconds.")
    else:
        logger.info("Hepatic inputs not available (ALAT and/or ANA missing). Skipping disease extraction.")

    return {
        "name": single_payload.name or "Unknown",
        "is_valid": is_valid_patient,
        "diseases": diseases or {},
        "blood_tests": blood_test_results.model_dump()
        if hasattr(blood_test_results, "model_dump")
        else blood_test_results.__dict__,
        "hepatic_inputs": hepatic_inputs,
        "timings": {"diseases_s": dis_elapsed, "blood_tests_s": bt_elapsed},
    }


###############################################################################
@router.post("", response_model=None, status_code=status.HTTP_202_ACCEPTED)
async def start_clinical_agent(
    payload: PatientData,
) -> PatientOutputReport | dict[str, Any]:
    logger.info(
        f"Starting clinical agent processing for patient: {payload.name or 'Unknown'}"
    )

    # If from_files is true, process all .txt files from default TASKS_PATH
    if payload.from_files:
        txt_files = [
            join(TASKS_PATH, f)
            for f in os.listdir(TASKS_PATH)
            if isfile(join(TASKS_PATH, f)) and f.lower().endswith(".txt")
        ]

        if not txt_files:
            logger.info(
                "No .txt files found in default path. Try upon adding new files or set from_files to False"
            )
            return {"status": "success", "processed": 0, "patients": []}

        results: list[dict[str, Any]] = []
        for path in txt_files:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except Exception as e:
                logger.error(f"Failed reading {path}: {e}")
                continue

            patient_name = Path(path).stem  # filename (no extension) as patient name
            patient = PatientData(name=patient_name, info=text, from_files=False)
            case = await process_single_patient(patient)
            results.append(case)

        return {"status": "success", "processed": len(results), "patients": results}

    # Fallback: process the provided single payload
    single_result = await process_single_patient(payload)
    if not single_result.get("ready_for_hepatic", False):
        return {
            "status": "unsuccess",
            "processed": 0,
            "reason": "Required hepatic inputs (ALAT, ANA) not found",
            "patients": [single_result],
        }
    return {"status": "success", "processed": 1, "patients": [single_result]}
