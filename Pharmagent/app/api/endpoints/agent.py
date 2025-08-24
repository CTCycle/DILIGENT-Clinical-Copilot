import time

import pandas as pd
from fastapi import APIRouter, status
from fastapi.concurrency import run_in_threadpool

from Pharmagent.app.utils.serializer import DataSerializer
from Pharmagent.app.utils.services.parser import PatientCase, DiseasesParser, BloodTestParser
from Pharmagent.app.api.schemas.clinical import PatientData, PatientOutputReport

from Pharmagent.app.constants import DATA_PATH, PARSER_MODEL
from Pharmagent.app.logger import logger

router = APIRouter(prefix="/agent", tags=["agent"])

patient = PatientCase()
serializer = DataSerializer()

###############################################################################
@router.post(
        "", 
        response_model=None, 
        status_code=status.HTTP_202_ACCEPTED)

async def start_clinical_agent(payload: PatientData) -> PatientOutputReport:
    logger.info(f'Starting clinical agent processing for patient: {payload.name or "Unknown"}') 

    # 1. Extract each section from the patient info and save entry into database
    # Text sections are: anamnesis, blood tests, additional tests, drugs 
    logger.info('Processing data to extract known sections and save new patient to database')
    sections, patient_table = await run_in_threadpool(
        patient.extract_sections_from_text, payload)    
    await run_in_threadpool(serializer.save_patients_info, patient_table)

    # 2. Initialize Ollama client and pull the model if not already done
    parser = DiseasesParser()
    logger.info(f'Extracting diseases from patient anamnesis using {parser.model}')

    start_time = time.time()
    diseases = await parser.extract_diseases(sections.get('anamnesis', None))
    elapsed = time.time() - start_time
    logger.info(f"Time elapsed for diseases extraction: {elapsed:.2f} seconds.")

    parser = BloodTestParser()
    logger.info(f'Extracting blood tests analysis from patient lab results using {parser.model}')
    start_time = time.time()
    blood_test_results = await parser.extract_blood_test_results(sections.get('blood_tests', None))
    elapsed = time.time() - start_time

   
    pass


    return {'status': 'success'}