from fastapi import APIRouter, status
from fastapi.concurrency import run_in_threadpool

from Pharmagent.app.utils.serializer import DataSerializer
from Pharmagent.app.utils.services.extraction import PatientInfoExtraction, DiseasesExtraction
from Pharmagent.app.api.models.server import OllamaClient
from Pharmagent.app.api.schemas.clinical import PatientData, PatientOutputReport

from Pharmagent.app.constants import DATA_PATH
from Pharmagent.app.logger import logger

router = APIRouter(prefix="/agent", tags=["agent"])


###############################################################################
@router.post(
        "", 
        response_model=None, 
        status_code=status.HTTP_202_ACCEPTED)

async def start_clinical_agent(payload: PatientData) -> PatientOutputReport:
    logger.info(f'Starting clinical agent processing for patient: {payload.name or "Unknown"}')

    processor = PatientInfoExtraction(payload)
    sections = processor.extract_textual_sections() 
    
    serializer = DataSerializer()
    serializer.save_patients_info(sections) 

    # do diseases extraction
    # to check validity
    extractor = DiseasesExtraction(model="llama3.1:8b")  # adjust model if needed
    clinical_text = sections if isinstance(sections, str) else "\n\n".join(
        [v for v in (sections.values() if isinstance(sections, dict) else []) if isinstance(v, str)])

    diseases_json = await run_in_threadpool(extractor.extract, clinical_text)
    logger.info("Disease extraction: %d found", len(diseases_json.get("diseases", [])))
    

    return {'status': 'success'}