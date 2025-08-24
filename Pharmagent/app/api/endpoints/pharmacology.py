from fastapi import APIRouter, HTTPException, Query

from Pharmagent.app.utils.serializer import DataSerializer
from Pharmagent.app.utils.services.scraper import LiverToxClient
from Pharmagent.app.api.schemas.clinical import PatientData

from Pharmagent.app.constants import SOURCES_PATH
from Pharmagent.app.logger import logger

router = APIRouter(prefix="/pharmacology", tags=["pharmacology"])

LT_client = LiverToxClient()

###############################################################################
@router.get(
    "/livertox/fetch",
    response_model=None,
    summary="Fetch LiverTox monograph data (FTP bulk or live web)")

async def fetch_bulk_livertox(
    convert_to_dataframe: bool = Query(
        False, 
        description="Extract data from the downloaded file and save it into database")):
      
    try:        
        results = await LT_client.download_bulk_data(SOURCES_PATH)
        if convert_to_dataframe:
            livertox_data = LT_client.convert_file_to_dataframe()        
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Bulk download failed: {e}")
    
    return results