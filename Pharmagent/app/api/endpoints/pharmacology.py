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

async def fetch_bulk_livertox():        
    try:        
        path = await LT_client.download_bulk_archive(SOURCES_PATH)        
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Bulk download failed: {e}")
    return {"saved_to": str(path)}