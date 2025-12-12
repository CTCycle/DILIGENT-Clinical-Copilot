from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from DILIGENT.server.database.database import database
from DILIGENT.server.database.schema import ClinicalSession, LiverToxData, DrugsCatalog
from DILIGENT.server.utils.logger import logger


###############################################################################
router = APIRouter(prefix="/browser", tags=["browser"])


###############################################################################
class TableDataResponse(BaseModel):
    columns: list[str]
    rows: list[dict]
    total_rows: int


TABLE_MAPPING = {
    "sessions": ClinicalSession.__tablename__,
    "livertox": LiverToxData.__tablename__,
    "drugs": DrugsCatalog.__tablename__,
}


###############################################################################
@router.get("/sessions", response_model=TableDataResponse)
async def get_sessions_data() -> TableDataResponse:
    """Fetch clinical sessions table data."""
    return await _fetch_table_data("sessions")


@router.get("/livertox", response_model=TableDataResponse)
async def get_livertox_data() -> TableDataResponse:
    """Fetch LiverTox catalog data."""
    return await _fetch_table_data("livertox")


@router.get("/drugs", response_model=TableDataResponse)
async def get_drugs_data() -> TableDataResponse:
    """Fetch drugs catalog data."""
    return await _fetch_table_data("drugs")


###############################################################################
async def _fetch_table_data(table_key: str) -> TableDataResponse:
    """Generic helper to fetch data from a table."""
    table_name = TABLE_MAPPING.get(table_key)
    if not table_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown table: {table_key}",
        )

    try:
        logger.info("Fetching data for table: %s", table_name)
        df = database.load_from_database(table_name)
        
        columns = df.columns.tolist()
        # Convert DataFrame to list of dicts, handling NaN and timestamps
        rows = df.fillna("").to_dict(orient="records")
        
        # Convert datetime objects to strings for JSON serialization
        for row in rows:
            for key, value in row.items():
                if hasattr(value, "isoformat"):
                    row[key] = value.isoformat()

        return TableDataResponse(
            columns=columns,
            rows=rows,
            total_rows=len(rows),
        )

    except Exception as e:
        logger.exception("Error fetching table data for %s: %s", table_name, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch data: {str(e)}",
        ) from e
