from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from DILIGENT.server.database.database import database
from DILIGENT.server.database.schema import ClinicalSession, LiverToxData, DrugsCatalog
from DILIGENT.server.utils.configurations import server_settings
from DILIGENT.server.utils.logger import logger


###############################################################################
router = APIRouter(prefix="/browser", tags=["browser"])

# Default page size from configuration
DEFAULT_PAGE_SIZE = server_settings.database.browser_page_size
OFFSET_DESCRIPTION = "Number of rows to skip"
LIMIT_DESCRIPTION = "Number of rows to fetch"


###############################################################################
class TableDataResponse(BaseModel):
    columns: list[str]
    rows: list[dict]
    total_rows: int
    has_more: bool


TABLE_MAPPING = {
    "sessions": ClinicalSession.__tablename__,
    "livertox": LiverToxData.__tablename__,
    "drugs": DrugsCatalog.__tablename__,
}


###############################################################################
@router.get("/sessions", response_model=TableDataResponse)
def get_sessions_data(
    offset: int = Query(0, ge=0, description=OFFSET_DESCRIPTION),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=10000, description=LIMIT_DESCRIPTION),
) -> TableDataResponse:
    """Fetch clinical sessions table data with pagination."""
    return _fetch_table_data("sessions", offset, limit)


@router.get("/livertox", response_model=TableDataResponse)
def get_livertox_data(
    offset: int = Query(0, ge=0, description=OFFSET_DESCRIPTION),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=10000, description=LIMIT_DESCRIPTION),
) -> TableDataResponse:
    """Fetch LiverTox catalog data with pagination."""
    return _fetch_table_data("livertox", offset, limit)


@router.get("/drugs", response_model=TableDataResponse)
def get_drugs_data(
    offset: int = Query(0, ge=0, description=OFFSET_DESCRIPTION),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=10000, description=LIMIT_DESCRIPTION),
) -> TableDataResponse:
    """Fetch drugs catalog data with pagination."""
    return _fetch_table_data("drugs", offset, limit)


###############################################################################
def _fetch_table_data(table_key: str, offset: int, limit: int) -> TableDataResponse:
    """Generic helper to fetch paginated data from a table."""
    table_name = TABLE_MAPPING.get(table_key)
    if not table_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown table: {table_key}",
        )

    try:
        logger.info("Fetching data for table: %s (offset=%d, limit=%d)", table_name, offset, limit)
        
        # Get total count for has_more calculation
        total_rows = database.count_rows(table_name)
        
        # Fetch only the requested page
        df = database.load_paginated(table_name, offset, limit)
        
        columns = df.columns.tolist()
        # Convert DataFrame to list of dicts, handling NaN and timestamps
        rows = df.fillna("").to_dict(orient="records")
        
        # Convert datetime objects to strings for JSON serialization
        for row in rows:
            for key, value in row.items():
                if hasattr(value, "isoformat"):
                    row[key] = value.isoformat()

        # Determine if there are more rows to fetch
        has_more = (offset + len(rows)) < total_rows

        return TableDataResponse(
            columns=columns,
            rows=rows,
            total_rows=total_rows,
            has_more=has_more,
        )

    except Exception as e:
        logger.exception("Error fetching table data for %s: %s", table_name, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch data: {str(e)}",
        ) from e

