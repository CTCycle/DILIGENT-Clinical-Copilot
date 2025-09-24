from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from Pharmagent.app.constants import SOURCES_PATH
from Pharmagent.app.utils.database.sqlite import Any, database
from Pharmagent.app.utils.services.scraper import LiverToxClient
from Pharmagent.app.utils.serializer import DataSerializer

router = APIRouter(prefix="/pharmacology", tags=["pharmacology"])

LT_client = LiverToxClient()
serializer = DataSerializer()


###############################################################################
@router.get(
    "/livertox/fetch",
    response_model=None,
    summary="Fetch LiverTox monograph data (FTP bulk or live web)",
)
async def fetch_bulk_livertox(
    convert_to_dataframe: bool = Query(
        False,
        description="Extract data from the downloaded file and save it into database",
    ),
) -> dict[str, Any]:
    try:
        download_info = await LT_client.download_bulk_data(SOURCES_PATH)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Bulk download failed: {exc}")

    archive_path = download_info.get("file_path")
    try:
        entries = LT_client.collect_monographs(archive_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process archive: {exc}")

    if not entries:
        raise HTTPException(
            status_code=500, detail="No LiverTox monographs were extracted from archive."
        )

    try:
        serializer.save_livertox_records(entries)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to persist data: {exc}")

    try:
        stored_count = database.count_rows("LIVERTOX_MONOGRAPHS")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to verify import: {exc}")

    if stored_count != len(entries):
        raise HTTPException(
            status_code=500,
            detail=(
                "Mismatch between extracted entries and stored rows; import verification failed."
            ),
        )

    if convert_to_dataframe:
        _ = LT_client.convert_file_to_dataframe()

    response_payload = dict(download_info)
    response_payload["records"] = stored_count
    response_payload["processed_entries"] = len(entries)
    return response_payload
