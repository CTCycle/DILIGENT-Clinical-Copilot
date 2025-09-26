from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from Pharmagent.app.constants import LIVERTOX_ARCHIVE, SOURCES_PATH
from Pharmagent.app.utils.database.sqlite import database
from Pharmagent.app.utils.jobs import JobManager
from Pharmagent.app.utils.services.retrieval import RxNavClient
from Pharmagent.app.utils.services.scraper import LiverToxClient
from Pharmagent.app.utils.serializer import DataSerializer

router = APIRouter(prefix="/pharmacology", tags=["pharmacology"])

LT_client = LiverToxClient()
rx_client = RxNavClient()
serializer = DataSerializer()
job_manager = JobManager()


# -----------------------------------------------------------------------------
def _normalize_archive_path(base_path: str) -> str:
    return os.path.abspath(os.path.join(base_path, LIVERTOX_ARCHIVE))


# -----------------------------------------------------------------------------
def _collect_local_archive_info(archive_path: str) -> dict[str, Any]:
    if not os.path.isfile(archive_path):
        raise HTTPException(
            status_code=404,
            detail="LiverTox archive not found; download is required before processing.",
        )
    file_size = os.path.getsize(archive_path)
    modified = datetime.fromtimestamp(os.path.getmtime(archive_path), UTC).isoformat()
    return {"file_path": archive_path, "size": file_size, "last_modified": modified}


type JobResult = dict[str, Any]
type JobState = dict[str, Any]


# -----------------------------------------------------------------------------
async def _run_livertox_job(
    job_id: str, convert_to_dataframe: bool, skip_download: bool, sources_path: str
) -> JobResult:
    await job_manager.set_progress(job_id, 0.01, "Preparing LiverTox ingestion job")

    archive_path = _normalize_archive_path(sources_path)

    if skip_download:
        await job_manager.set_progress(
            job_id, 0.05, "Validating existing LiverTox archive"
        )
        download_info = await asyncio.to_thread(
            _collect_local_archive_info, archive_path
        )
    else:
        await job_manager.set_progress(
            job_id, 0.1, "Downloading latest LiverTox archive"
        )
        try:
            download_info = await LT_client.download_bulk_data(sources_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Bulk download failed: {exc}")
        archive_file = download_info.get("file_path")
        if not isinstance(archive_file, str) or not archive_file:
            raise HTTPException(
                status_code=500, detail="LiverTox archive path missing."
            )
        archive_path = os.path.abspath(archive_file)
        download_info["file_path"] = archive_path

    await job_manager.set_progress(job_id, 0.3, "Extracting LiverTox monographs")
    try:
        entries = await asyncio.to_thread(LT_client.collect_monographs, archive_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to process archive: {exc}")
    if not entries:
        raise HTTPException(
            status_code=500,
            detail="No LiverTox monographs were extracted from archive.",
        )

    sanitized = await asyncio.to_thread(serializer.sanitize_livertox_records, entries)
    if sanitized.empty:
        raise HTTPException(
            status_code=500,
            detail="No valid LiverTox monographs were available after sanitization.",
        )
    entries = sanitized.to_dict(orient="records")

    await job_manager.set_progress(job_id, 0.45, "Enriching LiverTox records")
    tasks = [
        asyncio.to_thread(rx_client.fetch_drug_terms, entry["drug_name"])
        for entry in entries
    ]
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for entry, result in zip(entries, results, strict=False):
            if isinstance(result, Exception):
                entry["additional_names"] = None
                entry["synonyms"] = None
                continue
            names, synonyms = result
            entry["additional_names"] = ", ".join(names) if names else None
            entry["synonyms"] = ", ".join(synonyms) if synonyms else None
    else:
        for entry in entries:
            entry["additional_names"] = None
            entry["synonyms"] = None

    await job_manager.set_progress(job_id, 0.6, "Persisting LiverTox records")
    try:
        await asyncio.to_thread(serializer.save_livertox_records, entries)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to persist data: {exc}")

    await job_manager.set_progress(job_id, 0.75, "Verifying stored LiverTox records")
    try:
        stored_count = await asyncio.to_thread(
            database.count_rows, "LIVERTOX_MONOGRAPHS"
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to verify import: {exc}")

    if stored_count != len(entries):
        raise HTTPException(
            status_code=500,
            detail=(
                "Mismatch between extracted entries and stored rows; import verification failed."
            ),
        )

    if convert_to_dataframe:
        await job_manager.set_progress(job_id, 0.85, "Converting archive to DataFrame")
        await asyncio.to_thread(LT_client.convert_file_to_dataframe)

    await job_manager.set_progress(job_id, 1.0, "LiverTox ingestion completed")

    response_payload = dict(download_info)
    response_payload["records"] = stored_count
    response_payload["processed_entries"] = len(entries)
    return response_payload


###############################################################################
@router.get(
    "/livertox/fetch",
    response_model=None,
    summary="Fetch LiverTox monograph data (FTP bulk or live web)",
)
# -----------------------------------------------------------------------------
async def fetch_bulk_livertox(
    convert_to_dataframe: bool = Query(
        False,
        description="Extract data from the downloaded file and save it into database",
    ),
    skip_download: bool = Query(
        False,
        description="Skip downloading the LiverTox archive if a local copy is available",
    ),
) -> JobState:
    if skip_download:
        archive_path = _normalize_archive_path(SOURCES_PATH)
        if not os.path.isfile(archive_path):
            raise HTTPException(
                status_code=404,
                detail="LiverTox archive not found; download is required before processing.",
            )

    try:
        job_id = await job_manager.submit(
            _run_livertox_job, convert_to_dataframe, skip_download, SOURCES_PATH
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start job: {exc}")

    status = await job_manager.get_job(job_id)
    if status is None:
        raise HTTPException(status_code=500, detail="Failed to register job")
    return status


@router.get(
    "/livertox/status/{job_id}",
    response_model=None,
    summary="Get status of LiverTox import job",
)
# -----------------------------------------------------------------------------
async def get_livertox_job_status(job_id: str) -> JobState:
    status = await job_manager.get_job(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status
