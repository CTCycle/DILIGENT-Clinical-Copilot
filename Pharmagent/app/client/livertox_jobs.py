from __future__ import annotations

import asyncio
from typing import Any

import httpx

from Pharmagent.app.constants import (
    API_BASE_URL,
    PHARMACOLOGY_LIVERTOX_STATUS_ENDPOINT,
)


# -----------------------------------------------------------------------------
def _append_progress(progress_log: list[str], status: Any, detail: Any) -> None:
    if not isinstance(detail, str) or not detail:
        return
    label = status if isinstance(status, str) and status else "status"
    entry = f"{label}: {detail}"
    if entry not in progress_log:
        progress_log.append(entry)


# -----------------------------------------------------------------------------
def _format_progress_log(progress_log: list[str]) -> str:
    if not progress_log:
        return ""
    return "\nProgress log:\n" + "\n".join(progress_log)


# -----------------------------------------------------------------------------
async def _await_livertox_job(
    client: httpx.AsyncClient,
    initial_status: dict[str, Any],
    *,
    poll_interval: float = 2.0,
    timeout: float | None = None,
) -> tuple[dict[str, Any], list[str]] | str:
    job_id = initial_status.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return "[ERROR] Backend response did not include a job ID."

    progress_log: list[str] = []
    status = initial_status.get("status")
    detail = initial_status.get("detail")
    _append_progress(progress_log, status, detail)

    normalized_status = status.lower() if isinstance(status, str) else ""
    result = initial_status.get("result")
    if normalized_status == "failed":
        failure = detail if isinstance(detail, str) and detail else "Backend reported job failure."
        status_code = initial_status.get("status_code")
        if isinstance(status_code, int):
            failure = f"{failure} (status {status_code})"
        return "[ERROR] LiverTox import failed: " + failure + _format_progress_log(progress_log)
    if normalized_status == "completed":
        if isinstance(result, dict):
            return result, progress_log
        return "[ERROR] Backend did not provide job result on completion." + _format_progress_log(progress_log)

    status_url = f"{API_BASE_URL}{PHARMACOLOGY_LIVERTOX_STATUS_ENDPOINT}/{job_id}"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout if timeout is not None else None

    while True:
        if deadline is not None and loop.time() > deadline:
            progress_suffix = _format_progress_log(progress_log)
            waited = max(1, int(round(timeout))) if timeout is not None else 0
            message = (
                "[INFO] LiverTox import is still running after waiting "
                f"{waited} seconds."
                f"\nJob ID: {job_id}"
                f"\nStatus URL: {status_url}"
                "\nThe ingestion continues in the background; you can keep this tab "
                "open or retry later to refresh the status."
            )
            if progress_suffix:
                message += progress_suffix
            return message

        try:
            status_response = await client.get(status_url)
            status_response.raise_for_status()
        except httpx.TimeoutException:
            return "[ERROR] Polling job status timed out." + _format_progress_log(progress_log)
        except httpx.HTTPStatusError as exc:
            body = exc.response.text if exc.response is not None else ""
            code = exc.response.status_code if exc.response else "unknown"
            return (
                "[ERROR] Backend returned an error while checking job status."
                f"\nURL: {status_url}"
                f"\nStatus: {code}"
                f"\nResponse body:\n{body}"
                + _format_progress_log(progress_log)
            )
        except Exception as exc:  # noqa: BLE001
            return (
                f"[ERROR] Unexpected error while polling job status: {exc}"
                + _format_progress_log(progress_log)
            )

        try:
            status_payload = status_response.json()
        except ValueError:
            return "[ERROR] Backend status response was not valid JSON." + _format_progress_log(progress_log)

        if not isinstance(status_payload, dict):
            return "[ERROR] Unexpected status response format from backend." + _format_progress_log(progress_log)

        status = status_payload.get("status")
        detail = status_payload.get("detail")
        _append_progress(progress_log, status, detail)
        normalized_status = status.lower() if isinstance(status, str) else ""

        if normalized_status == "failed":
            failure = detail if isinstance(detail, str) and detail else "Backend reported job failure."
            status_code = status_payload.get("status_code")
            if isinstance(status_code, int):
                failure = f"{failure} (status {status_code})"
            return "[ERROR] LiverTox import failed: " + failure + _format_progress_log(progress_log)

        if normalized_status == "completed":
            result = status_payload.get("result")
            if isinstance(result, dict):
                return result, progress_log
            return "[ERROR] Backend did not provide job result on completion." + _format_progress_log(progress_log)

        await asyncio.sleep(poll_interval)
