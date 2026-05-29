from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any

import httpx

SUPPORTED_MONOGRAPH_EXTENSIONS = (".html", ".htm", ".xhtml", ".xml", ".nxml", ".pdf")
NBK_ID_PATTERN = re.compile(r"^NBK\d+$", re.IGNORECASE)
DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "DILIGENTClinicalCopilot/1.0 (contact=clinical-copilot@pharmagent.local)"
    )
}
DOWNLOAD_CHUNK_SIZE = 262_144
DOWNLOAD_PROGRESS_BYTE_INTERVAL = 5 * 1024 * 1024


def load_json(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError, OSError:
        return None


def save_masterlist_metadata(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def metadata_matches(stored: dict[str, Any], remote: dict[str, Any]) -> bool:
    return stored.get("last_modified") == remote.get("last_modified") and int(
        stored.get("size", 0)
    ) == int(remote.get("size", 0))


async def download_file(
    client: httpx.AsyncClient,
    url: str,
    destination: str,
    total_size: int,
    label: str,
    *,
    chunk_size: int,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_start: float = 0.0,
    progress_span: float = 0.0,
) -> None:
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        downloaded = 0
        last_reported = 0
        with open(destination, "wb") as output:
            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                if chunk:
                    output.write(chunk)
                    downloaded += len(chunk)
                    should_report = (
                        downloaded >= total_size > 0
                        or downloaded - last_reported >= DOWNLOAD_PROGRESS_BYTE_INTERVAL
                    )
                    if not should_report:
                        continue
                    last_reported = downloaded
                    if progress_callback is None or progress_span <= 0:
                        continue
                    if total_size > 0:
                        ratio = min(1.0, max(0.0, downloaded / total_size))
                        progress_value = progress_start + (ratio * progress_span)
                        message = (
                            f"Downloaded {downloaded:,}/{total_size:,} bytes for {label}"
                        )
                    else:
                        progress_value = progress_start
                        message = f"Downloaded {downloaded:,} bytes for {label}"
                    emit_progress(
                        progress_callback,
                        progress=progress_value,
                        message=message,
                    )


def emit_progress(
    progress_callback: Callable[[float, str], None] | None,
    *,
    progress: float,
    message: str,
) -> None:
    if progress_callback is None:
        return
    bounded_progress = min(100.0, max(0.0, float(progress)))
    progress_callback(bounded_progress, message)


def should_cancel(should_stop: Callable[[], bool] | None) -> bool:
    if should_stop is None:
        return False
    return bool(should_stop())
