# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

from common.utils.logger import logger
from services.updater.terminology.snapshot import StopFn, TerminologySourceUnavailable

USER_AGENT = "DILIGENT terminology updater"
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
BACKOFF_SECONDS = (1.0, 3.0, 9.0)
CHUNK_BYTES = 1 << 20

###############################################################################
def request_json(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    """GET JSON with bounded retries and Retry-After aware backoff."""
    last_error: Exception | None = None
    for attempt, delay in enumerate((0.0, *BACKOFF_SECONDS)):
        if delay:
            time.sleep(delay)
        try:
            response = client.get(url, params=params)
        except httpx.HTTPError as exc:
            last_error = exc
            continue
        if response.status_code in RETRY_STATUS:
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit() and attempt < len(BACKOFF_SECONDS):
                time.sleep(min(60.0, float(retry_after)))
            last_error = httpx.HTTPStatusError(
                f"HTTP {response.status_code}", request=response.request, response=response
            )
            continue
        response.raise_for_status()
        return response.json()
    raise TerminologySourceUnavailable(f"Request failed for {url}: {last_error}")

###############################################################################
def download_file(
    url: str,
    target: Path,
    *,
    timeout: float,
    should_stop: StopFn,
) -> tuple[Path, str]:
    """Download to ``target`` atomically and return (path, sha256).

    The previous file is kept until the new download is complete, so an
    interrupted refresh never leaves a truncated archive behind.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = target.with_suffix(target.suffix + ".metadata.json")
    metadata: dict[str, Any] = {}
    if target.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = {}
    headers = {"User-Agent": USER_AGENT}
    if metadata.get("url") == url:
        if metadata.get("etag"):
            headers["If-None-Match"] = str(metadata["etag"])
        if metadata.get("last_modified"):
            headers["If-Modified-Since"] = str(metadata["last_modified"])
    candidate = target.with_suffix(target.suffix + ".candidate")
    last_error: Exception | None = None
    for delay in (0.0, *BACKOFF_SECONDS):
        if delay:
            time.sleep(delay)
        if should_stop():
            raise TerminologySourceUnavailable("Download cancelled")
        digest = hashlib.sha256()
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
                with client.stream("GET", url) as response:
                    if response.status_code == 304 and metadata.get("sha256"):
                        logger.info("Terminology download unchanged: %s", url)
                        return target, str(metadata["sha256"])
                    if response.status_code in RETRY_STATUS:
                        last_error = RuntimeError(f"HTTP {response.status_code}")
                        continue
                    response.raise_for_status()
                    expected = response.headers.get("Content-Length")
                    written = 0
                    with candidate.open("wb") as handle:
                        for chunk in response.iter_bytes(CHUNK_BYTES):
                            if should_stop():
                                raise TerminologySourceUnavailable("Download cancelled")
                            handle.write(chunk)
                            digest.update(chunk)
                            written += len(chunk)
                    if expected and expected.isdigit() and int(expected) != written:
                        last_error = RuntimeError(
                            f"Incomplete download ({written} of {expected} bytes)"
                        )
                        continue
                    etag = response.headers.get("ETag")
                    last_modified = response.headers.get("Last-Modified")
        except httpx.HTTPError as exc:
            last_error = exc
            continue
        finally:
            if should_stop():
                candidate.unlink(missing_ok=True)
        candidate.replace(target)
        sha256 = digest.hexdigest()
        metadata_path.write_text(
            json.dumps(
                {"url": url, "etag": etag, "last_modified": last_modified, "sha256": sha256}
            ),
            encoding="utf-8",
        )
        return target, sha256
    candidate.unlink(missing_ok=True)
    raise TerminologySourceUnavailable(f"Download failed for {url}: {last_error}")
