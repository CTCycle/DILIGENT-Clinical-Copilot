from __future__ import annotations

import asyncio
import os
import re
import sys
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd

from common.utils.logger import logger
from services.updater import livertox_common, livertox_parse


###############################################################################
async def download_file(
    client: httpx.AsyncClient,
    url: str,
    destination: str,
    total_size: int,
    label: str,
    *,
    chunk_size: int,
) -> None:
    _ = sys
    _ = nullcontext
    await livertox_common.download_file(
        client,
        url,
        destination,
        total_size,
        label,
        chunk_size=chunk_size,
    )


###############################################################################

# Extracted from the facade module; functions intentionally accept the facade instance.

async def download_bulk_data(self, dest_path: str) -> dict[str, Any]:
    url = self.base_url + self.file_name
    async with httpx.AsyncClient(
        timeout=30.0, headers=self.http_headers, follow_redirects=True
    ) as client:
        head = await client.head(url)
        head.raise_for_status()
        metadata = {
            "size": int(head.headers.get("Content-Length", 0)),
            "last_modified": head.headers.get("Last-Modified"),
            "source_url": str(head.url),
        }
        dest_dir = os.path.abspath(dest_path)
        os.makedirs(dest_dir, exist_ok=True)
        file_path = os.path.join(dest_dir, self.file_name)

        stored_metadata = livertox_common.load_json(self.archive_metadata_path)
        if self.redownload:
            stored_metadata = None

        if (
            stored_metadata
            and os.path.isfile(file_path)
            and livertox_common.metadata_matches(stored_metadata, metadata)
        ):
            logger.info("LiverTox archive unchanged; skipping download")
            return {
                "file_path": file_path,
                "size": metadata.get("size", 0),
                "last_modified": metadata.get("last_modified"),
                "downloaded": False,
                "source_url": metadata["source_url"],
            }

        await asyncio.sleep(self.delay)
        await download_file(
            client,
            url,
            file_path,
            metadata.get("size", 0),
            self.file_name,
            chunk_size=self.chunk_size,
        )
        livertox_common.save_masterlist_metadata(self.archive_metadata_path, metadata)

    return {
        "file_path": file_path,
        "size": metadata.get("size", 0),
        "last_modified": metadata.get("last_modified"),
        "downloaded": True,
        "source_url": metadata["source_url"],
    }

def refresh_master_list(self) -> tuple[dict[str, Any], pd.DataFrame]:
    logger.info("Refreshing LiverTox master list")
    metadata = asyncio.run(download_master_list(self))

    frame = pd.read_excel(
        metadata["file_path"],
        engine="openpyxl",
        header=self.header_row,
        skiprows=0,
    )
    sanitized = livertox_parse.sanitize_livertox_master_list(self, frame)
    if sanitized is None:
        sanitized = pd.DataFrame()
    else:
        sanitized = sanitized.copy()
        sanitized["source_url"] = metadata.get("source_url")
        sanitized["source_last_modified"] = metadata.get("last_modified")
        if (
            "last_update" in sanitized.columns
            and pd.api.types.is_datetime64_any_dtype(sanitized["last_update"])
        ):
            sanitized["last_update"] = sanitized["last_update"].dt.strftime(  # type: ignore
                "%Y-%m-%d"
            )
    metadata["records"] = len(sanitized.index)

    return metadata, sanitized

async def download_master_list(self) -> dict[str, Any]:
    async with httpx.AsyncClient(
        timeout=30.0, headers=self.http_headers, follow_redirects=True
    ) as client:
        master_url = await resolve_master_list_url(self, client)
        head = await client.head(master_url)
        head.raise_for_status()
        metadata = {
            "size": int(head.headers.get("Content-Length", 0)),
            "last_modified": head.headers.get("Last-Modified"),
            "source_url": str(head.url),
        }
        stored_metadata = livertox_common.load_json(self.master_list_metadata_path)
        if self.redownload:
            stored_metadata = None
        if (
            stored_metadata
            and os.path.isfile(self.master_list_path)
            and livertox_common.metadata_matches(stored_metadata, metadata)
        ):
            logger.info("Master list unchanged; skipping download")
            return {
                "file_path": self.master_list_path,
                "size": metadata.get("size", 0),
                "last_modified": metadata.get("last_modified"),
                "downloaded": False,
                "source_url": metadata["source_url"],
            }

        await asyncio.sleep(self.delay)
        await download_file(
            client,
            master_url,
            self.master_list_path,
            metadata.get("size", 0),
            os.path.basename(self.master_list_path),
            chunk_size=self.chunk_size,
        )
        livertox_common.save_masterlist_metadata(self.master_list_metadata_path, metadata)

    return {
        "file_path": self.master_list_path,
        "size": metadata.get("size", 0),
        "last_modified": metadata.get("last_modified"),
        "downloaded": True,
        "source_url": metadata["source_url"],
    }

async def resolve_master_list_url(self, client: httpx.AsyncClient) -> str:
    try:
        return await resolve_master_list_from_bookshelf(self, client)
    except Exception as exc:
        logger.warning("Bookshelf Excel lookup failed: %s", exc)
    try:
        return await resolve_master_list_from_bin(self, client, self.base_url)
    except Exception as exc:
        logger.warning("Primary FTP lookup failed: %s", exc)
        fallback_url = await resolve_master_list_via_datagov(self, client)
        return fallback_url

async def resolve_master_list_from_bookshelf(
    self, client: httpx.AsyncClient
) -> str:
    report_url = "https://www.ncbi.nlm.nih.gov/books/NBK571102/?report=excel"
    head_response: httpx.Response | None = None
    try:
        head_response = await client.head(report_url, follow_redirects=False)
        head_response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in (301, 302, 303, 307, 308):
            head_response = None
        else:
            head_response = exc.response
    except httpx.HTTPError:
        head_response = None

    redirect_statuses = {301, 302, 303, 307, 308}
    if head_response is not None:
        if (
            head_response.status_code in redirect_statuses
            and head_response.headers.get("Location")
        ):
            candidate = httpx.URL(report_url).join(
                head_response.headers["Location"]
            )
            return await probe_master_list_candidate(self, client, str(candidate))
        content_type = (head_response.headers.get("Content-Type") or "").lower()
        disposition = (
            head_response.headers.get("Content-Disposition") or ""
        ).lower()
        if "excel" in content_type or ".xlsx" in disposition:
            return await probe_master_list_candidate(
                self, client, str(head_response.url)
            )

    get_response = await client.get(report_url, follow_redirects=False)
    if get_response.status_code in redirect_statuses and get_response.headers.get(
        "Location"
    ):
        candidate = httpx.URL(report_url).join(get_response.headers["Location"])
        return await probe_master_list_candidate(self, client, str(candidate))

    content_type = (get_response.headers.get("Content-Type") or "").lower()
    disposition = (get_response.headers.get("Content-Disposition") or "").lower()
    if "excel" in content_type or ".xlsx" in disposition:
        return await probe_master_list_candidate(self, client, str(get_response.url))

    html_content = get_response.text
    for pattern in (
        r"url=([^\"'>]+\.xlsx)",
        r"['\"]([^'\"]+\.xlsx)['\"]",
    ):
        for match in re.finditer(pattern, html_content, flags=re.IGNORECASE):
            candidate_url = match.group(1)
            candidate = httpx.URL(report_url).join(candidate_url)
            try:
                return await probe_master_list_candidate(
                    self, client, str(candidate)
                )
            except Exception as exc:  # pragma: no cover - network dependent
                logger.debug(
                    "Bookshelf candidate %s failed: %s", str(candidate), exc
                )
                continue

    raise RuntimeError("Unable to resolve master list via Bookshelf report page")

async def resolve_master_list_from_bin(
    self, client: httpx.AsyncClient, base_url: str
) -> str:
    bin_url = str(httpx.URL(base_url).join("bin/"))
    response = await client.get(bin_url)
    response.raise_for_status()
    content = response.text
    matches = re.finditer(
        r"<a[^>]+href=\"([^\"]+\.xlsx)\"[^>]*>([^<]*(?:<[^>]+>[^<]*)*)</a>",
        content,
        flags=re.IGNORECASE,
    )
    candidates: list[tuple[str, str]] = []
    for match in matches:
        href, label = match.groups()
        if "ipmcbook" in href.lower():
            continue
        normalized_label = re.sub(r"\s+", " ", label.lower())
        if "master" not in normalized_label and "excel" not in normalized_label:
            continue
        url = httpx.URL(bin_url).join(href)
        human_url = str(url)
        if not human_url.lower().startswith(
            "https://www.ncbi.nlm.nih.gov/"
        ) and not human_url.startswith(base_url):
            continue
        candidates.append((human_url, label.strip()))
    if not candidates:
        href_matches = re.finditer(
            r"href=\"([^\"]+\.xlsx)\"",
            content,
            flags=re.IGNORECASE,
        )
        for match in href_matches:
            href = match.group(1)
            if "ipmcbook" in href.lower():
                continue
            url = httpx.URL(bin_url).join(href)
            human_url = str(url)
            if not human_url.lower().startswith(
                "https://www.ncbi.nlm.nih.gov/"
            ) and not human_url.startswith(base_url):
                continue
            if "master" in os.path.basename(human_url).lower():
                candidates.append((human_url, os.path.basename(human_url)))
    if not candidates:
        raise RuntimeError(
            "Unable to locate LiverTox master list link on FTP bin page"
        )
    candidates.sort(key=lambda item: item[0])
    chosen_url = candidates[0][0]
    return chosen_url

async def resolve_master_list_via_datagov(self, client: httpx.AsyncClient) -> str:
    api_url = "https://catalog.data.gov/api/3/action/package_show"
    response = await client.get(api_url, params={"id": "livertox"})
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Unable to resolve FTP folder from Data.gov entry"
        ) from exc
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    resources = result.get("resources") if isinstance(result, dict) else None
    if not isinstance(resources, list):
        raise RuntimeError("Unable to resolve FTP folder from Data.gov entry")

    direct_candidates: list[str] = []
    direct_seen: set[str] = set()
    folder_candidates: list[str] = []
    folder_seen: set[str] = set()
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        raw_url = str(resource.get("url") or "").strip()
        if not raw_url:
            continue
        normalized_url = normalize_datagov_resource_url(self, raw_url)
        if not normalized_url:
            continue
        lowered = normalized_url.lower()
        if ".xlsx" in lowered:
            if normalized_url not in direct_seen:
                direct_seen.add(normalized_url)
                direct_candidates.append(normalized_url)
            continue
        nbk_match = re.search(r"/(nbk\d+)(?:/|$)", lowered)
        if nbk_match:
            nbk_id = nbk_match.group(1).upper()
            for template in (
                f"https://www.ncbi.nlm.nih.gov/books/{nbk_id}/?report=excel",
                f"https://www.ncbi.nlm.nih.gov/books/{nbk_id}/bin/{nbk_id}.xlsx",
            ):
                if template not in direct_seen:
                    direct_seen.add(template)
                    direct_candidates.append(template)
        if "ftp.ncbi.nlm.nih.gov" in lowered:
            if normalized_url.endswith("/"):
                base_candidate = normalized_url
            else:
                base_candidate = f"{normalized_url.rsplit('/', 1)[0]}/"
            if base_candidate not in folder_seen:
                folder_seen.add(base_candidate)
                folder_candidates.append(base_candidate)

    last_error: Exception | None = None
    for candidate in direct_candidates:
        try:
            resolved_direct = await probe_master_list_candidate(
                self, client, candidate
            )
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            logger.debug("Candidate Data.gov direct %s failed: %s", candidate, exc)
            continue
        return resolved_direct

    if not folder_candidates:
        raise RuntimeError("Unable to resolve FTP folder from Data.gov entry")

    original_base = self.base_url
    for base_candidate in folder_candidates:
        try:
            resolved = await resolve_master_list_from_bin(
                self, client, base_candidate
            )
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            logger.debug(
                "Candidate Data.gov base %s failed: %s", base_candidate, exc
            )
            continue
        self.base_url = base_candidate
        return resolved

    self.base_url = original_base
    if last_error is not None:
        raise RuntimeError(
            "Unable to resolve FTP folder from Data.gov entry"
        ) from last_error
    raise RuntimeError("Unable to resolve FTP folder from Data.gov entry")

def normalize_datagov_resource_url(self, url: str) -> str | None:
    normalized = url.strip()
    if not normalized:
        return None
    if normalized.startswith("ftp://"):
        normalized = "https://" + normalized[len("ftp://") :]
    if normalized.startswith("http"):
        return normalized
    if normalized.startswith("//"):
        return f"https:{normalized}"
    return None

async def probe_master_list_candidate(
    self, client: httpx.AsyncClient, candidate: str
) -> str:
    try:
        response = await client.head(candidate)
        response.raise_for_status()
    except httpx.HTTPStatusError:  # pragma: no cover - network dependent
        response = await fetch_candidate_with_get(self, client, candidate)
    except httpx.HTTPError:  # pragma: no cover - network dependent
        response = await fetch_candidate_with_get(self, client, candidate)
    content_type = (response.headers.get("Content-Type") or "").lower()
    if ".xlsx" not in candidate.lower() and "excel" not in content_type:
        disposition = (response.headers.get("Content-Disposition") or "").lower()
        if ".xlsx" in disposition:
            return str(response.url)
        raise RuntimeError("Candidate does not appear to be an Excel file")
    return str(response.url)

async def fetch_candidate_with_get(
    self, client: httpx.AsyncClient, candidate: str
) -> httpx.Response:
    probe_headers = dict(self.http_headers)
    probe_headers.setdefault("Range", "bytes=0-0")
    response = await client.get(
        candidate,
        headers=probe_headers,
        follow_redirects=True,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:  # pragma: no cover - network dependent
        if exc.response.status_code == 416:
            response = await client.get(
                candidate,
                headers=self.http_headers,
                follow_redirects=True,
            )
            response.raise_for_status()
        else:
            raise RuntimeError("Master list candidate returned HTTP error") from exc
    return response

def collect_local_archive_info(self, archive_path: str) -> dict[str, Any]:
    if not os.path.isfile(archive_path):
        raise RuntimeError(
            "LiverTox archive not found; enable REDOWNLOAD to fetch a fresh copy."
        )
    size = os.path.getsize(archive_path)
    modified = datetime.fromtimestamp(
        os.path.getmtime(archive_path), UTC
    ).isoformat()
    return {"file_path": archive_path, "size": size, "last_modified": modified}
