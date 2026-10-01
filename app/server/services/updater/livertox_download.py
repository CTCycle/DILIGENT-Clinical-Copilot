# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
import csv
import io
import re
import tarfile
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import httpx
import pandas as pd
from openpyxl import load_workbook

from common.constants import (
    DEFAULT_NCBI_CONTACT_EMAIL,
    LIVERTOX_BOOK_ACCESSION,
    LIVERTOX_MASTER_LIST_ACCESSION,
    NLM_LITARCH_BASE_URL,
    NLM_LITARCH_FILE_LIST_URL,
)
from common.utils.logger import logger
from configurations.startup import get_server_settings
from services.updater import livertox_common, livertox_parse
from services.updater.ncbi_client import (
    NCBIClientError,
    resolve_books_oai_master_list_url,
    search_bookshelf_record,
)

MASTER_LIST_MEMBER_PATTERN = re.compile(r"^masterlist.*\.xlsx$", re.IGNORECASE)
EXPECTED_ARCHIVE_FILENAME = f"livertox_{LIVERTOX_BOOK_ACCESSION}.tar.gz"


###############################################################################
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
    await livertox_common.download_file(
        client,
        url,
        destination,
        total_size,
        label,
        chunk_size=chunk_size,
        progress_callback=progress_callback,
        progress_start=progress_start,
        progress_span=progress_span,
    )


###############################################################################
async def download_bulk_data(
    self,
    dest_path: str,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    async with _http_client(self) as client:
        return await _download_archive(
            self,
            client,
            dest_path,
            progress_callback=progress_callback,
        )


###############################################################################
def refresh_master_list(
    self,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    logger.info("Refreshing LiverTox master list through NCBI machine services")
    metadata = asyncio.run(
        download_master_list(self, progress_callback=progress_callback)
    )
    try:
        frame = pd.read_excel(
            metadata["file_path"],
            engine="openpyxl",
            header=self.header_row,
            skiprows=0,
        )
    except (OSError, ValueError, ImportError) as exc:
        raise RuntimeError(
            "The downloaded LiverTox master list could not be read."
        ) from exc

    sanitized = livertox_parse.sanitize_livertox_master_list(self, frame)
    if sanitized is None or sanitized.empty:
        raise RuntimeError(
            "The downloaded LiverTox master list contained no usable rows."
        )
    sanitized = sanitized.copy()
    sanitized["source_url"] = metadata.get("source_url")
    sanitized["source_last_modified"] = metadata.get("last_modified")
    if "last_update" in sanitized.columns and pd.api.types.is_datetime64_any_dtype(
        sanitized["last_update"]
    ):
        sanitized["last_update"] = sanitized["last_update"].dt.strftime(  # type: ignore
            "%Y-%m-%d"
        )
    metadata["records"] = len(sanitized.index)
    return metadata, sanitized


###############################################################################
async def download_master_list(
    self,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    async with _http_client(self) as client:
        master_url = await resolve_master_list_url(self, client)
        if master_url is not None:
            try:
                return await _download_master_list_url(
                    self,
                    client,
                    master_url,
                    progress_callback=progress_callback,
                )
            except RuntimeError:
                logger.warning(
                    "Books-OAI master-list download was unusable; using LitArch fallback"
                )
        return await _download_master_list_from_archive(
            self,
            client,
            progress_callback=progress_callback,
        )


###############################################################################
async def _download_master_list_from_archive(
    self,
    client: httpx.AsyncClient,
    *,
    progress_callback: Callable[[float, str], None] | None,
) -> dict[str, Any]:
    archive_metadata = await _download_archive(
        self,
        client,
        self.sources_path,
        progress_callback=progress_callback,
    )
    archive_path = Path(str(archive_metadata["file_path"]))
    member_name = find_master_list_member(archive_path)
    metadata = {
        "size": archive_path.stat().st_size,
        "last_modified": archive_metadata.get("last_modified"),
        "source_url": archive_metadata.get("source_url"),
        "source_member": member_name,
    }
    stored_metadata = livertox_common.load_json(self.master_list_metadata_path)
    if self.redownload:
        stored_metadata = None
    if (
        stored_metadata
        and Path(self.master_list_path).is_file()
        and livertox_common.metadata_matches(stored_metadata, metadata)
    ):
        _validate_master_list_file(self, Path(self.master_list_path))
        logger.info("Master list unchanged; skipping archive extraction")
        return {
            "file_path": self.master_list_path,
            **metadata,
            "downloaded": False,
        }

    destination = Path(self.master_list_path)
    partial_path = _partial_path(destination)
    try:
        extract_master_list_member(archive_path, member_name, partial_path)
        _validate_master_list_file(self, partial_path)
        partial_path.replace(destination)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        _remove_partial(partial_path)
        raise RuntimeError("The LiverTox master list could not be validated.") from exc
    livertox_common.save_masterlist_metadata(self.master_list_metadata_path, metadata)
    return {
        "file_path": self.master_list_path,
        **metadata,
        "downloaded": True,
    }


###############################################################################
async def resolve_master_list_url(
    self,
    client: httpx.AsyncClient,
) -> str | None:
    contact_email = _contact_email(self)
    await search_bookshelf_record(
        client,
        contact_email=contact_email,
        accession=LIVERTOX_MASTER_LIST_ACCESSION,
    )
    try:
        return await resolve_books_oai_master_list_url(
            client,
            accession=LIVERTOX_MASTER_LIST_ACCESSION,
        )
    except NCBIClientError:
        logger.warning(
            "Books-OAI master-list discovery was unavailable; using LitArch fallback"
        )
        return None


###############################################################################
async def resolve_litarch_archive(
    client: httpx.AsyncClient,
    *,
    accession: str = LIVERTOX_BOOK_ACCESSION,
) -> str:
    try:
        response = await client.get(NLM_LITARCH_FILE_LIST_URL)
        response.raise_for_status()
        rows = csv.reader(io.StringIO(response.text))
        matches: set[str] = set()
        expected_filename = f"livertox_{accession}.tar.gz".lower()
        for row in rows:
            normalized_cells = [cell.strip() for cell in row]
            if not any(cell.upper() == accession.upper() for cell in normalized_cells):
                continue
            paths = [
                cell.replace("\\", "/")
                for cell in normalized_cells
                if cell.lower().endswith(expected_filename)
            ]
            matches.update(paths)
    except (httpx.HTTPError, csv.Error) as exc:
        raise RuntimeError("NLM LitArch file-list lookup failed.") from exc

    if not matches:
        raise RuntimeError(
            "NLM LitArch file list did not contain the LiverTox archive."
        )
    if len(matches) != 1:
        raise RuntimeError(
            "NLM LitArch file list contained conflicting LiverTox archives."
        )
    return _build_litarch_archive_url(next(iter(matches)), accession=accession)


###############################################################################
def find_master_list_member(archive_path: str | Path) -> str:
    path = Path(archive_path)
    if not path.is_file() or not tarfile.is_tarfile(path):
        raise RuntimeError("The LiverTox archive is not a valid tarball.")
    candidates: list[str] = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            _validate_tar_member(member)
            if not member.isfile():
                continue
            basename = PurePosixPath(member.name).name
            if MASTER_LIST_MEMBER_PATTERN.fullmatch(basename):
                candidates.append(member.name)
    if not candidates:
        raise RuntimeError(
            "NCBI's official Books-OAI/LitArch sources did not expose a LiverTox master list."
        )
    if len(candidates) != 1:
        raise RuntimeError("The LiverTox archive contained conflicting master lists.")
    return candidates[0]


###############################################################################
def extract_master_list_member(
    archive_path: str | Path,
    member_name: str,
    destination: str | Path,
) -> None:
    path = Path(archive_path)
    _validate_member_name(member_name)
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember(member_name)
        _validate_tar_member(member)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError("The LiverTox master-list member could not be read.")
        payload = extracted.read()
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("wb") as output:
        output.write(payload)


###############################################################################
def collect_local_archive_info(self, archive_path: str) -> dict[str, Any]:
    path = Path(archive_path)
    _validate_archive_file(self, path)
    modified = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    return {
        "file_path": str(path),
        "size": path.stat().st_size,
        "last_modified": modified,
    }


###############################################################################
async def _download_archive(
    self,
    client: httpx.AsyncClient,
    dest_path: str,
    *,
    progress_callback: Callable[[float, str], None] | None,
) -> dict[str, Any]:
    archive_url = self.resolved_archive_url
    if archive_url is None:
        archive_url = await resolve_litarch_archive(
            client,
            accession=LIVERTOX_BOOK_ACCESSION,
        )
        self.resolved_archive_url = archive_url
    remote_metadata = await _fetch_remote_metadata(client, archive_url)
    destination_dir = Path(dest_path).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    file_path = destination_dir / self.file_name
    stored_metadata = livertox_common.load_json(self.archive_metadata_path)
    if self.redownload:
        stored_metadata = None
    if (
        stored_metadata
        and file_path.is_file()
        and livertox_common.metadata_matches(stored_metadata, remote_metadata)
    ):
        _validate_archive_file(self, file_path)
        logger.info("LiverTox archive unchanged; skipping download")
        return {
            "file_path": str(file_path),
            **remote_metadata,
            "downloaded": False,
        }

    partial_path = _partial_path(file_path)
    try:
        await asyncio.sleep(self.delay)
        await download_file(
            client,
            archive_url,
            str(partial_path),
            int(remote_metadata.get("size", 0)),
            file_path.name,
            chunk_size=self.chunk_size,
            progress_callback=progress_callback,
            progress_start=20.0,
            progress_span=15.0,
        )
        _validate_archive_file(self, partial_path)
        partial_path.replace(file_path)
    except (OSError, RuntimeError, httpx.HTTPError) as exc:
        _remove_partial(partial_path)
        raise RuntimeError(
            "The LiverTox archive could not be downloaded safely."
        ) from exc
    livertox_common.save_masterlist_metadata(
        self.archive_metadata_path, remote_metadata
    )
    return {
        "file_path": str(file_path),
        **remote_metadata,
        "downloaded": True,
    }


###############################################################################
async def _download_master_list_url(
    self,
    client: httpx.AsyncClient,
    master_url: str,
    *,
    progress_callback: Callable[[float, str], None] | None,
) -> dict[str, Any]:
    remote_metadata = await _fetch_remote_metadata(client, master_url)
    destination = Path(self.master_list_path)
    stored_metadata = livertox_common.load_json(self.master_list_metadata_path)
    if self.redownload:
        stored_metadata = None
    if (
        stored_metadata
        and destination.is_file()
        and livertox_common.metadata_matches(stored_metadata, remote_metadata)
    ):
        _validate_master_list_file(self, destination)
        logger.info("Master list unchanged; skipping download")
        return {
            "file_path": str(destination),
            **remote_metadata,
            "downloaded": False,
        }

    partial_path = _partial_path(destination)
    try:
        await asyncio.sleep(self.delay)
        await download_file(
            client,
            master_url,
            str(partial_path),
            int(remote_metadata.get("size", 0)),
            destination.name,
            chunk_size=self.chunk_size,
            progress_callback=progress_callback,
            progress_start=5.0,
            progress_span=15.0,
        )
        _validate_master_list_file(self, partial_path)
        partial_path.replace(destination)
    except (
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
        httpx.HTTPError,
    ) as exc:
        _remove_partial(partial_path)
        raise RuntimeError(
            "The LiverTox master list could not be downloaded safely."
        ) from exc
    livertox_common.save_masterlist_metadata(
        self.master_list_metadata_path, remote_metadata
    )
    return {
        "file_path": str(destination),
        **remote_metadata,
        "downloaded": True,
    }


###############################################################################
async def _fetch_remote_metadata(
    client: httpx.AsyncClient,
    url: str,
) -> dict[str, Any]:
    try:
        response = await client.head(url)
        if response.status_code in {405, 501}:
            response = await client.get(url, headers={"Range": "bytes=0-0"})
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError("The NCBI source metadata request failed.") from exc
    try:
        size = int(response.headers.get("Content-Length", 0) or 0)
    except ValueError:
        size = 0
    return {
        "size": size,
        "last_modified": response.headers.get("Last-Modified"),
        "source_url": str(response.url),
    }


###############################################################################
def _validate_master_list_file(self, path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0 or not zipfile.is_zipfile(path):
        raise RuntimeError("The LiverTox master list is not a valid XLSX file.")
    workbook = None
    try:
        payload = path.read_bytes()
        workbook = load_workbook(io.BytesIO(payload), read_only=True, data_only=True)
        if not workbook.sheetnames:
            raise RuntimeError("The LiverTox master list has no worksheet.")
        frame = pd.read_excel(
            io.BytesIO(payload),
            engine="openpyxl",
            header=self.header_row,
            skiprows=0,
        )
        sanitized = livertox_parse.sanitize_livertox_master_list(self, frame)
        if sanitized is None or sanitized.empty:
            raise RuntimeError("The LiverTox master list has no usable rows.")
    except (OSError, ValueError, ImportError, zipfile.BadZipFile) as exc:
        raise RuntimeError("The LiverTox master list could not be validated.") from exc
    finally:
        if workbook is not None:
            workbook.close()


###############################################################################
def _validate_archive_file(self, path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0 or not tarfile.is_tarfile(path):
        raise RuntimeError("The LiverTox archive is not a valid tarball.")
    supported_members = 0
    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                _validate_tar_member(member)
                if (
                    member.isfile()
                    and PurePosixPath(member.name).suffix.lower()
                    in self.supported_extensions
                ):
                    supported_members += 1
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError("The LiverTox archive could not be validated.") from exc
    if supported_members == 0:
        raise RuntimeError("The LiverTox archive contained no supported monographs.")


###############################################################################
def _validate_tar_member(member: tarfile.TarInfo) -> None:
    if not member.isfile():
        if member.isdir():
            _validate_member_name(member.name)
            return
        raise RuntimeError("The LiverTox archive contained an unsupported member.")
    _validate_member_name(member.name)


###############################################################################
def _validate_member_name(member_name: str) -> None:
    if "\\" in member_name:
        raise RuntimeError("The LiverTox archive contained an unsafe path.")
    normalized = PurePosixPath(member_name)
    if (
        normalized.is_absolute()
        or not member_name
        or any(part in {"", ".", ".."} for part in normalized.parts)
        or member_name.startswith("..")
    ):
        raise RuntimeError("The LiverTox archive contained an unsafe path.")


###############################################################################
def _build_litarch_archive_url(path: str, *, accession: str) -> str:
    normalized = path.strip()
    if normalized.startswith("pub/litarch/"):
        normalized = normalized[len("pub/litarch/") :]
    expected_filename = f"livertox_{accession}.tar.gz".lower()
    normalized_lower = normalized.lower()
    if (
        not normalized
        or normalized.startswith("/")
        or "://" in normalized
        or "\\" in normalized
        or any(part in {"", ".", ".."} for part in PurePosixPath(normalized).parts)
        or not (
            normalized_lower == expected_filename
            or normalized_lower.endswith(f"/{expected_filename}")
        )
    ):
        raise RuntimeError("NLM LitArch returned an unsafe archive path.")
    url = f"{NLM_LITARCH_BASE_URL.rstrip('/')}/{normalized}"
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "ftp.ncbi.nlm.nih.gov":
        raise RuntimeError("NLM LitArch returned an unexpected archive host.")
    return url


###############################################################################
def _contact_email(self) -> str:
    return str(getattr(self, "ncbi_contact_email", DEFAULT_NCBI_CONTACT_EMAIL))


###############################################################################
def _http_client(self) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=get_server_settings().runtime.livertox_download_timeout,
        headers=self.http_headers,
        follow_redirects=True,
    )


###############################################################################
def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.part")


###############################################################################
def _remove_partial(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Unable to remove a temporary LiverTox download file")
