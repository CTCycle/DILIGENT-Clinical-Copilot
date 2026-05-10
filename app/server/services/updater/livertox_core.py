from __future__ import annotations

import asyncio
import html
import io
import json
import multiprocessing
import os
import re
import tarfile
import unicodedata
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from datetime import UTC, datetime
from typing import Any, cast
from collections.abc import Callable

import httpx
import pandas as pd
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pypdf import PdfReader
from tqdm import tqdm

from configurations.startup import server_settings
from common.constants import LIVERTOX_BASE_URL, ARCHIVES_PATH
from common.utils.logger import logger
from services.text.normalization import normalize_whitespace
from services.updater.sanitizer import LiverToxExcerptSanitizer
from repositories.serialization.data import DataSerializer

SUPPORTED_MONOGRAPH_EXTENSIONS = (".html", ".htm", ".xhtml", ".xml", ".nxml", ".pdf")
NBK_ID_PATTERN = re.compile(r"^NBK\d+$", re.IGNORECASE)

DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "DILIGENTClinicalCopilot/1.0 (contact=clinical-copilot@pharmagent.local)"
    )
}

DOWNLOAD_CHUNK_SIZE = 262_144


# -----------------------------------------------------------------------------
def load_json(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


# -----------------------------------------------------------------------------
def save_masterlist_metadata(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


# -----------------------------------------------------------------------------
def metadata_matches(stored: dict[str, Any], remote: dict[str, Any]) -> bool:
    return stored.get("last_modified") == remote.get("last_modified") and int(
        stored.get("size", 0)
    ) == int(remote.get("size", 0))


# -----------------------------------------------------------------------------
def process_monograph_payload(
    member_name: str,
    data: bytes,
) -> dict[str, str] | None:
    return livertox_parse.process_monograph_member(member_name, data)


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
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        with (
            open(destination, "wb") as output,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=label,
                ncols=80,
            ) as progress,
        ):
            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                if chunk:
                    output.write(chunk)
                    progress.update(len(chunk))


###############################################################################
from services.updater import livertox_download, livertox_index, livertox_parse

class LiverToxUpdater:
    def __init__(
        self,
        sources_path: str,
        *,
        redownload: bool,
        archive_name: str | None = None,
        monograph_max_workers: int | None = None,
        serializer: DataSerializer | None = None,
    ) -> None:
        self.supported_extensions = SUPPORTED_MONOGRAPH_EXTENSIONS
        self.http_headers = dict(DEFAULT_HTTP_HEADERS)
        self.delay = 0.5
        self.chunk_size = DOWNLOAD_CHUNK_SIZE

        self.sources_path = os.path.abspath(sources_path)
        self.redownload = redownload
        self.serializer = serializer or DataSerializer()
        self.excerpt_sanitizer = LiverToxExcerptSanitizer()
        self.header_row = 1

        self.base_url = LIVERTOX_BASE_URL
        self.file_name = (
            archive_name or server_settings.external_data.livertox_archive
        ).strip()
        self.monograph_max_workers = max(
            1,
            int(
                monograph_max_workers
                if monograph_max_workers is not None
                else server_settings.external_data.livertox_monograph_max_workers
            ),
        )
        self.tar_file_path = os.path.join(ARCHIVES_PATH, self.file_name)
        self.master_list_path = os.path.join(ARCHIVES_PATH, "LiverTox_Master_List.xlsx")
        self.master_list_metadata_path = os.path.join(
            ARCHIVES_PATH, "livertox_master_list.metadata.json"
        )
        self.archive_metadata_path = os.path.join(
            ARCHIVES_PATH, "livertox_archive.metadata.json"
        )

    # -------------------------------------------------------------------------
    def emit_progress(
        self,
        progress_callback: Callable[[float, str], None] | None,
        *,
        progress: float,
        message: str,
    ) -> None:
        if progress_callback is None:
            return
        bounded_progress = min(100.0, max(0.0, float(progress)))
        progress_callback(bounded_progress, message)

    # -------------------------------------------------------------------------
    def should_cancel(self, should_stop: Callable[[], bool] | None) -> bool:
        if should_stop is None:
            return False
        return bool(should_stop())

    # -------------------------------------------------------------------------
    def sanitize_livertox_master_list(self, data: pd.DataFrame) -> pd.DataFrame | None:
        return livertox_parse.sanitize_livertox_master_list(self, data)

    # -------------------------------------------------------------------------
    def clean_master_list_column(self, series: pd.Series) -> pd.Series:
        return livertox_parse.clean_master_list_column(self, series)

    # -------------------------------------------------------------------------
    async def download_bulk_data(self, dest_path: str) -> dict[str, Any]:
        return await livertox_download.download_bulk_data(self, dest_path)

    # -------------------------------------------------------------------------
    def refresh_master_list(self) -> tuple[dict[str, Any], pd.DataFrame]:
        return livertox_download.refresh_master_list(self)

    # -------------------------------------------------------------------------
    async def download_master_list(self) -> dict[str, Any]:
        return await livertox_download.download_master_list(self)

    # -------------------------------------------------------------------------
    async def resolve_master_list_url(self, client: httpx.AsyncClient) -> str:
        return await livertox_download.resolve_master_list_url(self, client)

    # -------------------------------------------------------------------------
    async def resolve_master_list_from_bookshelf(
        self, client: httpx.AsyncClient
    ) -> str:
        return await livertox_download.resolve_master_list_from_bookshelf(self, client)

    # -------------------------------------------------------------------------
    async def resolve_master_list_from_bin(
        self, client: httpx.AsyncClient, base_url: str
    ) -> str:
        return await livertox_download.resolve_master_list_from_bin(self, client, base_url)

    # -------------------------------------------------------------------------
    async def resolve_master_list_via_datagov(self, client: httpx.AsyncClient) -> str:
        return await livertox_download.resolve_master_list_via_datagov(self, client)

    # -------------------------------------------------------------------------
    def normalize_datagov_resource_url(self, url: str) -> str | None:
        return livertox_download.normalize_datagov_resource_url(self, url)

    # -------------------------------------------------------------------------
    async def probe_master_list_candidate(
        self, client: httpx.AsyncClient, candidate: str
    ) -> str:
        return await livertox_download.probe_master_list_candidate(self, client, candidate)

    # -------------------------------------------------------------------------
    async def fetch_candidate_with_get(
        self, client: httpx.AsyncClient, candidate: str
    ) -> httpx.Response:
        return await livertox_download.fetch_candidate_with_get(self, client, candidate)

    # -----------------------------------------------------------------------------
    def update_from_livertox(
        self,
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        return livertox_index.update_from_livertox(self, progress_callback=progress_callback, should_stop=should_stop)

    # -------------------------------------------------------------------------
    def collect_local_archive_info(self, archive_path: str) -> dict[str, Any]:
        return livertox_download.collect_local_archive_info(self, archive_path)

    # -----------------------------------------------------------------------------
    def build_unified_dataset(
        self,
        monographs: pd.DataFrame,
        master_frame: pd.DataFrame,
        master_metadata: dict[str, Any],
    ) -> pd.DataFrame:
        return livertox_index.build_unified_dataset(self, monographs, master_frame, master_metadata)

    # -------------------------------------------------------------------------
    def contains_symbol(self, value: str) -> bool:
        return livertox_parse.contains_symbol(self, value)

    # -------------------------------------------------------------------------
    def sanitize_unified_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        return livertox_index.sanitize_unified_dataset(self, frame)

    # -----------------------------------------------------------------------------
    def collect_monographs(
        self,
        archive_path: str | None = None,
        *,
        should_stop: Callable[[], bool] | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[dict[str, str]]:
        livertox_parse.ProcessPoolExecutor = ProcessPoolExecutor
        return livertox_parse.collect_monographs(self, archive_path, should_stop=should_stop, progress_callback=progress_callback)

    # -------------------------------------------------------------------------
    def emit_monograph_progress(
        self,
        *,
        progress_callback: Callable[[float, str], None] | None,
        processed_count: int,
        total_payloads: int,
    ) -> None:
        return livertox_parse.emit_monograph_progress(self, progress_callback=progress_callback, processed_count=processed_count, total_payloads=total_payloads)

    # -------------------------------------------------------------------------
    def drain_monograph_futures(
        self,
        *,
        in_flight: dict[Future[dict[str, str] | None], str],
        collected: list[dict[str, str]],
        processed_count: int,
        total_payloads: int,
        progress_callback: Callable[[float, str], None] | None,
        wait_for_one: bool,
    ) -> int:
        return livertox_parse.drain_monograph_futures(self, in_flight=in_flight, collected=collected, processed_count=processed_count, total_payloads=total_payloads, progress_callback=progress_callback, wait_for_one=wait_for_one)

    # -------------------------------------------------------------------------
    def sort_monograph_records(
        self, records: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        return livertox_parse.sort_monograph_records(self, records)

    # -------------------------------------------------------------------------
    @staticmethod
    def process_monograph_member(
        member_name: str,
        data: bytes,
    ) -> dict[str, str] | None:
        return livertox_parse.process_monograph_member(member_name, data)

    # -------------------------------------------------------------------------
    @staticmethod
    def convert_member_bytes(
        member_name: str, data: bytes
    ) -> tuple[str, str | None] | None:
        return livertox_parse.convert_member_bytes(member_name, data)

    # -------------------------------------------------------------------------
    @staticmethod
    def decode_markup(data: bytes) -> str:
        return livertox_parse.decode_markup(data)

    # -------------------------------------------------------------------------
    @staticmethod
    def pdf_to_text(data: bytes) -> str:
        return livertox_parse.pdf_to_text(data)

    # -------------------------------------------------------------------------
    @staticmethod
    def extract_nbk(member_name: str, content: str) -> str | None:
        return livertox_parse.extract_nbk(member_name, content)

    # -------------------------------------------------------------------------
    @staticmethod
    def derive_identifier(member_name: str) -> str:
        return livertox_parse.derive_identifier(member_name)

    # -------------------------------------------------------------------------
    @staticmethod
    def extract_title(html_text: str, plain_text: str, default: str) -> str:
        return livertox_parse.extract_title(html_text, plain_text, default)

    # -------------------------------------------------------------------------
    @staticmethod
    def clean_fragment(fragment: str) -> str:
        return livertox_parse.clean_fragment(fragment)

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_extracted_title(value: str) -> str:
        return livertox_parse.normalize_extracted_title(value)

    # -------------------------------------------------------------------------
    @staticmethod
    def html_to_text(html_text: str) -> str:
        return livertox_parse.html_to_text(html_text)

    # -------------------------------------------------------------------------
    @staticmethod
    def strip_punctuation(value: str) -> str:
        return livertox_parse.strip_punctuation(value)

    # -------------------------------------------------------------------------
    def sanitize_records(self, entries: list[dict[str, Any]]) -> pd.DataFrame:
        return livertox_parse.sanitize_records(self, entries)

    # -------------------------------------------------------------------------
    def sanitize_excerpt(self, value: Any) -> str | Any:
        return livertox_parse.sanitize_excerpt(self, value)

    # -------------------------------------------------------------------------
    def finalize_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        return livertox_index.finalize_dataset(self, frame)

    # -------------------------------------------------------------------------
    def normalize_nbk_id(self, value: Any) -> str | None:
        return livertox_parse.normalize_nbk_id(self, value)

