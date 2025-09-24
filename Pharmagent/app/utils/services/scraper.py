from __future__ import annotations

import html
import io
import os
import re
import tarfile
import unicodedata
from typing import Any

import httpx
import pandas as pd
from tqdm import tqdm

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except ModuleNotFoundError:
    pdfminer_extract_text = None

try:
    from pypdf import PdfReader
except ModuleNotFoundError:
    PdfReader = None

from Pharmagent.app.constants import (
    SOURCES_PATH,
    LIVERTOX_ARCHIVE,
    LIVERTOX_BASE_URL,
)


###############################################################################
class LiverToxClient:
    def __init__(self) -> None:
        self.base_url = LIVERTOX_BASE_URL
        self.file_name = LIVERTOX_ARCHIVE
        self.tar_file_path = os.path.join(SOURCES_PATH, self.file_name)
        self.chunk_size = 8192
        self.supported_extensions = (
            ".html",
            ".htm",
            ".xhtml",
            ".xml",
            ".nxml",
            ".pdf",
        )
        self.image_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".tiff",
        )
        self.extension_priority = {
            ".nxml": 0,
            ".xml": 1,
            ".html": 2,
            ".htm": 2,
            ".xhtml": 2,
            ".pdf": 3,
        }

    # -------------------------------------------------------------------------
    async def download_bulk_data(self, dest_path: str) -> dict[str, Any]:
        url = self.base_url + self.file_name
        async with httpx.AsyncClient(timeout=30.0) as client:
            # HEAD request for size and last-modified
            head_response = await client.head(url)
            head_response.raise_for_status()
            file_size = int(head_response.headers.get("Content-Length", 0))
            last_modified = head_response.headers.get("Last-Modified", None)

            dest_dir = os.path.abspath(dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            file_path = os.path.join(dest_dir, self.file_name)

            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with (
                    open(file_path, "wb") as f,
                    tqdm(
                        total=file_size,
                        unit="B",
                        unit_scale=True,
                        desc=self.file_name,
                        ncols=80,
                    ) as pbar,
                ):
                    async for chunk in response.aiter_bytes(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        return {
            "file_path": file_path,
            "size": file_size,
            "last_modified": last_modified,
        }

    # -------------------------------------------------------------------------
    def convert_file_to_dataframe(self) -> pd.DataFrame:
        records = []
        with tarfile.open(self.tar_file_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name.lower()
                if name.endswith(".csv") or name.endswith(".tsv"):
                    fileobj = tar.extractfile(member)
                    if fileobj is None:
                        continue
                    df = pd.read_csv(
                        fileobj, sep="\t" if name.endswith(".tsv") else ","
                    )
                    records.append(df)

        if not records:
            raise ValueError("No supported tabular files found in archive.")

        return pd.concat(records, ignore_index=True)

    # -----------------------------------------------------------------------------
    def collect_monographs(self, archive_path: str | None = None) -> list[dict[str, str]]:
        tar_path = archive_path or self.tar_file_path
        normalized_path = os.path.abspath(tar_path)
        if not os.path.isfile(normalized_path):
            raise FileNotFoundError(f"LiverTox archive missing at {normalized_path}")
        if not tarfile.is_tarfile(normalized_path):
            raise RuntimeError(f"Invalid LiverTox archive at {normalized_path}")

        collected: dict[str, dict[str, str]] = {}
        priorities: dict[str, int] = {}
        with tarfile.open(normalized_path, "r:gz") as archive:
            for member in tqdm(archive.getmembers(), desc="Extracting LiverTox files"):
                if not self._should_process_member(member):
                    continue
                payload = self._read_member_payload(archive, member)
                if payload is None:
                    continue
                plain_text, markup_text = payload
                if not plain_text:
                    continue
                nbk_id = self._extract_nbk(member.name, markup_text or plain_text)
                record_key = nbk_id or self._derive_identifier(member.name)
                if not record_key:
                    continue
                priority = self.extension_priority.get(
                    os.path.splitext(member.name.lower())[1],
                    len(self.extension_priority) + 1,
                )
                existing_priority = priorities.get(record_key)
                if existing_priority is not None and existing_priority < priority:
                    continue
                if existing_priority is not None and existing_priority == priority:
                    current_excerpt = collected[record_key]["excerpt"]
                    if len(current_excerpt) >= len(plain_text):
                        continue
                record_nbk = nbk_id or record_key
                drug_name = self._extract_title(markup_text or "", plain_text, record_nbk)
                cleaned_text = plain_text.strip()
                if not drug_name or not cleaned_text:
                    continue
                record = {
                    "nbk_id": record_nbk,
                    "drug_name": drug_name,
                    "excerpt": cleaned_text,
                    "text": cleaned_text,
                }
                collected[drug_name] = record
                priorities[drug_name] = priority
                
        return list(collected.values())

    # -----------------------------------------------------------------------------
    def _should_process_member(self, member: tarfile.TarInfo) -> bool:
        if not member.isfile():
            return False
        if member.size == 0:
            return False
        lower_name = member.name.lower()
        if lower_name.endswith(self.image_extensions):
            return False
        _, ext = os.path.splitext(lower_name)
        if ext not in self.supported_extensions:
            return False
        return True

    # -----------------------------------------------------------------------------
    def _read_member_payload(
        self, archive: tarfile.TarFile, member: tarfile.TarInfo
    ) -> tuple[str, str | None] | None:
        extracted = archive.extractfile(member)
        if extracted is None:
            return None
        data = extracted.read()
        if not data:
            return None
        return self._convert_member_bytes(member.name, data)

    # -----------------------------------------------------------------------------
    def _convert_member_bytes(
        self, member_name: str, data: bytes
    ) -> tuple[str, str | None] | None:
        lower_name = member_name.lower()
        if lower_name.endswith(".pdf"):
            text = self._pdf_to_text(data)
            if text.strip():
                return text, None
            decoded = self._decode_markup(data)
            return decoded, decoded
        markup = self._decode_markup(data)
        text = self._html_to_text(markup)
        return text, markup

    # -----------------------------------------------------------------------------
    def _decode_markup(self, data: bytes) -> str:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="ignore")

    # -----------------------------------------------------------------------------
    def _pdf_to_text(self, data: bytes) -> str:
        buffer = io.BytesIO(data)
        if pdfminer_extract_text is not None:
            try:
                buffer.seek(0)
                text = pdfminer_extract_text(buffer)
                if text:
                    return text
            except Exception:
                buffer.seek(0)
        if PdfReader is not None:
            try:
                buffer.seek(0)
                reader = PdfReader(buffer)
                collected: list[str] = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        collected.append(page_text)
                if collected:
                    return "\n".join(collected)
            except Exception:
                buffer.seek(0)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="ignore")

    # -----------------------------------------------------------------------------
    def _extract_nbk(self, member_name: str, content: str) -> str | None:
        match = re.search(r"NBK\d+", member_name, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        match = re.search(r"NBK\d+", content, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        return None

    # -----------------------------------------------------------------------------
    def _derive_identifier(self, member_name: str) -> str:
        base = os.path.basename(member_name)
        stem = os.path.splitext(base)[0]
        cleaned = self._normalize_whitespace(self._strip_punctuation(stem))
        return cleaned or base

    # -----------------------------------------------------------------------------
    def _extract_title(self, html_text: str, plain_text: str, default: str) -> str:
        patterns = (
            r"<title[^>]*>(.*?)</title>",
            r"<article-title[^>]*>(.*?)</article-title>",
            r"<h1[^>]*>(.*?)</h1>",
        )
        for pattern in patterns:
            match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                fragment = self._clean_fragment(match.group(1))
                if fragment:
                    return fragment
        for line in plain_text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return default

    # -----------------------------------------------------------------------------
    def _clean_fragment(self, fragment: str) -> str:
        return self._html_to_text(fragment)

    # -----------------------------------------------------------------------------
    def _html_to_text(self, html_text: str) -> str:
        stripped = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html_text)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        unescaped = html.unescape(stripped)
        return self._normalize_whitespace(unescaped)

    # -----------------------------------------------------------------------------
    def _normalize_whitespace(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    # -----------------------------------------------------------------------------
    def _strip_punctuation(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        folded = "".join(char for char in normalized if not unicodedata.combining(char))
        return re.sub(r"[-_,.;:()\[\]{}\/\\]", " ", folded)
