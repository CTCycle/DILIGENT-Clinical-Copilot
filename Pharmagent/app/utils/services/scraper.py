from __future__ import annotations

import html
import os
import re
import tarfile
import unicodedata
from typing import Any

import httpx
import pandas as pd
from tqdm import tqdm

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
        )
        self.image_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".tiff",
        )

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
        with tarfile.open(normalized_path, "r:gz") as archive:
            for member in archive.getmembers():
                # if not self._should_process_member(member):
                #     continue
                content = self._read_member_content(archive, member)
                if not content:
                    continue
                plain_text = self._html_to_text(content)
                nbk_id = self._extract_nbk(member.name, content)
                record_id = nbk_id or self._derive_identifier(member.name)
                title = self._extract_title(content, plain_text, record_id)
                if not title:
                    continue
                collected[title] = {
                    "nbk_id": record_id,                    
                    "excerpt": plain_text,
                }
        return list(collected.values())    

    # -----------------------------------------------------------------------------
    def _read_member_content(
        self, archive: tarfile.TarFile, member: tarfile.TarInfo
    ) -> str | None:
        extracted = archive.extractfile(member)
        if extracted is None:
            return None
        data = extracted.read()
        if not data:
            return None
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
