from __future__ import annotations

import asyncio
import html
import io
import json
import os
import re
import tarfile
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

import httpx
import pandas as pd
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pypdf import PdfReader
from tqdm import tqdm

from Pharmagent.app.constants import (
    LIVERTOX_ARCHIVE,
    LIVERTOX_BASE_URL,
    SOURCES_PATH,
)
from Pharmagent.app.logger import logger
from Pharmagent.app.utils.database.sqlite import database
from Pharmagent.app.utils.serializer import DataSerializer
from Pharmagent.app.utils.services.retrieval import RxNavClient


MATCHING_STOPWORDS = {
    "and",
    "apply",
    "caps",
    "capsule",
    "capsules",
    "chewable",
    "cream",
    "dose",
    "doses",
    "drink",
    "drops",
    "elixir",
    "enteric",
    "extended",
    "foam",
    "for",
    "free",
    "gel",
    "granules",
    "im",
    "inj",
    "injection",
    "intramuscular",
    "intravenous",
    "iv",
    "kit",
    "liquid",
    "lotion",
    "mg",
    "ml",
    "nasal",
    "ointment",
    "ophthalmic",
    "oral",
    "plus",
    "pack",
    "packet",
    "packets",
    "combo",
    "combination",
    "of",
    "or",
    "patch",
    "po",
    "powder",
    "prefilled",
    "release",
    "sc",
    "sol",
    "solution",
    "soln",
    "spray",
    "sterile",
    "subcutaneous",
    "suppository",
    "susp",
    "suspension",
    "sustained",
    "syringe",
    "syrup",
    "tablet",
    "tablets",
    "the",
    "topical",
    "treat",
    "treatment",
    "therapy",
    "vial",
    "use",
    "with",
    "without",
}

SUPPORTED_MONOGRAPH_EXTENSIONS = (".html", ".htm", ".xhtml", ".xml", ".nxml", ".pdf")

DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "PharmagentClinicalCopilot/1.0 (contact=clinical-copilot@pharmagent.local)"
    )
}

DOWNLOAD_CHUNK_SIZE = 262_144

# -----------------------------------------------------------------------------
def _load_json(path: str) -> dict[str, Any] | None:
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
def _metadata_matches(stored: dict[str, Any], remote: dict[str, Any]) -> bool:
    return (
        stored.get("last_modified") == remote.get("last_modified")
        and int(stored.get("size", 0)) == int(remote.get("size", 0))
    )


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
@dataclass(slots=True)
class MonographRecord:
    nbk_id: str
    drug_name: str
    normalized_name: str
    excerpt: str | None
    synonyms: dict[str, str]
    tokens: set[str]


###############################################################################
@dataclass(slots=True)
class LiverToxMatch:
    nbk_id: str
    matched_name: str
    confidence: float
    reason: str
    notes: list[str]
    record: MonographRecord | None = None


###############################################################################
class LiverToxUpdater:
    
    def __init__(
        self,
        sources_path: str,
        *,
        redownload: bool,
        rx_client: RxNavClient | None = None,
        serializer: DataSerializer | None = None,
        database_client=database,        
    ) -> None:        
        self.supported_extensions = SUPPORTED_MONOGRAPH_EXTENSIONS
        self.http_headers = dict(DEFAULT_HTTP_HEADERS)
        self.delay = 0.5
        self.chunk_size = DOWNLOAD_CHUNK_SIZE

        self.sources_path = os.path.abspath(sources_path)
        self.redownload = redownload
        self.rx_client = rx_client or RxNavClient()
        self.serializer = serializer or DataSerializer()
        self.database = database_client
        self.header_row = 1

        self.base_url = LIVERTOX_BASE_URL
        self.file_name = LIVERTOX_ARCHIVE
        self.tar_file_path = os.path.join(SOURCES_PATH, self.file_name)
        self.master_list_path = os.path.join(
            SOURCES_PATH, "LiverTox_Master_List.xlsx"
        )
        self.master_list_metadata_path = os.path.join(
            SOURCES_PATH, "livertox_master_list.metadata.json"
        )
        self.archive_metadata_path = os.path.join(
            SOURCES_PATH, "livertox_archive.metadata.json"
        )
        
    # -------------------------------------------------------------------------
    def sanitize_livertox_master_list(self, data: pd.DataFrame) -> pd.DataFrame | None:
        if data.empty:
            return 
        
        column_mapping = {
            "Count": "reference_count",
            "Ingredient": "ingredient",
            "Brand Name": "brand_name",
            "Likelihood Score": "likelihood_score",
            "Chapter Title": "chapter_title",
            "Last Update": "last_update",
            "Year Approved": "year_approved",
            "Type of Agent": "agent_classification",
            "In LiverTox": "include_in_livertox",
            "Primary Classification": "primary_classification",
            "Secondary Classification": "secondary_classification",
        }

        data = data.rename(columns=lambda s: re.sub(r'\s+', ' ', s).strip())
        data = data.rename(columns=column_mapping)

        required_columns = list(column_mapping.values())
        for column in required_columns:
            if column not in data.columns:
                data[column] = pd.NA

        data = data[required_columns]

        text_columns = [
            "ingredient",
            "brand_name",
            "likelihood_score",
            "chapter_title",
            "agent_classification",
            "primary_classification",
            "secondary_classification",
        ]
        for column in text_columns:
            data[column] = self._clean_master_list_column(data[column])

        data = data.dropna(subset=["ingredient", "brand_name"])

        invalid_headers = {
            "ingredient": {"ingredient", "count"},
            "brand_name": {"brand name"},
        }
        for column, values in invalid_headers.items():
            data = data[
                ~data[column]
                .fillna("")
                .str.lower()
                .isin(values)
            ]

        data["last_update"] = pd.to_datetime(data["last_update"], errors="coerce")
        data["reference_count"] = pd.to_numeric(
            data["reference_count"], errors="coerce"
        )
        data["year_approved"] = pd.to_numeric(
            data["year_approved"], errors="coerce"
        )

        data = data.drop_duplicates(subset=["ingredient", "brand_name"], keep="last")

        return data.reset_index(drop=True)

    # -------------------------------------------------------------------------
    def _clean_master_list_column(self, series: pd.Series) -> pd.Series:
        cleaned = series.fillna("").astype(str).str.strip()
        cleaned = cleaned.replace("", pd.NA)
        return cleaned

    # -------------------------------------------------------------------------
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
            stored_metadata = _load_json(self.archive_metadata_path)
            if (
                stored_metadata
                and os.path.isfile(file_path)
                and _metadata_matches(stored_metadata, metadata)
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
            save_masterlist_metadata(self.archive_metadata_path, metadata)

        return {
            "file_path": file_path,
            "size": metadata.get("size", 0),
            "last_modified": metadata.get("last_modified"),
            "downloaded": True,
            "source_url": metadata["source_url"],
        }

    # -------------------------------------------------------------------------
    def refresh_master_list(self) -> dict[str, Any]:
        logger.info("Refreshing LiverTox master list")
        metadata = asyncio.run(self._download_master_list())
        
        frame = pd.read_excel(
            metadata["file_path"],
            engine="openpyxl",
            header=self.header_row,
            skiprows=0,
        )
        sanitized = self.sanitize_livertox_master_list(frame)
        if sanitized is None or sanitized.empty:
            return {}
        
        self.serializer.save_livertox_master_list(
            sanitized,
            source_url=metadata["source_url"],
            last_modified=metadata.get("last_modified"),
        )
        metadata["records"] = len(sanitized.index)
        return metadata

    # -------------------------------------------------------------------------
    async def _download_master_list(self) -> dict[str, Any]:
        async with httpx.AsyncClient(
            timeout=30.0, headers=self.http_headers, follow_redirects=True
        ) as client:
            master_url = await self._resolve_master_list_url(client)
            head = await client.head(master_url)
            head.raise_for_status()
            metadata = {
                "size": int(head.headers.get("Content-Length", 0)),
                "last_modified": head.headers.get("Last-Modified"),
                "source_url": str(head.url),
            }
            stored_metadata = _load_json(self.master_list_metadata_path)
            if (
                stored_metadata
                and os.path.isfile(self.master_list_path)
                and _metadata_matches(stored_metadata, metadata)
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
            save_masterlist_metadata(self.master_list_metadata_path, metadata)

        return {
            "file_path": self.master_list_path,
            "size": metadata.get("size", 0),
            "last_modified": metadata.get("last_modified"),
            "downloaded": True,
            "source_url": metadata["source_url"],
        }
    
    # -------------------------------------------------------------------------
    async def _resolve_master_list_url(self, client: httpx.AsyncClient) -> str:
        try:
            return await self._resolve_master_list_from_bookshelf(client)
        except Exception as exc:
            logger.warning("Bookshelf Excel lookup failed: %s", exc)
        try:
            return await self._resolve_master_list_from_bin(client, self.base_url)
        except Exception as exc:
            logger.warning("Primary FTP lookup failed: %s", exc)
            fallback_url = await self._resolve_master_list_via_datagov(client)
            return fallback_url

    # -------------------------------------------------------------------------
    async def _resolve_master_list_from_bookshelf(
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
                return await self._probe_master_list_candidate(client, str(candidate))
            content_type = (head_response.headers.get("Content-Type") or "").lower()
            disposition = (
                head_response.headers.get("Content-Disposition") or ""
            ).lower()
            if "excel" in content_type or ".xlsx" in disposition:
                return await self._probe_master_list_candidate(
                    client, str(head_response.url)
                )

        get_response = await client.get(report_url, follow_redirects=False)
        if get_response.status_code in redirect_statuses and get_response.headers.get(
            "Location"
        ):
            candidate = httpx.URL(report_url).join(get_response.headers["Location"])
            return await self._probe_master_list_candidate(client, str(candidate))

        content_type = (get_response.headers.get("Content-Type") or "").lower()
        disposition = (get_response.headers.get("Content-Disposition") or "").lower()
        if "excel" in content_type or ".xlsx" in disposition:
            return await self._probe_master_list_candidate(client, str(get_response.url))

        html_content = get_response.text
        for pattern in (
            r"url=([^\"'>]+\.xlsx)",
            r"['\"]([^'\"]+\.xlsx)['\"]",
        ):
            for match in re.finditer(pattern, html_content, flags=re.IGNORECASE):
                candidate_url = match.group(1)
                candidate = httpx.URL(report_url).join(candidate_url)
                try:
                    return await self._probe_master_list_candidate(
                        client, str(candidate)
                    )
                except Exception as exc:  # pragma: no cover - network dependent
                    logger.debug(
                        "Bookshelf candidate %s failed: %s", str(candidate), exc
                    )
                    continue

        raise RuntimeError("Unable to resolve master list via Bookshelf report page")

    # -------------------------------------------------------------------------
    async def _resolve_master_list_from_bin(
        self, client: httpx.AsyncClient, base_url: str
    ) -> str:
        bin_url = str(httpx.URL(base_url).join("bin/"))
        response = await client.get(bin_url)
        response.raise_for_status()
        content = response.text
        matches = re.finditer(
            r"<a[^>]+href=\"([^\"]+\.xlsx)\"[^>]*>(.*?)</a>",
            content,
            flags=re.IGNORECASE | re.DOTALL,
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
            if (
                not human_url.lower().startswith("https://www.ncbi.nlm.nih.gov/")
                and not human_url.startswith(base_url)
            ):
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
                if (
                    not human_url.lower().startswith("https://www.ncbi.nlm.nih.gov/")
                    and not human_url.startswith(base_url)
                ):
                    continue
                if "master" in os.path.basename(human_url).lower():
                    candidates.append((human_url, os.path.basename(human_url)))
        if not candidates:
            raise RuntimeError("Unable to locate LiverTox master list link on FTP bin page")
        candidates.sort(key=lambda item: item[0])
        chosen_url = candidates[0][0]
        return chosen_url

    # -------------------------------------------------------------------------
    async def _resolve_master_list_via_datagov(
        self, client: httpx.AsyncClient
    ) -> str:
        api_url = "https://catalog.data.gov/api/3/action/package_show"
        response = await client.get(api_url, params={"id": "livertox"})
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Unable to resolve FTP folder from Data.gov entry") from exc
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
            normalized_url = self._normalize_datagov_resource_url(raw_url)
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
                resolved_direct = await self._probe_master_list_candidate(
                    client, candidate
                )
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                logger.debug(
                    "Candidate Data.gov direct %s failed: %s", candidate, exc
                )
                continue
            return resolved_direct

        if not folder_candidates:
            raise RuntimeError("Unable to resolve FTP folder from Data.gov entry")

        original_base = self.base_url
        for base_candidate in folder_candidates:
            try:
                resolved = await self._resolve_master_list_from_bin(client, base_candidate)
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
            raise RuntimeError("Unable to resolve FTP folder from Data.gov entry") from last_error
        raise RuntimeError("Unable to resolve FTP folder from Data.gov entry")

    # -------------------------------------------------------------------------
    def _normalize_datagov_resource_url(self, url: str) -> str | None:
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

    # -------------------------------------------------------------------------
    async def _probe_master_list_candidate(
        self, client: httpx.AsyncClient, candidate: str
    ) -> str:
        try:
            response = await client.head(candidate)
            response.raise_for_status()
        except httpx.HTTPStatusError:  # pragma: no cover - network dependent
            response = await self._fetch_candidate_with_get(client, candidate)
        except httpx.HTTPError:  # pragma: no cover - network dependent
            response = await self._fetch_candidate_with_get(client, candidate)
        content_type = (response.headers.get("Content-Type") or "").lower()
        if ".xlsx" not in candidate.lower() and "excel" not in content_type:
            disposition = (response.headers.get("Content-Disposition") or "").lower()
            if ".xlsx" in disposition:
                return str(response.url)
            raise RuntimeError("Candidate does not appear to be an Excel file")
        return str(response.url)

    # -------------------------------------------------------------------------
    async def _fetch_candidate_with_get(
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

    # -----------------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        logger.info("Starting LiverTox update")
        master_info = self.refresh_master_list()
        
        archive_path = os.path.join(self.sources_path, LIVERTOX_ARCHIVE)

        download_info: dict[str, Any] = {}
        if self.redownload:
            logger.info("Redownload flag enabled; fetching latest LiverTox archive")
            download_info = asyncio.run(self.download_bulk_data(self.sources_path))
        else:
            logger.info("Using existing LiverTox archive")

        local_info = self.collect_local_archive_info(archive_path)
        logger.info("Extracting LiverTox monographs from %s", archive_path)
        extracted = self.collect_monographs(archive_path)
        logger.info("Sanitizing %d extracted entries", len(extracted))
        records = self.sanitize_records(extracted)
        logger.info("Enriching %d sanitized entries with RxNav terms", len(records))
        enriched = self.enrich_records(records)
        logger.info("Persisting enriched records to database")
        self.serializer.save_livertox_records(enriched)
             
        payload = {**master_info, **download_info, **local_info}
        payload["processed_entries"] = len(enriched)
        payload["records"] = len(enriched)
        logger.info("LiverTox update completed successfully")

        return payload
    
    # -------------------------------------------------------------------------
    def collect_local_archive_info(self, archive_path: str) -> dict[str, Any]:
        if not os.path.isfile(archive_path):
            raise RuntimeError(
                "LiverTox archive not found; enable REDOWNLOAD to fetch a fresh copy."
            )
        size = os.path.getsize(archive_path)
        modified = datetime.fromtimestamp(os.path.getmtime(archive_path), UTC).isoformat()
        return {"file_path": archive_path, "size": size, "last_modified": modified}

    # -----------------------------------------------------------------------------
    def collect_monographs(
        self, archive_path: str | None = None
    ) -> list[dict[str, str]]:
        tar_path = archive_path or self.tar_file_path
        normalized_path = os.path.abspath(tar_path)
        if not os.path.isfile(normalized_path):
            raise FileNotFoundError(f"LiverTox archive missing at {normalized_path}")
        if not tarfile.is_tarfile(normalized_path):
            raise RuntimeError(f"Invalid LiverTox archive at {normalized_path}")

        collected: list[dict[str, str]] = []
        processed_files: set[str] = set()
        with tarfile.open(normalized_path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            allowed_members: list[tarfile.TarInfo] = []
            for member in members:
                normalized_name = os.path.normpath(member.name)
                if os.path.isabs(normalized_name) or normalized_name.startswith(".."):
                    logger.warning("Skipping unsafe archive member: %s", member.name)
                    continue
                extension = os.path.splitext(normalized_name.lower())[1]
                if extension not in self.supported_extensions:
                    continue
                allowed_members.append(member)

            for member in tqdm(
                allowed_members,
                desc="Processing LiverTox files",
                total=len(allowed_members),
            ):
                base_name = os.path.basename(member.name).lower()
                if base_name in processed_files:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                try:
                    data = extracted.read()
                finally:
                    extracted.close()
                if not data:
                    continue
                payload = self._convert_member_bytes(member.name, data)
                if payload is None:
                    continue
                plain_text, markup_text = payload
                if not plain_text:
                    continue
                nbk_id = self._extract_nbk(member.name, markup_text or plain_text)
                record_nbk = nbk_id or self._derive_identifier(member.name)
                if not record_nbk:
                    continue
                drug_name = self._extract_title(
                    markup_text or "", plain_text, record_nbk
                )
                cleaned_text = plain_text.strip()
                if not drug_name or not cleaned_text:
                    continue
                record = {
                    "nbk_id": record_nbk,
                    "drug_name": drug_name,
                    "excerpt": cleaned_text
                }
                collected.append(record)
                processed_files.add(base_name)

        return collected

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def _decode_markup(self, data: bytes) -> str:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="ignore")

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def _extract_nbk(self, member_name: str, content: str) -> str | None:
        match = re.search(r"NBK\d+", member_name, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        match = re.search(r"NBK\d+", content, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        return None

    # -------------------------------------------------------------------------
    def _derive_identifier(self, member_name: str) -> str:
        base = os.path.basename(member_name)
        stem = os.path.splitext(base)[0]
        cleaned = self._normalize_whitespace(self._strip_punctuation(stem))
        return cleaned or base

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def _clean_fragment(self, fragment: str) -> str:
        return self._html_to_text(fragment)

    # -------------------------------------------------------------------------
    def _html_to_text(self, html_text: str) -> str:
        stripped = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html_text)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        unescaped = html.unescape(stripped)
        return self._normalize_whitespace(unescaped)

    # -------------------------------------------------------------------------
    def _normalize_whitespace(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    # -------------------------------------------------------------------------
    def _strip_punctuation(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        folded = "".join(char for char in normalized if not unicodedata.combining(char))
        return re.sub(r"[-_,.;:()\[\]{}\/\\]", " ", folded)

    # -------------------------------------------------------------------------
    def sanitize_records(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sanitized = self.serializer.sanitize_livertox_records(entries)
        if sanitized.empty:
            raise RuntimeError("No valid LiverTox monographs were available after sanitization.")
        return sanitized.to_dict(orient="records")

    # -------------------------------------------------------------------------
    def enrich_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for entry in records:
            drug_name = entry.get("drug_name")
            if not isinstance(drug_name, str) or not drug_name.strip():
                entry["synonyms"] = None
                continue
            try:
                synonyms = self.rx_client.fetch_drug_terms(drug_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to enrich '%s': %s", drug_name, exc)
                entry["synonyms"] = None
                continue
            entry["synonyms"] = ", ".join(synonyms) if synonyms else None
        return records


  

###############################################################################
class LiverToxMatcher:
    DIRECT_CONFIDENCE = 1.0
    MASTER_CONFIDENCE = 0.92
    SYNONYM_CONFIDENCE = 0.90
    PARTIAL_CONFIDENCE = 0.86
    FUZZY_CONFIDENCE = 0.84
    FUZZY_THRESHOLD = 0.85
    TOKEN_MAX_FREQUENCY = 3
    MIN_CONFIDENCE = 0.40

    # -------------------------------------------------------------------------
    def __init__(
        self,
        livertox_df: pd.DataFrame,
        master_list_df: pd.DataFrame | None = None,
    ) -> None:
        self.livertox_df = livertox_df
        self.master_list_df = master_list_df
        self.match_cache: dict[str, LiverToxMatch | None] = {}
        self.records: list[MonographRecord] = []
        self.primary_index: dict[str, MonographRecord] = {}
        self.synonym_index: dict[str, tuple[MonographRecord, str]] = {}
        self.variant_catalog: list[tuple[str, MonographRecord, str, bool]] = []
        self.token_occurrences: dict[str, list[MonographRecord]] = {}
        self.token_index: dict[str, list[MonographRecord]] = {}
        self.brand_index: dict[str, list[tuple[str, str]]] = {}
        self.ingredient_index: dict[str, list[tuple[str, str]]] = {}
        self.rows_by_nbk: dict[str, dict[str, Any]] = {}
        self._build_records()
        self._build_master_list_aliases()
        self._finalize_token_index()

    # -------------------------------------------------------------------------
    async def match_drug_names(
        self, patient_drugs: list[str]
    ) -> list[LiverToxMatch | None]:
        total = len(patient_drugs)
        if total == 0:
            return []
        results: list[LiverToxMatch | None] = [None] * total
        if not self.records:
            return results
        normalized_queries = [self._normalize_name(name) for name in patient_drugs]
        for idx, normalized in enumerate(normalized_queries):
            if not normalized:
                continue
            cached = self.match_cache.get(normalized)
            if cached is not None or normalized in self.match_cache:
                results[idx] = cached
                continue
            lookup = self._match_query(normalized)
            if lookup is None:
                self.match_cache[normalized] = None
                continue
            record, confidence, reason, notes = lookup
            match = self._create_match(record, confidence, reason, notes)
            self.match_cache[normalized] = match
            results[idx] = match
        return results

    # -------------------------------------------------------------------------
    def build_patient_mapping(
        self,
        patient_drugs: list[str],
        matches: list[LiverToxMatch | None],
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        nbk_to_row = self._ensure_row_index()
        for original, match in zip(patient_drugs, matches, strict=False):
            row_data: dict[str, Any] | None = None
            excerpts: list[str] = []
            if match is not None:
                row_data = dict(nbk_to_row.get(match.nbk_id, {})) or None
                excerpt_value = row_data.get("excerpt") if row_data else None
                if match.record and match.record.excerpt:
                    excerpts.append(match.record.excerpt)
                if isinstance(excerpt_value, str) and excerpt_value:
                    excerpts.append(excerpt_value)
            unique_excerpts = list(dict.fromkeys(excerpts))
            entries.append(
                {
                    "drug_name": original,
                    "matched_livertox_row": row_data,
                    "extracted_excerpts": unique_excerpts,
                }
            )
        return entries

    # -------------------------------------------------------------------------
    def _match_query(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        if not normalized_query:
            return None
        direct = self._match_primary(normalized_query)
        if direct is not None:
            return direct
        master = self._match_master_list(normalized_query)
        if master is not None:
            return master
        synonym = self._match_synonym(normalized_query)
        if synonym is not None:
            return synonym
        partial = self._match_partial(normalized_query)
        if partial is not None:
            return partial
        return self._match_fuzzy(normalized_query)

    # -------------------------------------------------------------------------
    def _match_primary(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        record = self.primary_index.get(normalized_query)
        if record is None:
            return None
        return record, self.DIRECT_CONFIDENCE, "monograph_name", []

    # -------------------------------------------------------------------------
    def _match_master_list(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        alias_sources = (
            ("brand", self.brand_index),
            ("ingredient", self.ingredient_index),
        )
        for alias_type, index in alias_sources:
            entries = index.get(normalized_query)
            if not entries:
                continue
            for alias_value, chapter_title in entries:
                resolved = self._match_chapter_title(chapter_title)
                if resolved is None:
                    continue
                record, base_confidence, chapter_reason, chapter_notes = resolved
                notes = [
                    f"{alias_type}='{alias_value}'",
                    f"chapter='{chapter_title}'",
                ]
                notes.extend(chapter_notes)
                reason = f"{alias_type}_{chapter_reason}"
                confidence = min(self.MASTER_CONFIDENCE, base_confidence)
                return record, confidence, reason, notes
        return None

    # -------------------------------------------------------------------------
    def _match_synonym(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        alias = self.synonym_index.get(normalized_query)
        if alias is None:
            return None
        record, original = alias
        notes = [f"synonym='{original}'"]
        return record, self.SYNONYM_CONFIDENCE, "synonym_match", notes

    # -------------------------------------------------------------------------
    def _match_partial(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        tokens = [token for token in normalized_query.split() if self._is_token_valid(token)]
        if not tokens:
            return None
        candidate_scores: dict[str, int] = {}
        record_lookup: dict[str, MonographRecord] = {}
        for token in tokens:
            for record in self.token_index.get(token, []):
                key = record.nbk_id or record.normalized_name or record.drug_name.lower()
                record_lookup[key] = record
                candidate_scores[key] = candidate_scores.get(key, 0) + 1
        if not candidate_scores:
            return None
        best_key = max(candidate_scores, key=candidate_scores.get)
        best_score = candidate_scores[best_key]
        tied = [key for key, score in candidate_scores.items() if score == best_score]
        if len(tied) != 1:
            return None
        best_record = record_lookup[best_key]
        matched_tokens: list[str] = []
        for token in tokens:
            for candidate in self.token_index.get(token, []):
                key = candidate.nbk_id or candidate.normalized_name or candidate.drug_name.lower()
                if key == best_key:
                    matched_tokens.append(token)
                    break
        notes = [f"token='{token}'" for token in sorted(set(matched_tokens))]
        return best_record, self.PARTIAL_CONFIDENCE, "partial_synonym", notes

    # -------------------------------------------------------------------------
    def _match_fuzzy(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        if len(normalized_query) < 4:
            return None
        variant = self._find_best_variant(normalized_query)
        if variant is None:
            return None
        record, original, is_primary, score = variant
        reason = "fuzzy_primary" if is_primary else "fuzzy_synonym"
        notes: list[str] = [f"score={score:.2f}"]
        if not is_primary:
            notes.insert(0, f"variant='{original}'")
        return record, max(self.FUZZY_CONFIDENCE, score), reason, notes

    # -------------------------------------------------------------------------
    def _match_chapter_title(
        self, chapter_title: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        normalized_chapter = self._normalize_name(chapter_title)
        if not normalized_chapter:
            return None
        direct = self._match_primary(normalized_chapter)
        if direct is not None:
            record, confidence, _, _ = direct
            return record, confidence, "chapter_title", []
        synonym = self._match_synonym(normalized_chapter)
        if synonym is not None:
            record, confidence, _, notes = synonym
            return record, confidence, "chapter_synonym", notes
        variant = self._find_best_variant(normalized_chapter)
        if variant is None:
            return None
        record, original, is_primary, score = variant
        reason = "chapter_fuzzy_primary" if is_primary else "chapter_fuzzy_synonym"
        notes: list[str] = [f"score={score:.2f}"]
        if not is_primary:
            notes.insert(0, f"variant='{original}'")
        return record, max(self.FUZZY_CONFIDENCE, score), reason, notes

    # -------------------------------------------------------------------------
    def _find_best_variant(
        self, normalized_query: str
    ) -> tuple[MonographRecord, str, bool, float] | None:
        best: tuple[MonographRecord, str, bool, float] | None = None
        best_ratio = 0.0
        for candidate, record, original, is_primary in self.variant_catalog:
            ratio = SequenceMatcher(None, normalized_query, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = (record, original, is_primary, ratio)
        if best is None or best_ratio < self.FUZZY_THRESHOLD:
            return None
        return best

    # -------------------------------------------------------------------------
    def _build_records(self) -> None:
        if self.livertox_df is None or self.livertox_df.empty:
            return
        token_occurrences: dict[str, list[MonographRecord]] = {}
        for row in self.livertox_df.itertuples(index=False):
            raw_name = self._coerce_text(getattr(row, "drug_name", None))
            if raw_name is None:
                continue
            normalized_name = self._normalize_name(raw_name)
            if not normalized_name:
                continue
            nbk_raw = getattr(row, "nbk_id", None)
            nbk_id = str(nbk_raw).strip() if nbk_raw not in (None, "") else ""
            if not nbk_id:
                continue
            excerpt = self._coerce_text(getattr(row, "excerpt", None))
            synonyms_value = getattr(row, "synonyms", None)
            synonyms = self._parse_synonyms(synonyms_value)
            tokens = self._collect_tokens(raw_name, list(synonyms.values()))
            record = MonographRecord(
                nbk_id=nbk_id,
                drug_name=raw_name,
                normalized_name=normalized_name,
                excerpt=excerpt,
                synonyms=synonyms,
                tokens=tokens,
            )
            self.records.append(record)
            if normalized_name not in self.primary_index:
                self.primary_index[normalized_name] = record
            self.variant_catalog.append((normalized_name, record, record.drug_name, True))
            for normalized_synonym, original in synonyms.items():
                if normalized_synonym not in self.synonym_index:
                    self.synonym_index[normalized_synonym] = (record, original)
                self.variant_catalog.append(
                    (normalized_synonym, record, original, False)
                )
            for token in tokens:
                bucket = token_occurrences.setdefault(token, [])
                if record not in bucket:
                    bucket.append(record)
        self.records.sort(key=lambda record: record.drug_name.lower())
        self.variant_catalog.sort(key=lambda item: item[0])
        self.token_occurrences = token_occurrences

    # -------------------------------------------------------------------------
    def _build_master_list_aliases(self) -> None:
        self.brand_index = {}
        self.ingredient_index = {}
        if self.master_list_df is None or self.master_list_df.empty:
            return
        for row in self.master_list_df.itertuples(index=False):
            chapter_title = self._coerce_text(getattr(row, "chapter_title", None))
            if chapter_title is None:
                continue
            brand = self._coerce_text(getattr(row, "brand_name", None))
            ingredient = self._coerce_text(getattr(row, "ingredient", None))
            for alias_type, value in ("brand", brand), ("ingredient", ingredient):
                if value is None:
                    continue
                for variant in self._iter_alias_variants(value):
                    normalized_variant = self._normalize_name(variant)
                    if not normalized_variant:
                        continue
                    index = (
                        self.brand_index if alias_type == "brand" else self.ingredient_index
                    )
                    bucket = index.setdefault(normalized_variant, [])
                    entry = (variant, chapter_title)
                    if entry not in bucket:
                        bucket.append(entry)

    # -------------------------------------------------------------------------
    def _finalize_token_index(self) -> None:
        if not self.token_occurrences:
            self.token_index = {}
            return
        filtered: dict[str, list[MonographRecord]] = {}
        for token, records in self.token_occurrences.items():
            if len(records) > self.TOKEN_MAX_FREQUENCY:
                continue
            filtered[token] = sorted(records, key=lambda record: record.drug_name.lower())
        self.token_index = filtered

    # -------------------------------------------------------------------------
    def _coerce_text(self, value: Any) -> str | None:
        if value in (None, ""):
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    # -------------------------------------------------------------------------
    def _iter_alias_variants(self, value: str) -> list[str]:
        normalized_value = self._normalize_whitespace(value)
        if not normalized_value:
            return []
        variants: set[str] = {normalized_value}
        for segment in re.split(r"[;,/\n]+", value):
            candidate = self._normalize_whitespace(segment)
            if candidate:
                variants.add(candidate)
        return list(variants)

    # -------------------------------------------------------------------------
    def _parse_synonyms(self, value: Any) -> dict[str, str]:
        text = self._coerce_text(value)
        if text is None:
            return {}
        synonyms: dict[str, str] = {}
        candidates = re.split(r"[;,/\n]+", text)
        for candidate in candidates:
            for variant in self._expand_variant(candidate):
                normalized = self._normalize_name(variant)
                if not normalized:
                    continue
                if normalized in MATCHING_STOPWORDS:
                    continue
                if len(normalized) < 4 and " " not in normalized:
                    continue
                if normalized not in synonyms:
                    synonyms[normalized] = variant
        return synonyms

    # -------------------------------------------------------------------------
    def _expand_variant(self, value: str) -> list[str]:
        normalized = self._normalize_whitespace(value)
        if not normalized:
            return []
        variants = {normalized}
        for segment in re.split(r"[()]", normalized):
            candidate = segment.strip(" -")
            if candidate:
                variants.add(candidate)
        return list(variants)

    # -------------------------------------------------------------------------
    def _collect_tokens(self, primary: str, synonyms: list[str]) -> set[str]:
        tokens: set[str] = set()
        for source in [primary, *synonyms]:
            tokens.update(self._tokenize(source))
        return tokens

    # -------------------------------------------------------------------------
    def _tokenize(self, value: str) -> set[str]:
        normalized = self._normalize_name(value)
        if not normalized:
            return set()
        return {
            token
            for token in normalized.split()
            if self._is_token_valid(token)
        }

    # -------------------------------------------------------------------------
    def _is_token_valid(self, token: str) -> bool:
        if len(token) < 4:
            return False
        if token in MATCHING_STOPWORDS:
            return False
        return not token.isdigit()

    # -------------------------------------------------------------------------
    def _normalize_whitespace(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    # -------------------------------------------------------------------------
    def _ensure_row_index(self) -> dict[str, dict[str, Any]]:
        if self.rows_by_nbk:
            return self.rows_by_nbk
        if self.livertox_df is None or self.livertox_df.empty:
            return {}
        index: dict[str, dict[str, Any]] = {}
        for row in self.livertox_df.to_dict(orient="records"):
            nbk_id = str(row.get("nbk_id") or "").strip()
            if not nbk_id:
                continue
            index[nbk_id] = row
        self.rows_by_nbk = index
        return self.rows_by_nbk

    # -------------------------------------------------------------------------
    def _create_match(
        self,
        record: MonographRecord,
        confidence: float,
        reason: str,
        notes: list[str] | None,
    ) -> LiverToxMatch:
        normalized_confidence = round(min(max(confidence, self.MIN_CONFIDENCE), 1.0), 2)
        cleaned_notes = list(dict.fromkeys(note for note in (notes or []) if note))
        return LiverToxMatch(
            nbk_id=record.nbk_id,
            matched_name=record.drug_name,
            confidence=normalized_confidence,
            reason=reason,
            notes=cleaned_notes,
            record=record,
        )

    # -------------------------------------------------------------------------
    def _normalize_name(self, name: str) -> str:
        normalized = (
            unicodedata.normalize("NFKD", name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

