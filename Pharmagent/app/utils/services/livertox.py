from __future__ import annotations

import asyncio
import difflib
import html
import io
import json
import os
import re
import tarfile
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
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
    "vial",
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
    matching_pool: set[str]


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
@dataclass(slots=True)
class AliasEntry:
    record: MonographRecord
    alias_type: str
    display_name: str


###############################################################################
class LiverToxMatcher:
    DIRECT_CONFIDENCE = 1.0
    ALIAS_CONFIDENCE = 0.95
    MASTER_ALIAS_CONFIDENCE = 0.9
    FUZZY_MONOGRAPH_CONFIDENCE = 0.88
    FUZZY_ALIAS_CONFIDENCE = 0.85
    MIN_CONFIDENCE = 0.40
    FUZZY_CUTOFF = 0.88
    ALIAS_FUZZY_CUTOFF = 0.86

    # -------------------------------------------------------------------------
    def __init__(
        self,
        livertox_df: pd.DataFrame,
        *,
        master_list_df: pd.DataFrame | None = None,
    ) -> None:
        self.livertox_df = livertox_df
        self.master_list_df = master_list_df
        self.match_cache: dict[str, LiverToxMatch | None] = {}
        self.records: list[MonographRecord] = []
        self.records_by_normalized: dict[str, MonographRecord] = {}
        self.matching_pool_index: dict[str, list[MonographRecord]] = {}
        self.rows_by_nbk: dict[str, dict[str, Any]] = {}
        self.alias_index: dict[str, AliasEntry] = {}
        self.alias_keys: list[str] = []
        self._build_records()
        self._build_master_list_aliases()

    # -------------------------------------------------------------------------
    async def match_drug_names(
        self, patient_drugs: list[str]
    ) -> list[LiverToxMatch | None]:
        if not patient_drugs:
            return []
        if not self.records:
            return [None] * len(patient_drugs)
        results: list[LiverToxMatch | None] = []
        for name in patient_drugs:
            normalized = self._normalize_name(name) if name else ""
            if not normalized:
                results.append(None)
                continue
            if normalized in self.match_cache:
                results.append(self.match_cache[normalized])
                continue
            match = self._deterministic_lookup(normalized)
            self.match_cache[normalized] = match
            results.append(match)
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
    def _build_records(self) -> None:
        if self.livertox_df is None or self.livertox_df.empty:
            return
        processed: list[MonographRecord] = []
        normalized_map: dict[str, MonographRecord] = {}
        pool_index: dict[str, list[MonographRecord]] = {}
        for row in self.livertox_df.itertuples(index=False):
            raw_name = str(getattr(row, "drug_name", "") or "").strip()
            if not raw_name:
                continue
            normalized_name = self._normalize_name(raw_name)
            if not normalized_name:
                continue
            primary_variant = self._normalize_name(raw_name.split("(")[0])
            nbk_raw = getattr(row, "nbk_id", None)
            nbk_id = str(nbk_raw).strip() if nbk_raw not in (None, "") else ""
            excerpt_value = self._coerce_text(getattr(row, "excerpt", None))
            matching_pool = self._extract_matching_pool(
                getattr(row, "synonyms", None),
            )
            matching_pool.update(self._extract_parenthetical_tokens(raw_name))
            record = MonographRecord(
                nbk_id=nbk_id,
                drug_name=raw_name,
                normalized_name=normalized_name,
                excerpt=excerpt_value,
                matching_pool=matching_pool,
            )
            processed.append(record)
            normalized_map.setdefault(normalized_name, record)
            if primary_variant and primary_variant != normalized_name:
                normalized_map.setdefault(primary_variant, record)
            for token in matching_pool:
                bucket = pool_index.setdefault(token, [])
                if record not in bucket:
                    bucket.append(record)
        if not processed:
            return
        processed.sort(key=lambda item: item.drug_name.lower())
        self.records = processed
        self.records_by_normalized = {
            key: value for key, value in normalized_map.items() if value is not None
        }
        self.matching_pool_index = pool_index

    # -------------------------------------------------------------------------
    def _build_master_list_aliases(self) -> None:
        self.alias_index = {}
        self.alias_keys = []
        if self.master_list_df is None or self.master_list_df.empty:
            return
        for row in self.master_list_df.itertuples(index=False):
            chapter = self._coerce_text(getattr(row, "chapter_title", None))
            record = self._resolve_chapter_record(chapter)
            if record is None:
                continue
            if chapter:
                self._register_alias(chapter, "chapter", record)
            brand = self._coerce_text(getattr(row, "brand_name", None))
            if brand:
                self._register_alias(brand, "brand", record)
            ingredient = self._coerce_text(getattr(row, "ingredient", None))
            if ingredient:
                self._register_alias(ingredient, "ingredient", record)

    # -------------------------------------------------------------------------
    def _register_alias(
        self, value: str, alias_type: str, record: MonographRecord
    ) -> None:
        normalized = self._normalize_name(value)
        if not normalized:
            return
        if normalized in self.alias_index:
            return
        entry = AliasEntry(record=record, alias_type=alias_type, display_name=value)
        self.alias_index[normalized] = entry
        self.alias_keys.append(normalized)

    # -------------------------------------------------------------------------
    def _resolve_chapter_record(
        self, chapter_title: str | None
    ) -> MonographRecord | None:
        if not chapter_title:
            return None
        normalized = self._normalize_name(chapter_title)
        if not normalized:
            return None
        direct = self.records_by_normalized.get(normalized)
        if direct is not None:
            return direct
        fuzzy = self._find_fuzzy_monogram(normalized, cutoff=self.FUZZY_CUTOFF)
        if fuzzy is None:
            return None
        record, _ = fuzzy
        return record

    # -------------------------------------------------------------------------
    def _coerce_text(self, value: Any) -> str | None:
        if value in (None, ""):
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    # -------------------------------------------------------------------------
    def _extract_matching_pool(self, *values: Any) -> set[str]:
        tokens: set[str] = set()
        for value in values:
            text = self._coerce_text(value)
            if text is None:
                continue
            bracket_segments = re.findall(r"\[([^\]]+)\]", text)
            for segment in bracket_segments:
                tokens.update(self._tokenize_text(segment))
            tokens.update(self._tokenize_text(text))
        return tokens

    # -------------------------------------------------------------------------
    def _extract_parenthetical_tokens(self, text: str) -> set[str]:
        segments = re.findall(r"\(([^)]+)\)", text)
        tokens: set[str] = set()
        for segment in segments:
            tokens.update(self._tokenize_text(segment))
        return tokens

    # -------------------------------------------------------------------------
    def _tokenize_text(self, text: str) -> set[str]:
        ascii_text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        raw_tokens = re.findall(r"[A-Za-z]+", ascii_text)
        tokens: set[str] = set()
        for raw in raw_tokens:
            normalized = raw.lower()
            normalized = re.sub(r"[^a-z]", "", normalized)
            if len(normalized) < 3:
                continue
            if normalized in MATCHING_STOPWORDS:
                continue
            tokens.add(normalized)
        return tokens

    # -------------------------------------------------------------------------
    def _match_from_pool(
        self, normalized_value: str
    ) -> tuple[MonographRecord, str] | None:
        for token in self._tokenize_text(normalized_value):
            candidates = self.matching_pool_index.get(token)
            if not candidates:
                continue
            return candidates[0], token
        return None

    # -------------------------------------------------------------------------
    def _find_fuzzy_monogram(
        self, normalized_query: str, *, cutoff: float | None = None
    ) -> tuple[MonographRecord, str] | None:
        if not normalized_query or not self.records_by_normalized:
            return None
        keys = list(self.records_by_normalized.keys())
        threshold = cutoff if cutoff is not None else self.FUZZY_CUTOFF
        matches = difflib.get_close_matches(
            normalized_query, keys, n=1, cutoff=threshold
        )
        if not matches:
            return None
        key = matches[0]
        record = self.records_by_normalized.get(key)
        if record is None:
            return None
        return record, key

    # -------------------------------------------------------------------------
    def _find_fuzzy_alias(self, normalized_query: str) -> str | None:
        if not normalized_query or not self.alias_keys:
            return None
        matches = difflib.get_close_matches(
            normalized_query, self.alias_keys, n=1, cutoff=self.ALIAS_FUZZY_CUTOFF
        )
        if matches:
            return matches[0]
        return None

    # -------------------------------------------------------------------------
    def _match_master_list_alias(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        if not normalized_query or not self.alias_index:
            return None
        entry = self.alias_index.get(normalized_query)
        if entry is not None:
            reason = f"master_{entry.alias_type}_alias"
            note = f"{entry.alias_type}='{entry.display_name}'"
            return entry.record, self.MASTER_ALIAS_CONFIDENCE, reason, [note]
        fuzzy_key = self._find_fuzzy_alias(normalized_query)
        if fuzzy_key is None:
            return None
        entry = self.alias_index[fuzzy_key]
        reason = f"master_{entry.alias_type}_fuzzy"
        note = f"{entry.alias_type}='{entry.display_name}'"
        return entry.record, self.FUZZY_ALIAS_CONFIDENCE, reason, [note]

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
    def _deterministic_lookup(self, normalized_query: str) -> LiverToxMatch | None:
        if not normalized_query:
            return None
        direct = self.records_by_normalized.get(normalized_query)
        if direct is not None:
            return self._create_match(direct, self.DIRECT_CONFIDENCE, "direct_match", [])
        pool_match = self._match_from_pool(normalized_query)
        if pool_match is not None:
            record, token = pool_match
            note = f"token='{token}'"
            return self._create_match(record, self.ALIAS_CONFIDENCE, "alias_match", [note])
        fuzzy = self._find_fuzzy_monogram(normalized_query)
        if fuzzy is not None:
            record, _ = fuzzy
            note = f"matched='{record.drug_name}'"
            return self._create_match(
                record,
                self.FUZZY_MONOGRAPH_CONFIDENCE,
                "fuzzy_match",
                [note],
            )
        master_alias = self._match_master_list_alias(normalized_query)
        if master_alias is not None:
            record, confidence, reason, notes = master_alias
            return self._create_match(record, confidence, reason, notes)
        return None

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
