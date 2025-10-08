from __future__ import annotations

import asyncio
import html
import io
import json
import os
import re
import tarfile
import unicodedata
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pypdf import PdfReader
from tqdm import tqdm

from DILIGENT.app.constants import (
    LIVERTOX_ARCHIVE,
    LIVERTOX_BASE_URL,
    SOURCES_PATH,
)
from DILIGENT.app.logger import logger
from DILIGENT.app.utils.database.sqlite import database
from DILIGENT.app.utils.serializer import DataSerializer
from DILIGENT.app.utils.updater.rxnav import RxNavClient

__all__ = ["LiverToxUpdater"]

SUPPORTED_MONOGRAPH_EXTENSIONS = (".html", ".htm", ".xhtml", ".xml", ".nxml", ".pdf")

DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "DILIGENTClinicalCopilot/1.0 (contact=clinical-copilot@pharmagent.local)"
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

        data = data.rename(columns=lambda s: re.sub(r"\s+", " ", s).strip())
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

        data = data.dropna(subset=["chapter_title"])

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
        if self.redownload:
            stored_metadata = None
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
    def refresh_master_list(self) -> tuple[dict[str, Any], pd.DataFrame]:
        logger.info("Refreshing LiverTox master list")
        metadata = asyncio.run(self._download_master_list())

        frame = pd.read_excel(
            metadata["file_path"],
            engine="openpyxl",
            header=self.header_row,
            skiprows=0,
        )
        sanitized = self.sanitize_livertox_master_list(frame)
        if sanitized is None:
            sanitized = pd.DataFrame()
        else:
            sanitized = sanitized.copy()
            sanitized["source_url"] = metadata.get("source_url")
            sanitized["source_last_modified"] = metadata.get("last_modified")
            if "last_update" in sanitized.columns and pd.api.types.is_datetime64_any_dtype(
                sanitized["last_update"]
            ):
                sanitized["last_update"] = sanitized["last_update"].dt.strftime("%Y-%m-%d")
        metadata["records"] = len(sanitized.index)
        return metadata, sanitized

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
            if self.redownload:
                stored_metadata = None
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
    def update_from_livertox(self) -> dict[str, Any]:
        logger.info("Starting LiverTox update")
        master_metadata, master_frame = self.refresh_master_list()

        logger.info("Checking LiverTox archive metadata")
        archive_metadata = asyncio.run(self.download_bulk_data(self.sources_path))
        archive_path = archive_metadata.get("file_path") or os.path.join(
            self.sources_path, LIVERTOX_ARCHIVE
        )

        local_info = self.collect_local_archive_info(archive_path)
        logger.info("Extracting LiverTox monographs from %s", archive_path)
        extracted = self.collect_monographs(archive_path)
        logger.info("Sanitizing %d extracted entries", len(extracted))
        monograph_df = self.sanitize_records(extracted)
        logger.info("Combining LiverTox datasets")
        unified = self._build_unified_dataset(
            monograph_df,
            master_frame,
            master_metadata,
        )
        logger.info("Enriching %d unified records with RxNav terms", len(unified.index))
        enriched = self.enrich_records(unified)
        logger.info("Finalizing sanitized dataset")
        final_dataset = self._finalize_dataset(enriched)
        logger.info("Persisting enriched records to database")
        self.serializer.save_livertox_records(final_dataset)

        payload = {**master_metadata, **archive_metadata, **local_info}
        payload["processed_entries"] = len(final_dataset.index)
        payload["records"] = len(final_dataset.index)
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
    def _build_unified_dataset(
        self,
        monographs: pd.DataFrame,
        master_frame: pd.DataFrame,
        master_metadata: dict[str, Any],
    ) -> pd.DataFrame:
        base_columns = [
            "drug_name",
            "ingredient",
            "brand_name",
            "likelihood_score",
            "last_update",
            "reference_count",
            "year_approved",
            "agent_classification",
            "primary_classification",
            "secondary_classification",
            "include_in_livertox",
            "source_url",
            "source_last_modified",
        ]
        monograph_columns = ["drug_name", "nbk_id", "excerpt", "synonyms"]
        final_columns = base_columns + ["nbk_id", "excerpt", "synonyms"]

        if master_frame is None or master_frame.empty:
            master = pd.DataFrame(columns=base_columns)
        else:
            master = master_frame.copy()
            if "chapter_title" in master.columns:
                master = master.rename(columns={"chapter_title": "drug_name"})
            if "drug_name" not in master.columns:
                master["drug_name"] = pd.NA
            master["drug_name"] = master["drug_name"].astype(str).str.strip()
            master = master[master["drug_name"] != ""]
            for column in base_columns:
                if column not in master.columns:
                    master[column] = pd.NA
            if master.empty and master_metadata.get("source_url"):
                master = pd.DataFrame(columns=base_columns)
            else:
                master["source_url"] = master["source_url"].fillna(
                    master_metadata.get("source_url")
                )
                master["source_last_modified"] = master["source_last_modified"].fillna(
                    master_metadata.get("last_modified")
                )
            master = master[base_columns]

        if monographs.empty:
            monograph_df = pd.DataFrame(columns=monograph_columns)
        else:
            monograph_df = monographs.copy()
        for column in monograph_columns:
            if column not in monograph_df.columns:
                monograph_df[column] = pd.NA
        monograph_df = monograph_df[monograph_columns]

        if master.empty:
            dataset = monograph_df.copy()
            for column in base_columns:
                if column not in dataset.columns:
                    dataset[column] = pd.NA
            dataset = dataset[final_columns]
            return self._sanitize_unified_dataset(dataset)

        dataset = master.merge(monograph_df, on="drug_name", how="left")
        if not monograph_df.empty:
            matched = dataset["drug_name"].unique().tolist()
            unmatched = monograph_df[~monograph_df["drug_name"].isin(matched)]
            if not unmatched.empty:
                filler = unmatched.copy()
                for column in base_columns:
                    if column not in filler.columns:
                        filler[column] = pd.NA
                filler = filler[dataset.columns]
                dataset = pd.concat([dataset, filler], ignore_index=True)
        dataset = dataset[final_columns]
        return self._sanitize_unified_dataset(dataset)

    # -------------------------------------------------------------------------
    def _contains_symbol(self, value: str) -> bool:
        if not isinstance(value, str):
            return False
        return bool(re.search(r"[^A-Za-z0-9\s\-/(),'.+]", value))

    # -------------------------------------------------------------------------
    def _sanitize_unified_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        sanitized = frame.copy()
        sanitized["drug_name"] = sanitized["drug_name"].astype(str).str.strip()
        sanitized = sanitized[sanitized["drug_name"] != ""]
        numeric_mask = sanitized["drug_name"].str.fullmatch(r"\d+")
        sanitized = sanitized[~numeric_mask]
        symbol_mask = sanitized["drug_name"].apply(self._contains_symbol)
        sanitized = sanitized[~symbol_mask]

        for column in ("ingredient", "brand_name"):
            if column not in sanitized.columns:
                sanitized[column] = pd.NA
            sanitized[column] = sanitized[column].where(
                pd.notnull(sanitized[column]), pd.NA
            )
            sanitized[column] = sanitized[column].astype(str).str.strip()
            sanitized.loc[
                sanitized[column].isin(["", "nan", "None", "<NA>"]), column
            ] = pd.NA
            invalid_mask = sanitized[column].notna() & sanitized[column].apply(
                self._contains_symbol
            )
            invalid_mask = invalid_mask.fillna(False)
            sanitized = sanitized[~invalid_mask]

        sanitized["excerpt"] = sanitized["excerpt"].astype(str).str.strip()
        sanitized.loc[
            sanitized["excerpt"].isin(["", "nan", "None", "NaT"]), "excerpt"
        ] = "Not available"
        sanitized.loc[sanitized["excerpt"].isna(), "excerpt"] = "Not available"

        sanitized = sanitized.drop_duplicates(
            subset=["drug_name", "ingredient", "brand_name"], keep="first"
        )
        return sanitized.reset_index(drop=True)

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
    def sanitize_records(self, entries: list[dict[str, Any]]) -> pd.DataFrame:
        sanitized = self.serializer.sanitize_livertox_records(entries)
        if sanitized.empty:
            sanitized = pd.DataFrame(columns=["nbk_id", "drug_name", "excerpt", "synonyms"])
        sanitized = sanitized.copy()
        sanitized["drug_name"] = sanitized["drug_name"].astype(str).str.strip()
        sanitized = sanitized[sanitized["drug_name"] != ""]
        numeric_mask = sanitized["drug_name"].str.fullmatch(r"\d+")
        sanitized = sanitized[~numeric_mask]
        sanitized["excerpt"] = sanitized["excerpt"].astype(str).str.strip()
        sanitized.loc[sanitized["excerpt"] == "", "excerpt"] = pd.NA
        if "synonyms" not in sanitized.columns:
            sanitized["synonyms"] = pd.NA
        sanitized["synonyms"] = sanitized["synonyms"].where(pd.notnull(sanitized["synonyms"]), pd.NA)
        return sanitized.reset_index(drop=True)

    # -------------------------------------------------------------------------
    def enrich_records(self, records: pd.DataFrame) -> pd.DataFrame:
        if records.empty:
            return records.copy()
        enriched = records.copy()
        enriched["synonyms"] = pd.NA
        unit_stopwords = getattr(self.rx_client, "UNIT_STOPWORDS", set())
        synonyms_values: list[str | pd.NA] = []
        for row in enriched.itertuples(index=False):
            aliases = set()
            for attr in ("drug_name", "ingredient", "brand_name"):
                value = getattr(row, attr, None)
                if not isinstance(value, str):
                    continue
                normalized = value.strip()
                if not normalized or normalized.lower() == "not available":
                    continue
                aliases.add(normalized)
            collected: set[str] = set()
            for alias in aliases:
                try:
                    synonyms = self.rx_client.fetch_drug_terms(alias)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to enrich '%s': %s", alias, exc)
                    continue
                collected.update(synonyms)
            sanitized = self._sanitize_synonym_list(collected, unit_stopwords)
            if sanitized:
                synonyms_values.append(
                    ", ".join(sorted(sanitized, key=str.casefold))
                )
            else:
                synonyms_values.append(pd.NA)
        enriched["synonyms"] = synonyms_values
        return enriched

    # -------------------------------------------------------------------------
    def _sanitize_synonym_list(
        self, candidates: set[str], unit_stopwords: set[str]
    ) -> list[str]:
        sanitized: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            normalized = self._normalize_whitespace(candidate)
            if not normalized:
                continue
            if len(normalized) < 4:
                continue
            if normalized.isnumeric():
                continue
            if self._contains_symbol(normalized):
                continue
            tokens = normalized.split()
            filtered: list[str] = []
            for token in tokens:
                cleaned = re.sub(r"[^A-Za-z0-9'-]", "", token)
                if not cleaned:
                    continue
                if cleaned.lower() in unit_stopwords:
                    continue
                if cleaned.isnumeric():
                    continue
                if len(cleaned) < 2:
                    continue
                filtered.append(cleaned)
            refined = " ".join(filtered)
            refined = refined.strip()
            if len(refined) < 4:
                continue
            if self._contains_symbol(refined):
                continue
            key = refined.casefold()
            if key in seen:
                continue
            seen.add(key)
            sanitized.append(refined)
        return sanitized

    # -------------------------------------------------------------------------
    def _finalize_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        finalized = self._sanitize_unified_dataset(frame)
        fill_value = "Not available"
        for column in finalized.columns:
            if column == "drug_name":
                continue
            finalized[column] = finalized[column].where(
                pd.notnull(finalized[column]), fill_value
            )
            finalized[column] = finalized[column].astype(str).str.strip()
            finalized.loc[
                finalized[column].isin(["", "nan", "NaT", "None", "<NA>"]), column
            ] = fill_value
        finalized["synonyms"] = finalized["synonyms"].apply(
            lambda value: (
                value.strip()
                if isinstance(value, str)
                and value.strip()
                and value.strip().lower() not in {"<na>", "nan", "nat", "none"}
                else fill_value
            )
        )
        finalized["excerpt"] = finalized["excerpt"].astype(str).str.strip()
        finalized.loc[
            finalized["excerpt"].isin(["", "nan", "NaT", "None", "<NA>"]), "excerpt"
        ] = fill_value
        return finalized.reset_index(drop=True)


  

###############################################################################
