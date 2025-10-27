from __future__ import annotations

import asyncio
import math
import json
import os
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urljoin

import httpx
import pandas as pd

from DILIGENT.app.constants import (
    CHEMBL_API_BASE_URL,
    DAILYMED_SPL_BULK_BASE_URL,
    FDA_UNII_DOWNLOAD_URL,
    OPENFDA_NDC_ENDPOINT,
    PUBCHEM_PUG_REST_BASE_URL,
    RXNORM_EXPECTED_FILES,
    RXNORM_FULL_DOWNLOAD_URL,
    RXNORM_WEEKLY_DOWNLOAD_URL,
)
from DILIGENT.app.logger import logger

RXNCONSO_COLUMNS = [
    "RXCUI",
    "LAT",
    "TS",
    "LUI",
    "STT",
    "SUI",
    "ISPREF",
    "RXAUI",
    "SAUI",
    "SCUI",
    "SDUI",
    "SAB",
    "TTY",
    "CODE",
    "STR",
    "SRL",
    "SUPPRESS",
    "CVF",
]

RXNREL_COLUMNS = [
    "RXCUI1",
    "RXAUI1",
    "STYPE1",
    "REL",
    "RXCUI2",
    "RXAUI2",
    "STYPE2",
    "RELA",
    "RUI",
    "SRUI",
    "SAB",
    "SL",
    "DIR",
    "RG",
    "SUPPRESS",
    "CVF",
]

RXNSAT_COLUMNS = [
    "RXCUI",
    "LUI",
    "SAB",
    "CODE",
    "ATUI",
    "SATUI",
    "ATN",
    "ATV",
    "SUPPRESS",
    "CVF",
]

RXNCUICHANGES_COLUMNS = [
    "RXCUI",
    "RXAUI",
    "CODE",
    "SAB",
    "TTY",
    "CURSTATUS",
    "SUPPRESS",
    "CVF",
]

DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
PUBCHEM_CONCURRENCY = 8
OPENFDA_CONCURRENCY = 4
DAILYMED_CONCURRENCY = 4
CHEMBL_CONCURRENCY = 4


###############################################################################
class AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    ###########################################################################
    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.links.append(value)


###############################################################################
class RxNormReleaseManager:
    def __init__(
        self,
        sources_path: str,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.sources_path = os.path.abspath(sources_path)
        self.releases_path = os.path.join(self.sources_path, "rxnorm")
        os.makedirs(self.releases_path, exist_ok=True)
        default_headers = {"User-Agent": "DILIGENT-Clinical-Copilot/1.0"}
        if http_client is None:
            self.http_client = httpx.Client(
                timeout=DEFAULT_TIMEOUT,
                follow_redirects=True,
                headers=default_headers,
            )
        else:
            self.http_client = http_client
        self.owns_client = http_client is None

    # -------------------------------------------------------------------------
    def close(self) -> None:
        if self.owns_client:
            self.http_client.close()

    # -------------------------------------------------------------------------
    def ensure_release(
        self,
        *,
        redownload: bool,
        archive_path: str | None = None,
        monthly: bool = True,
    ) -> str:
        if archive_path and os.path.isfile(archive_path):
            return self.extract_archive(archive_path)
        download_url = RXNORM_FULL_DOWNLOAD_URL if monthly else RXNORM_WEEKLY_DOWNLOAD_URL
        archive_name = os.path.basename(download_url)
        destination = os.path.join(self.releases_path, archive_name)
        if redownload or not os.path.isfile(destination):
            logger.info("Downloading RxNorm release from %s", download_url)
            try:
                content = self.fetch_release_content(download_url)
            except httpx.HTTPStatusError:
                if not self.is_prescribe_url(download_url):
                    raise
                fallback_url = self.find_latest_prescribe_url(monthly)
                if fallback_url == download_url:
                    raise
                logger.info("Retrying RxNorm release download from %s", fallback_url)
                content = self.fetch_release_content(fallback_url)
            with open(destination, "wb") as handle:
                handle.write(content)
        return self.extract_archive(destination)

    # -------------------------------------------------------------------------
    def extract_archive(self, archive_path: str) -> str:
        extract_dir = os.path.join(
            self.releases_path, os.path.splitext(os.path.basename(archive_path))[0]
        )
        marker = os.path.join(extract_dir, ".complete")
        if os.path.isdir(extract_dir) and os.path.isfile(marker):
            return extract_dir
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(path=extract_dir)
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write(datetime.now(UTC).isoformat())
        return extract_dir

    # -------------------------------------------------------------------------
    def fetch_release_content(self, url: str) -> bytes:
        response = self.http_client.get(url)
        if response.status_code == 302 and "location" in response.headers:
            raise httpx.HTTPStatusError(
                "Unexpected redirect when downloading RxNorm release",
                request=response.request,
                response=response,
            )
        response.raise_for_status()
        if not self.is_zip_response(response):
            message = "Received unexpected content while downloading RxNorm release"
            if self.contains_uts_login(response):
                message = (
                    "RxNorm prescribe release requires UTS authentication; "
                    "use a dated link instead of the 'current' pointer"
                )
            raise httpx.HTTPStatusError(
                message,
                request=response.request,
                response=response,
            )
        return response.content

    # -------------------------------------------------------------------------
    def is_zip_response(self, response: httpx.Response) -> bool:
        content_type = response.headers.get("content-type", "").lower()
        if content_type:
            if "zip" in content_type or "octet-stream" in content_type:
                return True
            if "html" in content_type:
                return False
        disposition = response.headers.get("content-disposition", "").lower()
        if disposition and "zip" in disposition:
            return True
        return False

    # -------------------------------------------------------------------------
    def contains_uts_login(self, response: httpx.Response) -> bool:
        if self.is_zip_response(response):
            return False
        try:
            body = response.text
        except UnicodeDecodeError:
            return False
        login_markers = [
            "uts.nlm.nih.gov",
            "UTS Login",
            "Sign in",
        ]
        return any(marker.lower() in body.lower() for marker in login_markers)

    # -------------------------------------------------------------------------
    def is_prescribe_url(self, url: str) -> bool:
        return "prescribe" in url.lower()

    # -------------------------------------------------------------------------
    def find_latest_prescribe_url(self, monthly: bool) -> str:
        page_url = "https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html"
        logger.info("Discovering latest RxNorm prescribe release from %s", page_url)
        response = self.http_client.get(page_url)
        response.raise_for_status()
        release_type = "full" if monthly else "weekly"
        parser = AnchorCollector()
        parser.feed(response.text)
        pattern = re.compile(
            rf"RxNorm_{release_type}_prescribe_(\d{{8}})\.zip", re.IGNORECASE
        )
        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()
        for href in parser.links:
            absolute = urljoin(page_url, href)
            match = pattern.search(absolute)
            if not match:
                continue
            if absolute in seen:
                continue
            seen.add(absolute)
            candidates.append((match.group(1), absolute))
        if not candidates:
            raise httpx.HTTPStatusError(
                "Could not determine latest RxNorm prescribe release",
                request=response.request,
                response=response,
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]


###############################################################################
class RxNormCatalogBuilder:
    def __init__(self, release_dir: str) -> None:
        self.release_dir = release_dir

    # -------------------------------------------------------------------------
    def load_rrf(self, file_name: str, columns: Sequence[str]) -> pd.DataFrame:
        path = os.path.join(self.release_dir, file_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        frame = pd.read_csv(
            path,
            sep="|",
            names=list(columns),
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            encoding="utf-8",
            engine="python",
        )
        if frame.shape[1] > len(columns):
            frame = frame.iloc[:, : len(columns)]
        return frame

    # -------------------------------------------------------------------------
    def load_conso(self) -> pd.DataFrame:
        frame = self.load_rrf(RXNORM_EXPECTED_FILES["conso"], RXNCONSO_COLUMNS)
        frame = frame[frame["LAT"] == "ENG"]
        frame = frame[frame["SUPPRESS"] != "Y"]
        return frame

    # -------------------------------------------------------------------------
    def load_rel(self) -> pd.DataFrame:
        frame = self.load_rrf(RXNORM_EXPECTED_FILES["rel"], RXNREL_COLUMNS)
        frame = frame[frame["SUPPRESS"] != "Y"]
        return frame

    # -------------------------------------------------------------------------
    def load_sat(self) -> pd.DataFrame:
        frame = self.load_rrf(RXNORM_EXPECTED_FILES["sat"], RXNSAT_COLUMNS)
        frame = frame[frame["SUPPRESS"] != "Y"]
        return frame

    # -------------------------------------------------------------------------
    def load_changes(self) -> pd.DataFrame:
        return self.load_rrf(RXNORM_EXPECTED_FILES["changes"], RXNCUICHANGES_COLUMNS)

    # -------------------------------------------------------------------------
    def build_base_catalog(self) -> pd.DataFrame:
        conso = self.load_conso()
        rel = self.load_rel()
        sat = self.load_sat()
        changes = self.load_changes()
        status_map = self.extract_status_map(changes)
        preferred_names = self.get_preferred_names(conso)
        synonyms = self.collect_synonyms(conso)
        brand_links = self.collect_brand_links(rel)
        parent_links = self.collect_parent_links(rel)
        sat_map = self.collect_attributes(sat)
        records: list[dict[str, Any]] = []
        concept_types = {"IN", "PIN", "BN", "SCD", "SBD"}
        filtered = conso[conso["TTY"].isin(concept_types)]
        for rxcui, group in filtered.groupby("RXCUI"):
            concept_type = group["TTY"].iloc[0]
            preferred_name = preferred_names.get(rxcui) or group["STR"].iloc[0]
            syns = synonyms.get(rxcui, set()).copy()
            if preferred_name in syns:
                syns.discard(preferred_name)
            sat_entry = sat_map.get(rxcui, {})
            unii = self.first_value(sat_entry.get("UNII"))
            cas = self.first_value(sat_entry.get("CAS"))
            status = status_map.get(rxcui, "active")
            record = {
                "rxcui": rxcui,
                "preferred_name": preferred_name,
                "concept_type": concept_type,
                "synonyms": sorted(syns),
                "brands": self.resolve_brand_names(brand_links.get(rxcui, set()), preferred_names),
                "rxcui_parents": sorted(parent_links.get(rxcui, set())),
                "pubchem_cid": "",
                "inchikey": "",
                "unii": unii or "",
                "cas": cas or "",
                "xrefs": sat_entry,
                "status": status,
            }
            records.append(record)
        return pd.DataFrame.from_records(records)

    # -------------------------------------------------------------------------
    def extract_status_map(self, changes: pd.DataFrame) -> dict[str, str]:
        if changes.empty:
            return {}
        mapping: dict[str, str] = {}
        grouped = changes.groupby("RXCUI")
        for rxcui, group in grouped:
            latest = group.iloc[-1]
            status = latest.get("CURSTATUS") or "Active"
            mapping[str(rxcui)] = status.lower()
        return mapping

    # -------------------------------------------------------------------------
    def get_preferred_names(self, conso: pd.DataFrame) -> dict[str, str]:
        mapping: dict[str, str] = {}
        sorted_frame = conso.sort_values(
            by=["ISPREF", "TS"], ascending=[False, True], kind="mergesort"
        )
        for rxcui, group in sorted_frame.groupby("RXCUI"):
            entry = group.iloc[0]
            mapping[str(rxcui)] = entry.get("STR") or ""
        return mapping

    # -------------------------------------------------------------------------
    def collect_synonyms(self, conso: pd.DataFrame) -> dict[str, set[str]]:
        synonyms: dict[str, set[str]] = defaultdict(set)
        for row in conso.itertuples(index=False):
            name = getattr(row, "STR", "") or ""
            if not name:
                continue
            synonyms[str(getattr(row, "RXCUI"))].add(name)
        return synonyms

    # -------------------------------------------------------------------------
    def collect_brand_links(self, rel: pd.DataFrame) -> dict[str, set[str]]:
        brand_map: dict[str, set[str]] = defaultdict(set)
        tradename_mask = rel["RELA"].isin({"has_tradename", "tradename_of"})
        tradename_rel = rel[tradename_mask]
        for row in tradename_rel.itertuples(index=False):
            if row.RELA == "has_tradename":
                brand_map[str(row.RXCUI1)].add(str(row.RXCUI2))
            elif row.RELA == "tradename_of":
                brand_map[str(row.RXCUI2)].add(str(row.RXCUI1))
        return brand_map

    # -------------------------------------------------------------------------
    def collect_parent_links(self, rel: pd.DataFrame) -> dict[str, set[str]]:
        parent_map: dict[str, set[str]] = defaultdict(set)
        parent_rel = rel[rel["REL"].isin({"PAR", "RB"})]
        for row in parent_rel.itertuples(index=False):
            parent_map[str(row.RXCUI1)].add(str(row.RXCUI2))
        return parent_map

    # -------------------------------------------------------------------------
    def collect_attributes(self, sat: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
        sat_map: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for row in sat.itertuples(index=False):
            attribute = getattr(row, "ATN", "") or ""
            value = getattr(row, "ATV", "") or ""
            if not attribute or not value:
                continue
            rxcui = str(getattr(row, "RXCUI"))
            sat_map[rxcui][attribute].append(value)
        return {key: dict(value) for key, value in sat_map.items()}

    # -------------------------------------------------------------------------
    def resolve_brand_names(
        self,
        brand_rxcuis: set[str],
        preferred_names: dict[str, str],
    ) -> list[str]:
        names = [preferred_names.get(rxcui, "") for rxcui in brand_rxcuis]
        names = [name for name in names if name]
        return sorted(set(names))

    # -------------------------------------------------------------------------
    def first_value(self, values: dict[str, list[str]] | list[str] | None) -> str | None:
        if values is None:
            return None
        if isinstance(values, list):
            return values[0] if values else None
        if isinstance(values, dict):
            for key in values:
                selected = self.first_value(values[key])
                if selected:
                    return selected
        return None


###############################################################################
class PubChemClient:
    def __init__(self, *, concurrency: int = PUBCHEM_CONCURRENCY) -> None:
        self.semaphore = asyncio.Semaphore(max(1, concurrency))
        self.client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        await self.client.aclose()

    # -------------------------------------------------------------------------
    async def resolve_cid(self, candidate: str, identifier_type: str) -> str | None:
        if not candidate:
            return None
        if identifier_type == "name":
            path = f"/compound/name/{quote(candidate)}/cids/JSON"
        elif identifier_type == "unii":
            path = f"/compound/xref/UNII/{quote(candidate)}/cids/JSON"
        elif identifier_type == "cas":
            path = f"/compound/xref/CAS/{quote(candidate)}/cids/JSON"
        elif identifier_type == "inchikey":
            path = f"/compound/inchikey/{quote(candidate)}/cids/JSON"
        else:
            return None
        url = f"{PUBCHEM_PUG_REST_BASE_URL}{path}"
        async with self.semaphore:
            response = await self.client.get(url)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        information = payload.get("IdentifierList") or payload.get("InformationList")
        if not information:
            return None
        cids = information.get("CID") or information.get("CIDList")
        if isinstance(cids, list) and cids:
            return str(cids[0])
        return None

    # -------------------------------------------------------------------------
    async def fetch_properties(self, cid: str) -> dict[str, Any]:
        path = f"/compound/cid/{cid}/property/InChIKey,Title/JSON"
        url = f"{PUBCHEM_PUG_REST_BASE_URL}{path}"
        async with self.semaphore:
            response = await self.client.get(url)
        response.raise_for_status()
        payload = response.json()
        props = payload.get("PropertyTable", {}).get("Properties", [])
        if props:
            return props[0]
        return {}

    # -------------------------------------------------------------------------
    async def fetch_synonyms(self, cid: str) -> list[str]:
        path = f"/compound/cid/{cid}/synonyms/JSON"
        url = f"{PUBCHEM_PUG_REST_BASE_URL}{path}"
        async with self.semaphore:
            response = await self.client.get(url)
        if response.status_code == 404:
            return []
        response.raise_for_status()
        payload = response.json()
        synonyms = payload.get("InformationList", {}).get("Information", [])
        if not synonyms:
            return []
        values: list[str] = []
        for entry in synonyms:
            values.extend(entry.get("Synonym", []))
        return list(dict.fromkeys(values))

    # -------------------------------------------------------------------------
    async def fetch_xrefs(self, cid: str) -> dict[str, list[str]]:
        path = f"/compound/cid/{cid}/xrefs/JSON"
        url = f"{PUBCHEM_PUG_REST_BASE_URL}{path}"
        async with self.semaphore:
            response = await self.client.get(url)
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        payload = response.json()
        information = payload.get("InformationList", {}).get("Information", [])
        xrefs: dict[str, list[str]] = defaultdict(list)
        for entry in information:
            for key, value in entry.items():
                if key == "CID":
                    continue
                if isinstance(value, list):
                    xrefs[key].extend([str(item) for item in value])
                else:
                    xrefs[key].append(str(value))
        return {key: list(dict.fromkeys(values)) for key, values in xrefs.items()}

    # -------------------------------------------------------------------------
    async def enrich_record(self, record: dict[str, Any]) -> None:
        if record.get("pubchem_cid") and record.get("inchikey"):
            return
        candidates: list[tuple[str, str]] = []
        unii = record.get("unii")
        cas = record.get("cas")
        inchikey = record.get("inchikey")
        preferred_name = record.get("preferred_name")
        synonyms = record.get("synonyms", [])
        if unii:
            candidates.append((unii, "unii"))
        if cas:
            candidates.append((cas, "cas"))
        if inchikey:
            candidates.append((inchikey, "inchikey"))
        for name in [preferred_name, *synonyms]:
            if name:
                candidates.append((name, "name"))
        cid = record.get("pubchem_cid")
        for candidate, identifier_type in candidates:
            cid = await self.resolve_cid(candidate, identifier_type)
            if cid:
                break
        if not cid:
            return
        record["pubchem_cid"] = cid
        properties = await self.fetch_properties(cid)
        record["inchikey"] = properties.get("InChIKey", record.get("inchikey", ""))
        pubchem_title = properties.get("Title")
        if pubchem_title:
            synonyms = list(synonyms)
            synonyms.append(pubchem_title)
            record["synonyms"] = list(dict.fromkeys(synonyms))
        synonyms_payload = await self.fetch_synonyms(cid)
        if synonyms_payload:
            merged = list(dict.fromkeys([*record.get("synonyms", []), *synonyms_payload]))
            record["synonyms"] = merged
        xrefs = await self.fetch_xrefs(cid)
        if xrefs:
            existing = record.get("xrefs") or {}
            for key, values in xrefs.items():
                existing_values = existing.get(key, []) or []
                merged = list(dict.fromkeys([*existing_values, *values]))
                existing[key] = merged
            record["xrefs"] = existing
        if not record.get("unii"):
            unii_values = record.get("xrefs", {}).get("UNII", [])
            if unii_values:
                record["unii"] = unii_values[0]
        if not record.get("cas"):
            cas_values = record.get("xrefs", {}).get("CAS", [])
            if cas_values:
                record["cas"] = cas_values[0]

    # -------------------------------------------------------------------------
    def enrich(self, records: list[dict[str, Any]]) -> None:
        async def runner() -> None:
            tasks = [self.enrich_record(record) for record in records]
            if not tasks:
                return
            chunks: list[list[asyncio.Task[None]]] = []
            batch: list[asyncio.Task[None]] = []
            for task in tasks:
                batch.append(asyncio.create_task(task))
                if len(batch) >= PUBCHEM_CONCURRENCY * 2:
                    chunks.append(batch.copy())
                    batch.clear()
            if batch:
                chunks.append(batch)
            for group in chunks:
                await asyncio.gather(*group, return_exceptions=True)

        asyncio.run(runner())


###############################################################################
class FdaUniiClient:
    def __init__(
        self,
        sources_path: str,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.sources_path = os.path.abspath(sources_path)
        self.dataset_path = os.path.join(self.sources_path, "fda_unii.csv")
        self.http_client = http_client or httpx.Client(timeout=DEFAULT_TIMEOUT)
        self.owns_client = http_client is None

    # -------------------------------------------------------------------------
    def close(self) -> None:
        if self.owns_client:
            self.http_client.close()

    # -------------------------------------------------------------------------
    def ensure_dataset(self, *, redownload: bool) -> str:
        if not redownload and os.path.isfile(self.dataset_path):
            return self.dataset_path
        response = self.http_client.get(FDA_UNII_DOWNLOAD_URL)
        response.raise_for_status()
        with open(self.dataset_path, "wb") as handle:
            handle.write(response.content)
        return self.dataset_path

    # -------------------------------------------------------------------------
    def load_dataset(self, path: str) -> pd.DataFrame:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        return frame

    # -------------------------------------------------------------------------
    def build_index(self, frame: pd.DataFrame) -> dict[str, dict[str, str]]:
        index: dict[str, dict[str, str]] = {}
        columns = {name.lower(): name for name in frame.columns}
        unii_column = columns.get("unii") or columns.get("ingredient unii")
        inchikey_column = columns.get("inchikey") or columns.get("substance key")
        name_column = columns.get("preferred substance name") or columns.get("ingredient name")
        if not unii_column and not inchikey_column:
            return index
        records = frame.to_dict(orient="records")
        for row in records:
            entry: dict[str, str] = {}
            if unii_column:
                entry["unii"] = row.get(unii_column, "")
            if inchikey_column:
                entry["inchikey"] = row.get(inchikey_column, "")
            if name_column:
                entry["preferred_name"] = row.get(name_column, "")
            key = entry.get("inchikey") or entry.get("unii")
            if key:
                index[str(key)] = entry
        return index

    # -------------------------------------------------------------------------
    def enrich(self, records: list[dict[str, Any]], *, redownload: bool) -> None:
        dataset_path = self.ensure_dataset(redownload=redownload)
        frame = self.load_dataset(dataset_path)
        index = self.build_index(frame)
        if not index:
            return
        for record in records:
            key = record.get("inchikey") or record.get("unii")
            if not key:
                continue
            entry = index.get(str(key))
            if not entry:
                continue
            if entry.get("inchikey") and not record.get("inchikey"):
                record["inchikey"] = entry["inchikey"]
            if entry.get("unii") and not record.get("unii"):
                record["unii"] = entry["unii"]
            preferred_name = entry.get("preferred_name")
            if preferred_name and preferred_name not in record.get("synonyms", []):
                synonyms = list(record.get("synonyms", []))
                synonyms.append(preferred_name)
                record["synonyms"] = list(dict.fromkeys(synonyms))


###############################################################################
class OpenFdaClient:
    def __init__(self, *, concurrency: int = OPENFDA_CONCURRENCY) -> None:
        self.semaphore = asyncio.Semaphore(max(1, concurrency))
        self.client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        await self.client.aclose()

    # -------------------------------------------------------------------------
    async def fetch_ndc_records(self, ingredient: str) -> list[dict[str, Any]]:
        if not ingredient:
            return []
        params = {
            "search": f"active_ingredients.name:\"{ingredient}\"",
            "limit": 100,
        }
        async with self.semaphore:
            response = await self.client.get(OPENFDA_NDC_ENDPOINT, params=params)
        if response.status_code == 404:
            return []
        response.raise_for_status()
        payload = response.json()
        return payload.get("results", [])

    # -------------------------------------------------------------------------
    async def enrich_record(self, record: dict[str, Any]) -> None:
        brands = set(record.get("brands", []))
        ingredient = record.get("preferred_name")
        ndc_records = await self.fetch_ndc_records(ingredient)
        for ndc in ndc_records:
            brand_name = ndc.get("brand_name") or ndc.get("generic_name")
            if brand_name:
                brands.add(brand_name)
            product_ndcs = ndc.get("product_ndc")
            if product_ndcs:
                xrefs = record.get("xrefs") or {}
                existing = xrefs.get("NDC", []) or []
                combined = list(dict.fromkeys([*existing, *([product_ndcs] if isinstance(product_ndcs, str) else product_ndcs)]))
                xrefs["NDC"] = combined
                record["xrefs"] = xrefs
        record["brands"] = sorted(brands)

    # -------------------------------------------------------------------------
    def enrich(self, records: list[dict[str, Any]]) -> None:
        async def runner() -> None:
            tasks = [self.enrich_record(record) for record in records]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run(runner())


###############################################################################
class DailyMedClient:
    def __init__(self, *, concurrency: int = DAILYMED_CONCURRENCY) -> None:
        self.semaphore = asyncio.Semaphore(max(1, concurrency))
        self.client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        await self.client.aclose()

    # -------------------------------------------------------------------------
    async def fetch_by_unii(self, unii: str) -> list[dict[str, Any]]:
        if not unii:
            return []
        params = {"ingredient": unii, "page": 1}
        url = f"{DAILYMED_SPL_BULK_BASE_URL}.json"
        async with self.semaphore:
            response = await self.client.get(url, params=params)
        if response.status_code == 404:
            return []
        response.raise_for_status()
        payload = response.json()
        return payload.get("data", [])

    # -------------------------------------------------------------------------
    async def enrich_record(self, record: dict[str, Any]) -> None:
        unii = record.get("unii")
        entries = await self.fetch_by_unii(unii)
        if not entries:
            return
        brands = set(record.get("brands", []))
        xrefs = record.get("xrefs") or {}
        for entry in entries:
            brand = entry.get("title") or entry.get("setid")
            if brand:
                brands.add(brand)
            set_id = entry.get("setid")
            if set_id:
                existing = xrefs.get("SPL", []) or []
                existing.append(set_id)
                xrefs["SPL"] = list(dict.fromkeys(existing))
        record["brands"] = sorted(brands)
        record["xrefs"] = xrefs

    # -------------------------------------------------------------------------
    def enrich(self, records: list[dict[str, Any]]) -> None:
        async def runner() -> None:
            tasks = [self.enrich_record(record) for record in records]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run(runner())


###############################################################################
class ChemblClient:
    def __init__(self, *, concurrency: int = CHEMBL_CONCURRENCY) -> None:
        self.semaphore = asyncio.Semaphore(max(1, concurrency))
        self.client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        await self.client.aclose()

    # -------------------------------------------------------------------------
    async def fetch_by_inchikey(self, inchikey: str) -> dict[str, Any] | None:
        if not inchikey:
            return None
        path = f"{CHEMBL_API_BASE_URL}/molecule.json"
        params = {"molecule_structures__standard_inchi_key": inchikey}
        async with self.semaphore:
            response = await self.client.get(path, params=params)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        if payload.get("page_meta", {}).get("total_count", 0) == 0:
            return None
        objects = payload.get("molecules") or payload.get("objects")
        if not objects:
            return None
        return objects[0]

    # -------------------------------------------------------------------------
    async def enrich_record(self, record: dict[str, Any]) -> None:
        inchikey = record.get("inchikey")
        payload = await self.fetch_by_inchikey(inchikey)
        if not payload:
            return
        chembl_id = payload.get("molecule_chembl_id")
        if chembl_id:
            xrefs = record.get("xrefs") or {}
            values = xrefs.get("ChEMBL", []) or []
            values.append(chembl_id)
            xrefs["ChEMBL"] = list(dict.fromkeys(values))
            record["xrefs"] = xrefs
        synonyms = payload.get("molecule_synonyms", [])
        if synonyms:
            names = [synonym.get("synonyms") or synonym.get("synonym") for synonym in synonyms]
            names = [name for name in names if name]
            merged = list(dict.fromkeys([*record.get("synonyms", []), *names]))
            record["synonyms"] = merged

    # -------------------------------------------------------------------------
    def enrich(self, records: list[dict[str, Any]]) -> None:
        async def runner() -> None:
            tasks = [self.enrich_record(record) for record in records]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run(runner())


###############################################################################
class DrugCatalogDeduplicator:
    def merge(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        identity_map: dict[str, dict[str, Any]] = {}
        for record in records:
            key = record.get("inchikey") or record.get("rxcui")
            if key not in identity_map:
                identity_map[key] = record
                continue
            existing = identity_map[key]
            existing["synonyms"] = list(
                dict.fromkeys([*existing.get("synonyms", []), *record.get("synonyms", [])])
            )
            existing["brands"] = sorted(
                set([*existing.get("brands", []), *record.get("brands", [])])
            )
            existing["rxcui_parents"] = list(
                dict.fromkeys([*existing.get("rxcui_parents", []), *record.get("rxcui_parents", [])])
            )
            for key_name in ["pubchem_cid", "inchikey", "unii", "cas", "status"]:
                if not existing.get(key_name) and record.get(key_name):
                    existing[key_name] = record[key_name]
            existing_xrefs = existing.get("xrefs") or {}
            incoming_xrefs = record.get("xrefs") or {}
            for xref_key, values in incoming_xrefs.items():
                combined = list(
                    dict.fromkeys([*(existing_xrefs.get(xref_key, []) or []), *values])
                )
                existing_xrefs[xref_key] = combined
            existing["xrefs"] = existing_xrefs
        return list(identity_map.values())


###############################################################################
class DrugCatalogUpdater:
    def __init__(
        self,
        sources_path: str,
        *,
        redownload: bool,
        rxnorm_archive: str | None = None,
        database_client=None,
    ) -> None:
        from DILIGENT.app.utils.repository.database import database

        self.sources_path = os.path.abspath(sources_path)
        os.makedirs(self.sources_path, exist_ok=True)
        self.redownload = redownload
        self.rxnorm_archive = rxnorm_archive
        self.database = database_client or database
        self.rxnorm_manager = RxNormReleaseManager(self.sources_path)
        self.pubchem_client = PubChemClient()
        self.fda_unii_client = FdaUniiClient(self.sources_path)
        self.openfda_client = OpenFdaClient()
        self.dailymed_client = DailyMedClient()
        self.chembl_client = ChemblClient()
        self.deduplicator = DrugCatalogDeduplicator()

    # -------------------------------------------------------------------------
    def serialize_list(self, values: Iterable[str]) -> str:
        unique = [value for value in dict.fromkeys(values) if value]
        return json.dumps(unique, ensure_ascii=False)

    # -------------------------------------------------------------------------
    def serialize_dict(self, data: dict[str, Any]) -> str:
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, list):
                cleaned[key] = [item for item in dict.fromkeys(value) if item]
            elif value:
                cleaned[key] = value
        return json.dumps(cleaned, ensure_ascii=False)

    # -------------------------------------------------------------------------
    def normalize_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [item for item in value if item]
        if isinstance(value, str):
            return [value] if value else []
        if value is None:
            return []
        if isinstance(value, float) and math.isnan(value):
            return []
        if isinstance(value, tuple) or isinstance(value, set):
            return [item for item in value if item]
        return []

    # -------------------------------------------------------------------------
    def normalize_dict(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {}

    # -------------------------------------------------------------------------
    def update_catalog(self) -> dict[str, Any]:
        release_dir = self.rxnorm_manager.ensure_release(
            redownload=self.redownload, archive_path=self.rxnorm_archive
        )
        builder = RxNormCatalogBuilder(release_dir)
        base_catalog = builder.build_base_catalog()
        records = base_catalog.to_dict(orient="records")
        self.pubchem_client.enrich(records)
        self.fda_unii_client.enrich(records, redownload=self.redownload)
        self.openfda_client.enrich(records)
        self.dailymed_client.enrich(records)
        self.chembl_client.enrich(records)
        merged_records = self.deduplicator.merge(records)
        updated_at = datetime.now(UTC)
        frame = pd.DataFrame.from_records(merged_records)
        if frame.empty:
            logger.warning("Drug catalog updater produced no records")
            return {"rows": 0, "status": "empty"}
        frame["synonyms"] = frame["synonyms"].apply(self.normalize_list).apply(self.serialize_list)
        frame["brands"] = frame["brands"].apply(self.normalize_list).apply(self.serialize_list)
        frame["rxcui_parents"] = (
            frame["rxcui_parents"].apply(self.normalize_list).apply(self.serialize_list)
        )
        frame["xrefs"] = frame["xrefs"].apply(self.normalize_dict).apply(self.serialize_dict)
        frame["status"] = frame["status"].fillna("unknown")
        frame["updated_at"] = updated_at
        self.database.upsert_into_database(frame, "DRUG_CATALOG")
        summary = {"rows": int(len(frame)), "updated_at": updated_at.isoformat()}
        logger.info("Drug catalog updated with %s rows", summary["rows"])
        return summary

    # -------------------------------------------------------------------------
    def close(self) -> None:
        self.rxnorm_manager.close()
        self.fda_unii_client.close()
        asyncio.run(self.pubchem_client.close())
        asyncio.run(self.openfda_client.close())
        asyncio.run(self.dailymed_client.close())
        asyncio.run(self.chembl_client.close())

