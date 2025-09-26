from __future__ import annotations

import asyncio
import os
import re
import tarfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

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

from pdfminer.high_level import extract_text as pdfminer_extract_text
from pypdf import PdfReader


from Pharmagent.app.constants import (
    LIVERTOX_ARCHIVE,
    LIVERTOX_BASE_URL,
    SOURCES_PATH,
)



import pandas as pd

from Pharmagent.app.api.models.prompts import (
    LIVERTOX_MATCH_LIST_USER_PROMPT,
    LIVERTOX_MATCH_SYSTEM_PROMPT,
)
from Pharmagent.app.api.models.providers import initialize_llm_client
from Pharmagent.app.api.schemas.clinical import LiverToxBatchMatchSuggestion
from Pharmagent.app.configurations import ClientRuntimeConfig
from Pharmagent.app.constants import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    LIVERTOX_ARCHIVE,
    LIVERTOX_LLM_TIMEOUT_SECONDS,
    LIVERTOX_SKIP_DETERMINISTIC_RATIO,
    LIVERTOX_YIELD_INTERVAL,
    LLM_NULL_MATCH_NAMES,
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
        self.sources_path = os.path.abspath(sources_path)
        self.redownload = redownload  
        self.rx_client = rx_client or RxNavClient()
        self.serializer = serializer or DataSerializer()
        self.database = database_client

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

    # -----------------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        logger.info("Starting LiverTox update")       
        archive_path = os.path.join(self.sources_path, LIVERTOX_ARCHIVE)

        if self.redownload:
            logger.info("Redownload flag enabled; fetching latest LiverTox archive")
            download_info = self.download_archive()            
        else:
            logger.info("Using existing LiverTox archive")

        download_info = self.collect_local_archive_info(archive_path)
        logger.info("Extracting LiverTox monographs from %s", archive_path)
        extracted = self.collect_monographs(archive_path)
        logger.info("Sanitizing %d extracted entries", len(extracted))
        records = self.sanitize_records(extracted)
        logger.info("Enriching %d sanitized entries with RxNav terms", len(records))
        enriched = self.enrich_records(records)
        logger.info("Persisting enriched records to database")
        self.serializer.save_livertox_records(enriched)
             
        payload = dict(download_info)
        payload["file_path"] = archive_path
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

    # -------------------------------------------------------------------------
    def download_archive(self) -> dict[str, Any]:
        try:
            return asyncio.run(self.download_bulk_data(self.sources_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to download LiverTox archive: {exc}") from exc
        
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
    def collect_monographs(
        self, archive_path: str | None = None
    ) -> list[dict[str, str]]:
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
                drug_name = self._extract_title(
                    markup_text or "", plain_text, record_nbk
                )
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
                entry["additional_names"] = None
                entry["synonyms"] = None
                continue
            try:
                names, synonyms = self.rx_client.fetch_drug_terms(drug_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to enrich '%s': %s", drug_name, exc)
                entry["additional_names"] = None
                entry["synonyms"] = None
                continue
            entry["additional_names"] = ", ".join(names) if names else None
            entry["synonyms"] = ", ".join(synonyms) if synonyms else None
        return records


  

###############################################################################
class LiverToxMatcher:
    DIRECT_CONFIDENCE = 1.0
    ALIAS_CONFIDENCE = 0.95
    MIN_CONFIDENCE = 0.40
    LLM_DEFAULT_CONFIDENCE = 0.65
    LLM_TIMEOUT_SECONDS = LIVERTOX_LLM_TIMEOUT_SECONDS
    YIELD_INTERVAL = LIVERTOX_YIELD_INTERVAL
    DETERMINISTIC_SKIP_RATIO = LIVERTOX_SKIP_DETERMINISTIC_RATIO

    # -------------------------------------------------------------------------
    def __init__(
        self,
        livertox_df: pd.DataFrame,
        *,
        llm_client: Any | None = None,
    ) -> None:
        self.livertox_df = livertox_df
        self.llm_client = llm_client or initialize_llm_client(
            purpose="parser", timeout_s=DEFAULT_LLM_TIMEOUT_SECONDS
        )
        self.match_cache: dict[str, LiverToxMatch | None] = {}
        self.records: list[MonographRecord] = []
        self.records_by_normalized: dict[str, MonographRecord] = {}
        self.matching_pool_index: dict[str, list[MonographRecord]] = {}
        self.rows_by_nbk: dict[str, dict[str, Any]] = {}
        self.candidate_prompt_block: str | None = None
        self._build_records()

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

        # Step 1: normalize input names once and reuse throughout the flow.
        normalized_queries = [self._normalize_name(name) for name in patient_drugs]
        unresolved_indices: list[int] = []
        deterministic_matches = 0
        eligible_total = 0

        # Step 2: attempt cache hits and deterministic matches before any LLM call.
        for idx, normalized in enumerate(normalized_queries):            
            if not normalized:
                unresolved_indices.append(idx)
                continue
            eligible_total += 1
            cached = self.match_cache.get(normalized)
            if cached is not None:
                results[idx] = cached
                if cached.reason != "llm_fallback":
                    deterministic_matches += 1
                else:
                    unresolved_indices.append(idx)
                continue
            deterministic = self._deterministic_lookup(normalized)
            if deterministic is None:
                self.match_cache.setdefault(normalized, None)
                unresolved_indices.append(idx)
                continue
            record, confidence, reason, notes = deterministic
            match = self._create_match(record, confidence, reason, notes)
            self.match_cache[normalized] = match
            results[idx] = match
            if reason != "llm_fallback":
                deterministic_matches += 1
            else:
                unresolved_indices.append(idx)

        if not unresolved_indices:
            return results

        # Step 3: fall back to the language model only when deterministic coverage is low.
        deterministic_ratio = deterministic_matches / max(eligible_total, 1)
        if deterministic_ratio >= self.DETERMINISTIC_SKIP_RATIO:
            return results

        fallback_matches = await self._llm_batch_match_lookup(
            patient_drugs,
            normalized_queries=normalized_queries,
        )

        # Step 4: merge LLM suggestions back into the response cache.
        for idx in unresolved_indices:
            normalized = normalized_queries[idx]
            if not normalized:
                continue
            fallback = fallback_matches[idx] if idx < len(fallback_matches) else None
            if fallback is None:
                self.match_cache[normalized] = None
                continue
            record, confidence, reason, notes = fallback
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
                getattr(row, "additional_names", None),
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
        self.candidate_prompt_block = "\n".join(
            f"- {record.drug_name}" for record in self.records
        )

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
    def _deterministic_lookup(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        if not normalized_query:
            return None
        direct = self.records_by_normalized.get(normalized_query)
        if direct is not None:
            return direct, self.DIRECT_CONFIDENCE, "direct_match", []
        pool_match = self._match_from_pool(normalized_query)
        if pool_match is not None:
            record, token = pool_match
            note = f"token='{token}'"
            return record, self.ALIAS_CONFIDENCE, "alias_match", [note]
        return None

    # -------------------------------------------------------------------------
    async def _llm_batch_match_lookup(
        self,
        patient_drugs: list[str],
        *,
        normalized_queries: list[str],
    ) -> list[tuple[MonographRecord, float, str, list[str]] | None]:
        total = len(patient_drugs)
        if total == 0 or not self.records:
            return []
        # The prompt block is cached so repeated calls do not rebuild the monograph list.
        candidate_block = self.candidate_prompt_block or ""
        if not candidate_block:
            return [None] * total
        drugs_block = "\n".join(f"- {name}" if name else "-" for name in patient_drugs)
        prompt = LIVERTOX_MATCH_LIST_USER_PROMPT.format(
            patient_drugs=drugs_block,
            candidates=candidate_block,
        )
        model_name = ClientRuntimeConfig.get_parsing_model()
        logger.debug(
            "Dispatching batch LLM match for %s drugs against %s candidates (prompt length=%s chars)",
            total,
            len(self.records),
            len(prompt),
        )
        start_time = time.perf_counter()
        try:
            suggestion = await asyncio.wait_for(
                self.llm_client.llm_structured_call(
                    model=model_name,
                    system_prompt=LIVERTOX_MATCH_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    schema=LiverToxBatchMatchSuggestion,
                    temperature=0.0,
                ),
                timeout=self.LLM_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            duration = time.perf_counter() - start_time
            logger.error(
                "LLM batch match timed out after %.2fs using model '%s'",
                duration,
                model_name,
            )
            return [None] * total
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - start_time
            logger.error(
                "LLM batch match failed after %.2fs: %s",
                duration,
                exc,
            )
            return [None] * total
        duration = time.perf_counter() - start_time
        logger.debug(
            "Batch LLM matching completed in %.2fs using model '%s'",
            duration,
            model_name,
        )
        matches = suggestion.matches if suggestion else []
        if not matches:
            logger.warning(
                "LLM returned no matches for %s patient drugs",
                total,
            )
            return [None] * total
        if len(matches) != total:
            logger.warning(
                "LLM returned %s matches for %s patient drugs",
                len(matches),
                total,
            )
        normalized_to_items: dict[str, list[Any]] = {}
        for item in matches:
            normalized_drug = self._normalize_name(getattr(item, "drug_name", "") or "")
            if not normalized_drug:
                continue
            bucket = normalized_to_items.setdefault(normalized_drug, [])
            bucket.append(item)
        results: list[tuple[MonographRecord, float, str, list[str]] | None] = [
            None
        ] * total
        for idx, original in enumerate(patient_drugs):
            normalized_query = (
                normalized_queries[idx] if idx < len(normalized_queries) else ""
            )
            if not normalized_query:
                continue
            item: Any | None = None
            if idx < len(matches):
                candidate = matches[idx]
                candidate_normalized = self._normalize_name(
                    getattr(candidate, "drug_name", "") or ""
                )
                if candidate_normalized == normalized_query:
                    item = candidate
                    bucket = normalized_to_items.get(normalized_query)
                    if bucket:
                        try:
                            bucket.remove(candidate)
                        except ValueError:
                            pass
            if item is None:
                bucket = normalized_to_items.get(normalized_query)
                if bucket:
                    item = bucket.pop(0)
            if item is None:
                logger.debug(
                    "LLM did not return a usable match for '%s'",
                    original,
                )
                continue
            match_name = (getattr(item, "match_name", "") or "").strip()
            normalized_match = self._normalize_name(match_name)
            if normalized_match in LLM_NULL_MATCH_NAMES:
                logger.debug(
                    "LLM explicitly reported no match for '%s'",
                    original,
                )
                continue
            if not normalized_match:
                continue
            record: MonographRecord | None = self.records_by_normalized.get(
                normalized_match
            )
            confidence_raw = getattr(item, "confidence", None)
            confidence = (
                float(confidence_raw)
                if confidence_raw is not None
                else self.LLM_DEFAULT_CONFIDENCE
            )
            notes: list[str] = [f"LLM selected '{match_name}' for '{original}'"]
            rationale = (getattr(item, "rationale", "") or "").strip()
            if rationale:
                notes.append(rationale)
            if record is None:
                pool_match = self._match_from_pool(normalized_match)
                if pool_match is None and normalized_match != normalized_query:
                    pool_match = self._match_from_pool(normalized_query)
                if pool_match is not None:
                    record, token = pool_match
                    notes.append(f"token='{token}'")
                    confidence = max(confidence, self.ALIAS_CONFIDENCE)
            if record is None and normalized_match != normalized_query:
                record = self.records_by_normalized.get(normalized_query)
            if record is None:
                logger.debug(
                    "Unable to map LLM suggestion '%s' for '%s' to a monograph",
                    match_name,
                    original,
                )
                continue
            results[idx] = (record, confidence, "llm_fallback", notes)
        return results

    # -------------------------------------------------------------------------
    def _normalize_name(self, name: str) -> str:
        normalized = (
            unicodedata.normalize("NFKD", name)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        # Always lowercase before stripping punctuation to keep matching case-insensitive.
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
