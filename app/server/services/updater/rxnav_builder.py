from __future__ import annotations

import asyncio
import codecs
import json
import os
import re
import time
import unicodedata
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pandas as pd

from common.constants import RXNAV_CURATED_ALIASES_PATH
from common.utils.logger import logger
from repositories.serialization.data import DataSerializer
from services.text.normalization import normalize_drug_name
from services.text.vocabulary import get_text_normalization_snapshot
from services.updater.rxnav_client import (
    RxNavClient,
    run_with_semaphore,
)


class RxNavDrugCatalogBuilder:
    TERMS_URL = "https://rxnav.nlm.nih.gov/REST/RxTerms/allconcepts.json"
    CHUNK_SIZE = 131_072
    MAX_RETRIES = 3
    RETRY_STATUS = {429, 500, 502, 503, 504}
    TIMEOUT = 30.0
    BACKOFF_TIME = (0.8, 1.6, 3.2)
    TABLE_NAME = "drug_aliases"
    BATCH_SIZE = 200
    SYNONYM_WORKERS = 12
    TOKEN_SPLIT_PATTERN = re.compile(r"[^A-Za-z0-9']+")
    SINGLE_TOKEN_DIGIT_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")
    SHORT_TOKEN_EXCEPTIONS = {"id"}

    def __init__(
        self,
        rx_client: RxNavClient | None = None,
        *,
        serializer: DataSerializer | None = None,
        curated_aliases_path: str | None = None,
    ) -> None:
        vocabulary = get_text_normalization_snapshot()
        combined: set[str] = set(vocabulary.rxnav_name_stopwords)
        synonym_stopwords = {
            word.casefold() for word in vocabulary.rxnav_synonym_stopwords
        }
        combined.update(synonym_stopwords)
        self.stopwords = combined
        self.synonym_stopwords = synonym_stopwords
        self.brand_pattern = re.compile(r"\[([^\]]+)\]")
        self.rx_client = rx_client or RxNavClient()
        self.alias_cache: dict[str, list[str]] = {}
        self.rxcui_cache: dict[str, list[str]] = {}
        self.total_records: int | None = None
        self.last_logged_count = 0
        self.serializer = serializer or DataSerializer()
        resolved_path = curated_aliases_path or RXNAV_CURATED_ALIASES_PATH
        self.curated_aliases_path = os.path.abspath(resolved_path)
        self.curated_aliases_by_canonical = self.load_curated_aliases()

    # -------------------------------------------------------------------------
    def load_curated_aliases(self) -> dict[str, list[tuple[str, str]]]:
        path = self.curated_aliases_path
        if not os.path.exists(path):
            logger.info("RxNav curated alias file not found at '%s'; skipping", path)
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to read RxNav curated alias file '%s': %s",
                path,
                exc,
            )
            return {}

        records: list[Any]
        if isinstance(payload, dict):
            candidate_records = payload.get("aliases")
            records = candidate_records if isinstance(candidate_records, list) else []
        elif isinstance(payload, list):
            records = payload
        else:
            records = []

        curated: dict[str, dict[tuple[str, str], tuple[str, str]]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            canonical_name = record.get("canonical_name")
            if not isinstance(canonical_name, str):
                continue
            canonical_norm = normalize_drug_name(canonical_name)
            if not canonical_norm:
                continue
            raw_kind = record.get("alias_kind")
            alias_kind = (
                str(raw_kind).strip().lower()
                if isinstance(raw_kind, str) and str(raw_kind).strip()
                else "synonym"
            )
            aliases: list[str] = []
            alias_single = record.get("alias")
            if isinstance(alias_single, str):
                aliases.append(alias_single)
            alias_list = record.get("aliases")
            if isinstance(alias_list, list):
                for value in alias_list:
                    if isinstance(value, str):
                        aliases.append(value)
            for raw_alias in aliases:
                alias = raw_alias.strip()
                if not alias:
                    continue
                alias_norm = normalize_drug_name(alias)
                if not alias_norm:
                    continue
                by_alias = curated.setdefault(canonical_norm, {})
                key = (alias_norm, alias_kind)
                if key not in by_alias:
                    by_alias[key] = (alias, alias_kind)

        return {
            canonical: sorted(values.values(), key=lambda item: item[0].casefold())
            for canonical, values in curated.items()
        }

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
    def update_drug_catalog(
        self,
        *,
        total_records: int | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        self.total_records = total_records
        self.last_logged_count = 0
        attempt = 0
        last_error: Exception | None = None
        self.emit_progress(
            progress_callback,
            progress=2.0,
            message="Downloading RxNav catalog payload",
        )
        while attempt < self.MAX_RETRIES:
            try:
                with httpx.stream(
                    "GET", self.TERMS_URL, timeout=self.TIMEOUT
                ) as response:
                    if self.should_cancel(should_stop):
                        raise RuntimeError("RxNav update cancelled by user request")
                    if (
                        response.status_code in self.RETRY_STATUS
                        and attempt + 1 < self.MAX_RETRIES
                    ):
                        time.sleep(
                            self.BACKOFF_TIME[min(attempt, len(self.BACKOFF_TIME) - 1)]
                        )
                        attempt += 1
                        continue
                    response.raise_for_status()
                    started = time.perf_counter()
                    self.emit_progress(
                        progress_callback,
                        progress=8.0,
                        message="Processing RxNav records",
                    )
                    result = self.persist_catalog(
                        response.iter_bytes(self.CHUNK_SIZE),
                        progress_callback=progress_callback,
                        should_stop=should_stop,
                    )
                    elapsed = time.perf_counter() - started
                    self.emit_progress(
                        progress_callback,
                        progress=98.0,
                        message="Finalizing RxNav catalog refresh",
                    )
                    return self.compose_catalog_payload(
                        response,
                        result,
                        attempts=attempt + 1,
                        elapsed=elapsed,
                    )
            except httpx.HTTPStatusError as exc:  # pragma: no cover - network dependent
                last_error = exc
                if (
                    exc.response is not None
                    and exc.response.status_code in self.RETRY_STATUS
                    and attempt + 1 < self.MAX_RETRIES
                ):
                    time.sleep(
                        self.BACKOFF_TIME[min(attempt, len(self.BACKOFF_TIME) - 1)]
                    )
                    attempt += 1
                    continue
                break
            except httpx.RequestError as exc:  # pragma: no cover - network dependent
                last_error = exc
                if attempt + 1 < self.MAX_RETRIES:
                    time.sleep(
                        self.BACKOFF_TIME[min(attempt, len(self.BACKOFF_TIME) - 1)]
                    )
                    attempt += 1
                    continue
                break
        if last_error is not None:
            raise RuntimeError("Failed to download RxNav drug catalog") from last_error
        raise RuntimeError("Failed to download RxNav drug catalog")

    # -------------------------------------------------------------------------
    def compose_catalog_payload(
        self,
        response: httpx.Response,
        result: dict[str, Any],
        *,
        attempts: int,
        elapsed: float,
    ) -> dict[str, Any]:
        headers = response.headers
        try:
            content_length = int(headers.get("Content-Length", 0) or 0)
        except (TypeError, ValueError):
            content_length = 0
        payload = {
            "source_url": str(response.request.url)
            if response.request is not None
            else self.TERMS_URL,
            "downloaded": True,
            "status_code": response.status_code,
            "content_length": content_length,
            "content_type": headers.get("Content-Type"),
            "last_modified": headers.get("Last-Modified"),
            "attempts": attempts,
            "elapsed_seconds": elapsed,
        }
        payload.update(result)
        count = result.get("count", 0)
        payload.setdefault("records", count)
        payload.setdefault("processed_entries", count)
        return payload

    # -------------------------------------------------------------------------
    def persist_catalog(
        self,
        chunks: Iterator[bytes],
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        count = 0
        concepts_batch: list[dict[str, Any]] = []
        for concept in self.stream_min_concepts(chunks):
            if self.should_cancel(should_stop):
                raise RuntimeError("RxNav update cancelled by user request")
            concepts_batch.append(concept)
            if len(concepts_batch) >= self.BATCH_SIZE:
                persisted = self.persist_concept_batch(concepts_batch)
                count += persisted
                concepts_batch.clear()
                logger.info("Total records upserted into database: %d", count)
                self.emit_catalog_progress(progress_callback, count=count)
        if concepts_batch:
            if self.should_cancel(should_stop):
                raise RuntimeError("RxNav update cancelled by user request")
            persisted = self.persist_concept_batch(concepts_batch)
            count += persisted
            logger.info("Total records upserted into database: %d", count)
            self.emit_catalog_progress(progress_callback, count=count)

        return {"table_name": self.TABLE_NAME, "count": count}

    # -------------------------------------------------------------------------
    def emit_catalog_progress(
        self,
        progress_callback: Callable[[float, str], None] | None,
        *,
        count: int,
    ) -> None:
        if progress_callback is None:
            return
        denominator = float(self.total_records or 50_000)
        ratio = min(1.0, max(0.0, count / denominator))
        progress = 8.0 + (ratio * 87.0)
        self.emit_progress(
            progress_callback,
            progress=progress,
            message=f"Upserted {count} RxNav records",
        )

    # -------------------------------------------------------------------------
    def persist_concept_batch(self, concepts: list[dict[str, Any]]) -> int:
        if not concepts:
            return 0
        self.prefetch_concept_queries(concepts)
        payload_batch: list[dict[str, Any]] = []
        for concept in concepts:
            payload = self.sanitize_concept(concept)
            if payload is None:
                continue
            payload_batch.append(payload)
        if not payload_batch:
            return 0
        self.persist_batch(payload_batch)
        return len(payload_batch)

    # -------------------------------------------------------------------------
    def prefetch_concept_queries(self, concepts: list[dict[str, Any]]) -> None:
        pending_alias_queries: dict[str, str] = {}
        pending_synonym_identifiers: dict[str, str] = {}
        for concept in concepts:
            if not isinstance(concept, dict):
                continue
            full_name = concept.get("fullName")
            if isinstance(full_name, str):
                stripped_full_name = full_name.strip()
                if stripped_full_name:
                    full_name_key = stripped_full_name.casefold()
                    if full_name_key not in self.alias_cache:
                        pending_alias_queries[full_name_key] = stripped_full_name
                    sanitized_name = self.sanitize_name(stripped_full_name)
                    if sanitized_name:
                        sanitized_key = sanitized_name.casefold()
                        if sanitized_key not in self.alias_cache:
                            pending_alias_queries[sanitized_key] = sanitized_name
            rxcui = concept.get("rxcui")
            rxcui_identifier = str(rxcui).strip()
            if rxcui_identifier and rxcui_identifier not in self.rxcui_cache:
                pending_synonym_identifiers[rxcui_identifier] = rxcui_identifier
        if not pending_alias_queries and not pending_synonym_identifiers:
            return
        alias_results, synonym_results = self.fetch_bulk_pending_queries(
            pending_alias_queries,
            pending_synonym_identifiers,
        )
        self.alias_cache.update(alias_results)
        self.rxcui_cache.update(synonym_results)

    # -------------------------------------------------------------------------
    def fetch_bulk_pending_queries(
        self,
        pending_alias_queries: dict[str, str],
        pending_synonym_identifiers: dict[str, str],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        if not pending_alias_queries and not pending_synonym_identifiers:
            return {}, {}
        return asyncio.run(
            self.fetch_bulk_pending_queries_async(
                pending_alias_queries,
                pending_synonym_identifiers,
            )
        )

    # -------------------------------------------------------------------------
    async def fetch_bulk_pending_queries_async(
        self,
        pending_alias_queries: dict[str, str],
        pending_synonym_identifiers: dict[str, str],
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        alias_results: dict[str, list[str]] = {}
        synonym_results: dict[str, list[str]] = {}
        concurrency_limit = max(1, int(self.rx_client.max_concurrency))
        semaphore = asyncio.Semaphore(concurrency_limit)
        chunk_size = max(concurrency_limit * 4, concurrency_limit)

        async with httpx.AsyncClient(
            timeout=self.rx_client.timeout,
            limits=self.rx_client._build_limits(),
        ) as client:
            alias_entries = list(pending_alias_queries.items())
            for start in range(0, len(alias_entries), chunk_size):
                chunk = alias_entries[start : start + chunk_size]
                tasks = {
                    cache_key: asyncio.create_task(
                        run_with_semaphore(
                            semaphore,
                            lambda query=query: self.rx_client.fetch_drug_terms_async(
                                query, client=client
                            ),
                        )
                    )
                    for cache_key, query in chunk
                }
                for cache_key, task in tasks.items():
                    try:
                        alias_results[cache_key] = await task
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to prefetch RxNav aliases for '%s': %s",
                            pending_alias_queries.get(cache_key),
                            exc,
                        )
                        alias_results[cache_key] = []

            synonym_entries = list(pending_synonym_identifiers)
            for start in range(0, len(synonym_entries), chunk_size):
                chunk = synonym_entries[start : start + chunk_size]
                tasks = {
                    identifier: asyncio.create_task(
                        run_with_semaphore(
                            semaphore,
                            lambda identifier=identifier: (
                                self.rx_client.fetch_rxcui_synonyms_async(
                                    identifier, client=client
                                )
                            ),
                        )
                    )
                    for identifier in chunk
                }
                for identifier, task in tasks.items():
                    try:
                        synonym_results[identifier] = await task
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to prefetch RxNav synonyms for '%s': %s",
                            identifier,
                            exc,
                        )
                        synonym_results[identifier] = []
        return alias_results, synonym_results

    # -------------------------------------------------------------------------
    def persist_batch(self, batch: list[dict[str, Any]]) -> None:
        frame = pd.DataFrame(batch)
        if frame.empty:
            return
        self.serializer.upsert_drugs_catalog_records(
            frame,
            curated_aliases_by_canonical=self.curated_aliases_by_canonical,
        )

    # -------------------------------------------------------------------------
    def stream_min_concepts(self, chunks: Iterator[bytes]) -> Iterator[dict[str, Any]]:
        utf8_decoder = codecs.getincrementaldecoder("utf-8")()
        decoder = json.JSONDecoder()
        buffer = ""
        in_array = False
        for chunk in chunks:
            text = utf8_decoder.decode(chunk)
            if not text:
                continue
            buffer += text
            while True:
                if not in_array:
                    index = buffer.find("[")
                    if index == -1:
                        if len(buffer) > 1024:
                            buffer = buffer[-1024:]
                        break
                    in_array = True
                    buffer = buffer[index + 1 :]
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == "]":
                    in_array = False
                    buffer = buffer[1:]
                    continue
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                try:
                    parsed, offset = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                if isinstance(parsed, dict):
                    yield parsed
                buffer = buffer[offset:]
        buffer += utf8_decoder.decode(b"", final=True)
        while True:
            if not in_array:
                index = buffer.find("[")
                if index == -1:
                    break
                in_array = True
                buffer = buffer[index + 1 :]
            buffer = buffer.lstrip()
            if not buffer:
                break
            if buffer[0] == "]":
                in_array = False
                buffer = buffer[1:]
                continue
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue
            try:
                parsed, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                break
            if isinstance(parsed, dict):
                yield parsed
            buffer = buffer[offset:]

    # -------------------------------------------------------------------------
    def sanitize_concept(self, concept: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(concept, dict):
            return None
        full_name = concept.get("fullName")
        if not isinstance(full_name, str):
            return None
        rxcui = concept.get("rxcui")
        rxcui_str = str(rxcui).strip()
        if not rxcui_str:
            return None
        term_type = concept.get("termType")
        sanitized_name = self.sanitize_name(full_name) or None
        brands = sorted(self.extract_brands(full_name), key=str.casefold)
        formatted_brands = self.format_brand_names(brands)
        synonyms = self.collect_synonyms(
            rxcui_str,
            sanitized_name,
            full_name.strip(),
            brands,
        )
        payload = {
            "rxcui": rxcui_str,
            "term_type": term_type.strip() if isinstance(term_type, str) else "",
            "raw_name": full_name.strip(),
            "name": sanitized_name,
            "brand_names": formatted_brands,
            "synonyms": synonyms,
        }
        if (
            payload["name"] is None
            and not payload["brand_names"]
            and not payload["synonyms"]
        ):
            return None
        return payload

    # -------------------------------------------------------------------------
    def collect_synonyms(
        self,
        rxcui: str,
        name: str | None,
        raw_name: str,
        brand_names: list[str],
    ) -> list[str]:
        normalized_name = name.casefold() if isinstance(name, str) else None
        normalized_brands = {
            brand.casefold()
            for brand in brand_names
            if isinstance(brand, str) and brand
        }
        aliases: dict[str, str] = {}

        queries = []
        if isinstance(name, str):
            queries.append(name)
        if isinstance(raw_name, str):
            queries.append(raw_name)

        pending_alias_queries: dict[str, str] = {}

        for query in queries:
            stripped = query.strip()
            if not stripped:
                continue
            cache_key = stripped.casefold()
            cached = self.alias_cache.get(cache_key)
            if cached is None:
                pending_alias_queries[cache_key] = stripped
                continue
            for term in cached:
                for variant in self.expand_synonym_variants(term):
                    self.register_alias_candidate(
                        variant,
                        aliases,
                        normalized_name,
                        normalized_brands,
                    )

        identifier = rxcui.strip()
        pending_synonym_identifier: str | None = None
        if identifier:
            cached_synonyms = self.rxcui_cache.get(identifier)
            if cached_synonyms is None:
                pending_synonym_identifier = identifier
            else:
                for term in cached_synonyms:
                    for variant in self.expand_synonym_variants(term):
                        self.register_alias_candidate(
                            variant,
                            aliases,
                            normalized_name,
                            normalized_brands,
                        )

        if pending_alias_queries or pending_synonym_identifier:
            alias_results, synonym_results = self.fetch_pending_queries(
                pending_alias_queries,
                pending_synonym_identifier,
            )
            for cache_key, filtered_terms in alias_results.items():
                self.alias_cache[cache_key] = filtered_terms
                for term in filtered_terms:
                    for variant in self.expand_synonym_variants(term):
                        self.register_alias_candidate(
                            variant,
                            aliases,
                            normalized_name,
                            normalized_brands,
                        )
            if pending_synonym_identifier is not None:
                self.rxcui_cache[pending_synonym_identifier] = synonym_results
                for term in synonym_results:
                    for variant in self.expand_synonym_variants(term):
                        self.register_alias_candidate(
                            variant,
                            aliases,
                            normalized_name,
                            normalized_brands,
                        )

        for cache_key, stripped in pending_alias_queries.items():
            cached = self.alias_cache.get(cache_key, [])
            for term in cached:
                for variant in self.expand_synonym_variants(term):
                    self.register_alias_candidate(
                        variant,
                        aliases,
                        normalized_name,
                        normalized_brands,
                    )

        if pending_synonym_identifier is not None:
            cached_synonyms = self.rxcui_cache.get(pending_synonym_identifier, [])
            for term in cached_synonyms:
                for variant in self.expand_synonym_variants(term):
                    self.register_alias_candidate(
                        variant,
                        aliases,
                        normalized_name,
                        normalized_brands,
                    )

        if not aliases:
            return []
        return sorted(aliases.values(), key=str.casefold)

    # -------------------------------------------------------------------------
    def fetch_pending_queries(
        self,
        pending_alias_queries: dict[str, str],
        pending_synonym_identifier: str | None,
    ) -> tuple[dict[str, list[str]], list[str]]:
        if not pending_alias_queries and pending_synonym_identifier is None:
            return {}, []
        return asyncio.run(
            self.fetch_pending_queries_async(
                pending_alias_queries,
                pending_synonym_identifier,
            )
        )

    # -------------------------------------------------------------------------
    async def fetch_pending_queries_async(
        self,
        pending_alias_queries: dict[str, str],
        pending_synonym_identifier: str | None,
    ) -> tuple[dict[str, list[str]], list[str]]:
        alias_results: dict[str, list[str]] = {}
        synonym_results: list[str] = []
        async with httpx.AsyncClient(
            timeout=self.rx_client.timeout,
            limits=self.rx_client._build_limits(),
        ) as client:
            alias_tasks = {
                cache_key: asyncio.create_task(
                    self.rx_client.fetch_drug_terms_async(stripped, client=client)
                )
                for cache_key, stripped in pending_alias_queries.items()
            }
            synonym_task = (
                asyncio.create_task(
                    self.rx_client.fetch_rxcui_synonyms_async(
                        pending_synonym_identifier, client=client
                    )
                )
                if pending_synonym_identifier
                else None
            )
            for cache_key, task in alias_tasks.items():
                try:
                    alias_results[cache_key] = await task
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to fetch RxNav aliases for '%s': %s",
                        pending_alias_queries.get(cache_key),
                        exc,
                    )
                    alias_results[cache_key] = []
            if synonym_task is not None:
                try:
                    synonym_results = await synonym_task
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to fetch RxNav synonyms for '%s': %s",
                        pending_synonym_identifier,
                        exc,
                    )
                    synonym_results = []
        return alias_results, synonym_results

    # -------------------------------------------------------------------------
    def expand_synonym_variants(self, candidate: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", candidate)
        normalized = re.sub(r"\s+", " ", normalized)
        fragments: list[str] = []
        seen: set[str] = set()
        for match in self.brand_pattern.findall(normalized):
            stripped = match.strip()
            if not stripped:
                continue
            key = stripped.casefold()
            if key in seen:
                continue
            seen.add(key)
            fragments.append(stripped)
        base = self.brand_pattern.sub(" ", normalized).strip()
        if base:
            key = base.casefold()
            if key not in seen:
                seen.add(key)
                fragments.append(base)
        return fragments

    ###########################################################################
    def register_alias_candidate(
        self,
        candidate: str,
        aliases: dict[str, str],
        normalized_name: str | None,
        normalized_brands: set[str],
    ) -> None:
        cleaned = unicodedata.normalize("NFKC", candidate).strip()
        if not cleaned:
            return
        cleaned = re.sub(r"\s+", " ", cleaned)
        tokens = [token for token in self.TOKEN_SPLIT_PATTERN.split(cleaned) if token]
        if not tokens:
            return
        sanitized_tokens: list[str] = []
        seen_tokens: set[str] = set()
        for token in tokens:
            stripped = token.strip("'")
            if not stripped:
                continue
            if any(char.isdigit() for char in stripped):
                continue
            lowered = stripped.casefold()
            if lowered in self.stopwords:
                continue
            if lowered.rstrip("s") in self.stopwords:
                continue
            if len(lowered) <= 2 and lowered not in self.SHORT_TOKEN_EXCEPTIONS:
                continue
            if lowered in seen_tokens:
                continue
            seen_tokens.add(lowered)
            sanitized_tokens.append(stripped)
        if not sanitized_tokens:
            return
        cleaned_alias = " ".join(sanitized_tokens)
        cleaned_alias = re.sub(r"\s+", " ", cleaned_alias).strip()
        if not cleaned_alias:
            return
        if (
            len(cleaned_alias) <= 2
            and cleaned_alias.casefold() not in self.SHORT_TOKEN_EXCEPTIONS
        ):
            return
        if self.SINGLE_TOKEN_DIGIT_PATTERN.match(cleaned_alias):
            return
        key = cleaned_alias.casefold()
        if key in self.stopwords:
            return
        aliases.setdefault(key, cleaned_alias)

    # -------------------------------------------------------------------------
    def sanitize_name(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        normalized = self.brand_pattern.sub(" ", normalized)
        normalized = re.sub(r"\([^)]*\)", " ", normalized)
        normalized = normalized.replace("/", " ")
        normalized = re.sub(r"[^A-Za-z\s'-]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        tokens = normalized.split(" ")
        cleaned: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            stripped = re.sub(r"[^A-Za-z']", "", token)
            if not stripped:
                continue
            lowered = stripped.lower()
            if len(lowered) < 3:
                continue
            if lowered in self.stopwords or lowered.rstrip("s") in self.stopwords:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(lowered)
        return " ".join(cleaned)

    # -------------------------------------------------------------------------
    def extract_brands(self, value: str) -> list[str]:
        seen: set[str] = set()
        brands: list[str] = []
        for match in self.brand_pattern.findall(value):
            normalized = unicodedata.normalize("NFKC", match)
            normalized = re.sub(r"\s+", " ", normalized).strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            brands.append(normalized)
        return brands

    # -------------------------------------------------------------------------
    def format_brand_names(self, brands: list[str]) -> str | None:
        if not brands:
            return None
        seen: set[str] = set()
        formatted: list[str] = []
        for brand in brands:
            stripped = brand.strip()
            if not stripped:
                continue
            key = stripped.casefold()
            if key in seen:
                continue
            seen.add(key)
            formatted.append(stripped)
        if not formatted:
            return None
        if len(formatted) == 1:
            return formatted[0]
        return ", ".join(formatted)

