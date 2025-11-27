from __future__ import annotations

import codecs
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any
from collections.abc import Iterator

import httpx
import pandas as pd

from DILIGENT.server.packages.logger import logger
from DILIGENT.server.packages.constants import RXNAV_SYNONYM_STOPWORDS
from DILIGENT.server.packages.utils.repository.serializer import DataSerializer



###############################################################################
@dataclass(slots=True)
class RxNormCandidate:
    value: str
    kind: str


###############################################################################
class RxNavClient:
    BASE_URL = "https://rxnav.nlm.nih.gov/REST/drugs.json"
    RXCUI_PROPERTY_URL = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/property.json"
    MAX_RETRIES = 3
    BACKOFF_TIME = (0.6, 1.2, 2.4)
    RETRY_STATUS = {429, 500, 502, 503, 504}
    TIMEOUT = 10.0
    SALT_STOPWORDS = {
        "acetate",
        "adipate",
        "aluminum",
        "bitartrate",
        "bromide",
        "besylate",
        "calcium",
        "carbonate",
        "chloride",
        "citrate",
        "diacetate",
        "dihydrate",
        "disodium",
        "fumarate",
        "hydrobromide",
        "hydrochloride",
        "hydrogen",
        "isosorbide",
        "lactate",
        "maleate",
        "mesylate",
        "nitrate",
        "phosphate",
        "potassium",
        "sesquihydrate",
        "sodium",
        "succinate",
        "sulfate",
        "tartrate",
        "trihydrate",
    }
    FORM_STOPWORDS = {
        "capsule",
        "capsules",
        "tablet",
        "tablets",
        "oral",
        "solution",
        "suspension",
        "injection",
        "intravenous",
        "intramuscular",
        "subcutaneous",
        "topical",
        "cream",
        "ointment",
        "patch",
        "spray",
        "gel",
        "drops",
        "ophthalmic",
        "nasal",
        "powder",
        "elixir",
        "syrup",
        "kit",
        "pack",
        "dose",
        "doses",
        "film",
        "coated",
        "delayed",
        "extended",
        "release",
        "chewable",
        "lozenge",
        "suppository",
        "for",
        "use",
        "intrathecal",
        "intralesional",
        "implant",
        "inhalation",
        "sustained",
        "concentrate",
        "reconstituted",
        "resin",
        "prefilled",
        "prefill",
        "pre-filled",
        "auto",
        "injector",
        "autoinjector",
        "pen",
        "device",
        "syringe",
        "syringes",
    }
    UNIT_STOPWORDS = {
        "mg",
        "mcg",
        "g",
        "kg",
        "ml",
        "l",
        "iu",
        "unit",
        "units",
        "percent",
        "%",
        "meq",
        "mmol",
    }
    NAME_STOPWORDS = (
        SALT_STOPWORDS
        | FORM_STOPWORDS
        | UNIT_STOPWORDS
        | {
            "solution",
            "suspensions",
        }
    )
    TERM_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
    BRACKET_PATTERN = re.compile(r"\[([^\]]+)\]")

    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = True
        self.cache: dict[str, dict[str, RxNormCandidate]] = {}
        self.synonym_cache: dict[str, list[str]] = {}

    # -------------------------------------------------------------------------
    def fetch_drug_terms(self, raw_name: str) -> list[str]:
        payload = self.request(raw_name)
        collected: dict[str, str] = {}

        def store(term: str) -> None:
            normalized = self.standardize_term(term)
            if not normalized:
                return
            key = normalized.casefold()
            if key not in collected:
                collected[key] = normalized

        if payload is not None:
            drug_group = payload.get("drugGroup")
            if isinstance(drug_group, dict):
                groups = drug_group.get("conceptGroup")
                if isinstance(groups, list):
                    for group in groups:
                        if not isinstance(group, dict):
                            continue
                        props = group.get("conceptProperties")
                        if not isinstance(props, list):
                            continue
                        for prop in props:
                            if not isinstance(prop, dict):
                                continue
                            for value in self.gather_property_values(prop):
                                for term in self.extract_core_names(value):
                                    store(term)

        for term in self.extract_core_names(raw_name):
            store(term)
        store(raw_name)
        return sorted(collected.values(), key=str.casefold)

    # -------------------------------------------------------------------------
    def fetch_rxcui_synonyms(self, rxcui: str) -> list[str]:
        identifier = str(rxcui).strip()
        if not identifier:
            return []
        cached = self.synonym_cache.get(identifier)
        if cached is not None:
            return cached
        url = self.RXCUI_PROPERTY_URL.format(rxcui=identifier)
        params = {"propName": "RxNorm Synonym"}
        payload: dict[str, Any] | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = httpx.get(url, params=params, timeout=self.TIMEOUT)
            except httpx.RequestError as exc:
                if attempt + 1 == self.MAX_RETRIES:
                    logger.debug(
                        "RxNorm rxcui request failed for '%s': %s",
                        identifier,
                        exc,
                    )
                    self.synonym_cache[identifier] = []
                    return []
                time.sleep(
                    self.BACKOFF_TIME[min(attempt, len(self.BACKOFF_TIME) - 1)]
                )
                continue
            if response.status_code in self.RETRY_STATUS:
                if attempt + 1 == self.MAX_RETRIES:
                    logger.debug(
                        "RxNorm property service returned %s for '%s'",
                        response.status_code,
                        identifier,
                    )
                    self.synonym_cache[identifier] = []
                    return []
                time.sleep(
                    self.BACKOFF_TIME[min(attempt, len(self.BACKOFF_TIME) - 1)]
                )
                continue
            if response.status_code >= 400:
                logger.debug(
                    "RxNorm property service returned %s for '%s'",
                    response.status_code,
                    identifier,
                )
                self.synonym_cache[identifier] = []
                return []
            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                logger.debug(
                    "RxNorm property JSON decode failed for '%s': %s",
                    identifier,
                    exc,
                )
                self.synonym_cache[identifier] = []
                return []
            break
        if payload is None:
            self.synonym_cache[identifier] = []
            return []
        collected: dict[str, str] = {}
        group = payload.get("propConceptGroup")
        concepts: list[dict[str, Any]] = []
        if isinstance(group, dict):
            raw_concepts = group.get("propConcept")
            if isinstance(raw_concepts, list):
                concepts.extend(
                    concept for concept in raw_concepts if isinstance(concept, dict)
                )
            elif isinstance(raw_concepts, dict):
                concepts.append(raw_concepts)
        for concept in concepts:
            prop_name = concept.get("propName")
            if isinstance(prop_name, str) and "synonym" not in prop_name.lower():
                continue
            value = concept.get("propValue")
            if not isinstance(value, str):
                continue
            normalized_value = self.standardize_term(value)
            if normalized_value:
                collected[normalized_value.casefold()] = normalized_value
            for fragment in self.extract_core_names(value):
                refined = self.standardize_term(fragment)
                if refined:
                    collected[refined.casefold()] = refined
        synonyms = sorted(collected.values(), key=str.casefold)
        self.synonym_cache[identifier] = synonyms
        return synonyms

    # -------------------------------------------------------------------------
    def gather_property_values(self, prop: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("name", "synonym", "prescribableName", "psn"):
            value = prop.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value)
        return values

    # -------------------------------------------------------------------------
    def extract_core_names(self, raw_value: str | None) -> set[str]:
        if not isinstance(raw_value, str):
            return set()
        text = unicodedata.normalize("NFKC", raw_value)
        segments: list[str] = []
        segments.extend(self.BRACKET_PATTERN.findall(text))
        text = self.BRACKET_PATTERN.sub(" ", text)
        segments.extend(part.strip() for part in text.split(","))
        extracted: set[str] = set()

        for segment in segments:
            cleaned = segment.strip()
            if not cleaned:
                continue
            tokens: list[str] = []

            def flush() -> None:
                if not tokens:
                    return
                term = " ".join(tokens).strip()
                if term:
                    extracted.add(term)
                tokens.clear()

            for match in self.TERM_PATTERN.finditer(cleaned):
                token = match.group(0)
                if not any(char.isalpha() for char in token):
                    flush()
                    continue
                normalized = token.lower().strip("-'")
                base = re.sub(r"[^a-z0-9]", "", normalized)
                if not base:
                    flush()
                    continue
                if (
                    base in self.NAME_STOPWORDS
                    or base.rstrip("s") in self.NAME_STOPWORDS
                ):
                    flush()
                    continue
                tokens.append(token.strip("-'").strip())
            flush()

        standardized = {
            self.standardize_term(term) for term in extracted if term.strip()
        }
        return {term for term in standardized if term}

    # -------------------------------------------------------------------------
    def standardize_term(self, term: str) -> str:
        normalized = unicodedata.normalize("NFKC", term).strip()
        if not normalized:
            return ""
        normalized = re.sub(r"\s+", " ", normalized)
        words = normalized.split()
        formatted: list[str] = []
        for word in words:
            simplified = re.sub(r"[-'\s]", "", word)
            if simplified.isupper() and len(simplified) > 1:
                formatted.append(word)
            elif simplified.islower():
                formatted.append(word.capitalize())
            else:
                formatted.append(word)
        return " ".join(formatted)

    # -------------------------------------------------------------------------
    def expand(self, raw_name: str) -> dict[str, str]:
        normalized_key = self.normalize_value(raw_name)
        if not normalized_key:
            return {}
        if not self.enabled:
            return {normalized_key: "original"}
        cached = self.cache.get(normalized_key)
        if cached is not None:
            return {key: info.kind for key, info in cached.items()}
        candidates = self.collect_candidates(raw_name)
        if normalized_key not in candidates:
            candidates[normalized_key] = RxNormCandidate(
                value=normalized_key,
                kind="original",
            )
        else:
            candidates[normalized_key].kind = "original"
        if len(candidates) == 1:
            logger.debug("RxNorm expansion returned no alternates for '%s'", raw_name)
        self.cache[normalized_key] = candidates
        return {key: info.kind for key, info in candidates.items()}

    # -------------------------------------------------------------------------
    def get_candidate_kind(self, original: str, candidate: str) -> str:
        normalized_key = self.normalize_value(original)
        if not normalized_key:
            return "unknown"
        cached = self.cache.get(normalized_key)
        if not cached:
            return "unknown"
        info = cached.get(candidate)
        if info is None:
            return "unknown"
        return info.kind

    # -------------------------------------------------------------------------
    def collect_candidates(self, raw_name: str) -> dict[str, RxNormCandidate]:
        payload = self.request(raw_name)
        if payload is None:
            return {}
        drug_group = payload.get("drugGroup")
        if not isinstance(drug_group, dict):
            return {}
        groups = drug_group.get("conceptGroup")
        if not isinstance(groups, list):
            return {}
        collected: dict[str, RxNormCandidate] = {}
        for group in groups:
            if not isinstance(group, dict):
                continue
            props = group.get("conceptProperties")
            if not isinstance(props, list):
                continue
            for prop in props:
                if not isinstance(prop, dict):
                    continue
                suppress = prop.get("suppress")
                if suppress and suppress != "N":
                    continue
                for raw_value, source in self.extract_property_values(prop):
                    normalized_value = self.normalize_value(raw_value)
                    if not normalized_value:
                        continue
                    kind = self.classify_kind(raw_value, source)
                    self.store_candidate(collected, normalized_value, kind)
                    ingredient_variants = self.derive_ingredients(raw_value)
                    if ingredient_variants:
                        if len(ingredient_variants) > 1:
                            combo = " / ".join(ingredient_variants)
                            self.store_candidate(collected, combo, "ingredient_combo")
                        for variant in ingredient_variants:
                            self.store_candidate(collected, variant, "ingredient")
        return collected

    # -------------------------------------------------------------------------
    def request(self, raw_name: str) -> dict[str, Any] | None:
        params = {"name": raw_name, "expand": "psn"}
        for attempt in range(self.MAX_RETRIES):
            try:
                response = httpx.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.TIMEOUT,
                )
            except httpx.RequestError as exc:
                if attempt + 1 == self.MAX_RETRIES:
                    logger.debug("RxNorm request failed for '%s': %s", raw_name, exc)
                    return None
                time.sleep(
                    self.BACKOFF_TIME[min(attempt, len(self.BACKOFF_TIME) - 1)]
                )
                continue
            if response.status_code in self.RETRY_STATUS:
                if attempt + 1 == self.MAX_RETRIES:
                    logger.debug(
                        "RxNorm service returned %s for '%s'",
                        response.status_code,
                        raw_name,
                    )
                    return None
                time.sleep(
                    self.BACKOFF_TIME[min(attempt, len(self.BACKOFF_TIME) - 1)]
                )
                continue
            if response.status_code >= 400:
                logger.debug(
                    "RxNorm service returned %s for '%s'",
                    response.status_code,
                    raw_name,
                )
                return None
            try:
                return response.json()
            except json.JSONDecodeError as exc:
                logger.debug("RxNorm JSON decode failed for '%s': %s", raw_name, exc)
                return None
        return None

    # -------------------------------------------------------------------------
    def extract_property_values(self, prop: dict[str, Any]) -> list[tuple[str, str]]:
        values: list[tuple[str, str]] = []
        for key in ("name", "synonym", "prescribableName", "psn"):
            value = prop.get(key)
            if not isinstance(value, str):
                continue
            value = value.strip()
            if not value:
                continue
            source = "psn" if key in {"prescribableName", "psn"} else key
            values.append((value, source))
            for brand in self.extract_brands(value):
                values.append((brand, "brand"))
        return values

    # -------------------------------------------------------------------------
    def classify_kind(self, raw_value: str, source: str) -> str:
        combo = self.normalize_combo(raw_value)
        if " / " in combo or "+" in raw_value or "," in raw_value:
            return "ingredient_combo"
        if source == "brand" or self.looks_like_brand(raw_value):
            return "brand"
        if source in {"prescribableName", "psn"}:
            return "psn"
        return "ingredient"

    # -------------------------------------------------------------------------
    def extract_brands(self, value: str) -> set[str]:
        brands: set[str] = set()
        for match in re.findall(r"\[([^\]]+)\]", value):
            cleaned = match.strip()
            if cleaned:
                brands.add(cleaned)
        prefix = value.split("[")[0].strip()
        if prefix:
            prefix_words = prefix.split()
            if prefix_words and not any(token.isdigit() for token in prefix_words):
                brands.add(prefix)
        return brands

    # -------------------------------------------------------------------------
    def derive_ingredients(self, raw_value: str) -> list[str]:
        stripped = re.sub(r"\[.*?\]", " ", raw_value)
        stripped = re.sub(r"\(.*?\)", " ", stripped)
        parts = re.split(r"\s*(?:/|\+|,)\s*", stripped)
        normalized_parts: list[str] = []
        for part in parts:
            normalized = self.normalize_single_ingredient(part)
            if normalized and normalized not in normalized_parts:
                normalized_parts.append(normalized)
        if not normalized_parts:
            single = self.normalize_single_ingredient(stripped)
            if single:
                normalized_parts.append(single)
        return normalized_parts

    # -------------------------------------------------------------------------
    def store_candidate(
        self,
        collected: dict[str, RxNormCandidate],
        normalized_value: str,
        kind: str,
    ) -> None:
        if not normalized_value:
            return
        existing = collected.get(normalized_value)
        if existing is None:
            collected[normalized_value] = RxNormCandidate(
                value=normalized_value,
                kind=kind,
            )
            return
        if existing.kind == "unknown" and kind != "unknown":
            existing.kind = kind
            return
        if existing.kind == "brand" and kind in {"ingredient", "ingredient_combo"}:
            existing.kind = kind

    # -------------------------------------------------------------------------
    def normalize_value(self, value: str) -> str:
        normalized = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        normalized = normalized.lower()
        normalized = normalized.replace("/", " / ")
        normalized = re.sub(r"[^a-z0-9/\s]", " ", normalized)
        normalized = re.sub(r"\s+/\s+", " / ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    # -------------------------------------------------------------------------
    def normalize_single_ingredient(self, value: str) -> str:
        stripped = re.sub(r"\s+", " ", value).strip()
        if not stripped:
            return ""
        tokens = [token for token in stripped.split(" ") if token]
        if not tokens:
            return ""
        kept: list[str] = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower.isdigit():
                continue
            if token_lower in self.UNIT_STOPWORDS:
                continue
            if token_lower in self.FORM_STOPWORDS:
                break
            if token_lower in self.SALT_STOPWORDS:
                break
            kept.append(token_lower)
        normalized = " ".join(kept)
        return self.normalize_value(normalized)

    # -------------------------------------------------------------------------
    def normalize_combo(self, value: str) -> str:
        combo = value.replace("/", " / ")
        combo = re.sub(r"\s+/\s+", " / ", combo)
        combo = re.sub(r"\s+", " ", combo)
        return combo.strip()

    # -------------------------------------------------------------------------
    def looks_like_brand(self, value: str) -> bool:
        stripped = value.strip()
        if not stripped:
            return False
        if stripped.startswith("[") and stripped.endswith("]"):
            return True
        if any(char.isdigit() for char in stripped):
            return False
        words = stripped.split()
        if len(words) == 1:
            return True
        if len(words) == 2 and all(word[0].isupper() for word in words if word):
            return True
        return False

    # -------------------------------------------------------------------------
    def is_bracketed(self, value: str) -> bool:
        stripped = value.strip()
        return stripped.startswith("[") and stripped.endswith("]")


###############################################################################
class RxNavDrugCatalogBuilder:
    TERMS_URL = "https://rxnav.nlm.nih.gov/REST/RxTerms/allconcepts.json"
    CHUNK_SIZE = 131_072
    MAX_RETRIES = 3
    RETRY_STATUS = {429, 500, 502, 503, 504}
    TIMEOUT = 30.0
    BACKOFF_TIME = (0.8, 1.6, 3.2)
    TABLE_NAME = "DRUGS_CATALOG"
    BATCH_SIZE = 200
    SYNONYM_WORKERS = 12
    TOKEN_SPLIT_PATTERN = re.compile(r"[^A-Za-z0-9']+")
    SINGLE_TOKEN_DIGIT_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")
    SHORT_TOKEN_EXCEPTIONS = {"id"}

    def __init__(self, rx_client: RxNavClient | None = None) -> None:
        combined: set[str] = set()
        for attr in ("SALT_STOPWORDS", "FORM_STOPWORDS", "UNIT_STOPWORDS"):
            values = getattr(RxNavClient, attr, set())
            combined.update(word.lower() for word in values)
        combined.update({
            "sterile",
            "single",
            "multi",
            "dose",
            "kit",
            "pack",
            "per",
            "each",
            "day",
            "days",
            "hour",
            "hours",
        })
        synonym_stopwords = {word.casefold() for word in RXNAV_SYNONYM_STOPWORDS}
        combined.update(synonym_stopwords)
        self.stopwords = combined
        self.synonym_stopwords = synonym_stopwords
        self.brand_pattern = re.compile(r"\[([^\]]+)\]")
        self.rx_client = rx_client or RxNavClient()
        self.alias_cache: dict[str, list[str]] = {}
        self.rxcui_cache: dict[str, list[str]] = {}
        self.total_records: int | None = None
        self.last_logged_count = 0
        self.serializer = DataSerializer()

    # -------------------------------------------------------------------------
    def update_drug_catalog(self, *, total_records: int | None = None) -> dict[str, Any]:
        self.total_records = total_records
        self.last_logged_count = 0
        attempt = 0
        last_error: Exception | None = None
        while attempt < self.MAX_RETRIES:
            try:
                with httpx.stream("GET", self.TERMS_URL, timeout=self.TIMEOUT) as response:
                    if (
                        response.status_code in self.RETRY_STATUS
                        and attempt + 1 < self.MAX_RETRIES
                    ):
                        time.sleep(self.BACKOFF_TIME[min(attempt, len(self.BACKOFF_TIME) - 1)])
                        attempt += 1
                        continue
                    response.raise_for_status()
                    started = time.perf_counter()
                    result = self.persist_catalog(response.iter_bytes(self.CHUNK_SIZE))
                    elapsed = time.perf_counter() - started
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
    def persist_catalog(self, chunks: Iterator[bytes]) -> dict[str, Any]:
        count = 0
        batch: list[dict[str, Any]] = []
        for concept in self.stream_min_concepts(chunks):
            payload = self.sanitize_concept(concept)
            if payload is None:
                continue
            batch.append(payload)
            if len(batch) >= self.BATCH_SIZE:
                self.persist_batch(batch)
                count += len(batch)
                batch.clear()
                logger.info('Total records upserted into database: %d', count)
        if batch:
            self.persist_batch(batch)
            count += len(batch)
            logger.info('Total records upserted into database: %d', count)
            
        return {"table_name": self.TABLE_NAME, "count": count}

    # -------------------------------------------------------------------------
    def persist_batch(self, batch: list[dict[str, Any]]) -> None:
        frame = pd.DataFrame(batch)
        if frame.empty:
            return
        self.serializer.upsert_drugs_catalog_records(frame)

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

        fetch_tasks: dict[Any, tuple[str, str, str]] = {}
        if pending_alias_queries or pending_synonym_identifier:
            total_tasks = len(pending_alias_queries)
            if pending_synonym_identifier is not None:
                total_tasks += 1
            max_workers = min(self.SYNONYM_WORKERS, max(1, total_tasks))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for cache_key, stripped in pending_alias_queries.items():
                    future = executor.submit(self.rx_client.fetch_drug_terms, stripped)
                    fetch_tasks[future] = ("alias", cache_key, stripped)
                if pending_synonym_identifier is not None:
                    future = executor.submit(
                        self.rx_client.fetch_rxcui_synonyms,
                        pending_synonym_identifier,
                    )
                    fetch_tasks[future] = (
                        "synonym",
                        pending_synonym_identifier,
                        pending_synonym_identifier,
                    )
                for future in as_completed(fetch_tasks):
                    kind, cache_key, original = fetch_tasks[future]
                    try:
                        fetched_terms = future.result()
                    except Exception as exc:  # noqa: BLE001
                        if kind == "alias":
                            logger.warning(
                                "Failed to fetch RxNav aliases for '%s': %s",
                                original,
                                exc,
                            )
                        else:
                            logger.warning(
                                "Failed to fetch RxNav aliases for rxcui '%s': %s",
                                original,
                                exc,
                            )
                        filtered_terms: list[str] = []
                    else:
                        filtered_terms = [
                            term for term in fetched_terms if isinstance(term, str)
                        ]
                    if kind == "alias":
                        self.alias_cache[cache_key] = filtered_terms
                    else:
                        self.rxcui_cache[cache_key] = filtered_terms
                    for term in filtered_terms:
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
        tokens = [
            token
            for token in self.TOKEN_SPLIT_PATTERN.split(cleaned)
            if token
        ]
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
        normalized = re.sub(r"\(.*?\)", " ", normalized)
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
