from __future__ import annotations

import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterator

import httpx
import pandas as pd

from DILIGENT.app.logger import logger
from DILIGENT.app.utils.repository.serializer import (
    DRUGS_CATALOG_COLUMNS,
    DataSerializer,
)

__all__ = ["RxNavClient", "DrugsCatalogUpdater"]


###############################################################################
@dataclass(slots=True)
class RxNormCandidate:
    value: str
    kind: str
    display: str


# -----------------------------------------------------------------------------
def is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


###############################################################################
class RxNavClient:
    BASE_URL = "https://rxnav.nlm.nih.gov/REST/drugs.json"
    MAX_RETRIES = 3
    BACKOFF_SECONDS = (0.6, 1.2, 2.4)
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
        candidates = self.get_candidates(raw_name)
        return {key: info.kind for key, info in candidates.items()}

    # -------------------------------------------------------------------------
    def get_candidates(self, raw_name: str) -> dict[str, RxNormCandidate]:
        normalized_key = self.normalize_value(raw_name)
        if not normalized_key:
            return {}
        if not self.enabled:
            display = self.standardize_term(raw_name) or raw_name
            return {
                normalized_key: RxNormCandidate(
                    value=normalized_key,
                    kind="original",
                    display=display,
                )
            }
        cached = self.cache.get(normalized_key)
        if cached is not None:
            return cached
        candidates = self.collect_candidates(raw_name)
        if normalized_key not in candidates:
            candidates[normalized_key] = RxNormCandidate(
                value=normalized_key,
                kind="original",
                display=self.standardize_term(raw_name) or raw_name,
            )
        else:
            info = candidates[normalized_key]
            info.kind = "original"
            if not info.display:
                info.display = self.standardize_term(raw_name) or raw_name
        if len(candidates) == 1:
            logger.debug("RxNorm expansion returned no alternates for '%s'", raw_name)
        self.cache[normalized_key] = candidates
        return candidates

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
                    display_value = self.standardize_term(raw_value) or raw_value
                    self.store_candidate(
                        collected,
                        normalized_value,
                        kind,
                        display_value,
                    )
                    ingredient_variants = self.derive_ingredients(raw_value)
                    if ingredient_variants:
                        if len(ingredient_variants) > 1:
                            combo = " / ".join(ingredient_variants)
                            self.store_candidate(
                                collected,
                                combo,
                                "ingredient_combo",
                                self.standardize_term(combo),
                            )
                        for variant in ingredient_variants:
                            self.store_candidate(
                                collected,
                                variant,
                                "ingredient",
                                self.standardize_term(variant),
                            )
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
                    self.BACKOFF_SECONDS[min(attempt, len(self.BACKOFF_SECONDS) - 1)]
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
                    self.BACKOFF_SECONDS[min(attempt, len(self.BACKOFF_SECONDS) - 1)]
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
        display: str | None,
    ) -> None:
        if not normalized_value:
            return
        display_value = display or normalized_value
        existing = collected.get(normalized_value)
        if existing is None:
            collected[normalized_value] = RxNormCandidate(
                value=normalized_value,
                kind=kind,
                display=self.standardize_term(display_value) or display_value,
            )
            return
        if not existing.display and display_value:
            existing.display = self.standardize_term(display_value) or display_value
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
class DrugsCatalogUpdater:
    RXTERMS_URL = "https://rxnav.nlm.nih.gov/REST/RxTerms/allconcepts.json"
    STREAM_CHUNK_SIZE = 131_072
    BATCH_SIZE = 400
    RXNAV_MAX_WORKERS = 8

    def __init__(
        self,
        *,
        rx_client: RxNavClient | None = None,
        serializer: DataSerializer | None = None,
    ) -> None:
        self.rx_client = rx_client or RxNavClient()
        self.serializer = serializer or DataSerializer()
        self.alias_cache: dict[str, dict[str, RxNormCandidate]] = {}

    # -------------------------------------------------------------------------
    def update_catalog(self) -> dict[str, Any]:
        logger.info("Refreshing RxNav drugs catalog")
        frame = self.build_catalog_frame()
        if frame.empty:
            logger.warning("RxNav catalog returned no entries; clearing table")
        self.serializer.save_drugs_catalog(frame)
        return {"records": int(frame.shape[0])}

    # -------------------------------------------------------------------------
    def build_catalog_frame(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        seen_rxcui: set[str] = set()
        for batch in self.iter_concept_batches(self.BATCH_SIZE):
            enriched = self.enrich_batch(batch)
            for record in enriched:
                rxcui = record.get("rxcui")
                if not rxcui or rxcui in seen_rxcui:
                    continue
                seen_rxcui.add(rxcui)
                records.append(record)
        if not records:
            return pd.DataFrame(columns=DRUGS_CATALOG_COLUMNS)
        frame = pd.DataFrame(records)
        return frame.reindex(columns=DRUGS_CATALOG_COLUMNS)

    # -------------------------------------------------------------------------
    def iter_concept_batches(
        self, batch_size: int
    ) -> Iterator[list[dict[str, Any]]]:
        batch: list[dict[str, Any]] = []
        for concept in self.iter_concepts():
            batch.append(concept)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # -------------------------------------------------------------------------
    def iter_concepts(self) -> Iterator[dict[str, Any]]:
        for raw in self.stream_concepts(self.STREAM_CHUNK_SIZE):
            normalized = self.normalize_concept(raw)
            if normalized is not None:
                yield normalized

    # -------------------------------------------------------------------------
    def stream_concepts(self, chunk_size: int) -> Iterator[dict[str, Any]]:
        decoder = json.JSONDecoder()
        buffer = ""
        key = '"conceptProperties"'
        started = False
        max_prefix = len(key)
        timeout = getattr(self.rx_client, "TIMEOUT", 10.0) or 10.0
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            with client.stream("GET", self.RXTERMS_URL) as response:
                response.raise_for_status()
                for chunk in response.iter_text(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    buffer += chunk
                    while True:
                        if not started:
                            key_index = buffer.find(key)
                            if key_index == -1:
                                buffer = buffer[-max_prefix:]
                                break
                            bracket_index = buffer.find("[", key_index)
                            if bracket_index == -1:
                                buffer = buffer[key_index:]
                                break
                            buffer = buffer[bracket_index + 1 :]
                            started = True
                        buffer = buffer.lstrip()
                        if not buffer:
                            break
                        if buffer[0] == "]":
                            started = False
                            buffer = buffer[1:]
                            break
                        try:
                            item, offset = decoder.raw_decode(buffer)
                        except json.JSONDecodeError:
                            break
                        if isinstance(item, dict):
                            yield item
                        buffer = buffer[offset:]
                        buffer = buffer.lstrip()
                        if buffer.startswith(","):
                            buffer = buffer[1:]

    # -------------------------------------------------------------------------
    def normalize_concept(self, concept: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(concept, dict):
            return None
        rxcui = str(concept.get("rxcui") or "").strip()
        full_name = str(concept.get("fullName") or "").strip()
        if not rxcui or not full_name:
            return None
        term_type = str(concept.get("termType") or "").strip()
        alias_candidates: list[str] = []
        for key, value in concept.items():
            if not isinstance(value, str):
                continue
            if "name" not in key.lower():
                continue
            cleaned = value.strip()
            if cleaned:
                alias_candidates.append(cleaned)
        aliases = list(dict.fromkeys(alias_candidates))
        if full_name and full_name not in aliases:
            aliases.insert(0, full_name)
        return {
            "rxcui": rxcui,
            "full_name": full_name,
            "term_type": term_type,
            "aliases": aliases,
        }

    # -------------------------------------------------------------------------
    def collect_aliases(self, concept: dict[str, Any]) -> list[str]:
        aliases: list[str] = []
        seen: set[str] = set()
        for alias in concept.get("aliases", []):
            if not isinstance(alias, str):
                continue
            cleaned = alias.strip()
            if not cleaned:
                continue
            if cleaned.lower() == "not available":
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            aliases.append(cleaned)
        return aliases

    # -------------------------------------------------------------------------
    def enrich_batch(
        self, concepts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not concepts:
            return []
        alias_groups: list[tuple[dict[str, Any], list[str]]] = []
        unique_aliases: list[str] = []
        seen_aliases: set[str] = set()
        for concept in concepts:
            aliases = self.collect_aliases(concept)
            alias_groups.append((concept, aliases))
            for alias in aliases:
                if alias in seen_aliases:
                    continue
                seen_aliases.add(alias)
                unique_aliases.append(alias)

        cache: dict[str, dict[str, RxNormCandidate]] = {}
        pending: list[str] = []
        for alias in unique_aliases:
            cached = self.alias_cache.get(alias)
            if cached is not None:
                cache[alias] = cached
                continue
            pending.append(alias)

        max_workers = self.resolve_max_workers()
        per_request = float(getattr(self.rx_client, "TIMEOUT", 10.0) or 10.0)
        if pending:
            estimated = (len(pending) * per_request) / max_workers
            logger.info(
                "Preparing RxNav enrichment for %d unique lookup(s) across %d worker(s); "
                "worst-case duration %.1fs (%.1f min)",
                len(pending),
                max_workers,
                estimated,
                estimated / 60,
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.rx_client.get_candidates, alias): alias
                    for alias in pending
                }
                for future in as_completed(futures):
                    alias = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to enrich '%s': %s", alias, exc)
                        result = {}
                    cache[alias] = result
                    self.alias_cache[alias] = result

        records: list[dict[str, Any]] = []
        for concept, aliases in alias_groups:
            record = self.build_record(concept, aliases, cache)
            if record is not None:
                records.append(record)
        return records

    # -------------------------------------------------------------------------
    def build_record(
        self,
        concept: dict[str, Any],
        aliases: list[str],
        cache: dict[str, dict[str, RxNormCandidate]],
    ) -> dict[str, Any] | None:
        rxcui = str(concept.get("rxcui") or "").strip()
        full_name = str(concept.get("full_name") or "").strip()
        if not rxcui or not full_name:
            return None
        term_type = str(concept.get("term_type") or "").strip()
        ingredients: set[str] = set()
        brands: set[str] = set()
        synonyms: set[str] = set()
        for alias in aliases:
            formatted_alias = self.format_catalog_value(alias)
            if formatted_alias is not None:
                synonyms.add(formatted_alias)
            candidates = cache.get(alias, {})
            for candidate in candidates.values():
                display = candidate.display or candidate.value
                formatted = self.format_catalog_value(display)
                if formatted is None:
                    continue
                if candidate.kind in {"ingredient", "ingredient_combo"}:
                    ingredients.add(formatted)
                elif candidate.kind == "brand":
                    brands.add(formatted)
                else:
                    synonyms.add(formatted)
        for value in brands:
            synonyms.add(value)
        for value in ingredients:
            synonyms.add(value)
        record = {
            "rxcui": rxcui,
            "full_name": full_name,
            "term_type": term_type,
            "ingredient": self.serialize_values(ingredients),
            "brand_name": self.serialize_values(brands),
            "synonyms": self.serialize_values(synonyms),
        }
        return record

    # -------------------------------------------------------------------------
    def resolve_max_workers(self) -> int:
        raw = getattr(self, "RXNAV_MAX_WORKERS", 8)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 8
        if value < 1:
            return 1
        return value

    # -------------------------------------------------------------------------
    def format_catalog_value(self, value: str | None) -> str | None:
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if lowered in {"not available", "none", "na", "n/a"}:
            return None
        formatted = self.rx_client.standardize_term(cleaned) or cleaned
        normalized = formatted.strip()
        if not normalized:
            return None
        if len(normalized) < 2 and " " not in normalized:
            return None
        return normalized

    # -------------------------------------------------------------------------
    def serialize_values(self, values: set[str]) -> str:
        if not values:
            return json.dumps([], ensure_ascii=False)
        ordered = sorted(values, key=str.casefold)
        return json.dumps(ordered, ensure_ascii=False)
