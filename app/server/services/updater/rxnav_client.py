from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from common.utils.logger import logger
from configurations.startup import server_settings
from domain.rxnav import RxNormCandidate
from services.text.vocabulary import get_text_normalization_snapshot


###############################################################################
async def run_with_semaphore(
    semaphore: asyncio.Semaphore,
    task_factory: Callable[[], Awaitable[Any]],
):
    async with semaphore:
        return await task_factory()


###############################################################################
class RxNavClient:
    BASE_URL = "https://rxnav.nlm.nih.gov/REST/drugs.json"
    RXCUI_PROPERTY_URL = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/property.json"
    MAX_RETRIES = 3
    BACKOFF_TIME = (0.6, 1.2, 2.4)
    RETRY_STATUS = {429, 500, 502, 503, 504}
    TIMEOUT = 10.0
    TERM_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
    BRACKET_PATTERN = re.compile(r"\[([^\]]+)\]")

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        request_timeout: float | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.enabled = True
        external_settings = server_settings.runtime
        configured_timeout = (
            float(request_timeout)
            if request_timeout is not None
            else float(external_settings.rxnav_request_timeout)
        )
        configured_concurrency = (
            int(max_concurrency)
            if max_concurrency is not None
            else int(external_settings.rxnav_max_concurrency)
        )
        self.timeout = max(configured_timeout, 1.0)
        self.max_concurrency = max(configured_concurrency, 1)
        self.cache: dict[str, dict[str, RxNormCandidate]] = {}
        self.synonym_cache: dict[str, list[str]] = {}
        vocabulary = get_text_normalization_snapshot()
        self.salt_stopwords = set(vocabulary.rxnav_salt_stopwords)
        self.form_stopwords = set(vocabulary.rxnav_form_stopwords)
        self.unit_stopwords = set(vocabulary.rxnav_unit_stopwords)
        self.name_stopwords = set(vocabulary.rxnav_name_stopwords)

    # -------------------------------------------------------------------------
    def _build_limits(self) -> httpx.Limits:
        return httpx.Limits(
            max_connections=self.max_concurrency,
            max_keepalive_connections=self.max_concurrency,
        )

    # -------------------------------------------------------------------------
    async def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any] | None:
        owns_client = client is None
        session = client or httpx.AsyncClient(
            timeout=self.timeout,
            limits=self._build_limits(),
        )
        try:
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = await session.get(url, params=params)
                except httpx.RequestError as exc:
                    if attempt + 1 == self.MAX_RETRIES:
                        logger.debug("RxNorm request failed for '%s': %s", url, exc)
                        return None
                    await asyncio.sleep(
                        self.BACKOFF_TIME[min(attempt, len(self.BACKOFF_TIME) - 1)]
                    )
                    continue
                if response.status_code in self.RETRY_STATUS:
                    if attempt + 1 == self.MAX_RETRIES:
                        logger.debug(
                            "RxNorm service returned %s for '%s'",
                            response.status_code,
                            url,
                        )
                        return None
                    await asyncio.sleep(
                        self.BACKOFF_TIME[min(attempt, len(self.BACKOFF_TIME) - 1)]
                    )
                    continue
                if response.status_code >= 400:
                    logger.debug(
                        "RxNorm service returned %s for '%s'",
                        response.status_code,
                        url,
                    )
                    return None
                try:
                    return response.json()
                except json.JSONDecodeError as exc:
                    logger.debug("RxNorm JSON decode failed for '%s': %s", url, exc)
                    return None
        finally:
            if owns_client:
                await session.aclose()
        return None

    # -------------------------------------------------------------------------
    async def fetch_drug_terms_async(
        self, raw_name: str, *, client: httpx.AsyncClient | None = None
    ) -> list[str]:
        payload = await self._request_json(
            self.BASE_URL,
            params={"name": raw_name, "expand": "psn"},
            client=client,
        )
        collected: dict[str, str] = {}

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
                                    self.store_term(collected, term)

        for term in self.extract_core_names(raw_name):
            self.store_term(collected, term)
        self.store_term(collected, raw_name)
        return sorted(collected.values(), key=str.casefold)

    # -------------------------------------------------------------------------
    def fetch_drug_terms(self, raw_name: str) -> list[str]:
        return asyncio.run(self.fetch_drug_terms_async(raw_name))

    # -------------------------------------------------------------------------
    async def fetch_rxcui_synonyms_async(
        self, rxcui: str, *, client: httpx.AsyncClient | None = None
    ) -> list[str]:
        identifier = str(rxcui).strip()
        if not identifier:
            return []
        cached = self.synonym_cache.get(identifier)
        if cached is not None:
            return cached
        payload = await self._request_json(
            self.RXCUI_PROPERTY_URL.format(rxcui=identifier),
            params={"propName": "RxNorm Synonym"},
            client=client,
        )
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
    def fetch_rxcui_synonyms(self, rxcui: str) -> list[str]:
        return asyncio.run(self.fetch_rxcui_synonyms_async(rxcui))

    # -------------------------------------------------------------------------
    def gather_property_values(self, prop: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("name", "synonym", "prescribableName", "psn"):
            value = prop.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value)
        return values

    # -------------------------------------------------------------------------
    def store_term(self, collected: dict[str, str], term: str) -> None:
        normalized = self.standardize_term(term)
        if not normalized:
            return
        key = normalized.casefold()
        if key not in collected:
            collected[key] = normalized

    # -------------------------------------------------------------------------
    @staticmethod
    def flush_extracted_tokens(tokens: list[str], extracted: set[str]) -> None:
        if not tokens:
            return
        term = " ".join(tokens).strip()
        if term:
            extracted.add(term)
        tokens.clear()

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

            for match in self.TERM_PATTERN.finditer(cleaned):
                token = match.group(0)
                if not any(char.isalpha() for char in token):
                    self.flush_extracted_tokens(tokens, extracted)
                    continue
                normalized = token.lower().strip("-'")
                base = re.sub(r"[^a-z0-9]", "", normalized)
                if not base:
                    self.flush_extracted_tokens(tokens, extracted)
                    continue
                if (
                    base in self.name_stopwords
                    or base.rstrip("s") in self.name_stopwords
                ):
                    self.flush_extracted_tokens(tokens, extracted)
                    continue
                tokens.append(token.strip("-'").strip())
            self.flush_extracted_tokens(tokens, extracted)

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
        payload = asyncio.run(
            self._request_json(
                self.BASE_URL,
                params={"name": raw_name, "expand": "psn"},
            )
        )
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
        stripped = re.sub(r"\[[^\]]*\]", " ", raw_value)
        stripped = re.sub(r"\([^)]*\)", " ", stripped)
        parts = re.split(r"[/+,]", stripped)
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
            if token_lower in self.unit_stopwords:
                continue
            if token_lower in self.form_stopwords:
                break
            if token_lower in self.salt_stopwords:
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

