from __future__ import annotations

import codecs
import json
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from DILIGENT.app.logger import logger

__all__ = ["RxNavClient", "RxNavDrugCatalogBuilder"]


###############################################################################
@dataclass(slots=True)
class RxNormCandidate:
    value: str
    kind: str


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
    BACKOFF_SECONDS = (0.8, 1.6, 3.2)

    def __init__(self) -> None:
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
        })
        self.stopwords = combined
        self.brand_pattern = re.compile(r"\[([^\]]+)\]")

    # -------------------------------------------------------------------------
    def build_catalog(self, destination: str) -> dict[str, Any]:
        directory = os.path.dirname(destination) or "."
        os.makedirs(directory, exist_ok=True)

        attempt = 0
        last_error: Exception | None = None
        while attempt < self.MAX_RETRIES:
            try:
                with httpx.stream("GET", self.TERMS_URL, timeout=self.TIMEOUT) as response:
                    if (
                        response.status_code in self.RETRY_STATUS
                        and attempt + 1 < self.MAX_RETRIES
                    ):
                        time.sleep(self.BACKOFF_SECONDS[min(attempt, len(self.BACKOFF_SECONDS) - 1)])
                        attempt += 1
                        continue
                    response.raise_for_status()
                    return self.write_catalog(
                        response.iter_bytes(self.CHUNK_SIZE), destination, directory
                    )
            except httpx.HTTPStatusError as exc:  # pragma: no cover - network dependent
                last_error = exc
                if (
                    exc.response is not None
                    and exc.response.status_code in self.RETRY_STATUS
                    and attempt + 1 < self.MAX_RETRIES
                ):
                    time.sleep(
                        self.BACKOFF_SECONDS[min(attempt, len(self.BACKOFF_SECONDS) - 1)]
                    )
                    attempt += 1
                    continue
                break
            except httpx.RequestError as exc:  # pragma: no cover - network dependent
                last_error = exc
                if attempt + 1 < self.MAX_RETRIES:
                    time.sleep(
                        self.BACKOFF_SECONDS[min(attempt, len(self.BACKOFF_SECONDS) - 1)]
                    )
                    attempt += 1
                    continue
                break
        if last_error is not None:
            raise RuntimeError("Failed to download RxNav drug catalog") from last_error
        raise RuntimeError("Failed to download RxNav drug catalog")

    # -------------------------------------------------------------------------
    def write_catalog(
        self,
        chunks: Iterator[bytes],
        destination: str,
        directory: str,
    ) -> dict[str, Any]:
        count = 0
        temp_path = ""
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=directory,
        ) as handle:
            temp_path = handle.name
            try:
                for concept in self.stream_min_concepts(chunks):
                    payload = self.sanitize_concept(concept)
                    if payload is None:
                        continue
                    json.dump(payload, handle, ensure_ascii=False)
                    handle.write("\n")
                    count += 1
            except Exception:  # noqa: BLE001
                handle.flush()
                handle.close()
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                raise
        os.replace(temp_path, destination)
        return {"file_path": destination, "count": count}

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
        sanitized_name = self.sanitize_name(full_name)
        brands = self.extract_brands(full_name)
        payload = {
            "rxcui": rxcui_str,
            "term_type": term_type.strip() if isinstance(term_type, str) else "",
            "raw_name": full_name.strip(),
            "name": sanitized_name or None,
            "brand_names": brands,
        }
        if payload["name"] is None and not payload["brand_names"]:
            return None
        return payload

    # -------------------------------------------------------------------------
    def sanitize_name(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        normalized = self.brand_pattern.sub(" ", normalized)
        normalized = re.sub(r"\(.*?\)", " ", normalized)
        normalized = normalized.replace("/", " ")
        normalized = re.sub(r"[^A-Za-z\s-']", " ", normalized)
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
