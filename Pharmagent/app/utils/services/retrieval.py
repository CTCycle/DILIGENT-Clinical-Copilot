from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

import httpx

from Pharmagent.app.logger import logger


###############################################################################
@dataclass(slots=True)
class RxNormCandidate:
    value: str
    kind: str


# -----------------------------------------------------------------------------
def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


RXNORM_EXPANSION_ENABLED = _is_truthy(os.getenv("PHARMAGENT_RXNORM_EXPANSION", "1"))


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

    # -------------------------------------------------------------------------
    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = RXNORM_EXPANSION_ENABLED if enabled is None else enabled
        self.cache: dict[str, dict[str, RxNormCandidate]] = {}

    # -------------------------------------------------------------------------
    def fetch_drug_terms(self, raw_name: str) -> tuple[list[str], list[str]]:
        payload = self._request(raw_name)
        if payload is None:
            return [], []
        drug_group = payload.get("drugGroup")
        if not isinstance(drug_group, dict):
            return [], []
        groups = drug_group.get("conceptGroup")
        if not isinstance(groups, list):
            return [], []
        collected_names: set[str] = set()
        collected_synonyms: set[str] = set()
        for group in groups:
            if not isinstance(group, dict):
                continue
            props = group.get("conceptProperties")
            if not isinstance(props, list):
                continue
            for prop in props:
                if not isinstance(prop, dict):
                    continue
                name = prop.get("name")
                if isinstance(name, str) and name.strip():
                    collected_names.add(name.strip())
                for key in ("synonym", "prescribableName", "psn"):
                    value = prop.get(key)
                    if isinstance(value, str) and value.strip():
                        collected_synonyms.add(value.strip())
        if raw_name.strip():
            collected_names.add(raw_name.strip())
        return sorted(collected_names), sorted(collected_synonyms)

    # -------------------------------------------------------------------------
    def expand(self, raw_name: str) -> dict[str, str]:
        normalized_key = self._normalize_value(raw_name)
        if not normalized_key:
            return {}
        if not self.enabled:
            return {normalized_key: "original"}
        cached = self.cache.get(normalized_key)
        if cached is not None:
            return {key: info.kind for key, info in cached.items()}
        candidates = self._collect_candidates(raw_name)
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
        normalized_key = self._normalize_value(original)
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
    def _collect_candidates(self, raw_name: str) -> dict[str, RxNormCandidate]:
        payload = self._request(raw_name)
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
                for raw_value, source in self._extract_property_values(prop):
                    normalized_value = self._normalize_value(raw_value)
                    if not normalized_value:
                        continue
                    kind = self._classify_kind(raw_value, source)
                    self._store_candidate(collected, normalized_value, kind)
                    ingredient_variants = self._derive_ingredients(raw_value)
                    if ingredient_variants:
                        if len(ingredient_variants) > 1:
                            combo = " / ".join(ingredient_variants)
                            self._store_candidate(collected, combo, "ingredient_combo")
                        for variant in ingredient_variants:
                            self._store_candidate(collected, variant, "ingredient")
        return collected

    # -------------------------------------------------------------------------
    def _request(self, raw_name: str) -> dict[str, Any] | None:
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
    def _extract_property_values(self, prop: dict[str, Any]) -> list[tuple[str, str]]:
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
            for brand in self._extract_brands(value):
                values.append((brand, "brand"))
        return values

    # -------------------------------------------------------------------------
    def _classify_kind(self, raw_value: str, source: str) -> str:
        combo = self._normalize_combo(raw_value)
        if " / " in combo or "+" in raw_value or "," in raw_value:
            return "ingredient_combo"
        if source == "brand" or self._looks_like_brand(raw_value):
            return "brand"
        if source in {"prescribableName", "psn"}:
            return "psn"
        return "ingredient"

    # -------------------------------------------------------------------------
    def _extract_brands(self, value: str) -> set[str]:
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
    def _derive_ingredients(self, raw_value: str) -> list[str]:
        stripped = re.sub(r"\[.*?\]", " ", raw_value)
        stripped = re.sub(r"\(.*?\)", " ", stripped)
        parts = re.split(r"\s*(?:/|\+|,)\s*", stripped)
        normalized_parts: list[str] = []
        for part in parts:
            normalized = self._normalize_single_ingredient(part)
            if normalized and normalized not in normalized_parts:
                normalized_parts.append(normalized)
        if not normalized_parts:
            single = self._normalize_single_ingredient(stripped)
            if single:
                normalized_parts.append(single)
        return normalized_parts

    # -------------------------------------------------------------------------
    def _store_candidate(
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
    def _normalize_value(self, value: str) -> str:
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
    def _normalize_single_ingredient(self, value: str) -> str:
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
        return self._normalize_value(normalized)

    # -------------------------------------------------------------------------
    def _normalize_combo(self, value: str) -> str:
        combo = value.replace("/", " / ")
        combo = re.sub(r"\s+/\s+", " / ", combo)
        combo = re.sub(r"\s+", " ", combo)
        return combo.strip()

    # -------------------------------------------------------------------------
    def _looks_like_brand(self, value: str) -> bool:
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
    def _is_bracketed(self, value: str) -> bool:
        stripped = value.strip()
        return stripped.startswith("[") and stripped.endswith("]")
