from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from xml.etree import ElementTree as ET

import httpx
from Pharmagent.app.api.models.prompts import (
    HEPATOTOXICITY_ANALYSIS_SYSTEM_PROMPT,
    HEPATOTOXICITY_ANALYSIS_USER_PROMPT,
)
from Pharmagent.app.api.models.providers import initialize_llm_client
from Pharmagent.app.api.schemas.clinical import (
    DrugHepatotoxicityAnalysis,
    DrugToxicityFindings,
    HepatotoxicityPatternScore,
    LiverToxMatchInfo,
    PatientData,
    PatientDrugToxicityBundle,
    PatientDrugs,
)
from Pharmagent.app.configurations import ClientRuntimeConfig
from Pharmagent.app.logger import logger


###############################################################################
@dataclass(slots=True)
class LiverToxMatch:
    nbk_id: str
    matched_name: str
    confidence: float
    reason: str


###############################################################################
@dataclass(slots=True)
class CandidateSummary:
    nbk_id: str
    title: str
    synonyms: set[str]


###############################################################################
@dataclass(slots=True)
class RxNormConcept:
    rxcui: str
    preferred_name: str | None
    synonyms: set[str]
    ingredients: set[str]
    tty: str | None


###############################################################################
@dataclass(slots=True)
class NameCandidate:
    origin: str
    name: str
    priority: int


###############################################################################
class HepatotoxicityPatternAnalyzer:
    # -----------------------------------------------------------------------------
    def analyze(self, payload: PatientData) -> HepatotoxicityPatternScore:
        alt_value = self._parse_marker_value(payload.alt)
        alt_max_value = self._parse_marker_value(payload.alt_max)
        alp_value = self._parse_marker_value(payload.alp)
        alp_max_value = self._parse_marker_value(payload.alp_max)

        alt_multiple = self._safe_ratio(alt_value, alt_max_value)
        alp_multiple = self._safe_ratio(alp_value, alp_max_value)

        r_score: float | None = None
        if alt_multiple is not None and alp_multiple not in (None, 0.0):
            r_score = alt_multiple / alp_multiple

        classification = "indeterminate"
        if r_score is not None:
            if r_score > 5:
                classification = "hepatocellular"
            elif r_score < 2:
                classification = "cholestatic"
            else:
                classification = "mixed"

        return HepatotoxicityPatternScore(
            alt_multiple=alt_multiple,
            alp_multiple=alp_multiple,
            r_score=r_score,
            classification=classification,
        )

    # -----------------------------------------------------------------------------
    def _parse_marker_value(self, raw: str | None) -> float | None:
        if raw is None:
            return None
        normalized = raw.replace(",", ".")
        match = re.search(r"[-+]?\d*\.?\d+", normalized)
        if not match:
            return None
        try:
            return float(match.group())
        except ValueError:
            return None

    # -----------------------------------------------------------------------------
    def _safe_ratio(self, value: float | None, reference: float | None) -> float | None:
        if value is None or reference is None:
            return None
        if reference == 0:
            return None
        return value / reference


###############################################################################
class DrugToxicityEssay:
    _match_cache: dict[str, LiverToxMatch] = {}
    _rxnorm_term_cache: dict[str, tuple[tuple[NameCandidate, ...], frozenset[str]]] = {}
    _rxnorm_concept_cache: dict[str, RxNormConcept] = {}

    # -----------------------------------------------------------------------------
    def __init__(self, drugs: PatientDrugs, *, timeout_s: float = 300.0) -> None:
        self.drugs = drugs
        self.timeout_s = float(timeout_s)
        self.client = initialize_llm_client(purpose="agent", timeout_s=self.timeout_s)
        self.model = ClientRuntimeConfig.get_agent_model()
        self.max_prompt_chars = 6000
        self.http_timeout = httpx.Timeout(30.0)
        self._search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self._summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        self._fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        self._rxnorm_base = "https://rxnav.nlm.nih.gov/REST"
        self._rxnorm_approx_url = f"{self._rxnorm_base}/approximateTerm.json"
        self._rxnorm_property_url = f"{self._rxnorm_base}/rxcui"
        self._rxnorm_headers = {"User-Agent": "PharmagentClinicalCopilot/1.0"}

    # -----------------------------------------------------------------------------
    async def run(self) -> PatientDrugToxicityBundle:
        results: list[DrugHepatotoxicityAnalysis] = []
        for entry in self.drugs.entries:
            drug_name = entry.name.strip()
            if not drug_name:
                continue
            analysis = await self._process_drug(drug_name)
            results.append(analysis)
        return PatientDrugToxicityBundle(entries=results)

    # -----------------------------------------------------------------------------
    async def _process_drug(self, drug_name: str) -> DrugHepatotoxicityAnalysis:
        source_text, match = await self._gather_livertox_text(drug_name)
        match_payload = self._match_to_schema(match)
        if not source_text:
            message = "No LiverTox monograph available for this drug."
            logger.warning("LiverTox data unavailable for drug '%s'", drug_name)
            return DrugHepatotoxicityAnalysis(
                drug_name=drug_name,
                source_text=None,
                analysis=None,
                error=message,
                livertox_match=match_payload,
            )

        prompt_text = self._prepare_prompt_text(source_text)
        user_prompt = HEPATOTOXICITY_ANALYSIS_USER_PROMPT.format(
            drug_name=drug_name,
            source_text=prompt_text,
        )

        try:
            findings = await self.client.llm_structured_call(
                model=self.model,
                system_prompt=HEPATOTOXICITY_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=DrugToxicityFindings,
                temperature=0.0,
                use_json_mode=True,
                max_repair_attempts=2,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "LLM hepatotoxicity analysis failed for '%s': %s", drug_name, exc
            )
            return DrugHepatotoxicityAnalysis(
                drug_name=drug_name,
                source_text=prompt_text,
                analysis=None,
                error=str(exc),
                livertox_match=match_payload,
            )

        return DrugHepatotoxicityAnalysis(
            drug_name=drug_name,
            source_text=prompt_text,
            analysis=findings,
            error=None,
            livertox_match=match_payload,
        )

    # -----------------------------------------------------------------------------
    async def _gather_livertox_text(
        self, drug_name: str
    ) -> tuple[str | None, LiverToxMatch | None]:
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            match = await self._search_livertox_id(client, drug_name)
            if not match:
                logger.info("No NBK identifier found for drug '%s'", drug_name)
                return None, None

            xml_payload = await self._fetch_livertox_entry(client, match.nbk_id)
            if not xml_payload:
                return None, match

        extracted = self._extract_livertox_sections(xml_payload)
        if not extracted:
            return None, match

        logger.info(
            "LiverTox entry resolved for '%s' as '%s' (%s, %.2f)",
            drug_name,
            match.matched_name,
            match.reason,
            match.confidence,
        )
        return extracted, match

    # -----------------------------------------------------------------------------
    async def _search_livertox_id(
        self,
        client: httpx.AsyncClient,
        drug_name: str,
    ) -> LiverToxMatch | None:
        normalized_name = self._normalize_for_lookup(drug_name)
        cached = self._match_cache.get(normalized_name)
        if cached:
            logger.info(
                "LiverTox cache hit for '%s': '%s' (%s, %.2f)",
                drug_name,
                cached.matched_name,
                cached.reason,
                cached.confidence,
            )
            return cached

        candidates: list[tuple[str, str]] = []
        ingredient_hints: set[str] = set()

        formatted_original = self._format_name_for_query(drug_name)
        if formatted_original:
            candidates.append(("direct", formatted_original))
            ids = await self._query_esearch(client, formatted_original, title_only=True)
            if ids:
                summaries = await self._fetch_candidate_summaries(client, [ids[0]])
                matched_name = summaries[0].title if summaries else formatted_original
                match = LiverToxMatch(
                    nbk_id=ids[0],
                    matched_name=matched_name or formatted_original,
                    confidence=1.0,
                    reason="direct_match",
                )
                self._cache_match(normalized_name, match)
                if matched_name:
                    normalized_matched = self._normalize_for_lookup(matched_name)
                    if normalized_matched:
                        self._match_cache.setdefault(normalized_matched, match)
                logger.info(
                    "LiverTox direct_match for '%s' -> '%s' (NBK%s)",
                    drug_name,
                    match.matched_name,
                    match.nbk_id,
                )
                return match

        rxnorm_candidates, rxnorm_hints = await self._resolve_candidate_names(
            client, drug_name
        )
        candidates.extend(rxnorm_candidates)
        ingredient_hints.update(rxnorm_hints)

        reason_map = {
            "rxnorm_ingredient": ("ingredient_match", 0.96),
            "rxnorm_brand": ("brand_resolved", 0.92),
            "rxnorm_synonym": ("synonym_match", 0.9),
            "rxnorm_name": ("rxnorm_name", 0.93),
        }

        for origin, candidate in rxnorm_candidates:
            direct_match = await self._query_esearch(client, candidate, title_only=True)
            if not direct_match:
                continue

            nbk_id = direct_match[0]
            summaries = await self._fetch_candidate_summaries(client, [nbk_id])
            matched_name = candidate
            if summaries:
                matched_name = summaries[0].title or matched_name

            reason, confidence = reason_map.get(origin, ("synonym_match", 0.9))
            match = LiverToxMatch(
                nbk_id=nbk_id,
                matched_name=matched_name,
                confidence=confidence,
                reason=reason,
            )
            self._cache_match(normalized_name, match)
            if summaries:
                normalized_matched = self._normalize_for_lookup(matched_name)
                if normalized_matched:
                    self._match_cache.setdefault(normalized_matched, match)
            logger.info(
                "LiverTox %s match for '%s' -> '%s' (NBK%s)",
                reason,
                drug_name,
                matched_name,
                nbk_id,
            )
            return match

        fallback = await self._fallback_livertox_lookup(
            client, drug_name, normalized_name, candidates, ingredient_hints
        )
        if fallback:
            self._cache_match(normalized_name, fallback)
            logger.info(
                "LiverTox %s fallback for '%s' -> '%s' (NBK%s)",
                fallback.reason,
                drug_name,
                fallback.matched_name,
                fallback.nbk_id,
            )
        return fallback

    # -----------------------------------------------------------------------------
    async def _fallback_livertox_lookup(
        self,
        client: httpx.AsyncClient,
        drug_name: str,
        normalized_name: str,
        candidates: list[tuple[str, str]],
        ingredient_hints: set[str],
    ) -> LiverToxMatch | None:
        search_terms = [candidate for _, candidate in candidates]
        if not search_terms:
            search_terms.append(self._format_name_for_query(drug_name))
        term = search_terms[0] or drug_name
        ids = await self._query_esearch(client, term, title_only=False)
        if not ids:
            return None

        summaries = await self._fetch_candidate_summaries(client, ids)
        best_match: LiverToxMatch | None = None
        best_score = 0.0
        for summary in summaries:
            score = self._token_set_ratio(normalized_name, summary.title)
            for synonym in summary.synonyms:
                score = max(score, self._token_set_ratio(normalized_name, synonym))

            normalized_title = self._normalize_for_lookup(summary.title)
            if normalized_title in ingredient_hints:
                score += 5.0

            if score > best_score:
                best_score = score
                best_match = LiverToxMatch(
                    nbk_id=summary.nbk_id,
                    matched_name=summary.title or term,
                    confidence=min(0.89, max(0.0, score / 100.0)),
                    reason="fuzzy_match",
                )

        if best_match and best_score >= 70.0:
            self._cache_match(normalized_name, best_match)
            return best_match

        first_summary = summaries[0] if summaries else None
        matched_name = term
        if first_summary and first_summary.title:
            matched_name = first_summary.title
        return LiverToxMatch(
            nbk_id=ids[0],
            matched_name=matched_name,
            confidence=0.40,
            reason="list_first",
        )

    # -----------------------------------------------------------------------------
    async def _fetch_livertox_entry(
        self,
        client: httpx.AsyncClient,
        nbk_id: str,
    ) -> str | None:
        params = {"db": "books", "id": nbk_id, "retmode": "xml"}

        try:
            response = await client.get(self._fetch_url, params=params)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch LiverTox entry '%s': %s", nbk_id, exc)
            return None

        return response.text

    # -----------------------------------------------------------------------------
    async def _query_esearch(
        self,
        client: httpx.AsyncClient,
        term: str,
        *,
        title_only: bool,
    ) -> list[str]:
        formatted = term.strip()
        if not formatted:
            return []

        quoted_term = f'"{formatted}"[title]' if title_only else formatted
        query = f"LiverTox[book] AND {quoted_term}"
        params = {
            "db": "books",
            "term": query,
            "retmode": "json",
            "retmax": "1" if title_only else "10",
        }

        try:
            response = await client.get(self._search_url, params=params)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to query LiverTox esearch for '%s': %s", formatted, exc)
            return []

        id_list = payload.get("esearchresult", {}).get("idlist", [])
        return [str(value) for value in id_list if value]

    # -----------------------------------------------------------------------------
    async def _fetch_candidate_summaries(
        self,
        client: httpx.AsyncClient,
        ids: list[str],
    ) -> list[CandidateSummary]:
        if not ids:
            return []

        params = {"db": "books", "id": ",".join(ids), "retmode": "json"}
        try:
            response = await client.get(self._summary_url, params=params)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to query LiverTox esummary for '%s': %s", ids, exc)
            return []

        result = payload.get("result", {})
        summaries: list[CandidateSummary] = []
        uid_list = result.get("uids", [])
        if not isinstance(uid_list, list):
            uid_list = [uid_list]

        for uid in uid_list:
            item = result.get(str(uid))
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            synonyms: set[str] = set()

            other = item.get("other")
            if isinstance(other, list):
                for value in other:
                    if isinstance(value, str) and value.strip():
                        synonyms.add(value.strip())
            elif isinstance(other, str) and other.strip():
                synonyms.add(other.strip())

            items = item.get("items")
            if isinstance(items, list):
                for entry in items:
                    if not isinstance(entry, dict):
                        continue
                    name = str(entry.get("name") or "").lower()
                    if name not in {"other term", "other terms", "synonym", "synonyms"}:
                        continue
                    value = entry.get("value")
                    if isinstance(value, list):
                        for token in value:
                            if isinstance(token, str) and token.strip():
                                synonyms.add(token.strip())
                    elif isinstance(value, str) and value.strip():
                        synonyms.add(value.strip())

            summaries.append(
                CandidateSummary(nbk_id=str(uid), title=title, synonyms=synonyms)
            )
        return summaries

    # -----------------------------------------------------------------------------
    async def _resolve_candidate_names(
        self, client: httpx.AsyncClient, drug_name: str
    ) -> tuple[list[tuple[str, str]], set[str]]:
        candidates, ingredient_hints = await self._lookup_rxnorm_candidates(
            client, drug_name
        )
        ordered = sorted(candidates, key=lambda item: item.priority)
        return [(item.origin, item.name) for item in ordered], ingredient_hints

    # -----------------------------------------------------------------------------
    async def _lookup_rxnorm_candidates(
        self, client: httpx.AsyncClient, drug_name: str
    ) -> tuple[list[NameCandidate], set[str]]:
        normalized = self._normalize_for_lookup(drug_name)
        cached = self._rxnorm_term_cache.get(normalized)
        if cached is not None:
            cached_candidates, cached_hints = cached
            return list(cached_candidates), set(cached_hints)

        term = drug_name.strip()
        if not term:
            self._rxnorm_term_cache[normalized] = (tuple(), frozenset())
            return [], set()

        payload = await self._safe_rxnorm_get(
            client,
            self._rxnorm_approx_url,
            params={"term": term, "maxEntries": "5"},
        )
        if not isinstance(payload, dict):
            self._rxnorm_term_cache[normalized] = (tuple(), frozenset())
            return [], set()

        approximate_group = payload.get("approximateGroup", {})
        raw_candidates = approximate_group.get("candidate", [])
        if isinstance(raw_candidates, dict):
            raw_candidates = [raw_candidates]

        candidates: list[NameCandidate] = []
        ingredient_hints: set[str] = set()
        seen: set[str] = set()

        def push(origin: str, name: str, priority: int) -> None:
            formatted = self._format_name_for_query(name)
            normalized_candidate = self._normalize_for_lookup(formatted)
            if not formatted or not normalized_candidate or normalized_candidate in seen:
                return
            candidates.append(NameCandidate(origin=origin, name=formatted, priority=priority))
            seen.add(normalized_candidate)
            if origin in {"rxnorm_ingredient", "rxnorm_brand"}:
                ingredient_hints.add(normalized_candidate)

        for entry in raw_candidates:
            rxcui = str(entry.get("rxcui") or "").strip()
            if not rxcui:
                continue
            concept = await self._describe_rxnorm_concept(client, rxcui)
            preferred = concept.preferred_name or str(entry.get("rxstring") or "")
            if preferred:
                push("rxnorm_name", preferred, 2)

            tty = (concept.tty or "").upper()
            ingredient_origin = "rxnorm_brand"
            if tty in {"IN", "PIN"}:
                ingredient_origin = "rxnorm_ingredient"
                if concept.preferred_name:
                    push("rxnorm_ingredient", concept.preferred_name, 1)

            for ingredient in sorted(concept.ingredients)[:5]:
                push(ingredient_origin, ingredient, 1)

            for synonym in sorted(concept.synonyms)[:5]:
                push("rxnorm_synonym", synonym, 3)

        ordered_candidates = sorted(candidates, key=lambda item: item.priority)
        self._rxnorm_term_cache[normalized] = (
            tuple(ordered_candidates),
            frozenset(ingredient_hints),
        )
        return ordered_candidates, ingredient_hints

    # -----------------------------------------------------------------------------
    async def _describe_rxnorm_concept(
        self, client: httpx.AsyncClient, rxcui: str
    ) -> RxNormConcept:
        cached = self._rxnorm_concept_cache.get(rxcui)
        if cached:
            return cached

        tty_payload = await self._safe_rxnorm_get(
            client,
            f"{self._rxnorm_property_url}/{rxcui}/property.json",
            params={"propName": "TTY"},
        )
        tty: str | None = None
        if isinstance(tty_payload, dict):
            group = tty_payload.get("propConceptGroup", {})
            concepts = group.get("propConcept", [])
            if isinstance(concepts, dict):
                concepts = [concepts]
            for concept in concepts:
                value = concept.get("propValue")
                if isinstance(value, str) and value.strip():
                    tty = value.strip()
                    break

        preferred_payload = await self._safe_rxnorm_get(
            client,
            f"{self._rxnorm_property_url}/{rxcui}/property.json",
            params={"propName": "RxNorm Name"},
        )
        preferred_name: str | None = None
        if isinstance(preferred_payload, dict):
            group = preferred_payload.get("propConceptGroup", {})
            concepts = group.get("propConcept", [])
            if isinstance(concepts, dict):
                concepts = [concepts]
            for concept in concepts:
                value = concept.get("propValue")
                if isinstance(value, str) and value.strip():
                    preferred_name = value.strip()
                    break

        synonyms_payload = await self._safe_rxnorm_get(
            client,
            f"{self._rxnorm_property_url}/{rxcui}/synonyms.json",
        )
        synonyms: set[str] = set()
        if isinstance(synonyms_payload, dict):
            group = synonyms_payload.get("synonymGroup", {})
            values = group.get("synonym")
            if isinstance(values, list):
                for entry in values:
                    if isinstance(entry, str) and entry.strip():
                        synonyms.add(entry.strip())
            elif isinstance(values, str) and values.strip():
                synonyms.add(values.strip())

        related_payload = await self._safe_rxnorm_get(
            client,
            f"{self._rxnorm_property_url}/{rxcui}/related.json",
            params={"tty": "IN+PIN"},
        )
        ingredients: set[str] = set()
        if isinstance(related_payload, dict):
            group = related_payload.get("relatedGroup", {})
            concept_groups = group.get("conceptGroup", [])
            if isinstance(concept_groups, dict):
                concept_groups = [concept_groups]
            for item in concept_groups:
                concepts = item.get("conceptProperties", [])
                if isinstance(concepts, dict):
                    concepts = [concepts]
                for concept in concepts:
                    name = concept.get("name")
                    if isinstance(name, str) and name.strip():
                        ingredients.add(name.strip())

        concept = RxNormConcept(
            rxcui=rxcui,
            preferred_name=preferred_name,
            synonyms=synonyms,
            ingredients=ingredients,
            tty=tty,
        )
        self._rxnorm_concept_cache[rxcui] = concept
        return concept

    # -----------------------------------------------------------------------------
    async def _safe_rxnorm_get(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        params: dict[str, str] | None = None,
    ) -> dict[str, object] | None:
        try:
            response = await client.get(
                url,
                params=params,
                headers=self._rxnorm_headers,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("RxNorm request failed for '%s': %s", url, exc)
            return None

    # -----------------------------------------------------------------------------
    def _format_name_for_query(self, name: str) -> str:
        normalized = self._normalize_for_lookup(name)
        if not normalized:
            return ""
        tokens = normalized.split()
        if not tokens:
            return ""
        formatted: list[str] = []
        for token in tokens:
            if token in {"and", "or", "of"}:
                formatted.append(token)
            else:
                formatted.append(token.capitalize())
        return " ".join(formatted)

    # -----------------------------------------------------------------------------
    @staticmethod
    def _normalize_for_lookup(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value or "")
        stripped = "".join(
            character for character in normalized if not unicodedata.combining(character)
        )
        lowered = stripped.lower()
        sanitized = re.sub(r"[^a-z0-9]+", " ", lowered)
        return re.sub(r"\s+", " ", sanitized).strip()

    # -----------------------------------------------------------------------------
    def _token_set_ratio(self, reference: str, candidate: str) -> float:
        normalized_reference = self._normalize_for_lookup(reference)
        normalized_candidate = self._normalize_for_lookup(candidate)
        if not normalized_reference or not normalized_candidate:
            return 0.0
        reference_tokens = sorted(set(normalized_reference.split()))
        candidate_tokens = sorted(set(normalized_candidate.split()))
        if not reference_tokens or not candidate_tokens:
            return 0.0
        reference_joined = " ".join(reference_tokens)
        candidate_joined = " ".join(candidate_tokens)
        return SequenceMatcher(None, reference_joined, candidate_joined).ratio() * 100.0

    # -----------------------------------------------------------------------------
    def _cache_match(self, normalized_name: str, match: LiverToxMatch) -> None:
        if not normalized_name:
            return
        self._match_cache[normalized_name] = match

    # -----------------------------------------------------------------------------
    @staticmethod
    def _match_to_schema(match: LiverToxMatch | None) -> LiverToxMatchInfo | None:
        if not match:
            return None
        return LiverToxMatchInfo(
            nbk_id=match.nbk_id,
            matched_name=match.matched_name,
            confidence=max(0.0, min(1.0, match.confidence)),
            reason=match.reason,
        )

    # -----------------------------------------------------------------------------
    def _extract_livertox_sections(self, xml_payload: str) -> str | None:
        try:
            root = ET.fromstring(xml_payload)
        except ET.ParseError as exc:
            logger.error("Failed to parse LiverTox XML: %s", exc)
            return None

        sections: list[str] = []
        excluded_titles = {"References"}

        for sect in root.findall(".//sect1"):
            title = (sect.findtext("title") or "").strip()
            if title in excluded_titles:
                continue

            paragraphs: list[str] = []
            for para in sect.findall("p"):
                text = "".join(para.itertext()).strip()
                if text:
                    paragraphs.append(text)

            if not paragraphs:
                continue

            if title:
                sections.append(f"{title}: {' '.join(paragraphs)}")
            else:
                sections.append(" ".join(paragraphs))

        if not sections:
            return None

        return "\n\n".join(sections)

    # -----------------------------------------------------------------------------
    def _prepare_prompt_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= self.max_prompt_chars:
            return normalized
        return normalized[: self.max_prompt_chars]

