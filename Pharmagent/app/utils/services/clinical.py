from __future__ import annotations

import asyncio
import random
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from xml.etree import ElementTree as ET

import httpx
from httpx import Timeout
from typing import Any
from Pharmagent.app.api.models.prompts import (
    HEPATOTOXICITY_ANALYSIS_SYSTEM_PROMPT,
    HEPATOTOXICITY_ANALYSIS_USER_PROMPT,
)
from Pharmagent.app.api.models.providers import initialize_llm_client
from Pharmagent.app.api.schemas.clinical import (
    DrugEntry,
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
    notes: list[str]


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
    """NCBI LiverTox ingestion orchestrator.

    Brief and clear description:
        Resolves drug names to LiverTox chapters via ESearch/ESummary scoped by
        NBK547852[BACI], applies retry/backoff with caching layers, extracts
        key sections, and logs normalized queries plus ranking scores.
    Keyword arguments:
        None.
    Return value:
        Not applicable. Class provides async workflow utilities.

    Tests:
        - Cover common inputs (Aspirin, Xarelto/rivaroxaban, Pantozol/pantoprazole).
        - Include ambiguous or misspelled queries to exercise fuzzy matching.
    """

    _EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    _RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
    _LIVERTOX_SCOPE = "NBK547852[BACI]"
    _CACHE_TTL = 60 * 60 * 6
    _CONTENT_TTL = 60 * 60 * 24
    _MAX_RETRIES = 4
    _BACKOFF_BASE = 0.5
    _BACKOFF_MAX = 8.0
    _MIN_REQUEST_INTERVAL = 0.12
    _CONFIDENCE_FUZZY_PENALTY = 0.07

    _match_cache: dict[str, tuple[float, LiverToxMatch]] = {}
    _rxnorm_term_cache: dict[str, tuple[float, tuple[list[NameCandidate], set[str]]]] = {}
    _rxnorm_concept_cache: dict[str, tuple[float, RxNormConcept]] = {}
    _esearch_cache: dict[str, tuple[float, list[str]]] = {}
    _esummary_cache: dict[str, tuple[float, list[CandidateSummary]]] = {}
    _content_cache: dict[str, tuple[float, dict[str, str]]] = {}

    # -----------------------------------------------------------------------------
    def __init__(self, drugs: PatientDrugs, *, timeout_s: float = 300.0) -> None:
        self.drugs = drugs
        self._timeout = Timeout(timeout_s)
        self._llm_client = initialize_llm_client(purpose="agent", timeout_s=timeout_s)
        self._tool_params = {"tool": "pharmagent_clinical", "email": "support@example.com"}
        self._last_request_ts = 0.0
        self._debug_records: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------------
    async def run_analysis(self) -> PatientDrugToxicityBundle:
        async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
            client.headers.update({
                "User-Agent": "PharmagentClinical/1.0 (+https://github.com/pharmagent)",
                "Accept": "application/json, text/plain;q=0.8",
            })

            tasks: list[DrugHepatotoxicityAnalysis] = []
            for entry in self.drugs.entries:
                result = await self._process_single_drug(client, entry)
                tasks.append(result)
        return PatientDrugToxicityBundle(entries=tasks)

    # -----------------------------------------------------------------------------
    async def _process_single_drug(
        self, client: httpx.AsyncClient, entry: DrugEntry
    ) -> DrugHepatotoxicityAnalysis:
        match_notes: list[str] = []
        try:
            match = await self._search_livertox_id(client, entry.name, notes=match_notes)
        except Exception as exc:  # Defensive safety net, surfaced as error payload.
            logger.error("Failed LiverTox matching for %s: %s", entry.name, exc)
            return DrugHepatotoxicityAnalysis(
                drug_name=entry.name,
                analysis=None,
                error=f"LiverTox search failed: {exc}",
                livertox_match=None,
            )

        if match is None:
            explanation = "No LiverTox chapter matched this query"
            if match_notes:
                explanation = f"{explanation}: {'; '.join(match_notes)}"
            return DrugHepatotoxicityAnalysis(
                drug_name=entry.name,
                analysis=None,
                error=explanation,
                livertox_match=None,
            )

        try:
            content = await self._fetch_livertox_content(client, match.nbk_id)
        except Exception as exc:
            logger.error("Failed retrieving content for %s (%s): %s", entry.name, match.nbk_id, exc)
            return DrugHepatotoxicityAnalysis(
                drug_name=entry.name,
                analysis=None,
                error=f"Failed retrieving LiverTox content: {exc}",
                livertox_match=LiverToxMatchInfo(
                    nbk_id=match.nbk_id,
                    matched_name=match.matched_name,
                    confidence=match.confidence,
                    reason=match.reason,
                    notes=match.notes,
                ),
            )

        llm_payload = None
        try:
            llm_payload = await self._invoke_llm(entry.name, content)
        except Exception as exc:
            logger.error("LLM hepatotoxicity analysis failed for %s: %s", entry.name, exc)
            return DrugHepatotoxicityAnalysis(
                drug_name=entry.name,
                analysis=None,
                error=f"LLM analysis failed: {exc}",
                source_text=content.get("summary"),
                livertox_match=LiverToxMatchInfo(
                    nbk_id=match.nbk_id,
                    matched_name=match.matched_name,
                    confidence=match.confidence,
                    reason=match.reason,
                    notes=match.notes,
                ),
            )

        return DrugHepatotoxicityAnalysis(
            drug_name=entry.name,
            source_text=content.get("hepatotoxicity") or content.get("summary"),
            analysis=llm_payload,
            livertox_match=LiverToxMatchInfo(
                nbk_id=match.nbk_id,
                matched_name=match.matched_name,
                confidence=match.confidence,
                reason=match.reason,
                notes=match.notes,
            ),
        )

    # -----------------------------------------------------------------------------
    async def _invoke_llm(
        self, drug_name: str, content: dict[str, str]
    ) -> DrugToxicityFindings:
        sections: list[str] = []
        ordered_keys: list[tuple[str, str]] = [
            ("summary", "Summary"),
            ("hepatotoxicity", "Hepatotoxicity"),
            ("mechanism", "Mechanism of Injury"),
            ("latency_pattern_severity", "Latency, Pattern, and Severity"),
            ("clinical_course", "Clinical Course"),
            ("outcome", "Outcome and Management"),
        ]
        for key, heading in ordered_keys:
            value = content.get(key)
            if not value:
                continue
            sections.append(f"## {heading}\n{value.strip()}")
        excerpt = "\n\n".join(sections).strip()
        if len(excerpt) > 8000:
            excerpt = excerpt[:8000]
        prompt = HEPATOTOXICITY_ANALYSIS_USER_PROMPT.format(
            drug_name=drug_name,
            source_text=excerpt or (content.get("raw_html") or ""),
        )
        model_name = ClientRuntimeConfig.get_agent_model()
        return await self._llm_client.llm_structured_call(
            model=model_name,
            system_prompt=HEPATOTOXICITY_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=prompt,
            schema=DrugToxicityFindings,
            temperature=0.0,
        )

    # -----------------------------------------------------------------------------
    async def _search_livertox_id(
        self, client: httpx.AsyncClient, drug_name: str, *, notes: list[str] | None = None
    ) -> LiverToxMatch | None:
        normalized_query = self._normalize_name(drug_name)
        holder: list[str] = notes if notes is not None else []
        cached = self._read_cache(self._match_cache, normalized_query, self._CONTENT_TTL)
        if cached is not None:
            return cached

        candidates = self._build_name_candidates(drug_name)
        rx_candidates, rx_synonyms = await self._lookup_rxnorm_candidates(client, drug_name)
        if rx_candidates:
            candidates.extend(rx_candidates)
        candidates = self._deduplicate_candidates(candidates)

        synonym_pool: set[str] = set(rx_synonyms)

        match = await self._search_candidates(
            client,
            normalized_query,
            candidates,
            synonym_pool,
            holder,
        )
        if match is not None:
            self._match_cache[normalized_query] = (self._now(), match)
        return match

    # -----------------------------------------------------------------------------
    async def _search_candidates(
        self,
        client: httpx.AsyncClient,
        normalized_query: str,
        candidates: Sequence[NameCandidate],
        synonym_pool: set[str],
        notes: list[str],
    ) -> LiverToxMatch | None:
        # First attempt exact title matches.
        for candidate in sorted(candidates, key=lambda item: item.priority):
            direct_term = self._title_case(candidate.name)
            uids = await self._query_esearch(client, direct_term, title_only=True)
            if not uids:
                continue
            summaries = await self._fetch_candidate_summaries(client, uids)
            match = self._rank_candidates(
                normalized_query,
                summaries,
                synonym_pool,
                notes,
                source_candidate=candidate,
            )
            if match and match.reason != "list_first":
                return match

        # General search across all query variants.
        aggregated: set[str] = set()
        for candidate in sorted(candidates, key=lambda item: item.priority):
            ids = await self._query_esearch(client, candidate.name, title_only=False)
            aggregated.update(ids)

        if not aggregated:
            notes.append("NCBI ESearch returned no results")
            return None

        summaries = await self._fetch_candidate_summaries(client, list(aggregated))
        if not summaries:
            notes.append("ESummary yielded no LiverTox candidates")
            return None

        match = self._rank_candidates(
            normalized_query,
            summaries,
            synonym_pool,
            notes,
            source_candidate=None,
        )
        if match:
            return match

        fallback = summaries[0]
        fallback_notes = list(dict.fromkeys(notes + ["defaulted to first search result"]))
        return LiverToxMatch(
            nbk_id=fallback.nbk_id,
            matched_name=fallback.title,
            confidence=0.40,
            reason="list_first",
            notes=fallback_notes,
        )

    # -----------------------------------------------------------------------------
    def _rank_candidates(
        self,
        normalized_query: str,
        summaries: Sequence[CandidateSummary],
        synonym_pool: set[str],
        notes: list[str],
        *,
        source_candidate: NameCandidate | None,
    ) -> LiverToxMatch | None:
        rankings: list[tuple[float, str, CandidateSummary]] = []
        for summary in summaries:
            normalized_title = self._normalize_name(summary.title)
            summary_synonyms = {self._normalize_name(item) for item in summary.synonyms}
            combined_synonyms = synonym_pool | summary_synonyms
            score, reason = self._score_candidate(
                normalized_query,
                normalized_title,
                combined_synonyms,
            )
            if reason == "brand_resolved" and source_candidate is not None:
                notes.append(
                    f"Brand '{source_candidate.name}' mapped to '{summary.title}'"
                )
            elif reason == "direct_match" and source_candidate is not None:
                if source_candidate.origin != "input_raw":
                    notes.append(
                        f"Matched after {source_candidate.origin.replace('_', ' ')} normalization"
                    )
            rankings.append((score, reason, summary))

        if not rankings:
            return None

        rankings.sort(key=lambda item: item[0], reverse=True)
        best_score, best_reason, best_summary = rankings[0]

        if len(rankings) > 1 and rankings[1][0] >= best_score - 0.05:
            alternates = [
                f"{entry[2].title} ({entry[2].nbk_id})"
                for entry in rankings[1:3]
            ]
            if alternates:
                notes.append("Ambiguous matches: " + ", ".join(alternates))

        normalized_notes = list(dict.fromkeys(notes))
        return LiverToxMatch(
            nbk_id=best_summary.nbk_id,
            matched_name=best_summary.title,
            confidence=best_score,
            reason=best_reason,
            notes=normalized_notes,
        )

    # -----------------------------------------------------------------------------
    def _score_candidate(
        self,
        normalized_query: str,
        normalized_title: str,
        synonyms: set[str],
    ) -> tuple[float, str]:
        if normalized_query and normalized_query == normalized_title:
            return 1.0, "direct_match"
        if normalized_query in synonyms:
            return 0.95, "brand_resolved"

        query_tokens = {token for token in normalized_query.split() if token}
        title_tokens = {token for token in normalized_title.split() if token}
        overlap = 0.0
        if query_tokens:
            overlap = len(query_tokens & title_tokens) / len(query_tokens)

        ratio = SequenceMatcher(None, normalized_query, normalized_title).ratio()
        if ratio >= 0.70 or overlap >= 0.50:
            penalty = self._CONFIDENCE_FUZZY_PENALTY if ratio < 0.99 else 0.0
            score = max(ratio - penalty, overlap)
            score = round(max(0.4, min(score, 0.95)), 2)
            return score, "fuzzy_match"

        return 0.40, "list_first"

    # -----------------------------------------------------------------------------
    async def _query_esearch(
        self, client: httpx.AsyncClient, term: str, *, title_only: bool
    ) -> list[str]:
        normalized = term.strip()
        if not normalized:
            return []
        collapsed = re.sub(r"\s+", " ", normalized)
        if title_only:
            title_case = " ".join(token.capitalize() for token in collapsed.split())
            field = f'"{title_case}"[Title]'
        else:
            field = collapsed
        scoped_term = f"{field} AND {self._LIVERTOX_SCOPE}"
        cache_key = f"esearch::{scoped_term}"
        cached = self._read_cache(self._esearch_cache, cache_key, self._CACHE_TTL)
        if cached is not None:
            return cached
        params = {
            "db": "books",
            "retmode": "json",
            "term": scoped_term,
            **self._tool_params,
        }
        data = await self._request(
            client,
            f"{self._EUTILS_BASE}/esearch.fcgi",
            params=params,
            expect_json=True,
        )
        idlist = data.get("esearchresult", {}).get("idlist", [])
        ids = [uid for uid in idlist if isinstance(uid, str) and uid]
        self._esearch_cache[cache_key] = (self._now(), ids)
        self._record_debug({"action": "esearch", "term": scoped_term, "uids": ids})
        return ids

    # -----------------------------------------------------------------------------
    async def _fetch_candidate_summaries(
        self, client: httpx.AsyncClient, ids: Sequence[str]
    ) -> list[CandidateSummary]:
        unique_ids = sorted({uid for uid in ids if uid})
        if not unique_ids:
            return []
        cache_key = "esummary::" + ",".join(unique_ids)
        cached = self._read_cache(self._esummary_cache, cache_key, self._CACHE_TTL)
        if cached is not None:
            return cached
        params = {
            "db": "books",
            "retmode": "json",
            "id": ",".join(unique_ids),
            **self._tool_params,
        }
        data = await self._request(
            client,
            f"{self._EUTILS_BASE}/esummary.fcgi",
            params=params,
            expect_json=True,
        )
        summaries: list[CandidateSummary] = []
        result = data.get("result", {})
        for uid in result.get("uids", []):
            record = result.get(uid) or {}
            booktitle = str(record.get("booktitle") or "")
            if "livertox" not in booktitle.lower():
                continue
            nbk = self._extract_nbk(record)
            if not nbk:
                continue
            title = str(record.get("title") or record.get("sorttitle") or nbk)
            synonyms = self._extract_summary_synonyms(record)
            summaries.append(CandidateSummary(nbk_id=nbk, title=title, synonyms=synonyms))

        self._esummary_cache[cache_key] = (self._now(), summaries)
        self._record_debug({"action": "esummary", "ids": unique_ids, "count": len(summaries)})
        return summaries

    # -----------------------------------------------------------------------------
    async def _lookup_rxnorm_candidates(
        self, client: httpx.AsyncClient, drug_name: str
    ) -> tuple[list[NameCandidate], set[str]]:
        normalized = self._normalize_name(drug_name)
        cached = self._read_cache(self._rxnorm_term_cache, normalized, self._CACHE_TTL)
        if cached is not None:
            return cached
        params = {
            "term": drug_name,
            "maxEntries": 5,
        }
        try:
            data = await self._request(
                client,
                f"{self._RXNORM_BASE}/approximateTerm.json",
                params=params,
                expect_json=True,
            )
        except Exception as exc:
            logger.warning("RxNorm lookup failed for %s: %s", drug_name, exc)
            return [], set()

        entries = data.get("approximateGroup", {}).get("candidate") or []
        if isinstance(entries, dict):
            entries = [entries]

        candidates: list[NameCandidate] = []
        synonyms: set[str] = set()
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("term")
            rxcui = entry.get("rxcui")
            if not name or not rxcui:
                continue
            normalized_name = self._normalize_name(name)
            if normalized_name in seen:
                continue
            seen.add(normalized_name)
            rank = entry.get("rank")
            priority = int(rank) if isinstance(rank, str) and rank.isdigit() else len(candidates)
            candidates.append(
                NameCandidate(origin="rxnorm_brand", name=name, priority=priority)
            )
            synonyms.add(normalized_name)

        candidates = self._deduplicate_candidates(candidates)
        normalized_synonyms = {syn for syn in synonyms if syn}
        payload = (candidates, normalized_synonyms)
        self._rxnorm_term_cache[normalized] = (self._now(), payload)
        if normalized_synonyms:
            self._record_debug(
                {"action": "rxnorm", "query": drug_name, "synonyms": sorted(normalized_synonyms)}
            )
        return payload

    # -----------------------------------------------------------------------------
    async def _fetch_livertox_content(
        self, client: httpx.AsyncClient, nbk_id: str
    ) -> dict[str, str]:
        cached = self._read_cache(self._content_cache, nbk_id, self._CONTENT_TTL)
        if cached is not None:
            return cached
        url = f"https://www.ncbi.nlm.nih.gov/books/{nbk_id}/?report=xml"
        xml_text = await self._request(client, url, params=None, expect_json=False)
        sections = self._extract_sections_from_xml(xml_text)
        sections["raw_html"] = xml_text
        self._content_cache[nbk_id] = (self._now(), sections)
        return sections

    # -----------------------------------------------------------------------------
    def _extract_sections_from_xml(self, xml_text: str) -> dict[str, str]:
        sections: dict[str, str] = {
            "summary": "",
            "hepatotoxicity": "",
            "mechanism": "",
            "latency_pattern_severity": "",
            "clinical_course": "",
            "outcome": "",
            "references": "",
        }
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            cleaned = self._normalize_whitespace(re.sub(r"<[^>]+>", " ", xml_text))
            if cleaned:
                sections["summary"] = cleaned[:2000]
            return sections

        heading_map = {
            "introduction": "summary",
            "summary": "summary",
            "hepatotoxicity": "hepatotoxicity",
            "mechanism of injury": "mechanism",
            "mechanism": "mechanism",
            "latency to onset, pattern and severity": "latency_pattern_severity",
            "latency to onset, pattern, and severity": "latency_pattern_severity",
            "latency, pattern and severity": "latency_pattern_severity",
            "outcome and management": "clinical_course",
            "clinical course": "clinical_course",
            "outcome": "outcome",
        }

        references: list[str] = []

        for section in root.iter():
            if self._strip_namespace(section.tag) != "section":
                continue
            heading: str | None = None
            paragraphs: list[str] = []
            for child in section:
                tag = self._strip_namespace(child.tag)
                if tag == "title" and heading is None:
                    heading = (child.text or "").strip()
                elif tag in {"p", "para"}:
                    paragraphs.append(self._flatten_xml(child))
                elif tag in {"list", "ul", "ol"}:
                    paragraphs.append(self._flatten_xml(child))
            if not heading:
                continue
            normalized_heading = heading.lower()
            key = heading_map.get(normalized_heading)
            if key is None:
                continue
            content = "\n".join(part for part in paragraphs if part)
            sections[key] = content.strip()

        for ref in root.iter():
            if self._strip_namespace(ref.tag) not in {"ref", "reference"}:
                continue
            text = self._flatten_xml(ref)
            if text:
                references.append(text)
            if len(references) >= 10:
                break

        if references:
            sections["references"] = "\n".join(references)

        if not sections["summary"]:
            first_section = next(
                (
                    self._flatten_xml(child)
                    for child in root.iter()
                    if self._strip_namespace(child.tag) == "p"
                ),
                "",
            )
            sections["summary"] = first_section.strip()

        return sections

    # -----------------------------------------------------------------------------
    def _extract_nbk(self, record: dict[str, Any]) -> str | None:
        article_ids = record.get("articleids") or []
        for item in article_ids:
            if not isinstance(item, dict):
                continue
            value = item.get("value")
            if isinstance(value, str) and value.startswith("NBK"):
                return value
        for key in ("elocationid", "source", "bookaccession"):
            value = record.get(key)
            if isinstance(value, str) and "NBK" in value:
                match = re.search(r"NBK\d+", value)
                if match:
                    return match.group(0)
        return None

    # -----------------------------------------------------------------------------
    def _extract_summary_synonyms(self, record: dict[str, Any]) -> set[str]:
        synonyms: set[str] = set()
        for key in ("othername", "otherterm"):
            entry = record.get(key)
            if isinstance(entry, list):
                synonyms.update(str(item) for item in entry if item)
            elif isinstance(entry, str):
                synonyms.add(entry)
        mesh = record.get("meshheadinglist")
        if isinstance(mesh, list):
            synonyms.update(str(item) for item in mesh if item)
        return {syn.strip() for syn in synonyms if syn}

    # -----------------------------------------------------------------------------
    def _build_name_candidates(self, drug_name: str) -> list[NameCandidate]:
        candidates: list[NameCandidate] = []
        stripped = drug_name.strip()
        if stripped:
            candidates.append(NameCandidate(origin="input_raw", name=stripped, priority=0))
        normalized = self._normalize_name(drug_name)
        if normalized and normalized != stripped.lower():
            candidates.append(NameCandidate(origin="normalized", name=normalized, priority=1))
        ascii_variant = self._ascii_fold(drug_name)
        if ascii_variant and ascii_variant.lower() != normalized:
            candidates.append(NameCandidate(origin="ascii", name=ascii_variant, priority=2))
        for typo in self._generate_typo_variants(normalized):
            candidates.append(NameCandidate(origin="typo", name=typo, priority=3))
        return self._deduplicate_candidates(candidates)

    # -----------------------------------------------------------------------------
    def _deduplicate_candidates(self, items: Sequence[NameCandidate]) -> list[NameCandidate]:
        seen: set[str] = set()
        unique: list[NameCandidate] = []
        for item in sorted(items, key=lambda entry: entry.priority):
            key = self._normalize_name(item.name)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    # -----------------------------------------------------------------------------
    def _generate_typo_variants(self, normalized: str) -> set[str]:
        variants: set[str] = set()
        if not normalized or len(normalized) < 4:
            return variants
        # Single deletion variants (limit to first three positions to contain growth).
        for index in range(min(len(normalized), 6)):
            candidate = normalized[:index] + normalized[index + 1 :]
            if len(candidate) >= 4:
                variants.add(candidate)
        swaps = normalized.replace("ph", "f")
        if swaps != normalized:
            variants.add(swaps)
        if "y" in normalized:
            variants.add(normalized.replace("y", "i"))
        return variants

    # -----------------------------------------------------------------------------
    def _normalize_name(self, value: str) -> str:
        folded = self._ascii_fold(value)
        cleaned = self._strip_punctuation(folded.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    # -----------------------------------------------------------------------------
    def _ascii_fold(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        return "".join(char for char in normalized if not unicodedata.combining(char))

    # -----------------------------------------------------------------------------
    def _title_case(self, value: str) -> str:
        return " ".join(token.capitalize() for token in value.split())

    # -----------------------------------------------------------------------------
    def _strip_punctuation(self, value: str) -> str:
        return re.sub(r"[\-_,.;:()\[\]{}\/\\]", " ", value)

    # -----------------------------------------------------------------------------
    def _flatten_xml(self, node: ET.Element) -> str:
        text = "".join(node.itertext())
        return self._normalize_whitespace(text)

    # -----------------------------------------------------------------------------
    def _strip_namespace(self, tag: str) -> str:
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    # -----------------------------------------------------------------------------
    def _normalize_whitespace(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    # -----------------------------------------------------------------------------
    def _record_debug(self, payload: dict[str, Any]) -> None:
        self._debug_records.append(payload)
        logger.debug("LiverTox debug event: %s", payload)

    # -----------------------------------------------------------------------------
    async def _request(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        params: dict[str, Any] | None,
        expect_json: bool,
    ) -> Any:
        delay = self._BACKOFF_BASE
        for attempt in range(self._MAX_RETRIES):
            try:
                await self._respect_rate_limit()
                response = await client.get(url, params=params)
                response.raise_for_status()
                self._last_request_ts = self._now()
                if expect_json:
                    return response.json()
                return response.text
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                logger.warning("HTTP request failed (%s): %s", url, exc)
                if attempt + 1 >= self._MAX_RETRIES:
                    raise
                jitter = random.uniform(0.0, delay * 0.25)
                await asyncio.sleep(min(self._BACKOFF_MAX, delay) + jitter)
                delay *= 2

        raise RuntimeError("HTTP request retries exceeded")

    # -----------------------------------------------------------------------------
    async def _respect_rate_limit(self) -> None:
        elapsed = self._now() - self._last_request_ts
        if elapsed < self._MIN_REQUEST_INTERVAL:
            await asyncio.sleep(self._MIN_REQUEST_INTERVAL - elapsed)

    # -----------------------------------------------------------------------------
    def _read_cache(
        self,
        cache: dict[str, tuple[float, Any]],
        key: str,
        ttl: float,
    ) -> Any | None:
        entry = cache.get(key)
        if not entry:
            return None
        timestamp, payload = entry
        if self._now() - timestamp > ttl:
            cache.pop(key, None)
            return None
        return payload

    # -----------------------------------------------------------------------------
    def _now(self) -> float:
        return time.monotonic()
        