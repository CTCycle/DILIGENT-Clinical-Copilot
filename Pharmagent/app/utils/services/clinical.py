from __future__ import annotations

import html
import importlib
import io
import os
import re
import tarfile
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from Pharmagent.app.api.models.prompts import (
    HEPATOTOXICITY_ANALYSIS_SYSTEM_PROMPT,
    HEPATOTOXICITY_ANALYSIS_USER_PROMPT,
    LIVERTOX_MATCH_SYSTEM_PROMPT,
    LIVERTOX_MATCH_USER_PROMPT,
)
from Pharmagent.app.api.models.providers import initialize_llm_client
from Pharmagent.app.api.schemas.clinical import (
    DrugEntry,
    DrugHepatotoxicityAnalysis,
    DrugToxicityFindings,
    HepatotoxicityPatternScore,
    LiverToxMatchInfo,
    LiverToxMatchSuggestion,
    PatientData,
    PatientDrugToxicityBundle,
    PatientDrugs,
)
from Pharmagent.app.constants import LIVERTOX_ARCHIVE, SOURCES_PATH
from Pharmagent.app.configurations import ClientRuntimeConfig
from Pharmagent.app.logger import logger
from Pharmagent.app.utils.serializer import DataSerializer
from Pharmagent.app.utils.services.scraper import LiverToxClient

_pdfminer_extract_text = None
_pdfminer_package_spec = importlib.util.find_spec("pdfminer")
if _pdfminer_package_spec is not None:
    _pdfminer_spec = importlib.util.find_spec("pdfminer.high_level")
    if _pdfminer_spec is not None:
        _pdfminer_module = importlib.import_module("pdfminer.high_level")
        _pdfminer_extract_text = getattr(_pdfminer_module, "extract_text", None)

_pdf_reader_cls = None
_pypdf_spec = importlib.util.find_spec("pypdf")
if _pypdf_spec is not None:
    _pypdf_module = importlib.import_module("pypdf")
    _pdf_reader_cls = getattr(_pypdf_module, "PdfReader", None)
else:
    _pypdf2_spec = importlib.util.find_spec("PyPDF2")
    if _pypdf2_spec is not None:
        _pypdf2_module = importlib.import_module("PyPDF2")
        _pdf_reader_cls = getattr(_pypdf2_module, "PdfReader", None)


###############################################################################
@dataclass(slots=True)
class MonographRecord:
    nbk_id: str
    drug_name: str
    normalized_name: str
    tokens: set[str]
    excerpt: str | None


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
@dataclass(slots=True)
class ArchiveEntry:
    nbk_id: str
    title: str
    aliases: set[str]
    member_name: str
    normalized_title: str
    primary_normalized_title: str
    normalized_aliases: set[str]
    keyword_tokens: set[str]


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
    DIRECT_CONFIDENCE = 1.0
    ALIAS_CONFIDENCE = 0.95
    MIN_CONFIDENCE = 0.40
    DETERMINISTIC_THRESHOLD = 0.86
    LLM_DEFAULT_CONFIDENCE = 0.65
    SUPPORTED_EXTENSIONS: tuple[str, ...] = (
        ".html",
        ".htm",
        ".xml",
        ".xhtml",
        ".nxml",
        ".pdf",
    )
    IMAGE_EXTENSIONS: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".tiff",
        ".bmp",
    )
    SECTION_ALIASES: dict[str, set[str]] = {
        "summary": {"summary", "introduction", "overview"},
        "hepatotoxicity": {"hepatotoxicity", "liver injury"},
        "mechanism": {"mechanism", "mechanism of injury"},
        "latency_pattern_severity": {
            "latency, pattern, and severity",
            "latency pattern and severity",
            "latency and pattern",
            "latency",
        },
        "clinical_course": {"clinical course"},
        "outcome": {"outcome", "management", "outcome and management"},
        "references": {"references", "bibliography", "further reading"},
    }

    # -----------------------------------------------------------------------------
    def __init__(
        self,
        drugs: PatientDrugs,
        *,
        ensure_download: bool = True,
        timeout_s: float = 300.0,
        archive_path: str | None = None,
    ) -> None:
        self.drugs = drugs
        normalized_archive_path = archive_path or os.path.join(
            SOURCES_PATH, LIVERTOX_ARCHIVE
        )
        if os.path.isdir(normalized_archive_path):
            normalized_archive_path = os.path.join(
                normalized_archive_path, LIVERTOX_ARCHIVE
            )
        self.archive_path = os.path.abspath(normalized_archive_path)
        self.auto_download = ensure_download
        self.llm_client = initialize_llm_client(purpose="agent", timeout_s=timeout_s)
        self.match_cache: dict[str, LiverToxMatch] = {}
        self.content_cache: dict[str, dict[str, str]] = {}
        self.serializer = DataSerializer()
        self.monograph_records: list[MonographRecord] = []
        self.records_by_normalized: dict[str, MonographRecord] = {}
        self._candidate_prompt_block: str | None = None
        self.entries: list[ArchiveEntry] = []
        self.entry_by_nbk: dict[str, ArchiveEntry] = {}
        self.archive_ready = False
    # -----------------------------------------------------------------------------
    async def run_analysis(self) -> PatientDrugToxicityBundle:
        await self._ensure_index_loaded()
        self._ensure_livertox_records()
        results: list[DrugHepatotoxicityAnalysis] = []
        for entry in self.drugs.entries:
            result = await self._process_single_drug(entry)
            results.append(result)
        return PatientDrugToxicityBundle(entries=results)

    # -----------------------------------------------------------------------------
    async def _process_single_drug(self, entry: DrugEntry) -> DrugHepatotoxicityAnalysis:
        match_notes: list[str] = []
        try:
            match = await self._search_livertox_id(entry.name, notes=match_notes)
        except Exception as exc:
            logger.error("Failed LiverTox lookup for %s: %s", entry.name, exc)
            return DrugHepatotoxicityAnalysis(
                drug_name=entry.name,
                source_text=None,
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
                source_text=None,
                analysis=None,
                error=explanation,
                livertox_match=None,
            )

        record = match.record
        info_notes = list(match.notes)
        content: dict[str, str] = {}
        try:
            content = await self._fetch_livertox_content(match.nbk_id)
        except Exception as exc:
            logger.error(
                "Failed retrieving local LiverTox content for %s (%s): %s",
                entry.name,
                match.nbk_id,
                exc,
            )
            if record and record.excerpt:
                content = {
                    "summary": record.excerpt,
                    "hepatotoxicity": record.excerpt,
                    "raw_html": record.excerpt,
                }
                info_notes.append("Used stored LiverTox excerpt from database")
            else:
                notes_payload = list(dict.fromkeys(info_notes))
                return DrugHepatotoxicityAnalysis(
                    drug_name=entry.name,
                    source_text=None,
                    analysis=None,
                    error=f"Failed retrieving LiverTox content: {exc}",
                    livertox_match=LiverToxMatchInfo(
                        nbk_id=match.nbk_id,
                        matched_name=match.matched_name,
                        confidence=match.confidence,
                        reason=match.reason,
                        notes=notes_payload,
                    ),
                )

        if record and record.excerpt:
            content.setdefault("summary", record.excerpt)
            content.setdefault("hepatotoxicity", record.excerpt)

        try:
            llm_payload = await self._invoke_llm(entry.name, content)
        except Exception as exc:
            logger.error("LLM hepatotoxicity analysis failed for %s: %s", entry.name, exc)
            source_text = (
                content.get("hepatotoxicity")
                or content.get("summary")
                or (record.excerpt if record else None)
            )
            notes_payload = list(dict.fromkeys(info_notes))
            return DrugHepatotoxicityAnalysis(
                drug_name=entry.name,
                analysis=None,
                error=f"LLM analysis failed: {exc}",
                source_text=source_text,
                livertox_match=LiverToxMatchInfo(
                    nbk_id=match.nbk_id,
                    matched_name=match.matched_name,
                    confidence=match.confidence,
                    reason=match.reason,
                    notes=notes_payload,
                ),
            )

        source_text = (
            content.get("hepatotoxicity")
            or content.get("summary")
            or (record.excerpt if record else None)
        )
        notes_payload = list(dict.fromkeys(info_notes))
        return DrugHepatotoxicityAnalysis(
            drug_name=entry.name,
            source_text=source_text,
            analysis=llm_payload,
            error=None,
            livertox_match=LiverToxMatchInfo(
                nbk_id=match.nbk_id,
                matched_name=match.matched_name,
                confidence=match.confidence,
                reason=match.reason,
                notes=notes_payload,
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
        return await self.llm_client.llm_structured_call(
            model=model_name,
            system_prompt=HEPATOTOXICITY_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=prompt,
            schema=DrugToxicityFindings,
            temperature=0.0,
        )

    # -----------------------------------------------------------------------------
    async def _ensure_archive_ready(self) -> None:
        if self.archive_ready:
            return
        archive_path = self.archive_path
        archive_dir = os.path.dirname(archive_path) or os.curdir
        os.makedirs(archive_dir, exist_ok=True)
        if not os.path.isfile(archive_path):
            if not self.auto_download:
                raise FileNotFoundError(f"LiverTox archive missing at {archive_path}")
            client = LiverToxClient()
            await client.download_bulk_data(archive_dir)
        if not tarfile.is_tarfile(archive_path):
            raise RuntimeError(f"Invalid LiverTox archive at {archive_path}")
        self.archive_ready = True

    # -----------------------------------------------------------------------------
    async def _ensure_index_loaded(self) -> None:
        if self.entries:
            return
        await self._ensure_archive_ready()
        self.entries = self._build_index(self.archive_path)
        self.entry_by_nbk = {entry.nbk_id: entry for entry in self.entries}
        if not self.entries:
            raise RuntimeError("Local LiverTox archive is empty")

    # -----------------------------------------------------------------------------
    def _ensure_livertox_records(self) -> None:
        if self.monograph_records:
            return
        try:
            dataset = self.serializer.get_livertox_records()
        except Exception as exc:
            logger.error("Failed loading LiverTox monographs from database: %s", exc)
            return

        processed: list[MonographRecord] = []
        normalized_map: dict[str, MonographRecord] = {}
        if dataset is None or dataset.empty:
            return

        for row in dataset.itertuples(index=False):
            raw_name = str(getattr(row, "drug_name", "") or "").strip()
            if not raw_name:
                continue
            normalized_name = self._normalize_name(raw_name)
            if not normalized_name:
                continue
            nbk_raw = getattr(row, "nbk_id", None)
            nbk_id = str(nbk_raw).strip() if nbk_raw is not None else ""
            excerpt_raw = getattr(row, "excerpt", None)
            excerpt = str(excerpt_raw) if excerpt_raw not in (None, "") else None
            tokens = {token for token in normalized_name.split() if token}
            monograph = MonographRecord(
                nbk_id=nbk_id,
                drug_name=raw_name,
                normalized_name=normalized_name,
                tokens=tokens,
                excerpt=excerpt,
            )
            processed.append(monograph)
            if normalized_name not in normalized_map:
                normalized_map[normalized_name] = monograph
            primary_name = self._normalize_name(raw_name.split("(")[0])
            if primary_name and primary_name not in normalized_map:
                normalized_map[primary_name] = monograph

        if not processed:
            return

        processed.sort(key=lambda item: item.drug_name.lower())
        self.monograph_records = processed
        self.records_by_normalized = normalized_map
        self._candidate_prompt_block = "\n".join(
            f"- {record.drug_name}" for record in self.monograph_records
        )

    # -----------------------------------------------------------------------------
    async def _search_livertox_id(
        self, drug_name: str, *, notes: list[str] | None = None
    ) -> LiverToxMatch | None:
        await self._ensure_index_loaded()
        self._ensure_livertox_records()
        normalized_query = self._normalize_name(drug_name)
        if not normalized_query:
            if notes is not None:
                notes.append("Drug name is empty after normalization")
            return None
        if not self.monograph_records:
            if notes is not None:
                notes.append("LiverTox database is empty")
            return None
        cached = self.match_cache.get(normalized_query)
        if cached is not None:
            return cached

        deterministic = self._deterministic_lookup(normalized_query)
        if deterministic is not None:
            record, confidence, reason, extra_notes = deterministic
            if extra_notes and notes is not None:
                notes.extend(extra_notes)
            match = self._create_match(record, confidence, reason, extra_notes)
            self.match_cache[normalized_query] = match
            return match

        if notes is not None:
            notes.append("Deterministic match failed; invoking LLM fallback")

        try:
            fallback = await self._llm_match_lookup(drug_name)
        except Exception as exc:
            if notes is not None:
                notes.append(f"LLM fallback failed: {exc}")
            logger.error("LLM fallback match failed for %s: %s", drug_name, exc)
            return None
        if fallback is None:
            if notes is not None:
                notes.append("No matching entry found in LiverTox database")
            return None

        record, confidence, reason, extra_notes = fallback
        if extra_notes and notes is not None:
            notes.extend(extra_notes)
        match = self._create_match(record, confidence, reason, extra_notes)
        self.match_cache[normalized_query] = match
        return match

    # -----------------------------------------------------------------------------
    def _create_match(
        self,
        record: MonographRecord,
        confidence: float,
        reason: str,
        notes: list[str] | None,
    ) -> LiverToxMatch:
        normalized_confidence = round(
            min(max(confidence, self.MIN_CONFIDENCE), 1.0), 2
        )
        cleaned_notes = list(
            dict.fromkeys(note for note in (notes or []) if note)
        )
        return LiverToxMatch(
            nbk_id=record.nbk_id,
            matched_name=record.drug_name,
            confidence=normalized_confidence,
            reason=reason,
            notes=cleaned_notes,
            record=record,
        )

    # -----------------------------------------------------------------------------
    def _deterministic_lookup(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        record = self.records_by_normalized.get(normalized_query)
        if record is not None:
            return record, self.DIRECT_CONFIDENCE, "direct_match", []

        query_tokens = {token for token in normalized_query.split() if token}
        token_match = self._match_via_tokens(query_tokens)
        if token_match is not None:
            record, subset_score = token_match
            notes = [f"Matched token subset within '{record.drug_name}'"]
            confidence = max(self.ALIAS_CONFIDENCE, subset_score)
            return record, confidence, "alias_match", notes

        best = self._find_best_record(normalized_query)
        if best is None:
            return None
        record, score = best
        if score < self.DETERMINISTIC_THRESHOLD:
            return None
        notes = [f"Closest database name '{record.drug_name}' (score={score:.2f})"]
        return record, score, "fuzzy_match", notes

    # -----------------------------------------------------------------------------
    def _match_via_tokens(
        self, query_tokens: set[str]
    ) -> tuple[MonographRecord, float] | None:
        if not query_tokens:
            return None
        best: tuple[MonographRecord, float] | None = None
        for record in self.monograph_records:
            if not record.tokens:
                continue
            if query_tokens.issubset(record.tokens):
                subset_score = len(query_tokens) / len(record.tokens)
                if best is None or subset_score > best[1]:
                    best = (record, subset_score)
        return best

    # -----------------------------------------------------------------------------
    def _find_best_record(
        self, normalized_query: str
    ) -> tuple[MonographRecord, float] | None:
        best: tuple[MonographRecord, float] | None = None
        for record in self.monograph_records:
            score = self._compute_similarity(normalized_query, record)
            if best is None or score > best[1]:
                best = (record, score)
        return best

    # -----------------------------------------------------------------------------
    def _compute_similarity(
        self, normalized_query: str, record: MonographRecord
    ) -> float:
        candidate = record.normalized_name
        if not candidate:
            return 0.0
        seq_score = SequenceMatcher(None, normalized_query, candidate).ratio()
        query_tokens = {token for token in normalized_query.split() if token}
        token_score = 0.0
        if query_tokens and record.tokens:
            intersection = len(query_tokens & record.tokens)
            union = len(query_tokens | record.tokens)
            if union:
                token_score = intersection / union
        lev_distance = self._levenshtein_distance(normalized_query, candidate)
        max_len = max(len(normalized_query), len(candidate))
        levenshtein_score = 1.0 - (lev_distance / max_len) if max_len else 1.0
        return max(seq_score, (seq_score + token_score + levenshtein_score) / 3)

    # -----------------------------------------------------------------------------
    def _levenshtein_distance(self, left: str, right: str) -> int:
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, start=1):
            current = [i]
            for j, right_char in enumerate(right, start=1):
                insert_cost = previous[j] + 1
                delete_cost = current[j - 1] + 1
                replace_cost = previous[j - 1] + (0 if left_char == right_char else 1)
                current.append(min(insert_cost, delete_cost, replace_cost))
            previous = current
        return previous[-1]

    # -----------------------------------------------------------------------------
    async def _llm_match_lookup(
        self, drug_name: str
    ) -> tuple[MonographRecord, float, str, list[str]] | None:
        if not self.monograph_records:
            return None
        candidate_block = self._candidate_prompt_block or ""
        if not candidate_block:
            candidate_block = "\n".join(
                f"- {record.drug_name}" for record in self.monograph_records
            )
            self._candidate_prompt_block = candidate_block
        if not candidate_block:
            return None

        prompt = LIVERTOX_MATCH_USER_PROMPT.format(
            drug_name=drug_name,
            candidates=candidate_block,
        )
        model_name = ClientRuntimeConfig.get_agent_model()
        suggestion = await self.llm_client.llm_structured_call(
            model=model_name,
            system_prompt=LIVERTOX_MATCH_SYSTEM_PROMPT,
            user_prompt=prompt,
            schema=LiverToxMatchSuggestion,
            temperature=0.0,
        )

        match_name = (suggestion.match_name or "").strip()
        if not match_name:
            return None
        normalized_match = self._normalize_name(match_name)
        record = self.records_by_normalized.get(normalized_match)

        confidence = (
            suggestion.confidence
            if suggestion.confidence is not None
            else self.LLM_DEFAULT_CONFIDENCE
        )
        notes: list[str] = [f"LLM selected '{match_name}'"]
        if suggestion.rationale:
            notes.append(suggestion.rationale.strip())

        if record is None:
            best = self._find_best_record(normalized_match)
            if best is None:
                return None
            record, score = best
            notes.append(
                f"Mapped suggestion to '{record.drug_name}' (score={score:.2f})"
            )
            confidence = min(max(confidence, score), 1.0)

        return record, confidence, "llm_fallback", notes

    # -----------------------------------------------------------------------------
    async def _fetch_livertox_content(self, nbk_id: str) -> dict[str, str]:
        cached = self.content_cache.get(nbk_id)
        if cached is not None:
            return cached
        await self._ensure_index_loaded()
        entry = self.entry_by_nbk.get(nbk_id)
        if entry is None:
            raise KeyError(f"No entry for NBK id {nbk_id}")

        html_text = self._read_archive_member(entry.member_name)

        sections = self._extract_sections(html_text)
        sections.setdefault("title", entry.title)
        sections.setdefault("nbk_id", entry.nbk_id)
        self.content_cache[nbk_id] = sections
        return sections

    # -----------------------------------------------------------------------------
    def _extract_sections(self, html_text: str) -> dict[str, str]:
        sections = {key: "" for key in self.SECTION_ALIASES}
        sections["raw_html"] = html_text
        blocks = self._iter_heading_blocks(html_text)
        if blocks:
            first_block = blocks[0]
            sections["summary"] = self._html_to_text(html_text[: first_block[1]]).strip()
        else:
            sections["summary"] = self._html_to_text(html_text).strip()

        for heading_text, _heading_start, content_start, content_end in blocks:
            body_html = html_text[content_start:content_end]
            body_text = self._html_to_text(body_html).strip()
            if not body_text:
                continue
            key = self._map_heading_to_key(heading_text)
            if key is None:
                continue
            if sections[key]:
                sections[key] = f"{sections[key]}\n\n{body_text}".strip()
            else:
                sections[key] = body_text

        if not sections["summary"]:
            sections["summary"] = self._html_to_text(html_text).strip()
        return sections

    # -----------------------------------------------------------------------------
    def _iter_heading_blocks(self, html_text: str) -> list[tuple[str, int, int, int]]:
        pattern = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
        matches = list(pattern.finditer(html_text))
        blocks: list[tuple[str, int, int, int]] = []
        for index, match in enumerate(matches):
            heading_text = self._clean_fragment(match.group(2))
            heading_start = match.start()
            content_start = match.end()
            content_end = matches[index + 1].start() if index + 1 < len(matches) else len(html_text)
            blocks.append((heading_text, heading_start, content_start, content_end))
        return blocks

    # -----------------------------------------------------------------------------
    def _map_heading_to_key(self, heading: str) -> str | None:
        normalized = self._normalize_name(heading)
        if not normalized:
            return None
        for key, aliases in self.SECTION_ALIASES.items():
            for alias in aliases:
                alias_normalized = self._normalize_name(alias)
                if normalized.startswith(alias_normalized) or alias_normalized in normalized:
                    return key
        return None

    # -----------------------------------------------------------------------------
    def _build_index(self, archive_path: str) -> list[ArchiveEntry]:
        entries: list[ArchiveEntry] = []
        try:
            tar = tarfile.open(archive_path, "r:gz")
        except tarfile.ReadError as exc:
            raise RuntimeError(f"Failed to open LiverTox archive {archive_path}") from exc
        with tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name_lower = member.name.lower()
                if any(name_lower.endswith(ext) for ext in self.IMAGE_EXTENSIONS):
                    continue
                if not any(name_lower.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS):
                    continue
                document_text = self._read_text_from_tar_member(tar, member)
                if document_text is None:
                    continue
                plain_text = self._html_to_text(document_text)
                nbk_id = self._extract_nbk(member.name, document_text)
                if nbk_id is None:
                    continue
                title = self._extract_title(document_text, plain_text, nbk_id)
                aliases = self._extract_aliases(title, plain_text)
                normalized_title = self._normalize_name(title)
                primary_normalized = self._normalize_name(title.split("(")[0])
                normalized_aliases = set()
                for alias in aliases:
                    normalized_alias = self._normalize_name(alias)
                    if normalized_alias:
                        normalized_aliases.add(normalized_alias)
                keyword_tokens = self._tokens_from_values(
                    [
                        normalized_title,
                        primary_normalized,
                        *normalized_aliases,
                        self._normalize_name(nbk_id),
                    ]
                )
                entries.append(
                    ArchiveEntry(
                        nbk_id=nbk_id,
                        title=title,
                        aliases=aliases,
                        member_name=member.name,
                        normalized_title=normalized_title,
                        primary_normalized_title=primary_normalized,
                        normalized_aliases=normalized_aliases,
                        keyword_tokens=keyword_tokens,
                    )
                )
        if not entries:
            logger.warning("No LiverTox entries found in archive %s", archive_path)
        return entries

    # -----------------------------------------------------------------------------
    def _read_archive_member(self, member_name: str) -> str:
        try:
            with tarfile.open(self.archive_path, "r:gz") as tar:
                try:
                    member = tar.getmember(member_name)
                except KeyError as exc:
                    raise KeyError(f"Archive member for {member_name} not found") from exc
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    raise ValueError(f"Unable to read archive member {member_name}")
                raw = fileobj.read()
                return self._convert_member_bytes(member.name, raw)
        except tarfile.ReadError as exc:
            raise RuntimeError(f"Failed to read LiverTox archive {self.archive_path}") from exc

    # -----------------------------------------------------------------------------
    def _read_text_from_tar_member(
        self, tar: tarfile.TarFile, member: tarfile.TarInfo
    ) -> str | None:
        fileobj = tar.extractfile(member)
        if fileobj is None:
            return None
        raw = fileobj.read()
        return self._convert_member_bytes(member.name, raw)

    # -----------------------------------------------------------------------------
    def _convert_member_bytes(self, member_name: str, data: bytes) -> str:
        lower_name = member_name.lower()
        if lower_name.endswith(".pdf"):
            return self._pdf_to_text(data)
        return self._decode_markup(data)

    # -----------------------------------------------------------------------------
    def _decode_markup(self, data: bytes) -> str:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="ignore")

    # -----------------------------------------------------------------------------
    def _pdf_to_text(self, data: bytes) -> str:
        buffer = io.BytesIO(data)
        if _pdfminer_extract_text is not None:
            try:
                buffer.seek(0)
                text = _pdfminer_extract_text(buffer)
                if text:
                    return text
            except Exception:
                buffer.seek(0)
        if _pdf_reader_cls is not None:
            try:
                buffer.seek(0)
                reader = _pdf_reader_cls(buffer)
                pages = getattr(reader, "pages", [])
                collected: list[str] = []
                for page in pages:
                    extractor = getattr(page, "extract_text", None)
                    if extractor is None:
                        continue
                    page_text = extractor()
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
    def _extract_nbk(self, member_name: str, html_text: str) -> str | None:
        match = re.search(r"NBK\d+", member_name, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        match = re.search(r"NBK\d+", html_text, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        return None

    # -----------------------------------------------------------------------------
    def _extract_title(self, html_text: str, plain_text: str, default: str) -> str:
        for pattern in (r"<title[^>]*>(.*?)</title>", r"<h1[^>]*>(.*?)</h1>"):
            match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                cleaned = self._clean_fragment(match.group(1))
                if cleaned:
                    return cleaned
        for line in plain_text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return default

    # -----------------------------------------------------------------------------
    def _extract_aliases(self, title: str, plain_text: str) -> set[str]:
        aliases: set[str] = set()
        for fragment in re.findall(r"\(([^)]+)\)", title):
            for part in re.split(r"[,;/]|\bor\b", fragment):
                cleaned = part.strip()
                if cleaned:
                    aliases.add(cleaned)
        patterns = [
            r"^synonyms?[:\s-]+(.+)$",
            r"^other names?[:\s-]+(.+)$",
            r"^brand names?[:\s-]+(.+)$",
            r"^trade names?[:\s-]+(.+)$",
            r"^also known as[:\s-]+(.+)$",
        ]
        for line in plain_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            for pattern in patterns:
                match = re.match(pattern, stripped, flags=re.IGNORECASE)
                if not match:
                    continue
                chunk = re.split(r"[.;]", match.group(1), maxsplit=1)[0]
                for part in re.split(r"[,/]|\bor\b", chunk):
                    cleaned = part.strip()
                    if cleaned:
                        aliases.add(cleaned)
        return aliases

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
    def _tokens_from_values(self, values: Iterable[str]) -> set[str]:
        tokens: set[str] = set()
        for value in values:
            for token in value.split():
                if token:
                    tokens.add(token)
        return tokens

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
    def _strip_punctuation(self, value: str) -> str:
        return re.sub(r"[-_,.;:()\[\]{}\/\\]", " ", value)

    # -----------------------------------------------------------------------------
    def _normalize_whitespace(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()
