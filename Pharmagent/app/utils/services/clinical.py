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
from Pharmagent.app.constants import LIVERTOX_ARCHIVE, SOURCES_PATH
from Pharmagent.app.configurations import ClientRuntimeConfig
from Pharmagent.app.logger import logger
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
class LiverToxMatch:
    nbk_id: str
    matched_name: str
    confidence: float
    reason: str
    notes: list[str]


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
    _DIRECT_CONFIDENCE = 1.0
    _ALIAS_CONFIDENCE = 0.95
    _MIN_CONFIDENCE = 0.40
    _SUPPORTED_EXTENSIONS: tuple[str, ...] = (
        ".html",
        ".htm",
        ".xml",
        ".xhtml",
        ".nxml",
        ".pdf",
    )
    _IMAGE_EXTENSIONS: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".tiff",
        ".bmp",
    )
    _SECTION_ALIASES: dict[str, set[str]] = {
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
        self._archive_path = os.path.abspath(normalized_archive_path)
        self._auto_download = ensure_download
        self._llm_client = initialize_llm_client(purpose="agent", timeout_s=timeout_s)
        self._match_cache: dict[str, LiverToxMatch] = {}
        self._content_cache: dict[str, dict[str, str]] = {}
        self._entries: list[ArchiveEntry] = []
        self._entry_by_nbk: dict[str, ArchiveEntry] = {}
        self._archive_ready = False
    # -----------------------------------------------------------------------------
    async def run_analysis(self) -> PatientDrugToxicityBundle:
        await self._ensure_index_loaded()
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
            content = await self._fetch_livertox_content(match.nbk_id)
        except Exception as exc:
            logger.error(
                "Failed retrieving local LiverTox content for %s (%s): %s",
                entry.name,
                match.nbk_id,
                exc,
            )
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
    async def _ensure_archive_ready(self) -> None:
        if self._archive_ready:
            return
        archive_path = self._archive_path
        archive_dir = os.path.dirname(archive_path) or os.curdir
        os.makedirs(archive_dir, exist_ok=True)
        if not os.path.isfile(archive_path):
            if not self._auto_download:
                raise FileNotFoundError(f"LiverTox archive missing at {archive_path}")
            client = LiverToxClient()
            await client.download_bulk_data(archive_dir)
        if not tarfile.is_tarfile(archive_path):
            raise RuntimeError(f"Invalid LiverTox archive at {archive_path}")
        self._archive_ready = True

    # -----------------------------------------------------------------------------
    async def _ensure_index_loaded(self) -> None:
        if self._entries:
            return
        await self._ensure_archive_ready()
        self._entries = self._build_index(self._archive_path)
        self._entry_by_nbk = {entry.nbk_id: entry for entry in self._entries}
        if not self._entries:
            raise RuntimeError("Local LiverTox archive is empty")

    # -----------------------------------------------------------------------------
    async def _search_livertox_id(
        self, drug_name: str, *, notes: list[str] | None = None
    ) -> LiverToxMatch | None:
        await self._ensure_index_loaded()
        holder = notes if notes is not None else []
        normalized_query = self._normalize_name(drug_name)
        if not normalized_query:
            holder.append("Drug name is empty after normalization")
            return None
        cached = self._match_cache.get(normalized_query)
        if cached is not None:
            return cached

        best: tuple[float, ArchiveEntry, str, str | None] | None = None
        for entry in self._entries:
            score, reason, detail = self._score_entry(normalized_query, entry)
            if best is None or score > best[0]:
                best = (score, entry, reason, detail)

        if best is None or best[0] < self._MIN_CONFIDENCE:
            holder.append("No matching entry found in local LiverTox archive")
            return None

        score, entry, reason, detail = best
        notes_payload: list[str] = []
        if reason == "alias_match" and detail:
            notes_payload.append(f"Matched alias '{detail}'")
        elif reason == "fuzzy_match":
            notes_payload.append("Fuzzy matched against archive content")
            if detail:
                notes_payload.append(f"Closest alias '{detail}'")

        if notes is not None and notes_payload:
            notes.extend(notes_payload)

        match = LiverToxMatch(
            nbk_id=entry.nbk_id,
            matched_name=entry.title,
            confidence=round(min(max(score, self._MIN_CONFIDENCE), 1.0), 2),
            reason=reason,
            notes=list(dict.fromkeys(notes_payload)),
        )
        self._match_cache[normalized_query] = match
        return match

    # -----------------------------------------------------------------------------
    def _score_entry(
        self, normalized_query: str, entry: ArchiveEntry
    ) -> tuple[float, str, str | None]:
        if normalized_query in (
            entry.normalized_title,
            entry.primary_normalized_title,
        ):
            return self._DIRECT_CONFIDENCE, "direct_match", entry.title
        if normalized_query in entry.normalized_aliases:
            alias = next(
                (alias for alias in entry.aliases if self._normalize_name(alias) == normalized_query),
                entry.title,
            )
            return self._ALIAS_CONFIDENCE, "alias_match", alias

        best_alias = None
        best_alias_score = 0.0
        for alias in entry.aliases:
            normalized_alias = self._normalize_name(alias)
            if not normalized_alias:
                continue
            alias_score = SequenceMatcher(None, normalized_query, normalized_alias).ratio()
            if alias_score > best_alias_score:
                best_alias_score = alias_score
                best_alias = alias

        ratio = SequenceMatcher(None, normalized_query, entry.normalized_title).ratio()
        token_overlap = self._token_overlap(normalized_query, entry.keyword_tokens)
        score = max(ratio, token_overlap, best_alias_score)
        detail = best_alias if best_alias_score >= ratio and best_alias_score >= token_overlap else None
        return score, "fuzzy_match", detail

    # -----------------------------------------------------------------------------
    def _token_overlap(self, normalized_query: str, tokens: set[str]) -> float:
        query_tokens = [token for token in normalized_query.split() if token]
        if not query_tokens:
            return 0.0
        hits = sum(1 for token in query_tokens if token in tokens)
        return hits / len(query_tokens)

    # -----------------------------------------------------------------------------
    async def _fetch_livertox_content(self, nbk_id: str) -> dict[str, str]:
        cached = self._content_cache.get(nbk_id)
        if cached is not None:
            return cached
        await self._ensure_index_loaded()
        entry = self._entry_by_nbk.get(nbk_id)
        if entry is None:
            raise KeyError(f"No entry for NBK id {nbk_id}")

        html_text = self._read_archive_member(entry.member_name)

        sections = self._extract_sections(html_text)
        sections.setdefault("title", entry.title)
        sections.setdefault("nbk_id", entry.nbk_id)
        self._content_cache[nbk_id] = sections
        return sections

    # -----------------------------------------------------------------------------
    def _extract_sections(self, html_text: str) -> dict[str, str]:
        sections = {key: "" for key in self._SECTION_ALIASES}
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
        for key, aliases in self._SECTION_ALIASES.items():
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
                if any(name_lower.endswith(ext) for ext in self._IMAGE_EXTENSIONS):
                    continue
                if not any(name_lower.endswith(ext) for ext in self._SUPPORTED_EXTENSIONS):
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
            with tarfile.open(self._archive_path, "r:gz") as tar:
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
            raise RuntimeError(f"Failed to read LiverTox archive {self._archive_path}") from exc

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
