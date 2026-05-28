from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, Literal

from common.utils.logger import logger
from configurations.llm_configs import LLMRuntimeConfig
from configurations.startup import get_server_settings
from domain.clinical.entities import (
    ClinicalLabEntry,
    LiverInjuryOnsetContext,
    PatientData,
    PatientLabTimeline,
)
from domain.clinical.extras import LabExtractionPayload
from services.catalogs.runtime import get_reference_catalog_snapshot
from services.llm.client_runtime import ensure_runtime_client
from services.llm.prompts import CLINICAL_LAB_EXTRACTION_PROMPT
from services.llm.provider_factory import select_llm_provider
from services.text.vocabulary import get_text_normalization_snapshot

###############################################################################
RATE_LIMIT_WAIT_HINT_RE = re.compile(
    r"please\s+try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)
NUMERIC_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
DATE_RE = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})\b"
)


def _load_marker_aliases() -> dict[str, tuple[str, ...]]:
    snapshot = get_reference_catalog_snapshot()
    entries = snapshot.entries("clinical_extraction", "laboratory_markers")
    by_key: dict[str, list[str]] = {}
    for entry in entries:
        by_key.setdefault(entry.key.upper(), []).append(entry.value.casefold())
    if by_key:
        return {key: tuple(dict.fromkeys(values)) for key, values in by_key.items()}
    return {
        "ALT": ("alt", "alat", "gpt"),
        "AST": ("ast", "asat", "got"),
        "ALP": ("alp", "alkp", "alkaline phosphatase"),
        "TBIL": ("tbil", "total bilirubin", "bilirubin total", "bilirubin"),
        "DBIL": ("dbil", "direct bilirubin", "bilirubin direct"),
        "GGT": ("ggt", "gamma gt", "gamma-glutamyl transferase"),
        "INR": ("inr",),
        "ALB": ("albumin", "alb"),
    }


MARKER_ALIASES: dict[str, tuple[str, ...]] = _load_marker_aliases()
HEPATIC_PATTERN_RE = re.compile(
    r"\b(?:hepatic\s+pattern|injury\s+pattern|pattern)\s*[:=]?\s*(hepatocellular|cholestatic|mixed|indeterminate)\b",
    re.IGNORECASE,
)
RUCAM_SCORE_TEXT_RE = re.compile(
    r"\brucam\b\s*(?:score)?\s*[:=]?\s*(-?\d{1,2})\b",
    re.IGNORECASE,
)


def normalize_lab_marker(marker_name: str, aliases: dict[str, str]) -> str:
    normalized = (marker_name or "").strip().casefold()
    return aliases.get(normalized, marker_name)


###############################################################################
class ClinicalLabExtractor:
    CHUNK_MAX_CHARS = 2600
    LOCAL_LLM_CHUNK_TIMEOUT_CAP_S = 45.0
    CLOUD_LLM_CHUNK_TIMEOUT_CAP_S = 30.0

    def __init__(
        self,
        *,
        client: Any | None = None,
        temperature: float = 0.0,
        timeout_s: float = get_server_settings().runtime.default_llm_timeout,
    ) -> None:
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.client: Any | None = client
        self.model: str = ""
        # Prefer fast deterministic fallback over long retry loops.
        self.extraction_retry_attempts = 1
        self.client_lock = asyncio.Lock()
        self.client_loop_id: int | None = None
        self.forced_provider: str | None = None
        self.forced_model: str | None = None
        if client is None:
            self.client_provider: str | None = None
            self.runtime_revision = -1
        else:
            self.client_provider = "injected"
            self.runtime_revision = LLMRuntimeConfig.get_revision()

    # -------------------------------------------------------------------------
    async def ensure_client(self) -> None:
        revision = LLMRuntimeConfig.get_revision()
        resolved_provider, resolved_model = LLMRuntimeConfig.resolve_provider_and_model(
            "parser"
        )
        provider = self.forced_provider or resolved_provider
        model = self.forced_model or resolved_model
        await ensure_runtime_client(
            self,
            provider=provider,
            model=model,
            revision=revision,
            client_factory=lambda selected_provider, selected_model: (
                select_llm_provider(
                    provider=selected_provider,
                    default_model=selected_model,
                    timeout_s=self.timeout_s,
                    max_retries=0,
                )
            ),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def emit_progress(
        progress_callback: Callable[[float], None] | None,
        fraction: float,
    ) -> None:
        if progress_callback is None:
            return
        bounded_fraction = min(1.0, max(0.0, float(fraction)))
        progress_callback(bounded_fraction)

    # -------------------------------------------------------------------------
    def clean_text(self, text: str | None) -> str:
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    def extract_explicit_hepatic_pattern(self, text: str) -> str | None:
        if not text:
            return None
        match = HEPATIC_PATTERN_RE.search(text)
        if match is None:
            return None
        value = match.group(1).strip().lower()
        if value in {"hepatocellular", "cholestatic", "mixed", "indeterminate"}:
            return value
        return None

    # -------------------------------------------------------------------------
    def extract_explicit_rucam_score(self, text: str) -> int | None:
        if not text:
            return None
        match = RUCAM_SCORE_TEXT_RE.search(text)
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    def calculate_hepatic_pattern_from_lab_timeline(
        self,
        timeline: PatientLabTimeline,
    ) -> str | None:
        alt_entries = [
            entry
            for entry in timeline.entries
            if entry.marker_name == "ALT"
            and entry.value is not None
            and entry.upper_limit_normal is not None
            and entry.upper_limit_normal > 0
        ]
        alp_entries = [
            entry
            for entry in timeline.entries
            if entry.marker_name == "ALP"
            and entry.value is not None
            and entry.upper_limit_normal is not None
            and entry.upper_limit_normal > 0
        ]
        if not alt_entries or not alp_entries:
            return None
        peak_alt = max(
            (
                float(entry.value) / float(entry.upper_limit_normal)
                for entry in alt_entries
                if entry.value is not None and entry.upper_limit_normal is not None
            ),
            default=0.0,
        )
        peak_alp = max(
            (
                float(entry.value) / float(entry.upper_limit_normal)
                for entry in alp_entries
                if entry.value is not None and entry.upper_limit_normal is not None
            ),
            default=0.0,
        )
        if peak_alp <= 0:
            return None
        r_ratio = peak_alt / peak_alp
        if r_ratio >= 5.0:
            return "hepatocellular"
        if r_ratio <= 2.0:
            return "cholestatic"
        return "mixed"

    # -------------------------------------------------------------------------
    def chunk_text(self, text: str) -> list[str]:
        normalized = self.clean_text(text)
        if not normalized:
            return []
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        chunks: list[str] = []
        current_lines: list[str] = []
        current_size = 0
        for line in lines:
            line_size = len(line) + 1
            if current_lines and current_size + line_size > self.CHUNK_MAX_CHARS:
                chunks.append("\n".join(current_lines))
                current_lines = [line]
                current_size = line_size
                continue
            current_lines.append(line)
            current_size += line_size
        if current_lines:
            chunks.append("\n".join(current_lines))
        return chunks or [normalized]

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_numeric(raw: str | None) -> float | None:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        text = str(raw).strip()
        if not text:
            return None
        match = NUMERIC_RE.search(text)
        if not match:
            return None
        try:
            return float(match.group().replace(",", "."))
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    def extract_entries_from_text(
        self,
        *,
        text: str,
        source: Literal["laboratory_analysis", "anamnesis", "merged"],
        visit_date: date | None,
    ) -> list[ClinicalLabEntry]:
        if not text:
            return []
        entries: list[ClinicalLabEntry] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            sample_date = self.extract_date_from_text(line, visit_date=visit_date)
            entries.extend(
                self.extract_entries_from_line(
                    line=line,
                    source=source,
                    sample_date=sample_date,
                )
            )
        return entries

    # -------------------------------------------------------------------------
    def extract_entries_from_line(
        self,
        *,
        line: str,
        source: Literal["laboratory_analysis", "anamnesis", "merged"],
        sample_date: str | None,
    ) -> list[ClinicalLabEntry]:
        entries: list[ClinicalLabEntry] = []
        normalized_line = line.casefold()
        for canonical, aliases in MARKER_ALIASES.items():
            alias_token = self.find_marker_token(normalized_line, aliases)
            if alias_token is None:
                continue
            marker_position = normalized_line.find(alias_token)
            if marker_position < 0:
                continue
            tail = line[marker_position:]
            value = self.parse_numeric(tail)
            if value is None:
                continue
            upper_limit = self.extract_upper_limit(tail)
            entries.append(
                ClinicalLabEntry(
                    marker_name=canonical,
                    value=value,
                    value_text=str(value),
                    upper_limit_normal=upper_limit,
                    upper_limit_text=str(upper_limit)
                    if upper_limit is not None
                    else None,
                    sample_date=sample_date,
                    evidence=line[:500],
                    source=source,
                )
            )
        return entries

    # -------------------------------------------------------------------------
    @staticmethod
    def find_marker_token(text: str, aliases: tuple[str, ...]) -> str | None:
        for alias in aliases:
            token = alias.casefold()
            if re.search(rf"\b{re.escape(token)}\b", text):
                return token
        return None

    # -------------------------------------------------------------------------
    def extract_date_from_text(
        self, text: str, *, visit_date: date | None
    ) -> str | None:
        match = DATE_RE.search(text)
        if match is None:
            return None
        return self.normalize_date_with_visit_year(match.group(0), visit_date)

    # -------------------------------------------------------------------------
    def extract_upper_limit(self, text: str) -> float | None:
        patterns = (
            r"\bULN\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)",
            r"upper\s+limit(?:\s+normal)?\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match is None:
                continue
            parsed = self.parse_numeric(match.group(1))
            if parsed is not None:
                return parsed
        return None

    # -------------------------------------------------------------------------
    def normalize_marker_name(self, raw: str | None) -> str:
        text = (raw or "").strip().lower()
        if not text:
            return "UNKNOWN"
        text = normalize_lab_marker(
            text, get_text_normalization_snapshot().lab_marker_aliases
        ).lower()
        for canonical, aliases in MARKER_ALIASES.items():
            if text == canonical.lower():
                return canonical
            for alias in aliases:
                if alias in text:
                    return canonical
        return text.upper()

    # -------------------------------------------------------------------------
    @staticmethod
    def try_parse_date(value: str) -> date | None:
        cleaned = value.strip()
        if not cleaned:
            return None
        iso_candidate = cleaned.replace(".", "-").replace("/", "-")
        try:
            return date.fromisoformat(iso_candidate)
        except ValueError:
            pass
        for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d", "%d.%m.%Y", "%Y.%m.%d"):
            try:
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
        return None

    # -------------------------------------------------------------------------
    def normalize_date_with_visit_year(
        self,
        raw_date: str | None,
        visit_date: date | None,
    ) -> str | None:
        if raw_date is None:
            return None
        text = str(raw_date).strip()
        if not text:
            return None
        normalized = text.replace("/", "-").replace(".", "-").replace(",", "-")
        tokens = [token for token in normalized.split("-") if token]
        candidates: list[str] = []
        if visit_date is not None and len(tokens) == 2:
            day, month = tokens
            candidates.extend(
                [
                    f"{day.zfill(2)}-{month.zfill(2)}-{visit_date.year}",
                    f"{month.zfill(2)}-{day.zfill(2)}-{visit_date.year}",
                    f"{visit_date.year}-{month.zfill(2)}-{day.zfill(2)}",
                ]
            )
        candidates.extend(["-".join(tokens), text, normalized])
        checked: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in checked:
                continue
            checked.add(candidate)
            parsed = self.try_parse_date(candidate)
            if parsed is not None:
                return parsed.isoformat()
        return text

    # -------------------------------------------------------------------------
    @staticmethod
    def dedupe_key(entry: ClinicalLabEntry) -> tuple[str, str, str, str]:
        value_token = (
            f"{entry.value:.6f}"
            if isinstance(entry.value, (int, float))
            else (entry.value_text or "")
        )
        date_token = (entry.sample_date or "").strip().lower()
        return (
            entry.marker_name.strip().upper(),
            date_token,
            value_token.strip().lower(),
            entry.source,
        )

    # -------------------------------------------------------------------------
    @classmethod
    def lab_entry_sort_key(cls, item: ClinicalLabEntry) -> tuple[int, str, str]:
        if item.sample_date:
            parsed = cls.try_parse_date(item.sample_date)
            if parsed is not None:
                return (0, parsed.isoformat(), item.marker_name)
        return (1, item.relative_time or "", item.marker_name)

    # -------------------------------------------------------------------------
    def normalize_entry(
        self,
        entry: ClinicalLabEntry,
        *,
        visit_date: date | None,
    ) -> ClinicalLabEntry | None:
        marker = self.normalize_marker_name(entry.marker_name)
        if marker == "UNKNOWN":
            return None
        normalized_date = self.normalize_date_with_visit_year(
            entry.sample_date, visit_date
        )
        return ClinicalLabEntry(
            marker_name=marker,
            value=entry.value
            if entry.value is not None
            else self.parse_numeric(entry.value_text),
            value_text=entry.value_text,
            unit=entry.unit,
            upper_limit_normal=(
                entry.upper_limit_normal
                if entry.upper_limit_normal is not None
                else self.parse_numeric(entry.upper_limit_text)
            ),
            upper_limit_text=entry.upper_limit_text,
            sample_date=normalized_date,
            relative_time=(entry.relative_time or "").strip() or None,
            evidence=(entry.evidence or "").strip() or None,
            source=entry.source,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def extract_rate_limit_wait_hint_seconds(exc: Exception) -> float | None:
        message = str(exc)
        match = RATE_LIMIT_WAIT_HINT_RE.search(message)
        if match is None:
            return None
        try:
            parsed = float(match.group(1))
        except TypeError, ValueError:
            return None
        if parsed <= 0:
            return None
        return min(parsed + 0.25, 30.0)

    # -------------------------------------------------------------------------
    def retry_backoff_seconds(
        self, attempt: int, *, exc: Exception | None = None
    ) -> float:
        if exc is not None:
            hinted_wait = self.extract_rate_limit_wait_hint_seconds(exc)
            if hinted_wait is not None:
                return hinted_wait
        normalized_attempt = max(int(attempt), 1)
        return min(2.0, 0.5 * (2 ** (normalized_attempt - 1)))

    # -------------------------------------------------------------------------
    @staticmethod
    def has_explicit_lab_signal(text: str) -> bool:
        lowered = (text or "").casefold()
        if not lowered:
            return False
        snapshot = get_reference_catalog_snapshot()
        marker_tokens = tuple(
            value.casefold()
            for value in snapshot.values("clinical_extraction", "laboratory_markers")
        ) or (
            "alat",
            "alt",
            "asat",
            "ast",
            "ggt",
            "alp",
            "bilirubina",
            "bilirubin",
            "inr",
            "albumina",
            "albumin",
        )
        unit_tokens = tuple(
            value.casefold()
            for value in snapshot.values("clinical_extraction", "laboratory_units")
        ) or ("u/l", "ui/l", "micromol", "µmol", "mg/dl", "g/l")
        has_marker = any(token in lowered for token in marker_tokens)
        has_unit = any(token in lowered for token in unit_tokens)
        has_number = NUMERIC_RE.search(lowered) is not None
        return has_marker and has_number and has_unit

    # -------------------------------------------------------------------------
    async def llm_extract_chunk(
        self,
        *,
        chunk: str,
        chunk_index: int,
        total_chunks: int,
        reinforced: bool,
    ) -> LabExtractionPayload:
        user_prompt = (
            "Extract longitudinal liver-related labs and onset clues from this clinical chunk.\n"
            f"[Chunk {chunk_index}/{total_chunks}]\n{chunk}"
        )
        if reinforced:
            user_prompt = (
                f"{user_prompt}\n\n"
                "Important: this chunk contains explicit lab values. "
                "Extract every liver-related marker/value pair found (e.g., ALAT/ALT, ASAT/AST, GGT, ALP, "
                "bilirubina totale/diretta), preserving unit text and available dates."
            )
        parsed: LabExtractionPayload | None = None
        if self.client is None:
            raise RuntimeError("LLM client is not initialized for lab extraction")
        chunk_timeout_s = min(
            max(5.0, float(self.timeout_s)),
            self.CLOUD_LLM_CHUNK_TIMEOUT_CAP_S
            if LLMRuntimeConfig.is_cloud_enabled()
            else self.LOCAL_LLM_CHUNK_TIMEOUT_CAP_S,
        )
        for attempt in range(1, self.extraction_retry_attempts + 1):
            try:
                parsed = await asyncio.wait_for(
                    self.client.llm_structured_call(
                        model=self.model,
                        system_prompt=CLINICAL_LAB_EXTRACTION_PROMPT.strip(),
                        user_prompt=user_prompt,
                        schema=LabExtractionPayload,
                        temperature=self.temperature,
                        use_json_mode=True,
                        max_repair_attempts=1,
                    ),
                    timeout=chunk_timeout_s,
                )
                break
            except Exception as exc:
                if attempt >= self.extraction_retry_attempts:
                    raise
                delay = self.retry_backoff_seconds(attempt, exc=exc)
                logger.warning(
                    (
                        "Retrying clinical lab extraction for chunk %d/%d "
                        "(attempt %d/%d, delay %.2fs): %s"
                    ),
                    chunk_index,
                    total_chunks,
                    attempt,
                    self.extraction_retry_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        if parsed is None:
            raise RuntimeError("Failed to extract clinical labs from chunk")
        return parsed

    # -------------------------------------------------------------------------
    async def extract_from_payload(
        self,
        payload: PatientData,
        *,
        already_cleaned: bool = False,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[PatientLabTimeline, LiverInjuryOnsetContext | None]:
        primary_labs_text = (
            (payload.laboratory_analysis or "")
            if already_cleaned
            else self.clean_text(payload.laboratory_analysis)
        )
        deterministic_entries: list[ClinicalLabEntry] = []
        timeline_entries: list[ClinicalLabEntry] = []
        onset_context: LiverInjuryOnsetContext | None = None
        self.emit_progress(progress_callback, 0.0)

        deterministic_entries.extend(
            self.extract_entries_from_text(
                text=primary_labs_text,
                source="laboratory_analysis",
                visit_date=payload.visit_date,
            )
        )
        self.emit_progress(progress_callback, 0.2)

        merged_source_text = primary_labs_text
        if merged_source_text:
            try:
                await self.ensure_client()
                if self.client is None:
                    raise RuntimeError(
                        "LLM client is not initialized for lab extraction"
                    )
                chunks = self.chunk_text(merged_source_text)
                llm_entries: list[ClinicalLabEntry] = []
                llm_onset: LiverInjuryOnsetContext | None = None
                llm_unavailable = False
                for index, chunk in enumerate(chunks, start=1):
                    try:
                        parsed = await self.llm_extract_chunk(
                            chunk=chunk,
                            chunk_index=index,
                            total_chunks=len(chunks),
                            reinforced=False,
                        )
                    except Exception as exc:
                        logger.warning(
                            (
                                "Clinical lab extraction unavailable for chunk %d/%d "
                                "after %d attempts; using deterministic parser output only: %s"
                            ),
                            index,
                            len(chunks),
                            self.extraction_retry_attempts,
                            exc,
                        )
                        parsed = LabExtractionPayload(entries=[], onset_context=None)
                        llm_unavailable = True

                    # If the first LLM pass returns empty despite clear lab cues, retry once
                    # with a reinforced instruction before accepting an empty chunk.
                    if not parsed.entries and self.has_explicit_lab_signal(chunk):
                        try:
                            reinforced = await self.llm_extract_chunk(
                                chunk=chunk,
                                chunk_index=index,
                                total_chunks=len(chunks),
                                reinforced=True,
                            )
                            if reinforced.entries:
                                parsed = reinforced
                        except Exception as exc:
                            logger.warning(
                                "Reinforced clinical lab extraction failed for chunk %d/%d: %s",
                                index,
                                len(chunks),
                                exc,
                            )

                    llm_entries.extend(parsed.entries)
                    if llm_onset is None and parsed.onset_context is not None:
                        llm_onset = parsed.onset_context
                    self.emit_progress(
                        progress_callback, 0.2 + ((index / max(len(chunks), 1)) * 0.5)
                    )
                timeline_entries.extend(llm_entries)
                if llm_unavailable or (not llm_entries and deterministic_entries):
                    if not llm_entries and deterministic_entries:
                        logger.warning(
                            "LLM lab extraction returned no entries despite detectable lab markers; using deterministic lab parser output."
                        )
                    timeline_entries.extend(deterministic_entries)
                onset_context = llm_onset
            except Exception as exc:
                logger.warning(
                    "Clinical lab extraction unavailable; using deterministic parser output only: %s",
                    exc,
                )
                timeline_entries.extend(deterministic_entries)
        else:
            timeline_entries.extend(deterministic_entries)

        self.emit_progress(progress_callback, 0.85)

        normalized: list[ClinicalLabEntry] = []
        seen: set[tuple[str, str, str, str]] = set()
        for entry in timeline_entries:
            prepared = self.normalize_entry(entry, visit_date=payload.visit_date)
            if prepared is None:
                continue
            key = self.dedupe_key(prepared)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(prepared)

        normalized.sort(key=self.lab_entry_sort_key)
        self.emit_progress(progress_callback, 1.0)
        return PatientLabTimeline(entries=normalized), onset_context
