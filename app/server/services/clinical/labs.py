from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, Literal

from common.prompts.lab_extraction import (
    CLINICAL_LAB_EXTRACTION_SYSTEM_PROMPT,
    LOCAL_LAB_EXTRACTION_SYSTEM_PROMPT,
    build_lab_extraction_user_prompt,
)
from common.utils.logger import logger
from services.llm.runtime_config import LLMRuntimeConfig
from services.llm.generation_policy import GenerationPurpose
from configurations.startup import get_server_settings
from domain.clinical.entities import (
    ClinicalLabEntry,
    LiverInjuryOnsetContext,
    PatientData,
    PatientLabTimeline,
)
from domain.clinical.extras import LabExtractionPayload
from domain.clinical.extractor_contracts import (
    LocalLabExtractionPayload,
)
from services.catalogs.runtime import get_reference_catalog_snapshot
from services.clinical.lab_normalization import get_marker_aliases
from services.clinical import labs_extraction
from services.llm.client_runtime import ensure_runtime_client
from services.llm.provider_factory import select_llm_provider
from services.text.vocabulary import get_text_normalization_snapshot


###############################################################################
def _sanitize_optional_text(
    value: str | float | int | None,
    *,
    max_length: int,
) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    return text[:max_length]


###############################################################################
RATE_LIMIT_WAIT_HINT_RE = re.compile(
    r"please\s+try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)
NUMERIC_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
DATE_RE = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b"
)
VALUE_UNIT_RE = re.compile(
    r"(?P<value>[-+]?\d+(?:[.,]\d+)?)\s*(?P<unit>u/l|ui/l|µmol/l|umol/l|mg/dl|ml/min(?:/1\.73m2)?)?",
    re.IGNORECASE,
)
RELATIVE_TIME_VALUE_PREFIX_RE = re.compile(
    r"(?:^|[\s.;,:(-])(?:day|week|month|year|giorno|settimana|mese|anno|d)\s*$",
    re.IGNORECASE,
)
SINGLE_VALUE_MARKERS = frozenset({"CR", "EGFR", "INR", "ALB"})

###############################################################################
HEPATIC_PATTERN_RE = re.compile(
    r"\b(?:hepatic\s+pattern|injury\s+pattern|pattern)\s*[:=]?\s*(hepatocellular|cholestatic|mixed|indeterminate)\b",
    re.IGNORECASE,
)
RUCAM_SCORE_TEXT_RE = re.compile(
    r"\brucam\b\s*(?:score)?\s*[:=]?\s*(-?\d{1,2})\b",
    re.IGNORECASE,
)


###############################################################################
def normalize_lab_marker(marker_name: str, aliases: dict[str, str]) -> str:
    normalized = (marker_name or "").strip().casefold()
    return aliases.get(normalized, marker_name)


###############################################################################
class ClinicalLabExtractor:
    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        client: Any | None = None,
        temperature: float = 0.0,
        timeout_s: float = get_server_settings().runtime.parser_llm_timeout,
    ) -> None:
        runtime_settings = get_server_settings().runtime
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.minimum_timeout_floor_s = float(
            getattr(runtime_settings, "minimum_llm_timeout", 1.0)
        )
        self.cloud_llm_timeout_cap_s = float(
            getattr(runtime_settings, "cloud_llm_timeout_cap", self.timeout_s)
        )
        self.local_llm_timeout_cap_s = float(
            getattr(runtime_settings, "local_llm_timeout_cap", self.timeout_s)
        )
        self.LOCAL_LLM_CHUNK_TIMEOUT_CAP_S = self.local_llm_timeout_cap_s
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
    def minimum_timeout_s_default() -> float:
        return float(getattr(get_server_settings().runtime, "minimum_llm_timeout", 1.0))

    # -------------------------------------------------------------------------
    @staticmethod
    def cloud_timeout_cap_s_default() -> float:
        return float(
            getattr(get_server_settings().runtime, "cloud_llm_timeout_cap", 1.0)
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def local_timeout_cap_s_default() -> float:
        return float(
            getattr(get_server_settings().runtime, "local_llm_timeout_cap", 1.0)
        )

    # -------------------------------------------------------------------------
    def resolve_request_timeout_s(self) -> float:
        timeout_floor = max(self.minimum_timeout_floor_s, float(self.timeout_s))
        timeout_cap = (
            self.cloud_llm_timeout_cap_s
            if LLMRuntimeConfig.is_cloud_enabled()
            else float(self.LOCAL_LLM_CHUNK_TIMEOUT_CAP_S)
        )
        return min(timeout_floor, max(timeout_cap, self.minimum_timeout_floor_s))

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
        for raw_line in self.iter_logical_lab_lines(text):
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
    def iter_logical_lab_lines(self, text: str) -> list[str]:
        logical_lines: list[str] = []
        current = ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            has_marker = bool(self.find_marker_matches(line.casefold()))
            if has_marker:
                if current:
                    logical_lines.append(current)
                current = line
                continue
            if current and self.line_looks_like_lab_continuation(line):
                current = f"{current} {line}"
                continue
            if current:
                logical_lines.append(current)
                current = ""
            logical_lines.append(line)
        if current:
            logical_lines.append(current)
        return logical_lines

    # -------------------------------------------------------------------------
    @staticmethod
    def line_looks_like_lab_continuation(line: str) -> bool:
        lowered = line.casefold()
        has_number = NUMERIC_RE.search(line) is not None
        has_unit = any(
            token in lowered
            for token in ("u/l", "ui/l", "umol/l", "µmol/l", "mg/dl", "ml/min")
        )
        has_temporal_word = any(
            token in lowered for token in ("zenit", "rialzo", "fluttuante", "range")
        )
        return has_number and (has_unit or has_temporal_word)

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
        marker_matches = self.find_marker_matches(normalized_line)
        for index, marker_match in enumerate(marker_matches):
            canonical, alias_token, marker_position = marker_match
            next_position = (
                marker_matches[index + 1][2]
                if index + 1 < len(marker_matches)
                else len(line)
            )
            segment = line[marker_position:next_position]
            values = self.extract_values_from_marker_segment(
                segment,
                marker=canonical,
            )
            if not values:
                continue
            upper_limit = self.extract_upper_limit(segment)
            for value, value_text, unit in values:
                entries.append(
                    ClinicalLabEntry(
                        marker_name=canonical,
                        value=value,
                        value_text=value_text,
                        unit=unit,
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
        for alias in sorted(aliases, key=len, reverse=True):
            token = alias.casefold()
            if re.search(rf"\b{re.escape(token)}\b", text):
                return token
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def find_marker_matches(text: str) -> list[tuple[str, str, int]]:
        matches: list[tuple[str, str, int]] = []
        occupied: list[range] = []
        alias_items: list[tuple[str, str]] = []
        for canonical, aliases in get_marker_aliases().items():
            for alias in aliases:
                alias_items.append((canonical, alias.casefold()))
        for canonical, alias in sorted(
            alias_items, key=lambda item: len(item[1]), reverse=True
        ):
            for match in re.finditer(rf"\b{re.escape(alias)}\b", text):
                span_range = range(match.start(), match.end())
                if any(
                    match.start() < item.stop and match.end() > item.start
                    for item in occupied
                ):
                    continue
                matches.append((canonical, alias, match.start()))
                occupied.append(span_range)
        return sorted(matches, key=lambda item: item[2])

    # -------------------------------------------------------------------------
    def extract_values_from_marker_segment(
        self,
        segment: str,
        *,
        marker: str,
    ) -> list[tuple[float, str, str | None]]:
        date_spans = [match.span() for match in DATE_RE.finditer(segment)]
        upper_limit_spans = [
            match.span(1)
            for pattern in (
                r"\bULN\b\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)",
                r"upper\s+limit(?:\s+normal)?\s*[:=]?\s*([0-9]+(?:[.,][0-9]+)?)",
            )
            for match in re.finditer(pattern, segment, flags=re.IGNORECASE)
        ]
        values: list[tuple[float, str, str | None]] = []
        seen: set[tuple[str, str | None]] = set()
        for match in VALUE_UNIT_RE.finditer(segment):
            start, end = match.span("value")
            if any(
                start < date_end and end > date_start
                for date_start, date_end in date_spans
            ):
                continue
            if any(
                start < limit_end and end > limit_start
                for limit_start, limit_end in upper_limit_spans
            ):
                continue
            if self.value_looks_like_relative_timepoint(segment[:start]):
                continue
            raw_value = match.group("value")
            parsed = self.parse_numeric(raw_value)
            if parsed is None:
                continue
            unit = match.group("unit")
            normalized_unit = unit.strip() if unit else None
            key = (raw_value.replace(",", "."), normalized_unit)
            if key in seen:
                continue
            seen.add(key)
            values.append((parsed, raw_value, normalized_unit))
            if marker in SINGLE_VALUE_MARKERS:
                break
        return values

    # -------------------------------------------------------------------------
    @staticmethod
    def value_looks_like_relative_timepoint(prefix: str) -> bool:
        return RELATIVE_TIME_VALUE_PREFIX_RE.search(prefix[-32:]) is not None

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
        for canonical, aliases in get_marker_aliases().items():
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
        for fmt in (
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%Y-%m-%d",
            "%d.%m.%Y",
            "%Y.%m.%d",
        ):
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
        except (TypeError, ValueError):
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
    def is_local_runtime(provider: str | None) -> bool:
        return (provider or "").strip().lower() == "ollama"

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_local_payload(
        parsed: LocalLabExtractionPayload,
    ) -> LabExtractionPayload:
        onset_context = None
        if parsed.onset_context is not None:
            raw_onset_basis = (
                (parsed.onset_context.onset_basis or "unknown").strip().lower()
            )
            if raw_onset_basis == "first_symptom":
                onset_basis: Literal[
                    "first_symptom",
                    "first_abnormal_lab",
                    "visit_proxy",
                    "unknown",
                ] = "first_symptom"
            elif raw_onset_basis == "first_abnormal_lab":
                onset_basis = "first_abnormal_lab"
            elif raw_onset_basis == "visit_proxy":
                onset_basis = "visit_proxy"
            else:
                onset_basis = "unknown"
            onset_context = LiverInjuryOnsetContext(
                onset_date=_sanitize_optional_text(
                    parsed.onset_context.onset_date,
                    max_length=120,
                ),
                onset_basis=onset_basis,
                evidence=_sanitize_optional_text(
                    parsed.onset_context.evidence,
                    max_length=500,
                ),
            )
        return LabExtractionPayload(
            entries=[
                ClinicalLabEntry(
                    marker_name=entry.marker_name,
                    value_text=_sanitize_optional_text(
                        entry.value_text, max_length=100
                    ),
                    unit=_sanitize_optional_text(entry.unit, max_length=50),
                    sample_date=_sanitize_optional_text(
                        entry.sample_date, max_length=120
                    ),
                    evidence=_sanitize_optional_text(entry.evidence, max_length=500),
                    source="laboratory_analysis",
                )
                for entry in parsed.entries
            ],
            onset_context=onset_context,
        )

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
    async def llm_extract_full_text(
        self,
        *,
        text: str,
        reinforced: bool,
        expected_candidates: list[ClinicalLabEntry] | None = None,
        validation_feedback: list[str] | None = None,
        previous_wrong_output: str | None = None,
    ) -> LabExtractionPayload:
        candidates_text = self.format_expected_candidates(expected_candidates or [])
        user_prompt = build_lab_extraction_user_prompt(
            source_text=text,
            reinforced=reinforced,
            candidate_checklist=candidates_text,
            validation_feedback=validation_feedback,
            previous_output=previous_wrong_output,
        )
        parsed: LabExtractionPayload | None = None
        if self.client is None:
            raise RuntimeError("LLM client is not initialized for lab extraction")
        active_provider = self.forced_provider or self.client_provider or "ollama"
        use_local_schema = self.is_local_runtime(active_provider)
        schema: type[LabExtractionPayload] | type[LocalLabExtractionPayload] = (
            LocalLabExtractionPayload if use_local_schema else LabExtractionPayload
        )
        system_prompt = (
            LOCAL_LAB_EXTRACTION_SYSTEM_PROMPT
            if use_local_schema
            else CLINICAL_LAB_EXTRACTION_SYSTEM_PROMPT
        )
        request_timeout_s = self.resolve_request_timeout_s()
        for attempt in range(1, self.extraction_retry_attempts + 1):
            try:
                raw_parsed = await asyncio.wait_for(
                    self.client.llm_structured_call(
                        model=self.model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        schema=schema,
                        purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
                        use_json_mode=True,
                        max_repair_attempts=1,
                    ),
                    timeout=request_timeout_s,
                )
                parsed = (
                    self.normalize_local_payload(raw_parsed)
                    if use_local_schema
                    else raw_parsed
                )
                break
            except Exception as exc:
                if attempt >= self.extraction_retry_attempts:
                    raise
                delay = self.retry_backoff_seconds(attempt, exc=exc)
                logger.warning(
                    (
                        "Retrying clinical lab extraction "
                        "(attempt %d/%d, delay %.2fs): %s"
                    ),
                    attempt,
                    self.extraction_retry_attempts,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        if parsed is None:
            raise RuntimeError("Failed to extract clinical labs from full text")
        return parsed

    # -------------------------------------------------------------------------
    @staticmethod
    def format_expected_candidates(entries: list[ClinicalLabEntry]) -> str:
        lines: list[str] = []
        for entry in entries[:30]:
            parts = [
                entry.marker_name,
                str(entry.value_text or entry.value or ""),
            ]
            if entry.unit:
                parts.append(entry.unit)
            if entry.sample_date:
                parts.append(f"date={entry.sample_date}")
            if entry.evidence:
                parts.append(f"evidence={entry.evidence[:160]}")
            line = " | ".join(part for part in parts if part)
            if line:
                lines.append(f"- {line}")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    def validate_lab_entries_against_candidates(
        self,
        entries: list[ClinicalLabEntry],
        candidates: list[ClinicalLabEntry],
    ) -> tuple[list[str], list[ClinicalLabEntry]]:
        normalized_entries = [
            prepared
            for entry in entries
            if (prepared := self.normalize_entry(entry, visit_date=None)) is not None
        ]
        present = {
            (
                entry.marker_name.upper(),
                str(entry.value_text or entry.value or "").replace(",", "."),
            )
            for entry in normalized_entries
        }
        missing = [
            candidate
            for candidate in candidates
            if (
                candidate.marker_name.upper(),
                str(candidate.value_text or candidate.value or "").replace(",", "."),
            )
            not in present
        ]
        feedback: list[str] = []
        if missing:
            feedback.append(
                "Missing grounded lab candidates: "
                + "; ".join(
                    f"{entry.marker_name} {entry.value_text or entry.value}"
                    for entry in missing[:12]
                )
            )
        for entry in normalized_entries:
            evidence = (entry.evidence or "").replace(",", ".")
            value_text = str(entry.value_text or entry.value or "").replace(",", ".")
            if value_text and evidence and value_text not in evidence:
                feedback.append(
                    f"{entry.marker_name} value {value_text} is not present in its evidence snippet."
                )
        return feedback, missing

    # -------------------------------------------------------------------------
    async def extract_from_payload(
        self,
        payload: PatientData,
        *,
        already_cleaned: bool = False,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[PatientLabTimeline, LiverInjuryOnsetContext | None]:
        return await labs_extraction.extract_from_payload(
            self,
            payload,
            already_cleaned=already_cleaned,
            progress_callback=progress_callback,
        )

    # -------------------------------------------------------------------------
    async def extract_from_payload_with_audit(
        self,
        payload: PatientData,
        *,
        already_cleaned: bool = False,
        progress_callback: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        return await labs_extraction.extract_from_payload_with_audit(
            self,
            payload,
            already_cleaned=already_cleaned,
            progress_callback=progress_callback,
        )
