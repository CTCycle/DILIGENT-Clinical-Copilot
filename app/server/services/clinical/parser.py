from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import Callable
from datetime import date
from typing import Any, Literal, cast

from services.llm.prompts import (
    ANAMNESIS_DRUG_EXTRACTION_PROMPT,
    DRUG_EXTRACTION_PROMPT,
)
from services.llm.client_runtime import ensure_runtime_client
from services.llm.provider_factory import select_llm_provider
from domain.clinical.entities import (
    DrugEntry,
    PatientDrugs,
)
from configurations.startup import server_settings
from configurations.llm_configs import LLMRuntimeConfig
from services.text.normalization import normalize_token
from services.clinical.drug_blocks import isolate_drug_blocks
from services.clinical.parser_extraction import (
    BRACKET_TRAIL_RE,
    BULLET_RE,
    DATE_LIKE_SCHEDULE_RE,
    DOSE_CUE_RE,
    DOSAGE_TEMPORAL_SPLIT_RE,
    NAME_TEMPORAL_SPLIT_RE,
    ROUTE_PATTERNS,
    SCHEDULE_RE,
    START_DATE_RE,
    START_EVENT_RE,
    SUSPENSION_DATE_RE,
    SUSPENSION_EVENT_RE,
    SUSPENSION_RE,
    TRAILING_ROUTE_TOKEN_RE,
)
from services.clinical.parser_prompting import ANAMNESIS_CHUNK_MAX_CHARS
from services.clinical.parser_validation import (
    NON_DRUG_CONTAINS,
    NON_DRUG_EXACT_NAMES,
    NON_DRUG_PREFIXES,
    NON_THERAPY_LINE_PREFIXES,
    WEEKDAY_TOKENS,
)
from common.utils.logger import logger
from services.text.vocabulary import get_text_normalization_snapshot
from common.utils.patterns import (
    FORM_DESCRIPTORS,
    FORM_TOKENS,
    UNIT_TOKENS,
)


###############################################################################
class DrugsParser:
    LLM_CLIENT_NOT_INITIALIZED_ERROR = (
        "LLM client is not initialized for drug extraction"
    )
    SCHEDULE_RE = SCHEDULE_RE
    DATE_LIKE_SCHEDULE_RE = DATE_LIKE_SCHEDULE_RE
    BULLET_RE = BULLET_RE
    BRACKET_TRAIL_RE = BRACKET_TRAIL_RE
    SUSPENSION_RE = SUSPENSION_RE
    SUSPENSION_DATE_RE = SUSPENSION_DATE_RE
    START_DATE_RE = START_DATE_RE
    ROUTE_PATTERNS = ROUTE_PATTERNS
    DOSE_CUE_RE = DOSE_CUE_RE
    DOSAGE_TEMPORAL_SPLIT_RE = DOSAGE_TEMPORAL_SPLIT_RE
    NAME_TEMPORAL_SPLIT_RE = NAME_TEMPORAL_SPLIT_RE
    TRAILING_ROUTE_TOKEN_RE = TRAILING_ROUTE_TOKEN_RE
    START_EVENT_RE = START_EVENT_RE
    SUSPENSION_EVENT_RE = SUSPENSION_EVENT_RE
    ANAMNESIS_CHUNK_MAX_CHARS = ANAMNESIS_CHUNK_MAX_CHARS
    NON_DRUG_EXACT_NAMES = NON_DRUG_EXACT_NAMES
    NON_DRUG_PREFIXES = NON_DRUG_PREFIXES
    NON_DRUG_CONTAINS = NON_DRUG_CONTAINS
    WEEKDAY_TOKENS = WEEKDAY_TOKENS
    NON_THERAPY_LINE_PREFIXES = NON_THERAPY_LINE_PREFIXES

    def __init__(
        self,
        *,
        client: Any | None = None,
        temperature: float = 0.0,
        timeout_s: float = server_settings.runtime.default_llm_timeout,
    ) -> None:
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.client: Any | None = client
        self.model: str = ""
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
        lines: list[str] = []
        for raw_line in normalized.split("\n"):
            stripped = raw_line.strip()
            if not stripped:
                continue
            stripped = self.BULLET_RE.sub("", stripped)
            if not stripped:
                continue
            lines.append(stripped)
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    def conservative_prepare_drug_section_text(self, text: str | None) -> str:
        prepared = self.clean_text(text)
        if not prepared:
            return ""
        return "\n".join(line.rstrip() for line in prepared.split("\n") if line.rstrip())

    # -------------------------------------------------------------------------
    def parse_drug_list(self, text: str | None) -> PatientDrugs:
        cleaned_text = self.clean_text(text)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.extract_drugs_from_therapy(cleaned_text))
        raise RuntimeError(
            "parse_drug_list cannot be used inside a running event loop; use"
            " 'await extract_drugs_from_therapy(...)' instead."
        )

    # -------------------------------------------------------------------------
    async def extract_drugs_from_therapy(
        self,
        text: str | None,
        *,
        already_cleaned: bool = False,
        progress_callback: Callable[[float], None] | None = None,
    ) -> PatientDrugs:
        cleaned = (
            (text or "")
            if already_cleaned
            else self.conservative_prepare_drug_section_text(text)
        )
        if not cleaned:
            return PatientDrugs(entries=[])
        self.emit_progress(progress_callback, 0.0)
        combined = await self.extract_drugs_from_therapy_hybrid(
            cleaned,
            progress_callback=progress_callback,
        )
        self.emit_progress(progress_callback, 1.0)
        return PatientDrugs(entries=combined)

    async def extract_drugs_from_therapy_hybrid(
        self,
        cleaned: str,
        *,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[DrugEntry]:
        lines = [
            block.text.strip()
            for block in isolate_drug_blocks(cleaned)
            if block.text.strip() and not self.is_non_therapy_line(block.text.strip())
        ]
        parsed, fallback = self.rule_based_parse(lines)
        deterministic_entries = [entry for _, entry in parsed]
        if not fallback:
            self.emit_progress(progress_callback, 1.0)
            return self.deduplicate_drug_entries(deterministic_entries)

        if self.client is None or not hasattr(self.client, "llm_structured_call"):
            self.emit_progress(progress_callback, 1.0)
            return self.deduplicate_drug_entries(deterministic_entries)

        fallback_lines = [line for _, line in fallback]
        try:
            structured = await self.llm_extract_drugs(
                fallback_lines,
                progress_callback=progress_callback,
                progress_start=0.2,
                progress_span=0.8,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Therapy LLM extraction failed for %s unresolved fragments; using deterministic entries only: %s",
                len(fallback_lines),
                exc,
            )
            structured = PatientDrugs(entries=[])

        llm_entries: list[DrugEntry] = []
        for index, entry in enumerate(structured.entries):
            raw_line = fallback_lines[index] if index < len(fallback_lines) else ""
            post_processed = self.post_process_llm_entry(
                entry,
                raw_line=raw_line,
                source="therapy",
                historical_flag=False,
            )
            if post_processed is not None:
                llm_entries.append(post_processed)
        return self.deduplicate_drug_entries([*deterministic_entries, *llm_entries])

    # -------------------------------------------------------------------------
    async def llm_extract_drugs(
        self,
        lines: list[str],
        *,
        progress_callback: Callable[[float], None] | None = None,
        progress_start: float = 0.0,
        progress_span: float = 1.0,
    ) -> PatientDrugs:
        if self.client is None:
            raise RuntimeError(self.LLM_CLIENT_NOT_INITIALIZED_ERROR)

        if not lines:
            return PatientDrugs(entries=[])

        single_entry_prompt = (
            f"{DRUG_EXTRACTION_PROMPT.strip()}\n\n"
            "For this task, return a PatientDrugs object whose `entries` array contains "
            "exactly one DrugEntry describing the drug provided in the user prompt."
        )

        entries: list[DrugEntry] = []
        total_lines = max(len(lines), 1)
        bounded_start = min(1.0, max(0.0, float(progress_start)))
        bounded_span = max(0.0, float(progress_span))
        for index, line in enumerate(lines, start=1):
            # Request a deterministic parse for each drug-sized chunk
            parsed = await asyncio.wait_for(
                self.client.llm_structured_call(
                    model=self.model,
                    system_prompt=single_entry_prompt,
                    user_prompt=(
                        "Extract the structured representation for the following drug entry:\n"
                        f"{line}"
                    ),
                    schema=PatientDrugs,
                    temperature=self.temperature,
                    use_json_mode=True,
                    max_repair_attempts=1,
                ),
                timeout=max(5.0, float(self.timeout_s)),
            )
            entries.extend(parsed.entries)
            self.emit_progress(
                progress_callback,
                bounded_start + ((index / total_lines) * bounded_span),
            )

        return PatientDrugs(entries=entries)

    async def llm_extract_drugs_from_section(
        self,
        text: str,
        *,
        source: Literal["anamnesis", "therapy"],
        historical_flag: bool,
        progress_callback: Callable[[float], None] | None = None,
    ) -> PatientDrugs:
        await self.ensure_client()
        if self.client is None:
            raise RuntimeError(self.LLM_CLIENT_NOT_INITIALIZED_ERROR)
        if not text.strip():
            return PatientDrugs(entries=[])
        if not hasattr(self.client, "llm_structured_call"):
            logger.warning(
                "LLM structured call unavailable; falling back to deterministic parsing for %s section.",
                source,
            )
            entries: list[DrugEntry]
            if source == "anamnesis":
                entries = self.extract_drugs_from_anamnesis_rule_based(text)
            else:
                lines = [
                    block.text.strip()
                    for block in isolate_drug_blocks(text)
                    if block.text.strip() and not self.is_non_therapy_line(block.text.strip())
                ]
                entries = []
                for line in lines:
                    candidate = self.parse_line(line)
                    normalized_entry = self.normalize_entry(
                        candidate,
                        source=source,
                        historical_flag=historical_flag,
                    )
                    if normalized_entry is not None:
                        entries.append(normalized_entry)
            self.emit_progress(progress_callback, 1.0)
            return PatientDrugs(entries=entries)

        if source == "anamnesis":
            chunks = self.chunk_anamnesis_text(text)
            parsed_entries: list[DrugEntry] = []
            for index, chunk in enumerate(chunks, start=1):
                parsed = await asyncio.wait_for(
                    self.client.llm_structured_call(
                        model=self.model,
                        system_prompt=ANAMNESIS_DRUG_EXTRACTION_PROMPT.strip(),
                        user_prompt=(
                            "Extract all drugs mentioned in the following patient anamnesis chunk:\n\n"
                            f"[Chunk {index}/{len(chunks)}]\n{chunk}"
                        ),
                        schema=PatientDrugs,
                        temperature=self.temperature,
                        use_json_mode=True,
                        max_repair_attempts=1,
                    ),
                    timeout=max(5.0, float(self.timeout_s)),
                )
                parsed_entries.extend(parsed.entries)
                self.emit_progress(progress_callback, index / max(len(chunks), 1))
            normalized_candidates: list[DrugEntry | None] = [
                self.normalize_entry(
                    entry,
                    source=source,
                    historical_flag=historical_flag,
                )
                for entry in parsed_entries
            ]
            normalized = [entry for entry in normalized_candidates if entry is not None]
            return PatientDrugs(entries=normalized)

        lines = [
            block.text.strip()
            for block in isolate_drug_blocks(text)
            if block.text.strip() and not self.is_non_therapy_line(block.text.strip())
        ]
        parsed, fallback = self.rule_based_parse(lines)
        normalized = [entry for _, entry in parsed]
        if not fallback:
            return PatientDrugs(entries=normalized)
        fallback_lines = [line for _, line in fallback]
        structured = await self.llm_extract_drugs(
            fallback_lines,
            progress_callback=progress_callback,
            progress_start=0.2,
            progress_span=0.8,
        )
        for index, entry in enumerate(structured.entries):
            raw_line = fallback_lines[index] if index < len(fallback_lines) else ""
            post_processed = self.post_process_llm_entry(
                entry,
                raw_line=raw_line,
                source=source,
                historical_flag=historical_flag,
            )
            if post_processed is not None:
                normalized.append(post_processed)
        return PatientDrugs(entries=normalized)

    def chunk_anamnesis_text(self, text: str) -> list[str]:
        normalized = self.clean_text(text)
        if not normalized:
            return []
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        chunks: list[str] = []
        current_lines: list[str] = []
        current_size = 0
        for line in lines:
            line_size = len(line) + 1
            if (
                current_lines
                and current_size + line_size > self.ANAMNESIS_CHUNK_MAX_CHARS
            ):
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
    def extract_drugs_from_anamnesis_rule_based(
        self, anamnesis: str
    ) -> list[DrugEntry]:
        lines = [line.strip() for line in anamnesis.split("\n") if line.strip()]
        entries: list[DrugEntry] = []
        for line in lines:
            if not self.is_likely_medication_line(line):
                continue
            candidate = self.parse_line(line)
            normalized = self.normalize_entry(
                candidate,
                source="anamnesis",
                historical_flag=True,
            )
            if normalized is not None:
                entries.append(normalized)
        return entries

    # -------------------------------------------------------------------------
    def is_likely_medication_line(self, line: str) -> bool:
        lowered = line.lower()
        if self.SCHEDULE_RE.search(line):
            return True
        if self.DOSE_CUE_RE.search(line):
            return True
        if self.SUSPENSION_RE.search(line):
            return True
        if self.START_DATE_RE.search(line):
            return True
        if self.detect_route(line):
            return True
        if any(
            token in lowered
            for token in (" mg", " ml", " mcg", " cpr", " caps", " fiala", " sir ")
        ):
            return True
        return False

    # -------------------------------------------------------------------------
    def deduplicate_drug_entries(self, entries: list[DrugEntry]) -> list[DrugEntry]:
        selected: dict[tuple[str, str | None], DrugEntry] = {}
        order: list[tuple[str, str | None]] = []
        for entry in entries:
            normalized_name = normalize_token(entry.name)
            if not normalized_name:
                continue
            key = (normalized_name, entry.source)
            existing = selected.get(key)
            if existing is None:
                selected[key] = entry
                order.append(key)
                continue
            if self.entry_information_score(entry) > self.entry_information_score(
                existing
            ):
                selected[key] = entry
        return [selected[key] for key in order if key in selected]

    # -------------------------------------------------------------------------
    def entry_information_score(self, entry: DrugEntry) -> int:
        score = 1
        for field_name in (
            "dosage",
            "administration_mode",
            "route",
            "administration_pattern",
            "suspension_status",
            "suspension_date",
            "therapy_start_status",
            "therapy_start_date",
        ):
            value = getattr(entry, field_name, None)
            if value is not None and value != []:
                score += 1
        return score

    # -------------------------------------------------------------------------
    async def extract_drugs_from_anamnesis(
        self,
        anamnesis: str | None,
        *,
        already_cleaned: bool = False,
        progress_callback: Callable[[float], None] | None = None,
    ) -> PatientDrugs:
        """
        Extract drug mentions from free-text anamnesis using the LLM.

        Unlike the therapy list extraction (which uses rules first),
        anamnesis extraction is primarily LLM-based with a deterministic
        fallback for medication-like lines.
        """
        if not anamnesis or not anamnesis.strip():
            return PatientDrugs(entries=[])

        cleaned_anamnesis = (
            (anamnesis or "")
            if already_cleaned
            else self.conservative_prepare_drug_section_text(anamnesis)
        )
        self.emit_progress(progress_callback, 0.0)
        raw_llm_entries = 0
        try:
            structured = await self.llm_extract_drugs_from_section(
                cleaned_anamnesis,
                source="anamnesis",
                historical_flag=True,
                progress_callback=progress_callback,
            )
            raw_llm_entries = len(structured.entries)
            merged_entries = self.deduplicate_drug_entries(structured.entries)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Anamnesis LLM extraction failed; using deterministic fallback: %s",
                exc,
            )
            merged_entries = []
        if not merged_entries:
            merged_entries = self.extract_drugs_from_anamnesis_rule_based(
                cleaned_anamnesis
            )
        logger.info(
            "Anamnesis extraction produced %s normalized drugs (%s raw LLM entries)",
            len(merged_entries),
            raw_llm_entries,
        )
        self.emit_progress(progress_callback, 1.0)
        return PatientDrugs(entries=merged_entries)

    # -------------------------------------------------------------------------
    def rule_based_parse(
        self, lines: list[str]
    ) -> tuple[list[tuple[int, DrugEntry]], list[tuple[int, str]]]:
        parsed: list[tuple[int, DrugEntry]] = []
        fallback: list[tuple[int, str]] = []
        for index, line in enumerate(lines):
            entry = self.parse_line(line)
            if entry is None:
                fallback.append((index, line))
                continue
            parsed.append((index, entry))
        return parsed, fallback

    # -------------------------------------------------------------------------
    def parse_line(self, line: str) -> DrugEntry | None:
        if not self.has_alpha_token(line):
            return None

        schedule_match = self.SCHEDULE_RE.search(line)
        if schedule_match and self.is_date_like_schedule(
            schedule_match.group("schedule")
        ):
            schedule_match = None
        schedule_text = schedule_match.group("schedule") if schedule_match else None
        schedule_values = self.parse_schedule(schedule_text) if schedule_text else []
        administration_pattern = (
            self.normalize_schedule_pattern(schedule_text) if schedule_text else None
        )
        if schedule_match:
            before = line[: schedule_match.start()].strip(" ,;:\t")
            tail = line[schedule_match.end() :].strip()
        else:
            before = line.strip(" ,;:\t")
            tail = line
        bracket_match = self.BRACKET_TRAIL_RE.search(before)
        if bracket_match:
            before = before[: bracket_match.start()].strip()
        before = self.strip_temporal_name_tail(before)
        name, dosage, administration_mode = self.split_heading(before)
        if not name:
            name = before or line.strip()
        route = self.detect_route(line)
        suspension_status, suspension_date = self.detect_suspension(line, tail)
        start_status, start_date = self.detect_start(line, tail)
        candidate = DrugEntry(
            name=name,
            dosage=dosage,
            administration_mode=administration_mode,
            route=route,
            administration_pattern=administration_pattern,
            daytime_administration=schedule_values,
            suspension_status=suspension_status,
            suspension_date=suspension_date,
            therapy_start_status=start_status,
            therapy_start_date=start_date,
        )
        return self.normalize_entry(candidate, source="therapy", historical_flag=False)

    # -------------------------------------------------------------------------
    def parse_schedule(self, text: str | None) -> list[float]:
        if not text:
            return []
        if self.is_date_like_schedule(text):
            return []
        slots: list[float] = []
        for token in re.split(r"[-\s]+", text):
            normalized = token.strip()
            if not normalized:
                continue
            normalized = normalized.replace(",", ".")
            try:
                value = float(normalized)
            except ValueError:
                continue
            slots.append(value)
            if len(slots) >= 4:
                break
        return slots

    # -------------------------------------------------------------------------
    def normalize_schedule_pattern(self, text: str | None) -> str | None:
        if not text:
            return None
        if self.is_date_like_schedule(text):
            return None
        parts: list[str] = []
        for token in text.split("-"):
            normalized = token.strip().replace(",", ".")
            if not normalized:
                continue
            try:
                value = float(normalized)
                if value.is_integer():
                    parts.append(str(int(value)))
                else:
                    parts.append(f"{value:g}")
            except ValueError:
                parts.append(normalized)
        return "-".join(parts) if parts else None

    # -------------------------------------------------------------------------
    def is_date_like_schedule(self, text: str | None) -> bool:
        if not text:
            return False
        return bool(self.DATE_LIKE_SCHEDULE_RE.fullmatch(text.strip()))

    # -------------------------------------------------------------------------
    def split_heading(self, text: str) -> tuple[str | None, str | None, str | None]:
        if not text:
            return None, None, None
        tokens = text.split()
        if not tokens:
            return None, None, None
        first_numeric = None
        for idx, token in enumerate(tokens):
            if self.token_has_numeric(token):
                first_numeric = idx
                break
        if first_numeric is None:
            name = self.strip_trailing_route_token(" ".join(tokens).strip())
            return name or None, None, None
        name_tokens = tokens[:first_numeric]
        remainder = tokens[first_numeric:]
        mode_tokens: list[str] = []
        self.extract_mode_from_prefix(name_tokens, mode_tokens)
        dosage_tokens: list[str] = []
        for token in remainder:
            normalized = normalize_token(token)
            if normalized in FORM_TOKENS:
                mode_tokens.append(token)
                continue
            if normalized in FORM_DESCRIPTORS:
                mode_tokens.append(token)
                continue
            if (
                self.token_has_numeric(token)
                or normalized in UNIT_TOKENS
                or "/" in token
            ):
                dosage_tokens.append(token)
                continue
            if dosage_tokens:
                dosage_tokens.append(token)
                continue
            if normalized in {"per", "os"}:
                mode_tokens.append(token)
                continue
            name_tokens.append(token)
        if not dosage_tokens and remainder:
            dosage_tokens = remainder
        name = " ".join(name_tokens).strip() or None
        name = self.strip_trailing_route_token(name)
        dosage = " ".join(dosage_tokens).strip() or None
        administration_mode = " ".join(mode_tokens).strip() or None
        return name, dosage, administration_mode

    # -------------------------------------------------------------------------
    def strip_temporal_name_tail(self, value: str | None) -> str:
        if not value:
            return ""
        stripped = re.sub(
            r"\([^)]*(?:linea\s+precedente|sospes[oaie]|discontinued?|stopp?ed)[^)]*\)\s*$",
            "",
            value,
            flags=re.IGNORECASE,
        )
        stripped = self.NAME_TEMPORAL_SPLIT_RE.sub("", stripped)
        return stripped.strip(" ,;:\t")

    # -------------------------------------------------------------------------
    def strip_trailing_route_token(self, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = self.TRAILING_ROUTE_TOKEN_RE.sub("", value).strip(" ,;:\t")
        return stripped or None

    # -------------------------------------------------------------------------
    def is_non_therapy_line(self, line: str) -> bool:
        normalized = self.normalize_filter_key(line)
        return any(
            normalized.startswith(prefix) for prefix in self.NON_THERAPY_LINE_PREFIXES
        )

    # -------------------------------------------------------------------------
    def extract_mode_from_prefix(
        self, name_tokens: list[str], mode_tokens: list[str]
    ) -> None:
        idx = len(name_tokens)
        trailing: list[str] = []
        saw_form = False
        while idx > 0:
            token = name_tokens[idx - 1]
            normalized = normalize_token(token)
            if normalized in FORM_TOKENS:
                saw_form = True
                trailing.append(token)
                idx -= 1
                continue
            if normalized in FORM_DESCRIPTORS:
                trailing.append(token)
                idx -= 1
                continue
            break
        if not saw_form:
            return
        del name_tokens[idx:]
        trailing.reverse()
        mode_tokens.extend(trailing)

    # -------------------------------------------------------------------------
    def token_has_numeric(self, token: str) -> bool:
        return any(ch.isdigit() for ch in token)

    # -------------------------------------------------------------------------
    def detect_route(self, text: str) -> str | None:
        normalized = text.strip()
        if not normalized:
            return None
        for route_name, route_re in self.ROUTE_PATTERNS:
            if route_re.search(normalized):
                return route_name
        return None

    # -------------------------------------------------------------------------
    def detect_suspension(
        self, full_line: str, tail: str
    ) -> tuple[bool | None, str | None]:
        status = True if self.SUSPENSION_RE.search(full_line) else None
        date_match = self.SUSPENSION_DATE_RE.search(
            tail
        ) or self.SUSPENSION_DATE_RE.search(full_line)
        if date_match:
            date_value = self.normalize_date_token(date_match.group("date"))
        elif status:
            date_value = self.extract_event_detail(
                full_line,
                event_re=self.SUSPENSION_EVENT_RE,
            )
        else:
            date_value = None
        return status, date_value

    # -------------------------------------------------------------------------
    def detect_start(self, full_line: str, tail: str) -> tuple[bool | None, str | None]:
        for segment in (tail, full_line):
            if not segment:
                continue
            for match in self.START_DATE_RE.finditer(segment):
                prefix_end = match.start()
                if prefix_end >= 0:
                    context = segment[max(0, prefix_end - 15) : prefix_end].lower()
                    if "sospes" in context:
                        continue
                date_token = match.group("date")
                normalized = self.normalize_date_token(date_token)
                return True, normalized
        detail = self.extract_event_detail(
            tail or full_line,
            event_re=self.START_EVENT_RE,
        )
        if detail:
            return True, detail
        return None, None

    # -------------------------------------------------------------------------
    def extract_event_detail(
        self,
        text: str,
        *,
        event_re: re.Pattern[str],
    ) -> str | None:
        match = event_re.search(text)
        if not match:
            return None
        tail = match.groupdict().get("tail") or ""
        raw = tail.strip(" ,;:.")
        if not raw:
            return None
        return self.sanitize_text_field(raw)

    # -------------------------------------------------------------------------
    def has_alpha_token(self, text: str | None) -> bool:
        if not text:
            return False
        return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", text))

    # -------------------------------------------------------------------------
    def sanitize_name(self, value: str | None) -> str | None:
        if value is None:
            return None
        raw_text = str(value)
        if "\n" in raw_text or "\r" in raw_text:
            return None
        normalized = re.sub(r"\s+", " ", raw_text).strip(" \t,;:.-")
        if not normalized:
            return None
        if len(normalized.split()) > 8:
            return None
        if not self.has_alpha_token(normalized):
            return None
        if re.search(r"[.;:!?]{2,}", normalized):
            return None
        return normalized

    # -------------------------------------------------------------------------
    def sanitize_text_field(self, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = re.sub(r"\s+", " ", str(value)).strip()
        return normalized or None

    # -------------------------------------------------------------------------
    def sanitize_dosage_field(self, value: str | None) -> str | None:
        cleaned = self.sanitize_text_field(value)
        if cleaned is None:
            return None
        stripped = self.DOSAGE_TEMPORAL_SPLIT_RE.sub("", cleaned).strip(" ,;")
        return stripped or None

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_filter_key(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        normalized = normalized.encode("ascii", "ignore").decode("ascii")
        normalized = normalized.lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    # -------------------------------------------------------------------------
    def is_non_drug_fragment_name(self, value: str) -> bool:
        normalized = self.normalize_filter_key(value)
        snapshot = get_text_normalization_snapshot()
        non_drug_exact = set(self.NON_DRUG_EXACT_NAMES) | set(
            snapshot.drug_non_mentions
        )
        weekday_tokens = set(self.WEEKDAY_TOKENS) | set(snapshot.drug_weekday_words)
        duration_words = set(snapshot.drug_duration_words)
        if not normalized:
            return True
        if normalized in non_drug_exact:
            return True
        if any(normalized.startswith(prefix) for prefix in self.NON_DRUG_PREFIXES):
            return True
        if any(fragment in normalized for fragment in self.NON_DRUG_CONTAINS):
            return True
        tokens = normalized.split()
        if tokens and all(
            token.isdigit() or token in duration_words for token in tokens
        ):
            return True
        if tokens and all(token in weekday_tokens for token in tokens):
            return True
        return False

    # -------------------------------------------------------------------------
    def derive_temporal_classification(self, entry: DrugEntry) -> str:
        schedule_present = bool(entry.administration_pattern) or bool(
            entry.daytime_administration
        )
        start_present = entry.therapy_start_status is not None or bool(
            entry.therapy_start_date
        )
        suspension_present = entry.suspension_status is not None or bool(
            entry.suspension_date
        )
        if schedule_present or start_present or suspension_present:
            return "temporal_known"
        return "temporal_uncertain"

    def drug_entry_has_temporal_information(self, entry: DrugEntry) -> bool:
        if entry.temporal_classification == "temporal_known":
            return True
        return bool(
            entry.administration_pattern
            or entry.daytime_administration
            or entry.therapy_start_date
            or entry.therapy_start_status is not None
            or entry.suspension_date
            or entry.suspension_status is not None
        )

    # -------------------------------------------------------------------------
    def normalize_entry(
        self,
        entry: DrugEntry | None,
        *,
        source: Literal["therapy", "anamnesis"],
        historical_flag: bool,
    ) -> DrugEntry | None:
        if entry is None:
            return None
        name = self.sanitize_name(entry.name)
        if name is None:
            return None
        if self.is_non_drug_fragment_name(name):
            return None
        normalized = entry.model_copy(deep=True)
        normalized.name = name
        normalized.dosage = self.sanitize_dosage_field(normalized.dosage)
        normalized.administration_mode = self.sanitize_text_field(
            normalized.administration_mode
        )
        normalized.route = self.sanitize_text_field(normalized.route)
        normalized.administration_pattern = self.sanitize_text_field(
            normalized.administration_pattern
        )
        normalized.suspension_date = self.sanitize_text_field(
            normalized.suspension_date
        )
        normalized.therapy_start_date = self.sanitize_text_field(
            normalized.therapy_start_date
        )
        normalized.source = source
        normalized.historical_flag = historical_flag
        normalized.temporal_classification = cast(
            Literal["temporal_known", "temporal_uncertain"] | None,
            self.derive_temporal_classification(normalized),
        )
        return normalized

    # -------------------------------------------------------------------------
    def enrich_entry_from_line(self, entry: DrugEntry, raw_line: str) -> DrugEntry:
        normalized = entry.model_copy(deep=True)
        schedule_match = self.SCHEDULE_RE.search(raw_line)
        if schedule_match and self.is_date_like_schedule(
            schedule_match.group("schedule")
        ):
            schedule_match = None
        if schedule_match:
            schedule_text = schedule_match.group("schedule")
            schedule_values = self.parse_schedule(schedule_text)
            if schedule_values:
                normalized.daytime_administration = schedule_values
            schedule_pattern = self.normalize_schedule_pattern(schedule_text)
            if schedule_pattern:
                normalized.administration_pattern = schedule_pattern
        route = self.detect_route(raw_line)
        if route:
            normalized.route = route
        suspension_status, suspension_date = self.detect_suspension(raw_line, raw_line)
        if suspension_status is not None:
            normalized.suspension_status = suspension_status
        if suspension_date:
            normalized.suspension_date = suspension_date
        start_status, start_date = self.detect_start(raw_line, raw_line)
        if start_status is not None:
            normalized.therapy_start_status = start_status
        if start_date:
            normalized.therapy_start_date = start_date
        return normalized

    # -------------------------------------------------------------------------
    def post_process_llm_entry(
        self,
        entry: DrugEntry | None,
        *,
        raw_line: str,
        source: Literal["therapy", "anamnesis"],
        historical_flag: bool,
    ) -> DrugEntry | None:
        if entry is None:
            return None
        enriched = self.enrich_entry_from_line(entry, raw_line)
        return self.normalize_entry(
            enriched,
            source=source,
            historical_flag=historical_flag,
        )

    # -------------------------------------------------------------------------
    def normalize_date_token(self, token: str | None) -> str | None:
        if not token:
            return None
        stripped = token.strip(" .,:;")
        match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})(?:[./-](\d{4}))?", stripped)
        if not match:
            return stripped or None
        day, month, year = match.groups()
        if year:
            try:
                return date(int(year), int(month), int(day)).isoformat()
            except ValueError:
                return stripped
        return f"{day.zfill(2)}.{month.zfill(2)}"

