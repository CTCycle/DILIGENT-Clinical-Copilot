from __future__ import annotations

import asyncio
import contextlib
import re
import unicodedata
from datetime import date
from typing import Any, Literal

from DILIGENT.server.models.prompts import (
    ANAMNESIS_DRUG_EXTRACTION_PROMPT,
    DRUG_EXTRACTION_PROMPT,
)
from DILIGENT.server.models.providers import initialize_llm_client
from DILIGENT.server.entities.clinical import (
    DrugEntry,
    PatientDrugs,
)
from DILIGENT.server.configurations import LLMRuntimeConfig, server_settings
from DILIGENT.server.services.text.normalization import normalize_token
from DILIGENT.common.utils.logger import logger
from DILIGENT.common.utils.patterns import (
    DRUG_BRACKET_TRAIL_RE,
    DRUG_BULLET_RE,
    DRUG_SCHEDULE_RE,
    DRUG_START_DATE_RE,
    DRUG_SUSPENSION_DATE_RE,
    DRUG_SUSPENSION_RE,
    FORM_DESCRIPTORS,
    FORM_TOKENS,
    UNIT_TOKENS,
)


###############################################################################
class DrugsParser:
    SCHEDULE_RE = DRUG_SCHEDULE_RE
    BULLET_RE = DRUG_BULLET_RE
    BRACKET_TRAIL_RE = DRUG_BRACKET_TRAIL_RE
    SUSPENSION_RE = DRUG_SUSPENSION_RE
    SUSPENSION_DATE_RE = DRUG_SUSPENSION_DATE_RE
    START_DATE_RE = DRUG_START_DATE_RE
    ROUTE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("oral", re.compile(r"\b(?:po|p\.?o\.?|per\s+os|orale|oral)\b", re.IGNORECASE)),
        (
            "iv",
            re.compile(
                r"\b(?:iv|i\.?v\.?|ev|e\.?v\.?|endovenos[ao]?|intraven(?:ous|osa)?)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "im",
            re.compile(
                r"\b(?:im|i\.?m\.?|intramuscolar[ei]|intramuscular)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "sc",
            re.compile(
                r"\b(?:sc|s\.?c\.?|sottocutane[ao]?|subcut(?:aneous|anea)?)\b",
                re.IGNORECASE,
            ),
        ),
        ("topical", re.compile(r"\b(?:topical|topic[ao]|cutane[ao])\b", re.IGNORECASE)),
    )
    DOSE_CUE_RE = re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|ug|ml|u|ui|units?)\b",
        re.IGNORECASE,
    )
    ANAMNESIS_CHUNK_MAX_CHARS = 2400

    def __init__(
        self,
        *,
        client: Any | None = None,
        temperature: float = 0.0,
        timeout_s: float = server_settings.external_data.default_llm_timeout,
    ) -> None:
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.client: Any | None = client
        self.model: str = ""
        self.client_lock = asyncio.Lock()
        if client is None:
            self.client_provider: str | None = None
            self.runtime_revision = -1
        else:
            self.client_provider = "injected"
            self.runtime_revision = LLMRuntimeConfig.get_revision()

    # -------------------------------------------------------------------------
    async def ensure_client(self) -> None:
        async with self.client_lock:
            revision = LLMRuntimeConfig.get_revision()
            provider, model = LLMRuntimeConfig.resolve_provider_and_model("parser")
            if self.client_provider == "injected" and self.client is not None:
                self.model = model
                self.runtime_revision = revision
                return
            needs_refresh = (
                self.client is None
                or self.client_provider != provider
                or self.runtime_revision != revision
            )
            if needs_refresh:
                # Tear down stale clients so provider/model switches are honoured
                if self.client is not None:
                    with contextlib.suppress(Exception):
                        await self.client.close()
                self.client = initialize_llm_client(
                    purpose="parser", timeout_s=self.timeout_s
                )
                self.client_provider = provider
            self.runtime_revision = revision
            self.model = model
            if (
                self.client is not None
                and model
                and hasattr(self.client, "default_model")
            ):
                self.client.default_model = model  # type: ignore[attr-defined]

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
    def parse_drug_list(self, text: str | None) -> PatientDrugs:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.extract_drugs_from_therapy(text))
        raise RuntimeError(
            "parse_drug_list cannot be used inside a running event loop; use"
            " 'await extract_drugs_from_therapy(...)' instead."
        )

    # -------------------------------------------------------------------------
    async def extract_drugs_from_therapy(self, text: str | None) -> PatientDrugs:
        cleaned = self.clean_text(text)
        if not cleaned:
            return PatientDrugs(entries=[])
        lines = [
            segment
            for segment in (entry.strip() for entry in cleaned.split("\n"))
            if segment
        ]
        grouped_lines = self.group_lines_for_llm(lines)
        parsed_chunks, fallback_chunks = self.rule_based_parse(grouped_lines)
        ordered_entries: list[DrugEntry | None] = [None] * len(grouped_lines)
        for index, entry in parsed_chunks:
            normalized = self.normalize_entry(
                entry,
                source="therapy",
                historical_flag=False,
            )
            ordered_entries[index] = normalized

        if fallback_chunks:
            await self.ensure_client()
            if self.client is None:
                raise RuntimeError("LLM client is not initialized for drug extraction")
            try:
                # Ask the LLM for structured entries only for lines with true rule failures.
                structured = await self.llm_extract_drugs(
                    [line for _, line in fallback_chunks]
                )
            except Exception as exc:  # pragma: no cover - passthrough for visibility
                raise RuntimeError("Failed to extract drugs via LLM") from exc

            llm_entries = list(structured.entries)
            for offset, (target_index, raw_line) in enumerate(fallback_chunks):
                llm_entry = llm_entries[offset] if offset < len(llm_entries) else None
                normalized = self.post_process_llm_entry(
                    llm_entry,
                    raw_line=raw_line,
                    source="therapy",
                    historical_flag=False,
                )
                ordered_entries[target_index] = normalized
            for entry in llm_entries[len(fallback_chunks) :]:
                normalized = self.normalize_entry(
                    entry,
                    source="therapy",
                    historical_flag=False,
                )
                ordered_entries.append(normalized)

        combined = [entry for entry in ordered_entries if entry is not None]
        return PatientDrugs(entries=combined)

    # -------------------------------------------------------------------------
    async def llm_extract_drugs(self, lines: list[str]) -> PatientDrugs:
        if self.client is None:
            raise RuntimeError("LLM client is not initialized for drug extraction")

        if not lines:
            return PatientDrugs(entries=[])

        single_entry_prompt = (
            f"{DRUG_EXTRACTION_PROMPT.strip()}\n\n"
            "For this task, return a PatientDrugs object whose `entries` array contains "
            "exactly one DrugEntry describing the drug provided in the user prompt."
        )

        grouped_lines = self.group_lines_for_llm(lines)

        entries: list[DrugEntry] = []
        for line in grouped_lines:
            # Request a deterministic parse for each drug-sized chunk
            parsed = await self.client.llm_structured_call(
                model=self.model,
                system_prompt=single_entry_prompt,
                user_prompt=(
                    "Extract the structured representation for the following drug entry:\n"
                    f"{line}"
                ),
                schema=PatientDrugs,
                temperature=self.temperature,
                use_json_mode=True,
                max_repair_attempts=2,
            )
            entries.extend(parsed.entries)

        return PatientDrugs(entries=entries)

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
            if current_lines and current_size + line_size > self.ANAMNESIS_CHUNK_MAX_CHARS:
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
    def extract_drugs_from_anamnesis_rule_based(self, anamnesis: str) -> list[DrugEntry]:
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
        if any(token in lowered for token in (" mg", " ml", " mcg", " cpr", " caps", " fiala", " sir ")):
            return True
        return False

    # -------------------------------------------------------------------------
    def deduplicate_drug_entries(self, entries: list[DrugEntry]) -> list[DrugEntry]:
        selected: dict[str, DrugEntry] = {}
        order: list[str] = []
        for entry in entries:
            normalized_name = normalize_token(entry.name)
            if not normalized_name:
                continue
            existing = selected.get(normalized_name)
            if existing is None:
                selected[normalized_name] = entry
                order.append(normalized_name)
                continue
            if self.entry_information_score(entry) > self.entry_information_score(existing):
                selected[normalized_name] = entry
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
    async def extract_drugs_from_anamnesis(self, anamnesis: str | None) -> PatientDrugs:
        """
        Extract drug mentions from free-text anamnesis using the LLM.

        Unlike the therapy list extraction (which uses rules first),
        anamnesis extraction is primarily LLM-based with a deterministic
        fallback for medication-like lines.
        """
        if not anamnesis or not anamnesis.strip():
            return PatientDrugs(entries=[])

        await self.ensure_client()
        if self.client is None:
            raise RuntimeError("LLM client is not initialized for drug extraction")

        cleaned_anamnesis = self.clean_text(anamnesis)
        chunks = self.chunk_anamnesis_text(cleaned_anamnesis)
        try:
            parsed_entries: list[DrugEntry] = []
            for index, chunk in enumerate(chunks, start=1):
                parsed = await self.client.llm_structured_call(
                    model=self.model,
                    system_prompt=ANAMNESIS_DRUG_EXTRACTION_PROMPT.strip(),
                    user_prompt=(
                        "Extract all drugs mentioned in the following patient anamnesis chunk:\n\n"
                        f"[Chunk {index}/{len(chunks)}]\n{chunk}"
                    ),
                    schema=PatientDrugs,
                    temperature=self.temperature,
                    use_json_mode=True,
                    max_repair_attempts=2,
                )
                parsed_entries.extend(parsed.entries)
        except Exception as exc:
            raise RuntimeError("Failed to extract drugs from anamnesis via LLM") from exc

        entries: list[DrugEntry] = []
        for entry in parsed_entries:
            normalized = self.normalize_entry(
                entry,
                source="anamnesis",
                historical_flag=True,
            )
            if normalized is not None:
                entries.append(normalized)
        fallback_entries = self.extract_drugs_from_anamnesis_rule_based(cleaned_anamnesis)
        merged_entries = self.deduplicate_drug_entries([*entries, *fallback_entries])
        logger.info(
            "Anamnesis extraction produced %s normalized drugs (%s raw LLM entries, %s fallback candidates)",
            len(merged_entries),
            len(parsed_entries),
            len(fallback_entries),
        )
        return PatientDrugs(entries=merged_entries)

    # -------------------------------------------------------------------------
    def group_lines_for_llm(self, lines: list[str]) -> list[str]:
        grouped = [line.strip() for line in lines if line and line.strip()]
        return grouped if grouped else lines

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
            return " ".join(tokens).strip() or None, None, None
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
        dosage = " ".join(dosage_tokens).strip() or None
        administration_mode = " ".join(mode_tokens).strip() or None
        return name, dosage, administration_mode

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
        date_value = (
            self.normalize_date_token(date_match.group("date")) if date_match else None
        )
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
        return None, None

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
    def derive_temporal_classification(self, entry: DrugEntry) -> str:
        schedule_present = bool(entry.administration_pattern) or bool(
            entry.daytime_administration
        )
        start_present = (
            entry.therapy_start_status is not None or bool(entry.therapy_start_date)
        )
        suspension_present = (
            entry.suspension_status is not None or bool(entry.suspension_date)
        )
        if schedule_present or start_present or suspension_present:
            return "temporal_known"
        return "temporal_uncertain"

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
        normalized = entry.model_copy(deep=True)
        normalized.name = name
        normalized.dosage = self.sanitize_text_field(normalized.dosage)
        normalized.administration_mode = self.sanitize_text_field(
            normalized.administration_mode
        )
        normalized.route = self.sanitize_text_field(normalized.route)
        normalized.administration_pattern = self.sanitize_text_field(
            normalized.administration_pattern
        )
        normalized.suspension_date = self.sanitize_text_field(normalized.suspension_date)
        normalized.therapy_start_date = self.sanitize_text_field(
            normalized.therapy_start_date
        )
        normalized.source = source
        normalized.historical_flag = historical_flag
        normalized.temporal_classification = self.derive_temporal_classification(
            normalized
        )
        return normalized

    # -------------------------------------------------------------------------
    def enrich_entry_from_line(self, entry: DrugEntry, raw_line: str) -> DrugEntry:
        normalized = entry.model_copy(deep=True)
        schedule_match = self.SCHEDULE_RE.search(raw_line)
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

