from __future__ import annotations

import asyncio
import contextlib
import re
import unicodedata
from datetime import date
from typing import Any

from DILIGENT.src.app.server.models.prompts import DRUG_EXTRACTION_PROMPT
from DILIGENT.src.app.server.models.providers import initialize_llm_client
from DILIGENT.src.app.server.schemas.clinical import (
    DrugEntry,
    PatientDrugs,
)
from DILIGENT.src.packages.configurations import LLMRuntimeConfig, configurations
from DILIGENT.src.packages.patterns import (
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

    def __init__(
        self,
        *,
        client: Any | None = None,
        temperature: float = 0.0,
        timeout_s: float = configurations.server.external_data.default_llm_timeout,
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
            if not lines:
                lines.append(stripped)
                continue

            has_schedule = bool(self.SCHEDULE_RE.search(stripped))
            has_metadata = bool(
                self.SUSPENSION_RE.search(stripped)
                or self.SUSPENSION_DATE_RE.search(stripped)
                or self.START_DATE_RE.search(stripped)
            )

            if has_schedule:
                lines.append(stripped)
            elif has_metadata:
                # Attach metadata lines to the preceding drug entry
                lines[-1] = f"{lines[-1]} {stripped}"
            else:
                lines.append(stripped)
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    def parse_drug_list(self, text: str | None) -> PatientDrugs:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.extract_drug_list(text))
        raise RuntimeError(
            "parse_drug_list cannot be used inside a running event loop; use"
            " 'await extract_drug_list(...)' instead."
        )

    # -------------------------------------------------------------------------
    async def extract_drug_list(self, text: str | None) -> PatientDrugs:
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
        if not fallback_chunks:
            ordered = [entry for _, entry in parsed_chunks]
            return PatientDrugs(entries=ordered)

        await self.ensure_client()
        if self.client is None:
            raise RuntimeError("LLM client is not initialized for drug extraction")
        try:
            # Ask the LLM for a structured list following the PatientDrugs schema
            structured = await self.llm_extract_drugs(
                [line for _, line in fallback_chunks]
            )
        except Exception as exc:  # pragma: no cover - passthrough for visibility
            raise RuntimeError("Failed to extract drugs via LLM") from exc

        ordered_entries: list[DrugEntry | None] = [None] * len(grouped_lines)
        for index, entry in parsed_chunks:
            ordered_entries[index] = entry
        llm_entries = list(structured.entries)
        for offset, entry in enumerate(llm_entries):
            if offset < len(fallback_chunks):
                target_index = fallback_chunks[offset][0]
                ordered_entries[target_index] = entry
            else:
                ordered_entries.append(entry)
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

    # -------------------------------------------------------------------------
    def group_lines_for_llm(self, lines: list[str]) -> list[str]:
        grouped: list[str] = []
        metadata_buffer: list[str] = []
        prefix_buffer: list[str] = []
        for index, line in enumerate(lines):
            schedule_match = self.SCHEDULE_RE.search(line)
            has_schedule = bool(schedule_match)
            has_metadata = bool(
                self.SUSPENSION_RE.search(line)
                or self.SUSPENSION_DATE_RE.search(line)
                or self.START_DATE_RE.search(line)
            )
            if has_metadata and not has_schedule:
                metadata_buffer.append(line)
                continue

            has_numeric = bool(re.search(r"\d", line))
            if not has_numeric and not has_schedule:
                next_line = lines[index + 1] if index + 1 < len(lines) else ""
                next_has_schedule = (
                    bool(self.SCHEDULE_RE.search(next_line)) if next_line else False
                )
                next_has_numeric = (
                    bool(re.search(r"\d", next_line)) if next_line else False
                )
                if next_has_numeric or next_has_schedule:
                    prefix_buffer.append(line)
                    continue

            if grouped and schedule_match and not metadata_buffer and not prefix_buffer:
                prefix = line[: schedule_match.start()]
                prefix = self.BULLET_RE.sub("", prefix).strip(" \t,.;:-/")
                if not prefix:
                    grouped[-1] = f"{grouped[-1]} {line}".strip()
                    continue

            combined_parts = metadata_buffer + prefix_buffer + [line]
            # Deliver dosage, schedule, and annotations to the model as a single chunk
            combined = " ".join(part for part in combined_parts if part).strip()
            if combined:
                grouped.append(combined)
            else:
                grouped.append(line)
            metadata_buffer.clear()
            prefix_buffer.clear()

        leftover_parts = metadata_buffer + prefix_buffer
        if leftover_parts and grouped:
            # Append trailing metadata that never received its own schedule line
            grouped[-1] = f"{grouped[-1]} {' '.join(leftover_parts)}".strip()

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
        schedule_match = self.SCHEDULE_RE.search(line)
        if not schedule_match:
            return None
        schedule_text = schedule_match.group("schedule")
        schedule_values = self.parse_schedule(schedule_text)
        before = line[: schedule_match.start()].strip(" ,;:\t")
        tail = line[schedule_match.end() :].strip()
        bracket_match = self.BRACKET_TRAIL_RE.search(before)
        if bracket_match:
            before = before[: bracket_match.start()].strip()
        name, dosage, administration_mode = self.split_heading(before)
        if not name:
            name = before or line.strip()
        suspension_status, suspension_date = self.detect_suspension(line, tail)
        start_status, start_date = self.detect_start(line, tail)
        return DrugEntry(
            name=name,
            dosage=dosage,
            administration_mode=administration_mode,
            daytime_administration=schedule_values,
            suspension_status=suspension_status,
            suspension_date=suspension_date,
            therapy_start_status=start_status,
            therapy_start_date=start_date,
        )

    # -------------------------------------------------------------------------
    def parse_schedule(self, text: str) -> list[float]:
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
            if value.is_integer():
                slots.append(int(value))
            else:
                slots.append(value)
        if len(slots) >= 4:
            return slots[:4]
        return []

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
            normalized = self.normalize_token(token)
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
            normalized = self.normalize_token(token)
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
    def normalize_token(self, token: str) -> str:
        return re.sub(r"[.,;:]+$", "", token.lower())

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
    def normalize_date_token(self, token: str | None) -> str | None:
        if not token:
            return None
        stripped = token.strip(" .,:;")
        match = re.fullmatch(r"(\d{1,2})[./](\d{1,2})(?:[./](\d{4}))?", stripped)
        if not match:
            return stripped or None
        day, month, year = match.groups()
        if year:
            try:
                return date(int(year), int(month), int(day)).isoformat()
            except ValueError:
                return stripped
        return f"{day.zfill(2)}.{month.zfill(2)}"
