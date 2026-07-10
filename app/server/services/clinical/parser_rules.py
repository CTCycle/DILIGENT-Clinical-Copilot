from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from datetime import date
from typing import Literal, cast

from common.utils.logger import logger
from common.utils.patterns import (
    FORM_DESCRIPTORS,
    FORM_TOKENS,
    UNIT_TOKENS,
)
from domain.clinical.entities import (
    DrugEntry,
    PatientDrugs,
)
from services.clinical.deterministic_extraction import (
    DeterministicDrugExtractionResult,
    extract_regimen_drug_candidates,
    line_has_regimen_signal,
)
from common.utils.text_utils import normalize_token
from services.text.vocabulary import get_text_normalization_snapshot
from services.clinical.parser_host import ParserHost


class DrugRulesMixin(ParserHost):
    def extract_drugs_from_anamnesis_rule_based(
        self, anamnesis: str
    ) -> list[DrugEntry]:
        return self.extract_drugs_from_anamnesis_deterministic(anamnesis).entries

    # -------------------------------------------------------------------------
    def extract_drugs_from_anamnesis_deterministic(
        self, anamnesis: str
    ) -> DeterministicDrugExtractionResult:
        lines = [line.strip() for line in anamnesis.split("\n") if line.strip()]
        entries: list[DrugEntry] = []
        unresolved_lines: list[str] = []
        regimen_lines: list[str] = []
        for line in lines:
            if line_has_regimen_signal(line):
                regimen_lines.append(line)
                regimen_entries = extract_regimen_drug_candidates(
                    line,
                    normalize_date_token=self.normalize_date_token,
                    normalize_entry=self.normalize_entry,
                )
                if regimen_entries:
                    entries.extend(regimen_entries)
                    continue
            if self.is_likely_medication_line(line):
                candidate = self.parse_line(line)
                normalized = self.normalize_entry(
                    candidate,
                    source="anamnesis",
                    historical_flag=True,
                )
                if normalized is not None:
                    entries.append(normalized)
                    continue
            if line_has_regimen_signal(line) or re.search(
                r"\b(antibiotic|antibiotic[ao]|farmac|chemioterap|protocollo)\b",
                line,
                re.IGNORECASE,
            ):
                unresolved_lines.append(line)
        return DeterministicDrugExtractionResult(
            entries=self.deduplicate_drug_entries(entries),
            unresolved_lines=unresolved_lines,
            regimen_lines=regimen_lines,
        )

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
        deterministic_result = self.extract_drugs_from_anamnesis_deterministic(
            cleaned_anamnesis
        )

        fallback_entries: list[DrugEntry] = []
        for entry in deterministic_result.entries:
            grounded = self.attach_source_grounding(
                entry,
                source_text=cleaned_anamnesis,
                historical_flag=True,
                require_medication_syntax=False,
            )
            fallback_entries.append(grounded or entry)
        merged_entries: list[DrugEntry] = list(fallback_entries)
        raw_llm_entries = 0
        llm_input_text = cleaned_anamnesis.strip()
        if llm_input_text:
            try:
                structured = await self.llm_extract_drugs_from_section(
                    llm_input_text,
                    source="anamnesis",
                    historical_flag=True,
                    progress_callback=progress_callback,
                )
                raw_llm_entries = len(structured.entries)
                merged_entries = self.deduplicate_drug_entries(structured.entries)
            except Exception as exc:
                logger.warning(
                    "Anamnesis LLM enrichment failed; keeping deterministic extraction only: %s",
                    exc,
                )

        logger.info(
            "Anamnesis extraction produced %s normalized drugs (%s raw LLM entries, %s unresolved lines)",
            len(merged_entries),
            raw_llm_entries,
            len(deterministic_result.unresolved_lines),
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
        raw_text = self.BULLET_RE.sub("", raw_text)
        normalized = re.sub(r"\s+", " ", raw_text).strip(" \t,;:.-")
        normalized = re.sub(r"\s*\(=\s*$", "", normalized).strip(" \t,;:.-")
        normalized = self._strip_drug_form_suffixes(normalized).strip(" \t,;:.-")
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
    def extract_embedded_drug_name(self, value: str) -> str:
        if not value:
            return value
        contextual_match = re.search(
            r"\b(?:con|with)\s+([A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'/-]+(?:\s+[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'/-]+){0,3})\b",
            value,
        )
        if contextual_match and re.search(
            r"\b(?:terapia|antibiotic[ao]|farmacolog)\b",
            value,
            re.IGNORECASE,
        ):
            return contextual_match.group(1).strip()
        normalized = self.normalize_filter_key(value)
        for alias, replacement in self._embedded_aliases:
            if alias in normalized:
                return replacement
        return value

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
        if normalized in self.FUNCTION_WORD_NAMES:
            return True
        if re.fullmatch(
            r"(?:(?:ultima\s+dose\s+)?ricevut[oaie]?|interrott[oaie]?|"
            r"iniziat[oaie]?|sospes[oaie]?|ultima?(?:\s+dose)?|termine)"
            r"(?:\s+(?:il|dal|al))?",
            normalized,
        ):
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
        if tokens and all(token in self._non_drug_tokens for token in tokens):
            return True
        if tokens and all(
            token in self._non_drug_tokens
            or not any(character.isalpha() for character in token)
            for token in tokens
        ):
            return True
        if len(tokens) <= 3 and tokens[:2] == ["terapie", "eseguite"]:
            return True
        # Guardrail: fragments that look like lab measurements or markers
        # are frequently LLM extraction artefacts, not medication names.
        if self._lab_measurement_name_re.search(normalized) is not None:
            return True
        if self._lab_marker_name_re.search(normalized) is not None:
            return True
        # Guardrail: multi-word fragments whose tokens are all-lowercase
        # in the original text are clinical narrative fragments rather than
        # drug names. Drug names almost always contain at least one
        # capitalized token in Italian clinical text.
        if len(tokens) >= 3:
            original_tokens = value.split()
            has_uppercase_token = any(
                tok and tok[0].isupper() for tok in original_tokens
            )
            if not has_uppercase_token:
                func_word_count = sum(
                    1 for tok in tokens if tok in self.FUNCTION_WORD_NAMES
                )
                if func_word_count >= 2:
                    return True
        return False

    # -------------------------------------------------------------------------
    @staticmethod
    def is_truncated_compound_name(
        raw_name: str,
        *,
        source_text: str,
        name_start: int,
        name_end: int,
    ) -> bool:
        if len(raw_name.split()) != 1 or name_start < 0:
            return False
        line_end = source_text.find("\n", name_end)
        if line_end < 0:
            line_end = len(source_text)
        trailing = source_text[name_end:line_end]
        return bool(
            re.match(
                r"\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'-]{1,}(?:\s|$)",
                trailing,
            )
        )

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

    # -------------------------------------------------------------------------
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
        name = self.extract_embedded_drug_name(name)
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
    def validate_llm_drug_entry_grounding(
        self,
        entry: DrugEntry,
        *,
        source_text: str,
        source: Literal["therapy", "anamnesis"],
        historical_flag: bool,
    ) -> DrugEntry | None:
        if source == "therapy":
            return self.attach_source_grounding(
                entry,
                source_text=source_text,
                historical_flag=historical_flag,
                require_medication_syntax=False,
            )
        return self.attach_source_grounding(
            entry,
            source_text=source_text,
            historical_flag=historical_flag,
            require_medication_syntax=True,
        )

    # -------------------------------------------------------------------------
    def attach_source_grounding(
        self,
        entry: DrugEntry,
        *,
        source_text: str,
        historical_flag: bool,
        require_medication_syntax: bool,
    ) -> DrugEntry | None:
        if not source_text.strip():
            return None
        evidence = self.sanitize_text_field(entry.evidence)
        source_fold = source_text.casefold()
        evidence_start = source_fold.find(evidence.casefold()) if evidence else -1
        raw_name = (entry.name or "").strip()
        name_start, name_end = self.find_grounded_name_span(raw_name, source_text)
        if self.is_truncated_compound_name(
            raw_name,
            source_text=source_text,
            name_start=name_start,
            name_end=name_end,
        ):
            return None
        if evidence_start < 0 and name_start < 0:
            return None
        anchor_start = evidence_start if evidence_start >= 0 else name_start
        anchor_end = (
            evidence_start + len(evidence)
            if evidence_start >= 0 and evidence is not None
            else name_end
        )
        evidence_text = (
            source_text[evidence_start : evidence_start + len(evidence)]
            if evidence_start >= 0 and evidence is not None
            else source_text[name_start:name_end]
        )
        if require_medication_syntax and not self.has_grounded_medication_identity(
            entry,
            source_text=source_text,
            name_start=name_start,
            name_end=name_end,
            evidence_start=evidence_start,
            evidence_end=evidence_start + len(evidence)
            if evidence_start >= 0 and evidence is not None
            else -1,
        ):
            return None
        return entry.model_copy(
            update={
                "evidence": evidence_text,
                "source_span": entry.source_span or [anchor_start, anchor_end],
                "confidence": entry.confidence or "moderate",
                "attribution": entry.attribution or "patient",
                "current_status": entry.current_status
                or ("past" if historical_flag else "unclear"),
            }
        )

    # -------------------------------------------------------------------------
    def find_grounded_name_span(self, name: str, source_text: str) -> tuple[int, int]:
        if not name:
            return -1, -1
        source_fold = source_text.casefold()
        direct_start = source_fold.find(name.casefold())
        if direct_start >= 0:
            return direct_start, direct_start + len(name)
        normalized_name = self.normalize_filter_key(name)
        for alias, replacement in self._embedded_aliases:
            if self.normalize_filter_key(replacement) != normalized_name:
                continue
            alias_pattern = re.compile(
                r"\b"
                + r"[\s-]+".join(re.escape(part) for part in alias.split())
                + r"\b",
                re.IGNORECASE,
            )
            match = alias_pattern.search(source_text)
            if match is not None:
                return match.start(), match.end()
        return -1, -1

    # -------------------------------------------------------------------------
    def has_grounded_medication_identity(
        self,
        entry: DrugEntry,
        *,
        source_text: str,
        name_start: int,
        name_end: int,
        evidence_start: int,
        evidence_end: int,
    ) -> bool:
        if self.drug_entry_has_temporal_information(entry):
            return True
        if entry.dosage or entry.route or entry.administration_mode:
            return True
        if name_start >= 0 and self.name_has_medication_syntax(
            source_text,
            name_start=name_start,
            name_end=name_end,
        ):
            return True
        if evidence_start >= 0:
            window_start = max(0, evidence_start - 80)
            window_end = min(len(source_text), evidence_end + 80)
            evidence_context = source_text[window_start:window_end]
            return self.has_medication_context_signal(evidence_context)
        return False

    # -------------------------------------------------------------------------
    def name_has_medication_syntax(
        self,
        source_text: str,
        *,
        name_start: int,
        name_end: int,
    ) -> bool:
        before = source_text[max(0, name_start - 50) : name_start]
        after = source_text[name_end : min(len(source_text), name_end + 80)]
        if self.MEDICATION_NAME_PREFIX_RE.search(before):
            return True
        if self.SCHEDULE_RE.search(after):
            return True
        if self.DOSE_CUE_RE.search(after):
            return True
        if self.detect_route(after):
            return True
        return False

    # -------------------------------------------------------------------------
    def has_medication_context_signal(self, text: str) -> bool:
        return bool(self.MEDICATION_CONTEXT_RE.search(text))

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
        normalized = self.normalize_entry(
            enriched,
            source=source,
            historical_flag=historical_flag,
        )
        if normalized is None:
            return None
        evidence = (normalized.evidence or "").strip()
        if evidence and evidence in raw_line:
            start = raw_line.index(evidence)
            return normalized.model_copy(
                update={
                    "source_span": normalized.source_span
                    or [start, start + len(evidence)],
                    "confidence": normalized.confidence or "high",
                    "attribution": normalized.attribution or "patient",
                    "current_status": normalized.current_status
                    or ("past" if historical_flag else "unclear"),
                }
            )
        raw_name = (normalized.name or "").strip()
        start = raw_line.casefold().find(raw_name.casefold()) if raw_name else -1
        if start >= 0:
            end = start + len(raw_name)
            return normalized.model_copy(
                update={
                    "evidence": raw_line[start:end],
                    "source_span": [start, end],
                    "confidence": normalized.confidence or "moderate",
                    "attribution": normalized.attribution or "patient",
                    "current_status": normalized.current_status
                    or ("past" if historical_flag else "unclear"),
                }
            )
        return normalized.model_copy(
            update={
                "confidence": "low",
                "attribution": normalized.attribution or "unclear",
                "current_status": normalized.current_status or "unclear",
            }
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
