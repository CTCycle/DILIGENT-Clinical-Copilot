from __future__ import annotations

import asyncio
import re
from typing import Any

from services.llm.runtime_config import LLMRuntimeConfig
from configurations.startup import get_server_settings
from services.catalogs.runtime import get_reference_catalog_snapshot
from services.clinical.parser_extraction import (
    BRACKET_TRAIL_RE,
    BULLET_RE,
    DATE_LIKE_SCHEDULE_RE,
    SCHEDULE_RE,
    START_DATE_RE,
    SUSPENSION_DATE_RE,
    SUSPENSION_RE,
    build_dosage_temporal_split_re,
    build_dose_cue_re,
    build_name_temporal_split_re,
    build_route_patterns,
    build_start_event_re,
    build_suspension_event_re,
    build_trailing_route_token_re,
)
from services.clinical.parser_validation import get_parser_validation_data
from services.clinical.parser_llm import DrugLlmExtractionMixin
from services.clinical.parser_rules import DrugRulesMixin


###############################################################################
class DrugsParser(DrugLlmExtractionMixin, DrugRulesMixin):
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
    ROUTE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = ()
    DOSE_CUE_RE = re.compile(r"$^")
    DOSAGE_TEMPORAL_SPLIT_RE = re.compile(r"$^")
    NAME_TEMPORAL_SPLIT_RE = re.compile(r"$^")
    TRAILING_ROUTE_TOKEN_RE = re.compile(r"$^")
    START_EVENT_RE = re.compile(r"$^")
    SUSPENSION_EVENT_RE = re.compile(r"$^")
    DRUG_FORM_SUFFIX_RE = re.compile(r"$^")
    LAB_MEASUREMENT_NAME_RE = re.compile(r"\b(?:u/l|ui/l|mg/dl|uln)\b", re.IGNORECASE)
    LAB_MARKER_NAME_RE = re.compile(r"\b(?:alt|ast|alp|bilirubin|inr)\b", re.IGNORECASE)
    MEDICATION_CONTEXT_RE = re.compile(
        r"\b(?:"
        r"drug|drugs|medication|medications|medicine|therapy|treatment|"
        r"farmac[io]|terapia|terapie|trattamento|assume|assunta|assunto|"
        r"somministrat[aoie]|prescritt[aoie]|sospes[aoie]|antibiotic[ao]|"
        r"chemioterapia|protocollo|allergi[aoe]"
        r")\b",
        re.IGNORECASE,
    )
    MEDICATION_NAME_PREFIX_RE = re.compile(
        r"\b(?:"
        r"con|with|assume|assunta|assunto|somministrat[aoie]|prescritt[aoie]|"
        r"sospes[aoie]|terapia|therapy|treatment|farmac[io]|antibiotic[ao]|"
        r"protocollo"
        r")\s*$",
        re.IGNORECASE,
    )
    FUNCTION_WORD_NAMES = {
        "a",
        "al",
        "alla",
        "con",
        "da",
        "dal",
        "dalla",
        "del",
        "della",
        "di",
        "il",
        "in",
        "nel",
        "nella",
        "per",
        "the",
        "to",
        "with",
    }

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        client: Any | None = None,
        temperature: float = 0.0,
        timeout_s: float = get_server_settings().runtime.parser_llm_timeout,
    ) -> None:
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.client: Any | None = client
        self.model: str = ""
        self.client_lock = asyncio.Lock()
        self.client_loop_id: int | None = None
        self.forced_provider: str | None = None
        self.forced_model: str | None = None
        self._parser_validation_data = get_parser_validation_data()
        if client is None:
            self.client_provider: str | None = None
            self.runtime_revision = -1
        else:
            self.client_provider = "injected"
            self.runtime_revision = LLMRuntimeConfig.get_revision()
        self._embedded_aliases = self._load_embedded_aliases()
        self._non_drug_tokens = self._load_non_drug_tokens()
        self.NON_DRUG_EXACT_NAMES = self._load_non_drug_exact_names()
        self.NON_DRUG_PREFIXES = self._load_non_drug_prefixes()
        self.NON_DRUG_CONTAINS = self._load_non_drug_contains()
        self.WEEKDAY_TOKENS = self._load_weekday_terms()
        self.NON_THERAPY_LINE_PREFIXES = self._load_non_therapy_line_prefixes()
        self.ROUTE_PATTERNS = build_route_patterns()
        self.DOSE_CUE_RE = build_dose_cue_re()
        self.DOSAGE_TEMPORAL_SPLIT_RE = build_dosage_temporal_split_re()
        self.NAME_TEMPORAL_SPLIT_RE = build_name_temporal_split_re()
        self.TRAILING_ROUTE_TOKEN_RE = build_trailing_route_token_re()
        self.START_EVENT_RE = build_start_event_re()
        self.SUSPENSION_EVENT_RE = build_suspension_event_re()
        self._lab_measurement_name_re = self._build_lab_measurement_pattern()
        self._lab_marker_name_re = self._build_lab_marker_pattern()
        self.DRUG_FORM_SUFFIX_RE = self._build_drug_form_suffix_re()

    # -------------------------------------------------------------------------
    def _load_embedded_aliases(self) -> tuple[tuple[str, str], ...]:
        snapshot = get_reference_catalog_snapshot()
        entries = snapshot.entries("drug_matching", "catalog_fallback_aliases")
        aliases: list[tuple[str, str]] = []
        for entry in entries:
            normalized_alias = self.normalize_filter_key(entry.value)
            replacement = entry.key.strip()
            if normalized_alias and replacement:
                aliases.append((normalized_alias, replacement))
        return tuple(aliases)

    # -------------------------------------------------------------------------
    def _load_non_drug_tokens(self) -> frozenset[str]:
        snapshot = get_reference_catalog_snapshot()
        return frozenset(
            self.normalize_filter_key(value)
            for value in snapshot.values(
                "clinical_extraction",
                "drug_non_name_tokens",
            )
            if value
        )

    # -------------------------------------------------------------------------
    def _load_non_drug_exact_names(self) -> set[str]:
        snapshot = get_reference_catalog_snapshot()
        values = set(
            snapshot.values("clinical_extraction", "drug_non_name_exact", key="default")
        )
        values.update(self._parser_validation_data["NON_DRUG_EXACT_NAMES"])
        return values

    # -------------------------------------------------------------------------
    def _load_non_drug_prefixes(self) -> tuple[str, ...]:
        snapshot = get_reference_catalog_snapshot()
        return tuple(
            dict.fromkeys(
                [
                    *snapshot.values(
                        "clinical_extraction",
                        "drug_non_name_prefixes",
                        key="default",
                    ),
                    *self._parser_validation_data["NON_DRUG_PREFIXES"],
                ]
            )
        )

    # -------------------------------------------------------------------------
    def _load_non_drug_contains(self) -> tuple[str, ...]:
        snapshot = get_reference_catalog_snapshot()
        return tuple(
            dict.fromkeys(
                [
                    *snapshot.values(
                        "clinical_extraction",
                        "drug_non_name_contains",
                        key="default",
                    ),
                    *self._parser_validation_data["NON_DRUG_CONTAINS"],
                ]
            )
        )

    # -------------------------------------------------------------------------
    def _load_weekday_terms(self) -> set[str]:
        snapshot = get_reference_catalog_snapshot()
        values = set(
            snapshot.values("clinical_extraction", "weekday_terms", key="default")
        )
        values.update(self._parser_validation_data["WEEKDAY_TOKENS"])
        return values

    # -------------------------------------------------------------------------
    def _load_non_therapy_line_prefixes(self) -> tuple[str, ...]:
        snapshot = get_reference_catalog_snapshot()
        return tuple(
            dict.fromkeys(
                [
                    *snapshot.values("clinical_extraction", "drug_line_prefixes"),
                    *self._parser_validation_data["NON_THERAPY_LINE_PREFIXES"],
                ]
            )
        )

    # -------------------------------------------------------------------------
    def _build_lab_measurement_pattern(self) -> re.Pattern[str]:
        snapshot = get_reference_catalog_snapshot()
        values = list(
            snapshot.values("clinical_extraction", "lab_measurement_indicators")
        )
        values.extend(snapshot.values("clinical_extraction", "laboratory_uln_labels"))
        escaped = [re.escape(value) for value in values if value]
        if not escaped:
            return self.LAB_MEASUREMENT_NAME_RE
        return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)

    # -------------------------------------------------------------------------
    def _build_lab_marker_pattern(self) -> re.Pattern[str]:
        snapshot = get_reference_catalog_snapshot()
        values = list(snapshot.values("clinical_extraction", "lab_marker_indicators"))
        escaped = [re.escape(value) for value in values if value]
        if not escaped:
            return self.LAB_MARKER_NAME_RE
        return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)

    # -------------------------------------------------------------------------
    def _build_drug_form_suffix_re(self) -> re.Pattern[str]:
        snapshot = get_reference_catalog_snapshot()
        values = list(snapshot.values("clinical_extraction", "drug_form_suffixes"))
        if not values:
            return self.DRUG_FORM_SUFFIX_RE
        escaped = [re.escape(value.strip()) for value in values if value.strip()]
        return re.compile(
            r"\s+(?:" + "|".join(escaped) + r")\s*$",
            re.IGNORECASE,
        )

    # -------------------------------------------------------------------------
    def _strip_drug_form_suffixes(self, name: str) -> str:
        return self.DRUG_FORM_SUFFIX_RE.sub("", name)

    # -------------------------------------------------------------------------
