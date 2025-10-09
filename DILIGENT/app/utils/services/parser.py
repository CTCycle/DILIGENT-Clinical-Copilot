from __future__ import annotations

import asyncio
import contextlib
import re
import unicodedata
from collections.abc import Iterable, Iterator
from datetime import date, datetime
from typing import Any

import pandas as pd

from DILIGENT.app.api.models.prompts import (
    DISEASE_EXTRACTION_PROMPT,
    DRUG_EXTRACTION_PROMPT,
)
from DILIGENT.app.api.models.providers import initialize_llm_client
from DILIGENT.app.api.schemas.clinical import (
    BloodTest,
    DrugEntry,
    PatientBloodTests,
    PatientDiseases,
    PatientDrugs,
)
from DILIGENT.app.configurations import ClientRuntimeConfig
from DILIGENT.app.constants import DEFAULT_LLM_TIMEOUT_SECONDS
from DILIGENT.app.utils.patterns import (
    CUTOFF_IN_PAREN_RE,
    DATE_PATS,
    DRUG_BRACKET_TRAIL_RE,
    DRUG_BULLET_RE,
    DRUG_SCHEDULE_RE,
    DRUG_START_DATE_RE,
    DRUG_SUSPENSION_DATE_RE,
    DRUG_SUSPENSION_RE,
    FORM_DESCRIPTORS,
    FORM_TOKENS,
    ITALIAN_MONTHS,
    NUMERIC_RE,
    PATIENT_SECTION_HEADER_RE,
    TITER_RE,
    UNIT_TOKENS,
)

ALT_LABELS = {"ALT", "ALAT"}
ALP_LABELS = {"ALP"}


###############################################################################
class PatientCase:
    def __init__(self) -> None:
        self.HEADER_RE = PATIENT_SECTION_HEADER_RE
        self.expected_tags = ("ANAMNESIS", "BLOOD TESTS", "ADDITIONAL TESTS", "DRUGS")
        self.response = {
            "name": "Unknown",
            "sections": {},
            "unknown_headers": [],
            "missing_tags": list(self.expected_tags),
            "all_tags_present": False,
        }

    # -------------------------------------------------------------------------
    def clean_patient_info(self, text: str) -> str:
        # Normalize unicode width/compatibility (e.g., μ → μ, fancy quotes → ASCII where possible)
        processed_text = unicodedata.normalize("NFKC", text)
        # Normalize newlines
        processed_text = processed_text.replace("\r\n", "\n").replace("\r", "\n")
        # Strip trailing spaces on each line
        processed_text = "\n".join(line.rstrip() for line in processed_text.split("\n"))
        # Collapse 3+ blank lines to max 2, and leading/trailing blank lines
        processed_text = re.sub(r"\n{3,}", "\n\n", processed_text).strip()

        return processed_text

    # -------------------------------------------------------------------------
    def split_text_by_tags(self, text: str, name: str | None = None) -> dict[str, Any]:
        hits = [
            (m.group(1).strip(), m.start(), m.end())
            for m in self.HEADER_RE.finditer(text)
        ]
        if not hits:
            return self.response

        sections = {
            title.replace(" ", "_").lower(): text[
                end : (hits[i + 1][1] if i + 1 < len(hits) else len(text))
            ].strip()
            for i, (title, _start, end) in enumerate(hits)
        }

        exp_lower = {e.lower() for e in self.expected_tags}
        found_map = {k.lower(): k for k in sections}
        missing = [e for e in self.expected_tags if e.lower() not in found_map]
        unknown = [
            orig
            for low, orig in ((k.lower(), k) for k in sections)
            if low not in exp_lower
        ]

        self.response["name"] = name or "Unknown"
        self.response["sections"] = sections
        self.response["unknown_headers"] = unknown
        self.response["missing_tags"] = missing
        self.response["all_tags_present"] = not missing

        return sections

    # -------------------------------------------------------------------------
    def extract_sections_from_text(
        self, text: str, name: str | None = None
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        full_text = self.clean_patient_info(text)
        sections = self.split_text_by_tags(full_text, name)

        # Use DataFrame constructor for a list of dict rows (typed correctly)
        patient_table = pd.DataFrame([sections])
        patient_table["name"] = self.response["name"]

        return sections, patient_table


###############################################################################
class DiseasesParser:
    def __init__(
        self, timeout_s: float = DEFAULT_LLM_TIMEOUT_SECONDS, temperature: float = 0.0
    ) -> None:
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.client: Any | None = None
        self.client_provider: str | None = None
        self.model: str = ""
        self.runtime_revision = -1
        self._client_lock = asyncio.Lock()
        self.JSON_schema = {"diseases": list[str], "hepatic_diseases": list[str]}

    # -------------------------------------------------------------------------
    async def _ensure_client(self) -> None:
        async with self._client_lock:
            revision = ClientRuntimeConfig.get_revision()
            provider, model = ClientRuntimeConfig.resolve_provider_and_model("parser")
            needs_refresh = (
                self.client is None
                or self.client_provider != provider
                or self.runtime_revision != revision
            )
            if needs_refresh:
                if self.client is not None:
                    with contextlib.suppress(Exception):
                        await self.client.close()
                self.client = initialize_llm_client(
                    purpose="parser", timeout_s=self.timeout_s
                )
                self.client_provider = provider
            self.runtime_revision = revision
            self.model = model
            if self.client is not None and model and hasattr(self.client, "default_model"):
                self.client.default_model = model  # type: ignore[attr-defined]

    # -------------------------------------------------------------------------
    def normalize_unique(self, lst: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for x in lst:
            norm = x.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)

        return result

    # uses lanchain as wrapper to perform persing and validation to patient diseases model
    # -------------------------------------------------------------------------
    async def extract_diseases(self, text: str | None) -> dict[str, Any]:
        if text is None:
            return {"diseases": [], "hepatic_diseases": []}
        await self._ensure_client()
        if self.client is None:
            raise RuntimeError("LLM client is not initialized for disease extraction")
        try:
            parsed: Any = await self.client.llm_structured_call(
                model=self.model,
                system_prompt=DISEASE_EXTRACTION_PROMPT,
                user_prompt=text,
                schema=PatientDiseases,
                temperature=self.temperature,
                use_json_mode=True,
                max_repair_attempts=2,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to extract diseases (structured): {e}") from e

        diseases = self.normalize_unique(parsed.diseases)
        hepatic = [
            h for h in self.normalize_unique(parsed.hepatic_diseases) if h in diseases
        ]

        return {"diseases": diseases, "hepatic_diseases": hepatic}

    # -------------------------------------------------------------------------
    def validate_json_schema(self, output: dict) -> dict:
        for key in ["diseases", "hepatic_diseases"]:
            if key not in output or not isinstance(output[key], list):
                raise ValueError(f"Missing or invalid field: '{key}'. Must be a list.")
            if not all(isinstance(x, str) for x in output[key]):
                raise ValueError(f"All entries in '{key}' must be strings.")

        diseases = self.normalize_unique(output["diseases"])
        hepatic_diseases = self.normalize_unique(output["hepatic_diseases"])

        # Subset validation
        if not set(hepatic_diseases).issubset(set(diseases)):
            missing = set(hepatic_diseases) - set(diseases)
            raise ValueError("hepatic diseases were not validated")

        return {"diseases": diseases, "hepatic_diseases": hepatic_diseases}


###############################################################################
class BloodTestParser:
    """
    Minimal parser that assumes the input `text` is already the blood-test section.
    Strategy:
    1) Try LLM structured extraction to `PatientBloodTests`.
    2) If it fails, fall back to deterministic parsing.
    3) Post-process: dedupe + light normalization; always return a valid `PatientBloodTests`.

    """

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    ) -> None:
        default_model = (
            ClientRuntimeConfig.get_cloud_model()
            if ClientRuntimeConfig.is_cloud_enabled()
            else ClientRuntimeConfig.get_parsing_model()
        )
        self.model = (model or default_model).strip()
        self.temperature = float(temperature)
        self.client = initialize_llm_client(purpose="parser", timeout_s=timeout_s)

    # -------------------------------------------------------------------------
    def normalize_strings(self, s: str | None) -> str | None:
        if s is None:
            return None
        s2 = re.sub(r"\s+", " ", s).strip().rstrip(",:;.- ")
        return s2 or None

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def clean_text(self, text: str) -> str:
        t = unicodedata.normalize("NFKC", text or "")
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = "\n".join(line.rstrip() for line in t.split("\n")).strip()
        return t

    # -------------------------------------------------------------------------
    def dedupe_and_tidy(self, items: list[BloodTest]) -> list[BloodTest]:
        seen: set[tuple[Any, ...]] = set()
        out: list[BloodTest] = []
        for it in items or []:
            norm_name = self.normalize_strings(it.name)
            norm_value_text = self.normalize_strings(it.value_text)
            norm_unit = self.normalize_strings(it.unit)
            unit_clean = norm_unit.rstrip(".") if norm_unit is not None else None
            norm_cutoff_unit = self.normalize_strings(it.cutoff_unit)
            cutoff_unit_clean = (
                norm_cutoff_unit.rstrip(".") if norm_cutoff_unit is not None else None
            )
            norm_note = self.normalize_strings(it.note)
            norm_context_date = self.normalize_strings(it.context_date)

            key = (
                norm_name or "",
                it.value,
                norm_value_text,
                unit_clean,
                it.cutoff,
                cutoff_unit_clean,
                norm_note,
                norm_context_date,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(
                BloodTest(
                    name=norm_name or "",
                    value=it.value,
                    value_text=norm_value_text,
                    unit=unit_clean,
                    cutoff=it.cutoff,
                    cutoff_unit=cutoff_unit_clean,
                    note=norm_note,
                    context_date=norm_context_date,
                )
            )

        return out

    # -------------------------------------------------------------------------
    def parse_blood_test_results(self, text: str) -> Iterator[BloodTest]:
        for ctx_date, segment in self.iterate_date_segments(text):
            # 1) titer-like first (avoid numeric overlap)
            used_spans: list[tuple[int, int]] = []
            for m in TITER_RE.finditer(segment):
                name = m.group("name").strip()
                ratio = m.group("ratio").replace(" ", "")
                start, end = m.span()
                used_spans.append((start, end))
                yield BloodTest(
                    name=name,
                    value=None,
                    value_text=ratio,
                    unit=None,
                    cutoff=None,
                    cutoff_unit=None,
                    note=None,
                    context_date=ctx_date,
                )

            # 2) numeric values (with optional units/notes/cutoffs)
            for cand in self.split_candidates(segment):
                for m in NUMERIC_RE.finditer(cand):
                    start, end = m.span()
                    if any(not (end <= s or start >= e) for s, e in used_spans):
                        continue  # skip overlaps already captured as titers

                    name = m.group("name").strip()
                    raw_val = m.group("value")
                    unit = self.clean_unit(m.group("unit"))
                    paren = m.group("paren")
                    cutoff = None
                    note = None
                    cutoff_unit = None

                    if paren:
                        cut = CUTOFF_IN_PAREN_RE.search(paren)
                        if cut:
                            cutoff = float(cut.group(1).replace(",", "."))
                            cutoff_unit = unit
                        else:
                            note = paren.strip("() ").strip()

                    try:
                        value = float(raw_val.replace(",", "."))
                        value_text = None
                    except ValueError:
                        value = None
                        value_text = raw_val

                    # trim common leading/trailing noise around names
                    name = re.sub(
                        r"\b(Labor|BLOOD TESTS)\b[:\s]*$", "", name, flags=re.I
                    ).strip()
                    if not name:
                        continue

                    yield BloodTest(
                        name=name,
                        value=value,
                        value_text=value_text,
                        unit=unit,
                        cutoff=cutoff,
                        cutoff_unit=cutoff_unit,
                        note=note,
                        context_date=ctx_date,
                    )

    # -------------------------------------------------------------------------
    def _normalize_text(self, text: str) -> str:
        text = text.replace("\u00b5", "μ")
        text = re.sub(r"[ \t]+", " ", text)  # compact spaces
        text = re.sub(r"\s*\)\s*,", "),", text)  # tidy '),'
        return text

    # -----------------------------------------------------------------------------
    def _format_marker_value(self, value: str | None, unit: str | None) -> str | None:
        if not value:
            return None
        unit_part = unit.strip() if unit else ""
        return f"{value} {unit_part}".strip()

    # -----------------------------------------------------------------------------
    def parse_hepatic_markers(self, section: str | None) -> dict[str, Any]:
        markers: dict[str, Any] = {
            "alt": None,
            "alt_max": None,
            "alp": None,
            "alp_max": None,
        }
        if not section:
            return markers

        for match in NUMERIC_RE.finditer(section):
            raw_name = (match.group("name") or "").replace(":", "").strip().upper()
            normalized = raw_name.replace(" ", "")
            formatted_value = self._format_marker_value(
                match.group("value"), match.group("unit")
            )
            cutoff_value = self._extract_cutoff(match.group("paren"))
            if normalized in ALT_LABELS:
                markers["alt"] = formatted_value
                markers["alt_max"] = cutoff_value
            elif normalized in ALP_LABELS:
                markers["alp"] = formatted_value
                markers["alp_max"] = cutoff_value

        return markers

    # -------------------------------------------------------------------------
    def parse_date_string(self, s: str | None) -> str | None:
        if not s:
            return None
        # dd.mm.yyyy
        m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", s)
        if m:
            d, mth, y = map(int, m.groups())
            try:
                return date(y, mth, d).isoformat()
            except ValueError:
                return s
        # Month DD YYYY (Italian month)
        m2 = re.fullmatch(r"([A-Za-zÀ-ÿ]+)\s+(\d{1,2})[.,]?\s*(\d{4})", s, flags=re.I)
        if m2:
            mon_name, d, y = m2.groups()
            mon = ITALIAN_MONTHS.get(mon_name.lower())
            if mon:
                try:
                    return date(int(y), mon, int(d)).isoformat()
                except ValueError:
                    return s
        return None

    # -------------------------------------------------------------------------
    def _extract_cutoff(self, text: str | None) -> str | None:
        if not text:
            return None
        cutoff_match = CUTOFF_IN_PAREN_RE.search(text)
        if cutoff_match:
            return cutoff_match.group(1)
        max_match = re.search(r"max[: ]*([0-9]+(?:[.,][0-9]+)?)", text, re.IGNORECASE)
        if max_match:
            return max_match.group(1)
        return None

    # -------------------------------------------------------------------------
    def iterate_date_segments(self, text: str) -> Iterator[tuple[str | None, str]]:
        text = self._normalize_text(text)

        markers: list[tuple[int, int, str]] = []
        for pat in DATE_PATS:
            for m in pat.finditer(text):
                raw = m.groupdict().get("d") or (
                    f"{m.group('m')} {m.group('day')}.{m.group('year')}"
                    if "m" in m.groupdict()
                    else None
                )
                if raw is None:
                    continue
                markers.append((m.start(), m.end(), raw))

        markers.sort(
            key=lambda x: (x[0], -(x[1] - x[0]))
        )  # leftmost, prefer longer span
        pruned: list[tuple[int, int, str]] = []
        last_end = -1
        for s, e, raw in markers:
            if s >= last_end:
                pruned.append((s, e, raw))
                last_end = e

        if not pruned:
            yield (None, text)
            return

        prev_end = 0
        current_date: str | None = None
        for s, e, raw in pruned:
            if s > prev_end:
                seg = text[prev_end:s].strip(" \n:")
                if seg:
                    yield (
                        self.parse_date_string(current_date) if current_date else None,
                        seg,
                    )
            current_date = raw
            prev_end = e

        tail = text[prev_end:].strip(" \n:")
        if tail:
            yield (self.parse_date_string(current_date) if current_date else None, tail)

    # -------------------------------------------------------------------------
    def clean_unit(self, u: str | None) -> str | None:
        if not u:
            return None
        s = u.strip()
        s = re.split(r"[,;]|(?=\s[A-Za-zÀ-ÿ])", s)[0]  # stop at delimiter or new word
        return s.rstrip(".")

    # -------------------------------------------------------------------------
    def split_candidates(self, segment: str) -> Iterable[str]:
        """Split around commas/newlines but keep commas inside parentheses."""
        tmp = []
        depth = 0
        for ch in segment:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            tmp.append("§" if (ch == "," and depth > 0) else ch)
        safe = "".join(tmp)
        for p in re.split(r"[,\n]+", safe):
            yield p.replace("§", ",").strip(" .;:")

    # -------------------------------------------------------------------------
    async def extract_blood_test_results(self, text: str) -> PatientBloodTests:
        cleaned = self.clean_text(text)
        if not cleaned:
            return PatientBloodTests(entries=[])

        parsed = entries = list(self.parse_blood_test_results(cleaned))
        entries = self.dedupe_and_tidy(parsed)

        return PatientBloodTests(entries=entries)

    # -------------------------------------------------------------------------
    def parse_date_iso_format(self, s: str | None) -> datetime | None:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    def get_latest_by_name(
        self, entries: list[BloodTest], target: str
    ) -> BloodTest | None:
        target_low = target.strip().lower()
        dated: list[tuple[datetime | None, BloodTest]] = []
        fallback: BloodTest | None = None
        for e in entries or []:
            name = (e.name or "").strip().lower()
            if name != target_low:
                continue
            fallback = e  # keep last seen in case dates are missing
            dt = self.parse_date_iso_format(e.context_date)
            dated.append((dt, e))

        if dated:
            dated.sort(key=lambda t: (t[0] is None, t[0]))
            return dated[-1][1]
        return fallback

    # -------------------------------------------------------------------------
    def extract_hepatic_markers(self, blood_tests: PatientBloodTests) -> dict[str, Any]:
        """Return latest ALAT and ANA entries in a compact dict.

        Output example:
        {
            "ALAT": {"value": 189.0, "value_text": None, "unit": "U/L", "date": "2025-06-26"},
            "ANA":  {"value": None,  "value_text": "1:80", "unit": None,  "date": "2025-06-26"}
        }
        """
        entries = getattr(blood_tests, "entries", []) or []
        latest_alat = self.get_latest_by_name(entries, "ALAT")
        latest_alp = self.get_latest_by_name(entries, "ALP")
        latest_ana = self.get_latest_by_name(entries, "ANA")

        out: dict[str, Any] = {}
        if latest_alat is not None:
            out["ALAT"] = {
                "value": latest_alat.value,
                "value_text": latest_alat.value_text,
                "unit": latest_alat.unit,
                "date": latest_alat.context_date,
            }
        if latest_alp is not None:
            out["ALP"] = {
                "value": latest_alp.value,
                "value_text": latest_alp.value_text,
                "unit": latest_alp.unit,
                "date": latest_alp.context_date,
            }
        if latest_ana is not None:
            out["ANA"] = {
                "value": latest_ana.value,
                "value_text": latest_ana.value_text,
                "unit": latest_ana.unit,
                "date": latest_ana.context_date,
            }

        return out


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
        timeout_s: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.client: Any | None = client
        self.model: str = ""
        self._client_lock = asyncio.Lock()
        if client is None:
            self.client_provider: str | None = None
            self.runtime_revision = -1
        else:
            self.client_provider = "injected"
            self.runtime_revision = ClientRuntimeConfig.get_revision()

    async def _ensure_client(self) -> None:
        async with self._client_lock:
            revision = ClientRuntimeConfig.get_revision()
            provider, model = ClientRuntimeConfig.resolve_provider_and_model("parser")
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
                if self.client is not None:
                    with contextlib.suppress(Exception):
                        await self.client.close()
                self.client = initialize_llm_client(
                    purpose="parser", timeout_s=self.timeout_s
                )
                self.client_provider = provider
            self.runtime_revision = revision
            self.model = model
            if self.client is not None and model and hasattr(self.client, "default_model"):
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

    async def extract_drug_list(self, text: str | None) -> PatientDrugs:
        cleaned = self.clean_text(text)
        if not cleaned:
            return PatientDrugs(entries=[])
        await self._ensure_client()
        if self.client is None:
            raise RuntimeError("LLM client is not initialized for drug extraction")
        try:
            structured = await self._llm_extract_drugs(cleaned)
        except Exception as exc:  # pragma: no cover - passthrough for visibility
            raise RuntimeError("Failed to extract drugs via LLM") from exc

        return PatientDrugs(entries=list(structured.entries))

    async def _llm_extract_drugs(self, text: str) -> PatientDrugs:
        if self.client is None:
            raise RuntimeError("LLM client is not initialized for drug extraction")
        return await self.client.llm_structured_call(
            model=self.model,
            system_prompt=DRUG_EXTRACTION_PROMPT,
            user_prompt=text,
            schema=PatientDrugs,
            temperature=self.temperature,
            use_json_mode=True,
            max_repair_attempts=2,
        )

    def _parse_line(self, line: str) -> DrugEntry | None:
        schedule_match = self.SCHEDULE_RE.search(line)
        if not schedule_match:
            return None
        schedule_text = schedule_match.group("schedule")
        schedule_values = self._parse_schedule(schedule_text)
        before = line[: schedule_match.start()].strip(" ,;:\t")
        tail = line[schedule_match.end() :].strip()
        bracket_match = self.BRACKET_TRAIL_RE.search(before)
        if bracket_match:
            before = before[: bracket_match.start()].strip()
        name, dosage, administration_mode = self._split_heading(before)
        if not name:
            name = before or line.strip()
        suspension_status, suspension_date = self._detect_suspension(line, tail)
        start_status, start_date = self._detect_start(line, tail)
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
    def _parse_schedule(self, text: str) -> list[float]:
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
        if len(slots) == 4:
            return slots
        if len(slots) > 4:
            return slots[:4]
        return []

    # -------------------------------------------------------------------------
    def _split_heading(self, text: str) -> tuple[str | None, str | None, str | None]:
        if not text:
            return None, None, None
        tokens = text.split()
        if not tokens:
            return None, None, None
        first_numeric = None
        for idx, token in enumerate(tokens):
            if self._token_has_numeric(token):
                first_numeric = idx
                break
        if first_numeric is None:
            return " ".join(tokens).strip() or None, None, None
        name_tokens = tokens[:first_numeric]
        remainder = tokens[first_numeric:]
        mode_tokens: list[str] = []
        self._extract_mode_from_prefix(name_tokens, mode_tokens)
        dosage_tokens: list[str] = []
        for token in remainder:
            normalized = self._normalize_token(token)
            if normalized in FORM_TOKENS:
                mode_tokens.append(token)
                continue
            if mode_tokens and (
                normalized in FORM_DESCRIPTORS or not self._token_has_numeric(token)
            ):
                mode_tokens.append(token)
                continue
            if (
                self._token_has_numeric(token)
                or normalized in UNIT_TOKENS
                or "/" in token
            ):
                dosage_tokens.append(token)
                continue
            if dosage_tokens:
                dosage_tokens.append(token)
            else:
                name_tokens.append(token)
        if not dosage_tokens and remainder:
            dosage_tokens = remainder
        name = " ".join(name_tokens).strip() or None
        dosage = " ".join(dosage_tokens).strip() or None
        administration_mode = " ".join(mode_tokens).strip() or None
        return name, dosage, administration_mode

    # -------------------------------------------------------------------------
    def _extract_mode_from_prefix(
        self, name_tokens: list[str], mode_tokens: list[str]
    ) -> None:
        while name_tokens:
            normalized = self._normalize_token(name_tokens[-1])
            if normalized in FORM_TOKENS:
                mode_tokens.insert(0, name_tokens.pop())
                continue
            if mode_tokens and normalized in FORM_DESCRIPTORS:
                mode_tokens.insert(0, name_tokens.pop())
                continue
            break

    # -------------------------------------------------------------------------
    def _token_has_numeric(self, token: str) -> bool:
        return any(ch.isdigit() for ch in token)

    # -------------------------------------------------------------------------
    def _normalize_token(self, token: str) -> str:
        return re.sub(r"[.,;:]+$", "", token.lower())

    # -------------------------------------------------------------------------
    def _detect_suspension(
        self, full_line: str, tail: str
    ) -> tuple[bool | None, str | None]:
        status = True if self.SUSPENSION_RE.search(full_line) else None
        date_match = self.SUSPENSION_DATE_RE.search(
            tail
        ) or self.SUSPENSION_DATE_RE.search(full_line)
        date_value = (
            self._normalize_date_token(date_match.group("date")) if date_match else None
        )
        return status, date_value

    # -------------------------------------------------------------------------
    def _detect_start(
        self, full_line: str, tail: str
    ) -> tuple[bool | None, str | None]:
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
                normalized = self._normalize_date_token(date_token)
                return True, normalized
        return None, None

    # -------------------------------------------------------------------------
    def _normalize_date_token(self, token: str | None) -> str | None:
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
