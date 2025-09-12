from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Iterable, Iterator
from datetime import date, datetime
from typing import Any

import pandas as pd

from Pharmagent.app.api.models.prompts import DISEASE_EXTRACTION_PROMPT
from Pharmagent.app.api.models.providers import OllamaClient
from Pharmagent.app.api.schemas.clinical import (
    BloodTest,
    PatientBloodTests,
    PatientData,
    PatientDiseases,
)
from Pharmagent.app.api.schemas.regex import (
    CUTOFF_IN_PAREN_RE,
    DATE_PATS,
    ITALIAN_MONTHS,
    NUMERIC_RE,
    TITER_RE,
)
from Pharmagent.app.constants import PARSER_MODEL


###############################################################################
class PatientCase:
    def __init__(self) -> None:
        self.HEADER_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+(.+?)\s*$", re.MULTILINE)
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
        self, payload: PatientData
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        full_text = self.clean_patient_info(payload.info)
        sections = self.split_text_by_tags(full_text, payload.name)

        # Use DataFrame constructor for a list of dict rows (typed correctly)
        patient_table = pd.DataFrame([sections])
        patient_table["name"] = self.response["name"]

        return sections, patient_table


###############################################################################
class DiseasesParser:
    def __init__(self, timeout_s: float = 300.0, temperature: float = 0.0) -> None:
        self.temperature = float(temperature)
        self.client = OllamaClient(base_url=None, timeout_s=timeout_s)
        self.JSON_schema = {"diseases": list[str], "hepatic_diseases": list[str]}
        self.model = PARSER_MODEL

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
        timeout_s: float = 300.0,
    ) -> None:
        self.model = (model or PARSER_MODEL).strip()
        self.temperature = float(temperature)
        self.client = OllamaClient(base_url=None, timeout_s=timeout_s)

    # -------------------------------------------------------------------------
    def normalize_strings(self, s: str | None) -> str | None:
        if s is None:
            return None
        s2 = re.sub(r"\s+", " ", s).strip().rstrip(",:;.- ")
        return s2 or None

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
        s = re.split(r"[,;]|(?=\s[A-Za-z�-�])", s)[0]  # stop at delimiter or new word
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
            return PatientBloodTests(source_text="", entries=[])

        parsed = entries = list(self.parse_blood_test_results(cleaned))
        entries = self.dedupe_and_tidy(parsed)

        return PatientBloodTests(source_text=cleaned, entries=entries)

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
        latest_ana = self.get_latest_by_name(entries, "ANA")

        out: dict[str, Any] = {}
        if latest_alat is not None:
            out["ALAT"] = {
                "value": latest_alat.value,
                "value_text": latest_alat.value_text,
                "unit": latest_alat.unit,
                "date": latest_alat.context_date,
            }
        if latest_ana is not None:
            out["ANA"] = {
                "value": latest_ana.value,
                "value_text": latest_ana.value_text,
                "unit": latest_ana.unit,
                "date": latest_ana.context_date,
            }

        return out
