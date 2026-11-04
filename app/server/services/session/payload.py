# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"[ \t]+")
DRUG_BULLET_PREFIX_RE = re.compile(
    r"^[\-\u2022\u2023\u2043\*\u25A0\u25AA\u25CF\u25E6\u2219\u00B7]+\s*"
)
DRUG_ALLOWED_SYMBOLS_RE = re.compile(r"[^\w\s.,;:/()\-+%[\]'\"<>=]")
ANAMNESIS_ALLOWED_SYMBOLS_RE = re.compile(r"[^\w\s.,;:/()\-+%[\]'\"<>=]")
MAX_DRUGS_TEXT_LENGTH = 20000


# HELPERS


###############################################################################
def sanitize_field(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


###############################################################################
def strip_html(value: str) -> str:
    unescaped = html.unescape(value)
    return HTML_TAG_RE.sub(" ", unescaped)


###############################################################################
def sanitize_drug_line(value: str) -> str:
    stripped_html = strip_html(value)
    without_bullet = DRUG_BULLET_PREFIX_RE.sub("", stripped_html)
    without_marks = without_bullet.replace("!", " ").replace("?", " ")
    without_symbols = DRUG_ALLOWED_SYMBOLS_RE.sub(" ", without_marks)
    compact = MULTISPACE_RE.sub(" ", without_symbols).strip(" \t,;:-")
    return compact


###############################################################################
def sanitize_drugs_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw_line in normalized.split("\n"):
        cleaned = sanitize_drug_line(raw_line)
        if cleaned:
            lines.append(cleaned)
    if not lines:
        return None
    joined = "\n".join(lines)
    if len(joined) <= MAX_DRUGS_TEXT_LENGTH:
        return joined
    # Keep the longest safe prefix at line boundaries to avoid schema failures.
    clipped = joined[:MAX_DRUGS_TEXT_LENGTH]
    last_newline = clipped.rfind("\n")
    if last_newline >= 0:
        clipped = clipped[:last_newline]
    clipped = clipped.strip()
    return clipped or joined[:MAX_DRUGS_TEXT_LENGTH].strip() or None


###############################################################################
def sanitize_anamnesis_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    stripped_html = strip_html(normalized)
    without_marks = stripped_html.replace("!", " ").replace("?", " ")
    without_symbols = ANAMNESIS_ALLOWED_SYMBOLS_RE.sub(" ", without_marks)
    cleaned_lines: list[str] = []
    for raw_line in without_symbols.split("\n"):
        compact = MULTISPACE_RE.sub(" ", raw_line).strip()
        if compact:
            cleaned_lines.append(compact)
    if not cleaned_lines:
        return None
    return "\n".join(cleaned_lines)


###############################################################################
def sanitize_laboratory_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    stripped_html = strip_html(normalized)
    cleaned_lines: list[str] = []
    for raw_line in stripped_html.split("\n"):
        compact = MULTISPACE_RE.sub(" ", raw_line).strip()
        if compact:
            cleaned_lines.append(compact)
    if not cleaned_lines:
        return None
    return "\n".join(cleaned_lines)


###############################################################################
class PayloadSanitizationService:

    # -------------------------------------------------------------------------
    @staticmethod
    def sanitize_dili_payload(
        *,
        patient_name: str | None,
        visit_date: datetime | date | dict[str, Any] | str | None,
        anamnesis: str | None,
        drugs: str | None,
        laboratory_analysis: str | None,
        evidence_sources: Sequence[str],
    ) -> dict[str, Any]:
        return {
            "name": sanitize_field(patient_name),
            # PatientData is the single authority for visit-date validation.
            "visit_date": visit_date,
            "anamnesis": sanitize_anamnesis_text(anamnesis),
            "drugs": sanitize_drugs_text(drugs),
            "laboratory_analysis": sanitize_laboratory_text(laboratory_analysis),
            "evidence_sources": list(evidence_sources),
        }
