from __future__ import annotations

import re

from common.utils.patterns import (
    DRUG_BULLET_RE,
    DRUG_BRACKET_TRAIL_RE,
    DRUG_SCHEDULE_RE,
    DRUG_START_DATE_RE,
    DRUG_SUSPENSION_DATE_RE,
    DRUG_SUSPENSION_RE,
)

SCHEDULE_RE = DRUG_SCHEDULE_RE
DATE_LIKE_SCHEDULE_RE = re.compile(r"^\d{4}\s*-\s*\d{1,2}\s*-\s*\d{1,2}$")
BULLET_RE = DRUG_BULLET_RE
BRACKET_TRAIL_RE = DRUG_BRACKET_TRAIL_RE
SUSPENSION_RE = DRUG_SUSPENSION_RE
SUSPENSION_DATE_RE = DRUG_SUSPENSION_DATE_RE
START_DATE_RE = DRUG_START_DATE_RE
ROUTE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("oral", re.compile(r"\b(?:p\.?o\.?|per\s+os|oral(?:e)?)\b", re.IGNORECASE)),
    (
        "iv",
        re.compile(
            r"\b(?:i\.?v\.?|e\.?v\.?|endovenos[ao]?|intraven(?:ous|osa)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "im",
        re.compile(
            r"\b(?:i\.?m\.?|intramuscolar[ei]|intramuscular)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sc",
        re.compile(
            r"\b(?:s\.?c\.?|sottocutane[ao]?|subcut(?:aneous|anea)?)\b",
            re.IGNORECASE,
        ),
    ),
    ("topical", re.compile(r"\b(?:topical|topic[ao]|cutane[ao])\b", re.IGNORECASE)),
)
DOSE_CUE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|ug|ml|u|ui|units?)\b",
    re.IGNORECASE,
)
DOSAGE_TEMPORAL_SPLIT_RE = re.compile(
    r"""
    (?:[,;]\s*|\s+)
    (?:
        iniziat[oaie]|
        avviat[oaie]|
        start(?:ed|ing)?|
        began|
        begin|
        sospes[oaie]|
        interrott[aoie]|
        suspend(?:ed|ere|ing)?|
        stopp?ed|
        discontinued?|
        alla\s+comparsa|
        dal(?:la)?|
        da(?:ll['’])?|
        since|
        from|
        on
    )\b.*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
NAME_TEMPORAL_SPLIT_RE = re.compile(
    r"""
    (?:[,;]\s*|\s+)
    (?:
        ultima\s+somministrazione|
        linea\s+precedente|
        iniziat[oaie]|
        avviat[oaie]|
        start(?:ed|ing)?|
        began|
        begin|
        sospes[oaie]|
        interrott[aoie]|
        suspend(?:ed|ere|ing)?|
        stopp?ed|
        discontinued?|
        alla\s+comparsa|
        dal(?:la)?|
        da(?:ll['’])?|
        since|
        from|
        on
    )\b.*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
TRAILING_ROUTE_TOKEN_RE = re.compile(
    r"\b(?:p\.?o\.?|e\.?v\.?|i\.?v\.?|i\.?m\.?|s\.?c\.?|po|ev|iv|im|sc)\s*$",
    re.IGNORECASE,
)
START_EVENT_RE = re.compile(
    r"""
    \b(?:iniz(?:io|iat[oaie])|avviat[oaie]|ripres[oaie]|riprend[ei]re|
    assunzion[ei]|in\s+terapia|in\s+trattamento|terapia|trattamento|
    start(?:ed|ing)?|initiat(?:ed|ion)|began|begin|resume[sd]?|taking)
    \b(?P<tail>[^,;\n]*)
    """,
    re.IGNORECASE | re.VERBOSE,
)
SUSPENSION_EVENT_RE = re.compile(
    r"""
    \b(?:sospes[oaie]|interrott[aoie]|suspend(?:ed|ere|ing)?|stopp?ed|discontinued?)
    \b(?P<tail>[^,;\n]*)
    """,
    re.IGNORECASE | re.VERBOSE,
)
