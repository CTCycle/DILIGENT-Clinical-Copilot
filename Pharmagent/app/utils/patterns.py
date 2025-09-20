from __future__ import annotations

import re

# -----------------------------------------------------------------------------
# Patient section parsing
# -----------------------------------------------------------------------------
PATIENT_SECTION_HEADER_RE = re.compile(r"^[ 	]*#{1,6}[ 	]+(.+?)\s*$", re.MULTILINE)

# -----------------------------------------------------------------------------
# Drug parsing patterns
# -----------------------------------------------------------------------------
DRUG_SCHEDULE_RE = re.compile(
    r"(?P<schedule>\d+(?:[.,]\d+)?(?:\s*-\s*\d+(?:[.,]\d+)?){1,3})"
)
DRUG_BULLET_RE = re.compile(r"^[\-\u2022\u2023\u2043\*]+\s*")
DRUG_BRACKET_TRAIL_RE = re.compile(r"\[(?P<content>[^\]]+)\]\s*$")
DRUG_SUSPENSION_RE = re.compile(r"\bsospes[oa]\b", re.IGNORECASE)
DRUG_SUSPENSION_DATE_RE = re.compile(
    r"\bsospes[oa](?:\s+(?:dal|dall'))?\s*(?P<date>\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?)",
    re.IGNORECASE,
)

# -----------------------------------------------------------------------------
# Blood test and date parsing patterns
# -----------------------------------------------------------------------------
ITALIAN_MONTHS = {
    "gennaio": 1,
    "febbraio": 2,
    "marzo": 3,
    "aprile": 4,
    "maggio": 5,
    "giugno": 6,
    "luglio": 7,
    "agosto": 8,
    "settembre": 9,
    "ottobre": 10,
    "novembre": 11,
    "dicembre": 12,
}

DATE_PATS = [
    re.compile(r"\((?P<d>\d{1,2}\.\d{1,2}\.\d{4})\)"),
    re.compile(r"(?<!\d)(?P<d>\d{1,2}\.\d{1,2}\.\d{4})(?!\d)"),
    re.compile(
        r"(?P<m>\b[A-Za-zÀ-ÿ]+)\s+(?P<day>\d{1,2})[.,]?\s*(?P<year>\d{4})",
        re.IGNORECASE,
    ),
]

TITER_RE = re.compile(
    r"(?P<name>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-/ ]{0,40}?)\s*(?P<ratio>\d+\s*:\s*\d+)\b"
)

NUMERIC_RE = re.compile(
    r"""
    (?P<name>[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9\-/ ]{0,50}?)    # test name (lazy)
    \s*
    (?P<value>\d+(?:[.,]\d+)?)                        # number
    \s*
    (?P<unit>[A-Za-zµ/%\.0-9\-/\^]+)?                # unit (optional)
    \s*
    (?P<paren>\([^)]+\))?                             # parentheses (optional)
    """,
    re.VERBOSE,
)

CUTOFF_IN_PAREN_RE = re.compile(r"cut[\s\-]?off[: ]*\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)



# -----------------------------------------------------------------------------
# Drugs names and info parsing patterns
# -----------------------------------------------------------------------------
FORM_TOKENS = {
        "cpr",
        "compresse",
        "compressa",
        "caps",
        "capsule",
        "capsula",
        "sir",
        "scir",
        "sciroppo",
        "gtt",
        "gocce",
        "fiale",
        "fiala",
        "spray",
        "gel",
        "crema",
        "granulato",
        "bustine",
        "supp",
        "supposta",
        "supposte",
        "unguento",
        "pomata",
        "sol",
        "soluzione",
        "sospensione",
        "collirio",
        "aerosol",
        "tbl",
        "cp",
        "drg",
    }
FORM_DESCRIPTORS = {
        "rivestite",
        "retard",
        "oro",
        "sublinguale",
        "sublinguali",
        "prolungato",
        "prolungata",
        "rilascio",
        "modificato",
        "masticabile",
        "depot",
        "lp",
    }
UNIT_TOKENS = {
        "mg",
        "mcg",
        "ug",
        "g",
        "kg",
        "ml",
        "l",
        "ui",
        "u",
        "dose",
        "dosi",
        "puff",
        "puffs",
    }



__all__ = [
    "PATIENT_SECTION_HEADER_RE",
    "DRUG_SCHEDULE_RE",
    "DRUG_BULLET_RE",
    "DRUG_BRACKET_TRAIL_RE",
    "DRUG_SUSPENSION_RE",
    "DRUG_SUSPENSION_DATE_RE",
    "ITALIAN_MONTHS",
    "DATE_PATS",
    "TITER_RE",
    "NUMERIC_RE",
    "CUTOFF_IN_PAREN_RE",
]