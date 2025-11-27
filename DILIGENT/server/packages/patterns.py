from __future__ import annotations

import re

# -----------------------------------------------------------------------------
# Patient section parsing
# -----------------------------------------------------------------------------
PATIENT_SECTION_HEADER_RE = re.compile(
    r"^[ 	]*#{1,6}[ 	]+(.+?)\s*$", re.MULTILINE
)

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
DRUG_START_DATE_RE = re.compile(
    r"""
    (?P<prefix>
        \b(?:iniz(?:io|iat[oaie])|avviat[oaie]|ripres[oaie]|riprend[ei]re|
        assunzion[ei]|in\s+terapia|in\s+trattamento|terapia|trattamento)
        (?:\s+(?:dal|da|dall['’]|il))?
        |
        \b(?:dal|da|dall['’]|il)\b
    )
    \s*
    (?P<date>\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# -----------------------------------------------------------------------------
# LiverTox excerpt sanitation patterns
# -----------------------------------------------------------------------------
LIVERTOX_HEADER_RE = re.compile(
    r"^\s*livertox\s+LiverTox[:\s-]+Clinical and Research Information on Drug-Induced"
    r" Liver Injury\s+\d{4}\s+National Institute of Diabetes and Digestive and Kidney"
    r" Diseases\s+Bethesda\s*\(MD\)\s+books-source-type\.??\s*",
    re.IGNORECASE,
)

LIVERTOX_FOOTER_RE = re.compile(
    r"OTHER REFERENCE LINKS\s+Recent References on\s+.+?:\s+from\s+PubMed\.gov\s+"
    r"Trials on\s+.+?:\s+from\s+ClinicalTrials\.gov\.?\s*$",
    re.IGNORECASE,
)


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
    "ret",
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
    "DRUG_START_DATE_RE", 
    "LIVERTOX_HEADER_RE",
    "LIVERTOX_FOOTER_RE",
]
