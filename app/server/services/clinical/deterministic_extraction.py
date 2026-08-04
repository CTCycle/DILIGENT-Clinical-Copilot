from __future__ import annotations

import re

from domain.clinical.entities import (
    DiseaseContextEntry,
    DeterministicDiseaseExtractionResult,
    DeterministicDrugExtractionResult,
    DrugEntry,
    PatientDiseaseContext,
)
from services.catalogs.runtime import get_reference_catalog_snapshot

__all__ = [
    "DATE_SEQUENCE_RE",
    "DeterministicDrugExtractionResult",
    "extract_deterministic_diseases",
    "extract_regimen_drug_candidates",
    "line_has_regimen_signal",
]

DATE_TOKEN_RE = r"\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?"
HISTORICAL_RANGE_RE = re.compile(
    rf"\bdal\s+(?P<start>{DATE_TOKEN_RE})\s+(?:al|-)\s+(?P<end>{DATE_TOKEN_RE})\b",
    re.IGNORECASE,
)
DATE_SEQUENCE_RE = re.compile(DATE_TOKEN_RE, re.IGNORECASE)
REGIMEN_SIGNAL_RE = re.compile(
    r"\b("
    r"chemioterap|protocollo\s+con|terapia\s+con|schema|aggiunta\s+di|"
    r"associazion[ea]\s+con|ultima\s+somministrazione|weekly|ciclo|linea"
    r")\b",
    re.IGNORECASE,
)
SUSPENSION_SIGNAL_RE = re.compile(r"\b(sospes[oa]|interrott[oa]|stop)\b", re.IGNORECASE)
CAPITALIZED_DRUG_TOKEN_RE = re.compile(
    r"\b([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*)\b"
)
LOWER_DISEASE_PATTERNS: tuple[tuple[re.Pattern[str], dict[str, object]], ...] = (
    (
        re.compile(r"\bpyoderma\s+gangrenosum\b", re.IGNORECASE),
        {"name": "Pyoderma gangrenosum", "chronic": True, "hepatic_related": False},
    ),
    (
        re.compile(r"\bcalcifilassi\b", re.IGNORECASE),
        {"name": "Calcifilassi", "hepatic_related": False},
    ),
    (
        re.compile(r"\barteriopatia\s+periferica\s+obliterante\b", re.IGNORECASE),
        {
            "name": "Arteriopatia periferica obliterante",
            "chronic": True,
            "hepatic_related": False,
        },
    ),
    (
        re.compile(r"\binfezione\s+da\s+virus\s+influenza\s+a\b", re.IGNORECASE),
        {"name": "Infezione da virus Influenza A", "hepatic_related": False},
    ),
    (
        re.compile(r"\bpan[-\s]?ipopuitarismo\b", re.IGNORECASE),
        {"name": "Pan-ipopuitarismo", "chronic": True, "hepatic_related": False},
    ),
    (
        re.compile(r"\bmacroadenoma\s+ipofisario\b", re.IGNORECASE),
        {"name": "Macroadenoma ipofisario", "chronic": True, "hepatic_related": False},
    ),
    (
        re.compile(r"\binsufficienza\s+surrenalica\s+acuta\b", re.IGNORECASE),
        {"name": "Insufficienza surrenalica acuta", "hepatic_related": False},
    ),
    (
        re.compile(r"\binsufficienza\s+renale\s+acuta\b", re.IGNORECASE),
        {"name": "Insufficienza renale acuta", "hepatic_related": False},
    ),
    (
        re.compile(r"\bkdigo\s+g3b\b", re.IGNORECASE),
        {
            "name": "Malattia renale cronica KDIGO G3b",
            "chronic": True,
            "hepatic_related": False,
        },
    ),
    (
        re.compile(r"\bfibrillazione\s+atriale\b", re.IGNORECASE),
        {"name": "Fibrillazione atriale", "chronic": True, "hepatic_related": False},
    ),
    (
        re.compile(r"\banemia\s+normocit[ai]a\s+normocromica\b", re.IGNORECASE),
        {"name": "Anemia normocitica normocromica", "hepatic_related": False},
    ),
    (
        re.compile(r"\bcardiopatia\s+ischemica\b", re.IGNORECASE),
        {"name": "Cardiopatia ischemica", "chronic": True, "hepatic_related": False},
    ),
    (
        re.compile(r"\bcoronaropatia\s+trivasale\b", re.IGNORECASE),
        {"name": "Coronaropatia trivasale", "chronic": True, "hepatic_related": False},
    ),
    (
        re.compile(
            r"\bmalattia\s+da\s+reflusso\s+gastro[-\s]?esofageo\b", re.IGNORECASE
        ),
        {
            "name": "Malattia da reflusso gastro-esofageo",
            "chronic": True,
            "hepatic_related": False,
        },
    ),
    (
        re.compile(r"\biperplasia\s+prostatica\b", re.IGNORECASE),
        {"name": "Iperplasia prostatica", "chronic": True, "hepatic_related": False},
    ),
    (
        re.compile(r"\bmiopatia\s+da\s+statina\b", re.IGNORECASE),
        {"name": "Miopatia da statina", "hepatic_related": False},
    ),
    (
        re.compile(r"\bpolineuropatia\s+degenerativa\b", re.IGNORECASE),
        {
            "name": "Polineuropatia degenerativa",
            "chronic": True,
            "hepatic_related": False,
        },
    ),
    (
        re.compile(r"\bcolecistolitiasi\b", re.IGNORECASE),
        {"name": "Colecistolitiasi", "hepatic_related": True},
    ),
    (
        re.compile(r"\bsteatosi\s+epatica\b", re.IGNORECASE),
        {"name": "Steatosi epatica", "chronic": True, "hepatic_related": True},
    ),
    (
        re.compile(r"\bepatit(?:e|is)\b", re.IGNORECASE),
        {"name": "Epatite", "hepatic_related": True},
    ),
    (
        re.compile(r"\bcirr(?:osi|hosis)\b", re.IGNORECASE),
        {"name": "Cirrosi", "chronic": True, "hepatic_related": True},
    ),
    (
        re.compile(r"\bcolest(?:asi|asis)\b", re.IGNORECASE),
        {"name": "Colestasi", "hepatic_related": True},
    ),
    (
        re.compile(r"\bcolecistit(?:e|is)\s+acuta\b", re.IGNORECASE),
        {"name": "Colecistite acuta", "hepatic_related": True},
    ),
    (
        re.compile(r"\bpolmonit(?:e|is)\b", re.IGNORECASE),
        {"name": "Polmonite", "hepatic_related": False},
    ),
    (
        re.compile(r"\bipertension(?:e|)\b", re.IGNORECASE),
        {"name": "Ipertensione", "chronic": True, "hepatic_related": False},
    ),
    (
        re.compile(r"\bdiabet(?:e|es)\b", re.IGNORECASE),
        {"name": "Diabete", "chronic": True, "hepatic_related": False},
    ),
    (
        re.compile(r"\bobesit(?:a|y)\b", re.IGNORECASE),
        {"name": "Obesità", "chronic": True, "hepatic_related": False},
    ),
    (
        re.compile(r"\bcarcinosi\s+peritoneale\b", re.IGNORECASE),
        {"name": "Carcinosi peritoneale", "hepatic_related": False},
    ),
)
CARCINOMA_PHRASE_RE = re.compile(
    r"\b(?P<name>(?:high\s+grade\s+)?[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\s/-]{0,80}carcinoma)\b",
    re.IGNORECASE,
)
def _non_drug_tokens() -> frozenset[str]:
    return frozenset(
        get_reference_catalog_snapshot().values(
            "clinical_extraction",
            "deterministic_non_drug_tokens",
        )
    )

###############################################################################
def line_has_regimen_signal(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return False
    if HISTORICAL_RANGE_RE.search(stripped):
        return True
    if REGIMEN_SIGNAL_RE.search(stripped):
        return True
    return bool(re.search(r"(?<!\d)\+(?!\d)", stripped)) and bool(
        CAPITALIZED_DRUG_TOKEN_RE.search(stripped)
    )

###############################################################################
def extract_regimen_drug_candidates(
    line: str,
    *,
    normalize_date_token,
    normalize_entry,
) -> list[DrugEntry]:
    stripped = re.sub(r"\s+", " ", (line or "")).strip()
    if not stripped:
        return []

    date_range = HISTORICAL_RANGE_RE.search(stripped)
    start_date = normalize_date_token(date_range.group("start")) if date_range else None
    end_date = normalize_date_token(date_range.group("end")) if date_range else None
    if start_date is None:
        dates = [
            normalize_date_token(match.group(0))
            for match in DATE_SEQUENCE_RE.finditer(stripped)
        ]
        normalized_dates = [value for value in dates if value]
        if normalized_dates:
            start_date = normalized_dates[0]
        if len(normalized_dates) >= 2:
            end_date = normalized_dates[1]

    seen: set[str] = set()
    entries: list[DrugEntry] = []
    non_drug_tokens = _non_drug_tokens()
    for match in CAPITALIZED_DRUG_TOKEN_RE.finditer(stripped):
        candidate_name = match.group(1).strip()
        lowered = candidate_name.casefold()
        if lowered in seen or lowered in non_drug_tokens:
            continue
        if len(candidate_name) <= 2:
            continue
        raw_entry = DrugEntry(
            name=candidate_name,
            dosage=None,
            administration_mode=None,
            route=None,
            administration_pattern=None,
            therapy_start_status=True if start_date else None,
            therapy_start_date=start_date,
            suspension_status=True
            if end_date
            else (True if SUSPENSION_SIGNAL_RE.search(stripped) else None),
            suspension_date=end_date,
        )
        normalized = normalize_entry(
            raw_entry,
            source="anamnesis",
            historical_flag=True,
        )
        if normalized is None:
            continue
        seen.add(lowered)
        entries.append(normalized)
    return entries

###############################################################################
def extract_deterministic_diseases(
    anamnesis: str,
) -> DeterministicDiseaseExtractionResult:
    lines = [line.strip() for line in (anamnesis or "").splitlines() if line.strip()]
    matched_lines: list[str] = []
    unresolved_lines: list[str] = []
    entries: list[DiseaseContextEntry] = []
    seen: set[str] = set()

    for line in lines:
        line_entries: list[DiseaseContextEntry] = []
        for pattern, defaults in LOWER_DISEASE_PATTERNS:
            if not pattern.search(line):
                continue
            name = str(defaults["name"])
            chronic_value = defaults.get("chronic")
            chronic: bool | None = (
                chronic_value if isinstance(chronic_value, bool) else None
            )
            hepatic_related_value = defaults.get("hepatic_related")
            hepatic_related: bool | None = (
                hepatic_related_value
                if isinstance(hepatic_related_value, bool)
                else None
            )
            key = name.casefold()
            if key in seen:
                continue
            line_entries.append(
                DiseaseContextEntry(
                    name=name,
                    chronic=chronic,
                    hepatic_related=hepatic_related,
                    evidence=line[:500],
                )
            )
            seen.add(key)

        for match in CARCINOMA_PHRASE_RE.finditer(line):
            name = re.sub(r"\s+", " ", match.group("name")).strip(" ,;:.")
            key = name.casefold()
            if key in seen:
                continue
            line_entries.append(
                DiseaseContextEntry(
                    name=name,
                    chronic=True,
                    hepatic_related=False,
                    evidence=line[:500],
                )
            )
            seen.add(key)

        if line_entries:
            matched_lines.append(line)
            entries.extend(line_entries)
        elif re.search(
            r"\b(carcinom|carcinosi|epatit|cirr|steatosi|colecistit|polmonit|ipertension|diabet|obes|pyoderma|calcifilassi|arteriopatia|influenza|ipopuitarismo|macroadenoma|surrenalica|renale|fibrillazione|anemia|cardiopatia|coronaropatia|reflusso|iperplasia|miopatia|polineuropatia|colecistolitiasi)\b",
            line,
            re.IGNORECASE,
        ):
            unresolved_lines.append(line)

    return DeterministicDiseaseExtractionResult(
        context=PatientDiseaseContext(entries=entries),
        matched_lines=matched_lines,
        unresolved_lines=unresolved_lines,
    )
