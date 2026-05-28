from __future__ import annotations

import re
from dataclasses import dataclass

from domain.clinical.entities import DiseaseContextEntry, DrugEntry, PatientDiseaseContext

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
NON_DRUG_TOKENS = frozenset(
    {
        "dal",
        "al",
        "alla",
        "all",
        "con",
        "senza",
        "schema",
        "protocollo",
        "terapia",
        "chemioterapia",
        "secondo",
        "linea",
        "ultima",
        "somministrazione",
        "entrambi",
        "sola",
        "weekly",
        "pd",
        "pfi",
        "mrcp",
        "pet",
        "ct",
        "tc",
        "eco",
        "labor",
        "bil",
        "alp",
        "alt",
        "ast",
        "pcr",
        "high",
        "grade",
        "figo",
        "stato",
        "diagnosi",
        "nozione",
        "previsto",
        "nega",
        "paziente",
        "carcinosi",
        "carcinoma",
        "sludge",
        "magnetic",
        "resonance",
        "cholangio",
        "pancreatography",
    }
)


@dataclass(frozen=True)
class DeterministicDrugExtractionResult:
    entries: list[DrugEntry]
    unresolved_lines: list[str]
    regimen_lines: list[str]


@dataclass(frozen=True)
class DeterministicDiseaseExtractionResult:
    context: PatientDiseaseContext
    matched_lines: list[str]
    unresolved_lines: list[str]


def line_has_regimen_signal(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return False
    if HISTORICAL_RANGE_RE.search(stripped):
        return True
    if REGIMEN_SIGNAL_RE.search(stripped):
        return True
    return "+" in stripped and bool(CAPITALIZED_DRUG_TOKEN_RE.search(stripped))


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
        dates = [normalize_date_token(match.group(0)) for match in DATE_SEQUENCE_RE.finditer(stripped)]
        normalized_dates = [value for value in dates if value]
        if normalized_dates:
            start_date = normalized_dates[0]
        if len(normalized_dates) >= 2:
            end_date = normalized_dates[1]

    seen: set[str] = set()
    entries: list[DrugEntry] = []
    for match in CAPITALIZED_DRUG_TOKEN_RE.finditer(stripped):
        candidate_name = match.group(1).strip()
        lowered = candidate_name.casefold()
        if lowered in seen or lowered in NON_DRUG_TOKENS:
            continue
        if len(candidate_name) <= 2:
            continue
        raw_entry = DrugEntry(
            name=candidate_name,
            therapy_start_status=True if start_date else None,
            therapy_start_date=start_date,
            suspension_status=True if end_date else (True if SUSPENSION_SIGNAL_RE.search(stripped) else None),
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


def extract_deterministic_diseases(anamnesis: str) -> DeterministicDiseaseExtractionResult:
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
            key = name.casefold()
            if key in seen:
                continue
            line_entries.append(
                DiseaseContextEntry(
                    name=name,
                    chronic=defaults.get("chronic") if isinstance(defaults.get("chronic"), bool) else None,
                    hepatic_related=defaults.get("hepatic_related") if isinstance(defaults.get("hepatic_related"), bool) else None,
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
        elif re.search(r"\b(carcinom|carcinosi|epatit|cirr|steatosi|colecistit|polmonit|ipertension|diabet|obes)\b", line, re.IGNORECASE):
            unresolved_lines.append(line)

    return DeterministicDiseaseExtractionResult(
        context=PatientDiseaseContext(entries=entries),
        matched_lines=matched_lines,
        unresolved_lines=unresolved_lines,
    )
