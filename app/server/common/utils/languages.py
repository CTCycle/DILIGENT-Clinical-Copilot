from __future__ import annotations

import re


SUPPORTED_REPORT_LANGUAGES: tuple[str, ...] = ("en", "it", "de", "fr", "es")
TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
MISSING_VISIT_LABEL_BY_LANGUAGE: dict[str, str] = {
    "en": "Not provided",
    "it": "Non disponibile",
    "de": "Nicht angegeben",
    "fr": "Non renseignée",
    "es": "No proporcionada",
}

LANGUAGE_HINTS: dict[str, set[str]] = {
    "en": {
        "patient",
        "anamnesis",
        "drug",
        "drugs",
        "visit",
        "bilirubin",
        "jaundice",
        "liver",
        "history",
        "therapy",
        "suspended",
        "alkaline",
        "phosphatase",
        "elevated",
        "improved",
        "worsening",
    },
    "it": {
        "paziente",
        "anamnesi",
        "farmaco",
        "farmaci",
        "visita",
        "bilirubina",
        "ittero",
        "fegato",
        "terapia",
        "sospeso",
        "sospensione",
        "epatotossicita",
        "epatotossicità",
        "miglioramento",
        "peggioramento",
        "transaminasi",
    },
    "de": {
        "patient",
        "anamnese",
        "arzneimittel",
        "medikament",
        "medikamente",
        "besuch",
        "bilirubin",
        "ikterus",
        "leber",
        "therapie",
        "abgesetzt",
        "erhoht",
        "erhöht",
        "verschlechterung",
        "besserung",
        "transaminasen",
    },
    "fr": {
        "patient",
        "anamnese",
        "anamnesis",
        "médicament",
        "médicaments",
        "medicament",
        "medicaments",
        "visite",
        "bilirubine",
        "ictère",
        "ictere",
        "foie",
        "thérapie",
        "therapie",
        "arrêt",
        "arret",
        "aggravation",
        "amélioration",
        "amelioration",
    },
    "es": {
        "paciente",
        "anamnesis",
        "fármaco",
        "fármacos",
        "farmaco",
        "farmacos",
        "medicamento",
        "medicamentos",
        "visita",
        "bilirrubina",
        "ictericia",
        "hígado",
        "higado",
        "terapia",
        "suspendido",
        "empeoramiento",
        "mejoria",
        "mejoría",
        "transaminasas",
    },
}

LANGUAGE_FUNCTION_HINTS: dict[str, set[str]] = {
    "en": {"the", "and", "with", "without", "after", "before", "during", "from"},
    "it": {"il", "la", "con", "senza", "dopo", "prima", "durante", "dal", "della"},
    "de": {"der", "die", "das", "mit", "ohne", "nach", "vor", "wahrend", "während"},
    "fr": {"le", "la", "les", "avec", "sans", "apres", "après", "avant", "pendant"},
    "es": {"el", "la", "los", "las", "con", "sin", "despues", "después", "antes"},
}

LANGUAGE_PHRASE_HINTS: dict[str, tuple[str, ...]] = {
    "en": ("liver injury", "drug induced", "alkaline phosphatase"),
    "it": ("danno epatico", "epatotossicita", "fosfatasi alcalina"),
    "de": ("leberschaden", "arzneimittel induziert", "alkalische phosphatase"),
    "fr": ("atteinte hepatique", "lésion hépatique", "phosphatase alcaline"),
    "es": ("lesion hepatica", "lesión hepática", "fosfatasa alcalina"),
}

LANGUAGE_DIACRITIC_HINTS: dict[str, set[str]] = {
    "en": set(),
    "it": {"à", "è", "é", "ì", "ò", "ù"},
    "de": {"ä", "ö", "ü", "ß"},
    "fr": {"à", "â", "ç", "é", "è", "ê", "ë", "î", "ï", "ô", "ù", "û", "ü", "œ"},
    "es": {"á", "é", "í", "ó", "ú", "ü", "ñ"},
}

VALIDATION_MESSAGE_BUNDLES: dict[str, dict[str, str]] = {
    "en": {
        "missing_anamnesis": "Provide the anamnesis.",
        "missing_visit_date": "Provide the visit date.",
        "missing_timed_drug": (
            "Provide at least one drug with start, stop, or other timing information."
        ),
        "insufficient_labs": (
            "Provide laboratory data sufficient to determine hepatotoxicity pattern, "
            "ideally dated ALT or AST, ALP, and bilirubin."
        ),
    },
    "it": {
        "missing_anamnesis": "Fornire l’anamnesi.",
        "missing_visit_date": "Fornire la data della visita.",
        "missing_timed_drug": (
            "Fornire almeno un farmaco con data di inizio, sospensione o altra "
            "informazione temporale."
        ),
        "insufficient_labs": (
            "Fornire dati laboratoristici sufficienti per determinare il pattern di "
            "epatotossicità, idealmente ALT o AST datati, ALP e bilirubina."
        ),
    },
    "de": {
        "missing_anamnesis": "Bitte Anamnese angeben.",
        "missing_visit_date": "Bitte Besuchsdatum angeben.",
        "missing_timed_drug": (
            "Bitte mindestens ein Arzneimittel mit Start-, Stopp- oder anderen "
            "Zeitangaben angeben."
        ),
        "insufficient_labs": (
            "Bitte ausreichend Laborwerte zur Bestimmung des Hepatotoxizitätsmusters "
            "angeben, idealerweise datiertes ALT oder AST, ALP und Bilirubin."
        ),
    },
    "fr": {
        "missing_anamnesis": "Veuillez fournir l’anamnèse.",
        "missing_visit_date": "Veuillez fournir la date de visite.",
        "missing_timed_drug": (
            "Veuillez fournir au moins un médicament avec une date de début, d’arrêt "
            "ou une autre information temporelle."
        ),
        "insufficient_labs": (
            "Veuillez fournir des données biologiques suffisantes pour déterminer le "
            "profil d’hépatotoxicité, idéalement ALT ou AST datés, PAL et bilirubine."
        ),
    },
    "es": {
        "missing_anamnesis": "Proporcione la anamnesis.",
        "missing_visit_date": "Proporcione la fecha de la visita.",
        "missing_timed_drug": (
            "Proporcione al menos un fármaco con fecha de inicio, suspensión u otra "
            "información temporal."
        ),
        "insufficient_labs": (
            "Proporcione datos de laboratorio suficientes para determinar el patrón de "
            "hepatotoxicidad, idealmente ALT o AST con fecha, FA y bilirrubina."
        ),
    },
}


def resolve_supported_language_code(language: str | None) -> str:
    normalized = (language or "").strip().casefold()
    if not normalized:
        return "en"
    for code in SUPPORTED_REPORT_LANGUAGES:
        if normalized.startswith(code):
            return code
    return "en"
