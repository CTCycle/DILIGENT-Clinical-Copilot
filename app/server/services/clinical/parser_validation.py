from __future__ import annotations

NON_DRUG_EXACT_NAMES = {
    "in riserva",
    "al bisogno",
    "se necessario",
    "dopo",
    "paziente femmina",
    "paziente maschio",
}
NON_DRUG_PREFIXES = (
    "ulteriore ciclo",
    "eventuale inizio",
)
NON_DRUG_CONTAINS = ("originariamente previsto",)
WEEKDAY_TOKENS = {
    "il",
    "la",
    "lunedi",
    "martedi",
    "mercoledi",
    "giovedi",
    "venerdi",
    "sabato",
    "domenica",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}
NON_THERAPY_LINE_PREFIXES = (
    "farmaci non assunti",
    "farmaci non in uso",
    "non assunti",
    "not taking",
    "not currently taking",
)
