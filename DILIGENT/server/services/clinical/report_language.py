from __future__ import annotations

from DILIGENT.server.common.utils.languages import resolve_supported_language_code
from DILIGENT.server.domain.clinical.entities import DrugRucamAssessment

SUPPORTED_REPORT_LANGUAGE_CODES = ("en", "it", "es", "fr", "de", "pt")

_PHRASES = {
    "en": {
        "rucam_source_reported": "RUCAM score reported directly from trusted source.",
        "rucam_structured_score": "Structured RUCAM score: {score} ({category}).",
        "rucam_not_calculated": "RUCAM not calculated due to insufficient criteria-level evidence.",
        "rucam_insufficient_data": "Insufficient criteria-level evidence for RUCAM.",
        "rucam_score_source": "Score source",
        "rucam_causality_category": "Causality category",
        "rucam_limitations": "RUCAM limitations",
        "livertox_missing": "No matching LiverTox record available.",
        "livertox_ambiguous": "Ambiguous LiverTox match.",
        "unresolved_mentions": "Unresolved Drug Mentions",
        "evidence_quality": "Evidence quality",
        "report_section_summary": "Global Synthesis and Clinical Recommendations",
        "report_section_per_drug": "Per-drug assessment",
        "unnamed_drug": "Unnamed drug",
        "candidate_matches": "Candidate matches: {candidates}.",
        "manual_curation": "Manual curation is required before causality assessment.",
        "no_matching_record": "No matching drug record found in the local knowledge base.",
        "matched_no_excerpt": "Matched drug record found, but no local LiverTox excerpt is available.",
        "deterministic_section_unavailable": "Could not produce a deterministic matched-drug section.",
    },
    "it": {
        "rucam_source_reported": "Punteggio RUCAM riportato direttamente da fonte attendibile.",
        "rucam_structured_score": "Punteggio RUCAM strutturato: {score} ({category}).",
        "rucam_not_calculated": "RUCAM non calcolato per evidenze cliniche insufficienti a livello di criteri.",
        "rucam_insufficient_data": "Dati insufficienti per il calcolo RUCAM.",
        "rucam_score_source": "Fonte del punteggio",
        "rucam_causality_category": "Categoria di causalità",
        "rucam_limitations": "Limiti RUCAM",
        "livertox_missing": "Nessun record LiverTox corrispondente disponibile.",
        "livertox_ambiguous": "Corrispondenza LiverTox ambigua.",
        "unresolved_mentions": "Menzioni Farmaco Non Risolte",
        "evidence_quality": "Qualità dell'evidenza",
        "report_section_summary": "Sintesi Globale e Raccomandazioni Cliniche",
        "report_section_per_drug": "Valutazione per farmaco",
        "unnamed_drug": "Farmaco senza nome",
        "candidate_matches": "Corrispondenze candidate: {candidates}.",
        "manual_curation": "È richiesta una revisione manuale prima della valutazione di causalità.",
        "no_matching_record": "Nessun record del farmaco corrispondente trovato nella base di conoscenza locale.",
        "matched_no_excerpt": "Record del farmaco trovato, ma nessun estratto LiverTox locale disponibile.",
        "deterministic_section_unavailable": "Impossibile produrre una sezione deterministica per il farmaco associato.",
    },
    "es": {
        "rucam_source_reported": "Puntuación RUCAM informada directamente por una fuente confiable.",
        "rucam_structured_score": "Puntuación RUCAM estructurada: {score} ({category}).",
        "rucam_not_calculated": "RUCAM no calculado por evidencia insuficiente a nivel de criterios.",
        "rucam_insufficient_data": "Evidencia insuficiente para calcular RUCAM.",
        "rucam_score_source": "Fuente de la puntuación",
        "rucam_causality_category": "Categoría de causalidad",
        "rucam_limitations": "Limitaciones de RUCAM",
        "livertox_missing": "No hay registro LiverTox coincidente disponible.",
        "livertox_ambiguous": "Coincidencia LiverTox ambigua.",
        "unresolved_mentions": "Menciones de Fármacos No Resueltas",
        "evidence_quality": "Calidad de la evidencia",
        "report_section_summary": "Síntesis Global y Recomendaciones Clínicas",
        "report_section_per_drug": "Evaluación por fármaco",
        "unnamed_drug": "Fármaco sin nombre",
        "candidate_matches": "Coincidencias candidatas: {candidates}.",
        "manual_curation": "Se requiere revisión manual antes de la evaluación de causalidad.",
        "no_matching_record": "No se encontró un registro de fármaco coincidente en la base local.",
        "matched_no_excerpt": "Se encontró registro del fármaco, pero no hay extracto local de LiverTox.",
        "deterministic_section_unavailable": "No fue posible generar una sección determinista del fármaco.",
    },
    "fr": {
        "rucam_source_reported": "Score RUCAM rapporté directement depuis une source fiable.",
        "rucam_structured_score": "Score RUCAM structuré : {score} ({category}).",
        "rucam_not_calculated": "RUCAM non calculé faute de preuves suffisantes au niveau des critères.",
        "rucam_insufficient_data": "Données insuffisantes pour calculer le RUCAM.",
        "rucam_score_source": "Source du score",
        "rucam_causality_category": "Catégorie de causalité",
        "rucam_limitations": "Limites du RUCAM",
        "livertox_missing": "Aucun enregistrement LiverTox correspondant disponible.",
        "livertox_ambiguous": "Correspondance LiverTox ambiguë.",
        "unresolved_mentions": "Mentions de Médicaments Non Résolues",
        "evidence_quality": "Qualité des preuves",
        "report_section_summary": "Synthèse Globale et Recommandations Cliniques",
        "report_section_per_drug": "Évaluation par médicament",
        "unnamed_drug": "Médicament sans nom",
        "candidate_matches": "Correspondances candidates : {candidates}.",
        "manual_curation": "Une revue manuelle est requise avant l'évaluation de causalité.",
        "no_matching_record": "Aucun enregistrement médicamenteux correspondant dans la base locale.",
        "matched_no_excerpt": "Enregistrement du médicament trouvé, mais aucun extrait local LiverTox n'est disponible.",
        "deterministic_section_unavailable": "Impossible de produire une section déterministe pour le médicament associé.",
    },
    "de": {
        "rucam_source_reported": "RUCAM-Wert direkt aus vertrauenswürdiger Quelle übernommen.",
        "rucam_structured_score": "Strukturierter RUCAM-Score: {score} ({category}).",
        "rucam_not_calculated": "RUCAM wurde wegen unzureichender kriterienspezifischer Evidenz nicht berechnet.",
        "rucam_insufficient_data": "Unzureichende Evidenz für die RUCAM-Berechnung.",
        "rucam_score_source": "Score-Quelle",
        "rucam_causality_category": "Kausalitätskategorie",
        "rucam_limitations": "RUCAM-Einschränkungen",
        "livertox_missing": "Kein passender LiverTox-Eintrag verfügbar.",
        "livertox_ambiguous": "Mehrdeutige LiverTox-Zuordnung.",
        "unresolved_mentions": "Nicht Aufgelöste Arzneimittel-Nennungen",
        "evidence_quality": "Evidenzqualität",
        "report_section_summary": "Globale Synthese und Klinische Empfehlungen",
        "report_section_per_drug": "Arzneimittelbezogene Bewertung",
        "unnamed_drug": "Unbenanntes Arzneimittel",
        "candidate_matches": "Mögliche Treffer: {candidates}.",
        "manual_curation": "Vor der Kausalitätsbewertung ist eine manuelle Prüfung erforderlich.",
        "no_matching_record": "Kein passender Arzneimitteleintrag in der lokalen Wissensbasis gefunden.",
        "matched_no_excerpt": "Arzneimitteleintrag gefunden, aber kein lokaler LiverTox-Auszug verfügbar.",
        "deterministic_section_unavailable": "Ein deterministischer Abschnitt zum zugeordneten Arzneimittel konnte nicht erstellt werden.",
    },
    "pt": {
        "rucam_source_reported": "Pontuação RUCAM informada diretamente por fonte confiável.",
        "rucam_structured_score": "Pontuação RUCAM estruturada: {score} ({category}).",
        "rucam_not_calculated": "RUCAM não calculado por evidência insuficiente em nível de critérios.",
        "rucam_insufficient_data": "Evidência insuficiente para cálculo do RUCAM.",
        "rucam_score_source": "Fonte da pontuação",
        "rucam_causality_category": "Categoria de causalidade",
        "rucam_limitations": "Limitações do RUCAM",
        "livertox_missing": "Nenhum registro LiverTox correspondente disponível.",
        "livertox_ambiguous": "Correspondência LiverTox ambígua.",
        "unresolved_mentions": "Menções de Fármacos Não Resolvidas",
        "evidence_quality": "Qualidade da evidência",
        "report_section_summary": "Síntese Global e Recomendações Clínicas",
        "report_section_per_drug": "Avaliação por fármaco",
        "unnamed_drug": "Fármaco sem nome",
        "candidate_matches": "Correspondências candidatas: {candidates}.",
        "manual_curation": "É necessária curadoria manual antes da avaliação de causalidade.",
        "no_matching_record": "Nenhum registro de fármaco correspondente foi encontrado na base local.",
        "matched_no_excerpt": "Registro do fármaco encontrado, mas sem excerto local do LiverTox.",
        "deterministic_section_unavailable": "Não foi possível produzir uma seção determinística para o fármaco associado.",
    },
}


def resolve_report_language(code: str | None) -> str:
    resolved = resolve_supported_language_code(code)
    if resolved == "en" and (code or "").strip().lower().startswith("pt"):
        return "pt"
    return resolved if resolved in SUPPORTED_REPORT_LANGUAGE_CODES else "en"


def phrase(key: str, language: str, **values: object) -> str:
    lang = resolve_report_language(language)
    table = _PHRASES.get(lang, _PHRASES["en"])
    if key not in table:
        raise KeyError(f"Missing phrase key: {key}")
    return table[key].format(**values)


def rucam_summary_text(assessment: DrugRucamAssessment, language: str) -> str:
    lang = resolve_report_language(language)
    if assessment.calculation_method == "source_reported":
        return phrase("rucam_source_reported", lang)
    if assessment.total_score is None:
        return phrase("rucam_not_calculated", lang)
    return phrase(
        "rucam_structured_score",
        lang,
        score=assessment.total_score,
        category=assessment.causality_category,
    )


def report_heading(key: str, language: str) -> str:
    return phrase(key, language)


def requires_language_repair(text: str, expected_language: str) -> bool:
    lang = resolve_report_language(expected_language)
    if lang == "en":
        return False
    lower = (text or "").lower()
    return "estimated rucam" in lower or "confidence" in lower or "limitations" in lower
