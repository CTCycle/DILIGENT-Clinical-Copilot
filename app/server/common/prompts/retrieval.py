from __future__ import annotations

DILI_RAG_QUERY_TEMPLATE = """{drug_name} drug induced liver injury (DILI) {pattern_classification} pattern. Observed pattern: {r_score_summary}. Focus on latency, observed-versus-known pattern match, severity, risk factors, case reports, rechallenge outcomes, likelihood grading, management, contradictions, and association strength. Clinical context: {clinical_context}"""


def build_dili_rag_query(
    *,
    drug_name: str,
    pattern_classification: str,
    r_score_summary: str,
    clinical_context: str,
) -> str:
    return DILI_RAG_QUERY_TEMPLATE.format(
        drug_name=drug_name,
        pattern_classification=pattern_classification,
        r_score_summary=r_score_summary,
        clinical_context=clinical_context,
    )
