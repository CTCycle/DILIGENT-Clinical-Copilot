from __future__ import annotations

DILI_RAG_QUERY_PROMPT = (
    "{name} drug induced liver injury (DILI) {classification} pattern. "
    "Observed pattern: {r_part}. "
    "Focus on latency, observed-vs-known pattern match, severity, risk factors, "
    "case reports, rechallenge outcomes, likelihood grading, management, "
    "contradictions, and association strength. Clinical context: {clinical}"
)
