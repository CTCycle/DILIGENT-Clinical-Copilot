from __future__ import annotations

PATIENT_TIMELINE_EXTRACTION_PROMPT = """
Extract chronological, patient-specific clinical events from the provided case context.
Return a JSON object matching the provided schema.

Coverage when evidence exists:
- Therapy starts, changes, discontinuations/suspensions.
- Disease manifestations and symptom onset.
- Laboratory milestones, especially liver-related tests.
- Other clinically relevant dated, relative, duration, recurring, uncertain, or
  ordering-only events.

Rules:
- Extract only source-supported events; never invent dates, treatments, diseases, or labs.
- Keep titles concise and descriptions factual.
- Use event_type: therapy, disease, lab, or other.
- Use timing_type: explicit_date, relative, duration, recurring, uncertain, or ordering.
- Use ISO dates only when explicit or inferable with high confidence.
- Use event_date_end for an explicit date range; do not collapse ranges into a point.
- Set date_precision to day, month, or year. Set date_certainty to explicit, inferred,
  or uncertain and explain uncertainty_reason when applicable.
- Preserve exact timing phrases, source snippets, source provenance, confidence, and rationale.
- Link related events by event_id when timing depends on another event.
"""
