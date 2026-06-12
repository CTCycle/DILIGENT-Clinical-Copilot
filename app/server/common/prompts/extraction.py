from __future__ import annotations

DRUG_EXTRACTION_PROMPT = """
Extract structured drug regimens from free-text clinical notes. Return only data
that validates against the provided JSON schema.

Rules:
- One entry per drug explicitly present in the input; never fabricate drugs.
- Preserve original drug names, dosage, and administration mode except for whitespace.
- `daytime_administration` is four numeric slots: morning, midday, afternoon, night.
  Fill missing mentioned slots with 0; use decimals for half doses; use [] when no
  schedule is stated.
- Capture therapy start/suspension status and dates when stated; keep ISO dates unchanged.
- If therapy is explicitly not started, set `therapy_start_status` to false.
- Use null for missing scalar fields and [] for missing schedules.

Return a JSON object matching `PatientDrugs` with an `entries` array.
"""

ANAMNESIS_DRUG_EXTRACTION_PROMPT = """
Extract drug mentions from free-text patient anamnesis/medical history.

Rules:
- Extract every drug name mentioned, including previous treatments, allergies, and
  medication history; never fabricate drugs.
- Use the drug name as written.
- Capture dosage, administration mode, start date/status, and suspension date/status
  only when stated.
- Populate `daytime_administration` only for explicit schedules; otherwise use [].
- Set `source` to "anamnesis" for all entries.

Return a JSON object matching `PatientDrugs` with an `entries` array.
"""

ANAMNESIS_DISEASE_EXTRACTION_PROMPT = """
Extract structured disease information from anamnesis/medical history, including
non-English notes.

Rules:
- Extract only explicitly mentioned clinically relevant diseases/conditions.
- Prefer specific labels over generic labels.
- For each condition, return name, occurrence_time, timeline, severity,
  diagnosis_status, symptoms, clinical_context, chronic, hepatic_related, and
  a short supporting evidence snippet when available.
- Normalize spacing only; do not invent diseases or attributes.

Return a JSON object matching `PatientDiseaseContext` with an `entries` array.
"""

CLINICAL_LAB_EXTRACTION_PROMPT = """
Extract longitudinal liver-related labs and onset clues from free-text clinical
sections. Always match the provided JSON schema.

Rules:
- Extract only labs explicitly present in the text.
- Capture ALT, AST, ALP, total/direct bilirubin, GGT, INR, albumin, and other
  explicit liver-related markers when present.
- For each lab entry, include original marker text, numeric value or raw value text,
  unit, ULN or raw ULN text, sample date, relative timing, evidence snippet, and
  source (`laboratory_analysis` or `anamnesis`).
- Extract onset clues when present: onset_date, onset_basis
  (first_symptom, first_abnormal_lab, visit_proxy, unknown), and evidence.
- Use null/empty structures for missing information; do not invent values.

Return {"entries": [ClinicalLabEntry...], "onset_context": LiverInjuryOnsetContext | null}.
"""

CLINICAL_SECTION_EXTRACTION_PROMPT = """
Deterministic parsing failed. Return JSON only with exactly:
- anamnesis
- therapy
- lab_analysis

Rules:
- Values are verbatim excerpts from the source text.
- Preserve language and wording; do not summarize, translate, infer, normalize, or fabricate.
- If missing, return an empty string.
- Do not use markdown fences.
"""
