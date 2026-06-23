from __future__ import annotations

DRUG_EXTRACTION_PROMPT = """
Extract structured drug regimens from free-text clinical notes. Return only data
that validates against the provided JSON schema. Completeness matters: scan every
line and every bullet before answering.

Rules:
- One entry per drug explicitly present in the input; never fabricate drugs.
- Extract brand names, generic names, fixed combinations, oncology regimens,
  biologics, hormones, supplements, vitamins, and minerals when they are stated
  as medication exposure.
- Extract medication names from narrative therapy phrases such as "therapy with",
  "terapia con", "protocol with", "started on", and equivalent non-English
  wording, even when no dose is provided.
- Do not extract diagnoses, tumor staging, lab markers, procedures, dates, units,
  symptoms, or scoring systems as drugs.
- Preserve original drug names, dosage, and administration mode except for whitespace.
- For every entry, include exact source `evidence` copied from the input and
  `source_span` character offsets when possible.
- `daytime_administration` is four numeric slots: morning, midday, afternoon, night.
  Fill missing mentioned slots with 0; use decimals for half doses; use [] when no
  schedule is stated.
- Capture therapy start/suspension status and dates when stated; keep ISO dates unchanged.
- If therapy is explicitly not started, set `therapy_start_status` to false.
- Set `source` to "therapy", `historical_flag` to false, `attribution` to
  "patient", and `current_status` to "current" unless the text explicitly states
  otherwise.
- Use null for missing scalar fields and [] for missing schedules.
- Before returning, verify that each medication-like line or bullet has either
  a corresponding entry or is truly not a medication.

Return a JSON object matching `PatientDrugs` with an `entries` array.
"""

ANAMNESIS_DRUG_EXTRACTION_PROMPT = """
Extract drug mentions from free-text patient anamnesis/medical history.
Completeness matters: scan every line, bullet, and narrative treatment phrase
before answering.

Rules:
- Extract every drug name mentioned, including previous treatments, allergies, and
  medication history; never fabricate drugs.
- Extract medication exposure from phrases such as "therapy with", "terapia con",
  "protocol with", "started on", "treated with", "allergy to", and equivalent
  non-English wording, even when dose or schedule is absent.
- Extract brand names, generic names, fixed combinations, oncology regimens,
  biologics, hormones, supplements, vitamins, and minerals when they are stated
  as medication exposure.
- Use the drug name as written.
- Capture dosage, administration mode, start date/status, and suspension date/status
  only when stated.
- Populate `daytime_administration` only for explicit schedules; otherwise use [].
- Set `source` to "anamnesis" for all entries.
- For every entry, include exact source `evidence` copied from the input and
  `source_span` character offsets when possible.
- Classify `confidence` as high, moderate, or low based on source grounding.
- Classify `attribution` as patient, family_history, allergy, negated, or unclear.
- Classify `current_status` as current, past, suspected, ruled_out, or unclear.
- Do not extract diagnoses, tumor staging, lab markers, procedures, dates, units,
  symptoms, or scoring systems as drugs.
- Do not treat family-history, allergy-only, negated, or ruled-out mentions as
  active patient conditions unless the text explicitly says they apply to the patient.
- Before returning, verify that each medication-like line or phrase has either
  a corresponding entry or is truly not a medication.

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
- Include exact source evidence and `source_span` character offsets when possible.
- Classify `confidence` as high, moderate, or low and `attribution` as patient,
  family_history, negated, or unclear.
- Distinguish confirmed, suspected, ruled-out, past, and unclear diagnoses in
  `diagnosis_status`.
- Do not extract family history, allergy-only context, or ruled-out diagnoses as
  active patient diseases unless explicitly marked as patient conditions.
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
