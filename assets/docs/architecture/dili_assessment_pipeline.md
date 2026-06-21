# DILI Assessment Pipeline
Last updated: 2026-06-21

## Section Extraction Contract
`POST /api/clinical/jobs` uses deterministic section extraction for structural input splitting. The extractor preserves source-verbatim section bodies after newline normalization and records canonical key, payload key, raw and normalized heading, match strategy, confidence score, heading line span, body line span, character span, verbatim coherence, review requirement, and source hash.

LLM fallback is not part of section extraction. Content inference, fallback assignment, low-confidence semantic matches, duplicate headings, or ambiguous headings are surfaced as diagnostics or review signals.

Aggregate section confidence is the minimum confidence across required extracted sections. Preflight blocks analysis when aggregate confidence is below `0.65`; confidence from `0.65` through below `0.85` is non-blocking but requires operator review acknowledgement in the DILI workspace before starting a job.

## Structured Extraction Contract
Drug, disease, and laboratory extraction use provider-agnostic structured LLM calls for both cloud and local providers. The active runtime provider and model are resolved from persisted model configuration.

Structured LLM output is validated against Pydantic schemas and semantic guardrails. Invalid or contaminated output is retried with the rejected output and validation feedback. If bounded LLM attempts fail, the pipeline falls back to direct rule-based parsers and records fallback warnings.

Structured output parsing accepts strict JSON objects only. Prose-wrapped JSON or trailing prose is rejected by default; repair logs include schema name, attempt count, output length, short output hash, and error type, but not raw model output.

## Strategy Matrix
Therapy, anamnesis, disease, and laboratory extraction attempt structured LLM extraction first when a configured provider is available.

- `llm`: structured LLM extraction produced schema-valid, semantically acceptable output.
- `hybrid`: structured LLM extraction succeeded but direct rule-based evidence is retained for audit or fallback support.
- `deterministic`: bounded LLM attempts failed or no structured client was available, so direct rule-based parsing was used.

## Hepatic Pattern Resolution
If an explicit hepatic pattern is present in the laboratory source, it becomes the final pattern with source `provided`. The calculated R-ratio pattern is preserved separately. If both exist and differ, the pipeline emits `hepatic_pattern_source_calculation_conflict`.

If no explicit pattern exists, the calculated value is used. If neither is available, the final value is `indeterminate` with source `undetermined`.

## Match Audit
LiverTox and RxNav matching keeps raw extracted name, normalized name, matched LiverTox name, RxNorm RXCUI when available, match status, confidence, reason, candidates, rejected candidates, origins, raw mentions, and extraction metadata.

Pipeline issues are emitted for missing LiverTox matches, ambiguous LiverTox matches, low-confidence matches, and unvalidated RxNav aliases.

Per-drug clinical assessments carry claim envelopes and narrative limits. Claim review output distinguishes source-text claims, RUCAM-linked claims, unsupported or unknown-source claims, and generated limitations so report consumers can see which statements require review.

## Failure Modes
- Missing required sections block structural preprocessing.
- Duplicate required headings block deterministic section extraction.
- Ambiguous or low-confidence section assignments require review.
- Missing or ambiguous external drug matches do not force a match.
- LLM structured output failures fall back to direct deterministic parsing after bounded retries.
- Session persistence is mandatory for successful clinical and revision jobs. Persistence write failures raise a service dependency error instead of returning an apparently successful unpersisted report.
- Failed job payloads omit raw clinical input and patient image content; job-level errors expose generic failure text plus sanitized failure metadata.

## Testing Matrix
Regression fixtures under `app/tests/fixtures/dili_pipeline_audit` cover clean, noisy, incomplete, and adversarial clinical documents, including duplicate headings, ambiguous headings, structured and unstructured therapy, structured and noisy laboratories, hepatic-pattern conflicts, family-history disease mentions, allergy-only drug mentions, negated diagnoses, combination therapy, brand/generic variants, and missing LiverTox matches.
