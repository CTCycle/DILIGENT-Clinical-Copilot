# DILI Assessment Pipeline
Last updated: 2026-06-19

## Section Extraction Contract
`POST /api/clinical/jobs` uses deterministic section extraction for structural input splitting. The extractor preserves source-verbatim section bodies after newline normalization and records canonical key, payload key, raw and normalized heading, match strategy, confidence score, heading line span, body line span, character span, verbatim coherence, review requirement, and source hash.

LLM fallback is not part of section extraction. Content inference, fallback assignment, low-confidence semantic matches, duplicate headings, or ambiguous headings are surfaced as diagnostics or review signals.

## Tool Schema Contract
Deterministic extraction tool manifests live in `app/resources/tools/extraction_tools.json`. Runtime dispatch lives in `app/server/services/extraction_tools`.

Every tool definition includes an OpenAI-compatible function schema, version, supported section types, default regex profile, allowed profiles, and a deterministic return contract. Tool results include match text, normalized value, exact character span, line number, source section, pattern id, confidence, and warnings.

## Strategy Matrix
Therapy and laboratory extraction run deterministic parsing first.

- `deterministic`: structured coverage and evidence-span coverage meet thresholds, with no unresolved meaningful lines.
- `hybrid`: deterministic extraction found useful evidence but unresolved or ambiguous fragments remain.
- `llm`: deterministic structure is insufficient for the section.

Anamnesis remains LLM-preferred for free-text enrichment, while deterministic extraction remains available for obvious entities and fallback.

## Hepatic Pattern Resolution
If an explicit hepatic pattern is present in the laboratory source, it becomes the final pattern with source `provided`. The calculated R-ratio pattern is preserved separately. If both exist and differ, the pipeline emits `hepatic_pattern_source_calculation_conflict`.

If no explicit pattern exists, the calculated value is used. If neither is available, the final value is `indeterminate` with source `undetermined`.

## Match Audit
LiverTox and RxNav matching keeps raw extracted name, normalized name, matched LiverTox name, RxNorm RXCUI when available, match status, confidence, reason, candidates, rejected candidates, origins, raw mentions, and extraction metadata.

Pipeline issues are emitted for missing LiverTox matches, ambiguous LiverTox matches, low-confidence matches, and unvalidated RxNav aliases.

## Failure Modes
- Missing required sections block structural preprocessing.
- Duplicate required headings block deterministic section extraction.
- Ambiguous or low-confidence section assignments require review.
- Missing or ambiguous external drug matches do not force a match.
- Unsupported native LLM tool calling returns a structured provider error.

## Testing Matrix
Regression fixtures under `app/tests/fixtures/dili_pipeline_audit` cover clean, noisy, incomplete, and adversarial clinical documents, including duplicate headings, ambiguous headings, structured and unstructured therapy, structured and noisy laboratories, hepatic-pattern conflicts, family-history disease mentions, allergy-only drug mentions, negated diagnoses, combination therapy, brand/generic variants, and missing LiverTox matches.
