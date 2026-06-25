# DILI Assessment Pipeline
Last updated: 2026-06-25

## Section Extraction Contract
`POST /api/clinical/jobs` uses deterministic section extraction for structural input splitting. The extractor preserves source-verbatim section bodies after newline normalization and records canonical key, payload key, raw and normalized heading, match strategy, confidence score, heading line span, body line span, character span, verbatim coherence, review requirement, and source hash.

LLM fallback is not part of section extraction. Content inference, fallback assignment, low-confidence semantic matches, duplicate headings, or ambiguous headings are surfaced as diagnostics or review signals.

Aggregate section confidence is the minimum confidence across required extracted sections. Preflight blocks analysis when aggregate confidence is below `0.65`; confidence from `0.65` through below `0.85` is non-blocking but requires operator review acknowledgement in the DILI workspace before starting a job.

## Structured Extraction Contract
Drug, disease, and laboratory extraction use provider-agnostic structured LLM calls for both cloud and local providers. The active runtime provider and model are resolved from persisted model configuration.

Structured LLM output is validated against Pydantic schemas and semantic guardrails. Invalid or contaminated output is retried with the rejected output and validation feedback. If bounded LLM attempts fail, the pipeline falls back to direct rule-based parsers and records fallback warnings.

Therapy extraction is always full-section and context-aware when a structured model is available. The complete normalized therapy corpus is submitted in one structured request so medication names, schedules, and continuation lines remain associated across line breaks. Deterministic extraction is retained as a grounding reference and failure fallback, not as a reason to skip corpus-level extraction.

Structured output parsing accepts strict JSON objects only. Prose-wrapped JSON or trailing prose is rejected by default; repair logs include schema name, attempt count, output length, short output hash, and error type, but not raw model output.

For local Ollama extraction, disease and laboratory stages use compact structured payloads and schema-echo repair guards to reduce failure rates on smaller local models while preserving the persisted downstream result shape.

## Strategy Matrix
Therapy, anamnesis, disease, and laboratory extraction attempt structured LLM extraction first when a configured provider is available.

- `llm`: structured LLM extraction produced schema-valid, semantically acceptable output.
- `hybrid`: structured LLM extraction succeeded but direct rule-based evidence is retained for audit or fallback support.
- `deterministic`: bounded LLM attempts failed or no structured client was available, so direct rule-based parsing was used.

Clinical job result payloads include runtime diagnostics for extraction stages: resolved provider and model, stage elapsed time, fallback reasons, and structured failure kinds when a bounded structured stage falls back.

## Hepatic Pattern Resolution
If an explicit hepatic pattern is present in the laboratory source, it becomes the final pattern with source `provided`. The calculated R-ratio pattern is preserved separately. If both exist and differ, the pipeline emits `hepatic_pattern_source_calculation_conflict`.

If no explicit pattern exists, the calculated value is used. If neither is available, the final value is `indeterminate` with source `undetermined`.

## Match Audit
Drug matching uses a first-class local resolution decision layer. Extracted mentions are normalized once, regimen parents and components are preserved separately, RxNav candidates are built from the persisted RxNorm-backed catalog, LiverTox candidates are built from local monographs, and a deterministic policy decides whether a match can be accepted automatically.

LiverTox and RxNav matching keeps raw extracted name, normalized name, matched LiverTox name, RxNorm RXCUI when available, RxNav validation status, match status, confidence, reason, candidates, rejected candidates, origins, raw mentions, extraction metadata, and the full `DrugResolutionDecision`.

Candidate admission is intentionally high recall: structurally plausible medication labels are not discarded merely because they lack an exact local alias or recognized INN suffix. Exact, alias, normalized, structured ingredient/brand, and bounded unique spelling resolution run before a match is declared missing.

Before candidate selection and downstream RUCAM, retrieval, matching, and consultation, therapy and anamnesis mentions are deduplicated by normalized identity. Raw section entries remain available for audit, while the deduplicated artifact records origins, raw mentions, evidence snippets, and the selected clinical entry.

When a plausible mention remains unresolved, ambiguous, or low-confidence, the configured parser model may propose generic names, alternate names, or active ingredients in one structured batch. These proposals are candidate generation only. Every proposed identity is re-run through the local RxNav/LiverTox resolver, and no LiverTox evidence is accepted unless the local policy produces a unique accepted match. Unvalidated or ambiguous proposals remain unresolved and auditable.

Resolution statuses are:
- `accepted_exact_livertox`
- `accepted_rxnav_validated`
- `accepted_livertox_without_rxnav`
- `ambiguous_requires_review`
- `missing_rxnav`
- `missing_livertox`
- `rejected_false_positive`

An available LiverTox excerpt means evidence text is available for the accepted monograph. It is not treated as proof that a ranked or ambiguous candidate is clinically correct.

Pipeline issues are emitted for missing LiverTox matches, ambiguous LiverTox matches, low-confidence matches, and unvalidated RxNav aliases.

Per-drug clinical assessments carry claim envelopes and narrative limits. Claim review output distinguishes source-text claims, RUCAM-linked claims, unsupported or unknown-source claims, and generated limitations so report consumers can see which statements require review.

The rendered per-drug report consolidates evidence-match status, the matched local record, evidence warnings, claim-review requirements, and RUCAM limitations into one localized clinical commentary. The structured claim and evidence fields remain unchanged in persisted audit data.

## RAG Readiness
Clinical preflight reports whether the configured RAG embedding backend is ready. When the active vector index depends on Ollama embeddings, Ollama and the configured embedding model must be available before a RAG-enabled job starts.

If Ollama is unavailable, the DILI Agent offers three explicit choices: retry after starting Ollama, run the pending assessment once without RAG, or cancel. Running without RAG does not change the saved model configuration. Job submission repeats the readiness check to prevent a stale successful preflight from starting a RAG-enabled job after the dependency becomes unavailable.

If retrieval becomes unavailable after a job starts, the report continues safely without supporting RAG documents and records one aggregated pipeline warning listing the affected drugs.

## Failure Modes
- Missing required sections block structural preprocessing.
- Duplicate required headings block deterministic section extraction.
- Ambiguous or low-confidence section assignments require review.
- Missing or ambiguous external drug matches do not force a match.
- Ambiguous drug matches are included for review but are not used as authoritative LiverTox evidence.
- Broad categories and rejected false-positive extracted text remain audit-only and do not become concrete drug matches.
- LLM structured output failures fall back to direct deterministic parsing after bounded retries.
- Session persistence is mandatory for successful clinical and revision jobs. Persistence write failures raise a service dependency error instead of returning an apparently successful unpersisted report.
- Failed job payloads omit raw clinical input and patient image content; job-level errors expose generic failure text plus sanitized failure metadata.

## Testing Matrix
Regression fixtures under `app/tests/fixtures/dili_pipeline_audit` cover clean, noisy, incomplete, and adversarial clinical documents, including duplicate headings, ambiguous headings, structured and unstructured therapy, structured and noisy laboratories, hepatic-pattern conflicts, family-history disease mentions, allergy-only drug mentions, negated diagnoses, combination therapy, brand/generic variants, and missing LiverTox matches.
