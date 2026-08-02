# DILI Assessment Pipeline
Last updated: 2026-08-02

## Section Extraction Contract
`POST /api/clinical/jobs` uses deterministic section extraction for structural input splitting. The extractor preserves source-verbatim section bodies after newline normalization and records canonical key, payload key, raw and normalized heading, match strategy, confidence score, heading line span, body line span, character span, verbatim coherence, review requirement, and source hash.

LLM fallback is not part of section extraction. Content inference, fallback assignment, low-confidence semantic matches, duplicate headings, or ambiguous headings are surfaced as diagnostics or review signals.

Aggregate section confidence is the minimum confidence across required extracted sections. Preflight blocks analysis when aggregate confidence is below `0.65`; confidence from `0.65` through below `0.85` is non-blocking but requires operator review acknowledgement in the DILI workspace before starting a job.

## Structured Extraction Contract
Drug, disease, and laboratory extraction use provider-agnostic structured LLM calls for both cloud and local providers. The active runtime provider and model are resolved from persisted model configuration.

Structured LLM output is validated against Pydantic schemas and semantic guardrails. Invalid or contaminated output is retried with the rejected output and validation feedback. If bounded LLM attempts fail, the pipeline falls back to direct rule-based parsers and records fallback warnings.

Pre-flight preserves deterministic extraction counts as diagnostics, but it never asserts that anamnesis is disease-free solely because deterministic matching found no entries. Disease absence is a clinical conclusion, not a parser-gateway condition.

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

The structured adjudication layer calculates R ratio at the first qualifying
paired ALT/ALP date and at the peak ALT date. Boundary values follow LiverTox:
`R >= 5` is hepatocellular, `R <= 2` is cholestatic, and values between 2 and
5 are mixed.

The paired sample at the highest ALT/ULN is the primary clinical injury anchor.
Earlier baseline and later recovery pairs remain available as longitudinal audit
points but cannot replace the peak injury classification merely because they
occur first. Candidate selection compares normalized multiples, not raw values,
so changing laboratory ULNs are handled consistently.

## Structured DILI Adjudication
The final report is generated from a persisted `DiliEvidenceBundle` before any
LLM summary. The bundle contains case completeness, longitudinal exposure and
laboratory events, first and peak R-ratio assessments, deterministic phenotype
candidates, a mandatory competing-cause checklist, Hy's Law status, explicit
severity grade, conservative drug identity resolution, per-drug componentized
RUCAM, and a separate DILIN-like causality category.

The clinical source hierarchy is AASLD, LiverTox, FDA DILI guidance, then
DILIN/RUCAM. RUCAM is supportive and never treated as dispositive. LiverTox
likelihood describes the drug's prior hepatotoxic potential and is kept
separate from patient-specific causality. Missing follow-up is represented as
missing, not as a negative dechallenge or rechallenge.

The deterministic dossier provides the FDA-style fourteen-section report
structure. The bundle also carries twelve acceptance-question answers with
supporting quotes and explicit missing-data statements so the final
adjudication can be audited directly.

The persisted `final_report` is the human-readable clinical DILI report. It
restores per-drug clinical narrative from the consultation synthesis and appends
a concise deterministic adjudication summary. The full rendered deterministic
dossier is persisted separately as `pipeline_artifacts.structured_dili_report`,
with the unchanged `dili_evidence_bundle` as the authoritative structured audit
contract. LLM clinical synthesis must remain evidence-bounded and is not allowed
to introduce unsupported clinical facts.

Blocking faithfulness issues are finalization blockers. The workflow may retain
generated artifacts for audit, but the persisted session status must not remain
`successful` when blocking faithfulness issues exist. Such cases require human
review before clinical reliance.

Generated clinical narrative is also checked against the authoritative evidence
bundle before finalization. If the narrative claims that unresolved competing
causes were excluded, asserts Hy's Law when the structured status is not
`meets_criteria`, conflates LiverTox likelihood with patient-level causality, or
uses definitive/lifelong causality language while structured causality is limited,
the workflow records a blocking faithfulness issue and persists the session as
requiring human review rather than `successful`.

Longitudinal adjudication is clinically conservative by default: dose changes,
restart or rechallenge mentions, first symptoms, bilirubin or jaundice timing,
marker-specific peaks, dechallenge direction, recovery versus persistence, and
worsening after discontinuation remain explicit timeline events or missing-data
statements instead of being inferred as negatives.

Mandatory competing causes use explicit four-state outputs:
- `excluded`
- `not_excluded`
- `unknown`
- `missing_data`

Hy's Law evaluation is same-episode aware and records baseline multiples,
initial cholestasis, alternative-cause exclusion, exposure compatibility, and
individual-patient versus trial-signal context. Rechallenge remains a safety
signal only; the workflow never recommends rechallenge.

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

When RAG is enabled, retrieved text is used only as hidden model context. The final report bibliography lists compact references to retrieved documents by filename and page number only.

LiverTox input preparation is bounded for cloud and local providers. A timeout
produces an explicit pipeline warning and safe evidence-free continuation instead
of leaving the job indefinitely in Step 12. Progress text distinguishes
LiverTox-only preparation from RAG/vector-enabled retrieval.

Default Ollama embedding configuration must use an immutable model tag rather
than `:latest`, and vector collection reset is opt-in rather than enabled by
default.

## Timeline Grounding
Patient timeline extraction keeps month-only tokens such as `YYYY-MM` as
month-precision values and does not promote them to exact calendar days.

LLM-generated timeline events require preserved source evidence to remain in the
normalized event set. Events without source evidence are dropped rather than
persisted as clinically grounded chronology.

Deterministic fallback timeline events keep `timing_type="uncertain"` and do
not reuse the patient visit date as an invented exact event date.

## Failure Modes
- Missing required sections block structural preprocessing.
- Duplicate required headings block deterministic section extraction.
- Ambiguous or low-confidence section assignments require review.
- Missing or ambiguous external drug matches do not force a match.
- Ambiguous drug matches are included for review but are not used as authoritative LiverTox evidence.
- Broad categories and rejected false-positive extracted text remain audit-only and do not become concrete drug matches.
- LLM structured output failures fall back to direct deterministic parsing after bounded retries.
- Session persistence is mandatory for successful clinical jobs. Persistence write failures raise a service dependency error instead of returning an apparently successful unpersisted report.
- Failed job payloads omit raw clinical input and patient image content; job-level errors expose generic failure text plus sanitized failure metadata.

## Testing Matrix
Regression fixtures under `app/tests/fixtures/dili_pipeline_audit` cover clean, noisy, incomplete, and adversarial clinical documents, including duplicate headings, ambiguous headings, structured and unstructured therapy, structured and noisy laboratories, hepatic-pattern conflicts, family-history disease mentions, allergy-only drug mentions, negated diagnoses, combination therapy, brand/generic variants, and missing LiverTox matches.

