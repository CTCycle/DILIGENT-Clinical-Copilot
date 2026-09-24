# Local timeline structured response diagnostics — 2026-09-24

## Scope and isolation

This validation revisited `sessions.timeline` and `model.provider.local-ollama`,
the next coherent slice in the canonical ledger. It started on `develop` at
`a0566a6a77a5744c90e534c520ff94cd7d444184`, equal to `origin/develop`.

The official Windows launcher ran the app against the task-owned
`assets/QA/timeline-local-model-diagnostics-20260924/runtime-data` SQLite root.
The database contained only two seeded synthetic sessions: acetaminophen use in
January 2025, symptoms on 2025-01-17, and ALT 75 U/L on that date. Both exact
routes were selected in the isolated Settings UI: `ollama / qwen3.5:2b` and
`ollama / qwen3.5:9b`. No shared application database or settings were used.

## Outcome and failure attribution

| Attempt | Observed result | Classification |
|---|---|---|
| `ollama / qwen3.5:2b`, synthetic session 1 | The saved timeline is a deterministic fallback with `generation_error_code=invalid_response`. The request log records a `RuntimeError` wrapping eight Pydantic validation errors for `PatientTimelineExtraction`; the returned object was the extraction JSON Schema shape rather than a timeline instance. | Model-response structured-output/conformance failure for this run. It was not a timeout, polling, connectivity, or rendering failure. One sample does not establish that model size or performance is the root cause. |
| `ollama / qwen3.5:9b`, synthetic session 2 | The saved timeline has `generation_status=llm_generated`, three events, and no generation error. | Successful model-generated timeline for this exact lane and synthetic case. |
| Initial PowerShell timing wrapper | The wrapper exited with `0xC0000005` before emitting request results. On the subsequent check, the Ollama endpoint was unavailable. | Harness/runner failure with the failing stage unknown. This attempt provides no evidence about either model and is excluded from model-quality findings. |

The runner was replaced by a small Python standard-library HTTP probe after
restarting the task-owned Ollama service. All four requests completed and
returned `done=true`:

| Exact Ollama model | Request state | Wall time | Server load time |
|---|---|---:|---:|
| `qwen3.5:2b` | cold | 14.98 s | 14.80 s |
| `qwen3.5:2b` | warm | 0.079 s | 0.003 s |
| `qwen3.5:9b` | cold | 39.22 s | 38.14 s |
| `qwen3.5:9b` | warm | 0.605 s | 0.005 s |

Each probe used `/api/generate`, the non-clinical prompt `Reply exactly OK.`,
`stream=false`, and `num_predict=4`. The timing probes characterize the local
service only; they do not exercise the application schema, evidence guard, or
timeline quality. Successful direct requests show the service responded during
the retest. They do not retroactively identify the cause of the earlier
2026-09-24 pair's `unknown` fallbacks.

## Persistence, evidence, and rendered behavior

The official launcher served the saved timelines at
`/sessions/1/timetable/1` and `/sessions/2/timetable/2`. Both pages were
reloaded and still showed the same generation status and exact model.

- The 2B page visibly reports **Fallback chronology** and **Failure class:
  Invalid structured provider response**. Its three `fallback_parser` events
  each retain one source-evidence item. Medication remains month precision
  (`2025-01`); symptoms and ALT remain day precision (`2025-01-17`).
- The 9B page visibly reports **LLM generated**, model `qwen3.5:9b`, and three
  events. The event inspector confirmed source evidence for all three: the
  acetaminophen month statement (`drugs`), ALT value/date statement
  (`laboratory_analysis`), and symptom date statement (`anamnesis`). Month/day
  precision and normalized dates matched the synthetic sources.
- A read-only SQLite check returned `integrity_check=ok` and zero foreign-key
  violations. The two persisted rows retain exact `ollama` provider and local
  source provenance; the 2B row is `fallback / invalid_response` and the 9B row
  is `llm_generated / no error`. Every event in both records has evidence.

## Focused regression and lint

From working directory `app/server`:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -m pytest ..\tests\unit\test_timeline_error_diagnostics.py -q -o 'cache_dir=..\..\runtimes\cache\pytest\timeline-error-diagnostics-20260924' --basetemp=..\..\runtimes\cache\pytest\timeline-error-diagnostics-20260924-temp
.\.venv\Scripts\ruff.exe check services\inspection\timeline.py ..\tests\unit\test_timeline_error_diagnostics.py --no-cache
```

The focused test passed **9 tests** with one existing Google GenAI
`_UnionGenericAlias` deprecation warning; Ruff reported **all checks passed**.
The test used task-specific cache and basetemp paths, which were removed after
the run.

The regression confirms type-based classification for a bare `TimeoutError`;
it does not prove that the prior `unknown` failures were timeouts. The captured
2B `invalid_response` and the PowerShell `0xC0000005` belong to different
failure classes and must remain separate in future summaries.

## Gate status

- `sessions.timeline` — **PARTIAL**. The current 9B lane produced a grounded,
  evidence-backed LLM timeline and survived reload. The current 2B response
  violated the structured-output contract and correctly fell back. The two
  earlier same-day `unknown` records remain unattributed; the broader timeline
  matrix is not covered by this pair.
- `model.provider.local-ollama` — **PARTIAL**. Both exact model identities and
  provenance were verified, with one current 9B success and one current 2B
  structured-response failure. More than one run and the remaining local
  provider/recovery cases are needed before promoting the gate. The observed
  2B output defect is not evidence that the PowerShell runner crash was a model
  failure.
- `test.automated-regression` — **PARTIAL**. The focused local regression and
  Ruff passed. Hosted live-provider execution still requires workflow dispatch
  and a repository secret; conditional E2E still requires its provider,
  embedding, Ollama-model, and persisted-session prerequisites.

## Other incomplete gates reviewed and left separate

- `model.provider.opencode-go` and `revision.agentic-lifecycle` remain
  **PARTIAL** because their broader provider failure/retry/QA and revision
  lifecycle matrices are independent of this local timeline pair.
- `clinical.analysis.pipeline` remains **VALIDATED** for its named synthetic
  workflow, while the additional provider failure/retry variants and RAG-off
  lane remain separate validation debt.
- `data.inspection.catalogs` and `data.sources.refresh` remain **PARTIAL** while
  NCBI Bookshelf serves a human-verification CAPTCHA for the LiverTox list.
- `auth.access-key-management` remains **BLOCKED** pending an approved
  disposable credential and explicit credential-cleanup authorization.
- `release.desktop.v3-4-0` remains **BLOCKED** pending release authorization,
  signing material, synchronized/tagged release state, hosted Windows CI, and
  clean-machine install evidence.
- `api.local-boundaries` remains **WORKING** pending the complete route catalog;
  `runtime.containerized` remains **NOT_IMPLEMENTED**.
- RAG duplicate-file behavior still needs a product policy decision. The
  existing RAG gate remains validated only for its documented fixture scope.
  The broader keyboard, screen-reader, and responsive accessibility audit also
  remains open beyond the exercised desktop viewport.

## Cleanup and remaining limitations

The synthetic SQLite runtime, startup artifacts, and task-specific pytest
cache were removed after verifying executable paths and command lines for the
task-started services and stopping them. Ports 7690, 9847, and 11434 were clear
after cleanup. Thirty-one untracked `__pycache__` directories produced by the
focused run were removed after confirming they contained no tracked files.
Shared data and settings were not changed. The unretained historical
`unknown` exceptions, broader provider and clinical cases, hosted conditional
E2E, and independent gates listed above remain open. No code defect was found
in the inspected error-classification path; the existing focused regression
passed.
