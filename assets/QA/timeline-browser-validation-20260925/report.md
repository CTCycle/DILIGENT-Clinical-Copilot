# Timeline browser validation — 2026-09-25

## Selected scope

This run covered the next coherent slice identified by the validation ledger:
`sessions.timeline`, `model.provider.local-ollama`, and the current local
regression/build boundary. It revisited the exact local `ollama` routes for
`qwen3.5:2b` and `qwen3.5:9b`, using a fresh SQLite database and three
synthetic sessions with identical source facts:

- acetaminophen taken in `2025-01`;
- symptoms starting on `2025-01-17`;
- ALT 75 U/L on `2025-01-17`.

The run started from clean `develop` HEAD `11dd344970371a75173ec961d3b99d50af7f3aee`,
equal to `origin/develop`. The official Windows launcher initialized and
started the source runtime on ports `7690` and `9847`. The isolated database
was task-owned; the shared database, settings, credentials, and source data
were not changed.

## Current implementation and regression checks

The existing empty-extraction guard remained active. A populated clinical
source cannot be persisted as a successful empty `llm_generated` timeline when
the model produces no evidence-backed events. The live 2B result exercised the
guard and was persisted as a classified fallback. No application code change
was needed during this validation slice.

| Check | Result |
|---|---|
| Focused backend timeline/extraction/repository suite | `67 passed`, 1 existing Google GenAI deprecation warning |
| Targeted Ruff check | Passed: `All checks passed!` |
| Angular/Vitest suite | `24 files, 105 tests passed` |
| Current production frontend build | Passed with Angular bundle generation complete |
| Exact installed Ollama models | `qwen3.5:2b` digest `324d162be6ca5629ae4517c8710434d0bd2d665bc94dbad46e9af8fbf8a2f0df`; `qwen3.5:9b` digest `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` |

No model was downloaded or substituted.

## Rendered browser evidence

The Codex in-app Browser rendered the current build at a `1280 × 720` capture;
the prior “Widen application window to continue” desktop gate was absent.
The browser showed the exact installed local models in Settings and persisted
the Timeline-role change from `qwen3.5:2b` to `qwen3.5:9b` in the isolated
database.

### Exact qwen3.5:2b case

Session 1 generated timeline 1. The Clinical Sessions page showed:

- `Fallback chronology`;
- `Failure class: Invalid structured provider response`;
- `qwen3.5:2b`, local/Ollama provenance;
- three events, all three with source evidence.

The Patient Timeline page rendered the therapy event at month precision
(`2025-01`) and the symptom and laboratory events at day precision
(`2025-01-17`). The event inspector showed the source text, normalized date,
date precision, `fallback_parser` source, and source evidence. A page reload
retained the fallback status, model, three events, source labels, evidence,
and date precision.

This is safe degradation evidence. It is not grounded model acceptance.

### Exact qwen3.5:9b control

After saving the isolated Timeline role as `qwen3.5:9b`, session 3 generated
timeline 2. The page showed `LLM generated`, exact local/Ollama provenance,
three events, and three events with source evidence. The rendered event
inspector confirmed:

- acetaminophen from `drugs`, month precision `2025-01`;
- ALT from `laboratory_analysis`, day precision `2025-01-17`;
- symptom onset from `anamnesis`, day precision `2025-01-17`.

The event evidence matched the persisted synthetic session fields. Reload
retained the generated status, model, events, provenance, source mappings, and
date precision. This is a grounded control for this fixture only; it does not
establish repeatable 2B behavior or universal 9B reliability.

The rendered captures were inspected directly in the Codex in-app Browser;
the sanitized visual record is in [`browser-capture-notes.md`](browser-capture-notes.md).

## API and persistence evidence

The isolated runtime returned HTTP 200 for `/api/health`, `/api/model-config`,
`/api/inspection/sessions/1/timelines`,
`/api/inspection/sessions/1/timelines/1`, and
`/api/inspection/sessions/3/timelines/2`. The model configuration response
reported `use_cloud_services=false`, both exact Ollama models available, and
the saved Timeline role as `qwen3.5:9b`.

The persisted rows were independently read from SQLite:

| Timeline | Session | Status | Error | Model/provider | Events/evidence |
|---|---:|---|---|---|---:|
| 1 | 1 | `fallback` | `invalid_response` | `qwen3.5:2b` / `ollama` | 3 / 3 |
| 2 | 3 | `llm_generated` | none | `qwen3.5:9b` / `ollama` | 3 / 3 |

`PRAGMA integrity_check` returned `ok`; `PRAGMA foreign_key_check` returned no
rows. The selected sanitized API and persistence fields are recorded in
[`api-persistence-evidence.md`](api-persistence-evidence.md).

## Gate disposition and scope decision

The exact `qwen3.5:2b` result is recorded as a model/task compatibility
limitation, not as an unfinished application gate. The application correctly
rejects the unsupported structured response and preserves a source-backed
fallback. The supported local acceptance boundary is therefore the validated
Ollama transport/provenance/fallback path plus grounded generation on the
`qwen3.5:9b` route exercised here. No further 2B retry is scheduled unless
the model, prompt contract, or timeline implementation changes.

| Gate | Final status | Evidence and remaining limitation |
|---|---|---|
| `sessions.timeline` | `VALIDATED` for supported scope | Source-backed fallback classification, rendered event/source/date evidence, reload persistence, and grounded generation on the exact `qwen3.5:9b` local route passed. `qwen3.5:2b` is recorded as a non-blocking compatibility limitation for this structured task, not as a pending application gate. |
| `model.provider.local-ollama` | `VALIDATED` for supported scope | Exact local Ollama model discovery, provenance, fail-closed fallback behavior, rendered persistence, and grounded `qwen3.5:9b` output passed. `qwen3.5:2b` cannot be promoted to grounded-model acceptance for this task under the tested conditions; no further retry loop is required. |
| `api.local-boundaries` | `WORKING` | Health, model configuration, session list, timeline generation/read APIs, and persisted records returned successfully. The complete route catalog and every response variant remain outside this slice. |
| `test.automated-regression` | `PARTIAL` | Current focused backend tests, Ruff, frontend tests, and production build passed. No hosted result exists for this final working tree; dispatch-only live-provider E2E and conditional provider/Ollama/embedding/persisted-session cases remain unrun. |
| `ui.application-shell` | `VALIDATED` | Current in-app Browser rendered at 1280px without the desktop-width gate; timeline shell, provenance, event cards, inspector, and reload state were visible. Full accessibility/screen-reader coverage remains separate. |

## Independent incomplete gates retained

`model.provider.opencode-go` and `revision.agentic-lifecycle` remain
`PARTIAL` because their broader provider, QA, and lifecycle matrices were not
part of this local Ollama slice. `data.inspection.catalogs` and
`data.sources.refresh` remain `PARTIAL` while NCBI Bookshelf presents the
LiverTox human-verification challenge. `auth.access-key-management` remains
`BLOCKED` pending an approved disposable credential and explicit cleanup
authorization. `release.desktop.v3-4-0` remains `BLOCKED` pending signed/tagged
release material, hosted packaging, clean-machine evidence, and publication
prerequisites. `runtime.containerized` remains `NOT_IMPLEMENTED`.

The RAG duplicate-file policy, full API catalog, clinical provider-failure and
RAG-off cases, and full accessibility audit remain separate validation debt.

## Cleanup

The task-owned DILIGENT listeners, Ollama service, isolated database, and
pytest workspace were removed after evidence collection. Ports `7690`,
`9847`, and `11434` were verified clear. No shared database, settings,
credential material, or unrelated process was changed.
