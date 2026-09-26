# Clinical resilience validation — 2026-09-26

## Scope and decision

This slice covered the next actionable local boundary after the API/preflight
revalidation: a real rendered clinical run with RAG disabled, local Ollama
model provenance, terminal report generation, and persisted-session recovery.
It also revisited the related preflight, job, report-safety, and persistence
contracts. All patient content was synthetic.

The supported local RAG-off boundary is **PASS / VALIDATED** for the exercised
fixture. The run completed with a safe deterministic fallback for model
extraction gaps, produced a rendered report, and persisted session 25/version
1. This does not promote the separate cloud-provider, credential, source
refresh, release, revision-matrix, or container gates.

## Environment and startup boundary

- Branch: `develop`; the working tree was clean before the slice.
- The official launcher was attempted first against a fresh task-owned SQLite
  database on source ports `7690`/`9847`. Its frontend production build exited
  with Windows status `-1073741819` (`0xC0000005`) before application startup.
- The source client then built successfully with the Angular cache redirected to
  the task QA directory and the repository configuration was restored exactly.
- A manual source backend/frontend fallback used a disposable clone of
  `app/resources/database.db` on `127.0.0.1:7691` and `127.0.0.1:9849`.
  The clone contained the local LiverTox, RxNav, and DILIrank snapshots needed
  for the clinical preflight; the shared database and settings were not
  modified.
- Ollama was live at `127.0.0.1:11434`. A catalog refresh discovered the
  installed `qwen3.5:2b` and `qwen3.5:9b` models. The run used exact local
  `qwen3.5:9b` for parser and clinical roles with cloud services disabled.

## Rendered workflow evidence

The in-app Browser was used for the actual UI interaction and rendered
observation.

1. A fresh empty runtime correctly surfaced blocking structured-source and
   clinical-input issues before job submission.
2. On the populated clone, the same synthetic case passed
   `POST /api/clinical/validate-input` with `ready=true` and
   `rag_readiness.requested=false`.
3. The DILI Agent page submitted job `b7b404c4`. During execution the visible
   RAG checkbox was unchecked and disabled, the Run control changed to
   `Stop analysis`, and the live status advanced through the 15-step pipeline.
4. The job reached `completed` with progress `100.0` after approximately
   `921.3` seconds. The rendered report showed:
   - patient and visit metadata for `Synthetic Resilience Subject`;
   - a mixed hepatotoxicity pattern with R-score `4.85`;
   - one detected and locally resolved drug, Amoxicillin;
   - the four laboratory values and their ULN data;
   - structured RUCAM limitations, missing-data warnings, and clinical review
     language; and
   - local source bibliography text without a RAG bibliography section.
5. The Browser navigated to Clinical Sessions. The new row was visible as
   `Synthetic Resilience Subject — Session 25 · Version 1 — Successful`, and
   selecting it rendered the persisted report and evidence tables.

The completed API result independently recorded `session_id=25`,
`runtime_snapshot.use_rag=false`, `rag_retrieval_enabled=false`, zero
retrieved/canonical RAG references, no raw retrieved text, and
`rag_reference_audit.contract_valid=true`. The report's `faithfulness_audit`
was `faithful` with no blocking or non-blocking discrepancies.

The local 9B lane was deliberately observed as-is. Its therapy extraction
completed with one normalized entry. Anamnesis enrichment returned no
semantically valid entries and the backend retained deterministic extraction;
laboratory LLM extraction also failed closed and deterministic candidates were
merged. The pipeline continued to a successful report without silently
switching provider or model. This is safe degradation evidence, not a claim of
perfect local-model extraction quality.

## Automated evidence

Using the existing `app/server/.venv` and a task-owned pytest cache:

```text
42 passed, 1 skipped in 8.67s
```

Covered files were the assessment-preflight, clinical-validation,
session-workflow-report, runtime-job, and clinical-session-repository suites.
The host emitted two existing deprecation/cache warnings; the protected
default cache path was not modified.

## Gate disposition

| Gate | Final status | Evidence and remaining limitation |
|---|---|---|
| `clinical.input-preflight` | `PASS` for this scope | Populated-clone no-RAG validation returned ready; fresh empty runtime blocked safely before submission. |
| `clinical.analysis.pipeline` | `VALIDATED` for supported local RAG-off scope | Real rendered qwen3.5:9b job completed and persisted. Synthetic data, local provider, and one fixture only; cloud-provider and broader clinical variants remain open. |
| `model.provider.local-ollama` | `VALIDATED` for the exercised local boundary | Exact catalog discovery and qwen3.5:9b provenance were retained. Slow/degraded extraction behavior was observed and safely handled; broader model/provider matrices remain outside scope. |
| `sessions.crud-persistence` | `PASS` for creation/list/selection | Session 25/version 1 was listed and selected after completion. Deletion, binary image storage, and file-picker bridge remain separate. |
| `test.automated-regression` | `PARTIAL` | Focused local slice passed; provider-key-dependent and conditional lanes remain open. |
| `model.provider.opencode-go` | `PARTIAL` | `OPENCODE_GO_API_KEY` is still absent, so no new cloud request or output is claimed. |
| `auth.access-key-management` | `BLOCKED` | No approved disposable credential was available; only metadata-safe paths remain validated. |
| `data.inspection.catalogs` / `data.sources.refresh` | `PARTIAL` | NCBI LiverTox still requires human verification; no refresh was started or bypassed. |
| `release.desktop.v3-4-0` | `BLOCKED` | Signed/tagged/hosted/clean-machine/publication evidence remains independent. |
| `runtime.containerized` | `NOT_IMPLEMENTED` | No supported container runtime exists. |

No application defect requiring an in-scope source fix was found. The
temporary Angular-cache redirect and all task-owned runtime/database/cache
artifacts were removed after capture; only this report remains from the
slice. No Narrator or Speech Recap observation was performed.
