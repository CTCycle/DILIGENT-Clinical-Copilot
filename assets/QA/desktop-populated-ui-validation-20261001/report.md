# Packaged populated-workflow validation — 2026-10-01

Date: 2026-10-01 (Europe/Rome)

## Scope and baseline

This slice covers the previously UNTESTED/BLOCKED packaged-app validation gates
on the portable v3.4.0 lane by driving the packaged Tauri/WebView2 window with
native UI automation against a **populated** isolated data root. It is the
direct continuation of the
[packaged interactive-UI slice](../desktop-interactive-ui-validation-20261001/report.md),
which covered startup, Settings persistence, the preflight boundary, and clean
close/restart on a fresh root.

Candidate source SHA: `2f804d251769f4e01375d02459e41dd9970ec8eb` (develop;
origin/develop equal).

Artifacts verified against the release manifest:

- `DILIGENT-v3.4.0-windows-x64-portable.exe` SHA-256
  `0209cf25a8acf3a39762ffeec48488c28103e1b96c7c5a57ed8beb813574e167`
- `DILIGENT-v3.4.0-windows-x64.msi` SHA-256
  `1df49329361b600edb2e0c07949fe00bc113b25122d55af9ba93cb52f67fd508`

Environment: portable EXE launched with an isolated `LOCALAPPDATA` data root
under the short temp path `C:\Users\THOMAS~1\AppData\Local\Temp\opencode\dcpop-…`
(matching `smoke_release.ps1` semantics). The user-started Ollama service on
`11434` was used read-only as the model lane (`qwen3.5:9b` / `qwen3.5:2b`
installed). Only synthetic data was used; the shared database, settings,
credentials, and provider caches were never touched. No patient information,
provider credentials, tokens, or secrets were entered.

## Launch and API boundary

| Check | Result |
|---|---|
| Ready payload | `{"pid": …, "port": 56970, "release_version": "3.4.0"}` |
| `/api/health` | HTTP 200 |
| `/` (SPA root) | HTTP 200 |
| `/api/settings` (no cookie) | HTTP 401 (desktop boundary) |
| Window title | `DILIGENT Clinical Copilot` |

## Populating structured sources (D09/D07 enabler)

The packaged **Data Inspection → Update all** flow was started and ran the real
ordered pipeline (RxNav → LiverTox → DILIrank). RxNav's full refresh iterates
every RxNorm drug term with multiple HTTP requests each and had only reached the
letter "c" after ~12 minutes. The run was then cancelled **cooperatively
through the packaged UI**: the modal reported "0 of 3 source updates completed.
The operation was cancelled." with all three source rows `Cancelled`. This is
itself D11 cooperative-cancellation evidence for the packaged update job.

Because the full RxNav refresh is N-request-slow (hours), the isolated packaged
database was then seeded deterministically using the repository's own
update-persistence APIs (`DrugCatalogRepository.upsert_drugs_catalog_records`,
`KnowledgeRepository.save_livertox_records`,
`DiliRankRepository.replace_records` — the same persistence path the release-gate
live-provider seeding uses). Four synthetic drugs were seeded
(amoxicillin, acetaminophen, clarithromycin, ibuprofen) with RxCUI codes,
LiverTox monographs, and DILIrank records. All 4 DILIrank records linked.

The packaged Data Inspection then rendered the populated catalogs:

- **Drug Catalog (RxNav)** — acetaminophen, amoxicillin, clarithromycin,
  ibuprofen with alias/delete actions.
- **LiverTox** — the four monographs with excerpt/delete actions.
- **DILIrank** — four records with LTK ids and comments.

The DILI Agent preflight subsequently no longer reported empty-catalog
blockers.

## D07 — Full populated multi-drug analysis

Settings → Models in the packaged window assigned `qwen3.5:9b` to all four
roles (Clinical, Text extraction, Revision, Timeline) and **Save Configuration**
persisted them (DB: `application_configuration.payload` shows all four roles =
`qwen3.5:9b`).

A full synthetic multi-drug analysis was submitted from the packaged DILI Agent
(Amoxicillin + Clarithromycin + Ibuprofen, visit date 2026-09-09):

- Job `1b0e1e57` ran through all 15 pipeline steps and completed at 100%
  (`Progress: 100.0% | Clinical analysis completed.`; `Job 1b0e1e57 completed
  successfully`; consultation required ~379 s).
- The packaged window rendered the clinical report with the R-score (4.85),
  mixed hepatocellular-cholestatic pattern, per-drug clinical commentary,
  LiverTox scores (Amoxicillin D), and Hy's Law / competing-cause discussion.
- Persistence (read-only SQLite): session 1 (`standard_assessment`),
  `clinical_session_versions` 1/2/3, 3 matched `clinical_drug_mentions`
  (Amoxicillin, Clarithromycin, Ibuprofen → `matched`, unresolved 0), and the
  full result payload with report, evidence bundle, RUCAM assessments.
- Session 1 is stored as `failed`, which here is the **hard safety-gate
  fail-closed** semantic: the faithfulness audit flagged
  `clinical_narrative_recommends_rechallenge` and
  `clinical_narrative_overstates_causality`, so `clinical_validity =
  requires_human_review` and `manual_review_required = True`. This matches the
  documented safety-gate design (the generated qwen3.5:9b narrative was
  blocked, not silently accepted).

## D08 — Sessions, timeline, edits, revisions, lineage, reload

- **Clinical Sessions** rendered the populated session list and the session
  detail (Patient, anamnesis, classification, R-score, RUCAM evidence, safety
  issues, report).
- **Timeline**: the packaged Timeline workspace generated timeline #1 with the
  configured model. The `qwen3.5:9b` extraction call did not respond within the
  configured timeout, so the app **failed closed** and saved a deterministic
  **Fallback chronology** with 3 events, all with evidence
  (provenance note rendered: "Local timeline extraction did not complete…").
  The timeline view rendered the events, evidence, date coverage, and the
  fail-closed provenance. `clinical_session_timelines = 1`.
- **Metadata**: a JSON metadata payload was entered and saved through the
  packaged editor and persisted
  (`{"manual_metadata": {"reviewer": "packaged-validation", …}}`).
- **Text Editor / manual edit**: a manual report edit was saved. Version 1 was
  superseded and version 2 became `current` with `revision_kind = manual_edit`
  and a full `manual_edit_audit` (previous/new text SHA-256 hashes,
  `edited_fields: ["report_text"]`, actor unverified).
- **Revision**: the agentic revision workspace launched ("Planning
  revision…") and **failed closed** at Step 1/11 with an actionable diagnostic
  in the agent trace: "LLM call failed: Timed out waiting for Ollama chat
  response". No corrupted state; the workspace returned to a retryable
  `Start revision` state. A `draft_revision` version row was recorded.
- **Reload**: after a real process close/relaunch, both sessions, all versions,
  the timeline, and the model configuration remained (see Phase 8).

## D09 — RAG folder, ingestion, retrieval, citations

- The packaged RAG surface rendered with the default RAG source folder
  (`…\DILIGENT\data\resources\sources\documents`) and a native
  "Select RAG folder" picker button. The native `IFileDialog` was invoked but
  returned the host's last-used folder and could not be reliably driven in this
  automated session (no dialog window materialized for interaction), so the
  packaged **default RAG source folder** was used and this boundary is recorded
  honestly rather than claimed as a pass.
- Three synthetic documents (one a byte-identical copy of another) were staged
  in the default RAG source folder.
- **Update Embeddings** (job `4a8b122c`) downloaded the pinned multilingual
  Granite ONNX embedding model into the data-root cache and rebuilt the vector
  store: "Serialized 2 documents into 2 vector chunks", "RAG embeddings
  refreshed using the multilingual Granite ONNX runtime (2 documents,
  2 chunks)", "Job 4a8b122c completed successfully".
- The packaged RAG documents list rendered the ingested files with the vector
  model (`onnxruntime:ibm-granite/granite-embedding-97m-multilingual-r2`) and
  flagged the duplicate: "Duplicate of amoxicillin-copy.txt".
- A **RAG-on analysis** (job `d9970bdc`) completed successfully as session 2
  (`session_status = successful`, `metadata = {"use_rag": true}`). The rendered
  report included per-drug "Bibliography source: LiverTox; RxNav (RxCUI …);
  FDA DILIrank 2.0" lines and a `## Bibliography` section citing the retrieved
  source document (`acetaminophen.txt, p. 1`).

## D11 — Cancellation, provider failure, retry, recovery

- **Update All** cooperative cancellation (see above).
- **Clinical job cancellation**: a fresh analysis was started and cancelled via
  the packaged "Stop analysis" button; the UI showed
  "[INFO] Clinical analysis cancelled." and **no partial session was persisted**
  (only sessions 1 and 2 exist).
- **Timeline LLM timeout** → fail-closed fallback chronology with the timeout
  provenance rendered; the "Generate Timeline" control returned to retryable.
- **Revision planner timeout** → fail-closed diagnostic in the agent trace;
  retryable state preserved.

## D06 leftover — Access-key lifecycle in the packaged surface

Settings → Models → OpenAI "Manage keys" was exercised in the packaged window:

1. Created key `sk-…-0001` → stored encrypted (Fernet `encrypted_value`,
   fingerprint only; **plaintext never displayed**, masked as `********`).
2. Activated it → `is_active=1`, "ACTIVE" badge, button "Key is active".
3. Created key `…-0002` (inactive by design).
4. Activated key 2 → key 1 deactivated (`is_active=0`): **one-active-key
   rotation** confirmed in UI and DB.
5. Deleted both keys → "No keys stored for this provider."; `access_keys` empty.

## Restart persistence and cleanup

- Clean close (`CloseMainWindow`/Alt+F4) terminated the packaged backend with no
  leftover DILIGENT process or listener.
- Relaunch on the same isolated root: health 200, and the packaged UI retained
  both sessions (Session 2 Successful, Session 1 Failed), the version lineage,
  the model configuration, the timeline, the RAG vector store, and the deleted
  key state.
- Final clean close left no processes or listeners. The isolated data root and
  all temporary automation artifacts were removed; the shared database,
  settings, `.env`, and protected caches were untouched.

## Deliverables in this QA folder

- `s01-data-inspection-dilirank.png` — populated DILIrank catalog.
- `s02-dili-analysis-running.png` — packaged analysis running (job progress).
- `s03-dili-report-rendered.png` — rendered multi-drug report.
- `s04-patient-timeline.png` — generated timeline view (fallback chronology).
- `s05-rag-on-report.png` — RAG-on report (Bibliography).
- `s06-rag-ingested-documents.png` — RAG documents list with dedup.
- `backend-log-excerpt.txt` — sanitized backend log evidence (progress,
  completions, cancellation, timeouts).
- This report.

## Gate disposition

| Gate | Status after this run | Remaining boundary |
|---|---|---|
| `D07` full populated multi-drug analysis | `PASS` for the packaged lane | Job ran to 100%, report/evidence/R-score rendered, session + version + drug mentions persisted. Session 1 stored `failed` = safety-gate requires_human_review (rechallenge/causality flags). |
| `D08` sessions/timeline/edits/revisions/lineage | `PASS` for the exercised packaged scope | Sessions + detail, generated timeline (fail-closed fallback), metadata save, manual edit → version lineage + audit, revision workspace fail-closed with diagnostic, restart reload. |
| `D09` RAG folder/ingestion/retrieval/citations | `PARTIAL` | Ingestion (2 unique docs / 2 chunks), dedup flag, retrieval + bibliography citations all confirmed. Native `IFileDialog` could not be reliably driven in this automated session; the packaged default RAG source folder was used instead. |
| `D11` cancellation / failure / retry | `PASS` for the exercised cases | Update-All cancel, clinical-job cancel (no partial session), timeline + revision timeouts fail-closed with retryable state. Broader injected fault matrix remains outside scope. |
| `D06` access-key lifecycle (leftover) | `PASS` for the packaged surface | Encrypted storage, redaction, one-active-key rotation, deletion all DB-verified. Provider-side acceptance remains a provider gate. |
| `D03` startup/restart/cleanup | `PASS` (reconfirmed on populated root) | Clean close/relaunch persistence; no leftover processes/listeners. |
| `D12` API boundary | `PASS` (rechecked) | health 200, root 200, unauth settings 401. |
| `D13`/`D14` MSI lifecycle + upgrade | `BLOCKED` | Non-administrator token; no elevation attempted. |
| Signing / clean-machine distribution | `PENDING` | Separate distribution procedures. |

## Remaining boundaries (not claimed)

- MSI install/upgrade/uninstall/reinstall and genuine 3.3.0→3.4.0 upgrade
  (D13/D14) remain blocked on an administrator-capable host.
- Signing, Authenticode verification, offline WebView2 packaging, and
  clean-machine smoke remain pending distribution procedures.
- The native RAG folder dialog was not reliably drivable in this automated
  session (returns the host's last-used folder; no interactive dialog window
  materialized); the packaged default RAG source folder was used and the
  boundary is recorded rather than inferred as a pass.
- The local `qwen3.5:9b` lane times out for the timeline and revision LLM calls;
  the app fails closed with rendered diagnostics and retryable state. This is
  consistent with the documented model/task compatibility limitation and is not
  a claimed pass for those LLM outputs.
- Broader D10 file-dialog / WebView2 edge cases remain outside this slice.