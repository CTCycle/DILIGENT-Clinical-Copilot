# DILI Assessment Workflow
Last updated: 2026-10-30

## Open The DILI Agent
Open **DILI Agent** from the sidebar.

This is the main assessment page. It collects clinical context and submits a structured request to the backend analysis endpoint.

Typical input areas include:
- patient identifier or case identifier
- patient age or demographics
- suspected medication or exposure
- clinical history
- laboratory values
- symptoms
- timing information
- notes or free-text clinical context
- optional file upload where enabled
- run or submit action
- a live progress indicator and a **View report** link

On a first empty assessment, the page may show a small optional **Get started with DILI Agent** callout. Use **Show me** for the three-step tour of clinical input, patient name/date, and review/run, or use **Open Settings** to choose a runtime and the four role models. The tour can be closed with its X button and reopened from header **Help**.

## Enter Clinical Context
Use clear, specific, structured text. Prefer input like:

```text
Patient: 54-year-old adult
Suspected medication: ExampleDrug
Exposure timing: Started 21 days before liver enzyme rise
Labs: ALT 820, AST 610, ALP 160, total bilirubin 3.2
Symptoms: fatigue, jaundice
Relevant negatives: no known viral hepatitis in available records
Clinical question: assess whether this pattern is compatible with DILI
```

Avoid vague input like:

```text
Patient has liver issue. Check DILI.
```

When available, include the laboratory-specific upper limits of normal, pretreatment or baseline liver chemistry, exact drug start and stop dates, follow-up liver chemistry after discontinuation, and explicit alternative-cause workup. These details materially affect case qualification, R-ratio interpretation, dechallenge, and causality review.

## Run The Assessment
Select **Run DILI analysis**. A short setup dialog opens with four steps:
**Preflight → Evidence Sources → Models → Review**. Nothing is sent to a model
until you select **Start analysis**.

1. **Preflight** runs the existing safety checks for the exact session
   configuration. Blocking issues stop the setup; use **Fix in input** to jump
   to the field, correct it, and reopen the dialog (it restarts at Preflight
   and re-runs the checks). Warnings can be accepted. Below the issues,
   **What DILIGENT understood** shows the deterministic section split
   (Anamnesis, Current therapy, Laboratory analysis) with each section's
   heading basis, line range, and source excerpt; sections flagged for review
   are expanded. **Edit in clinical input** closes the dialog and selects that
   passage in the original text. This is a preliminary section assignment, not
   the final clinical extraction.
2. **Evidence Sources** chooses what the assessment may consult for this
   session only: **LiverTox** (recommended, selected by default), **FDA
   DILIrank 2.0**, and **Indexed documents** (your RAG library). Selecting none
   runs on LLM knowledge only. Unticking LiverTox shows an accuracy warning:
   its drug-level hepatotoxicity evidence is then removed from the causality
   assessment as well as from the narrative. The selection is enforced by the
   backend, and the report states which sources were used.
3. **Models** shows the provider and the text-extraction and clinical models.
   Overrides apply to this session only; **Restore defaults** returns to the
   saved configuration. Settings are never changed from here.
4. **Review** summarizes patient, visit date, input readiness, sources,
   effective models, and the preflight result. If you changed a source or
   model, the checks re-run before **Start analysis** is enabled. Start can
   only be triggered once per run.

**Cancel** closes the dialog without starting and keeps your input; your
choices are kept until the analysis starts or you select **Clear all**.

While the analysis runs, the progress panel in the right column shows the
current pipeline stage (for example *Normalizing drug names* or *Retrieving
evidence*), the step number, the backend message, and the elapsed time. It
stays visible when the clinical text is enlarged. DILI runs can take a long
time; the browser request timeout is one hour and does not cancel the
background job. **Stop analysis** requests cancellation.

Open **View report** at any time: while the job runs, the report page shows
the same progress and fills in when the analysis completes.

9. You may navigate to another page and return to the DILI Agent during the
   same browser session; the in-memory page state remains available. A full
   refresh resets input that was never saved as a draft. If an analysis is
   active, the existing job tracker may reconnect to it after refresh; draft
   recovery remains a separate action and does not start or reconnect a job.

## Save And Resume A Working Draft

Use **Save draft** when you want to keep an unfinished DILI Agent input. The
first save is explicit; after it succeeds, later edits autosave. On a later
visit, choose **Resume** to restore the saved patient name, visit date, and
exact clinical text. The draft is not an assessment, clinical
session, report, or approval, and Resume does not run preflight or start a job.

Drafts are stored in the currently configured SQLite or PostgreSQL database.
PostgreSQL may be on another machine. Draft content is not stored in browser
storage. Patient images are excluded, so select the optional image again after
resuming. **Discard** removes the saved draft; **Clear all** also clears the
current form. Both ask for confirmation. An unsaved initial form still resets
on refresh by design.

Expected result:
- the application submits structured clinical input and the session
  configuration to the backend
- the backend uses the configured provider with any session model overrides
- the report page shows the generated DILI assessment or a clear error message

**Settings** controls the retrieval and reranking setup of the document
library; the Evidence Sources step decides whether a given session uses it.
If the library is selected but unavailable, the setup blocks until you
deselect it or repair the cache. A selected source that is not imported
(for example DILIrank) is reported as unavailable in the report, which is not a
negative finding.

During evidence retrieval the progress message states whether selected
sources are being consulted or retrieval is skipped (LLM knowledge only). If
evidence preparation exceeds its bounded runtime, the assessment continues
without that prepared evidence and reports a warning for review.

If the run fails:
- confirm backend health
- confirm model provider configuration is saved
- confirm an active access key exists if required
- confirm Ollama is running for local Ollama models
- review backend console output for structured errors

## Review The Generated Report
Reports open on a dedicated page (**View report** in the workspace, or
**Open report** from a saved session in Clinical Sessions, `/report/:sessionId`).
The header shows the patient, visit date, evidence sources, and models used
(session overrides are tagged). **Back to workspace** returns to the
unchanged input.

Treat the report as a decision-support draft, not a final diagnosis.

Review for:
- consistency between exposure timeline and reported interpretation
- correct interpretation of the presentation liver chemistry pattern
- correct use of laboratory-specific ULNs in the R ratio
- explicit handling of pretreatment or baseline abnormalities
- mention of confounders and alternative causes
- consistency between conclusion and supplied evidence
- unsupported assumptions or invented facts
- drug-resolution review flags for ambiguous, missing, or unvalidated RxNav/LiverTox matches
- explicit missing-data statements rather than silent negatives
- the twelve acceptance-question answers and their supporting quotes
- drug-specific rather than global attribution of dechallenge evidence

The main report is a readable clinical DILI evaluation with per-drug narrative
commentary and a concise deterministic adjudication summary. The full structured
DILI dossier is retained as an audit artifact in the session result payload and
should be used to verify the clinical narrative before reuse.

The structured dossier explicitly reports:
- liver-injury case qualification separately from drug causality
- pretreatment baseline status when dated baseline chemistry is available
- presentation R-ratio phenotype and longitudinal laboratory context
- longitudinal exposure and lab timeline events
- dose changes, restart or rechallenge mentions, and drug-specific dechallenge evidence
- Hy's Law state and why it is or is not met
- competing-cause states as `excluded`, `not_excluded`, `unknown`, or `missing_data`
- a patient-record RUCAM score only when one is explicitly supplied for the current patient
- otherwise a non-scoring RUCAM criterion-evidence checklist with limitations
- an internal structured causality synthesis for review, not a validated DILIN score

LiverTox likelihood is drug-level evidence about known hepatotoxic potential. It is not the current patient's causality probability and must not be interpreted as a patient-specific diagnosis or RUCAM score.

Drug matching statuses can include:
- `accepted_exact_livertox`
- `accepted_rxnav_validated`
- `accepted_livertox_without_rxnav`
- `ambiguous_requires_review`
- `missing_rxnav`
- `missing_livertox`
- `rejected_false_positive`

Ambiguous matches require review and are not treated as authoritative LiverTox evidence. A LiverTox excerpt indicates available monograph text, not automatic proof that the drug identity is clinically correct.

If the report is incomplete or wrong:
1. Add missing clinical details.
2. Correct the input.
3. Re-run the assessment.
4. Compare the new output with the previous one.

Agentic revisions are subject to the same deterministic clinical safety checks as the primary report before a revised report is persisted as successful. A revision cannot rely on LLM QA alone to override unresolved competing causes, unsupported Hy's Law assertions, overstated causality, LiverTox/RUCAM provenance conflicts, or rechallenge safeguards.

## Copy Or Export Output
Use **Copy**, **Expand**, and **Download** on the report page to move the report into the local workflow.

Before reusing output:
- verify dates
- verify lab values, ULNs, and units
- verify drug names and dosing
- verify whether any RUCAM score was actually supplied for this patient
- remove placeholders and unsupported statements
- add human reviewer attribution according to local policy
