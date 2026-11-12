# Sessions, Timeline, And Data
Last updated: 2026-11-12

## Inspect Saved Clinical Sessions
Open **Clinical Sessions** from the sidebar.

The selected session's tabs have contextual help that changes with the selected page:
**Preview** is read-only, **Text Editor** saves manual edits, **Metadata** stores
attached evidence and JSON, **Revision** creates a new draft, and **Timeline**
manages generated chronologies.

Clinical Sessions lists every session in one place: drafts, queued and
running analyses, completed reports, and cancelled or failed analyses. Each
row shows the patient or session number, its status, when it was created and
last updated, a thin progress bar while it is active, and whether a report or
timeline exists. Sessions created in the last 90 days are loaded once and
filtered instantly in the browser: tick any combination of statuses, search by
name or `#id`, pick a date preset or custom range on the created, updated or
visit date, and sort by last update or creation date. Filters combine (any
selected status, and the text, and the date). Older sessions are not preloaded;
choose **Search older** (or pick a created-date range before the 90-day window)
to query the full history, and the line under the filters always says which
range is shown. The list refreshes itself while any analysis is active; the selected session and tab stay in the address, so refresh and Back
keep your place.

- **Open report** shows a running analysis with live progress, or a finished
  report.
- **Edit draft** reopens a draft in the DILI Agent; launch it from there.
- Cancelled and failed sessions keep their input and any partial report.
- Queued or running sessions cannot be deleted; cancel them first. A session
  with revised child sessions cannot be deleted until its children are.
- Revisions and manual report edits are available once a report exists.

Recommended workflow:
1. Open **Clinical Sessions**.
2. Locate the target session by identifier, date, or metadata.
3. Select the session.
4. Use **Text Editor** for direct in-place manual report edits.
   The Source view preserves Markdown, whitespace, blank lines, and unsaved drafts while Rendered shows a read-only preview of that same draft. Save persists the source text directly.
5. Use **LLM Revision** to create a new draft revision version when a model-assisted rewrite is needed.
6. Use **Official Version History** to inspect version lineage separately from **Manual Edit History**.
7. Use **Version Comparison** to compare the selected official version against its source or another persisted official version using backend-computed entity and report diffs.
8. Use **Human Clinical Review** to mark a revision version `under_review`, `approved_by_human`, or `rejected_by_human`.
9. Review **Revision QA And Artifacts**, **Revision Consultation Provenance**, **Revision Finalization Provenance**, **Persisted Revision Entities**, and persisted pipeline steps before approving an LLM-assisted revision.

If a session is missing, confirm that the assessment completed successfully and that local persistence is initialized.

Important distinctions:
- Manual report edits do not create a new official version.
- If the report changed after you opened the editor, Save is refused; use **Reload latest** to load the current report while keeping your draft text.
- Metadata attachments are downloaded through the app. HTML, SVG, script, and executable files are not accepted.
- LLM-assisted revision creates a new versioned draft and keeps the previous version unchanged.
- Human clinical review status is separate from LLM QA status.
- Backend-provided matched drugs, structured case fields, revision entities, RxNav identifiers, LiverTox matches, and linked FDA DILIrank identifiers are authoritative persisted evidence only within their documented source semantics.
- A DILIrank category is a drug-level hepatotoxicity prior, not a patient-level causality score.
- **Detected drugs** shows each entered name with its identity status: **Verified**, **Probable — review** (matched by spelling correction), **Ambiguous** (several candidate drugs, no evidence attached), or **Unresolved**. An arrow shows the resolved ingredient or combination. For an ambiguous name you can click the correct candidate to confirm it; the confirmation applies to future assessments and does not change the stored session.
- Frontend-derived display fallbacks are labeled as **Display fallback** or **Not backend-confirmed**. These values are navigation aids only and must not be interpreted as RxNav, LiverTox, DILIrank, RUCAM, or backend-confirmed clinical evidence.

## Use Patient Timeline from a session
Select a session in **Clinical Sessions**, then open its **Timeline** tab. Saved timeline previews open at `/sessions/:sessionId/timetable/:timelineId`; the route without `:timelineId` starts from the session workspace. There is no separate Patient Timeline item in the primary sidebar.

Use it to review event order, clinical sequence, and patient chronology where data is available.

Recommended workflow:
1. Open **Clinical Sessions**.
2. Locate and select the relevant patient or case.
3. Open the session's **Timeline** tab.
4. Generate a new timeline when needed from the session timeline workspace.
5. Reopen any previously generated timeline from the saved timeline preview list instead of regenerating it.
6. Review timeline entries in chronological order.

Timelines belong to exactly one session. A draft can generate a timeline from
its input without running the DILI analysis, and its status stays *Draft*. If
the input changes after a timeline was generated, the timeline is kept and
marked as out of date with a **Regenerate** action.
7. Compare exposure dates against lab abnormalities and symptoms.
8. Use the timeline to refine DILI Agent input if needed.

In the **Timeline** tab, the generation action uses the Timeline configuration of the selected agent (the default agent unless another active agent with Timeline enabled is chosen). Use **Manage agents** when that
assignment needs to change. Saved timelines appear as compact rows that record
the run's provider, model, date range, event count, and evidence-quality
warnings. Use **Open** to reopen a specific saved timeline or **Delete** to
remove only that saved timeline after confirmation.

The timetable page has three panels: the generation summary (status, event count,
events needing timing review, date coverage, model), and the **Clinical chronology**
panel whose toolbar stays pinned while you scroll. The toolbar switches between two views:

- **List** groups events by date (day, month-only, or year-only). Every event is a
  one-line row with its category, title, and review flags; select a row to expand its
  description, timing, precision and certainty, source, model confidence, verbatim source
  evidence, review note, and linked events. Date groups collapse too, and the
  **Date not reported** group starts collapsed. **Expand all** / **Collapse all** apply to
  the visible events. Chips such as `+13 d` between groups show the gap between day-level dates.
- **Timeline** shows the same dated groups as evenly spaced columns on a horizontally
  scrollable track, with one lane per category (Clinical, Medications, Laboratory, Other)
  and gap chips between columns. A bar under a lane marks a date range. Hollow markers are
  month- or year-only dates; dashed borders mark uncertain or inferred timing. Select an
  event to show its details below the track. Undated events are listed under the track.

Category chips hide or show categories, **Show uncertain timing** hides approximate and
undated events, and **Previous** / **Next** step through the visible events (Escape clears
the selection). **Approximate**, **Inferred date**, a **Fallback chronology**, and
**Missing evidence** are visible warnings, not clinical confirmation. These controls do
not alter the saved timeline.

An **Inferred date** is a date the model supplied that the quoted evidence does not state
at that precision (for example a day derived from relative timing); its review note
explains why. When the model's quoted evidence cannot be found in the session's clinical
text, that event is discarded and the summary states how many events were discarded.

Timeline generation may show a fallback notice when model extraction does not complete. For an explicitly selected OpenCode Go model, a temporary model-catalog outage does not prevent the known routed request from being attempted. The notice identifies the failure class, such as provider network unavailable, provider timeout, authentication rejected, rate limited, upstream error, invalid structured response, or incomplete configuration. Transient network, timeout, rate-limit, and upstream failures are retried with bounded backoff before fallback.

Fallback events are built deterministically from persisted session fields. A source field contributes a date only when it states exactly one distinct, valid calendar date (ISO, numeric such as `17.02.2025` in the record's day/month order, month-year, a written month name, or a year introduced by words such as "in" or "since"); doses and laboratory values are never read as dates, and the fallback preserves that token's original precision and marks it explicit. Missing, invalid, relative, or ambiguous dates remain undated and uncertain. Fallback never infers an event date from the visit timestamp. Treat fallback events as navigation aids rather than model-extracted chronology, then retry after correcting the reported condition.

For LLM-generated timelines, events without preserved source evidence are not part of the persisted clinical chronology contract. In the UI, missing source evidence should be treated as a warning rather than as clinically grounded support.

## Inspect Local Data
Open **Knowledge** from the sidebar.

Expected capabilities:
- data-source or table selection for the Drug Catalog (RxNav), LiverTox, Terminology, FDA DILIrank 2.0, and RAG
- refresh controls
- record counts or metadata
- table-style inspection
- search, filter, or pagination where supported
- embedding or data-update status where supported

The **Terminology** view shows source coverage (concepts, resolvable aliases, verified products, linked LiverTox monographs), each source's last import, version, license and attribution, open cross-source conflicts, and recent import reports. Use **Resolve** to test how a drug name, brand, or abbreviation is identified; the result lists the status, method, source, ingredients, and candidates. **Confirm** and **Reject mapping** record a reviewer decision that applies to all future assessments. **Update Terminology** refreshes DrugCentral, ChEMBL, and Swissmedic; a source can be skipped for one run, and unchanged releases are skipped unless a full re-import is chosen.

The **DILIrank** view is read-only. It shows FDA compound name, linked canonical drug when a safe deterministic link exists, LTKB ID, DILI concern class, severity class, labeling section, and source comment. `Unlinked` is an intentional state: it means the FDA row is preserved but is not eligible to contribute to clinical consultation. Do not infer a missing mapping manually from display similarity.

Recommended workflow:
1. Open **Knowledge**.
2. Select the data source or dataset.
3. Refresh the view.
4. Confirm expected records are present.
5. Use filters or pagination to inspect specific records.
6. For DILIrank, distinguish linked records from preserved unlinked source rows before interpreting the clinical knowledge available for a drug.

Do not edit local database files manually while the application is running.

## Update Local Data Sources
Some data sources or embeddings may require initialization or refresh through the Knowledge update controls or through:

```text
setup_and_maintenance.ps1
```

The Knowledge sidebar groups Drug Catalog, LiverTox and DILIrank under *Knowledge sources*, Terminology under *Mapping*, and RAG under *Retrieval*; the **?** button beside *Sources* opens a Knowledge help dialog covering sources, updates, terminology mapping, dependencies and update status. In Clinical Sessions the refresh icon beside Search reloads the persisted sessions while keeping the query, filters and selection.

**Update all** refreshes RxNav (RxNorm Prescribable Content and curated aliases), LiverTox, Terminology, and DILIrank in that order. DrugCentral and ChEMBL data are licensed CC BY-SA and are downloaded to your local database by this update; they are never bundled with the application. The Update all dialog has an **Include terminology mapping** checkbox (on by default); clearing it skips terminology for that run and shows it as excluded. The Terminology update dialog uses a segmented refresh mode (skip unchanged or full re-import) and a checkbox per source (DrugCentral, ChEMBL, Swissmedic) with purpose and licence notes.

DILIrank updates use the official FDA workbook. The default update may reuse a validated local workbook while checking HTTP freshness metadata; **Download fresh source** forces a new download. A downloaded candidate is validated before it replaces the last-known-good cache.

Use the launcher menu options for database initialization, dependency maintenance, test execution, log cleanup, or cache cleanup when those broader maintenance operations are needed.

Expected result:
- progress is reported
- long-running jobs show status
- refreshed data sources become available after completion
- restarting the application after maintenance is recommended
