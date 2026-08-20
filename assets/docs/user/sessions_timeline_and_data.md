# Sessions, Timeline, And Data
Last updated: 2026-08-18

## Inspect Saved Clinical Sessions
Open **Clinical Sessions** from the sidebar.

The section tabs have contextual help that changes with the selected page:
**Preview** is read-only, **Text Editor** saves manual edits, **Metadata** stores
attached evidence and JSON, **Revision** creates a new draft, and **Timeline**
manages generated chronologies.

Expected capabilities:
- view a list of sessions
- select a session
- review session metadata
- review generated content
- filter or refresh records where supported

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
- LLM-assisted revision creates a new versioned draft and keeps the previous version unchanged.
- Human clinical review status is separate from LLM QA status.
- Backend-provided matched drugs, structured case fields, revision entities, RxNav identifiers, and LiverTox matches are treated as authoritative persisted evidence.
- Frontend-derived display fallbacks are labeled as **Display fallback** or **Not backend-confirmed**. These values are navigation aids only and must not be interpreted as RxNav, LiverTox, RUCAM, or backend-confirmed clinical evidence.

## Use Patient Timeline
Open **Patient Timeline** from the sidebar. Saved timeline previews open at `/sessions/:sessionId/timetable/:timelineId`; the route without `:timelineId` starts from the session workspace.

Use it to review event order, clinical sequence, and patient chronology where data is available.

Recommended workflow:
1. Open **Patient Timeline**.
2. Locate the relevant patient or case.
3. Generate a new timeline when needed from the session timeline workspace.
4. Reopen any previously generated timeline from the saved timeline preview list instead of regenerating it.
5. Review timeline entries in chronological order.
6. Compare exposure dates against lab abnormalities and symptoms.
7. Use the timeline to refine DILI Agent input if needed.

In the **Timeline** tab, the generation action uses the model assigned to the
Timeline role in Model Configurations. Use **Manage model roles** when that
assignment needs to change. Saved timelines appear as compact rows that record
the run's provider, model, date range, event count, and evidence-quality
warnings. Use **Open** to reopen a specific saved timeline or **Delete** to
remove only that saved timeline after confirmation.

The timetable presents a vertical chronology grouped by canonical date so events on
the same day remain readable without overlapping cards. Clinical, Medication,
Laboratory, Uncertain, and Date not reported categories remain explicit through
labels and category controls. Use its evidence filter, category collapse controls,
dense/compact/comfortable density, and previous/next navigation to focus review.
Selecting an event opens the desktop Event inspector. Approximate placement, a **Fallback chronology**, and **Missing
source evidence** are visible warnings, not clinical confirmation.

Use the help popover beside **Review controls** when the filter names need context. Evidence filters describe source support, density changes reading comfort, uncertain timing keeps approximate events visible, and **Inspect details** opens the event's source and confidence rationale. These controls do not alter the saved timeline.

Timeline generation may show a fallback notice when model extraction does not complete. For an explicitly selected OpenCode Go model, a temporary model-catalog outage does not prevent the known routed request from being attempted. The notice now identifies the failure class, such as provider network unavailable, provider timeout, authentication rejected, rate limited, upstream error, invalid structured response, or incomplete configuration. Transient network, timeout, rate-limit, and upstream failures are retried with bounded backoff before fallback. In that case, the timetable is built deterministically from persisted session fields with uncertain timing and no invented exact dates. Treat fallback events as navigation aids rather than model-extracted chronology, then retry after correcting the reported condition.

For LLM-generated timelines, events without preserved source evidence are not part of the persisted clinical chronology contract. In the UI, missing source evidence should be treated as a warning rather than as clinically grounded support.

## Inspect Local Data
Open **Data Inspection** from the sidebar.

Expected capabilities:
- resource or table selection
- refresh controls
- record counts or metadata
- table-style inspection
- search, filter, or pagination where supported
- embedding or resource update status where supported

Recommended workflow:
1. Open **Data Inspection**.
2. Select the resource or dataset.
3. Refresh the view.
4. Confirm expected records are present.
5. Use filters or pagination to inspect specific records.

Do not edit local database files manually while the application is running.

## Update Local Resources
Some resources or embeddings may require initialization or refresh through:

```text
start_on_windows.ps1
```

Use its menu options for database initialization, dependency maintenance, test execution, log cleanup, or cache cleanup.

Expected result:
- progress is reported
- long-running jobs show status
- refreshed resources become available after completion
- restarting the application after maintenance is recommended
