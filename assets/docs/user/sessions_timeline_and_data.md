# Sessions, Timeline, And Data
Last updated: 2026-06-12

## Review Saved Clinical Sessions
Open **Clinical Sessions** from the sidebar.

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

## Use Patient Timeline
Open **Patient Timeline** from the sidebar.

Use it to review event order, clinical sequence, and patient chronology where data is available.

Recommended workflow:
1. Open **Patient Timeline**.
2. Locate the relevant patient or case.
3. Review timeline entries in chronological order.
4. Compare exposure dates against lab abnormalities and symptoms.
5. Use the timeline to refine DILI Agent input if needed.

Timeline generation may show a fallback notice when local model extraction is unavailable. In that case, the timetable is built deterministically from persisted session fields and should be treated as a navigation aid rather than a model-extracted chronology.

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
setup_and_maintenance.bat
```

Use its menu options for database initialization, dependency maintenance, Python cache cleanup, or embedding updates.

Expected result:
- progress is reported
- long-running jobs show status
- refreshed resources become available after completion
- restarting the application after maintenance is recommended
