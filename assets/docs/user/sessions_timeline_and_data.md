# Sessions, Timeline, And Data
Last updated: 2026-06-03

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
4. Review saved input and output.
5. Use copy or inspection controls if needed.

If a session is missing, confirm that the assessment completed successfully and that local persistence is initialized.

## Use Patient Timeline
Open **Patient Timeline** from the sidebar.

Use it to review event order, clinical sequence, and patient chronology where data is available.

Recommended workflow:
1. Open **Patient Timeline**.
2. Locate the relevant patient or case.
3. Review timeline entries in chronological order.
4. Compare exposure dates against lab abnormalities and symptoms.
5. Use the timeline to refine DILI Agent input if needed.

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

Use its menu options for database initialization, dependency maintenance, or embedding updates.

Expected result:
- progress is reported
- long-running jobs show status
- refreshed resources become available after completion
- restarting the application after maintenance is recommended
