# Experience
Last updated: 2026-10-30

## Page Structure
- Routes:
  - `/` for DILI analysis
  - `/report` for the active DILI report and `/report/:sessionId` for a saved session's report
  - `/clinical-sessions` for Clinical Sessions
  - `/data` for Knowledge
  - `/settings` (and `/settings/models` for model configuration)
  - `/sessions/:sessionId/timetable` for Patient Timeline
  - `/sessions/:sessionId/timetable/:timelineId` for a saved timeline
- App shell uses the root shell plus shared navigation through `NavSidebarComponent`.

## Core UX Rules
- Core journeys must remain consistent:
  - run clinical analysis
  - browse, edit, annotate, revise, and generate timelines for clinical sessions
  - configure models, providers, and keys
  - inspect reference catalogs and run update jobs
- Use clear user-safe messages from centralized API error normalization.
- Always provide explicit loading and empty-state messaging.
- Keep job terminal states explicit: `completed`, `failed`, `cancelled`.

## Clinical Sessions UX Rules
- Session detail is the canonical UI surface for report preview, parser output, metadata, revision audit, and timeline entry points.
- Timeline generation from Session detail must preserve prior generated timelines as a revisitable history, not overwrite the user's only navigable entry point.
- Session lists should stay in bounded scroll containers so page height remains stable with large histories.
- Preview reports render as formatted HTML, not raw Markdown or plain text.
- Drug evidence indicators should prefer persisted pipeline match metadata before catalog fallback.
- Laboratory previews should show all retrieved `lab_timeline` occurrences when available.
- Metadata UI should summarize `documents` and `images` from the same persisted metadata JSON.
- Revision Mode should keep full-session reprocessing as the default while still allowing a selected excerpt and free-text revision instruction.
- Model catalog reads reuse persisted state; **Refresh** is the explicit provider-contact action, and valid cached catalogs remain visible when a later refresh fails.
- Access-key dialogs keep new keys inactive until explicit activation and expose only masked fingerprints after storage.

## DILI Agent Workspace
- The workspace is a single-screen two-column layout: the clinical text on the
  left fills the viewport height (no fixed heights) and keeps vertical resize;
  enlarging it lets the page scroll naturally, with no nested scroll area.
- The right column is sticky and holds patient details, the analysis progress
  panel, Run/Stop, Clear all, a discreet **View report** link, and draft
  recovery. Textarea resizing can never push the progress panel out of view.
- The report canvas is not on the workspace; reports live on `/report`, which
  reuses the shared report viewer (Copy, Expand, Download) and handles empty,
  running, failed, cancelled, and completed states.
- Progress maps backend `progress_stage` keys to eight stages and shows a
  determinate bar driven by backend progress without a numeric percentage
  (checkpoint weights are not time-proportional). Stage changes are announced
  through a polite live region.
- **Run DILI analysis** opens the pre-run setup dialog (Preflight, Evidence
  Sources, Models, Review) built on the shared modal shell, with a step list
  using `aria-current="step"`; later steps are disabled while preflight
  blocks.

## DILI Agent Draft Recovery
- Saving is explicit the first time. Only after a successful Save draft, or an
  explicit Resume, do later form changes autosave after a short debounce.
- A saved draft appears in a non-modal recovery banner and is never applied
  automatically. Resume restores patient name, visit date, and clinical text;
  it clears the image and tells the user to select it again.
- Resuming over different current input requires confirmation. Resume is
  disabled while an analysis job is active and does not start or replace a job.
- Save state is announced through a polite live region. Unsaved changes,
  failures, and stale-draft/database conflicts remain visible; conflicts pause
  autosave until the user reloads the saved version.
- Discard and Clear all confirm before deleting a saved draft. A failed delete
  leaves the current input intact. Keep the recovery and confirmation controls
  keyboard reachable with the shared visible focus treatment.
- Draft contents live in the configured database and are not written to
  browser storage. Patient images are not part of a draft.

## Desktop viewport
- DILIGENT is a desktop application delivered through web technologies.
- The supported viewport is at least `1100px` wide in CSS pixels.
- At narrower widths, keep the desktop interface behind the minimum-window-size notice; do not switch to a mobile navigation or stacked layout.
- Optimize for mouse and keyboard use, information density, and horizontal workspace utilization.
- Windows tablets are supported only when they present a normal desktop browser viewport of at least `1100px`.
- Keep horizontal scrolling for genuinely dense tables and vertical scrolling for long workspaces.

## Accessibility
- Support keyboard navigation for navigation, tabs, modal actions, and key controls.
- Keep visible focus indicators through `--focus-ring`.
- Use ARIA attributes when interactive semantics are not native.
- Do not use color alone to indicate status.
- Respect reduced-motion preferences for non-essential transitions.

## Contextual Guidance
- Help is available from the existing header and opens the concise **Tips & Tricks** modal.
- Only the DILI Agent first-assessment callout appears automatically, and only until it is seen or dismissed in that browser.
- The optional DILI walkthrough has three steps for clinical input, patient name/date, and review/run. It can be closed at any point, supports Back and restart from Help, and uses stable `data-guidance-target` anchors.
- Runtime/RAG, Clinical Session sections, timeline generation, and Patient Timeline review controls use click-or-keyboard popovers rather than automatic tutorials.
- Guidance state is browser-local, versioned per definition, and tolerant of unavailable local storage. A revised content version can reintroduce only the affected guidance.
- Guidance popovers and tours must restore focus, remain keyboard navigable, avoid covering the highlighted control, and honor `prefers-reduced-motion`.

## Design Principles
- Consistency over one-off styling.
- Clarity and predictability over decorative complexity.
- Reuse tokens first and add new ones only when reusable across multiple views.
- Consolidate overrides when touching older blocks.
- Render report output as formatted in-app content.
- Expanded report view should show only Collapse, Copy, and Download controls.
- Raw Markdown is export data, not the primary on-screen presentation.
