# Experience
Last updated: 2026-06-15

## Page Structure
- Routes:
  - `/` for DILI analysis
  - `/clinical-sessions` for Clinical Sessions
  - `/data` for Data Inspection
  - `/model-config` for Model Configurations
  - `/sessions/:sessionId/timetable` for Patient Timeline
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

## Responsiveness
- Preserve breakpoints around:
  - `1100px` for main grid collapse
  - `1080px` for inspection and model layout fallback
  - `720px` for mobile stacking and table overflow
- Avoid clipped controls on mobile.
- Enable horizontal scrolling for dense tables.

## Accessibility
- Support keyboard navigation for navigation, tabs, modal actions, and key controls.
- Keep visible focus indicators through `--focus-ring`.
- Use ARIA attributes when interactive semantics are not native.
- Do not use color alone to indicate status.
- Respect reduced-motion preferences for non-essential transitions.

## Design Principles
- Consistency over one-off styling.
- Clarity and predictability over decorative complexity.
- Reuse tokens first and add new ones only when reusable across multiple views.
- Consolidate overrides when touching older blocks.
- Render report output as formatted in-app content.
- Expanded report view should show only Collapse, Copy, and Download controls.
- Raw Markdown is export data, not the primary on-screen presentation.
