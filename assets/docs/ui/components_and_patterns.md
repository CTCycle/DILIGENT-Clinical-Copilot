# Components And Patterns
Last updated: 2026-07-17

## Page Layout Patterns
- DILI page uses a responsive grid through `.stitch-dili-grid` and a sticky sidebar on desktop.
- Model configuration uses a two-column or two-row layout through `.model-config-layout`.
- Clinical Sessions uses a list-detail workspace with AI preview, session editing, metadata summaries, revision actions, and timeline actions.
- Data Inspection uses tabbed sections with scroll-aware tables and lists for reference data and RAG resources.

## Shared Component Rules
### Buttons
- Preserve visible hover, active, and disabled states.
- Keep icon buttons labeled for accessibility.

### Inputs And Forms
- Use shared control sizing and focus states.
- Maintain clear invalid and feedback messaging.

### Modals
- Use `ModalShellComponent`.
- Keep close actions consistent.
- DILI pre-flight issues are aggregated in one constrained modal before job submission.
- Blocking issues allow only a return to the input panel.
- Non-blocking warnings require an explicit choice between returning to the input panel and continuing with accepted limitations.
- RAG readiness is represented as a non-blocking pre-flight warning. Continuing disables RAG only for the pending assessment.
- Modal headers and footers remain visible while only long content regions scroll.
- Shared modal behavior traps focus, restores focus, blocks background interaction, and supports Escape through the same safe close path.

### Navigation
- Sidebar and tab patterns must support keyboard navigation.

### Tables And Scroll Areas
- Keep fixed action-column sizing where needed.
- Preserve responsive overflow behavior for dense inspection views.

## Component Usage Rules
- Prefer shared sizing tokens before introducing new values:
  - `--control-height-sm`
  - `--control-height-md`
  - `--control-height-lg`
  - `--radius-sm` through `--radius-xl`
  - `--shadow-sm` through `--shadow-lg`
- Interactive controls should always have:
  - visible hover
  - visible `:focus-visible`
  - distinct disabled state
  - comfortable hit area
- Page-local color values are acceptable only for page-specific illustration or background treatments.
- Dense horizontal navigation should wrap or scroll instead of clipping on narrow widths.

## Do And Do Not
| Do | Do not |
| --- | --- |
| Reuse spacing, radius, and color tokens. | Add near-duplicate one-off colors or spacing values without a reuse case. |
| Keep heading hierarchy limited and predictable. | Create new hierarchy through arbitrary font sizes alone. |
| Preserve visible keyboard focus. | Rely on color-only hover states as the only interaction cue. |
| Let dense controls wrap or scroll on small screens. | Force controls into cramped rows that clip labels. |
| Keep decorative backgrounds separate from functional surfaces. | Encode functional meaning only through background hue. |
