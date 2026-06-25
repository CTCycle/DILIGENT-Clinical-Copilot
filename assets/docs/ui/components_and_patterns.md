# Components And Patterns
Last updated: 2026-06-25

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
- RAG dependency failures use a blocking decision modal with retry, run-without-RAG, and cancel actions. The run-without-RAG action applies only to the pending assessment.

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
