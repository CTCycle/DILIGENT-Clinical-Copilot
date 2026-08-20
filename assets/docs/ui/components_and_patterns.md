# Components And Patterns
Last updated: 2026-08-20

## Page Layout Patterns
- DILI page uses a desktop split grid through `.stitch-dili-grid` with a persistent sidebar.
- Model configuration uses a desktop multi-column layout through `.model-config-layout`.
- Clinical Sessions uses a list-detail workspace with AI preview, session editing, metadata summaries, revision actions, and timeline actions.
- Data Inspection uses tabbed sections with scroll-aware tables and lists for reference data and RAG resources.
- Clinical Sessions keeps preview rendering and related view state in focused preview components; page-level orchestration should not reabsorb that logic.
- Model Configurations and access-key dialogs use signal-backed local state, explicit loading/saving states, and provider-response sequencing.

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

### Contextual Guidance
- Use `HelpPopoverComponent` for a small explanation tied to a non-obvious control.
- Use `FeatureTipComponent` for a one-time, dismissible first-use callout with a short action.
- Use `GuidedTourComponent` only for a genuinely multi-step workflow. Definitions belong in `core/guidance/guidance-content.ts` and should remain short, target stable `data-guidance-target` attributes, and provide a useful fallback when a target is unavailable.
- Use `TipsAndTricksComponent` for manually reopened, concise workflow reminders. It is hosted by the existing `ModalShellComponent` and must remain optional.
- Persist guidance with `GuidanceStateService`; do not add backend tables or couple tutorial state to clinical session state.
- Do not add automatic guidance to routine navigation, common controls, or Data Inspection, which already explains its page-level actions.

### Tables And Scroll Areas
- Keep fixed action-column sizing where needed.
- Preserve bounded desktop overflow behavior for dense inspection views.

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
- Main navigation remains a single desktop row; the root viewport notice handles unsupported narrow widths.

## Do And Do Not
| Do | Do not |
| --- | --- |
| Reuse spacing, radius, and color tokens. | Add near-duplicate one-off colors or spacing values without a reuse case. |
| Keep heading hierarchy limited and predictable. | Create new hierarchy through arbitrary font sizes alone. |
| Preserve visible keyboard focus. | Rely on color-only hover states as the only interaction cue. |
| Use the available desktop width for dense controls and tables. | Add mobile-only navigation or compact layouts. |
| Keep decorative backgrounds separate from functional surfaces. | Encode functional meaning only through background hue. |
