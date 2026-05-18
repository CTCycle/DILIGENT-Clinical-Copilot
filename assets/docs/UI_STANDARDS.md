# UI Standards

Last updated: 2026-05-17

Scope: `DILIGENT/client/src` (Angular + SCSS).

## 1. Typography

- Primary families:
  - `"Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif`
  - `"Manrope", "Inter", sans-serif` for emphasized headings.
- Tokenized size scale from `styles.scss`:
  - `--font-xs: 11px`
  - `--font-sm: 12px`
  - `--font-base: 14px`
  - `--font-md: 16px`
  - `--font-lg: 18px`
  - `--font-xl: 20px`
  - `--font-2xl: 28px`
  - `--font-3xl: 32px`
- Readability:
  - Body text line-height >= `1.5`
  - Headings around `1.1` to `1.2` line-height.

## 2. Layout and Spacing

- Spacing tokens:
  - `--space-xs: 4px` through `--space-3xl: 32px`.
- Control heights:
  - `--control-height-sm: 36px`
  - `--control-height-md: 40px`
  - `--control-height-lg: 44px`
- Main page layouts:
  - DILI page uses responsive grid (`.stitch-dili-grid`) and sticky sidebar on desktop.
  - Model config uses two-column/two-row layout (`.model-config-layout`).
  - Clinical Sessions uses a list/detail workspace with AI preview, session editing, document/image metadata summaries, revision, and timeline actions.
  - Inspection page uses tabbed sections with scroll-aware tables/lists for reference data and RAG resources.

## 3. Color System

- Theme model:
  - Light theme in `:root`
  - Dark theme in `:root[data-theme="dark"]`
- Core palette tokens:
  - Brand: `--color-brand`, `--color-brand-light`, `--color-brand-bg`
  - Text: `--color-text-primary`, `--color-text-secondary`, `--color-text-muted`, `--color-text-subtle`
  - Surfaces/borders: `--color-surface`, `--color-surface-alt`, `--color-border`, `--color-border-subtle`, `--color-divider`
- Semantic status tokens:
  - info, success, error each with text/background/border variables.

## 4. Components and Patterns

- Buttons:
  - Preserve visible hover, active, and disabled states.
  - Keep icon buttons labeled for accessibility.
- Inputs/forms:
  - Use shared control sizing and focus states.
  - Maintain clear invalid/feedback messaging patterns.
- Modals:
  - Use `ModalShellComponent` and consistent close actions.
- Navigation:
  - Sidebar/tab patterns must support keyboard navigation.
- Tables and scroll areas (inspection):
  - Keep fixed action-column sizing and responsive overflow behavior.

## 5. Page Structure

- Routes:
  - `/` -> DILI analysis page
  - `/clinical-sessions` -> Clinical Sessions page
  - `/data` -> Data inspection page
  - `/model-config` -> Model configuration page
  - `/sessions/:sessionId/timetable` -> patient timeline page
- App shell:
  - Root shell plus shared navigation (`NavSidebarComponent`) with page-level composition.

## 6. User Experience Rules

- Core journeys must remain consistent:
  - Run clinical analysis job
  - Browse, edit, annotate, revise, and generate timelines for clinical sessions
  - Configure models/providers and keys
  - Inspect reference catalogs and run update jobs
- Error feedback:
  - Use clear user-safe messages from centralized API error normalization.
- Loading and empty states:
  - Always provide explicit loading status and empty-state messaging.
- Job UX:
  - Keep terminal states explicit (`completed`, `failed`, `cancelled`).
- Clinical Sessions:
  - Session detail is the canonical UI source for report preview, parser output, metadata, revision audit, and timeline entry points.
  - Metadata UI should summarize `documents` and `images` from the same persisted metadata JSON instead of storing parallel attachment state.
  - Revision Mode should keep full-session reprocessing as the default while allowing a selected excerpt and free-text revision instruction to focus the second pass.

## 7. Responsiveness

- Existing breakpoints to preserve:
  - around `1100px` (main grid collapse)
  - around `1080px` (inspection/model layout fallback)
  - around `720px` (mobile stacking and table overflow)
- Mobile constraints:
  - Avoid clipped controls; enable horizontal scrolling for dense tables.

## 8. Accessibility

- Keyboard navigation must be supported for nav, tabs, modal actions, and key controls.
- Keep visible focus indicators (`--focus-ring` behavior).
- Use ARIA attributes where interactive semantics are not native.
- Do not use color alone to indicate status; pair with text/icons.
- Respect reduced-motion preferences for non-essential transitions and animations.

## 9. Component usage rules

- Prefer shared sizing tokens before introducing new values:
  - `--control-height-sm`, `--control-height-md`, `--control-height-lg`
  - `--radius-sm` through `--radius-xl`
  - `--shadow-sm` through `--shadow-lg`
- Interactive controls should use:
  - a visible hover state,
  - a visible `:focus-visible` state,
  - a distinct disabled state,
  - and a minimum comfortable hit area aligned with the control-height scale.
- Page-local color values are acceptable only when they express a page-specific illustration/background treatment. Text, borders, surfaces, and semantic states should prefer shared color tokens.
- Dense horizontal navigation (tabs, pill filters, compact toolbars) should wrap or scroll instead of clipping at narrow widths.

## 10. Do / Don’t

| Do | Don’t |
| --- | --- |
| Reuse spacing, radius, and color tokens. | Add near-duplicate one-off colors or spacing values without a reuse case. |
| Keep heading hierarchy limited and predictable. | Create new visual hierarchy through arbitrary font sizes alone. |
| Preserve visible keyboard focus. | Rely on color-only hover states as the only interaction cue. |
| Let dense controls wrap or scroll on small screens. | Force controls into cramped rows that clip labels. |
| Keep decorative page backgrounds separate from functional surfaces. | Encode functional meaning only through background hue. |

## 11. Design Principles

- Consistency over one-off styling.
- Clarity and predictability over decorative complexity.
- Reuse tokens first; add new tokens only when reusable across multiple views.
- Consolidate overrides when touching older blocks; avoid layered duplicate rules.

- Report output renders Markdown as formatted content in-app.
- Expanded report view uses full-page layout and only shows Collapse, Copy, Download controls.
- Raw Markdown is export data and should not be shown as the primary report presentation.
