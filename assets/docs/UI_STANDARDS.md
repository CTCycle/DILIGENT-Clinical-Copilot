# UI Standards (Frontend)

Scope: `DILIGENT/client/src`.
Goal: Preserve current product design while enforcing consistency, accessibility, and maintainability.

## Spacing Scale
- Base rhythm: `4px` step, prefer `8px` cadence for layout spacing.
- Tokens:
  - `--space-xs: 4px`
  - `--space-sm: 8px`
  - `--space-md: 12px`
  - `--space-lg: 16px`
  - `--space-xl: 20px`
  - `--space-2xl: 24px`
  - `--space-3xl: 32px`
- Rules:
  - Use token spacing in paddings/gaps/margins before introducing raw pixel values.
  - Keep inter-control spacing at `--space-xs`/`--space-sm` and section spacing at `--space-md`+.

## Typography Scale
- Font tokens:
  - `--font-xs: 11px`
  - `--font-sm: 12px`
  - `--font-base: 14px`
  - `--font-md: 16px`
  - `--font-lg: 18px`
  - `--font-xl: 20px`
  - `--font-2xl: 28px`
  - `--font-3xl: 32px`
- Rules:
  - Headings: page `h1` uses `--font-2xl` (responsive downshift on small screens), section `h2` uses `--font-xl`.
  - Body text should default to `--font-base`; helper/meta text should use `--font-sm` or `--font-xs`.
  - Keep mixed-case labels consistent (title-case for navigation items).

## Color System
- Primary brand:
  - `--color-brand`, `--color-brand-light`, `--color-brand-ui`, `--color-brand-ui-light`
- Text hierarchy:
  - `--color-text-primary`, `--color-text-secondary`, `--color-text-muted`, `--color-text-subtle`
- Surfaces and separators:
  - `--color-surface`, `--color-surface-alt`, `--color-border`, `--color-border-subtle`, `--color-divider`
- Semantic status colors:
  - Info: `--color-status-info-*`
  - Success: `--color-status-success-*`
  - Error: `--color-status-error-*`
- Rules:
  - Prefer semantic tokens over hardcoded hex values in component styling.
  - Do not rely on color alone for state; pair with text/icon/state labels.

## Component Usage Rules
- Buttons:
  - Use `.btn` + variant (`.btn-primary`, `.btn-secondary`, `.btn-tertiary`).
  - Control heights: `--control-height-sm`, `--control-height-md`, `--control-height-lg`.
  - Icon-only actions should keep minimum target size at least `--icon-action-size`.
- Inputs/selects/textarea:
  - Keep focus rings visible and consistent with `--focus-ring`.
  - Use tokenized control heights and paddings.
- Tabs:
  - If using `role="tablist"/"tab"/"tabpanel"`, include `aria-controls`, `aria-labelledby`, keyboard arrow/Home/End behavior, and active `tabIndex` management.
- Modals:
  - Use consistent close affordance (`.modal-close` with icon) and clear labels.

## Do and Don’t
- Do:
  - Use existing tokens and shared classes before adding new one-off values.
  - Keep responsive behavior explicit in breakpoint blocks.
  - Preserve visible focus states for keyboard users.
  - Keep diffs small and scoped to specific UI problems.
- Don’t:
  - Introduce duplicate selectors for the same component without clear intent.
  - Shrink interactive targets below accessible touch/click sizes.
  - Add decorative style variance that changes product identity.
  - Hide important state only behind color differences.
