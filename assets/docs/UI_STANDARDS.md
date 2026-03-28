# UI Standards (Frontend)

Last updated: 2026-03-28

Scope: `DILIGENT/client/src` and `DILIGENT/client/src/styles.css`.
Goal: keep UI coherent, accessible, and maintainable.

## 1. Design token usage

- Use existing CSS variables in `styles.css` before adding raw values.
- Prefer tokenized spacing, typography, colors, control sizes, and focus ring values.
- New tokens must be reusable across at least two UI areas.

## 2. Layout and spacing

- Maintain the current spacing rhythm (`4px` base, `8px` preferred cadence).
- Use consistent section spacing and avoid ad-hoc one-off margins/paddings.
- Keep responsive behavior explicit in media-query blocks.

## 3. Typography and labels

- Use the existing font-size tokens and heading scale.
- Keep UI labels and navigation naming consistent across pages.
- Avoid mixed conventions for similar controls.

## 4. Color and status semantics

- Use semantic color tokens, not hardcoded hex values.
- Do not communicate state with color only; include text/icons where relevant.
- Ensure contrast remains acceptable in both light and dark themes.

## 5. Component rules

- Buttons:
  - use shared button classes/variants,
  - keep accessible touch/click target size,
  - preserve disabled/loading states.
- Inputs/selects/textarea:
  - keep consistent control heights and paddings,
  - keep visible `:focus-visible` styling.
- Tabs:
  - implement complete ARIA semantics (`tablist`, `tab`, `tabpanel`, keyboard nav).
- Modals:
  - use consistent close affordance and labeling.

## 6. Accessibility baseline

- Keyboard navigation must work for all interactive controls.
- Preserve visible focus states.
- Use meaningful `aria-*` attributes for dynamic state and tab/modal patterns.
- Keep icon-only controls labelled for assistive technology.

## 7. Maintainability rules

- Keep diffs scoped to task intent.
- Avoid duplicating selector blocks for the same component.
- When redesigning a section, consolidate legacy CSS instead of layering conflicting overrides.

## 8. Relationship to audit report

`assets/docs/UI_AUDIT_REPORT.md` tracks current debt and concrete remediation items.
This standards file defines the target quality bar for new work.
