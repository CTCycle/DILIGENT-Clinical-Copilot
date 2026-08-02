# UI Standards
Last updated: 2026-08-02

## Spacing Scale
- Use the root spacing tokens from `app/client/src/styles.scss`: `--space-xs` through `--space-3xl`.
- Default to the 8px rhythm: `--space-sm` for tight gaps, `--space-lg` for grouped controls, `--space-xl` and `--space-2xl` for panel padding.
- Avoid raw pixel spacing in page styles unless it is intrinsic to a fixed-format surface such as table columns, timelines, or canvas-like layouts.
- Use `--control-height-sm`, `--control-height-md`, and `--control-height-lg` for interactive control height.

## Typography Scale
- Use `--font-xs`, `--font-sm`, `--font-base`, `--font-md`, `--font-lg`, `--font-xl`, `--font-2xl`, and `--font-3xl`.
- Body and form text should normally use `--font-base`; compact metadata and helper text should use `--font-sm` or `--font-xs`.
- Use `"Manrope", "Inter", sans-serif` for headings and `"Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif` for body text.
- Do not introduce fractional font sizes for visual tuning; choose the closest token.

## Color System
- Use theme tokens for reusable colors: `--color-brand`, `--color-brand-bg`, text tokens, surface tokens, border tokens, and semantic status tokens.
- Direct colors are acceptable only for domain-specific visualization, generated event colors, or non-reusable illustration details.
- Disabled states must remain visible and readable through both color and opacity/state, not color alone.
- Status UI must use semantic text/background/border tokens and include text labels or icons where color conveys meaning.

## Component Rules
- Buttons must have a visible hover state, visible `:focus-visible`, distinct disabled state, and comfortable hit area.
- Inputs must have associated labels or explicit ARIA labels, tokenized padding, and a visible focus state.
- Use `ModalShellComponent` for app dialogs unless a page has a documented specialized modal pattern.
- Tabs and dense navigation may wrap or scroll on narrow screens; they must not clip labels.
- Cards and panels should use `--radius-sm` through `--radius-xl` and `--shadow-sm` through `--shadow-lg`.

## Do And Don't
| Do | Don't |
| --- | --- |
| Reuse existing tokens before adding a new value. | Add near-duplicate colors, radii, or fractional font sizes for one screen. |
| Preserve current workflow and information architecture. | Redesign a page when a token or spacing polish is enough. |
| Keep dense data tables and timeline canvases horizontally scrollable. | Force dense content into narrow columns that clip controls. |
| Store validation notes and screenshots in `assets/QA/`. | Scatter audit artifacts across source folders. |
