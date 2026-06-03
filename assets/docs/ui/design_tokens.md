# Design Tokens
Last updated: 2026-06-03

## Scope
Frontend styling guidance applies to `app/client/src`.

## Typography
- Primary families:
  - `"Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif`
  - `"Manrope", "Inter", sans-serif` for emphasized headings
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
  - body text line-height at least `1.5`
  - headings around `1.1` to `1.2`

## Layout And Spacing
- Spacing tokens:
  - `--space-xs: 4px` through `--space-3xl: 32px`
- Control heights:
  - `--control-height-sm: 36px`
  - `--control-height-md: 40px`
  - `--control-height-lg: 44px`

## Color System
- Theme model:
  - light theme in `:root`
  - dark theme in `:root[data-theme="dark"]`
- Core palette tokens:
  - `--color-brand`
  - `--color-brand-light`
  - `--color-brand-bg`
  - `--color-text-primary`
  - `--color-text-secondary`
  - `--color-text-muted`
  - `--color-text-subtle`
  - `--color-surface`
  - `--color-surface-alt`
  - `--color-border`
  - `--color-border-subtle`
  - `--color-divider`
- Semantic status tokens exist for info, success, and error text, background, and border states.
