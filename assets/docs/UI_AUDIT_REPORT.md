# UI/UX Audit Report

Date: 2026-03-27
Scope: `DILIGENT/client/src` (React + TypeScript + global CSS)

## 1) Discovery

### Project structure and routing
- Shell and routing: `DILIGENT/client/src/App.tsx`
- Main pages:
  - `DILIGENT/client/src/pages/DiliAgentPage.tsx`
  - `DILIGENT/client/src/pages/ModelConfigPage.tsx`
  - `DILIGENT/client/src/pages/DataInspectionPage.tsx`
- Shared components:
  - `DILIGENT/client/src/components/*`

### Styling approach
- Single global stylesheet: `DILIGENT/client/src/styles.css`
- CSS custom properties for colors/typography/spacing at `styles.css:6-117`
- No utility framework (no Tailwind), no CSS modules, no CSS-in-JS

### Repeated patterns and duplication
- The stylesheet contains duplicated selectors and layered overrides (legacy + redesign) for Model Config and some layout blocks.
- Notable duplicates/overrides in `styles.css` include:
  - `.model-config-layout` (`236`, `2122`, plus responsive overrides)
  - `.model-config-search` (`306`, `2220`, `2580`)
  - `.model-config-row-header-top` (`299`, `2216`, `2575`)
  - `.model-config-availability-pill` and variants (`408`, `2476`)
  - `.inspection-active-view` (`2658`, `2726`, `3156`)

### Hardcoded values and inline styles
- Many one-off pixel values remain in layout and controls (for example `10px`, `26px`, `34px`, `360px`) across `styles.css`.
- Inline style usage in TSX is mostly for dynamic progress/position and is acceptable:
  - `DiliAgentPage.tsx:506, 519`
  - `ModelConfigPage.tsx:593-595`

## 2) Findings by screen/component

## A. Shared / Global CSS

1. **High** — Conflicting duplicated selector blocks increase drift risk and produce non-obvious overrides.
- Files: `DILIGENT/client/src/styles.css:236-732`, `2122-2607`
- Root cause: Two style generations for Model Config coexist in one file.
- Minimal fix: Keep one canonical block and remove duplicate declarations where selectors are identical; keep intentional overrides only.
- Category: Structural improvement.

2. **Medium** — Spacing and sizing scales are partially tokenized but inconsistent with one-off values.
- Files: `DILIGENT/client/src/styles.css` (multiple controls in sections around `2618-3045`)
- Root cause: Repeated hardcoded `px` values for paddings/heights/gaps.
- Minimal fix: Introduce/standardize control size tokens and replace one-off values in high-traffic controls.
- Category: Structural improvement.

3. **Medium** — Focus states are incomplete for several interactive controls.
- Files: `DILIGENT/client/src/styles.css:1608-1621` plus inspection controls at `2831-2877`, `2926-2958`
- Root cause: Focus ring list omits some controls (e.g., inspection icon buttons and pills).
- Minimal fix: Expand `:focus-visible` coverage and normalize focus ring usage.
- Category: Quick win.

## B. Navigation / Shell

4. **Medium** — Brand logo text colors are hardcoded and may have weak contrast in dark theme.
- Files: `DILIGENT/client/src/components/NavSidebar.tsx:19-23`
- Root cause: SVG text fill is fixed to dark shades.
- Minimal fix: Use `currentColor` and theme-aware color through CSS.
- Category: Quick win.

5. **Low** — Navigation width constraint may create awkward empty space or wrapping edge cases.
- Files: `DILIGENT/client/src/styles.css:1741-1745`
- Root cause: `width: clamp(560px, 40%, 760px)` for `.tab-items`.
- Minimal fix: Use `width: min(100%, 760px)` and rely on flex behavior.
- Category: Needs verification (visual intent-dependent).

## C. DILI Agent page

6. **Low** — Lab ratio visual controls use very small indicator/thumb dimensions.
- Files: `DILIGENT/client/src/styles.css:980-995`
- Root cause: Tiny thumb (`5px`) and track (`6px`) reduce readability.
- Minimal fix: Slightly increase dimensions for better visibility while preserving layout.
- Category: Quick win.

7. **Low** — Button stack has mixed min-height values (`38/40/44`) across contexts.
- Files: `DILIGENT/client/src/styles.css:1106,1122,1594`
- Root cause: Layered button overrides.
- Minimal fix: Normalize button control heights by semantic size tokens.
- Category: Structural improvement.

## D. Model Config page

8. **High** — Model Config layout and subcomponents are defined multiple times, increasing maintenance cost and mismatch risk.
- Files: `DILIGENT/client/src/styles.css:236-732`, `2122-2607`
- Root cause: Redesign overrides appended without consolidating prior block.
- Minimal fix: Consolidate selectors and remove duplicates with equivalent declarations.
- Category: Structural improvement.

9. **Medium** — Mobile/tablet alignment inconsistency for cloud block padding after border removal.
- Files: `DILIGENT/client/src/styles.css:2568-2571`
- Root cause: Border removed but left padding remains clamped.
- Minimal fix: Set consistent left padding (`0` or tokenized shared value) for narrow layouts.
- Category: Quick win.

10. **Low** — Role action controls are icon-only with compact size and no textual affordance.
- Files: `DILIGENT/client/src/pages/ModelConfigPage.tsx:622-643`, `styles.css:2465-2473`
- Root cause: Interaction model optimized for compactness over discoverability.
- Minimal fix: Keep aria labels, increase touch target slightly, and add clearer active-state contrast.
- Category: Quick win.

## E. Data Inspection page

11. **High** — Action icon buttons are below recommended touch/target size.
- Files: `DILIGENT/client/src/styles.css:2926-2937`, usage at `DataInspectionPage.tsx:359-361,395-397,448-449`
- Root cause: `26x26` action buttons and dense tables.
- Minimal fix: Increase to at least `34-36px`, maintain table density via spacing adjustments.
- Category: Quick win.

12. **High** — Tablist semantics are incomplete for robust keyboard/screen-reader behavior.
- Files: `DILIGENT/client/src/pages/DataInspectionPage.tsx:470-487`
- Root cause: `role="tab"` is present but no `id/tabIndex` management and panel not tied via `aria-labelledby`.
- Minimal fix: Add deterministic tab ids, `tabIndex` state, arrow-key navigation, and panel `aria-labelledby`.
- Category: Quick win.

13. **Medium** — Modal close affordance is inconsistent (`"X"` text) and visually weaker than shared icon-based close controls.
- Files: `DILIGENT/client/src/pages/DataInspectionPage.tsx:504,525`
- Root cause: Local inline close button implementation diverges from shared modal style.
- Minimal fix: Reuse icon-based close affordance for consistency.
- Category: Quick win.

14. **Medium** — Update button labels (`Stop/Update`) do not expose pressed/running state to assistive tech.
- Files: `DILIGENT/client/src/pages/DataInspectionPage.tsx:382,418`
- Root cause: Missing ARIA state metadata on toggling action.
- Minimal fix: Add `aria-pressed` and `aria-live` message consistency.
- Category: Quick win.

15. **Low** — Inconsistent capitalization in dataset labels.
- Files: `DILIGENT/client/src/pages/DataInspectionPage.tsx:44`
- Root cause: Label string uses sentence inconsistency (`"drug catalog"`).
- Minimal fix: Normalize to title case (`"Drug Catalog"`).
- Category: Quick win.

## F. Modals and overlays

16. **Medium** — Modal interaction model relies on overlay rendering without explicit keyboard dismissal handling in page-level dialogs.
- Files: `DILIGENT/client/src/pages/DataInspectionPage.tsx:499-534`, `DILIGENT/client/src/components/ConfirmModal.tsx:27-52`
- Root cause: No explicit escape/backdrop close handling in custom modal wrappers.
- Minimal fix: Add keyboard/backdrop close behavior where appropriate and safe.
- Category: Needs verification (behavioral expectations).

## 3) Quick wins vs structural

### Quick wins
- Increase interactive target sizes for inspection icon buttons and mini controls.
- Expand focus-visible states to all key interactive inspection/model controls.
- Fix data inspection tab semantics (`tabIndex`, ids, keyboard arrow navigation, `aria-labelledby`).
- Standardize close buttons in inspection modals to icon-based style.
- Normalize minor text consistency (`Drug Catalog`).
- Improve dark-mode logo text contrast.

### Structural improvements
- Consolidate duplicated Model Config CSS blocks.
- Rationalize control sizes and spacing via shared tokens.
- Clean dead/legacy selectors after consolidation.

## 4) Items marked “needs verification”
- Exact visual intent for nav width behavior on large desktop (`styles.css:1741-1745`).
- Modal dismissal behavior requirements (strict confirm flows vs easy dismissal).
- Any remaining contrast edge cases requiring runtime visual measurement (especially translucent surfaces in dark mode).
