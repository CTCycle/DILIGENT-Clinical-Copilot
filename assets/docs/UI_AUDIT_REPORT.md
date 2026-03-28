# UI/UX Audit Report

Audit date: 2026-03-28  
Scope: `DILIGENT/client/src` + `DILIGENT/client/src/styles.css`

This report captures current UI debt and prioritization.
Use `assets/docs/UI_STANDARDS.md` as the target standard.

## 1. High-priority findings

1. Duplicate CSS blocks increase override drift risk
- Area: Model Config and inspection-related selectors in `styles.css`
- Impact: hard-to-predict rendering, fragile maintenance
- Priority: High
- Recommended action: consolidate duplicated selectors into one canonical block per component

2. Some icon-only action controls remain below comfortable target size
- Area: dense table/action controls (notably inspection views)
- Impact: accessibility and usability degradation on touch/trackpad
- Priority: High
- Recommended action: standardize minimum interactive target size and update spacing to preserve density

3. Tab semantics are not consistently complete in all tab-like UIs
- Area: inspection and similar tab containers
- Impact: keyboard/screen-reader behavior inconsistency
- Priority: High
- Recommended action: enforce full tab semantics (`tablist/tab/tabpanel`, keyboard nav, labeling links)

## 2. Medium-priority findings

1. Focus-visible styles are uneven across custom controls
- Impact: keyboard discoverability issues
- Action: normalize `:focus-visible` coverage

2. Mixed spacing and control heights still appear in some sections
- Impact: visual inconsistency and harder long-term maintenance
- Action: migrate one-off values to shared spacing/control-size tokens

3. Modal close affordances vary by page
- Impact: interaction inconsistency
- Action: standardize modal close controls and behavior patterns

## 3. Low-priority findings

1. Minor copy/capitalization inconsistencies in labels
2. Layout tuning opportunities for large desktop widths

## 4. Execution plan

1. Quick wins:
- raise small target sizes
- complete tab semantics
- normalize focus-visible coverage
- unify modal close patterns

2. Structural cleanup:
- consolidate duplicate CSS sections
- remove obsolete selectors after verification
- reduce one-off px values in high-traffic components

## 5. Validation checklist after UI changes

- Keyboard-only navigation works end-to-end.
- Focus state is visible on all interactive elements.
- No regressions in primary pages (`DiliAgentPage`, `ModelConfigPage`, `DataInspectionPage`).
- Desktop and mobile layouts remain usable.
