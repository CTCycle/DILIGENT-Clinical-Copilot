# Frontend consistency refinement QA

**Run date:** 2026-09-25

**Scope:** DILI Agent, Clinical Sessions, Data Inspection, Patient Timeline, and all Settings sections.

## Rendered comparison

The capture harness rendered 22 route/theme/viewport states. At each route, the shared header measured 59px high and the page frame began at y=59px. The frame was x=0 with width 1,356px at 1366×900, x=0 with width 1,090px at 1100×900, and centered at x=187 with the 1,536px container cap at 1920×1080. All captures reported no document-level horizontal or vertical overflow.

The app showed the existing desktop-width notice at 1099×900 and kept the application content hidden without introducing horizontal overflow. Light and dark theme captures retained the existing product colors. At 1100px, the Settings model controls reflowed into a two-column arrangement with the catalog below.

A transient service error was used to check notification placement. At 1366×900 the toast was measured at x=453, y=836; at 1100×900 it was x=320, y=836. It stays below the 59px header instead of being clipped over the navigation.

Screenshots and geometry are in this folder. `visual-metrics.json` records each viewport and route; `visual-capture.log` contains the compact comparison output.

## Automated checks

- `npm run build -- --no-progress` — passed.
- `npm test -- --watch=false --progress=false` — 24 files and 105 tests passed.
- Existing app flow coverage — 7 passed: model settings navigation, legacy `/model-config` redirect, navigation scroll restoration, Data Inspection navigation, invalid timeline id handling, keyboard navigation, and deterministic timeline chronology/inspector layout.

## Scope limits

The backend was not running during visual validation, so backend-backed screens correctly showed unavailable or empty states. The capture harness supplied read-only synthetic settings and timeline fixtures to render those screens; no form was submitted and no persisted data was changed. Route-flow tests covered navigation and presentation contracts, not backend persistence. Pytest reported the repository's `cache_dir` option as unknown while the cache plugin was disabled; all seven selected tests passed.
