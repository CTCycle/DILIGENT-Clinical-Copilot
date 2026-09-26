# UI shell keyboard and responsive validation

Last updated: 2026-09-26

## Scope and baseline

This slice revisited the `ui.application-shell` validation debt for keyboard
operation and responsive desktop rendering. It started on `develop` at
`e3a8fcf07adccecec66ae0cd430a382f568558ac` with the working tree change that
adds focus management to the primary navigation tabs. No LLM provider request,
credential mutation, source refresh, or clinical analysis was performed.

The official launcher was attempted first with a task-owned SQLite path. It
stopped before application startup when the local Angular production build
exited with `-1073741819` (`0xC0000005`). A listener was observed on source
port `9847` during fallback, so the current source build used an isolated
Angular cache and port `9848` while ownership was checked. The listener was
verified as task-created and stopped during cleanup; the backend used an
isolated SQLite database on `7690`.

## Finding and remediation

`VAL-20260926-002` — Primary navigation handled Arrow/Home/End selection and
`tabindex` updates, but did not move focus to the newly selected tab. The live
AX tree showed the old tab focused after ArrowRight navigation, leaving the
roving-tabindex interaction incomplete.

The navigation component now focuses the target button in a microtask after
emitting the route change. The browser regression test asserts both
`aria-selected="true"` and DOM focus on the target tab.

## Current implementation evidence

| Check | Result |
|---|---|
| Isolated backend startup | Fresh task-owned SQLite migration completed and `GET /api/health` returned `200`. |
| Angular/Vitest suite | `24` files and `105` tests passed. |
| Production frontend build | `npm run build -- --progress=false` passed with the Angular cache isolated under this QA directory; the normal protected cache path was not modified. |
| Selected browser E2E | `8 passed, 17 deselected`: tab traversal, ArrowRight focus, form labels/focus, Settings routes, legacy redirect, scroll restoration, and Data Inspection navigation. |
| Default viewport | In-app Browser rendered the DILI Agent shell at `1280 × 720`; all four primary workspaces were present in the accessibility tree. |
| Narrow supported viewport | At `1100 × 900`, DILI Agent rendered without the minimum-width alert; the AX tree exposed all four primary tabs, clinical input, patient fields, RAG checkbox, and Run/Clear controls. |
| Wide supported viewport | At `1920 × 1080`, Settings rendered with its persistent section navigation and General controls visible. |
| Below-minimum viewport | At `1099 × 900`, the app presented the accessible alert `Widen the application window to continue` with the documented 1100-pixel requirement. |
| Keyboard focus | Tab traversal reached the primary controls; ArrowRight moved from DILI Agent to Clinical Sessions, changed selection, and left Clinical Sessions focused. |

The Browser screenshots and AX observations were inspected live. No Narrator
or Speech Recap observation was performed, so this report makes no claim about
spoken screen-reader output.

## Gate disposition

| Gate | Final status | Remaining limitation |
|---|---|---|
| `ui.application-shell` | `VALIDATED` for the exercised keyboard, rendered-shell, and responsive desktop scope | Spoken screen-reader output and a broader component-level accessibility audit remain open. |
| `test.automated-regression` | `PARTIAL` | The local frontend suite/build and selected browser slice passed; provider-key-dependent E2E and conditional hosted lanes remain independent. |
| `runtime.startup.source-launcher` | `VALIDATED` for its previously established source-launcher contract | This run reproduced the host-specific local Angular build failure before launch; isolated-cache build succeeded, so no launcher regression is inferred. |

The provider, credential, NCBI LiverTox, desktop-release, revision-matrix,
RAG duplicate-policy, and container-runtime dispositions remain unchanged from
the current ledgers. The exact local Ollama 2B compatibility result was not
retried.

## Cleanup

The task-owned backend, Angular server, and preview listener were stopped after
validation. The isolated database, cache, key-material path, pytest workspace,
and temporary server logs were removed. Protected pre-existing repository
cache residue was preserved. No patient-identifying data or secret values were
written to this report.
