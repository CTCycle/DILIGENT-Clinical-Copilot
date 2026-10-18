# Launcher option 9 and data-root validation

Date: 2026-10-18

## Launcher

Ran `start_on_windows.ps1` in an interactive PowerShell terminal. Selected option 9, chose Back (6) in the remove-artifacts submenu, dismissed the return prompt, and exited with option 15. The submenu rendered, the launcher reported `[OK] Remove release artifacts completed`, returned to the main menu, and exited with code 0. No release artifacts were selected for deletion.

The full launcher script passed PowerShell parsing. No launcher-integrity test was added to the repository.

## Data-root migration

Focused runtime-layout and legacy-data-migration checks passed: 5 passed. Ruff passed on the changed Python files; `TRY004` was ignored for two pre-existing findings in the archive validator. A read-only check of the existing packaged data location found six legacy entries and no destination conflicts; the updated packaged backend will migrate them at startup.
