# Exact local timeline repeatability — 2026-09-24

## Scope and setup

This validation covered the current sessions.timeline implementation and the exact local route ollama / qwen3.5:2b: two independent synthetic sessions plus one qwen3.5:9b control.

- Started from clean develop HEAD 38b6a5380d69cfc1421d3580f6100f7592cf8c2e, equal to origin/develop.
- Confirmed the installed Ollama tags before generation. qwen3.5:2b digest: 324d162be6ca5629ae4517c8710434d0bd2d665bc94dbad46e9af8fbf8a2f0df; qwen3.5:9b digest: 6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7. No model was downloaded and no route was substituted.
- Started task-owned Ollama and used the official Windows launcher with an isolated SQLite path under this QA directory. The shared database, settings file, and .env were not changed.
- Seeded three synthetic sessions with the same source facts: acetaminophen in January 2025, symptoms beginning 2025-01-17, and ALT 75 U/L on 2025-01-17. The seed helper is seed_synthetic_sessions.py.
- Settings showed Local Ollama and qwen3.5:2b as the Timeline role for both independent 2B runs. Only the Timeline role was then changed and saved as qwen3.5:9b for the control.
- Used the rendered Clinical Sessions and Patient Timeline pages in Chrome. The in-app Browser was initially too narrow for the app’s 1100 px desktop minimum, so the same local runtime was opened in Chrome for the completed rendered checks.

## Results

| Case | Session / timeline | Exact route | Persisted outcome | Rendered and reload outcome |
|---|---|---|---|---|
| 2B A | Session 1 / timeline 1 | ollama / qwen3.5:2b | llm_generated, local/Ollama provenance, no generation note, 0 events, 0 evidence events | The detail page identified qwen3.5:2b and displayed 0 of 0 events. Reload retained the same record and provenance. No source date precision could be evaluated because the saved result had no events. |
| 2B B | Session 2 / timeline 2 | ollama / qwen3.5:2b | llm_generated, local/Ollama provenance, no generation note, 0 events, 0 evidence events | The detail page identified qwen3.5:2b and displayed 0 of 0 events. Reload retained the same record and provenance. No source date precision could be evaluated because the saved result had no events. |
| 9B control | Session 3 / timeline 3 | ollama / qwen3.5:9b | llm_generated, local/Ollama provenance, 3 events, all 3 with source evidence | The detail page identified qwen3.5:9b, rendered all three events and their evidence, and retained them after reload. |

The 9B events were:

- Acetaminophen intake, source drugs, with the verbatim source quote and normalized date 2025-01 at month precision.
- ALT elevation, source laboratory_analysis, with the verbatim source quote and normalized date 2025-01-17 at day precision.
- Symptom onset, source anamnesis, with the verbatim source quote and normalized date 2025-01-17 at day precision.

The retained Ollama HTTP log shows two successful `/api/chat` completions during each 2B job (1m51s and 1m35s for session 1; 1m26s and 1m32s for session 2) and one completion for the 9B control (3m41s). The backend log confirms exactly three timeline jobs and that each completed successfully. Ollama’s log does not include response bodies, and the retained backend extract does not attribute why each 2B job issued two chat requests. The implementation uses structured-output repair attempts, but the available run trace does not prove that repair caused the second request or explain the final empty result. Treat each persisted empty timeline as the observed case outcome; do not infer that every individual response was empty or attribute the omission to a specific retry/repair step. The app accepted each final empty event array under the extraction schema and stored it as `llm_generated`, without a generation error or fallback. No schema or evidence guard was weakened, and no implementation change was made because the trace did not identify an application defect to repair.

The service log records successful HTTP 200 responses for all five `/api/chat` calls. Several service-readiness `/api/tags` probes returned HTTP 500 before Ollama was ready; exact installed model availability was confirmed before generation. Supporting traces are [Ollama service HTTP log](ollama.stdout.log) and [backend job completion extract](backend-job-completions.log).

The 9B `/api/chat` call took 3m41s and the final timeline contained three evidence-backed events. It is a successful control for this fixture, not evidence that 9B is universally reliable or faster.

## Persistence and checks

The isolated database contains timeline IDs 1, 2, and 3 for sessions 1, 2, and 3 respectively. A direct read confirmed each record’s source_model, source_kind=local, and model_provider=ollama; the 9B source quotes match the persisted session fields. SQLite returned integrity_check=ok and no rows from foreign_key_check.

The focused timeline error diagnostics suite passed: 9 tests, with one existing Google GenAI deprecation warning. Ruff passed for the timeline service, diagnostics test, and seed helper. Ruff initially identified import ordering in the new seed helper; that formatting issue was corrected and the complete targeted Ruff command then passed. No application behavior or public API/type/database schema changed.

Validation commands (launcher and seed ran from the repository root; pytest ran from `app/server`; Ruff ran from the repository root):

    .\start_on_windows.ps1 -Action InitializeDatabase
    .\start_on_windows.ps1 -Action Launch
    .\app\server\.venv\Scripts\python.exe assets\QA\timeline-local-model-repeatability-20260924\seed_synthetic_sessions.py
    # From app/server
    .\.venv\Scripts\python.exe -m pytest ..\tests\unit\test_timeline_error_diagnostics.py -q -o 'cache_dir=..\..\runtimes\cache\pytest\timeline-local-repeatability-20260924' --basetemp='..\..\runtimes\cache\pytest\timeline-local-repeatability-20260924-temp'
    # From repository root
    .\app\server\.venv\Scripts\ruff.exe check app\server\services\inspection\timeline.py app\tests\unit\test_timeline_error_diagnostics.py assets\QA\timeline-local-model-repeatability-20260924\seed_synthetic_sessions.py --no-cache

The launcher and seed command used the process-level DILIGENT_SQLITE_PATH set to assets/QA/timeline-local-model-repeatability-20260924/runtime-data/resources/database.db, DATABASE_BACKEND=sqlite, and PYTHONPATH=app/server.

## Gate status and remaining limits

- sessions.timeline remains PARTIAL: the 9B control passed for this fixture, but both required independent 2B cases returned empty timelines. This slice does not establish repeatable grounded 2B generation.
- model.provider.local-ollama remains PARTIAL: exact 9B behavior is grounded for the control; exact 2B behavior did not cover available source events. The earlier unknown failures remain unattributed, and broader provider failure/recovery coverage is still required.
- test.automated-regression remains PARTIAL: the focused local diagnostics and Ruff passed, but the hosted live-provider dispatch lane and prerequisite-dependent E2E cases were not run here. The current GitHub CLI authentication was reported invalid; dispatch also requires the repository secret. Push-triggered CI is separate and does not run the dispatch-only live-provider lane.
- model.provider.opencode-go and revision.agentic-lifecycle remain PARTIAL; their provider and lifecycle matrix is independent of this local timeline fixture.
- auth.access-key-management remains BLOCKED pending an approved disposable credential and cleanup authorization. release.desktop.v3-4-0 remains BLOCKED pending its signed/tagged, hosted, packaging, and clean-machine prerequisites.
- data.sources.refresh and data.inspection.catalogs remain PARTIAL while the NCBI LiverTox master-list endpoint presents a human-verification challenge.
- The full API route catalog, clinical provider-failure and RAG-off cases, RAG duplicate-file policy, accessibility audit, and runtime.containerized (NOT_IMPLEMENTED) remain outside this slice.

The previous same-day 2B and 9B `unknown` failures remain unattributed. They are not reclassified by this run. The task-owned app and Ollama service were stopped, the isolated runtime database and uniquely named focused pytest cache were removed, and ports 7690, 9847, and 11434 were confirmed clear. Ordinary cache cleanup was denied by Windows; after verifying the exact absolute target was within the repository cache root, the task-owned cache was removed with elevated cleanup. The separate pytest basetemp path was absent. No unrelated processes or cache paths were changed.
