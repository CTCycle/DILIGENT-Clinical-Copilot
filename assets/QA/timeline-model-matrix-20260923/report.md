# Tier 4 timeline model and recovery validation

Date: 2026-09-23
Branch: develop
Starting commit: e933d3008a5773e52b21a676bd8d9b10964ca75d

## Scope and setup

The official Windows launcher ran the application on ports 7690 and 9847
against an isolated SQLite backup under
runtimes/cache/qa/timeline-model-matrix-20260923/timeline-model-matrix.db.
The shared database and saved runtime settings were not modified. The clone
retained the existing OpenCode Go configuration for the DeepSeek lane; no
credential value was printed or changed.

Three synthetic sessions used the same source facts:

- Anamnesis: “Symptoms started on 2025-01-17.”
- Drugs: “Acetaminophen was taken in 2025-01.”
- Laboratory analysis: “ALT was 75 U/L on 2025-01-17.”

The in-app Browser showed both Ollama models as installed after refreshing the
local catalog. Each generation used the displayed Timeline role model. The
OpenCode Go catalog identified deepseek-v4-flash as DeepSeek V4.1 Flash.

## Live model matrix

| Session | Configured lane | Persisted provenance | Outcome and evidence |
|---:|---|---|---|
| 25 | Local Ollama, qwen3.5:2b | local / ollama / qwen3.5:2b | Saved as **Fallback chronology** with invalid_response (“Invalid structured provider response”). Three fallback-parser events retained the source text at month precision for medication and day precision for symptoms and ALT. The UI visibly labeled the fallback and its failure class. Reload retained that history entry. No LLM-generated result was produced for this model. |
| 26 | Local Ollama, qwen3.5:9b | local / ollama / qwen3.5:9b | Saved as **LLM generated**, with three events. The symptom and ALT events had explicit day precision on 2025-01-17 and exact matching source-evidence quotes. The medication quote matched the drugs text and remained month precision (2025-01); the UI marked it inferred and showed a month-level review note. The event-level Source field was “Not reported” and confidence was “Not scored” on all three events. Reload retained the model provenance and chronology. |
| 27 | Cloud OpenCode Go, deepseek-v4-flash (catalog identity: DeepSeek V4.1 Flash) | cloud / opencode_go / deepseek-v4-flash | Saved as **LLM generated**, with three evidence-backed events. Symptoms and ALT were placed on 2025-01-17 at day precision; acetaminophen was placed in January 2025 at month precision. The displayed sources were anamnesis, laboratory_analysis, and drugs, respectively, with exact matching evidence. Reload retained the route provenance and chronology. |

The read-only SQLite check found exactly one timeline per synthetic session.
Session 25 stored fallback status and invalid_response, with all three event
sources set to fallback_parser. Session 26 stored llm_generated, the exact
Ollama model/provider, and three events whose event-level Source fields were
null. Session 27 stored llm_generated, opencode_go, the exact cloud model,
and the three expected source fields. The Browser showed no console warnings
or errors. The timeline page was visually inspected inline; the Browser did
not expose a disk-export path, so no screenshot file is claimed.

## Controlled recovery regressions

| Check | Result |
|---|---|
| Focused backend timeline tests | 45 passed across test_timeline_retry_behavior.py, test_timeline_error_diagnostics.py, and test_data_inspection_repository.py. |
| Retry behavior | Synthetic timeout and rate-limit errors retried and succeeded on the second call. Authentication and invalid-response errors were not retried. |
| Persisted fallback diagnostics | Synthetic timeout, authentication, and rate-limit failures persisted as fallback timelines with their exact error codes. No real authentication failure or rate limit was triggered. |
| Timeline component spec | 12 passed, including visible timeout, authentication, and rate-limit labels. |
| Ruff | All changed Python files passed with DTZ001 ignored. The two focused retry/diagnostic files also passed without ignores. A strict run over the existing repository test file reports 14 naive-datetime findings on untouched lines; the new test timestamp is timezone-aware UTC. |

## Gate result

sessions.timeline remains **PARTIAL** because qwen3.5:2b still returned
invalid structured data and did not produce grounded LLM output. The qwen3.5:9b
and OpenCode Go DeepSeek V4.1 Flash lanes did produce LLM-generated timelines
with evidence matching the synthetic source, subject to the 9B event Source
field limitation above. The local Ollama gate also remains **PARTIAL**. This
timeline slice does not promote the broader model.provider.opencode-go gate.

No API, public type, or database schema changes were made. The temporary
database, test basetemp folders, and task-started launcher processes were
removed after evidence collection; ports 7690 and 9847 were verified free.
The user-started Ollama service was left running.

The synthetic seed helper is in
assets/QA/timeline-model-matrix-20260923/seed_synthetic_sessions.py.
