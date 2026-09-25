# API and persistence evidence — 2026-09-25

All requests below targeted the task-owned source runtime at `127.0.0.1`.

| Request/check | Result |
|---|---|
| `GET /api/health` | HTTP 200, `{"status":"ok"}` |
| `GET /api/model-config` | HTTP 200; `use_cloud_services=false`, Timeline role `qwen3.5:9b`, local catalog available, `qwen3.5:2b` and `qwen3.5:9b` installed |
| `GET /api/inspection/sessions/1/timelines` | HTTP 200; timeline 1, fallback, 3 events, 3 source-evidence events, `qwen3.5:2b`, `invalid_response` |
| `GET /api/inspection/sessions/1/timelines/1` | HTTP 200; therapy `2025-01`, symptom/lab `2025-01-17`, all `fallback_parser` evidence |
| `GET /api/inspection/sessions/3/timelines/2` | HTTP 200; timeline 2, `llm_generated`, 3 events, 3 source-evidence events, `qwen3.5:9b` |
| SQLite `PRAGMA integrity_check` | `ok` |
| SQLite `PRAGMA foreign_key_check` | no rows |

Persisted timeline summary:

```text
timeline=1 session=1 status=fallback error=invalid_response
source_kind=local model_provider=ollama source_model=qwen3.5:2b events=3 evidence_events=3

timeline=2 session=3 status=llm_generated error=None
source_kind=local model_provider=ollama source_model=qwen3.5:9b events=3 evidence_events=3
```

The source facts used by both synthetic sessions were the therapy month
`2025-01`, symptom day `2025-01-17`, and ALT day `2025-01-17`. No patient data
or credential value was used.
