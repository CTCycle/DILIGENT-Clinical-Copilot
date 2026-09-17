# Runtime Settings UI
Last updated: 2026-09-17

## Scope
DILIGENT exposes a centralized Settings area for configuration that is already represented in `settings/configurations.json` and can be applied safely to new runtime work. The UI does not expose environment variables, credentials, deployment configuration, database configuration, ports, filesystem/bootstrap paths, or other `.env` values.

The Settings routes are:

- `/settings/general`
- `/settings/models`
- `/settings/data`
- `/settings/integrations`
- `/settings/advanced`

The legacy `/model-config` route redirects to `/settings/models`.

## Source of Truth and Precedence
1. `.env` and process environment variables remain startup, deployment, database, and host configuration. They are not readable or writable through the Settings API.
2. `settings/configurations.json` is the persisted source of truth for the operational settings listed below. Successful updates are written atomically and the in-process `ConfigurationManager` snapshot is reloaded immediately.
3. Model provider, model role, reasoning, and persisted RAG overrides remain owned by the existing model-configuration persistence and API. The Models Settings page reuses that implementation directly.
4. For RAG, `settings/configurations.json` remains the base configuration while persisted model-configuration RAG values override the corresponding runtime-adjustable RAG fields.
5. Code defaults are fallback values when configuration is absent. Reset-to-default in the Settings API restores the checked-in defaults for the exposed operational settings.

Packaged desktop bootstrap behavior is unchanged. Existing mutable runtime settings are preserved rather than overwritten on every launch.

## Runtime-Managed JSON Settings

### General
| JSON field | UI control | Runtime behavior |
| --- | --- | --- |
| `jobs.polling_interval` | bounded numeric input | Applied to newly created job polling responses. |

### Models
`/settings/models` renders the existing `ModelConfigPageComponent` and keeps its existing services, validation, provider/model discovery, access-key workflow, catalog cache behavior, persistence, reasoning controls, model role assignments, and RAG controls. No second model-configuration form or persistence path exists.

### Data Processing
| JSON field | UI control | Runtime behavior |
| --- | --- | --- |
| `ingestion.drug_name_min_length` | bounded integer input | Applied by runtime drug-name validation. |
| `ingestion.drug_name_max_length` | bounded integer input | Applied by runtime drug-name validation. |
| `ingestion.drug_name_max_tokens` | bounded integer input | Applied by runtime drug-name validation. |

### Integrations
| JSON field | UI control | Runtime behavior |
| --- | --- | --- |
| `runtime.livertox_download_timeout` | bounded numeric input | Applied to new LiverTox download requests. |
| `runtime.rxnav_request_timeout` | bounded numeric input | Applied to new RxNav clients and update jobs. |
| `runtime.rxnav_max_concurrency` | bounded integer input | Applied to new RxNav clients and update jobs. |

### Advanced
| JSON field | UI control | Runtime behavior |
| --- | --- | --- |
| `runtime.default_llm_timeout` | bounded numeric input | Applied when new model clients resolve the default runtime timeout. |
| `runtime.clinical_llm_timeout` | bounded numeric input | Applied to new clinical reasoning services. |
| `runtime.livertox_llm_timeout` | bounded numeric input | Applied to new LiverTox-assisted extraction work. |
| `runtime.minimum_llm_timeout` | bounded numeric input | Applied when runtime model timeout budgets are resolved. |
| `runtime.cloud_llm_timeout_cap` | bounded numeric input | Applied to new cloud-model timeout resolution. |
| `runtime.local_llm_timeout_cap` | bounded numeric input | Applied to new local-model timeout resolution. |
| `runtime.max_excerpt_length` | bounded integer input | Applied to newly created clinical evidence-processing services. |

All exposed operational settings apply to new work after a successful save. No exposed field requires an application restart.

## JSON Settings Deliberately Not Exposed
The following configuration remains in `settings/configurations.json` because it is static, cached at import/startup, unused by a runtime consumer, security-sensitive in effect, or already owned by Model Configuration:

- `rag.allow_local_filesystem_access`
- `rag.vector_collection_name`
- `rag.hybrid_vector_weight`, `rag.hybrid_text_weight`, `rag.embedding_offline_mode` as JSON base values because the existing Models page already owns persisted runtime RAG overrides for these fields
- `rag.vector_index_metric`, `rag.vector_index_type`
- `runtime.parser_llm_timeout`, `runtime.disease_llm_timeout`, `runtime.ollama_server_start_timeout`
- `runtime.livertox_archive`
- `runtime.livertox_yield_interval`, `runtime.livertox_skip_deterministic_ratio`, `runtime.livertox_monograph_max_workers` because no runtime consumer currently uses these loaded values
- `session_pipeline.*` because the current keys have no active runtime consumer
- `clinical_language_detection.*` because the detector reads and caches these thresholds independently
- `drugs_matcher.*` because matcher constants and related caches are initialized outside the new runtime update path

These fields remain available to the existing configuration loader and are preserved verbatim by Settings writes.

## API
The operational settings API is typed and narrow:

- `GET /api/settings`: returns only the supported operational settings, defaults, source metadata, and `environment_editable: false`.
- `PATCH /api/settings`: applies partial category updates after schema and cross-field validation.
- `POST /api/settings/reset/{category}`: resets one operational category to checked-in defaults.

Unknown fields are rejected. Persistence errors use the standard application error contract. The API never returns `.env` values.

## Persistence Safety
Operational updates merge only the supported fields into the existing JSON document. Unmanaged JSON sections and keys are preserved. The file is written to a temporary sibling, flushed, and atomically replaced before the cached runtime configuration is reloaded.
