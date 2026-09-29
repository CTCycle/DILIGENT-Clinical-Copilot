# Runtime Settings UI
Last updated: 2026-09-29

## Scope

DILIGENT exposes one centralized Settings area for all operator-editable
configuration. Non-`.env` values are persisted in the singleton
`application_configuration` database row and are applied to newly created
runtime work. In-flight jobs keep the configuration captured when they were
created.

The UI does not expose environment variables, credentials, deployment
configuration, database connection settings, hosts, ports, resource paths,
Ollama endpoints, or other `.env`-owned values. Immutable reference catalogs
under `resources/catalogs/` remain source-controlled reference data.

The Settings routes are:

- `/settings/general`
- `/settings/models`
- `/settings/data`
- `/settings/integrations`
- `/settings/matching`
- `/settings/advanced`

The legacy `/model-config` route redirects to `/settings/models`.

## Source of truth and startup order

1. Environment-only settings are loaded from `.env` and the process environment.
2. Database migrations run and seed a complete typed default payload when the
   singleton is missing. Existing database model selections are preserved while
   missing operational defaults are added.
3. The persisted singleton is loaded into the in-process `ConfigurationManager`
   and validated before providers, jobs, and startup validation run.
4. Settings API and Models page writes update the same singleton document and
   refresh the manager for newly created work.

The former standalone settings file is not read, written, copied into desktop
archives, or required for startup. Its values are not imported; fresh defaults
are defined by typed application models.

## Settings categories

| Category | Ownership | Controls |
| --- | --- | --- |
| General | Jobs | Polling interval |
| Models | Application configuration | Provider, access-key workflow, model roles, reasoning, and the complete RAG payload |
| Data Processing | Ingestion, session pipeline, language detection | Drug-name limits, batch/concurrency limits, and clinical language thresholds |
| Integrations | Runtime integrations | NCBI/LiverTox contact and download/archive/worker controls, plus RxNav timeout/concurrency |
| Drug Matching | Matcher runtime | Confidence thresholds, cache sizes, catalog index size, token limits, and spelling distances |
| Advanced | LLM/evidence runtime | Role-specific timeouts, timeout floors/caps, Ollama startup timeout, and excerpt length |

RAG is represented once as `rag_settings` in the database payload. The Models
page and retrieval/settings services use that same typed payload; there is no
JSON base layer or second RAG override store.

## API

The typed operational API is:

- `GET /api/settings`: returns all five operational categories, typed defaults,
  database source metadata, and the singleton timestamp.
- `PATCH /api/settings`: applies partial category updates after schema and
  cross-field validation.
- `POST /api/settings/reset/{category}`: resets one operational category to
  typed defaults and persists the result.

Unknown fields and environment-owned fields are rejected. The API never
returns secrets or `.env` values. Model/provider/RAG updates use the existing
model API, which writes the same `application_configuration` row.

## Validation and persistence

Controls are typed numeric, boolean, email, text, or select inputs. Backend models
enforce bounds and relationships such as drug-name minimum/maximum lengths,
language confidence ordering, selected retrieval documents not exceeding
candidate documents, and timeout caps not falling below the minimum timeout.

The Integrations category includes `ncbi_contact_email`. The value is trimmed
and validated as a basic developer email address, then persisted in the
`application_configuration` payload with the other runtime settings. Older
payloads and blank values resolve to the compatibility default
`clinical-copilot@pharmagent.local`, which is also restored by an Integrations
reset. The UI shows the effective value and recommends a real developer or
organization address registered with NCBI for production use; this field is a
contact identity, not an API key or credential.

Every successful update returns the database `updated_at` timestamp. Reset and
reload use the same singleton, so persistence is observable after navigation
and application reload. No application restart is required for newly created
work.
