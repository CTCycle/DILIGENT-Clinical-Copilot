# Runtime Settings UI
Last updated: 2026-10-29

## Scope

DILIGENT exposes one centralized Settings area for operator-managed
configuration. Database selection and connection parameters are stored in
`<data-root>/database_config.json`. All other UI-managed settings are persisted
in the active database's `application_configuration` singleton and are applied
to newly created runtime work. In-flight jobs keep the configuration captured
when they were created.

The UI does not expose process environment variables, deployment configuration,
data-root selection, application API URLs, bind hosts or ports, CORS origins,
Ollama endpoints, or other process-owned values. These source-mode network
values are read from `settings/.env` and are not stored in the
`application_configuration` singleton. The
Database page exposes database connection credentials and options; a saved
PostgreSQL password is protected with Windows DPAPI and never returned by the
API. Immutable reference catalogs under `data/catalogs/` remain
source-controlled reference data.

The Settings routes are:

- `/settings/general`
- `/settings/models`
- `/settings/data`
- `/settings/integrations`
- `/settings/matching`
- `/settings/advanced`
- `/settings/database`

## Source of truth and startup order

1. Process and deployment settings are loaded from `.env` and the process
   environment. Database connection settings are loaded from the independent,
   versioned `database_config.json` file; `.env` and process environment
   variables never configure database access, and a missing configuration file
   is created with the default embedded SQLite configuration.
2. The selected database is connected and migrated. A new application schema
   or empty reference catalog is seeded before startup validation.
3. Database migrations seed a complete typed default payload when the
   singleton is missing. Existing database model selections are preserved while
   missing operational defaults are added.
4. The persisted singleton is loaded into the in-process `ConfigurationManager`
   and validated before providers, jobs, and startup validation run.
5. Settings API and Models page writes update the same singleton document and
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
| Integrations | Runtime integrations | NCBI/LiverTox contact and download/archive/worker controls, RxNav request timeout, terminology download timeout, and the DrugCentral (connection), ChEMBL (clinical phase), and Swissmedic source switches |
| Drug Matching | Terminology resolver | Spelling-correction length, short-name threshold, edit bounds, runner-up margin, shortlist size, resolution cache size, and the RxNav fallback switch, per-name request budget, and negative-result lifetime |
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
- `GET /api/database/configuration`: returns the selected backend and
  non-secret connection settings.
- `POST /api/database/test`: tests reachability and reports whether the
  application schema is ready without activating the candidate.
- `PUT /api/database/configuration`: initializes and verifies a target, then
  switches the live runtime.

Unknown fields and environment-owned fields are rejected. The API never
returns secrets or `.env` values. Model/provider/RAG updates use the existing
model API, which writes the same `application_configuration` row.

## Validation and persistence

Controls are typed numeric, boolean, email, text, or select inputs. Backend models
enforce bounds and relationships such as drug-name minimum/maximum lengths,
language confidence ordering, selected retrieval documents not exceeding
candidate documents, and timeout caps not falling below the minimum timeout.

Valid edits in operational Settings categories are persisted automatically
400 ms after the latest change. Cross-field or field-level validation errors
keep the current draft unsaved until corrected. The UI reflects the draft
immediately and shows inline save status; a failed write leaves the unsaved
draft visible with an error. The Models page also saves valid provider, model,
reasoning, and RAG changes automatically. Provider/model changes that do not
yet assign all required roles remain pending with guidance in the page. Reset
to defaults persists the selected operational category immediately.

The browser serializes settings writes in invocation order. Operational updates
and model/RAG updates merge into the latest singleton payload so concurrent
changes preserve unrelated settings. Successful writes refresh the in-process
configuration used for newly created runtime work; in-flight jobs keep their
captured configuration. Pending input debounce is flushed when leaving a
settings section or closing the Models page.

The Integrations category includes `ncbi_contact_email`. The value is trimmed
and validated as a basic email address, then persisted in the
`application_configuration` payload with the other runtime settings. It may be
blank, and an Integrations reset clears it. LiverTox and combined structured
source updates send it as NCBI contact identity when it is a real address;
blank and `.local` values fall back to tool-name-only identification. A saved
`.local` value remains visible until the user replaces or clears it. The field
is a contact identity, not an API key or credential.

Every successful update returns the database `updated_at` timestamp. Reset and
reload use the same singleton, so persistence is observable after navigation
and application reload. No application restart is required for newly created
work.

## Database settings and switching

The Database page supports SQLite and PostgreSQL through `psycopg`. It exposes
the SQLite file path, or PostgreSQL host, port, database, username, password,
SSL requirement, and optional CA file. Connection timeout and batch sizes are
in the advanced options. Test connection reports reachability and schema
readiness without changing the active backend.

Switching is coordinated by the backend. It rejects new API work temporarily,
waits up to 60 seconds for existing API requests to finish, and blocks the
switch while any background job is pending or running. The target is
initialized and schema-checked before the configuration file is replaced.
Activation failure restores the previous file, repositories, settings, and
services. Application settings are then loaded from the selected database;
records are never copied between databases. The Settings page reloads after a
successful switch so frontend service state matches the active backend.

If the target PostgreSQL database does not exist, Save and switch attempts to
create it; the configured PostgreSQL role must have `CREATEDB` permission. The
separate Test connection action requires the database to exist already. A
restart is still needed when changing process-level startup settings such as
the data root or packaged runtime. Restart after an activation error only if
the UI reports that automatic recovery could not restore services.

Database edits remain a local draft. Testing a connection does not persist or
activate that draft. Save and switch validates and initializes the target
before updating `database_config.json`; failed activation restores the previous
file and active runtime services.
