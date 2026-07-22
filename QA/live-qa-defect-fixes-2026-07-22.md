# Live QA: confirmed-defect fixes

Date: 2026-07-22

## Environment

- Backend: `http://127.0.0.1:7690/api/health` responded after launcher startup.
- Frontend: `http://127.0.0.1:9847` was exercised in the Codex in-app browser.
- Existing OpenCode Go configuration remained selected with clinical and extraction model `deepseek-v4-flash`, and the UI retained its automatic clinical-reliability generation behavior.

## Access keys

- A synthetic placeholder OpenAI value was submitted directly to the local API.
- The API returned HTTP 422; no value was persisted or recorded in this artifact.
- The active OpenAI key identifier was unchanged before and after the request.
- The OpenAPI document marks `provider` as required for `GET /api/access-keys`.
- Browser validation confirmed the OpenAI key dialog states that newly saved keys remain inactive until explicitly activated.

## Timeline and configuration UI

- Browser validation showed `Live provider catalog loaded.` for OpenCode Go.
- `deepseek-v4-flash` remained selected for both clinical and text-extraction roles.
- Timeline unit validation confirms that a single explicit ISO date in preserved source evidence replaces a conflicting model date before chronological sorting.

## Regression checks

- `pytest app/tests/unit/test_revision_agent_skeleton.py app/tests/unit/test_access_keys.py app/tests/unit/test_patient_timeline_extraction.py -q`: 21 passed.
- Earlier focused run including data-inspection persistence: 26 passed.
- Angular production build passed. Existing stylesheet budget warnings remained in the Clinical Sessions and Model Configurations page stylesheets.
- Angular test suite: 38 passed. One stale revision-template assertion was aligned with the current `setRevisionModelProvider` handler before the successful rerun.
