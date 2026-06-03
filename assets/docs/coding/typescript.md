# TypeScript Rules
Last updated: 2026-06-03

## Type Safety And Contracts
- Keep strict typing and avoid `any` for untrusted inputs.
- Centralize shared API contracts in `app/client/src/app/core/models/types.ts`.
- Normalize backend payloads before rendering.

## Frontend Architecture
- Keep HTTP transport and error normalization in `app/client/src/app/core/services/http-api.ts`.
- Keep domain API calls in `app/client/src/app/core/services/*-api.ts`.
- Keep page orchestration in `app/client/src/app/pages/*`.
- Keep reusable UI controls in `app/client/src/app/components/*`.
- Keep shared app state in `app/client/src/app/core/state/app-state.service.ts`.

## Interaction And UX Behavior
- Preserve deterministic job state transitions:
  - `pending`
  - `running`
  - `completed`
  - `failed`
  - `cancelled`
- Disable conflicting actions during active operations.
- Preserve keyboard accessibility and ARIA semantics.

## Tooling And Verification
- Use project scripts in `app/client/package.json`.
- Run frontend build validation after frontend code changes:

```powershell
npm run build
```
