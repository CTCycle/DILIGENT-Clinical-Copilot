# TypeScript Guidelines (DILIGENT Client)

Last updated: 2026-04-09

Scope: `DILIGENT/client` (Angular standalone + TypeScript 5).

Also mandatory: apply `assets/docs/ERROR_HANDLING.md`.

## 1. Type safety

- Keep strict TypeScript settings enabled.
- Prefer `unknown` over `any` for untrusted/external values.
- Narrow types via explicit guards before use.
- Keep shared API contracts in `src/app/core/models/types.ts` and reuse them consistently.

## 2. Structure and ownership

- Centralize network calls in `src/app/core/services/*-api.ts`, and keep shared HTTP transport/error handling in `src/app/core/services/http-api.ts`.
- Keep app-wide state in `src/app/core/state/app-state.service.ts`.
- Keep page-level orchestration in `src/app/pages/*`.
- Keep reusable UI controls in `src/app/components/*`.
- Keep shared constants/defaults in `src/app/core/constants.ts`.

## 3. Angular implementation rules

- Use standalone components by default.
- Use signals for local/global UI state and explicit cleanup for timers/listeners/object URLs/pollers.
- Keep form-heavy clinical workflows deterministic (template-driven or reactive forms are both acceptable).
- Keep side effects in lifecycle hooks or explicit service methods, never in template expressions.

## 4. API/runtime coherence

- Keep frontend API base path compatible with `/api`.
- When adding/modifying endpoints, update together:
  - backend API/domain models
  - impacted module service under `src/app/core/services/*-api.ts`
  - `src/app/core/models/types.ts`
  - impacted Angular pages/components
- Handle non-success responses explicitly; do not silently ignore failures.

## 5. Job and polling UX

- Disable conflicting actions during active operations (`isRunning`, `isPulling`, update jobs).
- Use shared terminal states: `completed`, `failed`, `cancelled`.
- Keep progress/result messaging stable and user-safe.

## 6. Accessibility and UI consistency

- Follow `assets/docs/UI_STANDARDS.md`.
- Preserve meaningful labels, keyboard reachability, and `aria-*` semantics.
- Do not rely on color alone for state communication.

## 7. Security

- Never expose plaintext provider keys in UI logs or persistent client state.
- Treat backend payloads as untrusted; normalize before rendering.

## 8. Testing expectations

- Validate frontend behavior through `tests/e2e`.
- Add/update E2E coverage when changing user-visible flows or API interactions.
- Prefer behavior assertions over implementation details.
