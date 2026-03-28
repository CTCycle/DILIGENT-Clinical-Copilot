# TypeScript Guidelines (DILIGENT Client)

Scope: `DILIGENT/client` (React 18 + Vite + TypeScript 5).

Also mandatory: apply `assets/docs/ERROR_HANDLING.md`.

## 1. Type safety

- Keep strict TypeScript settings enabled.
- Prefer `unknown` over `any` for untrusted/external values.
- Narrow types via explicit guards before use.
- Keep shared API contracts in `src/types.ts` and reuse them consistently.

## 2. Structure and ownership

- Centralize network calls in `src/services/api.ts`.
- Keep app-wide state in `src/context/AppStateContext.tsx`.
- Keep page-level orchestration in `src/pages/*`.
- Keep reusable UI controls in `src/components/*`.
- Keep shared constants/defaults in `src/constants.ts`.

## 3. React implementation rules

- Use function components with explicit prop interfaces.
- Use controlled inputs for form-heavy clinical workflows.
- Clean up all side effects (`useEffect` timers, listeners, object URLs, polling loops).
- Memoize only non-trivial derived values.

## 4. API/runtime coherence

- Keep frontend API base path compatible with `/api`.
- When adding/modifying endpoints, update together:
  - backend API/domain models
  - `src/services/api.ts`
  - `src/types.ts`
  - impacted UI pages/components
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
