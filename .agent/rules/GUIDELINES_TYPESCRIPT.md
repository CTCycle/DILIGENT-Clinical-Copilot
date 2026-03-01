## TypeScript Guidelines (DILIGENT Client)

These rules apply to `DILIGENT/client` (React 18 + Vite + TypeScript 5).

## 1. Type Safety

- Keep `tsconfig` strictness enabled (`"strict": true`).
- Prefer `unknown` over `any` for external or loosely typed values.
- Narrow types explicitly with guards before use.
- Model API payloads/responses in `src/types.ts` and reuse them across pages/components/services.

## 2. Project Structure

- Keep data fetching in `src/services/api.ts`; avoid ad-hoc `fetch` in UI components.
- Keep shared runtime/form/app state in `src/context/AppStateContext.tsx`.
- Keep route/page-level orchestration in `src/pages/*`.
- Keep dumb UI pieces in `src/components/*`.
- Keep constants and defaults in `src/constants.ts`.

## 3. React Patterns

- Use function components and explicit prop interfaces.
- Use controlled inputs for clinical forms.
- Clean up side effects (`useEffect` cleanup for timers, listeners, object URLs, pollers).
- Avoid duplicating derived values; use `useMemo` where serialization/filtering is non-trivial.

## 4. API and Runtime Contracts

- Respect `API_BASE_URL` contract and keep frontend API pathing `/api` compatible.
- When adding endpoints, update:
  - backend route + entity models
  - `src/services/api.ts`
  - frontend `src/types.ts` contracts
- Handle non-200 responses with clear `[ERROR]` messages; never silently swallow API failures.

## 5. UX and Resilience

- Disable conflicting actions during long-running operations (`isRunning`, `isSaving`, `isPulling`).
- Treat polling terminal states consistently: `completed`, `failed`, `cancelled`.
- Preserve accessibility basics: labels, button roles, and meaningful `aria-*` attributes.

## 6. Testing Expectations

- Frontend behavior is validated through Playwright E2E tests in `tests/e2e`.
- When adding UI flows or API interactions, add or update corresponding E2E coverage.
- Prefer behavior assertions over internal implementation details.

## 7. Security

- Never expose plaintext access keys in UI state, logs, or responses.
- Treat all backend payloads as untrusted and validate/normalize before rendering.
