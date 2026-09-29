# Session and revision API local-boundary validation

Date: 2026-09-29
Scope: current session inspection and revision HTTP contracts on the
disposable deterministic browser runtime

## Outcome

The selected `api.local-boundaries` slice is validated for the session
inspection and revision route groups exercised below. The implementation had
one real boundary defect: a missing session sent to
`PUT /api/inspection/sessions/{session_id}/report` returned a sanitized `500`
instead of `404`. The repository guard now checks session existence before
requiring a persisted version, and the public route returns the expected
not-found response. Existing sessions without a persisted version still retain
the explicit runtime error because they cannot accept a manual report audit.

The complete 68-path response/error and mutating-route matrix remains outside
this slice. RAG duplicate-file policy, broader provider/model behavior, spoken
screen-reader output, desktop delivery, and container runtime status were not
promoted by this run.

## Live HTTP evidence

The canonical Windows `Full` harness used a task-owned SQLite database seeded
with one synthetic session and one persisted timeline. It ran the existing
deterministic browser suite together with `test_api_local_boundaries.py`.

| Boundary | Result |
|---|---|
| Session list/detail/version/manual-edit reads | PASS: seeded session was located and typed records were returned. |
| Session metadata mutation | PASS: `PUT /sessions/{id}` returned the persisted session without changing the report. |
| Manual report audit mutation | PASS: report update returned the nested session/audit payload, distinct text hashes, persisted reviewer note, and cleanup restored the original report. |
| Session/timeline missing-record errors | PASS: missing session, report, timeline, and timeline-job routes returned `404` with a safe detail. |
| Revision validation and missing-record errors | PASS: invalid start returned `422`; missing session, job, pipeline run, retry, steps, artifacts, entities, reviews, and clinical-review target returned `404`. No provider request was started. |
| First live attempt | FOUND: 47 passed and the missing-session report mutation returned `500`; this reproduced the defect fixed in the current source. |
| Final live attempt | PASS: **48 passed, 0 failed, 0 skipped**. |

## Focused regression and quality evidence

- Focused repository/API/revision/settings/access-key suites after the fix:
  **113 passed**, 3 existing dependency/deprecation warnings.
- The new repository regression asserts that a missing session returns `None`
  from `update_current_report_text_with_manual_audit`.
- Changed-file Ruff checks: PASS.
- Full server Pyright: **0 errors, 0 warnings, 0 informations**.
- `git diff --check`: PASS.
- The harness stopped its backend, frontend, and fake-Ollama process trees;
  ports `7690`, `9847`, and `11435` were free after the run.

## Final gate disposition

| Gate | Final status | Evidence boundary |
|---|---|---|
| `api.local-boundaries` | `WORKING` for the exercised session/revision slice | Live success, mutation, validation, and not-found contracts passed. The complete 68-path response/error and mutating-route matrix remains open. |
| `sessions.crud-persistence` | `VALIDATED` for the exercised session update/manual-audit boundary | Metadata and manual report persistence/reload were checked on the disposable seeded session; binary image storage and Chrome-specific file-picker behavior remain outside scope. |
| `revision.agentic-lifecycle` | `VALIDATED` for its existing exact-provider synthetic scope; API error boundary rechecked here | This run did not start a provider-backed revision and does not expand provider/model or lifecycle permutation coverage. |
| `test.automated-regression` | `VALIDATED` for the local deterministic Full suite | The current local Full suite is green with zero skips; hosted exact-commit CI remains the authoritative PostgreSQL/Windows/security confirmation after push. |

## Remaining limitations

`ISSUE-005` remains open pending a product decision on byte-identical RAG
deduplication or canonical citation policy. The broader route matrix, wider
provider/model and fault matrix, spoken Narrator/Speech Recap output,
desktop `v3.4.0` delivery, and `runtime.containerized` remain independent
follow-ups.
