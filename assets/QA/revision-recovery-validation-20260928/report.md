# Revision recovery validation

Last updated: 2026-09-28

Date: 2026-09-28
Revision: `e8ab2eeadf89247be9bb1f3fac3fff7e598f5f61` (`develop` and
`origin/develop`)

## Scope

This slice validates persisted revision recovery when the in-memory worker is
gone. It uses a disposable SQLite-backed serializer and a fresh `JobManager`
with no registered worker for a persisted `running` revision. The test checks
the status contract, persisted failure metadata, source-session preservation,
and retry completion with a deterministic test runner.

The fixture stores `ollama / qwen3.5:9b` as revision configuration metadata
only. It does not contact Ollama or any other provider, and it does not claim
fresh clinical or provider acceptance.

## Evidence

`test_missing_revision_worker_is_recoverable_and_retry_preserves_source` in
`app/tests/unit/test_revision_agent_skeleton.py` verified that:

1. A persisted running revision whose worker is absent returns `status=failed`
   with `result.recoverable=true`, the persisted pipeline/version identifiers,
   and the sanitized message: “Revision job worker is no longer available.
   Reload the persisted revision run and retry if needed.”
2. The persisted pipeline run changes to `failed` with the same sanitized
   message; no internal exception or provider detail is exposed.
3. The source session and its report remain unchanged before retry.
4. Retrying creates a new job/run, completes with `revision_status=llm_qa_passed`
   under the deterministic runner, and leaves the original source session
   unchanged.

The test uses a temporary SQLite database and pytest temporary paths. No
shared database, provider cache, application process, or credential state was
mutated.

## Results

- Focused RAG/revision suite: **44 passed**, 1 warning.
- Affected backend/API regression suite: **129 passed**, 3 dependency/deprecation
  warnings.
- Ruff lint on changed tests: passed.
- Pyright from `app/server`: **0 errors, 0 warnings, 0 informations**.
- `git diff --check`: passed; the repository's normal LF/CRLF conversion
  warnings remain on two existing test files.

## Final gate status

`revision.agentic-lifecycle` remains `PARTIAL`. The persisted missing-worker
failure, sanitized recovery status, source preservation, and deterministic retry
branch are now validated. Fresh provider execution, provider/model consistency,
timeout and tool-failure injection, true process-restart behavior, and every
other lifecycle branch remain outside this local slice.

`revision.accepted-session-finalization` remains validated only for its exact
previous synthetic accepted path. The OpenCode Go gate remains `PARTIAL` because
no local `OPENCODE_GO_API_KEY` or live provider request was available; the
LiverTox refresh/catalog gates remain `PARTIAL` behind the NCBI challenge;
access-key lifecycle remains `BLOCKED`; desktop `v3.4.0` release remains
`BLOCKED`; and spoken Narrator/Speech Recap output remains unvalidated.
