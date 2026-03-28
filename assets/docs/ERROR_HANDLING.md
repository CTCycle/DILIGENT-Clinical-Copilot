# Error Handling Principles

Apply these rules across backend and frontend code.

## 1. Exception boundaries

- Handle failures at clear boundaries (API handlers, integration clients, background runners).
- Catch specific exception types where possible.
- Avoid broad `except Exception` unless converting to a safe boundary response.
- Never silently swallow exceptions.

## 2. Required failure classes

Every external interaction must consider:
- invalid input or schema mismatch
- timeout
- network/dependency unavailability
- malformed upstream responses
- missing files/resources

Retries are allowed only for transient failures and must define:
- max attempts
- backoff policy
- hard stop condition

## 3. Timeouts and cancellation

- Use explicit timeouts for all I/O and network calls.
- In async/threaded workflows, propagate cancellation signals and stop safely.
- Long-running jobs must be cancellable at safe checkpoints.

## 4. User-safe failures

- User-facing errors must be concise and non-technical.
- Never expose stack traces, secrets, access keys, or internal infrastructure details.
- Prefer degraded/partial output over total failure when safe.

## 5. Logging and diagnostics

Log enough context to debug:
- error class
- operation/context name
- request/job correlation ID when available

Never log:
- secrets or tokens
- raw provider keys
- sensitive personal data

## 6. Resource cleanup

Always clean up resources on success and failure:
- files and temp artifacts
- DB/network handles
- thread/async control state

Prefer context managers and explicit finalization paths.

## 7. Test expectations

Tests must cover failure paths for:
- validation errors
- dependency failures
- timeout and retry behavior
- cancellation behavior (for background jobs)
- safe user-visible error messaging

Expected result: failure does not crash the app and does not leak internal details.
