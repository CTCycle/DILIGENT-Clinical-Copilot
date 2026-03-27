# Error Handling Principles

All code must implement explicit, structured error handling that prioritizes reliability and safety.

---

## Exception Strategy

- Wrap risky operations in targeted `try/except` blocks.
- Catch specific exception types whenever possible.
- Do not broadly swallow exceptions except at top-level safety boundaries.

---

## Failure Handling

Provide safe handling and fallback behavior for:

- network failures
- missing files
- invalid inputs
- timeout conditions
- dependency failures
- malformed external responses

Retries are allowed only for transient failures and must include:

- retry limits
- backoff strategy
- clear stop conditions

---

## Timeouts

Always use explicit timeouts for:

- I/O operations
- network requests
- external service calls

Operations without timeouts are considered unsafe.

---

## Resource Safety

Always guarantee proper cleanup of resources:

- files
- sockets
- locks
- temporary data
- database connections

Prefer context managers or structured cleanup logic.

---

## Application Stability

Unhandled exceptions must never:

- reach the user interface
- crash the application process

Instead:

- return structured error states
- fail gracefully
- stop invalid states early using guard clauses and precondition checks

---

## User-Facing Error Messages

User-visible errors must:

- be short and calm
- use clear, plain language
- never expose technical details or internals

Messages should explain:

- what happened
- what the user can try next
- whether retrying may resolve the issue

When safe, provide partial results instead of failing completely.

---

## Logging and Diagnostics

On failure, log sufficient diagnostic context for debugging:

- error type
- relevant state
- correlation or request IDs when applicable

Never log:

- secrets
- authentication tokens
- sensitive personal data

Comments should only be added when they clarify:

- safeguards
- recovery paths
- non-obvious defensive logic

---

## Architectural Expectations

Error handling should be centralized at system boundaries.

System design should ensure:

- failures in one component do not crash the entire system
- operations degrade safely when dependencies fail
- safe defaults or fallback values exist where possible

Concurrency and asynchronous code must safely handle:

- cancellation
- timeouts
- race conditions
- shared-state protection

---

## Testing Requirements for Errors

Tests must explicitly cover failure scenarios, including:

- invalid inputs
- malformed data
- dependency failures
- retries and backoff behavior
- timeouts
- partial failures and fallback behavior

Tests must verify:

- the system does not crash under expected failures
- user-facing error messages remain clean and safe
- recovery and rollback behavior functions correctly
