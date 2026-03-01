## WEB SEARCH
Use web search to verify facts and stay current on tools, frameworks, and industry standards when it improves accuracy.

## REQUIRED DOCUMENTATION REVIEW
Before significant implementation work, review the relevant files in `.agent/rules`:

- `GENERAL_RULES.md`, mandatory for every task
- `GUIDELINES_PYTHON.md`, when using Python
- `GUIDELINES_TYPESCRIPT.md`, when using TypeScript
- `GUIDELINES_TESTS.md`, when writing tests
- `ARCHITECTURE.md`, system structure and APIs
- `BACKGROUND_JOBS.md`, background job management
- `README_WRITING.md`, required README structure and standards

## DOCUMENTATION UPDATES
If changes materially affect behavior, architecture, runtime, or usage, update the relevant `.agent/rules` files and notify the user.

## CROSS-LANGUAGE PRINCIPLES

### Code quality
- Prefer consistent style, clear naming, and small single-purpose components.
- Optimize for readability, testability, and low coupling.

### Testing and automation
- Enforce CI checks: formatting, linting, type checks, tests, and security scans.

### Security
- Apply standard secure coding practices: input validation, correct auth handling, secret protection, minimal attack surface.

## EXECUTION RULES
- On Windows, prefer PowerShell for local commands.
- Use `cmd /c` when invoking `.bat` workflows or when command semantics require `cmd`.

## FILE CHANGE NOTICE
- Any significant change requires updating relevant `.agent/rules` docs and informing the user.



