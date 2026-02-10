# Pandas 3.0 Migration Note

## Scope
- Target runtime: Python 3.14.x
- Target dependency: `pandas==3.0.0`
- Audited areas: repository, services, updater flows, scripts, and tests for pandas usage.

## What Changed
- Replaced shallow DataFrame ownership copies with explicit independent copies in:
  - `DILIGENT/server/services/clinical/livertox.py`
- Made numeric date parsing deterministic in:
  - `DILIGENT/server/repositories/serialization/data.py`
  - `normalize_date()` now uses explicit units for numeric epoch-like inputs (`s`, `ms`, `us`, `ns`) and explicit `%Y%m%d` parsing for 8-digit dates.

## Why
- Pandas 3 uses copy-on-write semantics by default. Explicit copies prevent accidental reliance on shared-memory behavior.
- Pandas 3 datetime resolution inference can differ for numeric values. Explicit units keep date normalization stable across versions.

## Validation Added
- New focused tests in `tests/unit/test_pandas_migration.py` cover:
  - String dtype handling during LiverTox master-list sanitization.
  - Copy isolation for internal DataFrame storage.
  - Deterministic numeric datetime normalization.
  - Ordering expectations in clinical-session append (`concat`) flow.

## Known Constraints
- No `value_counts(sort=False)` or `pd.offsets.Day` usages were found in application code paths.
- No `dtype == object` string-detection logic was found.
- Full end-to-end test execution still depends on external runtime prerequisites (browser/server orchestration) as documented in `tests/run_tests.bat`.
