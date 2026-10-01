# RAG edge validation and gate triage

Last updated: 2026-09-28

Date: 2026-09-28
Revision: `e8ab2eeadf89247be9bb1f3fac3fff7e598f5f61` (`develop` and
`origin/develop`)
Scope: current RAG source-selection, unsupported-file, arbitrary-folder, and
duplicate-content boundaries, error contracts for empty/malformed/missing and
non-directory inputs, plus a read-only recheck of the external gates that
determine the next validation work.

## Slice selection

The canonical ledger was current through 2026-09-27. The next locally actionable
debt was the RAG edge boundary: the inspection table exposed a Supported column,
but its backend enumerated only ingestible files, and legacy `.doc` files were
counted as supported even though the loader skipped them. The duplicate-content
policy remains a separate product decision and was rechecked without changing
its semantics.

The OpenCode Go provider, access-key lifecycle, NCBI-dependent source refresh,
spoken screen-reader output, and desktop publication gates were not silently
folded into this code slice. Their prerequisites were checked and their
documented partial/blocked boundaries were retained.

## Current implementation changes

- `DocumentSerializer.collect_file_paths()` now enumerates every file under the
  selected folder for inspection.
- `collect_document_paths()` filters that inventory to the formats with active
  loaders: `.pdf`, `.txt`, `.xml`, and `.docx`.
- Legacy binary `.doc` is no longer advertised as ingestible. It remains
  visible in the inspection listing with `supported_for_ingestion=false`.
- The Data Inspection RAG listing now uses the all-file inventory, so users can
  see unsupported files instead of receiving a misleading incomplete inventory.
- Byte-identical files in different relative paths still receive distinct
  path-based document IDs. This preserves current provenance behavior while
  leaving `ISSUE-005` open for a future deduplication or canonical-citation
  decision.

## Validation evidence

The temporary fixtures covered a nested folder containing an uppercase `.TXT`,
a legacy `.doc`, and an unrelated `.png`. They verified that:

- all three files are visible to inspection;
- only `.TXT` is returned by the embedding document inventory;
- `.doc` is explicitly marked unsupported;
- the supported text file still loads and preserves metadata;
- two byte-identical `guide.txt` files in different folders remain two source
  paths with two distinct document IDs; and
- the inspection service returns both supported and unsupported rows with the
  correct flag.

## Error-boundary follow-up — 2026-09-28

The current implementation was revalidated with isolated fixtures for empty
folders, unsupported-only folders (`.doc` and `.png`), an empty supported text
file, a malformed `.docx`, a missing path, and a path that resolves to a file
instead of a directory. The follow-up covered both repository behavior and
the affected inspection/update contracts:

- empty and unsupported-only inventories produce no ingestible documents;
- empty and malformed supported files are ignored by document loading rather
  than creating empty or invalid chunks;
- zero supported files and zero produced chunks fail the RAG update before the
  existing manifest/vector state can be replaced;
- a successful controlled update writes the manifest only after chunks exist;
- missing and non-directory roots fail during updater preflight before vector
  database setup; and
- browse/listing error responses retain the existing sanitized 403/404/422
  contracts while unsupported files remain visible with an explicit false
  `supported_for_ingestion` flag.

The revision recovery test is recorded separately in the
[revision restart/tool-failure validation](../revision-restart-recovery-validation-20260928/report.md)
and is not counted as live provider or vector-store execution.

Focused commands and results:

| Check | Result |
|---|---|
| Focused RAG/revision regression (`test_document_chunking.py`, `test_rag_update_jobs.py`, `test_revision_agent_skeleton.py`) | `44 passed`, 1 warning |
| Affected backend/API regression (RAG, inspection, session-update, data-inspection, clinical/API contracts, and revision tests) | `129 passed`, 3 dependency/deprecation warnings |
| Ruff lint on changed Python tests | Passed |
| Pyright from `app/server` using the project configuration | `0 errors, 0 warnings, 0 informations` |
| `git diff --check` | Passed; only normal repository line-ending conversion warnings were reported for two existing test files |

No embedding cache, vector store, provider, or long-running application process
was changed by this slice. The prior 99-file / 1,317-chunk rendered RAG run
continues to supply the main ingestion/retrieval evidence; this follow-up
extends the validated boundary to local error handling without claiming a
fresh provider-backed embedding run. No frontend files changed, so the
canonical Angular suite was not rerun for this backend/test-only slice.

## External gate rechecks

- `OPENCODE_GO_API_KEY` was absent from the local process. `gh secret list
  --repo CTCycle/DILIGENT-Clinical-Copilot` completed successfully but returned
  no repository secret names. No live provider request was claimed.
- Both current NCBI Bookshelf master-list URLs returned HTTP 200 HTML containing
  `recaptcha` and `challengepage` markers rather than a machine-readable Excel
  response. `data.sources.refresh` therefore remains `PARTIAL`, with the
  external human-verification blocker open.
- No disposable credential was available or authorized, so
  `auth.access-key-management` remains `BLOCKED`.
- The signed/tagged/hosted/clean-machine publication workflow was not attempted;
  `release.desktop.v3-4-0` remains `BLOCKED`.
- The existing keyboard/rendered accessibility evidence does not establish
  spoken Narrator/Speech Recap output; no accessibility status was promoted.

## Final status

`rag.ingestion-retrieval` remains `VALIDATED` for its stated scope, now including
recursive arbitrary-folder inventory, unsupported-file visibility, and the
tested empty/malformed/missing/non-directory and fail-closed update boundaries.
Filesystem permission failures, broader provider-backed vector failures, and
the duplicate-file policy remain outside this local slice. `ISSUE-005` remains
open and unchanged in severity because resolving it requires a product
decision, not a safe inferred implementation choice.

The provider/revision regression gates remain `PARTIAL` because no fresh live
provider request ran in this environment. The NCBI source-refresh gate remains
`PARTIAL` for the current upstream challenge, access-key management remains
`BLOCKED` for missing approved credential material, and desktop release remains
`BLOCKED` for its independent release prerequisites.
