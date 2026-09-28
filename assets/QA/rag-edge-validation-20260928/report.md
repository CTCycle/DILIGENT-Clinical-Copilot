# RAG edge validation and gate triage

Last updated: 2026-09-28

Date: 2026-09-28
Revision: `ed0b59f8401adb3139b6973650d9a61c61143042` (`develop` and
`origin/develop`)
Scope: current RAG source-selection, unsupported-file, arbitrary-folder, and
duplicate-content boundaries, plus a read-only recheck of the external gates
that determine the next validation work.

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

Focused commands and results:

| Check | Result |
|---|---|
| `python -m pytest` on document serialization, inspection security, RAG contract/readiness/provenance/settings/preflight, session-update, and data-inspection repository tests | `77 passed`, 3 existing dependency/deprecation warnings |
| Ruff on changed Python source/tests | Passed |
| Pyright on changed backend source from `app/server` | `0 errors, 0 warnings, 0 informations` |

No embedding cache, vector store, provider, or long-running application process
was changed by this slice. The prior 99-file / 1,317-chunk rendered RAG run
continues to supply the main ingestion/retrieval evidence; this run closes the
previously untested source-inventory boundary only.

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
recursive arbitrary-folder inventory and unsupported-file visibility. Its
remaining validation debt is the duplicate-file policy and any broader
unsupported-folder/error matrix not exercised here. `ISSUE-005` remains open and
unchanged in severity because resolving it requires a product decision, not a
safe inferred implementation choice.

The provider/revision regression gates remain `PARTIAL` because no fresh live
provider request ran in this environment. The NCBI source-refresh gate remains
`PARTIAL` for the current upstream challenge, access-key management remains
`BLOCKED` for missing approved credential material, and desktop release remains
`BLOCKED` for its independent release prerequisites.
