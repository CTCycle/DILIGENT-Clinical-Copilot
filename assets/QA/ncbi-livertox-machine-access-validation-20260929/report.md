# NCBI LiverTox machine-access validation

Date: 2026-09-29
Repository: `CTCycle/DILIGENT-Clinical-Copilot`
Branch: `develop`
Scope: official NCBI discovery replacement, Settings contact persistence, atomic source validation, and local UI rendering

## Outcome

The implementation replaces automated Bookshelf HTML discovery with official
NCBI machine-access services:

- E-utilities ESearch/ESummary identify the master-list record by exact
  Bookshelf RID and send the fixed tool identity plus the effective Settings
  contact.
- Books-OAI uses the stable master-list identifier and validates candidate
  spreadsheet links against the expected HTTPS host, accession path, and
  `masterlist*.xlsx` filename pattern.
- LitArch `file_list.csv` resolves the LiverTox archive dynamically instead of
  assuming a two-level directory.
- If OAI does not expose a usable spreadsheet, a safe tar-member fallback
  extracts exactly one `masterlist*.xlsx` member.
- XLSX and tarball downloads use `.part` files and replace the last-good file
  only after format, content, and safe-member validation.

The contact field is a non-secret database-backed setting. Missing and blank
legacy values resolve to the bundled compatibility fallback; the report does
not reproduce the address value.

## Automated validation

| Check | Result |
|---|---|
| Focused updater/settings/timeout tests | **33 passed** |
| Full backend unit suite | **832 passed** |
| Ruff (`app/server` and `app/tests`) | **Passed** |
| Pyright | **0 errors, 0 warnings, 0 informations** |
| Angular suite | **24 files / 105 tests passed** |
| Production frontend build | **Passed** with `npm run build` |
| OpenAPI synchronization | **68 paths / 104 schemas**, JSON parses successfully; both Integration schemas contain the new field |

The focused updater tests cover E-utilities query identity, exact RID
selection, OAI identifier and link safety, LitArch accession/path validation,
archive-member ambiguity/traversal, OAI-to-LitArch fallback, partial-download
preservation, unchanged metadata optimization, and contact redaction from
exceptions.

## Live NCBI metadata preflight

Read-only calls to the official services completed successfully:

1. ESearch and ESummary identified the expected Bookshelf record by RID.
2. Books-OAI exposed `nbk_ftext` for the expected stable identifier.
3. The LitArch file list resolved the archive path dynamically.
4. A metadata request reported an archive size of 195,497,807 bytes.

The current OAI spreadsheet candidate returned HTML challenge content rather
than a usable XLSX payload. The implementation therefore follows the intended
LitArch fallback path; the large live archive was not downloaded during this
validation run.

## Browser/UI evidence

An isolated disposable SQLite backend was started with the in-app Browser at
1280×720. The Integrations screen visibly rendered:

- the NCBI developer contact field and `NCBI / LiverTox` scope;
- the effective compatibility default, shown only in the local UI and not
  copied into this report;
- `type=email` and `autocomplete=email` semantics;
- production-use guidance recommending a real registered developer or
  organization contact; and
- a client-side validation alert with Save disabled for a malformed address.

The temporary frontend/backend processes were stopped, and ports 9847 and
7690 had no listening process afterward. The disposable database was not part
of the repository.

## Not claimed by this run

- No 195 MB live archive download or live master-list extraction was completed.
- No LiverTox catalog replacement or restart-persistence check was claimed.
- No ordered RxNav → LiverTox → DILIrank Update All success was claimed.
- No real production developer contact was supplied or persisted.

Accordingly, `data.inspection.catalogs`, `data.sources.refresh`, and
`ISSUE-003` remain `PARTIAL`/open pending the full disposable-database live
refresh and failure-preservation revalidation.

## Official references

- [NCBI Books-OAI documentation](https://www.ncbi.nlm.nih.gov/sites/books/NBK554844/)
- [NCBI LitArch documentation](https://www.ncbi.nlm.nih.gov/sites/books/NBK554846/)
