# Local Deployment
Last updated: 2026-09-22

## Supported Runtime
- DILIGENT supports local single-user operation.
- On Windows, `start_on_windows.ps1` prepares only the warm-launch runtime prerequisites on a normal launch; backend/runtime repair and explicit installation use the tracked `app/server/uv.lock`. The frontend build is normally produced by install option 2, can be rebuilt independently with option 3 or `-Action RebuildFrontend`, and is rebuilt by option 1 only when its deterministic build state is missing, unreadable, stale, or its output is missing. Option 6 checks `origin/main` without changing the checkout, while option 7 updates source only from a non-detached, clean `main` checkout with `git pull --ff-only origin main`; it does not switch branches or modify local changes.
- The release frontend runtime is Node.js 22.13.0, matching the launcher and CI.
- Desktop release builds use Rust 1.95.0 with the `x86_64-pc-windows-msvc` target, Python 3.14.7, uv 0.11.30, and PyInstaller 6.21.0.
- RAG requires `numpy`, `onnxruntime`, and `tokenizers`; the canonical artifact is a pinned AVX2 `uint8` ONNX model. PyTorch and Sentence Transformers are not required.
- Manual macOS and Linux startup requires compatible Python, Node.js, and npm installations.

## Dependency Locks
- `app/server/uv.lock` is tracked and release-relevant workflows use `uv sync --locked`.
- `app/client/package-lock.json`
- `app/desktop/package-lock.json`
- `app/desktop/src-tauri/Cargo.lock`

## Source-mode frontend build policy

The source-mode Angular output is current only when both
`app/client/dist/browser/index.html` and
`app/client/dist/.diligent-build-state.json` exist. The state marker records
`schema_version`, a deterministic `build_fingerprint`, a separate
`dependency_fingerprint`, the pinned `node_version`, and the canonical
`build_command` (`npm run build`). Fingerprints use sorted normalized relative
paths and SHA-256 content digests; mtimes are not used.

The build fingerprint covers production files under `app/client/src/` except
`*.spec.ts`, `app/client/public/**`, `package.json`, `package-lock.json`,
`angular.json`, `tsconfig.json`, `tsconfig.app.json`, the pinned launcher Node
version, and the build command. Changes to spec files, `tsconfig.spec.json`,
`vitest.config.ts`, preview/dev-server scripts, `settings/.env`, backend
Python, documentation, and QA files do not invalidate the production build.

Automatic source-mode behavior is:

- valid current marker and output: reuse exactly as-is; no Angular build or npm dependency synchronization
- missing output, missing marker, corrupt marker, or fingerprint/contract mismatch: rebuild
- dependency-manifest or pinned-Node change, or an unknown prior marker: run `npm ci` from the tracked lockfile before rebuilding
- option 2 / `-Action Install`: synchronize backend and frontend dependencies and rebuild
- option 3 / `-Action RebuildFrontend`: rebuild unconditionally
- update/check-for-update actions: do not build the frontend

The launcher calculates the source fingerprint before and after `npm run build`
and writes the marker only when the inputs stayed stable. The marker is written
through an atomic replacement. Desktop-release staging output keeps its
existing output contract and does not require this source-mode marker.

## Database migrations

- Alembic revisions live under `app/server/migrations` and are bundled into the
  frozen backend. The application uses the synchronous SQLAlchemy engine and
  runs `upgrade head` before serving requests.
- From `app/server`, generate development revisions with `uv run alembic -c
  alembic.ini revision --autogenerate -m "describe the schema change"`, review
  the script, then run
  `uv run alembic -c alembic.ini upgrade head` and
  `uv run alembic -c alembic.ini current --check-heads`, followed by
  `uv run alembic -c alembic.ini check`.
- Keep one linear head. Back up production data before upgrades. Populated
  databases without an Alembic revision are rejected; the runtime does not
  guess or stamp unversioned migration history. Older or divergent schemas
  require an explicit administrative conversion plan.
- Use Alembic downgrade commands only for reviewed development or recovery
  procedures. The initializer's `--drop-existing` option is the explicit reset
  workflow and destroys application rows before rebuilding to head.

## Deployment Constraints
- Network deployment, reverse proxies, and unauthenticated multi-user access are unsupported.
- No supported container deployment path exists.
- Backend resources and the frontend build must remain aligned within the local repository checkout.
- Offline source deployments must pre-populate and verify `runtimes/cache/embeddings/<revision>/`; packaged deployments use `%LOCALAPPDATA%\DILIGENT\data\cache\embeddings\<revision>\`. A complete rebuild is mandatory after this model migration.

## Windows desktop distribution

Build from a Windows x64 host with Rust/Cargo, the Windows build toolchain, and the pinned frontend/backend dependencies. The current source manifests report synchronized unpublished candidate version `3.4.0`:

```powershell
.\start_on_windows.ps1 -Action BuildDesktopRelease -Version 3.4.0 -DesktopTarget All
```

The build produces `DILIGENT-v<version>-windows-x64-portable.exe`, `DILIGENT-v<version>-windows-x64.msi`, and a matching `.sha256` file under `release/`. The portable EXE is a single-file Tauri distribution; the MSI installs the same shell and packaged runtime. Use `-DesktopTarget Portable` or `-DesktopTarget Msi` for one artifact. Add `-OfflineWebView2` only for an MSI when an offline WebView2 installer is required. Release builds reject dirty worktrees unless `-AllowDirtyTree` is supplied.

The portable executable embeds the PyInstaller backend and deterministic runtime archive. At runtime it extracts immutable content to `%LOCALAPPDATA%\DILIGENT\runtime\<version>\<payload-sha256>`, starts the backend on a random localhost port, and keeps mutable user data under `%LOCALAPPDATA%\DILIGENT\data`. It uses the system WebView2 runtime. MSI uninstall removes installed program files but preserves `%LOCALAPPDATA%\DILIGENT\data`.

Before distribution, verify the `.sha256` file and perform the tracked Windows host smoke script for the portable EXE and MSI. Clean-machine, MSI upgrade/uninstall, WebView2 offline, code-signing, and enterprise deployment tests remain separate release gates.

The annotated `v3.4.0` tag must not be created until hosted CI, full browser and live-provider E2E, vulnerability-scan review, and Windows EXE/MSI smoke evidence are green and `main` is intentionally synchronized with the candidate source. Pushing that tag later invokes `.github/workflows/release.yml` on Windows; this session does not create or publish it.
