# Local Deployment
Last updated: 2026-09-01

## Supported Runtime
- DILIGENT supports local single-user operation.
- On Windows, `start_on_windows.ps1` prepares portable runtimes and dependencies before launching the local services; the frontend build is normally produced by install option 2, can be rebuilt independently with option 3 or `-Action RebuildFrontend`, and is rebuilt by option 1 when recovery detects missing or unusable environments or frontend output. Option 6 checks `origin/main` without changing the checkout, while option 7 updates source only from a non-detached, clean `main` checkout with `git pull --ff-only origin main`; it does not switch branches or modify local changes.
- The release frontend runtime is Node.js 22.13.0, matching the launcher and CI.
- Desktop release builds use Rust 1.95.0 with the `x86_64-pc-windows-msvc` target, Python 3.14.2, uv 0.11.30, and PyInstaller 6.21.0.
- RAG requires `numpy`, `onnxruntime`, and `tokenizers`; the canonical artifact is a pinned AVX2 `uint8` ONNX model. PyTorch and Sentence Transformers are not required.
- Manual macOS and Linux startup requires compatible Python, Node.js, and npm installations.

## Dependency Locks
- `app/server/uv.lock` is tracked and release-relevant workflows use `uv sync --locked`.
- `app/client/package-lock.json`
- `app/desktop/package-lock.json`
- `app/desktop/src-tauri/Cargo.lock`

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
- Offline deployments must pre-populate and verify `app/resources/models/embeddings/<revision>/`; a complete rebuild is mandatory after this model migration.

## Windows desktop distribution

Build from a Windows x64 host with Rust/Cargo, the Windows build toolchain, and the pinned frontend/backend dependencies. The current source manifests report version `3.3.0`:

```powershell
.\start_on_windows.ps1 -Action BuildDesktopRelease -Version 3.3.0 -DesktopTarget All -Force
```

The build produces `DILIGENT-v<version>-windows-x64-portable.exe`, `DILIGENT-v<version>-windows-x64.msi`, and a matching `.sha256` file under `release/`. The portable EXE is a single-file Tauri distribution; the MSI installs the same shell and packaged runtime. Use `-DesktopTarget Portable` or `-DesktopTarget Msi` for one artifact. Add `-OfflineWebView2` only for an MSI when an offline WebView2 installer is required. Release builds reject dirty worktrees unless `-AllowDirtyTree` is supplied.

The portable executable embeds the PyInstaller backend and deterministic runtime archive. At runtime it extracts immutable content to `%LOCALAPPDATA%\DILIGENT\runtime\<version>\<payload-sha256>`, starts the backend on a random localhost port, and keeps mutable user data under `%LOCALAPPDATA%\DILIGENT\data`. It uses the system WebView2 runtime. MSI uninstall removes installed program files but preserves `%LOCALAPPDATA%\DILIGENT\data`.

Before distribution, verify the `.sha256` file and perform a Windows host smoke test of the portable EXE. Clean-machine, MSI upgrade/uninstall, WebView2 offline, code-signing, and enterprise deployment tests are separate release gates.

Pushing the annotated `v3.3.0` tag invokes `.github/workflows/release.yml` on Windows. The workflow rebuilds both desktop targets from the tag and attaches the portable EXE, MSI, and checksum manifest to the corresponding GitHub Release.
