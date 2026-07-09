# Deployment And Packaging
Last updated: 2026-07-09

## Desktop Packaging
- Packaging command:

```cmd
release\tauri\build_with_tauri.bat
```

- Underlying release workflow invokes:
  - `npm run tauri:build:release`
- Versioned desktop sources stay under `app/src-tauri` and include Rust source, configuration, icons, capabilities, and required build metadata.
- Generated desktop outputs under `app/src-tauri/target`, `app/src-tauri/bundle`, `app/src-tauri/gen`, and `release/windows` are build artifacts and must not be committed.
- Desktop binaries and installers are published as release artifacts instead of tracked repository files.

## Exported Windows Artifacts
- `release/windows/installers`
- `release/windows/portable`

## Build Prerequisites
- Portable runtimes available, usually after `start_on_windows.bat`
- Rust and Cargo toolchain installed
- Frontend dependencies installed

## Dependency Locks
- `runtimes/uv.lock`
- `app/client/package-lock.json`

## Packaging Constraints
- Desktop packaging is Windows-focused in the current repository.
- The packaged application is localhost-only. Network deployment, reverse
  proxies, and unauthenticated multi-user access are unsupported.
- No supported container deployment path exists.
- Packaged runtime depends on the bundled frontend dist plus backend resources being present and aligned.
