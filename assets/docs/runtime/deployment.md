# Deployment And Packaging
Last updated: 2026-06-21

## Desktop Packaging
- Packaging command:

```cmd
release\tauri\build_with_tauri.bat
```

- Underlying release workflow invokes:
  - `npm run tauri:build:release`

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
- Supported deployment mode is `local_single_user`; startup rejects other deployment modes.
- No supported container deployment path exists.
- Packaged runtime depends on the bundled frontend dist plus backend resources being present and aligned.
