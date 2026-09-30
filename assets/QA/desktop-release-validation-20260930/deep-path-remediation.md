# DILIGENT 3.4.0 deep-path remediation

Date: 2026-09-30 (Europe/Rome)

## Reproduction

The pre-fix portable candidate was launched from an executable path of 230
characters with an isolated `%LOCALAPPDATA%` path of 107 characters. The
desktop shell wrote a ready payload and showed the branded window, but the
backend refused loopback connections and logged:

```text
ImportError: DLL load failed while importing onnxruntime_pybind11_state:
Nome del file o estensione troppo lunga.
```

The traceback entered `onnxruntime` during the frozen backend import path. The
pre-fix artifacts are retained under
`prebuild-stale-artifacts-deep-fix-20260930/`.

## Correction

The runtime remains versioned and addressed by the full embedded archive
SHA-256. Only the internal directory names used by the native backend were
shortened:

- `%LOCALAPPDATA%\DILIGENT\runtime` became `%LOCALAPPDATA%\DILIGENT\rt`.
- The packaged backend destination `backend` became `b`.
- Rust extraction validation, backend launch, archive validation, smoke
  assertions, and release documentation were synchronized.

This preserves `%LOCALAPPDATA%\DILIGENT\data` and the manifest/marker checks
while reducing the native-extension path used by Windows DLL resolution.

## Repair-build evidence

The dirty repair build used the pinned Windows toolchain and completed the
frontend build, PyInstaller build, runtime archive validation, frozen-backend
smoke, Tauri build, MSI bundle, and portable artifact publication.

- Repair source commit: `749e86c9c33b49ce76668bae03a91c2673ce524d`
- Runtime archive SHA-256: `3bbc9bd6d6a1cc92909b3defd58d899c325ce87a9398c0b0d985753f943e80d1`
- Runtime files: `1108`
- Archive-required backend: `b/DILIGENTBackend.exe`
- Repair portable SHA-256: `e9a4d1f977b74864b32975ab42512722c863f111c6670fa2c15de960f1139ceb`

The standard portable smoke exited `0` with two health-200 launches, the
branded native title, clean backend/port shutdown, and ready-file replacement.

## Formerly failing boundary

The repaired artifact was copied to the same 230-character executable path
and launched twice with the same 107-character isolated `%LOCALAPPDATA%`:

| Launch | Backend PID | Port | Health | Settings | Title | Backend gone | Port closed |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 8596 | 49249 | 200 | 401 | `DILIGENT Clinical Copilot` | yes | yes |
| 2 | 8428 | 60302 | 200 | 401 | `DILIGENT Clinical Copilot` | yes | yes |

The final log scan found no `onnxruntime_pybind11_state`, import, or
filename-length error. Temporary executable/data roots were removed after the
replay.

## Clean candidate revalidation

A clean pinned-toolchain rebuild was then completed from repair commit
`2fadf0f22410665a814efc24b316cf68753e1cd6` with `dirty_tree=false`.

- Runtime archive SHA-256: `b1f20b52b4cc211b3da4aa5a4a2aa796f61216a571b5bf66acd1d049ad3fc433`
- Runtime files: `1108`
- Archive-required backend: `b/DILIGENTBackend.exe`
- Portable SHA-256: `b525df274191b3a6e7f04bd0ff070d853100a177e2b094bc6e72d0067b7c5da5`
- MSI SHA-256: `7708cb8b5e7495c3e29193780a0434d952011d5e699e3b2319e25534bf9f7759`

The standard portable smoke passed with two health-200 launches, the branded
title, clean backend/port shutdown, extraction under `DILIGENT\\rt`, and
stale-ready-file replacement. The exact formerly failing boundary was replayed
twice with executable path length `230` and isolated `%LOCALAPPDATA%` length
`107`:

| Launch | Backend PID | Port | Health | Settings | Title | Backend gone | Port closed | ONNX path error |
|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 36652 | 65336 | 200 | 401 | `DILIGENT Clinical Copilot` | yes | yes | no |
| 2 | 15316 | 58501 | 200 | 401 | `DILIGENT Clinical Copilot` | yes | yes | no |

The clean candidate therefore closes the deep-path portable gate. Temporary
executable/data roots were removed after the clean replay.

## Boundary

The deep-path portable gate is now validated for the clean candidate. Native
packaged UI interaction, MSI installation/upgrade/uninstall, signing, and
clean-machine distribution certification remain separate open gates.
