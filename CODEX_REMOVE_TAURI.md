You are working on the DILIGENT-Clinical-Copilot repository checked out at the current directory.

TASK: Remove all Tauri packaging infrastructure, consolidate the launcher scripts into a single PowerShell menu, and update all documentation. Do NOT modify any Python, TypeScript, or Angular source code — only config files, CI, docs, scripts, and batch files.

## Step 1: Create `app.ps1` at repo root

Replace both `start_on_windows.bat` and `setup_and_maintenance.bat` with a single `app.ps1` interactive menu.

The PowerShell script must present this menu and implement each option:

```
=========================================
    DILIGENT Clinical Copilot
=========================================
1.  Launch application
2.  Install / update dependencies
3.  Initialize database
4.  Run test suite
5.  Remove logs
6.  Clear cache
7.  Uninstall application
8.  Exit
=========================================
Select an option (1-8):
```

**OPTION 1 — Launch application:**
Read the existing `start_on_windows.bat` and replicate its logic in PowerShell:
- Ensure portable runtimes directory `runtimes/` exists
- For each runtime (Python embeddable, uv, Node.js): detect if installed, download ZIP from official URL if missing using Invoke-WebRequest, extract using Expand-Archive, patch python314._pth to enable site packages
- Set environment: UV_CACHE_DIR, UV_PROJECT_ENVIRONMENT pointing to app/server/.venv, PYTHONHOME unset, PYTHONPATH unset, PYTHONNOUSERSITE unset
- Run `uv sync --python <python_exe>` in app/server/ to install Python deps (with --all-extras if OPTIONAL_DEPENDENCIES=true)
- Install npm dependencies in app/client/ (npm ci if package-lock.json exists, else npm install)
- Build frontend: npm run build
- Load settings/.env into environment variables (parse key=value, skip comments/blanks)
- Kill any processes on FASTAPI_PORT and UI_PORT using netstat + taskkill
- Start uvicorn via `start` as background process: python.exe -m uvicorn app:app --app-dir <root>/app --host <FASTAPI_HOST> --port <FASTAPI_PORT>
- Wait for backend health check via Invoke-WebRequest on /api/health (up to 60s, 1s intervals)
- Start frontend preview via npm run preview -- --host <UI_HOST> --port <UI_PORT> --strictPort
- Open browser to http://<UI_HOST>:<UI_PORT>
- Print success message
- NOTE: Reuse the exact port defaults and .env loading logic from the original start_on_windows.bat (FASTAPI_PORT=8000, UI_PORT=7861, etc.)
- Helper functions to write: download_and_extract.ps1, patch_pth.ps1, find_uv.ps1, check_pyver.ps1, health_check.ps1
- Use Write-Host with colors for step output (like the batch file's [STEP], [OK], [FATAL] messages)

**OPTION 2 — Install / update dependencies:**
- Same runtime detection + download + extraction logic as Option 1
- uv sync for Python deps
- npm ci (or install) + npm run build
- Prune uv cache (remove UV_CACHE_DIR)
- Print success

**OPTION 3 — Initialize database:**
- Run: uv run --project app/server --python <python_exe> python app/scripts/initialize_database.py --drop-existing --seed-catalogs --force-reseed-catalogs
- Handle errors gracefully

**OPTION 4 — Run test suite:**
- Execute app/tests/run_tests.bat
- Report exit code

**OPTION 5 — Remove logs:**
- Delete all *.log files from app/resources/logs/

**OPTION 6 — Clear cache:**
- Remove all __pycache__ directories recursively under repo root
- Remove uv cache directory

**OPTION 7 — Uninstall application:**
- Remove: runtimes/, app/server/.venv, .venv (root), app/client/node_modules, app/client/.angular, app/client/dist, app/client/package-lock.json, app/server/uv.lock, uv.lock (root), all __pycache__ dirs
- Do NOT remove settings/, resources/ (database), or user data

**OPTION 8 — Exit**
- Break the menu loop

After any option completes (except Exit), prompt "Press any key to return to menu..." and wait for keypress before looping.

Menu navigation: use Read-Host for selection, switch statement, ValidateSet or regex validation on the input.

## Step 2: Delete old batch files

Remove from repo root:
- start_on_windows.bat
- setup_and_maintenance.bat

## Step 3: Delete all Tauri / Cargo / Rust artifacts

Directories to delete (entire trees):
- app/src-tauri/ (full Rust + Tauri project, including vendored crates app/src-tauri/vendor/cookie-0.18.1/ and app/src-tauri/vendor/tauri-2.6.2/)
- release/tauri/ (build scripts and scripts/ subdirectory)
- release/windows/ (if exists)

Files to delete:
- .github/workflows/desktop-release.yml

## Step 4: Update .gitignore

Add entries for the new app.ps1's helper scripts (if any are generated at runtime). Remove entries specific to Tauri build outputs like:
- app/src-tauri/target/
- app/src-tauri/bundle/
- app/src-tauri/gen/
- release/windows/

## Step 5: Update package.json

Edit app/client/package.json:
- Remove `"@tauri-apps/cli"` from devDependencies if present
- Remove the `"build:tauri"` script if it exists (keep the regular `"build"` script)

## Step 6: Update README.md

Read the current README.md and make these changes:
- Replace the Quick Start section to only reference `app.ps1` — remove dual-mode documentation
- Remove the "Switch to local Tauri profile" instructions and the associated `copy /Y settings\.env.local.example settings\.env` block
- Remove the entire "Desktop Packaging (Tauri)" section (currently section 5, starting with "The desktop shell lives in `app/src-tauri`...")
- Remove the "Build Windows desktop artifacts: release\tauri\build_with_tauri.bat" reference
- Remove the "release/windows/installers" reference
- Simplify "Runtime Profiles" to only cover local mode
- Update any file references: change start_on_windows.bat and setup_and_maintenance.bat to app.ps1
- Keep the LICENSE, CI badges, and any existing screenshots

## Step 7: Update assets/docs/

Scan these directories for any Tauri/desktop/packaging references and update or remove them:
- assets/docs/runtime/
- assets/docs/architecture/
- assets/docs/ui/
- assets/docs/user/
- assets/docs/coding/

Specifically:
- If assets/docs/runtime/modes.md exists and describes Tauri vs webapp modes, simplify to only webapp mode or remove the file
- Update any installation/setup docs to reference app.ps1 instead of the batch files

## Step 8: Verify

After all changes:
1. Confirm start_on_windows.bat and setup_and_maintenance.bat are deleted
2. Confirm app/src-tauri/ is gone
3. Confirm release/tauri/ is gone
4. Confirm release/windows/ is gone (if it existed)
5. Confirm .github/workflows/desktop-release.yml is deleted
6. Confirm app.ps1 exists at repo root with correct shebang and menu
7. Confirm no references to "Tauri", "desktop-release", "build_with_tauri", "tauri.conf", "Cargo.toml" remain in docs
8. List all changed files at the end