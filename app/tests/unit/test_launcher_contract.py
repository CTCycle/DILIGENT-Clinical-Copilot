# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER_PATH = REPO_ROOT / "start_on_windows.ps1"

###############################################################################
def _launcher_text() -> str:
    return LAUNCHER_PATH.read_text(encoding="utf-8")

###############################################################################
def _function_body(source: str, name: str) -> str:
    marker = f"function {name}"
    start = source.index(marker)
    next_function = source.find("\nfunction ", start + len(marker))
    end = len(source) if next_function == -1 else next_function
    return source[start:end]

###############################################################################
def test_launch_port_cleanup_uses_general_deduplicated_holders() -> None:
    source = _launcher_text()
    process_discovery = _function_body(source, "Get-ApplicationProcessRecords")
    port_wait = _function_body(source, "Wait-ForLauncherPorts")
    owned_stop = _function_body(source, "Stop-ApplicationProcessRecords")
    explicit_cleanup = _function_body(source, "Stop-ApplicationProcesses")
    port_stop = _function_body(source, "Stop-PortConflictProcesses")
    launch_resolution = _function_body(source, "Resolve-LaunchPortConflicts")

    assert "function Stop-PortListeners" not in source
    assert "function Get-ListeningPortRecords" in source
    assert "function Get-PortConflictProcesses" in source
    assert "Get-ListeningProcessIds" not in process_discovery
    assert "taskkill.exe" not in port_wait
    assert "taskkill.exe /PID $process.ProcessId /T /F" in owned_stop
    assert "Stop-Process -Id $processId -Force" in port_stop
    assert "Get-ApplicationProcessRecords" not in launch_resolution
    assert "Assert-NoForeignPortListeners" not in launch_resolution
    assert "Get-PortConflictProcesses -PortRecord" in launch_resolution
    assert "Get-LauncherPortStates" in explicit_cleanup
    assert "Assert-NoForeignPortListeners" in explicit_cleanup

###############################################################################
def test_launch_resolves_only_api_port_and_starts_one_backend_process() -> None:
    source = _launcher_text()
    start_application = _function_body(source, "Start-Application")

    assert start_application.count("Resolve-LaunchPortConflicts") == 1
    assert start_application.index("Resolve-LaunchPortConflicts") < start_application.index(
        "Start-Process -FilePath $VenvPython"
    )
    assert "-FrontendPort" not in start_application
    assert "UI_PORT" not in start_application
    assert "run preview" not in start_application
    assert "source_entry" in start_application
    assert "api/health" in start_application
    assert "JavaScript asset" in start_application
    assert "Test-DependenciesReady" not in source
    assert start_application.count("Set-LauncherEnvironment") == 1
    assert "Test-LaunchRuntimeReady" in start_application
    assert "Get-FrontendBuildStatus" in start_application

###############################################################################
def test_explicit_cleanup_retains_repository_qualified_process_ownership() -> None:
    process_discovery = _function_body(_launcher_text(), "Get-ApplicationProcessRecords")

    assert "$repositoryPattern" in process_discovery
    assert "uvicorn.*app:app" in process_discovery
    assert "-m\\s+source_entry" in process_discovery
    assert "(?:run\\s+preview|vite)" in process_discovery
    assert "--port" in process_discovery
    assert "candidateIds.Contains($parentId)" in process_discovery

###############################################################################
def test_launch_shutdown_is_graceful_then_limited_to_the_owned_process_tree() -> None:
    source = _launcher_text()
    shutdown = _function_body(source, "Stop-OwnedApplication")
    supervisor = _function_body(source, "Wait-OwnedApplication")

    assert "/api/desktop/shutdown" in shutdown
    assert "X-DILIGENT-Desktop-Session" in shutdown
    assert "WaitForExit(15000)" in shutdown
    assert "taskkill.exe /PID $processId /T /F" in shutdown
    assert "Wait-ForLauncherPorts" in shutdown
    assert "Stop-OwnedApplication" in source[source.index("finally {", source.index("function Invoke-RemoveDesktopReleaseMenu")):]
    assert "while (-not $process.HasExited)" in supervisor
    assert "Wait-OwnedApplication" in source[source.index("if ($Action)"):]

###############################################################################
def test_launch_only_prepares_node_for_a_stale_frontend_build() -> None:
    source = _launcher_text()
    runtime_check = _function_body(source, "Test-LaunchRuntimeReady")
    start_application = _function_body(source, "Start-Application")

    assert "$NodeExe" not in runtime_check
    assert "preview server" not in runtime_check
    assert "Initialize-PortableNodeRuntime" in start_application
    assert start_application.index("Get-FrontendBuildStatus") < start_application.index(
        "Initialize-PortableNodeRuntime"
    )

###############################################################################
def test_launch_conflict_messages_include_pid_port_and_safe_confirmation_boundaries() -> None:
    source = _launcher_text()
    conflict_message = _function_body(source, "New-PortConflictMessage")
    resolution = _function_body(source, "Resolve-LaunchPortConflicts")

    assert "PID(s)" in conflict_message
    assert "Configured port conflict" in conflict_message
    assert "settings/.env" in conflict_message
    assert "LauncherInteractive" in resolution
    assert "No port holders were terminated" in resolution
    assert "All configured-port holders were left running" in resolution

###############################################################################
def test_frontend_build_state_is_deterministic_and_published_after_stable_build() -> None:
    source = _launcher_text()
    input_files = _function_body(source, "Get-FrontendBuildInputFiles")
    fingerprint = _function_body(source, "Get-FrontendInputFingerprint")
    status = _function_body(source, "Get-FrontendBuildStatus")
    build = _function_body(source, "Build-Frontend")

    assert "dist/.diligent-build-state.json" in source
    assert "Get-FileHash" in fingerprint
    assert "Sort-Object -Unique" in input_files
    assert r"\.spec\.ts$" in input_files
    assert "schema_version" in source
    assert "build_fingerprint" in status
    assert "dependency_fingerprint" in status
    assert "Get-FrontendBuildFingerprint" in build
    assert "Write-FrontendBuildState" in build
    assert "VITE_API_BASE_URL" in fingerprint
    assert "__launcher__/api-base-url" in fingerprint
    assert "DILIGENT_DESKTOP_BUILD" in build
    assert "[IO.File]::Replace" in source
    assert "changed while Angular was building" in build

###############################################################################
def test_warm_runtime_repair_does_not_force_frontend_dependency_or_build_work() -> None:
    source = _launcher_text()
    start_application = _function_body(source, "Start-Application")
    install = _function_body(source, "Install-ApplicationDependencies")
    explicit_install = _function_body(source, "Install-OrUpdateApplication")

    assert "-InstallFrontendDependencies $false" in start_application
    assert "-ForceFrontendDependencySync" in explicit_install
    assert "Install-FrontendDependencies -ForceSync:$frontendBuildStatus.DependencyInputsChanged" in start_application
    assert "BuildFrontend requires InstallFrontendDependencies" in install
