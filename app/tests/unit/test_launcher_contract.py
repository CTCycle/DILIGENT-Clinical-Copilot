from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER_PATH = REPO_ROOT / "start_on_windows.ps1"

###############################################################################
def _launcher_text() -> str:
    return LAUNCHER_PATH.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    marker = f"function {name}"
    start = source.index(marker)
    next_function = source.find("\nfunction ", start + len(marker))
    end = len(source) if next_function == -1 else next_function
    return source[start:end]


###############################################################################
def test_launcher_never_terminates_unowned_port_listeners() -> None:
    source = _launcher_text()
    process_discovery = _function_body(source, "Get-ApplicationProcessRecords")
    port_wait = _function_body(source, "Wait-ForLauncherPorts")
    owned_stop = _function_body(source, "Stop-ApplicationProcessRecords")
    explicit_cleanup = _function_body(source, "Stop-ApplicationProcesses")

    assert "function Stop-PortListeners" not in source
    assert "Get-ListeningProcessIds" not in process_discovery
    assert "taskkill.exe" not in port_wait
    assert "taskkill.exe /PID $process.ProcessId /T /F" in owned_stop
    assert "Get-LauncherPortStates" in explicit_cleanup
    assert "Assert-NoForeignPortListeners" in explicit_cleanup


def test_launch_preflight_checks_both_ports_before_startup_or_cleanup() -> None:
    source = _launcher_text()
    start_application = _function_body(source, "Start-Application")
    preflight = _function_body(source, "Invoke-LaunchPortPreflight")

    preflight_positions = []
    search_start = 0
    while True:
        position = start_application.find("Invoke-LaunchPortPreflight", search_start)
        if position == -1:
            break
        preflight_positions.append(position)
        search_start = position + 1

    assert len(preflight_positions) == 2
    assert preflight_positions[0] < start_application.index("Test-DependenciesReady")
    assert preflight_positions[1] < start_application.index("Start-Process -FilePath 'cmd.exe'")
    assert preflight.index("Assert-NoForeignPortListeners") < preflight.index(
        "Stop-ApplicationProcessRecords"
    )
    assert "BackendPort" in preflight and "FrontendPort" in preflight


def test_launcher_process_ownership_requires_repository_and_launch_signatures() -> None:
    process_discovery = _function_body(_launcher_text(), "Get-ApplicationProcessRecords")

    assert "$repositoryPattern" in process_discovery
    assert "uvicorn.*app:app" in process_discovery
    assert "(?:run\\s+preview|vite)" in process_discovery
    assert "--port" in process_discovery
    assert "candidateIds.Contains($parentId)" in process_discovery


def test_launch_conflict_messages_preserve_process_and_remediation_details() -> None:
    source = _launcher_text()
    conflict_message = _function_body(source, "New-PortConflictMessage")
    preflight = _function_body(source, "Invoke-LaunchPortPreflight")

    assert "PID(s)" in conflict_message
    assert "DILIGENT did not terminate an unowned listener" in conflict_message
    assert "settings/.env" in conflict_message
    assert "LauncherInteractive" in preflight
    assert "KillApplicationProcesses" in preflight
