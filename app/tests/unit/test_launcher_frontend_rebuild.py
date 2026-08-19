from __future__ import annotations

from pathlib import Path


###############################################################################
def _launcher_text() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    return (repository_root / "start_on_windows.ps1").read_text(encoding="utf-8")


###############################################################################
def test_launcher_exposes_frontend_only_rebuild_action() -> None:
    script = _launcher_text()

    rebuild_frontend = script.index("function Rebuild-Frontend")
    install_dependencies = script.index("function Install-ApplicationDependencies")
    rebuild_frontend_end = script.index("function Test-DependenciesReady")

    assert "[ValidateSet('Launch', 'Install', 'RebuildFrontend'" in script
    assert "'RebuildFrontend' { Rebuild-Frontend }" in script
    assert "'^3$' { Rebuild-Frontend }" in script
    assert "Write-MenuOption -Number '3.' -Label 'Rebuild frontend'" in script
    assert "function Install-FrontendDependencies" in script
    assert "function Build-Frontend" in script
    assert rebuild_frontend > install_dependencies
    assert "Install-ApplicationDependencies" not in script[rebuild_frontend:rebuild_frontend_end]
    assert "Initialize-PortableRuntimes" not in script[rebuild_frontend:rebuild_frontend_end]
    assert "Set-LauncherEnvironment" not in script[rebuild_frontend:rebuild_frontend_end]
