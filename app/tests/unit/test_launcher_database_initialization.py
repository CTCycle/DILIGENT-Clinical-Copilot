from __future__ import annotations

from pathlib import Path

###############################################################################
def _launcher_text() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    return (repository_root / "start_on_windows.ps1").read_text(encoding="utf-8")

###############################################################################
def test_launcher_keeps_database_initialization_explicit() -> None:
    script = _launcher_text()
    start_application = script.index("function Start-Application")
    install_application = script.index("function Install-OrUpdateApplication")
    initialize_database = script.index("function Initialize-Database")
    install_end = initialize_database

    assert "Initialize-Database" not in script[start_application:install_application]
    assert initialize_database > start_application
    assert "app/scripts/initialize_database.py" in script[initialize_database:]
    assert "'InitializeDatabase' { Initialize-Database }" in script
    assert "'^4$' { Initialize-Database }" in script
    assert "Initialize-Database" in script[install_application:install_end]
    initialize_database_end = script.index("function Invoke-TestSuite")
    assert "Write-Step 'Synchronizing database schema'" in script[
        initialize_database:initialize_database_end
    ]
    assert script.count("Write-Step 'Synchronizing database schema'") == 1
    assert "select version_num from alembic_version" in script
