from __future__ import annotations

from pathlib import Path

###############################################################################
def _launcher_text() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    return (repository_root / "start_on_windows.ps1").read_text(encoding="utf-8")

###############################################################################
def test_launcher_exposes_create_and_remove_artifact_submenus() -> None:
    script = _launcher_text()

    assert "function Read-DesktopArtifactSelection" in script
    assert "function Invoke-CreateDesktopReleaseMenu" in script
    assert "function Invoke-RemoveDesktopReleaseMenu" in script
    assert "Write-MenuOption -Number '12.' -Label 'Create release artifacts'" in script
    assert "Write-MenuOption -Number '13.' -Label 'Remove release artifacts'" in script
    assert "'1' { return [pscustomobject]@{ Target = 'Portable'" in script
    assert "'2' { return [pscustomobject]@{ Target = 'Msi'" in script
    assert "'3' { return [pscustomobject]@{ Target = 'Checksum'" in script
    assert "'4' { return [pscustomobject]@{ Target = 'All'" in script
    assert "'^12$' { Invoke-CreateDesktopReleaseMenu }" in script
    assert "'^13$' { Invoke-RemoveDesktopReleaseMenu }" in script

###############################################################################
def test_launcher_keeps_checksum_and_generated_state_cleanup_target_aware() -> None:
    script = _launcher_text()

    removal_start = script.index("function Remove-DesktopRelease")
    removal_end = script.index("# ============================================================\n# Interactive menu", removal_start)
    removal = script[removal_start:removal_end]

    assert "[ValidateSet('Portable', 'Msi', 'Checksum', 'All')]" in removal
    assert "Write-DesktopChecksums -Version $resolvedVersion" in removal
    assert "function Remove-DesktopGeneratedState" in script
    assert "Where-Object { $_.Name -ne '.gitkeep' }" in script
    assert "other generated state was preserved" in removal
