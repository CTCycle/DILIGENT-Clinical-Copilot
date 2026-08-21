from __future__ import annotations

from pathlib import Path

###############################################################################
def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]

###############################################################################
def test_launcher_uses_split_runtime_and_test_cache_roots() -> None:
    script = (_repository_root() / "start_on_windows.ps1").read_text(encoding="utf-8")

    assert "Join-Path $RuntimesDir 'cache'" in script
    assert "Join-Path $TestsDir 'cache'" in script
    assert "Join-Path $RepoRoot 'assets/cache'" in script
    assert "$script:UvCacheDir = Join-Path $RuntimeCacheDir 'uv'" in script
    assert "$script:PytestCacheDir = Join-Path $TestCacheDir 'pytest'" in script
    assert "$script:RuffCacheDir = Join-Path $TestCacheDir 'ruff'" in script
    assert "Remove-CacheContents -RootPath $cacheRoot" in script

###############################################################################
def test_launcher_cache_cleanup_skips_locked_entries() -> None:
    script = (_repository_root() / "start_on_windows.ps1").read_text(encoding="utf-8")

    cleanup_start = script.index("function Remove-PathSafely")
    cleanup_end = script.index("function Uninstall-Application")
    cleanup = script[cleanup_start:cleanup_end]

    assert "catch {" in cleanup
    assert "Skipped locked or inaccessible cache item" in cleanup
    assert "-ErrorAction Stop" in cleanup
    assert "rerun as administrator" in cleanup


###############################################################################
def test_launcher_groups_source_control_and_user_data_actions() -> None:
    script = (_repository_root() / "start_on_windows.ps1").read_text(encoding="utf-8")

    assert "'Update', 'CheckForUpdates', 'RemoveAllData'" in script
    assert "'Update' { Update-Application }" in script
    assert "'CheckForUpdates' { Check-ForUpdates }" in script
    assert "'RemoveAllData' { Remove-AllData }" in script
    assert "ls-remote origin refs/heads/main" in script
    assert "@('pull', 'origin', 'main')" in script
    assert "Write-MenuSection -Title 'SOURCE CONTROL'" in script
    assert "Write-MenuSection -Title 'DATA & CLEANUP'" in script
    assert "Write-MenuOption -Number '10.' -Label 'Remove all data'" in script
    assert "Confirm-RemoveAllData -Force:$Force" in script
