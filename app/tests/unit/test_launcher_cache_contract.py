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
