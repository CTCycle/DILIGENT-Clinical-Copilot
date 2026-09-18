from __future__ import annotations

import os
import subprocess
from pathlib import Path

from common.paths import EMBEDDING_MODELS_PATH
from common.runtime_layout import resolve_runtime_layout
from configurations.runtime_bootstrap import configure_cache_environment

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_CACHE_ROOT = REPO_ROOT / "runtimes" / "cache"

###############################################################################
def _tracked_text() -> str:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    contents: list[str] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        path = REPO_ROOT / raw_path.decode("utf-8")
        try:
            contents.append(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(contents)

###############################################################################
def test_source_and_packaged_layouts_have_one_cache_root(tmp_path: Path, monkeypatch) -> None:
    source_layout = resolve_runtime_layout()
    assert source_layout.packaged is False
    assert source_layout.cache_root == CANONICAL_CACHE_ROOT
    assert EMBEDDING_MODELS_PATH == CANONICAL_CACHE_ROOT / "embeddings"

    runtime_root = tmp_path / "runtime"
    data_root = tmp_path / "data"
    monkeypatch.setenv("DILIGENT_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("DILIGENT_DATA_ROOT", str(data_root))
    cache_environment_names = (
        "CARGO_TARGET_DIR",
        "COVERAGE_FILE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_ASSETS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "MYPY_CACHE_DIR",
        "NPM_CONFIG_CACHE",
        "PIP_CACHE_DIR",
        "PLAYWRIGHT_BROWSERS_PATH",
        "PYTHONPYCACHEPREFIX",
        "PYTEST_ADDOPTS",
        "RUFF_CACHE_DIR",
        "UV_CACHE_DIR",
    )
    original_cache_environment = {
        name: os.environ.get(name) for name in cache_environment_names
    }
    resolve_runtime_layout.cache_clear()
    try:
        packaged_layout = resolve_runtime_layout()
        assert packaged_layout.packaged is True
        assert packaged_layout.runtime_root == runtime_root.resolve()
        assert packaged_layout.cache_root == (data_root / "cache").resolve()
        configure_cache_environment()
        assert Path(os.environ["HF_HOME"]) == (
            data_root / "cache" / "huggingface"
        ).resolve()
        assert Path(os.environ["HF_HUB_CACHE"]) == (
            data_root / "cache" / "huggingface" / "hub"
        ).resolve()
        assert Path(os.environ["PYTHONPYCACHEPREFIX"]) == (
            data_root / "cache" / "python"
        ).resolve()
        assert str(data_root / "cache" / "pytest") in os.environ["PYTEST_ADDOPTS"]
    finally:
        for name, value in original_cache_environment.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        resolve_runtime_layout.cache_clear()

###############################################################################
def test_cache_consumers_resolve_under_canonical_root() -> None:
    expected_fragments = {
        "start_on_windows.ps1": (
            "RuntimeCacheDir = Join-Path $RuntimesDir 'cache'",
            "HF_HUB_CACHE",
            "CARGO_TARGET_DIR",
            "ClearCache",
        ),
        "app/tests/pytest.ini": ("../../runtimes/cache/pytest",),
        "app/tests/run_tests.bat": (
            "%PROJECT_ROOT%\\runtimes\\cache",
            "%CACHE_DIR%\\pytest",
        ),
        "app/tests/ci/run_browser_e2e.ps1": (
            "runtimes/cache",
            "PYTEST_ADDOPTS",
        ),
        "app/server/pyproject.toml": (
            "../../runtimes/cache/ruff",
            "../../runtimes/cache/coverage/.coverage",
        ),
        "app/client/angular.json": ("../../runtimes/cache/angular",),
        "app/client/tsconfig.app.json": ("../../runtimes/cache/angular/out-tsc/app",),
        "app/client/tsconfig.spec.json": ("../../runtimes/cache/angular/out-tsc/spec",),
        "app/client/vitest.config.ts": (
            "../../runtimes/cache/coverage/angular",
        ),
        "app/client/.npmrc": ("../../runtimes/cache/npm",),
        "app/desktop/.npmrc": ("../../runtimes/cache/npm",),
        ".cargo/config.toml": ("runtimes/cache/cargo/target",),
        ".vscode/launch.json": ("runtimes/cache/pytest",),
        ".github/workflows/ci.yml": ("runtimes/cache/pytest",),
        ".github/workflows/release.yml": ("runtimes/cache/pytest",),
    }
    for relative_path, fragments in expected_fragments.items():
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for fragment in fragments:
            assert fragment in text, f"{relative_path} is missing {fragment!r}"

    tracked_text = _tracked_text()
    forbidden_references = (
        "/".join(("app", "tests", "cache")),  # noqa: FLY002 - keep the contract source free of this literal
        "/".join(("assets", "cache")),  # noqa: FLY002 - keep the contract source free of this literal
        "_".join(("DILIGENT", "TEST", "CACHE", "ROOT")),  # noqa: FLY002 - keep the contract source free of this literal
        "/".join(("models", "embeddings")),  # noqa: FLY002 - keep the contract source free of this literal
        "-".join(("desktop", "cargo", "target")),  # noqa: FLY002 - keep the contract source free of this literal
        "-".join(("desktop", "release", "uv", "cache")),  # noqa: FLY002 - keep the contract source free of this literal
    )
    for reference in forbidden_references:
        assert reference not in tracked_text
