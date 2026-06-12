from __future__ import annotations

import importlib
from pathlib import Path

import common.paths as common_paths


###############################################################################
def test_database_file_path_honors_sqlite_env_override(monkeypatch) -> None:
    override_path = Path("C:/temp/diligent-sqlite-override.db")
    monkeypatch.setenv("DILIGENT_SQLITE_PATH", str(override_path))

    reloaded_paths = importlib.reload(common_paths)

    try:
        assert reloaded_paths.DATABASE_FILE_PATH == override_path
    finally:
        monkeypatch.delenv("DILIGENT_SQLITE_PATH", raising=False)
        importlib.reload(common_paths)
