# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from configurations import database_config
from configurations.management import environment_snapshot_from_os_env


###############################################################################
def _patch_configuration_paths(monkeypatch, tmp_path: Path) -> Path:  # type: ignore[no-untyped-def]
    config_path = tmp_path / "database_config.json"
    env_path = tmp_path / "settings" / ".env"
    env_path.parent.mkdir(parents=True)
    monkeypatch.setattr(database_config, "DATABASE_CONFIG_FILE_PATH", config_path)
    monkeypatch.setattr(database_config.paths, "ENV_FILE_PATH", env_path)
    return config_path


###############################################################################
def test_legacy_dotenv_is_imported_once_and_scrubbed(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config_path = _patch_configuration_paths(monkeypatch, tmp_path)
    env_path = database_config.paths.ENV_FILE_PATH
    env_path.write_text(
        "FASTAPI_PORT=7777\n"
        "DATABASE_BACKEND=sqlite\n"
        "DATABASE_SQLITE_PATH=legacy-store.sqlite3\n"
        "DATABASE_WRITE_BATCH_SIZE=250\n"
        "DATABASE_READ_PAGE_SIZE=500\n",
        encoding="utf-8",
    )
    for key in database_config.LEGACY_DATABASE_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)

    loaded = database_config.load_database_configuration()

    assert loaded.backend == "sqlite"
    assert loaded.sqlite_path == str(Path("legacy-store.sqlite3").resolve())
    assert loaded.write_batch_size == 250
    assert loaded.read_page_size == 500
    assert config_path.is_file()
    remaining_env = env_path.read_text(encoding="utf-8")
    assert "FASTAPI_PORT=7777" in remaining_env
    assert "DATABASE_" not in remaining_env


###############################################################################
def test_database_file_is_authoritative_over_environment(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config_path = _patch_configuration_paths(monkeypatch, tmp_path)
    expected = database_config.DatabaseConfiguration(
        backend="sqlite",
        sqlite_path=str(tmp_path / "selected.sqlite3"),
    )
    database_config.save_database_configuration(expected)
    monkeypatch.setenv("DATABASE_BACKEND", "postgresql")
    monkeypatch.setenv("DATABASE_HOST", "ignored.invalid")

    loaded = database_config.load_database_configuration()

    assert loaded == expected
    assert json.loads(config_path.read_text(encoding="utf-8"))["database"]["backend"] == "sqlite"


###############################################################################
def test_corrupt_database_file_fails_closed_without_legacy_fallback(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    config_path = _patch_configuration_paths(monkeypatch, tmp_path)
    config_path.write_text('{"schema_version":99}', encoding="utf-8")
    monkeypatch.setenv("DATABASE_BACKEND", "sqlite")

    with pytest.raises(RuntimeError, match="Database configuration is invalid"):
        database_config.load_database_configuration()

    assert config_path.read_text(encoding="utf-8") == '{"schema_version":99}'


###############################################################################
def test_database_configuration_is_excluded_from_general_environment_snapshot(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DATABASE_URL", "postgresql://ignored:secret@example.invalid/db")
    monkeypatch.setenv("DATABASE_BACKEND", "postgresql")
    snapshot = environment_snapshot_from_os_env()

    assert "database" not in snapshot.model_dump()
    assert "DATABASE_URL" not in str(snapshot.model_dump())


###############################################################################
@pytest.mark.skipif(os.name != "nt", reason="Credential protection uses Windows DPAPI")
def test_postgres_password_is_dpapi_protected_in_configuration_file(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    config_path = _patch_configuration_paths(monkeypatch, tmp_path)
    secret = "temporary-test-secret"
    protected = database_config.protect_database_password(secret)
    configuration = database_config.DatabaseConfiguration(
        backend="postgresql",
        host="127.0.0.1",
        database_name="configuration_test",
        username="test_user",
        password_protected=protected,
    )

    database_config.save_database_configuration(configuration)
    content = config_path.read_text(encoding="utf-8")

    assert secret not in content
    assert database_config.unprotect_database_password(protected) == secret
    assert Path(config_path).read_bytes()
