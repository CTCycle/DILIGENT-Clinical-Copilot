# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from pathlib import Path

import pytest
from common.runtime_layout import RuntimeLayout
from configurations.runtime_bootstrap import migrate_legacy_data_layout


###############################################################################
def _runtime_layout(
    runtime_root: Path, data_root: Path, *, packaged: bool
) -> RuntimeLayout:
    return RuntimeLayout(
        packaged=packaged,
        runtime_root=runtime_root,
        application_root=runtime_root / "app",
        settings_template_root=runtime_root / "settings",
        settings_root=data_root / "settings",
        immutable_data_root=runtime_root / "data",
        mutable_data_root=data_root,
        cache_root=data_root / "cache",
        client_dist_root=runtime_root / "app" / "client" / "dist" / "browser",
    )


###############################################################################
def test_packaged_data_migration_merges_without_replacing_existing_state(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    data_root = tmp_path / "user-data"
    legacy_root = data_root / "resources"
    (legacy_root / "sources" / "documents").mkdir(parents=True)
    (legacy_root / "exports").mkdir()
    (legacy_root / "state").mkdir()
    (legacy_root / "database.db").write_bytes(b"sqlite-data")
    (legacy_root / "database.db-shm").write_bytes(b"shared-memory")
    (legacy_root / "database.db-wal").write_bytes(b"write-ahead-log")
    (legacy_root / "sources" / "documents" / "source.txt").write_text(
        "source", encoding="utf-8"
    )
    (legacy_root / "exports" / "report.csv").write_text(
        "id,value\n1,ok\n", encoding="utf-8"
    )
    (legacy_root / "state" / "previous.json").write_text("{}", encoding="utf-8")
    ready_file = data_root / "state" / "desktop-backend-ready.json"
    ready_file.parent.mkdir(parents=True)
    ready_file.write_text('{"ready":true}', encoding="utf-8")

    migrate_legacy_data_layout(_runtime_layout(runtime_root, data_root, packaged=True))

    assert (data_root / "database.db").read_bytes() == b"sqlite-data"
    assert (data_root / "database.db-shm").read_bytes() == b"shared-memory"
    assert (data_root / "database.db-wal").read_bytes() == b"write-ahead-log"
    assert (data_root / "sources" / "documents" / "source.txt").read_text(
        encoding="utf-8"
    ) == "source"
    assert (data_root / "exports" / "report.csv").is_file()
    assert (data_root / "state" / "previous.json").is_file()
    assert ready_file.read_text(encoding="utf-8") == '{"ready":true}'
    assert not legacy_root.exists()


###############################################################################
def test_source_data_migration_preserves_catalogs_and_merges_sources(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "repository"
    old_root = runtime_root / "resources"
    new_root = runtime_root / "data"
    (old_root / "catalogs").mkdir(parents=True)
    (old_root / "sources").mkdir()
    (new_root / "catalogs").mkdir(parents=True)
    (new_root / "sources").mkdir()
    (old_root / "catalogs" / "old.json").write_text("{}", encoding="utf-8")
    (old_root / "database.db").write_bytes(b"database")
    (old_root / "sources" / "aliases.json").write_text("same", encoding="utf-8")
    (new_root / "sources" / "aliases.json").write_text("same", encoding="utf-8")
    (old_root / "sources" / "document.txt").write_text("document", encoding="utf-8")

    migrate_legacy_data_layout(_runtime_layout(runtime_root, new_root, packaged=False))

    assert (new_root / "database.db").read_bytes() == b"database"
    assert (new_root / "sources" / "aliases.json").read_text(encoding="utf-8") == "same"
    assert (new_root / "sources" / "document.txt").read_text(
        encoding="utf-8"
    ) == "document"
    assert (old_root / "catalogs" / "old.json").is_file()


###############################################################################
def test_data_migration_stops_before_moving_anything_on_conflict(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    data_root = tmp_path / "user-data"
    legacy_root = data_root / "resources"
    legacy_root.mkdir(parents=True)
    (legacy_root / "database.db").write_bytes(b"old")
    (legacy_root / "access-key-material.json").write_text("old-key", encoding="utf-8")
    data_root.mkdir(exist_ok=True)
    (data_root / "database.db").write_bytes(b"new")

    with pytest.raises(RuntimeError, match="no data was changed"):
        migrate_legacy_data_layout(
            _runtime_layout(runtime_root, data_root, packaged=True)
        )

    assert (legacy_root / "database.db").read_bytes() == b"old"
    assert (legacy_root / "access-key-material.json").read_text(
        encoding="utf-8"
    ) == "old-key"
    assert (data_root / "database.db").read_bytes() == b"new"
