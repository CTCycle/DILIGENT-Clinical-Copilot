from __future__ import annotations

import sqlite3
from pathlib import Path

from repositories.database.sqlite import SQLiteRepository

###############################################################################
def test_sqlite_connection_uses_durable_wal_for_embedded_workspace_writes(
    tmp_path: Path,
) -> None:
    connection = sqlite3.connect(tmp_path / "persistence.db")

    SQLiteRepository._configure_connection(connection, None)

    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        connection.close()
