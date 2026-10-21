# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Snapshot stable application data, then verify it after real MSI transitions."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def inspect_database(path: Path) -> dict:
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError("SQLite integrity check failed.")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("SQLite foreign-key check failed.")
        sessions = connection.execute(
            "SELECT id, patient_name, visit_date, anamnesis FROM clinical_sessions ORDER BY id"
        ).fetchall()
        versions = connection.execute(
            "SELECT id, session_id, version_number, report_text, revision_kind "
            "FROM clinical_session_versions ORDER BY id"
        ).fetchall()
        configuration = connection.execute(
            "SELECT payload FROM application_configuration WHERE id = 1"
        ).fetchone()
        if not sessions or not versions or not configuration:
            raise RuntimeError(
                "Upgrade fixture must contain sessions, versions, and settings."
            )
        payload = json.loads(configuration[0])
        return {
            "sessions": sessions,
            "versions": versions,
            "polling_interval": payload.get("polling_interval"),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--expected-head")
    args = parser.parse_args()
    actual = json.loads(json.dumps(inspect_database(args.database)))
    if args.capture:
        if actual["polling_interval"] != 2.0:
            raise RuntimeError(
                "The baseline application's nondefault setting was not saved."
            )
        args.snapshot.write_text(json.dumps(actual, indent=2), encoding="utf-8")
    elif actual != json.loads(args.snapshot.read_text(encoding="utf-8")):
        raise RuntimeError(
            "Sessions, versions, or settings changed across the MSI lifecycle."
        )
    if args.expected_head:
        with sqlite3.connect(
            f"{args.database.resolve().as_uri()}?mode=ro", uri=True
        ) as connection:
            if connection.execute(
                "SELECT version_num FROM alembic_version"
            ).fetchall() != [(args.expected_head,)]:
                raise RuntimeError(
                    "The upgraded database did not reach the expected Alembic head."
                )


if __name__ == "__main__":
    main()
