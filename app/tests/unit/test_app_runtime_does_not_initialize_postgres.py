from __future__ import annotations

import importlib
import sys


def test_fastapi_app_import_does_not_trigger_database_initializer(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    called = {"postgres": False}

    def _fake_ensure_postgres_database(_settings):  # type: ignore[no-untyped-def]
        called["postgres"] = True
        raise AssertionError(
            "PostgreSQL initializer should never run during app runtime import"
        )

    monkeypatch.setattr(
        "repositories.database.initializer.ensure_postgres_database",
        _fake_ensure_postgres_database,
    )

    sys.modules.pop("app", None)
    importlib.import_module("app")

    assert called["postgres"] is False

