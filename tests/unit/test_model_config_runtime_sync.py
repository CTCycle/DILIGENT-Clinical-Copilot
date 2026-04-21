from __future__ import annotations

from DILIGENT.server.configurations import model_runtime


def test_sync_runtime_model_config_applies_snapshot(monkeypatch) -> None:
    sentinel_snapshot = object()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        model_runtime.ModelConfigService,
        "ensure_defaults",
        lambda self: observed.setdefault("snapshot", sentinel_snapshot),
    )

    model_runtime.sync_runtime_model_config()

    assert observed["snapshot"] is sentinel_snapshot
