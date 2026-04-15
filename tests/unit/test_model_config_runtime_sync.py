from __future__ import annotations

from DILIGENT.server.api import model_config


def test_sync_runtime_model_config_applies_snapshot(monkeypatch) -> None:
    sentinel_snapshot = object()
    applied: dict[str, object] = {}

    monkeypatch.setattr(
        model_config.endpoint,
        "ensure_defaults",
        lambda: sentinel_snapshot,
    )

    def fake_apply(snapshot: object) -> None:
        applied["snapshot"] = snapshot

    monkeypatch.setattr(
        model_config.endpoint,
        "apply_runtime_snapshot",
        fake_apply,
    )

    model_config.sync_runtime_model_config()

    assert applied["snapshot"] is sentinel_snapshot
