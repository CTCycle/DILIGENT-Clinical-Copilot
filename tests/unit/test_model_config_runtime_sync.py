from __future__ import annotations

from DILIGENT.server.configurations import model_runtime


def test_sync_runtime_model_config_applies_snapshot(monkeypatch) -> None:
    sentinel_snapshot = object()
    applied: dict[str, object] = {}

    monkeypatch.setattr(
        model_runtime.ModelConfigService,
        "ensure_defaults",
        lambda self: sentinel_snapshot,
    )

    def fake_apply(self, snapshot: object) -> None:
        applied["snapshot"] = snapshot

    monkeypatch.setattr(
        model_runtime.ModelConfigService,
        "apply_runtime_snapshot",
        fake_apply,
    )

    model_runtime.sync_runtime_model_config()

    assert applied["snapshot"] is sentinel_snapshot
