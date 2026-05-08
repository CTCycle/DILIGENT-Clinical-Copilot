from __future__ import annotations

from services.llm.model_config import ModelConfigService


def test_model_config_service_ensure_defaults_applies_snapshot(monkeypatch) -> None:
    sentinel_snapshot = object()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        ModelConfigService,
        "ensure_defaults",
        lambda self: observed.setdefault("snapshot", sentinel_snapshot),
    )

    ModelConfigService().ensure_defaults()

    assert observed["snapshot"] is sentinel_snapshot
