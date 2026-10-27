# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from services.runtime import system_status as system_status_module
from services.runtime import system_telemetry as telemetry_module
from services.runtime.system_status import SystemStatusService
from services.runtime.system_telemetry import SystemTelemetry


###############################################################################
def _descriptor(provider_id: str, scope: str, status: str) -> Any:
    return SimpleNamespace(id=provider_id, credential_scope=scope, catalog_status=status)


###############################################################################
class _StubModelConfig:
    def __init__(self, descriptors: list[Any]) -> None:
        self.descriptors = descriptors

    def load_current_snapshot(self) -> None:
        return None

    async def discover_provider_descriptors(self, snapshot: None) -> list[Any]:
        return self.descriptors


###############################################################################
class _StubAccessKeys:
    def __init__(self, scopes_with_keys: set[str]) -> None:
        self.scopes_with_keys = scopes_with_keys

    def get_active_key(self, scope: str) -> object | None:
        return object() if scope in self.scopes_with_keys else None


###############################################################################
def test_provider_count_requires_active_key_and_usable_catalog() -> None:
    descriptors = [
        _descriptor("openai", "openai", "available"),
        _descriptor("gemini", "gemini", "cached"),
        _descriptor("anthropic", "anthropic", "authentication_required"),
        _descriptor("deepseek", "deepseek", "available"),
        _descriptor("opencode_go", "opencode", "not_loaded"),
    ]
    service = SystemStatusService(
        model_config=_StubModelConfig(descriptors),  # type: ignore[arg-type]
        access_keys=_StubAccessKeys({"openai", "gemini", "anthropic", "opencode"}),  # type: ignore[arg-type]
        telemetry=SystemTelemetry(),
    )

    availability = asyncio.run(service._count_providers())

    assert (availability.active, availability.total) == (2, 5)


###############################################################################
@pytest.mark.parametrize(("status_code", "expected"), [(200, True), (500, False)])
def test_ollama_probe_uses_configured_host(
    monkeypatch: pytest.MonkeyPatch, status_code: int, expected: bool
) -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(status_code)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        system_status_module.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setattr(
        system_status_module,
        "get_server_settings",
        lambda: SimpleNamespace(
            llm_defaults=SimpleNamespace(ollama_host_default="http://ollama.test:9999/")
        ),
    )
    service = SystemStatusService(
        model_config=_StubModelConfig([]),  # type: ignore[arg-type]
        access_keys=_StubAccessKeys(set()),  # type: ignore[arg-type]
        telemetry=SystemTelemetry(),
    )

    assert asyncio.run(service._probe_ollama()) is expected
    assert requested == ["http://ollama.test:9999/api/tags"]


###############################################################################
def test_vram_is_unavailable_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry_module.shutil, "which", lambda name: None)

    assert SystemTelemetry().read_vram() is None
