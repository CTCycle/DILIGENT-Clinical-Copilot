from __future__ import annotations

import os
import sys
import types

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

if "gradio" not in sys.modules:
    gradio_stub = types.ModuleType("gradio")

    def update(**kwargs):  # type: ignore[override]
        return kwargs

    gradio_stub.update = update
    sys.modules["gradio"] = gradio_stub

if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class _AsyncClient:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            return None

        async def __aenter__(self) -> "_AsyncClient":
            raise AssertionError("httpx.AsyncClient should not be used in tests")

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class RequestError(Exception):
        pass

    def get(*args, **kwargs):
        raise AssertionError("httpx.get should be patched in tests")

    httpx_stub.AsyncClient = _AsyncClient  # type: ignore[attr-defined]
    httpx_stub.RequestError = RequestError  # type: ignore[attr-defined]
    httpx_stub.HTTPStatusError = Exception  # type: ignore[attr-defined]
    httpx_stub.ConnectError = Exception  # type: ignore[attr-defined]
    httpx_stub.TimeoutException = Exception  # type: ignore[attr-defined]
    httpx_stub.get = get  # type: ignore[attr-defined]
    sys.modules["httpx"] = httpx_stub

if "Pharmagent.app.api.models.providers" not in sys.modules:
    providers_stub = types.ModuleType("Pharmagent.app.api.models.providers")

    class OllamaError(Exception):
        pass

    class OllamaTimeout(OllamaError):
        pass

    class OllamaClient:  # type: ignore[override]
        async def __aenter__(self) -> "OllamaClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def pull(self, *args, **kwargs):  # type: ignore[override]
            raise AssertionError("Unexpected pull call in tests")

        async def start_server(self):  # type: ignore[override]
            return "started"

        async def is_server_online(self) -> bool:  # type: ignore[override]
            return True

        async def preload_models(self, *args, **kwargs):  # type: ignore[override]
            return [], []

    class _StubLLMClient:  # type: ignore[override]
        async def llm_structured_call(self, *args, **kwargs):
            raise AssertionError("LLM calls are not expected in tests")

    def initialize_llm_client(*args, **kwargs):  # type: ignore[override]
        return _StubLLMClient()

    providers_stub.OllamaClient = OllamaClient
    providers_stub.OllamaError = OllamaError
    providers_stub.OllamaTimeout = OllamaTimeout
    providers_stub.initialize_llm_client = initialize_llm_client
    sys.modules["Pharmagent.app.api.models.providers"] = providers_stub

from Pharmagent.app.client import controllers as module
from Pharmagent.app.configurations import ClientRuntimeConfig


# -----------------------------------------------------------------------------
def test_clear_agent_fields_resets_values() -> None:
    cleared = module.clear_agent_fields()
    assert cleared == (
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        [],
        False,
        False,
        False,
        "",
    )


# -----------------------------------------------------------------------------
def test_toggle_cloud_services_updates_runtime() -> None:
    ClientRuntimeConfig.reset_defaults()
    updates = module.toggle_cloud_services(True)
    assert ClientRuntimeConfig.is_cloud_enabled() is True
    provider_update, model_update, button_update_a, button_update_b, temperature_update, reasoning_update = updates
    assert provider_update["interactive"] is True
    assert model_update["interactive"] is True
    assert button_update_a["interactive"] is False
    assert button_update_b["interactive"] is False
    assert temperature_update["interactive"] is False
    assert reasoning_update["interactive"] is False
    ClientRuntimeConfig.reset_defaults()
