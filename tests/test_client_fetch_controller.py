import asyncio
import io
import os
import sys
import tarfile
import types

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

if "gradio" not in sys.modules:
    gradio_stub = types.ModuleType("gradio")

    def update(**kwargs):  # type: ignore[override]
        return kwargs

    gradio_stub.update = update
    sys.modules["gradio"] = gradio_stub

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

        async def pull(self, *args, **kwargs):
            raise AssertionError("Unexpected pull call in tests")

        async def start_server(self):
            return "started"

        async def is_server_online(self) -> bool:
            return True

        async def preload_models(self, *args, **kwargs):
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

from Pharmagent.app.constants import LIVERTOX_ARCHIVE


# -----------------------------------------------------------------------------
def _build_archive(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        data = b"payload"
        info = tarfile.TarInfo(name="NBK/test.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


###############################################################################
class _DummyResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload

    @property
    def text(self) -> str:
        return ""


###############################################################################
class _DummyClient:
    def __init__(self, recorder: dict[str, object], payload: dict[str, object]) -> None:
        self.recorder = recorder
        self.payload = payload

    async def __aenter__(self) -> "_DummyClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str, params=None):
        self.recorder["url"] = url
        self.recorder["params"] = params
        return _DummyResponse(self.payload)


# -----------------------------------------------------------------------------
def test_fetch_clinical_data_uses_skip_flag(monkeypatch, tmp_path) -> None:
    from Pharmagent.app.client import controllers as module

    archive_path = tmp_path / LIVERTOX_ARCHIVE
    _build_archive(str(archive_path))

    payload = {
        "file_path": str(archive_path),
        "size": archive_path.stat().st_size,
        "last_modified": "local",
        "processed_entries": 1,
        "records": 1,
    }
    recorder: dict[str, object] = {}

    monkeypatch.setattr(module, "SOURCES_PATH", str(tmp_path))
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _DummyClient(recorder, payload),
    )

    message = asyncio.run(module.fetch_clinical_data(True))

    assert recorder["params"] == {"skip_download": "true"}
    assert "Processed monographs" in message
