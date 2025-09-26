import os
import sys
import types

if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class RequestError(Exception):
        pass

    httpx_stub.RequestError = RequestError
    httpx_stub.get = lambda *args, **kwargs: None
    sys.modules["httpx"] = httpx_stub

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from Pharmagent.app.utils.services.retrieval import RxNavClient


###############################################################################
class _DummyResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    # -------------------------------------------------------------------------
    def json(self) -> dict[str, object]:
        return self._payload


# -----------------------------------------------------------------------------
def _build_payload(entries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "drugGroup": {
            "conceptGroup": [
                {
                    "conceptProperties": entries,
                }
            ]
        }
    }


# -----------------------------------------------------------------------------
def test_rxnorm_expansion_includes_brand_and_ingredient(monkeypatch):
    payload = _build_payload(
        [
            {
                "name": "DULoxetine 60 MG Oral Capsule [Cymbalta]",
                "synonym": "Cymbalta Oral Capsule",
                "prescribableName": "Cymbalta",
                "suppress": "N",
            }
        ]
    )

    def _fake_get(*args, **kwargs):
        return _DummyResponse(payload)

    monkeypatch.setattr(
        "Pharmagent.app.utils.services.retrieval.httpx.get",
        _fake_get,
    )
    client = RxNavClient()
    candidates = client.expand("Cymbalta")
    assert "cymbalta" in candidates
    assert "duloxetine" in candidates


# -----------------------------------------------------------------------------
def test_rxnorm_expansion_handles_multi_ingredient(monkeypatch):
    payload = _build_payload(
        [
            {
                "name": "Amlodipine Besylate 5 MG / Atorvastatin Calcium 10 MG Oral Tablet",
                "synonym": "Caduet",
                "suppress": "N",
            }
        ]
    )

    def _fake_get(*args, **kwargs):
        return _DummyResponse(payload)

    monkeypatch.setattr(
        "Pharmagent.app.utils.services.retrieval.httpx.get",
        _fake_get,
    )
    client = RxNavClient()
    candidates = client.expand("Caduet")
    assert "amlodipine" in candidates
    assert "atorvastatin" in candidates
    assert "amlodipine / atorvastatin" in candidates


# -----------------------------------------------------------------------------
def test_rxnorm_handles_failures(monkeypatch):
    request_error = (
        __import__("Pharmagent.app.utils.services.retrieval", fromlist=["httpx"])
        .httpx.RequestError
    )

    def _raise(*args, **kwargs):
        raise request_error("boom")

    monkeypatch.setattr(
        "Pharmagent.app.utils.services.retrieval.httpx.get",
        _raise,
    )
    client = RxNavClient()
    candidates = client.expand("Unknown")
    assert candidates == {"unknown": "original"}


# -----------------------------------------------------------------------------
def test_rxnorm_caches_responses(monkeypatch):
    payload = _build_payload(
        [
            {
                "name": "Acetylsalicylic Acid 325 MG Oral Tablet [Aspirin]",
                "synonym": "Aspirin",
                "suppress": "N",
            }
        ]
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return _DummyResponse(payload)

    monkeypatch.setattr(
        "Pharmagent.app.utils.services.retrieval.httpx.get",
        _fake_get,
    )
    client = RxNavClient()
    first = client.expand("Aspirin")
    second = client.expand("Aspirin")
    assert first == second
    assert len(calls) == 1
    assert "aspirin" in first
