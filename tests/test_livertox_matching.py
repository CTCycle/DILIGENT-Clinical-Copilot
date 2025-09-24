from __future__ import annotations

import asyncio
import os
import sys
import types

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

if "pydantic" not in sys.modules:
    pydantic_stub = types.ModuleType("pydantic")

    class BaseModel:  # type: ignore[override]
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, *, default_factory=None, **kwargs):  # type: ignore[override]
        if default_factory is not None:
            return default_factory()
        return default

    def field_validator(*args, **kwargs):  # type: ignore[override]
        def decorator(func):
            return func

        return decorator

    def model_validator(*args, **kwargs):  # type: ignore[override]
        def decorator(func):
            return func

        return decorator

    pydantic_stub.BaseModel = BaseModel
    pydantic_stub.Field = Field
    pydantic_stub.field_validator = field_validator
    pydantic_stub.model_validator = model_validator
    sys.modules["pydantic"] = pydantic_stub

if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class Timeout:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class AsyncClient:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            raise RuntimeError("httpx stub cannot perform network operations")

        async def aclose(self):  # compatibility with real client
            return None

    class Response:  # type: ignore[override]
        def __init__(self, status_code=200, text=""):
            self.status_code = status_code
            self.text = text

        def raise_for_status(self):
            return None

        def json(self):
            return {}

    class RequestError(Exception):
        pass

    class HTTPStatusError(Exception):
        pass

    class TimeoutException(Exception):
        pass

    httpx_stub.AsyncClient = AsyncClient
    httpx_stub.Timeout = Timeout
    httpx_stub.Response = Response
    httpx_stub.RequestError = RequestError
    httpx_stub.HTTPStatusError = HTTPStatusError
    httpx_stub.TimeoutException = TimeoutException
    sys.modules["httpx"] = httpx_stub

if "Pharmagent.app.api.models.providers" not in sys.modules:
    providers_stub = types.ModuleType("Pharmagent.app.api.models.providers")

    class _StubLLMClient:
        async def llm_structured_call(self, *args, **kwargs):
            raise RuntimeError("LLM call not expected in tests")

    def initialize_llm_client(*args, **kwargs):  # type: ignore[override]
        return _StubLLMClient()

    providers_stub.initialize_llm_client = initialize_llm_client
    sys.modules["Pharmagent.app.api.models.providers"] = providers_stub

if "Pharmagent.app.logger" not in sys.modules:
    logger_stub = types.ModuleType("Pharmagent.app.logger")

    class _StubLogger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

    logger_stub.logger = _StubLogger()
    sys.modules["Pharmagent.app.logger"] = logger_stub

from Pharmagent.app.api.schemas.clinical import PatientDrugs
from Pharmagent.app.utils.services.clinical import (
    CandidateSummary,
    DrugToxicityEssay,
    NameCandidate,
)


###############################################################################
class _DummyLLMClient:
    # -----------------------------------------------------------------------------
    async def llm_structured_call(self, *args, **kwargs):
        raise RuntimeError("LLM call not expected in tests")


# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch_llm_client(monkeypatch):
    monkeypatch.setattr(
        "Pharmagent.app.utils.services.clinical.initialize_llm_client",
        lambda *args, **kwargs: _DummyLLMClient(),
    )
    yield


# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_caches():
    DrugToxicityEssay._match_cache.clear()
    DrugToxicityEssay._rxnorm_term_cache.clear()
    DrugToxicityEssay._rxnorm_concept_cache.clear()
    yield
    DrugToxicityEssay._match_cache.clear()
    DrugToxicityEssay._rxnorm_term_cache.clear()
    DrugToxicityEssay._rxnorm_concept_cache.clear()


# -----------------------------------------------------------------------------
def test_direct_match_case_sensitive(monkeypatch):
    essay = DrugToxicityEssay(PatientDrugs(entries=[]))

    async def fake_esearch(client, term, *, title_only):
        if title_only:
            return ["NBK100"] if "Acetaminophen" in term else []
        return []

    async def fake_summary(client, ids):
        return [CandidateSummary(nbk_id="NBK100", title="Acetaminophen", synonyms=set())]

    monkeypatch.setattr(essay, "_query_esearch", fake_esearch)
    monkeypatch.setattr(essay, "_fetch_candidate_summaries", fake_summary)

    match = asyncio.run(essay._search_livertox_id(object(), "Acetaminophen"))
    assert match is not None
    assert match.nbk_id == "NBK100"
    assert match.reason == "direct_match"
    assert match.matched_name == "Acetaminophen"
    assert match.confidence == 1.0


# -----------------------------------------------------------------------------
def test_direct_match_case_insensitive(monkeypatch):
    essay = DrugToxicityEssay(PatientDrugs(entries=[]))

    async def fake_esearch(client, term, *, title_only):
        if title_only:
            return ["NBK101"] if "Acetaminophen" in term else []
        return []

    async def fake_summary(client, ids):
        return [CandidateSummary(nbk_id="NBK101", title="Acetaminophen", synonyms=set())]

    monkeypatch.setattr(essay, "_query_esearch", fake_esearch)
    monkeypatch.setattr(essay, "_fetch_candidate_summaries", fake_summary)

    match = asyncio.run(essay._search_livertox_id(object(), "acetaminophen"))
    assert match is not None
    assert match.nbk_id == "NBK101"
    assert match.reason == "direct_match"
    assert match.matched_name == "Acetaminophen"
    assert match.confidence == 1.0


# -----------------------------------------------------------------------------
def test_brand_name_resolves_to_active(monkeypatch):
    essay = DrugToxicityEssay(PatientDrugs(entries=[]))

    async def fake_esearch(client, term, *, title_only):
        if title_only:
            if "Acetaminophen" in term:
                return ["NBK200"]
            return []
        return []

    async def fake_summary(client, ids):
        return [CandidateSummary(nbk_id="NBK200", title="Acetaminophen", synonyms={"Tylenol"})]

    async def fake_lookup(client, drug_name):
        candidate = NameCandidate(origin="rxnorm_brand", name="Acetaminophen", priority=0)
        return [candidate], {"acetaminophen"}

    monkeypatch.setattr(essay, "_query_esearch", fake_esearch)
    monkeypatch.setattr(essay, "_fetch_candidate_summaries", fake_summary)
    monkeypatch.setattr(essay, "_lookup_rxnorm_candidates", fake_lookup)

    match = asyncio.run(essay._search_livertox_id(object(), "Tylenol"))
    assert match is not None
    assert match.nbk_id == "NBK200"
    assert match.matched_name == "Acetaminophen"
    assert match.reason == "brand_resolved"
    assert match.confidence >= 0.9


# -----------------------------------------------------------------------------
def test_fuzzy_match_with_misspelling(monkeypatch):
    essay = DrugToxicityEssay(PatientDrugs(entries=[]))

    async def fake_esearch(client, term, *, title_only):
        if title_only:
            return []
        return ["NBK300", "NBK301"]

    async def fake_summary(client, ids):
        return [
            CandidateSummary(
                nbk_id="NBK300",
                title="Acetaminophen",
                synonyms={"Paracetamol"},
            )
        ]

    async def fake_lookup(client, drug_name):
        return [], set()

    monkeypatch.setattr(essay, "_query_esearch", fake_esearch)
    monkeypatch.setattr(essay, "_fetch_candidate_summaries", fake_summary)
    monkeypatch.setattr(essay, "_lookup_rxnorm_candidates", fake_lookup)

    match = asyncio.run(essay._search_livertox_id(object(), "Acetaminophein"))
    assert match is not None
    assert match.nbk_id == "NBK300"
    assert match.matched_name == "Acetaminophen"
    assert match.reason == "fuzzy_match"
    assert match.confidence == 0.89


# -----------------------------------------------------------------------------
def test_list_first_fallback(monkeypatch):
    essay = DrugToxicityEssay(PatientDrugs(entries=[]))

    async def fake_esearch(client, term, *, title_only):
        if title_only:
            return []
        return ["NBK400", "NBK401"]

    async def fake_summary(client, ids):
        return [CandidateSummary(nbk_id="NBK400", title="Unrelated", synonyms=set())]

    async def fake_lookup(client, drug_name):
        return [], set()

    monkeypatch.setattr(essay, "_query_esearch", fake_esearch)
    monkeypatch.setattr(essay, "_fetch_candidate_summaries", fake_summary)
    monkeypatch.setattr(essay, "_lookup_rxnorm_candidates", fake_lookup)

    match = asyncio.run(essay._search_livertox_id(object(), "MysteryDrug"))
    assert match is not None
    assert match.nbk_id == "NBK400"
    assert match.matched_name == "Unrelated"
    assert match.reason == "list_first"
    assert match.confidence == 0.40
