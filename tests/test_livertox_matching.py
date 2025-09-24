import asyncio
import io
import os
import sys
import tarfile
import types
from pathlib import Path

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

from Pharmagent.app.api.schemas.clinical import DrugEntry, PatientDrugs
from Pharmagent.app.utils.services.clinical import DrugToxicityEssay
from Pharmagent.app.constants import LIVERTOX_ARCHIVE


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
@pytest.fixture()
def sample_archive(tmp_path: Path) -> Path:
    archive_path = tmp_path / LIVERTOX_ARCHIVE
    html_one = """
<html><head><title>Acetaminophen (Tylenol)</title></head>
<body>
<p>Synonyms: Tylenol, Paracetamol.</p>
<h2>Hepatotoxicity</h2>
<p>High doses cause liver injury.</p>
<h2>Mechanism of Injury</h2>
<p>Metabolites form reactive intermediates.</p>
<h2>Outcome and Management</h2>
<p>Supportive care is recommended.</p>
</body></html>
"""
    html_two = """
<html><head><title>Amoxicillin</title></head>
<body>
<p>Synonyms: Amoxil.</p>
<h2>Hepatotoxicity</h2>
<p>Rare hypersensitivity reactions.</p>
<h2>Mechanism</h2>
<p>Likely immune mediated.</p>
</body></html>
"""
    with tarfile.open(archive_path, "w:gz") as tar:
        for name, html_text in {
            "NBK100.html": html_one,
            "NBK200.html": html_two,
        }.items():
            data = html_text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive_path


# -----------------------------------------------------------------------------
def test_search_direct_match(sample_archive: Path):
    essay = DrugToxicityEssay(PatientDrugs(entries=[]), archive_path=str(sample_archive))
    match = asyncio.run(essay._search_livertox_id("Acetaminophen"))
    assert match is not None
    assert match.nbk_id == "NBK100"
    assert match.reason == "direct_match"
    assert match.confidence == pytest.approx(1.0)


# -----------------------------------------------------------------------------
def test_search_alias_match(sample_archive: Path):
    essay = DrugToxicityEssay(PatientDrugs(entries=[]), archive_path=str(sample_archive))
    match = asyncio.run(essay._search_livertox_id("Tylenol"))
    assert match is not None
    assert match.nbk_id == "NBK100"
    assert match.reason in {"alias_match", "fuzzy_match"}
    assert match.confidence >= 0.90


# -----------------------------------------------------------------------------
def test_fetch_content_sections(sample_archive: Path):
    essay = DrugToxicityEssay(PatientDrugs(entries=[]), archive_path=str(sample_archive))
    sections = asyncio.run(essay._fetch_livertox_content("NBK100"))
    assert sections["summary"]
    assert "High doses" in sections["hepatotoxicity"]
    assert "Supportive care" in sections["outcome"]
    assert sections["title"] == "Acetaminophen (Tylenol)"


# -----------------------------------------------------------------------------
def test_run_analysis_reports_no_match(sample_archive: Path):
    drugs = PatientDrugs(entries=[DrugEntry(name="ImaginaryDrug")])
    essay = DrugToxicityEssay(drugs, archive_path=str(sample_archive))
    result = asyncio.run(essay.run_analysis())
    assert result.entries[0].analysis is None
    assert "No LiverTox" in (result.entries[0].error or "")


# -----------------------------------------------------------------------------
def test_archive_download_trigger(sample_archive: Path, tmp_path: Path, monkeypatch):
    target_path = tmp_path / LIVERTOX_ARCHIVE

    async def fake_download(dest_path):
        dest_dir = Path(dest_path)
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = dest_dir / LIVERTOX_ARCHIVE
        copied.write_bytes(sample_archive.read_bytes())
        return {"file_path": str(copied), "size": copied.stat().st_size, "last_modified": None}

    monkeypatch.setattr(
        "Pharmagent.app.utils.services.scraper.LiverToxClient.download_bulk_data",
        fake_download,
    )

    essay = DrugToxicityEssay(PatientDrugs(entries=[]), archive_path=str(target_path))
    match = asyncio.run(essay._search_livertox_id("Amoxicillin"))
    assert match is not None
    assert match.nbk_id == "NBK200"
